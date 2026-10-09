import numpy as np
import pytest

from motile_tracker.membrane.features import (
    RAW_FEATURES,
    add_membrane_features,
    compute_membrane_features,
)
from motile_tracker.membrane.refine import (
    RefineParams,
    boundary_signal,
    drop_islands,
    raw_cavity,
    refine_boundaries,
)
from motile_tracker.membrane.score import (
    ScoreParams,
    blend_for_segmentation,
    embryo_from_raw,
    membrane_score,
    mitotic_nodes,
    sheet_score,
)

ISO = (1.0, 1.0, 1.0)


def _two_cells(shape=(12, 30, 60), old_x=26):
    labels = np.zeros(shape, np.uint32)
    labels[:, :, :old_x] = 1
    labels[:, :, old_x:] = 2
    nuclei = np.zeros(shape, np.uint32)
    nuclei[5:7, 13:16, 8:11] = 1
    nuclei[5:7, 13:16, 48:51] = 2
    return labels, nuclei


# --- score --------------------------------------------------------------------


def test_sheet_score_prefers_sheets_over_fibres():
    img = np.zeros((30, 40, 40), np.float32)
    img[:, :, 20] = 1.0  # a sheet (plane x = 20)
    img[:, 10, 8] = 1.0  # a fibre along z
    s = sheet_score(img, 1.0, ISO)
    assert s[15, 30, 20] > 5 * s[15, 10, 8]


def test_sheet_score_handles_anisotropic_voxels():
    img = np.zeros((20, 60, 60), np.float32)
    img[:, :, 30] = 1.0
    s = sheet_score(img, 1.0, (1.0, 0.26, 0.26))
    assert s[10, 30, 30] == s[:, :, 25:36].max()


def test_membrane_score_blanks_mitotic_chromatin():
    raw = np.full((20, 40, 40), 50.0, np.float32)
    raw[:, :, 20] = 400.0  # membrane far from the nucleus
    raw[8:12, 5:9, 5:9] = 1000.0  # bright spindle-like blob at the nucleus
    nuclei = np.zeros(raw.shape, np.uint32)
    nuclei[9:11, 6:8, 6:8] = 7
    kept = membrane_score(raw, nuclei, ISO)
    blanked = membrane_score(raw, nuclei, ISO, mitotic=[7])
    assert blanked[10, 7, 7] == 0 and blanked[10, 30, 20] > 0
    assert kept[10, 30, 20] == pytest.approx(blanked[10, 30, 20])


def test_embryo_from_raw_fills_each_slice():
    raw = np.zeros((5, 30, 30), np.float32)
    raw[:, 5:25, 5:25] = 10
    raw[:, 12:18, 12:18] = 0  # blank inside (a cavity)
    emb = embryo_from_raw(raw)
    assert emb[2, 15, 15] and not emb[2, 1, 1]


def test_blend_is_uint16_and_boosts_membranes():
    raw = np.full((10, 30, 30), 50.0, np.float32)
    score = np.zeros(raw.shape, np.float32)
    score[:, :, 15] = 1.0
    b = blend_for_segmentation(raw, score, ISO)
    assert b.dtype == np.uint16
    assert b[5, 15, 15] > b[5, 15, 5]


def test_mitotic_nodes(solution_tracks_3d):
    # Node 1 (t=0) divides into 2 and 3 (t=1).
    assert mitotic_nodes(solution_tracks_3d, [1], before=1, after=0) == {1}
    assert mitotic_nodes(solution_tracks_3d, [2, 3], before=0, after=1) == {2, 3}
    assert mitotic_nodes(solution_tracks_3d, [2, 3], before=1, after=0) == set()


# --- refinement --------------------------------------------------------------


def test_boundary_moves_onto_clear_membrane():
    labels, nuclei = _two_cells(old_x=26)
    score = np.zeros(labels.shape, np.float32)
    # The real membrane 2 voxels from the old boundary, inside the band
    # (30 % of the cell radius of ~13 voxels, so +-3.9).
    score[:, :, 28] = 1.0
    out, st = refine_boundaries(labels, score, nuclei, ISO)
    assert st["moved"] == 1
    assert (out[:, :, :28] == 1).all() and (out[:, :, 29:] == 2).all()


def test_boundary_stays_without_membrane():
    labels, nuclei = _two_cells(old_x=26)
    out, st = refine_boundaries(labels, np.zeros(labels.shape, np.float32), nuclei, ISO)
    np.testing.assert_array_equal(out, labels)
    assert st["moved"] == 0


def test_boundary_ignores_membrane_outside_band():
    labels, nuclei = _two_cells(old_x=26)
    score = np.zeros(labels.shape, np.float32)
    score[:, :, 45] = 1.0  # far away, beyond the band
    params = RefineParams(band_max_um=3.0)
    out, _ = refine_boundaries(labels, score, nuclei, ISO, params)
    np.testing.assert_array_equal(out, labels)


def test_local_move_follows_partial_membrane_only():
    labels, nuclei = _two_cells(old_x=26)
    score = np.zeros(labels.shape, np.float32)
    score[:6, :, 28] = 1.0  # membrane visible in the upper half only
    out, _ = refine_boundaries(
        labels,
        score,
        nuclei,
        ISO,
        RefineParams(strong=2.0),  # force local mode
    )
    assert (out[1, :, 27] == 1).all()  # moved where the membrane is
    assert (out[10, :, 26] == 2).all()  # stayed where there is none


def test_drop_islands_gives_strays_to_neighbour():
    labels = np.zeros((6, 20, 20), np.uint32)
    labels[:, :, :10] = 1
    labels[:, :, 10:] = 2
    labels[2:4, 5:7, 15:17] = 1  # stray piece of cell 1 inside cell 2
    out, removed = drop_islands(labels, ISO)
    assert removed == 8
    assert (out[2:4, 5:7, 15:17] == 2).all()


def test_raw_cavity_finds_blank_inside_not_rim():
    raw = np.zeros((20, 60, 60), np.float32)
    raw[:, 5:55, 5:55] = 100.0
    raw[4:16, 20:40, 20:40] = 0.0  # cavity
    rng = np.random.default_rng(0)
    rim = np.zeros(raw.shape, bool)
    rim[:, 5:8, 5:55] = True
    raw[rim & (rng.random(raw.shape) < 0.7)] = 0.0  # blank speckle at the rim
    emb = embryo_from_raw(raw)
    nuclei = np.zeros(raw.shape, np.uint32)
    cav = raw_cavity(raw, nuclei, ISO, emb, RefineParams(cavity_min_um3=500))
    assert cav[10, 30, 30]
    assert not cav[:, 5:8].any()


def test_boundary_signal_per_cell():
    labels, _ = _two_cells(old_x=26)
    score = np.zeros(labels.shape, np.float32)
    score[:, :, 25:27] = 0.8
    sig = boundary_signal(labels, score)
    assert sig[1] == pytest.approx(0.8) and sig[2] == pytest.approx(0.8)


# --- whole run --------------------------------------------------------------


@pytest.fixture
def raw_and_membrane():
    """Membrane labels as in test_features, raw image with a membrane sheet."""
    membrane = np.zeros((2, 100, 100, 100), dtype=np.uint32)
    membrane[0, 25:76, 25:76, 25:50] = 5
    membrane[0, 25:76, 25:76, 50:76] = 6
    membrane[1, 5:80, 30:70, 25:95] = 9
    raw = np.where(membrane > 0, 60.0, 0.0).astype(np.float32)
    raw[1, 5:80, 30:70, 63] = 600.0  # a membrane between nuclei 2 and 3
    return membrane, raw


def test_compute_with_raw_adds_boundary_signal(solution_tracks_3d, raw_and_membrane):
    membrane, raw = raw_and_membrane
    merged, feats = compute_membrane_features(
        solution_tracks_3d, membrane, raw=raw, division_window=0
    )
    assert set(np.unique(merged[1])) == {0, 2, 3}
    for n in (1, 2, 3):
        assert "membrane_boundary_signal" in feats[n]
    add_membrane_features(solution_tracks_3d, feats)
    assert "membrane_boundary_signal" in solution_tracks_3d.features
    assert feats[2]["membrane_volume"] == (merged[1] == 2).sum()


def test_compute_without_raw_has_no_raw_features(solution_tracks_3d, raw_and_membrane):
    membrane, _ = raw_and_membrane
    _, feats = compute_membrane_features(solution_tracks_3d, membrane)
    add_membrane_features(solution_tracks_3d, feats)
    for key in RAW_FEATURES:
        assert key not in solution_tracks_3d.features


def test_compute_with_raw_parallel_matches_sequential(
    solution_tracks_3d, raw_and_membrane
):
    membrane, raw = raw_and_membrane
    seq, seq_feats = compute_membrane_features(solution_tracks_3d, membrane, raw=raw)
    par, par_feats = compute_membrane_features(
        solution_tracks_3d, membrane, raw=raw, workers=2
    )
    np.testing.assert_array_equal(seq, par)
    assert seq_feats.keys() == par_feats.keys()
    for n in seq_feats:
        for k, v in seq_feats[n].items():
            w = par_feats[n][k]
            assert (v == w) or (np.isnan(v) and np.isnan(w))


def test_score_params_are_used(solution_tracks_3d, raw_and_membrane):
    membrane, raw = raw_and_membrane
    _, feats = compute_membrane_features(
        solution_tracks_3d,
        membrane,
        raw=raw,
        score_params=ScoreParams(mitotic_before=0, mitotic_after=0),
    )
    assert feats[2]["membrane_boundary_signal"] >= 0


def test_blend_frame_is_zero_outside_embryo_and_full_size():
    from motile_tracker.membrane.score import blend_frame

    raw = np.zeros((10, 40, 40), np.float32)
    raw[:, 10:30, 10:30] = 50.0
    raw[:, 10:30, 20] = 400.0
    out = blend_frame(raw, np.zeros(raw.shape, np.uint32), ISO)
    assert out.shape == raw.shape and out.dtype == np.uint16
    assert not out[:, :5].any()
    assert out[5, 20, 20] > out[5, 20, 15]


def test_sym3_eigvals_matches_numpy():
    from motile_tracker.membrane.score import sym3_eigvals

    rng = np.random.default_rng(1)
    m = rng.normal(size=(2000, 3, 3))
    m = m + m.transpose(0, 2, 1)
    m[:5] = 0.0  # zero matrices
    m[5:10] = np.eye(3)  # repeated eigenvalues
    ref = np.linalg.eigvalsh(m)
    got = sym3_eigvals(
        m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], m[:, 0, 1], m[:, 0, 2], m[:, 1, 2]
    )
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_embryo_from_raw_ignores_sparse_background_noise():
    rng = np.random.default_rng(0)
    raw = np.zeros((10, 60, 60), np.float32)
    raw[:, 15:45, 15:45] = 100.0
    noise = rng.random(raw.shape) < 0.02  # sparse non-zero background voxels
    raw[noise & (raw == 0)] = 5.0
    emb = embryo_from_raw(raw)
    assert emb[5, 30, 30]
    assert not emb[:, :10].any() and not emb[:, :, 50:].any()


def test_nucleus_without_cell_gets_one_with_raw(solution_tracks_3d):
    """Node 1's fragment is missing from the labels: with raw, it still gets a
    cell, grown from its nucleus and the embryo outline of the raw image."""
    membrane = np.zeros((2, 100, 100, 100), dtype=np.uint32)
    membrane[1, 5:80, 30:70, 25:95] = 9  # frame 0 has no membrane label at all
    raw = np.zeros(membrane.shape, np.float32)
    raw[:, 20:85, 20:85, 20:85] = 60.0
    _, plain = compute_membrane_features(solution_tracks_3d, membrane, max_frames=1)
    merged, feats = compute_membrane_features(
        solution_tracks_3d, membrane, raw=raw, max_frames=1
    )
    assert plain[1]["membrane_qc"] == "no_membrane"
    assert feats[1]["membrane_qc"] == "from_nucleus"
    assert feats[1]["membrane_id"] == 1
    assert (merged[0][20:85, 20:85, 20:85] == 1).mean() > 0.95


def test_nucleus_inside_other_cell_splits_it_with_raw(solution_tracks_3d):
    """At t=1 the label covers all of nucleus 2 but under half of nucleus 3, so
    only 2 seeds it and 3 gets no cell from the merge; with raw, 3's nucleus
    lies inside 2's cell, which is then split between the two."""
    membrane = np.zeros((2, 100, 100, 100), dtype=np.uint32)
    membrane[0, 25:76, 25:76, 25:76] = 5
    membrane[1, 5:85, 25:75, 46:95] = 9  # nucleus 3 spans x 30..61: ~48 % inside
    raw = np.zeros(membrane.shape, np.float32)
    raw[:, 5:85, 25:75, 25:95] = 60.0
    _, plain = compute_membrane_features(
        solution_tracks_3d, membrane, division_window=0, start_frame=1
    )
    merged, feats = compute_membrane_features(
        solution_tracks_3d, membrane, raw=raw, division_window=0, start_frame=1
    )
    assert plain[3]["membrane_qc"] == "no_membrane"
    assert feats[3]["membrane_qc"] == "from_nucleus"
    assert (merged[0] == 2).any() and (merged[0] == 3).any()
    # nucleus 3's own voxels belong to its cell
    nuc3 = np.zeros(merged[0].shape, bool)
    nuc3[45:76, 35:66, 30:61] = True
    assert (merged[0][nuc3] == 3).mean() > 0.9
