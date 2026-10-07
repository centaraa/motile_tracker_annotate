import numpy as np
import pytest
from funtracks.import_export import import_from_geff

from motile_tracker.import_export.geff_io import write_geff_over
from motile_tracker.membrane.features import (
    MEMBRANE_FEATURES,
    add_membrane_features,
    compute_membrane_features,
    dividing_pairs,
)
from motile_tracker.membrane.io import (
    MEMBRANE_ARRAY,
    load_membrane_labels,
    save_membrane_labels,
)
from motile_tracker.membrane.merge import QC_DIVIDING, QC_OK, QC_SPLIT


@pytest.fixture
def membrane_3d():
    """t=0: node 1's box cut into two slabs. t=1: one fragment over nodes 2 and 3."""
    membrane = np.zeros((2, 100, 100, 100), dtype=np.uint32)
    membrane[0, 25:76, 25:76, 25:50] = 5
    membrane[0, 25:76, 25:76, 50:76] = 6
    membrane[1, 5:80, 30:70, 25:95] = 9
    return membrane


def test_dividing_pairs_finds_sisters(solution_tracks_3d):
    assert dividing_pairs(solution_tracks_3d, [2, 3], window=1) == [(2, 3)]
    assert dividing_pairs(solution_tracks_3d, [2, 3], window=0) == []
    assert dividing_pairs(solution_tracks_3d, [1], window=1) == []


def test_compute_merges_and_splits_dividing_cell(solution_tracks_3d, membrane_3d):
    merged, feats = compute_membrane_features(solution_tracks_3d, membrane_3d)

    assert set(np.unique(merged[0])) == {0, 1}
    assert feats[1]["membrane_n_fragments"] == 2
    assert feats[1]["membrane_qc"] == QC_OK
    assert set(np.unique(merged[1])) == {0, 2, 3}
    assert feats[2]["membrane_qc"] == feats[3]["membrane_qc"] == QC_DIVIDING
    assert feats[3]["membrane_id"] == 3


def test_compute_splits_without_division_window(solution_tracks_3d, membrane_3d):
    merged, feats = compute_membrane_features(
        solution_tracks_3d, membrane_3d, division_window=0
    )
    assert set(np.unique(merged[1])) == {0, 2, 3}
    assert feats[2]["membrane_qc"] == QC_SPLIT


def test_shape_mismatch_raises(solution_tracks_3d):
    with pytest.raises(ValueError, match="shape"):
        compute_membrane_features(solution_tracks_3d, np.zeros((2, 10, 10, 10)))


def test_geff_round_trip(solution_tracks_3d, membrane_3d, tmp_path):
    tracks = solution_tracks_3d
    merged, feats = compute_membrane_features(tracks, membrane_3d)
    add_membrane_features(tracks, feats)
    path = tmp_path / "out.geff"
    write_geff_over(tracks, path)
    save_membrane_labels(path, merged)

    loaded = import_from_geff(path)
    for key in MEMBRANE_FEATURES:
        assert key in loaded.features
    assert loaded.get_node_attr(1, "membrane_n_fragments") == 2
    assert loaded.get_node_attr(3, "membrane_qc") == QC_DIVIDING
    np.testing.assert_array_equal(np.asarray(load_membrane_labels(path)), merged)

    # A re-save rewrites the geff metadata but keeps the array in the store.
    write_geff_over(loaded, path)
    assert (path / MEMBRANE_ARRAY).exists()


def test_max_frames_limits_processing(solution_tracks_3d, membrane_3d):
    merged, feats = compute_membrane_features(
        solution_tracks_3d, membrane_3d, max_frames=1
    )
    assert merged.shape == (1, 100, 100, 100)
    assert set(feats) == {1}


def test_read_label_file_tiff_folder(tmp_path):
    import tifffile

    from motile_tracker.membrane.io import read_label_file

    for t in (2, 1, 3):
        tifffile.imwrite(tmp_path / f"m_T{t:04d}.tif", np.full((2, 3, 4), t, np.uint16))
    stack = read_label_file(tmp_path)
    assert stack.shape == (3, 2, 3, 4)
    assert stack[0][0, 0, 0] == 1
    assert stack[2][0, 0, 0] == 3


def test_streams_frames_into_zarr_and_geff(solution_tracks_3d, membrane_3d, tmp_path):
    from motile_tracker.membrane.io import create_label_store

    store = create_label_store(tmp_path / "merged.zarr", membrane_3d.shape)
    merged, feats = compute_membrane_features(
        solution_tracks_3d, membrane_3d, out=store
    )
    assert merged is store
    in_memory, _ = compute_membrane_features(solution_tracks_3d, membrane_3d)
    np.testing.assert_array_equal(store[...], in_memory)

    add_membrane_features(solution_tracks_3d, feats)
    path = tmp_path / "out.geff"
    write_geff_over(solution_tracks_3d, path)
    save_membrane_labels(path, store)
    np.testing.assert_array_equal(np.asarray(load_membrane_labels(path)), in_memory)

    # Saving the store's own array back onto itself keeps the data.
    loaded = load_membrane_labels(path)
    save_membrane_labels(path, loaded)
    np.testing.assert_array_equal(np.asarray(load_membrane_labels(path)), in_memory)


def test_start_frame(solution_tracks_3d, membrane_3d, tmp_path):
    from motile_tracker.membrane.io import create_label_store

    merged, feats = compute_membrane_features(
        solution_tracks_3d, membrane_3d, start_frame=1
    )
    assert merged.shape == (1, 100, 100, 100)
    assert set(feats) == {2, 3}

    store = create_label_store(tmp_path / "m.zarr", membrane_3d.shape)
    compute_membrane_features(solution_tracks_3d, membrane_3d, start_frame=1, out=store)
    assert not store[0].any()
    np.testing.assert_array_equal(store[1], merged[0])


def test_parallel_matches_sequential(solution_tracks_3d, membrane_3d, tmp_path):
    from motile_tracker.membrane.io import create_label_store

    seq, seq_feats = compute_membrane_features(solution_tracks_3d, membrane_3d)
    store = create_label_store(tmp_path / "p.zarr", membrane_3d.shape)
    done = []
    par, par_feats = compute_membrane_features(
        solution_tracks_3d,
        membrane_3d,
        out=store,
        workers=2,
        progress=lambda i, n: done.append(i),
    )
    np.testing.assert_array_equal(par[...], seq)
    assert par_feats == seq_feats
    assert sorted(done) == [0, 1]


def test_keep_cavities_from_switches_per_frame(solution_tracks_3d):
    """A cavity inside cell 1 (t=0) is filled; the same at t=1 (cell 2) is kept."""
    membrane = np.zeros((2, 100, 100, 100), dtype=np.uint32)
    membrane[0, 25:76, 25:76, 25:76] = 5
    membrane[0, 40:46, 40:46, 60:66] = 0  # cavity away from nucleus 1
    membrane[1, 0:40, 30:70, 60:100] = 9
    membrane[1, 15:21, 45:51, 92:98] = 0  # cavity away from nucleus 2

    # A 1-voxel ball does not close the 6-voxel cavities (the default 6 would,
    # by design: only cavities wider than the ball count as cavities).
    from motile_tracker.membrane import MembraneMergeParams

    merged, _ = compute_membrane_features(
        solution_tracks_3d,
        membrane,
        params=MembraneMergeParams(closing_radius_um=1),
        keep_cavities_from=1,
        division_window=0,
    )

    assert merged[0, 43, 43, 63] == 1  # before the switch: filled
    assert merged[1, 18, 48, 95] == 0  # from the switch on: kept empty


def test_save_copies_zarr_store_file_by_file(solution_tracks_3d, membrane_3d, tmp_path):
    from motile_tracker.membrane.io import create_label_store

    store = create_label_store(tmp_path / "m.zarr", membrane_3d.shape)
    merged, feats = compute_membrane_features(
        solution_tracks_3d, membrane_3d, out=store
    )
    add_membrane_features(solution_tracks_3d, feats)
    path = tmp_path / "out.geff"
    write_geff_over(solution_tracks_3d, path)

    save_membrane_labels(path, store)

    # One chunk file per frame, copied as is.
    src_files = sorted(f.name for f in (tmp_path / "m.zarr").iterdir())
    assert sorted(f.name for f in (path / MEMBRANE_ARRAY).iterdir()) == src_files
    np.testing.assert_array_equal(np.asarray(load_membrane_labels(path)), store[...])

    # Saving the geff onto a new path copies its own array along.
    loaded = load_membrane_labels(path)
    other = tmp_path / "other.geff"
    write_geff_over(solution_tracks_3d, other)
    save_membrane_labels(other, loaded)
    np.testing.assert_array_equal(np.asarray(load_membrane_labels(other)), store[...])
