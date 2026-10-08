"""Stage classification, centrosome detection, candidates and the geff round
trip of the spindle features (synthetic data only)."""

import numpy as np
import pytest
from funtracks.import_export import import_from_geff

from motile_tracker.import_export.geff_io import write_geff_over
from motile_tracker.spindle import (
    SPINDLE_FEATURES,
    SpindleParams,
    add_spindle_features,
    classify_stage,
    find_candidates,
    measure_tracks,
)
from motile_tracker.spindle.core import detect_centrosomes, stage_metrics

SP = (1.0, 0.26, 0.26)
P = SpindleParams(spacing=SP)


def _case(dna):
    """Stage-metric input: the whole crop is the target nucleus."""
    return {"dna": dna.astype(np.float32), "target": np.ones(dna.shape, bool)}


def _coords(shape):
    c = (np.asarray(shape) - 1) / 2
    return (np.indices(shape).T - c).T * np.asarray(SP)[:, None, None, None]


SHAPE = (31, 100, 100)


def test_flat_plate_is_metaphase():
    z, y, x = _coords(SHAPE)
    dna = ((np.abs(x) <= 1.0) & (np.hypot(z, y) <= 8)) * 1000.0
    stage, _ = stage_metrics(_case(dna), P)
    assert stage["flatness"] < 0.4
    assert classify_stage(stage, 15.0, P, 2.0)[0] == "metaphase"
    assert classify_stage(stage, 15.0, P, 2.0)[2] == 1


def test_round_blob_is_prometaphase():
    z, y, x = _coords(SHAPE)
    dna = (np.sqrt(z**2 + y**2 + x**2) <= 6) * 1000.0
    stage, _ = stage_metrics(_case(dna), P)
    cls, rule, is_meta, _ = classify_stage(stage, 12.0, P, 2.0)
    assert (cls, is_meta) == ("prometaphase", 0)
    assert rule.startswith("not flat")


def test_two_masses_are_split():
    z, y, x = _coords(SHAPE)
    dna = ((np.abs(np.abs(x) - 5) <= 1.0) & (np.hypot(z, y) <= 7)) * 1000.0
    stage, _ = stage_metrics(_case(dna), P)
    assert classify_stage(stage, 14.0, P, 2.0)[0] == "split"


def test_no_plate_rim_and_thick_plate_are_prometaphase():
    stage = {
        "mass2_frac": 0.0,
        "flatness": 0.3,
        "disc": 0.9,
        "dna_axes_um": [18.0, 16.0, 5.0],
    }
    cls, rule, _, rim = classify_stage(
        stage, 2.5, P, 3.0
    )  # rosette: no edge at the rim
    assert cls == "prometaphase" and rule.startswith("no plate rim")
    assert rim == pytest.approx(2.5 / 18.0)
    cls, rule, _, _ = classify_stage(stage, 16.0, P, 8.5)
    assert cls == "prometaphase" and rule.startswith("thick plate")
    assert classify_stage(stage, 16.0, P, 2.5)[0] == "metaphase"
    # without a Spindle3D result the two plate rules cannot be tested
    cls, rule, _, rim = classify_stage(stage, np.nan, P)
    assert cls == "metaphase" and "not tested" in rule and np.isnan(rim)


def _aligned_spindle(with_puncta):
    """DNA-aligned frame (0.25 um voxels): spindle body along z, optional asters."""
    shape = (121, 61, 61)
    o = np.array([60, 30, 30])
    zz, yy, xx = np.indices(shape)
    lat = np.hypot(yy - o[1], xx - o[2]) * 0.25
    ax = (zz - o[0]) * 0.25
    body = (np.abs(ax) <= 9) & (lat <= 3 * (1 - np.abs(ax) / 11))
    tub = 20 + 300 * body.astype(float)
    if with_puncta:
        for zc in (o[0] - 44, o[0] + 44):  # 11 um from the plate, 2 um past the tips
            tub += 1500 * np.exp(
                -(((zz - zc) ** 2 + (yy - o[1]) ** 2 + (xx - o[2]) ** 2) / (2 * 2.0**2))
            )
    return tub, body, o


@pytest.mark.parametrize("with_puncta", [True, False])
def test_centrosome_detection(with_puncta):
    tub, body, o = _aligned_spindle(with_puncta)
    cents = detect_centrosomes(tub, body, o, 0.25, P, half_plate_um=1.0, peak_min=150.0)
    assert len(cents) == (2 if with_puncta else 0)
    for c in cents:
        assert abs(abs(c["pos"][0] - o[0]) - 44) <= 2


def test_candidates_from_division(solution_tracks_3d):
    # node 1 (t=0) divides into 2 and 3 (t=1)
    assert find_candidates(solution_tracks_3d, None, 6) == [(1, 0, 1, 1)]
    assert find_candidates(solution_tracks_3d, 0, 6) == [(1, 0, 1, 1)]
    assert find_candidates(solution_tracks_3d, -1, 6) == []


def test_measure_tracks_records_every_candidate(solution_tracks_3d):
    shape = tuple(solution_tracks_3d.segmentation.shape)
    rng = np.random.default_rng(0)
    tubulin = rng.normal(100, 5, shape).astype(np.float32)
    dna = rng.normal(100, 5, shape).astype(np.float32)
    params = SpindleParams(spacing=(1.0, 1.0, 1.0), half_size_um=12.0, z_extra_um=0.0)
    out = list(measure_tracks(solution_tracks_3d, tubulin, dna, params))
    assert len(out) == 1
    row, _, res = out[0]
    assert row["node"] == 1
    assert row["spindle_frames_before_division"] == 1
    assert row["spindle_status"] == res["status"]
    # noise only: Spindle3D fails, the failure is recorded, the stage still set
    assert row["spindle_status"].startswith("failed: ")
    assert row["spindle_stage"] in {"metaphase", "prometaphase", "split", "unclear"}


def test_bad_correction_raises(solution_tracks_3d):
    with pytest.raises(ValueError, match="correction"):
        list(
            measure_tracks(
                solution_tracks_3d, None, None, SpindleParams(correction="inpaint")
            )
        )


def test_geff_round_trip(solution_tracks_3d, tmp_path):
    tracks = solution_tracks_3d
    row = {
        "node": 1,
        "spindle_status": "ok",
        "spindle_stage": "metaphase",
        "spindle_stage_rule": "flat disc",
        "spindle_is_metaphase": 1,
        "spindle_frames_before_division": 1,
        "spindle_length_um": 18.5,
        "spindle_n_centrosomes": 2,
        "spindle_axis_z": 0.1,
    }
    add_spindle_features(tracks, [row])
    path = tmp_path / "out.geff"
    write_geff_over(tracks, path)
    loaded = import_from_geff(path)
    for key in SPINDLE_FEATURES:
        assert key in loaded.features
    assert loaded.get_node_attr(1, "spindle_length_um") == pytest.approx(18.5)
    assert loaded.get_node_attr(1, "spindle_stage") == "metaphase"
    assert loaded.get_node_attr(1, "spindle_n_centrosomes") == 2
    # a node that was not a candidate keeps the defaults
    assert loaded.get_node_attr(2, "spindle_status") == "not_measured"
    assert np.isnan(loaded.get_node_attr(2, "spindle_length_um"))
    assert loaded.get_node_attr(2, "spindle_is_metaphase") == -1
