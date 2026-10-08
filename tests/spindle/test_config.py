"""The TOML config of the spindle measurement (synthetic, fast)."""

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from motile_tracker.spindle import core
from motile_tracker.spindle.config import (
    DEFAULT_CONFIG_HEADER,
    SpindleConfig,
    default_config_path,
    dump_config,
    load_config,
    write_config,
)
from motile_tracker.spindle.core import SpindleParams, measure_crop
from motile_tracker.spindle.spindle3d import JavaSettings

# Spindle3DSettings.java (v0.8.0) defaults
JAVA_V080 = {
    "voxel_size_for_analysis": 0.25,
    "metaphase_plate_width_derivative_delta": 1.0,
    "metaphase_plate_length_derivative_delta": 2.0,
    "spindle_fragment_inclusion_zone": 3.0,
    "axial_pole_refinement_radius": 1.0,
    "lateral_pole_refinement_radius": 2.0,
    "voxel_size_for_initial_dna_threshold": 1.5,
    "initial_dna_threshold_factor": 0.5,
    "minimal_dynamic_range": 7,
}

SCRIPT = Path(__file__).parents[2] / "scripts" / "spindle_features.py"


def test_defaults_file_dataclasses_and_java_agree():
    path = default_config_path()
    assert path.exists()
    assert path.read_text(encoding="utf-8") == dump_config(
        SpindleConfig(), DEFAULT_CONFIG_HEADER
    )
    assert load_config(path) == SpindleConfig()
    java = dataclasses.asdict(JavaSettings())
    for key, value in JAVA_V080.items():
        assert java[key] == value
    assert java["pole_profile_radius"] == "java"


def test_no_parameter_in_both_sections():
    java = {f.name for f in dataclasses.fields(JavaSettings)}
    pipe = {f.name for f in dataclasses.fields(SpindleParams)}
    assert not java & pipe


def test_round_trip(tmp_path):
    cfg = SpindleConfig(
        spindle3d=JavaSettings(
            spindle_fragment_inclusion_zone=4.5, pole_profile_radius="um"
        ),
        pipeline=SpindleParams(
            spacing=(1.0, 0.26, 0.26), max_frame=454, correction="none"
        ),
    )
    path = tmp_path / "c.toml"
    write_config(cfg, path, "header\nline 2")
    loaded = load_config(path)
    assert loaded == cfg
    assert dump_config(loaded) == dump_config(cfg)


def test_file_overrides_defaults_and_cli_overrides_file(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        "[pipeline]\nthreshold_exclude_um = 2.0\nhalf_size_um = 15\n"
        "[spindle3d]\nspindle_fragment_inclusion_zone = 5\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.pipeline.threshold_exclude_um == 2.0
    assert cfg.pipeline.half_size_um == 15.0  # int in the file -> float
    assert cfg.spindle3d.spindle_fragment_inclusion_zone == 5.0
    assert cfg.pipeline.plate_flat_max == SpindleParams().plate_flat_max
    cfg = load_config(path, {"pipeline": {"threshold_exclude_um": 0.5}})
    assert cfg.pipeline.threshold_exclude_um == 0.5
    assert cfg.pipeline.half_size_um == 15.0


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("[pipeline]\nno_such_key = 1\n", "unknown key"),
        ("[spindle3d]\nthreshold_exclude_um = 1.0\n", "unknown key"),
        ("[other]\nx = 1\n", "unknown config section"),
        ("[pipeline]\ntimepoints_before = 2.5\n", "integer"),
        ("[pipeline]\ncorrection = 1\n", "string"),
        ('[pipeline]\nspacing = [1, "a"]\n', "list of numbers"),
    ],
)
def test_bad_config_raises(tmp_path, text, match):
    path = tmp_path / "c.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_all_settings_reach_spindle3d(monkeypatch, solution_tracks_3d):
    seen = {}

    def fake_measure(tub, dna, spacing, st, keep=False, threshold_exclude_um=0.0):
        seen["st"] = st
        seen["excl"] = threshold_exclude_um
        raise RuntimeError("stop")

    monkeypatch.setattr(core.s3d, "measure", fake_measure)
    shape = tuple(solution_tracks_3d.segmentation.shape)
    img = np.random.default_rng(0).normal(100, 5, shape).astype(np.float32)
    settings = JavaSettings(
        spindle_fragment_inclusion_zone=4.5,
        voxel_size_for_analysis=0.5,
        minimal_dynamic_range=3.0,
        pole_profile_radius="um",
    )
    params = SpindleParams(spacing=(1.0, 1.0, 1.0), half_size_um=12.0, z_extra_um=0.0)
    rows = list(core.measure_tracks(solution_tracks_3d, img, img, params, settings))
    assert seen["st"] == settings
    assert seen["excl"] == params.threshold_exclude_um
    assert rows[0][0]["spindle_status"] == "failed: RuntimeError: stop"
    # correction "none" switches the exclusion off
    list(
        core.measure_tracks(
            solution_tracks_3d,
            img,
            img,
            dataclasses.replace(params, correction="none"),
            settings,
        )
    )
    assert seen["excl"] == 0.0


def test_analysis_voxel_size_changes_the_result():
    from test_spindle3d import SPACING, phantom

    tub, dna = phantom()
    out = {}
    for vs in (0.25, 0.4):
        job = {
            "tub": tub,
            "dna": dna,
            "params": SpindleParams(spacing=SPACING).to_dict(),
            "settings": dataclasses.asdict(JavaSettings(voxel_size_for_analysis=vs)),
            "crop_lo_px": [0, 0, 0],
        }
        out[vs] = measure_crop(job)
    assert out[0.25]["status"] == out[0.4]["status"] == "ok"
    assert out[0.4]["qc"]["iso_voxel_um"] == 0.4
    assert out[0.4]["qc"]["iso_shape"] != out[0.25]["qc"]["iso_shape"]
    assert (
        out[0.4]["features"]["spindle_length_um"]
        != out[0.25]["features"]["spindle_length_um"]
    )


def test_script_writes_the_default_config(tmp_path):
    spec = importlib.util.spec_from_file_location("spindle_features_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = tmp_path / "default.toml"
    mod.main(["--write-default-config", str(path)])
    assert path.read_text(encoding="utf-8") == default_config_path().read_text(
        encoding="utf-8"
    )
