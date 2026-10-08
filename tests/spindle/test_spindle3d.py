"""Synthetic tests of the Java-faithful Spindle3D re-implementation and the
per-crop measurement (no image data, no GUI)."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from motile_tracker.spindle import SpindleParams, measure_crop
from motile_tracker.spindle import spindle3d as s3d
from motile_tracker.spindle.core import unit_sign

SPACING = (1.0, 0.26, 0.26)


def phantom(half_len=9.0, radius=4.0, plate_r=7.0, plate_t=2.0, shape=(31, 115, 115)):
    """DNA plate (normal along x) inside a bipolar spindle along x.

    Returns (tubulin, dna) with anisotropic voxels SPACING; the spindle is an
    ellipsoid with half-length `half_len` um along x and `radius` um across.
    """
    sp = np.asarray(SPACING)
    c = (np.asarray(shape) - 1) / 2
    z, y, x = (np.indices(shape).T - c).T * sp[:, None, None, None]
    plate = (np.abs(x) <= plate_t / 2) & (np.hypot(z, y) <= plate_r)
    body = (x / half_len) ** 2 + (z / radius) ** 2 + (y / radius) ** 2 <= 1
    rng = np.random.default_rng(0)
    tub = 50 + 600 * body + rng.normal(0, 5, shape)
    dna = 20 + 2000 * plate + rng.normal(0, 5, shape)
    sig = 0.3 / sp
    return (
        ndi.gaussian_filter(tub, sig).astype(np.float32),
        ndi.gaussian_filter(dna, sig).astype(np.float32),
    )


@pytest.fixture(scope="module")
def phantom_result():
    tub, dna = phantom()
    job = {
        "tub": tub,
        "dna": dna,
        "params": SpindleParams(spacing=SPACING).to_dict(),
        "crop_lo_px": [0, 0, 0],
    }
    return measure_crop(job)


def test_phantom_geometry(phantom_result):
    res = phantom_result
    assert res["status"] == "ok", res.get("traceback")
    f = res["features"]
    # poles end about 1 um inside the tips (inner face of the refinement box)
    assert 14.0 < f["spindle_length_um"] < 19.0
    assert 6.0 < f["spindle_width_avg_um"] < 10.5
    assert f["spindle_plate_width_um"] < 4.0
    assert 10.0 < f["spindle_plate_length_um"] < 16.0
    # spindle along x: in the coverslip plane, along the plate normal
    assert f["spindle_angle_deg"] < 10.0
    assert f["spindle_axis_dna_offset_deg"] < 10.0
    assert abs(f["spindle_axis_x"]) > 0.98
    assert f["spindle_azimuth_deg"] < 10.0 or f["spindle_azimuth_deg"] > 170.0
    assert f["spindle_mask_components"] == 1
    assert f["plate_axis_mask_fraction"] == pytest.approx(1.0)


def test_tilt_from_axis_equals_spindle_angle(phantom_result):
    f = phantom_result["features"]
    tilt = np.degrees(np.arcsin(abs(f["spindle_axis_z"])))
    assert tilt == pytest.approx(f["spindle_angle_deg"], abs=1e-6)


def test_threshold_exclusion_lowers_threshold():
    tub, dna = phantom()
    # a DNA leak into the tubulin channel at the plate
    tub = tub + 0.3 * dna
    out = {}
    for corr in ("ignore", "none"):
        job = {
            "tub": tub,
            "dna": dna,
            "params": SpindleParams(spacing=SPACING, correction=corr).to_dict(),
            "crop_lo_px": [0, 0, 0],
        }
        out[corr] = measure_crop(job)["diag"]
    assert out["none"]["threshold_used"] == pytest.approx(
        out["none"]["threshold_java_rim"]
    )
    assert out["ignore"]["threshold_used"] < out["ignore"]["threshold_java_rim"]


def test_failure_is_recorded():
    tub, _ = phantom()
    job = {
        "tub": tub,
        "dna": np.zeros_like(tub),
        "params": SpindleParams(spacing=SPACING).to_dict(),
        "crop_lo_px": [0, 0, 0],
    }
    res = measure_crop(job)
    assert res["status"].startswith("failed: ")
    assert res["features"] == {}


def test_pole_edges_sign_split_with_gap():
    # mask along the axis with a gap around the plate: the left pole must be the
    # outermost rising edge (z < 0), the right pole a falling edge at z > 0
    mask = np.zeros((41, 5, 5), bool)
    mask[5:17, 2, 2] = True  # z = -15 .. -4
    mask[23:36, 2, 2] = True  # z = +3 .. +15
    zl, zr = s3d.pole_edges_along_z(mask, (20, 2, 2), radius=1.0, vs=1.0)
    assert zl < 0 < zr
    # Java's centred 2-voxel derivative puts the first maximum of the rising edge
    # one voxel outside the mask and the first minimum on its last voxel
    assert zl == pytest.approx(-16.0)
    assert zr == pytest.approx(15.0)


def test_central_regions_any_voxel_rule():
    mask = np.zeros((40, 40, 40), bool)
    # long rod whose centroid is far from the centre, one end touching it
    mask[20, 20, 20:40] = True
    # compact blob nowhere near the centre
    mask[2:5, 2:5, 2:5] = True
    out = s3d.central_regions_mask(mask, (20, 20, 20), radius=3)
    assert out[20, 20, 39]
    assert not out[3, 3, 3]


def test_otsu256():
    rng = np.random.default_rng(1)
    vals = np.concatenate([rng.normal(10, 1, 1000), rng.normal(50, 1, 1000)])
    thr = s3d.otsu256(vals)
    # the threshold separates the two modes (first maximum of the between-class
    # variance, i.e. just above the lower mode for well separated modes)
    assert (vals[:1000] < thr).mean() > 0.99
    assert (vals[1000:] > thr).all()
    assert s3d.otsu256(np.full(10, 3.0)) == 3.0


def test_rescale_exact_quarter_micron_grid():
    img = np.tile(np.arange(40, dtype=float), (11, 40, 1))  # x ramp, 0.26 um voxels
    out = s3d.create_rescaled(img, np.asarray(SPACING) / 0.25)
    assert out.shape == (41, 41, 41)
    # output voxel j sits at j * 0.25 um, i.e. input index j * 0.25 / 0.26
    j = np.arange(5, 35)
    np.testing.assert_allclose(out[20, 20, j], j * 0.25 / 0.26, atol=1e-3)


def test_unit_sign_convention():
    np.testing.assert_allclose(unit_sign([-2, 0, 0]), [1, 0, 0])
    np.testing.assert_allclose(
        unit_sign([0, -1, 1]), [0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
    )
    np.testing.assert_allclose(unit_sign([0, 0, -3]), [0, 0, 1])
