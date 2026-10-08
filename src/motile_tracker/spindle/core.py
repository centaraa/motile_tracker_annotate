"""Spindle measurement of tracked nuclei shortly before they divide.

Per candidate node (a node with two successors, plus up to `timepoints_before - 1`
predecessors on its track):

1. crop a box (um) centred on the nucleus from the microtubule (488) and DNA
   (561) channels; outside the volume the crop is padded;
2. subtract the background of both channels per crop; in the DNA channel set
   the other nuclei to 0 (the microtubule channel is not modified);
3. classify the mitotic stage from the chromatin shape (`classify_stage`);
4. run the Java-faithful Spindle3D measurement (`spindle3d.measure`) with the
   chromatin + `threshold_exclude_um` kept out of the spindle threshold;
5. detect centrosome/aster puncta (never required) and compute the axis,
   centre and QC features in um of the original frame.

Every candidate is measured; the stage is a classification, not a gate.
"""

from __future__ import annotations

import dataclasses
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as ndi

from . import spindle3d as s3d

if TYPE_CHECKING:
    from funtracks.data_model import Tracks

ISO = 0.25  # um, default Spindle3D voxelSizeForAnalysis


@dataclass
class SpindleParams:
    """Parameters of the spindle measurement (lengths in um)."""

    spacing: tuple = ()  # voxel size (z, y, x); () = from the tracks
    max_frame: int = -1  # only divisions up to this frame; -1 = all
    timepoints_before: int = 6  # frames per division
    half_size_um: float = 20.0  # half box size in y/x
    z_extra_um: float = 4.0  # extra half size in z, padded beyond the volume
    other_dilate_um: float = 1.0  # other nuclei grown by this before zeroing
    # "ignore": 488 used as recorded; the DNA mask grown by threshold_exclude_um
    # is kept out of the spindle threshold. "none": plain Java threshold rim.
    correction: str = "ignore"
    threshold_exclude_um: float = 1.0
    # stage classification
    psf_sigma_z_um: float = 0.0
    plate_flat_max: float = 0.40
    plate_disc_min: float = 0.60
    split_min_frac: float = 0.20
    plate_rim_min_ratio: float = 0.5
    plate_width_max_um: float = 4.0
    min_mass_um3: float = 5.0
    # centrosome detection (stored separately, does not change the mask)
    centrosome_contrast_min: float = 2.0
    centrosome_lateral_um: float = 4.0
    centrosome_beyond_um: float = 6.0
    # QC flags
    axis_vs_plate_flag_deg: float = 10.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spacing"] = list(d["spacing"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SpindleParams:
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "spacing" in d:
            d["spacing"] = tuple(d["spacing"])
        return cls(**d)


# Spindle3D result key -> feature key
JAVA_KEYS = {
    "spindle_length_um": "spindle_length_um",
    "spindle_volume_um3": "spindle_volume_um3",
    "spindle_width_avg_um": "spindle_width_avg_um",
    "spindle_width_min_um": "spindle_width_min_um",
    "spindle_width_max_um": "spindle_width_max_um",
    "spindle_aspect_ratio": "spindle_aspect_ratio",
    "spindle_angle_deg": "spindle_angle_deg",
    "spindle_axis_dna_offset_deg": "spindle_axis_dna_offset_deg",
    "metaphase_plate_width_um": "spindle_plate_width_um",
    "metaphase_plate_length_um": "spindle_plate_length_um",
    "chromatin_volume_um3": "spindle_chromatin_volume_um3",
}


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
def find_candidates(
    tracks: Tracks, max_frame: int | None, n_before: int
) -> list[tuple[int, int, int, int]]:
    """(node, t, frames_before_division, division_node), sorted by time.

    A division node is a node with two successors and time <= max_frame (all if
    None); it is frame 1 before the division, its single-predecessor chain gives
    frames 2..n_before (the chain stops at another division or a merge).
    """
    out = []
    for n in tracks.graph_full.node_ids():
        if len(list(tracks.successors(n))) != 2:
            continue
        t = tracks.get_time(n)
        if max_frame is not None and t > max_frame:
            continue
        chain, cur = [n], n
        while len(chain) < n_before:
            preds = list(tracks.predecessors(cur))
            if len(preds) != 1 or len(list(tracks.successors(preds[0]))) != 1:
                break
            cur = preds[0]
            chain.append(cur)
        for i, c in enumerate(chain):
            out.append((int(c), int(tracks.get_time(c)), i + 1, int(n)))
    out.sort(key=lambda r: (r[1], r[0]))
    return out


# ---------------------------------------------------------------------------
# crop and background
# ---------------------------------------------------------------------------
def padded_crop(arr, lo, hi, fill):
    """arr[lo:hi] with the parts outside the array set to `fill`."""
    out_shape = tuple(int(h - lo_) for lo_, h in zip(lo, hi, strict=True))
    dtype = np.float32 if isinstance(fill, float) else arr.dtype
    out = np.full(out_shape, fill, dtype=dtype)
    src = tuple(
        slice(max(lo_, 0), min(h, n))
        for lo_, h, n in zip(lo, hi, arr.shape, strict=True)
    )
    dst = tuple(
        slice(s.start - lo_, s.stop - lo_) for s, lo_ in zip(src, lo, strict=True)
    )
    out[dst] = arr[src]
    return out


def extract_crop(node, t, a_frame, d_frame, seg_frame, bbox, p: SpindleParams) -> dict:
    """Raw crop of both channels (NaN outside the volume) and of the labels."""
    sp = np.asarray(p.spacing, float)
    z0, y0, x0, z1, y1, x1 = (int(v) for v in bbox)
    sub = seg_frame[z0:z1, y0:y1, x0:x1] == node
    if not sub.any():
        raise RuntimeError("node label not found in segmentation")
    cen = np.argwhere(sub).mean(0) + np.array([z0, y0, x0])
    half_um = np.array([p.half_size_um + p.z_extra_um, p.half_size_um, p.half_size_um])
    half_px = np.ceil(half_um / sp).astype(int)
    c = np.round(cen).astype(int)
    lo, hi = c - half_px, c + half_px + 1
    return {
        "node": int(node),
        "t": int(t),
        "cen": cen,
        "lo": lo,
        "padded": bool(np.any(lo < 0) or np.any(hi > np.array(a_frame.shape))),
        "a": padded_crop(a_frame, lo, hi, np.nan),
        "d": padded_crop(d_frame, lo, hi, np.nan),
        "seg": padded_crop(seg_frame, lo, hi, 0).astype(np.int32),
    }


def build_case(raw: dict, p: SpindleParams) -> dict:
    """Per-crop background subtraction.

    Background = median of the voxels more than 2 um from the target nucleus and
    outside the other nuclei. The microtubule channel is the background-subtracted
    488, otherwise unchanged; in the DNA channel the other nuclei (grown by
    other_dilate_um) are set to 0.
    """
    if p.correction not in ("ignore", "none"):
        raise ValueError(f"unknown correction {p.correction!r} (use ignore or none)")
    sp = np.asarray(p.spacing, float)
    node = raw["node"]
    a, d, seg = raw["a"], raw["d"], raw["seg"]
    valid = np.isfinite(a)
    target = seg == node
    others = (seg > 0) & ~target
    if others.any():
        near_other = ndi.distance_transform_edt(~others, sampling=sp)
        others = (near_other <= p.other_dilate_um) & ~target
    dist_t = ndi.distance_transform_edt(~target, sampling=sp)
    bgsel = valid & ~others & (dist_t > 2.0)
    bg_a = float(np.median(a[bgsel])) if bgsel.any() else float(np.nanmin(a))
    bg_d = float(np.median(d[bgsel])) if bgsel.any() else float(np.nanmin(d))
    a_b = np.where(valid, a - bg_a, 0).astype(np.float32)
    d_b = np.where(valid, d - bg_d, 0).astype(np.float32)
    dna = np.clip(d_b, 0, None).astype(np.float32)
    dna[others] = 0.0
    tub = np.clip(a_b, 0, None).astype(np.float32)
    meta = {
        "node": int(node),
        "t": int(raw["t"]),
        "centroid_px": raw["cen"].tolist(),
        "crop_lo_px": raw["lo"].tolist(),
        "crop_shape": list(a.shape),
        "padded_beyond_volume": raw["padded"],
        "correction": p.correction,
        "n_other_nuclei": int(len(np.unique(seg[(seg > 0) & ~target]))),
    }
    return {
        "a_b": a_b,
        "tub": tub,
        "dna": dna,
        "seg": seg,
        "target": target,
        "others": others,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# stage classification
# ---------------------------------------------------------------------------
def chromatin_mask(dna, target, sp):
    """561 > Otsu over the voxels within 2 um of the target nucleus mask."""
    from skimage.filters import threshold_otsu

    near = ndi.distance_transform_edt(~target, sampling=sp) <= 2.0
    vals = dna[near]
    thr = float(threshold_otsu(vals)) if vals.size and np.ptp(vals) > 0 else np.inf
    return (dna > thr) & near, thr


def stage_metrics(case: dict, p: SpindleParams) -> tuple[dict, np.ndarray]:
    """Chromatin shape metrics from the 561 crop and the largest chromatin mass.

    mass2_frac: second / largest chromatin mass; dna_axes_um: +-2 sd lengths of
    the largest mass along its PCA axes (longest first); flatness = shortest /
    longest; disc = middle / longest.
    """
    from skimage.measure import label

    sp = np.asarray(p.spacing, float)
    m, thr = chromatin_mask(case["dna"], case["target"], sp)
    lab = label(m, connectivity=3)
    sizes = np.bincount(lab.ravel())[1:] * float(np.prod(sp))
    res = {
        "stage_threshold": thr,
        "n_masses": 0,
        "mass2_frac": np.nan,
        "flatness": np.nan,
        "disc": np.nan,
        "dna_axes_um": [np.nan] * 3,
        "plate_normal_z": np.nan,
    }
    if sizes.size == 0:
        return res, m
    order = np.argsort(sizes)[::-1]
    res["n_masses"] = int(sum(s >= p.min_mass_um3 for s in sizes))
    res["mass2_frac"] = (
        float(sizes[order[1]] / sizes[order[0]]) if sizes.size > 1 else 0.0
    )
    largest = lab == order[0] + 1
    coords = np.argwhere(largest) * sp
    if coords.shape[0] < 4:
        return res, largest
    w, V = np.linalg.eigh(np.cov(coords, rowvar=False))
    order3 = np.argsort(w)[::-1]
    ev = np.clip(w[order3], 0, None).copy()
    nz = float(abs(V[:, order3[2]][0]))
    ev[2] = max(ev[2] - (nz * p.psf_sigma_z_um) ** 2, (0.25 * sp[1]) ** 2)
    L = 4.0 * np.sqrt(ev)
    res.update(
        dna_axes_um=L.tolist(),
        plate_normal_z=nz,
        flatness=float(L[2] / L[0]) if L[0] > 0 else np.nan,
        disc=float(L[1] / L[0]) if L[0] > 0 else np.nan,
    )
    return res, largest


def classify_stage(
    stage: dict,
    plate_length_um: float,
    p: SpindleParams,
    plate_width_um: float = np.nan,
) -> tuple[str, str, int, float]:
    """Stage class; the first matching rule decides.

    split: second chromatin mass >= split_min_frac of the largest;
    prometaphase: not flat (shortest/longest > plate_flat_max), or no plate rim
    (Spindle3D plate length < plate_rim_min_ratio x chromatin extent, i.e. the
    radial DNA profile has no edge at the rim, as in a lobed rosette), or a thick
    plate (Spindle3D plate width > plate_width_max_um);
    unclear: not a disc (middle/longest < plate_disc_min); metaphase otherwise.

    Returns (stage, rule, is_metaphase (1/0), plate rim ratio).
    """
    ext = stage["dna_axes_um"][0] if stage.get("dna_axes_um") else np.nan
    rim = (
        plate_length_um / ext
        if np.isfinite(plate_length_um) and np.isfinite(ext) and ext > 0
        else np.nan
    )
    if stage["mass2_frac"] >= p.split_min_frac:
        return (
            "split",
            f"2nd mass {stage['mass2_frac']:.2f} >= {p.split_min_frac}",
            0,
            rim,
        )
    if not stage["flatness"] <= p.plate_flat_max:
        rule = f"not flat: short/long {stage['flatness']:.2f} > {p.plate_flat_max}"
        return "prometaphase", rule, 0, rim
    if np.isfinite(rim) and rim < p.plate_rim_min_ratio:
        rule = (
            f"no plate rim: plate length {plate_length_um:.1f} um = {rim:.2f} x "
            f"chromatin extent {ext:.1f} um < {p.plate_rim_min_ratio}"
        )
        return "prometaphase", rule, 0, rim
    if np.isfinite(plate_width_um) and plate_width_um > p.plate_width_max_um:
        rule = (
            f"thick plate: plate width {plate_width_um:.2f} um > {p.plate_width_max_um}"
        )
        return "prometaphase", rule, 0, rim
    if not stage["disc"] >= p.plate_disc_min:
        return (
            "unclear",
            f"not a disc: mid/long {stage['disc']:.2f} < {p.plate_disc_min}",
            0,
            rim,
        )
    rule = "flat disc" + ("" if np.isfinite(rim) else " (plate rim not tested)")
    return "metaphase", rule, 1, rim


# ---------------------------------------------------------------------------
# centrosomes (detected separately; the spindle mask is not changed)
# ---------------------------------------------------------------------------
def detect_centrosomes(
    tub, mask, center, vs, p: SpindleParams, half_plate_um, peak_min
):
    """Compact bright puncta at the axial ends of the spindle.

    Works in the DNA-aligned frame (z = plate normal through the chromatin centre
    `center`, index units). Candidates are local maxima of the tubulin blurred
    with sigma 0.4 um, within centrosome_lateral_um of the axis, beyond 60 % of
    the mask's axial extent on that side and at most centrosome_beyond_um past
    its end, with peak >= peak_min and peak / median of a 1.5-3 um shell >=
    centrosome_contrast_min. The best candidate per side is returned.
    """
    sm = ndi.gaussian_filter(tub.astype(np.float32), 0.4 / vs)
    w = max(int(round(1.5 / vs)), 3)
    lm = (sm == ndi.maximum_filter(sm, size=w)) & (sm >= peak_min)
    cz, cy, cx = center
    zz, yy, xx = np.nonzero(mask)
    sel = np.hypot(yy - cy, xx - cx) * vs <= 3.0
    ext = {-1: 0.0, 1: 0.0}
    if sel.any():
        ext[-1] = max((cz - zz[sel].min()) * vs, 0.0)
        ext[1] = max((zz[sel].max() - cz) * vs, 0.0)
    r_in, r_out = int(round(1.5 / vs)), int(round(3.0 / vs))
    oz, oy, ox = np.mgrid[-r_out : r_out + 1, -r_out : r_out + 1, -r_out : r_out + 1]
    rr = np.sqrt(oz**2 + oy**2 + ox**2)
    shell = (rr >= r_in) & (rr <= r_out)
    best: dict[int, dict] = {}
    for q in np.argwhere(lm):
        ax = (q[0] - cz) * vs
        side = 1 if ax > 0 else -1
        lat = float(np.hypot(q[1] - cy, q[2] - cx) * vs)
        a = abs(ax)
        if lat > p.centrosome_lateral_um:
            continue
        if (
            a < max(0.6 * ext[side], half_plate_um + 1.5)
            or a > ext[side] + p.centrosome_beyond_um
        ):
            continue
        lo, hi = q - r_out, q + r_out + 1
        if np.any(lo < 0) or np.any(hi > np.array(tub.shape)):
            continue
        bg = float(np.median(tub[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]][shell]))
        peak = float(sm[tuple(q)])
        contrast = peak / max(bg, 1e-3)
        if contrast >= p.centrosome_contrast_min and (
            side not in best or contrast > best[side]["contrast"]
        ):
            best[side] = {
                "pos": q.astype(float),
                "contrast": contrast,
                "peak": peak,
                "in_mask": bool(mask[tuple(q)]),
            }
    return list(best.values())


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def unit_sign(v) -> np.ndarray:
    """Unit vector with the sign convention z >= 0, else y >= 0, else x >= 0."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        return np.full(3, np.nan)
    v = v / n
    for i in range(3):
        if abs(v[i]) > 1e-9:
            return -v if v[i] < 0 else v
    return v


def angle_deg(u, v) -> float:
    """Angle between two axes (sign-free, 0-90 deg)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def iso_to_frame_um(p_iso, crop_lo_px, spacing, vs: float = ISO) -> np.ndarray:
    """Iso crop voxel index -> um of the frame (iso index j <-> j * vs um)."""
    return np.asarray(p_iso, float) * vs + np.asarray(crop_lo_px, float) * np.asarray(
        spacing, float
    )


# ---------------------------------------------------------------------------
# one crop
# ---------------------------------------------------------------------------
def measure_crop(job: dict) -> dict:
    """Spindle3D + centrosomes + features for one crop (worker entry point).

    job: tub, dna (3D arrays), params (SpindleParams as dict, spacing resolved),
    settings (JavaSettings as dict; default Java v0.8.0), crop_lo_px. Returns
    status ("ok" or "failed: <reason>"), features, qc (masks/points in the iso
    crop grid, voxel size qc["iso_voxel_um"]), diag (spindle thresholds) and
    seconds.
    """
    p = SpindleParams.from_dict(job["params"])
    st = s3d.JavaSettings(**job.get("settings", {}))
    vs = st.voxel_size_for_analysis
    excl = p.threshold_exclude_um if p.correction == "ignore" else 0.0
    t0 = time.perf_counter()
    res = {"status": "ok", "traceback": ""}
    feats: dict = {}
    qc: dict = {}
    diag: dict = {}
    try:
        r = s3d.measure(
            job["tub"], job["dna"], p.spacing, st, keep=True, threshold_exclude_um=excl
        )
        for jk, fk in JAVA_KEYS.items():
            feats[fk] = float(r[jk]) if r.get(jk) is not None else np.nan
        sp = np.asarray(p.spacing, float)
        lo = job["crop_lo_px"]
        poles_iso = s3d.a_to_iso_points(r["poles_um"], r)
        P = np.array([iso_to_frame_um(q, lo, sp, vs) for q in poles_iso])
        v = P[1] - P[0]
        u = unit_sign(v)
        mid = P.mean(0)
        feats.update(
            spindle_axis_z=u[0],
            spindle_axis_y=u[1],
            spindle_axis_x=u[2],
            spindle_azimuth_deg=float(np.degrees(np.arctan2(u[1], u[2])) % 180.0),
            spindle_center_z_um=mid[0],
            spindle_center_y_um=mid[1],
            spindle_center_x_um=mid[2],
        )
        # lateral box edge only: the axial inner face is hit systematically
        feats["spindle_flag_pole_on_box_edge"] = int(
            any(e[1] for e in r["pole_on_box_edge"])
        )
        feats["pole_on_axial_box_edge"] = int(any(e[0] for e in r["pole_on_box_edge"]))
        feats["pole_box_offsets_zyx_vox"] = str(r["pole_box_offsets"])
        feats["spindle_flag_axis_vs_plate"] = int(
            r["spindle_axis_dna_offset_deg"] > p.axis_vs_plate_flag_deg
        )
        # centrosomes in the DNA-aligned frame
        o = r["origin_index"]
        cents = detect_centrosomes(
            r["A_tub"],
            r["spindle_mask_A"],
            o,
            vs,
            p,
            0.5 * r["metaphase_plate_width_um"],
            r["spindle_threshold"],
        )
        c_um_A = [(c["pos"] - o) * vs for c in cents]
        c_frame = (
            [iso_to_frame_um(q, lo, sp, vs) for q in s3d.a_to_iso_points(c_um_A, r)]
            if cents
            else []
        )
        # centrosome 1 = nearest pole A, centrosome 2 = nearest pole B
        if len(c_frame) == 2 and np.linalg.norm(c_frame[0] - P[0]) > np.linalg.norm(
            c_frame[1] - P[0]
        ):
            c_frame = c_frame[::-1]
            cents = cents[::-1]
        elif len(c_frame) == 1 and np.linalg.norm(c_frame[0] - P[0]) > np.linalg.norm(
            c_frame[0] - P[1]
        ):
            c_frame = [np.full(3, np.nan), c_frame[0]]
        feats["spindle_n_centrosomes"] = len(cents)
        for i in (1, 2):
            q = c_frame[i - 1] if len(c_frame) >= i else np.full(3, np.nan)
            feats[f"spindle_centrosome{i}_z_um"] = q[0]
            feats[f"spindle_centrosome{i}_y_um"] = q[1]
            feats[f"spindle_centrosome{i}_x_um"] = q[2]
        if len(cents) == 2:
            cv = c_frame[1] - c_frame[0]
            cu = unit_sign(cv)
            feats.update(
                spindle_centrosome_distance_um=float(np.linalg.norm(cv)),
                spindle_centrosome_axis_z=cu[0],
                spindle_centrosome_axis_y=cu[1],
                spindle_centrosome_axis_x=cu[2],
                spindle_axis_vs_centrosome_deg=angle_deg(v, cv),
            )
        feats["centrosome_in_mask"] = ";".join(str(int(c["in_mask"])) for c in cents)
        # continuity of the spindle mask through the plate (A frame)
        sm = r["spindle_mask_A"]
        hz = max(int(round(0.5 * r["metaphase_plate_width_um"] / vs)), 1)
        zsl = slice(max(o[0] - hz, 0), o[0] + hz + 1)
        yy, xx = np.indices(sm.shape[1:])
        core_disc = np.hypot(yy - o[1], xx - o[2]) * vs <= 1.0
        feats["plate_axis_mask_fraction"] = float(sm[zsl][:, core_disc].mean())
        feats["spindle_mask_components"] = int(ndi.label(sm, structure=s3d.S6)[1])
        feats["n_width_angles"] = r["n_width_angles"]
        # where the refined poles sit relative to the mask tips and centrosomes
        PA = [np.asarray(q, float) / vs + o for q in r["poles_um"]]  # A-frame index
        ua = (PA[1] - PA[0]) / max(np.linalg.norm(PA[1] - PA[0]), 1e-9)
        rel = np.argwhere(sm).astype(float) - PA[0]
        proj = rel @ ua
        latd = np.linalg.norm(rel - np.outer(proj, ua), axis=1) * vs
        near = latd <= 2.0
        if near.any():
            pb = (PA[1] - PA[0]) @ ua
            feats["pole_tip_offset_um"] = (
                f"{-proj[near].min() * vs:.2f};{(proj[near].max() - pb) * vs:.2f}"
            )
        if cents:
            cd = [
                min(
                    np.linalg.norm(pf - cf) for cf in c_frame if np.all(np.isfinite(cf))
                )
                for pf in P
            ]
            feats["pole_to_nearest_centrosome_um"] = f"{cd[0]:.2f};{cd[1]:.2f}"
        diag.update(
            threshold_used=float(r["spindle_threshold"]),
            threshold_java_rim=float(r["threshold_java_rim"]),
        )
        # QC / napari: masks and points in the iso crop grid
        qc["iso_shape"] = r["iso_shape"]
        qc["iso_voxel_um"] = vs
        qc["dna_mask"] = s3d.a_mask_to_iso(r["dna_mask_A"], r)
        qc["spindle_mask"] = s3d.a_mask_to_iso(r["spindle_mask_A"], r)
        qc["poles"] = poles_iso
        qc["init_poles"] = s3d.a_to_iso_points(r["init_poles_um"], r)
        if cents:
            qc["centrosomes"] = s3d.a_to_iso_points(c_um_A, r)
        qc["threshold_exclusion"] = s3d.a_mask_to_iso(r["threshold_exclusion_A"], r)
        qc["threshold_shell"] = s3d.a_mask_to_iso(r["threshold_shell_A"], r)
    except Exception as e:  # noqa: BLE001 - a failed node is recorded, not fatal
        msg = str(e).replace("→", "->").strip().splitlines()
        res["status"] = f"failed: {type(e).__name__}: {msg[0][:160] if msg else ''}"
        res["traceback"] = traceback.format_exc()
    res["seconds"] = time.perf_counter() - t0
    res["features"] = feats
    res["qc"] = qc
    res["diag"] = diag
    return res


# ---------------------------------------------------------------------------
# rows (one per candidate node)
# ---------------------------------------------------------------------------
def _base_row(cand, case, stage) -> dict:
    from .features import FEATURE_DEFAULTS

    node, t, fb, dnode = cand
    row = {"node": node, "t": t, "division_node": dnode}
    row.update(FEATURE_DEFAULTS)
    row["spindle_frames_before_division"] = fb
    if case is not None:
        m = case["meta"]
        row.update(
            correction=m["correction"],
            padded_beyond_volume=m["padded_beyond_volume"],
            n_other_nuclei=m["n_other_nuclei"],
        )
    if stage is not None:
        row.update(
            spindle_plate_flatness=stage["flatness"],
            spindle_plate_disc=stage["disc"],
            spindle_split_ratio=stage["mass2_frac"],
            plate_normal_z=stage.get("plate_normal_z", np.nan),
            dna_axes_um=";".join(f"{v:.1f}" for v in stage["dna_axes_um"]),
        )
    return row


def _finish_row(row: dict, case: dict, res: dict, p: SpindleParams) -> dict:
    row["spindle_status"] = res["status"]
    row.update(res["features"])
    row["seconds"] = round(res["seconds"], 2)
    ok = res["status"] == "ok"
    pl = res["features"].get("spindle_plate_length_um", np.nan) if ok else np.nan
    pw = res["features"].get("spindle_plate_width_um", np.nan) if ok else np.nan
    stg, rule, ismeta, rim = classify_stage(case["stage"], pl, p, pw)
    row.update(
        spindle_stage=stg,
        spindle_stage_rule=rule,
        spindle_is_metaphase=ismeta,
        spindle_plate_rim_ratio=rim,
    )
    return row


def prepare_case(node, t, a_frame, d_frame, seg_frame, bbox, p: SpindleParams) -> dict:
    """Crop, background and stage metrics of one node."""
    raw = extract_crop(node, t, a_frame, d_frame, seg_frame, bbox, p)
    case = build_case(raw, p)
    case["stage"], case["stage_mask"] = stage_metrics(case, p)
    return case


def resolve_spacing(params: SpindleParams, tracks: Tracks) -> SpindleParams:
    """params with an empty spacing replaced by the voxel size of the tracks."""
    if params.spacing:
        return params
    scale = getattr(tracks, "scale", None)
    if not scale:
        raise ValueError(
            "no voxel size: set [pipeline] spacing or store it with the tracks"
        )
    return dataclasses.replace(params, spacing=tuple(float(v) for v in scale[1:]))


def measure_node(
    tracks: Tracks,
    node: int,
    tubulin,
    dna,
    p: SpindleParams | None = None,
    settings: s3d.JavaSettings | None = None,
):
    """Measure one node (e.g. for the viewer). Returns (row, case, result).

    tubulin, dna: lazy (t, z, y, x) stacks indexable by frame, e.g. from
    `motile_tracker.membrane.io.read_label_file`.
    """
    p = resolve_spacing(p or SpindleParams(), tracks)
    settings = settings or s3d.JavaSettings()
    t = int(tracks.get_time(node))
    seg = np.asarray(tracks.segmentation[t])
    case = prepare_case(
        node,
        t,
        np.asarray(tubulin[t]),
        np.asarray(dna[t]),
        seg,
        tracks.get_node_attr(node, "bbox"),
        p,
    )
    res = measure_crop(
        {
            "tub": case["tub"],
            "dna": case["dna"],
            "params": p.to_dict(),
            "settings": dataclasses.asdict(settings),
            "crop_lo_px": case["meta"]["crop_lo_px"],
        }
    )
    cand = (int(node), t, 0, int(node))
    return _finish_row(_base_row(cand, case, case["stage"]), case, res, p), case, res


def measure_tracks(
    tracks: Tracks,
    tubulin,
    dna,
    params: SpindleParams | None = None,
    settings: s3d.JavaSettings | None = None,
    max_frame: int | None = None,
    timepoints_before: int | None = None,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[tuple[dict, dict, dict]]:
    """Measure every candidate node; yields (row, case, result) per node.

    Args:
        tracks: Tracks with a nuclei segmentation (node id = label) and bbox.
        tubulin, dna: microtubule and DNA images (t, z, y, x), indexable by
            frame (lazy is fine; frames are read once each).
        params: Pipeline parameters ([pipeline]); an empty spacing is taken
            from the tracks.
        settings: Spindle3D settings ([spindle3d]); default Java v0.8.0.
        max_frame: Only divisions at or before this frame (default
            params.max_frame; -1 or None = all).
        timepoints_before: Frames per division, the division node and its
            predecessors (default params.timepoints_before).
        workers: Crops measured in parallel processes.
        progress: Called with (number done, number of candidates).
    """
    p = params or SpindleParams()
    if p.correction not in ("ignore", "none"):
        raise ValueError(f"unknown correction {p.correction!r} (use ignore or none)")
    p = resolve_spacing(p, tracks)
    settings = settings or s3d.JavaSettings()
    if max_frame is None:
        max_frame = p.max_frame
    if max_frame is not None and max_frame < 0:
        max_frame = None
    if timepoints_before is None:
        timepoints_before = p.timepoints_before
    cands = find_candidates(tracks, max_frame, timepoints_before)
    by_t: dict[int, list] = {}
    for c in cands:
        by_t.setdefault(c[1], []).append(c)
    total = len(cands)
    done = 0
    pdict = p.to_dict()
    sdict = dataclasses.asdict(settings)
    pool = None
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=workers)
    pending: list = []
    try:
        for t in sorted(by_t):
            a_frame = np.asarray(tubulin[t])
            d_frame = np.asarray(dna[t])
            seg_frame = np.asarray(tracks.segmentation[t])
            for cand in by_t[t]:
                node = cand[0]
                try:
                    case = prepare_case(
                        node,
                        t,
                        a_frame,
                        d_frame,
                        seg_frame,
                        tracks.get_node_attr(node, "bbox"),
                        p,
                    )
                except Exception as e:  # noqa: BLE001 - recorded per node
                    row = _base_row(cand, None, None)
                    row["spindle_status"] = f"failed: crop: {e}"
                    done += 1
                    if progress is not None:
                        progress(done, total)
                    yield (
                        row,
                        {},
                        {"status": row["spindle_status"], "qc": {}, "diag": {}},
                    )
                    continue
                job = {
                    "tub": case["tub"],
                    "dna": case["dna"],
                    "params": pdict,
                    "settings": sdict,
                    "crop_lo_px": case["meta"]["crop_lo_px"],
                }
                row = _base_row(cand, case, case["stage"])
                if pool is None:
                    res = measure_crop(job)
                    done += 1
                    if progress is not None:
                        progress(done, total)
                    yield _finish_row(row, case, res, p), case, res
                else:
                    pending.append((row, case, pool.submit(measure_crop, job)))
            del a_frame, d_frame, seg_frame
            still = []
            for row, case, fut in pending:
                if fut.done():
                    done += 1
                    if progress is not None:
                        progress(done, total)
                    res = fut.result()
                    yield _finish_row(row, case, res, p), case, res
                else:
                    still.append((row, case, fut))
            pending = still
        for row, case, fut in pending:
            res = fut.result()
            done += 1
            if progress is not None:
                progress(done, total)
            yield _finish_row(row, case, res, p), case, res
    finally:
        if pool is not None:
            pool.shutdown()
