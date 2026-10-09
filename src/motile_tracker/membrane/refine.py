"""Move merged cell boundaries onto the membrane signal, where there is one.

Starting from merged labels (one cell per nucleus), every pair of touching
cells is re-split inside a band around their shared boundary, by a seeded
watershed from the parts of both cells outside the band:

- Strong membrane: if a free re-split over the membrane score puts the
  boundary on a clearly brighter ridge (mean score >= `strong` and >=
  `strong_ratio` x the old boundary), it is taken as is.
- Otherwise the re-split runs over the smoothed score plus a weak ridge at
  the old boundary position (`mu`), so the boundary moves only where a
  membrane is stronger than that ridge and stays put where the band is flat;
  it is kept unless its mean score is lower than the old boundary's.

Labels outside the band are never touched, so where the labels were already
right, or the membrane signal is too weak, the result stays the merged one.

Optional Voronoi correction (`voronoi`), before the refinement: the embryo is
divided among the nuclei by distance from their surfaces (a Voronoi partition
in um, in which a larger nucleus gets a larger region). Gaps the segmentation
left take the cell of the nearest nucleus instead of the nearest cell, and a
cell part lying more than `voronoi_margin_um` beyond its own nucleus' Voronoi
region goes to the Voronoi neighbour. Within the margin the segmentation is
kept, and the refinement still moves the boundaries onto the membrane.

Also here: dropping disconnected pieces of a cell, the cavity from blank raw
signal, and the per-cell boundary signal used for QC. Lengths are in um.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from .merge import contact_areas, fill_within


@dataclass
class RefineParams:
    """Parameters of the boundary refinement and the raw-image cavity."""

    band_frac: float = 0.3
    """Band half-width as a fraction of the smaller cell's radius."""
    band_min_um: float = 1.0
    band_max_um: float = 15.0
    mu: float = 0.03
    """Height of the ridge at the old boundary (membrane score units)."""
    ridge_sigma_um: float = 1.0
    smooth_um: float = 0.5
    """Smoothing of the score for the local re-split."""
    strong: float = 0.15
    """Minimum mean boundary score for taking a free re-split as is."""
    strong_ratio: float = 2.0
    cavity_share: float = 0.5
    """A voxel is cavity when more than this share of its surroundings is blank."""
    cavity_smooth_um: float = 1.5
    cavity_depth_um: float = 4.0
    """Cavity must lie this far inside the embryo surface (excludes the rim)."""
    cavity_min_um3: float = 5000.0
    voronoi: bool = False
    """Correct the cells with a Voronoi partition from the nuclei (see above);
    off, gaps take the nearest cell."""
    voronoi_margin_um: float = 3.0
    """How far (um) a cell may reach beyond its nucleus' Voronoi region."""


def _faces(lab: np.ndarray, score: np.ndarray, a: int, b: int) -> np.ndarray:
    """Score on the voxel faces between labels a and b (max of both voxels)."""
    vals = []
    for ax in range(lab.ndim):
        x = np.moveaxis(lab, ax, 0)
        s = np.moveaxis(score, ax, 0)
        m = ((x[:-1] == a) & (x[1:] == b)) | ((x[:-1] == b) & (x[1:] == a))
        vals.append(np.maximum(s[:-1][m], s[1:][m]))
    return np.concatenate(vals) if vals else np.zeros(0)


def _mean_faces(lab, score, a, b) -> float:
    v = _faces(lab, score, a, b)
    return float(v.mean()) if v.size else 0.0


def _interface_boxes(labels: np.ndarray) -> dict[tuple[int, int], tuple]:
    """Bounding box (lo, hi inclusive per axis) of each touching pair's interface.

    One pass over the volume: for every axis, the voxels on both sides of a
    face between two different non-zero labels are collected with their pair
    code, and the per-pair minimum and maximum coordinates are taken.
    """
    k = int(labels.max()) + 1
    codes, coords = [], []
    for ax in range(labels.ndim):
        sl0 = [slice(None)] * labels.ndim
        sl1 = [slice(None)] * labels.ndim
        sl0[ax] = slice(None, -1)
        sl1[ax] = slice(1, None)
        a, b = labels[tuple(sl0)], labels[tuple(sl1)]
        m = (a != b) & (a > 0) & (b > 0)
        if not m.any():
            continue
        idx = np.nonzero(m)
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        code = lo * k + hi
        pos = np.stack(idx, axis=1)
        nxt = pos.copy()
        nxt[:, ax] += 1
        codes += [code, code]
        coords += [pos, nxt]
    if not codes:
        return {}
    code = np.concatenate(codes)
    pos = np.concatenate(coords)
    order = np.argsort(code, kind="stable")
    code, pos = code[order], pos[order]
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    mins = np.minimum.reduceat(pos, starts, axis=0)
    maxs = np.maximum.reduceat(pos, starts, axis=0)
    return {
        (int(c // k), int(c % k)): (mn, mx)
        for c, mn, mx in zip(code[starts], mins, maxs, strict=True)
    }


def refine_boundaries(
    labels: np.ndarray,
    score: np.ndarray,
    nuclei: np.ndarray,
    spacing,
    params: RefineParams | None = None,
) -> tuple[np.ndarray, dict]:
    """Re-place the boundaries between touching cells on the membrane score.

    Args:
        labels: merged labels (values are nucleus node ids).
        score: membrane score in [0, 1] of the same region.
        nuclei: nuclei labels of the same region; a cell's own nucleus always
            stays in it.
        spacing: voxel size (z, y, x) in um.

    Returns:
        The refined labels and counts: pairs, moved, strong, changed_voxels.
    """
    params = params or RefineParams()
    spacing = tuple(float(s) for s in spacing)
    labels = labels.copy()
    voxel = float(np.prod(spacing))
    pairs, _ = contact_areas(labels, spacing)
    boxes = ndimage.find_objects(labels)
    iface_boxes = _interface_boxes(labels)
    vols = np.bincount(labels.ravel()).astype(float)
    smooth = ndimage.gaussian_filter(score, [params.smooth_um / s for s in spacing])
    st = {"pairs": 0, "moved": 0, "strong": 0, "changed_voxels": 0}
    for (a, b), _ in sorted(pairs.items(), key=lambda kv: -kv[1]):
        ba, bb = boxes[a - 1], boxes[b - 1]
        if ba is None or bb is None:
            continue
        st["pairs"] += 1
        radius = min((3 * vols[i] * voxel / (4 * np.pi)) ** (1 / 3) for i in (a, b))
        width = float(
            np.clip(params.band_frac * radius, params.band_min_um, params.band_max_um)
        )
        # Work only around the shared boundary: its bounding box grown by the
        # band width (plus a margin, as earlier pairs may have shifted it a
        # little). The band, the seeds next to it and the flood inside it all
        # lie there; the rest of both cells cannot change.
        lo, hi = iface_boxes.get((a, b), iface_boxes.get((b, a), (None, None)))
        if lo is None:
            continue
        box = tuple(
            slice(
                max(int(l_) - int(np.ceil(width / sp)) - 3, 0),
                min(int(h_) + int(np.ceil(width / sp)) + 4, n),
            )
            for l_, h_, sp, n in zip(lo, hi, spacing, labels.shape, strict=True)
        )
        sub = labels[box]
        in_a, in_b = sub == a, sub == b
        both = in_a | in_b
        old = in_a.astype(np.uint8) + 2 * in_b.astype(np.uint8)
        iface = (in_a & ndimage.binary_dilation(in_b)) | (
            in_b & ndimage.binary_dilation(in_a)
        )
        if not iface.any():
            continue
        d_old = ndimage.distance_transform_edt(~iface, sampling=spacing)
        band = both & (d_old <= width)
        markers = np.zeros(sub.shape, np.int32)
        markers[in_a & ~band] = 1
        markers[in_b & ~band] = 2
        n_sub = nuclei[box]
        markers[(n_sub == a) & in_a] = 1
        markers[(n_sub == b) & in_b] = 2
        if not (markers == 1).any() or not (markers == 2).any():
            continue
        s_sub = score[box]
        old_m = _mean_faces(old, s_sub, 1, 2)
        # Only the band changes; the flood into it starts from the seeds
        # touching it. Restricting the watershed to the band plus that shell
        # gives the same result without walking through both cell interiors.
        work = both & (band | ndimage.binary_dilation(band))
        free = np.where(both, old, 0).astype(np.uint8)
        free[work] = watershed(s_sub, markers=markers * work, mask=work)[work]
        free_m = _mean_faces(free, s_sub, 1, 2)
        if free_m >= params.strong and free_m >= params.strong_ratio * old_m:
            new = free
            st["strong"] += 1
        else:
            land = smooth[box] + params.mu * np.exp(
                -((d_old / params.ridge_sigma_um) ** 2)
            )
            new = np.where(both, old, 0).astype(np.uint8)
            new[work] = watershed(land, markers=markers * work, mask=work)[work]
            if _mean_faces(new, s_sub, 1, 2) < old_m:
                continue
        changed = (new != old) & both
        if changed.any():
            sub[changed & (new == 1)] = a
            sub[changed & (new == 2)] = b
            st["moved"] += 1
            st["changed_voxels"] += int(changed.sum())
    return labels, st


def voronoi_correct(
    labels: np.ndarray,
    nuclei: np.ndarray,
    inside: np.ndarray,
    spacing,
    margin_um: float,
) -> tuple[np.ndarray, int]:
    """Correct cells towards the Voronoi partition of their nuclei.

    Every voxel's Voronoi owner is the nucleus whose surface is nearest (um),
    among the nuclei that have a cell. A cell keeps its voxels unless they lie
    more than `margin_um` closer to another nucleus than to its own; those go
    to their Voronoi owner, and so do the empty voxels in `inside`.

    Args:
        labels: cell labels; a cell's value is the node id of its nucleus.
        nuclei: nuclei labels of the same region.
        inside: where empty voxels are filled (the embryo without cavities).
        spacing: voxel size (z, y, x) in um.
        margin_um: tolerated reach of a cell beyond its Voronoi region.

    Returns:
        The corrected labels and the number of cell voxels reassigned.
    """
    ids = np.unique(labels)
    ids = ids[ids > 0]
    seeds = np.where(np.isin(nuclei, ids), nuclei, 0)
    if not seeds.any():
        return labels, 0
    d_min, idx = ndimage.distance_transform_edt(
        seeds == 0, sampling=spacing, return_indices=True
    )
    owner = seeds[tuple(idx)]
    del idx
    out = labels.copy()
    moved = 0
    cell_boxes = ndimage.find_objects(labels)
    nuc_boxes = ndimage.find_objects(seeds)
    for n in ids:
        cb = cell_boxes[n - 1]
        nb = nuc_boxes[n - 1] if n <= len(nuc_boxes) else None
        if cb is None or nb is None:
            continue
        # The cell and its nucleus lie in the union box, so the distance to
        # the nucleus computed there is exact.
        box = tuple(
            slice(min(a.start, b.start), max(a.stop, b.stop))
            for a, b in zip(cb, nb, strict=True)
        )
        d_own = ndimage.distance_transform_edt(seeds[box] != n, sampling=spacing)
        off = (labels[box] == n) & (owner[box] != n) & (d_own - d_min[box] > margin_um)
        if off.any():
            out[box][off] = owner[box][off]
            moved += int(off.sum())
    gaps = inside & (out == 0)
    out[gaps] = owner[gaps]
    return out, moved


def drop_islands(labels: np.ndarray, spacing) -> tuple[np.ndarray, int]:
    """Keep each label's largest connected piece; give the rest to the nearest cell."""
    out = labels.copy()
    removed = 0
    for lab_id, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        m = labels[box] == lab_id
        comp, n = ndimage.label(m)
        if n <= 1:
            continue
        sizes = np.bincount(comp.ravel())[1:]
        small = m & (comp != (sizes.argmax() + 1))
        out[box][small] = 0
        removed += int(small.sum())
    if removed:
        out = fill_within(out, labels > 0, spacing)
    return out, removed


def raw_cavity(
    raw: np.ndarray,
    nuclei: np.ndarray,
    spacing,
    embryo: np.ndarray,
    params: RefineParams | None = None,
) -> np.ndarray:
    """A cavity (e.g. the blastocoel) as the blank inside of the embryo.

    The deconvolved raw image is exactly 0 in the cavity, but mixed with tiny
    values, so a voxel counts when more than `cavity_share` of its smoothed
    surroundings is blank. Voxels within `cavity_depth_um` of the embryo
    surface (per slice) are excluded, which removes the blank speckle of the
    outer rim, and only regions above `cavity_min_um3` are kept.
    """
    params = params or RefineParams()
    spacing = tuple(float(s) for s in spacing)
    depth = np.zeros(raw.shape, np.float32)
    for z in range(raw.shape[0]):
        if embryo[z].any():
            depth[z] = ndimage.distance_transform_edt(embryo[z], sampling=spacing[1:])
    share = ndimage.gaussian_filter(
        (raw <= 0).astype(np.float32), [params.cavity_smooth_um / s for s in spacing]
    )
    dark = (
        embryo
        & (depth > params.cavity_depth_um)
        & (share > params.cavity_share)
        & (nuclei == 0)
    )
    comp, _ = ndimage.label(dark)
    sizes = np.bincount(comp.ravel()) * float(np.prod(spacing))
    sizes[0] = 0
    return np.isin(comp, np.flatnonzero(sizes > params.cavity_min_um3))


def boundary_signal(labels: np.ndarray, score: np.ndarray) -> dict[int, float]:
    """Mean membrane score on each cell's faces with other cells.

    Low values mean the cell's boundaries are guesses rather than placed on a
    visible membrane. Cells without neighbours get NaN.
    """
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for ax in range(labels.ndim):
        x = np.moveaxis(labels, ax, 0)
        s = np.moveaxis(score, ax, 0)
        a, b = x[:-1], x[1:]
        m = (a != b) & (a > 0) & (b > 0)
        v = np.maximum(s[:-1][m], s[1:][m]).astype(np.float64)
        for side in (a[m], b[m]):
            sums_side = np.bincount(side, weights=v)
            cnt_side = np.bincount(side)
            for i in np.flatnonzero(cnt_side):
                sums[int(i)] = sums.get(int(i), 0.0) + float(sums_side[i])
                counts[int(i)] = counts.get(int(i), 0) + int(cnt_side[i])
    ids = [int(i) for i in np.unique(labels) if i]
    return {i: sums[i] / counts[i] if counts.get(i) else float("nan") for i in ids}
