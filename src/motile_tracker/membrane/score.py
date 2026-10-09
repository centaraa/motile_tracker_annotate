"""Membrane signal from the raw membrane image, kept apart from the spindle.

The membrane channel also shows microtubules: during mitosis the spindle is the
brightest structure in the cell, so brightness alone pulls boundaries through
it. Two properties separate them:

- A membrane is a sheet; spindle fibres are lines. A Hessian "sheet" filter
  (bright, flat structures high; fibres and blobs low) removes most of the
  spindle, and weighting it by brightness removes the speckle the filter
  picks up in dim cytoplasm.
- Around the chromatin of a dividing cell the spindle core still looks partly
  flat, so the score is blanked there for nuclei the tracks mark as mitotic
  (a few frames before and after a division). Interphase nuclei are left
  alone: in dense stages membranes lie within a few um of the nuclei.

All lengths are in um; pass the physical voxel size as `spacing` (z, y, x).
The raw image is assumed to be 0 outside the embryo, as in deconvolved crops.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class ScoreParams:
    """Parameters of the membrane score and the Cellpose blend."""

    sheet_um: float = 1.0
    """Scale (um) of the sheet filter."""
    floor_pct: float = 60.0
    """Score percentile in the embryo below which the score is set to 0."""
    top_pct: float = 99.5
    """Score percentile in the embryo that maps to 1."""
    mitotic_exclusion_um: float = 8.0
    """Blank the score this close to the chromatin of mitotic nuclei."""
    mitotic_before: int = 3
    """Frames before a division in which a nucleus counts as mitotic."""
    mitotic_after: int = 2
    """Frames after a division in which a nucleus counts as mitotic."""
    blend_smooth_um: float = 0.5
    """Smoothing (um) of the score before blending it with the raw image."""


def embryo_from_raw(raw: np.ndarray) -> np.ndarray:
    """The embryo as the filled, slightly closed non-zero raw signal, per slice.

    Deconvolved images can carry sparse non-zero noise outside the embryo
    (about 2 % of the background voxels in Pos_23): a voxel only counts when
    more than half of its surroundings (~1 z-slice, 4 pixels) is non-zero,
    and only the largest connected region is kept. Filling per z-slice keeps
    a cavity that opens towards the top or bottom of the stack inside the
    embryo, which a 3D fill would leave out.
    """
    share = ndimage.gaussian_filter((raw > 0).astype(np.float32), (1, 4, 4))
    sig = share > 0.5
    lab, n = ndimage.label(sig)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        sig = lab == sizes.argmax()
    out = np.zeros(sig.shape, bool)
    for z in range(sig.shape[0]):
        if sig[z].any():
            out[z] = ndimage.binary_fill_holes(
                ndimage.binary_closing(sig[z], iterations=3)
            )
    return out


def normalise(raw: np.ndarray, embryo: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Raw intensities divided by their `pct` percentile inside the embryo."""
    raw = raw.astype(np.float32)
    ref = np.percentile(raw[embryo], pct) if embryo.any() else raw.max()
    return raw / max(float(ref), 1e-6)


def sym3_eigvals(xx, yy, zz, xy, xz, yz) -> np.ndarray:
    """Eigenvalues of symmetric 3x3 matrices, closed form, on whole arrays.

    The trigonometric solution of the characteristic cubic (Smith 1961),
    vectorised: much faster than a general eigen solver per matrix. Returns an
    array (..., 3) in ascending order. Computed in float64 for accuracy.
    """
    a = [np.asarray(v, np.float64) for v in (xx, yy, zz, xy, xz, yz)]
    xx, yy, zz, xy, xz, yz = a
    q = (xx + yy + zz) / 3.0
    p1 = xy**2 + xz**2 + yz**2
    p2 = (xx - q) ** 2 + (yy - q) ** 2 + (zz - q) ** 2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    safe = p > 1e-30
    inv = np.where(safe, 1.0 / np.where(safe, p, 1.0), 0.0)
    b_xx, b_yy, b_zz = (xx - q) * inv, (yy - q) * inv, (zz - q) * inv
    b_xy, b_xz, b_yz = xy * inv, xz * inv, yz * inv
    det = (
        b_xx * (b_yy * b_zz - b_yz**2)
        - b_xy * (b_xy * b_zz - b_yz * b_xz)
        + b_xz * (b_xy * b_yz - b_yy * b_xz)
    )
    phi = np.arccos(np.clip(det / 2.0, -1.0, 1.0)) / 3.0
    e1 = q + 2.0 * p * np.cos(phi)
    e3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    return np.sort(np.stack([e1, e2, e3], axis=-1), axis=-1)


def sheet_score(
    img: np.ndarray, sigma_um: float, spacing, where: np.ndarray | None = None
) -> np.ndarray:
    """Scale-normalised plate measure for bright sheets (Frangi-like).

    Hessian eigenvalues sorted by magnitude |l1| <= |l2| <= |l3|: a bright
    sheet has one large negative l3 and small l1, l2. The measure is
    exp(-(l2/l3)^2 / 0.5) * |l| for l3 < 0, so fibres and blobs, where l2 is
    as large as l3, score low. Derivatives are taken in um, so anisotropic
    voxels are handled exactly. Eigenvalues are computed in slabs of z-slices
    to bound memory. With `where`, the measure is computed only there (0
    elsewhere), which saves the eigenvalue work on voxels that do not matter.
    """
    spacing = tuple(float(s) for s in spacing)
    sm = ndimage.gaussian_filter(
        img.astype(np.float32), [sigma_um / s for s in spacing]
    )
    grads = np.gradient(sm, *spacing)
    del sm
    hess = {}
    for i in range(3):
        g2 = np.gradient(grads[i], *spacing)
        for j in range(i, 3):
            hess[i, j] = (g2[j] * sigma_um**2).astype(np.float32)
        del g2
    del grads
    out = np.zeros(img.shape, np.float32)
    for z0 in range(0, img.shape[0], 8):
        z1 = min(z0 + 8, img.shape[0])
        sel = np.ones(img[z0:z1].shape, bool) if where is None else where[z0:z1]
        if not sel.any():
            continue
        h = {k: v[z0:z1][sel] for k, v in hess.items()}
        ev = sym3_eigvals(h[0, 0], h[1, 1], h[2, 2], h[0, 1], h[0, 2], h[1, 2])
        ev = np.take_along_axis(ev, np.argsort(np.abs(ev), axis=-1), axis=-1)
        ra = np.abs(ev[..., 1]) / (np.abs(ev[..., 2]) + 1e-12)
        mag = np.sqrt((ev**2).sum(axis=-1))
        out[z0:z1][sel] = np.where(ev[..., 2] < 0, np.exp(-(ra**2) / 0.5) * mag, 0.0)
    return out


def mitotic_nodes(tracks, nodes: Iterable[int], before: int, after: int) -> set[int]:
    """Nodes up to `before` frames before or `after` frames after a division."""
    out = set()
    for n in nodes:
        cur = int(n)
        for _ in range(before + 1):  # forward: is a division coming?
            succ = tracks.successors(cur)
            if len(succ) > 1:
                out.add(int(n))
                break
            if len(succ) != 1:
                break
            cur = int(succ[0])
        cur = int(n)
        for _ in range(after):  # backward: did a division just happen?
            pred = tracks.predecessors(cur)
            if len(pred) != 1:
                break
            if len(tracks.successors(int(pred[0]))) > 1:
                out.add(int(n))
                break
            cur = int(pred[0])
    return out


def membrane_score(
    raw: np.ndarray,
    nuclei: np.ndarray,
    spacing,
    mitotic: Iterable[int] = (),
    params: ScoreParams | None = None,
    embryo: np.ndarray | None = None,
) -> np.ndarray:
    """Membrane score in [0, 1]: sheet filter x brightness, spindle suppressed.

    Args:
        raw: raw membrane image of one frame (or an embryo crop of it).
        nuclei: nuclei labels of the same region; values are node ids.
        spacing: voxel size (z, y, x) in um.
        mitotic: node ids of mitotic nuclei; the score is blanked within
            `params.mitotic_exclusion_um` of them.
        params: score parameters.
        embryo: the embryo mask; computed from `raw` if not given.
    """
    params = params or ScoreParams()
    embryo = embryo_from_raw(raw) if embryo is None else embryo
    norm = normalise(raw, embryo)
    # The score is multiplied by the brightness, so it is 0 wherever the raw
    # image is; the sheet filter is skipped there.
    score = sheet_score(norm, params.sheet_um, spacing, where=norm > 0)
    score *= np.clip(norm, 0, 1)
    if embryo.any():
        lo, hi = np.percentile(score[embryo], [params.floor_pct, params.top_pct])
    else:
        lo, hi = 0.0, 1.0
    score = np.clip((score - lo) / max(float(hi - lo), 1e-9), 0, 1).astype(np.float32)
    mitotic = list(mitotic)
    if mitotic:
        near = ndimage.distance_transform_edt(
            ~np.isin(nuclei, mitotic), sampling=spacing
        )
        score[near < params.mitotic_exclusion_um] = 0
    return score


def blend_for_segmentation(
    raw: np.ndarray,
    score: np.ndarray,
    spacing,
    params: ScoreParams | None = None,
    embryo: np.ndarray | None = None,
    scale: float = 4000.0,
) -> np.ndarray:
    """Raw image with membranes boosted by the score, as input for Cellpose.

    Half the normalised raw image plus half the (slightly smoothed,
    renormalised) membrane score, scaled to uint16 [0, `scale`].
    """
    params = params or ScoreParams()
    embryo = embryo_from_raw(raw) if embryo is None else embryo
    norm = normalise(raw, embryo)
    sm = ndimage.gaussian_filter(score, [params.blend_smooth_um / s for s in spacing])
    if embryo.any():
        sm = sm / max(float(np.percentile(sm[embryo], 99.5)), 1e-6)
    blend = 0.5 * np.clip(norm, 0, 1) + 0.5 * np.clip(sm, 0, 1)
    return (blend * scale).astype(np.uint16)


def blend_frame(
    raw: np.ndarray,
    nuclei: np.ndarray,
    spacing,
    mitotic: Iterable[int] = (),
    params: ScoreParams | None = None,
) -> np.ndarray:
    """The Cellpose input for one whole frame (uint16, same shape as `raw`).

    Score and blend are computed on the embryo's bounding box only; outside
    it the result is 0, like the raw image.
    """
    out = np.zeros(raw.shape, np.uint16)
    embryo = embryo_from_raw(raw)
    if not embryo.any():
        return out
    box = tuple(
        slice(max(s.start - 8, 0), min(s.stop + 8, n))
        for s, n in zip(
            ndimage.find_objects(embryo.astype(np.uint8))[0], raw.shape, strict=True
        )
    )
    r, emb = raw[box], embryo[box]
    score = membrane_score(r, nuclei[box], spacing, mitotic, params, embryo=emb)
    out[box] = blend_for_segmentation(r, score, spacing, params, embryo=emb)
    return out
