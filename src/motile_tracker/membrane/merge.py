"""Nucleus-guided agglomeration of membrane label fragments, one frame at a time.

Membrane segmentations are typically over-segmented (one cell cut into several
fragments) or under-segmented (one fragment covering several cells). The tracked
nuclei are a much more reliable signal, so they are used as seeds:

1. A fragment is *seeded* by a nucleus when it covers at least
   ``min_nucleus_overlap`` of that nucleus' volume.
2. Fragments are joined greedily along a region adjacency graph, best shared
   surface first. The score of a contact is ``contact_area / min(surface_a,
   surface_b)``: 1 means the smaller fragment is fully wrapped by the larger.
   Two groups seeded by different nuclei are never joined (cannot-link), and
   an unseeded fragment only joins a seeded group whose nucleus centroid lies
   within ``max_dist_um`` of the fragment centroid. An unseeded fragment lying
   *between* cells (it touches several, none with at least
   ``ambiguous_share`` of its contact with cells) is ambiguous: neither it
   nor what it collects joins any cell, and the gap filling divides it
   among its neighbours voxel by voxel instead of handing it whole to one.
3. A group still holding several nuclei (under-segmentation, or a cell in
   ana-/telophase before cytokinesis) is split by a seeded watershed with the
   nuclei as markers, so every nucleus gets its own cell. The landscape is the
   distance to the nearest nucleus (a Voronoi split; for two daughters, the
   plane halfway between them). Sister nuclei of a recent division
   (`dividing_pairs`) are split the same way, but flagged "dividing", since
   their membrane may not have separated yet.
   Before that, unseeded fragments fully embedded in one seeded group (they
   touch that group and no background) are absorbed into it, whatever their
   contact score.
4. The output labels each cell with the node id of its nucleus, so it shares
   the track colormap of the nuclei layer. Unseeded groups become background.
5. Gap filling: every unlabelled voxel inside the embryo outline is given the
   label of the nearest cell, so cells have no holes and no gaps between them
   and grow outward until they reach the outline, but never past it. The
   outline is by default the raw labels (and nuclei) of the frame, closed with
   a ball of radius ``closing_radius_um`` and with enclosed holes filled: that
   closes fjord-like inlets narrower than the ball but keeps the bulges of
   the cells on the surface, and the shallow notches between them, as they
   are. The alternative, the convex hull of all labels, fills every dent and
   so flattens the surface.

All lengths are in µm (pass ``spacing`` as the voxel size in µm per axis).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

QC_OK = "ok"
QC_SPLIT = "split"
QC_DIVIDING = "dividing"
QC_SHARED = "shared"  # several nuclei, one label (use_watershed_split=False)
QC_NO_MEMBRANE = "no_membrane"


@dataclass
class MembraneMergeParams:
    """Thresholds of the merge; tune them on real data."""

    min_nucleus_overlap: float = 0.5
    """Fraction of a nucleus' volume a fragment must cover to be seeded by it."""
    min_shared_frac: float = 0.2
    """Minimum contact_area / min(surface_a, surface_b) to join two fragments."""
    min_contact_area_um2: float = 0.0
    """Minimum absolute contact area (µm²) to join two fragments."""
    max_dist_um: float = np.inf
    """Max distance (µm) from an unseeded fragment centroid to the nucleus."""
    min_cell_volume_um3: float = 0.0
    """Seeded cells smaller than this (µm³) are reported as no_membrane."""
    use_watershed_split: bool = True
    """Split groups holding several nuclei; otherwise they keep a shared label."""
    merge_embedded: bool = True
    """Absorb unseeded fragments enclosed by a single seeded group."""
    ambiguous_share: float = 0.6
    """An unseeded fragment touching cells of several nuclei joins none of them
    unless one has at least this share of its contact with cells; the gap
    filling then divides it. 0 disables."""
    fill_gaps: bool = True
    """Give every unlabelled voxel inside the embryo outline its nearest cell."""
    outline: str = "closed"
    """Embryo outline for gap filling: "closed" (raw mask closed with a ball of
    `closing_radius_um`, then hole-filled) or "convex_hull" of all raw labels
    (fills every dent, flattening the cells' bulges on the surface)."""
    closing_radius_um: float = 6.0
    """Ball radius (µm) of the "closed" outline: inlets up to about twice this
    wide are closed, wider dents are kept."""
    fill_embryo_holes: bool = True
    """Treat enclosed cavities as inside the embryo. Turn off once a cavity
    (e.g. a blastocoel) should stay empty; it is then cut out of either outline."""


@dataclass
class MergeResult:
    labels: np.ndarray
    """Merged membrane labels; the value is the node id of the cell's nucleus."""
    features: dict[int, dict] = field(default_factory=dict)
    """Per nucleus node id: membrane_id, membrane_volume, membrane_n_fragments,
    membrane_merge_score, membrane_n_nuclei, membrane_qc."""


class _UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent = {i: i for i in items}

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root


def _face_areas(spacing: tuple[float, ...]) -> list[float]:
    """Area (µm²) of a voxel face orthogonal to each axis."""
    total = float(np.prod(spacing))
    return [total / s for s in spacing]


def contact_areas(
    labels: np.ndarray, spacing: tuple[float, ...]
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """Shared surface between touching labels and total surface per label.

    Surfaces are counted as voxel faces, so they are staircase approximations;
    that is fine for the ratios used here. Faces on the image border are not
    counted.
    """
    pairs: dict[tuple[int, int], float] = {}
    surface: dict[int, float] = {}
    for axis, face in enumerate(_face_areas(spacing)):
        # Compare neighbours on views; only the (few) boundary voxels are copied.
        a = np.moveaxis(labels, axis, 0)[:-1]
        b = np.moveaxis(labels, axis, 0)[1:]
        diff = a != b
        a, b = a[diff], b[diff]
        for lab, counts in zip(
            *np.unique(np.concatenate([a, b]), return_counts=True), strict=True
        ):
            if lab:
                surface[int(lab)] = surface.get(int(lab), 0.0) + counts * face
        both = (a > 0) & (b > 0)
        lo = np.minimum(a[both], b[both])
        hi = np.maximum(a[both], b[both])
        if lo.size:
            uniq, counts = np.unique(np.stack([lo, hi]), axis=1, return_counts=True)
            for (i, j), c in zip(uniq.T, counts, strict=True):
                key = (int(i), int(j))
                pairs[key] = pairs.get(key, 0.0) + c * face
    return pairs, surface


def _centroids(labels: np.ndarray, ids: list[int], spacing) -> dict[int, np.ndarray]:
    if not ids:
        return {}
    coms = ndimage.center_of_mass(labels > 0, labels, ids)
    return {i: np.asarray(c) * spacing for i, c in zip(ids, coms, strict=True)}


def _counts(labels: np.ndarray) -> dict[int, int]:
    """Voxel count per non-zero label; bincount, which beats a sort on big frames."""
    flat = labels.ravel()
    if flat.size and int(flat.max()) < 1 << 24:
        c = np.bincount(flat)
        ids = np.flatnonzero(c)
        return {int(i): int(c[i]) for i in ids if i}
    ids, c = np.unique(flat, return_counts=True)
    return {int(i): int(n) for i, n in zip(ids, c, strict=True) if i}


def _crop(
    membrane: np.ndarray, nuclei: np.ndarray, margin: int
) -> tuple[slice, ...] | None:
    """Bounding box of all labelled voxels plus `margin`, or None if empty."""
    fg = (membrane > 0) | (nuclei > 0)
    slices = []
    for axis in range(fg.ndim):
        other = tuple(i for i in range(fg.ndim) if i != axis)
        hits = np.flatnonzero(fg.any(axis=other))
        if hits.size == 0:
            return None
        lo = max(int(hits[0]) - margin, 0)
        hi = min(int(hits[-1]) + 1 + margin, fg.shape[axis])
        slices.append(slice(lo, hi))
    return tuple(slices)


def merge_frame(
    membrane: np.ndarray,
    nuclei: np.ndarray,
    spacing: Iterable[float] | None = None,
    params: MembraneMergeParams | None = None,
    dividing_pairs: Iterable[Iterable[int]] = (),
) -> MergeResult:
    """Merge the membrane fragments of one frame onto the tracked nuclei.

    Args:
        membrane: raw membrane labels of one frame (0 = background).
        nuclei: nuclei labels of the same frame; values are node ids.
        spacing: voxel size in µm per axis (default 1).
        params: merge thresholds.
        dividing_pairs: node-id pairs of sister nuclei shortly after a
            division. They are split like any other nuclei, but get
            membrane_qc "dividing" instead of "ok"/"split".
    """
    params = params or MembraneMergeParams()
    # All work happens in the bounding box of the labelled voxels; the margin
    # keeps a background shell around it, so contacts with background and the
    # embryo closing come out exactly as on the full frame.
    sp = tuple(float(x) for x in (spacing or (1.0,) * membrane.ndim))
    margin = int(np.ceil(params.closing_radius_um / min(sp))) + 1
    box = _crop(membrane, nuclei, margin=margin)
    out = np.zeros(membrane.shape, dtype=np.uint32)
    if box is None:
        return MergeResult(labels=out)
    res = _merge_cropped(
        np.ascontiguousarray(membrane[box]),
        np.ascontiguousarray(nuclei[box]),
        spacing,
        params,
        dividing_pairs,
    )
    out[box] = res.labels
    res.labels = out
    return res


def _merge_cropped(
    membrane: np.ndarray,
    nuclei: np.ndarray,
    spacing: Iterable[float] | None,
    params: MembraneMergeParams,
    dividing_pairs: Iterable[Iterable[int]],
) -> MergeResult:
    spacing = tuple(float(s) for s in (spacing or (1.0,) * membrane.ndim))
    voxel = float(np.prod(spacing))
    dividing = {frozenset(int(n) for n in p) for p in dividing_pairs}

    frag_ids = list(_counts(membrane))
    nuc_vol = _counts(nuclei)
    nuc_ids = list(nuc_vol)

    # Seeding: overlap of every (fragment, nucleus) pair.
    both = (membrane > 0) & (nuclei > 0)
    seeds: dict[int, set[int]] = {f: set() for f in frag_ids}
    if both.any():
        uniq, counts = np.unique(
            np.stack([membrane[both], nuclei[both]]), axis=1, return_counts=True
        )
        for (f, n), c in zip(uniq.T, counts, strict=True):
            if c / nuc_vol[n] >= params.min_nucleus_overlap:
                seeds[int(f)].add(int(n))

    pairs, surface = contact_areas(membrane, spacing)
    frag_cent = _centroids(membrane, frag_ids, spacing)
    nuc_cent = _centroids(nuclei, nuc_ids, spacing)

    uf = _UnionFind(frag_ids)
    group_seeds = {f: set(s) for f, s in seeds.items()}
    group_members = {f: [f] for f in frag_ids}
    group_score = dict.fromkeys(frag_ids, 1.0)
    ambiguous = {
        f: f in _ambiguous_fragments(seeds, pairs, params.ambiguous_share)
        for f in frag_ids
    }

    def close_enough(frag: int, nucs: set[int]) -> bool:
        return any(
            np.linalg.norm(frag_cent[frag] - nuc_cent[n]) <= params.max_dist_um
            for n in nucs
        )

    edges = []
    for (i, j), area in pairs.items():
        score = area / min(surface[i], surface[j])
        if score >= params.min_shared_frac and area >= params.min_contact_area_um2:
            edges.append((score, i, j))
    edges.sort(reverse=True)

    for score, i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        si, sj = group_seeds[ri], group_seeds[rj]
        if si and sj and si != sj:
            continue  # cannot-link: different nuclei
        if (ambiguous[ri] and sj) or (ambiguous[rj] and si):
            continue  # between cells: left for the gap filling to divide
        # An unseeded group joining a seeded one must lie near its nucleus.
        if si and not sj and not all(close_enough(f, si) for f in group_members[rj]):
            continue
        if sj and not si and not all(close_enough(f, sj) for f in group_members[ri]):
            continue
        uf.parent[rj] = ri
        group_seeds[ri] = si | sj
        group_members[ri] += group_members.pop(rj)
        group_score[ri] = min(group_score[ri], group_score.pop(rj), score)
        ambiguous[ri] = ambiguous[ri] or ambiguous.pop(rj)
        del group_seeds[rj]

    if params.merge_embedded:
        _absorb_embedded(
            uf,
            group_seeds,
            group_members,
            pairs,
            surface,
            skip={r for r, a in ambiguous.items() if a},
        )

    out = np.zeros(membrane.shape, dtype=np.uint32)
    features: dict[int, dict] = {}
    lut = np.zeros(int(membrane.max()) + 1, dtype=np.uint32)
    for root, members in group_members.items():
        lut[members] = root
    groups = lut[membrane]

    for root, nucs in group_seeds.items():
        if not nucs:
            continue  # unseeded group: background
        region = groups == root
        n_frag = len(group_members[root])
        score = group_score[root]
        if len(nucs) == 1 or not params.use_watershed_split:
            label = min(nucs)
            out[region] = label
            qc = QC_OK if len(nucs) == 1 else QC_SHARED
            vol = region.sum() * voxel
            for n in nucs:
                features[n] = _record(label, vol, n_frag, score, len(nucs), qc)
            continue
        # Under-segmentation: seeded watershed inside the region.
        # Flooding the distance to the nearest nucleus is a Voronoi split in µm,
        # which does not depend on the region having a background border.
        markers = np.where(region & np.isin(nuclei, list(nucs)), nuclei, 0)
        dist = ndimage.distance_transform_edt(markers == 0, sampling=spacing)
        split = watershed(dist, markers=markers, mask=region)
        out[region] = split[region]
        for n in nucs:
            vol = (split == n).sum() * voxel
            features[n] = _record(n, vol, n_frag, score, len(nucs), QC_SPLIT)

    for n in nuc_ids:
        rec = features.get(n)
        if rec is None or rec["membrane_volume"] < params.min_cell_volume_um3:
            if rec is not None and rec["membrane_qc"] != QC_SHARED:
                out[out == n] = 0
            features[n] = _record(0, 0.0, 0, np.nan, 0, QC_NO_MEMBRANE)

    # Flag sisters of a recent division, whichever way their cells came about.
    for pair in dividing:
        for n in pair:
            rec = features.get(n)
            if rec is not None and rec["membrane_qc"] != QC_NO_MEMBRANE:
                rec["membrane_qc"] = QC_DIVIDING

    if params.fill_gaps and out.any():
        # Nuclei seed the filling too, so a nucleus reaching into a dissolved
        # (ambiguous or unseeded) fragment keeps that part in its own cell.
        cell_of = np.zeros(int(nuclei.max()) + 1, dtype=np.uint32)
        for n, rec in features.items():
            cell_of[n] = rec["membrane_id"]
        out = np.where(out == 0, cell_of[nuclei], out)
        embryo = embryo_mask(
            membrane,
            nuclei,
            params.closing_radius_um,
            params.fill_embryo_holes,
            convex=params.outline == "convex_hull",
            spacing=spacing,
        )
        out = fill_within(out, embryo, spacing)
        volume = {i: c * voxel for i, c in _counts(out).items()}
        for rec in features.values():
            if rec["membrane_id"]:
                rec["membrane_volume"] = volume.get(rec["membrane_id"], 0.0)
    return MergeResult(labels=out, features=features)


def _ambiguous_fragments(
    seeds: dict[int, set[int]],
    pairs: dict[tuple[int, int], float],
    share: float,
) -> set[int]:
    """Unseeded fragments lying between cells of different nuclei.

    For each unseeded fragment, its contact area with seeded fragments is
    summed per nucleus set. It is ambiguous when it touches at least two and
    the largest takes less than `share` of that contact.
    """
    if share <= 0:
        return set()
    touch: dict[int, dict[frozenset, float]] = {}
    for (i, j), area in pairs.items():
        for f, other in ((i, j), (j, i)):
            if not seeds[f] and seeds[other]:
                key = frozenset(seeds[other])
                per = touch.setdefault(f, {})
                per[key] = per.get(key, 0.0) + area
    return {
        f
        for f, per in touch.items()
        if len(per) > 1 and max(per.values()) < share * sum(per.values())
    }


def _absorb_embedded(
    uf, group_seeds, group_members, pairs, surface, skip=frozenset()
) -> None:
    """Merge unseeded groups enclosed by exactly one seeded group.

    Unseeded groups that touch each other are handled together, so a cluster
    of nucleus-free fragments inside one cell is absorbed as a whole. A
    cluster qualifies when none of its surface faces background (border faces
    of the image are not counted as background), all its labelled neighbours
    belong to the same seeded group, and it holds no group from `skip`.
    """
    bg = dict(surface)
    group_adj: dict[int, set[int]] = {r: set() for r in group_members}
    for (i, j), area in pairs.items():
        bg[i] -= area
        bg[j] -= area
        ri, rj = uf.find(i), uf.find(j)
        if ri != rj:
            group_adj[ri].add(rj)
            group_adj[rj].add(ri)

    unseeded = {r for r in group_members if not group_seeds[r]}
    seen: set[int] = set()
    for start in unseeded:
        if start in seen:
            continue
        cluster, stack = [], [start]
        seen.add(start)
        while stack:
            r = stack.pop()
            cluster.append(r)
            for nb in group_adj[r]:
                if nb in unseeded and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        hosts = {nb for r in cluster for nb in group_adj[r] if nb not in unseeded}
        touches_bg = any(bg[f] > 1e-9 for r in cluster for f in group_members[r])
        if len(hosts) != 1 or touches_bg or skip.intersection(cluster):
            continue
        host = hosts.pop()
        for r in cluster:
            uf.parent[r] = host
            group_members[host] += group_members.pop(r)
            del group_seeds[r]


def embryo_mask(
    membrane: np.ndarray,
    nuclei: np.ndarray,
    closing_radius_um: float = 6.0,
    fill_holes: bool = True,
    convex: bool = False,
    spacing=None,
) -> np.ndarray:
    """The embryo outline from the raw membrane and nuclei labels.

    convex=True gives their convex hull; otherwise the labelled voxels closed
    with a ball of `closing_radius_um` (µm, using `spacing`). Enclosed
    cavities belong to the embryo when `fill_holes`, and are cut out otherwise.
    """
    labelled = (membrane > 0) | (nuclei > 0)
    if convex:
        hull = convex_hull_mask(labelled)
        if hull is not None:
            if not fill_holes:
                closed = embryo_mask(
                    membrane, nuclei, closing_radius_um, False, spacing=spacing
                )
                hull &= ~(ndimage.binary_fill_holes(closed) & ~closed)
            return hull
    mask = close_ball(labelled, closing_radius_um, spacing)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    return mask


def close_ball(mask: np.ndarray, radius: float, spacing=None) -> np.ndarray:
    """Morphological closing with a Euclidean ball of `radius` (µm).

    Dilation and erosion are thresholds on distance transforms, so the cost
    does not grow with the radius and anisotropic voxels are handled exactly.
    The mask is padded by the radius so the erosion does not eat the border.
    """
    if radius <= 0 or not mask.any():
        return mask
    spacing = tuple(float(s) for s in (spacing or (1.0,) * mask.ndim))
    pad = [int(np.ceil(radius / s)) + 1 for s in spacing]
    padded = np.pad(mask, [(k, k) for k in pad])
    dilated = ndimage.distance_transform_edt(~padded, sampling=spacing) <= radius
    closed = ndimage.distance_transform_edt(dilated, sampling=spacing) > radius
    closed |= padded  # rounding at the ball's rim must never drop a voxel
    return closed[tuple(slice(k, -k) for k in pad)]


def convex_hull_mask(mask: np.ndarray, tol: float = 1e-6) -> np.ndarray | None:
    """The convex hull of a binary mask, or None if it is flat or empty.

    The hull of a mask is the hull of the first and last foreground voxel of
    each line along the last axis, which cuts the points handed to qhull from
    millions to two per line. Likewise a convex body crosses every such line in
    one interval, so the mask is filled line by line from the hull's
    half-spaces instead of testing every voxel against every facet. The hull
    runs through voxel centres, so it is tight: it holds every voxel of the
    mask but adds nothing beyond its outermost voxels (skimage's
    convex_hull_image, built on voxel corners, is about half a voxel wider).
    `tol` (voxels) only absorbs rounding for centres lying on the hull.
    """
    from scipy.spatial import ConvexHull, QhullError

    lines = mask.reshape(-1, mask.shape[-1])
    has = lines.any(axis=1)
    if not has.any():
        return None
    first = lines.argmax(axis=1)
    last = mask.shape[-1] - 1 - lines[:, ::-1].argmax(axis=1)
    rows = np.flatnonzero(has)
    row_coords = np.stack(np.unravel_index(rows, mask.shape[:-1]), axis=1)
    pts = np.concatenate(
        [
            np.column_stack([row_coords, first[rows]]),
            np.column_stack([row_coords, last[rows]]),
        ]
    ).astype(float)
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return None

    # Facet i: normal[i] . p + offset[i] <= 0 inside (unit normals).
    normal, offset = hull.equations[:, :-1], hull.equations[:, -1]
    a_last, a_rest = normal[:, -1], normal[:, :-1]
    # Only lines through the hull's bounding box can cross it.
    lo_box = np.floor(pts.min(axis=0)).astype(int)
    hi_box = np.ceil(pts.max(axis=0)).astype(int)
    grids = np.meshgrid(
        *[np.arange(lo_box[d], hi_box[d] + 1) for d in range(mask.ndim - 1)],
        indexing="ij",
    )
    cand = np.stack([g.ravel() for g in grids], axis=1)
    out = np.zeros(mask.shape, dtype=bool)
    x = np.arange(mask.shape[-1])
    pos, neg = a_last > 1e-12, a_last < -1e-12
    flat = ~(pos | neg)
    for start in range(0, len(cand), 4096):
        c = cand[start : start + 4096]
        # Per facet: a_last * x <= tol - offset - a_rest . c
        rhs = tol - offset[None, :] - c.astype(float) @ a_rest.T
        upper = np.min(rhs[:, pos] / a_last[pos], axis=1, initial=np.inf)
        lower = np.max(rhs[:, neg] / a_last[neg], axis=1, initial=-np.inf)
        ok = np.all(rhs[:, flat] >= 0, axis=1)
        inside = (
            ok[:, None]
            & (x[None, :] >= lower[:, None])
            & (x[None, :] <= upper[:, None])
        )
        out[tuple(c.T)] = inside
    return out


def fill_within(labels: np.ndarray, mask: np.ndarray, spacing) -> np.ndarray:
    """Give every voxel in `mask` the nearest label (µm distance); 0 outside."""
    idx = ndimage.distance_transform_edt(
        labels == 0, sampling=spacing, return_distances=False, return_indices=True
    )
    filled = labels[tuple(idx)]
    del idx
    filled[~mask] = 0
    return filled


def _record(label, volume, n_frag, score, n_nuc, qc) -> dict:
    return {
        "membrane_id": int(label),
        "membrane_volume": float(volume),
        "membrane_n_fragments": int(n_frag),
        "membrane_merge_score": float(score),
        "membrane_n_nuclei": int(n_nuc),
        "membrane_qc": qc,
    }
