"""Run the membrane merge over a whole time series and store it as node features.

Following the hoct pattern (royerlab/hoct, features/features.py), every value
is a pure function of one node's data and is written back onto the graph by
node id; registering the keys with `Tracks.add_feature` makes them show up in
the table and tree view, and makes `write_to_geff` save them as node props.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np

from .merge import (
    QC_FROM_NUCLEUS,
    MembraneMergeParams,
    fill_within,
    merge_frame,
    split_region,
)
from .refine import (
    RefineParams,
    boundary_signal,
    drop_islands,
    raw_cavity,
    refine_boundaries,
    voronoi_correct,
)
from .score import ScoreParams, embryo_from_raw, membrane_score, mitotic_nodes

if TYPE_CHECKING:
    from funtracks.data_model import Tracks
    from funtracks.features import Feature


def _feature(value_type: str, display_name: str, default) -> Feature:
    return {
        "feature_type": "node",
        "value_type": value_type,
        "num_values": 1,
        "display_name": display_name,
        "default_value": default,
    }


MEMBRANE_FEATURES: dict[str, Feature] = {
    "membrane_id": _feature("int", "Membrane ID", 0),
    "membrane_volume": _feature("float", "Membrane Volume", 0.0),
    "membrane_n_fragments": _feature("int", "Membrane Fragments", 0),
    "membrane_merge_score": _feature("float", "Membrane Merge Score", float("nan")),
    "membrane_n_nuclei": _feature("int", "Membrane Nuclei", 0),
    "membrane_qc": _feature("str", "Membrane QC", ""),
}

# Only computed when the raw membrane image is given (see `raw` below).
RAW_FEATURES: dict[str, Feature] = {
    "membrane_boundary_signal": _feature(
        "float", "Membrane Boundary Signal", float("nan")
    ),
}


def dividing_pairs(
    tracks: Tracks, nodes: Iterable[int], window: int
) -> list[tuple[int, int]]:
    """Sister nodes at most `window` frames after their parent divided.

    For each node, walk back along single-predecessor links for up to `window`
    steps; two nodes that reach the same dividing parent are a pair.
    """
    if window < 1:
        return []
    by_parent: dict[int, list[int]] = {}
    for node in nodes:
        current = int(node)
        for _ in range(window):
            preds = tracks.predecessors(current)
            if len(preds) != 1:
                break
            parent = int(preds[0])
            if len(tracks.successors(parent)) > 1:
                by_parent.setdefault(parent, []).append(int(node))
                break
            current = parent
    pairs = []
    for children in by_parent.values():
        pairs += [(a, b) for i, a in enumerate(children) for b in children[i + 1 :]]
    return pairs


def compute_membrane_features(
    tracks: Tracks,
    membrane,
    params: MembraneMergeParams | None = None,
    division_window: int = 1,
    progress: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
    out=None,
    start_frame: int = 0,
    workers: int = 1,
    keep_cavities_from: int | None = None,
    raw=None,
    raw_spacing=None,
    score_params: ScoreParams | None = None,
    refine_params: RefineParams | None = None,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Merge the membrane labels of every frame onto the tracked nuclei.

    Args:
        tracks: Tracks with a segmentation (the nuclei); node id = label.
        membrane: Raw membrane labels (t, [z], y, x), same shape as the
            nuclei segmentation. May be lazy (zarr/dask); read frame by frame.
        params: Merge thresholds.
        division_window: Sister nuclei up to this many frames after a division
            are flagged membrane_qc "dividing" (0 disables). Their cells are
            split between them like any others.
        progress: Called with (index among processed frames, their number).
        max_frames: Merge at most this many frames (default: to the end).
        out: Where to write the merged frames, e.g. a zarr array on disk from
            `io.create_label_store`, so that only one frame is in memory at a
            time. It must have the full shape of the segmentation, and frame t
            is written at index t, so frames that are not processed stay 0 (and
            take no disk space in zarr). Default: a new in-memory array
            holding only the processed frames, the first at index 0.
        start_frame: First frame to merge.
        keep_cavities_from: From this frame on, enclosed cavities (e.g. a
            forming blastocoel) are left empty instead of being filled, as with
            `params.fill_embryo_holes=False`. Earlier frames use `params` as
            given. Default: use `params` for every frame.
        workers: Frames merged at the same time, each in its own process.
            Frames are independent, so the result is the same as with 1. This
            process reads the frames and writes the results; at most `workers`
            frames are in flight at once, so memory grows by about one frame's
            merge (~2 GB for a 107 x 594 x 616 frame) per worker; with
            `raw`, about 4 GB.
        raw: The raw membrane image (t, z, y, x), same shape, read frame by
            frame. When given, each merged frame is refined on it: the
            boundaries between touching cells move onto the membrane signal
            where it is clear (`refine.refine_boundaries`), with the spindle
            of mitotic cells suppressed (`score.membrane_score`); stray
            pieces of cells are dropped; from `keep_cavities_from` on, the
            blank inside of the embryo (the cavity) is removed from the cells;
            and each node gets `membrane_boundary_signal`.
        raw_spacing: Physical voxel size (z, y, x) in um for the raw-image
            steps. Default: the tracks' scale. The merge itself always uses
            the tracks' scale.
        score_params, refine_params: Parameters of those steps.

    Returns:
        The merged labels (`out` if given; values are node ids) and per-node
        feature records. Nodes in frames not processed get no record.
    """
    if tracks.segmentation is None:
        raise ValueError("The tracks have no nuclei segmentation.")
    seg_shape = tuple(tracks.segmentation.shape)
    if not 0 <= start_frame < seg_shape[0]:
        raise ValueError(f"start_frame {start_frame} outside 0..{seg_shape[0] - 1}")
    n_frames = seg_shape[0] - start_frame
    if max_frames is not None:
        n_frames = min(max_frames, n_frames)
    stop = start_frame + n_frames
    mem_shape = tuple(membrane.shape)
    if mem_shape[1:] != seg_shape[1:] or mem_shape[0] < stop:
        raise ValueError(
            f"Membrane labels have shape {mem_shape}, "
            f"the nuclei segmentation {seg_shape}."
        )
    scale = tracks.scale or [1.0] * len(seg_shape)
    spacing = tuple(float(s) for s in scale[1:])
    if raw is not None and (
        tuple(raw.shape)[1:] != seg_shape[1:] or raw.shape[0] < stop
    ):
        raise ValueError(
            f"Raw image has shape {tuple(raw.shape)}, the nuclei segmentation "
            f"{seg_shape}."
        )
    raw_spacing = tuple(float(s) for s in (raw_spacing or spacing))
    score_params = score_params or ScoreParams()
    refine_params = refine_params or RefineParams()

    if out is None:
        # Labels are node ids; uint32 halves memory over uint64 for long movies.
        merged = np.zeros((n_frames, *seg_shape[1:]), dtype=np.uint32)
        offset = start_frame
    elif tuple(out.shape) != seg_shape:
        raise ValueError(f"out has shape {tuple(out.shape)}, need {seg_shape}.")
    else:
        merged, offset = out, 0
    features: dict[int, dict] = {}

    base = params or MembraneMergeParams()
    keep = dataclasses.replace(base, fill_embryo_holes=False)

    def job(t: int) -> tuple:
        # Node ids fit in uint32, which halves what is sent to a worker.
        nuclei = np.asarray(tracks.segmentation[t]).astype(np.uint32)
        nodes = [int(n) for n in np.unique(nuclei) if n]
        pairs = dividing_pairs(tracks, nodes, division_window)
        late = keep_cavities_from is not None and t >= keep_cavities_from
        p = keep if late else base
        refine = None
        if raw is not None:
            mitotic = mitotic_nodes(
                tracks, nodes, score_params.mitotic_before, score_params.mitotic_after
            )
            refine = (
                np.asarray(raw[t]),
                raw_spacing,
                mitotic,
                score_params,
                refine_params,
                late,
            )
        return t, np.asarray(membrane[t]), nuclei, spacing, p, pairs, refine

    def collect(t: int, labels: np.ndarray, feats: dict, done: int) -> None:
        merged[t - offset] = labels
        features.update(feats)
        if progress is not None:
            progress(done, n_frames)

    frames = range(start_frame, stop)
    if workers <= 1:
        for done, t in enumerate(frames):
            collect(*_merge_job(job(t)), done)
        return merged, features

    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    todo = iter(frames)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        running = set()
        for t in todo:
            running.add(pool.submit(_merge_job, job(t)))
            if len(running) < workers:
                continue
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                collect(*fut.result(), done)
                done += 1
        for fut in running:
            collect(*fut.result(), done)
            done += 1
    return merged, features


def _merge_job(args: tuple) -> tuple[int, np.ndarray, dict]:
    """Merge (and refine) one frame; module level so workers can run it."""
    t, membrane, nuclei, spacing, params, pairs, refine = args
    res = merge_frame(
        membrane, nuclei, spacing=spacing, params=params, dividing_pairs=pairs
    )
    labels, feats = res.labels, res.features
    if refine is not None:
        labels = _refine_frame(labels, feats, nuclei, spacing, *refine)
    return t, labels, feats


def _refine_frame(
    labels, feats, nuclei, merge_spacing, raw, spacing, mitotic, sp, rp, cavity
) -> np.ndarray:
    """Refine merged labels on the raw image; updates `feats` in place."""
    from scipy import ndimage

    embryo = embryo_from_raw(raw, spacing, sp.outline_frac, sp.outline_close_um)
    fg = embryo | (labels > 0)
    if not fg.any():
        return labels
    box = tuple(
        slice(max(s.start - 8, 0), min(s.stop + 8, n))
        for s, n in zip(
            ndimage.find_objects(fg.astype(np.uint8))[0], fg.shape, strict=True
        )
    )
    r, n_sub, emb = raw[box], nuclei[box], embryo[box]
    score = membrane_score(r, n_sub, spacing, mitotic, sp, embryo=emb)
    cav = raw_cavity(r, n_sub, spacing, emb, rp) if cavity else None
    sub = labels[box].copy()
    # A nucleus the segmentation gave no cell (e.g. a thin outer cell it
    # missed) seeds its own cell from its nucleus.
    # If its nucleus lies inside another cell, that cell is split between the
    # two nuclei, as the merge does for a label holding several nuclei.
    for n, rec in feats.items():
        if rec["membrane_id"] != 0:
            continue
        own = n_sub == n
        if not own.any():
            continue
        hosts = sub[own]
        hosts = hosts[hosts > 0]
        if hosts.size:
            host = int(np.bincount(hosts).argmax())
            region = sub == host
            rbox = ndimage.find_objects(region.astype(np.uint8))[0]
            if (n_sub[rbox][region[rbox]] == host).any():
                split = split_region(region[rbox], n_sub[rbox], [host, n], spacing)
                sub[rbox][region[rbox]] = split[region[rbox]]
            else:
                sub[own] = n
        else:
            sub[own] = n
        rec.update(membrane_id=int(n), membrane_qc=QC_FROM_NUCLEUS)
    # The embryo outline from the raw image covers what the segmentation
    # left empty; every voxel in it (but not in the cavity) goes to the
    # nearest cell before the boundaries move onto the membrane.
    inside = emb if cav is None else emb & ~cav
    if rp.voronoi and sub.any():
        # Gaps go to the nearest nucleus' cell, and cell parts reaching far
        # into another nucleus' Voronoi region are handed over.
        sub, _ = voronoi_correct(sub, n_sub, inside, spacing, rp.voronoi_margin_um)
    elif sub.any() and inside.any():
        sub = fill_within(sub, inside | (sub > 0), spacing)
    sub, _ = refine_boundaries(sub, score, n_sub, spacing, rp)
    sub, _ = drop_islands(sub, spacing)
    if cav is not None:
        sub[cav] = 0
    labels = labels.copy()
    labels[box] = sub
    signal = boundary_signal(sub, score)
    voxel = float(np.prod(merge_spacing))
    counts = np.bincount(sub.ravel())
    for rec in feats.values():
        mid = rec["membrane_id"]
        if mid:
            rec["membrane_volume"] = (
                float(counts[mid] if mid < counts.size else 0) * voxel
            )
        rec["membrane_boundary_signal"] = signal.get(mid, float("nan"))
    return labels


def add_membrane_features(tracks: Tracks, features: dict[int, dict]) -> None:
    """Register the membrane feature keys and write the per-node values.

    The raw-image features are registered only when the records carry them.
    """
    nodes = list(features)
    keys = dict(MEMBRANE_FEATURES)
    for key, feature in RAW_FEATURES.items():
        if any(key in features[n] for n in nodes):
            keys[key] = feature
    for key, feature in keys.items():
        if key not in tracks.features:
            tracks.add_feature(key, feature)
    if not nodes:
        return
    for key, feature in keys.items():
        default = feature["default_value"]
        tracks._set_nodes_attr(
            nodes, key, [features[n].get(key, default) for n in nodes]
        )
