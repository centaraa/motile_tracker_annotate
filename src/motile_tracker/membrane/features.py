"""Run the membrane merge over a whole time series and store it as node features.

Following the hoct pattern (royerlab/hoct, features/features.py), every value
is a pure function of one node's data and is written back onto the graph by
node id; registering the keys with `Tracks.add_feature` makes them show up in
the table and tree view, and makes `write_to_geff` save them as node props.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np

from .merge import MembraneMergeParams, merge_frame

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

    if out is None:
        # Labels are node ids; uint32 halves memory over uint64 for long movies.
        merged = np.zeros((n_frames, *seg_shape[1:]), dtype=np.uint32)
        offset = start_frame
    elif tuple(out.shape) != seg_shape:
        raise ValueError(f"out has shape {tuple(out.shape)}, need {seg_shape}.")
    else:
        merged, offset = out, 0
    features: dict[int, dict] = {}
    for t in range(start_frame, stop):
        nuclei = np.asarray(tracks.segmentation[t])
        nodes = [int(n) for n in np.unique(nuclei) if n]
        res = merge_frame(
            np.asarray(membrane[t]),
            nuclei,
            spacing=spacing,
            params=params,
            dividing_pairs=dividing_pairs(tracks, nodes, division_window),
        )
        merged[t - offset] = res.labels
        features.update(res.features)
        if progress is not None:
            progress(t - start_frame, n_frames)
    return merged, features


def add_membrane_features(tracks: Tracks, features: dict[int, dict]) -> None:
    """Register the membrane feature keys and write the per-node values."""
    for key, feature in MEMBRANE_FEATURES.items():
        if key not in tracks.features:
            tracks.add_feature(key, feature)
    nodes = list(features)
    if not nodes:
        return
    for key in MEMBRANE_FEATURES:
        tracks._set_nodes_attr(nodes, key, [features[n][key] for n in nodes])
