"""Tests for the mask-only measurement columns this fork keeps up to date.

Measurement columns imported from a pipeline geff belong to no annotator, so
funtracks never refreshes them and a newly painted node keeps the column
default forever. `attach_mask_features` hands the two mask-derived ones to a
`RegionpropsAnnotator` subclass. Without these tests an upstream merge or a
funtracks bump can silently take that ownership away again while the rest of
the suite stays green.
"""

import numpy as np
import pytest
from funtracks.annotators import RegionpropsAnnotator

from motile_tracker.mask_features import (
    EQUIV_DIAMETER_KEY,
    EXTRA_FEATURES,
    NUM_PIXELS_KEY,
    MaskFeatureAnnotator,
    attach_mask_features,
)

# The fixture masks are filled boxes, so the voxel count of each node is simply
# the volume of its bbox. graph_3d uses 41, 21 and 31 voxel cubes.
EXPECTED_VOXELS = {1: 41**3, 2: 21**3, 3: 31**3}


def _graph(tracks):
    """Return the tracks graph, preferring the non-deprecated accessor."""
    if hasattr(tracks, "graph_solution"):
        return tracks.graph_solution
    return tracks.graph


def _add_extra_columns(tracks, keys=None):
    """Register the measurement columns the way an imported geff would."""
    for key in EXTRA_FEATURES if keys is None else keys:
        tracks.add_feature(key, EXTRA_FEATURES[key][0])


def _regionprops_annotators(tracks):
    return [a for a in tracks.annotators if isinstance(a, RegionpropsAnnotator)]


def _is_enabled(annotator, key):
    return bool(annotator.all_features[key][1])


def test_annotator_owns_the_extra_features(solution_tracks_3d):
    """Both keys are registered, switched off, and mapped to regionprops."""
    annotator = MaskFeatureAnnotator(solution_tracks_3d)

    for key, (_feature, regionprops_attr) in EXTRA_FEATURES.items():
        assert key in annotator.all_features
        assert not _is_enabled(annotator, key)  # off until enabled
        assert annotator.regionprops_names[key] == regionprops_attr

    # subclassing must not lose what upstream already owns
    assert "area" in annotator.all_features
    assert "pos" in annotator.all_features


def test_attach_adopts_existing_columns(solution_tracks_3d):
    """The plain annotator is replaced and the present columns switched on."""
    tracks = solution_tracks_3d
    _add_extra_columns(tracks)

    adopted = attach_mask_features(tracks)

    assert sorted(adopted) == sorted(EXTRA_FEATURES)
    annotators = _regionprops_annotators(tracks)
    assert len(annotators) == 1
    assert isinstance(annotators[0], MaskFeatureAnnotator)
    for key in EXTRA_FEATURES:
        assert _is_enabled(annotators[0], key)


def test_only_present_columns_are_adopted(solution_tracks_3d):
    """No column is invented for data that never carried it."""
    tracks = solution_tracks_3d
    _add_extra_columns(tracks, [NUM_PIXELS_KEY])

    assert attach_mask_features(tracks) == [NUM_PIXELS_KEY]
    assert EQUIV_DIAMETER_KEY not in tracks.features


def test_recompute_fills_the_adopted_columns(solution_tracks_3d):
    """With recompute the values come from the masks, not from the default."""
    tracks = solution_tracks_3d
    _add_extra_columns(tracks)

    attach_mask_features(tracks, recompute=True)

    graph = _graph(tracks)
    for node_id, voxels in EXPECTED_VOXELS.items():
        attrs = graph.nodes[node_id]
        assert int(attrs[NUM_PIXELS_KEY]) == voxels
        # sphere of equal volume
        assert attrs[EQUIV_DIAMETER_KEY] == pytest.approx(
            (6 * voxels / np.pi) ** (1 / 3), rel=1e-3
        )


def test_attach_keeps_pipeline_values_by_default(solution_tracks_3d):
    """Without recompute, existing values are left exactly as they were."""
    tracks = solution_tracks_3d
    _add_extra_columns(tracks, [NUM_PIXELS_KEY])
    graph = _graph(tracks)
    tracks._set_node_attr(1, NUM_PIXELS_KEY, 12345)

    attach_mask_features(tracks)

    assert int(graph.nodes[1][NUM_PIXELS_KEY]) == 12345


def test_attach_is_idempotent(solution_tracks_3d):
    """A second load must not swap our own subclass out again."""
    tracks = solution_tracks_3d
    _add_extra_columns(tracks)

    first = attach_mask_features(tracks)
    second = attach_mask_features(tracks)

    assert sorted(second) == sorted(first)
    annotators = _regionprops_annotators(tracks)
    assert len(annotators) == 1
    assert isinstance(annotators[0], MaskFeatureAnnotator)
    for key in EXTRA_FEATURES:
        assert _is_enabled(annotators[0], key)


def test_attach_preserves_already_enabled_features(solution_tracks_3d):
    """Features switched on before the swap stay on after it."""
    tracks = solution_tracks_3d
    _add_extra_columns(tracks)
    tracks.enable_features(["area"])

    attach_mask_features(tracks)

    annotator = _regionprops_annotators(tracks)[0]
    assert _is_enabled(annotator, "area")


def test_attach_without_segmentation_is_a_no_op(
    solution_tracks_3d_without_segmentation,
):
    """Nothing can be measured without a segmentation, so nothing is claimed."""
    tracks = solution_tracks_3d_without_segmentation
    _add_extra_columns(tracks)

    assert attach_mask_features(tracks) == []


def test_attach_without_matching_columns_is_a_no_op(solution_tracks_3d):
    """Data without these columns keeps the unmodified upstream annotator."""
    tracks = solution_tracks_3d

    assert attach_mask_features(tracks) == []

    annotators = _regionprops_annotators(tracks)
    assert len(annotators) == 1
    assert type(annotators[0]) is RegionpropsAnnotator
