"""Keep mask-derived pipeline measurements live when nodes are edited.

Measurement columns that arrive inside an imported geff (num_pixels,
equivalent_diameter_area, ...) were produced by an external pipeline and belong
to no annotator, so funtracks never refreshes them: a newly painted node keeps
the column default forever. The two columns below are pure functions of the
mask, so the regionprops annotator can adopt them, after which they are
recomputed on every AddNode / UpdateNodeSeg, exactly like `area` and `pos`.

Columns needing information the graph does not carry are deliberately left
alone: the intensity columns need the raw image, and scaled_x/y/z need a voxel
size, which varies per microscope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.annotators import RegionpropsAnnotator

if TYPE_CHECKING:
    from funtracks.data_model import Tracks
    from funtracks.features import Feature

NUM_PIXELS_KEY = "num_pixels"
EQUIV_DIAMETER_KEY = "equivalent_diameter_area"

# Fork-owned features, each mapped onto the regionprops attribute that computes
# it. voxel_count is funtracks' own ExtendedRegionProperties property;
# equivalent_diameter_area comes from skimage.
EXTRA_FEATURES: dict[str, tuple[Feature, str]] = {
    NUM_PIXELS_KEY: (
        {
            "feature_type": "node",
            "value_type": "int",
            "num_values": 1,
            "display_name": "Num Pixels",
            "default_value": None,
        },
        "voxel_count",
    ),
    EQUIV_DIAMETER_KEY: (
        {
            "feature_type": "node",
            "value_type": "float",
            "num_values": 1,
            "display_name": "Equivalent Diameter",
            "default_value": None,
        },
        "equivalent_diameter_area",
    ),
}


class MaskFeatureAnnotator(RegionpropsAnnotator):
    """A RegionpropsAnnotator that also owns the mask-only pipeline columns.

    The extra features are registered after construction rather than by
    overriding the private _define_features, whose signature differs between
    funtracks versions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, (feature, regionprops_attr) in EXTRA_FEATURES.items():
            self.all_features[key] = (feature, False)
            self.regionprops_names[key] = regionprops_attr


def attach_mask_features(tracks: Tracks, recompute: bool = False) -> list[str]:
    """Adopt the mask-only measurement columns that `tracks` already carries.

    Swaps the plain RegionpropsAnnotator in the registry for one that also owns
    num_pixels and equivalent_diameter_area, then enables whichever of the two
    the loaded data actually has. Only existing columns are adopted, so no new
    column appears for data that never had them.

    Args:
        tracks: The tracks to annotate. Without a segmentation the regionprops
            annotator cannot compute anything, so nothing is done.
        recompute: Whether to recompute the adopted columns for every existing
            node. False keeps the values that came from the pipeline and only
            fills nodes edited from now on. True overwrites all of them with
            funtracks' own measurements.

    Returns:
        The feature keys that were adopted.
    """
    annotators = getattr(tracks, "annotators", None)
    if annotators is None or tracks.segmentation is None:
        return []

    keys = [key for key in EXTRA_FEATURES if key in tracks.features]
    if not keys:
        return []

    index = next(
        (i for i, a in enumerate(annotators) if type(a) is RegionpropsAnnotator),
        None,
    )
    if index is not None:
        try:
            replacement = MaskFeatureAnnotator(
                tracks, pos_key=annotators[index].pos_key
            )
        except TypeError:
            replacement = MaskFeatureAnnotator(tracks)
        # Carry over which features were already switched on
        for key, value in annotators[index].all_features.items():
            if key in replacement.all_features:
                replacement.all_features[key] = value
        annotators[index] = replacement
    elif not any(isinstance(a, MaskFeatureAnnotator) for a in annotators):
        # No regionprops annotator to extend, and ours is not already in place
        return []

    tracks.enable_features(keys, recompute=recompute)
    return keys
