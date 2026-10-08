"""The spindle_* node features and how they are written onto the tracks.

Like membrane/features.py: registering the keys with `Tracks.add_feature` makes
them show up in the table and makes `write_to_geff` save them as node props.
Only geometric / localisation values are stored; intensity-derived values of
Spindle3D (SNR, tubulin mean, thresholds, chromatin dilation) are not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


_NAN = float("nan")

SPINDLE_FEATURES: dict[str, Feature] = {
    # status and stage
    "spindle_status": _feature("str", "Spindle status", "not_measured"),
    "spindle_stage": _feature("str", "Spindle stage", ""),
    "spindle_stage_rule": _feature("str", "Spindle stage rule", ""),
    "spindle_is_metaphase": _feature("int", "Spindle metaphase (1/0)", -1),
    "spindle_frames_before_division": _feature("int", "Frames before division", 0),
    # Spindle3D geometry
    "spindle_length_um": _feature("float", "Spindle length (um)", _NAN),
    "spindle_volume_um3": _feature("float", "Spindle volume (um3)", _NAN),
    "spindle_width_avg_um": _feature("float", "Spindle width avg (um)", _NAN),
    "spindle_width_min_um": _feature("float", "Spindle width min (um)", _NAN),
    "spindle_width_max_um": _feature("float", "Spindle width max (um)", _NAN),
    "spindle_aspect_ratio": _feature("float", "Spindle aspect ratio", _NAN),
    "spindle_angle_deg": _feature("float", "Spindle tilt to coverslip (deg)", _NAN),
    "spindle_plate_width_um": _feature("float", "Plate width (um)", _NAN),
    "spindle_plate_length_um": _feature("float", "Plate length (um)", _NAN),
    "spindle_chromatin_volume_um3": _feature("float", "Chromatin volume (um3)", _NAN),
    # axis and position (um of the original frame)
    "spindle_azimuth_deg": _feature("float", "Spindle azimuth (deg)", _NAN),
    "spindle_axis_dna_offset_deg": _feature(
        "float", "Axis vs plate normal (deg)", _NAN
    ),
    "spindle_axis_z": _feature("float", "Spindle axis z", _NAN),
    "spindle_axis_y": _feature("float", "Spindle axis y", _NAN),
    "spindle_axis_x": _feature("float", "Spindle axis x", _NAN),
    "spindle_center_z_um": _feature("float", "Spindle centre z (um)", _NAN),
    "spindle_center_y_um": _feature("float", "Spindle centre y (um)", _NAN),
    "spindle_center_x_um": _feature("float", "Spindle centre x (um)", _NAN),
    # stage classification metrics
    "spindle_plate_flatness": _feature("float", "Chromatin short/long axis", _NAN),
    "spindle_plate_disc": _feature("float", "Chromatin mid/long axis", _NAN),
    "spindle_split_ratio": _feature("float", "Chromatin 2nd/largest mass", _NAN),
    "spindle_plate_rim_ratio": _feature(
        "float", "Plate length / chromatin extent", _NAN
    ),
    # centrosomes
    "spindle_n_centrosomes": _feature("int", "Centrosomes detected", -1),
    "spindle_centrosome1_z_um": _feature("float", "Centrosome 1 z (um)", _NAN),
    "spindle_centrosome1_y_um": _feature("float", "Centrosome 1 y (um)", _NAN),
    "spindle_centrosome1_x_um": _feature("float", "Centrosome 1 x (um)", _NAN),
    "spindle_centrosome2_z_um": _feature("float", "Centrosome 2 z (um)", _NAN),
    "spindle_centrosome2_y_um": _feature("float", "Centrosome 2 y (um)", _NAN),
    "spindle_centrosome2_x_um": _feature("float", "Centrosome 2 x (um)", _NAN),
    "spindle_centrosome_distance_um": _feature(
        "float", "Centrosome distance (um)", _NAN
    ),
    "spindle_centrosome_axis_z": _feature("float", "Centrosome axis z", _NAN),
    "spindle_centrosome_axis_y": _feature("float", "Centrosome axis y", _NAN),
    "spindle_centrosome_axis_x": _feature("float", "Centrosome axis x", _NAN),
    "spindle_axis_vs_centrosome_deg": _feature(
        "float", "Axis vs centrosome line (deg)", _NAN
    ),
    # QC flags
    "spindle_flag_pole_on_box_edge": _feature("int", "Pole on lateral box edge", -1),
    "spindle_flag_axis_vs_plate": _feature(
        "int", "Axis > 10 deg from plate normal", -1
    ),
}

FEATURE_DEFAULTS: dict = {k: f["default_value"] for k, f in SPINDLE_FEATURES.items()}

_CASTS = {"float": float, "int": int, "str": str}


def add_spindle_features(tracks: Tracks, rows: list[dict]) -> None:
    """Register the spindle feature keys and write the per-node values.

    rows: one dict per measured node with "node" and the feature keys (missing
    keys get the default). Nodes without a row keep the defaults (NaN, -1,
    "not_measured").
    """
    for key, feature in SPINDLE_FEATURES.items():
        if key not in tracks.features:
            tracks.add_feature(key, feature)
    if not rows:
        return
    nodes = [int(r["node"]) for r in rows]
    for key, feature in SPINDLE_FEATURES.items():
        cast = _CASTS[feature["value_type"]]
        values = []
        for r in rows:
            v = r.get(key, feature["default_value"])
            values.append(cast(v) if v is not None else feature["default_value"])
        tracks._set_nodes_attr(nodes, key, values)
