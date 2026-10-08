"""Show one node's spindle measurement in napari, recomputed on the fly.

Usage:
    python scripts/spindle_view.py OUT.geff TUBULIN DNA --node ID [--dry-run]
    python scripts/spindle_view.py OUT.geff TUBULIN DNA --best
    python scripts/spindle_view.py OUT.geff TUBULIN DNA --list

OUT.geff is written by spindle_features.py; the parameters are read from
<stem>_params.json next to it, so the recomputed result matches the batch run.
--best picks the ok metaphase node without QC flags and with two centrosomes
whose axis is closest to the centrosome line. --dry-run prints the layers and
values instead of opening napari.

Layers (um, at the crop's position in the frame): 488 (background-subtracted,
the image the measurement uses), 561 (other nuclei removed), nuclei
segmentation, spindle-threshold exclusion region and shell (contours), DNA
mask, spindle mask (with centrosomes), spindle axis, refined and initial poles,
detected centrosomes. The measured values are shown as a text overlay.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from funtracks.import_export import import_from_geff

from motile_tracker.membrane.io import read_label_file
from motile_tracker.spindle import SpindleParams, measure_node
from motile_tracker.spindle.core import ISO

SHOW_KEYS = [
    "spindle_status",
    "spindle_stage",
    "spindle_stage_rule",
    "spindle_frames_before_division",
    "spindle_length_um",
    "spindle_width_avg_um",
    "spindle_volume_um3",
    "spindle_aspect_ratio",
    "spindle_axis_dna_offset_deg",
    "spindle_angle_deg",
    "spindle_azimuth_deg",
    "spindle_plate_length_um",
    "spindle_plate_width_um",
    "spindle_plate_flatness",
    "spindle_plate_rim_ratio",
    "spindle_n_centrosomes",
    "spindle_centrosome_distance_um",
    "spindle_axis_vs_centrosome_deg",
    "spindle_flag_pole_on_box_edge",
    "spindle_flag_axis_vs_plate",
    "spindle_is_metaphase",
]


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, float | np.floating) else str(v)


def _rows(tracks):
    rows = []
    for n in tracks.graph_full.node_ids():
        st = tracks.get_node_attr(n, "spindle_status")
        if st is None or st == "not_measured":
            continue
        rows.append(
            {
                "node": int(n),
                "t": int(tracks.get_time(n)),
                **{k: tracks.get_node_attr(n, k) for k in SHOW_KEYS},
            }
        )
    rows.sort(key=lambda r: (r["t"], r["node"]))
    return rows


def _best(rows):
    ok = [
        r
        for r in rows
        if r["spindle_status"] == "ok" and r["spindle_is_metaphase"] == 1
    ]
    clean = [
        r
        for r in ok
        if r["spindle_flag_pole_on_box_edge"] == 0
        and r["spindle_flag_axis_vs_plate"] == 0
        and r["spindle_n_centrosomes"] == 2
        and np.isfinite(r["spindle_axis_vs_centrosome_deg"])
    ]
    if clean:
        return min(clean, key=lambda r: r["spindle_axis_vs_centrosome_deg"])["node"]
    if ok:
        return min(ok, key=lambda r: r["spindle_axis_dna_offset_deg"])["node"]
    raise SystemExit("no ok metaphase node in this geff")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("geff", type=Path)
    p.add_argument("tubulin", type=Path)
    p.add_argument("dna", type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--node", type=int)
    g.add_argument("--best", action="store_true")
    g.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print, do not open napari")
    args = p.parse_args(argv)

    tracks = import_from_geff(args.geff)
    if "spindle_status" not in tracks.features:
        raise SystemExit("the geff has no spindle_* features (run spindle_features.py)")
    rows = _rows(tracks)
    if args.list:
        cols = ["node", "t", *SHOW_KEYS[3:7], "spindle_stage", "spindle_status"]
        print("\t".join(cols))
        for r in rows:
            print("\t".join(_fmt(r[c]) for c in cols))
        return
    node = _best(rows) if args.best else args.node
    ppath = args.geff.parent / f"{args.geff.stem}_params.json"
    if ppath.exists():
        params = SpindleParams.from_dict(
            json.loads(ppath.read_text(encoding="utf-8"))["params"]
        )
    else:
        print(f"warning: {ppath} not found, using default parameters")
        params = SpindleParams()
    row, case, res = measure_node(
        tracks, node, read_label_file(args.tubulin), read_label_file(args.dna), params
    )
    vals = {k: tracks.get_node_attr(node, k) for k in SHOW_KEYS}
    sp = np.asarray(params.spacing, float)
    origin = np.asarray(case["meta"]["crop_lo_px"], float) * sp  # um of crop voxel 0
    qc = res.get("qc", {})
    iso = (ISO,) * 3
    layers = [
        (
            "image",
            case["a_b"],
            {"name": "488 (bg-subtracted, used)", "colormap": "green", "scale": sp},
        ),
        (
            "image",
            case["dna"],
            {
                "name": "561 (other nuclei removed)",
                "colormap": "magenta",
                "blending": "additive",
                "scale": sp,
            },
        ),
        (
            "labels",
            case["seg"],
            {"name": "nuclei segmentation", "visible": False, "scale": sp},
        ),
    ]
    for key, name, val in [
        ("threshold_exclusion", "threshold exclusion (DNA mask + margin)", 5),
        ("threshold_shell", "threshold shell (Otsu sampled here)", 6),
    ]:
        if key in qc:
            layers.append(
                (
                    "labels",
                    qc[key].astype(np.uint8) * val,
                    {"name": name, "scale": iso, "opacity": 0.7, "contour": 1},
                )
            )
    if "dna_mask" in qc:
        layers.append(
            (
                "labels",
                qc["dna_mask"].astype(np.uint8),
                {"name": "DNA mask", "scale": iso, "opacity": 0.4},
            )
        )
    if "spindle_mask" in qc:
        layers.append(
            (
                "labels",
                qc["spindle_mask"].astype(np.uint8) * 2,
                {
                    "name": "spindle mask (incl. centrosomes)",
                    "scale": iso,
                    "opacity": 0.35,
                },
            )
        )
    if "poles" in qc:
        P = qc["poles"] * ISO
        layers.append(
            (
                "shapes",
                [P],
                {
                    "name": "spindle axis",
                    "shape_type": "line",
                    "edge_color": "yellow",
                    "edge_width": 0.3,
                },
            )
        )
        layers.append(
            (
                "points",
                P,
                {"name": "refined poles", "size": 1.2, "face_color": "yellow"},
            )
        )
    if "init_poles" in qc:
        layers.append(
            (
                "points",
                qc["init_poles"] * ISO,
                {
                    "name": "initial poles (mask profile)",
                    "size": 0.8,
                    "face_color": "orange",
                    "visible": False,
                },
            )
        )
    if "centrosomes" in qc:
        layers.append(
            (
                "points",
                qc["centrosomes"] * ISO,
                {
                    "name": "centrosomes (detected)",
                    "size": 1.5,
                    "face_color": "red",
                    "symbol": "x",
                },
            )
        )
    title = (
        f"node {node} t{case['meta']['t']} | {vals['spindle_status']} | stage {vals['spindle_stage']} | "
        f"L {_fmt(vals['spindle_length_um'])} um, W {_fmt(vals['spindle_width_avg_um'])} um, "
        f"V {_fmt(vals['spindle_volume_um3'])} um3"
    )
    diag = res.get("diag", {})
    print(title)
    print(
        f"  params: {ppath if ppath.exists() else 'defaults'}; correction {params.correction}; "
        f"spindle threshold {diag.get('threshold_used', np.nan):.1f} "
        f"(plain Java rim {diag.get('threshold_java_rim', np.nan):.1f})"
    )
    print("  geff values:", {k: _fmt(v) for k, v in vals.items()})
    print(
        f"  recomputed: status {row['spindle_status']}, stage {row['spindle_stage']}, "
        f"L {row['spindle_length_um']:.2f}, V {row['spindle_volume_um3']:.0f}, "
        f"centrosomes {row['spindle_n_centrosomes']}"
    )
    if args.dry_run:
        for kind, data, kw in layers:
            print(f"  {kind:7s} {kw['name']:40s} shape={np.shape(data)}")
        return

    import napari

    viewer = napari.Viewer(title=title)
    for kind, data, kw in layers:
        kw = dict(kw)
        contour = kw.pop("contour", None)
        layer = getattr(viewer, f"add_{kind}")(data, translate=origin, **kw)
        if contour:
            layer.contour = contour  # not a constructor argument in napari 0.7
    viewer.dims.axis_labels = ["z (um)", "y (um)", "x (um)"]
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "\n".join(
        f"{k.replace('spindle_', '')}: {_fmt(vals[k])}" for k in SHOW_KEYS
    )
    viewer.text_overlay.font_size = 9
    napari.run()


if __name__ == "__main__":
    main()
