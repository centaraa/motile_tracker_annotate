"""Measure spindles of tracked nuclei before division and store them as features.

Usage:
    python scripts/spindle_features.py TRACKS TUBULIN DNA OUT.geff
        [--spacing 1,0.26,0.26] [--max-frame T] [--timepoints-before 6]
        [--correction ignore|none] [--threshold-exclude-um 1.0]
        [--pole-profile-radius java|um] [--half-size-um 20] [--z-extra-um 4]
        [--workers N] [--qc-dir DIR | --no-qc]

TRACKS is a geff store (corrected tracks; node id = label in the segmentation).
TUBULIN (microtubules, e.g. 488) and DNA (e.g. 561) are zarr arrays, tiffs or
folders of one tiff per timepoint, with the same shape as the segmentation.
OUT.geff gets a copy of the tracks with the spindle_* node features; next to it
<stem>_spindle.csv (one row per candidate), <stem>_params.json (read by
spindle_view.py), <stem>_threshold_diagnostic.csv and, unless --no-qc, QC PNGs
in --qc-dir (default <out dir>/qc_<stem>, one per candidate + overview.png).
See src/motile_tracker/spindle/README.md.
"""

import argparse
import csv
import functools
import json
import time
from pathlib import Path

from funtracks.import_export import import_from_geff

from motile_tracker.import_export.geff_io import write_geff_over
from motile_tracker.membrane.io import read_label_file
from motile_tracker.spindle import SpindleParams, add_spindle_features, measure_tracks

print = functools.partial(print, flush=True)  # noqa: A001


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("tracks", type=Path)
    p.add_argument("tubulin", type=Path)
    p.add_argument("dna", type=Path)
    p.add_argument("out_geff", type=Path)
    p.add_argument(
        "--spacing",
        default=None,
        help="voxel size z,y,x in um (default: from the tracks)",
    )
    p.add_argument(
        "--max-frame", type=int, default=None, help="only divisions up to this frame"
    )
    p.add_argument(
        "--timepoints-before", type=int, default=6, help="frames per division"
    )
    p.add_argument("--correction", choices=["ignore", "none"], default="ignore")
    p.add_argument(
        "--threshold-exclude-um", type=float, default=SpindleParams.threshold_exclude_um
    )
    p.add_argument("--pole-profile-radius", choices=["java", "um"], default="java")
    p.add_argument("--half-size-um", type=float, default=SpindleParams.half_size_um)
    p.add_argument("--z-extra-um", type=float, default=SpindleParams.z_extra_um)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--qc-dir", type=Path, default=None)
    p.add_argument(
        "--no-qc", action="store_true", help="no QC PNGs (no matplotlib needed)"
    )
    args = p.parse_args(argv)

    in_geff, out_geff = args.tracks.resolve(), args.out_geff.resolve()
    if in_geff == out_geff:
        raise SystemExit("OUT.geff must differ from the input")
    tracks = import_from_geff(in_geff)
    if args.spacing:
        spacing = tuple(float(v) for v in args.spacing.split(","))
    else:
        scale = tracks.scale or [1.0] * tracks.segmentation.ndim
        spacing = tuple(float(s) for s in scale[1:])
        if all(s == 1.0 for s in spacing):
            print("warning: the tracks store voxel size 1; pass --spacing z,y,x in um")
    params = SpindleParams(
        spacing=spacing,
        correction=args.correction,
        threshold_exclude_um=args.threshold_exclude_um,
        pole_profile_radius=args.pole_profile_radius,
        half_size_um=args.half_size_um,
        z_extra_um=args.z_extra_um,
    )
    run = out_geff.stem
    out_dir = out_geff.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = None if args.no_qc else (args.qc_dir or out_dir / f"qc_{run}")
    if qc_dir is not None:
        from motile_tracker.spindle.qc import contact_sheet, draw_qc

        qc_dir.mkdir(parents=True, exist_ok=True)
        for old in qc_dir.glob("*.png"):
            old.unlink()
    (out_dir / f"{run}_params.json").write_text(
        json.dumps(
            {
                "params": params.to_dict(),
                "input_geff": str(in_geff),
                "tubulin": str(args.tubulin),
                "dna": str(args.dna),
                "max_frame": args.max_frame,
                "timepoints_before": args.timepoints_before,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tubulin = read_label_file(args.tubulin)
    dna = read_label_file(args.dna)
    t0 = time.perf_counter()
    rows, diag, thumbs = [], [], []
    for row, case, res in measure_tracks(
        tracks,
        tubulin,
        dna,
        params,
        max_frame=args.max_frame,
        timepoints_before=args.timepoints_before,
        workers=args.workers,
    ):
        rows.append(row)
        if res.get("diag"):
            diag.append({"node": row["node"], "t": row["t"], **res["diag"]})
        if res.get("traceback"):
            (out_dir / "logs").mkdir(exist_ok=True)
            (out_dir / "logs" / f"{run}_node{row['node']}_t{row['t']}.txt").write_text(
                res["traceback"], encoding="utf-8"
            )
        if qc_dir is not None and case:
            tag = row["spindle_status"].split(":")[0]
            draw_qc(
                case,
                res,
                row,
                params.spacing,
                qc_dir / f"node{row['node']}_t{row['t']}_{tag}.png",
                thumbs,
            )
        print(
            f"  node {row['node']} t{row['t']} -{row['spindle_frames_before_division']}: "
            f"{row['spindle_status']} | {row.get('spindle_stage', '')} | "
            f"L={row['spindle_length_um']:.1f} W={row['spindle_width_avg_um']:.1f} "
            f"V={row['spindle_volume_um3']:.0f} vsN={row['spindle_axis_dna_offset_deg']:.1f} "
            f"nC={row['spindle_n_centrosomes']}"
        )

    rows.sort(key=lambda r: (r["t"], r["node"]))
    if qc_dir is not None:
        order = {r["node"]: i for i, r in enumerate(rows)}
        thumbs.sort(key=lambda th: order.get(int(th[0].split()[0]), 0))
        contact_sheet(thumbs, qc_dir / "overview.png")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out_dir / f"{run}_spindle.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if diag:
        with open(
            out_dir / f"{run}_threshold_diagnostic.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as fh:
            w = csv.DictWriter(fh, fieldnames=list(diag[0]))
            w.writeheader()
            w.writerows(sorted(diag, key=lambda r: (r["t"], r["node"])))
    add_spindle_features(tracks, rows)
    write_geff_over(tracks, out_geff)
    n_ok = sum(r["spindle_status"] == "ok" for r in rows)
    n_meta = sum(
        r["spindle_status"] == "ok" and r.get("spindle_is_metaphase") == 1 for r in rows
    )
    print(
        f"[{run}] {n_ok} ok ({n_meta} metaphase), {len(rows) - n_ok} failed of {len(rows)} "
        f"candidates; {time.perf_counter() - t0:.0f}s; geff {out_geff}"
    )


if __name__ == "__main__":
    main()
