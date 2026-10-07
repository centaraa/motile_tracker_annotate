"""Write the Cellpose input for membrane segmentation: raw image + membrane score.

Usage:
    python scripts/export_membrane_blend.py TRACKS RAW OUT_DIR
        [--spacing 1,0.26,0.26] [--start-frame T] [--max-frames N]
        [--workers 8] [--overwrite]

TRACKS is the corrected geff (or a motile run folder), RAW the raw membrane
image (a folder of one tiff per timepoint, a tiff or zarr). For every frame,
OUT_DIR gets one uint16 tiff: half the normalised raw image, half the membrane
score (a sheet filter that keeps membranes and drops spindle fibres, with the
spindle of mitotic cells blanked using the tracks). Files keep the raw file
names when RAW is a tiff folder, so DestiNuc's cell_seg.py pairs them with the
nuclei channel by timepoint; point its membrane_dir at OUT_DIR.

Then segment OUT_DIR with Cellpose and run merge_membranes.py on the masks
with --raw RAW, which refines the merged boundaries on the same score.
"""

import argparse
import functools
import time
from pathlib import Path

import numpy as np
import tifffile
from funtracks.import_export import import_from_geff

from motile_tracker.membrane.io import TiffStack, read_label_file
from motile_tracker.membrane.score import ScoreParams, blend_frame, mitotic_nodes

print = functools.partial(print, flush=True)  # noqa: A001


def _job(args):
    t, raw, nuclei, spacing, mitotic, params, path = args
    tifffile.imwrite(
        path, blend_frame(raw, nuclei, spacing, mitotic, params), compression="zlib"
    )
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("tracks", type=Path)
    p.add_argument("raw", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--spacing", default=None, help="z,y,x in um (default: tracks)")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--workers", type=int, default=1, help="~3 GB RAM each")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    geff = args.tracks
    if not (geff / "nodes").exists() and (geff / "tracks.geff" / "nodes").exists():
        geff = geff / "tracks.geff"
    tracks = import_from_geff(geff)
    raw = read_label_file(args.raw)
    seg_shape = tuple(tracks.segmentation.shape)
    if tuple(raw.shape)[1:] != seg_shape[1:]:
        raise SystemExit(f"raw {tuple(raw.shape)} vs nuclei {seg_shape}")
    if args.spacing:
        spacing = tuple(float(v) for v in args.spacing.split(","))
    else:
        spacing = tuple(float(s) for s in (tracks.scale or [1.0] * 4)[1:])
    params = ScoreParams()
    start = args.start_frame
    stop = min(raw.shape[0], seg_shape[0])
    if args.max_frames is not None:
        stop = min(stop, start + args.max_frames)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def out_path(t):
        if isinstance(raw, TiffStack):
            return args.out_dir / raw.files[t].name
        return args.out_dir / f"blend_T{t + 1:04d}.tif"

    def job(t):
        nuclei = np.asarray(tracks.segmentation[t]).astype(np.uint32)
        nodes = [int(n) for n in np.unique(nuclei) if n]
        mitotic = mitotic_nodes(
            tracks, nodes, params.mitotic_before, params.mitotic_after
        )
        return t, np.asarray(raw[t]), nuclei, spacing, mitotic, params, out_path(t)

    frames = [
        t for t in range(start, stop) if args.overwrite or not out_path(t).exists()
    ]
    print(f"{len(frames)} frames to write to {args.out_dir} (voxel {spacing} um)")
    t0 = time.time()
    if args.workers <= 1:
        for i, t in enumerate(frames, 1):
            _job(job(t))
            print(f"frame {i}/{len(frames)}  {time.time() - t0:.0f} s")
        return

    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        running = set()
        for t in frames:
            running.add(pool.submit(_job, job(t)))
            if len(running) < args.workers:
                continue
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                fut.result()
                done += 1
                print(f"frame {done}/{len(frames)}  {time.time() - t0:.0f} s")
        for fut in running:
            fut.result()
            done += 1
            print(f"frame {done}/{len(frames)}  {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
