"""Merge raw membrane labels onto the tracked nuclei of a geff.

Usage:
    python scripts/merge_membranes.py IN.geff MEMBRANE LABELS.zarr
        [--out-geff OUT.geff] [--start-frame T] [--max-frames N] [--view]
        [--min-nucleus-overlap 0.5] [--min-shared-frac 0.2]
        [--min-contact-area 0] [--max-dist inf] [--min-cell-volume 0]
        [--division-window 1] [--no-split] [--no-embedded] [--no-fill]
        [--outline closed|convex_hull] [--closing-radius 6] [--keep-cavities]

MEMBRANE is a zarr array, a tiff, or a folder of one tiff per timepoint.
Merged frames are streamed into LABELS.zarr one at a time, so memory use stays
at a few frames. LABELS.zarr spans the full time range, with each frame at its
own index (frames not processed stay 0 and take no disk space). --out-geff also writes the tracks with the membrane_* node
features, plus a copy of the merged labels inside the geff store.

All lengths are in µm, using the voxel size stored with the tracks (voxels if
the geff stores none).
"""

import argparse
import functools
import time
from collections import Counter
from pathlib import Path

import numpy as np
from funtracks.import_export import import_from_geff

from motile_tracker.import_export.geff_io import write_geff_over
from motile_tracker.membrane import MembraneMergeParams
from motile_tracker.membrane.features import (
    add_membrane_features,
    compute_membrane_features,
)
from motile_tracker.membrane.io import (
    create_label_store,
    read_label_file,
    save_membrane_labels,
)

# Flush every line, so a log file is current while napari is still open.
print = functools.partial(print, flush=True)  # noqa: A001


def _lazy(frames, n, frame_shape, dtype):
    """A dask stack that reads frame t only when napari shows it."""
    import dask
    import dask.array as da

    return da.stack(
        [
            da.from_delayed(dask.delayed(frames)(t), frame_shape, dtype=dtype)
            for t in range(n)
        ]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("geff", type=Path)
    p.add_argument("membrane", type=Path)
    p.add_argument("labels_zarr", type=Path)
    p.add_argument("--out-geff", type=Path, default=None)
    p.add_argument("--start-frame", type=int, default=0, help="first frame index")
    p.add_argument("--max-frames", type=int, default=None, help="at most N frames")
    p.add_argument("--view", action="store_true", help="show result in napari")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="frames merged in parallel; each needs ~2-3 GB RAM",
    )
    p.add_argument("--min-nucleus-overlap", type=float, default=0.5)
    p.add_argument("--min-shared-frac", type=float, default=0.2)
    p.add_argument("--min-contact-area", type=float, default=0.0, help="µm²")
    p.add_argument("--max-dist", type=float, default=np.inf, help="µm")
    p.add_argument("--min-cell-volume", type=float, default=0.0, help="µm³")
    p.add_argument("--division-window", type=int, default=1, help="frames")
    p.add_argument("--no-split", action="store_true")
    p.add_argument("--no-embedded", action="store_true")
    p.add_argument("--no-fill", action="store_true")
    p.add_argument("--outline", choices=["closed", "convex_hull"], default="closed")
    p.add_argument(
        "--closing-radius", type=float, default=6.0, help="µm, ball for 'closed'"
    )
    p.add_argument("--keep-cavities", action="store_true")
    args = p.parse_args()

    params = MembraneMergeParams(
        min_nucleus_overlap=args.min_nucleus_overlap,
        min_shared_frac=args.min_shared_frac,
        min_contact_area_um2=args.min_contact_area,
        max_dist_um=args.max_dist,
        min_cell_volume_um3=args.min_cell_volume,
        use_watershed_split=not args.no_split,
        merge_embedded=not args.no_embedded,
        fill_gaps=not args.no_fill,
        outline=args.outline,
        closing_radius_um=args.closing_radius,
        fill_embryo_holes=not args.keep_cavities,
    )
    geff = args.geff
    # A motile run folder keeps its geff in tracks.geff; accept either.
    if not (geff / "nodes").exists() and (geff / "tracks.geff" / "nodes").exists():
        geff = geff / "tracks.geff"
    tracks = import_from_geff(geff)
    membrane = read_label_file(args.membrane)
    seg_shape = tuple(tracks.segmentation.shape)
    start = args.start_frame
    n = seg_shape[0] - start
    if args.max_frames is not None:
        n = min(args.max_frames, n)
    print(
        f"tracks {seg_shape}, membrane {tuple(membrane.shape)}, "
        f"frames {start}..{start + n - 1}"
    )
    print(f"voxel size (t, [z], y, x): {tracks.scale}")

    store = create_label_store(args.labels_zarr, seg_shape)
    t0 = time.time()
    merged, feats = compute_membrane_features(
        tracks,
        membrane,
        params=params,
        division_window=args.division_window,
        progress=lambda t, total: print(
            f"frame {t + 1}/{total}  {time.time() - t0:.0f} s", flush=True
        ),
        max_frames=n,
        out=store,
        start_frame=start,
        workers=args.workers,
    )
    add_membrane_features(tracks, feats)
    print("wrote", args.labels_zarr)
    print("QC:", dict(Counter(f["membrane_qc"] for f in feats.values())))

    if args.out_geff is not None:
        write_geff_over(tracks, args.out_geff)
        print("wrote", save_membrane_labels(args.out_geff, merged))

    if args.view:
        import napari

        frame_shape, n_all = seg_shape[1:], seg_shape[0]
        nuclei = _lazy(
            lambda t: np.asarray(tracks.segmentation[t]).astype(np.uint32),
            n_all,
            frame_shape,
            np.uint32,
        )
        raw = _lazy(
            lambda t: np.asarray(membrane[t]), n_all, frame_shape, membrane.dtype
        )
        viewer = napari.Viewer(title=f"membrane merge: {args.geff.name}")
        viewer.add_labels(nuclei, name="nuclei", scale=tracks.scale)
        viewer.add_labels(raw, name="membrane_raw", scale=tracks.scale, visible=False)
        layer = viewer.add_labels(
            merged, name="membrane_merged", scale=tracks.scale, opacity=0.6
        )
        # Spread consecutive node ids over the colour wheel so neighbouring
        # cells (whose ids often differ by 1) never look alike.
        layer.colormap = napari.utils.colormaps.label_colormap(
            49, seed=0.5, background_value=0
        )
        viewer.dims.set_current_step(0, start)
        napari.run()


if __name__ == "__main__":
    main()
