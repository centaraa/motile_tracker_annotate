"""QC images of the spindle measurement (optional; needs matplotlib).

matplotlib is imported lazily: it is only a transitive dependency (via napari).
"""

from __future__ import annotations

import numpy as np

from .core import ISO as _ISO


def _extent(shape2, sc2):
    (ny, nx), (sy, sx) = shape2, sc2
    return [-sx / 2, (nx - 0.5) * sx, (ny - 0.5) * sy, -sy / 2]


def _contour(ax, mask2d, sc, color, lw=1.0):
    if mask2d is None or not mask2d.any():
        return
    ax.contour(
        np.arange(mask2d.shape[1]) * sc[1],
        np.arange(mask2d.shape[0]) * sc[0],
        mask2d.astype(float),
        levels=[0.5],
        colors=color,
        linewidths=lw,
    )


def draw_qc(case: dict, res: dict, row: dict, spacing, png_path, thumbs=None) -> None:
    """One PNG per node: MIPs along z and y of 488, 561 and the masks/poles.

    thumbs: if a list, a thumbnail of the overlay panel is appended for
    `contact_sheet`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = np.asarray(spacing, float)
    qc = res.get("qc", {}) if res else {}
    ISO = qc.get("iso_voxel_um", _ISO)  # noqa: N806 - isotropic analysis voxel (um)
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    chans = [
        ("488 (bg-subtracted, used as is)", case["a_b"], "gray"),
        ("488 + threshold exclusion (orange) / shell (yellow)", case["tub"], "gray"),
        ("561 DNA (other nuclei removed)", case["dna"], "magma"),
        (
            (
                "overlay: DNA mask (magenta), spindle mask (cyan), threshold exclusion "
                "(orange),\nrefined poles (lime), centrosomes (red x)"
            ),
            case["tub"],
            "gray",
        ),
    ]
    views = [(0, "MIP along z (XY)", (1, 2)), (1, "MIP along y (XZ)", (0, 2))]
    for r, (pax, lab, keep) in enumerate(views):
        for c, (title, arr, cmap) in enumerate(chans):
            ax = axes[r, c]
            mip = arr.max(axis=pax)
            ext = _extent(mip.shape, sp[list(keep)])
            vmax = max(float(np.percentile(mip, 99.7)), 1e-3)
            ax.imshow(
                mip, cmap=cmap, vmin=0, vmax=vmax, extent=ext, interpolation="nearest"
            )
            ax.set_title(f"{title}\n{lab}", fontsize=8)
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)" if r == 0 else "z (um)")
            if c == 1:
                for key, col in (
                    ("threshold_exclusion", "orange"),
                    ("threshold_shell", "yellow"),
                ):
                    if key in qc:
                        _contour(ax, qc[key].max(axis=pax), (ISO, ISO), col, 0.8)
            if c == 3:
                if "dna_mask" in qc:
                    _contour(ax, qc["dna_mask"].max(axis=pax), (ISO, ISO), "magenta")
                elif case.get("stage_mask") is not None:
                    _contour(
                        ax, case["stage_mask"].max(axis=pax), sp[list(keep)], "magenta"
                    )
                if "spindle_mask" in qc:
                    _contour(ax, qc["spindle_mask"].max(axis=pax), (ISO, ISO), "cyan")
                if (
                    "threshold_exclusion" in qc
                    and case["meta"]["correction"] == "ignore"
                ):
                    _contour(
                        ax,
                        qc["threshold_exclusion"].max(axis=pax),
                        (ISO, ISO),
                        "orange",
                    )
                if "poles" in qc:
                    P = qc["poles"] * ISO
                    ax.plot(
                        P[:, keep[1]],
                        P[:, keep[0]],
                        "o-",
                        ms=7,
                        mfc="none",
                        mec="lime",
                        color="lime",
                        lw=0.8,
                        mew=2,
                    )
                if "centrosomes" in qc:
                    C = qc["centrosomes"] * ISO
                    ax.plot(C[:, keep[1]], C[:, keep[0]], "x", ms=9, color="red", mew=2)
            ax.set_xlim(ext[:2])
            ax.set_ylim(ext[2:])
    st = f"{row['spindle_status']} | stage: {row.get('spindle_stage', '')}"
    fig.suptitle(
        f"node {row['node']}  frame {row['t']}  frames before division "
        f"{row['spindle_frames_before_division']}  (division node {row['division_node']})\n"
        f"length {row['spindle_length_um']:.1f} um  width {row['spindle_width_avg_um']:.1f} um  "
        f"volume {row['spindle_volume_um3']:.0f} um3  vs plate normal "
        f"{row['spindle_axis_dna_offset_deg']:.1f} deg  centrosomes {row['spindle_n_centrosomes']}  "
        f"plate flatness {row['spindle_plate_flatness']:.2f}\nstatus: {st}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=70)
    if thumbs is not None:
        fig.canvas.draw()
        bb = axes[0, 3].get_window_extent()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        H = buf.shape[0]
        label = (
            f"{row['node']} t{row['t']} -{row['spindle_frames_before_division']} "
            f"{str(row.get('spindle_stage', '')).upper()}\n{row['spindle_status'][:40]}"
        )
        thumbs.append(
            (
                label,
                buf[int(H - bb.y1) : int(H - bb.y0), int(bb.x0) : int(bb.x1)].copy(),
            )
        )
    plt.close(fig)


def contact_sheet(thumbs, path, ncol: int = 8) -> None:
    """All thumbnails of `draw_qc` on one page."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not thumbs:
        return
    nrow = int(np.ceil(len(thumbs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.9 * nrow))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (lab, img) in zip(axes.ravel(), thumbs, strict=False):
        ax.imshow(img)
        ax.set_title(lab, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
