"""Read raw membrane labels and keep merged ones inside a saved geff store.

The merged labels are written as a zarr array *inside* the geff store, next to
`nodes`/`edges`, the same way other tracks_saved listeners keep their extras
there (see `import_export/geff_io.py`): rewriting the geff only replaces
geff-controlled groups, so the array survives re-saves. The geff metadata is
rewritten on every save, though, so `save_membrane_labels` (re)registers the
array in `related_objects` each time it is called.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import zarr

MEMBRANE_ARRAY = "membrane_labels"
MEMBRANE_NODE_PROP = "membrane_id"

# Tracks carry no slot for a second label volume, so the merged membrane labels
# ride along as a plain attribute of the (promoted) tracks object the list holds.
_TRACKS_ATTR = "_motile_membrane_labels"


def get_membrane(tracks):
    """The merged membrane labels attached to `tracks`, or None."""
    return getattr(tracks, _TRACKS_ATTR, None)


def set_membrane(tracks, labels) -> None:
    """Attach merged membrane labels (t, [z], y, x) to `tracks`."""
    setattr(tracks, _TRACKS_ATTR, labels)


class TiffStack:
    """A folder of per-timepoint tiffs, read lazily one frame at a time.

    Frames are ordered by file name, so names must sort in time order (e.g.
    zero-padded `...T0001...`). Only integer indexing is supported.
    """

    def __init__(self, files: list[Path]):
        import tifffile

        self.files = files
        with tifffile.TiffFile(files[0]) as tif:
            series = tif.series[0]
            frame_shape, self.dtype = tuple(series.shape), series.dtype
        self.shape = (len(files), *frame_shape)
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, t: int) -> np.ndarray:
        import tifffile

        return tifffile.imread(self.files[t])


def read_label_file(path: str | Path):
    """Open raw membrane labels from zarr, a tiff file or a folder of tiffs.

    Zarr arrays and tiff folders are returned lazily; a zarr group must hold
    exactly one array, and a tiff folder one 3D (or 2D) frame per file.
    """
    path = Path(path)
    if path.is_dir() and not (path / ".zgroup").exists():
        files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in {".tif", ".tiff"}
        )
        if files:
            return TiffStack(files)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        return tifffile.imread(path)
    node = zarr.open(str(path), mode="r")
    if isinstance(node, zarr.Array):
        return node
    arrays = list(node.array_keys())
    if len(arrays) != 1:
        raise ValueError(
            f"{path} is a zarr group with arrays {arrays}; pass the array path"
        )
    return node[arrays[0]]


def _chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    # One chunk (one file) per frame: a frame is written and read in one go,
    # and a long movie stays at a few hundred files instead of tens of
    # thousands, which matters on network drives.
    return (1, *shape[1:])


def create_label_store(path: str | Path, shape, dtype=np.uint32) -> zarr.Array:
    """Create an empty, compressed zarr array on disk to stream merged frames into.

    It uses zarr format 2, like the geff stores written here, so that
    `save_membrane_labels` can copy it into a geff file by file.
    """
    return zarr.create_array(
        str(path),
        shape=tuple(shape),
        dtype=dtype,
        chunks=_chunks(tuple(shape)),
        fill_value=0,
        overwrite=True,
        zarr_format=2,
    )


def _array_dir(labels) -> Path | None:
    """The directory of a zarr array kept in a local store, else None."""
    if not isinstance(labels, zarr.Array):
        return None
    root = getattr(labels.store, "root", None)
    if root is None:
        return None
    return Path(root) / labels.path


def _same_array(labels, path: Path) -> bool:
    """Whether `labels` is the zarr array stored at `path`."""
    src = _array_dir(labels)
    if src is None or not src.exists() or not path.exists():
        return False
    try:
        return src.samefile(path)
    except OSError:
        return src.resolve() == path.resolve()


def save_membrane_labels(geff_path: str | Path, labels) -> Path:
    """Write merged labels into the geff store and register them in its metadata.

    A zarr source of the same format as the geff is copied file by file, with
    no decompressing and recompressing; anything else is copied frame by
    frame, so a lazy source never has to fit in memory. If `labels` already is
    the array in this store (tracks loaded from and saved back to the same
    geff), only the metadata is rewritten.

    Args:
        geff_path: A saved geff store (the directory holding `nodes`/`edges`).
        labels: Merged labels (t, [z], y, x); values are nucleus node ids.

    Returns:
        The path of the written array.
    """
    geff_path = Path(geff_path)
    root = zarr.open_group(str(geff_path), mode="a")
    target = geff_path / MEMBRANE_ARRAY
    src = _array_dir(labels)
    if _same_array(labels, target):
        pass
    elif src is not None and labels.metadata.zarr_format == root.metadata.zarr_format:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
    else:
        arr = root.create_array(
            MEMBRANE_ARRAY,
            shape=labels.shape,
            dtype=labels.dtype,
            chunks=_chunks(tuple(labels.shape)),
            fill_value=0,
            overwrite=True,
        )
        for t in range(labels.shape[0]):
            arr[t] = np.asarray(labels[t])

    attrs = dict(root.attrs)
    geff_meta = dict(attrs.get("geff", {}))
    related = [
        r
        for r in geff_meta.get("related_objects") or []
        if r.get("path") != MEMBRANE_ARRAY
    ]
    related.append(
        {"type": "labels", "path": MEMBRANE_ARRAY, "node_prop": MEMBRANE_NODE_PROP}
    )
    geff_meta["related_objects"] = related
    root.attrs["geff"] = geff_meta
    return geff_path / MEMBRANE_ARRAY


def load_membrane_labels(geff_path: str | Path):
    """The merged membrane labels kept in a geff store, or None if it has none."""
    geff_path = Path(geff_path)
    if not (geff_path / MEMBRANE_ARRAY).exists():
        return None
    root = zarr.open_group(str(geff_path), mode="r")
    return root[MEMBRANE_ARRAY]
