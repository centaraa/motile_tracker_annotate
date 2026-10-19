"""Membrane labels inside a geff store (written by DestiNuc, shown here)."""

import numpy as np
import zarr

from motile_tracker.import_export.geff_io import write_geff_over
from motile_tracker.membrane.io import (
    MEMBRANE_ARRAY,
    create_label_store,
    get_membrane,
    load_membrane_labels,
    read_label_file,
    save_membrane_labels,
    set_membrane,
)


def _labels(shape=(2, 20, 30, 30)):
    lab = np.zeros(shape, np.uint32)
    lab[0, 5:15, 5:25, 5:25] = 1
    lab[1, 2:12, 3:13, 3:13] = 2
    lab[1, 2:12, 15:28, 15:28] = 3
    return lab


def test_read_label_file_tiff_folder(tmp_path):
    import tifffile

    lab = _labels()
    for t in range(lab.shape[0]):
        tifffile.imwrite(tmp_path / f"frame_{t:03d}.tif", lab[t])
    stack = read_label_file(tmp_path)
    assert tuple(stack.shape) == lab.shape
    np.testing.assert_array_equal(np.asarray(stack[1]), lab[1])


def test_geff_round_trip_keeps_labels_and_metadata(solution_tracks_3d, tmp_path):
    lab = _labels((2, 100, 100, 100))
    path = tmp_path / "out.geff"
    write_geff_over(solution_tracks_3d, path)
    assert load_membrane_labels(path) is None
    save_membrane_labels(path, lab)
    np.testing.assert_array_equal(np.asarray(load_membrane_labels(path)), lab)
    meta = zarr.open_group(str(path), mode="r").attrs["geff"]
    assert any(r.get("path") == MEMBRANE_ARRAY for r in meta["related_objects"])
    # re-saving the tracks (e.g. from the GUI) keeps the labels
    write_geff_over(solution_tracks_3d, path)
    save_membrane_labels(path, load_membrane_labels(path))
    np.testing.assert_array_equal(np.asarray(load_membrane_labels(path)), lab)


def test_zarr_source_is_copied_file_by_file(solution_tracks_3d, tmp_path):
    lab = _labels((2, 100, 100, 100))
    store = create_label_store(tmp_path / "m.zarr", lab.shape)
    store[...] = lab
    path = tmp_path / "out.geff"
    write_geff_over(solution_tracks_3d, path)
    save_membrane_labels(path, store)
    src = sorted(f.name for f in (tmp_path / "m.zarr").iterdir())
    assert sorted(f.name for f in (path / MEMBRANE_ARRAY).iterdir()) == src
    np.testing.assert_array_equal(np.asarray(load_membrane_labels(path)), lab)


def test_tracks_carry_the_labels(solution_tracks_3d):
    lab = _labels((2, 100, 100, 100))
    assert get_membrane(solution_tracks_3d) is None
    set_membrane(solution_tracks_3d, lab)
    assert get_membrane(solution_tracks_3d) is lab
