from unittest.mock import MagicMock

import numpy as np
import pytest

from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
from motile_tracker.import_export.geff_io import write_geff_over
from motile_tracker.membrane.features import (
    add_membrane_features,
    compute_membrane_features,
)
from motile_tracker.membrane.io import (
    MEMBRANE_ARRAY,
    load_membrane_labels,
    save_membrane_labels,
)


@pytest.fixture(autouse=True)
def clear_viewer_layers(viewer):
    yield
    viewer.layers.clear()


@pytest.fixture
def membrane_geff(solution_tracks_3d, tmp_path):
    membrane = np.zeros((2, 100, 100, 100), dtype=np.uint32)
    membrane[0, 25:76, 25:76, 25:76] = 5
    membrane[1, 5:80, 30:70, 25:95] = 9
    merged, feats = compute_membrane_features(solution_tracks_3d, membrane)
    add_membrane_features(solution_tracks_3d, feats)
    path = tmp_path / "with_membrane.geff"
    write_geff_over(solution_tracks_3d, path)
    save_membrane_labels(path, merged)
    return path, merged


def test_load_shows_membrane_overlay_and_save_keeps_it(viewer, membrane_geff, tmp_path):
    path, merged = membrane_geff
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_list = tracks_viewer.tracks_list
    tracks_list.dropdown_menu.setCurrentText("Tracks (geff)")
    tracks_list.file_dialog.exec_ = MagicMock(return_value=True)
    tracks_list.file_dialog.selectedFiles = MagicMock(return_value=[str(path)])

    tracks_list.load_tracks()

    names = [layer.name for layer in viewer.layers]
    assert "with_membrane_membrane" in names
    assert "with_membrane_seg" in names
    layer = viewer.layers["with_membrane_membrane"]
    assert not layer.editable
    np.testing.assert_array_equal(np.asarray(layer.data), merged)

    out_dir = tmp_path / "saved"
    tracks_list.save_dir_line.setText(str(out_dir))
    tracks_list.save_name_line.setText("resaved")
    tracks_list.save_tracks(tracks_list.tracks_list.item(0))

    out = out_dir / "resaved.geff"
    assert (out / MEMBRANE_ARRAY).exists()
    np.testing.assert_array_equal(np.asarray(load_membrane_labels(out)), merged)


def test_no_overlay_without_membrane(viewer, solution_tracks_3d):
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.tracks_list.add_tracks(solution_tracks_3d, "plain")
    assert not any(layer.name.endswith("_membrane") for layer in viewer.layers)
