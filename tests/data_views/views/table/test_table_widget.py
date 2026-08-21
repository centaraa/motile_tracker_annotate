import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QSpinBox,
    QStyleOptionViewItem,
    QWidget,
)

from motile_tracker.data_views.views.table.custom_table_widget import (
    AnnotationDelegate,
    ColoredTableWidget,
    TrackTableModel,
    is_checked,
    parse_bool,
)
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


@pytest.fixture(autouse=True)
def clear_viewer_layers(viewer):
    """Clear viewer layers between tests."""
    yield
    viewer.layers.clear()


@pytest.fixture
def setup_tracks_viewer(viewer, solution_tracks_2d):
    """Create a TracksViewer with tracks loaded."""
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks=solution_tracks_2d, name="test")

    return viewer, tracks_viewer


@pytest.fixture
def colored_table_widget(qtbot, setup_tracks_viewer):
    viewer, tracks_viewer = setup_tracks_viewer

    # Build dataframe from tracks
    nodes = tracks_viewer.tracks.graph.node_ids()

    df = pd.DataFrame(
        {
            "ID": nodes,
            "value": list(range(len(nodes))),
        }
    )

    tracks_viewer.track_df = df

    widget = ColoredTableWidget(viewer)
    qtbot.addWidget(widget)

    return widget, tracks_viewer


def test_table_population(colored_table_widget):
    widget, _ = colored_table_widget

    table = widget._table_widget

    assert table.model().rowCount() > 0
    assert table.model().columnCount() >= 1


def test_table_selection_updates_tracksviewer(colored_table_widget, qtbot):
    widget, tracks_viewer = colored_table_widget
    table = widget._table_widget

    first_row_node = widget._table["ID"][0]

    with qtbot.waitSignal(tracks_viewer.selected_nodes.selection_updated, timeout=1000):
        table.selectRow(0)

    assert first_row_node in tracks_viewer.selected_nodes.as_list


def test_tracksviewer_selection_updates_table(colored_table_widget, qtbot):
    widget, tracks_viewer = colored_table_widget
    table = widget._table_widget

    # Select first two nodes in viewer
    nodes = widget._table["ID"][:2].tolist()
    tracks_viewer.selected_nodes.add_list(nodes, append=False)

    qtbot.wait(50)

    selected_rows = sorted({index.row() for index in table.selectedIndexes()})

    assert selected_rows == [0, 1]


def test_no_infinite_selection_loop(colored_table_widget, qtbot):
    widget, tracks_viewer = colored_table_widget
    table = widget._table_widget

    spy_count = {"calls": 0}

    def spy():
        spy_count["calls"] += 1

    tracks_viewer.selected_nodes.selection_updated.connect(spy)

    table.selectRow(0)
    qtbot.wait(50)

    assert spy_count["calls"] == 1


def test_center_from_table_triggers_viewer(colored_table_widget, qtbot):
    widget, tracks_viewer = colored_table_widget
    table = widget._table_widget

    index = table.model().index(0, 0)

    with qtbot.waitSignal(tracks_viewer.center_node, timeout=1000):
        widget.center_node(index)


def test_center_from_tracksviewer_scrolls_table(colored_table_widget, qtbot):
    widget, _ = colored_table_widget
    table = widget._table_widget

    # Force small viewport so only 2 rows fit
    row_height = 30
    table.setFixedHeight(row_height * 2)
    for i in range(table.model().rowCount()):
        table.setRowHeight(i, row_height)

    widget.show()
    qtbot.wait(50)
    qtbot.waitExposed(widget)  # Ensure widget is rendered
    qtbot.wait(50)

    table.verticalScrollBar().setValue(0)
    qtbot.wait(50)

    node = widget._table["ID"][5]
    target_row_index = widget._find_row(ID=node)

    # Scroll to node and check that it is visible
    widget.scroll_to_node(node)
    qtbot.wait(50)

    QApplication.processEvents()
    row_rect = table.visualRect(table.model().index(target_row_index, 0))
    viewport_rect = table.viewport().rect()

    assert row_rect.intersects(viewport_rect), (
        "The target row should be visible after scroll"
    )

    assert widget._syncing is False


def test_sort_preserves_functionality(colored_table_widget, qtbot):
    widget, tracks_viewer = colored_table_widget

    widget._sort_table(0)
    qtbot.wait(50)

    # Try selecting again after sort
    widget._table_widget.selectRow(0)
    qtbot.wait(50)

    assert len(tracks_viewer.selected_nodes.as_list) >= 1


def test_table_widget_keybinds(colored_table_widget, qtbot):
    """Test all keyboard shortcuts in the table widget.

    The table widget supports keybinds that are delegated to tracks_viewer:
    - D / Delete: delete_node
    - A: create_edge
    - B: delete_edge
    - S: swap_nodes
    - Z: undo
    - R: redo
    - Escape: deselect
    - E: restore_selection
    """
    from unittest.mock import MagicMock

    widget, tracks_viewer = colored_table_widget
    table_widget = widget._table_widget

    # Set focus to the table widget so it receives key events
    table_widget.setFocus()

    # Mock all tracks_viewer methods to verify they're called
    delete_mock = MagicMock()
    tracks_viewer.delete_node = delete_mock

    create_edge_mock = MagicMock()
    tracks_viewer.create_edge = create_edge_mock

    delete_edge_mock = MagicMock()
    tracks_viewer.delete_edge = delete_edge_mock

    swap_mock = MagicMock()
    tracks_viewer.swap_nodes = swap_mock

    undo_mock = MagicMock()
    tracks_viewer.undo = undo_mock

    redo_mock = MagicMock()
    tracks_viewer.redo = redo_mock

    deselect_mock = MagicMock()
    tracks_viewer.deselect = deselect_mock

    restore_mock = MagicMock()
    tracks_viewer.restore_selection = restore_mock

    # Test D key calls delete_node
    qtbot.keyPress(table_widget, Qt.Key_D)
    delete_mock.assert_called_once()

    # Test Delete key also calls delete_node
    delete_mock.reset_mock()
    qtbot.keyPress(table_widget, Qt.Key_Delete)
    delete_mock.assert_called_once()

    # Test A key calls create_edge
    qtbot.keyPress(table_widget, Qt.Key_A)
    create_edge_mock.assert_called_once()

    # Test B key calls delete_edge
    qtbot.keyPress(table_widget, Qt.Key_B)
    delete_edge_mock.assert_called_once()

    # Test S key calls swap_nodes
    qtbot.keyPress(table_widget, Qt.Key_S)
    swap_mock.assert_called_once()

    # Test Z key calls undo
    qtbot.keyPress(table_widget, Qt.Key_Z)
    undo_mock.assert_called_once()

    # Test R key calls redo
    qtbot.keyPress(table_widget, Qt.Key_R)
    redo_mock.assert_called_once()

    # Test Escape key calls deselect
    qtbot.keyPress(table_widget, Qt.Key_Escape)
    deselect_mock.assert_called_once()

    # Test E key calls restore_selection
    qtbot.keyPress(table_widget, Qt.Key_E)
    restore_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Manual annotation columns
#
# The tests above are upstream's and only prove that the widget builds,
# populates, selects and sorts. The tests below cover this fork's own feature:
# annotating nodes directly in the table. Without them, an upstream merge can
# silently break annotations while the suite stays green.
# ---------------------------------------------------------------------------

BOOL_SPEC = {"key": "checked", "value_type": "bool"}
INT_SPEC = {"key": "score", "value_type": "int"}


def _column_index(model, name: str) -> int:
    """Return the index of the column displayed under the given header."""
    for col in range(model.columnCount()):
        if model.headerData(col, Qt.Horizontal) == name:
            return col
    raise AssertionError(f"column {name!r} not found")


def _graph(tracks):
    """Return the tracks graph, preferring the non-deprecated accessor."""
    if hasattr(tracks, "graph_solution"):
        return tracks.graph_solution
    return tracks.graph


@pytest.fixture
def annotation_model():
    """A model holding one bool and one int manual annotation column."""
    model = TrackTableModel()
    model.set_table(
        {
            "ID": np.array([1, 2, 3]),
            "value": np.array([10.0, 20.0, 30.0]),
            "Checked": np.array([False, True, False]),
            "Score": np.array([0, 5, 7]),
        },
        None,
        {"Checked": BOOL_SPEC, "Score": INT_SPEC},
    )
    return model


def test_check_state_helpers_are_binding_agnostic():
    """Check states are compared as ints, so all Qt bindings behave alike."""
    assert is_checked(Qt.Checked)
    assert not is_checked(Qt.Unchecked)
    assert not is_checked(Qt.PartiallyChecked)
    assert is_checked(2)
    assert not is_checked(0)
    assert not is_checked(None)


def test_parse_bool_accepts_stored_representations():
    """Stored feature values may be bools, numpy bools, ints or text."""
    assert parse_bool(True) is True
    assert parse_bool(np.bool_(True)) is True
    assert parse_bool(False) is False
    assert parse_bool(None) is False
    assert parse_bool(0) is False
    assert parse_bool(3) is True
    assert parse_bool("true") is True
    assert parse_bool("nonsense") is False


def test_only_annotation_columns_are_editable(annotation_model):
    model = annotation_model
    bool_index = model.index(0, _column_index(model, "Checked"))
    int_index = model.index(0, _column_index(model, "Score"))
    plain_index = model.index(0, _column_index(model, "value"))

    assert bool(model.flags(bool_index) & Qt.ItemIsUserCheckable)
    assert not bool(model.flags(bool_index) & Qt.ItemIsEditable)

    assert bool(model.flags(int_index) & Qt.ItemIsEditable)
    assert not bool(model.flags(int_index) & Qt.ItemIsUserCheckable)

    assert not bool(model.flags(plain_index) & Qt.ItemIsEditable)
    assert not bool(model.flags(plain_index) & Qt.ItemIsUserCheckable)


def test_bool_column_shows_check_box_without_text(annotation_model):
    model = annotation_model
    col = _column_index(model, "Checked")
    unchecked = model.index(0, col)
    checked = model.index(1, col)

    # the check box carries the value, so no redundant text is displayed
    assert model.data(unchecked, Qt.DisplayRole) == ""
    assert not is_checked(model.data(unchecked, Qt.CheckStateRole))
    assert is_checked(model.data(checked, Qt.CheckStateRole))


def test_toggling_check_box_updates_model_and_announces_edit(annotation_model, qtbot):
    model = annotation_model
    index = model.index(0, _column_index(model, "Checked"))

    with qtbot.waitSignal(model.annotation_edited, timeout=1000) as blocker:
        assert model.setData(index, Qt.Checked, Qt.CheckStateRole) is True

    assert blocker.args == [1, "checked", True]
    assert is_checked(model.data(index, Qt.CheckStateRole))

    # and back again
    assert model.setData(index, Qt.Unchecked, Qt.CheckStateRole) is True
    assert not is_checked(model.data(index, Qt.CheckStateRole))


def test_editing_int_column_updates_model_and_announces_edit(annotation_model, qtbot):
    model = annotation_model
    index = model.index(0, _column_index(model, "Score"))

    with qtbot.waitSignal(model.annotation_edited, timeout=1000) as blocker:
        assert model.setData(index, 42, Qt.EditRole) is True

    assert blocker.args == [1, "score", 42]
    assert model.data(index, Qt.EditRole) == 42
    assert model.data(index, Qt.DisplayRole) == "42"


def test_invalid_and_read_only_edits_are_rejected(annotation_model):
    model = annotation_model
    bool_index = model.index(0, _column_index(model, "Checked"))
    int_index = model.index(0, _column_index(model, "Score"))
    plain_index = model.index(0, _column_index(model, "value"))

    # garbage never reaches the tracks
    assert model.setData(int_index, "not a number", Qt.EditRole) is False

    # each column type only accepts its own role
    assert model.setData(bool_index, 1, Qt.EditRole) is False
    assert model.setData(int_index, Qt.Checked, Qt.CheckStateRole) is False

    # regular columns stay read only
    assert model.setData(plain_index, 1, Qt.EditRole) is False
    assert model.data(plain_index, Qt.DisplayRole) == "10"


def test_edit_works_on_read_only_numpy_column(annotation_model):
    """track_df can hand out read-only views, so edits must copy on write."""
    model = TrackTableModel()
    stored = np.array([False, False])
    stored.flags.writeable = False
    model.set_table(
        {"ID": np.array([7, 8]), "Checked": stored},
        None,
        {"Checked": BOOL_SPEC},
    )
    index = model.index(1, _column_index(model, "Checked"))

    assert model.setData(index, Qt.Checked, Qt.CheckStateRole) is True
    assert is_checked(model.data(index, Qt.CheckStateRole))
    assert not stored[1]  # the original read-only array is untouched


def test_int_column_uses_spin_box_editor(annotation_model, qtbot):
    """Integer annotations are edited with a spin box, bools are not."""
    parent = QWidget()
    qtbot.addWidget(parent)
    delegate = AnnotationDelegate(parent)
    option = QStyleOptionViewItem()

    int_index = annotation_model.index(0, _column_index(annotation_model, "Score"))
    bool_index = annotation_model.index(0, _column_index(annotation_model, "Checked"))

    editor = delegate.createEditor(parent, option, int_index)
    assert isinstance(editor, QSpinBox)

    delegate.setEditorData(editor, int_index)
    assert editor.value() == 0

    editor.setValue(13)
    delegate.setModelData(editor, annotation_model, int_index)
    assert annotation_model.data(int_index, Qt.EditRole) == 13

    assert not isinstance(
        delegate.createEditor(parent, option, bool_index), QSpinBox
    )


@pytest.mark.parametrize(
    ("value_type", "default_value", "edited_value"),
    [("bool", False, True), ("int", 0, 42)],
)
def test_annotation_column_round_trips_to_tracks(
    colored_table_widget, value_type, default_value, edited_value
):
    """Creating a column and editing a cell must persist on the tracks."""
    widget, tracks_viewer = colored_table_widget
    tracks = tracks_viewer.tracks
    name = f"manual_{value_type}"

    widget._create_manual_annotation_column(name, value_type, default_value)

    assert name in tracks.features
    assert widget._is_manual_annotation_column(name)

    model = widget._table_widget.model()
    index = model.index(0, _column_index(model, name))
    node_id = model.node_id(0)
    assert node_id is not None

    if value_type == "bool":
        assert model.setData(index, Qt.Checked, Qt.CheckStateRole) is True
    else:
        assert model.setData(index, edited_value, Qt.EditRole) is True

    stored = _graph(tracks).nodes[int(node_id)][name]
    if value_type == "bool":
        assert parse_bool(stored) is True
    else:
        assert int(stored) == edited_value

    # the edit must not leave the widget stuck in its syncing guard
    assert widget._syncing is False
