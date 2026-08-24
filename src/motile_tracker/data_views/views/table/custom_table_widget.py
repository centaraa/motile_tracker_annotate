import contextlib

import napari
import numpy as np
import pandas as pd
from funtracks.user_actions import UserUpdateNodesAttrs
from napari.utils import DirectLabelColormap
from qtpy.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    Qt,
    QTimer,
    Signal,
)
from qtpy.QtGui import QColor, QKeyEvent, QMouseEvent, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from motile_tracker.data_views.keybindings_config import GENERAL_KEY_ACTIONS
from motile_tracker.data_views.views.layers.click_utils import (
    detect_side_button,
)
from motile_tracker.data_views.views.tree_view.tree_widget_utils import (
    get_features_from_tracks,
)
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer

# Qt.CheckState.Checked as a plain int, so the comparison works the same on
# PyQt5, PyQt6 and PySide (where the enum type differs).
CHECKED = 2


def parse_bool(value) -> bool:
    """Interpret a stored feature value as a boolean."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def is_checked(state) -> bool:
    """True if a Qt check state (enum member or int) means Checked."""
    raw = getattr(state, "value", state)
    try:
        return int(raw) == CHECKED
    except (TypeError, ValueError):
        return False


class TrackTableModel(QAbstractTableModel):
    """Lazy table model backing the tracks table.

    Holds the data as columns of numpy arrays (one row per node) and serves
    values/colors on demand, so the view only ever realizes the handful of rows
    currently visible. This keeps memory and populate-time O(visible rows)
    regardless of the total node count (a ``QTableWidget`` would instead create
    one ``QTableWidgetItem`` per cell, which blows up memory and freezes the UI
    for large datasets).

    Columns registered as manual annotation columns are editable: ``bool``
    features are exposed as check boxes, ``int`` features as editable numbers.
    Edits are written into the numpy column and announced via
    ``annotation_edited`` so the widget can persist them on the tracks.
    """

    #: node id, feature key, new value
    annotation_edited = Signal(int, str, object)

    def __init__(self, parent=None, decimals: int = 3):
        super().__init__(parent)
        self._table: dict[str, np.ndarray] = {}
        self._columns: list[str] = []
        self._nrows = 0
        self._decimals = decimals
        self._bg: list[QColor] = []
        self._fg: list[QColor] = []
        self._manual_cols: dict[str, dict[str, str]] = {}

    def set_table(
        self,
        table: dict[str, np.ndarray],
        colormap,
        manual_cols: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Replace the table contents and precompute per-row colors.

        Args:
            table (dict[str, np.ndarray]): one numpy array per column.
            colormap: colormap mapping node ids to colors, or None.
            manual_cols (dict[str, dict[str, str]] | None): manual annotation
                columns, keyed by displayed column name, each holding the
                feature ``key`` and its ``value_type`` ('bool' or 'int').
        """
        self.beginResetModel()
        self._table = table
        self._columns = list(table.keys())
        self._nrows = len(next(iter(table.values()))) if table else 0
        self._manual_cols = dict(manual_cols) if manual_cols else {}

        # Precompute one background/foreground color per row (O(rows), cheap;
        # no per-cell widget objects are created).
        self._bg = []
        self._fg = []
        ids = table.get("ID")
        if ids is not None and len(ids) > 0 and colormap is not None:
            # Single vectorized colormap.map call over all row labels: colormap.map
            # has a large fixed per-call overhead, so mapping row-by-row is O(rows)
            # slow. Map once, then build the per-row QColors.
            mapped = colormap.map(np.asarray(ids))
            for rgba in mapped:
                if rgba[3] == 0:
                    rgba = [0, 0, 0, 0]
                r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                self._bg.append(QColor(r, g, b))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                self._fg.append(
                    QColor(0, 0, 0) if luminance > 140 else QColor(255, 255, 255)
                )
        self.endResetModel()

    def rowCount(self, parent=None) -> int:
        if parent is not None and parent.isValid():
            return 0  # flat table: child rows don't exist
        return self._nrows

    def columnCount(self, parent=None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._columns[section]
        return None

    def _format(self, value) -> str:
        try:
            number = float(value)
        except (ValueError, TypeError):
            return str(value)
        if float(number).is_integer():
            return str(int(number))
        return f"{number:.{self._decimals}f}"

    def manual_spec(self, column: int) -> dict[str, str] | None:
        """Return the manual annotation spec of a column, or None.

        Args:
            column (int): the column index.

        Returns:
            dict[str, str] | None: the feature ``key`` and ``value_type`` of the
                manual annotation column, or None for regular columns.
        """
        if column < 0 or column >= len(self._columns):
            return None
        return self._manual_cols.get(self._columns[column])

    def node_id(self, row: int) -> int | None:
        """Return the node id displayed in the given row, if any.

        Args:
            row (int): the row index.

        Returns:
            int | None: the node id, or None if it cannot be determined.
        """
        ids = self._table.get("ID")
        if ids is None or row < 0 or row >= len(ids):
            return None
        try:
            return int(ids[row])
        except (TypeError, ValueError):
            return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        spec = self.manual_spec(col)
        if spec is not None:
            value = self._table[self._columns[col]][row]
            if spec["value_type"] == "bool":
                if role == Qt.CheckStateRole:
                    return Qt.Checked if parse_bool(value) else Qt.Unchecked
                if role in (Qt.DisplayRole, Qt.EditRole):
                    # the check box carries the value, no text next to it
                    return ""
            elif role == Qt.EditRole:
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return 0
        if role == Qt.DisplayRole:
            return self._format(self._table[self._columns[col]][row])
        if role == Qt.BackgroundRole:
            return self._bg[row] if row < len(self._bg) else None
        if role == Qt.ForegroundRole:
            return self._fg[row] if row < len(self._fg) else None
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        item_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        spec = self.manual_spec(index.column())
        if spec is not None:
            if spec["value_type"] == "bool":
                item_flags |= Qt.ItemIsUserCheckable
            else:
                item_flags |= Qt.ItemIsEditable
        return item_flags

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:
        """Store an edited manual annotation value and announce the change.

        Only manual annotation columns are writable. The numpy column is updated
        copy-on-write (the dataframe may hand out read-only views), and
        ``annotation_edited`` is emitted so the widget can persist the value on
        the tracks.
        """
        if not index.isValid():
            return False
        spec = self.manual_spec(index.column())
        if spec is None:
            return False

        if spec["value_type"] == "bool":
            if role != Qt.CheckStateRole:
                return False
            new_value = is_checked(value)
        else:
            if role != Qt.EditRole:
                return False
            try:
                new_value = int(float(value))
            except (TypeError, ValueError):
                return False

        node_id = self.node_id(index.row())
        if node_id is None:
            return False

        column = self._columns[index.column()]
        col_arr = self._table[column]
        if not col_arr.flags.writeable:
            col_arr = col_arr.copy()
            self._table[column] = col_arr
        col_arr[index.row()] = new_value

        self.dataChanged.emit(index, index, [role, Qt.DisplayRole])
        self.annotation_edited.emit(node_id, spec["key"], new_value)
        return True


class NoSelectionHighlightDelegate(QStyledItemDelegate):
    """Prevents Qt from painting the default selection background,
    preserving each row's custom background color, and draws a cyan border instead."""

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)

        # Read the selection state straight off the style option: Qt already
        # computed it for this cell. Asking the view instead (via
        # selectedIndexes()) would rebuild the full selection on every single
        # cell paint, making each repaint O(visible cells x selected cells) --
        # ~15s for a 25k-row table with everything selected.
        # With SelectRows behavior every cell of a selected row carries this
        # flag, so a whole-row border still gets drawn.
        selected = bool(opt.state & QStyle.State_Selected)
        opt.state &= ~QStyle.State_Selected

        # Paint normally first (preserving the model's Background + Foreground roles)
        super().paint(painter, opt, index)

        # Draw a cyan border around the *entire row* if selected
        if selected:
            painter.setPen(QPen(Qt.cyan, 2))
            painter.drawRect(opt.rect.adjusted(1, 1, -2, -2))


class AnnotationDelegate(NoSelectionHighlightDelegate):
    """Adds an integer editor for manual annotation columns.

    Keeps the custom row-color painting of NoSelectionHighlightDelegate, and
    provides a spin box (instead of a free text field) for int annotation
    columns, so invalid input cannot reach the tracks in the first place.
    """

    def _is_int_annotation(self, index) -> bool:
        model = index.model()
        spec = model.manual_spec(index.column()) if model is not None else None
        return spec is not None and spec["value_type"] == "int"

    def createEditor(self, parent, option, index):
        if not self._is_int_annotation(index):
            return super().createEditor(parent, option, index)
        editor = QSpinBox(parent)
        editor.setRange(-2147483648, 2147483647)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        if isinstance(editor, QSpinBox):
            try:
                editor.setValue(int(index.data(Qt.EditRole)))
            except (TypeError, ValueError):
                editor.setValue(0)
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QSpinBox):
            editor.interpretText()
            model.setData(index, editor.value(), Qt.EditRole)
            return
        super().setModelData(editor, model, index)


class CustomTableWidget(QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verticalHeader().setSectionsClickable(False)
        self._drag_start_row = None

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse click events and check modifiers for different behaviors:
        - Plain click on an annotation check box: toggle it.
        - Plain click: single selection, toggle if already selected
        - Shift: append to selection.
        - Ctrl/CMD: center node, should not affect selection.
        - Side buttons (back/forward): navigate selection history.
        """
        # Intercept mouse side buttons for selection history navigation
        side_button = detect_side_button(event)

        if side_button is not None:
            self.parent().tracks_viewer.select_node_set_from_history(
                previous=side_button == 4
            )
            return

        # Handle other clicks for new selection and centering
        index = self.indexAt(event.pos())
        if not index.isValid():
            return

        row = index.row()
        modifiers = event.modifiers()

        ctrl = modifiers & Qt.ControlModifier
        shift = modifiers & Qt.ShiftModifier

        model = self.model()

        # Plain click on a manual annotation check box toggles it
        if (
            not ctrl
            and not shift
            and model is not None
            and bool(model.flags(index) & Qt.ItemIsUserCheckable)
        ):
            checked = is_checked(model.data(index, Qt.CheckStateRole))
            model.setData(
                index,
                Qt.Unchecked if checked else Qt.Checked,
                Qt.CheckStateRole,
            )
            event.accept()
            return

        sel_model = self.selectionModel()
        model_index = model.index(row, 0)

        if ctrl:
            self.parent().center_node(model_index)
            event.accept()
            return

        if shift:
            # Append single row
            sel_model.select(
                model_index, QItemSelectionModel.Select | QItemSelectionModel.Rows
            )
            self._drag_start_row = row
            event.accept()
            return

        # Plain click: single selection, toggle if already selected
        if sel_model.isSelected(model_index):
            sel_model.select(
                model_index, QItemSelectionModel.Deselect | QItemSelectionModel.Rows
            )
            self._drag_start_row = None
        else:
            sel_model.clearSelection()
            sel_model.select(
                model_index, QItemSelectionModel.Select | QItemSelectionModel.Rows
            )
            self._drag_start_row = row

        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Record mouse drag events to select a range. In combination with shift, it is
        possible to select multiple ranges.
        """

        if not (event.buttons() & Qt.LeftButton):
            return

        index = self.indexAt(event.pos())
        if not index.isValid() or self._drag_start_row is None:
            return

        current_row = index.row()
        start = self._drag_start_row
        end = current_row

        top = min(start, end)
        bottom = max(start, end)

        selection = QItemSelection(
            self.model().index(top, 0),
            self.model().index(bottom, self.model().columnCount() - 1),
        )

        modifiers = event.modifiers()

        if modifiers & Qt.ShiftModifier:
            # add range
            self.selectionModel().select(selection, QItemSelectionModel.Select)
        else:
            # replace selection with this range
            self.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_start_row = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events for common tracksviewer actions."""
        # Get the parent ColoredTableWidget to access tracks_viewer
        parent = self.parent()
        if parent is None or not hasattr(parent, "tracks_viewer"):
            super().keyPressEvent(event)
            return

        tracks_viewer = parent.tracks_viewer

        # Get the action name from the general keybind mapping
        action_name = GENERAL_KEY_ACTIONS.get(event.key())
        if action_name:
            method = getattr(tracks_viewer, action_name, None)
            if method:
                method()
                event.accept()
                return

        # Allow parent class to handle other events
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        index = self.indexAt(event.pos())
        model = self.model()
        if (
            index.isValid()
            and model is not None
            and bool(model.flags(index) & Qt.ItemIsEditable)
        ):
            self.edit(index)  # force-start editing, bypassing pressedIndex check
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ColoredTableWidget(QWidget):
    """Customized table widget with colored rows based on label colors in a napari Labels layer"""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.tracks_viewer = TracksViewer.get_instance(viewer)
        self.tracks_viewer.update_track_df(
            initialization=True, refresh_view=True
        )  # make sure tracks_viewer initializes/updates the track df
        self.tracks_viewer.table_widget_present = True
        self.tracks_viewer.tracks_updated.connect(self.update_data)
        self._table_widget = CustomTableWidget()
        self._model = TrackTableModel(self._table_widget)
        self._table_widget.setModel(self._model)
        self.special_selection = []
        self.ascending = False  # for choosing whether to sort ascending or descending
        self._syncing = False

        self._table: dict[str, np.ndarray] = {}
        self.colormap = None
        self._id_to_row: dict[int, int] = {}
        self._manual_annotation_cols = {}
        self.update_data()

        add_col_btn = QPushButton("Add Annotation Column")
        add_col_btn.clicked.connect(self._add_annotation_column_dialog)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(add_col_btn)
        controls_layout.addStretch()

        # Connect to single click in the header to sort the table.
        self._table_widget.horizontalHeader().sectionClicked.connect(self._sort_table)

        # Instruction label to explain mouse and keyboard functions.
        label = QLabel(
            "Use left mouse click to select and center a label. Use Ctrl/CMD to center a node, Shift to append to selection. Use mouse drag to select a range. Annotation columns are editable: click a check box, double click a number."
        )
        label.setWordWrap(True)
        font = label.font()
        font.setItalic(True)
        label.setFont(font)

        main_layout = QVBoxLayout()
        main_layout.addWidget(label)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self._table_widget)
        self.setLayout(main_layout)
        self.setMinimumHeight(300)

        # Selection behavior
        self._table_widget.setStyleSheet("""
            QTableView::item:selected {
                border: 2px solid cyan;
            }
        """)

        self._table_widget.verticalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: rgb(40,40,40);       /* normal */
                color: white;
                padding: 4px;
                border: 1px solid #555;
            }

            QHeaderView::section:selected {            /* when the row is selected */
                background-color: cyan;
                color: black;
            }

            QHeaderView::section:pressed {
                background-color: cyan;
                color: black;
            }
        """)

        self._table_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self._table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table_widget.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )

        delegate = AnnotationDelegate(self._table_widget)
        self._table_widget.setItemDelegate(delegate)

        self._table_widget.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self._model.annotation_edited.connect(self._on_annotation_edited)
        self.tracks_viewer.node_selection_updated.connect(self._update_selected)
        self.tracks_viewer.center_node.connect(self.scroll_to_node)

    def cleanup(self) -> None:
        """Stop following the TracksViewer.

        Called by MenuManager when the dock is destroyed, so we can stop listening to
        TracksViewer update signals to rebuild the table.
        """
        self.tracks_viewer.table_widget_present = False
        for signal, slot in (
            (self.tracks_viewer.tracks_updated, self.update_data),
            (self.tracks_viewer.node_selection_updated, self._update_selected),
            (self.tracks_viewer.center_node, self.scroll_to_node),
        ):
            with contextlib.suppress(ValueError, KeyError, RuntimeError):
                signal.disconnect(slot)

    def update_data(self, **kwargs) -> None:
        """Update the displayed data based on the tracks_df on TracksViewer"""
        if self._syncing:
            return

        columns_to_display = ["node_id"] + get_features_from_tracks(
            self.tracks_viewer.tracks, features_to_ignore=["Bounding box"]
        )
        self.set_data(self.tracks_viewer.track_df, columns_to_display)

    def _update_selected(self) -> None:
        """Select the rows belonging to the nodes that are in the selection list of the
        TracksViewer
        """
        if self._syncing:
            return

        self._syncing = True
        try:
            selected_nodes = self.tracks_viewer.selected_nodes.as_list
            rows = [
                self._find_row(ID=node) for node in selected_nodes if node is not None
            ]

            self._table_widget.clearSelection()
            self._select_rows(rows)

        finally:
            self._syncing = False

    def _select_rows(self, rows: list[int]) -> None:
        """Replace current table selection with given rows.

        Args:
            rows (list[int]): list of indices to be selected.
        """

        if not rows:
            return

        model = self._table_widget.model()
        selection_model = self._table_widget.selectionModel()

        selection = QItemSelection()

        for row in rows:
            if row is None:
                continue
            index = model.index(row, 0)
            selection.select(index, index)

        selection_model.select(
            selection,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )

    def _on_selection_changed(self, *args) -> None:
        """Update the node selection list on TracksViewer based on the rows selected in
        the table.
        """
        if self._syncing:
            return  # skip if selection was changed programmatically

        rows = sorted({index.row() for index in self._table_widget.selectedIndexes()})
        if not rows:
            return

        labels = [self._table["ID"][row] for row in rows]

        # Ensure we do not call this when it is still updating.
        self._syncing = True
        try:
            self.tracks_viewer.selected_nodes.add_list(labels)
        finally:
            self._syncing = False

        QTimer.singleShot(0, self._update_label_colormap)

    def center_node(self, index: int) -> None:
        """Call TracksViewer to center Viewer on the node of current index

        Args:
            index (int): the index in the table corresponding to the to be centered node.
        """
        if self._syncing:
            return

        self._syncing = True
        try:
            row = index.row()
            node = self._table["ID"][row]
            self.tracks_viewer.center_on_node(node)
        finally:
            self._syncing = False

    def scroll_to_node(self, node: int) -> None:
        """Identify the index of the node that was selected, and scroll to that index.

        Args:
            node (int): the node to scroll to.
        """

        if self._syncing:
            return

        self._syncing = True
        try:
            index = self._find_row(ID=node)
            if index is not None:
                selection_model = self._table_widget.selectionModel()

                model_index = self._table_widget.model().index(index, 0)

                if (
                    selection_model.isSelected(model_index)
                    and len(selection_model.selectedRows()) == 1
                ):
                    return

                self.scroll_to_row(index)

        finally:
            self._syncing = False

    def scroll_to_row(self, index: int) -> None:
        """Scroll to make sure the row is in view

        Args:
            index (int): the index to scroll to
        """
        self._table_widget.scrollTo(
            self._table_widget.model().index(index, 0),
            QAbstractItemView.PositionAtCenter,
        )

    def _find_row(self, **conditions) -> int | None:
        """
        Find the first row matching the given conditions (e.g. label=12, time_point=5)
        Returns: row index or None
        """

        # Fast path: lookup by node id (the only condition used in practice).
        if set(conditions) == {"ID"} and conditions["ID"] is not None:
            try:
                return self._id_to_row.get(int(conditions["ID"]))
            except (ValueError, TypeError):
                return None

        n_rows = len(self._table.get("ID", []))
        for row in range(n_rows):
            # Only check conditions that are not None
            if all(
                float(self._table[col][row]) == float(val)
                for col, val in conditions.items()
                if val is not None
            ):
                return row

        return None

    @staticmethod
    def _build_id_to_row(table: dict[str, np.ndarray]) -> dict[int, int]:
        """Build the node id -> row index lookup used by _find_row.

        Args:
            table (dict[str, np.ndarray]): the current table columns.

        Returns:
            dict[int, int]: node id to row index.
        """
        ids = table.get("ID")
        if ids is None:
            return {}

        mapping: dict[int, int] = {}
        for row, node_id in enumerate(ids):
            try:
                mapping[int(node_id)] = row
            except (TypeError, ValueError):
                continue
        return mapping

    def set_data(
        self, df: pd.DataFrame, columns_to_display: list[str] | None = None
    ) -> None:
        """Set the content of the table from a dataframe.

        Args:
            df (pd.DataFrame): dataframe holding the tree widget data, one row per node.
            columns_to_display (list[str] | None): optional list of column headers to
                filter on (should correspond to the tracks features). Column 'node_id'
                should always be included.
        """

        if columns_to_display is not None and len(df.columns) > 0:
            df = df[[col for col in columns_to_display if col in df.columns]]
        df = df.rename(columns={"node_id": "ID"})

        table: dict[str, np.ndarray] = {col: df[col].to_numpy() for col in df.columns}
        self._table = table
        self.colormap = self._get_colormap()
        self._manual_annotation_cols = self._get_manual_annotation_columns()
        self._id_to_row = self._build_id_to_row(table)

        # Hand the columns to the model (which serves cells lazily) instead of
        # creating one item widget per cell.
        self._syncing = True
        try:
            self._model.set_table(table, self.colormap, self._manual_annotation_cols)
        finally:
            self._syncing = False

    def _get_colormap(self) -> DirectLabelColormap:
        """Get a DirectLabelColormap that maps node ids to their track ids, and then
        uses the tracks_viewer.colormap to map from track_id to color.

        Returns:
            DirectLabelColormap: A map from node ids to colors based on track id
        """
        tracks = self.tracks_viewer.tracks
        if tracks is not None:
            nodes = tracks.graph.node_ids()
            track_ids = tracks.get_track_ids(nodes)
            # Single vectorized colormap.map call: ~290x faster than per-node
            # calls because colormap.map has a large fixed per-call overhead.
            # Copy per node (distinct rows) so each color is an independent array.
            if len(track_ids) > 0:
                mapped = self.tracks_viewer.colormap.map(np.asarray(track_ids))
                colors = [color.copy() for color in mapped]
            else:
                colors = []
        else:
            nodes = []
            colors = []

        return DirectLabelColormap(
            color_dict={
                **dict(zip(nodes, colors, strict=True)),
                None: [0, 0, 0, 0],
            }
        )

    def _update_label_colormap(self) -> None:
        """
        Highlight the labels of selected rows. Assumes the layer already has a
        DirectLabelColormap.
        """

        # find selected rows, and set highlight matching labels
        selected_rows = sorted(
            {index.row() for index in self._table_widget.selectedIndexes()}
        )
        if not selected_rows:
            self._reset_layer_colormap()
            return

        selected_labels = [self._table["ID"][row] for row in selected_rows]
        for key, color in self.colormap.color_dict.items():
            if key is not None and key != 0:
                color[-1] = 0.6
        for key in selected_labels:
            if key in self.colormap.color_dict:
                self.colormap.color_dict[key][-1] = 1

    def _sort_table(self, column_index: int) -> None:
        """Sorts the table in ascending or descending order

        Args:
            column_index (int): The index of the clicked column header
        """

        selected_column = list(self._table.keys())[column_index]
        df = pd.DataFrame(self._table).sort_values(
            by=selected_column, ascending=self.ascending
        )
        self.ascending = not self.ascending

        self.set_data(df)

    def _get_manual_annotation_columns(self) -> dict[str, dict[str, str]]:
        tracks = self.tracks_viewer.tracks
        if tracks is None:
            return {}

        manual_cols: dict[str, dict[str, str]] = {}
        for key, feature in tracks.features.items():
            if feature.get("feature_type") != "node":
                continue
            if not feature.get("manual_annotation", False):
                continue
            value_type = feature.get("value_type")
            if value_type not in ("bool", "int"):
                continue
            display_name = feature.get("display_name", key)
            if isinstance(display_name, (list, tuple)):
                continue
            manual_cols[str(display_name)] = {
                "key": str(key),
                "value_type": str(value_type),
            }
        return manual_cols

    def _is_manual_annotation_column(self, column_name: str) -> bool:
        return column_name in self._manual_annotation_cols

    def _parse_bool(self, value) -> bool:
        return parse_bool(value)

    def _create_manual_annotation_column(
        self, name: str, value_type: str, default_value: int | bool
    ) -> None:
        tracks = self.tracks_viewer.tracks
        if tracks is None:
            return

        if name in tracks.features:
            QMessageBox.warning(
                self, "Column exists", f"Feature '{name}' already exists."
            )
            return

        new_feature = {
            "feature_type": "node",
            "value_type": value_type,
            "num_values": 1,
            "display_name": name,
            "default_value": default_value,
            "manual_annotation": True,
        }
        tracks.add_feature(name, new_feature)

        nodes = [int(n) for n in tracks.graph.node_ids()]
        if nodes:
            UserUpdateNodesAttrs(
                tracks=tracks,
                nodes=nodes,
                attrs={name: [default_value] * len(nodes)},
            )

        self.tracks_viewer.update_track_df(initialization=False, refresh_view=False)
        self.update_data()

    def _add_annotation_column_dialog(self) -> None:
        tracks = self.tracks_viewer.tracks
        if tracks is None:
            QMessageBox.warning(self, "No tracks", "Load tracks first.")
            return

        name, ok = QInputDialog.getText(self, "New annotation", "Column name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            return

        value_type, ok = QInputDialog.getItem(
            self,
            "Type",
            "Column type:",
            ["bool", "int"],
            0,
            False,
        )
        if not ok:
            return

        if value_type == "bool":
            default_text, ok = QInputDialog.getItem(
                self,
                "Default value",
                "Default bool value:",
                ["False", "True"],
                0,
                False,
            )
            if not ok:
                return
            default_value = default_text == "True"
        else:
            default_value, ok = QInputDialog.getInt(
                self,
                "Default value",
                "Default int value:",
                0,
                -2147483648,
                2147483647,
                1,
            )
            if not ok:
                return

        self._create_manual_annotation_column(name, value_type, default_value)

    def _on_annotation_edited(self, node_id: int, feature_key: str, value) -> None:
        """Persist an annotation edit made in the table on the tracks.

        Args:
            node_id (int): the node whose attribute was edited.
            feature_key (str): the key of the edited node feature.
            value: the new value (bool or int).
        """
        if self._syncing:
            return

        tracks = self.tracks_viewer.tracks
        if tracks is None:
            return

        self._syncing = True
        try:
            UserUpdateNodesAttrs(
                tracks=tracks,
                nodes=[int(node_id)],
                attrs={feature_key: [value]},
            )
        finally:
            self._syncing = False
