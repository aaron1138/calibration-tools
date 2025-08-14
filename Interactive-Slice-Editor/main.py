import sys
import os
import tempfile
import cv2
import numpy as np
import copy
import re

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget,
    QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, QDockWidget,
    QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QTabWidget, QInputDialog
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPixmap, QImage, QAction, QPen
from PySide6.QtOpenGLWidgets import QOpenGLWidget

import uvtools_wrapper
from project import Project, DrawRectangleCommand, DrawGradientCommand, DrawTextCommand
from tool_widgets import RectangleToolWidget, GradientToolWidget, TextToolWidget

class InteractiveCanvas(QGraphicsView):
    rectangle_drawn = Signal(QRectF)
    point_clicked = Signal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self._pixmap_item = self.scene.addPixmap(QPixmap())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setViewport(QOpenGLWidget())
        self.drawing_mode = None
        self.start_point = None
        self.preview_item = None

    def set_image(self, image_np: np.ndarray):
        h, w = image_np.shape
        bytes_per_line = w
        q_image = QImage(image_np.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self._pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def set_drawing_mode(self, mode: str):
        self.drawing_mode = mode
        if self.drawing_mode:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            self.start_point = self.mapToScene(event.pos())
            if self.drawing_mode in ['rectangle', 'gradient']:
                rect = QRectF(self.start_point, self.start_point)
                self.preview_item = QGraphicsRectItem(rect)
                self.preview_item.setPen(QPen(Qt.red, 2, Qt.DashLine))
                self.scene.addItem(self.preview_item)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start_point and self.preview_item and self.drawing_mode in ['rectangle', 'gradient']:
            current_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current_point).normalized()
            self.preview_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start_point and event.button() == Qt.LeftButton:
            if self.drawing_mode in ['rectangle', 'gradient'] and self.preview_item:
                self.rectangle_drawn.emit(self.preview_item.rect())
                self.scene.removeItem(self.preview_item)
                self.preview_item = None
            elif self.drawing_mode == 'text':
                self.point_clicked.emit(self.start_point)
            self.start_point = None
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Slice Editor")
        self.resize(1200, 800)
        self.project: Project = Project()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.clipboard = None
        self._setup_ui()
        self._setup_menus()
        self._connect_tool_signals()
        self.update_ui_for_project_state()

    def _setup_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        self.image_canvas = InteractiveCanvas()
        main_layout.addWidget(self.image_canvas, 1)
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Layer:"))
        self.layer_input = QLineEdit("0")
        self.layer_input.setFixedWidth(60)
        self.layer_input.setAlignment(Qt.AlignCenter)
        self.layer_input.editingFinished.connect(self.go_to_layer_from_input)
        nav_layout.addWidget(self.layer_input)
        self.layer_slider = QSlider(Qt.Horizontal)
        self.layer_slider.setRange(0, 0)
        self.layer_slider.valueChanged.connect(self.slider_moved)
        nav_layout.addWidget(self.layer_slider)
        main_layout.addLayout(nav_layout)
        self.statusBar().showMessage("Ready. Open a file to begin.")

        tool_dock = QDockWidget("Tools")
        self.addDockWidget(Qt.LeftDockWidgetArea, tool_dock)

        self.tool_tabs = QTabWidget()
        self.rect_tool_widget = RectangleToolWidget()
        self.gradient_tool_widget = GradientToolWidget()
        self.text_tool_widget = TextToolWidget()
        self.tool_tabs.addTab(self.rect_tool_widget, "Rectangle")
        self.tool_tabs.addTab(self.gradient_tool_widget, "Gradient")
        self.tool_tabs.addTab(self.text_tool_widget, "Text")
        tool_dock.setWidget(self.tool_tabs)

    def _setup_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        edit_menu = menu_bar.addMenu("&Edit")

        open_slice_action = QAction("&Open Slice File...", self)
        open_slice_action.triggered.connect(self.open_slice_file)
        file_menu.addAction(open_slice_action)

        open_single_png_action = QAction("Open Single &PNG...", self)
        open_single_png_action.triggered.connect(self.open_single_png)
        file_menu.addAction(open_single_png_action)

        open_png_seq_action = QAction("Open PNG &Sequence...", self)
        open_png_seq_action.triggered.connect(self.open_png_sequence)
        file_menu.addAction(open_png_seq_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.undo_action = QAction("&Undo", self)
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("&Redo", self)
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        self.copy_action = QAction("&Copy Last Action", self)
        self.copy_action.triggered.connect(self.copy_last_action)
        edit_menu.addAction(self.copy_action)

        self.paste_action = QAction("&Paste to Current Layer", self)
        self.paste_action.triggered.connect(self.paste_to_current_layer)
        edit_menu.addAction(self.paste_action)

        self.paste_n_action = QAction("Paste to &N Layers...", self)
        self.paste_n_action.triggered.connect(self.paste_to_n_layers)
        edit_menu.addAction(self.paste_n_action)

        edit_menu.addSeparator()

        self.add_layer_action = QAction("&Add New Blank Layer", self)
        self.add_layer_action.triggered.connect(self.add_new_layer)
        edit_menu.addAction(self.add_layer_action)

        self.copy_layer_action = QAction("C&opy Current Layer", self)
        self.copy_layer_action.triggered.connect(self.copy_current_layer)
        edit_menu.addAction(self.copy_layer_action)

        self.delete_layer_action = QAction("&Delete Current Layer", self)
        self.delete_layer_action.triggered.connect(self.delete_current_layer)
        edit_menu.addAction(self.delete_layer_action)

    def _connect_tool_signals(self):
        self.rect_tool_widget.draw_button.toggled.connect(self.toggle_draw_mode)
        self.gradient_tool_widget.draw_button.toggled.connect(self.toggle_draw_mode)
        self.text_tool_widget.place_button.toggled.connect(self.toggle_draw_mode)
        self.image_canvas.rectangle_drawn.connect(self.handle_rect_drawing)
        self.image_canvas.point_clicked.connect(self.handle_point_drawing)

    def toggle_draw_mode(self, checked):
        sender = self.sender()
        buttons = [self.rect_tool_widget.draw_button, self.gradient_tool_widget.draw_button, self.text_tool_widget.place_button]
        modes = ['rectangle', 'gradient', 'text']
        if not checked:
            self.image_canvas.set_drawing_mode(None)
            self.statusBar().showMessage("Draw mode disabled.")
            return
        for i, button in enumerate(buttons):
            if sender is button:
                self.image_canvas.set_drawing_mode(modes[i])
                self.statusBar().showMessage(f"{modes[i].capitalize()} mode enabled.")
            else:
                button.setChecked(False)

    def handle_rect_drawing(self, rect: QRectF):
        mode = self.image_canvas.drawing_mode
        if mode == 'rectangle': self.add_rectangle_from_mouse(rect)
        elif mode == 'gradient': self.add_gradient_from_mouse(rect)

    def handle_point_drawing(self, point: QPointF):
        if self.image_canvas.drawing_mode == 'text': self.add_text_from_mouse(point)

    def add_rectangle_from_mouse(self, rect: QRectF):
        if not self.project.layer_image_paths: return
        color_val = int(self.rect_tool_widget.color_edit.text())
        command = DrawRectangleCommand(layer_index=self.layer_slider.value(), x=int(rect.x()), y=int(rect.y()), width=int(rect.width()), height=int(rect.height()), color=color_val)
        self.project.add_edit(command)
        self.load_layer(self.layer_slider.value())
        self.statusBar().showMessage(f"Added rectangle.")
        self.rect_tool_widget.draw_button.setChecked(False)

    def add_gradient_from_mouse(self, rect: QRectF):
        if not self.project.layer_image_paths: return
        grad_values = self.gradient_tool_widget.get_values()
        if grad_values:
            command = DrawGradientCommand(layer_index=self.layer_slider.value(), x=int(rect.x()), y=int(rect.y()), width=int(rect.width()), height=int(rect.height()), start_color=grad_values['start_color'], end_color=grad_values['end_color'], gradient_type=grad_values['type'])
            self.project.add_edit(command)
            self.load_layer(self.layer_slider.value())
            self.statusBar().showMessage(f"Added gradient.")
        self.gradient_tool_widget.draw_button.setChecked(False)

    def add_text_from_mouse(self, point: QPointF):
        if not self.project.layer_image_paths: return
        text_values = self.text_tool_widget.get_values()
        if text_values:
            command = DrawTextCommand(layer_index=self.layer_slider.value(), position=(int(point.x()), int(point.y())), text=text_values['text'], font_scale=text_values['font_scale'], color=text_values['color'])
            self.project.add_edit(command)
            self.load_layer(self.layer_slider.value())
            self.statusBar().showMessage(f"Added text.")
        self.text_tool_widget.place_button.setChecked(False)

    def open_slice_file(self):
        uvtools_path = "C:\\Program Files\\UVTools\\UVToolsCmd.exe"
        if not os.path.exists(uvtools_path):
            QMessageBox.critical(self, "UVTools Not Found", f"Could not find UVToolsCmd.exe at:\n{uvtools_path}")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Slice File")
        if not file_path: return
        self.statusBar().showMessage(f"Extracting layers...")
        QApplication.processEvents()
        try:
            extracted_folder = uvtools_wrapper.extract_layers(uvtools_path, file_path, self.temp_dir.name)
            image_files = sorted([os.path.join(extracted_folder, f) for f in os.listdir(extracted_folder) if f.lower().endswith('.png')])
            if not image_files: raise RuntimeError("No PNG images were extracted.")
            self.load_new_project(image_files, file_path)
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            QMessageBox.critical(self, "Error Loading File", f"An error occurred:\n{e}")

    def open_single_png(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Single PNG", filter="PNG Files (*.png)")
        if not file_path: return
        self.load_new_project([file_path], file_path)

    def open_png_sequence(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder with PNG Sequence")
        if not dir_path: return
        try:
            image_files = sorted(
                [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.png')],
                key=lambda f: int(re.search(r'(\d+)', f).group(1)) if re.search(r'(\d+)', f) else -1
            )
            if not image_files: raise RuntimeError("No PNG images found in the selected directory.")
            self.load_new_project(image_files, dir_path)
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            QMessageBox.critical(self, "Error Loading Sequence", f"An error occurred:\n{e}")

    def load_new_project(self, image_paths: list, source_path: str):
        self.project = Project(source_file_path=source_path, layer_image_paths=image_paths)
        self.update_ui_for_project_state()
        self.load_layer(0)
        self.statusBar().showMessage(f"Loaded {self.project.total_layers} layers from {os.path.basename(source_path)}.")

    def slider_moved(self, value):
        self.load_layer(value)

    def go_to_layer_from_input(self):
        try:
            target_layer = int(self.layer_input.text())
            if 0 <= target_layer < self.project.total_layers:
                self.load_layer(target_layer)
            else:
                self.layer_input.setText(str(self.layer_slider.value()))
        except (ValueError, AttributeError):
             self.layer_input.setText("0")

    def load_layer(self, layer_index):
        if not self.project.layer_image_paths or not (0 <= layer_index < self.project.total_layers): return

        base_image_path = self.project.layer_image_paths[layer_index]
        if base_image_path is not None:
            image_np = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image_np = None

        if image_np is None:
            dims = self.project.get_project_dimensions()
            image_np = np.zeros(dims, dtype=np.uint8)

        image_np = image_np.copy()
        if layer_index in self.project.edits:
            for command in self.project.edits[layer_index]:
                image_np = command.apply(image_np)
        self.image_canvas.set_image(image_np)
        self.layer_slider.setValue(layer_index)
        self.layer_input.setText(str(layer_index))
        self.update_ui_for_project_state()

    def undo(self):
        command = self.project.perform_undo()
        if command:
            self.statusBar().showMessage(f"Undid action on layer {command.get_affected_layer()}")
            self.load_layer(self.layer_slider.value())
        self.update_ui_for_project_state()

    def redo(self):
        command = self.project.perform_redo()
        if command:
            self.statusBar().showMessage(f"Redid action on layer {command.get_affected_layer()}")
            self.load_layer(self.layer_slider.value())
        self.update_ui_for_project_state()

    def copy_last_action(self):
        if not self.project.undo_stack:
            self.statusBar().showMessage("No action to copy.")
            return
        self.clipboard = copy.deepcopy(self.project.undo_stack[-1])
        self.statusBar().showMessage("Copied last action to clipboard.")
        self.update_ui_for_project_state()

    def paste_to_current_layer(self):
        if not self.clipboard:
            self.statusBar().showMessage("Clipboard is empty.")
            return
        new_command = copy.deepcopy(self.clipboard)
        new_command.layer_index = self.layer_slider.value()
        self.project.add_edit(new_command)
        self.load_layer(self.layer_slider.value())
        self.statusBar().showMessage(f"Pasted action to layer {new_command.layer_index}.")

    def paste_to_n_layers(self):
        if not self.clipboard:
            self.statusBar().showMessage("Clipboard is empty.")
            return
        num, ok = QInputDialog.getInt(self, "Paste to N Layers", "Number of subsequent layers to paste to:", 1, 1, 1000)
        if ok and num > 0:
            start_layer = self.layer_slider.value()
            end_layer = min(start_layer + num, self.project.total_layers - 1)
            for i in range(start_layer, end_layer + 1):
                new_command = copy.deepcopy(self.clipboard)
                new_command.layer_index = i
                self.project.add_edit(new_command)
            self.load_layer(self.layer_slider.value())
            self.statusBar().showMessage(f"Pasted action to layers {start_layer} through {end_layer}.")

    def add_new_layer(self):
        if not self.project.layer_image_paths: return
        current_index = self.layer_slider.value()
        self.project.add_blank_layer(current_index + 1)
        self.update_ui_for_project_state()
        self.load_layer(current_index + 1)
        self.statusBar().showMessage(f"Added new blank layer at {current_index + 1}.")

    def copy_current_layer(self):
        if not self.project.layer_image_paths: return
        current_index = self.layer_slider.value()
        self.project.copy_layer(current_index)
        self.update_ui_for_project_state()
        self.load_layer(current_index + 1)
        self.statusBar().showMessage(f"Copied layer {current_index} to {current_index + 1}.")

    def delete_current_layer(self):
        if not self.project.layer_image_paths: return
        if self.project.total_layers <= 1:
            self.statusBar().showMessage("Cannot delete the last layer.")
            return

        current_index = self.layer_slider.value()
        self.project.delete_layer(current_index)

        # After deleting, load the new layer at the same index (or the last one if it was the last)
        new_index = min(current_index, self.project.total_layers - 1)
        self.update_ui_for_project_state()
        self.load_layer(new_index)
        self.statusBar().showMessage(f"Deleted layer {current_index}.")

    def update_ui_for_project_state(self):
        has_project = self.project.total_layers > 0
        self.layer_slider.setRange(0, self.project.total_layers - 1)

        self.undo_action.setEnabled(bool(self.project.undo_stack))
        self.redo_action.setEnabled(bool(self.project.redo_stack))
        self.copy_action.setEnabled(bool(self.project.undo_stack))
        self.paste_action.setEnabled(self.clipboard is not None and has_project)
        self.paste_n_action.setEnabled(self.clipboard is not None and has_project)
        self.add_layer_action.setEnabled(has_project)
        self.copy_layer_action.setEnabled(has_project)
        self.delete_layer_action.setEnabled(has_project and self.project.total_layers > 1)

    def closeEvent(self, event):
        self.temp_dir.cleanup()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
