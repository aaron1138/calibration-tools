# tool_widgets.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox
)
from PySide6.QtGui import QIntValidator

class GradientToolWidget(QWidget):
    """A widget for defining the parameters of a gradient to be drawn."""
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        grid_layout = QGridLayout()

        # Validators
        color_validator = QIntValidator(0, 255, self)

        # Controls
        self.gradient_type_combo = QComboBox()
        self.gradient_type_combo.addItems(["Linear", "Radial"])
        grid_layout.addWidget(QLabel("Type:"), 0, 0)
        grid_layout.addWidget(self.gradient_type_combo, 0, 1)

        self.start_color_edit = QLineEdit("0")
        self.start_color_edit.setValidator(color_validator)
        grid_layout.addWidget(QLabel("Start Color:"), 1, 0)
        grid_layout.addWidget(self.start_color_edit, 1, 1)

        self.end_color_edit = QLineEdit("255")
        self.end_color_edit.setValidator(color_validator)
        grid_layout.addWidget(QLabel("End Color:"), 2, 0)
        grid_layout.addWidget(self.end_color_edit, 2, 1)

        main_layout.addLayout(grid_layout)

        self.draw_button = QPushButton("Draw Gradient")
        self.draw_button.setCheckable(True)
        main_layout.addWidget(self.draw_button)

        main_layout.addStretch(1)

    def get_values(self):
        """Returns the values from the input fields."""
        try:
            return {
                "type": self.gradient_type_combo.currentText().lower(),
                "start_color": int(self.start_color_edit.text()),
                "end_color": int(self.end_color_edit.text()),
            }
        except ValueError:
            return None


class RectangleToolWidget(QWidget):
    """A widget for defining the parameters of a rectangle to be drawn."""
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        grid_layout = QGridLayout()

        # Validators
        uint_validator = QIntValidator(0, 99999, self) # For coords and size
        color_validator = QIntValidator(0, 255, self) # For grayscale color

        # Input Fields
        self.x_edit = QLineEdit("100")
        self.x_edit.setValidator(uint_validator)
        grid_layout.addWidget(QLabel("X:"), 0, 0)
        grid_layout.addWidget(self.x_edit, 0, 1)

        self.y_edit = QLineEdit("100")
        self.y_edit.setValidator(uint_validator)
        grid_layout.addWidget(QLabel("Y:"), 1, 0)
        grid_layout.addWidget(self.y_edit, 1, 1)

        self.width_edit = QLineEdit("200")
        self.width_edit.setValidator(uint_validator)
        grid_layout.addWidget(QLabel("Width:"), 2, 0)
        grid_layout.addWidget(self.width_edit, 2, 1)

        self.height_edit = QLineEdit("150")
        self.height_edit.setValidator(uint_validator)
        grid_layout.addWidget(QLabel("Height:"), 3, 0)
        grid_layout.addWidget(self.height_edit, 3, 1)

        self.color_edit = QLineEdit("255") # Default to white
        self.color_edit.setValidator(color_validator)
        grid_layout.addWidget(QLabel("Color (0-255):"), 4, 0)
        grid_layout.addWidget(self.color_edit, 4, 1)

        main_layout.addLayout(grid_layout)

        self.add_button = QPushButton("Add via Coords")
        main_layout.addWidget(self.add_button)

        self.draw_button = QPushButton("Draw with Mouse")
        self.draw_button.setCheckable(True) # Make it a toggle button
        main_layout.addWidget(self.draw_button)

        main_layout.addStretch(1) # Pushes everything to the top

    def get_values(self):
        """Returns the integer values from the input fields."""
        try:
            return {
                "x": int(self.x_edit.text()),
                "y": int(self.y_edit.text()),
                "width": int(self.width_edit.text()),
                "height": int(self.height_edit.text()),
                "color": int(self.color_edit.text()),
            }
        except ValueError:
            # Return None or raise an exception if fields are empty/invalid
            return None


class TextToolWidget(QWidget):
    """A widget for defining the parameters of text to be drawn."""
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        grid_layout = QGridLayout()

        color_validator = QIntValidator(0, 255, self)
        fontsize_validator = QIntValidator(1, 500, self)

        self.text_edit = QLineEdit("Calibrate")
        grid_layout.addWidget(QLabel("Text:"), 0, 0)
        grid_layout.addWidget(self.text_edit, 0, 1)

        self.font_size_edit = QLineEdit("50")
        self.font_size_edit.setValidator(fontsize_validator)
        grid_layout.addWidget(QLabel("Font Size (px):"), 1, 0)
        grid_layout.addWidget(self.font_size_edit, 1, 1)

        self.color_edit = QLineEdit("255")
        self.color_edit.setValidator(color_validator)
        grid_layout.addWidget(QLabel("Color (0-255):"), 2, 0)
        grid_layout.addWidget(self.color_edit, 2, 1)

        main_layout.addLayout(grid_layout)

        self.place_button = QPushButton("Place with Mouse")
        self.place_button.setCheckable(True)
        main_layout.addWidget(self.place_button)

        main_layout.addStretch(1)

    def get_values(self):
        """Returns the values from the input fields."""
        try:
            # Note: font scale in OpenCV is not pixels. This is a rough conversion.
            font_scale = max(0.1, int(self.font_size_edit.text()) / 20.0)
            return {
                "text": self.text_edit.text(),
                "font_scale": font_scale,
                "color": int(self.color_edit.text()),
            }
        except (ValueError, TypeError):
            return None
