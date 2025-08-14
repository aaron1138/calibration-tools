# project.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

class EditCommand:
    """Base class for a non-destructive edit."""
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the edit to the given image (a NumPy array) and returns the modified image.
        """
        raise NotImplementedError

    def get_affected_layer(self) -> int:
        """Returns the layer index this command belongs to."""
        raise NotImplementedError

@dataclass
class Project:
    """Manages the state of an editing project."""
    source_file_path: str = ""
    layer_image_paths: List[str] = field(default_factory=list)

    # Stores edits per layer: {layer_index: [EditCommand, ...]}
    edits: Dict[int, List[EditCommand]] = field(default_factory=dict)

    # For Undo/Redo functionality
    undo_stack: List[EditCommand] = field(default_factory=list)
    redo_stack: List[EditCommand] = field(default_factory=list)

    @property
    def total_layers(self) -> int:
        """Returns the total number of layers in the project."""
        return len(self.layer_image_paths)

    def add_edit(self, command: EditCommand):
        """Adds a new edit command to a specific layer and the undo stack."""
        layer_index = command.get_affected_layer()
        if layer_index not in self.edits:
            self.edits[layer_index] = []

        self.edits[layer_index].append(command)
        self.undo_stack.append(command)
        # Adding a new edit clears the redo stack
        self.redo_stack.clear()

    def _find_and_remove_command(self, command_to_remove: EditCommand):
        """Finds a specific command in the edits dictionary and removes it."""
        layer_index = command_to_remove.get_affected_layer()
        if layer_index in self.edits:
            try:
                self.edits[layer_index].remove(command_to_remove)
            except ValueError:
                # This can happen if the command was already removed but is still in the stack
                pass

    def perform_undo(self) -> Optional[EditCommand]:
        """
        Moves the last command from the undo stack to the redo stack and removes
        it from the list of active edits. Returns the command that was undone.
        """
        if not self.undo_stack:
            return None
        command = self.undo_stack.pop()
        self._find_and_remove_command(command)
        self.redo_stack.append(command)
        return command

    def perform_redo(self) -> Optional[EditCommand]:
        """
        Moves the last command from the redo stack back to the undo stack and
        re-adds it to the list of active edits. Returns the command that was redone.
        """
        if not self.redo_stack:
            return None
        command = self.redo_stack.pop()

        # Re-add the command to the edits list
        layer_index = command.get_affected_layer()
        if layer_index not in self.edits:
            self.edits[layer_index] = []
        self.edits[layer_index].append(command)

        self.undo_stack.append(command)
        return command

@dataclass
class DrawRectangleCommand(EditCommand):
    """An command to draw a rectangle."""
    layer_index: int
    x: int
    y: int
    width: int
    height: int
    color: int # Grayscale value 0-255

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Draws the rectangle on the image."""
        # The image is expected to be a CV2 image (numpy array)
        # Note: cv2.rectangle modifies the image in-place.
        pt1 = (self.x, self.y)
        pt2 = (self.x + self.width, self.y + self.height)
        # -1 thickness fills the rectangle
        cv2.rectangle(image, pt1, pt2, (self.color,), thickness=-1)
        return image

    def get_affected_layer(self) -> int:
        return self.layer_index

@dataclass
class DrawTextCommand(EditCommand):
    """A command to draw text."""
    layer_index: int
    text: str
    position: tuple[int, int] # (x, y) of the bottom-left corner of the text
    font_scale: float
    color: int
    thickness: int = 2 # A reasonable default thickness

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Draws the text on the image."""
        output_image = image.copy()

        # Use a standard OpenCV font
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            img=output_image,
            text=self.text,
            org=self.position,
            fontFace=font,
            fontScale=self.font_scale,
            color=(self.color,),
            thickness=self.thickness
        )
        return output_image

    def get_affected_layer(self) -> int:
        return self.layer_index

@dataclass
class DrawGradientCommand(EditCommand):
    """An command to draw a gradient within a rectangular region."""
    layer_index: int
    x: int
    y: int
    width: int
    height: int
    start_color: int
    end_color: int
    gradient_type: str = "linear"

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Draws the gradient within the defined rectangle on the image."""
        if self.gradient_type == "linear":
            # Define the bounding box for the gradient
            x_start, y_start = self.x, self.y
            x_end, y_end = self.x + self.width, self.y + self.height

            # Clamp the bounding box to the image dimensions
            h, w = image.shape
            x_start_c = max(0, x_start)
            y_start_c = max(0, y_start)
            x_end_c = min(w, x_end)
            y_end_c = min(h, y_end)

            # If the rect is completely outside the image, do nothing
            if x_start_c >= x_end_c or y_start_c >= y_end_c:
                return image

            # Define the gradient vector (top-left to bottom-right of the original rect)
            grad_vec = np.array([self.width, self.height], dtype=float)
            grad_len_sq = grad_vec[0]**2 + grad_vec[1]**2

            if grad_len_sq < 1e-6: # Handle zero-size rect
                return image

            # Create coordinate grid ONLY for the relevant part of the image
            y_coords, x_coords = np.mgrid[y_start_c:y_end_c, x_start_c:x_end_c]

            # Vectors from the gradient start point to each pixel in the sub-grid
            pixel_vecs = np.stack([x_coords.ravel() - x_start, y_coords.ravel() - y_start], axis=1)

            # Project and normalize
            t = np.dot(pixel_vecs, grad_vec) / grad_len_sq
            t = np.clip(t, 0, 1)

            # Interpolate colors
            gradient_values = (self.start_color * (1 - t) + self.end_color * t).astype(np.uint8)
            gradient_patch = gradient_values.reshape((y_end_c - y_start_c, x_end_c - x_start_c))

            # Apply the patch to the output image
            output_image = image.copy()
            output_image[y_start_c:y_end_c, x_start_c:x_end_c] = gradient_patch

            return output_image

        return image

    def get_affected_layer(self) -> int:
        return self.layer_index
