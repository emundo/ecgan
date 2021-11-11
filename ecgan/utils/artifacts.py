"""Supported artifact types which are used as templates for tracking experiments."""
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from ecgan.utils.plotting import matplotlib_render_to_array


class Artifact:
    """Abstract base class for artifacts supported by the tracker."""

    def __init__(self, name: str):
        self.name = name


class ImageArtifact(Artifact):
    """Artifact wrapper for an image encoded as np.ndarray or a mpl Figure."""

    def __init__(self, name: str, image: Union[np.ndarray, plt.Figure]):
        super().__init__(name)
        self.figure = None
        if isinstance(image, plt.Figure):
            self.image = matplotlib_render_to_array(image)
            self.figure = image

        else:
            self.image = image

    def __del__(self):
        """Close the internal figure when the object is destroyed."""
        if self.figure is not None:
            plt.close(self.figure)


class ValueArtifact(Artifact):
    """Create an artifact which stores a single value (e.g. metric)."""

    def __init__(self, name: str, value: Union[float, Dict]):
        super().__init__(name)
        self.value = value


class FileArtifact(Artifact):
    """Create an artifact containing a dictionary which shall be saved into a yaml or pickle file."""

    def __init__(self, name: str, data: Any, file_name: str):
        """
        Initialize a file artifact.

        Args:
            name: Name of the saved data - ONLY USED FOR LOGGING, does not have to be a file name.
            data: Data that shall be saved.
            file_name: Saving location. Has to point to a .yml or .pkl file.
        """
        super().__init__(name)
        self.data = data
        self.file_name = file_name
