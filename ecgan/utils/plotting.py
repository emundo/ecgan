"""Utility functions for rendering matplotlib figures to numpy arrays."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def matplotlib_render_to_array(fig: Figure) -> np.ndarray:
    """
    Return a numpy array representing the plot of a matplotlib figure.

    A bit hacky.
    """
    canvas = FigureCanvas(fig)
    canvas.draw()
    plt.close()
    # noinspection PyProtectedMember
    return np.array(fig.canvas.get_renderer()._renderer)  # pylint: disable=protected-access
