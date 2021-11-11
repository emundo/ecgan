"""Functions to visualize evaluation metrics."""
from logging import getLogger
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from ecgan.utils.plotting import matplotlib_render_to_array

getLogger('matplotlib.font_manager').disabled = True
logger = getLogger(__name__)


def boxplot(
    data: Union[List[np.ndarray], object],
    label: List,
    title: str,
) -> np.ndarray:
    """
    Create a boxplot using mpl.

    Args:
        data: List of data points or List of Lists containing data points if multiple metrics are tracked.
        label: List of labels for each plot.
        title: Title of the plot (usually the name of the metric).

    Returns:
        Boxplot image encoded as np.ndarray.
    """
    figure = plt.figure(1)
    axes = figure.add_subplot(111)
    axes.set_title(title)
    axes.boxplot(data, labels=label, vert=False, whis=0.75)
    plt.tight_layout()

    return matplotlib_render_to_array(figure)


def visualize_roc(true_labels: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    """
    Calculate and draw the ROC curve for our model using the predicted labels.

    Args:
        true_labels: Ground truth.
        predicted_labels: Labels predicted by model.

    Returns:
        ROC curve image encoded as np.ndarray.
    """
    figure = plt.figure(2)
    axes = figure.add_subplot(111)
    axes.set_title('AUROC')
    fpr, tpr, _thresholds = roc_curve(true_labels, predicted_labels, pos_label=0)

    plt.plot(fpr, tpr)

    return matplotlib_render_to_array(figure)
