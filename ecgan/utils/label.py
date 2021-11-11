"""Functions to label synthetic data."""
from logging import getLogger
from typing import List, Optional

from torch import Tensor, count_nonzero, ge, mean, tensor, var

from ecgan.utils.custom_types import LabelingStrategy

logger = getLogger(__name__)


def label_generated_data_pointwise(anomaly_scores: Tensor, tau: float) -> Tensor:
    """
    Labeling of generated data depending on some tau.

    Args:
        anomaly_scores: Pointwise anomaly scores
        tau: Anomaly threshold.

    Returns:
        The labels of generated series/data points.
    """
    return ge(anomaly_scores, tau)


def label_data_by_summation(anomaly_scores: Tensor, tau, channelwise: bool = True) -> Tensor:
    """
    Calculate one label per channel (or series if channelwise=False).

    Utilizes the sum of pointwise anomaly scores and checks if the **average** anomaly
    score is below tau.

    Args:
        anomaly_scores: Pointwise anomaly scores.
        tau: (Pointwise) anomaly threshold.
        channelwise: Flag to indicate if the data should be labeled channelwise.

    Returns:
        One label for each series or channel, meaning anomaly_scores.shape[0] labels for
        serieswise scoring and respectively anomaly_score.shape[0] * anomaly_score.shape[2]
        labels for channelwise detection will be returned.
    """
    if channelwise:
        return tensor([[mean(channel) > tau for channel in series] for series in anomaly_scores])
    if len(anomaly_scores.size()) == 0:
        return (mean(anomaly_scores) > tau).clone().detach()  # type: ignore
    return tensor([mean(series) > tau for series in anomaly_scores])


def label_data_by_variance(anomaly_scores: Tensor, tau: float, channelwise: bool = True) -> Tensor:
    """
    Calculate one label per channel (or series if channelwise=False).

    Utilizes the variance of pointwise anomaly scores and checks if the anomaly score
    is below the given tau.

    Args:
        anomaly_scores: Pointwise anomaly scores.
        tau: (Pointwise) anomaly threshold.
        channelwise: Flag indicating if you want to return channelwise or serieswise anomaly scores.

    Returns:
        One label for each series or channel, meaning anomaly_scores.shape[0] labels for
        serieswise scoring and respectively anomaly_score.shape[0] * anomaly_score.shape[2]
        labels for channelwise detection will be returned.
    """
    if channelwise:
        return tensor([[float(var(channel)) > tau for channel in series] for series in anomaly_scores])
    return tensor([float(var(series)) > tau for series in anomaly_scores])


def label_absolute(
    anomaly_scores: Tensor,
    tau: float = 0.2,
    anomaly_lower_bound: Optional[int] = None,
) -> Tensor:
    """
    Label channels based on the absolute amount of pointwise anomalies.

    A channel is labeled as anomalous if more than `anomaly_lower_bound` samples are
    labeled during the pointwise detection.
    """
    if anomaly_lower_bound is None:
        anomaly_lower_bound = int(anomaly_scores.shape[1] / 20)
    labels = label_generated_data_pointwise(anomaly_scores=anomaly_scores, tau=tau)
    labels_with_bound: List = [
        [count_nonzero(channel).int() > anomaly_lower_bound for channel in series] for series in labels
    ]

    return tensor(labels_with_bound)


def label(
    anomaly_scores: Tensor,
    strategy: LabelingStrategy = LabelingStrategy.POINTWISE,
    tau: float = 0.2,
) -> Tensor:
    """
    Label synthetic data based on the respective anomaly scores.

    Args:
        anomaly_scores: Series of pointwise anomaly scores.
        strategy: Labeling strategy: either pointwise, channelwise or serieswise.
        tau: Anomaly threshold.

    Returns:
        Labels for each data point, channel or series. User has to ensure the correct
        format.
    """
    if strategy == LabelingStrategy.POINTWISE:
        return label_generated_data_pointwise(anomaly_scores=anomaly_scores, tau=tau)
    if strategy == LabelingStrategy.ACCUMULATE_CHANNELWISE:
        return label_data_by_summation(anomaly_scores=anomaly_scores, tau=tau)
    if strategy == LabelingStrategy.ACCUMULATE_SERIESWISE:
        return label_data_by_summation(anomaly_scores=anomaly_scores, tau=tau, channelwise=False)
    if strategy == LabelingStrategy.VARIANCE_CHANNELWISE:
        return label_data_by_variance(anomaly_scores=anomaly_scores, tau=tau)
    if strategy == LabelingStrategy.VARIANCE_SERIESWISE:
        return label_data_by_variance(anomaly_scores=anomaly_scores, tau=tau, channelwise=False)
    raise ValueError("Unknown LabelingStrategy: {}.".format(strategy))
