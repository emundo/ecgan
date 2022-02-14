"""Optimize the parameters of a model by maximizing the performance on a given metric."""
import itertools
from logging import getLogger
from typing import List, Tuple

import numpy as np
import torch
from sklearn import svm
from torch import Tensor, from_numpy

from ecgan.evaluation.metrics.classification import ClassificationMetric, ClassificationMetricFactory
from ecgan.utils.custom_types import LabelingStrategy, MetricType, SklearnSVMKernels
from ecgan.utils.label import label

logger = getLogger(__name__)


def optimize_svm(
    metric: MetricType,
    errors: List[Tensor],
    labels: Tensor,
    kernel: SklearnSVMKernels = SklearnSVMKernels.RBF,
) -> Tuple[Tensor, svm.SVC]:
    """
    Optimize metric via Support Vector Machines (SVMs).

    Args:
        metric: The metric that has to be optimized.
        errors: The errors used to train the SVM.
        labels: Real input labels.
        kernel: Kernel used in SVM.

    Returns:
        Label predictions from SVM.
    """
    clf = svm.SVC(kernel=kernel.value)
    data_ = np.array([list_.cpu().numpy() for list_ in errors]).T
    clf.fit(data_, labels.cpu().numpy())

    predicted_labels: np.ndarray = clf.predict(data_)
    classification_metric = ClassificationMetricFactory()(metric)
    metric_score = classification_metric.calculate(
        labels.cpu().numpy(),
        predicted_labels,
    )
    logger.info("Metric score after optimizing svm {0} {1}.".format(metric.value, metric_score))

    return from_numpy(predicted_labels), clf


def query_svm(clf: svm.SVC, errors: List[Tensor], labels: Tensor) -> Tensor:
    """
    Query an already trained SVM (usually during train/vali) on test data.

    Args:
        clf: Trained classifier (SVM).
        errors: The errors used to test the SVM.
        labels: Real input labels.

    Returns:
        Label predictions from classifier.
    """
    data_ = np.array([list_.cpu().numpy() for list_ in errors]).T
    predicted_labels: np.ndarray = clf.predict(data_)
    fscore_metric = ClassificationMetricFactory()(MetricType.FSCORE)
    fscore = fscore_metric.calculate(labels.cpu().numpy(), predicted_labels)
    mcc_metric = ClassificationMetricFactory()(MetricType.MCC)
    mcc = mcc_metric.calculate(labels.cpu().numpy(), predicted_labels)
    logger.info("Metric score after using pretrained svm: F-score: {0}.".format(fscore))
    logger.info("Metric score after using pretrained svm: MCC: {0}.".format(mcc))

    return from_numpy(predicted_labels)


def optimize_metric(
    metric: MetricType,
    errors: List[Tensor],
    taus: List[float],
    params: List[List[float]],
    ground_truth_labels: Tensor,
) -> np.ndarray:
    r"""
    Optimize the given metric by weighting multiple errors using a grid-search approach.

    To achieve this, a weighted (anomaly) score will be created. If there is only one parameter, the score will be
    aggregated and it will be checked if it exceeds a given value :math:`\tau`. If the score is higher than a given
    :math:`\tau`, the value is labeled as an anomaly.

    If there are multiple error components :math:`e_i`, all combinations of error weights (params) which are less or
    equal than 1 are used to calculate an error score.
    For two errors (e.g. :ref:`AnoGAN`), a datum will be anomalous if
    :math:`\lambda_1 \cdot e_1+(1-\lambda_1) \cdot e_2 >= \tau`.

    This holds true for multiple errors :math:`e_i` and lambdas :math:`\lambda_i`:

    .. math::

        \lambda_1 \cdot e_1+\lambda_2 \cdot e_2+....+\lambda_{n-1}+e_{n-1}+\left(1 - \sum_i{\lambda_i}\right) \cdot e_n
        \geq \tau

    .. note::   While :math:`\tau` can take arbitrary values, the weighting factors have to add up to 1! To avoid
                overwhelming error components, you might want to normalize the errors.

    Args:
        errors: List of error Tensors.
        metric: The type of the metric that should be optimized.
        taus: Search range for optimizing the threshold tau.
        params: Ranges of weighting parameters (requires n-1 weights for n tensors).
            Only params adding up to <1 are considered, you do not need to ensure this tho.
        ground_truth_labels: The real labels.

    Returns:
          An array of the 10 best scores for the specified metric given the parameterization.
          The shape will be [scores, taus, params].
    """
    # Holds the 10 highest scores with the corresponding parameters:
    # (score, tau, lambda(s))
    best_weights: np.ndarray = np.full((10, len(params) + 2), -1, dtype=np.float64)

    metric_classifier = ClassificationMetricFactory()(metric)
    # Get combinations of all params (except tau) that sum up to <= 1
    combined_params = list(itertools.product(*params))
    valid_params: List[Tuple] = [param for param in combined_params if sum(param) <= 1]

    for tau in taus:
        for _, valid_param_tuple in enumerate(valid_params):
            valid_param_list = list(valid_param_tuple)

            if len(errors) == 1:  # only tau needs to be optimized
                weighted_errors = errors[0]
                weights = []
            else:
                weighted_errors = get_weighted_error(errors, valid_param_list)
                weights = valid_param_list

            best_weights = label_score_get_best(
                ground_truth_labels,
                weighted_errors,
                tau,
                metric_classifier,
                best_weights,
                weights,
            )

    return best_weights


def get_weighted_error(errors: List[Tensor], params: List[float]) -> Tensor:
    r"""
    Calculate the weighted error.

    Given n errors and n-1 parameters :math:`\lambda_i, i\in\{1,...n-1\}`, the calculation is based on the formula:

    .. math::

        \lambda_1 \cdot e_1+\lambda_2 \cdot e_2+...+\lambda_{n-1} + e_{n-1} + \left(1-\sum_i{\lambda_i}\right) \cdot e_n

    """
    if not len(errors) == len(params) + 1:
        raise ValueError("Calculating the weighted error requires exactly n errors and n-1 weights.")
    if sum(params) > 1:
        raise AttributeError("Sum or parameters may not be above 1 for weighting.")
    if len(params) == 0:  # If no params are given: only one error is used -> only the else case is triggered.
        params = [0.0]

    weighted_error_list = [
        (errors[index] * params[index]).tolist()
        if index is not len(errors) - 1
        else (errors[index] * (1.0 - sum(params))).tolist()
        for index in range(0, len(errors))
    ]
    return torch.sum(torch.Tensor(weighted_error_list), dim=0)


def label_score_get_best(
    ground_truth_labels: Tensor,
    weighted_errors: Tensor,
    tau: float,
    metric_classifier: ClassificationMetric,
    best_weights: np.ndarray,
    weights: List[float],
) -> np.ndarray:
    """
    Generate labels based on an absolute threshold and calculate the metric.

    Check if the returned metric is one of the best weights and return the updated array of best weights.
    """
    predicted_labels = label(weighted_errors, LabelingStrategy.POINTWISE, tau)

    avg_score = metric_classifier.calculate(
        ground_truth_labels.numpy(),
        predicted_labels.numpy(),
    )

    new_best_weights = best_weights
    # To reproduce the labeling we require tau as well as the weights.
    # Additionally we return the metric score to give some context on the quality of the returned weights.
    for idx, entry in enumerate(new_best_weights):
        if entry[0] < avg_score:
            new_best_weights[idx] = np.concatenate(([avg_score, tau], weights), axis=None)
            break

    return new_best_weights


def retrieve_labels_from_weights(errors: List[Tensor], tau: float, weighting_params: List[float]) -> torch.Tensor:
    """Retrieve labels from a given pair of weighting parameters."""
    weighted_error: Tensor = get_weighted_error(errors, weighting_params)

    return label(weighted_error, LabelingStrategy.POINTWISE, tau)


def optimize_grid_search(
    metric: MetricType, errors: List[Tensor], labels: Tensor, params: List[List[float]], taus: List[float]
) -> Tensor:
    """Optimize anomaly detection via grid-search."""
    best_params = optimize_metric(
        metric,
        errors=errors,
        taus=taus,
        params=params,
        ground_truth_labels=labels,
    )
    logger.info("Best params from grid search: {}.".format(best_params))
    # Best params format: [0]: fscore, [1]: tau, [2...n]: params
    return retrieve_labels_from_weights(errors, best_params[0][1], best_params[0][1:])


def optimize_tau_single_error(
    true_labels: Tensor, error: Tensor, tau_range: List, metric: MetricType = MetricType.FSCORE
) -> float:
    """
    Optimize threshold given a metric for a single error.

    Args:
        true_labels: Real labels of the data.
        error: List of errors (1D Tensor) using any metric which can be used to formulate a thresholdable error score.
        tau_range: Range of taus grid searched.
        metric: Metric to optimize on.

    Returns:
        Highest score.
    """
    best_score: float = 0.0
    metric_classifier = ClassificationMetricFactory()(metric)

    for tau in tau_range:
        predicted_labels = label(error, LabelingStrategy.POINTWISE, tau)

        avg_score = metric_classifier.calculate(
            true_labels.numpy(),
            predicted_labels.numpy(),
        )
        if avg_score > best_score:
            best_score = avg_score

    return best_score
