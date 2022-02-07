"""Calculations and wrappers of various metrics."""
from abc import ABC, abstractmethod
from logging import getLogger
from typing import List, Union

import numpy as np
import sklearn.metrics as skm
import torch
from sklearn.metrics import roc_auc_score

from ecgan.utils.custom_types import MetricType, SklearnAveragingOptions
from ecgan.utils.miscellaneous import to_numpy

logger = getLogger(__name__)


class ClassificationMetric(ABC):
    """Classification metric base class."""

    @abstractmethod
    def calculate(self, y: Union[torch.Tensor, np.ndarray], y_hat: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Calculate the metric based on ground truth labels y and predicted labels y_hat.

        Labels can be either integer or boolean arrays but have to be numpy arrays for the flattening.
        """
        raise NotImplementedError("ClassificationMetric needs to implement the `calculate` method.")


class FScoreMetric(ClassificationMetric):
    """Create F-score objects."""

    def __init__(
        self,
        beta: float = 1,
        average: str = SklearnAveragingOptions.WEIGHTED.value,
    ):
        """
        Initialize a F-score metric object.

        Args:
            beta: The weighting of the precision. Only required for the f-score
            average: Determines how you want to average the F-score.
                NOTE: we currently assume `weighted`, be careful with `None` or `binary`.
        """
        self.beta = beta
        self.average = average

    def calculate(self, y: Union[torch.Tensor, np.ndarray], y_hat: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Calculate the f-beta score (beta defaults to 1) and track it if desired.

        If the f-score is chosen as a metric, the precision and recall will also be saved but not returned to the user.

        Args:
            y: Ground truth labels.
            y_hat: Predicted labels.

        Returns:
            The resulting F_beta score. Usually weighted to account for imbalanced classes,
                returns average score if the classwise F-score was calculated.
        """
        # Can return a single value if the F_beta-score is weighted, can also return a list of values (score per class).
        y = to_numpy(y)
        y_hat = to_numpy(y_hat)

        _, _, fscore, _ = skm.precision_recall_fscore_support(
            np.ndarray.flatten(y),
            np.ndarray.flatten(y_hat),
            beta=self.beta,
            average=self.average,
            zero_division=1,
        )

        if isinstance(fscore, float):
            return float(fscore)

        if isinstance(fscore, list):
            for index, score in enumerate(fscore):
                logger.debug('F-{0} score for class {1} is {2}.\n '.format(self.beta, index, score))
            # Be careful when working with the unweighted F-score.
            logger.info(
                'WARNING: you are working with an averaged F-score, \
                you might want to consider the class (im)balance.'
            )
            return float(np.average(fscore))
        raise ValueError('Unexpected type of F-score: {}'.format(type(fscore)))


class MCCMetric(ClassificationMetric):
    """Create a MCC object."""

    def calculate(self, y: Union[torch.Tensor, np.ndarray], y_hat: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Calculate the mcc score and track if desired.

        Args:
            y: Ground truth labels of shape (num_samples,).
            y_hat: Predicted labels of shape (num_samples,).

        Returns:
            float: The average mcc value.
        """
        y = to_numpy(y)
        y_hat = to_numpy(y_hat)
        mcc_values: List[float] = []

        mcc: float = skm.matthews_corrcoef(
            y,
            y_hat,
        )
        mcc_values.append(mcc)

        average_mcc = np.average(mcc_values)

        return float(average_mcc)


class AUROCMetric(ClassificationMetric):
    """Create an AUROC object."""

    def __init__(self, average: str = 'weighted'):
        """
        Initialize an AUROC object.

        More information: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

        Args:
            average: Type of data averaging. Default: weighted average.
        """
        self.average = average

    def calculate(self, y: Union[np.ndarray, torch.Tensor], y_hat: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Calculate the AUROC score and track if desired.

        If only one class label is present, AUROC is ill-defined. To avoid exceptions, we set the auroc to 0 if only one
        class is present. Since this can distort results you should make sure that the batches are large enough.

        Args:
            y: Ground truth labels of shape (num_samples,).
            y_hat: Predicted labels of shape (num_samples,).

        Returns:
            AUROC score.
        """
        y = to_numpy(y)
        y_hat = to_numpy(y_hat)

        if np.unique(y).__len__() == 1:
            logger.debug('Only one class label. Returning 0 auroc.')
            return 0.0
        if np.unique(y).__len__() > 2:
            logger.debug('Multiple classes: num classes is {}.'.format(np.unique(y)))

        auroc = roc_auc_score(y, y_hat, average=self.average, multi_class='ovr')

        if isinstance(auroc, float):
            return auroc

        if isinstance(auroc, list):
            # Be careful when working with the unweighted AUROC.
            logger.info(
                'WARNING: you are working with an averaged AUROC, you might want to consider the class (im)balance.'
            )
            return float(np.average(auroc))
        raise ValueError('Unexpected type of AUROC: {0}'.format(type(auroc)))


class AvgPrecisionMetric(ClassificationMetric):
    """Create an AP object."""

    def calculate(self, y: Union[torch.Tensor, np.ndarray], y_hat: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Calculate the average precision score.

        Args:
            y: Ground truth labels of shape (num_samples,).
            y_hat: Predicted labels of shape (num_samples,).

        Returns:
            float: The average precision score.
        """
        y = to_numpy(y)
        y_hat = to_numpy(y_hat)

        average_precision: float = skm.average_precision_score(
            y,
            y_hat,
        )

        return average_precision


class ClassificationMetricFactory:
    """Meta module for creating classification metric objects."""

    def __call__(
        self,
        metric: MetricType,
        **kwargs,
    ) -> ClassificationMetric:
        """Return implemented module when a Loss object is created."""
        metrics = {
            MetricType.FSCORE: FScoreMetric(**kwargs),
            MetricType.MCC: MCCMetric(),
            MetricType.AUROC: AUROCMetric(**kwargs),
            MetricType.AP: AvgPrecisionMetric(),
        }
        try:
            return metrics[metric]
        except KeyError as err:
            raise AttributeError('Argument {0} is not set correctly.'.format(metric.value)) from err
