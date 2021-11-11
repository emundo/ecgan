"""Definition of a base PyTorch classifier."""
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import Tensor

from ecgan.evaluation.metrics.classification import AUROCMetric, FScoreMetric, MCCMetric


class BaseClassifier(ABC):
    """
    Abstract baseclass for classification models.

    Each classifier is expected to handle queries of incoming data.
    """

    @abstractmethod
    def classify(self, data: Tensor) -> Tensor:
        """Return a classification score."""
        raise NotImplementedError("Classifier needs to implement the `classify` method.")

    @staticmethod
    def get_classification_metrics(
        real_label: np.ndarray,
        prediction_labels: np.ndarray,
        stage: str = 'metrics/',
        get_fscore_weighted: bool = False,
        get_fscore_micro: bool = False,
        get_prec_recall_fscore=True,
        get_accuracy: bool = True,
        get_mcc: bool = True,
        get_auroc: bool = True,
    ) -> Dict:
        """
        Compute classification metrics for given input data and prediction.

        Args:
            real_label: Real input label (y).
            prediction_labels: Predicted label (y_hat).
            stage: String identifier for the logging stage.
            get_prec_recall_fscore: Flag to indicate if precision, recall, F-score (macro) and/or support are computed.
            get_fscore_weighted: Flag to indicate if the weighted F-score should be computed.
            get_fscore_micro: Flag to indicate if the micro F-score should be computed.
            get_accuracy: Flag indicating if the accuracy should be computed.
            get_mcc: Flag indicating if the MCC should be computed.
            get_auroc: Flag indicating if the AUROC should be computed.

        Returns:
            Dict containing all metrics that were marked as to be computed.
        """
        metrics_dict = {}
        if get_accuracy:
            acc = accuracy_score(real_label, prediction_labels)
            metrics_dict['{0}{1}'.format(stage, 'accuracy')] = acc
        if get_fscore_weighted:
            fscore_weighted = FScoreMetric(average='weighted').calculate(real_label, prediction_labels)
            metrics_dict['{0}{1}'.format(stage, 'fscore_weighted')] = fscore_weighted
        if get_fscore_micro:
            fscore_micro = FScoreMetric(average='micro').calculate(real_label, prediction_labels)
            metrics_dict['{0}{1}'.format(stage, 'fscore_micro')] = fscore_micro
        if get_prec_recall_fscore:
            prec, recall, fscore_macro, _support = precision_recall_fscore_support(
                real_label, prediction_labels, warn_for=tuple(), average='macro'
            )
            metrics_dict['{0}{1}'.format(stage, 'precision')] = prec
            metrics_dict['{0}{1}'.format(stage, 'recall')] = recall
            metrics_dict['{0}{1}'.format(stage, 'fscore_macro')] = fscore_macro
        if get_mcc:
            mcc = MCCMetric().calculate(real_label, prediction_labels)
            metrics_dict['{0}{1}'.format(stage, 'mcc')] = mcc
        if get_auroc:
            # Auroc not always defined if not all labels are inside each batch.
            # We use binary Auroc (class 0: normal, class 1: all abnormal classes) as a heuristic.
            # Should be changed in the proper evaluation of the test dataset
            auroc = AUROCMetric().calculate(real_label > 0, prediction_labels > 0)
            metrics_dict['{0}{1}'.format(stage, 'auroc')] = auroc

        return metrics_dict
