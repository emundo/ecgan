"""Implementation of AD algorithms based on classification."""
from abc import ABC
from logging import getLogger
from typing import Dict, Optional

from torch import Tensor, argmax, tensor

from ecgan.anomaly_detection.detector.base_detector import AnomalyDetector
from ecgan.evaluation.tracker import BaseTracker
from ecgan.modules.classifiers.base import BaseClassifier
from ecgan.utils.custom_types import AnomalyDetectionStrategies
from ecgan.utils.miscellaneous import get_num_workers

logger = getLogger(__name__)


class ClassificationDetector(AnomalyDetector, ABC):
    """
    Base class for anomaly detectors which directly classify data.

    The anomalousness is asserted based on the classification score.
    """

    def __init__(self, module: BaseClassifier, tracker: BaseTracker):
        super().__init__(module, tracker)
        self._labels: Optional[Tensor] = None
        self._scores: Optional[Tensor] = None

    def _get_data_to_save(self) -> Dict:
        """Select data that shall be saved after anomaly detection."""
        return {
            'labels': self._labels.tolist() if self._labels is not None else None,
            'scores': self._scores.tolist() if self._scores is not None else None,
        }

    def load(self, saved_data: Dict):
        """Load data from dict."""
        self._labels = tensor(saved_data['labels']) if saved_data['labels'] is not None else None
        self._scores = tensor(saved_data['scores']) if saved_data['scores'] is not None else None


class ArgmaxClassifierDetector(ClassificationDetector):
    """Detector which utilizes the maximum output of a classifier to predict labels."""

    def __init__(self, module: BaseClassifier, tracker: BaseTracker):
        super().__init__(module, tracker)
        self.classifier = module

    def _detect(self, data: Tensor) -> Tensor:
        """
        Detect anomalies.

        Args:
            data: Tensor (usually of size [series, channel, data points]) of data which shall be classified.

        Returns:
            A Tensor with the corresponding labels.
        """
        self._scores: Tensor = self.classifier.classify(data)
        self._labels = argmax(self._scores, dim=1).cpu()

        return self._labels

    @staticmethod
    def configure() -> Dict:
        """Configure the default settings for an RNNClassifierDetector."""
        return {
            'detection': {
                'BATCH_SIZE': 64,
                'NUM_WORKERS': get_num_workers(),
                'DETECTOR': AnomalyDetectionStrategies.ARGMAX.value,
                'AMOUNT_OF_RUNS': 1,
                'SAVE_DATA': False,
            }
        }
