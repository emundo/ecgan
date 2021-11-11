"""Base class used for anomaly detection."""
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Dict, Union

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecgan.config import get_global_ad_config
from ecgan.evaluation.tracker import BaseTracker
from ecgan.modules.base import BaseModule
from ecgan.modules.classifiers.base import BaseClassifier
from ecgan.training.datasets import SeriesDataset
from ecgan.utils.artifacts import FileArtifact
from ecgan.utils.configurable import Configurable

logger = getLogger(__name__)


class AnomalyDetector(Configurable, ABC):
    """
    A baseclass for various (PyTorch based) anomaly detectors.

    This class can be used to implement general anomaly detection algorithms and it is not limited to deep learning/
    machine learning approaches. Each :code:`AnomalyDetector` should be able to

        #. Assert labels to a given time series.
        #. Save/load relevant evaluation data to/from a pkl file. This includes at least the labels but can be expanded
           for arbitrary information such as anomaly scores or data obtained from reconstructions.

    The labeling can depend on the type of detection (e.g. one score per series, channel or point). Thus, no specific
    format will be enforced. In general, we will save the labels per series and if scores are used to determine if some
    point is anomalous, we use pointwise scoring whenever possible, since channel-/serieswise anomalies can usually be
    reconstructed based on that data. The actual performance measures are controlled by the AnomalyManager based on the
    predicted labels.
    """

    def __init__(self, module: Union[BaseModule, BaseClassifier], tracker: BaseTracker):
        self.module = module
        self.cfg = get_global_ad_config()
        self.tracker = tracker

    @abstractmethod
    def _detect(self, data: Tensor) -> Tensor:
        """
        Detect anomalies based on the desired detection scheme and return the asserted class labels.

        Args:
            data: Tensor (usually of size [batch_size, seq_len, channels]) of data which shall be classified.

        Returns:
            A Tensor with the label predictions for `data`.
        """
        raise NotImplementedError("AnomalyDetector needs to implement the `_detect` method.")

    def detect(self, test_x: Tensor, test_y: Tensor) -> np.ndarray:
        """
        Detect anomalies based on the desired detection scheme and return the asserted class labels.

        Data is expected to be shuffled when passed to the detect method. It is then fed into a DataLoader, chunked
        into batches on which anomalies are detected.

        The function calls the abstract `._detect` method and logs wall time.

        Args:
            test_x: The shuffled test data.
            test_y: The labels corresponding to test_y.

        Returns:
            Predicted labels.
        """
        predicted_labels = np.empty(test_y.shape)

        test_dataset = SeriesDataset(
            test_x.float(),
            test_y.float(),
        )

        dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg.detection_config.BATCH_SIZE,
            num_workers=self.cfg.detection_config.NUM_WORKERS,
            pin_memory=True,
        )

        batch_size = dataloader.batch_size if dataloader.batch_size is not None else 0

        for batch_idx, batch in enumerate(tqdm(dataloader, leave=False)):
            data = batch['data'].to(self.module.device) if isinstance(self.module, BaseModule) else batch['data']
            predicted_batch_labels = self._detect(data=data)
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = (1 + batch_idx) * batch_size
            predicted_labels[batch_start_idx:batch_end_idx] = predicted_batch_labels.numpy()

        return predicted_labels

    @abstractmethod
    def _get_data_to_save(self) -> Dict:
        """Select list of objects which shall be saved using the tracker."""
        raise NotImplementedError("AnomalyDetector needs to implement the `_get_data_to_save` method.")

    def save(self, run_id: str) -> None:
        """Save anomaly detection results to tracker."""
        self.tracker.log_artifacts(
            FileArtifact(
                'Anomaly Detection Data',
                self._get_data_to_save(),
                'detection_data_{}.pkl'.format(run_id),
            )
        )

    @abstractmethod
    def load(self, saved_data: Dict) -> None:
        """
        Load AD data from dict.

        The provided dict is usually part of an output of a previous AD run.
        The `load` method loads the saved data to the instantiated detector  and can subsequently used in to e.g.
        create embeddings without reprocessing all data. The user is tasked to retrieve the saved dict by themself.

        Args:
            saved_data: Previously saved data, loaded into variables of the respective detector.
        """
        raise NotImplementedError("AnomalyDetector needs to implement the `load` method.")
