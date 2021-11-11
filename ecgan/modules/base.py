"""Abstract base module for learning algorithms which use training and validation steps."""
from __future__ import annotations

from abc import abstractmethod
from logging import getLogger
from typing import List

from ecgan.config import ModuleConfig, get_global_config
from ecgan.utils.artifacts import Artifact
from ecgan.utils.configurable import Configurable
from ecgan.utils.custom_types import SampleDataset
from ecgan.utils.datasets import DatasetFactory
from ecgan.utils.miscellaneous import select_device
from ecgan.utils.sampler import DataSampler
from ecgan.visualization.plotter import PlotterFactory

logger = getLogger(__name__)


class BaseModule(Configurable):
    """Base class from which all implemented modules should inherit."""

    def __init__(
        self,
        cfg: ModuleConfig,
        seq_len: int,
        num_channels: int,
    ):
        self.cfg = cfg
        exp_cfg = get_global_config().experiment_config
        trainer_cfg = get_global_config().trainer_config
        self.dataset = DatasetFactory()(exp_cfg.DATASET)
        self.plotter = PlotterFactory.from_config(trainer_cfg)

        self.seq_len = seq_len
        self.num_channels = num_channels

        self.device = select_device(gpu_flag=exp_cfg.TRAIN_ON_GPU)

        logger.info('Using device {0}.'.format(self.device))

        self.train_dataset_sampler: DataSampler = DataSampler(
            None, self.device, num_channels, seq_len, name=SampleDataset.TRAIN.value
        )
        self.vali_dataset_sampler: DataSampler = DataSampler(
            None, self.device, num_channels, seq_len, name=SampleDataset.VALI.value
        )

    @abstractmethod
    def training_step(
        self,
        batch: dict,
    ) -> dict:
        """
        Declare what the model should do during a training step using a given batch.

        The returned metrics are concatenated across batches and **averaged** before logging.
        Note: this is important if you want to log min/max values!

        Args:
            batch: A batch of data.

        Return:
            A dict containing the metrics from optimization or evaluation which shall be logged.
        """
        raise NotImplementedError("BaseModule needs to implement the `training_step` method.")

    @abstractmethod
    def validation_step(
        self,
        batch: dict,
    ) -> dict:
        """
        Declare what the model should do during a validation step.

        Args:
            batch: A batch of data.

        Return:
            A dict containing the metrics from optimization or evaluation which shall be logged.
        """
        raise NotImplementedError("BaseModule needs to implement the `validation_step` method.")

    @abstractmethod
    def save_checkpoint(self) -> dict:
        """Return current model parameters."""
        raise NotImplementedError("BaseModule needs to implement the `save_checkpoint` method.")

    @abstractmethod
    def load(self, model_reference: str, load_optim: bool = False) -> BaseModule:
        """Load a trained module from disk (file path) or wand reference."""
        raise NotImplementedError("BaseModule needs to implement the `load` method.")

    @property
    @abstractmethod
    def watch_list(self) -> List:
        """Return torch nn.Modules that should be watched during training."""
        pass

    @staticmethod
    def _print_metric(epoch: int, metrics: dict) -> None:
        """Print formatted metrics that were produced by the model."""
        formatted_metrics: str = 'Ep. {0} -> '.format(epoch)
        for key, value in metrics.items():
            formatted_metrics += '{0}: {1:.4f} | '.format(key, value)
        logger.info(formatted_metrics)

    @classmethod
    def print_metric(cls, epoch, metrics) -> None:
        """Allow overwriting the static `_print_metric` method."""
        cls._print_metric(epoch, metrics)

    @abstractmethod
    def on_epoch_end(self, epoch: int, sample_interval: int, batch_size: int) -> List[Artifact]:
        """
        Set actions to be executed after epoch ends.

        Declare what should be done upon finishing an epoch (e.g. save artifacts or
        evaluate some metric).

        Args:
            epoch: Current training epoch.
            sample_interval: Regular sampling interval to save modules independent of their performance.
            batch_size: Size of batch.

        Returns:
            List containing all :class:`ecgan.utils.artifacts.Artifact` s which shall be logged upon epoch end.
        """
        raise NotImplementedError("Module needs to implement the `on_epoch_end` method.")
