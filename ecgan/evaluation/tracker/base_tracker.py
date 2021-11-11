"""Base class for trackers."""
import os
from abc import ABC, abstractmethod
from statistics import mean
from typing import Dict, List, Union

import torch
import yaml

from ecgan.utils.artifacts import Artifact, FileArtifact
from ecgan.utils.miscellaneous import save_pickle
from ecgan.visualization.plotter import BasePlotter


class BaseTracker(ABC):
    """Base class for trackers."""

    def __init__(self, entity: str, project: str, run_name: str, save_pdf: bool = False):
        """Set basic tracking parameters."""
        self.entity = entity
        self.project = project
        self.run_name = run_name
        self.step = 1
        self.plotter = BasePlotter()
        self.fold = 1
        self.run_dir = self._init_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)
        self.save_pdf = save_pdf

    def advance_step(self):
        """Advance by one step."""
        self.step += 1

    def advance_fold(self):
        """Advance by one fold."""
        self.fold += 1

    @abstractmethod
    def _init_run_dir(self) -> str:
        """Initialize the run directory and returns the file path."""
        raise NotImplementedError("Tracker needs to implement the `_init_run_dir` method.")

    @abstractmethod
    def close(self):
        """Close training run."""
        raise NotImplementedError("Tracker needs to implement the `close` method.")

    @staticmethod
    def collate_metrics(metrics: List[Dict]) -> Dict:
        """
        Transform a list with dictionaries to a single dictionary.

        All values are collated in their respective keys and are then averaged.

        Args:
            metrics: A list of metrics.

        Returns:
            Dictionary with the collated metrics.
        """
        metrics_collated = {}
        key_list: List = []
        for metric_dict in metrics:
            key_list.extend(metric_dict.keys())
        key_list = list(set(key_list))

        for key in key_list:
            metric_list = [metric.get(key) for metric in metrics if metric.get(key) is not None]
            metrics_collated.update({key: mean(metric_list)})  # type: ignore
        return metrics_collated

    @abstractmethod
    def watch(self, model: Union[torch.nn.Module, List[torch.nn.Module]]) -> None:
        """Pass a module that should be watched during training."""
        pass

    @abstractmethod
    def log_config(self, cfg: Dict) -> None:
        """Log parameters for training setup."""
        raise NotImplementedError("Tracker needs to implement the `log_config` method.")

    @abstractmethod
    def log_metrics(self, metrics: Dict) -> None:
        """
        Take a dictionary with metrics and log content.

        Args:
            metrics: Dict in shape {metric: val, ...}
        """
        raise NotImplementedError("Tracker needs to implement the `log_metrics` method.")

    @abstractmethod
    def log_checkpoint(self, module_checkpoint: Dict, fold: int) -> None:
        """
        Save a module checkpoint.

        The checkpoint is a dictionary containing its state dict and
        optimizer parameters.

        Args:
            module_checkpoint: Dictionary with model weights.
            fold: Current fold. Should be extracted before beginning to upload to avoid concurrency.
        """
        raise NotImplementedError("Tracker needs to implement the `log_checkpoint` method.")

    @abstractmethod
    def log_artifacts(self, artifacts: Union[Artifact, List[Artifact]]) -> None:
        """
        Log dictionary with artifacts.

        Args:
            artifacts: Dictionary containing artifacts to log.
        """
        raise NotImplementedError("Tracker needs to implement the `log_artifacts` method.")

    @abstractmethod
    def load_config(self, run_uri: str) -> Dict:
        """
        Load config.

        Args:
            run_uri: Path pointing to project root.
        """
        raise NotImplementedError("Tracker needs to implement the `load_config` method.")

    def _local_file_save(self, artifact: FileArtifact) -> str:
        """
        Self data to local file system.

        Args:
            artifact: artifact containing the file.

        Returns:
            The local path the file is saved to.
        """
        file_name: str = artifact.file_name
        save_path = os.path.join(self.run_dir, file_name)

        if file_name.endswith('.pdf'):
            save_path = os.path.join(self.run_dir, file_name)
            self.plotter.save_plot(artifact.data, save_path)
        elif file_name.endswith('.yml'):
            with open(save_path, 'w', encoding='utf-8') as out_file:
                yaml.dump(artifact.data, out_file)
        elif file_name.endswith('.pkl'):
            save_pickle(artifact.data, self.run_dir, file_name)
        else:
            with open(save_path, 'w', encoding='utf-8') as out_file:
                out_file.write(str(artifact.data))

        return save_path
