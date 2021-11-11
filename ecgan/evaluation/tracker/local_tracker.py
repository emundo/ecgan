"""Tracker storing data locally."""
import os
import uuid
from csv import DictReader, DictWriter
from logging import getLogger
from typing import Dict, List, Union

import numpy as np
import torch
import yaml

from ecgan.evaluation.tracker.base_tracker import BaseTracker
from ecgan.utils.artifacts import Artifact, FileArtifact, ImageArtifact, ValueArtifact
from ecgan.utils.miscellaneous import load_yml

logger = getLogger(__name__)


class LocalTracker(BaseTracker):
    """Class to manage calculating metrics and logging them on the local file system."""

    def __init__(
        self,
        entity: str,
        project: str,
        run_name: str,
        save_pdf: bool = True,
        base_root: str = 'results',
    ):
        self.base_root = base_root
        super().__init__(entity, project, run_name, save_pdf)
        logger.info('Local tracking enabled. Your run is saved in "{}".'.format(self.run_dir))

        self.metrics_buffer: Dict = {}

    def _init_run_dir(self) -> str:
        """
        Return a randomly generated directory name.

        Returns:
            Path consisting of './<base_root>/<entity>/<project>/<run_name>-<random_hex>'.
        """
        while True:
            run_dir = os.path.join(
                self.base_root,
                self.entity,
                self.project,
                self.run_name + '-' + uuid.uuid4().hex,
            )
            if not os.path.exists(run_dir):
                break

        return run_dir

    def close(self):
        """Close training run."""
        self.metrics_buffer = {}

    def watch(self, model: Union[torch.nn.Module, List[torch.nn.Module]]) -> None:
        """Watch models during training - not supported in LocalTracker."""
        pass

    def log_config(self, cfg: Dict) -> None:
        """Log parameters for training setup."""
        # The local config name will be replaced by 'config.yml' in any case.
        # This allows to easily retrieve configs from a given run.
        path = os.path.join(self.run_dir, 'config.yml')
        with open(path, 'w', encoding='utf-8') as out_file:
            yaml.dump(cfg, out_file)

    def log_metrics(self, metrics: Dict) -> None:
        """
        Take a dictionary with metrics and log content.

        Args:
            metrics: Dict in shape {metric: val, ...}.
        """
        metrics_path = "{}/metrics.csv".format(self.run_dir)
        if not os.path.exists(metrics_path):
            open(metrics_path, "x", encoding='utf-8')  # pylint: disable=R1732

        with open(metrics_path, 'r', encoding='utf-8') as f:
            csv_dict_reader = DictReader(f)
            try:
                header = list(next(csv_dict_reader))
            except StopIteration:
                header = []

        header_list = list(set(header + list(metrics.keys())))

        with open(metrics_path, "w", encoding='utf-8') as f:
            csv_writer = DictWriter(f, fieldnames=header_list)
            csv_writer.writeheader()

        with open(metrics_path, "a", encoding='utf-8') as f:
            csv_writer = DictWriter(f, fieldnames=header_list)
            csv_writer.writerow(metrics)

        for key, value in metrics.items():
            if self.metrics_buffer.get(key) is None:
                self.metrics_buffer[key] = [value]

            else:
                self.metrics_buffer[key].append(value)

            fig = self.plotter.create_plot(np.array(self.metrics_buffer[key]), label=key)

            file_type = 'pdf' if self.save_pdf else 'png'
            if isinstance(key, str) and key.__contains__('/'):
                key = key.replace('/', '_')
            self.plotter.save_plot(fig, '{}/{}.{}'.format(self.run_dir, key, file_type))

    def log_checkpoint(self, module_checkpoint: Dict, fold: int) -> None:
        """
        Save a module checkpoint.

        The checkpoint is a dictionary containing its state dict and optimizer parameters.

        Args:
            module_checkpoint: Dictionary with model weights.
            fold: Current fold. Should be extracted before beginning to upload to avoid concurrency.
        """
        # Checkpoint for model
        model_dir = os.path.join(self.run_dir, 'MODELS')
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            module_checkpoint,
            os.path.join(model_dir, 'model_ep_{}_fold{}.pt'.format(self.step, fold)),
        )

    def log_artifacts(self, artifacts: Union[Artifact, List[Artifact]]) -> None:
        """
        Log dictionary with artifacts.

        Args:
            artifacts: Dictionary containing artifacts to log.
        """
        artifacts = artifacts if isinstance(artifacts, List) else [artifacts]

        for artifact in artifacts:
            if isinstance(artifact, ImageArtifact):
                img_dir = os.path.join(self.run_dir, artifact.name.replace(' ', '_').upper())
                os.makedirs(img_dir, exist_ok=True)
                file_type = 'pdf' if self.save_pdf else 'png'
                self.plotter.save_plot(
                    file_location=os.path.join(img_dir, str(self.step) + '.' + file_type),
                    plot=artifact.image,
                )

            elif isinstance(artifact, ValueArtifact):
                if isinstance(artifact.value, float):
                    self.log_metrics({artifact.name: artifact.value})
                if isinstance(artifact.value, Dict):
                    value_dict = self._unfold_inner_dict(artifact_dict=artifact.value, artifact_name=artifact.name)
                    self.log_metrics(value_dict)

            elif isinstance(artifact, FileArtifact):
                path = self._local_file_save(artifact)
                logger.info('Saved {0} in {1}'.format(artifact.name, path))
            else:
                logger.warning("Artifact type was not found: {0}.".format(type(artifact)))

    def load_config(self, run_uri: str) -> Dict:
        """
        Load config.

        Args:
            run_uri: Path pointing to project root.

        Returns:
            Loaded yml config as dict.
        """
        if os.path.isabs(run_uri):
            run_dir = run_uri
        else:
            run_dir = os.path.join(self.base_root, run_uri)

        if not os.path.isdir(run_dir):
            raise ValueError('{} does not point to a valid directory.'.format(run_dir))

        path = os.path.join(run_dir, 'config.yml')
        if not os.path.exists(path):
            raise ValueError('{} does not contain a valid config.yml.'.format(run_dir))

        return load_yml(path)

    @staticmethod
    def _unfold_inner_dict(artifact_dict: Dict, artifact_name: str) -> Dict:
        value_dict = {}
        for outer_key, val_dict in artifact_dict.items():
            logger.info("unfolding {} {}".format(outer_key, val_dict))
            if not isinstance(val_dict, dict):
                value_dict.update({'{}_'.format(artifact_name) + outer_key: val_dict})
            else:
                for inner_key, val in val_dict.items():
                    value_dict.update({'{}_'.format(artifact_name) + outer_key + '_' + inner_key: val})

        return value_dict
