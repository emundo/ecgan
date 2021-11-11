"""Tracker storing data using weights and biases."""
import os
import threading
import time
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import wandb

from ecgan.evaluation.tracker.base_tracker import BaseTracker
from ecgan.utils.artifacts import Artifact, FileArtifact, ImageArtifact, ValueArtifact
from ecgan.utils.miscellaneous import load_wandb_config

logger = getLogger(__name__)


class WBTracker(BaseTracker):
    """
    Class to manage calculating metrics and logging them with Weights & Biases (W&B).

    Args:
        entity: Weights and Biases entity name (given by W&B).
        project: Weights and Biases project name (chosen by user).
        run_name: Weights and Biases run name (chosen by user).
        save_pdf: Flag to save data as pdf **ADDITIONALLY** to normal plots.
        save_locally: Flag to retain local storage of data.
    """

    PDF_DIR = "pdfs"

    def __init__(
        self,
        entity: str,
        project: str,
        run_name: str,
        save_locally: bool = False,
        save_pdf: bool = False,
        save_checkpoint_s3: bool = False,
    ):
        """Set basic tracking parameters."""
        self.step = 1

        self.run = wandb.init(project=project, name=run_name, entity=entity, reinit=True)
        self.run_id = wandb.run.id  # type: ignore

        # Data will be saved locally to self.temp_dir during training and removed
        # if the `save_locally` flag is not explicitly set to True.
        self.temp_dir: Optional[TemporaryDirectory] = None
        self.save_locally = save_locally
        self.save_pdf = save_pdf

        self.thread_list: List[threading.Thread] = []

        super().__init__(entity, project, run_name, save_pdf)
        if save_checkpoint_s3:
            import boto3  # pylint: disable=C0415

            self.s3_bucket = boto3.resource('s3').Bucket(entity)
        self.save_checkpoint_s3 = save_checkpoint_s3

    def _init_run_dir(self) -> str:
        """
        Initialize the run directory.

        Returns:
            The file path.
        """
        if self.save_locally:
            run_dir = os.path.join('results', 'runs', self.run_id)
        else:
            self.temp_dir = TemporaryDirectory()  # pylint: disable=R1732
            run_dir = self.temp_dir.name

        if self.save_pdf:
            os.makedirs("{}/{}".format(run_dir, WBTracker.PDF_DIR))

        return run_dir

    def close(self):
        """
        Close training run.

        The uploading might require some time. The run is interrupted if the upload did not complete after 90 seconds.
        You can manually upload the data afterwards with a local sync.
        """
        timeout = 90
        start = time.time()
        still_alive = True
        remaining_time = timeout
        while still_alive:
            still_alive = False
            for thread in self.thread_list:
                thread.join(timeout=0.001)
                if thread.is_alive():
                    still_alive = True
            print(
                'Waiting for threaded upload to finish. Force quit in {0:3} second(s) if '
                'upload does not complete.'.format(remaining_time),
                end='\r',
            )

            if time.time() - start > timeout:
                break
            time.sleep(1)
            remaining_time -= 1

        self.run.finish()

        if not self.save_locally and self.temp_dir is not None:
            self.temp_dir.cleanup()

    def watch(self, model: Union[torch.nn.Module, List[torch.nn.Module]]) -> None:
        """Pass a module that should be watched during training."""
        if isinstance(model, list):
            for mod in model:
                wandb.watch(mod, log_freq=1)
        else:
            wandb.watch(model, log_freq=1)

    def log_config(self, cfg: Dict) -> None:
        """Log parameters for training setup."""
        wandb.config.update(cfg)  # pylint: disable=E1101

    def log_metrics(self, metrics: Dict) -> None:
        """
        Take a dictionary with metrics and log content.

        Args:
            metrics: Dict in shape {metric: val, ...}.
        """
        wandb.log(metrics, step=self.step)

    def log_checkpoint(self, module_checkpoint: Dict, fold: int) -> None:
        """
        Save a module checkpoint.

        The checkpoint is a dictionary containing its state dict and
        optimizer parameters.

        Args:
            module_checkpoint: Dictionary with model weights.
            fold: Current fold. Should be extracted before beginning to upload to avoid concurrency.
        """
        self._start_thread(
            target=self._upload_checkpoint,
            args=(
                module_checkpoint,
                fold,
            ),
        )

    def _start_thread(self, target: Any, args: Iterable) -> None:
        thread = threading.Thread(target=target, args=args)
        self.thread_list.append(thread)
        thread.start()

    @staticmethod
    def _upload_file(path: str) -> None:
        """Upload file to wandb."""
        wandb.save(path)

    def _upload_checkpoint(self, module_checkpoint: Dict, fold: int) -> None:
        """Upload checkpoint to S3 and log it to W&B."""
        path = os.path.join(self.run_dir, 'model_ep_{}_fold_{}.pt'.format(self.step, fold))

        torch.save(module_checkpoint, path)
        model_id = str(self.run_id) + '_fold' + str(fold)
        art = wandb.Artifact(model_id, type='model')

        if self.save_checkpoint_s3:
            s3_path = path.replace(self.run_dir, self.run_id)
            self.s3_bucket.upload_file(path, s3_path)
            s3_reference = 's3://' + self.entity + '/' + s3_path
            art.add_reference(s3_reference)
        else:
            art.add_file(path)
        wandb.log_artifact(art)

        if self.save_checkpoint_s3:
            logger.info('Uploaded checkout to s3.')
        else:
            logger.info('Uploaded checkout to wandb.')

    def log_artifacts(self, artifacts: Union[Artifact, List[Artifact]]) -> None:
        """
        Log dictionary with artifacts to W&B.

        Args:
            artifacts: Dictionary containing artifacts to log.
        """
        metrics_log: Dict = {}

        artifacts = artifacts if isinstance(artifacts, List) else [artifacts]

        for artifact in artifacts:
            if isinstance(artifact, ImageArtifact):
                metrics_log.update({artifact.name: wandb.Image(artifact.image)})

                if self.save_pdf and artifact.figure is not None:
                    file_name = artifact.name.replace(" ", "_")
                    pdf_artifact = FileArtifact(
                        artifact.name, artifact.figure, "{}_{}.pdf".format(file_name, self.step)
                    )
                    path = self._local_file_save(pdf_artifact)
                    self._start_thread(target=self._upload_file, args=("{}".format(path),))

            elif isinstance(artifact, ValueArtifact):
                metrics_log.update({artifact.name: artifact.value})
            elif isinstance(artifact, FileArtifact):
                path = self._local_file_save(artifact)
                self._start_thread(target=self._upload_file, args=(path,))

                logger.info('Saved {0} in {1}'.format(artifact.name, path))
            else:
                logger.warning("Artifact type was not found: {0}.".format(type(artifact)))

        if metrics_log:
            wandb.log(metrics_log, step=self.step)

    def load_config(self, run_uri: str) -> Dict:
        """Load config as dict."""
        return load_wandb_config(run_uri)
