"""Tracker factory to retrieve the instance of a tracker defined in the configuration file."""
from typing import Union

from ecgan.config import AdExperimentConfig, ExperimentConfig
from ecgan.evaluation.tracker.base_tracker import BaseTracker
from ecgan.evaluation.tracker.local_tracker import LocalTracker
from ecgan.evaluation.tracker.wb_tracker import WBTracker
from ecgan.utils.custom_types import TrackerType


class TrackerFactory:
    """Create desired tracker."""

    def __call__(self, config: Union[ExperimentConfig, AdExperimentConfig]) -> BaseTracker:
        """Return a BaseTracker instance."""
        tracker = config.TRACKER
        entity = tracker.ENTITY
        project = tracker.PROJECT
        run_name = tracker.EXPERIMENT_NAME
        save_pdf = tracker.SAVE_PDF

        if tracker.tracker_name == TrackerType.LOCAL:
            return LocalTracker(entity, project, run_name, save_pdf=save_pdf)

        if tracker.tracker_name == TrackerType.WEIGHTS_AND_BIASES:
            save_locally = tracker.LOCAL_SAVE
            upload_checkpoint_to_s3 = tracker.S3_CHECKPOINT_UPLOAD
            return WBTracker(
                entity,
                project,
                run_name,
                save_locally=save_locally,
                save_pdf=save_pdf,
                save_checkpoint_s3=upload_checkpoint_to_s3,
            )

        raise AttributeError('Argument {0} is not set correctly.'.format(tracker.tracker_name.value))
