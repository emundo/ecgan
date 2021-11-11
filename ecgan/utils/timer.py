"""Utility functions for time tracking."""
import datetime
from logging import getLogger
from typing import Optional

from ecgan.evaluation.tracker import BaseTracker

logger = getLogger(__name__)


class Timer:
    """Utility class for tracking wall time."""

    def __init__(self, name: str, tracker: Optional[BaseTracker], metric_name: Optional[str]):
        self.name = name
        self.tracker = tracker
        self.metric_name = metric_name
        self.start = datetime.datetime.now()
        self.stop = datetime.datetime.now()
        self.delta = self.start - self.stop

    def __enter__(self):
        """Print start information on entering."""
        logger.info('~~~~~~~> Started {0} <~~~~~~~'.format(self.name))
        self.start = datetime.datetime.now()

    def __exit__(self, *args) -> None:
        """Print and log wall time on exiting."""
        self.stop = datetime.datetime.now()
        self.delta = self.stop - self.start
        seconds = self.delta.seconds
        minutes, seconds_of_minute = divmod(seconds, 60)
        hours, minutes_of_hour = divmod(minutes, 60)
        logger.info(
            '{0} took {1:02}:{2:02}:{3:02}.'.format(self.name, int(hours), int(minutes_of_hour), int(seconds_of_minute))
        )
        if self.tracker is not None and self.metric_name is not None:
            self.tracker.log_metrics({self.metric_name: self.delta.total_seconds()})
