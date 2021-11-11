"""Implementation of the data cleansing process."""
from logging import getLogger
from typing import Optional, Tuple

import numpy as np

logger = getLogger(__name__)


class DataCleanser:
    r"""
    Check for dead or faulty sensors, NaNs and correct shape.

    A :code:`DataCleanser` object can be used for some or all of the above tasks. Most often the
    :meth:`ecgan.preprocessing.cleansing.DataCleanser.should_cleanse` method is called which checks if the series
    fulfills all of the checks. Each check can also be called individually. The input series is generally expected to
    be a single 2D series of shape `(seq_len, features)` with `features` being the different data channels.
    By default, all values are accepted if no threshold/condition is set.

    Args:
        lower_fault_threshold: Lowest value accepted without removing the series from dataset.
        upper_fault_threshold: Highest value accepted without removing the series from dataset.
        nan_threshold: Upper limit of allowed percentage of NaNs. Remove series if
            more than :math:`(self.nan\_threshold \cdot 100)\%` of all values are NaN.
        target_shape: Accepted shape of series.
    """

    def __init__(
        self,
        lower_fault_threshold: Optional[int] = None,
        upper_fault_threshold: Optional[int] = None,
        nan_threshold: Optional[float] = None,
        target_shape: Optional[Tuple[int, int]] = None,
    ):
        self.cleansed_total = 0
        self.target_shape = target_shape
        self.upper_fault_threshold = upper_fault_threshold
        self.lower_fault_threshold = lower_fault_threshold
        self.nan_threshold = nan_threshold

    def should_cleanse(self, series: np.ndarray) -> bool:
        """
        Conduct checks for a given 2D time series to determine if it should be cleansed.

        Remove sample from dataset if any check fails.

        Performed checks:

        * :meth:`ecgan.preprocessing.cleansing.DataCleanser.check_shape`
        * :meth:`ecgan.preprocessing.cleansing.DataCleanser.check_for_nan`
        * :meth:`ecgan.preprocessing.cleansing.DataCleanser.check_for_dead_sensor`
        * :meth:`ecgan.preprocessing.cleansing.DataCleanser.check_for_faulty_sensor`

        Args:
            series: 2D series of shape `seq_len, features`.

        Returns:
            Flag indicating whether the sample should be removed from the final dataset.
        """
        cleanse_sample = (
            self.check_shape(series)
            or self.check_for_nan(series)
            or self.check_for_dead_sensor(series)
            or self.check_for_faulty_sensor(series)
        )

        if cleanse_sample:
            self.cleansed_total += 1

        return cleanse_sample

    def check_shape(self, series: np.ndarray) -> bool:
        """
        Check if the sample should be removed because its shape.

        If no target_shape is specified in the instance creation, the shape is assumed to be a simple 2D (seq_len,
        features).

        Args:
            series: 2D series of shape `seq_len, features`.

        Returns:
            Flag indicating whether the sample should be removed from the final dataset.
        """
        if self.target_shape is not None and series.shape != self.target_shape:
            return True

        if len(series.shape) != 2:
            return True

        return False

    def check_for_nan(self, series: np.ndarray) -> bool:
        r"""
        Check for NaN values in the data.

        Data is marked for cleansing when at least :math:`(self.nan\_threshold \cdot 100)\%` of values of one feature
        are NaN. The data is expected to be a single time series sample of shape (seq_len, features), i.e. a 2D array.
        Series with more than 0 but less NaNs than allowed can impute the remaining NaNs using the
        :class:`ecgan.preprocessing.preprocessor.BasePreprocessor`.

        Args:
            series: 2D series of shape `seq_len, features`.

        Returns:
            Flag indicating whether the sample should be removed from the final dataset.
        """
        if self.nan_threshold is not None:
            for feature in range(series.shape[1]):
                nan_count = np.count_nonzero(np.isnan(series[:, feature]))
                if nan_count > self.nan_threshold * series.shape[0]:
                    return True

        return False

    @staticmethod
    def check_for_dead_sensor(series: np.ndarray) -> bool:
        """
        Check for dead sensors in the data.

        Data is marked as dead and subsequently as 'to be cleansed' if the variance (and thus the standard deviation)
        of a sensor is close to zero.

        Args:
            series: 2D series of shape `seq_len, features`.

        Returns:
            Flag indicating whether the sample should be removed from the final dataset.
        """
        for feature in range(series.shape[1]):
            std = np.std(series[:, feature])
            if np.allclose(std, 0):
                return True

        return False

    def check_for_faulty_sensor(self, series: np.ndarray) -> bool:
        """
        Check for faulty sensors in the data.

        Data is marked for cleansing if certain values in the data exceed a threshold or if all values are NaN.

        Args:
            series: 2D series of shape `seq_len, features`.

        Returns:
            Flag indicating whether the sample should be removed from the final dataset.
        """
        if self.upper_fault_threshold is None and self.lower_fault_threshold is None:
            logger.debug('Threshold for faulty sensors not specified. Skipping fault checks.')
            return False

        lower_threshold = 0.0
        upper_threshold = 0.0

        if self.lower_fault_threshold != 0:
            lower_threshold = self.lower_fault_threshold or float('-inf')

        if self.upper_fault_threshold != 0:
            upper_threshold = self.upper_fault_threshold or float('inf')

        min_ = np.nanmin(series)
        max_ = np.nanmax(series)
        if min_ < lower_threshold or max_ > upper_threshold:
            return True
        return False
