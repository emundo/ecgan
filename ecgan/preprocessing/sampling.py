"""Helpers to down- or upsample time series data."""
from logging import getLogger
from typing import Optional

import numpy as np
from pylttb import lttb
from torch import from_numpy, transpose
from torch.nn.functional import interpolate as torch_interpolate

from ecgan.utils.custom_types import SamplingAlgorithm

logger = getLogger(__name__)


def downsampling_fixed_sample_rate(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Downsample dataset by returning every `sample_rate`-th value in every dimension.

    .. warning:
        Information on the shape is easily lost using fixed rate downsampling.

    Args:
        data: Tensor to be downsampled. Shape: :code:`(seq_len, channels)`.
        sample_rate: The fixed sample rate used to retain every `sample_rate`-th value.

    Returns:
        The downsampled series.
    """
    return data[::sample_rate]  # type: ignore


def downsample_largest_triangle_three_buckets(data: np.ndarray, threshold: int) -> np.ndarray:
    """
    Downsample the data according to `LTTB <https://skemman.is/handle/1946/15343>`_.

    Args:
        data: The unsampled data.
        threshold: The LTTB threshold (target size).

    Returns:
        The downsampled data.
    """
    index = np.arange(data.shape[0])
    ecg_sampled = np.empty((threshold, data.shape[1]))

    for i in range(data.shape[1]):
        _, down_y = lttb(index, data[:, i], threshold)
        ecg_sampled[:, i] = down_y

    return ecg_sampled


def interpolate(
    data: np.ndarray,
    target_frequency: int,
    interpolation_strategy: Optional[str] = None,
) -> np.ndarray:
    """
    Force an incoming multivariate series to conform to a fixed frequency using PyTorch.

    Required if measuring devices use a different sampling frequency. More information on sampling rates:
    `See here <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6085204/>`_.
    Requires 3D input for most interpolation strategies.

    Args:
        data: Input series which shall be upsampled. Shape: (seq_len, num_channels).
        target_frequency: Desired output frequency.
        interpolation_strategy: Strategy according to https://pytorch.org/docs/stable/nn.functional.html

    Returns:
         (NumPy) Tensor with the upsampled values.
    """
    if interpolation_strategy is None:
        interpolation_strategy = 'linear'
        logger.info('No interpolation strategy defined. Defaulting to linear interpolation.')

    # Using the PyTorch upsampling function we require float tensors.
    data_tensor = from_numpy(data).float()
    # 3D tensor of shape (1, num_channels, seq_len)).
    data_tensor = transpose(data_tensor, 0, 1).unsqueeze(0)

    sampled = torch_interpolate(input=data_tensor, size=target_frequency, mode=interpolation_strategy)
    # Return shape to (seq_len, num_channels).
    sampled_np: np.ndarray = sampled.squeeze(0).transpose(0, 1).numpy()

    return sampled_np


def resample(
    data: np.ndarray,
    target_rate: int,
    algorithm: SamplingAlgorithm = SamplingAlgorithm.LTTB,
    interpolation_strategy: Optional[str] = None,
) -> np.ndarray:
    """
    Sample data according to the specified SamplingAlgorithm.

    Args:
        data: The data that shall be sampled.
        algorithm: The sampling algorithm.
        target_rate: Has to be set for a fixed sampling rate.
        interpolation_strategy: According to the PyTorch interpolation strategies.
            Only required for SamplingAlgorithm.INTERPOLATE.

    Returns:
        The resampled data.
    """
    if algorithm == SamplingAlgorithm.LTTB:
        return downsample_largest_triangle_three_buckets(data, threshold=target_rate)
    if algorithm == SamplingAlgorithm.FIXED_DOWNSAMPLING_RATE:
        return downsampling_fixed_sample_rate(data, sample_rate=target_rate)
    if algorithm == SamplingAlgorithm.INTERPOLATE:
        return interpolate(
            data,
            target_frequency=target_rate,
            interpolation_strategy=interpolation_strategy,
        )

    raise ValueError('{0} is currently not supported.'.format(algorithm))
