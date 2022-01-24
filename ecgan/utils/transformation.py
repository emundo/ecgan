"""Implementation of various normalizers for time series data."""
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Dict, Union

import torch
from numpy import ndarray
from torch import Tensor, from_numpy

from ecgan.utils.custom_types import Transformation
from ecgan.utils.miscellaneous import to_torch

logger = getLogger(__name__)


class DataTransformation(ABC):
    """A base class for transformations to inherit from."""

    def __init__(self):
        self.params = None

    def fit(self, data: Union[ndarray, Tensor]) -> None:
        """Fit a transformation on a numpy array of data points."""
        data_: Tensor = to_torch(data)

        if data_.dim() == 2:
            return self._fit_2d(data_)
        if data_.dim() == 3:
            return self._fit_3d(data_)
        raise ValueError(
            'Array with shape {0} and {1} dimensions is not valid.'
            'Please provide a 2D or 3D Array'.format(data_.shape, data_.dim())
        )

    @abstractmethod
    def _fit_2d(self, data: Tensor) -> None:
        raise NotImplementedError("The selected DataTransformation needs to implement the `_fit_2d` method.")

    @abstractmethod
    def _fit_3d(self, data: Tensor) -> None:
        raise NotImplementedError("The selected DataTransformation needs to implement the `_fit_3d` method.")

    def transform(self, data: Union[ndarray, Tensor]) -> Tensor:
        """
        Apply a transformation on a numpy array of data points.

        Requires an already fitted transformation.

        Returns:
             Transformed data.
        """
        data_: Tensor = data if isinstance(data, Tensor) else from_numpy(data)
        if data_.dim() == 2:
            return self._transform_2d(data_)
        if data_.dim() == 3:
            return self._transform_3d(data_)
        raise ValueError(
            'Array with shape {0} and {1} dimensions is not valid.'
            'Please provide a 2D or 3D Array'.format(data_.shape, data_.dim())
        )

    @abstractmethod
    def _transform_2d(self, data: Tensor) -> Tensor:
        raise NotImplementedError("The selected DataTransformation needs to implement the `_transform_2d` method.")

    @abstractmethod
    def _transform_3d(self, data: Tensor) -> Tensor:
        raise NotImplementedError("The selected DataTransformation needs to implement the `_transform_3d` method.")

    def fit_transform(self, data: Tensor) -> Tensor:
        """
        First apply the fit and then perform the transformation on given data.

        The 2D case as well as the 3D case are transformed along the columns.
        In 2D this is useful for a typical feature matrix but not often useful for
        time series data where one might want to transform along the rows or the
        whole dataset. If you want to transform time series data, one way
        would be to use 3D transformation with shape (samples x sequence_length x 1).
        """
        self.fit(data)
        return self.transform(data)


class MinMaxTransformation(DataTransformation):
    """Min-Max normalizer: scales the input to [0,1]."""

    def _fit_2d(self, data: Tensor) -> None:
        self.params = {'min': torch.min(data, dim=0), 'max': torch.max(data, dim=0)}

    def _transform_2d(self, data: Tensor) -> Tensor:
        if self.params['min'] is None or self.params['max'] is None:
            raise ValueError(
                'Either min (value: {0}) or max (value: {1}) are not set during transform. '
                'Please fit your normalizer before transforming the data.'.format(
                    self.params['min'], self.params['max']
                )
            )
        _, columns = data.shape
        normalized_data = torch.zeros(data.shape)
        for column in range(columns):
            normalized_data[:, column] = (data[:, column] - self.params['min'][0][column]) / (
                self.params['max'][0][column] - self.params['min'][0][column]
            )
        return normalized_data

    def _fit_3d(self, data: Tensor) -> None:
        mins = []
        maxs = []
        _, _, columns = data.shape
        for column in range(columns):
            mins.append(torch.min(data[:, :, column]))
            maxs.append(torch.max(data[:, :, column]))
        self.params = {'min': mins, 'max': maxs}

    def _transform_3d(self, data: Tensor) -> Tensor:
        if self.params['min'] is None or self.params['max'] is None:
            raise ValueError(
                'Either min (values: {0}) or max (values: {1}) are not set during transform. '
                'Please fit your normalizer before transforming the data.'.format(
                    self.params['min'], self.params['max']
                )
            )
        _, _, columns = data.shape
        normalized_data = torch.zeros(data.shape)
        for column in range(columns):
            normalized_data[:, :, column] = (data[:, :, column] - self.params['min'][column]) / (
                self.params['max'][column] - self.params['min'][column]
            )
        return normalized_data

    def get_params(self) -> Dict:
        """Retrieve normalization parameters."""
        return self.params  # type: ignore

    def set_params(self, params: Dict) -> None:
        """Set existing normalization parameters."""
        self.params = params


class StandardizationTransformation(DataTransformation):
    """Standardize the data such that it is distributed to N(0,1)."""

    def _fit_2d(self, data: Tensor) -> None:
        self.params = {
            'mean': torch.mean(data, dim=0),
            'std': torch.std(data, dim=0),
        }

    def _transform_2d(self, data: Tensor) -> Tensor:
        if self.params['mean'] is None or self.params['std'] is None:
            raise ValueError(
                'Either mean (value: {0}) or std (value: {1}) are not set during transform. '
                'Please fit your normalizer before transforming the data.'.format(
                    self.params['mean'], self.params['std']
                )
            )
        _, columns = data.shape
        normalized_data = torch.zeros(data.shape)
        for column in range(columns):
            normalized_data[:, column] = (data[:, column] - self.params['mean'][column]) / self.params['std'][column]
        return normalized_data

    def _fit_3d(self, data: Tensor) -> None:
        means = []
        stds = []
        _, _, columns = data.shape
        for column in range(columns):
            means.append(torch.mean(data[:, :, column]))
            stds.append(torch.std(data[:, :, column]))
        self.params = {'mean': means, 'std': stds}

    def _transform_3d(self, data: Tensor) -> Tensor:
        if self.params['mean'] is None or self.params['std'] is None:
            raise ValueError(
                'Either means (values: {0}) or stds (values: {1}) are not set during transform. '
                'Please fit your normalizer before transforming the data.'.format(
                    self.params['mean'], self.params['std']
                )
            )

        _, _, columns = data.shape
        normalized_data = torch.zeros(data.shape)
        for column in range(columns):
            normalized_data[:, :, column] = (data[:, :, column] - self.params['mean'][column]) / self.params['std'][
                column
            ]
        return normalized_data


class WhiteningTransformation(DataTransformation):
    """
    Apply a Whitening transformation on data.

    The Whitening transformation returns decorrelated data i.e. data with unit covariance matrix.
    """

    def __init__(self, fudge=1e-16):
        super().__init__()
        self.fudge = fudge

    def _compute_whitening_matrix(self, column) -> Tensor:
        eigenvalues, eigenvectors = (
            self.params['eigenvalues'][column],
            self.params['eigenvectors'][column],
        )

        diag = torch.diag(1 / torch.sqrt(torch.abs(eigenvalues) + self.fudge))
        whitening = torch.mm(torch.mm(eigenvectors, diag), eigenvectors.t())
        return whitening

    def _fit_2d(self, data: Tensor) -> None:
        _, columns = data.shape
        mean = torch.mean(data, dim=0)
        eigenvalues, eigenvectors = [], []
        for column in range(columns):
            mean_ = mean[column]
            data_column = data[:, column]
            shifted_data = data_column - mean_
            cov = torch.dot(shifted_data.t(), shifted_data)
            eigenvalue, eigenvector = torch.symeig(cov.view(1, 1), eigenvectors=True)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)

        self.params = {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}

    def _fit_3d(self, data: Tensor) -> None:
        _, _, columns = data.shape
        eigenvalues, eigenvectors = [], []
        for column in range(columns):
            data_column = data[:, :, column].t()
            mean = torch.mean(data_column, dim=0)
            shifted_data = data_column - mean
            cov = torch.mm(shifted_data.t(), shifted_data)
            eigenvalue, eigenvector = torch.symeig(cov, eigenvectors=True)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)

        self.params = {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}

    def _transform_2d(self, data: Tensor) -> Tensor:
        if self.params['eigenvalues'] is None or self.params['eigenvectors'] is None:
            raise ValueError(
                'Either eigenvalues (value: {0}) or eigenvectors (value: {1}) are not set during transform. '
                'Please fit your normalizer before transforming the data.'.format(
                    self.params['eigenvalues'], self.params['eigenvectors']
                )
            )
        rows, columns = data.shape
        normalized_data = torch.zeros(data.shape)
        for column in range(columns):
            whitening = self._compute_whitening_matrix(column)
            data_column = data[:, column].view(1, rows)
            normalized_data[:, column] = torch.mm(data_column.t(), whitening)[:, 0]
        return normalized_data

    def _transform_3d(self, data: Tensor) -> Tensor:
        if self.params['eigenvalues'] is None or self.params['eigenvectors'] is None:
            raise ValueError(
                'Either eigenvalues (values: {0}) or eigenvectors (values: {1}) are not set during transform. '
                'Please fit your normalizer before transforming the data.'.format(
                    self.params['eigenvalues'], self.params['eigenvectors']
                )
            )
        _, _, columns = data.shape
        normalized_data = torch.zeros(data.shape)
        for column in range(columns):
            whitening = self._compute_whitening_matrix(column)
            data_column = data[:, :, column].t()
            normalized_data[:, :, column] = torch.mm(data_column, whitening).t()
        return normalized_data


class FFTTransformation(DataTransformation):
    """Compute the 2D or 3D discrete Fourier transform using the PyTorch FFT implementation."""

    def _fit_2d(self, data: Tensor) -> None:
        return

    def _fit_3d(self, data: Tensor) -> None:
        return

    def _transform_2d(self, data: Tensor) -> Tensor:
        """
        Return the 2D data transformed via FFT.

        The FFT maps each value in the data tensor to a complex number in the frequency domain. The method separates the
        two real and imaginary components into separate channels, alternating real and imaginary components of the value
        in frequency domain. Thus a tensor of size (batch_size, seq_length) is transformed to a tensor
        (batch_size, seq_length, 2), with floating point entries.
        """
        fourier_coeffs = torch.fft.fft(data, norm="ortho")
        zipped = list(zip(fourier_coeffs.real.tolist(), fourier_coeffs.imag.tolist()))
        zipped_tensor: Tensor = torch.tensor(zipped)
        return zipped_tensor.permute(0, 2, 1)

    def _transform_3d(self, data: Tensor) -> Tensor:
        """
        Return the 3D data transformed via FFT.

        The FFT maps each value in the data tensor to a complex number in the frequency domain. The method separates the
        two real and imaginary components into separate channels, alternating real and imaginary components of the value
        in frequency domain. Thus a tensor of size (batch_size, seq_length, num_channels) is transformed to a tensor
        (batch_size, seq_length, 2 * num_channels), with floating point entries.
        """
        batch_size, seq_length, num_channels = data.shape
        fourier_coeffs = torch.fft.fftn(data, norm="ortho")
        zipped = list(
            zip(
                fourier_coeffs.real.flatten().tolist(),
                fourier_coeffs.imag.flatten().tolist(),
            )
        )
        zipped_tensor: Tensor = torch.tensor(zipped)

        return zipped_tensor.reshape(batch_size, seq_length, 2 * num_channels)


class SamplewiseMinmaxTransformation(DataTransformation):
    """
    Scales each sample to the [0, 1] range.

    MinMaxTransformation scales in the same way but per channel, not per sample.
    """

    def _fit_2d(self, data: Tensor) -> None:
        pass

    def _transform_2d(self, data: Tensor) -> Tensor:
        raise NotImplementedError("2D samplewise minmax scaling not yet supported.")

    def _fit_3d(self, data: Tensor) -> None:
        pass

    @staticmethod
    def transform_1d(sample):
        """Scale individual series to 0 and 1."""
        normalized_data = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))

        return normalized_data

    def _transform_3d(
        self,
        data: Tensor,
    ) -> Tensor:
        data = data.permute(0, 2, 1)
        normalized_data = torch.zeros(data.shape)
        batch, columns, _ = normalized_data.shape
        for sample in range(batch):
            for col in range(columns):
                normalized_data[sample][col] = self.transform_1d(data[sample][col][:])
        normalized_data = normalized_data.permute(0, 2, 1)

        return normalized_data


class NoTransformation(DataTransformation):
    """Apply no transformation."""

    def _fit_2d(self, data: Tensor) -> None:
        pass

    def _transform_2d(self, data: Tensor) -> Tensor:
        return data

    def _fit_3d(self, data: Tensor) -> None:
        pass

    def _transform_3d(
        self,
        data: Tensor,
    ) -> Tensor:
        return data


def get_transformation(
    transformation: Transformation,
) -> DataTransformation:
    """Transform the data. The output range depends on the normalizer chosen."""
    transformations = {
        Transformation.MINMAX: MinMaxTransformation(),
        Transformation.WHITENING: WhiteningTransformation(),
        Transformation.STANDARDIZE: StandardizationTransformation(),
        Transformation.FOURIER: FFTTransformation(),
        Transformation.INDIVIDUAL: SamplewiseMinmaxTransformation(),
        Transformation.NONE: NoTransformation(),
    }
    try:
        return transformations[transformation]
    except KeyError:
        logger.warning('No known transformation with name {0}. Defaulting to no transformation.'.format(transformation))
        return NoTransformation()
