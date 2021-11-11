"""Implementation of different distance metrics."""
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Union

import numpy as np
import torch
from torch import Tensor, mean
from torch.nn import L1Loss, MSELoss

from ecgan.utils.miscellaneous import to_torch

logger = getLogger(__name__)


class DistanceMetric(ABC):
    """A base class for different distance metrics to inherit from."""

    def __call__(
        self,
        point_1: Union[np.ndarray, Tensor],
        point_2: Union[np.ndarray, Tensor],
    ) -> Tensor:
        """Call the calculate method."""
        return self.calculate(point_1, point_2)

    @abstractmethod
    def calculate(
        self,
        point_1: Union[np.ndarray, Tensor],
        point_2: Union[np.ndarray, Tensor],
    ) -> Tensor:
        """
        Calculate the distance between two points (arrays of same size).

        Args:
            point_1: Some data with at least 1 dimension.
            point_2: Some data with at least 1 dimension.

        Returns:
            The distance.
        """
        raise NotImplementedError("Distance needs to implement the `calculate` method.")

    @staticmethod
    def _reduction(pairwise_distance: Tensor, reduction: str) -> Tensor:
        if reduction == 'mean':
            return torch.mean(pairwise_distance)

        if reduction == 'sum':
            return torch.sum(pairwise_distance)

        return pairwise_distance


class MinkowskiDistance(DistanceMetric):
    """
    Implementation of the Minkowski distance of two vectors.

    p=1: Manhattan Distance, p=2: Euclidean distance. Default is p=3.
    """

    def __init__(self, order: int = 3, reduction: str = 'none'):
        self.order = order
        self.reduction = reduction

    def calculate(
        self,
        point_1: Union[np.ndarray, Tensor],
        point_2: Union[np.ndarray, Tensor],
    ) -> Tensor:
        """
        Calculate the Minkowski distance.

        Args:
            point_1: Coordinates of one point.
            point_2: Coordinate of another point.

        Returns:
             The Minkowski distance of point_1 and point_2.
        """
        point_1 = torch.from_numpy(point_1) if isinstance(point_1, np.ndarray) else point_1
        point_2 = torch.from_numpy(point_2) if isinstance(point_2, np.ndarray) else point_2

        pairwise_distance = (abs(point_1 - point_2) ** self.order) ** (1 / self.order)

        return self._reduction(pairwise_distance, self.reduction)


class L1Distance(DistanceMetric):
    """Implementation of the :math:`L_1`-distance."""

    def __init__(self, reduction: str = 'none'):
        self.reduction = reduction

    def calculate(
        self,
        point_1: Union[np.ndarray, Tensor],
        point_2: Union[np.ndarray, Tensor],
    ) -> Tensor:
        r"""
        Return the average :math:`L_1` distance per sample in the batch.

        The pairwise :math:`L_1` distance of any shape - usually :math:`(b \times c \times s)` or :math:`(b
        \times c)` is calculated, reshaped to :math:`(b, -1)` and returned.
        """
        return mean(
            L1Loss(reduction=self.reduction)(point_1, point_2).view(point_1.shape[0], -1),
            dim=1,
        )


class L2Distance(DistanceMetric):
    """Implementation of the :math:`L_2`-distance."""

    def __init__(self, reduction: str = 'none'):
        self.reduction = reduction

    def calculate(
        self,
        point_1: Union[np.ndarray, Tensor],
        point_2: Union[np.ndarray, Tensor],
    ) -> Tensor:
        r"""
        Return the average :math:`L_2` distance per sample in the batch.

        The pairwise :math:`L_2` distance of any shape - usually :math:`(b \times c \times s)` or :math:`(b
        \times c)` is calculated, reshaped to :math:`(b, -1)` and returned.
        """
        point_1 = torch.from_numpy(point_1) if isinstance(point_1, np.ndarray) else point_1
        point_2 = torch.from_numpy(point_2) if isinstance(point_2, np.ndarray) else point_2

        return mean(
            MSELoss(reduction=self.reduction)(point_1, point_2).view(point_1.shape[0], -1),
            dim=1,
        )


class RGANMedianPairwiseDistance(DistanceMetric):
    """
    Based on the tensorflow implementation from https://github.com/ratschlab/RGAN/blob/master/mmd.py.

    Wsed as a heuristic for the RBF bandwidth.
    """

    def calculate(
        self,
        point_1: Union[np.ndarray, Tensor],
        point_2: Union[np.ndarray, Tensor],
    ) -> Tensor:
        """
        Calculate the RGAN median pairwise distance.

        If y cannot be provided: pass x as y argument.
        """
        x = to_torch(point_1)
        y = to_torch(point_2)

        if len(x.shape) == 2:
            x_squarenorms = torch.einsum('...i,...i', x, x)
            y_squarenorms = torch.einsum('...i,...i', y, y)
            xy = torch.einsum('ia,ja', x, y)
        elif len(x.shape) == 3:
            # tensor -- this is computing the Frobenius norm
            x_squarenorms = torch.einsum('...ij,...ij', x, x)
            y_squarenorms = torch.einsum('...ij,...ij', y, y)
            xy = torch.einsum('iab,jab', x, y)
        else:
            raise ValueError(x)

        distances = torch.sqrt(x_squarenorms.view(-1, 1) - 2 * xy + y_squarenorms.view(1, -1))
        distances[torch.isnan(distances)] = 0  # torch numerical instability
        return torch.quantile(distances, q=0.5)
