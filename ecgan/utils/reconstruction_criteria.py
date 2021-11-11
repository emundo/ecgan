"""Utility functions required for the GAN based anomaly detection."""
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import CosineSimilarity

from ecgan.utils.distances import L1Distance, L2Distance, RGANMedianPairwiseDistance

logger = getLogger(__name__)


class SimilarityCriterions(ABC):
    """Optimization criterion based on the dissimilarity of samples."""

    @abstractmethod
    def calculate(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculate the (dis-)similarity between two Tensors x and y."""
        raise NotImplementedError("Need to implement abstract `calculate` method.")


class RBFSimilarityCriterion(SimilarityCriterions):
    """Optimization criterion based on the RBF similarity of samples."""

    def __init__(self, **kwargs):
        self.rbf_mode = kwargs.get('rbf_mode', 'gaussian')
        self.sigma = kwargs.get('sigma')

    def calculate(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculate the mean RBF dissimilarity between two tensors of arbitrary shape."""
        sim_matrix = rbf_kernel(
            x.unsqueeze(0),
            y.unsqueeze(0),
            rbf_mode=self.rbf_mode,
            sigma=self.sigma,
        )
        dissimilarity: Tensor = 1.0 - torch.mean(sim_matrix)
        return dissimilarity


class CosineSimilarityCriterion(SimilarityCriterions):
    """Optimization criterion based on the cosine similarity of samples."""

    def calculate(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculate the mean cosine dissimilarity between two tensors of arbitrary shape."""
        sim_matrix = CosineSimilarity(dim=0)(x.unsqueeze(0), y.unsqueeze(0))
        dissimilarity: Tensor = 1.0 - torch.mean(sim_matrix)
        return dissimilarity


class RganMmdCriterion(SimilarityCriterions):
    """Optimization criterion based on the MMD similarity of samples."""

    def __init__(self, sigma: Optional[float] = None):
        self.sigma = sigma

    def calculate(self, x: Tensor, y: Tensor):
        """
        Pytorch implementation of the RGAN MMD.

        The implementation is equivalent to the implementation of the
        :func:`ecgan.utils.reconstruction_criteria._mix_rbf_kernel function`
        from [RGAN repository, GitHub](https://github.com/ratschlab/RGAN/blob/master/mmd.py).
        The quadratic-time MMD with Gaussian RBF kernel is computed and - digressing from the original
        tensorflow implementation - only the K_XY kernel is returned.
        """
        if self.sigma is None:
            self.sigma = RGANMedianPairwiseDistance().calculate(x, x).detach().item()
        gamma = 1 / (2 * self.sigma ** 2) if self.sigma != 0.0 else 1 / (2 * 5 ** 2)

        if len(x.shape) == 2:
            xx = torch.matmul(x, x.transpose(0, 1))
            xy = torch.matmul(x, y.transpose(0, 1))
            yy = torch.matmul(y, y.transpose(0, 1))
        elif len(x.shape) == 3:
            xx = torch.tensordot(x, x, dims=[[1, 2], [1, 2]])
            xy = torch.tensordot(x, y, dims=[[1, 2], [1, 2]])
            yy = torch.tensordot(y, y, dims=[[1, 2], [1, 2]])
        else:
            raise ValueError('Function only accepts 2D or 3D matrices. Got {} dimensions'.format(len(x.shape)))
        x_squarenorms = torch.diag(xx).unsqueeze(0).expand(len(xy), -1)
        y_squarenorms = torch.diag(yy).unsqueeze(1).expand(-1, len(xy))
        k_xy = torch.exp(-gamma * (-2 * xy + x_squarenorms + y_squarenorms))
        return 1 - torch.diag(k_xy)


def rbf_kernel(x: Tensor, y: Tensor, rbf_mode: Optional[str] = None, sigma: Optional[float] = None) -> Tensor:
    """
    Calculate the Gaussian kernel function between two tensors.

    Gaussian kernel between samples of x and y.
    If sigma is not set, the kernel will infer the sigma value via median pairwise
    distance.

    Args:
        x: Tensor of shape (N x Features).
        y: Tensor of shape (N x Features).
        rbf_mode: Explicit choice of kernel: gaussian, exp or laplacian.
        sigma: Sigma for RBF bandwidth. Is usually set automatically by the object but can also be set manually.

    Returns:
        Kernel matrix K(X,Y) with shape [NX, NY]
    """
    if rbf_mode is None:
        rbf_mode = 'gaussian'

    dist = torch.cdist(x, y)

    # Infer sigma from median distance if no sigma is provided
    if sigma is None:
        sigma = float(torch.median(dist).detach())

    gamma = 1.0 / (2.0 * sigma ** 2)
    if rbf_mode == 'gaussian':
        return torch.exp(-gamma * dist ** 2)
    if rbf_mode == 'exp':
        return torch.exp(-gamma * dist)
    if rbf_mode == 'laplacian':
        gamma = 1.0 / sigma
        return torch.exp(-gamma * dist)

    raise ValueError(
        'RBF Mode {} is not known. Please use "gaussian", "exp" or ' '"laplacian" instead.'.format(rbf_mode)
    )


def get_reconstruction_criterion(criterion: str = 'residual') -> Callable[[Tensor, Tensor], Tensor]:
    """
    Select criterion function.

    Criteria are either distance based or similarity based. The target is usually to reduce either
    the distance or the dissimilarity (i.e. increase the similarity) between to samples.

    Returns:
        Callable reconstruction criterion.
    """
    func_dict = {
        'residual': L1Distance().calculate,
        'squared': L2Distance().calculate,
        'rbf': RBFSimilarityCriterion().calculate,
        'cosine': CosineSimilarityCriterion().calculate,
        'rgan': RganMmdCriterion().calculate,
    }

    if criterion not in func_dict.keys():
        raise ValueError('Criterion with name "{}" is unknown.'.format(criterion))

    return func_dict[criterion]
