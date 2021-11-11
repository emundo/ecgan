"""Implementation of the maximum mean discrepancy to measure distributional differences."""
from typing import Optional

import torch
from torch import Tensor


class MaxMeanDiscrepancy:
    """Maximum Mean Discrepancy handler. Implementation is inspired by https://github.com/SeldonIO/alibi-detect."""

    def __init__(self, sigma: Optional[float] = None):
        r"""Initialize object for calculating the :math:`MMD^2` metric."""
        self.sigma = sigma

    def infer_sigma(
        self,
        x: Tensor,
        y: Tensor,
    ) -> float:
        """
        Infer heuristic sigma value for gaussian kernel.

        Infer sigma used in the kernel by setting it to the median distance between each of the pairwise instances
        in x and y.

        Args:
            x: Tensor of shape (num_samples x Features).
            y: Tensor of shape (num_samples x Features).

        Returns:
            Determined sigma value.
        """
        dist = torch.cdist(x, y)
        self.sigma = float(torch.median(dist))
        return self.sigma

    def gaussian_kernel(self, x: Tensor, y: Tensor, sigma: Optional[float] = None) -> Tensor:
        """
        Calculate the Gaussian kernel function between two tensors.

        If sigma is not set, the kernel will infer the sigma value via median pairwise distance.

        Args:
            x: Tensor of shape (num_samples x Features).
            y: Tensor of shape (num_samples x Features).
            sigma: Sigma for RBF bandwidth. Is usually set automatically by the object but can also be set manually.

        Returns:
            Kernel matrix K(X,Y) with shape [NX, NY]
        """
        if sigma is None:
            sigma = self.sigma if self.sigma is not None else self.infer_sigma(x, y)

        beta = 1.0 / (2.0 * (sigma ** 2))
        dist = torch.cdist(x, y)
        return torch.exp(-beta * dist)

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
    ) -> float:
        r"""
        Compute the maximum mean discrepancy between two tensors.

        Tensors can be of arbitrary shape as long they are both equal. First dimension will be interpreted as number of
        samples from distribution.
        Formula overview:
        :math:`MMD^2(P,Q) = \\mathbb{E}_P[k(X,X)] + \\mathbb{E}_Q[k(Y,Y)]
        - 2\\mathbb{E}_{P,Q}[k(X,Y)] = cxx * kxx + cyy * kyy + cxy * kxy`
        Elaborated information on the calculation and math of MMD can be found in
        https://www.gatsby.ucl.ac.uk/~gretton/papers/cardiff.pdf on slides 15-17.

        Args:
            x: Tensor of shape (num_samples x *)
            y: Tensor of shape (num_samples x *)

        Returns:
            MMD^2 between the tensors x and y.
        """
        if x.shape != y.shape:
            raise RuntimeError('Shape mismatch: {} and {}'.format(x.shape, y.shape))

        num_x = x.shape[0]
        num_y = y.shape[0]

        x = x.reshape(num_x, -1)
        y = y.reshape(num_y, -1)

        if self.sigma is None:
            self.infer_sigma(x, y)

        cxx = 1 / (num_x * (num_x - 1))
        cyy = 1 / (num_y * (num_y - 1))
        cxy = 2 / (num_x * num_y)

        kxx = self.gaussian_kernel(x, x)
        kyy = self.gaussian_kernel(y, y)
        kxy = self.gaussian_kernel(x, y)

        mmd = cxx * (kxx.sum() - kxx.trace()) + cyy * (kyy.sum() - kyy.trace()) - cxy * kxy.sum()
        return max(float(mmd), 0.0)
