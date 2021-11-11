"""Interpolation schemes between two data points in latent space."""
from typing import List, Union

import torch


def slerp(mu, low, high):
    """
    Spherical linear interpolation based on `White et al. 2016 <https://arxiv.org/pdf/1609.04468.pdf>`_.

    Originally introduced in `Shoemake, 1985 <https://www.engr.colostate.edu/ECE481A2/Readings/Rotation_Animation.pdf>`_
    and additional visualizations can be found in
    `Huszár 2017 (Blogpost) <https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/>`_.
    Implementation adapted from
    `ptrblack, GitHub <https://github.com/ptrblck/prog_gans_pytorch_inference/blob/master/utils.py>`_
    and `soumith, GitHub <https://github.com/soumith/dcgan.torch/issues/14>`_.
    Most probability mass in high-dimensional Gaussian latent spaces is in an annulus around the origin
    and points around the origin are scarce. To account for this, a meaningful interpolation should
    traverse the annulus and not travel through the center of the hypersphere.

    Args:
        mu: Parameter moving from 0 to 1 the closer it gets to `high`.
        low: Sample from latent space, origin of the interpolation process.
        high: Sample from latent space, target of the interpolation process.

    Returns:
        Mu-based interpolated sample between low and high.
    """
    low_norm = torch.linalg.norm(low)
    high_norm = torch.linalg.norm(high)
    # avoid numerical instability norm is only zero if low is zero
    # denominator can then be set arbitrarily to avoid division by zero
    low_norm = torch.where(low_norm != 0, low_norm, torch.ones(low_norm.shape))
    high_norm = torch.where(high_norm != 0, high_norm, torch.ones(high_norm.shape))
    low_norm_scale = torch.div(low, low_norm)
    high_norm_scale = torch.div(high, high_norm)

    omega = torch.acos(torch.matmul(low_norm_scale, high_norm_scale.t()))
    sin_omega = torch.sin(omega)
    # L'Hopital's rule/LERP from https://github.com/soumith/dcgan.torch/issues/14
    if sin_omega == 0:
        return (1.0 - mu) * low + mu * high
    res = (torch.sin((1.0 - mu) * omega) / sin_omega) * low + (torch.sin(mu * omega) / sin_omega) * high

    return res


def spherical_interpolation(start, target, num_steps) -> torch.Tensor:
    """
    Perform the interpolation between two points in a specified amount of steps.

    Args:
        start: One point (in latent space), of dimensionality (n,).
        target: Point which is approached during interpolation.
        num_steps: Amount of steps/samples taken during interpolation.

    Returns:
        Tensor: (num_steps x samples.shape).
    """
    interpolation_steps = torch.linspace(start=1 / num_steps, end=1, steps=num_steps)
    interpolated_samples = [slerp(val, start, target).numpy() for val in interpolation_steps]

    return torch.as_tensor(interpolated_samples)


def latent_walk(
    base_sample: torch.Tensor,
    component: torch.nn.Module,
    walk_range: torch.Tensor,
    device: torch.device,
    latent_dims: Union[int, List],
) -> torch.Tensor:
    """
    Explore the latent space based on a single latent space sample.

    Up to 10 dims are visualized.

    Args:
        base_sample: Initial latent sample of dim [1,1,latent_dim].
        component: The generative module that is used to create new samples.
        walk_range: The area of the latent sample investigated.
        device: The device of the NN module.
        latent_dims: Amount of dims walked through. If more dims exist than selected: Use the first n latent_dims.
            Only up to 10 latent_dims are allowed for this visualization.

    Returns:
        A tensor of reconstructed samples where each latent dim is altered in the direction of walk_range.
    """
    latent_dim_range = latent_dims if isinstance(latent_dims, List) else range(0, min(latent_dims, 10))
    with torch.no_grad():
        samples = torch.empty(0).to(device)
        for dim in latent_dim_range:
            for offset in walk_range:
                base_clone = base_sample.clone()
                base_clone[0][0][dim] = base_clone[0][0][dim] + offset
                gen_samples = component(base_clone)
                samples = torch.cat((samples, gen_samples[:, :, :1]))
        return samples
