"""Custom torch layers for neural architectures."""
from functools import partial
from typing import Optional, cast

import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal
from torch.nn import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from ecgan.config import NormalInitializationConfig, UniformInitializationConfig, WeightInitializationConfig
from ecgan.utils.custom_types import WeightInitialization


class MinibatchDiscrimination(nn.Module):
    """Minibatch discrimination layer based on https://gist.github.com/t-ae/732f78671643de97bbe2c46519972491."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_dims: int = 16,
        calc_mean: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.calc_mean = calc_mean

        self.t_mat = nn.Parameter(Normal(0, 1).sample((in_features, out_features, kernel_dims)))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Minibatch Discriminator."""
        # x is NxA
        # T is AxBxC
        out = x.mm(self.t_mat.view(self.in_features, -1))
        out = out.view(-1, self.out_features, self.kernel_dims)

        out = out.unsqueeze(0)  # 1xNxBxC
        out_perm = out.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(out - out_perm).sum(3)  # NxNxB
        exp_norm = torch.exp(-norm)
        o_b = exp_norm.sum(0) - 1  # NxB, subtract self distance

        if self.calc_mean:
            o_b /= x.shape[0] - 1

        x = torch.cat([x, o_b], dim=1)
        return x


class MinibatchDiscriminationSimple(nn.Module):
    """From `Karras et al. 2018 <https://arxiv.org/pdf/1710.10196.pdf>`_."""

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        """Forward pass of the Minibatch Discriminator."""
        out_stds = torch.std(x, dim=0)
        out = torch.mean(out_stds).unsqueeze(0)
        return out.expand(x.shape[0], 1).detach()


def initialize_weights(
    network: nn.Module,
    init_config: WeightInitializationConfig,
) -> None:
    """
    Initialize weights of a Torch architecture.

    Currently supported are:

        - 'normal': Sampling from a normal distribution. Parameters: mean, std
        - 'uniform': Sampling from a uniform distribution. Parameters: upper_bound,
           lower_bound
        - 'he': He initialization . He, K. et al. (2015)
        - 'glorot': Glorot, X. & Bengio, Y. (2010)

    Biases and BatchNorm are not initialized with this function as different strategies are applicable for these
    tensors/layers. Therefore the standard initialization of PyTorch when creating the layers is taken in these cases.
    """
    weight_init = partial(_initialize_weights, init_cfg=init_config)
    network.apply(weight_init)


def _initialize_weights(layer: nn.Module, init_cfg: WeightInitializationConfig) -> None:
    """
    Initialize the weights of a given layer.

    Args:
        layer: Layer to initialize weights in.
        init_cfg: Configuration for weight initialization.
    """
    if is_normalization_layer(layer):
        return

    if init_cfg.weight_init_type == WeightInitialization.NORMAL:
        normal_cfg = cast(NormalInitializationConfig, init_cfg)
        _init_normal(layer, mean=normal_cfg.MEAN, std=normal_cfg.STD)
    elif init_cfg.weight_init_type == WeightInitialization.UNIFORM:
        uniform_cfg = cast(UniformInitializationConfig, init_cfg)
        _init_uniform(layer, lower_bound=uniform_cfg.LOWER_BOUND, upper_bound=uniform_cfg.UPPER_BOUND)
    elif init_cfg.weight_init_type == WeightInitialization.HE.value:
        _init_he(layer)
    elif init_cfg.weight_init_type == WeightInitialization.GLOROT_UNIFORM:
        _init_glorot_uniform(layer)
    elif init_cfg.weight_init_type == WeightInitialization.GLOROT_NORMAL:
        _init_glorot_normal(layer)
    else:
        raise ValueError('Initialization "{}" is not known.'.format(init_cfg.NAME))


def initialize_batchnorm(module: nn.Module, **kwargs):
    """Explicitly initialize batchnorm layers with a normal distribution."""
    for layer in module.modules():
        if isinstance(layer, _BatchNorm):
            layer.weight.data.normal_(kwargs.get('mean', 1.0), kwargs.get('std', 0.02))
            layer.bias.data.fill_(kwargs.get('bias', 0))


def _init_normal(module: nn.Module, mean: Optional[float] = None, std: Optional[float] = None) -> None:
    """Initialize a nn.Module by sampling from a normal distribution."""
    mean = 0.0 if mean is None else mean
    std = 0.02 if std is None else std

    if hasattr(module, 'weight'):
        nn.init.normal_(module.weight, mean, std)  # type: ignore


def _init_uniform(
    module: nn.Module,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> None:
    """Initialize a nn.Module by sampling from a uniform distribution."""
    lower_bound = 0.0 if lower_bound is None else lower_bound
    upper_bound = 1.0 if upper_bound is None else upper_bound

    if hasattr(module, 'weight'):
        nn.init.uniform_(module.weight, lower_bound, upper_bound)  # type: ignore


def _init_he(module: nn.Module) -> None:
    """Initialize a nn.Module with He initialization."""
    if hasattr(module, 'weight'):
        nn.init.kaiming_normal_(module.weight)


def _init_glorot_uniform(module: nn.Module) -> None:
    """Initialize a nn.Module with Glorot initialization."""
    if hasattr(module, 'weight'):
        nn.init.xavier_uniform_(module.weight)  # type: ignore


def _init_glorot_normal(layer: nn.Module) -> None:
    """Initialize a nn.Module with Glorot initialization."""
    if hasattr(layer, 'weight'):
        nn.init.xavier_normal_(layer.weight)  # type: ignore


def is_normalization_layer(module: nn.Module):
    """Check if a module is a input normalization layer."""
    if isinstance(module, (_BatchNorm, GroupNorm)):
        return True
    return False
