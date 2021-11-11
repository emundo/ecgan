"""Wrapper class for supported optimizer functions."""
from abc import abstractmethod
from logging import getLogger
from typing import Dict, List, Tuple, Union

import torch.optim
from adabelief_pytorch import AdaBelief as AdaBeliefOptimizer
from torch import Tensor

from ecgan.config import OptimizerConfig, get_
from ecgan.utils.configurable import Configurable
from ecgan.utils.custom_types import Optimizers

logger = getLogger(__name__)


class BaseOptimizer(Configurable):
    """Base optimizer class for custom optimizers."""

    def __init__(self, module_config, lr: float = 1e-4, weight_decay: float = 0.0) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.module_config = module_config
        self.param_group: List[Dict] = []

    def optimize(self, losses: Union[Tensor, List[Tuple[str, Tensor]]]):
        """Perform an optimization step given zero, one or several losses."""
        self.zero_grad()

        if isinstance(losses, List):
            for _, loss in losses:
                loss.backward()
        else:
            losses.backward()

        self.step()

    @property
    @abstractmethod
    def _optimizer(self) -> torch.optim.Optimizer:
        """Return the PyTorch optimizer used for the optimization."""
        raise NotImplementedError("Optimizer needs to implement the `_optimizer` method.")

    def state_dict(self) -> Dict:
        """Return the state dict of the PyTorch optimizer."""
        return self._optimizer.state_dict()  # type: ignore

    def zero_grad(self) -> None:
        """Zero the gradient of the optimizer."""
        self._optimizer.zero_grad()

    def step(self) -> None:
        """Perform an optimizer step."""
        self._optimizer.step()

    def set_param_group(self, updated_lr: float):
        """Set optimizer params for adaptive LR."""
        for group in self._optimizer.param_groups:
            group['lr'] = updated_lr

    def load_existing_optim(self, state_dict: Dict) -> None:
        """Load an already trained optim from an existing state_dict."""
        self._optimizer.load_state_dict(state_dict)

    @staticmethod
    def _configure(name: str, lr: float = 1e-4, weight_decay: float = 0.0) -> Dict:
        return {'OPTIMIZER': {'NAME': name, 'LR': lr, 'WEIGHT_DECAY': weight_decay}}

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for an optimizer."""
        return BaseOptimizer._configure(Optimizers.UNDEFINED.value)


class Adam(BaseOptimizer):
    """Adam optimizer wrapper around the PyTorch implementation."""

    def __init__(
        self,
        module_config,
        lr: float = 1e-4,
        weight_decay: float = 0,
        betas: Tuple[float, float] = None,
        eps: float = 1e-8,
    ):
        super().__init__(module_config, lr, weight_decay)
        if betas is None:
            betas = (0.5, 0.99)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self._optim = torch.optim.Adam(
            self.module_config,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
        )
        self.param_group = self._optim.param_groups

    @property
    def _optimizer(self):
        return self._optim

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the Adam optimizer."""
        config = BaseOptimizer._configure(Optimizers.ADAM.value)
        config['OPTIMIZER']['BETAS'] = [0.5, 0.99]
        return config


class StochasticGradientDescent(BaseOptimizer):
    """Stochastic gradient descent optimizer. For a Momentum variant see `Momentum`."""

    def __init__(self, module_config, lr: float = 1e-4, weight_decay: float = 0) -> None:
        super().__init__(module_config, lr, weight_decay)
        self._optim = torch.optim.SGD(module_config, lr=self.lr, momentum=0, weight_decay=self.weight_decay)
        self.param_group = self._optim.param_groups

    @property
    def _optimizer(self) -> torch.optim.Optimizer:
        return self._optim

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the Adam optimizer."""
        return BaseOptimizer._configure(Optimizers.STOCHASTIC_GRADIENT_DESCENT.value)


class Momentum(BaseOptimizer):
    """Momentum optimizer wrapper around the PyTorch implementation."""

    def __init__(
        self,
        module_config,
        lr: float = 1e-4,
        weight_decay: float = 0,
        momentum: float = 0.9,
        dampening: float = 0.0,
    ):
        super().__init__(module_config, lr, weight_decay)
        self.momentum = momentum
        self.dampening = dampening
        self._optim = torch.optim.SGD(
            module_config,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            dampening=self.dampening,
        )
        self.param_group = self._optim.param_groups

    @property
    def _optimizer(self) -> torch.optim.Optimizer:
        return self._optim

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the Momentum optimizer."""
        config = BaseOptimizer._configure(Optimizers.MOMENTUM.value)
        config['OPTIMIZER']['MOMENTUM'] = 0.9
        config['OPTIMIZER']['DAMPENING'] = 0.0
        return config


class RMSprop(BaseOptimizer):
    """Wrapper for the PyTorch RMSprop implementation."""

    def __init__(
        self,
        module_config,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        alpha: float = 0.99,
        eps: float = 1e-08,
        centered: bool = False,
    ):
        super().__init__(module_config, lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        self.alpha = alpha
        self.eps = eps
        self.centered = centered
        self._optim = torch.optim.RMSprop(
            module_config,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )
        self.param_group = self._optim.param_groups

    @property
    def _optimizer(self) -> torch.optim.Optimizer:
        return self._optim

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for a the RMSprop optimizer."""
        config = BaseOptimizer._configure(Optimizers.RMS_PROP.value)
        config['OPTIMIZER']['MOMENTUM'] = 0.9
        config['OPTIMIZER']['ALPHA'] = 0.99
        config['OPTIMIZER']['EPS'] = 1e-8
        config['OPTIMIZER']['CENTERED'] = False
        return config


class AdaBelief(BaseOptimizer):
    """
    Wrapper for the AdaBelief implementation.

    Not currently supported by PyTorch itself, taken from the official
    adabelief-pytorch repo until then.
    More information can be found at [Zhuang, GitHub Pages](https://juntang-zhuang.github.io/adabelief/).
    """

    def __init__(
        self,
        module_config,
        lr: float = 1e-3,
        betas: Tuple[float, float] = None,
        eps: float = 1e-16,
        weight_decay: float = 0,
    ):
        super().__init__(module_config, lr=lr, weight_decay=weight_decay)
        if betas is None:
            betas = (0.9, 0.999)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self._optim = AdaBeliefOptimizer(
            self.module_config,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
        )
        self.param_group = self._optim.param_groups

    @property
    def _optimizer(self) -> torch.optim.Optimizer:
        return self._optim  # type: ignore

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the Adabelief optimizer."""
        config = BaseOptimizer._configure(Optimizers.ADABELIEF.value)
        config['OPTIMIZER']['LR'] = 2e-4
        config['OPTIMIZER']['EPS'] = 1e-16
        config['OPTIMIZER']['BETAS'] = (0.9, 0.999)
        return config


class OptimizerFactory:
    """Meta module for creating an optimizer instance."""

    def __call__(self, module_config, optim_cfg: OptimizerConfig) -> BaseOptimizer:
        """Return an instance of an optimizer."""
        optimizer: Optimizers = Optimizers(optim_cfg.NAME)
        lr: float = float(get_(optim_cfg.LR, 1e-3))
        weight_decay: float = float(get_(optim_cfg.WEIGHT_DECAY, 0.0))

        if optimizer == Optimizers.STOCHASTIC_GRADIENT_DESCENT:
            return StochasticGradientDescent(module_config, lr=lr, weight_decay=weight_decay)

        if optimizer == Optimizers.MOMENTUM:
            momentum: float = get_(optim_cfg.MOMENTUM, 0.9)
            dampening: float = get_(optim_cfg.DAMPENING, 0)
            return Momentum(
                module_config,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                dampening=dampening,
            )

        if optimizer == Optimizers.ADAM:
            betas: Tuple[float, float] = get_(optim_cfg.BETAS, (0.5, 0.99))
            adam_eps: float = get_(optim_cfg.EPS, 1e-8)
            return Adam(
                module_config,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=adam_eps,
            )

        if optimizer == Optimizers.RMS_PROP:
            rms_momentum: float = get_(optim_cfg.MOMENTUM, 0.0)
            alpha: float = get_(optim_cfg.ALPHA, 0.99)
            rms_eps: float = get_(optim_cfg.EPS, 1e-8)
            centered = get_(optim_cfg.CENTERED, False)
            return RMSprop(
                module_config,
                lr=lr,
                weight_decay=weight_decay,
                momentum=rms_momentum,
                alpha=alpha,
                eps=rms_eps,
                centered=centered,
            )

        if optimizer == Optimizers.ADABELIEF:
            adabelief_betas: Tuple[float, float] = get_(optim_cfg.BETAS, (0.5, 0.99))
            eps: float = float(get_(optim_cfg.EPS, 1e-16))
            return AdaBelief(
                module_config,
                lr=lr,
                betas=adabelief_betas,
                eps=eps,
                weight_decay=weight_decay,
            )

        raise AttributeError('Argument {0} is not set correctly.'.format(optimizer))
