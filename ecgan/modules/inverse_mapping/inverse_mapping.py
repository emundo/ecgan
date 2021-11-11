r"""
Defines a module that includes an inverse mapping.

In the typical ECGAN use-case the module would consist of two different mappings: :math:`G: A \rightarrow B` and
:math:`Inv: B \rightarrow A`, where :math:`G` is a typical generator that maps a given distribution (commonly
a normal distribution) :math:`A` to some set :math:`B`. The Inv function is the inverting mapping, essentially
tasked with restoring the distribution :math:`A` from a given sample of :math:`B`.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import nn
from tqdm import tqdm

from ecgan.config import InverseConfig, ModuleConfig, get_global_config, get_global_inv_config
from ecgan.evaluation.tracker import BaseTracker, TrackerFactory
from ecgan.modules.base import BaseModule
from ecgan.utils.configurable import Configurable
from ecgan.utils.miscellaneous import load_model


class InvertibleBaseModule(BaseModule, Configurable):
    """
    The abstract base class for inverse mappings.

    Every implementation of this class

    #. gets at least a reference to some trained generator module and
    #. must implement an inverse method (:code:`invert`) that restores the input data for the generator module.
    """

    def __init__(
        self,
        inv_cfg: InverseConfig.Attribs,
        module_cfg: ModuleConfig,
        seq_len: int,
        num_channels: int,
        tracker: Optional[BaseTracker] = None,
    ):
        super().__init__(module_cfg, seq_len, num_channels)
        self._inv_config = inv_cfg
        self._generator_module = self._init_generator_module()
        self.inv = self._init_inv()
        self.inv = nn.DataParallel(self.inv)
        self.inv.to(self.device)

        self.exp_cfg = get_global_config().experiment_config

        if tracker is None:
            self.tracker = TrackerFactory()(config=self.exp_cfg)
            self.close_tracker = True
        else:
            self.tracker = tracker
            self.close_tracker = False
        self.tracker.log_config(get_global_inv_config().config_dict)

    @abstractmethod
    def invert(self, data) -> torch.Tensor:
        """
        Apply the inverse mapping for the provided data.

        Note that the function does not make any assumptions about the gradient and must be wrapped into a torch.no_grad
        if the gradient is not needed.
        """
        raise NotImplementedError("InvertibleModule needs to implement the `invert` method.")

    def training_step(
        self,
        batch: dict,
    ) -> dict:
        """
        Perform a training step that can be called by a :class:`ecgan.training.trainer.Trainer` class instance.

        Note that the function does not require training data and the batch is only required to figure out the
        appropriate batch size.
        """
        return self._training_step(batch['data'].shape[0])

    @abstractmethod
    def _training_step(
        self,
        batch_size: int,
    ) -> dict:
        """
        Perform a training step.

        This private method is more appropriate to the definition of the inverse mapping. The batch_size is required to
        sample an according number of noise.
        """
        raise NotImplementedError("InvertibleModule needs to implement the `_training_step` method.")

    @property
    def inv_cfg(self) -> InverseConfig.Attribs:
        return self._inv_config

    @property
    def generator_module(self) -> BaseModule:
        return self._generator_module

    @generator_module.setter
    def generator_module(self, module: BaseModule) -> None:
        self._generator_module = module

    @property
    def inv(self) -> Any:
        return self._inv

    @inv.setter
    def inv(self, value: Any) -> None:
        self._inv = value

    @abstractmethod
    def _init_generator_module(self) -> BaseModule:
        raise NotImplementedError("InvertibleModule needs to implement the `_init_generator_module` method.")

    @abstractmethod
    def _init_inv(self) -> Any:
        raise NotImplementedError("InvertibleModule needs to implement the `_init_inv` method.")

    def load(self, model_reference: str, load_optim: bool = False) -> InvertibleBaseModule:
        """
        Load an inverse mapping model.

        The modules have to decide to save/load optimizers by themselves.

        Args:
            model_reference: Reference used to load an existing model.
            load_optim: Flag to indicate if the optimizer params should be loaded.
        """
        model = load_model(model_reference, self.device)
        self._load_inv(model['INV'], load_optim)
        self._load_generator_module(self._inv_config.RUN_URI)

        return self

    def save_checkpoint(self) -> dict:
        """Save a checkpoint of the inverse mapping."""
        return {'G': self.generator_module.save_checkpoint(), 'INV': self._save_inv()}

    @abstractmethod
    def _load_inv(self, inv_dict: Dict, load_optim: bool = False) -> None:
        """Load inversion module to memory."""
        raise NotImplementedError("InvertibleModule needs to implement the `_load_inv` method.")

    @abstractmethod
    def _load_generator_module(self, model_reference: Any) -> None:
        """Load generator module to memory."""
        raise NotImplementedError("InvertibleModule needs to implement the `_load_generator_module` method.")

    @abstractmethod
    def _save_inv(self) -> Dict:
        raise NotImplementedError("InvertibleModule needs to implement the `_save_inv` method.")

    def train(self) -> None:
        """Train a inverse mapping."""
        epochs = self.inv_cfg.EPOCHS
        rounds = self.inv_cfg.ROUNDS
        batch_size = self.inv_cfg.BATCH_SIZE
        save_checkpoint = self.inv_cfg.SAVE_CHECKPOINT
        artifact_checkpoint = self.inv_cfg.ARTIFACT_CHECKPOINT
        for epoch in tqdm(range(1, epochs + 1)):
            train_metrics = []

            # TRAINING LOOP
            for _ in range(rounds):
                metrics = self._training_step(batch_size)
                train_metrics.append(metrics)

            # AFTER EPOCH ACTION
            artifacts = self.on_epoch_end(epoch, artifact_checkpoint, batch_size)

            # CHECKPOINT
            if epoch % save_checkpoint == 0:
                self.tracker.log_checkpoint(self.save_checkpoint(), self.tracker.fold)

            # HANDLE EPOCH ARTIFACTS AND TRACKING
            collated_train_metrics = self.tracker.collate_metrics(train_metrics)
            self.tracker.log_metrics(collated_train_metrics)
            self.tracker.log_artifacts(artifacts)
            self.tracker.advance_step()
            self.print_metric(epoch, collated_train_metrics)

        if self.close_tracker:
            self.tracker.close()
