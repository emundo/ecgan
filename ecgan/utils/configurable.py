"""Interfaces for configurable classes within ECGAN."""
from abc import ABC, abstractmethod
from typing import Dict

from torch import nn


class Configurable(ABC):
    """Interface of a class setup by the config."""

    @staticmethod
    @abstractmethod
    def configure() -> Dict:
        """Return the default configuration of a configurable class."""
        raise NotImplementedError("Configurable needs to implement the `configure` method.")


class ConfigurableTorchModule(nn.Module, Configurable):
    """Configurable variant of the torch nn.Module class."""

    @staticmethod
    @abstractmethod
    def configure() -> Dict:
        """Return the default configuration for a nn.Module."""
        raise NotImplementedError("ConfigurableTorchModule needs to implement the `configure` method.")
