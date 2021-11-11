"""RGAN architectures for the discriminator and generator."""
from typing import Dict

from torch import Tensor, nn
from torch.nn import Linear

from ecgan.config import BaseNNConfig, BaseRNNConfig
from ecgan.utils.configurable import ConfigurableTorchModule
from ecgan.utils.custom_types import WeightInitialization
from ecgan.utils.losses import WassersteinDiscriminatorLoss, WassersteinGeneratorLoss
from ecgan.utils.optimizer import Adam


class RGANGenerator(ConfigurableTorchModule):
    """Generator with the RGAN architecture."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        params: BaseNNConfig,
    ):
        super().__init__()

        if not isinstance(params.LAYER_SPECIFICATION, BaseRNNConfig):
            raise RuntimeError("Cannot instantiate RNN with config {0}.".format(type(params)))

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=params.LAYER_SPECIFICATION.HIDDEN_SIZE,
            num_layers=params.LAYER_SPECIFICATION.HIDDEN_DIMS,
            batch_first=True,
        )
        self.fully_connected = Linear(in_features=params.LAYER_SPECIFICATION.HIDDEN_SIZE, out_features=output_size)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the generator."""
        self.lstm.flatten_parameters()
        batch_size, seq_len, _ = x.shape

        x, (_, _) = self.lstm(x)
        x = x.reshape(-1, x.shape[2])
        x = self.fully_connected(x)
        x = self.tanh(x)
        x = x.reshape(batch_size, seq_len, -1)
        return x

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the generator of the RGAN module."""
        config = {
            'GENERATOR': {
                'LAYER_SPECIFICATION': {
                    'HIDDEN_DIMS': 1,
                    'HIDDEN_SIZE': 128,
                },
                'TANH_OUT': True,
                'WEIGHT_INIT': {'NAME': WeightInitialization.NORMAL.value, 'MEAN': 0.0, 'STD': 0.02},
            }
        }

        config['GENERATOR'].update(WassersteinGeneratorLoss.configure())
        config['GENERATOR'].update(Adam.configure())

        return config


class RGANDiscriminator(ConfigurableTorchModule):
    """Discriminator with the RGAN architecture with additional spectral normalization."""

    def __init__(self, input_size: int, params: BaseNNConfig):
        super().__init__()

        if not isinstance(params.LAYER_SPECIFICATION, BaseRNNConfig):
            raise RuntimeError("Cannot instantiate RNN with config {}.".format(type(params)))

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=params.LAYER_SPECIFICATION.HIDDEN_SIZE,
            num_layers=params.LAYER_SPECIFICATION.HIDDEN_DIMS,
            batch_first=True,
        )
        self.fully_connected = Linear(in_features=params.LAYER_SPECIFICATION.HIDDEN_SIZE, out_features=1)
        if params.SPECTRAL_NORM:
            self.fully_connected = nn.utils.spectral_norm(self.fully_connected)
        self.sig = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the discriminator."""
        self.lstm.flatten_parameters()

        x, (_, _) = self.lstm(x)
        x = x.reshape(-1, x.shape[2])
        x = self.fully_connected(x)
        x = self.sig(x)
        x = x.squeeze(-1)
        return x

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the discriminator of the RGAN module."""
        config = {
            'DISCRIMINATOR': {
                'SPECTRAL_NORM': False,
                'LAYER_SPECIFICATION': {
                    'HIDDEN_DIMS': 1,
                    'HIDDEN_SIZE': 128,
                },
                'LOSS': WassersteinDiscriminatorLoss.configure(),
                'OPTIMIZER': Adam.configure(),
                'WEIGHT_INIT': {'NAME': WeightInitialization.NORMAL.value, 'MEAN': 0.0, 'STD': 0.02},
            }
        }

        config['DISCRIMINATOR'].update(WassersteinDiscriminatorLoss.configure())
        config['DISCRIMINATOR'].update(Adam.configure())

        return config
