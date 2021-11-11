"""Simple RNN (LSTM) which can be used for classification."""
from typing import Dict

from torch import Tensor
from torch.nn import LSTM, Linear

from ecgan.utils.configurable import ConfigurableTorchModule
from ecgan.utils.custom_types import WeightInitialization


class RecurrentNeuralNetwork(ConfigurableTorchModule):
    """Generic Recurrent Neural Network classifier with LSTM blocks followed by a fully connected layer."""

    def __init__(self, num_channels: int, hidden_dim: int, hidden_size: int, n_classes: int):
        super().__init__()
        self.lstm = LSTM(input_size=num_channels, hidden_size=hidden_size, num_layers=hidden_dim, batch_first=True)
        self.fully_connected = Linear(in_features=hidden_size, out_features=n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the RNN."""
        self.lstm.flatten_parameters()
        x, (hidden_layers, _cell_state) = self.lstm(x)
        x = self.fully_connected(hidden_layers[-1])
        return x

    @staticmethod
    def _configure(hidden_dims: int, hidden_size: int) -> Dict:
        return {
            'module': {
                'LAYER_SPECIFICATION': {
                    'HIDDEN_DIMS': hidden_dims,
                    'HIDDEN_SIZE': hidden_size,
                },
                'WEIGHT_INIT': {'NAME': WeightInitialization.NORMAL.value, 'MEAN': 0.0, 'STD': 0.02},
            }
        }

    @staticmethod
    def configure() -> Dict:
        """Configure rnn."""
        return RecurrentNeuralNetwork._configure(
            hidden_dims=1,
            hidden_size=128,
        )
