"""Generic CNNs."""
from typing import Dict, List

from torch import Tensor, nn
from torch.nn import Linear

from ecgan.networks.helpers import conv_norm_relu, create_5_hidden_layer_convnet
from ecgan.utils.configurable import ConfigurableTorchModule
from ecgan.utils.custom_types import InputNormalization, InverseMappingType, WeightInitialization


class ConvolutionalNeuralNetwork(ConfigurableTorchModule):
    """Generic CNN which can be used for classification."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        n_classes: int,
        seq_len: int,
        input_norm: InputNormalization,
    ):
        super().__init__()

        self.cnn = create_5_hidden_layer_convnet(
            input_channels,
            hidden_channels,
            out_channels,
            seq_len=seq_len,
            input_norm=input_norm,
        )

        self.fully_connected = Linear(in_features=out_channels, out_features=n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the CNN."""
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1).squeeze(1)
        return x

    @staticmethod
    def _configure(hidden_channels: List[int]) -> Dict:
        return {
            'module': {
                'LAYER_SPECIFICATION': {
                    'HIDDEN_CHANNELS': hidden_channels,
                },
                'WEIGHT_INIT': {'NAME': WeightInitialization.NORMAL.value, 'MEAN': 0.0, 'STD': 0.02},
            }
        }

    @staticmethod
    def configure() -> Dict:
        """Return a default configuration of a CNN."""
        config = ConvolutionalNeuralNetwork._configure(
            hidden_channels=[32, 64, 128, 256, 512],
        )

        config['update'] = {
            'trainer': {
                'TRAIN_ONLY_NORMAL': False,
                'BINARY_LABELS': False,
                'SPLIT': (0.70, 0.15, 0.15),
            }
        }
        return config


class DownsampleCNN(ConfigurableTorchModule):
    """A CNN used for downsampling."""

    def __init__(
        self,
        kernel_sizes: List[int],
        pooling_kernel_size: int,
        input_channels: int,
        output_channels: int,
        seq_len: int,
        sampling_seq_len: int,
    ):
        super().__init__()

        paddings = [2, 2, 1, 1, 1]
        strides = [2, 2, 2, 2, 2]
        self.seq_len = seq_len
        self.sampling_seq_len = sampling_seq_len
        self.output_channels = output_channels

        hidden_channels = [16 * (2 ** i) for i in range(0, len(kernel_sizes))]
        before_pooling = seq_len // (2 ** 4)
        after_pooling = (before_pooling - pooling_kernel_size) // pooling_kernel_size + 1
        linear_layer_in_size = seq_len * input_channels * after_pooling

        linear_layer_out_size = self.sampling_seq_len * output_channels

        model = [
            conv_norm_relu(
                input_channels=input_channels,
                output_channels=hidden_channels[0],
                kernel_size=kernel_sizes[0],
                stride=strides[0],
                padding=paddings[0],
            ),
            conv_norm_relu(
                input_channels=hidden_channels[0],
                output_channels=hidden_channels[1],
                kernel_size=kernel_sizes[1],
                stride=strides[1],
                padding=paddings[1],
            ),
            conv_norm_relu(
                input_channels=hidden_channels[1],
                output_channels=hidden_channels[2],
                kernel_size=kernel_sizes[2],
                stride=strides[2],
                padding=paddings[2],
            ),
            conv_norm_relu(
                input_channels=hidden_channels[2],
                output_channels=seq_len,
                kernel_size=kernel_sizes[3],
                stride=strides[3],
                padding=paddings[3],
            ),
            nn.AvgPool1d(kernel_size=pooling_kernel_size),
            nn.Flatten(),
            nn.Linear(in_features=linear_layer_in_size, out_features=linear_layer_out_size),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward pass of the downsample CNN."""
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.shape
        x = self.model(x)
        return x.view(batch_size, self.sampling_seq_len, self.output_channels)

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the `DownsampleCNN`."""
        return {
            'INV_MODULE': {
                'NAME': InverseMappingType.SIMPLE.value,
                'KERNEL_SIZES': [5, 5, 3, 3, 3],
                'LOSS': 'L2',
            }
        }
