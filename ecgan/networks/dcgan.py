"""Adapted DCGAN generator and discriminator."""
from logging import getLogger
from typing import Dict, List

from torch import Tensor, nn
from torch.nn import LeakyReLU

from ecgan.config import BaseCNNConfig, BaseNNConfig, GeneratorConfig
from ecgan.networks.helpers import apply_input_normalization, conv1d_block, conv1d_trans_block
from ecgan.utils.configurable import ConfigurableTorchModule
from ecgan.utils.custom_types import InputNormalization, WeightInitialization
from ecgan.utils.datasets import SineDataset
from ecgan.utils.losses import WassersteinDiscriminatorLoss, WassersteinGeneratorLoss
from ecgan.utils.optimizer import Adam

logger = getLogger(__name__)


class DCGANGenerator(ConfigurableTorchModule):
    """A generator using an architecture similar to `Radford et al. 2015 <https://arxiv.org/abs/1511.06434>`_."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        params: GeneratorConfig,
        seq_len: int = 128,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        use_fully_connected = seq_len != SineDataset.default_seq_len

        if not isinstance(params.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN with config {}.".format(type(params)))

        hidden_channels = params.LAYER_SPECIFICATION.HIDDEN_CHANNELS
        kernels = [4, 4, 4, 4, 4, 4]
        paddings = [0, 1, 1, 1, 1, 1]
        strides = [1, 2, 2, 2, 2, 2]

        conv1t = conv1d_trans_block(input_channels, hidden_channels[0], k=kernels[0], s=strides[0], p=paddings[0])
        conv2t = conv1d_trans_block(hidden_channels[0], hidden_channels[1], k=kernels[1], s=strides[1], p=paddings[1])
        conv3t = conv1d_trans_block(hidden_channels[1], hidden_channels[2], k=kernels[2], s=strides[2], p=paddings[2])
        conv4t = conv1d_trans_block(hidden_channels[2], hidden_channels[3], k=kernels[3], s=strides[3], p=paddings[3])
        conv5t = conv1d_trans_block(hidden_channels[3], hidden_channels[4], k=kernels[4], s=strides[4], p=paddings[4])
        conv6t = conv1d_trans_block(hidden_channels[4], output_channels, k=kernels[5], s=strides[5], p=paddings[5])

        module_list: List[nn.Module] = [
            #####################################
            # CONV LAYER 1 OUT: 4
            #####################################
            conv1t,
            nn.BatchNorm1d(hidden_channels[0]),
            nn.ReLU(),
            #####################################
            # CONV LAYER 2 OUT: 8
            #####################################
            conv2t,
            nn.BatchNorm1d(hidden_channels[1]),
            nn.ReLU(),
            #####################################
            # CONV LAYER 3 OUT: 16
            #####################################
            conv3t,
            nn.BatchNorm1d(hidden_channels[2]),
            nn.ReLU(),
            #####################################
            # CONV LAYER 4 OUT: 32
            #####################################
            conv4t,
            nn.BatchNorm1d(hidden_channels[3]),
            nn.ReLU(),
            #####################################
            # CONV LAYER 5 OUT: 64
            #####################################
            conv5t,
            nn.BatchNorm1d(hidden_channels[4]),
            nn.ReLU(),
            #####################################
            # CONV LAYER 6 OUT: 128
            #####################################
            conv6t,
        ]

        if use_fully_connected:
            module_list.extend(
                [
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=SineDataset.default_seq_len, out_features=seq_len),
                ]
            )

        module_list.append(nn.Tanh())
        self.net = nn.Sequential(*module_list)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the generator."""
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the generator of the DCGAN module."""
        config = {
            'GENERATOR': {
                'SPECTRAL_NORM': False,
                'LAYER_SPECIFICATION': {
                    'HIDDEN_CHANNELS': [256, 128, 64, 32, 16],
                },
                'TANH_OUT': True,
                'WEIGHT_INIT': {'NAME': WeightInitialization.NORMAL.value, 'MEAN': 0.0, 'STD': 0.02},
            }
        }
        config['GENERATOR'].update(WassersteinGeneratorLoss.configure())
        config['GENERATOR'].update(Adam.configure())
        return config


class DCGANDiscriminator(ConfigurableTorchModule):
    """Slightly modified discriminator from `Radford et al. 2015 <https://arxiv.org/abs/1511.06434>`_."""

    def __init__(
        self,
        input_channels: int,
        params: BaseNNConfig,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.use_spectral_norm = params.SPECTRAL_NORM

        if not isinstance(params.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN with config {}.".format(type(params)))

        hidden_channels = params.LAYER_SPECIFICATION.HIDDEN_CHANNELS
        kernels = [5, 5, 3, 3, 3, 4]
        paddings = [2, 2, 1, 1, 1, 0]
        strides = [2, 2, 2, 2, 2, 1]

        #####################################
        # CONV LAYER 1 OUT: 64
        #####################################
        conv1 = conv1d_block(self.input_channels, hidden_channels[0], k=kernels[0], s=strides[0], p=paddings[0])
        #####################################
        # CONV LAYER 2 OUT: 32
        #####################################
        conv2 = conv1d_block(hidden_channels[0], hidden_channels[1], k=kernels[1], s=strides[1], p=paddings[1])
        #####################################
        # CONV LAYER 3 OUT: 16
        #####################################
        conv3 = conv1d_block(hidden_channels[1], hidden_channels[2], k=kernels[2], s=strides[2], p=paddings[2])
        #####################################
        # CONV LAYER 4 OUT: 8
        #####################################
        conv4 = conv1d_block(hidden_channels[2], hidden_channels[3], k=kernels[3], s=strides[3], p=paddings[3])
        #####################################
        # CONV LAYER 5 OUT: 4
        #####################################
        conv5 = conv1d_block(hidden_channels[3], hidden_channels[4], k=kernels[4], s=strides[4], p=paddings[4])
        #####################################
        # CONV LAYER 6 OUT: 1
        #####################################
        conv6 = conv1d_block(hidden_channels[4], 1, k=kernels[5], s=strides[5], p=paddings[5])

        input_norm = InputNormalization.BATCH
        norm1 = apply_input_normalization(hidden_channels[0], input_norm, track_running_stats=False)
        norm2 = apply_input_normalization(hidden_channels[1], input_norm, track_running_stats=False)
        norm3 = apply_input_normalization(hidden_channels[2], input_norm, track_running_stats=False)
        norm4 = apply_input_normalization(hidden_channels[3], input_norm, track_running_stats=False)
        norm5 = apply_input_normalization(hidden_channels[4], input_norm, track_running_stats=False)

        if self.use_spectral_norm:
            logger.info("Using weight normalization (spectral norm) in encoder net.")
            conv1 = nn.utils.spectral_norm(conv1)
            conv2 = nn.utils.spectral_norm(conv2)
            conv3 = nn.utils.spectral_norm(conv3)
            conv4 = nn.utils.spectral_norm(conv4)
            conv5 = nn.utils.spectral_norm(conv5)

        module_list = [
            conv1,
            norm1,
            LeakyReLU(),
            conv2,
            norm2,
            LeakyReLU(),
            conv3,
            norm3,
            LeakyReLU(),
            conv4,
            norm4,
            LeakyReLU(),
            conv5,
            norm5,
            LeakyReLU(),
            conv6,
            nn.AdaptiveAvgPool1d(1),
            nn.Sigmoid(),
        ]

        module_list = [mod for mod in module_list if mod is not None]
        self.net = nn.Sequential(*module_list)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the discriminator."""
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.view(-1)

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the discriminator of the DCGAN module."""
        config = {
            'DISCRIMINATOR': {
                'SPECTRAL_NORM': False,
                'LAYER_SPECIFICATION': {
                    'HIDDEN_CHANNELS': [16, 32, 64, 128, 256],
                },
                'WEIGHT_INIT': {'NAME': WeightInitialization.NORMAL.value, 'MEAN': 0.0, 'STD': 0.02},
            }
        }
        config['DISCRIMINATOR'].update(WassersteinDiscriminatorLoss.configure())
        config['DISCRIMINATOR'].update(Adam.configure())

        return config
