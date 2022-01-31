"""BeatGAN encoder, generator and discriminator from Zhou et al. 2019."""
from typing import Dict, List

from torch import nn

from ecgan.networks.helpers import create_5_hidden_layer_convnet, create_transpose_conv_net
from ecgan.utils.configurable import ConfigurableTorchModule
from ecgan.utils.custom_types import InputNormalization, WeightInitialization
from ecgan.utils.losses import AEGANDiscriminatorLoss, BceGeneratorLoss, L2Loss
from ecgan.utils.optimizer import Adam


class BeatganInverseEncoder(ConfigurableTorchModule):
    """Encoder of the BeatGAN model."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        output_channels: int,
        seq_len: int,
        input_norm: InputNormalization,
        spectral_norm: bool,
    ):
        super().__init__()
        self.net = create_5_hidden_layer_convnet(
            input_channels,
            hidden_channels,
            output_channels,
            seq_len,
            input_norm=input_norm,
            spectral_norm=spectral_norm,
            track_running_stats=True,
        )

    def forward(self, x):
        """Perform a forward pass."""
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the encoder of the BeatGAN module."""
        config = {
            'ENCODER': {
                'LAYER_SPECIFICATION': {
                    'HIDDEN_CHANNELS': [32, 64, 128, 256, 512],
                },
                'INPUT_NORMALIZATION': InputNormalization.BATCH.value,
                'SPECTRAL_NORM': False,
                'WEIGHT_INIT': {
                    'NAME': WeightInitialization.GLOROT_NORMAL.value,
                },
            }
        }
        config['ENCODER'].update(L2Loss.configure())
        config['ENCODER'].update(Adam.configure())

        return config


class BeatganDiscriminator(ConfigurableTorchModule):
    """Discriminator of the BeatGAN model."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        output_channels: int,
        seq_len: int,
        input_norm: InputNormalization,
        spectral_norm: bool,
    ):
        super().__init__()
        model: nn.Module = create_5_hidden_layer_convnet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            seq_len=seq_len,
            input_norm=input_norm,
            spectral_norm=spectral_norm,
            track_running_stats=True,
        )

        layers = list(model.children())
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        """Perform a forward pass."""
        x = x.permute(0, 2, 1)
        features = self.features(x)
        classifier = self.classifier(features).view(-1, 1).squeeze(1)
        features.permute(0, 2, 1)

        return classifier, features

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the discriminator of the BeatGAN model."""
        config = {
            'DISCRIMINATOR': {
                'SPECTRAL_NORM': True,
                'LAYER_SPECIFICATION': {
                    'HIDDEN_CHANNELS': [32, 64, 128, 256, 512],
                },
                'INPUT_NORMALIZATION': InputNormalization.NONE.value,
                'WEIGHT_INIT': {
                    'NAME': WeightInitialization.GLOROT_NORMAL.value,
                },
            }
        }

        config['DISCRIMINATOR'].update(AEGANDiscriminatorLoss.configure())
        config['DISCRIMINATOR'].update(Adam.configure())
        config['DISCRIMINATOR']['OPTIMIZER']['BETAS'] = [0.5, 0.999]  # type: ignore
        return config


class BeatganGenerator(ConfigurableTorchModule):
    """Generator of the BeatGAN model."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        latent_size: int,
        seq_len: int,
        input_norm: InputNormalization,
        spectral_norm: bool,
        tanh_out: bool,
    ):
        super().__init__()

        self.model = create_transpose_conv_net(
            input_channels=latent_size,
            hidden_channels=hidden_channels,
            output_channels=input_channels,
            seq_len=seq_len,
            input_norm=input_norm,
            spectral_norm=spectral_norm,
            track_running_stats=True,
        )
        if tanh_out:
            self.model.add_module('Tanh', nn.Tanh())
        else:
            self.model.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        """Perform a forward pass."""
        x = x.permute(0, 2, 1)
        x = self.model(x)
        x = x.permute(0, 2, 1)
        return x

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the generator of the BeatGAN module."""
        config = {
            'GENERATOR': {
                'LAYER_SPECIFICATION': {
                    'HIDDEN_CHANNELS': [512, 256, 128, 64, 32],
                },
                'TANH_OUT': True,
                'INPUT_NORMALIZATION': InputNormalization.BATCH.value,
                'SPECTRAL_NORM': False,
                'WEIGHT_INIT': {
                    'NAME': WeightInitialization.GLOROT_NORMAL.value,
                },
            }
        }
        config['GENERATOR'].update(BceGeneratorLoss.configure())
        config['GENERATOR'].update(Adam.configure())
        config['GENERATOR']['OPTIMIZER']['BETAS'] = [0.5, 0.999]  # type: ignore
        return config
