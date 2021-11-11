"""VAEGAN encoder."""
from logging import getLogger
from typing import Dict, List, Optional, Tuple

from torch import Tensor, nn
from torch.nn import Flatten, LeakyReLU, Linear

from ecgan.networks.helpers import apply_input_normalization, conv1d_block
from ecgan.utils.configurable import ConfigurableTorchModule
from ecgan.utils.custom_types import InputNormalization, WeightInitialization
from ecgan.utils.losses import VAEGANGeneratorLoss
from ecgan.utils.optimizer import AdaBelief

logger = getLogger(__name__)


class VAEEncoder(ConfigurableTorchModule):
    """Variational Convolutional Encoder Module."""

    def __init__(
        self,
        input_channels: int,
        latent_size: int,
        hidden_channels: List[int],
        seq_len: int,
        spectral_norm: bool = False,
        input_norm: Optional[InputNormalization] = None,
        track_running_stats: bool = True,
    ):
        """
        Initialize a variational encoder.

        The resulting mu, sigma can be used to create noise using the reparametrization trick.

        .. note::
            The sequence length needs to be divisible by 32 for the pooling kernel.
        """
        super().__init__()

        pooling_kernel = seq_len // 32
        # The calculations are for pooling_kernel=10 (seq_len=320).
        #####################################
        # CONV LAYER 1 IN: IN_CHANNELS x 320, OUT: hidden_channels x 160
        #####################################
        conv1 = conv1d_block(input_channels, hidden_channels[0], k=4, s=2, p=1)
        #####################################
        # CONV LAYER 2 OUT : HIDDEN x 80
        #####################################
        conv2 = conv1d_block(hidden_channels[0], hidden_channels[1], k=4, s=2, p=1)
        #####################################
        # CONV LAYER 3 OUT : HIDDEN x 40
        #####################################
        conv3 = conv1d_block(hidden_channels[1], hidden_channels[2], k=4, s=2, p=1)
        #####################################
        # CONV LAYER 4 OUT : HIDDEN x 20
        #####################################
        conv4 = conv1d_block(hidden_channels[2], hidden_channels[3], k=4, s=2, p=1)
        #####################################
        # CONV LAYER 5 OUT : HIDDEN x 10
        #####################################
        conv5 = conv1d_block(hidden_channels[3], hidden_channels[4], k=4, s=2, p=1)

        if spectral_norm:
            logger.info("Using weight normalization (spectral norm) in encoder net.")
            conv2 = nn.utils.spectral_norm(conv2)
            conv3 = nn.utils.spectral_norm(conv3)
            conv4 = nn.utils.spectral_norm(conv4)
            conv5 = nn.utils.spectral_norm(conv5)

        logger.info("Using {0} input normalization in encoder net.".format(input_norm))
        norm1 = apply_input_normalization(hidden_channels[1], input_norm, track_running_stats=track_running_stats)
        norm2 = apply_input_normalization(hidden_channels[2], input_norm, track_running_stats=track_running_stats)
        norm3 = apply_input_normalization(hidden_channels[3], input_norm, track_running_stats=track_running_stats)
        norm4 = apply_input_normalization(hidden_channels[4], input_norm, track_running_stats=track_running_stats)

        module_list = [
            conv1,
            nn.LeakyReLU(0.2, inplace=True),
            conv2,
            norm1,
            nn.LeakyReLU(0.2, inplace=True),
            conv3,
            norm2,
            nn.LeakyReLU(0.2, inplace=True),
            conv4,
            norm3,
            nn.LeakyReLU(0.2, inplace=True),
            conv5,
            norm4,
            nn.LeakyReLU(0.2, inplace=True),
        ]
        module_list = [mod for mod in module_list if mod is not None]
        self.enc = nn.Sequential(*module_list)  # type: ignore

        ##################################################
        # EXPECTED VALUE
        ##################################################

        self.mu = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channels[-1],
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(16),
            LeakyReLU(0.2),
            Flatten(),
            # Linear in_features = out_channels * output of last layer from net = (16)*(10) for a seq_len of 320.
            Linear(in_features=16 * pooling_kernel, out_features=latent_size),
        )

        ##################################################
        # LOG VARIANCE
        ##################################################
        self.log_var = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channels[-1],
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(16),
            LeakyReLU(0.2),
            Flatten(),
            Linear(in_features=16 * pooling_kernel, out_features=latent_size),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the VAEGAN encoder.

        Args:
            x: Input data.

        Returns:
            Tuple of (mu, log_var).
        """
        x = x.permute(0, 2, 1)
        x = self.enc(x)
        return self.mu(x).unsqueeze(1), self.log_var(x).unsqueeze(1)

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the encoder of the VAEGAN module."""
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
        # Loss is ignored during training, is part of generator loss.
        config['ENCODER'].update(VAEGANGeneratorLoss.configure())
        config['ENCODER'].update(AdaBelief.configure())

        return config
