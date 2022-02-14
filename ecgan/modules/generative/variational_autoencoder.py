"""
Implementation of a architecture using a variational autoencoder as generator.

No discriminator is used in this model, :ref:`ecgan.modules.generative.vaegan` additionally utilizes adversarial info.
"""
import logging
from itertools import chain
from typing import Any, Dict, cast

from torch import nn
from torch.distributions import Normal

from ecgan.config import BaseCNNConfig, OptimizerConfig, VariationalAutoEncoderConfig, nested_dataclass_asdict
from ecgan.modules.generative.autoencoder import AutoEncoder
from ecgan.networks.beatgan import BeatganGenerator, BeatganInverseEncoder
from ecgan.networks.vaegan import VAEEncoder
from ecgan.utils.custom_types import InputNormalization, LatentDistribution, WeightInitialization
from ecgan.utils.layers import initialize_batchnorm, initialize_weights
from ecgan.utils.losses import BceGeneratorLoss, VariationalAutoEncoderLoss
from ecgan.utils.optimizer import Adam, BaseOptimizer, OptimizerFactory
from ecgan.utils.sampler import VAEGANGeneratorSampler

logger = logging.getLogger(__name__)


class VariationalAutoEncoder(AutoEncoder):
    """Basic autoencoder model."""

    BATCHNORM_MEAN = 1.0
    BATCHNORM_STD = 0.02
    BATCHNORM_BIAS = 0

    def __init__(
        self,
        cfg: VariationalAutoEncoderConfig,
        seq_len: int,
        num_channels: int,
    ):
        self.distribution = Normal(0, 1)
        super().__init__(cfg, seq_len, num_channels)
        self.cfg: VariationalAutoEncoderConfig = cast(VariationalAutoEncoderConfig, self.cfg)

    @property
    def autoencoder_sampler(self) -> VAEGANGeneratorSampler:
        return cast(VAEGANGeneratorSampler, self._autoencoder_sampler)

    def _init_encoder(self) -> nn.Module:
        if not isinstance(self.cfg.ENCODER.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN with config {}.".format(type(self.cfg)))
        model = VAEEncoder(
            input_channels=self.num_channels,
            latent_size=self.latent_size,
            hidden_channels=self.cfg.ENCODER.LAYER_SPECIFICATION.HIDDEN_CHANNELS,
            seq_len=self.seq_len,
            input_norm=InputNormalization(self.cfg.ENCODER.INPUT_NORMALIZATION),
        )

        initialize_batchnorm(model, mean=self.BATCHNORM_MEAN, std=self.BATCHNORM_STD, bias=self.BATCHNORM_BIAS)
        initialize_weights(model, self.cfg.ENCODER.WEIGHT_INIT)

        return model

    def _init_decoder(self) -> nn.Module:
        if not isinstance(self.cfg.DECODER.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN with config {}.".format(type(self.cfg)))

        model = BeatganGenerator(
            self.num_channels,
            self.cfg.DECODER.LAYER_SPECIFICATION.HIDDEN_CHANNELS,
            self.latent_size,
            self.seq_len,
            input_norm=InputNormalization(self.cfg.DECODER.INPUT_NORMALIZATION),
            spectral_norm=self.cfg.DECODER.SPECTRAL_NORM,
            tanh_out=self.cfg.TANH_OUT,
        )
        initialize_batchnorm(model, mean=self.BATCHNORM_MEAN, std=self.BATCHNORM_STD, bias=self.BATCHNORM_BIAS)
        initialize_weights(model, self.cfg.DECODER.WEIGHT_INIT)

        return model

    def _init_autoencoder_sampler(self) -> VAEGANGeneratorSampler:
        return VAEGANGeneratorSampler(
            component=self.decoder,
            encoder=self.encoder,
            dev=self.device,
            num_channels=self.num_channels,
            sampling_seq_length=1,
            distribution=self.distribution,
        )

    def _init_criterion(self) -> Any:
        return VariationalAutoEncoderLoss(
            self.autoencoder_sampler,
            self.cfg.TANH_OUT,  # use MSE if Tanh out, use BCE if sig out
            self.cfg.KL_BETA,
            distribution=self.distribution,
            device=self.device,
        )

    def _init_optim(self) -> BaseOptimizer:
        # Optimizes the generator as well as the encoder, no additional optim for the encoder is used.
        return OptimizerFactory()(
            chain(self.encoder.parameters(), self.decoder.parameters()),
            OptimizerConfig(**nested_dataclass_asdict(self.cfg.DECODER.OPTIMIZER)),
        )

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the VAE model."""
        config = {
            'module': {
                'LATENT_SIZE': 5,
                'KL_BETA': 0.0001,
                'LATENT_SPACE': LatentDistribution.ENCODER_BASED.value,
                'TANH_OUT': True,
                'DECODER': {
                    'LAYER_SPECIFICATION': {
                        'HIDDEN_CHANNELS': [512, 256, 128, 64, 32],
                    },
                    'INPUT_NORMALIZATION': InputNormalization.BATCH.value,
                    'SPECTRAL_NORM': False,
                    'WEIGHT_INIT': {
                        'NAME': WeightInitialization.GLOROT_NORMAL.value,
                    },
                },
            }
        }

        config['module'].update(BeatganInverseEncoder.configure())  # type: ignore
        config['module']['DECODER'].update(BceGeneratorLoss.configure())  # type: ignore
        config['module']['DECODER'].update(Adam.configure())  # type: ignore
        config['module']['DECODER']['OPTIMIZER']['BETAS'] = [0.5, 0.999]  # type: ignore

        return config
