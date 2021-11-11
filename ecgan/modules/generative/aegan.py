"""
Implementation of a architecture using an autoencoder as generator and a discriminator.

Based on "BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series" by Zhou et al. 2019.
We extend their work by a validation loop that includes the discriminator error and additional flexibility,
especially allowing spectral weight normalization and similar configurable improvements such as the AdaBelief optimizer.
"""
import logging
from itertools import chain
from typing import Dict, Optional, Tuple, cast

import torch
from torch import Tensor, nn

from ecgan.config import BaseCNNConfig, EncoderGANConfig, OptimizerConfig, nested_dataclass_asdict
from ecgan.modules.generative.base import BaseEncoderGANModule
from ecgan.networks.beatgan import BeatganDiscriminator, BeatganGenerator, BeatganInverseEncoder
from ecgan.utils.custom_types import InputNormalization, LatentDistribution, LossMetricType
from ecgan.utils.layers import initialize_batchnorm, initialize_weights
from ecgan.utils.losses import AEGANDiscriminatorLoss, AEGANGeneratorLoss, BCELoss, GANBaseLoss, L2Loss
from ecgan.utils.optimizer import BaseOptimizer, OptimizerFactory
from ecgan.utils.sampler import EncoderBasedGeneratorSampler, FeatureDiscriminatorSampler, GeneratorSampler

logger = logging.getLogger(__name__)


class AEGAN(BaseEncoderGANModule):
    """
    GAN-Autoencoder model.

    Based on the `reference implementation of BeatGAN <https://github.com/Vniex/BeatGAN>`_.
    """

    BATCHNORM_MEAN = 1.0
    BATCHNORM_STD = 0.02
    BATCHNORM_BIAS = 0

    def __init__(
        self,
        cfg: EncoderGANConfig,
        seq_len: int,
        num_channels: int,
    ):

        self.data_sampler = None
        self.latent_size = cfg.LATENT_SIZE
        self.seq_len = seq_len
        self.num_channels = num_channels
        super().__init__(cfg=cfg, seq_len=seq_len, num_channels=num_channels)

        self.mse_criterion = L2Loss()
        self.bce_criterion = BCELoss()
        self.data_sampler = self.train_dataset_sampler

    @property
    def criterion_gen(self) -> AEGANGeneratorLoss:
        return cast(AEGANGeneratorLoss, self._criterion_gen)

    @property
    def criterion_disc(self) -> AEGANDiscriminatorLoss:
        return cast(AEGANDiscriminatorLoss, self._criterion_disc)

    def _init_inverse_mapping(self) -> nn.Module:
        self.cfg = cast(EncoderGANConfig, self.cfg)
        if not isinstance(self.cfg.ENCODER.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN with config {}.".format(type(self.cfg)))
        model = BeatganInverseEncoder(
            input_channels=self.num_channels,
            output_channels=self.latent_size,
            hidden_channels=self.cfg.ENCODER.LAYER_SPECIFICATION.HIDDEN_CHANNELS,
            seq_len=self.seq_len,
            input_norm=InputNormalization(self.cfg.ENCODER.INPUT_NORMALIZATION),
            spectral_norm=self.cfg.ENCODER.SPECTRAL_NORM,
        )

        initialize_batchnorm(model, mean=self.BATCHNORM_MEAN, std=self.BATCHNORM_STD, bias=self.BATCHNORM_BIAS)
        initialize_weights(model, self.cfg.ENCODER.WEIGHT_INIT)

        return model

    def _init_generator(self) -> nn.Module:

        if not isinstance(self.cfg.GENERATOR.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN with config {}.".format(type(self.cfg)))

        model = BeatganGenerator(
            self.num_channels,
            self.cfg.GENERATOR.LAYER_SPECIFICATION.HIDDEN_CHANNELS,
            self.latent_size,
            self.seq_len,
            input_norm=InputNormalization(self.cfg.GENERATOR.INPUT_NORMALIZATION),
            spectral_norm=self.cfg.GENERATOR.SPECTRAL_NORM,
            tanh_out=self.cfg.GENERATOR.TANH_OUT,
        )
        initialize_batchnorm(model, mean=self.BATCHNORM_MEAN, std=self.BATCHNORM_STD, bias=self.BATCHNORM_BIAS)
        initialize_weights(model, self.cfg.GENERATOR.WEIGHT_INIT)

        return model

    def _init_discriminator(self) -> nn.Module:
        if not isinstance(self.cfg.DISCRIMINATOR.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN with config {}.".format(type(self.cfg)))
        model = BeatganDiscriminator(
            self.num_channels,
            self.cfg.DISCRIMINATOR.LAYER_SPECIFICATION.HIDDEN_CHANNELS,
            1,
            seq_len=self.seq_len,
            input_norm=InputNormalization(self.cfg.DISCRIMINATOR.INPUT_NORMALIZATION),
            spectral_norm=self.cfg.DISCRIMINATOR.SPECTRAL_NORM,
        )

        initialize_batchnorm(model, mean=self.BATCHNORM_MEAN, std=self.BATCHNORM_STD, bias=self.BATCHNORM_BIAS)
        initialize_weights(model, self.cfg.DISCRIMINATOR.WEIGHT_INIT)
        return model

    def _init_generator_sampler(self) -> GeneratorSampler:
        return EncoderBasedGeneratorSampler(
            component=self.generator,
            encoder=self.encoder,
            dev=self.device,
            num_channels=self.num_channels,
            sampling_seq_length=1,
        )

    def _init_criterion_gen(self) -> GANBaseLoss:
        return AEGANGeneratorLoss(
            self.discriminator_sampler,
            cast(EncoderBasedGeneratorSampler, self.generator_sampler),
        )

    def _init_criterion_disc(self) -> AEGANDiscriminatorLoss:
        return AEGANDiscriminatorLoss(self.discriminator_sampler, self.generator_sampler)

    def _init_discriminator_sampler(self) -> FeatureDiscriminatorSampler:
        return FeatureDiscriminatorSampler(
            component=self.discriminator,
            dev=self.device,
            num_channels=self.num_channels,
            sampling_seq_length=1,
        )

    def _init_optim_gen(self) -> BaseOptimizer:
        # Optimizes the generator as well as the encoder, no additional optim for the encoder is used.
        return OptimizerFactory()(
            chain(self.generator.parameters(), self.encoder.parameters()),
            OptimizerConfig(**nested_dataclass_asdict(self.cfg.GENERATOR.OPTIMIZER)),
        )

    def _init_optim_disc(self) -> BaseOptimizer:
        return OptimizerFactory()(
            self.discriminator.parameters(),
            OptimizerConfig(**nested_dataclass_asdict(self.cfg.DISCRIMINATOR.OPTIMIZER)),
        )

    def _evaluate_train_step(self, disc_metrics: LossMetricType, gen_metrics: LossMetricType) -> Dict:
        """
        Evaluate and process metrics from the losses.

        Metrics are logged and the noise is concatenated for later use during on_epoch_end.

        Args:
            disc_metrics: Discriminator metrics retrieved from the loss function.
            gen_metrics: Generator metrics retrieved from the loss function.

        Returns:
            Dict with params to be logged to the tracker.
        """
        for idx, (key, val) in enumerate(gen_metrics):
            if key == 'noise':
                self.latent_vectors_train = torch.cat((self.latent_vectors_train, val), dim=0)
                del gen_metrics[idx]

        return {key: float(value) for (key, value) in disc_metrics + gen_metrics}

    def _get_validation_results(self, data: torch.Tensor) -> Dict:
        return {}

    def get_sample(
        self, num_samples: Optional[int] = None, data: Optional[torch.Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve the reconstructed sampled x_hat from our model."""
        if data is None:
            raise RuntimeError("Data tensor may not be empty.")

        return self.generator_sampler.sample_generator_encoder(data)

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the BeatGAN model."""
        config = {
            'module': {
                'LATENT_SIZE': 50,
                'DISCRIMINATOR_ROUNDS': 1,
                'LATENT_SPACE': LatentDistribution.ENCODER_BASED.value,
                'GENERATOR_ROUNDS': 1,
            }
        }

        config['module'].update(BeatganInverseEncoder.configure())
        config['module'].update(BeatganGenerator.configure())
        config['module'].update(BeatganDiscriminator.configure())

        return config
