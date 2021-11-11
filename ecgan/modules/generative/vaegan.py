"""
Module for a GAN sampling from a Variational Autoencoder (VAE).

The VAE is inspired by the implementation of "Survival-oriented embeddings with application to CT scans of colorectal
carcinoma patients with liver metastases" by Tobias Weber, 2021. The basic structure of the underlying AE-GAN
architecture is inspired by :class:`ecgan.modules.generative.aegan.AEGAN`.
"""
from logging import getLogger
from typing import Dict, List, cast

import torch
from torch import nn
from torch.distributions.normal import Normal

from ecgan.config import BaseCNNConfig, VAEGANConfig
from ecgan.modules.generative.aegan import AEGAN
from ecgan.networks.beatgan import BeatganDiscriminator, BeatganGenerator
from ecgan.networks.vaegan import VAEEncoder
from ecgan.utils.artifacts import Artifact, FileArtifact, ImageArtifact
from ecgan.utils.custom_types import InputNormalization, LatentDistribution, LossMetricType
from ecgan.utils.layers import initialize_batchnorm, initialize_weights
from ecgan.utils.losses import BCELoss, L2Loss, SupervisedLoss, VAEGANGeneratorLoss
from ecgan.utils.sampler import GeneratorSampler, VAEGANGeneratorSampler

logger = getLogger(__name__)


class VAEGAN(AEGAN):
    """Variational Autoencoder for encoding data to latent space, reconstructing it and judging the quality."""

    def __init__(self, cfg: VAEGANConfig, seq_len: int, num_channels: int):
        """Initialize the VAE GAN."""
        self.cfg: VAEGANConfig = cfg
        ############################################################
        # SET PARAMETERS
        ############################################################
        self.kl_warm_up = self.cfg.KL_WARMUP
        self.kl_anneal_rounds = self.cfg.KL_ANNEAL_ROUNDS
        self.kl_beta = float(self.cfg.KL_BETA)

        if self.cfg.GENERATOR.TANH_OUT:
            logger.info("Using MSE as reconstruction loss.")
            self.rec_loss: SupervisedLoss = L2Loss(reduction='sum')
        else:
            logger.info("Using BCE as reconstruction loss.")
            self.rec_loss = BCELoss(reduction='sum')

        self.distribution = Normal(0, 1)
        self.ep_counter = 0
        super().__init__(cfg, seq_len, num_channels)
        self.mu_train = torch.empty(0).to(self.device)
        self.mu_vali = torch.empty(0).to(self.device)

    @property
    def generator_sampler(self) -> VAEGANGeneratorSampler:
        return cast(VAEGANGeneratorSampler, self._generator_sampler)

    def _init_generator_sampler(self) -> GeneratorSampler:
        return VAEGANGeneratorSampler(
            component=self.generator,
            encoder=self.encoder,
            dev=self.device,
            num_channels=self.num_channels,
            sampling_seq_length=1,
            distribution=self.distribution,
        )

    @property
    def criterion_gen(self) -> VAEGANGeneratorLoss:  # type: ignore
        return cast(VAEGANGeneratorLoss, self._criterion_gen)

    def _init_criterion_gen(self) -> VAEGANGeneratorLoss:
        return VAEGANGeneratorLoss(
            self.discriminator_sampler,
            self.generator_sampler,
            self.rec_loss,
            self.distribution,
            self.kl_beta,
            self.device,
        )

    def _init_inverse_mapping(self) -> nn.Module:
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

    def get_kl_weight(self) -> float:
        """Get annealed weight for Kullback-Leibler Divergence."""
        if self.ep_counter < self.kl_warm_up:
            return 0.0
        if self.ep_counter >= self.kl_warm_up + self.kl_anneal_rounds:
            return 1.0

        progress = self.ep_counter - self.kl_warm_up
        return progress / self.kl_anneal_rounds

    def _prepare_train_step(self) -> None:
        self.criterion_gen.kl_weight = self.get_kl_weight()

    def _evaluate_train_step(self, disc_metrics: LossMetricType, gen_metrics: LossMetricType):
        for idx, (key, val) in enumerate(gen_metrics):
            if key == 'latent/z_mu':
                self.mu_train = torch.cat((self.mu_train, val), dim=0)
                avg_mu = float(torch.mean(val))
                gen_metrics[idx] = ('latent/z_mu_avg', avg_mu)
            if key == 'latent/noise':
                self.latent_vectors_train = torch.cat((self.latent_vectors_train, val), dim=0)
                del gen_metrics[idx]

        return {key: float(value) for (key, value) in disc_metrics + gen_metrics}

    def _get_validation_results(self, data: torch.Tensor):
        with torch.no_grad():
            _, (
                (_, kl_loss),
                _,
                (_, reconstruction_loss),
                (_, mu),
                _,
                _,
                (_, weighted_error),
            ) = self.criterion_gen(data)
        self.mu_vali = torch.cat((self.mu_vali, mu), dim=0)

        self.ep_counter += 1

        metrics = {
            'val/loss': float(weighted_error),
            'val/rec_loss_scaled': float(reconstruction_loss),
            'val/kl_loss': float(kl_loss),
        }
        return metrics

    def _on_epoch_end_addition(self, epoch: int, sample_interval: int) -> List[Artifact]:
        _result: List[Artifact] = []
        if not epoch % sample_interval == 0:
            return []
        mu_norm_train = torch.norm(self.mu_train.squeeze(), dim=1)
        mu_norm_vali_abnormal = torch.norm(self.mu_vali[self.label != 0].squeeze(), dim=1)

        _result.append(
            ImageArtifact(
                'Norm of mu vectors (normal train)',
                self.plotter.create_histogram(mu_norm_train.cpu().numpy(), 'Norm of mu vectors (normal train)'),
            )
        )

        _result.append(
            ImageArtifact(
                'Norm of mu vectors (abnormal validation)',
                self.plotter.create_histogram(
                    mu_norm_vali_abnormal.cpu().numpy(), 'Norm of mu vectors (abnormal validation)'
                ),
            )
        )

        _result.append(
            FileArtifact(
                'Mu vectors',
                {'mu_train': self.mu_train.cpu(), 'mu_vali': self.mu_vali.cpu(), 'labels': self.label.cpu()},
                'mu_data_{}.pkl'.format(epoch),
            )
        )
        return _result

    def _reset_internal_tensors(self):
        """Reset tensors which are filled internally during an epoch."""
        super()._reset_internal_tensors()
        self.mu_train = torch.empty(0).to(self.device)
        self.mu_vali = torch.empty(0).to(self.device)

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the VAEGAN model."""
        config = {
            'module': {
                'LATENT_SIZE': 50,
                'DISCRIMINATOR_ROUNDS': 1,
                'LATENT_SPACE': LatentDistribution.ENCODER_BASED.value,
                'GENERATOR_ROUNDS': 1,
                'KL_WARMUP': 0,
                'KL_BETA': 0.01,
                'KL_ANNEAL_ROUNDS': 0,
            }
        }

        config['module'].update(VAEEncoder.configure())
        config['module'].update(BeatganGenerator.configure())
        config['module'].update(BeatganDiscriminator.configure())
        config['update'] = {'trainer': {'BATCH_SIZE': 256, 'CHANNELS': 1}}

        return config
