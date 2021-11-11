"""Implementation of the DCGAN architecture."""
from typing import Dict

from torch import nn

from ecgan.config import GANModuleConfig, OptimizerConfig, nested_dataclass_asdict
from ecgan.modules.generative.base import BaseGANModule
from ecgan.networks.dcgan import DCGANDiscriminator, DCGANGenerator
from ecgan.utils.custom_types import LatentDistribution
from ecgan.utils.layers import initialize_weights
from ecgan.utils.losses import GANBaseLoss, GANLossFactory
from ecgan.utils.optimizer import BaseOptimizer, OptimizerFactory
from ecgan.utils.sampler import DiscriminatorSampler, GeneratorSampler, LatentDistributionFactory


class DCGAN(BaseGANModule):
    """
    Module for DCGAN.

    PyTorch implementation of DCGAN from `Radford et al. 2015 <https://arxiv.org/abs/1511.06434>`_ with improvements.
    """

    DEFAULT_LATENT_SIZE = 5
    DEFAULT_DISC_ROUNDS = 2
    DEFAULT_GEN_ROUNDS = 1

    def __init__(
        self,
        cfg: GANModuleConfig,
        seq_len: int = 128,
        num_channels: int = 12,
    ):
        super().__init__(
            cfg=cfg,
            seq_len=seq_len,
            num_channels=num_channels,
        )

    def _init_generator(self) -> nn.Module:
        """Initialize the generator."""
        gen = DCGANGenerator(
            input_channels=self.cfg.LATENT_SIZE,
            output_channels=self.num_channels,
            params=self.cfg.GENERATOR,
            seq_len=self.seq_len,
        )
        initialize_weights(gen, self.cfg.GENERATOR.WEIGHT_INIT)
        return gen

    def _init_discriminator(self) -> nn.Module:
        """Initialize the discriminator."""
        disc = DCGANDiscriminator(
            input_channels=self.num_channels,
            params=self.cfg.DISCRIMINATOR,
        )
        initialize_weights(disc, self.cfg.DISCRIMINATOR.WEIGHT_INIT)
        return disc

    def _init_generator_sampler(self) -> GeneratorSampler:
        """Initialize the sampler for the generator."""
        return GeneratorSampler(
            component=self.generator,
            dev=self.device,
            num_channels=self.num_channels,
            sampling_seq_length=1,
            latent_space=LatentDistributionFactory()(self.cfg.latent_distribution),
            latent_size=self.latent_size,
        )

    def _init_discriminator_sampler(self) -> DiscriminatorSampler:
        """Initialize the sampler for the discriminator."""
        return DiscriminatorSampler(
            component=self.discriminator,
            dev=self.device,
            num_channels=self.num_channels,
            sampling_seq_length=1,
        )

    def _init_optim_gen(self) -> BaseOptimizer:
        """Initialize the optimizer for the generator."""
        return OptimizerFactory()(
            self.generator.parameters(),
            OptimizerConfig(**nested_dataclass_asdict(self.cfg.GENERATOR.OPTIMIZER)),
        )

    def _init_optim_disc(self) -> BaseOptimizer:
        """Initialize the optimizer for the discriminator."""
        return OptimizerFactory()(
            self.discriminator.parameters(),
            OptimizerConfig(**nested_dataclass_asdict(self.cfg.DISCRIMINATOR.OPTIMIZER)),
        )

    def _init_criterion_gen(self) -> GANBaseLoss:
        """Initialize the criterion for the generator."""
        return GANLossFactory()(
            params=self.cfg.GENERATOR.LOSS,
            discriminator_sampler=self.discriminator_sampler,
            generator_sampler=self.generator_sampler,
        )

    def _init_criterion_disc(self) -> GANBaseLoss:
        """Initialize the criterion for the discriminator."""
        return GANLossFactory()(
            params=self.cfg.DISCRIMINATOR.LOSS,
            discriminator_sampler=self.discriminator_sampler,
            generator_sampler=self.generator_sampler,
        )

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration of a standard GAN."""
        base_config = BaseGANModule._configure(
            latent_size=5,
            latent_space=LatentDistribution.NORMAL,
            disc_rounds=2,
            gen_rounds=1,
        )
        base_config['module'].update(DCGANGenerator.configure())
        base_config['module'].update(DCGANDiscriminator.configure())
        return base_config
