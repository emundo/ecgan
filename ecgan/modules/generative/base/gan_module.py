"""Base class for GAN modules."""
from abc import abstractmethod
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.svm import SVC
from torch import nn, no_grad

from ecgan.config import GANModuleConfig, get_global_config
from ecgan.modules.generative.base.generative_module import BaseGenerativeModule
from ecgan.utils.artifacts import Artifact, ImageArtifact, ValueArtifact
from ecgan.utils.custom_types import LatentDistribution, Transformation
from ecgan.utils.losses import GANBaseLoss
from ecgan.utils.miscellaneous import load_model
from ecgan.utils.optimizer import BaseOptimizer
from ecgan.utils.sampler import DiscriminatorSampler, GeneratorSampler

logger = getLogger(__name__)


class BaseGANModule(BaseGenerativeModule):
    """Base class from which all implemented GANs should inherit."""

    DEFAULT_LATENT_SIZE = 0
    DEFAULT_DISC_ROUNDS = 0
    DEFAULT_GEN_ROUNDS = 0

    def __init__(
        self,
        cfg: GANModuleConfig,
        seq_len: int,
        num_channels: int,
    ):
        super().__init__(cfg, seq_len, num_channels)

        self.cfg: GANModuleConfig = cfg
        self.latent_size: int = self.cfg.LATENT_SIZE
        self.num_classes: int = self.dataset.NUM_CLASSES_BINARY
        self._generator = self._init_generator()
        self._generator = nn.DataParallel(self.generator)
        self._generator.to(self.device)

        self._discriminator = self._init_discriminator()
        self._discriminator = nn.DataParallel(self.discriminator)
        self._discriminator.to(self.device)

        self._optim_gen = self._init_optim_gen()
        self._optim_disc = self._init_optim_disc()

        self._generator_sampler = self._init_generator_sampler()
        self._discriminator_sampler = self._init_discriminator_sampler()

        self._criterion_gen = self._init_criterion_gen()
        self._criterion_disc = self._init_criterion_disc()

        self.num_fixed_samples: int = 8

        if self.cfg.latent_distribution != LatentDistribution.ENCODER_BASED:
            self.fixed_noise = self.generator_sampler.sample_z(self.num_fixed_samples)
        else:
            self.fixed_noise = torch.empty((self.num_fixed_samples, self.seq_len, self.latent_size)).to(self.device)
        # Tensors which can be filled during train/validation. Can be reset using self._reset_internal_tensors.
        self.reconstruction_error = torch.empty(0).to(self.device)
        self.discrimination_error = torch.empty(0).to(self.device)
        self.latent_vectors_train = torch.empty(0).to(self.device)
        self.latent_vectors_vali = torch.empty(0).to(self.device)
        self.label = torch.empty(0).to(self.device)

        # Required for validation/testing.
        self.svm: SVC = SVC()
        self.tau: float = 0.0
        self.lambda_: float = 0.0

    @abstractmethod
    def _init_generator(self) -> nn.Module:
        """Initialize the generator."""
        raise NotImplementedError("GANModule needs to implement the `_init_generator` method.")

    @property
    def generator(self) -> nn.Module:
        """Return the generator."""
        return self._generator

    @abstractmethod
    def _init_discriminator(self) -> nn.Module:
        """Initialize the discriminator."""
        raise NotImplementedError("GANModule needs to implement the `_init_discriminator` method.")

    @property
    def discriminator(self) -> nn.Module:
        """Return the generator."""
        return self._discriminator

    @abstractmethod
    def _init_generator_sampler(self) -> GeneratorSampler:
        """Initialize the sampler for the generator."""
        raise NotImplementedError("GANModule needs to implement the `_init_generator_sampler` method.")

    @property
    def generator_sampler(self) -> GeneratorSampler:
        """Return the sampler used to sample from the generator."""
        return self._generator_sampler

    @abstractmethod
    def _init_discriminator_sampler(self) -> DiscriminatorSampler:
        """Initialize the sampler for the discriminator."""
        raise NotImplementedError("GANModule needs to implement the `_init_discriminator_sampler` method.")

    @property
    def discriminator_sampler(self) -> DiscriminatorSampler:
        """Return the sampler used to sample from the discriminator."""
        return self._discriminator_sampler

    @abstractmethod
    def _init_optim_gen(self) -> BaseOptimizer:
        """Initialize the optimizer for the generator."""
        raise NotImplementedError("GANModule needs to implement the `_init_optim_gen` method.")

    @property
    def optim_gen(self) -> BaseOptimizer:
        """Return the optimizer for the generator."""
        return self._optim_gen

    @abstractmethod
    def _init_optim_disc(self) -> BaseOptimizer:
        """Initialize the optimizer for the discriminator."""
        raise NotImplementedError("GANModule needs to implement the `_init_optim_disc` method.")

    @property
    def optim_disc(self) -> BaseOptimizer:
        """Return the optimizer for the discriminator."""
        return self._optim_disc

    @abstractmethod
    def _init_criterion_gen(self) -> GANBaseLoss:
        """Initialize the criterion for the generator."""
        raise NotImplementedError("GANModule needs to implement the `_init_criterion_gen` method.")

    @property
    def criterion_gen(self) -> GANBaseLoss:
        """Return the criterion for the generator."""
        return self._criterion_gen

    @abstractmethod
    def _init_criterion_disc(self) -> GANBaseLoss:
        """Initialize the criterion for the discriminator."""
        raise NotImplementedError("GANModule needs to implement the `_init_criterion_disc` method.")

    @property
    def criterion_disc(self) -> GANBaseLoss:
        """Return the criterion for the discriminator."""
        return self._criterion_disc

    def _reset_internal_tensors(self):
        """Reset tensors which are filled internally during an epoch."""
        self.reconstruction_error = torch.empty(0).to(self.device)
        self.discrimination_error = torch.empty(0).to(self.device)
        self.label = torch.empty(0).to(self.device)
        self.latent_vectors_train = torch.empty(0).to(self.device)
        self.latent_vectors_vali = torch.empty(0).to(self.device)

    @staticmethod
    def _configure(
        latent_size: int,
        latent_space: LatentDistribution,
        disc_rounds: int,
        gen_rounds: int,
    ) -> Dict:
        return {
            'module': {
                'LATENT_SIZE': latent_size,
                'DISCRIMINATOR_ROUNDS': disc_rounds,
                'LATENT_SPACE': latent_space.value,
                'GENERATOR_ROUNDS': gen_rounds,
            }
        }

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration of a standard GAN."""
        return BaseGANModule._configure(
            latent_size=BaseGANModule.DEFAULT_LATENT_SIZE,
            latent_space=LatentDistribution.NORMAL,
            disc_rounds=BaseGANModule.DEFAULT_DISC_ROUNDS,
            gen_rounds=BaseGANModule.DEFAULT_GEN_ROUNDS,
        )

    def training_step(
        self,
        batch: dict,
    ) -> dict:
        """
        Declare what the model should do during a training step using a given batch.

        Args:
            batch: The batch of real data and labels. Labels are always 0 if trained on normal data only.

        Return:
            A dict containing the optimization metrics which shall be logged.
        """
        self.generator.train()
        self.discriminator.train()

        real_data = batch['data'].to(self.device)
        try:
            disc_metric_collection = []
            gen_metric_collection = []
            #########################################
            # Update discriminator
            #########################################
            for _ in range(self.cfg.DISCRIMINATOR_ROUNDS):
                # Retrieve losses and update gradients
                disc_losses, disc_metrics = self.criterion_disc(real_data)
                self.optim_disc.optimize(disc_losses)
                disc_metric_collection.extend(disc_metrics)

            #########################################
            # Update generator
            #########################################
            for _ in range(self.cfg.GENERATOR_ROUNDS):
                # Retrieve losses and update gradients internally
                gen_losses, gen_metrics = self.criterion_gen(real_data)
                self.optim_gen.optimize(gen_losses)
                gen_metric_collection.extend(gen_metrics)

            return {key: float(value) for (key, value) in disc_metric_collection + gen_metric_collection}

        except TypeError as err:
            raise TypeError('Error during training: Config parameter was not correctly set.') from err

    def validation_step(
        self,
        batch: dict,
    ) -> dict:
        """Declare what the model should do during a validation step."""
        return {}

    def save_checkpoint(self) -> dict:
        """Return current model parameters."""
        return {
            'GEN': self.generator.state_dict(),
            'DIS': self.discriminator.state_dict(),
            'GEN_OPT': self.optim_gen.state_dict(),
            'DIS_OPT': self.optim_disc.state_dict(),
            'ANOMALY_DETECTION': {
                'SVM': self.svm,
                'LAMBDA': self.lambda_,
                'TAU': self.tau,
            },
        }

    def load(self, model_reference: str, load_optim: bool = False):
        """Load a trained module from existing model_reference."""
        model = load_model(model_reference, self.device)

        self.generator.load_state_dict(model['GEN'], strict=False)
        self.discriminator.load_state_dict(model['DIS'], strict=False)

        if load_optim:
            self.optim_gen.load_existing_optim(model['GEN_OPT'])
            self.optim_disc.load_existing_optim(model['DIS_OPT'])
        logger.info('Loading existing {0} model completed.'.format(self.__class__.__name__))

        self.svm = model['ANOMALY_DETECTION']['SVM']
        self.tau = model['ANOMALY_DETECTION']['TAU']
        self.lambda_ = model['ANOMALY_DETECTION']['LAMBDA']

        return self

    @property
    def watch_list(self) -> List[nn.Module]:
        """Return models that should be watched during training."""
        return [self.generator, self.discriminator]

    def on_epoch_end(self, epoch: int, sample_interval: int, batch_size: int) -> List[Artifact]:
        """
        Set actions to be executed after epoch ends.

        Declare what should be done upon finishing an epoch (e.g. save artifacts or evaluate some metric).
        """
        result: List[Artifact] = []

        if not epoch % sample_interval == 0:
            return result
        ###################################################################
        # GENERATOR SAMPLES

        noise = torch.cat(
            [
                self.fixed_noise,
                self.generator_sampler.sample_z(len(self.fixed_noise)),
            ]
        )
        with no_grad():
            gen_samples = self.generator_sampler.sample(noise)

        train_cfg = get_global_config().trainer_config
        if train_cfg.transformation == Transformation.FOURIER:
            result.append(
                ImageArtifact(
                    'FFT Samples',
                    self.plotter.get_sampling_grid(
                        gen_samples,
                    ),
                )
            )
        else:
            if self.cfg.GENERATOR.TANH_OUT:
                gen_samples = (gen_samples / 2) + 0.5

            result.append(
                ImageArtifact(
                    'Generator Samples',
                    self.plotter.get_sampling_grid(
                        gen_samples,
                    ),
                )
            )

        mmd_score = self.get_mmd()
        result.append(ValueArtifact('MMD', mmd_score))

        tstr_dict = self.get_tstr()
        result.append(ValueArtifact('generative_metric/tstr', tstr_dict))

        return result

    def get_sample(
        self,
        num_samples: Optional[int] = None,
        data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a sample.

        Either based on random noise (requires the amount of samples) or original data
        if a reconstruction-based GAN is chosen.
        """
        if num_samples is None:
            raise RuntimeError("num_samples has to be set using the selected GAN module.")
        noise = self.generator_sampler.sample_z(num_samples)

        return self.generator_sampler.sample(noise), noise
