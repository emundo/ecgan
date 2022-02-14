"""Functions related to the reconstruction of series."""
import timeit
from abc import ABC, abstractmethod
from logging import getLogger
from typing import List, Optional, cast

import torch.nn
from torch import Tensor, cat, empty, mean, tensor

from ecgan.config import (
    InverseDetectorConfig,
    LatentWalkReconstructionConfig,
    OptimizerConfig,
    ReconstructionConfig,
    get_global_ad_config,
    get_global_inv_config_attribs,
    get_inv_run_config,
    get_model_path,
    set_global_inv_config,
)
from ecgan.config.initialization.inverse import init_inverse
from ecgan.evaluation.tracker import BaseTracker
from ecgan.modules.generative.base import BaseEncoderGANModule, BaseGANModule
from ecgan.modules.inverse_mapping.inverse_mapping import InvertibleBaseModule
from ecgan.modules.inverse_mapping.inversion import inverse_train
from ecgan.modules.inverse_mapping.vanilla_inverse_mapping import SimpleGANInverseMapping
from ecgan.utils.custom_types import ReconstructionType
from ecgan.utils.optimizer import BaseOptimizer, OptimizerFactory
from ecgan.utils.reconstruction_criteria import get_reconstruction_criterion
from ecgan.utils.sampler import EncoderBasedGeneratorSampler

logger = getLogger(__name__)


class Reconstructor(ABC):
    """Base class for different reconstruction strategies."""

    def __init__(self, reconstruction_cfg: ReconstructionConfig, module: BaseGANModule, **_kwargs):
        self.module = module
        self.reconstruction_cfg = reconstruction_cfg
        self.time_passed = empty(0)

    @abstractmethod
    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Reconstruct the latent representation of a given Tensor.

        Args:
            x: A single data sample which is to be reconstructed.

        Returns:
           Reconstructed series.
        """
        raise NotImplementedError("Reconstructor needs to implement the `reconstruct` method.")


class InterpolationReconstructor(Reconstructor):
    """
    Reconstruct samples based on the AnoGAN approach (`Schlegl et al. 2017 <https://arxiv.org/pdf/1703.05921.pdf>`_).

    Optimize through latent space to search for a series similar to the input series.

    Args:
        module: Generative module used for interpolation.
        reconstruction_cfg: Config containing relevant parameters for latent walk.
    """

    def __init__(self, module: BaseGANModule, reconstruction_cfg: LatentWalkReconstructionConfig, **_kwargs):
        super().__init__(reconstruction_cfg, module)
        self.rec_cfg = reconstruction_cfg
        self.criterion = get_reconstruction_criterion(reconstruction_cfg.CRITERION)
        self.adapt_lr = self.rec_cfg.ADAPT_LR

        verbose_steps = reconstruction_cfg.VERBOSE_STEPS
        self.verbose_steps = (
            verbose_steps if verbose_steps is not None else reconstruction_cfg.MAX_RECONSTRUCTION_ITERATIONS // 10
        )

        self.z_sequence = empty(0)
        self.series_samples = empty(0)
        self.losses: List[float] = []
        self.z_sequences = empty(0)  # Contains the first z_sequence of each batch. Used for visualization later on.
        self.total_z_distance = empty(0)

    def reconstruct(self, x: Tensor) -> Tensor:
        r"""
        Reconstruct the latent representation of a given Tensor.

        Procedure:

            #. Randomly sample data :math:`z_0` from the latent space of the model.
            #. Create a synthetic series :math:`G(z_0)`.
            #. Compare the similarity :math:`sim(x, G(z_0))`.
            #. Optimize through latent space to find :math:`z_1` which generates a series
               :math:`G(z_1)` which is more similar to :math:`x` than :math:`G(z_0)`.
            #. Repeat until :math:`G(z_i)` is similar enough, defined by:
               :math:`dissimilarity(x, G(z_i)) = 1-sim(x, G(z_i)) \leq \epsilon`.

        Args:
            x: The input data in shape (N x +) that shall be reconstructed.

        Returns:
           Reconstructed series.
        """
        start = timeit.default_timer()
        batch_size = x.shape[0]
        if isinstance(self.module, BaseEncoderGANModule):
            # Use for inverse mapping + latent optimization
            z_sample = self.module.generator_sampler.sample_encoder(x).detach()
            # Use for latent optimization only
            # z_sample: Tensor = torch.distributions.normal.Normal(0,1).sample((batch_size, 1, self.module.latent_size))
        else:
            z_sample = self.module.generator_sampler.sample_z(batch_size).clone()

        # Enables usage for RNN/CNN generator.
        if z_sample.shape[1] > 1:
            # If the sampler is an RNN generator sampler: latent space should only
            # have one latent space sample per sample and not per step.
            # Avoid optimizing (batch_size, seq_len, latent_space)
            # but optimizes (batch_size, 1, latent_space) instead.
            z_sample = z_sample[:, 0, :].unsqueeze(1).clone()
            z_sample.requires_grad = True
            z_sequence: Tensor = z_sample.expand(-1, self.module.seq_len, -1)
        else:
            # If the sampler is a CNN generator sampler: latent space will by default
            # only be of shape (1,1,latent_space)
            z_sample.requires_grad = True
            z_sequence = z_sample

        series_samples = x[0].unsqueeze(0)

        optimizer = OptimizerFactory()(
            [z_sample],
            OptimizerConfig(
                NAME=self.rec_cfg.LATENT_OPTIMIZER.NAME,
                LR=self.rec_cfg.LATENT_OPTIMIZER.LR,
            ),
        )

        reached_target = False
        losses = []
        z_sequences = z_sequence[0].unsqueeze(0).detach()
        loss: Tensor = tensor([float('inf')])
        reconstructed_series = empty(x.shape)
        total_z_distance = torch.zeros(batch_size).to(self.module.device)
        previous_z_tensor = z_sequence.clone().detach().view(batch_size, -1)
        for iteration in range(self.rec_cfg.MAX_RECONSTRUCTION_ITERATIONS):
            optimizer.zero_grad()

            reconstructed_series = self.module.generator_sampler.sample(z_sequence)
            loss = mean(self.criterion(reconstructed_series, x))
            # Future improvement: make sure that the gradient doesnt lead into area with high norm
            # Adapt LR -> Maybe longer latent space walk but better results
            self.adapt_learning_rate(float(loss), optimizer)

            if float(loss) < self.rec_cfg.EPSILON:
                logger.info(
                    'Loss has reached target epsilon of {0} in iteration {1}.'.format(self.rec_cfg.EPSILON, iteration)
                )
                reached_target = True
                break

            loss.backward()
            optimizer.step()
            current_z_sequence = z_sequence.clone().detach().view(batch_size, -1)

            # Add distance of current iteration to total distance
            total_z_distance = total_z_distance + torch.nn.PairwiseDistance()(previous_z_tensor, current_z_sequence).to(
                self.module.device
            )
            previous_z_tensor = current_z_sequence

            if iteration % self.verbose_steps == 0:
                logger.info('Iteration {0:4} | Loss: {1:2.3f}'.format(iteration, float(loss)))

                losses.append(float(loss))
                z_sequences = cat((z_sequences, z_sequence[0].unsqueeze(0).detach()))

                series_samples = cat((series_samples, reconstructed_series[0].unsqueeze(0).detach()))

        if not reached_target:
            logger.warning('Could not match epsilon quality criterion. Loss is {0:2.3f}'.format(float(loss)))

        self.series_samples = cat((series_samples, reconstructed_series[0].unsqueeze(0).detach()))
        self.z_sequences = cat((z_sequences, z_sequence[0].unsqueeze(0).detach()))
        self.z_sequence = z_sequence.detach()
        self.losses = losses
        self.total_z_distance = total_z_distance.detach().to(self.module.device)

        end = timeit.default_timer()
        avg_time = [(end - start) / batch_size] * batch_size
        self.time_passed = cat((self.time_passed, torch.tensor(avg_time)), 0)

        return reconstructed_series.detach()

    def adapt_learning_rate(self, loss: float, optimizer: BaseOptimizer) -> None:
        """Adapt the learning rate if the loss is below a previously set learning rate threshold."""
        if not (float(loss) < self.rec_cfg.LR_THRESHOLD and self.adapt_lr):
            return

        logger.info(
            "Error below threshold of {0}. Adapting LR, new LR at {1}.".format(
                self.rec_cfg.LR_THRESHOLD,
                self.rec_cfg.LATENT_OPTIMIZER.LR * 10 ** (-1),
            )
        )
        optimizer.set_param_group(self.rec_cfg.LATENT_OPTIMIZER.LR * 10 ** (-1))
        self.adapt_lr = False


class InverseMappingReconstructor(Reconstructor):
    """
    Reconstruct the samples based on the ALAD approach by `Zenati et al. 2018 <https://arxiv.org/abs/1802.06222>`_.

    Learn an inverse mapping from the data space to the latent space to avoid the costly interpolation of AnoGAN.
    The mapping cans be learned during training (e.g. using an autoencoder based GAN or CycleGAN) or after training.
    If the mapping is not learned during training (i.e. if the module is not an instance of
    :class:`ecgan.modules.generative.base.BaseEncoderGANModule`), we train this module during initialization
    of the inverse mapping reconstructor. This can take quite some time.
    """

    def __init__(self, module: BaseGANModule, reconstruction_cfg: ReconstructionConfig, **kwargs):
        tracker: Optional[BaseTracker] = kwargs.get('tracker', None)
        ad_cfg = get_global_ad_config()
        if not isinstance(ad_cfg.detection_config, InverseDetectorConfig):
            raise RuntimeError(
                "An InverseDetectorConfig has to be supplied if inverse mapping is selected. "
                "Current config: {0}".format(type(ad_cfg.detection_config))
            )
        detection_cfg: InverseDetectorConfig = cast(InverseDetectorConfig, ad_cfg.detection_config)
        super().__init__(reconstruction_cfg, module)
        if isinstance(self.module, BaseEncoderGANModule):
            self._inverse_mapping = self.module.encoder
        else:
            model_path = get_model_path(
                ad_cfg.ad_experiment_config.RUN_URI,
                ad_cfg.ad_experiment_config.RUN_VERSION,
            )
            if detection_cfg.INVERSE_MAPPING_URI is None:
                init_inverse(model_path, filename='inverse_config.yml')
                inv_module: InvertibleBaseModule = inverse_train(tracker=tracker)
                self._inverse_mapping = inv_module.inv
            else:
                set_global_inv_config(get_inv_run_config(ad_cfg).config_dict)
                inv_config = get_global_inv_config_attribs()

                if inv_config.RUN_URI != ad_cfg.ad_experiment_config.RUN_URI:
                    raise RuntimeError(
                        "Supplied URI of inverse mapping module ({0}) differs from "
                        "anomaly detection module URI ({1}).".format(
                            inv_config.RUN_URI, ad_cfg.ad_experiment_config.RUN_URI
                        )
                    )

                inverse_module = SimpleGANInverseMapping(
                    inv_cfg=inv_config,
                    module_cfg=self.module.cfg,
                    run_path=model_path,
                    seq_len=self.module.seq_len,
                    num_channels=self.module.num_channels,
                    tracker=tracker,
                )

                inverse_module.load(detection_cfg.INVERSE_MAPPING_URI)
                self._inverse_mapping = inverse_module.inv
        self.noise = empty(0)

    def reconstruct(self, x: Tensor) -> Tensor:
        r"""
        Reconstruct the latent representation of a given Tensor.

        Procedure:
            #. Query inverse mapping for latent representation of :math:`z_0`.
            #. Create a synthetic series :math:`G(z_0)`.

        Args:
            x: The input data in shape :math:`(N \times *)` that shall be reconstructed.

        Returns:
           Reconstructed series.
        """
        self._inverse_mapping.eval()
        self.module.discriminator.eval()
        self.module.generator.eval()
        sampler = cast(EncoderBasedGeneratorSampler, self.module.generator_sampler)

        with torch.no_grad():
            start = timeit.default_timer()
            x_hat, noise = sampler.sample_generator_encoder(data=x)
            end = timeit.default_timer()
            avg_time = [(end - start) / x_hat.shape[0]] * x_hat.shape[0]
            self.time_passed = cat((self.time_passed, torch.tensor(avg_time)), 0)

        self.noise = noise

        return x_hat


class ReconstructorFactory:
    """Factory module for creating :class:`ecgan.anomaly_detection.reconstruction.Reconstructor` objects."""

    def __call__(
        self, reconstructor: ReconstructionType, reconstruction_cfg: ReconstructionConfig, **kwargs
    ) -> Reconstructor:
        """Return a :class:`ecgan.anomaly_detection.reconstruction.Reconstructor` object."""
        if reconstructor == ReconstructionType.INTERPOLATE:
            reconstruction_cfg = cast(LatentWalkReconstructionConfig, reconstruction_cfg)
            return InterpolationReconstructor(reconstruction_cfg=reconstruction_cfg, **kwargs)
        if reconstructor == ReconstructionType.INVERSE_MAPPING:
            return InverseMappingReconstructor(reconstruction_cfg=reconstruction_cfg, **kwargs)

        raise ValueError(
            'Unknown reconstruction type: {0}. Please select a valid ReconstructionType.'.format(reconstructor)
        )
