"""Implementation of various loss functions for PyTorch models."""

from abc import abstractmethod
from typing import Callable, Dict, Optional, Tuple, cast

import torch
from torch import Tensor, nn

from ecgan.config import LossConfig
from ecgan.utils.configurable import Configurable
from ecgan.utils.custom_types import Losses, LossMetricType, LossType
from ecgan.utils.sampler import (
    DiscriminatorSampler,
    EncoderBasedGeneratorSampler,
    FeatureDiscriminatorSampler,
    GeneratorSampler,
    VAEGANGeneratorSampler,
)


class SupervisedLoss(Configurable):
    """Base class for supervised loss functions."""

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        """Return the loss of a given component."""
        return self.loss(input_tensor, target_tensor)

    def __call__(self, input_tensor: Tensor, target_tensor: Tensor):
        """Call the forward method implicitly."""
        return self.forward(input_tensor, target_tensor)

    @property
    def loss(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._loss()

    @abstractmethod
    def _loss(self) -> Callable[[Tensor, Tensor], Tensor]:
        raise NotImplementedError("Loss needs to implement the `_loss` method.")

    @staticmethod
    def _configure(name: str) -> Dict:
        return {'LOSS': {'NAME': name, 'REDUCTION': 'mean'}}

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for a general loss function."""
        return SupervisedLoss._configure(Losses.UNDEFINED.value)


class L2Loss(SupervisedLoss):
    """Wrapper over the mean squared error loss of the torch module."""

    def __init__(self, reduction: str = 'mean') -> None:
        self._internal_loss = torch.nn.MSELoss(reduction=reduction)
        super().__init__()

    def _loss(self):
        return self._internal_loss

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for a general loss function."""
        return SupervisedLoss._configure(Losses.L2.value)


class BCELoss(SupervisedLoss):
    """Wrapper over the binary cross entropy loss of the torch module."""

    def __init__(self, reduction: str = 'mean') -> None:
        self._internal_loss = torch.nn.BCELoss(reduction=reduction)
        super().__init__()

    def _loss(self):
        return self._internal_loss

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for a general loss function."""
        return SupervisedLoss._configure(Losses.BCE.value)


class CrossEntropyLoss(SupervisedLoss):
    """Wrapper over the cross entropy loss of the torch module."""

    def __init__(self, reduction: str = 'mean') -> None:
        self._internal_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        super().__init__()

    def _loss(self):
        return self._internal_loss

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for a general loss function."""
        config = SupervisedLoss._configure(Losses.CROSS_ENTROPY.value)
        config['LOSS']['REDUCTION'] = 'none'
        return config


class KLLoss:
    """Kullback-Divergence loss for the usage in variational auto-encoders."""

    @staticmethod
    def forward(mean_value: Tensor, log_var: Tensor) -> Tensor:
        """
        Calculate Kullback-Leibler divergence for standard Gaussian distribution.

        Calculate KL divergence for a given expected value and log variance.
        The input tensors are expected to be in shape (N x DIM) where N is the number of
        samples and DIM is the dimension of the multivariate Gaussian. The result will be
        the average KL-Divergence of a batch of distributions and a unit Gaussian.
        """
        kl_div = -0.5 * torch.sum(1 + log_var - mean_value.pow(2) - log_var.exp())

        return kl_div / log_var.shape[0]

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the BCE loss."""
        return {'LOSS': {'NAME': Losses.KL_DIV}}


class GANBaseLoss(Configurable):
    """Base loss class for custom GAN losses."""

    def __init__(
        self,
        discriminator_sampler: DiscriminatorSampler,
        generator_sampler: GeneratorSampler,
    ) -> None:
        super().__init__()
        self.discriminator_sampler = discriminator_sampler
        self.generator_sampler = generator_sampler

    @abstractmethod
    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Return the loss of a given component."""
        raise NotImplementedError("GANLoss needs to implement the `forward` method.")

    def __call__(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Loss-specific forward will be applied upon call."""
        return self.forward(training_data)

    @staticmethod
    def _configure(name: str) -> Dict:
        return {'LOSS': {'NAME': name}}

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for a general loss function."""
        return GANBaseLoss._configure(Losses.UNDEFINED.value)


class BceGeneratorLoss(GANBaseLoss):
    """BCE Loss using the PyTorch implementation."""

    def __init__(
        self,
        discriminator_sampler: DiscriminatorSampler,
        generator_sampler: GeneratorSampler,
        reduction: str,
    ):
        super().__init__(discriminator_sampler, generator_sampler)
        self.loss = nn.BCELoss(reduction=reduction)

    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Calculate the BCE loss and update a given optimizer to minimize the loss."""
        batch_size = training_data.shape[0]
        noise = self.generator_sampler.sample_z(batch_size)
        gen_samples = self.generator_sampler.sample(noise)
        fake_labels = self.discriminator_sampler.sample_label_zeros(batch_size)
        disc_out_fake = self.discriminator_sampler.sample(gen_samples.detach())
        # For GANs: input: disc_outfake, target: target_fake
        batch_loss = self.loss(disc_out_fake, fake_labels)

        return batch_loss, [('loss_gen', float(batch_loss))]

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the BCE loss."""
        config = GANBaseLoss._configure(Losses.BCE_GENERATOR_LOSS.value)
        config['LOSS']['REDUCTION'] = 'mean'
        return config


class BceDiscriminatorLoss(BceGeneratorLoss):
    """
    Two component BCE Loss using the PyTorch implementation.

    The class assumes that corresponding `BaseSampler`s for each component are implemented. The fake data is sampled by
    the `sample` method of the provided generator. The BCE loss is commonly used when optimizing the discriminator
    of a vanilla GAN.
    """

    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Calculate the two component BCE loss and update a given optimizer to minimize the loss."""
        batch_size = training_data.shape[0]

        noise = self.generator_sampler.sample_z(batch_size)
        with torch.no_grad():
            fake_data = self.generator_sampler.sample(noise)

        disc_out_fake = self.discriminator_sampler.sample(fake_data)
        disc_out_real = self.discriminator_sampler.sample(training_data)

        target_fake = self.discriminator_sampler.sample_label_zeros(batch_size)
        target_real = self.discriminator_sampler.sample_label_ones(batch_size)
        # Train with all-real batch
        loss_disc_real = self.loss(disc_out_real, target_real)
        # Train with all-fake batch
        loss_disc_fake = self.loss(disc_out_fake, target_fake)

        return [('loss_disc_real', loss_disc_real), ('loss_disc_fake', loss_disc_fake)], [
            ('loss_disc_real', float(loss_disc_real)),
            ('loss_disc_fake', float(loss_disc_fake)),
        ]

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the two component BCE loss."""
        config = BceDiscriminatorLoss._configure(Losses.TWO_COMPONENT_BCE.value)
        config['LOSS']['REDUCTION'] = 'mean'
        return config


class WassersteinDiscriminatorLoss(GANBaseLoss):
    """Wasserstein loss for the discriminator."""

    def __init__(
        self,
        discriminator_sampler: DiscriminatorSampler,
        generator_sampler: GeneratorSampler,
        gradient_penalty_weight: Optional[float] = None,
        clipping_bound: Optional[float] = None,
    ):
        super().__init__(discriminator_sampler, generator_sampler)
        self.gradient_penalty_weight = gradient_penalty_weight
        self.clipping_bound = clipping_bound

    @staticmethod
    def apply_gp(input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        """GP penalty is applied outside the forward call during optimization."""
        return input_tensor.mean() - target_tensor.mean()

    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Calculate the Wasserstein distance and minimize it using a given optimizer."""
        batch_size = training_data.shape[0]

        noise = self.generator_sampler.sample_z(batch_size)
        with torch.no_grad():
            fake_data = self.generator_sampler.sample(noise)
        disc_out_fake = self.discriminator_sampler.sample(fake_data.detach())
        disc_out_real = self.discriminator_sampler.sample(training_data)

        loss = self.apply_gp(disc_out_fake, disc_out_real)

        if self.gradient_penalty_weight is not None and self.gradient_penalty_weight > 0:
            with torch.backends.cudnn.flags(enabled=False):
                gp_penalty = self.get_gradient_penalty(training_data, fake_data)
            loss += self.gradient_penalty_weight * gp_penalty

        if (
            self.clipping_bound is not None
            and self.clipping_bound > 0
            and self.discriminator_sampler.component is not None
        ):
            for param in self.discriminator_sampler.component.parameters():
                param.data.clamp_(
                    min=-self.clipping_bound,  # pylint: disable=E1130
                    max=self.clipping_bound,
                )

        return loss, [('loss_disc', float(loss))]

    def get_gradient_penalty(self, real_data: Tensor, generated_data: Tensor) -> Tensor:
        """Based on https://github.com/EmilienDupont/wgan-gp/blob/master/training.py."""
        # Interpolate
        batch_size = real_data.size()[0]
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand_as(real_data)
        if real_data.is_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated.requires_grad_(True)
        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator_sampler.sample(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).cuda()
            if generated_data.is_cuda
            else torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, seq_len, num_channels),
        # so flatten to easily take norm per example in batch
        gradients = gradients.reshape(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the Wasserstein discriminator loss."""
        config = GANBaseLoss._configure(Losses.WASSERSTEIN_DISCRIMINATOR.value)
        config['LOSS']['GRADIENT_PENALTY_WEIGHT'] = 10
        config['LOSS']['CLIPPING_BOUND'] = None
        return config


class WassersteinGeneratorLoss(GANBaseLoss):
    """Wasserstein loss for the discriminator."""

    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Calculate the Wasserstein distance and minimize it using a given optimizer."""
        batch_size = training_data.shape[0]
        noise = self.generator_sampler.sample_z(batch_size)
        fake_data = self.generator_sampler.sample(noise)
        disc_out_fake = self.discriminator_sampler.sample(fake_data)
        batch_loss = -disc_out_fake.mean()
        return batch_loss, [('loss_gen', float(batch_loss))]

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the Wasserstein generator loss."""
        return GANBaseLoss._configure(Losses.WASSERSTEIN_GENERATOR.value)


class AEGANGeneratorLoss(GANBaseLoss):
    """Loss function for the auto-encoder based GANs."""

    def __init__(
        self,
        discriminator_sampler: FeatureDiscriminatorSampler,
        generator_sampler: EncoderBasedGeneratorSampler,
    ) -> None:
        super().__init__(discriminator_sampler, generator_sampler)
        self._internal_loss = torch.nn.MSELoss()

    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Perform the forward for the AEGAN generator loss."""
        faked_output, noise = self.generator_sampler.sample_generator_encoder(training_data)  # type: ignore

        feat_fake = cast(FeatureDiscriminatorSampler, self.discriminator_sampler).sample_features(faked_output)
        feat_real = cast(FeatureDiscriminatorSampler, self.discriminator_sampler).sample_features(training_data)

        adversarial_error = self._internal_loss(feat_fake, feat_real)
        reconstruction_error = self._internal_loss(faked_output, training_data)
        weighted_error = adversarial_error + reconstruction_error

        return weighted_error, [
            ('adversarial_loss', float(adversarial_error)),
            ('reconstruction_loss', float(reconstruction_error)),
            ('weighted_loss', float(weighted_error)),
            ('noise', noise.detach()),
        ]

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the AEGAN generator loss."""
        return GANBaseLoss._configure(Losses.AEGAN_GENERATOR.value)


class AEGANDiscriminatorLoss(GANBaseLoss):
    """Discriminator loss for a AEGAN module."""

    def __init__(
        self,
        discriminator_sampler: DiscriminatorSampler,
        generator_sampler: GeneratorSampler,
    ) -> None:
        super().__init__(discriminator_sampler, generator_sampler)
        self._internal_loss = torch.nn.BCELoss()

    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Perform a forward pass for the loss."""
        batch_size = training_data.shape[0]

        fake_labels = self.discriminator_sampler.sample_label_zeros(batch_size)
        real_labels = self.discriminator_sampler.sample_label_ones(batch_size)

        real_output = self.discriminator_sampler.sample(training_data)

        noise = cast(EncoderBasedGeneratorSampler, self.generator_sampler).sample_encoder(training_data)
        faked_output = self.generator_sampler.sample(noise)
        faked_output = self.discriminator_sampler.sample(faked_output)

        real_error = self._internal_loss(real_output, real_labels)
        fake_error = self._internal_loss(faked_output, fake_labels)

        disc_error: Tensor = real_error + fake_error

        return disc_error, [
            ('loss_disc_real', float(real_error)),
            ('loss_disc_fake', float(fake_error)),
        ]

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the AEGAN discriminator loss."""
        return GANBaseLoss._configure(Losses.AEGAN_DISCRIMINATOR.value)


class VAEGANGeneratorLoss(GANBaseLoss):
    """Generator loss for the VAEGAN module."""

    def __init__(
        self,
        discriminator_sampler: FeatureDiscriminatorSampler,
        generator_sampler: VAEGANGeneratorSampler,
        reconstruction_loss: SupervisedLoss,
        distribution: torch.distributions.Distribution,
        kl_beta: float,
        device,
    ) -> None:
        super().__init__(discriminator_sampler, generator_sampler)
        self._reconstruction_loss = reconstruction_loss
        self._kl_loss = KLLoss()
        self._adversarial_loss = L2Loss()
        self.kl_beta = kl_beta
        self._kl_weight = 1.0

        self.generator_sampler = cast(VAEGANGeneratorSampler, self.generator_sampler)
        self.discriminator_sampler = cast(FeatureDiscriminatorSampler, self.discriminator_sampler)
        self.distribution = distribution
        self.device = device

    @property
    def kl_weight(self) -> float:
        return self._kl_weight

    @kl_weight.setter
    def kl_weight(self, new_kl_weight: float) -> None:
        self._kl_weight = new_kl_weight

    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Perform a forward pass for the loss."""
        batch_size, num_channel, seq_len = training_data.shape
        rec_loss_scale_factor = 1 / (batch_size * num_channel * seq_len)

        mu, log_var = cast(VAEGANGeneratorSampler, self.generator_sampler).sample_mu_logvar(training_data)
        faked_output, noise = cast(VAEGANGeneratorSampler, self.generator_sampler).sample_pre_computed(mu, log_var)

        feat_fake = cast(FeatureDiscriminatorSampler, self.discriminator_sampler).sample_features(faked_output)
        feat_real = cast(FeatureDiscriminatorSampler, self.discriminator_sampler).sample_features(training_data)

        adversarial_error = self._adversarial_loss(feat_fake, feat_real)

        reconstruction_error = self._reconstruction_loss(faked_output, training_data) * rec_loss_scale_factor

        kl_loss: Tensor = self._kl_loss.forward(mu, log_var) * self.kl_beta * self.kl_weight
        weighted_error = kl_loss + adversarial_error + reconstruction_error

        return weighted_error, [
            ('gen_kl_loss', float(kl_loss)),
            ('gen_adversarial_loss', float(adversarial_error)),
            ('gen_reconstruction_loss', float(reconstruction_error)),
            ('latent/z_mu', mu.detach()),
            ('latent/noise', noise.detach()),
            ('latent/z_std_avg', float(torch.mean(torch.exp(0.5 * log_var)))),
            ('gen_weighted_loss', float(weighted_error)),
        ]

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the VAEGAN generator loss."""
        return GANBaseLoss._configure(Losses.VAEGAN_GENERATOR.value)


class SupervisedLossFactory:
    """Meta module for creating correct loss functions."""

    def __call__(self, params: LossConfig) -> SupervisedLoss:
        """Return implemented module when a Loss object is created."""
        loss = Losses(params.NAME)
        reduction = params.REDUCTION or 'mean'

        if loss == Losses.L2:
            return L2Loss(reduction=reduction)

        if loss == Losses.BCE:
            return BCELoss(reduction=reduction)

        if loss == Losses.CROSS_ENTROPY:
            return CrossEntropyLoss(reduction=reduction)

        raise AttributeError('Argument {0} is not set correctly.'.format(loss))


class GANLossFactory:
    """Meta module for creating correct GAN loss functions."""

    def __call__(
        self,
        params: LossConfig,
        discriminator_sampler: DiscriminatorSampler,
        generator_sampler: GeneratorSampler,
    ) -> GANBaseLoss:
        """Return implemented module when a Loss object is created."""
        loss = Losses(params.NAME)
        reduction = params.REDUCTION or 'mean'
        if loss == Losses.BCE_GENERATOR_LOSS:
            return BceGeneratorLoss(discriminator_sampler, generator_sampler, reduction=reduction)
        if loss == Losses.TWO_COMPONENT_BCE:
            return BceDiscriminatorLoss(discriminator_sampler, generator_sampler, reduction=reduction)
        if loss == Losses.WASSERSTEIN_DISCRIMINATOR:
            return WassersteinDiscriminatorLoss(
                discriminator_sampler=discriminator_sampler,
                generator_sampler=generator_sampler,
                gradient_penalty_weight=params.GRADIENT_PENALTY_WEIGHT,
                clipping_bound=params.CLIPPING_BOUND,
            )
        if loss == Losses.WASSERSTEIN_GENERATOR:
            return WassersteinGeneratorLoss(discriminator_sampler, generator_sampler)

        raise AttributeError('Argument {0} is not set correctly.'.format(loss))


class AutoEncoderLoss(Configurable):
    """Reconstruction based loss used by autoencoders."""

    def __init__(self, autoencoder_sampler: EncoderBasedGeneratorSampler, use_mse: bool) -> None:
        super().__init__()
        self.autoencoder_sampler = autoencoder_sampler
        self._internal_loss = torch.nn.MSELoss() if use_mse else torch.nn.BCELoss()

    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Perform a forward pass for the loss."""
        batch_size = training_data.shape[0]
        torch.zeros(batch_size, device=training_data.device)

        noise = self.autoencoder_sampler.sample_encoder(training_data)
        faked_output = self.autoencoder_sampler.sample(noise)

        error = self._internal_loss(faked_output, training_data)

        return error, [
            ('loss_mse', float(error)),
        ]

    def __call__(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Loss-specific forward will be applied upon call."""
        return self.forward(training_data)

    @staticmethod
    def _configure(name: str) -> Dict:
        return {'LOSS': {'NAME': name}}

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the autoencoder loss function."""
        return AutoEncoderLoss._configure(Losses.AUTOENCODER.value)


class VariationalAutoEncoderLoss(Configurable):
    r"""Base loss of VAEs considering reconstruction loss as well as KL divergence weighted by :math:`\beta`."""

    def __init__(
        self,
        autoencoder_sampler: GeneratorSampler,
        use_mse: bool,
        kl_beta: float,
        distribution: torch.distributions.Distribution,
        device,
    ) -> None:
        super().__init__()
        self._internal_loss = torch.nn.MSELoss() if use_mse else torch.nn.BCELoss()
        self._kl_loss = KLLoss()
        self.kl_beta = kl_beta
        self._kl_weight = 1.0

        self.generator_sampler = cast(VAEGANGeneratorSampler, autoencoder_sampler)
        self.distribution = distribution
        self.device = device

    def __call__(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Loss-specific forward will be applied upon call."""
        return self.forward(training_data)

    @property
    def kl_weight(self) -> float:
        return self._kl_weight

    @kl_weight.setter
    def kl_weight(self, new_kl_weight: float) -> None:
        self._kl_weight = new_kl_weight

    def forward(self, training_data: Tensor) -> Tuple[LossType, LossMetricType]:
        """Perform a forward pass for the loss."""
        batch_size, num_channel, seq_len = training_data.shape
        rec_loss_scale_factor = 1 / (batch_size * num_channel * seq_len)

        mu, log_var = cast(VAEGANGeneratorSampler, self.generator_sampler).sample_mu_logvar(training_data)
        faked_output, _noise = cast(VAEGANGeneratorSampler, self.generator_sampler).sample_pre_computed(mu, log_var)

        reconstruction_error = self._internal_loss(faked_output, training_data) * rec_loss_scale_factor

        kl_loss: Tensor = self._kl_loss.forward(mu, log_var) * self.kl_beta * self.kl_weight
        weighted_error = kl_loss + reconstruction_error

        return weighted_error, [
            ('gen_kl_loss', float(kl_loss)),
            ('gen_reconstruction_loss', float(reconstruction_error)),
            ('latent/z_std_avg', float(torch.mean(torch.exp(0.5 * log_var)))),
            ('gen_weighted_loss', float(weighted_error)),
        ]

    @staticmethod
    def _configure(name: str) -> Dict:
        return {'LOSS': {'NAME': name}}

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the VAE loss."""
        return VariationalAutoEncoderLoss._configure(Losses.VAE.value)
