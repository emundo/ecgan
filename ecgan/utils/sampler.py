"""
Implementation of a read-only sampling interface for module resources.

A sampler provides access to a module's resources such as its latent space, generator
and discriminator. The usage is exemplified in the implementation of loss functions where
samplers are used to compute the necessary gradients.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, cast

import torch
from torch import Tensor, nn, ones, zeros
from torch._C import device

from ecgan.training.datasets import BaseDataset, SeriesDataset
from ecgan.utils.custom_types import LatentDistribution
from ecgan.utils.distributions import TruncatedNormal


class BaseSampler(ABC):
    """
    Abstract sampler that provides read-only access to a component.

    Examples are modules such as the generator or discriminator, or data and labels.
    """

    def __init__(
        self,
        component: Optional[nn.Module],
        dev: device,
        num_channels: int,
        sampling_seq_length: int,
    ):
        self.device = dev
        self.component = component
        self.num_channels = num_channels
        self.sampling_seq_length = sampling_seq_length

    @abstractmethod
    def sample(self, data):
        """Sample from the component."""
        raise NotImplementedError("Sampler needs to implement the `sample` method.")

    def sample_label_ones(self, sampling_size: int) -> Tensor:
        """
        Return a tensor filled with ones based on the sampling size.

        RNN-based GANs can rely on a label for each timestep in every series instead of
        only one label per series. If an RNN is used, the sampling_seq_length should be
        set to the sequence length, 1 otherwise during initialization.
        """
        return ones(sampling_size * self.sampling_seq_length, device=self.device)

    def sample_label_zeros(self, sampling_size: int) -> Tensor:
        """
        Return a tensor filled with zeros based on the sampling size.

        RNN-based GANs can rely on a label for each timestep in every series instead of
        only one label per series. If an RNN is used, the sampling_seq_length should be
        set to the sequence length, 1 otherwise during initialization.
        """
        return zeros(sampling_size * self.sampling_seq_length, device=self.device)


class DataSampler(BaseSampler):
    """Sampler for a (PyTorch) dataset."""

    def __init__(
        self,
        component: Optional[nn.Module],
        dev: device,
        num_channels: int,
        seq_length: int,
        name: str,
        dataset: Optional[BaseDataset] = None,
    ):
        super().__init__(component, dev, num_channels, seq_length)

        self.name = name
        self.dataset: Optional[BaseDataset] = dataset

    def sample(self, data: int) -> Dict:
        """
        Sample `data` amount of data from the provided dataset.

        Args:
            data: Amount of data to draw from the dataset.

        Returns:
            Dict containing the sample values and labels.
        """
        if self.dataset is None:
            raise ValueError(
                'Attribute "{0}" dataset is None. '
                'Set it via the constructor or the `set_datasets` function.'.format(self.name)
            )
        return self.dataset.sample(data)

    def sample_class(self, num_samples: int, class_label: int) -> Dict:
        """
        Sample `num_samples` amount of samples belonging to a given class from the dataset.

        Args:
            num_samples: Amount of data to draw from the dataset.
            class_label: Class label.

        Returns:
            Dict containing the sampled values and (non-zero) labels.
        """
        if not isinstance(self.dataset, SeriesDataset):
            raise ValueError(
                'Attribute "{0}" dataset is None. '
                'Set it via the constructor or the `set_datasets` function.'.format(self.name)
            )

        class_data: Tensor = self.dataset.data[self.dataset.label == class_label]
        class_labels: Tensor = self.dataset.label[self.dataset.label == class_label]
        idx_permuted_selection: Tensor = torch.randperm(len(class_data))[:num_samples]

        anomalies: Dict = {
            'data': class_data[idx_permuted_selection],
            'label': class_labels[idx_permuted_selection],
        }

        return anomalies

    def set_dataset(self, dataset: BaseDataset) -> None:
        """Change or set datasets to sample from."""
        self.dataset = dataset

    def get_dataset_size(self, class_label: Optional[int] = None) -> int:
        """
        Retrieve the amount of data in the dataset or of a specific class.

        Args:
            class_label: Optional class label.

        Returns:
            Amount of samples in whole dataset if no class label is given, amount of samples in class otherwise.
        """
        if class_label is None and self.dataset is not None:
            return self.dataset.__len__()

        if not isinstance(self.dataset, SeriesDataset):
            raise ValueError(
                'Attribute "{0}" dataset is None. '
                'Set it via the constructor or the `set_datasets` function.'.format(self.name)
            )
        return len(self.dataset.label[self.dataset.label == class_label])


class DiscriminatorSampler(BaseSampler):
    """Sampler for a classification model (e.g. a GAN discriminator) to retrieve the classification scores."""

    def __init__(
        self,
        component: nn.Module,
        dev: device,
        num_channels: int,
        sampling_seq_length: int,
    ):
        super().__init__(component, dev, num_channels, sampling_seq_length)

    def sample(self, data: Tensor) -> Tensor:
        """
        Sample the classifier.

        Note that a gradient for the component is computed.
        You can wrap the method in a `torch.no_grad()` block in
        order to stop the computation of the gradient.

        Args:
            data: Input tensor that shall be judged by the discriminator.

        Returns:
            Probability scores for the data being real.
        """
        return self.component(data)  # type: ignore


class GeneratorSampler(BaseSampler):
    """
    Sampler for a generator.

    Can be used to either sample noise from the latent space provided during
    initialization, or to generate data based on a noise sample.
    """

    def __init__(
        self,
        component: nn.Module,
        dev: device,
        num_channels: int,
        sampling_seq_length: int,
        latent_space: Optional[torch.distributions.Distribution],
        latent_size: int,
    ):
        super().__init__(component, dev, num_channels, sampling_seq_length)
        self.latent_space = latent_space
        self.latent_size = latent_size

    def sample(self, data: Tensor) -> Tensor:
        """
        Sample the generator to synthesize data space of the training data.

        The resulting data is a time series with a predefined sequence length
        and a specified amount of channels.
        Note that that a gradient for the component is computed. You can wrap the
        method in a `torch.no_grad()` block in order to stop the computation of the
        gradient.

        Args:
            data: Input noise for the generator.

        Returns:
            Samples in the training data space.
        """
        return self.component(data)  # type: ignore

    def sample_z(self, sample_amount: int) -> Tensor:
        """
        Draw a sample from from the latent space.

        Sample n vectors of noise. The dimensionality of the noise should be set during
        initialization. The sequence length is set to 1 (contrary to what some
        LSTM-based GANs do). The noise is expanded if the sampling_seq_length is larger.
        """
        if self.latent_size is None:
            raise ValueError("Sampling of latent space is None. Operation requires setting a valid distribution.")
        self.latent_space = cast(torch.distributions.Distribution, self.latent_space)
        sampled_z: Tensor = self.latent_space.sample((sample_amount, 1, self.latent_size))
        if str(self.device) != 'cpu':
            sampled_z = sampled_z.cuda()

        if self.sampling_seq_length > 1:
            sampled_z = sampled_z.expand(-1, self.sampling_seq_length, -1)

        return sampled_z


class FeatureDiscriminatorSampler(DiscriminatorSampler):
    """Sampler for a discriminator model which returns a discrimination score and features."""

    def sample(self, data: Tensor) -> Tensor:
        """
        Sample the classifier.

        Note that a gradient for the component is computed.
        You can wrap the method in a `torch.no_grad()` block in
        order to stop the computation of the gradient.

        Args:
            data: Input tensor that shall be judged by the discriminator.

        Returns:
            Probability scores for the data being real.
        """
        score, _ = self.sample_score_features(data)
        return score

    def sample_features(self, data) -> Tensor:
        """
        Sample the model for the features.

        Args:
            data:

        Args:
            data: Input tensor that shall be judged by the discriminator.

        Returns:
            Returns the features as returned by the model.
        """
        _, features = self.sample_score_features(data)

        return features

    def sample_score_features(self, data) -> Tuple[Tensor, Tensor]:
        """
        Sample the model for the score and the features.

        Args:
            data: Input tensor that shall be judged by the discriminator.

        Returns:
            Returns the score and the features as returned by the model.
        """
        score, features = self.component(data)  # type: ignore

        return score, features


class EncoderBasedGeneratorSampler(GeneratorSampler):
    """
    Generator sampler for encoder based modules.

    Since these modules do not have a traditional latent distribution, calling `sample_z` will result in a
    `NotImplementedError`.
    """

    def __init__(
        self,
        component: nn.Module,
        encoder: nn.Module,
        dev: device,
        num_channels: int,
        sampling_seq_length: int,
    ):
        super().__init__(component, dev, num_channels, sampling_seq_length, None, -1)
        self.encoder = encoder

    def sample_encoder(self, data: Tensor) -> Tensor:
        """Sample the encoder of the module."""
        return self.encoder(data)  # type: ignore

    def sample_generator_encoder(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the result of the encoder and the generator."""
        latent_vector = self.sample_encoder(data)
        return self.component(latent_vector), latent_vector  # type: ignore

    def sample_z(self, sample_amount: int):
        """Raise `NotImplementedError` since the sampler has no latent distribution."""
        raise NotImplementedError("Sampling of latent space is not supported for encoder based models.")


class VAEGANGeneratorSampler(EncoderBasedGeneratorSampler):
    """Generator sampler for the VAEGAN module."""

    def __init__(
        self,
        component: nn.Module,
        encoder: nn.Module,
        dev: device,
        num_channels: int,
        sampling_seq_length: int,
        distribution: torch.distributions.Distribution,
    ):
        super().__init__(component, encoder, dev, num_channels, sampling_seq_length)
        self.distribution = distribution

    def sample_mu_logvar(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample mu and log variance for a given sample from VAEGAN encoder."""
        return self.encoder(data)  # type: ignore

    def sample_encoder(self, data: Tensor) -> Tensor:
        """
        Return the noise of the encoder for the VAEGAN module.

        Args:
            data: Training data.

        Returns:
            Learned mean value mu with reparameterized std.
        """
        mu, log_var = self.sample_mu_logvar(data)
        std = torch.exp(0.5 * log_var)
        eps = self.sample_eps(std.shape)
        noise = mu + eps * std
        return noise

    def sample_eps(self, sample_shape: Tuple) -> Tensor:
        """Sample epsilon for reparametrization from normal distribution."""
        sampled_eps: Tensor = self.distribution.sample(sample_shape).to(self.device)
        return sampled_eps

    def sample_pre_computed(self, mu: Tensor, log_var: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample the generator given a mean value and a log variance."""
        std = torch.exp(0.5 * log_var)
        eps = self.sample_eps(std.shape)
        noise = mu + eps * std
        return self.component(noise), noise  # type: ignore

    def sample_z(self, sample_amount: int):
        """Raise `NotImplementedError` since the sampler has no latent distribution."""
        raise NotImplementedError("Sampling of latent space is not supported for encoder based models.")


class EncoderDecoderSampler(GeneratorSampler):
    """
    Sampler for a Encoder/Decoder based model, without explicit latent distribution.

    As of yet it remains unused. But it is meant for usage in encoder-based architectures such as BeatGAN. These
    architectures do not have a typical distribution which can be queried for latent space vectors but have to query the
    encoder model for a latent vector.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        dev: device,
        num_channels: int,
        sampling_seq_length: int,
        latent_size: int,
    ):
        super().__init__(
            decoder,
            dev,
            num_channels,
            sampling_seq_length,
            LatentDistributionFactory()(LatentDistribution.ENCODER_BASED),
            latent_size,
        )
        self._data_sampler: Optional[DataSampler] = None
        self.encoder = encoder
        self.decoder = decoder

    @property
    def data_sampler(self) -> Optional[DataSampler]:
        return self._data_sampler

    @data_sampler.setter
    def data_sampler(self, data_sampler: DataSampler):
        self._data_sampler = data_sampler

    def sample(self, data: Tensor) -> Tensor:
        """Sample data by feeding input to encoder and then to decoder."""
        x = self.encoder(data)
        return self.decoder(x)  # type: ignore

    def sample_z(self, sample_amount: int) -> Tensor:
        """Sample the implicit latent space of the encoder."""
        if self.data_sampler is None:
            raise Exception("Dataset sampler for the Encoder/Decoder sampler was not set.")

        data = self.data_sampler.sample(sample_amount)['data'].to(self.device)
        return self.encoder(data)  # type: ignore


class LatentDistributionFactory:
    """Meta module for creating correct loss functions."""

    def __call__(self, distribution: LatentDistribution):
        """Return implemented module when a Loss object is created."""
        distributions = {
            LatentDistribution.NORMAL: torch.distributions.normal.Normal(0, 1),
            LatentDistribution.NORMAL_TRUNCATED: TruncatedNormal(),
            LatentDistribution.UNIFORM: torch.distributions.uniform.Uniform(0, 1),
            LatentDistribution.ENCODER_BASED: None,
        }
        try:
            return distributions[distribution]
        except KeyError as err:
            raise AttributeError('Argument {0} is not set correctly.'.format(distribution)) from err
