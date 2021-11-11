"""Functions and classes which are commonly used to calculate anomaly scores."""
from abc import ABC, abstractmethod
from logging import getLogger
from typing import cast

import torch.nn
from torch import Tensor, log
from torch import sum as torchsum

from ecgan.utils.custom_types import DiscriminationStrategy
from ecgan.utils.distances import L2Distance
from ecgan.utils.sampler import DiscriminatorSampler, FeatureDiscriminatorSampler

logger = getLogger(__name__)


class InferenceDiscriminator(ABC):
    """
    Base class for different discrimination inference strategies.

    .. note::   This base class is intended to be used for anomaly detection.
                It is intended to help deciding if some data is drawn from the data distribution by
                e.g. using the direct output of an already trained GAN discriminator but also by comparing the
                activations between real and synthetic data to evaluate if they are similar.

    Generally, this type of Discriminator is used to calculate a score which can be used to asses if data belongs to a
    given distribution. In this class we utilize only neural network based discriminators inspired by GAN architectures
    and :ref:`AnoGAN`. This discriminator can not be used for training.
    """

    def __init__(self, disc_sampler: DiscriminatorSampler):
        self.disc_sampler = disc_sampler

    @abstractmethod
    def discriminate(self, data: Tensor, **kwargs) -> Tensor:
        """
        Discriminate a Tensor.

        Args:
            data: A data sample that should be discriminated.

        Returns:
           Discrimination score.
        """
        raise NotImplementedError("Discriminator needs to implement the `discriminate` method.")


class RawInferenceDiscriminator(InferenceDiscriminator):
    """Return the output of the discriminator."""

    def discriminate(self, data: Tensor, **kwargs) -> Tensor:
        """
        Calculate the raw disc output and return the difference from the target value (1).

        The output depends a lot on the training state of the discriminator and is not necessarily reliable.
        """
        disc_scores = self.disc_sampler.sample(data)
        target_difference: Tensor = 1 - disc_scores.detach()
        return target_difference


class LogInferenceDiscriminator(InferenceDiscriminator):
    """Return the logarithmic output of the discriminator."""

    def discriminate(self, data: Tensor, **kwargs) -> Tensor:
        """
        Return the log of the disc output.

        The output depends a lot on the training state of the discriminator and is not necessarily reliable.
        """
        disc_scores = self.disc_sampler.sample(data)
        return log(disc_scores.detach())


class FeatureMatchingInferenceDiscriminator(InferenceDiscriminator):
    """Match the features of the real and reconstructed series."""

    def __init__(self, disc_sampler: DiscriminatorSampler):
        super().__init__(disc_sampler)
        self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.disc_sampler: FeatureDiscriminatorSampler = cast(FeatureDiscriminatorSampler, self.disc_sampler)

    def discriminate(self, data: Tensor, **kwargs) -> Tensor:
        """
        Check if the features of real and fake data are similar.

        The similarity is measured by the MSELoss/L2 distance between the real data and the reconstructed data. This
        also means that, in comparison to many other discriminators, we do not only need to reconstructed samples but
        the real samples as well.
        """
        if kwargs['real_data'] is None:
            raise AttributeError(
                "Feature matching requires the real data. "
                "Please pass the data or select a different discrimination strategy."
            )
        real_data = kwargs['real_data']
        with torch.no_grad():
            disc_out_fake_feat = self.disc_sampler.sample_features(data)
            disc_out_real_feat = self.disc_sampler.sample_features(real_data)

        return L2Distance()(disc_out_real_feat, disc_out_fake_feat)


class DiscriminatorFactory:
    """Factory module for creating Discriminator objects."""

    def __call__(
        self,
        disc_sampler: DiscriminatorSampler,
        strategy: DiscriminationStrategy,
    ) -> InferenceDiscriminator:
        """Return an InferenceDiscriminator instance."""
        strategies = {
            DiscriminationStrategy.TARGET_VALUE: RawInferenceDiscriminator(disc_sampler),
            DiscriminationStrategy.LOG: LogInferenceDiscriminator(disc_sampler),
            DiscriminationStrategy.FEATURE_MATCHING: FeatureMatchingInferenceDiscriminator(disc_sampler),
        }
        try:
            return strategies[strategy]
        except KeyError as err:
            raise AttributeError(
                'Unknown discriminator strategy: {0}. Please select a valid DiscriminationStrategy.'.format(strategy)
            ) from err


def get_pointwise_anomaly_score_anogan(
    discrimination_error: Tensor,
    reconstruction_error: Tensor,
    lambda_: float = 0.1,
) -> Tensor:
    r"""
    Calculate the series-wise anomaly scores.

    Calculate the series-wise anomaly scores including all channel- and pointwise anomaly scores. Currently based on the
    AnoGAN approach (:math:`lambda` is the weighting between discrimination error and reconstruction error).

    Args:
        discrimination_error: Pointwise discrimination error.
        reconstruction_error: Pointwise reconstruction error.
        lambda_: Weighting of the reconstruction error (the higher lambda,
            the less relevant the discrimination_error. The weight is assumed to be in [0, 1].

    Returns:
        The pointwise anomaly score.
    """
    if discrimination_error.shape != reconstruction_error.shape:
        raise RuntimeError('Anomaly Score Error: Shape of samples used for anomaly detection is not equal.')

    anomaly_scores: Tensor = lambda_ * reconstruction_error + (1 - lambda_) * discrimination_error

    return anomaly_scores


def get_anomaly_score(
    discrimination_error: Tensor,
    reconstruction_error: Tensor,
    pointwise: bool = True,
    lambda_: float = 0.1,
) -> Tensor:
    r"""
    Return an anomaly score.

    Anomaly score:

        1. Default: all points for each channel.
        2. All points inside one series summarized.

    Args:
        discrimination_error: Pointwise discrimination error.
        reconstruction_error: Pointwise reconstruction error.
        pointwise: Signals that the pointwise anomaly score should be returned.
            If :code:`pointwise=False`: Return serieswise anomaly score.
        lambda_: Weighting of the reconstruction error.

    Returns:
         The pointwise or channelwise reconstruction error for each time series.
    """
    assert (
        discrimination_error.shape == reconstruction_error.shape
    ), 'Anomaly Score: Shape of samples used for anomaly detection is not equal.'
    anomaly_score: Tensor = get_pointwise_anomaly_score_anogan(
        discrimination_error, reconstruction_error, lambda_=lambda_
    )
    # Returns anomaly values for every point in every univariate series of all multivariate series
    if pointwise:
        return anomaly_score

    # Returns a single anomaly value for each multivariate series.
    return torchsum(anomaly_score, dim=(2, 1))
