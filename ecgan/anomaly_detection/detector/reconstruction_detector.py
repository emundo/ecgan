"""Base class for reconstruction based AD and implementation of such algorithms."""
from abc import ABC
from logging import getLogger
from typing import Dict, List, Optional, cast

import numpy as np
import torch
from torch import Tensor, cat, empty, tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecgan.anomaly_detection.anomaly_assessment import DiscriminatorFactory
from ecgan.anomaly_detection.detector.base_detector import AnomalyDetector
from ecgan.anomaly_detection.reconstruction import (
    InterpolationReconstructor,
    InverseMappingReconstructor,
    ReconstructorFactory,
)
from ecgan.config import GANLatentWalkConfig
from ecgan.evaluation.optimization import optimize_grid_search, optimize_svm, query_svm, retrieve_labels_from_weights
from ecgan.evaluation.tracker import BaseTracker
from ecgan.modules.generative.base import BaseEncoderGANModule, BaseGANModule
from ecgan.training.datasets import SeriesDataset
from ecgan.utils.artifacts import ImageArtifact
from ecgan.utils.custom_types import (
    AnomalyDetectionStrategies,
    DiscriminationStrategy,
    MetricOptimization,
    MetricType,
    ReconstructionType,
)
from ecgan.utils.distances import L2Distance
from ecgan.utils.miscellaneous import get_num_workers
from ecgan.utils.transformation import MinMaxTransformation
from ecgan.visualization.plotter import visualize_reconstruction

logger = getLogger(__name__)


class ReconstructionDetector(AnomalyDetector, ABC):
    """
    Base class for anomaly detectors which reconstruct data.

    The reconstructed data is used to calculate the anomalousness of the data.
    """

    def __init__(
        self,
        module: BaseGANModule,
        reconstructor: ReconstructionType,
        tracker: BaseTracker,
    ):
        super().__init__(module, tracker)

        self.detection_cfg: GANLatentWalkConfig = cast(GANLatentWalkConfig, self.cfg.detection_config)
        self.module: BaseGANModule = module
        self.reconstructor = ReconstructorFactory()(
            reconstructor,
            reconstruction_cfg=self.detection_cfg.RECONSTRUCTION,
            module=self.module,
            tracker=self.tracker,
        )
        self._reconstructed_data: Tensor = empty(0, self.module.seq_len, self.module.num_channels).to(
            self.module.device
        )
        self._labels: Optional[Tensor] = None
        self._scores: Optional[Tensor] = None

    def get_reconstructed_data(self) -> Tensor:
        """Return the reconstructed data."""
        return self._reconstructed_data


class GANAnomalyDetector(ReconstructionDetector):
    r"""
    A GAN based anomaly detector which utilizes a reconstructed series.

    Data is reconstructed by latent interpolation from :ref:`AnoGAN`.
    Given an input sample x, an :math:`\epsilon` similar sample :math:`\hat{x}` is retrieved by
    interpolating through latent space (see :class:`ecgan.detection.reconstruction.InterpolationReconstructor`).
    Afterwards an anomaly score is calculated by

        #. Comparing real and synthetic data in data space (using e.g. :math:`L_1` /:math:`L_2` distance) using
           reconstruction error R(x). R(x) is e.g. the L2 distance or any other distance in data space.
        #. Comparing real and synthetic data using the output of the discriminator using discrimination error D(x).
           D(x) is e.g. the deviation of the output score from a target value. Since this can be unreliable and
           depends on the training progress of the discriminator, feature matching is most commonly used.

    Both components are weighted using :math:`\lambda` according to the :ref:`AnoGAN` paper.
    Additionally we allow using a second weight :math:`\gamma` to incorporate a third variable, the latent norm Z(x).
    Z(x) compares the norm of the latent vector: The distribution of the norm of training data usually follows the
    Chi distribution (albeit depending on the generative net used). The deviation from its mode can be used to
    measure how likely it is that the latent vector has produced the output data.
    """

    def __init__(
        self,
        module: BaseGANModule,
        reconstructor: ReconstructionType,
        tracker: BaseTracker,
    ):
        super().__init__(module, reconstructor, tracker)
        self.detection_cfg: GANLatentWalkConfig = cast(GANLatentWalkConfig, self.cfg.detection_config)
        self.module: BaseGANModule = module
        latent_seq_len = (
            self.module.generator_sampler.sampling_seq_length if self.module.generator_sampler is not None else 1
        )
        self._noise = empty(0, latent_seq_len, self.module.latent_size).to(self.module.device)
        self.discriminator = DiscriminatorFactory()(
            module.discriminator_sampler, self.detection_cfg.discrimination_strategy
        )
        self.z_norm = torch.Tensor(0).to(self.module.device)

    def detect(self, test_x: Tensor, test_y: Tensor) -> np.ndarray:
        """
        Detect anomalies in the test data and return predicted labels.

        Original `detect` method has to be overridden since part of our
        score (and label) optimization requires all reconstructed samples.
        """
        self.z_norm = torch.Tensor(0).to(self.module.device)
        num_test_samples = test_x.shape[0]
        test_dataset = SeriesDataset(test_x.float(), test_y.float())

        dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg.detection_config.BATCH_SIZE,
            num_workers=self.cfg.detection_config.NUM_WORKERS,
            pin_memory=True,
        )

        batch_size = self.detection_cfg.BATCH_SIZE

        reconstruction_error = torch.empty(num_test_samples)
        discriminator_error = torch.empty(num_test_samples)

        for batch_idx, batch in enumerate(tqdm(dataloader, leave=False)):
            data = batch['data'].to(self.module.device)
            labels = batch['label'].to(self.module.device)
            rec_series = self._reconstruct(data=data)

            self.track_batch(rec_series, data, labels)

            reconstruction_error[batch_idx * batch_size : (batch_idx + 1) * batch_size] = L2Distance()(data, rec_series)
            discriminator_error[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ] = self.discriminator.discriminate(rec_series, real_data=data)

        logger.info(
            "Detection complete. Batch size: {0}, avg_time: {1:.2e}, std: {2:.2e} (exact:{1:.6f}+-{2:.8f})".format(
                self.cfg.detection_config.BATCH_SIZE,
                torch.mean(self.reconstructor.time_passed).item(),
                torch.std(self.reconstructor.time_passed).item(),
            )
        )

        if self.detection_cfg.NORMALIZE_ERROR:
            disc_scaler = MinMaxTransformation()
            rec_scaler = MinMaxTransformation()
            if hasattr(self.module, 'normalization_params'):
                vali_normalization_params = self.module.normalization_params  # type: ignore
                disc_scaler.set_params(
                    {
                        'min': [vali_normalization_params['discrimination_error']['min']],
                        'max': [vali_normalization_params['discrimination_error']['max']],
                    }
                )
                rec_scaler.set_params(
                    {
                        'min': [vali_normalization_params['reconstruction_error']['min']],
                        'max': [vali_normalization_params['reconstruction_error']['max']],
                    }
                )
                discriminator_error = disc_scaler.transform(discriminator_error.unsqueeze(1)).squeeze()
                reconstruction_error = rec_scaler.transform(reconstruction_error.unsqueeze(1)).squeeze()
            else:
                logger.info("Unable to set normalization parameters. Retrieving new parameters")
                discriminator_error = disc_scaler.fit_transform(discriminator_error.unsqueeze(1)).squeeze()
                reconstruction_error = rec_scaler.fit_transform(reconstruction_error.unsqueeze(1)).squeeze()

        if self.detection_cfg.AD_SCORE_STRATEGY is None:
            raise ValueError("No AD score strategy selected.")

        self._labels = self._optimize_metric(reconstruction_error, discriminator_error, test_y)

        return self._labels.cpu().numpy()  # type: ignore

    def _detect(self, data: Tensor) -> Tensor:
        """Not implemented."""
        raise NotImplementedError("Method `_detect` is not implemented on class {0}".format(self.__class__))

    def _optimize_metric(  # pylint: disable=R0911
        self, reconstruction_error: Tensor, discriminator_error: Tensor, test_y: Tensor
    ) -> Tensor:
        """
        Predict labels based on the selected strategy.

        The deciding parameters (SVM or manual params) should be retrieved from the validation dataset.
        If none exist: train and evaluate on test data. This should be avoided.

        Returns:
            The predicted labels.
        """
        logger.info("Predicting labels using {0}.".format(self.detection_cfg.AD_SCORE_STRATEGY))
        errors = [reconstruction_error.cpu(), discriminator_error.cpu()]

        if self.detection_cfg.ad_score_strategy == MetricOptimization.RECONSTRUCTION_ERROR:
            # Tau_{rec} is not explicitly calculated on the validation data for this experiment.
            if not hasattr(self.module, 'tau'):
                logger.warning(
                    "DID NOT FIND PRETRAINED TAU. Optimizing on test data. This should be avoided and "
                    "needs to be accounted for in the interpretation."
                )

                return optimize_grid_search(
                    metric=MetricType.FSCORE,
                    labels=test_y.cpu(),
                    errors=errors,
                    taus=cast(List[float], np.linspace(0, 2, 100).tolist()),
                    params=[cast(List[float], np.linspace(0, 1, 50).tolist())],
                )

            logger.info("Loading tau from validation set...")
            return retrieve_labels_from_weights(errors, self.module.tau, [1.0])

        if self.detection_cfg.ad_score_strategy == MetricOptimization.GRID_SEARCH_LAMBDA:
            if not (hasattr(self.module, 'tau') and hasattr(self.module, 'lambda_')):
                logger.warning(
                    "DID NOT FIND PRETRAINED TAU AND LAMBDA. Optimizing on test data. This should be avoided and "
                    "needs to be accounted for in the interpretation."
                )

                return optimize_grid_search(
                    metric=MetricType.FSCORE,
                    labels=test_y.cpu(),
                    errors=errors,
                    taus=cast(List[float], np.linspace(0, 2, 100).tolist()),
                    params=[cast(List[float], np.linspace(0, 1, 50).tolist())],
                )

            logger.info("Loading tau and lambda from validation set...")
            return retrieve_labels_from_weights(errors, self.module.tau, [self.module.lambda_])

        if self.detection_cfg.ad_score_strategy == MetricOptimization.SVM_LAMBDA:
            if not hasattr(self.module, 'svm'):
                logger.warning('DID NOT FIND PRETRAINED SVM. Training on test data. Treat with care.')
                labels, _ = optimize_svm(
                    MetricType.FSCORE,
                    errors,
                    test_y.cpu(),
                )
                return labels

            return query_svm(self.module.svm, errors=errors, labels=test_y.cpu())

        if self.detection_cfg.ad_score_strategy == MetricOptimization.SVM_LAMBDA_GAMMA:
            if not isinstance(self.module, BaseEncoderGANModule):
                raise ValueError(
                    "Cannot select strategy {} for non encoder-GANs".format(self.detection_cfg.ad_score_strategy)
                )

            errors.append(self._get_scaled_norm())
            if not hasattr(self.module, 'svm_mu'):
                logger.warning('DID NOT FIND PRETRAINED SVM. Training on test data. Treat with care.')
                labels, _ = optimize_svm(
                    MetricType.FSCORE,
                    errors,
                    test_y.cpu(),
                )
                return labels
            return query_svm(self.module.svm_mu, errors=errors, labels=test_y.cpu())
        if self.detection_cfg.ad_score_strategy == MetricOptimization.GRID_SEARCH_LAMBDA_GAMMA:
            if not isinstance(self.module, BaseEncoderGANModule):
                raise ValueError(
                    "Cannot select strategy {} for non encoder-GANs".format(self.detection_cfg.ad_score_strategy)
                )

            if not (hasattr(self.module, 'tau') and hasattr(self.module, 'lambda_') and hasattr(self.module, 'gamma')):
                logger.warning(
                    "DID NOT FIND PRETRAINED TAU, LAMBDA and GAMMA. Optimizing on test data. "
                    "This should be avoided and needs to be accounted for in the interpretation."
                )
                params = [
                    np.linspace(0, 1, 50),  # Lambda
                    np.linspace(0, 1, 50),  # Gamma
                ]
                errors.append(self._get_scaled_norm())
                return optimize_grid_search(
                    metric=MetricType.FSCORE,
                    labels=test_y.cpu(),
                    errors=errors,
                    taus=cast(List[float], np.linspace(0, 2, 100)),
                    params=cast(List[List[float]], params),
                )

            logger.info("Loading tau and lambda from validation set...")
            return retrieve_labels_from_weights(errors, self.module.tau, [self.module.lambda_, self.module.gamma])

        supported_strategies = [optim_strategy.value for optim_strategy in MetricOptimization]
        raise ValueError(
            "Unknown optimization strategy {}. Select one of {}".format(
                self.detection_cfg.ad_score_strategy, supported_strategies
            )
        )

    def _get_scaled_norm(self):
        if not isinstance(self.module, BaseEncoderGANModule):
            raise ValueError(
                "Cannot select strategy {} for non encoder-GANs".format(self.detection_cfg.ad_score_strategy)
            )

        latent_norm = torch.norm(self._noise.squeeze(), dim=1)
        if self.detection_cfg.NORMALIZE_ERROR:
            latent_scaler = MinMaxTransformation()
            if hasattr(self.module, 'normalization_params'):
                vali_normalization_params = self.module.normalization_params
                latent_scaler.set_params(
                    {
                        'min': [vali_normalization_params['latent_error']['min']],
                        'max': [vali_normalization_params['latent_error']['max']],
                    }
                )
                return latent_scaler.transform((latent_norm.unsqueeze(1)) - self.module.z_mode).squeeze()
            return latent_scaler.fit_transform((latent_norm.unsqueeze(1)) - self.module.z_mode).squeeze()
        return latent_norm

    def _reconstruct(self, data: Tensor) -> Tensor:
        """
        Detect anomalies inside the `data` Tensor.

        Args:
            data: Tensor (usually of size [series, channel, seq_len]) of real data

        Returns:
            Tensor with the corresponding predicted labels.
        """
        rec_series = self.reconstructor.reconstruct(data)

        self._reconstructed_data = cat([self._reconstructed_data, rec_series], dim=0)

        self.tracker.advance_step()
        return rec_series

    def track_batch(
        self,
        reconstructed_series: Tensor,
        real_series: Tensor,
        labels: Tensor,
        plot_num_samples: int = 4,
    ) -> None:
        """
        Track a batch of reconstructed data.

        By default: visualize the first `plot_num_samples` samples for a visual comparison.
        If the reconstruction is an interpolation reconstructed: Track further metrics.
        This includes the norm of reconstructed data in the latent space, the
        L1 distance between real and fake samples and interpolation grids.

        Args:
            reconstructed_series: Reconstructed data.
            real_series: Real data.
            labels: Labels corresponding to real data.
            plot_num_samples: Amount of samples that shall be plotted.
        """
        # Visualize n reconstructed samples
        rec_name = (
            self.cfg.detection_config.DETECTOR
            + '_reconstructed_'
            + str(self._noise.shape[0])
            + str(self.cfg.detection_config.AMOUNT_OF_RUNS)
        )
        rec_plot = visualize_reconstruction(
            series=reconstructed_series[:plot_num_samples],
            plotter=self.tracker.plotter,
        )
        self.tracker.log_artifacts(ImageArtifact(rec_name, rec_plot))
        # Get healthy and unhealthy heatmap
        first_healthy_sample = (labels == 0).nonzero()[0]
        first_unhealthy_sample = (labels == 1).nonzero()[0]

        # Range is reset for each batch, samples across batches are not comparable.
        data_range = (
            0,
            max(
                torch.abs(
                    real_series[first_unhealthy_sample][0, :, :] - reconstructed_series[first_unhealthy_sample][0, :, :]
                )
            ),
        )

        heatmap_plot_healthy = self.tracker.plotter.create_error_plot(
            real_series[first_healthy_sample][0, :, :],
            reconstructed_series[first_healthy_sample][0, :, :],
            title='Healthy heatmap',
            data_range=data_range,
        )
        heatmap_plot_unhealthy = self.tracker.plotter.create_error_plot(
            real_series[first_unhealthy_sample][0, :, :],
            reconstructed_series[first_unhealthy_sample][0, :, :],
            title='Unhealthy heatmap',
            data_range=data_range,
        )
        self.tracker.log_artifacts(ImageArtifact(rec_name + 'healthy_heatmap', heatmap_plot_healthy))
        self.tracker.log_artifacts(ImageArtifact(rec_name + 'unhealthy_heatmap', heatmap_plot_unhealthy))

        if not isinstance(self.reconstructor, InterpolationReconstructor):
            return

        self.z_norm = cat((self.z_norm, self.reconstructor.total_z_distance))
        sample_diff = torch.abs(self.reconstructor.series_samples - real_series[0])

        self.tracker.log_artifacts(
            ImageArtifact(
                'Sample Difference',
                self.tracker.plotter.get_sampling_grid(
                    sample_diff,
                    color='red',
                ),
            )
        )

        start_encoding = self.reconstructor.z_sequences[0]
        label_ = labels[0]

        self.tracker.log_artifacts(
            ImageArtifact(
                'Interpolated Samples',
                self.tracker.plotter.get_sampling_grid(
                    self.reconstructor.series_samples,
                    scale_per_batch=True,
                    color='green' if label_ == 0 else 'red',
                ),
            )
        )

        for encoding in self.reconstructor.z_sequences:
            if label_ == 0:
                self.tracker.log_metrics(
                    {"encoding_differences_normal": torch.nn.MSELoss()(start_encoding, encoding).item()}
                )
            else:
                self.tracker.log_metrics(
                    {"encoding_differences_abnormal": torch.nn.MSELoss()(start_encoding, encoding).item()}
                )
        for loss in self.reconstructor.losses:
            self.tracker.log_metrics({"interpolation_loss": loss})

        self._noise = cat([self._noise, self.reconstructor.z_sequence], dim=0)

    def _get_data_to_save(self) -> Dict:
        """Select data that shall be saved after anomaly detection."""
        return {
            'labels': self._labels.detach().cpu().tolist() if isinstance(self._labels, Tensor) else None,
            'scores': self._scores.detach().cpu().tolist() if isinstance(self._scores, Tensor) else None,
            'z_norm': self.z_norm.detach().cpu().tolist(),
            'noise': self._noise.detach().cpu().tolist(),
        }

    def load(self, saved_data) -> None:
        """Load data from dict."""
        self._labels = tensor(saved_data['labels']) if saved_data['labels'] is not None else None
        self._scores = tensor(saved_data['scores']) if saved_data['scores'] is not None else None
        self.z_norm = tensor(saved_data['z_norm'])
        self._noise = tensor(saved_data['noise'])

    @staticmethod
    def configure():
        """Configure the default settings for an GANAnomalyDetector."""
        return {
            'detection': {
                'RECONSTRUCTION': {
                    'STRATEGY': ReconstructionType.INTERPOLATE.value,
                    'MAX_RECONSTRUCTION_ITERATIONS': 500,
                    'EPSILON': 0.005,
                    'LATENT_OPTIMIZER': {
                        'NAME': 'adam',
                        'LR': 1e-2,
                    },
                    'CRITERION': 'rgan',
                    'ADAPT_LR': True,
                    'LR_THRESHOLD': 0.05,
                },
                'DISCRIMINATION_STRATEGY': DiscriminationStrategy.FEATURE_MATCHING.value,
                'BATCH_SIZE': 64,
                'DETECTOR': AnomalyDetectionStrategies.ANOGAN.value,
                'AMOUNT_OF_RUNS': 5,
                'AD_SCORE_STRATEGY': MetricOptimization.GRID_SEARCH_LAMBDA.value,
                'EMBEDDING': {
                    'CREATE_UMAP': True,
                    'LOAD_PRETRAINED_UMAP': True,
                },
                'NORMALIZE_ERROR': True,
                'INVERSE_MAPPING_URI': None,
                'SAVE_DATA': False,
            }
        }


class GANInverseAnomalyDetector(GANAnomalyDetector):
    r"""
    Anomaly detector with an inverse mapping from data to latent space.

    The detector can use a pretrained network for mapping a datum to a latent vector.
    Alternatively, a novel mapping can be trained using a fully trained GAN.
    The resulting detection follows the :class:`ecgan.anomaly_detection.reconstruction_detector.GANAnomalyDetector`,
    the only difference is how the reconstructed sample is retrieved. Using the inverse mapping the sample
    is not necessarily :math:`\epsilon` similar but the process is significantly sped up and the runtime is linear.
    """

    def __init__(
        self,
        module: BaseGANModule,
        reconstructor: ReconstructionType,
        tracker: BaseTracker,
    ):
        super().__init__(module, reconstructor, tracker)
        self.reconstructor: InverseMappingReconstructor = cast(InverseMappingReconstructor, self.reconstructor)

    def _reconstruct(self, data: Tensor) -> Tensor:
        """
        Detect anomalies inside the `data` Tensor.

        Args:
            data: Tensor (usually of size [series, channel, seq_len]) of real data.

        Returns:
            Tensor with the corresponding predicted labels.
        """
        # Get mapping from data space x to latent space z
        x_hat = self.reconstructor.reconstruct(data).detach()
        noise = self.reconstructor.noise.detach()

        self._reconstructed_data = cat((self._reconstructed_data, x_hat), dim=0)
        self._noise = cat((self._noise, noise), dim=0)

        return x_hat

    def _detect(self, data: Tensor) -> Tensor:
        """Not implemented."""
        raise NotImplementedError("_detect is not implemented on {}".format(self.__class__))

    @staticmethod
    def configure():
        """Configure the default settings for a GANInverseAnomalyDetector."""
        return {
            'detection': {
                'RECONSTRUCTION': {
                    'STRATEGY': ReconstructionType.INVERSE_MAPPING.value,
                },
                'DISCRIMINATION_STRATEGY': DiscriminationStrategy.FEATURE_MATCHING.value,
                'BATCH_SIZE': 256,
                'NUM_WORKERS': get_num_workers(),
                'DETECTOR': AnomalyDetectionStrategies.INVERSE_MAPPING.value,
                'AMOUNT_OF_RUNS': 1,
                'AD_SCORE_STRATEGY': MetricOptimization.SVM_LAMBDA_GAMMA.value,
                'EMBEDDING': {
                    'CREATE_UMAP': False,
                    'LOAD_PRETRAINED_UMAP': True,
                },
                'NORMALIZE_ERROR': True,
                'INVERSE_MAPPING_URI': None,
                'SAVE_DATA': False,
            },
        }
