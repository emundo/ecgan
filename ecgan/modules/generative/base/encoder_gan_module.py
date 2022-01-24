"""Base class for encoder based GANs."""
from abc import abstractmethod
from logging import getLogger
from typing import Dict, List, Optional, cast

import torch
from numpy import argmax, histogram
from sklearn.svm import SVC
from torch import nn

from ecgan.config import EncoderGANConfig
from ecgan.evaluation.optimization import (
    optimize_metric,
    optimize_svm,
    optimize_tau_single_error,
    retrieve_labels_from_weights,
)
from ecgan.modules.generative.base.gan_module import BaseGANModule
from ecgan.utils.artifacts import Artifact, FileArtifact, ImageArtifact, ValueArtifact
from ecgan.utils.custom_types import LossMetricType, MetricType, SklearnSVMKernels
from ecgan.utils.distances import L2Distance
from ecgan.utils.interpolation import latent_walk
from ecgan.utils.miscellaneous import load_model
from ecgan.utils.sampler import EncoderBasedGeneratorSampler, FeatureDiscriminatorSampler
from ecgan.utils.transformation import MinMaxTransformation

logger = getLogger(__name__)


class BaseEncoderGANModule(BaseGANModule):
    """Base class for GANs with an autoencoder as generator."""

    def __init__(
        self,
        cfg: EncoderGANConfig,
        seq_len: int,
        num_channels: int,
    ):
        self.cfg = cast(EncoderGANConfig, cfg)
        self._encoder = self._init_inverse_mapping()
        super().__init__(
            cfg=cfg,
            seq_len=seq_len,
            num_channels=num_channels,
        )

        self._encoder = nn.DataParallel(self.encoder)
        self._encoder.to(self.device)
        # Have to be set after data sampler has been added. Not possible upon creation.
        self.fixed_samples: Optional[torch.Tensor] = None
        self.fixed_samples_labels: Optional[torch.Tensor] = None
        self.svm_mu: SVC = SVC()
        self.z_mu: float = 0.0
        self.z_mode: float = 0.0
        self.gamma: float = 0.0  # not currently supported via saved grid search - use svm_mu for improved results
        self.normalization_params: Dict = {'reconstruction_error': {}, 'discrimination_error': {}, 'latent_error': {}}

    @property
    def encoder(self):
        return self._encoder

    @property
    def discriminator_sampler(self) -> FeatureDiscriminatorSampler:
        return cast(FeatureDiscriminatorSampler, self._discriminator_sampler)

    @property
    def generator_sampler(self) -> EncoderBasedGeneratorSampler:
        return cast(EncoderBasedGeneratorSampler, self._generator_sampler)

    @abstractmethod
    def _init_inverse_mapping(self) -> nn.Module:
        raise NotImplementedError("EncoderGANModule needs to implement the `_init_inverse_mapping` method.")

    @property
    def watch_list(self) -> List[nn.Module]:
        """Return models that should be watched during training."""
        return [self.generator, self.discriminator, self.encoder]

    def training_step(
        self,
        batch: dict,
    ) -> dict:
        """
        Declare what the model should do during a training step using a given batch.

        Args:
            batch: The batch of real data.

        Return:
            A dict containing the optimization metrics which shall be logged.
        """
        real_data = batch['data'].to(self.device)
        self.generator.train()
        self.discriminator.train()
        self.encoder.train()
        self._prepare_train_step()
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

            return self._evaluate_train_step(disc_metrics=disc_metric_collection, gen_metrics=gen_metric_collection)

        except TypeError as err:
            raise TypeError('Error during training: Config parameter was not correctly set: {0}.'.format(err)) from err

    def _prepare_train_step(self):
        """Can be used to set dynamic variables."""
        pass

    def _evaluate_train_step(  # pylint: disable=R0201
        self, disc_metrics: LossMetricType, gen_metrics: LossMetricType
    ) -> Dict:
        """
        Can be used to evaluate data without impacting the training.

        If multiple gen/disc rounds are used, the behavior of the metric logging and
        this method is not sufficient in the current state.
        """
        return {key: float(value) for (key, value) in disc_metrics + gen_metrics}

    def validation_step(self, batch: dict) -> dict:
        """
        Perform a validation step.

        This method states the validation or inference process for one given batch.

        Args:
            batch: Dictionary containing training tensors.

        Returns:
            Dictionary with metrics to log (e.g. loss).
        """
        if not isinstance(self.discriminator_sampler, FeatureDiscriminatorSampler):
            raise AttributeError("Encoder based GANs currently require a feature discrimination implementation.")
        data = batch['data'].to(self.device)
        label = batch['label'].to(self.device)

        self.discriminator.eval()
        self.generator.eval()
        self.encoder.eval()

        l2_distance = L2Distance()

        if self.fixed_samples is None:
            self.set_fixed_samples()

        with torch.no_grad():
            x_hat, latent_vector = self.generator_sampler.sample_generator_encoder(data=data)
            features_fake = self.discriminator_sampler.sample_features(x_hat)
            features_real = self.discriminator_sampler.sample_features(data)
            rec_error = l2_distance(data, x_hat)
            disc_error = l2_distance(features_real, features_fake)

        # concatenate tensors to form tensors for all batches in one epoch
        self.reconstruction_error = torch.cat((self.reconstruction_error, rec_error), dim=0)
        self.latent_vectors_vali = torch.cat((self.latent_vectors_vali, latent_vector), dim=0)
        self.discrimination_error = torch.cat((self.discrimination_error, disc_error), dim=0)
        self.label = torch.cat((self.label, label), dim=0)

        return self._get_validation_results(data)

    @abstractmethod
    def _get_validation_results(self, data: torch.Tensor) -> Dict:
        raise NotImplementedError("GANEncoder models need to implement `_get_validation_results` which will be logged.")

    def on_epoch_end(self, epoch: int, sample_interval: int, batch_size: int) -> List[Artifact]:
        """
        Every `sample_interval`-th epoch.

        1. Sample the reconstruction of previously set fixed samples from the generator.
        2. Walk through latent space to check how data changes when walking through latent space.

        Every tenth epoch and for the last 30 epochs:
        Check metrics using the optimization procedure for the reconstruction and discrimination loss.
        """
        result: List[Artifact] = []

        # Min-max normalize error:
        scaler = MinMaxTransformation()
        self.reconstruction_error = scaler.fit_transform(self.reconstruction_error.unsqueeze(1)).squeeze()
        scaling_params = {key: value[0] for key, value in scaler.get_params().items()}
        self.normalization_params['reconstruction_error'] = scaling_params
        self.discrimination_error = scaler.fit_transform(self.discrimination_error.unsqueeze(1)).squeeze()
        scaling_params = {key: value[0] for key, value in scaler.get_params().items()}
        self.normalization_params['discrimination_error'] = scaling_params

        if epoch % sample_interval == 0:
            result.append(self._reconstruct_fixed_samples())
            if self.fixed_samples is None or self.fixed_samples_labels is None:
                raise RuntimeError("Fixed samples not set correctly.")

            # Interpolate through latent space for normal and abnormal samples
            result.append(self._get_interpolation_grid(self.fixed_samples[1], 'Normal Class'))
            result.append(self._get_interpolation_grid(self.fixed_samples[-1], 'Abnormal Class'))

            # Save the latent norms of the data
            result.append(
                FileArtifact(
                    'Latent vector distribution',
                    {
                        'latent_train': self.latent_vectors_train.cpu(),
                        'latent_vali': self.latent_vectors_vali.cpu(),
                        'labels': self.label.cpu(),
                    },
                    'latent_data_{}.pkl'.format(epoch),
                )
            )

            # Get distribution of latent norm
            latent_norm_train = torch.norm(self.latent_vectors_train.squeeze(), dim=1)
            logger.info(
                "latent norm train. Shape {}, mean {}, median {}, std {}".format(
                    latent_norm_train.shape,
                    torch.mean(latent_norm_train),
                    torch.median(latent_norm_train),
                    torch.std(latent_norm_train),
                )
            )
            latent_norm_vali_abnormal = torch.norm(self.latent_vectors_vali[self.label != 0].squeeze(), dim=1)
            logger.info(
                "latent norm vali.Shape {}, mean {}, median {}, std {}".format(
                    latent_norm_vali_abnormal.shape,
                    torch.mean(latent_norm_vali_abnormal),
                    torch.median(latent_norm_vali_abnormal),
                    torch.std(latent_norm_vali_abnormal),
                )
            )

            result.append(
                ImageArtifact(
                    'Norm of latent vectors (normal train)',
                    self.plotter.create_histogram(
                        latent_norm_train.cpu().numpy(), 'Norm of latent vectors (normal train)'
                    ),
                )
            )

            result.append(
                ImageArtifact(
                    'Norm of latent vectors (abnormal validation)',
                    self.plotter.create_histogram(
                        latent_norm_vali_abnormal.cpu().numpy(), 'Norm of latent vectors (abnormal vali)'
                    ),
                )
            )

            # Get train statistics
            result.append(ValueArtifact('latent/z_mean_min', torch.min(self.latent_vectors_train).item()))
            result.append(ValueArtifact('latent/z_mean_max', torch.max(self.latent_vectors_train).item()))
            self.z_mu = torch.mean(latent_norm_train).item()
            # Approximate mode (median can be used as a simple alternative given the distribution scatters a lot)
            mode_count, mode_val = histogram(latent_norm_train.cpu().numpy(), bins=50)
            self.z_mode = mode_val[argmax(mode_count)]
            # Get euclidean distance from origin
            latent_norm_vali = torch.norm(self.latent_vectors_vali.squeeze(), dim=1)

            # For each scaled norm: subtract mode of mu to approximately shift to center of chi distribution.
            scaled_latent_norm = scaler.fit_transform((latent_norm_vali.unsqueeze(1)) - self.z_mode).squeeze()
            scaling_params = {key: value[0] for key, value in scaler.get_params().items()}
            self.normalization_params['latent_error'] = scaling_params

            # Get anomaly scores:
            # 1. SVM on reconstruction error, discrimination error and latent norm
            pred_latent_scaled, self.svm_mu = optimize_svm(
                MetricType.FSCORE,
                [
                    self.reconstruction_error.cpu().detach(),
                    self.discrimination_error.cpu().detach(),
                    scaled_latent_norm.cpu(),
                ],
                self.label.cpu(),
            )

            result.extend(
                self._get_metrics(
                    self.label, pred_latent_scaled, 'svm/scaled_latent', log_fscore=True, log_auroc=True, log_mcc=True
                )
            )

            # 2. SVM on reconsturction error and discrimination error
            pred_minmax, self.svm = optimize_svm(
                MetricType.FSCORE,
                [
                    self.reconstruction_error.cpu().detach(),
                    self.discrimination_error.cpu().detach(),
                ],
                self.label.cpu(),
            )

            result.extend(
                self._get_metrics(self.label, pred_minmax, 'svm/minmax', log_fscore=True, log_auroc=True, log_mcc=True)
            )

            pred_linear_svm, _ = optimize_svm(
                MetricType.FSCORE,
                [
                    self.reconstruction_error.cpu().detach(),
                    self.discrimination_error.cpu().detach(),
                ],
                self.label.cpu(),
                kernel=SklearnSVMKernels.LINEAR,
            )
            result.extend(
                self._get_metrics(
                    self.label, pred_linear_svm, 'svm/linear', log_fscore=True, log_auroc=False, log_mcc=False
                )
            )

            mmd = self.get_mmd()
            result.append(ValueArtifact('generative_metric/mmd', mmd))
            tstr_dict = self.get_tstr()
            result.append(ValueArtifact('generative_metric/tstr', tstr_dict))

            # Evaluate lambda = 0, lambda=1 for anogan and gamma=1 for vaegan
            tau_range = torch.linspace(0, 2, 100).cpu().tolist()
            result.append(
                ValueArtifact(
                    'only_reconstruction_error',
                    optimize_tau_single_error(self.label.cpu(), self.reconstruction_error.cpu(), tau_range),
                )
            )
            result.append(
                ValueArtifact(
                    'only_disc_error',
                    optimize_tau_single_error(self.label.cpu(), self.discrimination_error.cpu(), tau_range),
                )
            )
            result.append(
                ValueArtifact(
                    'only_latent_error',
                    optimize_tau_single_error(self.label.cpu(), scaled_latent_norm.cpu(), tau_range),
                )
            )

        # Optimize F-score every 10 epochs
        if epoch % 10 == 0:
            # if False:
            lambda_search_range = [torch.linspace(0, 1, 50).numpy().tolist()]
            best_params = optimize_metric(
                MetricType.FSCORE,
                errors=[self.reconstruction_error.cpu(), self.discrimination_error.cpu()],
                taus=torch.linspace(0, 2, 100).numpy().tolist(),
                params=lambda_search_range,
                ground_truth_labels=self.label.cpu(),
            )
            logger.info(
                "Best params: {} for data {}.".format(best_params, torch.unique(self.label, return_counts=True))
            )
            self.tau = best_params[0][1]
            self.lambda_ = best_params[0][2]
            result.append(
                ValueArtifact(
                    'grid/lambda_tau',
                    float(self.tau),
                )
            )
            result.append(
                ValueArtifact(
                    'grid/lambda',
                    float(self.lambda_),
                )
            )
            predictions = retrieve_labels_from_weights(
                errors=[self.reconstruction_error.cpu(), self.discrimination_error.cpu()],
                tau=self.tau,
                weighting_params=[self.lambda_],
            )
            result.extend(
                self._get_metrics(
                    self.label,
                    predictions,
                    'grid/lambda',
                    log_fscore=True,
                    log_auroc=True,
                    log_mcc=True,
                )
            )

        result.extend(self._on_epoch_end_addition(epoch, sample_interval))
        self._reset_internal_tensors()
        return result

    def _on_epoch_end_addition(self, epoch: int, sample_interval: int) -> List[Artifact]:  # pylint: disable=R0201,W0613
        return []

    def _reconstruct_fixed_samples(
        self,
    ) -> ImageArtifact:
        """
        Visualize real (fixed) samples and the faked_samples which aim to reconstruct them.

        Returns:
            Artifact containing a comparison of the real and reconstructed fixed samples.
        """
        if self.fixed_samples is None or self.fixed_samples_labels is None:
            raise RuntimeError("Fixed samples not set correctly.")

        with torch.no_grad():
            faked_samples, _ = self.generator_sampler.sample_generator_encoder(data=self.fixed_samples)

        samples = torch.empty(
            (
                2 * self.num_fixed_samples,
                faked_samples.shape[1],
                faked_samples.shape[2],
            )
        )
        labels = torch.empty((2 * self.num_fixed_samples, 1))

        for i in range(self.num_fixed_samples):
            samples[2 * i] = self.fixed_samples[i]
            samples[2 * i + 1] = faked_samples[i]
            labels[2 * i] = self.fixed_samples_labels[i]
            labels[2 * i + 1] = self.fixed_samples_labels[i]

        return ImageArtifact(
            'Fixed Generator Samples',
            self.plotter.get_sampling_grid(
                samples,
                label=labels,
            ),
        )

    def save_checkpoint(self) -> dict:
        """Return current model parameters."""
        return {
            'GEN': self.generator.state_dict(),
            'DIS': self.discriminator.state_dict(),
            'ENC': self.encoder.state_dict(),
            'GEN_OPT': self.optim_gen.state_dict(),
            'DIS_OPT': self.optim_disc.state_dict(),
            'ANOMALY_DETECTION': {
                'SVM': self.svm,
                'LAMBDA': self.lambda_,
                'GAMMA': self.gamma,
                'TAU': self.tau,
                'Z_MU': self.z_mu,
                'Z_MODE': self.z_mode,
                'SVM_MU': self.svm_mu,
                'NORM_PARAMS': self.normalization_params,
            },
            'FIXED_SAMPLES': self.fixed_samples.detach().cpu().tolist()
            if isinstance(self.fixed_samples, torch.Tensor)
            else None,  # temporary for tracking and graphing
        }

    def load(self, model_reference: str, load_optim: bool = False):
        """Load a trained module from disk (file path) or wand reference."""
        model = load_model(model_reference, self.device)

        self.generator.load_state_dict(model['GEN'], strict=False)
        self.discriminator.load_state_dict(model['DIS'], strict=False)
        self.encoder.load_state_dict(model['ENC'], strict=False)

        if load_optim:
            self.optim_gen.load_existing_optim(model['GEN_OPT'])
            self.optim_disc.load_existing_optim(model['DIS_OPT'])

        logger.info('Loading existing {0} model completed.'.format(self.__class__.__name__))
        self.svm = model['ANOMALY_DETECTION']['SVM']
        self.tau = model['ANOMALY_DETECTION']['TAU']
        self.lambda_ = model['ANOMALY_DETECTION']['LAMBDA']
        self.gamma = model['ANOMALY_DETECTION']['GAMMA']
        self.z_mu = model['ANOMALY_DETECTION']['Z_MU']
        self.z_mode = model['ANOMALY_DETECTION']['Z_MODE']
        self.svm_mu = model['ANOMALY_DETECTION']['SVM_MU']
        self.normalization_params = model['ANOMALY_DETECTION']['NORM_PARAMS']
        return self

    def set_fixed_samples(self) -> None:
        """
        Set the fixed samples of the module.

        Utilized in validation_step or on_epoch_end to have comparable samples across epochs.
        It is made sure that approximately the same amount of samples belong to
        class 0 and 1.
        """
        fixed_samples_normal = self.vali_dataset_sampler.sample_class(self.num_fixed_samples // 2, 0)
        fixed_samples_anormal = self.vali_dataset_sampler.sample_class(
            self.num_fixed_samples - self.num_fixed_samples // 2, 1
        )

        self.fixed_samples = torch.cat((fixed_samples_normal['data'], fixed_samples_anormal['data'])).to(self.device)
        self.fixed_samples_labels = torch.cat((fixed_samples_normal['label'], fixed_samples_anormal['label'])).to(
            self.device
        )

    def _get_interpolation_grid(self, base_sample: torch.Tensor, class_name: str) -> ImageArtifact:
        interpolated_samples = self.get_interpolated_samples(base_sample.unsqueeze(0))
        return ImageArtifact(
            '{0} Interpolation Grid'.format(class_name),
            self.plotter.get_sampling_grid(
                interpolated_samples,
                row_width=11,
                scale_per_batch=True,
                max_num_series=interpolated_samples.shape[0],
            ),
        )

    def get_interpolated_samples(self, sample: torch.Tensor):
        """Interpolate through latent space based on fixed samples."""
        # Investigate the latent space using one of the fixed samples and performing a latent walk.
        with torch.no_grad():
            _x_hat, inverse_mapping = self.get_sample(data=sample)
            std = torch.std(inverse_mapping).item()
            logger.info("Standard error of latent space is {}.".format(std))
            walk_space = torch.linspace(-std, std, 11)
            interpolated_samples = latent_walk(
                inverse_mapping,
                self.generator,
                walk_range=walk_space,
                device=self.device,
                latent_dims=inverse_mapping.shape[2],
            )
        return interpolated_samples
