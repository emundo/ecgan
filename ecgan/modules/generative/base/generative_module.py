"""Abstract base generative module."""
from abc import abstractmethod
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
from torch import no_grad

from ecgan.config import ModuleConfig
from ecgan.evaluation.metrics.classification import AUROCMetric, AvgPrecisionMetric, FScoreMetric, MCCMetric
from ecgan.evaluation.metrics.mmd import MaxMeanDiscrepancy
from ecgan.evaluation.metrics.tstr import TSTR
from ecgan.modules.base import BaseModule
from ecgan.utils.artifacts import ValueArtifact

logger = getLogger(__name__)


class BaseGenerativeModule(BaseModule):
    """Abstract base generative module containing several generative metrics."""

    def __init__(
        self,
        cfg: ModuleConfig,
        seq_len: int,
        num_channels: int,
    ):
        super().__init__(cfg, seq_len, num_channels)
        self.num_classes: int = self.dataset.NUM_CLASSES_BINARY

    @abstractmethod
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
        raise NotImplementedError("GenerativeModule needs to implement the `get_sample` method.")

    def get_tstr(self) -> Dict:
        """
        Calculate TSTR values.

        Requires a validation dataset from which the data is drawn. Afterwards, data is generated either from
        randomly sampling the latent space (e.g. GAN based models which use a random z vector) or from retrieving a
        reconstructed sample from the validation data used for training.

        Returns:
            Dict containing TSTR statistics.
        """
        tstr_metric = TSTR(
            n_estimators=250,
            num_channels=self.num_channels,
            classifier='cnn',
            seq_len=self.seq_len,
            device=self.device,
            num_classes=self.num_classes,
        )
        # Train TSTR on binary labels without class imbalance: Use equal amount of normal and abnormal data.
        # Use 2/3 of the data for training and 1/3 for testing.
        lower_volume_class = (
            0 if self.vali_dataset_sampler.get_dataset_size(0) < self.vali_dataset_sampler.get_dataset_size(1) else 1
        )

        num_test_samples_per_class = (self.vali_dataset_sampler.get_dataset_size(lower_volume_class)) // 3
        num_train_samples_per_class = (
            self.vali_dataset_sampler.get_dataset_size(lower_volume_class) - num_test_samples_per_class
        )
        samples_per_class = num_test_samples_per_class + num_train_samples_per_class
        abnormal_data_samples = self.vali_dataset_sampler.sample_class(samples_per_class, class_label=1)
        normal_data_samples = self.vali_dataset_sampler.sample_class(samples_per_class, class_label=0)
        anomalous_data_train = {
            'data': abnormal_data_samples['data'][:num_train_samples_per_class],
            'label': abnormal_data_samples['label'][:num_train_samples_per_class],
        }
        anomalous_data_test = {
            'data': abnormal_data_samples['data'][:num_test_samples_per_class],
            'label': abnormal_data_samples['label'][:num_test_samples_per_class],
        }
        normal_data_train = {
            'data': normal_data_samples['data'][num_train_samples_per_class:],
            'label': normal_data_samples['label'][num_train_samples_per_class:],
        }
        normal_data_test = {
            'data': normal_data_samples['data'][num_test_samples_per_class:],
            'label': normal_data_samples['label'][num_test_samples_per_class:],
        }

        real_train_data = torch.cat((normal_data_train['data'], anomalous_data_train['data'])).cpu().numpy()
        real_test_data = torch.cat((normal_data_test['data'], anomalous_data_test['data'])).cpu().numpy()
        real_train_label = torch.cat((normal_data_train['label'], anomalous_data_train['label'])).cpu().numpy()
        real_test_label = torch.cat((normal_data_test['label'], anomalous_data_test['label'])).cpu().numpy()

        # Attention: transition of terminology.
        # Before: zero-Tensor related to the discriminator score > lower score was worse
        # Now: zero-Tensor means that the data was not anomalous i.e. all data fed into the GAN
        #
        # In the current version, the GAN is only trained to generate data from class 0 (normal/non-anomalous/healthy).
        # Training a classifier with only one label would result in not very useful information since no useful
        # separating hyperplane can be created. To allow the creation of such a hyperplane, anomalous data has to be
        # considered.

        with no_grad():
            fake_train_data, _ = self.get_sample(
                num_samples=num_test_samples_per_class,
                data=normal_data_train['data'].to(self.device),
            )
            fake_test_data, _ = self.get_sample(
                num_samples=num_test_samples_per_class,
                data=normal_data_test['data'].to(self.device),
            )

            fake_train_data = torch.cat((fake_train_data.cpu(), anomalous_data_train['data'])).cpu()
            fake_train_label = torch.cat((normal_data_train['label'], anomalous_data_train['label'])).cpu()
            fake_test_data = torch.cat((fake_test_data.cpu(), anomalous_data_test['data'])).cpu()
            fake_test_label = torch.cat((normal_data_test['label'], anomalous_data_test['label'])).cpu()

        tstr_score = tstr_metric(
            real_train_data=real_train_data,
            real_test_data=real_test_data,
            real_train_labels=real_train_label,
            real_test_labels=real_test_label,
            synth_train_data=fake_train_data.numpy(),
            synth_test_data=fake_test_data.numpy(),
            synth_train_labels=fake_train_label.numpy(),
            synth_test_labels=fake_test_label.numpy(),
        )

        logger.info(
            'Metrics: TSTR F1 synth at {0}, normalized with real classifier: {1}.'.format(
                tstr_score['synth']['f1'], tstr_score['normalized']
            )
        )

        return tstr_score

    def get_mmd(self, num_samples: int = 512, sigma: float = 5.0) -> float:
        """
        Calculate the maximum mean discrepancy.

        Args:
            num_samples: Amount of samples used for the MMD calculation.
            sigma: Sigma for Gaussian kernel during MMD.

        Returns:
            MMD score.
        """
        mmd_metric = MaxMeanDiscrepancy(sigma=sigma)

        samples = num_samples
        real_data = self.train_dataset_sampler.sample(samples)['data'].to(self.device)

        with no_grad():
            synthetic_data, _ = self.get_sample(num_samples=samples, data=real_data.to(self.device))

        mmd_score = mmd_metric(real_data, synthetic_data)

        return mmd_score

    @staticmethod
    def _get_metrics(
        real_labels: torch.Tensor,
        predicted_labels: torch.Tensor,
        identifier: str,
        log_fscore: bool = True,
        log_mcc: bool = True,
        log_auroc: bool = True,
        log_avg_prec: bool = True,
    ) -> List[ValueArtifact]:
        result: List = []
        if log_fscore:
            fscore = FScoreMetric().calculate(real_labels, predicted_labels)

            result.append(ValueArtifact('{}_fscore'.format(identifier), fscore))
        if log_mcc:
            mcc = MCCMetric().calculate(real_labels, predicted_labels)
            result.append(ValueArtifact('{}_mcc'.format(identifier), mcc))
        if log_auroc:
            auroc = AUROCMetric().calculate(real_labels, predicted_labels)
            result.append(ValueArtifact('{}_auroc'.format(identifier), auroc))
        if log_avg_prec:
            avg_prec = AvgPrecisionMetric().calculate(real_labels, predicted_labels)
            result.append(ValueArtifact('{}_ap'.format(identifier), avg_prec))

        return result
