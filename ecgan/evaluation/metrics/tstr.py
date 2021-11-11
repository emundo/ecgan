"""Implementation of TSTR from `Esteban et al. 2017 <https://arxiv.org/pdf/1706.02633.pdf>`_."""
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Dict

import numpy as np
import torch
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch import Tensor, from_numpy
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecgan.config import LossConfig, NormalInitializationConfig, OptimizerConfig
from ecgan.networks.cnn import ConvolutionalNeuralNetwork
from ecgan.training.datasets import SeriesDataset
from ecgan.utils.custom_types import (
    InputNormalization,
    Losses,
    Optimizers,
    SklearnAveragingOptions,
    SklearnSVMKernels,
    WeightInitialization,
)
from ecgan.utils.layers import initialize_weights
from ecgan.utils.losses import SupervisedLossFactory
from ecgan.utils.optimizer import OptimizerFactory

logger = getLogger(__name__)


class TSTRClassifier(ABC):
    """Abstract classification baseclass for TSTR."""

    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """Fit model on train data and labels."""
        raise NotImplementedError("TSTRClassifier needs to implement the `fit` method.")

    @abstractmethod
    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """Predict labels for test data."""
        raise NotImplementedError("TSTRClassifier needs to implement the `predict` method.")


class TSTRRandomForest(TSTRClassifier):
    """ECGAN wrapper for sklearn RandomForestClassifier."""

    def __init__(self, n_estimators: int):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        """Fit model on train data and labels."""
        train_x = train_x.reshape(train_x.shape[0], -1)
        return self.classifier.fit(train_x, train_y)

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """Predict labels for test data."""
        test_x = test_x.reshape(test_x.shape[0], -1)
        prediction: np.ndarray = self.classifier.predict(test_x)
        return prediction


class TSTRSVM(TSTRClassifier):
    """ECGAN wrapper for sklearn SVM classifier."""

    def __init__(self, kernel: SklearnSVMKernels = SklearnSVMKernels.RBF):
        self.classifier = svm.SVC(kernel=kernel.value)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        """Fit model on train data and labels."""
        train_x = train_x.reshape(train_x.shape[0], -1)
        return self.classifier.fit(train_x, train_y)

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """Predict labels for test data."""
        test_x = test_x.reshape(test_x.shape[0], -1)
        prediction: np.ndarray = self.classifier.predict(test_x)
        return prediction


class TSTRCNN(TSTRClassifier):
    """
    Fit a small CNN as a classifier for TSTR.

    This means that the `fit` and `predict` methods are implemented and
    accessed as wrappers around the usual training/evaluation steps.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        seq_len: int,
        device,
    ):
        self.epochs = 100
        self.cnn_classifier = ConvolutionalNeuralNetwork(
            num_channels, [32, 64, 128, 256, 512], num_classes, num_classes, seq_len, InputNormalization.BATCH
        )
        initialize_weights(
            self.cnn_classifier, NormalInitializationConfig(NAME=WeightInitialization.NORMAL.value, MEAN=0.0, STD=0.02)
        )
        self.cnn_classifier = DataParallel(self.cnn_classifier)  # type: ignore
        self.cnn_classifier.to(device)
        self.device = device
        loss_config = LossConfig(NAME=Losses.CROSS_ENTROPY.value, REDUCTION='mean')
        self.criterion = SupervisedLossFactory()(params=loss_config)

        optim_config = OptimizerConfig(NAME=Optimizers.ADAM.value, LR=0.0001)

        self.optimizer = OptimizerFactory()(self.cnn_classifier.parameters(), optim_config)
        self.batch_size = 256

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """Fit CNN."""
        epoch = 0
        self.cnn_classifier.train()
        for _ in tqdm(range(self.epochs), desc='TSTR CNN'):
            cnn_dataset = SeriesDataset(from_numpy(train_x), from_numpy(train_y))
            train_loader = DataLoader(
                dataset=cnn_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True,
            )
            for _batch_idx, batch in enumerate(train_loader):
                data = batch['data'].to(self.device)
                labels = batch['label'].long().to(self.device)
                prediction = self.cnn_classifier(data)
                loss = self.criterion.forward(prediction, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch += 1

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """Infer using trained CNN."""
        self.cnn_classifier.eval()
        test_data: Tensor = from_numpy(test_x) if not isinstance(test_x, Tensor) else test_x
        with torch.no_grad():
            predictions = self.cnn_classifier(test_data)
            prediction_labels: np.ndarray = torch.argmax(predictions, dim=1).cpu().numpy()

        return prediction_labels


class TSTRClassifierFactory:
    """Meta module for creating a classifier instance for TSTR."""

    def __call__(self, classifier: str, **kwargs) -> TSTRClassifier:
        """Return an instance of an optimizer."""
        if classifier == 'random_forest':
            n_estimators = kwargs['n_estimators'] if kwargs.get('n_estimators') is not None else 250
            return TSTRRandomForest(n_estimators=n_estimators)

        if classifier == 'cnn':
            num_channels = kwargs.get('num_channels', None)
            seq_len = kwargs.get('seq_len', None)
            num_classes = kwargs.get('num_classes', None)
            if num_channels is None or seq_len is None or num_classes is None:
                logger.warning(
                    "Cannot use CNN TSTR, missing parameters. Defaulting to random_forest with 250 estimators."
                )

                return TSTRRandomForest(n_estimators=250)

            return TSTRCNN(
                num_channels=num_channels,
                num_classes=num_classes,
                seq_len=seq_len,
                device=kwargs.get('device'),
            )
        if classifier == 'svm':
            kernel = (
                SklearnSVMKernels(kwargs['kernel']) if kwargs.get('kernel') is not None else SklearnSVMKernels('rbf')
            )
            return TSTRSVM(kernel=kernel)

        raise NotImplementedError('Classifier {} is not supported for the computation of TSTR.'.format(classifier))


class TSTR:
    """TSTR: train on synthetic, test on real data."""

    def __init__(self, classifier: str = 'random_forest', **kwargs):
        """Get a TSTR instance."""
        self.real_classifier = TSTRClassifierFactory()(classifier, **kwargs)
        self.synth_classifier = TSTRClassifierFactory()(classifier, **kwargs)

    def __call__(
        self,
        real_train_data: np.ndarray,
        real_test_data: np.ndarray,
        real_train_labels: np.ndarray,
        real_test_labels: np.ndarray,
        synth_train_data: np.ndarray,
        synth_test_data: np.ndarray,
        synth_train_labels: np.ndarray,
        synth_test_labels: np.ndarray,
        reverse: bool = False,
    ) -> Dict:
        """
        TSTR: train on synthetic, test on real data.

        The TSTR score can be used to evaluate the output of a GAN while avoiding subjective human judgement. Adapted
        from the original `RGAN implementation <https://github.com/ratschlab/RGAN>`_.
        The idea is rather intuitive:
        We train any classifier on both, real as well as synthetic train data and evaluate the performance based on
        metrics of our choice.

        Args:
            real_train_data: Real training samples for some classifier.
            real_train_labels: The known labels corresponding to `real_train_data`.
            real_test_data: Real samples to evaluate both, the classifier trained on real as well as synthetic data.
            real_test_labels: The known labels corresponding to `real_train_data`.
            synth_train_data: Synthetic training data. Usually synthesized to resemble real_train_data.
            synth_train_labels: Usually `real_train_labels`.
            synth_test_data: Only used for reverse TSTR. Usually created by reconstructing `real_test_data` or random
                sampling the latent space.
            synth_test_labels: Only used for reverse TSTR. Labels usually correspond to `real_test_labels`
            reverse: Flag indicating that we want to calculate the reverse TSTR (TRTS).

        Returns:
            Tensor containing the evaluation metrics
        """
        # Use synth test data for testing (TRTS)
        if reverse:
            if synth_test_data is None:
                raise RuntimeError('Synth data can not be empty for reverse TSTR.')

            real_test_data = synth_test_data
            real_test_labels = synth_test_labels

        # Fit classifiers
        self.real_classifier.fit(real_train_data, real_train_labels)
        self.synth_classifier.fit(synth_train_data, synth_train_labels)

        # Get predictions
        real_predicted_labels = self.real_classifier.predict(real_test_data)
        synth_predicted_labels = self.synth_classifier.predict(real_test_data)

        # Retrieve metrics
        real_metrics = self._get_metrics_dict(real_test_labels, real_predicted_labels)
        synth_metrics = self._get_metrics_dict(real_test_labels, synth_predicted_labels)
        return {'synth': synth_metrics, 'real': real_metrics, 'normalized': synth_metrics['f1'] / real_metrics['f1']}

    @staticmethod
    def _get_metrics_dict(real_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict:
        prec, recall, f1, support = precision_recall_fscore_support(
            real_labels,
            predicted_labels,
            average=SklearnAveragingOptions.WEIGHTED.value,
            zero_division=0,
        )
        accuracy = accuracy_score(real_labels, predicted_labels)
        try:
            auroc = roc_auc_score(real_labels, predicted_labels)
        except ValueError:
            auroc = 0

        return {
            'f1': f1,
            'precision': prec,
            'recall': recall,
            'support': support,
            'accuracy': accuracy,
            'auroc': auroc,
        }
