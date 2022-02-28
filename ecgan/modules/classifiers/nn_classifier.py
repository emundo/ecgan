"""Basic RNN/CNN with the aim to correctly predict labels based on input data."""
from abc import abstractmethod
from logging import getLogger
from typing import Dict, List

from torch import Tensor, argmax, nn, no_grad

from ecgan.config import BaseCNNConfig, BaseNNConfig, BaseRNNConfig, TrainerConfig, get_global_config
from ecgan.modules.base import BaseModule
from ecgan.modules.classifiers.base import BaseClassifier
from ecgan.networks.cnn import ConvolutionalNeuralNetwork
from ecgan.networks.rnn import RecurrentNeuralNetwork
from ecgan.utils.artifacts import Artifact
from ecgan.utils.custom_types import InputNormalization
from ecgan.utils.layers import initialize_weights
from ecgan.utils.losses import CrossEntropyLoss, SupervisedLoss, SupervisedLossFactory
from ecgan.utils.miscellaneous import load_model, scale_weights
from ecgan.utils.optimizer import Adam, BaseOptimizer, OptimizerFactory

logger = getLogger(__name__)


class NNClassifier(BaseModule, BaseClassifier):
    """NN used to predict labels, not used for forecasting which can also be used for AD."""

    def __init__(self, cfg: BaseNNConfig, seq_len: int, num_channels: int):
        super().__init__(cfg, seq_len, num_channels)
        self.train_cfg: TrainerConfig = get_global_config().trainer_config

        self.cfg: BaseNNConfig = cfg
        self._classifier = self._init_classifier(self.cfg)
        if self.device == 'gpu':
            self._classifier = nn.DataParallel(self.classifier)
        self._classifier.to(self.device)

        self._optim = self._init_optimizer()

        self._criterion = self._init_criterion()

    @abstractmethod
    def _init_classifier(self, cfg: BaseNNConfig) -> nn.Module:
        raise NotImplementedError("NNClassifier needs to implement the `_init_classifier` method.")

    @property
    def classifier(self) -> nn.Module:
        """Return the NN classifier."""
        return self._classifier

    def classify(self, data: Tensor) -> Tensor:
        """Return a classification score according to the NN."""
        self.classifier.eval()

        with no_grad():
            prediction: Tensor = self.classifier(data)
        return prediction

    def _init_optimizer(self) -> BaseOptimizer:
        """Initialize the optimizer for the network."""
        return OptimizerFactory()(self.classifier.parameters(), self.cfg.OPTIMIZER)

    @property
    def optimizer(self) -> BaseOptimizer:
        """Return the optimizer for the network."""
        return self._optim

    def _init_criterion(self) -> SupervisedLoss:
        """Initialize the criterion for the network."""
        return SupervisedLossFactory()(
            params=self.cfg.LOSS,
        )

    @property
    def criterion(self) -> SupervisedLoss:
        """Return the criterion for the network."""
        return self._criterion

    def training_step(
        self,
        batch: dict,
    ) -> Dict:
        """
        Declare what the model should do during a training step using a given batch.

        Args:
            batch: The batch of real data.

        Return:
            A dict containing the optimization metrics which shall be logged.
        """
        self.classifier.train()

        real_data = batch['data'].to(self.device)
        real_label = batch['label'].to(self.device)

        prediction = self.classifier(real_data)

        loss_per_sample = self.criterion.forward(prediction, real_label.long())

        avg_loss = scale_weights(real_label, loss_per_sample)

        self.optimizer.optimize(avg_loss)

        return {'training/loss': float(avg_loss.data)}

    def validation_step(
        self,
        batch: dict,
    ) -> dict:
        """Declare what the model should do during a validation step."""
        real_data = batch['data'].to(self.device)
        real_label = batch['label'].to(self.device)

        prediction = self.classify(real_data)
        with no_grad():
            vali_loss_per_sample = self.criterion.forward(prediction, real_label.long())
            avg_vali_loss = scale_weights(real_label, vali_loss_per_sample)

        real_label = real_label.cpu().numpy()
        prediction_labels = argmax(prediction, dim=1).cpu().numpy()

        metrics_dict = self.get_classification_metrics(
            real_label,
            prediction_labels,
            stage='validation/',
            get_prec_recall_fscore=True,
            get_fscore_micro=True,
            get_fscore_weighted=True,
            get_accuracy=True,
            get_mcc=True,
            get_auroc=True,
        )
        metrics_dict['validation/vali_loss'] = float(avg_vali_loss.data)

        return metrics_dict

    def save_checkpoint(self) -> dict:
        """Return current model parameters."""
        return {
            'NN': self.classifier.state_dict(),
            'NN_OPT': self.optimizer.state_dict(),
        }

    def load(self, model_reference: str, load_optim: bool = False):
        """Load a trained module from disk (file path) or wand reference."""
        model = load_model(model_reference, self.device)
        self.classifier.load_state_dict(model['NN'], strict=False)

        if load_optim:
            self.optimizer.load_existing_optim(model['NN_OPT'])

        logger.info('Loading existing model completed.')
        return self

    @property
    def watch_list(self) -> List[nn.Module]:
        """Return models that should be watched during training."""
        return [self.classifier]

    def on_epoch_end(self, epoch: int, sample_interval: int, batch_size: int) -> List[Artifact]:
        """
        Set actions to be executed after epoch ends.

        Declare what should be done upon finishing an epoch (e.g. save artifacts or
        evaluate some metric).
        """
        return []


class CNNClassifier(NNClassifier):
    """Argmax CNN classifier."""

    def _init_classifier(self, cfg: BaseNNConfig) -> nn.Module:
        """Initialize the CNN."""
        if not isinstance(cfg.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN classifier with config {}.".format(type(cfg)))
        num_classes = 1 if self.train_cfg.TRAIN_ONLY_NORMAL else self.dataset.num_classes
        cnn = ConvolutionalNeuralNetwork(
            input_channels=self.num_channels,
            hidden_channels=cfg.LAYER_SPECIFICATION.HIDDEN_CHANNELS,
            out_channels=num_classes,
            n_classes=num_classes,
            seq_len=self.seq_len,
            input_norm=InputNormalization.BATCH,
        )
        initialize_weights(cnn, self.cfg.WEIGHT_INIT)

        return cnn

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration of a standard CNN classifier."""
        config = ConvolutionalNeuralNetwork.configure()
        config['module'].update(CrossEntropyLoss.configure())
        config['module'].update(Adam.configure())

        return config


class RNNClassifier(NNClassifier):
    """Argmax RNN classifier."""

    def _init_classifier(self, cfg: BaseNNConfig) -> nn.Module:
        """Initialize the RNN."""
        if not isinstance(cfg.LAYER_SPECIFICATION, BaseRNNConfig):
            raise RuntimeError("Cannot instantiate RNN classifier with config {}.".format(type(cfg)))

        num_classes = self.dataset.NUM_CLASSES_BINARY if self.train_cfg.TRAIN_ONLY_NORMAL else self.dataset.num_classes
        rnn = RecurrentNeuralNetwork(
            num_channels=self.num_channels,
            hidden_dim=cfg.LAYER_SPECIFICATION.HIDDEN_DIMS,
            hidden_size=cfg.LAYER_SPECIFICATION.HIDDEN_SIZE,
            n_classes=num_classes,
        )
        initialize_weights(rnn, cfg.WEIGHT_INIT)

        return rnn

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration of a standard RNN classifier."""
        config = RecurrentNeuralNetwork.configure()
        config['module'].update(CrossEntropyLoss.configure())
        config['module'].update(Adam.configure())

        config['update'] = {
            'trainer': {
                'TRAIN_ONLY_NORMAL': False,
                'BINARY_LABELS': False,
                'SPLIT': (0.7, 0.15, 0.15),
            }
        }

        return config
