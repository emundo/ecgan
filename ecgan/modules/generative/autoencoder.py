"""
Implementation of a architecture using an autoencoder as generator.

No discriminator is used in this model, :ref:`ecgan.modules.generative.aegan` utilizes adversarial information.
"""
import logging
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from torch import Tensor, nn

from ecgan.config import AutoEncoderConfig, BaseCNNConfig, OptimizerConfig, nested_dataclass_asdict
from ecgan.evaluation.optimization import optimize_metric, optimize_tau_single_error, retrieve_labels_from_weights
from ecgan.modules.generative.base import BaseGenerativeModule
from ecgan.networks.beatgan import BeatganGenerator, BeatganInverseEncoder
from ecgan.utils.artifacts import Artifact, ImageArtifact, ValueArtifact
from ecgan.utils.custom_types import InputNormalization, LatentDistribution, MetricType, WeightInitialization
from ecgan.utils.distances import L2Distance
from ecgan.utils.layers import initialize_batchnorm, initialize_weights
from ecgan.utils.losses import AEGANGeneratorLoss, AutoEncoderLoss, BceGeneratorLoss, BCELoss, L2Loss
from ecgan.utils.miscellaneous import load_model
from ecgan.utils.optimizer import Adam, BaseOptimizer, OptimizerFactory
from ecgan.utils.sampler import EncoderBasedGeneratorSampler
from ecgan.utils.transformation import MinMaxTransformation

logger = logging.getLogger(__name__)


class AutoEncoder(BaseGenerativeModule):
    """Basic autoencoder model."""

    BATCHNORM_MEAN = 1.0
    BATCHNORM_STD = 0.02
    BATCHNORM_BIAS = 0

    def __init__(
        self,
        cfg: AutoEncoderConfig,
        seq_len: int,
        num_channels: int,
    ):
        super().__init__(cfg, seq_len, num_channels)

        self.cfg: AutoEncoderConfig = cfg
        self.latent_size: int = self.cfg.LATENT_SIZE
        self.num_classes: int = self.dataset.NUM_CLASSES_BINARY
        self._decoder = self._init_decoder()
        self._decoder = nn.DataParallel(self.decoder)
        self._decoder.to(self.device)

        self._encoder = self._init_encoder()
        self._encoder = nn.DataParallel(self.encoder)
        self._encoder.to(self.device)

        self._optim = self._init_optim()
        self._autoencoder_sampler = self._init_autoencoder_sampler()

        self._criterion = self._init_criterion()

        self.num_fixed_samples: int = 8

        if self.cfg.latent_distribution == LatentDistribution.ENCODER_BASED:
            self.fixed_noise = torch.empty((self.num_fixed_samples, self.seq_len, self.latent_size)).to(self.device)
        else:
            raise ValueError("Encoder latent space ('encoder') needs to be used for autoencoder.")
        # Tensors which can be filled during train/validation. Can be reset using self._reset_internal_tensors.
        self.reconstruction_error = torch.empty(0).to(self.device)
        self.label = torch.empty(0).to(self.device)

        # Required for validation/testing.
        self.tau: float = 0.0  # Threshold for the reconstruction error

        self.seq_len = seq_len
        self.num_channels = num_channels
        self.mse_criterion = L2Loss()
        self.bce_criterion = BCELoss()
        self.data_sampler = self.train_dataset_sampler
        self.fixed_samples: Optional[Tensor] = None
        self.fixed_samples_labels: Optional[Tensor] = None

    @property
    def criterion(self) -> AEGANGeneratorLoss:
        return cast(AEGANGeneratorLoss, self._criterion)

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    @property
    def autoencoder_sampler(self) -> EncoderBasedGeneratorSampler:
        return self._autoencoder_sampler

    def _init_decoder(self) -> nn.Module:
        if not isinstance(self.cfg.DECODER.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN with config {}.".format(type(self.cfg)))

        model = BeatganGenerator(
            self.num_channels,
            self.cfg.DECODER.LAYER_SPECIFICATION.HIDDEN_CHANNELS,
            self.latent_size,
            self.seq_len,
            input_norm=InputNormalization(self.cfg.DECODER.INPUT_NORMALIZATION),
            spectral_norm=self.cfg.DECODER.SPECTRAL_NORM,
            tanh_out=self.cfg.TANH_OUT,
        )
        initialize_batchnorm(model, mean=self.BATCHNORM_MEAN, std=self.BATCHNORM_STD, bias=self.BATCHNORM_BIAS)
        initialize_weights(model, self.cfg.DECODER.WEIGHT_INIT)

        return model

    def _init_encoder(self) -> nn.Module:
        if not isinstance(self.cfg.ENCODER.LAYER_SPECIFICATION, BaseCNNConfig):
            raise RuntimeError("Cannot instantiate CNN with config {}.".format(type(self.cfg)))
        model = BeatganInverseEncoder(
            input_channels=self.num_channels,
            output_channels=self.latent_size,
            hidden_channels=self.cfg.ENCODER.LAYER_SPECIFICATION.HIDDEN_CHANNELS,
            seq_len=self.seq_len,
            input_norm=InputNormalization(self.cfg.ENCODER.INPUT_NORMALIZATION),
            spectral_norm=self.cfg.ENCODER.SPECTRAL_NORM,
        )

        initialize_batchnorm(model, mean=self.BATCHNORM_MEAN, std=self.BATCHNORM_STD, bias=self.BATCHNORM_BIAS)
        initialize_weights(model, self.cfg.ENCODER.WEIGHT_INIT)

        return model

    def _init_autoencoder_sampler(self) -> EncoderBasedGeneratorSampler:
        return EncoderBasedGeneratorSampler(
            component=self.decoder,
            encoder=self.encoder,
            dev=self.device,
            num_channels=self.num_channels,
            sampling_seq_length=1,
        )

    def _init_criterion(self) -> Any:
        return AutoEncoderLoss(
            cast(EncoderBasedGeneratorSampler, self.autoencoder_sampler),
            self.cfg.TANH_OUT,  # use MSE if Tanh out, use BCE if sig out
        )

    def _init_optim(self) -> BaseOptimizer:
        # Optimizes the generator as well as the encoder, no additional optim for the encoder is used.
        return OptimizerFactory()(
            chain(self.encoder.parameters(), self.decoder.parameters()),
            OptimizerConfig(**nested_dataclass_asdict(self.cfg.DECODER.OPTIMIZER)),
        )

    def get_sample(
        self, num_samples: Optional[int] = None, data: Optional[torch.Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve the reconstructed sampled x_hat from our model."""
        if data is None:
            raise RuntimeError("Data tensor may not be empty.")

        return self.autoencoder_sampler.sample_generator_encoder(data)  # type: ignore

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the autoencoder model."""
        config = {
            'module': {
                'LATENT_SIZE': 5,
                'LATENT_SPACE': LatentDistribution.ENCODER_BASED.value,
                'TANH_OUT': True,
                'DECODER': {
                    'LAYER_SPECIFICATION': {
                        'HIDDEN_CHANNELS': [512, 256, 128, 64, 32],
                    },
                    'INPUT_NORMALIZATION': InputNormalization.BATCH.value,
                    'SPECTRAL_NORM': False,
                    'WEIGHT_INIT': {
                        'NAME': WeightInitialization.GLOROT_NORMAL.value,
                    },
                },
            }
        }

        config['module'].update(BeatganInverseEncoder.configure())  # type: ignore
        config['module']['DECODER'].update(BceGeneratorLoss.configure())  # type: ignore
        config['module']['DECODER'].update(Adam.configure())  # type: ignore
        config['module']['DECODER']['OPTIMIZER']['BETAS'] = [0.5, 0.999]  # type: ignore

        return config

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
        self.encoder.train()
        self.decoder.train()

        real_data = batch['data'].to(self.device)
        try:
            metric_collection = []

            # Retrieve losses and update gradients
            losses, metrics = self.criterion(real_data)
            self._optim.optimize(losses)
            metric_collection.extend(metrics)

            return {key: float(value) for (key, value) in metric_collection}

        except TypeError as err:
            raise TypeError('Error during training: Config parameter was not correctly set.') from err

    def validation_step(
        self,
        batch: dict,
    ) -> dict:
        """Declare what the model should do during a validation step."""
        data = batch['data'].to(self.device)
        label = batch['label'].to(self.device)

        self.decoder.eval()
        self.encoder.eval()

        l2_distance = L2Distance()

        if self.fixed_samples is None:
            self.set_fixed_samples()

        with torch.no_grad():
            x_hat, _latent_vector = self.autoencoder_sampler.sample_generator_encoder(data=data)
            rec_error = l2_distance(data, x_hat)

        # concatenate tensors to form tensors for all batches in one epoch
        self.reconstruction_error = torch.cat((self.reconstruction_error, rec_error), dim=0)
        self.label = torch.cat((self.label, label), dim=0)

        return {}

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

    def save_checkpoint(self) -> dict:
        """Return current model parameters."""
        return {
            'ENC': self.encoder.state_dict(),
            'DEC': self.decoder.state_dict(),
            'OPT': self._optim.state_dict(),
            'ANOMALY_DETECTION': {
                'TAU': self.tau,
            },
        }

    def load(self, model_reference: str, load_optim: bool = False):
        """Load a trained module from existing model_reference."""
        model = load_model(model_reference, self.device)

        self.encoder.load_state_dict(model['ENC'], strict=False)
        self.decoder.load_state_dict(model['DEC'], strict=False)

        if load_optim:
            self._optim.load_existing_optim(model['OPT'])
        logger.info('Loading existing {0} model completed.'.format(self.__class__.__name__))

        self.tau = model['ANOMALY_DETECTION']['TAU']
        return self

    @property
    def watch_list(self) -> List[nn.Module]:
        """Return models that should be watched during training."""
        return [self.decoder, self.encoder]

    def on_epoch_end(self, epoch: int, sample_interval: int, batch_size: int) -> List[Artifact]:
        """
        Set actions to be executed after epoch ends.

        Declare what should be done upon finishing an epoch (e.g. save artifacts or evaluate some metric).
        """
        result: List[Artifact] = []

        # Min-max normalize error:
        scaler = MinMaxTransformation()
        self.reconstruction_error = scaler.fit_transform(self.reconstruction_error.unsqueeze(1)).squeeze()

        if epoch % sample_interval == 0:
            result.append(self._reconstruct_fixed_samples())
            if self.fixed_samples is None or self.fixed_samples_labels is None:
                raise RuntimeError("Fixed samples not set correctly.")

            mmd = self.get_mmd()
            result.append(ValueArtifact('generative_metric/mmd', mmd))
            tstr_dict = self.get_tstr()
            result.append(ValueArtifact('generative_metric/tstr', tstr_dict))

            tau_range = torch.linspace(0, 1, 50).cpu().tolist()
            result.append(
                ValueArtifact(
                    'only_reconstruction_error',
                    optimize_tau_single_error(self.label.cpu(), self.reconstruction_error.cpu(), tau_range),
                )
            )
        # Optimize F-score every 10 epochs
        if epoch % 10 == 0:
            best_params = optimize_metric(
                MetricType.FSCORE,
                errors=[self.reconstruction_error.cpu()],
                taus=torch.linspace(0, 1, 50).numpy().tolist(),
                params=[],
                ground_truth_labels=self.label.cpu(),
            )
            logger.info(
                "Best params: {} for data {}.".format(best_params, torch.unique(self.label, return_counts=True))
            )
            self.tau = best_params[0][1]
            result.append(
                ValueArtifact(
                    'grid/lambda_tau',
                    float(self.tau),
                )
            )

            predictions = retrieve_labels_from_weights(
                errors=[self.reconstruction_error.cpu()],
                tau=self.tau,
                weighting_params=[],
            )
            result.extend(
                self._get_metrics(self.label, predictions, 'grid/lambda', log_fscore=True, log_auroc=True, log_mcc=True)
            )

        self._reset_internal_tensors()
        return result

    def _reconstruct_fixed_samples(self):
        if self.fixed_samples is None or self.fixed_samples_labels is None:
            raise RuntimeError("Fixed samples not set correctly.")

        with torch.no_grad():
            faked_samples, _ = self.autoencoder_sampler.sample_generator_encoder(data=self.fixed_samples)

        if self.cfg.TANH_OUT:
            faked_samples = (faked_samples / 2) + 0.5

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

    def _reset_internal_tensors(self):
        """Reset tensors which are filled internally during an epoch."""
        self.reconstruction_error = torch.empty(0).to(self.device)
        self.label = torch.empty(0).to(self.device)
