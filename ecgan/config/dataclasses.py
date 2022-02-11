"""Custom (partially nested) dataclasses describing configurations of individual components."""
# pylint: disable=C0103
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from ecgan.config.nested_dataclass import nested_dataclass
from ecgan.utils.custom_types import (
    DiscriminationStrategy,
    LatentDistribution,
    MetricOptimization,
    SamplingAlgorithm,
    TrackerType,
    Transformation,
    WeightInitialization,
)
from ecgan.utils.miscellaneous import generate_seed


@dataclass
class OptimizerConfig:
    """Type hints for Optimizer dicts."""

    _name = 'optimizer'
    NAME: str
    LR: float  # Learning rate
    WEIGHT_DECAY: Optional[float] = None
    MOMENTUM: Optional[float] = None
    DAMPENING: Optional[float] = None
    BETAS: Optional[Tuple[float, float]] = None
    EPS: Optional[float] = None
    ALPHA: Optional[float] = None
    CENTERED: Optional[bool] = None


@nested_dataclass
class InverseModuleConfig:
    """Type hints for the module config of an inverse mapping module."""

    KERNEL_SIZES: List[int]
    LOSS: str
    NAME: str
    OPTIMIZER: OptimizerConfig


@nested_dataclass
class ReconstructionConfig:
    """Type hints for ReconstructionType dicts."""

    STRATEGY: str


@nested_dataclass
class EmbeddingConfig:
    """Type hints for ReconstructionType dicts."""

    CREATE_UMAP: bool
    LOAD_PRETRAINED_UMAP: bool


@nested_dataclass
class LatentWalkReconstructionConfig(ReconstructionConfig):
    """Type hints for latent walk reconstructions."""

    MAX_RECONSTRUCTION_ITERATIONS: int
    EPSILON: float
    LATENT_OPTIMIZER: OptimizerConfig
    CRITERION: str
    ADAPT_LR: bool
    LR_THRESHOLD: float
    VERBOSE_STEPS: Optional[int] = None


@dataclass
class LossConfig:
    """Type hints for a generic loss configuration."""

    NAME: str
    GRADIENT_PENALTY_WEIGHT: Optional[float] = None
    CLIPPING_BOUND: Optional[float] = None
    REDUCTION: Optional[str] = None


@dataclass
class BaseCNNConfig:
    """Generalized configuration of an CNN module."""

    HIDDEN_CHANNELS: List[int]


@dataclass
class BaseRNNConfig:
    """Generalized configuration of an RNN module."""

    HIDDEN_DIMS: int  # Amount of layers
    HIDDEN_SIZE: int  # Size of each layer


@dataclass
class TrackingConfig:
    """Config for tracking and logging information."""

    TRACKER_NAME: str
    ENTITY: str
    PROJECT: str
    EXPERIMENT_NAME: str
    LOCAL_SAVE: bool
    SAVE_PDF: bool
    S3_CHECKPOINT_UPLOAD: bool  # Currently only supported for W&B tracker
    LOG_LEVEL: str = 'info'

    @property
    def tracker_name(self) -> TrackerType:
        return TrackerType(self.TRACKER_NAME)


@nested_dataclass
class ExperimentConfig:
    """
    Parameters regarding the experiment itself.

    Includes information on the experiment, the used dataset and the directory from where the dataset is loaded.
    """

    _name = 'experiment'
    TRACKER: TrackingConfig
    DATASET: str
    MODULE: str
    LOADING_DIR: str
    TRAIN_ON_GPU: bool

    @staticmethod
    def configure(  # pylint: disable=R0913
        entity: str,
        project: str,
        experiment_name: str,
        module: str,
        dataset: str,
        tracker: str = TrackerType.LOCAL.value,
        local_save: bool = False,
        save_pdf: bool = False,
        loading_dir: str = 'data',
        train_on_gpu: bool = True,
        s3_checkpoint_upload: bool = False,
        log_level: str = 'info',
    ) -> Dict:
        """Return a default experiment configuration."""
        return {
            'experiment': {
                'TRACKER': {
                    'TRACKER_NAME': tracker,
                    'PROJECT': project,
                    'EXPERIMENT_NAME': experiment_name,
                    'ENTITY': entity,
                    'LOCAL_SAVE': local_save,
                    'SAVE_PDF': save_pdf,
                    'S3_CHECKPOINT_UPLOAD': s3_checkpoint_upload,
                    'LOG_LEVEL': log_level,
                },
                'MODULE': module,
                'DATASET': dataset,
                'LOADING_DIR': loading_dir,
                'TRAIN_ON_GPU': train_on_gpu,
            }
        }

    @property
    def name(self):
        return self._name


@dataclass
class PreprocessingConfig:
    """Create a preprocessing config object."""

    _name = 'preprocessing'
    LOADING_DIR: str
    NUM_WORKERS: int
    WINDOW_LENGTH: int
    WINDOW_STEP_SIZE: int
    RESAMPLING_ALGORITHM: SamplingAlgorithm
    TARGET_SEQUENCE_LENGTH: int
    LOADING_SRC: Optional[str]
    NUM_SAMPLES: int

    @staticmethod
    def configure(
        loading_src: Optional[str],
        target_sequence_length: int,
        loading_dir: str = 'data',
        num_workers: int = 4,
        window_length: int = 0,
        window_step_size: int = 0,
        resampling_algo: str = 'lttb',
        num_samples: int = 0,
    ):
        """Return a default preprocessing configuration."""
        return {
            'preprocessing': {
                'LOADING_DIR': loading_dir,
                'LOADING_SRC': loading_src,
                'NUM_WORKERS': num_workers,
                'WINDOW_LENGTH': window_length,
                'WINDOW_STEP_SIZE': window_step_size,
                'RESAMPLING_ALGORITHM': resampling_algo,
                'TARGET_SEQUENCE_LENGTH': target_sequence_length,
                'NUM_SAMPLES': num_samples,
            }
        }

    @property
    def name(self):
        return self._name

    @property
    def resampling_algorithm(self) -> SamplingAlgorithm:
        return SamplingAlgorithm(self.RESAMPLING_ALGORITHM)


@dataclass
class SyntheticPreprocessingConfig(PreprocessingConfig):
    """Preprocessing configuration for synthetic datasets."""

    RANGE: Tuple[int, int]
    ANOMALY_PERCENTAGE: float
    NOISE_PERCENTAGE: float
    SYNTHESIS_SEED: int

    @staticmethod
    def configure(  # pylint: disable=R0913, W0221
        loading_src: Optional[str],
        target_sequence_length: int,
        loading_dir: str = 'data',
        num_workers: int = 4,
        window_length: int = 0,
        window_step_size: int = 0,
        resampling_algo: str = 'lttb',
        num_samples: int = 0,
        data_range: Tuple[int, int] = (0, 25),
        anomaly_percentage: float = 0.2,
        noise_percentage: float = 0.5,
        synthesis_seed: int = 1337,
    ) -> Dict:
        """Provide a default configuration for a synthetic dataset."""
        result_dict: Dict = PreprocessingConfig.configure(
            loading_src=loading_src,
            target_sequence_length=target_sequence_length,
            loading_dir=loading_dir,
            num_workers=num_workers,
            window_length=window_length,
            window_step_size=window_step_size,
            resampling_algo=resampling_algo,
            num_samples=num_samples,
        )

        update_dict: Dict = {
            "RANGE": data_range,
            "ANOMALY_PERCENTAGE": anomaly_percentage,
            "NOISE_PERCENTAGE": noise_percentage,
            "SYNTHESIS_SEED": synthesis_seed,
        }

        result_dict['preprocessing'].update(update_dict)

        return result_dict


@dataclass
class SinePreprocessingConfig(SyntheticPreprocessingConfig):
    """Preprocessing config for the synthetic sine dataset."""

    AMPLITUDE: float = 3.0
    FREQUENCY: float = 3.0
    PHASE: float = 5.0
    VERTICAL_TRANSLATION: float = 1.0

    @staticmethod
    def configure(  # pylint: disable=W0221, R0913
        loading_src: Optional[str],
        target_sequence_length: int,
        loading_dir: str = 'data',
        num_workers: int = 4,
        window_length: int = 0,
        window_step_size: int = 0,
        resampling_algo: str = 'lttb',
        num_samples: int = 0,
        data_range: Tuple[int, int] = (0, 25),
        anomaly_percentage: float = 0.2,
        noise_percentage: float = 0.5,
        synthesis_seed: int = 1337,
        amplitude: float = 3,
        frequency: float = 3,
        phase: float = 5,
        vertical_translation: float = 1,
    ) -> Dict:
        """Return the default configuration for the sine dataset."""
        result_dict = SyntheticPreprocessingConfig.configure(
            loading_src=loading_src,
            target_sequence_length=target_sequence_length,
            loading_dir=loading_dir,
            num_workers=num_workers,
            window_length=window_length,
            window_step_size=window_step_size,
            resampling_algo=resampling_algo,
            num_samples=num_samples,
            data_range=data_range,
            anomaly_percentage=anomaly_percentage,
            noise_percentage=noise_percentage,
            synthesis_seed=synthesis_seed,
        )

        update_dict = {
            "AMPLITUDE": amplitude,
            "FREQUENCY": frequency,
            "PHASE": phase,
            "VERTICAL_TRANSLATION": vertical_translation,
        }

        result_dict['preprocessing'].update(update_dict)

        return result_dict


@dataclass
class TrainerConfig:
    """Used to initialize a config for training."""

    _name = "trainer"
    NUM_WORKERS: int
    CHANNELS: Union[int, List[int]]
    EPOCHS: int
    BATCH_SIZE: int
    TRANSFORMATION: str
    SPLIT_PATH: str
    SPLIT_METHOD: str
    SPLIT: Tuple[float, float]
    TRAIN_ONLY_NORMAL: bool
    CROSS_VAL_FOLDS: int
    CHECKPOINT_INTERVAL: int
    SAMPLE_INTERVAL: int
    BINARY_LABELS: bool
    MANUAL_SEED: int

    @staticmethod
    def configure(  # pylint: disable=R0913
        transformation: Transformation = Transformation.NONE,
        num_workers: int = 4,
        epochs: int = 500,
        batch_size: int = 64,
        split_path: str = 'split.pkl',
        split_method: str = 'random',
        split: Tuple[float, float] = (0.85, 0.15),
        cross_val_folds: int = 5,
        checkpoint_interval: int = 10,
        sample_interval: int = 1,
        train_only_normal: bool = True,
        binary_labels: bool = True,
        channels: Union[int, List[int]] = 0,
        manual_seed: int = generate_seed(),
    ):
        """Return a default configuration for the trainer."""
        return {
            'trainer': {
                'NUM_WORKERS': num_workers,
                'CHANNELS': channels,
                'EPOCHS': epochs,
                'BATCH_SIZE': batch_size,
                'TRANSFORMATION': transformation.value,
                'SPLIT_PATH': split_path,
                'SPLIT_METHOD': split_method,
                'SPLIT': split,
                'CROSS_VAL_FOLDS': cross_val_folds,
                'CHECKPOINT_INTERVAL': checkpoint_interval,
                'SAMPLE_INTERVAL': sample_interval,
                'TRAIN_ONLY_NORMAL': train_only_normal,
                'BINARY_LABELS': binary_labels,
                'MANUAL_SEED': manual_seed,
            }
        }

    @property
    def name(self):
        return self._name

    @property
    def transformation(self) -> Transformation:
        """Return an instance of the internal enum class `Transformation`."""
        return Transformation(self.TRANSFORMATION)


@dataclass
class WeightInitializationConfig:
    """Base weight initialization config."""

    NAME: str

    @property
    def weight_init_type(self) -> WeightInitialization:
        return WeightInitialization(self.NAME)


@dataclass
class NormalInitializationConfig(WeightInitializationConfig):
    """Base weight initialization config for drawing from a normal distribution."""

    MEAN: float
    STD: float


@dataclass
class UniformInitializationConfig(WeightInitializationConfig):
    """Base weight initialization config for drawing from a uniform distribution."""

    LOWER_BOUND: float
    UPPER_BOUND: float


@nested_dataclass
class ModuleConfig:
    """Generalized configuration of a module."""

    _name = "module"

    @property
    def name(self):
        return self._name


@nested_dataclass
class BaseNNConfig(ModuleConfig):
    """Generic neural network configuration."""

    OPTIMIZER: OptimizerConfig
    LOSS: LossConfig
    LAYER_SPECIFICATION: Union[BaseCNNConfig, BaseRNNConfig]
    WEIGHT_INIT: Union[WeightInitializationConfig, NormalInitializationConfig, UniformInitializationConfig]
    SPECTRAL_NORM: bool = False
    INPUT_NORMALIZATION: Optional[str] = None


@nested_dataclass
class AutoEncoderConfig(ModuleConfig):
    """Generalized configuration of a AE module."""

    LATENT_SIZE: int
    ENCODER: BaseNNConfig
    DECODER: BaseNNConfig
    TANH_OUT: bool
    LATENT_SPACE: str

    @property
    def latent_distribution(self) -> LatentDistribution:
        """Convenience conversion to internal enum type."""
        return LatentDistribution(self.LATENT_SPACE)


@nested_dataclass
class VariationalAutoEncoderConfig(AutoEncoderConfig):
    """Generalized configuration of a VAE module."""

    KL_BETA: float


@nested_dataclass
class GeneratorConfig(BaseNNConfig):
    """Generic generator configuration."""

    TANH_OUT: bool = False


@nested_dataclass
class GANModuleConfig(ModuleConfig):
    """Generalized configuration of a GAN module."""

    LATENT_SIZE: int
    GENERATOR: GeneratorConfig
    DISCRIMINATOR: BaseNNConfig
    GENERATOR_ROUNDS: int
    DISCRIMINATOR_ROUNDS: int
    LATENT_SPACE: str

    @property
    def latent_distribution(self) -> LatentDistribution:
        """Convenience conversion to internal enum type."""
        return LatentDistribution(self.LATENT_SPACE)


@nested_dataclass
class EncoderGANConfig(GANModuleConfig):
    """Generalized configuration for the BeatGAN module."""

    ENCODER: BaseNNConfig


@nested_dataclass
class VAEGANConfig(EncoderGANConfig):
    """VAEGAN config."""

    KL_WARMUP: int
    KL_ANNEAL_ROUNDS: int
    KL_BETA: int


@nested_dataclass
class AdExperimentConfig:
    """Basic experimental settings for the anomaly detection process."""

    _name = 'ad_experiment'
    TRACKER: TrackingConfig
    RUN_URI: str
    RUN_VERSION: str
    FOLD: int
    SAVE_DIR: str

    @property
    def name(self):
        return self._name


@dataclass
class DetectionConfig:
    """Generalized configuration of a detection object."""

    _name = "detection"
    DETECTOR: str
    BATCH_SIZE: int
    NUM_WORKERS: int
    AMOUNT_OF_RUNS: int
    SAVE_DATA: bool

    @property
    def name(self) -> str:
        return self._name


@nested_dataclass
class ReconstructionDetectionConfig(DetectionConfig):
    """Generalized configuration of a reconstruction based detection config."""

    EMBEDDING: EmbeddingConfig


@nested_dataclass
class GANDetectorConfig(ReconstructionDetectionConfig):
    """Base config for GAN based anomaly detection."""

    DISCRIMINATION_STRATEGY: str
    AD_SCORE_STRATEGY: str
    NORMALIZE_ERROR: bool
    RECONSTRUCTION: Union[ReconstructionConfig, LatentWalkReconstructionConfig]

    @property
    def ad_score_strategy(self) -> MetricOptimization:
        return MetricOptimization(self.AD_SCORE_STRATEGY)

    @property
    def discrimination_strategy(self) -> DiscriminationStrategy:
        return DiscriminationStrategy(self.DISCRIMINATION_STRATEGY)


@nested_dataclass
class InverseDetectorConfig(GANDetectorConfig):
    """Config for anomaly detectors utilizing GAN inversion."""

    RECONSTRUCTION: ReconstructionConfig
    INVERSE_MAPPING_URI: Optional[str]


@nested_dataclass
class GANLatentWalkConfig(GANDetectorConfig):
    """Config for anomaly detectors utilizing latent walks to approximate the reconstructed series."""

    RECONSTRUCTION: LatentWalkReconstructionConfig
    INVERSE_MAPPING_URI: Optional[str]
