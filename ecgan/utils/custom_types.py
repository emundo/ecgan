"""Custom enums of supported functionality and types of for the ecgan library."""
from enum import Enum
from typing import Any, List, Tuple, Union

from torch import Tensor

# List of metrics which can trigger `save_epoch` if they are improved.
# TODO read from config
save_epoch_metrics = [
    'grid/lambda_fscore',
    'grid/lambda_mcc',
    'grid/lambda_auroc',
    'svm/minmax_fscore',
    'svm/minmax_auroc',
    'svm/minmax_mcc',
    'svm/scaled_latent_mcc',
    'svm/scaled_latent_fscore',
    'svm/scaled_latent_auroc',
    'only_disc_error',
    'only_reconstruction_error',
    'only_latent_error',
]


class AnomalyDetectionStrategies(Enum):
    """Implemented strategies for anomaly detection."""

    ANOGAN = 'anogan'
    ARGMAX = 'argmax'
    INVERSE_MAPPING = 'inverse_mapping'


class Transformation(Enum):
    """Implemented transformations which can be set before passing data to a module."""

    STANDARDIZE = 'standard'
    MINMAX = 'minmax'
    WHITENING = 'whitening'
    NONE = 'none'
    FOURIER = 'fourier'
    INDIVIDUAL = 'individual'


class SplitMethods(Enum):
    """Available split methods."""

    MIXED = 'mixed'
    NORMAL_ONLY = 'normal_only'


class SampleDataset(Enum):
    """Available datasets a DatasetSampler can sample from."""

    TRAIN = 'train'
    TEST = 'test'
    VALI = 'vali'


class InverseMappingType(Enum):
    """Available inverse mappings."""

    SIMPLE = 'simple'


class ReconstructionType(Enum):
    """Different ways to reconstruct data using an already trained GAN generator."""

    INTERPOLATE = 'interpolate'
    INVERSE_MAPPING = 'inverse_mapping'


class DiscriminationStrategy(Enum):
    """Different ways to discriminate data using an already trained GAN discriminator."""

    FEATURE_MATCHING = 'feature_matching'
    TARGET_VALUE = 'target_value'
    LOG = 'log_discrimination'


class SimilarityMeasures(Enum):
    """Methods to calculate the similarity between two series."""

    RBF_KERNEL = 'rbf_kernel'
    COSINE = 'cosine_similarity'


class Optimizers(Enum):
    """Supported optimizers."""

    UNDEFINED = 'undefined_optimizer'
    STOCHASTIC_GRADIENT_DESCENT = 'sgd'
    MOMENTUM = 'momentum'
    ADAM = 'adam'
    RMS_PROP = 'rms_prop'
    ADABELIEF = 'ada_belief'


class MetricOptimization(Enum):
    """Supported methods to optimize weighted errors regarding their weighting for a total anomaly score."""

    NONE = None
    SVM_LAMBDA = 'SVM_LAMBDA'
    SVM_LAMBDA_GAMMA = 'SVM_LAMBDA_GAMMA'
    GRID_SEARCH_LAMBDA = 'GRID_SEARCH_LAMBDA'
    GRID_SEARCH_LAMBDA_GAMMA = 'GRID_SEARCH_LAMBDA_GAMMA'
    RECONSTRUCTION_ERROR = 'RECONSTRUCTION_ERROR'


class Losses(Enum):
    """Supported losses."""

    UNDEFINED = 'undefined_loss'
    BCE = 'bce'
    BCE_GENERATOR_LOSS = 'bce_generator_loss'
    L2 = "l2"
    TWO_COMPONENT_BCE = 'two_component_bce'
    KL_DIV = 'kullback_leibler_divergence'
    WASSERSTEIN_DISCRIMINATOR = 'wasserstein_discriminator'
    WASSERSTEIN_GENERATOR = 'wasserstein_generator'
    CROSS_ENTROPY = 'cross_entropy'
    AEGAN_GENERATOR = 'aegan_generator'
    AEGAN_DISCRIMINATOR = 'aegan_discriminator'
    VAEGAN_GENERATOR = 'vaegan_generator'
    VAE = 'vae'
    AUTOENCODER = 'autoencoder'


class LabelingStrategy(Enum):
    """
    Determine how the points shall be labeled.

    WARNING: The strategy chosen has implications on the output format.
    POINTWISE returns a label for each datapoint in each series.
    ACCUMULATE_UNIVARIATE returns a label for each univariate series.
    ACCUMULATE_MULTIVARIATE returns a label for each multivariate series.
    """

    POINTWISE = 'pointwise'
    ACCUMULATE_CHANNELWISE = 'accumulate_channelwise'
    ACCUMULATE_SERIESWISE = 'accumulate_serieswise'
    VARIANCE_SERIESWISE = 'variance_serieswise'
    VARIANCE_CHANNELWISE = 'variance_channelwise'


class SamplingAlgorithm(Enum):
    """Different down- or upsampling algorithms which can be used during preprocessing."""

    LTTB = 'lttb'
    INTERPOLATE = 'interpolate'
    FIXED_DOWNSAMPLING_RATE = 'fixed_rate'


class MetricType(Enum):
    """Supported evaluation metrics."""

    FSCORE = 'fscore'
    MCC = 'mcc'
    AUROC = 'auroc'
    AP = 'average_precision'


class WeightInitialization(Enum):
    """Different strategies to inititalize weights in neural networks."""

    NORMAL = 'normal'
    UNIFORM = 'uniform'
    HE = 'he'
    GLOROT_UNIFORM = 'glorot_uniform'
    GLOROT_NORMAL = 'glorot_normal'


class InputNormalization(Enum):
    """Supported normalization layers."""

    BATCH = 'batch'
    GROUP = 'group'
    NONE = 'none'


class LatentDistribution(Enum):
    """Supported latent distributions which can be set via config."""

    NORMAL = 'normal_distribution'
    NORMAL_TRUNCATED = 'truncated_normal_distribution'
    UNIFORM = 'uniform_distribution'
    ENCODER_BASED = 'encoder'


class TrackerType(Enum):
    """Supported trackers that can be set via config."""

    LOCAL = 'local'
    WEIGHTS_AND_BIASES = 'wb'


class PlotterType(Enum):
    """Different types of plotters."""

    BASE = 'base'
    FOURIER = 'fourier'
    SCATTER = 'scatter'


class SklearnAveragingOptions(Enum):
    """
    Additional Sklearn type stubs for the available F-score averaging options.

    More information on the respective effects:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    """

    BINARY = 'binary'
    MICRO = 'micro'
    MACRO = 'macro'
    SAMPLES = 'samples'
    WEIGHTED = 'weighted'
    NONE = None


class SklearnSVMKernels(Enum):
    """
    SVM kernels supported by ECGAN from sklearn.

    More informations can be found in the official [docs]
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    """

    LINEAR = 'linear'
    POLY = 'poly'
    RBF = 'rbf'
    SIGMOID = 'sigmoid'


class SupportedModules(Enum):
    """Modules supported by the framework."""

    RGAN = 'rgan'
    DCGAN = 'dcgan'
    RDCGAN = 'rdcgan'
    AEGAN = 'aegan'
    VAEGAN = 'vaegan'
    RNN = 'rnn'
    CNN = 'cnn'
    AUTOENCODER = 'autoencoder'
    VAE = 'variational_autoencoder'


# Optimizer can either handle a single tensor or multiple losses as a tuple (NAME, LossTensor).
LossType = Union[Tensor, List[Tuple[str, Tensor]]]
LossMetricType = List[Tuple[str, Any]]
