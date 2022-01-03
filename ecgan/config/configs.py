"""Includes the config class, its default parameters and directly related factories."""
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Dict, Optional, Type, Union

import yaml

from ecgan.config.dataclasses import (
    AdExperimentConfig,
    AutoEncoderConfig,
    BaseNNConfig,
    DetectionConfig,
    EncoderGANConfig,
    ExperimentConfig,
    GANLatentWalkConfig,
    GANModuleConfig,
    InverseDetectorConfig,
    InverseModuleConfig,
    ModuleConfig,
    PreprocessingConfig,
    SinePreprocessingConfig,
    TrainerConfig,
    VAEGANConfig,
    VariationalAutoEncoderConfig,
)
from ecgan.config.nested_dataclass import nested_dataclass, nested_dataclass_asdict
from ecgan.utils.custom_types import AnomalyDetectionStrategies, SupportedModules, TrackerType
from ecgan.utils.datasets import SineDataset
from ecgan.utils.miscellaneous import get_num_workers, load_yml


class Config(ABC):
    """Base class for configurations which can be used to generate persistent config files or read from one."""

    _NAME = 'BaseConfig'

    def __init__(
        self,
        base_config: Optional[Union[str, Dict]] = 'config.yml',
        output_file: str = 'config.yml',
    ):
        if isinstance(base_config, str):
            self._config_dict = load_yml(base_config)
        elif base_config is None:
            self._config_dict = {}
            return
        else:
            self._config_dict = base_config
        self.file_name = output_file

    @property
    def config_dict(self):
        return self._config_dict

    @config_dict.setter
    def config_dict(self, value: dict):
        self._config_dict = value

    @abstractmethod
    def _update_internal_config_dict(self):
        pass

    def generate_config_file(self):
        """Generate a default configuration file with dummy values."""
        self._update_internal_config_dict()
        with open(self.file_name, 'w', encoding='utf-8') as outfile:
            for key, value in self.config_dict.items():
                outfile.write('################################################\n')
                outfile.write('### ' + key.upper() + '\n')
                outfile.write('################################################\n')
                yaml.dump({key: value}, outfile, indent=4, default_flow_style=False)
                outfile.write('\n')

    def get_property(self, property_name: str):  # noqa: D102
        if self._config_dict is None:
            return None
        if property_name not in self._config_dict.keys():
            return None
        return self._config_dict[property_name]


class PreprocessingConfigFactory:
    """Factory for preprocessing configs."""

    @staticmethod
    def choose_class(dataset: str) -> Type[PreprocessingConfig]:
        """Choose the correct class based on the provided dataset name."""
        if dataset == SineDataset.name:
            return SinePreprocessingConfig

        return PreprocessingConfig

    def __call__(self, dataset: str, **kwargs) -> PreprocessingConfig:
        """Return implemented module when a GANModule is created."""
        cls: Type[PreprocessingConfig] = PreprocessingConfigFactory.choose_class(dataset)

        # Ignore:  https://github.com/python/mypy/issues/5485
        return cls(**kwargs)


class TrainConfig(Config):
    """
    Create a config object.

    Creates config no base_config has yet been created.
    """

    def __init__(
        self,
        base_config: Optional[Union[str, Dict]] = 'config.yml',
        output_file: str = 'config.yml',
    ):
        super().__init__(base_config, output_file)

        self._name: Optional[str] = None
        if self.get_property('experiment') is not None:
            self._experiment_config = ExperimentConfig(**self.get_property('experiment'))  # type: ignore
        if self.get_property('trainer') is not None:
            self._trainer_config = TrainerConfig(**self.get_property('trainer'))
        if self.get_property('preprocessing') is not None:
            preprocessing_factory = PreprocessingConfigFactory()
            self._preprocessing_config = preprocessing_factory(
                self.experiment_config.DATASET, **self.get_property('preprocessing')
            )
        if self.get_property('module') is not None:
            module_factory = ModuleConfigFactory()
            self._module_config = module_factory(self.experiment_config.MODULE, **self.get_property('module'))

    def _update_internal_config_dict(self):
        self.config_dict.update({self.experiment_config.name: nested_dataclass_asdict(self._experiment_config)})
        self.config_dict.update({self.trainer_config.name: asdict(self._trainer_config)})
        self.config_dict.update({self.preprocessing_config.name: asdict(self._preprocessing_config)})
        self.config_dict.update({self.module_config.name: nested_dataclass_asdict(self._module_config)})

    @property
    def experiment_config(self) -> ExperimentConfig:
        return self._experiment_config

    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        return self._preprocessing_config

    @property
    def trainer_config(self) -> TrainerConfig:
        return self._trainer_config

    @property
    def module_config(self) -> ModuleConfig:
        return self._module_config


class InverseConfig(Config):
    """Configuration for an inverse mapping."""

    _NAME = 'inverse'

    def __init__(
        self,
        base_config: Optional[Union[str, Dict]] = 'config.yml',
        output_file: str = 'config.yml',
    ):
        if base_config is None:
            return
        super().__init__(base_config, output_file)
        # Ignore:  https://github.com/python/mypy/issues/5485
        self.attribs = InverseConfig.Attribs(**self.get_property(self._NAME))  # type: ignore

    @nested_dataclass
    class Attribs:
        """Attributes of the inverse config."""

        _name = 'inverse'
        RUN_URI: str
        FOLD: int
        RUN_VERSION: str
        INV_MODULE: InverseModuleConfig
        EPOCHS: int
        ROUNDS: int
        BATCH_SIZE: int
        ARTIFACT_CHECKPOINT: int
        SAVE_CHECKPOINT: int
        GPU: bool

    def _update_internal_config_dict(self):
        pass


class AnomalyDetectionConfig(Config):
    """Configuration used to detect anomalies."""

    _NAME = 'anomaly'

    def __init__(
        self,
        base_config: Optional[Union[str, Dict]] = 'ad_config.yml',
        output_file: str = 'ad_config.yml',
    ):
        if base_config is None:
            return
        super().__init__(base_config, output_file)
        if self.get_property('ad_experiment') is not None:
            self._ad_experiment_config = AdExperimentConfig(**self.get_property('ad_experiment'))  # type: ignore
        if self.get_property('detection') is not None:
            factory = DetectionConfigFactory()
            self._detection_config = factory(
                self.get_property('detection')['DETECTOR'], **self.get_property('detection')
            )

    @staticmethod
    def configure(
        entity: str,
        project: str,
        name: str,
        run_path: str,
        fold: int,
        run_version: str = 'latest',
        save_locally: bool = False,
        save_pdf: bool = False,
        s3_checkpoint_upload: bool = False,
        log_level: str = 'info',
    ) -> Dict:
        """Return the default configuration for the anomaly detection."""
        # Each anomaly detector takes care of the detection config.
        config = {
            'ad_experiment': {
                'TRACKER': {
                    'TRACKER_NAME': TrackerType.LOCAL.value,
                    'ENTITY': entity,
                    'PROJECT': project,
                    'EXPERIMENT_NAME': name,
                    'LOCAL_SAVE': save_locally,
                    'SAVE_PDF': save_pdf,
                    'S3_CHECKPOINT_UPLOAD': s3_checkpoint_upload,
                    'LOG_LEVEL': log_level,
                },
                'RUN_URI': run_path,
                'RUN_VERSION': run_version,
                'FOLD': fold,
                'SAVE_DIR': 'results/',
            },
            'detection': {
                'DETECTOR': None,
                'BATCH_SIZE': 64,
                'NUM_WORKERS': get_num_workers(),
                'AMOUNT_OF_RUNS': 1,
            },
        }

        return config

    @property
    def ad_experiment_config(self) -> AdExperimentConfig:
        return self._ad_experiment_config

    @property
    def detection_config(self) -> DetectionConfig:
        return self._detection_config

    def _update_internal_config_dict(self):
        self.config_dict.update({self.ad_experiment_config.name: nested_dataclass_asdict(self._ad_experiment_config)})

        self.config_dict.update({self.detection_config.name: nested_dataclass_asdict(self._detection_config)})


class DetectionConfigFactory:
    """Create an instance of `ModuleConfig` depending on the provided module."""

    @staticmethod
    def choose_class(detector: str) -> Type[DetectionConfig]:
        """Choose the correct class based on the provided module name."""
        ad_configs = {
            AnomalyDetectionStrategies.ANOGAN.value: GANLatentWalkConfig,
            AnomalyDetectionStrategies.INVERSE_MAPPING.value: InverseDetectorConfig,
            AnomalyDetectionStrategies.ARGMAX.value: DetectionConfig,
        }
        try:
            return ad_configs[detector]
        except KeyError as err:
            raise AttributeError('Argument {0} is not set correctly.'.format(detector)) from err

    def __call__(self, detector: str, **kwargs) -> DetectionConfig:
        """Return implemented module when a GANModule is created."""
        cls: Type[DetectionConfig] = DetectionConfigFactory.choose_class(detector)
        # noinspection PyArgumentList
        return cls(**kwargs)


class ModuleConfigFactory:
    """Create an instance of `ModuleConfig` depending on the provided module."""

    @staticmethod
    def choose_class(module: str) -> Type[ModuleConfig]:
        """Choose the correct class based on the provided module name."""
        module_configs = {
            SupportedModules.DCGAN.value: GANModuleConfig,
            SupportedModules.RGAN.value: GANModuleConfig,
            SupportedModules.RDCGAN.value: GANModuleConfig,
            SupportedModules.AEGAN.value: EncoderGANConfig,
            SupportedModules.VAEGAN.value: VAEGANConfig,
            SupportedModules.RNN.value: BaseNNConfig,
            SupportedModules.CNN.value: BaseNNConfig,
            SupportedModules.AUTOENCODER.value: AutoEncoderConfig,
            SupportedModules.VAE.value: VariationalAutoEncoderConfig,
        }

        try:
            return module_configs[module]
        except KeyError as err:
            raise AttributeError('Argument {0} is not set correctly.'.format(module)) from err

    def __call__(self, module: str, **kwargs) -> ModuleConfig:
        """Return implemented module when a GANModule is created."""
        cls: Type[ModuleConfig] = ModuleConfigFactory.choose_class(module)

        # Ignore:  https://github.com/python/mypy/issues/5485
        return cls(**kwargs)  # type: ignore
