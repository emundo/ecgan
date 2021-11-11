"""Helper function to access configurations from previous runs."""
from typing import Optional, TypeVar, Union

from ecgan.config.configs import AnomalyDetectionConfig, InverseConfig, TrainConfig
from ecgan.config.dataclasses import InverseDetectorConfig
from ecgan.utils.miscellaneous import is_wandb_model_link, load_wandb_config


def get_run_config(config: Union[InverseConfig.Attribs, AnomalyDetectionConfig]) -> TrainConfig:
    """Return a :code:`TrainerConfig` from a :code:`InverseConfig.Attribs` or :code:`AnomalyDetectionConfig`."""
    if isinstance(config, InverseConfig.Attribs):
        run_uri = config.RUN_URI

    else:
        run_uri = config.ad_experiment_config.RUN_URI

    if is_wandb_model_link(run_uri):
        config_dict = load_wandb_config(run_uri)
        return TrainConfig(config_dict)

    return TrainConfig("{}/config.yml".format(run_uri))


def get_inv_run_config(ad_config: AnomalyDetectionConfig) -> InverseConfig:
    """Retrieve inverse config."""
    detection_config = ad_config.detection_config
    if not isinstance(detection_config, InverseDetectorConfig):
        raise RuntimeError(
            "The detection configuration needs to be of type "
            "InverseDetectorConfig. Current config is {}.".format(type(detection_config))
        )
    if detection_config.INVERSE_MAPPING_URI is None:
        raise RuntimeError("Inverse mapping may not be None.")
    run_uri = detection_config.INVERSE_MAPPING_URI.split(':')[0]

    if is_wandb_model_link(run_uri):
        config_dict = load_wandb_config(run_uri)
        return InverseConfig(config_dict)

    return InverseConfig("{}/inverse_config.yml".format(run_uri))


def get_model_path(run_uri: str, run_version: str) -> str:
    """Return a model path from a run URI and a version."""
    if is_wandb_model_link(run_uri):
        return "{}:{}".format(run_uri, run_version)

    return "{}/MODELS/model_ep_{}.pt".format(run_uri, run_version)


T = TypeVar('T')


def get_(val: Optional[T], default: T) -> T:
    """Retrieve values from typed dict or set a default if None."""
    return val if val is not None else default
