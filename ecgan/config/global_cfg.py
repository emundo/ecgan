"""Definition of methods which can be used to get/set the global configuration."""
# pylint: disable=W0603, W0602
from typing import Dict, Optional, Union

from ecgan.config.configs import AnomalyDetectionConfig, InverseConfig, TrainConfig

# The global variables will be reimported during multiprocessing in pytorchs dataloader. Should not be a problem if the
# config is used as read only but keep this in mind.
GLOBAL_CONFIG: TrainConfig = TrainConfig(None)
GLOBAL_AD_CFG: AnomalyDetectionConfig = AnomalyDetectionConfig(None)
GLOBAL_INV_CFG: InverseConfig = InverseConfig(None)


def set_global_config(cfg: Optional[Union[str, Dict]] = 'config.yml'):
    """Set the global ECGAN config."""
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = TrainConfig(base_config=cfg)


def get_global_config() -> TrainConfig:
    """Return the global ECGAN config."""
    global GLOBAL_CONFIG
    return GLOBAL_CONFIG


def set_global_ad_config(cfg: Optional[Union[str, Dict]] = 'config.yml'):
    """Set the global anomaly detection config."""
    global GLOBAL_AD_CFG
    GLOBAL_AD_CFG = AnomalyDetectionConfig(base_config=cfg)


def get_global_ad_config() -> AnomalyDetectionConfig:
    """Return the attributes of the global anomaly detection config."""
    global GLOBAL_AD_CFG
    return GLOBAL_AD_CFG


def set_global_inv_config(cfg: Optional[Union[str, Dict]] = 'config.yml'):
    """Set the global inverse config."""
    global GLOBAL_INV_CFG
    GLOBAL_INV_CFG = InverseConfig(cfg)


def get_global_inv_config() -> InverseConfig:
    """Return the attributes of the global inverse config."""
    global GLOBAL_INV_CFG
    return GLOBAL_INV_CFG


def get_global_inv_config_attribs() -> InverseConfig.Attribs:
    """Return the attributes of the global inverse config."""
    return get_global_inv_config().attribs
