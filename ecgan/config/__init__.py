"""Configuration specifications to automatically generate a config file and allow correct use."""

from ecgan.config.configs import AnomalyDetectionConfig, Config, InverseConfig, PreprocessingConfig, TrainConfig
from ecgan.config.dataclasses import *
from ecgan.config.global_cfg import (
    get_global_ad_config,
    get_global_config,
    get_global_inv_config,
    get_global_inv_config_attribs,
    set_global_ad_config,
    set_global_config,
    set_global_inv_config,
)
from ecgan.config.helpers import get_, get_inv_run_config, get_model_path, get_run_config
from ecgan.config.nested_dataclass import nested_dataclass, nested_dataclass_asdict
