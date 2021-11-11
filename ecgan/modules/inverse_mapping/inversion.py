"""Inversion methods."""
from typing import Optional

from ecgan.config import (
    EncoderGANConfig,
    GANModuleConfig,
    get_global_config,
    get_global_inv_config_attribs,
    get_model_path,
    get_run_config,
    set_global_config,
)
from ecgan.evaluation.tracker import BaseTracker
from ecgan.modules.inverse_mapping.inverse_mapping import InvertibleBaseModule
from ecgan.modules.inverse_mapping.vanilla_inverse_mapping import SimpleGANInverseMapping


def inverse_train(tracker: Optional[BaseTracker] = None) -> InvertibleBaseModule:
    """Train an inverse mapping using an existing model."""
    # Get model and config
    inverse_config = get_global_inv_config_attribs()
    run_uri = inverse_config.RUN_URI
    run_version = inverse_config.RUN_VERSION
    run_fold = inverse_config.FOLD
    run_identifier = '{}_fold{}'.format(run_uri, run_fold)

    config = get_run_config(inverse_config)
    model_path = get_model_path(run_identifier, run_version)
    set_global_config(config.config_dict)

    # Set params from config, initialize and train inversion.
    seq_len = get_global_config().preprocessing_config.TARGET_SEQUENCE_LENGTH
    channels = get_global_config().trainer_config.CHANNELS
    num_of_channels = channels if isinstance(channels, int) else len(channels)

    gan_config = get_global_config().module_config

    if not isinstance(gan_config, GANModuleConfig) or isinstance(gan_config, EncoderGANConfig):
        raise NotImplementedError('Inverse mapping is currently only supported for non-encoder based GANs.')

    inverse_mapping = SimpleGANInverseMapping(
        inv_cfg=inverse_config,
        module_cfg=gan_config,
        run_path=model_path,
        seq_len=seq_len,
        num_channels=num_of_channels,
        tracker=tracker,
    )
    inverse_mapping.train()

    return inverse_mapping
