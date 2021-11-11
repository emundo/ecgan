"""Create a configuration file for the inverse mapping of a GAN module."""
from ecgan.config import get_global_inv_config, set_global_inv_config
from ecgan.modules.inverse_mapping.vanilla_inverse_mapping import SimpleGANInverseMapping
from ecgan.utils.miscellaneous import retrieve_model_specification


def init_inverse(path: str, filename: str) -> None:
    """
    Initialize an inverse mapping config.

    Args:
        path: Path to model storage.
        filename: Name of the file the config is saved to.
    """
    uri, fold, version = retrieve_model_specification(path)

    default_cfg = SimpleGANInverseMapping.configure()
    default_cfg['inverse']['RUN_URI'] = uri
    default_cfg['inverse']['FOLD'] = fold
    default_cfg['inverse']['RUN_VERSION'] = version
    set_global_inv_config(default_cfg)
    get_global_inv_config().file_name = filename
    get_global_inv_config().generate_config_file()
