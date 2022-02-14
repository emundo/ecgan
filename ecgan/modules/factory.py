"""Factory for creating module objects."""
from ecgan.config import ModuleConfig
from ecgan.modules.base import BaseModule
from ecgan.modules.classifiers.nn_classifier import CNNClassifier, RNNClassifier
from ecgan.modules.generative.aegan import AEGAN
from ecgan.modules.generative.autoencoder import AutoEncoder
from ecgan.modules.generative.dcgan import DCGAN
from ecgan.modules.generative.rdcgan import RDCGAN
from ecgan.modules.generative.rgan import RGAN
from ecgan.modules.generative.vaegan import VAEGAN
from ecgan.modules.generative.variational_autoencoder import VariationalAutoEncoder
from ecgan.utils.custom_types import SupportedModules


class ModuleFactory:
    """Meta module for creating correct model instances."""

    @staticmethod
    def choose_class(module_name: str):
        """Choose the correct class based on the provided module name."""
        available_modules = {
            SupportedModules.DCGAN.value: DCGAN,
            SupportedModules.RGAN.value: RGAN,
            SupportedModules.RDCGAN.value: RDCGAN,
            SupportedModules.AEGAN.value: AEGAN,
            SupportedModules.RNN.value: RNNClassifier,
            SupportedModules.VAEGAN.value: VAEGAN,
            SupportedModules.CNN.value: CNNClassifier,
            SupportedModules.AUTOENCODER.value: AutoEncoder,
            SupportedModules.VAE.value: VariationalAutoEncoder,
        }
        try:
            return available_modules[module_name]
        except KeyError as err:
            raise AttributeError('Argument `module_name` is not set correctly: {0}.'.format(module_name)) from err

    def __call__(
        self,
        cfg: ModuleConfig,
        module: str,
        seq_len: int,
        num_channels: int,
    ) -> BaseModule:
        """
        Return implemented module instance.

        Args:
            cfg: Config of the respective class. Has to inherit from :class:`ecgan.utils.config.ModuleConfig`.
            module: Identifier of a supported module.
            seq_len: Sequence length of the series.
            num_channels: Amount of channels of the series.

        Returns:
            Instance of selected module.
        """
        module_class = ModuleFactory.choose_class(module)

        base_module: BaseModule = module_class(cfg, seq_len, num_channels)
        return base_module
