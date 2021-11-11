r"""A simple inverse mapping module which expects a trained GAN module as function :math:`G: A \rightarrow B`."""
from typing import Dict, List, Optional

import torch

from ecgan.config import GANModuleConfig, InverseConfig, OptimizerConfig, get_global_config, nested_dataclass_asdict
from ecgan.evaluation.tracker import BaseTracker
from ecgan.modules.factory import ModuleFactory
from ecgan.modules.generative.base import BaseGANModule
from ecgan.modules.inverse_mapping.inverse_mapping import InvertibleBaseModule
from ecgan.networks.cnn import DownsampleCNN
from ecgan.utils.artifacts import Artifact, ImageArtifact
from ecgan.utils.optimizer import Adam, OptimizerFactory


class SimpleGANInverseMapping(InvertibleBaseModule):
    """
    Implementation of a simple inverse mapping module.

    The module expects a pre-trained generator model and trains a downsampling CNN based on the generator model.
    """

    POOLING_KERNEL_SIZE = 3

    def __init__(
        self,
        inv_cfg: InverseConfig.Attribs,
        module_cfg: GANModuleConfig,
        run_path: str,
        seq_len: int,
        num_channels: int,
        tracker: Optional[BaseTracker],
    ):
        self.run_path = run_path
        self._generator_module: BaseGANModule
        super().__init__(
            inv_cfg=inv_cfg,
            module_cfg=module_cfg,
            seq_len=seq_len,
            num_channels=num_channels,
            tracker=tracker,
        )
        self.cfg: GANModuleConfig = module_cfg

        self.inv_loss = torch.nn.L1Loss() if inv_cfg.INV_MODULE.LOSS == 'L1' else torch.nn.MSELoss()
        self.inv_optim = OptimizerFactory()(
            self.inv.parameters(),
            OptimizerConfig(**nested_dataclass_asdict(self.inv_cfg.INV_MODULE.OPTIMIZER)),
        )

    def _init_generator_module(self) -> BaseGANModule:
        exp_cfg = get_global_config().experiment_config
        module_cfg = get_global_config().module_config
        module = ModuleFactory()(module_cfg, exp_cfg.MODULE, self.seq_len, self.num_channels)
        module.load(self.run_path)
        if not isinstance(module, BaseGANModule):
            raise Exception(
                'The simple GAN inverse mapping expects a GAN module, got {0} instead'.format(module.__class__)
            )
        return module

    def _init_inv(self) -> torch.nn.Module:
        return DownsampleCNN(
            kernel_sizes=self.inv_cfg.INV_MODULE.KERNEL_SIZES,
            pooling_kernel_size=SimpleGANInverseMapping.POOLING_KERNEL_SIZE,
            input_channels=self.generator_module.num_channels,
            output_channels=self.generator_module.latent_size,
            seq_len=self.generator_module.seq_len,
            sampling_seq_len=self.generator_module.generator_sampler.sampling_seq_length,
        )

    @staticmethod
    def configure() -> Dict:
        """Return a default configuration for the module."""
        config = {
            'inverse': {
                'EPOCHS': 1024,
                'ROUNDS': 1024,
                'BATCH_SIZE': 128,
                'GPU': True,
                'ARTIFACT_CHECKPOINT': 10,
                'SAVE_CHECKPOINT': 10,
            }
        }
        inv_module_config = DownsampleCNN.configure()
        inv_module_config['INV_MODULE'].update(Adam.configure())
        config['inverse'].update(inv_module_config)
        return config

    def invert(self, data) -> torch.Tensor:
        """Apply the downsampling CNN."""
        return self.inv(data)  # type: ignore

    def _training_step(self, batch_size: int) -> Dict:
        noise = self.generator_module.generator_sampler.sample_z(batch_size)
        with torch.no_grad():
            output_data = self.generator_module.generator(noise)
        inv_vector = self.invert(output_data)
        loss = self.inv_loss(inv_vector, noise)

        loss.backward()
        self.inv_optim.step()

        return {'INV_LOSS': float(loss)}

    def _load_inv(self, inv_dict: Dict, load_optim: bool = False) -> None:
        self.inv.load_state_dict(inv_dict['MODULE'], strict=False)
        if load_optim:
            self.inv_optim.load_existing_optim(inv_dict['OPT'])

    def _load_generator_module(self, model_reference: str):
        return self.generator_module.load(model_reference)

    def _save_inv(self) -> Dict:
        return {'MODULE': self.inv.state_dict(), 'OPT': self.inv_optim.state_dict()}

    def validation_step(self, batch: dict) -> dict:
        """Move along. Nothing to see here."""
        pass

    @property
    def generator_module(self) -> BaseGANModule:
        return self._generator_module

    @generator_module.setter
    def generator_module(self, module: BaseGANModule) -> None:
        self._generator_module = module

    @property
    def watch_list(self) -> List:
        return [*self.generator_module.watch_list, *self.inv]

    def on_epoch_end(self, epoch: int, sample_interval: int, batch_size: int) -> List[Artifact]:
        """
        Perform artifact and metric logging in a sample interval.

        The function creates two types of sample images:

        #. Apply the generator module on some fixed noise and some randomly sampled noise.
        #. Apply the inverse mapping on the output of the generator and then re-apply the generator on the output of
           the downsampling.

        """
        result: List[Artifact] = []

        if not epoch % sample_interval == 0:
            return result
        noise = torch.cat(
            [
                self.generator_module.fixed_noise,
                self.generator_module.generator_sampler.sample_z(len(self.generator_module.fixed_noise)),
            ]
        )

        with torch.no_grad():
            gen_latent = self.generator_module.generator_sampler.sample(noise)
            inverted_latent = self.invert(gen_latent)
            gen_inverted = self.generator_module.generator(inverted_latent)
            if self.cfg.GENERATOR.TANH_OUT:
                gen_latent = (gen_latent / 2) + 0.5
                gen_inverted = (gen_inverted / 2) + 0.5

            difference = torch.abs(gen_latent - gen_inverted)

        result.append(
            ImageArtifact(
                'Generator Samples',
                self.plotter.get_sampling_grid(
                    gen_latent,
                    color='blue',
                    scale_per_batch=True,
                ),
            )
        )

        result.append(
            ImageArtifact(
                'Inverted Generator Samples',
                self.plotter.get_sampling_grid(
                    gen_inverted,
                    color='red',
                    scale_per_batch=True,
                ),
            )
        )
        result.append(
            ImageArtifact(
                'Difference Samples',
                self.plotter.get_sampling_grid(
                    difference,
                    color='green',
                    scale_per_batch=True,
                ),
            )
        )

        return result
