"""Basic trainer class used to create data splits, initialize training modules, fit a model and collect metrics."""
from logging import getLogger
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecgan.config import TrainerConfig, get_global_config
from ecgan.evaluation.tracker import TrackerFactory
from ecgan.modules.base import BaseModule
from ecgan.modules.factory import ModuleFactory
from ecgan.modules.generative.base import BaseGANModule
from ecgan.training.datasets import SeriesDataset
from ecgan.utils.artifacts import FileArtifact, ImageArtifact
from ecgan.utils.custom_types import SplitMethods, Transformation, save_epoch_metrics
from ecgan.utils.miscellaneous import (
    list_from_tuple_list,
    load_pickle,
    nested_list_from_dict_list,
    save_epoch,
    scale_to_unit_circle,
    to_torch,
    update_highest_metrics,
)
from ecgan.utils.splitting import create_splits, load_split, select_channels, verbose_channel_selection
from ecgan.utils.transformation import get_transformation
from ecgan.visualization.evaluation import boxplot
from ecgan.visualization.plotter import FourierPlotter, PlotterFactory

logger = getLogger(__name__)


class Trainer:
    """
    Load all required elements (data, config, tracking, plotter) and initialize the model.

    Requires a previously set config (via :func:`ecgan.config.global_cfg.set_global_config`).
    """

    def __init__(
        self,
        data: Union[Tensor, np.ndarray],
        label: Union[Tensor, np.ndarray],
    ):

        self.trainer_cfg = get_global_config().trainer_config
        self.exp_cfg = get_global_config().experiment_config
        self.module_cfg = get_global_config().module_config

        torch.manual_seed(self.trainer_cfg.MANUAL_SEED)
        self.tracker = TrackerFactory()(config=self.exp_cfg)
        logger.info(
            'Starting training process with dataset {0} using model {1}.'.format(
                self.exp_cfg.DATASET, self.exp_cfg.MODULE
            )
        )

        self.data = to_torch(data).float()
        self.label = to_torch(label).float()

        plotter = PlotterFactory.from_config(self.trainer_cfg)
        num_channels = (
            self.trainer_cfg.CHANNELS if isinstance(self.trainer_cfg.CHANNELS, int) else len(self.trainer_cfg.CHANNELS)
        )

        # If fourier transform is chosen: double the amount of channels (img+real)
        channel_multiplier = 2 if isinstance(plotter, FourierPlotter) else 1
        self.num_channels = channel_multiplier * num_channels

        module_factory = ModuleFactory()
        self.module = module_factory(
            cfg=self.module_cfg,
            module=self.exp_cfg.MODULE,
            seq_len=data.shape[1],
            num_channels=self.num_channels,
        )

        self.tracker.log_config(get_global_config().config_dict)
        self.tracker.watch(self.module.watch_list)
        self.tracker.plotter = plotter

    def fit(self) -> BaseModule:
        """Fit a model on a split."""
        split_indices = self.get_split()
        self._fit_cross_val(split_indices)
        self.tracker.close()

        return self.module

    def get_split(self) -> Dict:
        """Retrieve existing split or create new split with given amount of folds."""
        if self.trainer_cfg.SPLIT_METHOD == 'fixed':
            split_indices: Dict = load_pickle(self.trainer_cfg.SPLIT_PATH)
        else:
            split_method = SplitMethods.NORMAL_ONLY if self.trainer_cfg.TRAIN_ONLY_NORMAL else SplitMethods.MIXED

            split_indices = create_splits(
                self.data,
                self.label,
                seed=self.trainer_cfg.MANUAL_SEED,
                folds=self.trainer_cfg.CROSS_VAL_FOLDS,
                method=split_method,
                split=self.trainer_cfg.SPLIT,
            )

        self.tracker.log_artifacts(FileArtifact('split indices', split_indices, 'split.pkl'))

        return split_indices

    @staticmethod
    def prepare_data(
        train_x,
        test_x,
        vali_x,
        train_y,
        test_y,
        vali_y,
        trainer_cfg: TrainerConfig,
        rescale_to_unit_circle: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select channels, transform data, mask labels and possibly rescale data based on cfg.

        Args:
            train_x: Train data.
            test_x: Test data.
            vali_x: Validation data.
            train_y: Train labels.
            test_y: Test labels.
            vali_y: Validation labels.
            trainer_cfg: Trainer configuration.
            rescale_to_unit_circle: Flag indicating if the data shall be rescaled to unit circle.

        Returns:
            Transformed train, test and vali data and labels.
        """
        ############################################################
        # TRANSFORM DATA
        ############################################################
        verbose_channel_selection(test_x, trainer_cfg.CHANNELS)
        train_x = select_channels(train_x, trainer_cfg.CHANNELS)
        vali_x = select_channels(vali_x, trainer_cfg.CHANNELS)
        test_x = select_channels(test_x, trainer_cfg.CHANNELS)

        if trainer_cfg.TRANSFORMATION is not Transformation.NONE.value:
            logger.info("Applying data transformation: {0}.".format(trainer_cfg.TRANSFORMATION))
            transformer = get_transformation(trainer_cfg.transformation)
            train_x = transformer.fit_transform(train_x)
            test_x = transformer.transform(test_x)
            vali_x = transformer.transform(vali_x)

        if trainer_cfg.BINARY_LABELS:
            train_y[train_y != 0] = 1
            vali_y[vali_y != 0] = 1
            test_y[test_y != 0] = 1

        if rescale_to_unit_circle:
            logger.info("Rescaling data to unit circle (range [-1,1]).")
            train_x = scale_to_unit_circle(train_x)
            test_x = scale_to_unit_circle(test_x)
            vali_x = scale_to_unit_circle(vali_x)

        return train_x, test_x, vali_x, train_y, test_y, vali_y

    def _fit_split(self, train_x, vali_x, train_y, vali_y) -> Dict:
        """
        Fit training data using the model the trainer was initialized with.

        Args:
            train_x: Training data.
            vali_x: Validation data.
            train_y: Training labels.
            vali_y: Validation labels.

        Returns:
            Dictionary containing the highest metrics measured during training.
            The keys are metric names, values are (epoch, value) tuples.
        """
        ############################################################
        # CREATE DATASETS & DATA LOADERS
        ############################################################
        train_data = SeriesDataset(train_x, train_y)
        validation_data = SeriesDataset(vali_x, vali_y)
        self.module.train_dataset_sampler.set_dataset(train_data)
        self.module.vali_dataset_sampler.set_dataset(validation_data)

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.trainer_cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=self.trainer_cfg.NUM_WORKERS,
            pin_memory=True,
        )

        validation_loader = DataLoader(
            dataset=validation_data,
            shuffle=True,
            batch_size=self.trainer_cfg.BATCH_SIZE,
            num_workers=self.trainer_cfg.NUM_WORKERS,
            pin_memory=True,
        )

        ############################################################
        # CONDUCT TRAINING
        ############################################################
        logger.info('Start training with {0} epochs'.format(self.trainer_cfg.EPOCHS))

        # The highest metrics are (epoch, value) pairs.
        highest_metrics: Dict = {}

        for epoch in range(1, self.trainer_cfg.EPOCHS + 1):

            train_metrics: List[Dict] = []
            validation_metrics: List[Dict] = []

            # TRAINING LOOP
            for _batch_idx, batch in enumerate(tqdm(train_loader, leave=False, desc='Training epoch {}'.format(epoch))):
                metrics = self.module.training_step(batch)
                train_metrics.append(metrics)

            # VALIDATION LOOP
            for _batch_idx, batch in enumerate(
                tqdm(validation_loader, leave=False, desc='Validation epoch {}'.format(epoch))
            ):
                validation_results: Dict = self.module.validation_step(batch)

                if validation_results is not None or not validation_results:
                    validation_metrics.append(validation_results)

            # AFTER EPOCH ACTION
            artifacts = self.module.on_epoch_end(epoch, self.trainer_cfg.SAMPLE_INTERVAL, self.trainer_cfg.BATCH_SIZE)

            # HANDLE EPOCH ARTIFACTS AND TRACKING
            collated_train_metrics = self.tracker.collate_metrics(train_metrics)
            collated_validation_metrics = self.tracker.collate_metrics(validation_metrics)
            self.tracker.log_metrics(collated_train_metrics)
            self.tracker.log_metrics(collated_validation_metrics)
            self.tracker.log_artifacts(artifacts)
            self.tracker.advance_step()
            self.module.print_metric(epoch, collated_train_metrics)

            # CHECKPOINT
            highest_metrics = update_highest_metrics(
                collated_validation_metrics,
                artifacts,
                highest_metrics,
                epoch,
            )

            if save_epoch(
                highest_metrics,
                epoch,
                self.trainer_cfg.CHECKPOINT_INTERVAL,
                save_epoch_metrics,
                final_epoch=self.trainer_cfg.EPOCHS,
            ):
                fold = self.tracker.fold
                self.tracker.log_checkpoint(self.module.save_checkpoint(), fold)
        return highest_metrics

    def _fit_cross_val(self, split_indices: Dict):
        """Conduct training with cross validation."""
        metric_tuple_list: List[dict] = []
        rescale_to_unit_circle = False

        if isinstance(self.module, BaseGANModule):
            if self.module.cfg.GENERATOR.TANH_OUT:
                rescale_to_unit_circle = True

        for fold in range(1, self.trainer_cfg.CROSS_VAL_FOLDS + 1):

            train_x, test_x, vali_x, train_y, test_y, vali_y = load_split(
                self.data, self.label, index_dict=split_indices, fold=fold
            )

            train_x, _test_x, vali_x, train_y, _test_y, vali_y = self.prepare_data(
                train_x,
                test_x,
                vali_x,
                train_y,
                test_y,
                vali_y,
                self.trainer_cfg,
                rescale_to_unit_circle,
            )
            metrics = self._fit_split(train_x, vali_x, train_y, vali_y)
            metric_tuple_list.append(metrics)

            # Advance fold and create new module.
            if fold < self.trainer_cfg.CROSS_VAL_FOLDS:
                self.tracker.advance_fold()
                module_factory = ModuleFactory()
                self.module = module_factory(
                    cfg=self.module_cfg,
                    module=self.exp_cfg.MODULE,
                    seq_len=vali_x.shape[1],
                    num_channels=self.num_channels,
                )
        metric_dict_list = list_from_tuple_list(metric_tuple_list)

        self.tracker.log_artifacts(FileArtifact('metrics across all folds', split_indices, 'metrics.json'))
        metric_list, name_list = nested_list_from_dict_list(metric_dict_list)

        collated = self.tracker.collate_metrics(metric_dict_list)
        # Create a boxplot for all metrics
        self.tracker.log_artifacts(
            ImageArtifact(
                'Metric Boxplot Train',
                boxplot(
                    metric_list,
                    name_list,
                    "Metrics across {0} folds.".format(self.trainer_cfg.CROSS_VAL_FOLDS),
                ),
            )
        )

        logger.info("Collated metrics across folds: {0}.".format(collated))
