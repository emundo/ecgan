"""Anomaly manager containing the loading, execution and saving logic."""
from logging import getLogger
from typing import Optional, cast

import numpy as np
from torch import Tensor, cat, manual_seed

from ecgan.anomaly_detection.detector.detector_factory import AnomalyDetectorFactory
from ecgan.anomaly_detection.detector.reconstruction_detector import GANAnomalyDetector, ReconstructionDetector
from ecgan.anomaly_detection.embedder import Embedder, UMAPEmbedder
from ecgan.anomaly_detection.reconstruction import InterpolationReconstructor
from ecgan.config import AnomalyDetectionConfig, ReconstructionDetectionConfig, get_global_config, get_model_path
from ecgan.evaluation.metrics.classification import ClassificationMetricFactory
from ecgan.evaluation.tracker import BaseTracker
from ecgan.modules.factory import ModuleFactory
from ecgan.utils.artifacts import ImageArtifact
from ecgan.utils.custom_types import MetricType
from ecgan.utils.timer import Timer
from ecgan.visualization.plotter import PlotterFactory

logger = getLogger(__name__)


class AnomalyManager:
    """
    Load and set model, delegate work to anomaly detector and trigger visualization/evaluation logic.

    Args:
        cfg: Configuration used for anomaly detection, including reference to existing model.
        seq_len: Sequence length used by the module.
        tracker: Tracker to save evaluation.
        num_channels: Amount of channels used by the module.
    """

    def __init__(
        self,
        cfg: AnomalyDetectionConfig,
        seq_len: int,
        tracker: BaseTracker,
        num_channels,
    ):
        self.train_cfg = get_global_config()
        self.ad_cfg = cfg
        trainer_cfg = self.train_cfg.trainer_config
        module_cfg = self.train_cfg.module_config
        manual_seed(trainer_cfg.MANUAL_SEED)

        # Configure tracker
        self.tracker = tracker
        plotter = PlotterFactory.from_config(trainer_cfg)
        self.tracker.plotter = plotter
        self.tracker.log_config(cfg.config_dict)

        # Load module
        fold_uri = '{}_fold{}'.format(cfg.ad_experiment_config.RUN_URI, str(cfg.ad_experiment_config.FOLD))
        model_path = get_model_path(fold_uri, cfg.ad_experiment_config.RUN_VERSION)
        module_factory = ModuleFactory()
        self.module = module_factory(
            cfg=module_cfg,
            module=self.train_cfg.experiment_config.MODULE,
            seq_len=seq_len,
            num_channels=num_channels,
        ).load(model_path)

        anomaly_factory = AnomalyDetectorFactory()
        self.anomaly_detector = anomaly_factory(
            detector=cfg.detection_config.DETECTOR,
            module=self.module,
            tracker=self.tracker,
        )
        self.embedder: Optional[Embedder] = None

    def start_detection(
        self,
        train_x: Tensor,
        train_y: Tensor,
        test_x: Tensor,
        test_y: Tensor,
        vali_x: Tensor,
        vali_y: Tensor,
    ) -> None:
        """
        Triggers the anomaly detection and contains the relevant logic.

        Expects the same train and test data and labels as during training.

        Args:
            train_x: Train dataset from model fitting.
            train_y: Train labels from model fitting.
            test_x: Test dataset from model fitting.
            test_y: Test labels from model fitting.
            vali_x: Validation dataset from model fitting.
            vali_y: Validation labels from model fitting.
        """
        for run_id in range(self.ad_cfg.detection_config.AMOUNT_OF_RUNS):
            with Timer(
                name='Detection Timer',
                tracker=self.tracker,
                metric_name='detection_time',
            ):
                predicted_labels = self.anomaly_detector.detect(test_x, test_y)

            self.evaluate_performance(
                test_y.cpu().numpy(),
                predicted_labels,
            )
            if isinstance(self.ad_cfg.detection_config, ReconstructionDetectionConfig):
                embedding_config = cast(ReconstructionDetectionConfig, self.ad_cfg.detection_config).EMBEDDING
                if embedding_config.CREATE_UMAP:
                    self.create_embedding(
                        train_x=train_x,
                        test_x=test_x,
                        vali_x=vali_x,
                        train_y=train_y,
                        test_y=test_y,
                        vali_y=vali_y,
                    )
            self.tracker.advance_step()
            if self.ad_cfg.detection_config.SAVE_DATA:
                self.anomaly_detector.save(str(run_id))
        self.tracker.close()

    def evaluate_performance(
        self,
        test_y: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> None:
        """
        Evaluate the anomaly detection performance.

        Calculate the F1 and MCC score and log them using the tracker. If the detector is an interpolation
        reconstructor: also visualize the distribution of the norm of the latent vectors.

        Args:
            test_y: Real test labels.
            predicted_labels: Predicted test labels.
        """
        fscore = ClassificationMetricFactory()(MetricType.FSCORE).calculate(test_y, predicted_labels)
        logger.info("F-score of real and predicted labels is {}".format(fscore))
        mcc = ClassificationMetricFactory()(MetricType.MCC).calculate(test_y, predicted_labels)
        logger.info("MCC of real and predicted labels is {}".format(mcc))
        auroc = ClassificationMetricFactory()(MetricType.AUROC).calculate(test_y, predicted_labels)
        logger.info("AUROC of real and predicted labels is {}".format(auroc))

        if not (
            isinstance(self.anomaly_detector, GANAnomalyDetector)
            and isinstance(self.anomaly_detector.reconstructor, InterpolationReconstructor)
        ):
            return

        normal_norms = self.anomaly_detector.z_norm.cpu().numpy()[test_y == 0]
        abnormal_norms = self.anomaly_detector.z_norm.cpu().numpy()[test_y != 0]
        normal_plot = self.tracker.plotter.create_histogram(normal_norms, 'Normal class z norm')
        abnormal_plot = self.tracker.plotter.create_histogram(abnormal_norms, 'Abnormal class z norm')
        self.tracker.log_artifacts(ImageArtifact('Normal Class z-norm', normal_plot))
        self.tracker.log_artifacts(ImageArtifact('Abnormal Class z-norm', abnormal_plot))

    @property
    def embedder(self):
        return self._embedder

    @embedder.setter
    def embedder(self, embedder: Embedder):
        self._embedder = embedder

    def create_embedding(
        self,
        train_x: Tensor,
        test_x: Tensor,
        vali_x: Tensor,
        train_y: Tensor,
        test_y: Tensor,
        vali_y: Tensor,
    ) -> None:
        """Create an embedding trained on train and validation data with embedded test data and save the embedding."""
        if not isinstance(self.anomaly_detector, ReconstructionDetector):
            raise RuntimeError("Cannot create embedding with {0} anomaly detector.".format(type(self.anomaly_detector)))
        if self.embedder is None:
            self.embedder: UMAPEmbedder = UMAPEmbedder(
                cat((train_x, vali_x)),
                cat((train_y, vali_y)),
                self.train_cfg.experiment_config.DATASET,
            )

        test_embedding, test_labels = self.embedder.embed_test(test_x, test_y, include_initial_embedding=False)
        reconstructed_embedding, reconstructed_labels = self.embedder.embed(
            self.anomaly_detector.get_reconstructed_data(), labels=test_y
        )
        total_embedding = np.concatenate(
            (
                self.embedder.initial_embedding,
                test_embedding,
                reconstructed_embedding,
            )
        )

        total_labels = np.concatenate(
            (
                self.embedder.initial_labels,
                test_labels,
                reconstructed_labels,
            )
        )
        plot = self.embedder.get_plot(total_embedding, total_labels)
        self.tracker.log_artifacts(ImageArtifact('Interpolation Embedding', plot))
        interpol_path = self.embedder.draw_interpolation_path(
            cast(InterpolationReconstructor, self.anomaly_detector.reconstructor).series_samples,
            total_labels,
            total_embedding,
        )
        self.tracker.log_artifacts(ImageArtifact('Interpolation trace', interpol_path))
        logger.info("Completed embeddings.")
