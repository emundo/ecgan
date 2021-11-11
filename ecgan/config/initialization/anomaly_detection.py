"""Create a configuration file for the anomaly detection."""
import argparse

from ecgan.anomaly_detection.detector.detector_factory import AnomalyDetectorFactory
from ecgan.config import AnomalyDetectionConfig
from ecgan.utils.custom_types import AnomalyDetectionStrategies
from ecgan.utils.miscellaneous import retrieve_model_specification


def init_detection(args: argparse.Namespace) -> None:
    """
    Initialize and generate a config for anomaly detection.

    Args:
        args: Arguments parsed from CLI.
    """
    run_uri, fold, run_version = retrieve_model_specification(args.init[0])

    # Create the configuration file for anomaly detection using a given model.
    _, entity, project, run_name = args.init
    default_cfg = AnomalyDetectionConfig.configure(
        entity=entity,
        project=project,
        name=run_name,
        run_path=run_uri,
        run_version=run_version,
        fold=int(fold),
    )
    detection_config = AnomalyDetectorFactory.choose_class(
        AnomalyDetectionStrategies(args.anomaly_detector)
    ).configure()
    default_cfg['detection'].update(**detection_config['detection'])

    ad_config = AnomalyDetectionConfig(base_config=default_cfg, output_file=args.out)

    ad_config.generate_config_file()
