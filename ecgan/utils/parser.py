"""Utils for parsing CLI commands using argparse."""
import argparse

from ecgan.utils.custom_types import AnomalyDetectionStrategies, SupportedModules
from ecgan.utils.datasets import MitbihExtractedBeatsDataset


def config_parser() -> str:
    """Parse configuration file path command line args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        help='Choose the config. Default is: config.yml',
        type=str,
        nargs='?',
        default='config.yml',
    )
    args = parser.parse_args()
    return str(args.config)


def init_parser():
    """Parse the arguments required for ecgan-init."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--module',
        help='Choose the module. Possible values: {0}. Default is: {1}.'.format(
            [module.value for module in SupportedModules], SupportedModules.VAEGAN.value
        ),
        type=str,
        default=SupportedModules.VAEGAN.value,
    )
    parser.add_argument(
        '-o',
        '--out',
        help='Output path for config file. Default is: config.yml',
        type=str,
        metavar='PATH',
        default='config.yml',
    )
    parser.add_argument(
        '-l',
        '--data-location',
        help='Directory where data shall be saved to and loaded from. Default is: `data/`',
        type=str,
        metavar='DATA_LOCATION',
        default='data',
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        metavar='DATASET_NAME',
        help='Name of the dataset. Default is: {}'.format(MitbihExtractedBeatsDataset.name),
        default=MitbihExtractedBeatsDataset.name,
    )
    parser.add_argument('entity', help='Entity/Team name. Can contain multiple projects.', type=str)
    parser.add_argument(
        'project',
        help='Project belonging to one entity, possibly containing multiple named runs.',
        type=str,
    )
    parser.add_argument('name', help='Name of the experiment/run.', type=str)
    args = parser.parse_args()
    project = args.project
    entity = args.entity
    name = args.name
    module = args.module
    dataset = args.dataset
    output_file = args.out
    data_location = args.data_location
    return data_location, dataset, entity, module, name, output_file, project


def inverse_parser():
    """Parse the arguments required for ecgan-inverse."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--out',
        help='Output path for the inverse mapping config file. Default is: inverse_config.yml',
        type=str,
        nargs="?",
        default='inverse_config.yml',
    )
    parser.add_argument(
        '-i',
        '--init',
        help='Create a new configuration for the model.',
        metavar='MODULE_PATH:MODEL_VERSION',
        type=str,
    )
    args = parser.parse_args()
    return args


def detection_parser():
    """Parse the arguments required for ecgan-detect."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help='Output path for the anomaly detection config file. Default is: ad_config.yml.',
        type=str,
        nargs="?",
        default='ad_config.yml',
    )
    parser.add_argument(
        '-i',
        '--init',
        help='Create a new configuration for the model. Requires module identifier, entity, project and run name.',
        metavar='MODULE_NAME:MODEL_VERSION ENTITY_NAME PROJECT_NAME EXPERIMENT_NAME',
        type=str,
        nargs=4,
    )
    parser.add_argument(
        '-o',
        '--out',
        help='Output path for anomaly detection config file. Default is: ad_config.yml.',
        type=str,
        metavar='PATH',
        default='ad_config.yml',
    )
    parser.add_argument(
        '-a',
        '--anomaly-detector',
        help='Anomaly detection module. Supported: {0}. Default is {1}.'.format(
            [strategy.value for strategy in AnomalyDetectionStrategies],
            AnomalyDetectionStrategies.INVERSE_MAPPING.value,
        ),
        type=str,
        metavar='DETECTOR',
        default='inverse_mapping',
    )
    args = parser.parse_args()
    return args
