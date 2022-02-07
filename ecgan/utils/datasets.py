"""Descriptions of supported datasets."""
from dataclasses import dataclass
from typing import ClassVar, Dict, Type


@dataclass
class Dataset:
    """Base class for static descriptions of ECG datasets."""

    name: str
    num_channels: int
    num_classes: int
    default_seq_len: int
    beat_types: Dict
    percentage_normal: float
    loading_src: str
    # Should not be overwritten by subclasses:
    NUM_CLASSES_BINARY: ClassVar[int] = 2
    beat_types_binary = {
        'normal': 0,
        'abnormal': 1,
    }


@dataclass
class ShaoxingDataset(Dataset):
    """Static description of the Shaoxing dataset."""

    name = 'shaoxing'
    num_channels = 12
    num_classes = 11
    default_seq_len = 128
    loading_src = 'https://ndownloader.figshare.com/files/15652862'
    percentage_normal = 0.172
    # For more information see https://figshare.com/articles/dataset/RhythmNames_xlsx/8360414.
    beat_types = {
        'SR': 0,  # Sinus Rhythm
        'AF': 1,  # Atrial Flutter
        'AFIB': 2,  # Atrial Fibrillation
        'AT': 3,  # Atrial Tachycardia
        'AVNRT': 4,  # Atrioventricular Node Reentrant Tachycardia
        'AVRT': 5,  # Atrioventricular Reentrant Tachycardia
        'SI': 6,  # Sinus Irregularity
        'SA': 6,  # Sinus Irregularity
        'SAAWR': 7,  # Sinus Atrium to Atrial Wandering Rhythm
        'SB': 8,  # Sinus Bradycardia
        'ST': 9,  # Sinus Tachycardia
        'SVT': 10,  # Supraventricular Tachycardia
    }


@dataclass
class MitbihDataset(Dataset):
    """Static description of the Mitbih dataset."""

    name = 'mitbih'
    num_channels = 2
    num_classes = 2
    default_seq_len = 0  # Can be adjusted manually, not supported for training at the moment
    loading_src = 'mondejar/mitbih-database'
    percentage_normal = 0.828

    beat_types = {'Normal': 0, 'Arrhythmic': 1}


@dataclass
class WaferDataset(Dataset):
    """Static description of the Wafer dataset."""

    name = 'wafer'
    num_channels = 1
    num_classes = 2
    default_seq_len = 152
    loading_src = 'http://www.timeseriesclassification.com/Downloads/Wafer.zip'
    percentage_normal = 0.894

    beat_types = {'Normal': 0, 'Abnormal': 1}


@dataclass
class MitbihExtractedBeatsDataset(Dataset):
    """Static description of the MITBIH dataset with extracted and downsampled single beats."""

    name = 'mitbih_beats'
    num_channels = 1
    num_classes = 5
    default_seq_len = 187
    loading_src = 'shayanfazeli/heartbeat'
    percentage_normal = 0.828
    # Beat Types as classified by https://arxiv.org/abs/1805.00794 for the MITBIH dataset.
    beat_types = {
        'N': 0,  # Normal SR, left/right bundle branch block, atrial escape, nodal escape
        'S': 1,  # {Atrial, aberrant atrial, nodal, supra-ventricular} Premature
        'V': 2,  # Premature ventricular contraction, ventricular escape
        'F': 3,  # Fusion of ventricular and normal
        'Q': 4,  # Paced, fusion of paced and normal, unclassifiable
    }


@dataclass
class MitbihBeatganDataset(Dataset):
    """
    Static description of the MITBIH Beatgan dataset.

    .. note:
        This preprocessing requires additional investigation regarding its correctness.
        We recommend using 'mitbih_beats' for the beatwise segmented MITBIH data.
    """

    name = 'mitbih_beatgan'
    num_channels = 2
    num_classes = 5  # 2
    default_seq_len = 320
    loading_src = 'https://www.dropbox.com/sh/b17k2pb83obbrkn/AADzJigiIrottyTOyvAEU1LOa?dl=1'
    percentage_normal = 0.889
    beat_types = {
        'N': 0,  # Normal SR, left/right bundle branch block, atrial escape, nodal escape
        'S': 1,  # {Atrial, aberrant atrial, nodal, supra-ventricular} Premature
        'V': 2,  # Premature ventricular contraction, ventricular escape
        'F': 3,  # Fusion of ventricular and normal
        'Q': 4,  # Paced, fusion of paced and normal, unclassifiable
    }


@dataclass
class SineDataset(Dataset):
    """Static description of the sine dataset."""

    name = 'sine'
    num_channels = 1
    num_classes = 3
    default_seq_len = 128
    loading_src = ''
    percentage_normal = 0.8
    # Different types of data contained in the synthetic sine dataset.
    beat_types = {
        'NORMAL': 0,  # Sinus Rhythm
        'NOISE': 1,  # Gaussian distributed noise
        'SUPERIMPOSED': 2,  # Superimposed sine waves
    }


@dataclass
class PTBExtractedBeatsDataset(Dataset):
    """Static description of the PTB dataset."""

    name = 'ptb_beats'
    num_channels = 1
    num_classes = 2
    default_seq_len = 187
    loading_src = 'shayanfazeli/heartbeat'
    percentage_normal = 0.278
    # Beat Types as classified by https://arxiv.org/abs/1805.00794 for the PTB dataset.
    beat_types = {
        'N': 0,  # Normal SR
        'A': 1,  # Abnormal heartbeat
    }


class CMUMoCapDataset(Dataset):
    """Static description of the CMU MoCap subset."""

    name = 'cmu_mocap'
    num_channels = 4
    num_classes = 3
    default_seq_len = 64
    loading_src = 'maximdolg/cmu-mocap-dataset-as-used-in-beatgan'
    percentage_normal = 0.73
    beat_types = {'walking': 0, 'jogging': 1, 'jumping': 2}


class ExtendedCMUMoCapDataset(CMUMoCapDataset):
    """Static description of the extended CMU MoCap subset."""

    name = 'cmu_mocap_extended'
    num_channels = 4
    num_classes = 4
    loading_src = 'maximdolg/extended-cmu-mocap-dataset-for-beatgan'
    percentage_normal = 0.378
    beat_types = {'walking': 0, 'jogging': 1, 'jumping': 2, 'dancing': 3}


class DatasetFactory:
    """Meta module for creating datasets objects containing static data to describe the datasets."""

    def __call__(self, dataset: str) -> Type[Dataset]:
        """Return implemented AD module when a BaseModule is created."""
        datasets = {
            ShaoxingDataset.name: ShaoxingDataset,
            MitbihDataset.name: MitbihDataset,
            MitbihExtractedBeatsDataset.name: MitbihExtractedBeatsDataset,
            MitbihBeatganDataset.name: MitbihBeatganDataset,
            SineDataset.name: SineDataset,
            PTBExtractedBeatsDataset.name: PTBExtractedBeatsDataset,
            CMUMoCapDataset.name: CMUMoCapDataset,
            ExtendedCMUMoCapDataset.name: ExtendedCMUMoCapDataset,
            WaferDataset.name: WaferDataset,
        }
        try:
            return datasets[dataset]
        except KeyError as err:
            raise AttributeError('Argument {0} is not set correctly.'.format(dataset)) from err
