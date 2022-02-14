"""Classes specifying the retrieval/creation of datasets."""
import os
import urllib.request
import zipfile
from abc import abstractmethod
from logging import getLogger
from math import ceil, floor
from typing import Dict, List, Tuple, cast

import torch

import wandb
from ecgan.config import PreprocessingConfig, SinePreprocessingConfig
from ecgan.utils.configurable import Configurable
from ecgan.utils.custom_types import TrackerType, Transformation
from ecgan.utils.datasets import (
    CMUMoCapDataset,
    DatasetFactory,
    ExtendedCMUMoCapDataset,
    MitbihBeatganDataset,
    MitbihDataset,
    MitbihExtractedBeatsDataset,
    PTBExtractedBeatsDataset,
    ShaoxingDataset,
    SineDataset,
    WaferDataset,
)
from ecgan.utils.miscellaneous import get_num_workers, load_pickle, load_pickle_numpy, save_pickle
from ecgan.utils.splitting import load_split

logger = getLogger(__name__)


class DataRetriever(Configurable):
    """
    A :code:`DataRetriever` base class for retrieval of datasets.

    Objects of this class are used to download a given dataset and additional information on the dataset from a given
    source. More information on implemented datasets and how to add new datasets can be found in :ref:`Datasets`.

    Args:
        dataset: Name of the dataset which has to be supported by :class:`ecgan.utils.datasets.DatasetFactory`.
    """

    def __init__(self, dataset: str, cfg: PreprocessingConfig):
        self.dataset = DatasetFactory()(dataset)
        self.cfg = cfg

    @staticmethod
    def configure() -> Dict:
        """Return the default preprocessing configuration for a data retriever object."""
        num_workers = get_num_workers()
        config: Dict = PreprocessingConfig.configure(
            loading_src='None',
            target_sequence_length=0,
            num_workers=num_workers,
        )

        return config

    @abstractmethod
    def load(self) -> None:
        """Download the dataset to disk."""
        raise NotImplementedError("DataRetriever needs to implement data downloading using `load` method.")


class KaggleDataRetriever(DataRetriever):
    """
    A base class for downloading datasets from Kaggle.

    Since there is no rigid format for the datasets on Kaggle, the raw dataset from disk needs to be implemented and
    preprocessed by a custom :ref:`Preprocessor`.

    .. warning::
        Install the pip kaggle module if you want to download the data. It is included in the :code:`requirements.txt`
        or can be installed via :code:`pip install kaggle`. Create a file with your authentication information at
        :code:`~/.kaggle/kaggle.json.` or export the tokens using your command line (see `Kaggle on Github
        <https://github.com/Kaggle/kaggle-api>`_ for more information). If you cannot or do not want to use the kaggle
        API, download the data from the individual kaggle repositories and unzip them to
        :code:`<data_location>/<dataset_name>/raw`.
    """

    def load(self) -> None:
        """
        Load a dataset from Kaggle.

        The source url has to be given in the config as :code:`cfg.LOADING_SRC`.
        The target directory has to be given as :code:`cfg.LOADING_DIR`.
        """
        if self.cfg.LOADING_SRC is None:
            raise AttributeError('cfg.LOADING_SRC cannot be None. Need to supply Kaggle source repository.')

        path = os.path.join(self.cfg.LOADING_DIR, self.dataset.name)

        target = os.path.join(path, 'raw')
        os.makedirs(target, exist_ok=True)

        if not len(os.listdir(target)) == 0:
            logger.info(
                'Directory is not empty and download will be skipped.'
                'Make sure to point to an empty directory if you want to download '
                'the data once again. '
            )
            return

        try:
            logger.info('Downloading dataset {0} from kaggle...'.format(self.cfg.LOADING_SRC))

            import kaggle

            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(self.cfg.LOADING_SRC, target, unzip=True)
        except Exception as e:
            raise RuntimeError('Could not download data to {0}.'.format(self.cfg.LOADING_DIR)) from e

        logger.info('Download successful. Data has been saved to {0}.'.format(target))


class MitbihDataRetriever(KaggleDataRetriever):
    """
    The MITBIH dataset is downloaded via the regular :class:`ecgan.preprocessing.data_retrieval.KaggleDataLoader`.

    This class exists to configure the KaggleDataLoader correctly and supply relevant parameters required for further
    preprocessing. The given configuration is used only during initialization and can be changed if desired.

    The dataset is the raw original dataset and cannot be used for classification by default, requiring manual
    preprocessing steps. To use the MITBIH data you can either preprocess the downloaded data arbitrarily by yourself
    or use the supported preprocessed datasets :code:`mitbih_beats` or :code:`mitbih_beatgan` during initialization.

    | **Paper**:
    | `Moody and Mark 2001 <https://ieeexplore.ieee.org/abstract/document/932724>`_.
    | **Information on source**:
    | Original data can be found at `PhysioNet <https://physionet.org/content/mitdb/1.0.0/>`_.
      This framework **does not** use the original data source but an unofficial
      `kaggle mirror <https://www.kaggle.com/mondejar/mitbih-database/>`_. The data remains unchanged but
      is saved as csv for easier preprocessing.
    """

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the default MITBIH dataset."""
        num_workers = get_num_workers()
        config: Dict = PreprocessingConfig.configure(
            loading_src=MitbihDataset.loading_src,
            target_sequence_length=MitbihDataset.default_seq_len,
            num_workers=num_workers,
        )

        return config


class MitbihExtractedBeatsDataRetriever(KaggleDataRetriever):
    """
    Download the (beat-wise) segmented MITBIH dataset.

    The segmented MITBIH dataset is downloaded via the regular KaggleDataLoader.

    | **Paper**:
    | `Kachuee et al. 2018 <https://arxiv.org/abs/1805.00794>`_.
    | **Information on source**:
    | Data is downloaded from the authors `official kaggle repository <https://www.kaggle.com/shayanfazeli/heartbeat>`_.
    """

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the MITBIH dataset with extracted beats."""
        num_workers = get_num_workers()
        config: Dict = PreprocessingConfig.configure(
            loading_src=MitbihExtractedBeatsDataset.loading_src,
            target_sequence_length=MitbihExtractedBeatsDataset.default_seq_len,
            num_workers=num_workers,
        )

        return config


class PtbExtractedBeatsDataRetriever(KaggleDataRetriever):
    """
    Download the (beat-wise) segmented PTB dataset.

    The segmented PTB dataset is downloaded via the regular `KaggleDataRetriever`.

    | Paper: `Kachuee et al. 2018 <https://arxiv.org/abs/1805.00794>`_.
    | Information on source: Data is downloaded from the authors
      `official kaggle repository <https://www.kaggle.com/shayanfazeli/heartbeat>`_.
    """

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the MITBIH dataset with extracted beats."""
        num_workers = get_num_workers()
        config: Dict = PreprocessingConfig.configure(
            loading_src=PTBExtractedBeatsDataset.loading_src,
            target_sequence_length=PTBExtractedBeatsDataset.default_seq_len,
            num_workers=num_workers,
        )

        return config


class CMUMoCapDataRetriever(KaggleDataRetriever):
    """
    Download the subset of the CMU MoCap dataset used in BeatGAN.

    The dataset is downloaded via the regular `KaggleDataRetriever`.

    | Paper: `Zhou et al. 2019 <https://www.ijcai.org/proceedings/2019/0616.pdf>`_.
    | Information on source: Data is downloaded from a kaggle upload
      `unofficial kaggle repository <https://www.kaggle.com/maximdolg/cmu-mocap-dataset-as-used-in-beatgan>`_.
    """

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the CMU MoCap dataset."""
        num_workers = get_num_workers()
        config: Dict = PreprocessingConfig.configure(
            loading_src=CMUMoCapDataset.loading_src,
            target_sequence_length=CMUMoCapDataset.default_seq_len,
            num_workers=num_workers,
        )

        return config


class ExtendedCMUMoCapDataRetriever(KaggleDataRetriever):
    """
    Download a extended version of the subset of the CMU MoCap dataset used in BeatGAN.

    The dataset is downloaded via the regular `KaggleDataRetriever`.

    | Paper: `Zhou et al. 2019 <https://www.ijcai.org/proceedings/2019/0616.pdf>`_.
    | Information on source: Data is downloaded from a kaggle upload
      `unofficial kaggle repository <https://www.kaggle.com/maximdolg/cmu-mocap-dataset-as-used-in-beatgan>`_.
    """

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for extended CMU MoCap Dataset."""
        num_workers = get_num_workers()
        config: Dict = PreprocessingConfig.configure(
            loading_src=ExtendedCMUMoCapDataset.loading_src,
            target_sequence_length=ExtendedCMUMoCapDataset.default_seq_len,
            num_workers=num_workers,
        )

        return config


class SineDataRetriever(DataRetriever):
    """Class to generate a synthetic dataset containing sine waves."""

    def __init__(self, name, cfg: PreprocessingConfig):
        super().__init__(name, cfg)
        if not isinstance(self.cfg, SinePreprocessingConfig):
            raise RuntimeError("Config needs to be SinePreprocessingConfig for sine dataset.")
        self.cfg: SinePreprocessingConfig = self.cfg

    def load(self) -> None:
        """
        Generate a synthetic dataset with sine waves and save it.

        Configuration is currently limited to the amount of samples you want to create and the target sequence length.

        By default, the domain of sines will be between 0 and 25 which can lead to imperfect generated sine waves. This
        is intended behavior to have more variety in the FFT of generated sine waves and can be changed manually. The
        amplitude, frequency, phase and vertical translation will be chosen randomly. Furthermore, the dataset will be
        imbalanced: only 20% of the data will be anomalous. Half of the anomalous data consists of noisy sine waves
        (added gaussian noise) and the other half consists of superimposed sine waves. The resulting dataset can be used
        to asses the classification or generative capabilities of a given model.

        Since the resulting dataset will already in the target shape, no further preprocessing is currently supported
        and the data is saved as an already preprocessed dataset.
        """
        sine_lower_range, sine_upper_range = self.cfg.RANGE
        anomaly_percentage: float = self.cfg.ANOMALY_PERCENTAGE

        num_samples = self.cfg.NUM_SAMPLES
        if num_samples is None:
            raise ValueError("num_samples has to be defined in config.")
        num_channels = SineDataset.num_channels
        seq_len = self.cfg.TARGET_SEQUENCE_LENGTH

        torch.manual_seed(self.cfg.SYNTHESIS_SEED)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        amount_normal_samples = int((1 - anomaly_percentage) * num_samples)
        amount_abnormal_samples = num_samples - amount_normal_samples
        # func_in is the time dependency component of the sine wave.
        func_in = (
            torch.linspace(sine_lower_range, sine_upper_range, seq_len, device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(amount_normal_samples, 1, num_channels)
        )

        logger.info(
            'Generating sine dataset with shape ({0}, {1}, {2}) and {3}% anomalous series.'.format(
                self.cfg.NUM_SAMPLES,
                seq_len,
                num_channels,
                anomaly_percentage * 100,
            )
        )
        amplitude_range = self.cfg.AMPLITUDE
        frequency_range = self.cfg.FREQUENCY
        phase_range = self.cfg.PHASE
        v_translation = self.cfg.VERTICAL_TRANSLATION
        # Generate normal (non-anomalous) sine waves
        amplitude = (amplitude_range - -amplitude_range) * torch.rand(
            amount_normal_samples, 1, num_channels, device=device
        ).repeat(1, seq_len, 1) + -amplitude_range
        frequency = (frequency_range - -frequency_range) * torch.rand(
            amount_normal_samples, 1, 1, device=device
        ).repeat(1, seq_len, num_channels) + -frequency_range
        phase = (phase_range - -phase_range) * torch.rand(amount_normal_samples, 1, 1, device=device).repeat(
            1, seq_len, num_channels
        ) + -phase_range
        vertical_translation = (v_translation - -v_translation) * torch.rand(
            amount_normal_samples, 1, num_channels, device=device
        ).repeat(1, seq_len, 1) + -v_translation

        series = amplitude * torch.sin(frequency * func_in + phase) + vertical_translation

        # Generate anomalous sine waves:
        # noise_percentage % are superimposed sine waves, (1-noise_percentage)% is random noise.
        noise_percentage = self.cfg.NOISE_PERCENTAGE

        noise_series = torch.rand(
            ceil(noise_percentage * amount_abnormal_samples),
            seq_len,
            num_channels,
            device=device,
        )

        superimposed_sine = torch.stack(
            [
                1 / 3 * series[torch.randint(0, series.shape[0], (1,), device=device)]
                + 1 / 3 * series[torch.randint(0, series.shape[0], (1,), device=device)]
                + 1 / 3 * series[torch.randint(0, series.shape[0], (1,), device=device)]
                for _ in range(0, floor((1 - noise_percentage) * amount_abnormal_samples))
            ]
        ).squeeze(1)

        anomalous_series = torch.cat((noise_series, superimposed_sine))

        data_array = torch.cat((series, anomalous_series)).cpu().numpy()
        # Labels: Normal sine waves: 0; Noise: 1; Superimposed: 2
        labels = torch.cat(
            (
                torch.zeros(amount_normal_samples),
                torch.ones(ceil(amount_abnormal_samples * noise_percentage)),
                torch.full((floor(amount_abnormal_samples * (1 - noise_percentage)),), 2),
            )
        ).cpu()

        full_path = os.path.join(self.cfg.LOADING_DIR, SineDataset.name, 'processed')

        save_pickle(data_array, full_path, 'data.pkl')
        save_pickle(labels.numpy(), full_path, 'label.pkl')

        logger.info(
            'Successfully saved dataset to disc. Shape: {}, Distribution: {}'.format(
                data_array.shape, torch.unique(labels, return_counts=True)
            )
        )

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for an artificial sine dataset."""
        num_workers = get_num_workers()
        config: Dict = SinePreprocessingConfig.configure(
            loading_src='',
            target_sequence_length=SineDataset.default_seq_len,
            num_workers=num_workers,
        )

        config['preprocessing']['NUM_SAMPLES'] = 30000

        return config


class UrlDataRetriever(DataRetriever):
    """Class to download and extract zipped datasets from URLs."""

    def __init__(self, dataset: str, cfg: PreprocessingConfig, delete_zip: bool = False):
        super().__init__(dataset, cfg)
        self.delete_zip = delete_zip

    def load(self) -> None:
        """
        Load publicly available datasets which are saved as zips and extract them.

        The URLDataRetriever does not support additional authentication. If errors occur please check if the dataset is
        still available at the specified URL in the configuration file and please open an issue if this is not the case.

        Subclasses need to implement the abstract methods to define meta data and determine how to unzip the data.

        ..warning:
            The `urllib` request might require the installation of a Python certificate for Mac.

        """
        if self.cfg.LOADING_SRC is None:
            raise AttributeError('Dataset cannot be None using the UrlDataLoader.')
        path = os.path.join(self.cfg.LOADING_DIR, self.dataset.name)

        meta = self.get_meta()

        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'raw'), exist_ok=True)

        save_location = os.path.join(path, '{0}_raw.zip'.format(self.dataset.name))

        if not os.path.isfile(save_location):
            try:
                save_location = os.path.join(path, '{0}_raw.zip'.format(self.dataset.name))
                logger.info('Downloading {0} ECG data into {1} ...'.format(self.dataset.name, save_location))
                urllib.request.urlretrieve(self.cfg.LOADING_SRC, save_location)

                os.makedirs(os.path.join(path, 'raw'), exist_ok=True)
                for meta_file in meta:
                    meta_location = os.path.join(path, 'raw', meta_file[0])
                    logger.info('Downloading meta data into {0} ...'.format(meta_location))
                    urllib.request.urlretrieve(meta_file[1], meta_location)

            except Exception as e:
                raise RuntimeError('Could not download data to {0}.'.format(self.cfg.LOADING_DIR)) from e
            else:
                logger.info('Download successful. Data has been saved in {0}.'.format(save_location))
        else:
            logger.info('Skipping download, data has already been downloaded in {0}.'.format(save_location))

        unzip_location = os.path.join(path, 'raw')
        try:
            logger.info('Unzipping data in {0} ...'.format(path))
            self.extract_data(save_location, unzip_location)
            logger.info('Data has been successfully unzipped in {0}.'.format(unzip_location))

            if self.delete_zip:
                os.remove(save_location)

        except Exception as e:
            logger.info('Cleaning interrupted download {0}, deleting'.format(unzip_location))
            os.rmdir(unzip_location)
            raise RuntimeError('Could not download data to {0}.'.format(self.cfg.LOADING_DIR)) from e

    @abstractmethod
    def get_meta(self) -> List[Tuple]:
        """Get meta information on the downloaded files if required."""
        pass

    @abstractmethod
    def extract_data(self, save_location: str, unzip_location: str) -> None:
        """
        Extract data from zip file.

        Args:
            save_location: Reference to local directory where the zip is stored.
            unzip_location: Reference to local directory where the data shall be extracted to.
        """
        pass


class ShaoxingDataRetriever(UrlDataRetriever):
    """
    Download and extract the zipped Shaoxing dataset.

    | **Paper**:
    | `Zheng et al. 2020 <https://pubmed.ncbi.nlm.nih.gov/32051412/>`_.
    | **Information on source**:
    | Data is downloaded from their `official figshare mirror <https://figshare.com/collections/ChapmanECG/4560497/2>`_.
    """

    def get_meta(self) -> List[Tuple]:
        """Get meta information on the downloaded files."""
        return [
            ('RhythmNames.xlsx', 'https://ndownloader.figshare.com/files/15651296'),
            (
                'ConditionNames.xlsx',
                'https://ndownloader.figshare.com/files/15651293',
            ),
            (
                'AttributesDictionary.xlsx',
                'https://ndownloader.figshare.com/files/15653123',
            ),
            ('Diagnostics.xlsx', 'https://ndownloader.figshare.com/files/15651299'),
        ]

    def extract_data(self, save_location: str, unzip_location: str) -> None:
        """
        Extract data from zip file.

        Args:
            save_location: Reference to local directory where the zip is stored.
            unzip_location: Reference to local directory where the data shall be extracted to.
        """
        with zipfile.ZipFile(save_location, 'r') as zip_ref:
            zip_ref.extractall(unzip_location)

    @staticmethod
    def configure() -> Dict:
        """
        Return the default configuration for the Shaoxing dataset.

        The window_length, step size and target sequence length can be configured manually after initialization of the
        config file.
        """
        num_workers = get_num_workers()
        config: Dict = PreprocessingConfig.configure(
            loading_src=ShaoxingDataset.loading_src,
            target_sequence_length=ShaoxingDataset.default_seq_len,
            num_workers=num_workers,
            window_length=1250,
            window_step_size=250,
        )

        return config


class MitbihBeatganDataRetriever(UrlDataRetriever):
    """
    Download and extract the zipped MITBIH dataset based on the BeatGAN preprocessing.

    | **Paper**:
    | See `Zhou et al. 2019 <https://www.ijcai.org/Proceedings/2019/0616.pdf>`_.
    | **Information on source**:
    | Data is downloaded from the
      `official Dropbox mirror <https://www.dropbox.com/sh/b17k2pb83obbrkn/
      AABF9mUNVdaYwce9fnwXsg1ta/ano0?dl=0&subfolder_nav_tracking=1>`_.
    """

    def get_meta(self) -> List[Tuple]:
        """No metadata required."""
        return []

    def extract_data(self, save_location: str, unzip_location: str) -> None:
        """
        Extract data from zip file.

        Args:
            save_location: Reference to local directory where the zip is stored.
            unzip_location: Reference to local directory where the data shall be extracted to.
        """
        with zipfile.ZipFile(save_location, 'r') as zip_ref:
            # extract content into folder but remove toplevel dir.
            namelist = zip_ref.namelist()
            top_dir = namelist[1]
            zip_ref.extractall(unzip_location, members=namelist[1:])
            for item in namelist[2:]:
                rename_args = [
                    os.path.join(unzip_location, item),
                    os.path.join(unzip_location, os.path.basename(item)),
                ]
                os.rename(*rename_args)
            os.rmdir(os.path.join(unzip_location, top_dir))

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the MITBIH dataset based on the BeatGAN preprocessing."""
        num_workers = get_num_workers()
        config: Dict = PreprocessingConfig.configure(
            loading_src=MitbihBeatganDataset.loading_src,
            target_sequence_length=MitbihBeatganDataset.default_seq_len,
            num_workers=num_workers,
        )
        # Default transformation from the original paper is beat wise minmax normalization
        config['update'] = {'trainer': {'TRANSFORMATION': Transformation.INDIVIDUAL.value}}

        return config


class WaferDataRetriever(UrlDataRetriever):
    """
    Download the Wafer dataset from a public time series dataset collection.

    | **Paper**:
    | See `Olszewski 2001 <https://www.cs.cmu.edu/~bobski/pubs/tr01108-twosided.pdf>`_.
    | **Information on source**:
    | Data is downloaded from a
      `public repository for time series repository <http://www.timeseriesclassification.com/
      description.php?Dataset=Wafer>`_.
    """

    @staticmethod
    def configure() -> Dict:
        """Return the default configuration for the Wafer dataset."""
        num_workers = get_num_workers()
        config: Dict = PreprocessingConfig.configure(
            loading_src=WaferDataset.loading_src,
            target_sequence_length=WaferDataset.default_seq_len,
            num_workers=num_workers,
        )

        return config

    def get_meta(self) -> List[Tuple]:
        """No metadata required."""
        return []

    def extract_data(self, save_location: str, unzip_location: str) -> None:
        """
        Extract data from zip file.

        Args:
            save_location: Reference to local directory where the zip is stored.
            unzip_location: Reference to local directory where the data shall be extracted to.
        """
        with zipfile.ZipFile(save_location, 'r') as zip_ref:
            # extract content into folder but remove toplevel dir.
            zip_ref.extractall(unzip_location)


class DataRetrieverFactory:
    """Meta module for creating data retriever instances."""

    datasets = {
        MitbihDataset.name: MitbihDataRetriever,
        MitbihExtractedBeatsDataset.name: MitbihExtractedBeatsDataRetriever,
        ShaoxingDataset.name: ShaoxingDataRetriever,
        SineDataset.name: SineDataRetriever,
        MitbihBeatganDataset.name: MitbihBeatganDataRetriever,
        PTBExtractedBeatsDataset.name: PtbExtractedBeatsDataRetriever,
        CMUMoCapDataset.name: CMUMoCapDataRetriever,
        ExtendedCMUMoCapDataset.name: ExtendedCMUMoCapDataRetriever,
        WaferDataset.name: WaferDataRetriever,
    }

    def __call__(self, dataset: str, cfg: PreprocessingConfig) -> DataRetriever:
        """
        Retrieve a specified dataset and save it to disc.

        Args:
            dataset: String specifying the dataset to be downloaded.
            cfg: Configuration for preprocessing.

        Returns:
            DataRetriever instance.
        """
        try:
            return DataRetrieverFactory.datasets[dataset](dataset, cfg)  # type: ignore
        except KeyError as err:
            raise ValueError('Dataset {} is unknown.'.format(dataset)) from err

    @staticmethod
    def choose_class(dataset: str) -> DataRetriever:
        """
        Retrieve a specified dataset and save it to disc.

        Args:
            dataset: String specifying the dataset to be downloaded.

        Returns:
            DataRetriever instance.
        """
        try:
            return cast(DataRetriever, DataRetrieverFactory.datasets[dataset])
        except KeyError as err:
            raise ValueError('Dataset {0} is unknown.'.format(dataset)) from err


def retrieve_fold_from_existing_split(
    data_dir: str,
    split_path: str,
    split_file: str,
    fold: int,
    target_dir: str,
    location: TrackerType = TrackerType.WEIGHTS_AND_BIASES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and split data given an **existing** split file.

    The split file has to be previously saved by an instance of :class:`ecgan.evaluation.tracker.BaseTracker`.

    Args:
        data_dir: Directory containing the data/label pkl files (should be loaded from config used to create the split).
        split_path: Pointing to the run from which the split shall be loaded from.
            Format is usually <entity>/<project>/<run_id>.
        split_file: Pointing to the file inside :code:`split_path` containing the split indices.
        fold: The fold used during the training run that shall be evaluated.
        location: Tracker location of split file.
        target_dir: Directory the split file is saved to if it is retrieved from remote host.

    Returns:
        Tensors containing the train_x, test_x, vali_x, train_y, test_y, vali_y data from the given split.
    """
    data = torch.from_numpy(load_pickle_numpy(os.path.join(data_dir, 'data.pkl')))
    label = torch.from_numpy(load_pickle_numpy(os.path.join(data_dir, 'label.pkl')))

    logger.info('Retrieving data split from run {0} in file {1}.'.format(split_path, split_file))
    if location == TrackerType.WEIGHTS_AND_BIASES:
        api = wandb.Api()

        run = api.run(split_path)
        if split_file.__contains__('artifacts/'):
            split_file = split_file.replace('artifacts/', '')
        run.file(split_file).download(root=target_dir, replace=True)

    else:
        split_file = "{0}/{1}".format(split_path, split_file)
    split_location = os.path.join(target_dir, split_file)
    logger.info("Data split retrieved. Stored in {}.".format(split_location))
    split = load_pickle(split_location)

    return load_split(data, label, index_dict=split, fold=fold)
