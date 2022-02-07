"""Baseclass for preprocessing as well as the preprocessing classes for the supported datasets."""
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from functools import partial
from logging import getLogger
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import KNNImputer
from tqdm import tqdm

from ecgan.config import PreprocessingConfig
from ecgan.preprocessing.cleansing import DataCleanser
from ecgan.preprocessing.sampling import resample
from ecgan.utils.custom_types import SamplingAlgorithm
from ecgan.utils.datasets import (
    CMUMoCapDataset,
    DatasetFactory,
    ExtendedCMUMoCapDataset,
    MitbihBeatganDataset,
    MitbihDataset,
    MitbihExtractedBeatsDataset,
    PTBExtractedBeatsDataset,
    ShaoxingDataset,
    WaferDataset,
)
from ecgan.utils.miscellaneous import save_pickle

logger = getLogger(__name__)


class BasePreprocessor(ABC):
    """
    Base class for preprocessors.

    Generally, preprocessors expect data to be in :code:`<target_dir>/<dataset_name>/raw` (e.g. :code:`data/mitbih/raw`)
    with :code:`target_dir` being given in the config. Processed data should always be saved as :code:`data.pkl` and
    :code:`label.pkl` into :code:`<target_dir>/<dataset_name>/processed` (e.g. :code:`data/mitbih/processed`).

    Args:
        cfg: Configuration determining data source and desired preprocessing.
        dataset_name: Name of dataset which has to be preprocessed.
    """

    def __init__(self, cfg: PreprocessingConfig, dataset_name: str):
        self.cfg = cfg

        self.data: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.dataset = DatasetFactory()(dataset_name)

        self.target = os.path.join(self.cfg.LOADING_DIR, self.dataset.name, 'raw')
        self.save_dir = os.path.join(self.cfg.LOADING_DIR, self.dataset.name, 'processed')
        os.makedirs(self.save_dir, exist_ok=True)

    @abstractmethod
    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a given dataset.

        Returns:
             Data and label tensors.
        """
        raise NotImplementedError("Preprocessor needs to implement the `preprocess` method.")

    def save(self):
        """Save the data as well as the labels to `save_dir`."""
        if self.save_dir is None:
            raise ValueError('No saving location was provided')

        save_pickle(self.data, self.save_dir, 'data.pkl')
        save_pickle(self.labels, self.save_dir, 'label.pkl')
        logger.info('Data was saved to {}.'.format(self.save_dir))

    @staticmethod
    def _impute_nans(series: np.ndarray) -> np.ndarray:
        """
        Check if NaNs are contained in the data and impute them with a KNN imputer.

        Before impuration, you should make sure that the series do not contain too many NaNs to achieve good imputation.

        Args:
            series: Series of shape: :code:`(num_samples, seq_len, channels)`. May contain NaNs.

        Returns:
             All series with imputed values, i.e. without NaNs.
        """
        amount_nan = np.count_nonzero(np.isnan(series))
        if amount_nan == 0:
            logger.info('No NaNs in the dataset!')
            return series

        logger.info('{} NaNs found in data. Missing values are imputed.'.format(amount_nan))

        for i in range(series.shape[0]):
            series[i] = KNNImputer().fit_transform(series[i])

        if not np.nonzero(~np.isnan(series)):
            raise RuntimeError('The data still contains NaN-Values after imputation with KNN.')

        return series


class ExtractedBeatsPreprocessor(BasePreprocessor):
    """
    Preprocess the MITBIH/PTB dataset to retrieve beatwise segmented data.

    The data is already preprocessed according to `Kachuee et al. 2018 <https://arxiv.org/abs/1805.00794>`_.
    """

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataset.

        Load the raw CSV data, reshape the univariate series into multivariate series,
        extract labels, call the _preprocess_worker to cleanse and resample the data.
        Each initial raw sample has 187 values and contains the label in the last column.

        Returns:
            Tuple of data and labels in a framework compatible format.
        """
        logger.info('Loading csv data to memory.')
        src_files = self._get_src_files()
        raw_data = np.concatenate(
            [
                np.genfromtxt(
                    os.path.join(self.target, src_file),
                    delimiter=',',
                    invalid_raise=False,
                )
                for src_file in src_files
            ]
        )

        self.data = np.expand_dims(raw_data[:, :-1], axis=-1)
        self.labels = raw_data[:, -1]
        logger.info('Starting resampling and cleansing.')
        data_split = np.array_split(self.data, self.cfg.NUM_WORKERS)
        labels_split = np.array_split(self.labels, self.cfg.NUM_WORKERS)
        with mp.Pool(processes=self.cfg.NUM_WORKERS) as pool:
            results = [
                pool.apply_async(
                    self._preprocess_worker,
                    args=(
                        data_split[pos],
                        labels_split[pos],
                        self.cfg.TARGET_SEQUENCE_LENGTH,
                        pos,
                    ),
                )
                for pos in range(self.cfg.NUM_WORKERS)
            ]
            output = [p.get() for p in results]
            pool.close()
            pool.join()
            output.sort()

        self.data: np.ndarray = np.concatenate([out[1]['data'] for out in output])
        self.labels: np.ndarray = np.concatenate([out[1]['labels'] for out in output]).flatten().astype(dtype=np.int_)

        num_removed = sum([out[1]['removed_total'] for out in output])
        logger.info('{} samples were removed during cleansing process.'.format(num_removed))

        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError('The number of labels and samples does not match.')

        logger.info('Loaded and cleansed data.')

        self.data = self._impute_nans(self.data)
        logger.info(
            'Final dataset has {} samples with shape {}. Class distribution: {}'.format(
                len(self.data),
                self.data.shape,
                np.unique(self.labels, return_counts=True),
            )
        )

        return self.data, self.labels

    def _preprocess_worker(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        resampling_target: int,
        worker_id: int,
    ) -> Tuple[int, dict]:
        """
        Perform preprocessing using multiprocessing. Each worker is given a batch of data to cleanse and resample.

        Args:
            data: Raw downloaded data.
            labels: Labels corresponding to the data.
            resampling_target: Target sequence length.
            worker_id: Identifier of the multiprocessing worker.

        Returns:
            A tuple containing the ID of the multiprocessing worker as well as a Dict
            containing the preprocessed data, labels and the indices of the removed samples.
        """
        target_array = np.empty((data.shape[0], resampling_target, 1))
        cleansed_indices = []
        cleanser = DataCleanser(
            target_shape=(self.dataset.default_seq_len, 1),
            upper_fault_threshold=1,
            lower_fault_threshold=-1,
            nan_threshold=0.2,
        )
        should_resample = (
            self.cfg.RESAMPLING_ALGORITHM is not None and self.cfg.TARGET_SEQUENCE_LENGTH != self.data.shape[1]
        )

        for idx in tqdm(range(len(labels)), leave=False):

            if cleanser.should_cleanse(data[idx]):
                cleansed_indices.append(idx)

            # Resample the loaded ECG data if desired.
            if should_resample:
                target_array[idx] = resample(
                    data=data[idx],
                    algorithm=self.cfg.resampling_algorithm,
                    target_rate=resampling_target,
                    interpolation_strategy='linear',
                )

        data = target_array if should_resample else data
        data = np.delete(data, cleansed_indices, axis=0)
        labels = np.delete(labels, cleansed_indices)

        return worker_id, {
            'data': data,
            'labels': labels,
            'removed_total': cleanser.cleansed_total,
        }

    @abstractmethod
    def _get_src_files(self) -> List:
        raise NotImplementedError('ExtractedBeat Dataset needs to define source files.')


class MitbihExtractedBeatsPreprocessor(ExtractedBeatsPreprocessor):
    """Preprocess the MITBIH dataset from `Kachuee et al. 2018 <https://arxiv.org/abs/1805.00794>`_."""

    def _get_src_files(self) -> List:
        return ['mitbih_train.csv', 'mitbih_test.csv']


class PtbExtractedBeatsPreprocessor(ExtractedBeatsPreprocessor):
    """Preprocess the PTB dataset from `Kachuee et al. 2018 <https://arxiv.org/abs/1805.00794>`_."""

    def _get_src_files(self) -> List:
        return ['ptbdb_abnormal.csv', 'ptbdb_normal.csv']


class CMUMoCapPreprocessor(BasePreprocessor):
    """Preprocess the CMU MoCap subset used in `Zhou et al. 2019 <https://www.ijcai.org/proceedings/2019/0616.pdf>`_."""

    # Original strides used in BeatGAN preprocessing
    STRIDE = 5
    # Window size used in BeatGAN

    def _window(self, data: np.ndarray, stride: int) -> np.ndarray:
        window = self.cfg.TARGET_SEQUENCE_LENGTH
        data_length = data.shape[0]
        samples = []
        for start_idx in np.arange(0, data_length, stride):
            if start_idx + window >= data_length:
                break
            samples.append(data[start_idx : start_idx + window, :])

        return np.array(samples)

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataset.

        Load the raw CSV data, reshape the univariate series into multivariate series,
        extract labels, call the _preprocess_worker to cleanse and resample the data.

        Data is sampled as described in `Zhou et al. 2019 <https://www.ijcai.org/proceedings/2019/0616.pdf>`_.

        Returns:
            Tuple of data and labels in a framework compatible format.
        """
        raw_data_path = os.path.join(self.target, 'data.csv')
        raw_labels_path = os.path.join(self.target, 'labels.csv')

        raw_data = np.genfromtxt(raw_data_path, delimiter=",")
        raw_labels = np.genfromtxt(raw_labels_path, delimiter=",")

        walking_data = raw_data[raw_labels == CMUMoCapDataset.beat_types['walking']]
        jogging_data = raw_data[raw_labels == CMUMoCapDataset.beat_types['jogging']]
        jumping_data = raw_data[raw_labels == CMUMoCapDataset.beat_types['jumping']]

        walking_data = self._window(walking_data, self.STRIDE)
        jogging_data = self._window(jogging_data, 20)
        jumping_data = self._window(jumping_data, 5)

        walking_labels = np.zeros(walking_data.shape[0])
        jogging_labels = np.ones(jogging_data.shape[0])
        jumping_labels = np.ones(jumping_data.shape[0]) * 2

        self.data = np.concatenate([walking_data, jogging_data, jumping_data])
        self.labels = np.concatenate([walking_labels, jogging_labels, jumping_labels])

        return self.data, self.labels


class ExtendedCMUMoCapPreprocessor(CMUMoCapPreprocessor):
    """
    Preprocess the CMU MoCap subset used in `Zhou et al. 2019 <https://www.ijcai.org/proceedings/2019/0616.pdf>`_.

    The original dataset by Zhou et al. is extended an additional class _dancing_.
    """

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataset.

        Load the raw CSV data, reshape the univariate series into multivariate series,
        extract labels, call the _preprocess_worker to cleanse and resample the data.

        Data is sampled as described in `Zhou et al. 2019 <https://www.ijcai.org/proceedings/2019/0616.pdf>`_.
        Extended data of class _dancing_ is sampled same as the _walking_ class in the original.

        Returns:
            Tuple of data and labels in a framework compatible format.
        """
        raw_data_path = os.path.join(self.target, 'data.csv')
        raw_labels_path = os.path.join(self.target, 'labels.csv')
        raw_data = np.genfromtxt(raw_data_path, delimiter=",")
        raw_labels = np.genfromtxt(raw_labels_path, delimiter=",")

        data, labels = super().preprocess()

        dancing_data = raw_data[raw_labels == ExtendedCMUMoCapDataset.beat_types['dancing']]
        dancing_data = self._window(dancing_data, self.STRIDE)
        dancing_labels = np.ones(dancing_data.shape[0]) * 3
        self.data = np.concatenate([data, dancing_data])
        self.labels = np.concatenate([labels, dancing_labels])

        return self.data, self.labels


class WaferPreprocessor(BasePreprocessor):
    """
    Preprocess the Wafer Dataset from `Olszewski 2001 <https://www.cs.cmu.edu/~bobski/pubs/tr01108-twosided.pdf>`_.
    """

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataset.

        Returns:
            Tuple of data and labels in a framework compatible format.
        """
        logger.info('Loading arff data to memory.')
        file_list = [os.path.join(self.target, 'Wafer_TRAIN.arff'), os.path.join(self.target, 'Wafer_TEST.arff')]
        data = np.empty(0)
        label = np.empty(0)
        seq_len = WaferDataset.default_seq_len

        for file in file_list:
            try:
                data_ = arff.loadarff(file)[0]
                data_ = np.char.decode(np.array([list(tpl) for tpl in data_]), encoding='UTF-8').astype(float)
                data = np.append(data, data_[:, :seq_len])
                label = np.append(label, data_[:, seq_len])
            except Exception as e:
                raise RuntimeError('Could not preprocess data:{}'.format(e)) from e

        label[label == 1.0] = 0
        label[label == -1.0] = 1

        self.data = np.reshape(data, (-1, seq_len))
        self.data = np.expand_dims(self.data, -1)
        self.labels = label.astype(int)
        data_split = np.array_split(self.data, self.cfg.NUM_WORKERS)
        labels_split = np.array_split(self.labels, self.cfg.NUM_WORKERS)
        with mp.Pool(processes=self.cfg.NUM_WORKERS) as pool:
            results = [
                pool.apply_async(
                    self._preprocess_worker,
                    args=(
                        data_split[pos],
                        labels_split[pos],
                        self.cfg.TARGET_SEQUENCE_LENGTH,
                        pos,
                    ),
                )
                for pos in range(self.cfg.NUM_WORKERS)
            ]
            output = [p.get() for p in results]
            pool.close()
            pool.join()
            output.sort()

        self.data: np.ndarray = np.concatenate([out[1]['data'] for out in output])
        self.labels: np.ndarray = np.concatenate([out[1]['labels'] for out in output]).flatten().astype(dtype=np.int_)

        if len(self.data) != len(self.labels):
            raise ValueError('The number of labels and samples does not match.')

        logger.info('Loaded (and resampled) data.')

        logger.info(
            'Final dataset has {} samples with shape {}. Class distribution: {}'.format(
                len(self.data),
                self.data.shape,
                np.unique(self.labels, return_counts=True),
            )
        )
        return self.data, self.labels

    def _preprocess_worker(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        resampling_target: int,
        worker_id: int,
    ) -> Tuple[int, dict]:
        """
        Perform preprocessing using multiprocessing. Each worker is given a batch of data to cleanse and resample.

        Args:
            data: Raw downloaded data.
            labels: Labels corresponding to the data.
            resampling_target: Target sequence length.
            worker_id: Identifier of the multiprocessing worker.

        Returns:
            A tuple containing the ID of the multiprocessing worker as well as a Dict
            containing the preprocessed data, labels and the indices of the removed samples.
        """
        target_array = np.empty((data.shape[0], resampling_target, 1))

        should_resample = (
            self.cfg.RESAMPLING_ALGORITHM is not None and self.cfg.TARGET_SEQUENCE_LENGTH != self.data.shape[1]
        )

        for idx in tqdm(range(len(labels)), leave=False):
            # Resample the loaded ECG data if desired.
            if should_resample:
                target_array[idx] = resample(
                    data=data[idx],
                    algorithm=self.cfg.resampling_algorithm,
                    target_rate=resampling_target,
                    interpolation_strategy='linear',
                )

        data = target_array if should_resample else data

        return worker_id, {
            'data': data,
            'labels': labels,
        }


class MitbihBeatganPreprocessor(BasePreprocessor):
    """Preprocess the MITBIH dataset from `Zhou et al. 2019 <https://www.ijcai.org/Proceedings/2019/0616.pdf>`_."""

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the MITBIH BEATGAN dataset.

        Since the data itself is already preprocessed according to the paper, data classes are
        concatenated, reshaped and labeled according to the format used in ECGAN.
        Data can be additionally resampled, additional cleansing is not supported.

        It can be useful to remove the second dim (the second feature) since the leads
        differ across many series.

        .. warning:: The data has not been validated and might be erroneous.

        """
        logger.info('Loading csv data to memory.')
        raw_healthy = np.load(os.path.join(self.target, 'N_samples.npy'))
        self.labels: np.ndarray = np.zeros(raw_healthy.shape[0])

        raw_anomalies = np.empty((0, raw_healthy.shape[1], raw_healthy.shape[2]))
        abnormal_classes = ['F', 'V', 'Q', 'S']

        for i, abnormal_class in enumerate(abnormal_classes, 1):
            abnormal_data = np.load(os.path.join(self.target, '{}_samples.npy'.format(abnormal_class)))
            raw_anomalies = np.concatenate([raw_anomalies, abnormal_data])
            self.labels = np.concatenate([self.labels, np.repeat(i, abnormal_data.shape[0])])

        self.data = np.concatenate([raw_healthy, raw_anomalies]).transpose((0, 2, 1))

        # We assume that the data is already cleansed. It can be resampled if desired.
        if self.cfg.TARGET_SEQUENCE_LENGTH != self.data.shape[1]:
            logger.info('Starting resampling to target length {}.'.format(self.cfg.TARGET_SEQUENCE_LENGTH))
            if self.cfg.RESAMPLING_ALGORITHM is not None:
                with mp.Pool(processes=self.cfg.NUM_WORKERS) as pool:
                    data_split = np.array_split(self.data, self.cfg.NUM_WORKERS)
                    results = [
                        pool.apply_async(
                            self._preprocess_worker,
                            args=(
                                data_split[pos],
                                self.cfg.TARGET_SEQUENCE_LENGTH,
                                self.cfg.resampling_algorithm,
                                pos,
                            ),
                        )
                        for pos in range(self.cfg.NUM_WORKERS)
                    ]
                    output = [p.get() for p in results]
                    pool.close()
                    pool.join()
                    output.sort()
                    self.data: np.ndarray = np.concatenate([out[1] for out in output])

        if len(self.data) != len(self.labels):
            raise ValueError('The number of labels and samples does not match.')

        logger.info('Loaded (and resampled) data.')

        logger.info(
            'Final dataset has {} samples with shape {}. Class distribution: {}'.format(
                len(self.data),
                self.data.shape,
                np.unique(self.labels, return_counts=True),
            )
        )

        return self.data, self.labels

    @staticmethod
    def _preprocess_worker(
        data: np.ndarray,
        resampling_target: int,
        resample_algorithm: SamplingAlgorithm,
        worker_id: int,
    ) -> Tuple[int, np.ndarray]:
        """
        Worker method for resampling series.

        Args:
            data: Unprocessed data array.
            resample_algorithm: Resampling algorithm to be chosen.
            resampling_target: Target size of resampling.
            worker_id: ID of worker.

        Returns:
            Tuple with worker index and dictionary containing
            the resampled data.
        """
        resampled = np.empty((data.shape[0], resampling_target, 2))
        for idx in tqdm(range(len(data)), leave=False):
            resampled[idx] = resample(
                data=data[idx],
                algorithm=resample_algorithm,
                target_rate=resampling_target,
                interpolation_strategy='linear',
            )

        return worker_id, resampled


class ShaoxingPreprocessor(BasePreprocessor):
    """Preprocess the Shaoxing dataset from `Zheng et al. 2020 <https://www.nature.com/articles/s41597-020-0386-x>`_."""

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """Start preprocessing the Shaoxing data and save it to disc."""
        label_path = os.path.join(self.target, 'Diagnostics.xlsx')
        labels_to_files = self._get_labels(label_path)
        logger.info('Obtained list of labels and files to load.')

        self.data, self.labels = self._get_ecg_data(
            labels_to_files=labels_to_files,
            window_length=self.cfg.WINDOW_LENGTH,
            window_step_size=self.cfg.WINDOW_STEP_SIZE,
            resampling_algo=self.cfg.resampling_algorithm,
            resampling_threshold=self.cfg.TARGET_SEQUENCE_LENGTH,
            num_workers=self.cfg.NUM_WORKERS,
        )

        if len(self.data) != len(self.labels):
            raise ValueError('The number of labels and samples does not match.')

        logger.info('Loaded and cleansed data.')

        self.data = self._impute_nans(self.data)

        logger.info(
            'Final dataset has {} samples with shape {}. Class distribution: {}'.format(
                len(self.data),
                self.data.shape,
                np.unique(self.labels, return_counts=True),
            )
        )

        return self.data, self.labels

    @staticmethod
    def _get_labels(label_path: str) -> List[Tuple[int, str]]:
        """
        Load the list of labels, class labels are mapped to integers.

        Args:
            label_path: Path to label file.

        Returns:
            Tuple with encoded target vector and list of files that contain data matching the encoded versions.
        """
        df = pd.read_excel(label_path)

        df['FileName'] = df['FileName'].astype('category')
        df = df.sort_values(['FileName'])

        label_encoding = ShaoxingDataset.beat_types
        file_names = df['FileName'].tolist()
        rhythm_names = df['Rhythm'].tolist()
        result = []

        for file, rhythm in zip(file_names, rhythm_names):
            rhythm_label = label_encoding.get(rhythm)
            if rhythm_label is not None:
                result.append(
                    (
                        rhythm_label,
                        file + '.csv',
                    )
                )
        return result

    def _get_ecg_data(
        self,
        labels_to_files: List[Tuple[int, str]],
        window_length: int,
        window_step_size: int,
        resampling_algo: Optional[SamplingAlgorithm],
        resampling_threshold: int,
        num_workers: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the data files to memory, resample and concatenate them.

        Args:
            labels_to_files: List of files to load.
            window_length: Length of the windows that will be cut from the original length series.
            window_step_size: Number of steps the sliding window approach should progress on each slicing point.
            resampling_algo: Resampling algorithm to be chosen.
            resampling_threshold: Target size of resampling.
            num_workers: Number of workers to process in parallel.

        Returns:
            Data with shape :code:`(num_samples, seq_len, num_features)` and labels.
        """
        files_in_dir = os.listdir(self.target)
        if 'ECGDataDenoised' in files_in_dir:
            file_dir = os.path.join(self.target, 'ECGDataDenoised')
        elif 'ECGData' in files_in_dir:
            file_dir = os.path.join(self.target, 'ECGData')
        else:
            raise IOError('Directory does not contain a valid data directory.')

        with mp.Pool(processes=num_workers) as pool:
            func = partial(
                self._preprocess_worker,
                file_dir,
                resampling_algo,
                resampling_threshold,
                window_length,
                window_step_size,
            )
            results = list(
                tqdm(
                    pool.imap(func=func, iterable=labels_to_files),
                    total=len(labels_to_files),
                )
            )
            pool.close()
            pool.join()
        data = np.concatenate([result['data'] for result in results if result['data'] is not None])
        labels = (
            np.concatenate([result['labels'] for result in results if result['labels'] is not None])
            .flatten()
            .astype(dtype=np.int_)
        )

        if len(data) != len(labels):
            raise ValueError(
                'Length mismatch: Number of samples and labels are not '
                'the same: {} and {}'.format(len(data), len(labels))
            )

        num_removed = sum([result['removed_total'] for result in results])
        logger.info('{} samples were removed during cleansing process.'.format(num_removed))

        return data, labels

    @staticmethod
    def _preprocess_worker(
        file_dir: str,
        resampling_algo: Optional[SamplingAlgorithm],
        resampling_target: int,
        window_length: int,
        window_step_size: int,
        file_index: tuple,
    ) -> dict:
        """
        Worker method for loading and resampling time series files.

        Args:
            file_dir: Directory containing the files.
            file_index: Tuple containing (label, file_name).
            window_length: Length of the windows that will be cut from the original length series.
            window_step_size: Number of steps the sliding window approach should progress on each slicing point.
            resampling_algo: Resampling algorithm to be chosen.
            resampling_target: Target size of resampling.

        Returns:
            Dict containing data of shape :code:`(num_samples, seq_len, num_features)`,
            labels, the indices of the removed files and the total number of removed files.
        """
        has_removed = False
        cleanser = DataCleanser(
            target_shape=(window_length, 12),
            upper_fault_threshold=10000,
            lower_fault_threshold=-10000,
            nan_threshold=0.2,
        )

        # If one row in the data doesn't match column number then a
        # conversion warning is triggered. This warning is pointless
        # as the missing values will only be marked as NaNs, which
        # will be anyway removed if certain criteria are met
        # --> Suppress Conversion Warning
        ecg = np.genfromtxt(
            os.path.join(file_dir, file_index[1]),
            delimiter=',',
            usecols=np.arange(12),
            invalid_raise=False,
        )

        # Applies windowing to whole sequence
        windowed_ecgs = []
        seq_idx = 0
        while seq_idx < 5000 - window_length:
            windowed_ecg = ecg[seq_idx : seq_idx + window_length]
            seq_idx += window_step_size

            # Checks loaded file for different criteria
            # File is only appended to dataset if all checks succeed
            if cleanser.should_cleanse(windowed_ecg):
                has_removed = True
                continue

            # Resample the loaded ECG data if required
            if resampling_algo is not None:
                windowed_ecg = resample(
                    data=windowed_ecg,
                    algorithm=resampling_algo,
                    target_rate=resampling_target,
                )

            windowed_ecg = np.expand_dims(windowed_ecg, axis=0)
            windowed_ecgs.append(windowed_ecg)

        data: Optional[np.ndarray] = None
        labels: Optional[List] = None

        if len(windowed_ecgs) > 0:
            data = np.concatenate(windowed_ecgs)
            labels = [file_index[0] for _ in range(len(windowed_ecgs))]

            if len(labels) != len(windowed_ecgs):
                raise ValueError('Number of loaded labels and ECGs in list does not match.')

        result = {
            'data': data,
            'labels': labels,
            'removed': has_removed,
            'removed_total': cleanser.cleansed_total,
        }

        return result


class PreprocessorFactory:  # pylint: disable=R0911
    """Meta module for creating preprocessor instances."""

    def __call__(self, cfg: PreprocessingConfig, dataset: str) -> BasePreprocessor:
        """
        Initialize the preprocessor defined in the configuration file.

        Every preprocessor is determined by the dataset name, each dataset has
        exactly one preprocessor.

        Args:
            cfg: Configuration for preprocessors.
            dataset: Name of dataset.

        Returns:
            Instance of a Preprocessor.
        """
        if dataset == MitbihDataset.name:
            raise NotImplementedError(
                'We do not support preprocessing the raw MITBIH dataset. '
                'It has been downloaded in data/mitbih/raw and can be processed manually '
                '(save the data in data/mitbih/processed/data.pkl and the labels in '
                'data/mitbih/processed/label.pkl). To use a processed version of MITBIH, '
                'please select "mitbih_beats" or "mitbih_beatgan".'
            )
        if dataset == MitbihExtractedBeatsDataset.name:
            return MitbihExtractedBeatsPreprocessor(cfg, dataset)
        if dataset == ShaoxingDataset.name:
            return ShaoxingPreprocessor(cfg, dataset)
        if dataset == MitbihBeatganDataset.name:
            return MitbihBeatganPreprocessor(cfg, dataset)
        if dataset == PTBExtractedBeatsDataset.name:
            return PtbExtractedBeatsPreprocessor(cfg, dataset)
        if dataset == CMUMoCapDataset.name:
            return CMUMoCapPreprocessor(cfg, dataset)
        if dataset == ExtendedCMUMoCapDataset.name:
            return ExtendedCMUMoCapPreprocessor(cfg, dataset)
        if dataset == WaferDataset.name:
            return WaferPreprocessor(cfg, dataset)
        raise ValueError('Preprocessing mode {0} is unknown.'.format(dataset))
