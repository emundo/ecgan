"""Functions to split the dataset."""
from logging import getLogger
from math import ceil
from typing import Dict, List, Tuple, Union, cast

import numpy as np
import torch
from numpy import ndarray
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor

from ecgan.utils.custom_types import SplitMethods

logger = getLogger(__name__)


def create_splits(
    data: Union[Tensor, ndarray],
    label: Union[Tensor, ndarray],
    folds: int,
    seed: int,
    method: SplitMethods = SplitMethods.MIXED,
    split: Tuple[float, float] = (0.8, 0.2),
) -> Dict:
    r"""
    Take given data and label tensors and split them into n folds with train, test and validation data.

    Currently we support the creation of two kinds of splits:

    #. Mixed train split: A standard split of all data into train, test and validation using the
       sklearn :code:`StratifiedShuffleSplit`, frequently used in classification tasks.
       The split-tuple has to sum up to 1, containing the respective percentages for the (training+validation, test)
       set. Using at least 60% of data for training is common.
    #. Healthy only train split: Training is only performed on healthy data. To create the training set, all normal
       data (label==0) is shuffled and the training set contains
       :math:`train\_x = len(normal\_data) * split[0])` samples. The test/validation set contain
       the remaining normal samples and all anomalous samples.
       In this implementation we follow the cross-validation setup and use the same test set for all folds. This
       means that the abnormal validation data also remains the same.
       The split-tuple determines how many of the normal data is used in train/vali and test.
       Make sure to keep possible data imbalances in mind.

    Args:
        data: Input data of shape :code:`(num_samples, seq_len, channels)`.
        label: Input labels as tensor.
        folds: Amount of folds.
        seed: PyTorch/Numpy RNG seed to deterministically create splits.
        method: Indicator of how to split dataset. Based on random split
            ('mixed') or splitting the data such that only the normal class is used during training ('normal_only')
            and normal and abnormal classes are used during validation/testing.
            0 if no instance of the normal data are used, 1 if all instances are used in the test set.
        split: Fraction of data in the (train+vali, test) sets.

    Returns:
        Split indices dictionary containing n folds with indices to construct
        n datasets consisting of (train_x, test_x, vali_x, train_y, test_y, vali_y).
    """
    data = torch.from_numpy(data).float() if isinstance(data, ndarray) else data
    label = torch.from_numpy(label).int() if isinstance(label, ndarray) else label

    if sum(list(split)) != 1:
        raise Exception("Sum of all splits has to equal 1.")
    if folds <= 1:
        raise RuntimeError(
            "Invalid number of folds: {0} (type: {1}. Has to be an integer > 1.".format(folds, type(folds))
        )

    if method == SplitMethods.NORMAL_ONLY:
        return train_only_normal_split(label, folds, seed, split)
    if method == SplitMethods.MIXED:
        return mixed_split(data, label, folds, seed, split)

    raise ValueError('Method {0} is not applicable.'.format(method))


def train_only_normal_split(
    label: Tensor,
    folds: int,
    seed: int,
    split: Tuple[float, float] = (0.85, 0.15),
) -> Dict:
    r"""
    Take given data and label tensors and split them into a train, test and validation set.

    Training is only performed on healthy data. To create the training set, all normal
    data (label==0) is shuffled and the training set contains
    :math:`train\_x = len(normal\_data) * split[0])` samples. The test/validation set contain
    the remaining normal samples and all anomalous samples.
    In this implementation we follow the cross-validation setup and use the same test set for all folds. This
    means that the abnormal validation data also remains the same.
    The split-tuple determines how many of the normal data is used in train/vali and test.
    Make sure to keep possible data imbalances in mind.

    Args:
        label: Input labels as tensor.
        folds: Amount of splits performed.
        seed: Random seed.
        split: Fraction of data in the (train, test) set.

    Returns:
        Index dictionary with n folds containing indices used in train, test and validation set.
    """
    split_indices = {}
    rng = np.random.default_rng(seed)

    normal_mask: Tensor = cast(Tensor, label == 0)
    anomalous_mask = ~normal_mask

    normal_indices = torch.nonzero(normal_mask).view(-1)
    anomalous_indices = torch.nonzero(anomalous_mask).view(-1)
    shuffled_normal_idx = normal_indices[torch.randperm(normal_indices.size()[0])]
    shuffled_anomalous_idx = anomalous_indices[torch.randperm(anomalous_indices.size()[0])]
    test_idx_normal = shuffled_normal_idx[ceil(len(normal_indices) * split[0]) :]
    idx_train_vali_normal = shuffled_normal_idx[: ceil(len(normal_indices) * split[0])]

    samples_per_fold = ceil(len(idx_train_vali_normal) / folds)
    idx_fold = idx_train_vali_normal.split(samples_per_fold, dim=0)

    # Use same proportion of stratified abnormal data in test and vali
    test_size = (len(test_idx_normal)) / (len(test_idx_normal) + samples_per_fold)
    num_abnormal_test = ceil(len(shuffled_anomalous_idx) * test_size)
    test_idx_abnormal = shuffled_anomalous_idx[:num_abnormal_test]
    vali_idx_abnormal = shuffled_anomalous_idx[num_abnormal_test:].tolist()
    idx_test = test_idx_normal.tolist() + test_idx_abnormal.tolist()

    for idx, idx_test_upper_limit in enumerate(idx_fold, 1):
        idx_train = list(set(idx_train_vali_normal.tolist()) - set(idx_test_upper_limit.tolist()))
        idx_vali_normal = list(set(idx_train_vali_normal.tolist()) - set(idx_train))
        idx_vali = idx_vali_normal + vali_idx_abnormal

        rng.shuffle(idx_test)
        rng.shuffle(idx_vali)

        split_indices['fold_{}'.format(idx)] = {
            'train_ids': idx_train,
            'test_ids': idx_test,
            'vali_ids': idx_vali,
        }

    return split_indices


def mixed_split(
    data: Tensor,
    label: Tensor,
    folds: int,
    seed: int,
    split: Tuple[float, float] = (0.85, 0.15),
) -> Dict:
    """
    Take given data and label tensors and split them into a training and test set.

    Args:
        data: Input dataset as tensor.
        label: Input labels as tensor.
        folds: Amount of folds.
        seed: PyTorch/Numpy RNG seed to deterministically create splits.
        split: Fraction of data in the test set.

    Returns:
        Index dictionary with n folds containing indices used in train, test and validation set.
    """
    x: np.ndarray = data.cpu().numpy()
    y: np.ndarray = label.cpu().numpy()
    split_indices = {}

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=split[1], random_state=seed)

    for train_index, test_index in sss_test.split(x, y):

        sss_vali = StratifiedShuffleSplit(n_splits=folds, test_size=1 / folds, random_state=seed)
        for idx, (train_index_, vali_index) in enumerate(sss_vali.split(x[train_index], y[train_index]), 1):

            split_indices['fold_{}'.format(idx)] = {
                'train_ids': train_index[train_index_].tolist(),
                'test_ids': test_index.tolist(),
                'vali_ids': train_index[vali_index].tolist(),
            }

    return split_indices


def load_split(
    data: Tensor,
    label: Tensor,
    index_dict: Dict,
    fold: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load split from a given fold of a previous run."""
    fold_dict: Dict = index_dict['fold_{}'.format(fold)]
    if not {'train_ids', 'test_ids', 'vali_ids'} == fold_dict.keys():
        raise ValueError(
            'Index dict does not contain the required keys: '
            '"train_ids", "test_ids" and "vali_ids" or'
            '"test_anormal_ids", "test_normal_ids".'
        )
    train_ids = torch.tensor(fold_dict['train_ids'], dtype=torch.int)
    test_ids = torch.tensor(fold_dict['test_ids'], dtype=torch.int)
    vali_ids = torch.tensor(fold_dict['vali_ids'], dtype=torch.int)

    return (
        data.index_select(0, train_ids),
        data.index_select(0, test_ids),
        data.index_select(0, vali_ids),
        label.index_select(0, train_ids),
        label.index_select(0, test_ids),
        label.index_select(0, vali_ids),
    )


def select_channels(data: Tensor, channels: Union[int, List[int]]) -> Tensor:
    """
    Select channels based on their indices (given as a list or an int).

    If an int n is selected, the first n columns are used. However, it is usually
    preferred to pass a list of ints to be sure to select the correct columns.
    Passed channels are zero-indexed.

    Args:
        data: Tensor containing series of shape :code:`(num_samples, seq_len, num_channels)`.
        channels: Selected channels.
    """
    if isinstance(channels, int) and data.shape[2] > channels:
        data = data[:, :, :channels]
    elif isinstance(channels, list) and data.shape[2] > len(channels):
        data = data[:, :, channels]

    return data


def verbose_channel_selection(data: Tensor, channels: Union[int, List[int]]) -> None:
    """
    Verbose output corresponding to the select_channel function.

    Args:
        data: Tensor containing series of shape :code:`(num_samples, seq_len, num_channels)`.
        channels: Selected channels.
    """
    if isinstance(channels, int) and data.shape[2] == channels:
        logger.info('All channels will be used, no channels are removed.')
    elif isinstance(channels, int) and data.shape[2] > channels:
        logger.info(
            'Removing {0} channel(s), the first {1} channel(s) will remain.'.format(data.shape[2] - channels, channels)
        )
    elif isinstance(channels, list) and data.shape[2] > len(channels):
        logger.info('Selecting channels by index: taking channels {0}.'.format(channels))
