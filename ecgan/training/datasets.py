"""Dataset base class extending the PyTorch Dataset class specifications."""
import random
from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor, stack
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Extend PyTorch Dataset class with explicit sampling and __len__ function."""

    def sample(self, batch_size: int) -> Dict:
        """
        Sample a batch directly from the dataset.

        Args:
            batch_size: Amount of samples to sample.

        Returns:
            Dict containing the samples and its attributes.
        """
        indices = random.sample(range(len(self)), batch_size)
        sample_dicts = [self[idx] for idx in indices]
        collated_samples: Dict = {key: [] for key in sample_dicts[0].keys()}
        for sample_dict in sample_dicts:
            for key, val in sample_dict.items():
                collated_samples[key].append(val)

        return {key: stack(val) for key, val in collated_samples.items()}

    @abstractmethod
    def __len__(self) -> int:
        """
        Return number of samples in dataset.

        Returns:
            Number of samples in dataset.
        """
        raise NotImplementedError("Dataset needs to implement the `__len__` method.")


class SeriesDataset(BaseDataset):
    """PyTorch Dataset class for time series that are preprocessed using :code:`ecgan-preprocess`."""

    def __init__(self, data: Tensor, label: Tensor):
        """Load dataset to memory and transforms it to tensor."""
        self.data = data
        self.label = label

        self.num_classes = len(self.label.unique())

    def __len__(self) -> int:
        """
        Return number of samples in dataset.

        Returns:
            Number of samples in dataset.
        """
        return len(self.label)

    def __getitem__(self, idx: int) -> dict:
        """
        Given an index, return the corresponding data pair.

        Args:
            idx: Index of entry in dataset.

        Returns:
            Dict with the respective time series and its label.
        """
        return {'data': self.data[idx], 'label': self.label[idx]}
