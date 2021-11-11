"""Embedding logic for anomaly detection."""
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from torch import Tensor

from ecgan.config import get_global_config
from ecgan.utils.datasets import DatasetFactory
from ecgan.utils.embeddings import assert_and_reshape_dim, calculate_umap
from ecgan.utils.miscellaneous import to_numpy
from ecgan.visualization.plotter import ScatterPlotter

logger = getLogger(__name__)


class Embedder(ABC):
    """
    Abstract embedding class.

    Args:
        train_x: Data the initial embedding should be fit on.
        train_y: Labels the initial embedding should be fit on.
        dataset: Dataset identifier, has to be supported by :class:`ecgan.utils.datasets.DatasetFactory`
    """

    def __init__(
        self,
        train_x: Union[Tensor, np.ndarray],
        train_y: Union[Tensor, np.ndarray],
        dataset: str,
    ):
        self._train_x = to_numpy(train_x)
        self._train_y = to_numpy(train_y)
        self.classes = list(
            DatasetFactory()(dataset).beat_types_binary.keys()
            if get_global_config().trainer_config.BINARY_LABELS
            else list(DatasetFactory()(dataset).beat_types.keys())
        )

        self.dataset = dataset
        self.plotter = ScatterPlotter()
        self._initial_embedding: Optional[np.ndarray] = None
        self._reducer: Optional[BaseEstimator] = None

    @abstractmethod
    def get_initial_embedding(self) -> Tuple[np.ndarray, BaseEstimator]:
        """
        Create an initial embedding and train a reducer.

        Returns:
            The embedding data as well as the reducer.
        """
        raise NotImplementedError("Embedder needs to implement the `get_initial_embedding` method.")

    def embed_test(
        self,
        test_x: Union[Tensor, np.ndarray],
        test_y: Union[Tensor, np.ndarray],
        include_initial_embedding: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed test data into embedding pre-trained on the training data.

        Args:
            test_x: Test data.
            test_y: Test labels.
            include_initial_embedding: Flag to indicate if the embedding trained on train_x should be returned as well.

        Returns:
            (embedding, labels) tuple. The labels have shifted to avoid conflicts with existing train labels.
        """
        data = to_numpy(test_x)
        labels = to_numpy(test_y)
        return self._embed_test(data, labels, include_initial_embedding)

    @abstractmethod
    def _embed_test(
        self, test_x: np.ndarray, test_y: np.ndarray, include_initial_embedding: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed test data into embedding pre-trained on the training data.

        Args:
            test_x: Test data.
            test_y: Test labels.
            include_initial_embedding: Flag to indicate if the embedding trained on train_x should be returned as well.

        Returns:
            (embedding, labels) tuple. The labels have shifted to avoid conflicts with existing train labels.
        """
        raise NotImplementedError("Embedder needs to implement the `_embed_test` method.")

    def embed(self, embeddable_data: Tensor, labels: Optional[Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed novel data into given embedding. This data can be synthetic data.

        Args:
            embeddable_data: Any data which can be embedded into a trained embedding.
            labels: Labels corresponding to the embeddable data.

        Returns:
            (embedding, labels) tuple. The labels have shifted to avoid conflicts with existing train labels.
        """
        data_ = to_numpy(embeddable_data)
        labels_ = to_numpy(labels) if labels is not None else None

        return self._embed(data_, labels_)

    @abstractmethod
    def _embed(self, embeddable_data: np.ndarray, labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Embed the novel data into given train/test sets."""
        raise NotImplementedError("Embedder needs to implement the `_embed` method.")

    @abstractmethod
    def get_plot(self, embedding: np.ndarray, labels: np.ndarray) -> plt.Figure:
        """Retrieve a plot of the embedding as an np.ndarray."""
        raise NotImplementedError("Embedder needs to implement the `get_plot` method.")

    @abstractmethod
    def draw_interpolation_path(
        self,
        trace: Union[np.ndarray, Tensor],
        labels: Union[np.ndarray, Tensor],
        embedding: Optional[Union[np.ndarray, Tensor]] = None,
        fig_size: Tuple[float, float] = (15, 11.25),
    ) -> plt.Figure:
        """Draw an interpolation path of reconstructed data into an existing embedding."""
        raise NotImplementedError("Embedder needs to implement the `draw_interpolation_path` method.")

    @property
    def reducer(self) -> BaseEstimator:
        if self._reducer is None:
            _, reducer = self.get_initial_embedding()
            self._reducer = reducer

        return self._reducer

    @property
    def initial_embedding(self) -> np.ndarray:
        if self._initial_embedding is None:
            embedding, _ = self.get_initial_embedding()
            self._initial_embedding = embedding

        return self._initial_embedding

    @property
    def initial_labels(self) -> np.ndarray:
        return self._train_y


class UMAPEmbedder(Embedder):
    """Create UMAP embedding (on train/vali data) and allow embedding of novel (test) data."""

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        dataset,
    ):
        super().__init__(train_x, train_y, dataset)
        self.total_classes = self.classes
        self.len_fit_labels: int = len(self.classes)

    def get_initial_embedding(self) -> Tuple[np.ndarray, BaseEstimator]:
        """Create an initial embedding."""
        logger.info('Creating initial UMAP embedding...')

        initial_embedding, reducer = calculate_umap(self._train_x, self.initial_labels, supervised_umap=True)

        logger.info('Created initial UMAP embedding.')
        # Store the reducer for future inference of the remaining data.
        # Can be used to avoid recalculation each run since the embeddings will usually be similar.
        self._reducer = reducer
        self._initial_embedding = initial_embedding

        return initial_embedding, reducer

    def _embed_test(
        self,
        test_x: np.ndarray,
        test_y: np.ndarray,
        include_initial_embedding: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Embed remaining test data using a reducer fit onto the initial embedding."""
        if self.reducer is None or self.initial_embedding is None:
            self.get_initial_embedding()

        test_data = assert_and_reshape_dim(test_x)

        embedding_test: np.ndarray = self.reducer.transform(test_data)  # type: ignore
        # Shift all labels to avoid label conflicts
        test_labels: np.ndarray = test_y + self.len_fit_labels

        self.total_classes = self.total_classes + [classname + '_test' for classname in self.classes]

        if not include_initial_embedding:
            return embedding_test, test_labels

        embedding = np.concatenate((self.initial_embedding, embedding_test))
        labels = np.concatenate((self._train_y, test_labels))

        return embedding, labels

    def _embed(self, embeddable_data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create UMAP embedding of the real and synthetic data.

        Create initial embedding and reducer if this is the first run. The initial embedding contains all train data and
        a subset of the test data. This embedding is reused in subsequent runs to reduce the computational cost.
        """
        if self.reducer is None or self.initial_embedding is None:
            self.get_initial_embedding()

        embedding = self.reducer.transform(assert_and_reshape_dim(embeddable_data))  # type: ignore
        self.total_classes = self.total_classes + [classname + '_embedded' for classname in self.classes]

        if labels is None:
            len_interpolated_labels = 2 * self.len_fit_labels
            updated_labels = np.full((embeddable_data.shape[0],), len_interpolated_labels)
        else:
            updated_labels = labels + 2 * self.len_fit_labels

        return embedding, updated_labels

    def get_plot(self, embedding: np.ndarray, labels: np.ndarray) -> plt.Figure:
        """Retrieve a plot of the embedding as a matplotlib Figure."""
        return self.plotter.plot_scatter(
            data=embedding,
            target=labels,
            fig_title='',
            classes=self.total_classes,
        )

    def draw_interpolation_path(
        self,
        trace: Union[np.ndarray, Tensor],
        labels: Union[np.ndarray, Tensor],
        embedding: Optional[Union[np.ndarray, Tensor]] = None,
        fig_size: Tuple[float, float] = (15, 11.25),
    ) -> plt.Figure:
        """
        Plot a trace/path in latent space between data in the trace tensor.

        Args:
            trace: The trace to draw.
            labels: The data labels.
            embedding: The embedding the trace should be embedded into.
            fig_size: Size of the matplotlib figure (width, height).

        Returns:
            A matplotlib Figure of the visualized embedding.
        """
        embedding_np = to_numpy(embedding) if embedding is not None else self.initial_embedding
        labels_np = to_numpy(labels)
        trace_np: np.ndarray = to_numpy(trace)

        embedding_trace = self.reducer.transform(assert_and_reshape_dim(trace_np))

        return self.plotter.plot_interpolation_path(
            data=embedding_np,
            labels=labels_np,
            trace=embedding_trace,
            fig_size=fig_size,
            classes=self.total_classes,
        )
