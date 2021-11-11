"""Functions to create low dimensional embeddings."""
from logging import getLogger
from time import time
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = getLogger(__name__)


def calculate_tsne(
    data: np.ndarray,
    perplexity: float = 30,
    early_exaggeration: float = 12.0,
    n_components: int = 2,
) -> Tuple[np.ndarray, BaseEstimator]:
    """
    Calculate t-SNE.

    This is a wrapper function for the corresponding sklearn implementation. Can be applied to both, univariate as well
    as multivariate series. Can be visualized with the :class:`ecgan.visualization.plotter.ScatterPlotter`.
    Keep in mind rerunning t-SNE will not return the same embeddings on different runs because its cost function is not
    convex. t-SNE is slow in comparison to e.g. UMAP, to speed up training the reducer, one might want to train it on
    the GPU, a cuda implementation can be found `on GitHub (CannyLab) <https://github.com/CannyLab/tsne-cuda>`_.

    Args:
        data: Data whose dimensionality shall be reduced. Either (batch, seq_len) or (batch, seq_len, channel) format.
        perplexity: t-SNE perplexity (more information e.g. `here <https://distill.pub/2016/misread-tsne/>`_.
        early_exaggeration: Controls how tight the embedded points are packed.
        n_components: Dimension of the embedded space.

    References:
        `van der Maaten and Hinton, 2008 <https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf>`_

    Returns:
        The resulting low-dim embedding with shape (dims, samples) and the trained reducer
    """
    logger.info('Creating t-SNE embedding...')

    # Data is expected to be a Tensor of shape (a,b). (a,b,c) will be reshaped.
    data = assert_and_reshape_dim(data)

    start = time()
    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        init='pca',
        learning_rate='auto',
    )

    tsne_embedding: np.ndarray = reducer.fit_transform(data)

    logger.info(
        'Computed TSNE embedding in {0} seconds. KL divergence between both'
        ' spaces after optimization is {1}.'.format(time() - start, reducer.kl_divergence_)
    )
    return tsne_embedding, reducer


def calculate_umap(
    data: np.ndarray,
    target: Union[List[int], object],
    n_neighbors: int = 25,
    supervised_umap: bool = True,
    n_components: int = 2,
    rnd_seed: Optional[int] = None,
    low_memory: bool = True,
) -> Tuple[np.ndarray, BaseEstimator]:
    """
    UMAP embeddings according to `McInnes et al. 2018 <https://arxiv.org/abs/1802.03426>`_.

    Using the `public UMAP implementation <https://umap-learn.readthedocs.io/en/latest/>`_ for 2D visualizations.

    Args:
        data: Univariate or multivariate series as numpy array tensor.
        target: List of the target classes encoded as integers.
        n_neighbors: Amount of UMAP neighbors used to construct the graph in high dimensionality.
        supervised_umap: Flag indicating if we want to use supervised umap, utilizing the target info.
        n_components: Dimensionality of low dim. embedding.
        rnd_seed: Set random seed if you want to reproduce the embedding. Warning: Slows down performance!
        low_memory: Enables or disables the low memory mode. Should be True if you run into memory problems during
            NNDescent. More time required during computation if enabled.

    Returns:
        The resulting low-dim UMAP embedding of shape (dim, samples).
    """
    logger.info('Creating UMAP embedding...')
    from umap.umap_ import UMAP  # workaround until umap init times are fixed. pylint: disable=C0415

    data = assert_and_reshape_dim(data)
    time_start = time()
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=rnd_seed,
        low_memory=low_memory,
    )

    if supervised_umap:
        embedding = reducer.fit_transform(data, y=target)
    else:
        embedding = reducer.fit_transform(data)
    logger.info('Finished constructing UMAP embedding after {0} seconds.'.format((time() - time_start)))

    return embedding, reducer


def calculate_pca(
    data: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, BaseEstimator]:
    """
    PCA embeddings using the sklearn library.

    Args:
        data: Univariate or multivariate series as numpy array tensor.
        n_components: Dimensionality of low dim. embedding.

    Returns:
        The resulting low-dim PCA embedding of shape (dim, samples) and the trained reducer.
    """
    logger.info('Creating PCA embedding...')

    data = assert_and_reshape_dim(data)
    time_start = time()
    reducer = PCA(n_components=n_components)

    embedding: np.ndarray = reducer.fit_transform(data)
    logger.info('Finished constructing PCA embedding after {0} seconds.'.format((time() - time_start)))

    return embedding, reducer


def assert_and_reshape_dim(data: np.ndarray) -> np.ndarray:
    """
    Assert that data is either of shape (a,b) or (a,b,c) and reshape if required.

    Args:
        data: Data of arbitrary shape.

    Returns:
        Reshaped 2D np.ndarray.
    """
    if len(data.shape) != 2 and len(data.shape) != 3:
        raise RuntimeError('Can only handle data of shape (a,b) or (a,b,c). Not {0}.'.format(data.shape))

    if len(data.shape) == 2:
        return data

    # Reshape data if the data is of shape (a,b,c), e.g. if it is a multivariate time series:
    amount_of_series, amount_of_samples, amount_of_channels = data.shape
    return data.reshape((amount_of_series, amount_of_channels * amount_of_samples))
