"""Describes different plotting classes."""
import math
from typing import List, Optional, Tuple, Union

import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Axes, Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import fftpack
from torch import Tensor

from ecgan.config import TrainerConfig
from ecgan.utils.custom_types import PlotterType, Transformation
from ecgan.utils.miscellaneous import to_numpy, to_torch


def matplotlib_prep(size: Tuple[int, int], subplots: int = 1, y_lim: Optional[Tuple] = None) -> Tuple[Figure, Axes]:
    """
    Return a figure and an axis for a matplotlib plot with a given size.

    Args:
        size: Target (width, height) of the plot.
        subplots: amount of subplots of the figure.
        y_lim: Visual y limits of the plot.

    Returns:
        The processed figure and axes.
    """
    # Standard dpi in matplotlib is 100.
    dpi = 100
    fig, ax = plt.subplots(subplots, figsize=(size[0] / dpi, size[1] / dpi))
    plt.tight_layout()

    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    return fig, ax


class BasePlotter:
    """Base class for plotting classes. Creates its plots by simple call to plt.plot."""

    @staticmethod
    def _get_data_axes(data: Union[Tensor, np.ndarray]) -> np.ndarray:
        """
        Get the x and y axis corresponding to a given time series.

        The BasePlotter assumes a equidistantly sampled time series in which
        case the x-values of the plot are set to [0, 1,..., seq_len].
        """
        return np.array([range(data.shape[0]), data])

    def create_plot(
        self,
        data: np.ndarray,
        color: str = 'blue',
        size: Tuple[int, int] = (256, 256),
        y_lim: Optional[Tuple[float, float]] = None,
        label: Optional[int] = None,
    ) -> Figure:
        """
        Generate a plot from given data (default: 2D time series).

        How the plot is created is subject to the given class implementation. The plot is returned as a mpl Figure.

        Args:
            data: A list of data points representing a 2D time series.
            color: The color in which to draw the plot.
            size: The size the plot image is resized to. Resizing is done via cubic interpolation.
            y_lim: Limit the y-axis of the plot to the floating tuple.
            label: Optional label which can be drawn into the plot.

        Returns:
            The generated images.
        """
        fig, axs = matplotlib_prep(size, y_lim=y_lim)

        self._plot(data=data, axes=axs, color=color, label=label)

        return fig

    def get_sampling_grid(
        self,
        sample_data: Union[Tensor, np.ndarray],
        max_num_series: int = 16,
        row_width: int = 4,
        color: str = 'blue',
        scale_per_batch: bool = False,
        label: Optional[Union[Tensor, np.ndarray]] = None,
        x_axis: bool = True,
        y_axis: bool = True,
        fig_size: Optional[Tuple[float, float]] = None,
    ) -> Figure:
        """
        Get sampled time series data image grid.

        Args:
            sample_data: The data that shall be visualized with shape (amount_of_series, measurements, channels).
            max_num_series: Maximum amount of samples visualized.
            row_width: Row width.
            color: Sets the colour of the plots.
            scale_per_batch: Set the y-limit for each plot to min/max of the batch (per channel).
            label: Optional labels parameter, which is written to the plots.
            x_axis: Flag indicating whether the x-axis should be visible.
            y_axis: Flag indicating whether the y-axis should be visible.
            fig_size: Optional size for the figure in inches.

        Returns:
            Figure of the image grid.
        """
        sample_data_ = to_numpy(sample_data)

        if fig_size is None:
            # Make the figure wider if more than the default 4 figures are in one row.
            extended_row_width = row_width / 5 if row_width > 4 else 1.0
            arbitrary_extension = max_num_series / (row_width * 5)
            arbitrary_extension = 1.0 if arbitrary_extension < 1 else arbitrary_extension
            # DIN A4 size in inches: (11.69, 8.27)
            extended_height = 8.27 * arbitrary_extension
            extended_width = 11.69 * arbitrary_extension * extended_row_width
            fig_size = (extended_width, extended_height)

        fig = plt.figure(figsize=fig_size)

        num_data, seq_len, num_channels = sample_data_.shape
        num_data = min(num_data, max_num_series)

        # Univariate series
        if num_channels == 1:
            sample_data_.reshape((num_data, seq_len, num_channels))

        num_of_rows = math.ceil(num_data / row_width)
        outer_grid = GridSpec(num_of_rows, row_width, wspace=0.1, hspace=0.1)

        y_lim = None
        if scale_per_batch:
            y_lim = (np.min(sample_data_), np.max(sample_data_))

        for i in range(num_of_rows):
            for j in range(row_width):
                inner_grid = GridSpecFromSubplotSpec(
                    num_channels,
                    1,
                    subplot_spec=outer_grid[i * row_width + j],
                    wspace=0.3,
                    hspace=0.15,
                )
                for k in range(num_channels):
                    plot_number = i * row_width + j
                    if plot_number >= num_data:
                        break
                    ax = plt.Subplot(fig, inner_grid[k])

                    ax.get_xaxis().set_visible(x_axis)
                    ax.get_yaxis().set_visible(y_axis)

                    if y_lim is not None:
                        ax.set_ylim(y_lim[0], y_lim[1])
                    if k != num_channels - 1:
                        ax.get_xaxis().set_visible(False)

                    label_ = int(label[plot_number]) if label is not None else None
                    self._plot(
                        sample_data_[plot_number, :, k],
                        label=label_,
                        axes=ax,
                        color=color,
                    )
                    fig.add_subplot(ax)

        outer_grid.tight_layout(fig)
        return fig

    def _plot(
        self,
        data: np.ndarray,
        axes: Axes,
        color: str,
        label: Optional[int] = None,
    ) -> None:
        """
        Plot data to given axes.

        Args:
            data: Data to plot.
            axes: Axes to plot the data on.
            color: Graph color.
            label: Numeric label of the data class.
        """
        transformed_data = self._get_data_axes(data)
        x, y = transformed_data
        axes.plot(x, y, color=color)

        if label is not None:
            axes.set_title('Class: {}'.format(label), loc='center')

    def save_sampling_grid(
        self,
        sample_data: Union[Tensor, np.ndarray],
        file_location: str,
        color: str = 'blue',
        max_num_series: int = 16,
        scale_per_batch=False,
        row_width: int = 4,
        label: Optional[Union[Tensor, np.ndarray]] = None,
    ) -> None:
        """
        Save sampled time series data to an image grid.

        Args:
            sample_data: The data that shall be visualized with shape (amount_of_series, 1, measurements).
            color: Color of the plot.
            file_location: Path to file on local system. Should be a UNIQUE identifier.
            max_num_series: Maximum amount of samples visualized.
            row_width: Width of a row.
            scale_per_batch: Set the y-limit for each plot to min/max of the batch (per channel).
            label: Optional labels parameter, which is written to the plots.
        """
        image_grid = self.get_sampling_grid(
            sample_data,
            max_num_series,
            row_width,
            color=color,
            scale_per_batch=scale_per_batch,
            label=label,
        )
        image_grid.savefig(file_location)

    @staticmethod
    def create_histogram(
        data: np.ndarray,
        title: str,
        x_label: str = '',
        y_label: str = '',
        bins: int = 50,
        color: str = 'g',
    ) -> Figure:
        """Create a histogram of given data."""
        fig = plt.figure()
        plt.hist(data, bins, density=True, facecolor=color, alpha=0.75)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)

        return fig

    @staticmethod
    def save_plot(
        plot: Union[Figure, np.ndarray],
        file_location: str,
    ) -> None:
        """Save a plot (encoded as np.ndarray) to file_location."""
        if isinstance(plot, Figure):
            plot.savefig(file_location)
            plt.close(plot)

        else:
            plt.imsave(file_location, np.ascontiguousarray(plot))

    @staticmethod
    def create_error_plot(  # pylint: disable=R0913
        data_lined: Union[np.ndarray, Tensor],
        data_dashed: Union[np.ndarray, Tensor],
        heatmap_data: Optional[np.ndarray] = None,
        x_axis: Optional[np.ndarray] = None,
        data_range: Optional[Tuple[float, float]] = None,
        color_lined: str = 'blue',
        color_dashed: str = 'red',
        color_map: str = 'plasma',
        x_label: str = '',
        y_label: str = '',
        title: str = '',
    ) -> Figure:
        """
        Create a plot with two graphs visualizing the difference of the two samples and a heatmap.

        Args:
            data_lined: The sample data that is depicted as a solid line.
            data_dashed: The sample data that is depicted as a dashed line.
            heatmap_data: Optional data for the drawing of the heatmap.
                The default simply computes the absolute value of (data_dashed - data_lined)
            x_axis: Sampling of the x-axis. Default assumes range(1, len_of_samples).
            data_range: Range of the heatmap. Dynamic by default, requires (min,max) otherwise.
            color_lined: Color of the lined plot.
            color_dashed: Color of the dashed plot.
            color_map: Color map for the heatmap.
            x_label: Label of the x-axis.
            y_label: Label of the y-axis.
            title: Title for the plot.

        Returns:
            Heatmap Figure.
        """
        data_lined = to_numpy(data_lined)
        data_dashed = to_numpy(data_dashed)
        if x_axis is None:
            len_ = len(data_lined) if len(data_lined) > len(data_dashed) else len(data_dashed)
            x_axis = np.arange(len_)

        if heatmap_data is None:
            heatmap_data = abs(data_dashed - data_lined)

        _, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [6, 1]})

        axes[0].plot(x_axis, data_lined, color=color_lined)
        axes[0].plot(x_axis, data_dashed, color=color_dashed, linestyle="--")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        axes[1].get_yaxis().set_visible(False)

        if data_range is not None:
            norm = mplcolors.Normalize(vmin=data_range[0], vmax=data_range[1])
            heatmap = axes[1].imshow([heatmap_data], cmap=color_map, aspect='auto', norm=norm)
        else:
            heatmap = axes[1].imshow([heatmap_data], cmap=color_map, aspect='auto')
        plt.colorbar(heatmap, ax=axes)

        return plt.gcf()


class FourierPlotter(BasePlotter):
    """Plotter implementation to plot data in the Fourier-domain as a frequency-histogram."""

    @staticmethod
    def _transform_to_complex(data: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Transform a tensor to a complex tensors.

        Args:
            data: Input data of shape (batch_size, seq_len, 2*num_channels)

        Returns:
            Tensor of shape (batch_size, seq_length, num_channels) with complex entries.
        """
        data = to_torch(data)

        batch_size, seq_length, twice_num_channels = data.shape
        num_channels = twice_num_channels // 2
        dtype = torch.cfloat if data.data == torch.float else torch.cdouble
        result = torch.empty((num_channels, batch_size, seq_length), dtype=dtype)

        for channel in range(num_channels):
            real = data[:, :, 2 * channel]
            imag = data[:, :, 2 * channel + 1]

            result[channel] = torch.complex(real, imag)

        return result.permute(1, 2, 0)

    def get_sampling_grid(
        self,
        sample_data: Union[Tensor, np.ndarray],
        max_num_series: int = 16,
        row_width: int = 4,
        color: str = 'blue',
        scale_per_batch: bool = False,
        label: Optional[Union[Tensor, np.ndarray]] = None,
        x_axis: bool = True,
        y_axis: bool = True,
        fig_size: Optional[Tuple[float, float]] = None,
    ) -> Figure:
        """
        Get sampled time series data image grid.

        Args:
            sample_data: The data that shall be visualized. The tensor is assumed to be of shape
                (batch_size, seq_length, 2 * num_of_channels), where each two channels form the real and imaginary parts
                of the Fourier coefficients.
            max_num_series: Maximum amount of samples visualized.
            row_width: Amount of columns if grid shall not be symmetric.
            color: Sets the colour of the plots.
            scale_per_batch: Set the y-limit for each plot to min/max of the batch (per channel).
            label: Optional labels parameter, which is written to the plots.
            x_axis: Flag indicating whether the x-axis should be visible.
            y_axis: Flag indicating whether the y-axis should be visible.
            fig_size: Optional size for the figure in inches.

        Returns:
            Image grid as Figure.
        """
        # First restore the tensor to a complex number.
        complex_tensor = self._transform_to_complex(sample_data)
        # Now the sampling can continue as declared in the super class.
        return super().get_sampling_grid(
            complex_tensor,
            max_num_series,
            row_width,
            color,
            scale_per_batch,
            label,
        )

    @staticmethod
    def _transform_data(data: Union[Tensor, np.ndarray]) -> np.ndarray:
        """Transform a data tensor in Fourier domain to a histogram of form (freq, abs(data))."""
        data = to_numpy(data)
        seq_len = data.shape[0]
        sample_rate = 2 * seq_len  # Nyquist theorem
        freqs: np.ndarray = fftpack.fftfreq(len(data)) * sample_rate

        return freqs

    def create_plot(
        self,
        data: np.ndarray,
        color: str = 'blue',
        size: Tuple[int, int] = (256, 256),
        y_lim: Optional[Tuple[float, float]] = None,
        label: Optional[int] = None,
    ) -> Figure:
        """Create a plot of the data using the settings of the plotting class."""
        transformed_data = self._transform_data(data)
        frequencies = transformed_data
        abs_ = np.abs(data)
        x = range(data.shape[0])
        y = torch.fft.ifftn(torch.from_numpy(data), norm='ortho').numpy()
        y = np.real(y)
        bottom, top = plt.ylim()
        fig, ax = matplotlib_prep(size, subplots=2, y_lim=y_lim)
        ax[1].plot(x, y, color=color)
        plt.ylim(bottom, top)

        if label is not None:
            ax[0].set_title('Class: {}'.format(label), loc='center')

        ax[0].stem(frequencies, abs_, markerfmt=" ")

        return fig


class ScatterPlotter(BasePlotter):
    """Plotter specialized on scatter plots for large datasets."""

    @staticmethod
    def truncate_colormap(
        cmap: mplcolors.Colormap, min_val: float = 0.0, max_val: float = 1.0, n_cmap: int = 100
    ) -> mplcolors.LinearSegmentedColormap:
        """
        Truncate a given matplotlib colormap to a specific range.

        Args:
            cmap: matplotlib colormap object.
            min_val: Minimum bound of colormap.
            max_val: Maximum bound of colormap.
            n_cmap: Number of samples from original cmap.

        Returns:
            New colormap object
        """
        new_cmap = mplcolors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
            cmap(np.linspace(min_val, max_val, n_cmap)),
        )
        return new_cmap

    @staticmethod
    def plot_scatter(
        data: np.ndarray,
        target: Union[List[int], object],
        fig_title: Optional[str],
        classes: Optional[List[str]],
        dpi: int = 300,
        alpha: float = 1,
        cmap: str = 'plasma',
    ) -> Figure:
        """
        Visualizes the resulting low dimensional (2D or 3D) embedding.

        Args:
            data: Low dimensional embedding, either 2D or 3D plots, i.e. shape (n_samples, 2) or (n_samples, 3).
            target: Target values used for visualization encoded as integers.
            fig_title: Description of the model saved. Should include the name of the embedding.
            classes: Class names according to the encoding in `target`.
            dpi: DPI of the resulting figure.
            alpha: Alpha blending value. Is between 0 (transparent) and 1 (opaque).
            cmap: Colormap.

        Returns:
            Figure containing the scatter plot.
        """
        assert (
            data.shape[1] == 2 or data.shape[1] == 3
        ), 'Can only visualize 2D or 3D embeddings input shape was {0}.'.format(data.shape)

        fig = plt.figure(figsize=(14, 10))
        fig._set_dpi(dpi)  # pylint: disable=W0212

        if data.shape[1] == 2:
            ax = fig.add_subplot(111)
            scatterplot = ax.scatter(
                data[:, 0],
                data[:, 1],
                alpha=alpha,
                cmap=cmap,
                c=target,
                s=1,
            )

            plt.setp(ax, xticks=[], yticks=[])
        else:
            ax = fig.add_subplot(111, projection='3d')
            scaling = 3 / np.log10(data.shape[0]) if data.shape[0] > 5000 else 30 / np.log10(data.shape[0])
            scatterplot = ax.scatter(
                data[:, 0],
                data[:, 1],
                data[:, 2],
                s=scaling,
                alpha=1.0,
                cmap=cmap,
                c=target,
            )
            plt.setp(ax, xticks=[], yticks=[], zticks=[])

        if classes is not None:
            cbar = fig.colorbar(scatterplot, boundaries=np.arange((len(classes) + 1)) - 0.5)
            cbar.set_ticks(np.arange(len(classes)))
            cbar.set_ticklabels(classes)

        if fig_title is not None:
            plt.title(fig_title, fontsize=18, y=1.03)

        return fig

    @staticmethod
    def plot_interpolation_path(
        data: np.ndarray,
        labels: np.ndarray,
        trace: np.ndarray,
        classes: Optional[List[str]],
        fig_size: Tuple[float, float] = (10, 7.5),
        cmap: str = 'plasma',
        cmap_range: Tuple[float, float] = (0.0, 1.0),
        path_color: str = 'r',
        scatter_alpha: float = 1.0,
    ) -> Figure:
        """Plot a trace between points in a scatter plot."""
        fig, ax = plt.subplots(figsize=fig_size)

        # Plot latent embedding
        new_cmap = ScatterPlotter.truncate_colormap(plt.get_cmap(cmap), cmap_range[0], cmap_range[1])
        scatterplot = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=new_cmap, alpha=scatter_alpha, s=0.5)

        # Create legend
        if classes is not None:
            cb = fig.colorbar(scatterplot, shrink=0.75, boundaries=np.arange((len(classes) + 1)) - 0.5)
            cb.solids.set(alpha=1)
            cb.set_ticks(np.arange(len(classes)))
            cb.set_ticklabels(classes)

        # Plot walk route
        if trace is None:
            raise RuntimeError("No trace supplied to `ScatterPlotter.plot_interpolation_path`.")
        ax.plot(trace[:, 0], trace[:, 1], c=path_color)
        ax.scatter(trace[0, 0], trace[0, 1], s=0.5, c=path_color, marker='X', alpha=1.0)
        ax.arrow(
            trace[-2, 0],
            trace[-2, 1],
            trace[-1, 0] - trace[-2, 0],
            trace[-1, 1] - trace[-2, 1],
            color=path_color,
            head_width=0.1,
            head_length=0.1,
        )

        # fig.tight_layout()
        ax.axis('off')

        return fig


class PlotterFactory:
    """Used to retrieve instance of desired plotter."""

    @staticmethod
    def choose_class(plotter_type: PlotterType):
        """Choose the correct class based on the provided plotter name."""
        if plotter_type == PlotterType.FOURIER:
            return FourierPlotter
        if plotter_type == PlotterType.BASE:
            return BasePlotter
        if plotter_type == PlotterType.SCATTER:
            return ScatterPlotter

        raise AttributeError('Argument {0} is not set correctly.'.format(plotter_type))

    def __call__(self, plotter_type: PlotterType, **kwargs) -> BasePlotter:
        """Create and return Plotter object."""
        cls = PlotterFactory.choose_class(plotter_type)
        base_plotter: BasePlotter = cls()
        return base_plotter

    @staticmethod
    def from_config(train_cfg: TrainerConfig) -> BasePlotter:
        """Generate a plotter from a config dictionary."""
        if train_cfg.transformation == Transformation.FOURIER:
            plotter = PlotterFactory()(PlotterType.FOURIER)
        else:
            plotter = PlotterFactory()(PlotterType.BASE)

        return plotter


def visualize_reconstruction(
    series: torch.Tensor,
    plotter: BasePlotter,
    max_intermediate_samples: int = 10,
) -> Figure:
    """
    Visualize a fixed amount of series for a variable amount of input series.

    Total steps=max_intermediate_samples+original sample+final sample.

    Args:
        series: Tensor of series that shall be reconstructed.
        plotter: The plotter to use for the visualization.
        max_intermediate_samples: Maximum amount of interpolation steps.

    Returns:
        A mpl Figure containing a sequence of series.
    """
    max_intermediate_samples = len(series) if len(series) // max_intermediate_samples == 0 else max_intermediate_samples
    idx_list = list(range(0, len(series), len(series) // max_intermediate_samples))
    idx_list.append(len(series) - 1)

    return plotter.get_sampling_grid(series[idx_list])
