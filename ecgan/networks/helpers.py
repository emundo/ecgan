"""Helper functions and networks."""
# pylint: disable=C0103
from logging import getLogger
from typing import List, Optional

from torch import nn

from ecgan.utils.custom_types import InputNormalization

logger = getLogger(__name__)


def apply_input_normalization(
    channel_size: int, normalization: Optional[InputNormalization], **kwargs
) -> Optional[nn.Module]:
    """
    Apply input normalization to a layer of size `channel_size`.

    Args:
        channel_size: Size of the channel/layer.
        normalization: Selected normalization method.
        kwargs: Optional parameters which might be required for normalizations.
    """
    if normalization is InputNormalization.NONE or normalization is None:
        return None
    if normalization == InputNormalization.BATCH:
        return nn.BatchNorm1d(channel_size, track_running_stats=kwargs.get('track_running_stats', True))
    if normalization == InputNormalization.GROUP:
        return nn.GroupNorm(kwargs.get('n_groups', 2), channel_size)
    logger.warning("Invalid normalization {} - defaulting to no normalization.".format(normalization.value))
    return None


def conv1d_block(
    in_channels: int, out_channels: int, k: int = 4, s: int = 2, p: int = 1, bias: bool = False
) -> nn.Conv1d:
    """
    Abbreviate the creation of a Conv1d block.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        k: kernel size.
        s: stride.
        p: padding.
        bias: bias.

    Returns:
        A Conv1d block.
    """
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=k,
        stride=s,
        padding=p,
        bias=bias,
    )


def conv1d_trans_block(
    in_channels: int, out_channels: int, k: int = 4, s: int = 2, p: int = 1, bias: bool = False
) -> nn.ConvTranspose1d:
    """
    Abbreviate the creation of a 1d convolutional transpose block.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        k: kernel size.
        s: stride.
        p: padding.
        bias: bias.

    Returns:
        A ConvTranspose1d block.
    """
    return nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=k,
        stride=s,
        padding=p,
        bias=bias,
    )


def create_5_hidden_layer_convnet(
    input_channels: int,
    hidden_channels: List[int],
    output_channels: int,
    seq_len: int,
    input_norm: InputNormalization,
    spectral_norm: bool = False,
    track_running_stats: bool = True,
) -> nn.Sequential:
    """
    Generate a downsampling CNN architecture, with LeakyReLU activation and optionally weight/input normalization.

    .. note::
        seq_len has to be divisible by 32 for the pooling kernel.

    Args:
        input_channels: Amount of input channels.
        hidden_channels: List of hidden channel sizes. Should be of length 5.
        output_channels: Amount of output channels.
        seq_len: Sequence length of the data.
        input_norm: Type of input normalization.
        spectral_norm: Flag to indicate if spectral weight normalization should be performed
        track_running_stats: Flag to indicate if a BatchNorm layer should track the running statistics.

    Returns:
        A five hidden layer CNN as nn.Module.
    """
    pooling_kernel = seq_len // 32
    # Exemplary calculation for seq_len=320 -> pooling_kernel=10.
    #####################################
    # CONV LAYER 1 IN: IN_CHANNELS x 320, OUT: hidden_channels x 160
    #####################################
    conv1 = conv1d_block(input_channels, hidden_channels[0], k=4, s=2, p=1)
    #####################################
    # CONV LAYER 2 OUT : HIDDEN x 80
    #####################################
    conv2 = conv1d_block(hidden_channels[0], hidden_channels[1], k=4, s=2, p=1)
    #####################################
    # CONV LAYER 3 OUT : HIDDEN x 40
    #####################################
    conv3 = conv1d_block(hidden_channels[1], hidden_channels[2], k=4, s=2, p=1)
    #####################################
    # CONV LAYER 4 OUT : HIDDEN x 20
    #####################################
    conv4 = conv1d_block(hidden_channels[2], hidden_channels[3], k=4, s=2, p=1)
    #####################################
    # CONV LAYER 5 OUT : HIDDEN x 10
    #####################################
    conv5 = conv1d_block(hidden_channels[3], hidden_channels[4], k=4, s=2, p=1)
    #####################################
    # CONV LAYER 6 OUT : OUT_CHANNELS x 1
    #####################################
    conv6 = conv1d_block(hidden_channels[4], output_channels, k=pooling_kernel, s=1, p=0)

    if spectral_norm:
        logger.info("Using weight normalization spectral norm in conv net.")
        conv2 = nn.utils.spectral_norm(conv2)
        conv3 = nn.utils.spectral_norm(conv3)
        conv4 = nn.utils.spectral_norm(conv4)
        conv5 = nn.utils.spectral_norm(conv5)

    logger.info("Using {} input normalization in conv net.".format(input_norm))

    norm1 = apply_input_normalization(hidden_channels[1], input_norm, track_running_stats=track_running_stats)
    norm2 = apply_input_normalization(hidden_channels[2], input_norm, track_running_stats=track_running_stats)
    norm3 = apply_input_normalization(hidden_channels[3], input_norm, track_running_stats=track_running_stats)
    norm4 = apply_input_normalization(hidden_channels[4], input_norm, track_running_stats=track_running_stats)

    module_list = [
        conv1,
        nn.LeakyReLU(0.2, inplace=True),
        conv2,
        norm1,
        nn.LeakyReLU(0.2, inplace=True),
        conv3,
        norm2,
        nn.LeakyReLU(0.2, inplace=True),
        conv4,
        norm3,
        nn.LeakyReLU(0.2, inplace=True),
        conv5,
        norm4,
        nn.LeakyReLU(0.2, inplace=True),
        conv6,
    ]

    module_list = [mod for mod in module_list if mod is not None]
    net = nn.Sequential(*module_list)  # type: ignore

    return net


def create_transpose_conv_net(
    input_channels: int,
    hidden_channels: List[int],
    output_channels: int,
    seq_len: int,
    input_norm: InputNormalization,
    spectral_norm: bool = False,
    track_running_stats=True,
) -> nn.Sequential:
    """
    Create a 5 hidden layer conv transposed network.

    Args:
        input_channels: Amount of input channels.
        hidden_channels: List of hidden channel sizes. Should be of length 5.
        output_channels: Amount of output channels.
        seq_len: Sequence length of the data.
        input_norm: Type of input normalization.
        spectral_norm: Flag to indicate if spectral weight normalization should be performed
        track_running_stats: Flag to indicate if a BatchNorm layer should track the running statistics.

    Returns:
        A five hidden layer transposed CNN as nn.Module.
    """
    pooling_kernel = seq_len // 32
    #####################################
    # CONV LAYER 1 IN: LATENT_SIZE x 1 OUT : HIDDEN x 10 (pooling_kernel)
    #####################################
    conv1t = conv1d_trans_block(input_channels, hidden_channels[0], k=pooling_kernel, s=1, p=0)
    #####################################
    # CONV LAYER 2 OUT : HIDDEN x 20
    #####################################
    conv2t = conv1d_trans_block(hidden_channels[0], hidden_channels[1], k=4, s=2, p=1)
    #####################################
    # CONV LAYER 3 OUT : HIDDEN x 40
    #####################################
    conv3t = conv1d_trans_block(hidden_channels[1], hidden_channels[2], k=4, s=2, p=1)
    #####################################
    # CONV LAYER 4 OUT : HIDDEN x 80
    #####################################
    conv4t = conv1d_trans_block(hidden_channels[2], hidden_channels[3], k=4, s=2, p=1)
    #####################################
    # CONV LAYER 5 OUT : HIDDEN x 160
    #####################################
    conv5t = conv1d_trans_block(hidden_channels[3], hidden_channels[4], k=4, s=2, p=1)
    #####################################
    # CONV LAYER 6 OUT : IN_CHANNELS x 320
    #####################################
    conv6t = conv1d_trans_block(hidden_channels[4], output_channels, k=4, s=2, p=1)

    logger.info("Using {} input normalization in transpose net.".format(input_norm))

    norm1 = apply_input_normalization(hidden_channels[0], input_norm, track_running_stats=track_running_stats)
    norm2 = apply_input_normalization(hidden_channels[1], input_norm, track_running_stats=track_running_stats)
    norm3 = apply_input_normalization(hidden_channels[2], input_norm, track_running_stats=track_running_stats)
    norm4 = apply_input_normalization(hidden_channels[3], input_norm, track_running_stats=track_running_stats)
    norm5 = apply_input_normalization(hidden_channels[4], input_norm, track_running_stats=track_running_stats)

    if spectral_norm:
        logger.info("Using weight normalization spectral norm in transpose net.")
        conv2t = nn.utils.spectral_norm(conv2t)
        conv3t = nn.utils.spectral_norm(conv3t)
        conv4t = nn.utils.spectral_norm(conv4t)
        conv5t = nn.utils.spectral_norm(conv5t)

    module_list = [
        conv1t,
        norm1,
        nn.ReLU(inplace=True),
        conv2t,
        norm2,
        nn.ReLU(inplace=True),
        conv3t,
        norm3,
        nn.ReLU(inplace=True),
        conv4t,
        norm4,
        nn.ReLU(inplace=True),
        conv5t,
        norm5,
        nn.ReLU(inplace=True),
        conv6t,
    ]

    module_list = [mod for mod in module_list if mod is not None]
    net = nn.Sequential(*module_list)  # type: ignore

    return net


def conv_norm_relu(
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    bias: bool = False,
) -> nn.Sequential:
    """Chain convolutional layers with ReLU activations and batch norm."""
    return nn.Sequential(
        nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm1d(output_channels),
        nn.LeakyReLU(),
    )
