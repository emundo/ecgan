"""Miscellaneous utilities (especially saving, loading, logging)."""
import json
import os
import pickle
import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
import yaml

from ecgan.utils.artifacts import Artifact, ValueArtifact
from ecgan.utils.custom_types import save_epoch_metrics

logger = getLogger(__name__)


def load_model(model_reference: str, device) -> Dict:
    """Load a trained module from disk (file path) or wand reference to device."""
    artifact_dir = 'artifacts'
    downloaded_artifacts = os.listdir(artifact_dir)
    model_reference_root = ""
    if '.pt' in model_reference and os.path.exists(model_reference):
        model_path = model_reference

    elif model_reference in downloaded_artifacts:
        model_reference_root = os.path.join(artifact_dir, model_reference)
        model_path = [file for file in os.listdir(model_reference_root) if file.endswith('.pt')][0]

    else:
        if is_wandb_model_link(model_reference) and ':' not in model_reference:
            model_reference += ':latest'
        api = wandb.Api()
        logger.info("Loading model {}.".format(model_reference))
        artifact = api.artifact(model_reference, type='model')
        model_reference_root = artifact.download()
        model_path = [file for file in os.listdir(model_reference_root) if file.endswith('.pt')][0]
    model: dict = torch.load(os.path.join(model_reference_root, model_path), map_location=device)
    return model


def save_pickle(data: object, filepath: str, filename: str) -> None:
    """
    Save a generic object to a binary file.

    Args:
        data: Object to be saved.
        filepath: Saving destination.
        filename: Name of file `data` is saved to.
    """
    os.makedirs(filepath, exist_ok=True)
    full_path = os.path.join(filepath, filename)
    with open(full_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_tensor(data: torch.Tensor, filepath: str, filename: str) -> None:
    """
    Save a torch tensor to a binary file.

    Args:
        data: Object to be saved.
        filepath: Saving destination.
        filename:  Name of file `data` is saved to.
    """
    os.makedirs(filepath, exist_ok=True)
    full_path = os.path.join(filepath, filename)
    torch.save(data, full_path)


def load_pickle(filepath: str) -> Any:
    """
    Load a binary file to a python object.

    Args:
        filepath: File to be loaded
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_pickle_numpy(filepath: str) -> np.ndarray:
    """
    Load a binary file to a numpy array.

    Args:
        filepath: File to be loaded
    """
    return np.load(filepath, allow_pickle=True)  # type: ignore


def load_yml(filepath: str) -> Dict:
    """Load a yml file to memory as dict."""
    with open(filepath, 'r', encoding='utf-8') as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))


def is_wandb_model_link(filename: str) -> bool:
    """Differentiates between local data and W&B data."""
    return not os.path.exists(filename)


def load_wandb_config(run_uri: str) -> Dict:
    """Load a config from W&B and convert it to a local one."""
    api = wandb.Api()
    run = api.run(run_uri)
    return convert_wandb_config(run.json_config)


def convert_wandb_config(wandb_config: str) -> Dict:
    """
    Convert a wandb config into a config dict to reconstruct a run.

    The wandb includes some additional metadata, especially a `value` field for each key which has to be removed to fit
    the original form.

    Args:
        wandb_config: The downloaded wandb config.

    Returns:
        The cleaned config parsed into a dict.
    """
    wandb_json = json.loads(wandb_config)
    new_dict = {}
    for key, val in wandb_json.items():
        if isinstance(val, dict):
            new_dict.update({key: val.pop('value')})
        else:
            new_dict.update({key: val})
    return new_dict


def update_dicts(original_dict: Dict, update_dict: Dict, additional_dict: Dict) -> None:
    """Update dict and the update dict with an additional dict."""
    if 'update' in additional_dict:
        merge_dicts(update_dict, additional_dict['update'], duplicates=False)
        additional_dict.pop('update')
    original_dict.update(additional_dict)


def merge_dicts(dict_0: Dict, dict_1: Dict, duplicates=True) -> None:
    """
    Recursively merges two dicts into the first one.

    Params:
        dict_0: The dictionary that is merged into.
        dict_1: The dictionary that is merged from.
        duplicates: If `False` the merging will exit with `RunTimeError` when merging duplicate keys.
    """
    for (key, value) in dict_1.items():
        if isinstance(dict_0.get(key, None), dict) and isinstance(value, dict):
            merge_dicts(dict_0[key], value)

        else:
            if key in dict_0 and not duplicates:
                raise RuntimeError(
                    "Overwriting existing key when duplicated where prohibited: key: {} value: {} new "
                    "value: {}".format(key, dict_0[key], dict_1[key])
                )
            dict_0[key] = value


def calc_conv_output(in_size: int, kernel_size: Union[int, List[int]], stride: Union[int, List[int]]) -> int:
    """
    Calculate the output shape for a network with convolutions.

    The calculation is done for every dimension individually. That means if your
    training input is a 5D Tensor with shape (B x C x H x W x D) you need to call this
    function 3 times separately for H, W, D in case the dimensions differ. Hence input
    size is an int, which represents one dimension.

    Args:
        in_size: Input size in the CNN.
        kernel_size: All kernel sizes in the CNN layers.
        stride: All strides in the CNN layers.

    Returns:
        Output size in the given dimension.
    """
    if isinstance(kernel_size, int) and isinstance(stride, int):
        return ((in_size - kernel_size) // stride) + 1

    if isinstance(kernel_size, list) and isinstance(stride, list):
        for kernel, strd in zip(kernel_size, stride):
            in_size = ((in_size - kernel) // strd) + 1
        return in_size

    raise ValueError(
        'Parameters kernel_size and stride are not of same type. '
        'They should both be of type List or int.'
        'Current types: kernel size: {0} and stride: {1}'.format(type(kernel_size), type(stride))
    )


def calc_pool_output(in_size: int, kernel_size: int, stride: int) -> int:
    """Calculate the output size of a pooling layer."""
    return ((in_size - kernel_size) // stride) + 1


def select_device(gpu_flag: bool) -> torch.device:
    """
    Select device the model shall be trained on.

    Either GPU if GPU is set in config and GPU is available or
    CPU if GPU is selected but not available or CPU is selected.

    Args:
        gpu_flag: Flag indicating if GPU shall be used.

    Returns:
        Device for torch (:code:`gpu` or :code:`cpu`).
    """
    if gpu_flag and torch.cuda.is_available():
        return torch.device('cuda')
    if gpu_flag and not torch.cuda.is_available():
        logger.warning(
            '\n##############################################################\n'
            '### WARNING: GPU FLAG IS SET TO TRUE BUT CUDA IS NOT AVAILABLE\n'
            '### Please check your CUDA install. Defaulting to CPU now.'
            '\n##############################################################\n'
        )
        count = 3
        for _ in range(count):
            print('Resuming process in {0:3} second(s)'.format(count), end='\r')
            time.sleep(1)
            count -= 1
    return torch.device('cpu')


def save_epoch(
    highest_vals: Dict,
    epoch: int,
    checkpoint_interval: int,
    metrics: List[str],
    final_epoch: int,
) -> bool:
    """
    Check if epoch should be saved based on the performance on the validation data.

    It will be saved if:
    1. auroc/f1 are at its maximum,
    2. the model would not be saved due to the checkpoint interval anyways, and
    3. the auroc/f1 are above the threshold of 0.7 to avoid excessive saving during first epochs

    Args:
        highest_vals: Dictionary of the maximum epoch values.
        epoch: Current epoch.
        checkpoint_interval: Regular checkpoint interval.
        metrics: Additional metrics.
        final_epoch: Last epoch, saved by default.

    Returns:
        Flag, indicating if epoch should be saved.
    """
    if epoch % checkpoint_interval == 0:
        logger.info('Saving epoch: regular checkpoint interval.')
        return True
    if epoch == final_epoch:
        logger.info('Saving epoch: Current epoch is final epoch of fold.')
        return True

    for key, (high_epoch, high_val) in highest_vals.items():
        if high_val > 0.7 and high_epoch == epoch:
            if key in metrics:
                logger.info(
                    'Saving epoch: new highest metric score: {0} is {1} in epoch {2}.'.format(key, high_val, epoch)
                )
                return True
    return False


def update_highest_metrics(
    new_vali_metrics: Dict,
    artifacts: List[Artifact],
    highest_metrics: Dict[str, Tuple],
    epoch: int,
    minimum_metric_improvement: float = 0.005,
) -> Dict:
    """
    Compare validation metrics of current epoch with existing max values.

    A value is only saved as a new highest value if the previous highest value is exceeded
    by at least `minimum_metric_improvement`.
    This means that the real highest metric might be higher than the highest metric saved here
    but it will only be a slight improvement. This avoids too many saved checkpoints which
    would happen if any relevant metric is improved marginally (see :func:`ecgan.utils.miscellaneous.save_epoch`.

    Args:
        new_vali_metrics: Dict containing metric keys and float values for current epoch.
        artifacts: List of artifacts that is checked for valid metrics.
        highest_metrics: Dict containing highest metrics. Metric keys and values are Tuples of (epoch, max value).
        epoch: Current epoch.
        minimum_metric_improvement: Minimum required relative improvement of the metric.

    Returns:
        Updated dict with highest metric values.
    """
    for artifact in artifacts:
        if isinstance(artifact, ValueArtifact):
            if isinstance(artifact.value, float):  # nested dicts are not compared at the moment.
                new_vali_metrics.update({artifact.name: artifact.value})

    for key, value in new_vali_metrics.items():
        if isinstance(value, dict):
            # nested dicts are not tracked
            continue
        if highest_metrics.get(key) is None:
            highest_metrics[key] = (epoch, value)
        highest_val = highest_metrics.get(key)[1]  # type: ignore
        if value is not None and (value > highest_val * (1 + minimum_metric_improvement)):
            highest_metrics[key] = (epoch, value)
    return highest_metrics


def generate_seed() -> int:
    """Generate a random seed which can later be used as manual seed."""
    return int(torch.randint(0, 100000, (1,)).item())


def get_num_workers() -> int:
    """Return the number of available CPU cores (minus one)."""
    available_cores = os.cpu_count()
    num_workers: int = (
        0 if available_cores is None else available_cores if available_cores <= 1 else available_cores - 1
    )
    return num_workers


def list_from_tuple_list(metric_tuple_list: List[Dict], position: int = 1) -> List:
    """Retrieve all values at `position` of a tuple."""
    metric_list: List = []
    for fold in metric_tuple_list:
        fold_dict = {}
        for key in fold.keys():
            fold_dict.update({key: fold[key][position]})
        metric_list.append(fold_dict)

    return metric_list


def nested_list_from_dict_list(metric_list: List[Dict]):
    """Create a nested List from a given Dict."""
    nested_list = []
    metric_name_list: List[str] = []
    for metric in metric_list[0].keys():
        if any(save_metric in metric for save_metric in save_epoch_metrics):
            nested_list.append([fold[metric] for fold in metric_list])
            metric_name_list.append(metric)
    return nested_list, metric_name_list


def scale_to_unit_circle(data: torch.Tensor):
    """Rescales data to [-1,1] range."""
    min_ = torch.min(data)
    scaled_data = (data - min_) / (torch.max(data) - min_) * 2 - 1

    return scaled_data


def retrieve_model_specification(run_path: str) -> Tuple[str, str, str]:
    """
    Retrieve model uri, fold and version from existing model path.

    Args:
        run_path: Path of previous run.

    Returns:
        URI, fold and version of run.
    """
    if is_wandb_model_link(run_path):
        run_information, run_version = run_path.split(":")
        run_uri, fold = run_information.split('_fold')

    else:
        run_uri, fold = run_path.split('_fold')
        run_version = 'latest'

    return run_uri, fold, run_version


def scale_weights(
    real_label: torch.Tensor,
    loss: torch.Tensor,
    percentage_normal: Optional[float] = None,
) -> torch.Tensor:
    """
    Scale the loss of some input based on the training data imbalance.

    Assumes a binary classification. The imbalance weighting is calculated per batch.
    Manual reduction is possible because "reduction='none'" was passed during loss init.

    Args:
        real_label: Tensor of binary labels.
        loss: Network loss.
        percentage_normal: Share of labels in the whole dataset which are normal.

    Returns:
        The scaled average loss.
    """
    # Possible improvement: scale per class?
    if percentage_normal is None:
        # Avoid zero division:
        count_anomalies = real_label[real_label != 0].numel() if real_label[real_label != 0].numel() != 0 else 1
        imbalance = real_label[real_label == 0].numel() / count_anomalies
    else:
        imbalance = percentage_normal / (1.0 - percentage_normal)

    loss[real_label == 1] *= imbalance
    return torch.mean(loss)


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a Tensor of type Union[np.ndarray, torch.Tensor] to np.ndarray."""
    return tensor if isinstance(tensor, np.ndarray) else tensor.detach().cpu().numpy()  # type: ignore


def to_torch(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert a Tensor of type Union[np.ndarray, torch.Tensor] to torch.Tensor."""
    return torch.from_numpy(tensor) if isinstance(tensor, np.ndarray) else tensor
