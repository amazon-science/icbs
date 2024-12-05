# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import gc
from functools import reduce
from itertools import repeat

import numpy as np
import torch
from bitarray import bitarray


def calc_num_weights(model):
    """Returns the total number of weights in the model."""
    return sum(module.weight.numel() for module in get_modules_with_weights(model))


def get_modules_with_weights(model):
    """Returns a list of modules with weights in model."""
    modules_with_weights = []
    for module in model.modules():
        try:
            getattr(module, "weight")
        except AttributeError:
            pass
        else:
            modules_with_weights.append(module)
    return modules_with_weights


def get_named_weights(model, include_layers=None):
    """Returns a dict of weights in model, filtered by include_layers if provided."""
    # Create list of module names to keep
    keep_modules = []
    for name, module in model.named_modules():
        if include_layers is not None:
            if isinstance(module, include_layers):
                keep_modules.append(name)
        else:
            keep_modules.append(name)

    # Retrieve associated weight parameters
    named_weights = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".weight") & (name.replace(".weight", "") in keep_modules):
            named_weights[name] = parameter

    return named_weights


def calc_density(model):
    """Calculates the density of the model (weights only)."""
    modules_with_weights = get_modules_with_weights(model)

    num_nonzero = 0
    num_weights = 0
    for module in modules_with_weights:
        try:
            num_nonzero += module.weight.nonzero().size(0)
            num_weights += module.weight.numel()
        except AttributeError:
            continue
    return num_nonzero / num_weights


def calc_density_from_layer_names(model, layer_names):
    """Calculates the density of the given list of weights."""
    num_nonzero = 0
    num_weights = 0
    for layer_name in layer_names:
        layer = getattr_nested(model, layer_name)
        try:
            num_nonzero += layer.weight.nonzero().size(0)
            num_weights += layer.weight.numel()
        except AttributeError:
            continue
    return num_nonzero / num_weights


def calc_loss(model, loss_function, X, y, params=None):
    """Calculates the loss for model with respect to data X and labels y."""
    if params is None:
        with torch.amp.autocast(**get_autocast_params()):
            y_pred = model(X)
    else:
        with torch.amp.autocast(**get_autocast_params()):
            y_pred = torch.func.functional_call(model, params, X)
    loss = loss_function(y_pred, y)
    return loss


def is_flat(t):
    """Checks if a tensor is flat - has only one dimension with a size that is not 1."""
    if t.dim() > 2:
        raise ValueError(
            "is_flat() is implemented only for tensors with dimension 1 or 2"
        )
    return t.dim() == 1 or (t.dim() == 2 and t.size()[-1] == 1)


def get_square_size(T):
    """Returns the size of a square tensor T."""
    assert T.dim() == 2
    num_rows, num_cols = T.size()
    assert num_rows == num_cols
    return num_rows


def getattr_nested(model, name):
    """Gets a module/parameter from a model, given the name, like layer1.0.relu.

    Source:
    https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8
    """
    names = name.split(sep=".")
    return reduce(getattr, names, model)


def set_module_by_name(model, name, module):
    """Set a module in a model by its name, like layer1.0.relu."""
    names = name.split(sep=".")
    parent_names = names[:-1]
    parent_module = reduce(getattr, parent_names, model)

    last_name = names[-1]
    setattr(parent_module, last_name, module)


def get_log_w_names_step(named_weights, num_steps, min_num_steps=1):
    """Returns a weight name for each step, proportionally to the log of its size.

    If min_num_steps is provided, then an attempt will be made to assign at least this
    number of steps to each layer. However, if num_layers * min_num_steps > num_steps,
    then this is not possible. In this case, some layers might have a number of steps
    lower than min_num_steps, in order to make sure that the total is equal to num_steps.
    """
    if not named_weights:
        raise ValueError("named_weights cannot be empty.")
    if len(named_weights) * min_num_steps > num_steps:
        print(
            "WARNING: cannot adhere to min_num_steps - some layers will have a lower number of steps"
        )

    # Get the names and sizes of the weights
    w_names, w_sizes = [], []
    for w_name, weight in named_weights.items():
        w_names.append(w_name)
        w_sizes.append(weight.numel())
    w_sizes = np.array(w_sizes)

    # First round - number of steps for each layer
    w_fracs = np.log10(w_sizes)
    w_fracs /= sum(w_fracs)
    w_steps = np.round(w_fracs * num_steps).astype(int)

    # Impose the minimum number of steps for any layers that went below it
    w_steps[w_steps < min_num_steps] = min_num_steps

    # Recalculate the number of steps for each layer
    w_steps = np.round(num_steps * w_steps / sum(w_steps)).astype(int)

    # It can happen that the sum of the steps is not equal to the number of steps, due
    # to rounding and then summing the resulting integers. We deal with this by taking
    # the extra/missing steps and take them off the largest/smallest layers.
    extra_steps = sum(w_steps) - num_steps
    while extra_steps != 0:
        sign = np.sign(extra_steps)
        if sign == 1:
            target = max(w_steps)
        else:
            target = min(w_steps)
        candidates = np.where(w_steps == target)[0]
        candidate = np.random.choice(candidates, 1)
        w_steps[candidate] -= sign
        extra_steps -= sign

    # Create a list of weight names for each step
    w_names_step = []
    for w_name, w_step in zip(w_names, w_steps):
        w_names_step += [w_name] * w_step

    return w_names_step


def save_model_torchscript(model, model_filename):
    """Save model to model_filename, in the torchscript format."""
    model_torchscript = torch.jit.script(model)
    model_torchscript.save(model_filename)


def load_model_torchscript(model_filename):
    """Load model from model_filename, in the torchscript format."""
    model_torchscript = torch.jit.load(model_filename)
    return model_torchscript


def bitarray_from_list(l, n):
    """Returns a bitarray of length n from a list of integers.

    The corresponding indices are set to 1. For example, if l = [1, 3, 5], and n=7 then
    the bitarray will be [0, 1, 0, 1, 0, 1, 0].
    """
    a = bitarray(n)
    a[l] = 1
    return a


def extend_bitarray(b, n):
    """Extends a bitarray b to length n by adding zeros."""
    if n > len(b):
        b.extend(repeat(0, n - len(b)))


def get_autocast_params():
    """Returns dict with parameters for torch.omp.autocast."""
    if torch.cuda.is_available():
        return {"device_type": "cuda", "dtype": torch.float16}
    # Disable autocasting for CPU - leave as is
    return {"device_type": "cpu", "enabled": False}


def clear_memory_caches():
    """Clears the memory caches, CPU and GPU."""
    gc.collect()
    torch.cuda.empty_cache()


def print_total_gpu_memory_allocated():
    """Prints the total GPU memory allocated across all GPUs."""
    total_memory_allocated = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            memory_allocated = torch.cuda.memory_allocated(gpu_id)
            total_memory_allocated += memory_allocated

        print(
            f"Total memory allocated across all GPUs: {total_memory_allocated / 10**6:.2f} MB"
        )
    else:
        print("CUDA is not available on this machine.")


def array_slice(a, axis, start, end, step=1):
    """Slices array a on axis from start to end with step.

    Based on: https://stackoverflow.com/a/64436208/1265409
    """
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(start, end, step)
    return a[tuple(sl)]


def sorted_indices_2d_to_1d(ordering_2d, axis):
    """Converts 2D ordering (as returned from argsort) to 1D ordering."""
    # Generate 2D index arrays
    num_rows, num_cols = ordering_2d.shape
    if axis == 0:
        row_indices = ordering_2d
        col_indices = np.tile(np.arange(num_cols), (num_rows, 1))
    else:
        row_indices = np.tile(np.arange(num_rows), (num_cols, 1)).T
        col_indices = ordering_2d

    # Convert 2D indices to 1D indices for the flattened array
    ordering_1d = row_indices * num_cols + col_indices

    return ordering_1d
