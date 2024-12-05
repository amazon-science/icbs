# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from collections import OrderedDict

import numpy as np
import pytest
import torch
from bitarray import bitarray
from icbs.util import (
    array_slice,
    bitarray_from_list,
    calc_num_weights,
    extend_bitarray,
    get_log_w_names_step,
    get_named_weights,
    getattr_nested,
    set_module_by_name,
    sorted_indices_2d_to_1d,
)
from torch import nn


@pytest.fixture
def model():
    """Create a model."""
    conv = nn.Conv2d(1, 20, 5)
    relu = nn.ReLU()
    model = nn.Sequential(
        OrderedDict(
            [
                ("conv", conv),
                ("relu", relu),
            ]
        )
    )
    return model


@pytest.mark.parametrize("include_layers", [None, (nn.Conv2d,)])
def test_get_named_weights_typical(model, include_layers):
    """Get the named weights from a model."""
    named_weights = get_named_weights(model, include_layers)
    assert len(named_weights) == 1
    assert torch.equal(named_weights["conv.weight"], model.conv.weight)


def test_get_named_weights_include_layers(model):
    """Get the named weights from a model."""
    named_weights = get_named_weights(model, include_layers=(nn.Linear,))
    assert named_weights == {}


def test_calc_num_weights_typical():
    module = nn.Linear(20, 30)
    assert calc_num_weights(module) == 600


def test_getattr_nested_typical():
    """Get a module from a model, given the name, like layer1.0.relu."""
    conv = nn.Conv2d(1, 20, 5)
    relu = nn.ReLU()

    model = nn.Sequential(
        OrderedDict(
            [
                ("conv", conv),
                ("relu", relu),
            ]
        )
    )
    assert getattr_nested(model, "conv") == conv
    assert getattr_nested(model, "relu") == relu


def test_getattr_nested_nested():
    conv = nn.Conv2d(1, 20, 5)
    relu = nn.ReLU()
    sequential = nn.Sequential(conv, relu)

    model = nn.Sequential(
        OrderedDict(
            [
                ("sequential", sequential),
            ]
        )
    )

    assert getattr_nested(model, "sequential.0") == conv
    assert getattr_nested(model, "sequential.1") == relu


def test_set_module_by_name():
    conv = nn.Conv2d(1, 20, 5)
    relu = nn.ReLU()
    sequential = nn.Sequential(conv, relu)

    model = nn.Sequential(
        OrderedDict(
            [
                ("sequential", sequential),
            ]
        )
    )

    new_conv = nn.Conv2d(1, 20, 5)
    set_module_by_name(model, "sequential.0", new_conv)

    assert model.sequential[0] == new_conv


@pytest.mark.parametrize(
    "num_steps,w_sizes",
    [(22, (2222, 1444, 3)), (3, (100, 11111, 123)), (103, (100, 200, 300))],
)
def test_get_log_w_names_step(num_steps, w_sizes):
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Linear(w_sizes[0], 1)
            self.layer1 = nn.Linear(w_sizes[1], 1)
            self.layer2 = nn.Linear(w_sizes[2], 1)

    model = MockModel()
    min_num_steps = 2

    named_weights = {
        "layer0": model.layer0.weight,
        "layer1": model.layer1.weight,
        "layer2": model.layer2.weight,
    }
    w_names_step = get_log_w_names_step(named_weights, num_steps, min_num_steps)

    assert len(w_names_step) == num_steps
    values, counts = np.unique(w_names_step, return_counts=True)
    assert set(values) == set(named_weights.keys())
    if min_num_steps * len(named_weights) < num_steps:
        assert np.all(counts >= min_num_steps)
    assert sum(counts) == num_steps


@pytest.mark.parametrize("num_steps", [103, 27])
def test_get_log_w_names_step_sign(num_steps):
    # Test examples where sign=1 and sign=-1 are present
    layer_sizes = [100, 8, 4, 4, 2, 2]
    named_weights = {
        i: nn.Linear(layer_size, 1).weight for i, layer_size in enumerate(layer_sizes)
    }
    w_names_step = get_log_w_names_step(named_weights, num_steps=num_steps)
    assert len(w_names_step) == num_steps
    for i in range(len(layer_sizes)):
        assert i in w_names_step


def test_bitarray_from_list():
    l = [0, 3, 4]
    assert bitarray_from_list(l, n=7) == bitarray([1, 0, 0, 1, 1, 0, 0])


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_extend_bitarray_trivial(n):
    b = bitarray([1, 0, 1])
    extend_bitarray(b, n)
    assert b == bitarray([1, 0, 1])


def test_extend_bitarray_typical():
    b = bitarray([1, 0, 1])
    extend_bitarray(b, 5)
    assert b == bitarray([1, 0, 1, 0, 0])


def test_array_slice():
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # Slice first row
    assert np.all(array_slice(a, 0, 1, 3) == np.array([5, 6, 7, 8]))
    # Slice odd columns (tests step and using None at start and end)
    assert np.all(array_slice(a, 1, None, None, 2) == np.array([[1, 3], [5, 7]]))


@pytest.mark.parametrize("axis", [0, 1])
def test_sorted_indices_2d_to_1d(axis):
    a = np.array([[3, 1, 2], [9, 5, 6], [7, 8, 4]])
    indices_2d = np.argsort(a, axis)
    indices_1d = sorted_indices_2d_to_1d(indices_2d, axis)
    assert np.all(a.reshape(-1)[indices_1d] == np.sort(a, axis))
