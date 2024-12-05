# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from collections import OrderedDict
from functools import partial

import pytest
import torch
import torch.nn as nn
from datasets import load_dataset
from icbs.derivatives import (
    calc_grad_autograd,
    calc_grad_autograd_llm,
    calc_grad_functional,
    calc_grad_sample_autograd,
    calc_grad_sample_autograd_llm,
    calc_grad_sample_functional,
    calc_hessian,
    calc_hessian_functional,
    calc_hessian_piecewise,
    calc_hessian_rows,
)
from icbs.util import getattr_nested
from icbs.util_llm import load_llm, prep_loader_batch, prepare_llm

device = "cpu"


def allclose(input, other):
    """Returns true if all elements of the Tensors are close."""
    if device == "cpu":
        return torch.allclose(input, other)
    else:
        return torch.allclose(input, other, atol=0.001, rtol=0)


@pytest.fixture
def model():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                OrderedDict(
                    [
                        ("linear", nn.Linear(5 * 5, 3)),
                        ("relu", nn.ReLU()),
                    ]
                )
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    return TestModel().to(device)


@pytest.fixture
def X():
    return torch.rand(4, 1, 5, 5).to(device)  # 4 samples of 5 * 5 with 1 channel


@pytest.fixture
def y():
    return torch.randint(low=0, high=3, size=(4,)).to(device)  # 4 samples of 3 classes


@pytest.fixture
def X_y_model_device_llm():
    model = load_llm("stas/tiny-random-llama-2", device)
    model, tokenizer, seqlen, input_device = prepare_llm(model, device)

    train_dataset = load_dataset(
        "stas/c4-en-10k",
        split="train",
    )
    batch = train_dataset[:10]
    X, y = prep_loader_batch(batch, tokenizer, seqlen)

    return X, y, model, input_device


@pytest.fixture
def loss_function():
    return nn.CrossEntropyLoss()


@pytest.fixture
def weight_name():
    return "linear_relu_stack.linear.weight"


@pytest.fixture
def weight_name_llm():
    return "model.layers.1.mlp.up_proj.weight"


def test_calc_grad_consistency_autograd_vs_functional(
    model, loss_function, X, y, weight_name
):
    grad = calc_grad_autograd(model, loss_function, X, y, weight_name, device)
    grad_functional = calc_grad_functional(model, loss_function, X, y, weight_name)
    assert allclose(grad, grad_functional)


@pytest.mark.parametrize("max_batch_size", [1, 2, 3, 5])
def test_calc_grad_accumulation_consistency(
    model, loss_function, X, y, weight_name, max_batch_size
):
    grad = calc_grad_autograd(
        model, loss_function, X, y, weight_name, device, max_batch_size=None
    )
    grad_accum = calc_grad_autograd(
        model,
        loss_function,
        X,
        y,
        weight_name,
        device,
        max_batch_size=max_batch_size,
    )
    assert allclose(grad, grad_accum)


@pytest.mark.parametrize("max_batch_size", [2, 10])
def test_calc_grad_sample_consistency_autograd_vs_functional(
    model, loss_function, X, y, weight_name, max_batch_size
):
    grad_sample_autograd = calc_grad_sample_autograd(
        model, loss_function, X, y, weight_name, device
    )
    grad_sample_functional = calc_grad_sample_functional(
        model,
        loss_function,
        X,
        y,
        weight_name,
        device,
        max_batch_size=max_batch_size,
    )
    assert allclose(grad_sample_functional, grad_sample_autograd)


def test_calc_hessian_consistency_vs_functional(
    model, loss_function, X, y, weight_name
):
    weight_indices = [0, 12, 34, 42, 74, 23, 4]
    hess = calc_hessian(model, loss_function, X, y, weight_name, weight_indices, device)
    hess_functional = calc_hessian_functional(
        model, loss_function, X, y, weight_name, weight_indices
    )
    assert allclose(hess, hess_functional)


def test_calc_hessian_rows(model, loss_function, X, y, weight_name):
    y_pred = model(X)
    loss = loss_function(input=y_pred, target=y)
    layer_weights = getattr_nested(model, weight_name)
    weight_indices_chunk = [0, 1]
    weight_indices = range(6)

    # Calculate gradient of loss w.r.t. layer weights
    [grad] = torch.autograd.grad(
        outputs=loss, inputs=layer_weights, create_graph=True, allow_unused=False
    )
    grad = grad.view(-1)  # flatten

    hess_rows = calc_hessian_rows(layer_weights, weight_indices_chunk, grad)
    hess_rows = hess_rows[:, weight_indices]
    hess = calc_hessian(model, loss_function, X, y, weight_name, weight_indices, device)

    assert allclose(hess_rows.to(device), hess[weight_indices_chunk, :])


@pytest.mark.parametrize("chunk_size", [2, 7, 10])
def test_calc_hessian_consistency_vs_piecewise(
    model, loss_function, X, y, weight_name, chunk_size
):
    weight_indices = [12, 0, 34, 42, 74, 23, 4]
    hess = calc_hessian(model, loss_function, X, y, weight_name, weight_indices, device)
    hess_piecewise = calc_hessian_piecewise(
        model,
        loss_function,
        X,
        y,
        weight_name,
        weight_indices,
        device,
        chunk_size=chunk_size,
    )

    assert allclose(hess, hess_piecewise)


def test_calc_grad_autograd_llm_self_consistency(
    X_y_model_device_llm, loss_function, weight_name_llm
):
    X, y, model, input_device = X_y_model_device_llm
    calc_grad = partial(
        calc_grad_autograd_llm,
        model,
        loss_function,
        X,
        y,
        weight_name_llm,
        device=input_device,
    )
    grad = calc_grad(max_batch_size=None)
    for max_batch_size in [5]:
        grad2 = calc_grad(max_batch_size=max_batch_size)
        assert (grad - grad2).abs().max() < 1e-4


def test_calc_grad_sample_autograd_llm_self_consistency(
    X_y_model_device_llm, loss_function, weight_name_llm
):
    X, y, model, input_device = X_y_model_device_llm
    calc_grad_sample = partial(
        calc_grad_sample_autograd_llm,
        model,
        loss_function,
        X,
        y,
        weight_name_llm,
        input_device,
    )
    grad_sample = calc_grad_sample(max_batch_size=None)

    for max_batch_size in [1, 5]:
        grad_sample2 = calc_grad_sample(max_batch_size=max_batch_size)
        assert allclose(grad_sample, grad_sample2)
