# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from math import ceil

import numpy as np
import torch

from .util import calc_loss, clear_memory_caches, get_autocast_params, getattr_nested


def calc_hessian_accumulated(
    model, loss_function, X, y, weight_name, weight_indices, device
):
    """Borrows from gradient accumulation, but extends to second derivative.

    NOTE: May fail on some models, as some existing gradient functions (e.g.,
    aten::_scaled_dot_product_efficient_attention_backward) do not have known gradient
    functions.
    """
    # Retrieve model weights
    layer_weights = getattr_nested(model, weight_name)

    # Initialize the reduced Hessian
    num_weights = len(weight_indices)
    hess = torch.zeros(num_weights, num_weights, device=device)

    # Accumulate gradient over batches to reduce memory usage
    for i in range(X.shape[0]):
        X_batch = X[i : i + 1].to(device, non_blocking=True)
        y_batch = y[i : i + 1].to(device, non_blocking=True)

        with torch.amp.autocast(**get_autocast_params()):
            y_pred = model(X_batch)
            loss = loss_function(input=y_pred, target=y_batch)
        loss /= X.shape[0]

        # Calculate gradient of loss w.r.t. layer weights
        [grad] = torch.autograd.grad(
            outputs=loss, inputs=layer_weights, create_graph=True, allow_unused=False
        )

        # Retrieve + ravel 2D gradient matrix into 1D gradient vector
        grad = grad.view(-1)

        # Steps through the weight_indices and calculates the gradient of the respective
        # gradient elements, yielding the second derivatives.
        for i, weight_index in enumerate(weight_indices):
            [row] = torch.autograd.grad(
                outputs=grad[weight_index],
                inputs=layer_weights,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            row = row.view(-1)
            row = row[weight_indices]

            hess[i, :].add_(row.type_as(hess))  # Add this row to the Hessian

        model.zero_grad(set_to_none=True)

    return hess


def calc_hessian(
    model, loss_function, X, y, weight_name, weight_indices, device, max_batch_size=None
):
    """Calculates the Hessian of the loss function for weight_indices in weight_name.

    Calculates the gradient, and then calculates the gradient of the gradient, but only
    for weights in weight_indices. Uses torch.autograd. Adapted from:
    https://github.com/mariogeiger/hessian/blob/master/hessian/hessian.py
    """
    # Filter for layer of interest
    layer_weights = getattr_nested(model, weight_name)
    if not layer_weights.requires_grad:
        raise AttributeError(
            f"Weights {weight_name} is not attached to compute graph. Cannot calculate Hessian."
        )

    if max_batch_size is not None:
        hess = calc_hessian_accumulated(
            model, loss_function, X, y, layer_weights, weight_indices, device
        )
    else:
        # In case data is residing on CPU (or device other than where model is)
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(**get_autocast_params()):
            y_pred = model(X)
            loss = loss_function(input=y_pred, target=y)

        # Calculate gradient of loss w.r.t. layer weights
        [grad] = torch.autograd.grad(
            outputs=loss, inputs=layer_weights, create_graph=True, allow_unused=False
        )

        # Initialize the reduced Hessian
        num_weights = len(weight_indices)
        hess = torch.zeros(num_weights, num_weights, device=device)

        # Ravel 2D gradient matrix into 1D gradient vector
        grad = grad.view(-1)

        # Steps through the weight_indices and calculates the gradient of the respective
        # gradient elements, yielding the second derivatives.
        for i, weight_index in enumerate(weight_indices):
            [row] = torch.autograd.grad(
                outputs=grad[weight_index],
                inputs=layer_weights,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            row = row.view(-1)
            row = row[weight_indices]

            hess[i, :].add_(row.type_as(hess))  # Add this row to the Hessian

        model.zero_grad(set_to_none=True)

    return hess


def calc_hessian_functional(
    model, loss_function, X, y, weight_name, weight_indices=None
):
    """Calculates the Hessian of the loss function using torch.func.hessian().

    Note: this function may fail on models that use BatchNorm with tracking on. For such
    models it's better to use calc_hessian() which uses torch.autograd.grad().
    """
    # Filter for layer of interest
    layer_weights = getattr_nested(model, weight_name)

    # Calculate the Hessian. argnums tells hessian() which argument of calc_loss() to
    # calculate the Hessian with respect to, counting from 0.
    hess = torch.func.hessian(calc_loss, argnums=4)(
        model, loss_function, X, y, {weight_name: layer_weights}
    )
    hess = hess[weight_name][weight_name]

    # At this point for a weights matrix with num_rows x num_cols, the Hessian is a
    # tensor of dimensions [num_rows, num_cols, num_rows, num_cols]. We flatten it so
    # that it is a matrix of dimensions [num_rows*num_cols, num_rows*num_cols].
    num_weights = layer_weights.numel()
    hess = hess.view(num_weights, num_weights)

    if weight_indices is not None:
        hess = hess[:, weight_indices][weight_indices, :]

    return hess


def calc_hessian_piecewise(
    model,
    loss_function,
    X,
    y,
    weight_name,
    weight_indices,
    device,
    max_batch_size=None,
    chunk_size=200,
):
    """Calculates the Hessian of the loss function in chunks to avoid memory issues."""
    if len(weight_indices) <= chunk_size:
        hess = calc_hessian(
            model,
            loss_function,
            X,
            y,
            weight_name,
            weight_indices,
            device,
            max_batch_size,
        )

    else:
        num_weights = len(weight_indices)
        if not isinstance(weight_indices, np.ndarray):
            _weight_indices = np.array(weight_indices)
        else:
            _weight_indices = weight_indices

        # Filter for layer of interest
        layer_weights = getattr_nested(model, weight_name)
        if not layer_weights.requires_grad:
            raise AttributeError(
                f"Weights {weight_name} is not attached to compute graph. Cannot calculate Hessian."
            )

        # Split weight_indices into chunks, and process up to chunk_size at a time
        num_chunks = ceil(num_weights / chunk_size)
        # We split an index into weight_indices, rather than weight_indices itself
        index_splits = np.array_split(range(num_weights), num_chunks)

        # Calculate the loss just once
        if max_batch_size is not None:
            # Accumulate gradient over batches to reduce memory usage
            for i in range(X.shape[0]):
                X_batch = X[i : i + 1].to(device, non_blocking=True)
                y_batch = y[i : i + 1].to(device, non_blocking=True)

                with torch.amp.autocast(**get_autocast_params()):
                    y_pred = model(X_batch)
                    loss = loss_function(input=y_pred, target=y_batch)
                loss = loss / X.shape[0]
                loss.backward(inputs=layer_weights)

            grad = layer_weights._grad
        else:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(**get_autocast_params()):
                y_pred = model(X)
                loss = loss_function(input=y_pred, target=y)

            # Calculate gradient of loss w.r.t. layer weights
            [grad] = torch.autograd.grad(
                outputs=loss,
                inputs=layer_weights,
                create_graph=True,
                allow_unused=False,
            )

        # Ravel 2D gradient matrix into 1D gradient vector
        grad = grad.view(-1)

        # Calculate Hessian of each strip and aggregate
        hess = torch.zeros((num_weights, num_weights), device=device)
        for index_split in index_splits:
            # Find the weight indices corresponding to this index split - we need this
            # so that the Hessian is calculated with respect to the right weight indices.
            # But when indexing/slicing into hess or hess_block, we use index_split!
            weight_split = _weight_indices[index_split]
            # We calculate a row strip of the Hessian consisting of all columns and
            # just the weight indices in weight_split.
            hess_rows = calc_hessian_rows(layer_weights, weight_split, grad)
            hess[index_split, :] = hess_rows[range(len(index_split)), :][
                :, weight_indices
            ].to(device)

    model.zero_grad(set_to_none=True)

    return hess


def calc_hessian_rows(layer_weights, weight_indices, grad):
    """Helper that calculates rows of the Hessian given by weight_indices."""
    # Initialize the "strip" = rows of the Hessian
    num_weights = layer_weights.numel()  # All the weights
    num_weight_indices = len(weight_indices)  # Just the chosen ones
    hess_rows = torch.zeros(num_weight_indices, num_weights)

    # Steps through the weight_indices and calculates the gradient of the respective
    # gradient elements, yielding the second derivatives.
    for i, weight_index in enumerate(weight_indices):
        [row] = torch.autograd.grad(
            outputs=grad[weight_index],
            inputs=layer_weights,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        row = row.view(-1)
        hess_rows[i, :].add_(row.type_as(hess_rows))  # Add this row to the Hessian

    return hess_rows


def calc_grad_autograd(
    model,
    loss_function,
    X,
    y,
    weight_name,
    device,
    max_batch_size=None,
    indices_block=None,
):
    """Calculates the gradient of the loss function, using autograd."""
    # Filter for layer of interest
    layer_weights = getattr_nested(model, weight_name)

    if max_batch_size is not None:
        num_samples = X.shape[0]

        # Accumulate the gradients across sub-batches of size max_batch_size
        for X_batch, y_batch in zip(
            torch.split(X, max_batch_size, dim=0),
            torch.split(y, max_batch_size, dim=0),
        ):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            batch_size = X_batch.shape[0]

            with torch.amp.autocast(**get_autocast_params()):
                y_pred = model(X_batch)
                loss = loss_function(input=y_pred, target=y_batch)
            del X_batch, y_batch, y_pred

            loss = loss / num_samples * batch_size
            loss.backward(inputs=layer_weights)

        grad = layer_weights._grad

    else:
        grad = calc_grad_autograd_unbatched(
            model,
            loss_function,
            X,
            y,
            weight_name,
            device,
            create_graph=False,
            allow_unused=False,
        )

    # Ravel 2D gradient matrix into 1D gradient vector
    grad = grad.view(-1)
    # Select out indices for block, if provided
    if indices_block is not None:
        grad = grad[indices_block]

    # Clear gradient caches across model
    model.zero_grad(set_to_none=True)

    return grad


def calc_grad_autograd_unbatched(
    model, loss_function, X, y, weight_name, device, **kwargs
):
    """Calculates the full per-sample gradient using autograd.

    Note: additional kwargs are passed to torch.autograd.grad().
    """
    layer_weights = getattr_nested(model, weight_name)
    X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
    with torch.amp.autocast(**get_autocast_params()):
        y_pred = model(X)
        loss = loss_function(input=y_pred, target=y)
    del X, y, y_pred

    # Calculate gradient of loss w.r.t. layer weights
    [grad] = torch.autograd.grad(outputs=loss, inputs=layer_weights, **kwargs)

    return grad


def calc_grad_functional(model, loss_function, X, y, weight_name):
    """Calculates the gradient of the loss function using torch.func.grad().

    Note: this function may fail on models that use BatchNorm with tracking on. For such
    models it's better to use calc_grad_autograd() which uses torch.autograd.grad().
    """
    # Filter for layer of interest
    layer_weights = getattr_nested(model, weight_name)

    # Calculate the gradient. argnums tells grad() which argument of calc_loss() to
    # calculate the gradient with respect to, counting from 0.
    grad = torch.func.grad(calc_loss, argnums=4)(
        model, loss_function, X, y, {weight_name: layer_weights}
    )
    grad = grad[weight_name]

    # At this point for a weights matrix with num_rows x num_cols, the gradient is a
    # tensor of dimensions [num_rows, num_cols]. We flatten it so that it's a vector
    # of length [num_rows*num_cols].
    grad = grad.view(-1)

    return grad


def calc_grad_sample_autograd(
    model,
    loss_function,
    X,
    y,
    weight_name,
    device,
    max_batch_size=None,
    indices_block=None,
):
    """Calculates the per-sample gradient using autograd.

    Note: at present we don't have a way of calculating the per-sample grad for several
    samples at once with autograd. So, max_batch_size is unused and included only for
    consistency.
    """
    # Calculate the regular grad, sample by sample
    grad_sample = []
    num_samples = X.shape[0]
    for i in range(num_samples):
        X_sample = X[i : i + 1]
        y_sample = y[i : i + 1]
        grad = calc_grad_autograd_unbatched(
            model,
            loss_function,
            X_sample,
            y_sample,
            weight_name,
            device,
        )
        del X_sample, y_sample

        # Flatten dimensions
        grad = grad.view(-1)
        # Select out indices for block, if provided
        if indices_block is not None:
            grad = grad[indices_block]
        grad_sample.append(grad)

    return torch.stack(grad_sample)


def calc_grad_sample_functional(
    model,
    loss_function,
    X,
    y,
    weight_name,
    device,
    max_batch_size=None,
    indices_block=None,
):
    """Calculates the per-sample gradient, using the functional interface."""
    # Note: below we deliberately do not follow the pattern in the other
    # calc_grad_sample functions in which the computation for each batch is done in a
    # separate function with the "unbatched" suffix. The reason is that the set up is
    # more involved here, and we'd like to avoid repeating it for every batch (or
    # passing additional variables in, breaking the consistency).

    # Use underscore to avoid obfuscation with the calc_loss() that we use elsewhere
    def _calc_loss(params, buffers, X, y):
        y_pred = torch.func.functional_call(model, (params, buffers), (X,))
        loss = loss_function(input=y_pred, target=y)
        return loss

    # Define a function that returns the gradient of the loss. It has the same
    # arguments as _calc_loss(), leveraging variables from outer scope. This
    # gradient can be filtered if a set of indices indices_block is provided.
    def _calc_grad_filtered(*args):
        grad = torch.func.grad(_calc_loss)(*args)
        # Unpack the dictionary, which should have just one key - the name of this layer
        # and flatten it
        grad = grad[weight_name].view(-1)
        # Select out indices for block, if provided
        if indices_block is not None:
            # Note: this slice produces a copy as indicated in the PyTorch documentation:
            # "When accessing the contents of a tensor via indexing, PyTorch follows
            # Numpy behaviors that basic indexing returns views, while advanced indexing
            # returns a copy. Assignment via either basic or advanced indexing is
            # in-place."
            grad_ = grad[indices_block]
            del grad
        else:
            grad_ = grad

        return grad_

    # Vectorize the above function over the sample and target (3rd and 4th) args of
    # _calc_loss(), but use the same params and buffers (1st and 2nd) args for all
    # batches.
    calc_grad_sample = torch.func.vmap(_calc_grad_filtered, in_dims=(None, None, 0, 0))

    params = {k: v for k, v in model.named_parameters() if k == weight_name}
    buffers = {
        k: v
        for k, v in model.named_buffers()
        if k.startswith(weight_name.rstrip("weight"))
    }

    num_samples = X.shape[0]
    if max_batch_size is None:
        max_batch_size_ = num_samples
    else:
        max_batch_size_ = max_batch_size

    # Accumulate the gradients across sub-batches of size max_batch_size
    grad_sample = []
    for X_batch, y_batch in zip(
        torch.split(X, max_batch_size_, dim=0),
        torch.split(y, max_batch_size_, dim=0),
    ):
        # At this point y_batch is a 1D vector. We need to expand it to a 2D column
        # vector. Otherwise, the vectorization over a 1D vector will select "nothing"
        # at each iteration. With the expansion, it gets a single label, as expected.
        y_batch = y_batch.unsqueeze(1)
        # Similarly, we need an extra dimension here to get the right dimensions inside
        # the vectorization
        X_batch = X_batch.unsqueeze(1)

        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        with torch.no_grad():
            grad_sample_batch = calc_grad_sample(params, buffers, X_batch, y_batch)
        del X_batch, y_batch
        grad_sample.append(grad_sample_batch)

    return torch.vstack(grad_sample)


def calc_grad_autograd_unbatched_llm(
    model, loss_function, X_ids, X_mask, y_sample, weight_name, **kwargs
):
    """Calculates gradient of LLM using autograd.

    Note: additional kwargs are passed to torch.autograd.grad().
    """
    layer_weights = getattr_nested(model, weight_name)

    # This functionality was adopted from:
    # https://github.com/huggingface/transformers/blob/892399c5ffdc3c7d0e4f9b2c1c81511293c1581b/src/transformers/models/llama/modeling_llama.py#L707
    with torch.amp.autocast(**get_autocast_params()):
        prediction = model(input_ids=X_ids, attention_mask=X_mask)
        logits = prediction.logits

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = y_sample[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device, non_blocking=True)

        loss = loss_function(shift_logits, shift_labels)

    [grad] = torch.autograd.grad(outputs=loss, inputs=layer_weights, **kwargs)

    return grad


def calc_grad_sample_autograd_llm(
    model,
    loss_function,
    X,
    y,
    weight_name,
    device,
    max_batch_size=None,
    indices_block=None,
):
    """Calculates the per-sample gradient, using autograd. LLM version."""
    # Compile inputs and outputs
    X_ids = torch.stack(tuple(sample["input_ids"] for sample in X))
    X_mask = torch.stack(tuple(sample["attention_mask"] for sample in X))
    y = torch.stack(y)

    num_samples = len(X)
    if max_batch_size is None:
        max_batch_size_ = num_samples
    else:
        max_batch_size_ = max_batch_size

    grad_sample = []
    for X_ids_batch, X_mask_batch, y_batch in zip(
        torch.split(X_ids, max_batch_size_, dim=0),
        torch.split(X_mask, max_batch_size_, dim=0),
        torch.split(y, max_batch_size_, dim=0),
    ):
        # Move the whole batch to device
        X_ids_batch = X_ids_batch.to(device, non_blocking=True)
        X_mask_batch = X_mask_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Calculate the gradient for each sample in the batch
        for X_ids_sample, X_mask_sample, y_sample in zip(
            X_ids_batch, X_mask_batch, y_batch
        ):
            grad = calc_grad_autograd_unbatched_llm(
                model, loss_function, X_ids_sample, X_mask_sample, y_sample, weight_name
            )
            # Flatten dimensions
            grad = grad.view(-1)
            # Select out indices for block, if provided
            if indices_block is not None:
                grad = grad[indices_block]
            # Move this grad to CPU, so that we don't overload the GPU's memory
            grad_cpu = grad.detach().cpu()
            del grad
            grad_sample.append(grad_cpu)

            clear_memory_caches()

    return torch.stack(grad_sample)


def calc_grad_autograd_llm(
    model,
    loss_function,
    X,
    y,
    weight_name,
    device,
    max_batch_size=None,
    indices_block=None,
):
    """Calculates the gradient, using autograd. LLM version."""
    # Compile inputs and outputs
    X_ids = torch.stack(tuple(sample["input_ids"] for sample in X))
    X_mask = torch.stack(tuple(sample["attention_mask"] for sample in X))
    y = torch.stack(y)

    num_samples = len(X)
    if max_batch_size is None:
        max_batch_size_ = num_samples
    else:
        max_batch_size_ = max_batch_size

    grad_size = (
        len(indices_block)
        if indices_block is not None
        else getattr_nested(model, weight_name).numel()
    )
    grad = torch.zeros(grad_size, device=device)

    for X_ids_batch, X_mask_batch, y_batch in zip(
        torch.split(X_ids, max_batch_size_, dim=0),
        torch.split(X_mask, max_batch_size_, dim=0),
        torch.split(y, max_batch_size_, dim=0),
    ):
        # For batches of size>1 the X's are 3D with a trivial middle dimension - so we
        # squeeze it (if needed), and then move the batch to the device
        X_ids_batch = X_ids_batch.squeeze(dim=1).to(device, non_blocking=True)
        X_mask_batch = X_mask_batch.squeeze(dim=1).to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Calculate the gradient for this batch
        grad_batch = calc_grad_autograd_unbatched_llm(
            model, loss_function, X_ids_batch, X_mask_batch, y_batch, weight_name
        )
        # Flatten dimensions
        grad_batch = grad_batch.view(-1)
        # Select out indices for block, if provided
        if indices_block is not None:
            grad_batch = grad_batch[indices_block]

        # grad_batch is an average over the batch_size. Since the batches could differ
        # in size we need to multiply each batch's contribution by the batch_size, and
        # then at the end divide by num_samples.
        batch_size = X_ids_batch.shape[0]

        # When using multiple GPUs, grad_batch can end up on a different GPU, so we
        # need to move it over for the addition.
        grad += grad_batch.to(grad.device) * batch_size

        del grad_batch
        clear_memory_caches()

    grad /= num_samples
    return grad
