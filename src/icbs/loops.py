# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import itertools
import time

import numpy as np
import torch
from datasets import IterableDataset

from .util import get_autocast_params
from .util_llm import prep_loader_batch


def train_loop(dataloader, model, loss_function, optimizer, device, verbose=100):
    """Runs the training loop."""
    size = len(dataloader.dataset)

    # Set the model to training mode - important for batch normalization and dropout
    # layers
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        num_samples = len(X)
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Zero the gradients for each batch
        optimizer.zero_grad(set_to_none=True)

        # Compute prediction and loss
        with torch.amp.autocast(**get_autocast_params()):
            y_pred = model(X)
            loss = loss_function(y_pred, y)
        del X, y, y_pred

        # Backpropagation
        loss.backward()
        optimizer.step()

        if verbose and batch % verbose == 0:
            loss, current = loss.item(), (batch + 1) * num_samples
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def valid_loop(dataloader, model, loss_function, device, verbose=1):
    """Runs the validation loop."""
    # Set the model to evaluation mode - important for batch normalization and dropout
    # layers
    model.eval()
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    # Evaluating the model in inference mode ensures that no gradients are computed
    # during test mode, and reduces unnecessary gradient computations and memory usage
    # for tensors with requires_grad=True
    loss, correct = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(**get_autocast_params()):
                y_pred = model(X)
                loss += loss_function(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            del X, y, y_pred

    loss /= num_batches
    accuracy = correct / size * 100

    if verbose:
        print(f"  Accuracy: {accuracy:>0.2f}%, Loss: {loss:>8f}")

    return loss, accuracy


def valid_loop_llm(dataloader, model, loss_function, device, verbose=1):
    """Runs the validation loop. LLM version."""
    # Set the model to evaluation mode - important for batch normalization and dropout
    # layers
    model.eval()

    # Evaluating the model in inference mode ensures that no gradients are computed
    # during test mode also serves to reduce unnecessary gradient computations and
    # memory usage for tensors with requires_grad=True
    loss = 0
    num_samples = 0
    with torch.inference_mode():
        nlls = []  # List to store negative log likelihoods
        for i, batch in enumerate(dataloader):
            if verbose:
                print(f"Valid loop batch {i}")
            X, y = prep_loader_batch(batch, dataloader.tokenizer, model.seqlen)

            # Concatenate all input_ids and attention_mask tensors
            # They are of shape batch_size x seqlen
            in_ids = torch.cat(tuple(sample["input_ids"] for sample in X))
            in_ids.requires_grad = False
            in_mask = torch.cat(tuple(sample["attention_mask"] for sample in X))
            in_mask.requires_grad = False

            num_samples += in_ids.numel()

            with torch.amp.autocast(**get_autocast_params()):
                in_ids = in_ids.to(device, non_blocking=True)
                in_mask = in_mask.to(device, non_blocking=True)
                # Get the output of the model = logits
                # They are of dimension: batch_size x model.seqlen x vocab_size
                logits = model(input_ids=in_ids, attention_mask=in_mask).logits

                # Shift logits and labels for next token prediction
                # Remove last token
                shift_logits = logits[:, :-1, :].contiguous()
                # Shift to the right, effectively treating the current token as the
                # label for the next token prediction
                shift_labels = in_ids[:, 1:]

                batch_loss = loss_function(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                )

                # Calculate negative log likelihood
                neg_log_likelihood = batch_loss * model.seqlen * len(X)

                # Append to list of negative log likelihoods
                nlls.append(neg_log_likelihood)
                loss += batch_loss.item()

            del X, y

    num_batches = len(nlls)
    loss /= num_batches

    # Compute perplexity
    num_samples /= model.seqlen
    perplexity = torch.exp(torch.stack(nlls).sum() / (num_samples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    if verbose:
        print(f"  Perplexity: {perplexity:>0.2f}%, Avg loss: {loss:>8f}")

    return loss, perplexity.item()


def prune_loop(
    dataloader, model, pruner, device, w_names_step=None, max_steps=None, verbose=0
):
    """Prunes the model on calling step() using pruner given a batch from dataloader.

    Note: performs one full iteration through the dataloader, i.e., one epoch, at a
    given temperature.

    Args:
        dataloader (torch.utils.data.DataLoader ): A dataloader from which batches will
            be retrieved.
        model (torch.Module): The model that will be pruned.
        pruner (Pruner): An object that performs pruning, see Pruner.
        device (str): The device to use (a la torch), such as "cpu" or "gpu".
        w_names_step (iterable of str, optional): List of weight names to prune, one for
            each step, or None to use the pruner's default (equal probability). Defaults
            to None.
        max_steps (int, optional): Maximum number of steps to run. Defaults to None.
        verbose (int, optional): Whether to print progress every verbose steps.
            Defaults to 0 - no printing.
    """
    # Set the model to eval mode, to avoid in-place accrual operations in some
    # BatchNorm-type layers
    model.eval()

    if w_names_step is None:
        if max_steps is None:
            n_steps = len(dataloader)
        else:
            n_steps = max_steps

        w_names_step_ = itertools.repeat(None, n_steps)

    else:
        # Randomize order
        w_names_step_ = np.random.permutation(w_names_step)

        if max_steps is None or max_steps >= len(w_names_step):
            n_steps = len(w_names_step)

        elif max_steps < len(w_names_step):
            n_steps = max_steps
            w_names_step_ = w_names_step_[:n_steps]

    if not isinstance(dataloader.dataset, IterableDataset):
        # The below doesn't apply to infinite streaming datasets
        if n_steps > len(dataloader):
            print(
                f"WARNING: n_steps ({n_steps}) is larger than the number of batches in the dataloader ({len(dataloader)})."
                " n_steps will be reduced so that it is equal to the number of batches."
            )
            n_steps = len(dataloader)

    for batch_num, (batch, w_name) in enumerate(
        zip(dataloader, w_names_step_), start=1
    ):
        if verbose and batch_num % verbose == 0:
            print("-----------------------")
            print(f"Pruner step {batch_num} (of {n_steps})")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.perf_counter()
        try:
            seqlen = model.seqlen
            tokenizer = dataloader.tokenizer
        except AttributeError:
            seqlen = None
            tokenizer = None

        X, y = prep_loader_batch(batch, tokenizer, seqlen)
        pruner.step(X, y, w_name)
        del X, y

        if verbose and batch_num % verbose == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start
            print(f"    Pruner step took: {step_time:.2f}s")
