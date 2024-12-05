# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import time
from collections import deque
from itertools import count, takewhile

import numpy as np
import psutil
import torch
from bitarray import bitarray

from .derivatives import (
    calc_grad_autograd,
    calc_grad_autograd_llm,
    calc_grad_sample_autograd_llm,
    calc_grad_sample_functional,
    calc_hessian,
    calc_hessian_piecewise,
)
from .move import Move
from .problem import Problem
from .state import State
from .util import (
    array_slice,
    clear_memory_caches,
    get_autocast_params,
    getattr_nested,
    print_total_gpu_memory_allocated,
    sorted_indices_2d_to_1d,
)
from .util_llm import is_llm

all_init_methods = []
for init_method in ["wanda", "magnitude", "gradient", "random"]:
    all_init_methods.append(init_method)
    if init_method != "random":
        for suffix in ["_per_output", "_per_input"]:
            all_init_methods.append(init_method + suffix)


class Pruner:
    """A pruner object for pruning neural networks."""

    def __init__(
        self,
        model,
        loss_function,
        density,
        block_solver,
        block_size,
        k,
        layer_names,
        init_method="magnitude",
        selection_method="random",
        grad_multiplier=1.0,
        ridge_multiplier=0,
        calc_hessian_method="exact",
        seed=None,
        verbose=0,
        X_init=None,
        y_init=None,
        tabu_frac=0,
        fix_frac_prune=0,
        fix_frac_keep=0,
        max_batch_size=None,
        remove_zero_rows=False,
    ):
        """Instantiates a pruner object for pruning neural networks.

        Args:
            model (torch.Module): The model that will be pruned.
            loss_function (torch.Loss): Unused currently, retained only to fit the
                signature of the training and validation loop (and might be used in the
                future).
            density (float): Target pruning density (inverse of sparsity).
            block_size (int): Number of weights to select in each block.
            block_solver (Solver): Solve the block optimization problem. Has signature
                solver.solve(Q, k), where the objective is x.T*Q*x, and k is the
                cardinality, meaning the number of 1's desired in the optimized x.
            k (int): Number of weights to prune in each block.
            layer_names (list): List of layers to process.
            init_method (str, optional): Initialization method. "random" prunes randomly
                chosen weights. "magnitude" prunes the weights with the smallest
                absolute magnitude. "wanda" prunes the weights with the smallest product
                of the weight magnitude and the L2 norm of the respective activation.
                "gradient" prunes the weights with the smallest first order term, which
                is the product of the weight value and the respective gradient. Except
                for "random", the other methods can be suffixed by "_per_output" or
                "_per_input". Otherwise they prune "per layer". Defaults to "magnitude".
            selection_method (str, optional): Method to select the candidate weights for
                each block. See a description of the possible values under init_method.
            grad_multiplier (float): Multiplier for the gradient term in the
                optimization problem, or 0 to disable. Defaults to 1.0.
            ridge_multiplier (float): Multiplier for the ridge term in the optimization
                problem, or 0 to disable. Defaults to 0.
            calc_hessian_method (str): How to calculate the Hessian. Options are:
                "exact", "exact_piecewise", "gradient_per_sample", and "gradient_mean".
                Defaults to "exact".
            seed (int, optional): Seed for random number generator or None to set a
                random seed. Defaults to None.
            verbose (int, optional): Level of verbosity - higher provides more verbose
                output. Defaults to 0.
            X_init (torch.Tensor, optional): Data batch used for the initial pruning
                to: 1. cache model layer activations when using init_method="wanda", and
                2. calculate the gradient when using init_method="gradient". Defaults to
                None.
            y_init (torch.Tensor, optional): Target batch used for the initial pruning
                to calculate the gradient when using init_method="gradient". Defaults to
                None.
            tabu_frac (float, optional): Maximum fraction of each layer to hold in tabu
                list at any point in time. If additional weights are added to the tabu
                list for this layer above this fraction, the oldest weights (first in)
                will be ejected from the list. 0 disables this feature. Defaults to 0.
            fix_frac_prune (float, optional): Fraction of initial pruning set with
                lowest score to fix for the duration of the optimization. 0 disables
                this feature. Defaults to 0.
            fix_frac_keep (float, optional): Fraction of initial keep set with
                highest score to fix for the duration of the optimization. 0 disables
                this feature. Defaults to 0.
            max_batch_size (int, optional): Maximum of size of sub-batches to process
                when pushing data through the model. This is important in
                memory-constrained environments, like on GPU. NOTE: Setting to None will
                remove any gradient accumulation and sub-batching of data within the
                pruner. In this case, it is assumed the provided data batch is
                sufficiently small to fit in device memory. Defaults to None.
            remove_zero_rows (bool, optional): Whether to remove variables with dL=0
                from Q (always pruning them), or run them through the block solver.
                Defaults to False.
        """
        self.model = model
        self.loss_function = loss_function
        self.density = density
        self.block_size = block_size
        self.block_solver = block_solver
        self.k = k
        self.init_method = init_method
        self.selection_method = selection_method
        self.grad_multiplier = grad_multiplier
        self.ridge_multiplier = ridge_multiplier
        self.calc_hessian_method = calc_hessian_method
        self.layer_names = layer_names
        self.seed = seed
        self.verbose = verbose
        self.X_init = X_init
        self.y_init = y_init
        self.tabu_frac = tabu_frac
        self.fix_frac_prune = fix_frac_prune
        self.fix_frac_keep = fix_frac_keep
        self.max_batch_size = max_batch_size
        self.remove_zero_rows = remove_zero_rows

        self._model_device = next(iter(model.parameters())).device
        self._cuda_is_available = torch.cuda.is_available()

        if density <= 0 or density >= 1:
            raise ValueError("density must be between 0 and 1")

        if init_method.startswith("wanda") and (X_init is None):
            raise ValueError('X_init must be provided when using init_method="wanda"')

        if init_method.startswith("gradient") and (
            (X_init is None) or (y_init is None)
        ):
            raise ValueError(
                'X_init and y_init must be provided when using init_method="gradient"'
            )

        # Placeholders
        self._current_loss = None
        self._rng = None
        self._activations_squared = {}
        self._activation_hook_handles = {}

        # Diagnostics
        self.dL_track = []

        self._prep_weight_dict()
        self.reset()

    def _do_batched_forward(self, X):
        """Do model forward pass over batches of input X."""
        if self.verbose >= 2:
            if self._cuda_is_available:
                torch.cuda.synchronize()
            forward_start = time.perf_counter()

        num_samples = len(X)
        if self.max_batch_size is None:
            max_batch_size = num_samples
        else:
            max_batch_size = self.max_batch_size

        with torch.inference_mode():
            if is_llm(self.model):
                # Make forward pass to update cached activations
                self._do_batched_forward_llm(X, max_batch_size)
            else:
                # Make forward pass to update cached activations
                self._do_batched_forward_not_llm(X, max_batch_size)

        if self.verbose >= 2:
            if self._cuda_is_available:
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - forward_start
            print(f"    _do_batched_forward took: {forward_time:.2f}s")

    def _do_batched_forward_not_llm(self, X, max_batch_size):
        """Do model forward pass over batches of input data - for non-LLM models."""
        if self.verbose >= 2:
            print(
                f"Doing batched forward pass. Data shape: {X.shape}, batch size: {max_batch_size}"
            )
        for X_batch in torch.split(X, max_batch_size, dim=0):
            X_batch = X_batch.to(self._model_device, non_blocking=True)
            with torch.amp.autocast(**get_autocast_params()):
                self.model(X_batch)
            del X_batch

    def _do_batched_forward_llm(self, X, max_batch_size):
        """Do model forward pass over batches of input data for LLM."""
        # Compile inputs
        X_ids = torch.stack(tuple(sample["input_ids"] for sample in X))
        X_ids.requires_grad = False
        X_mask = torch.stack(tuple(sample["attention_mask"] for sample in X))
        X_mask.requires_grad = False

        if self.verbose >= 2:
            print(
                f"Doing LLM batched forward pass. Data shape: {X_ids.shape}, batch size: {max_batch_size}"
            )

        for X_ids_batch, X_mask_batch in zip(
            torch.split(X_ids, max_batch_size, dim=0),
            torch.split(X_mask, max_batch_size, dim=0),
        ):
            # For batches of size>1 the X's are 3D with a trivial middle dimension - so
            # we squeeze it (if needed), and then move the batch to the device
            X_ids_batch = X_ids_batch.squeeze(dim=1).to(
                self._model_device, non_blocking=True
            )
            X_mask_batch = X_mask_batch.squeeze(dim=1).to(
                self._model_device, non_blocking=True
            )

            # Forward pass
            with torch.amp.autocast(**get_autocast_params()):
                self.model(input_ids=X_ids_batch, attention_mask=X_mask_batch)

            del X_ids_batch, X_mask_batch

    def _get_layer_norm_dims(self, weight_name, layer_input):
        """Logic to identify dimensions to take torch.norm over.

        Note: Might not be necessary, if we remove any 2d/3d layers from pruning
        consideration upstream.
        """
        layer = getattr_nested(self.model, weight_name.replace(".weight", ""))
        if isinstance(layer, torch.nn.BatchNorm2d):
            # Keep dim=1, which is dimension of interest
            dims = [0] + list(range(2, len(layer_input.shape)))
        elif isinstance(layer, (torch.nn.LayerNorm, torch.nn.Linear)):
            # Assume N dims, keep last dim (N-1)
            dims = list(range(0, len(layer_input.shape) - 1))
        else:
            raise NotImplementedError(f"Unable to process layer type: {type(layer)}")

        del layer

        return dims

    def _reset_activations(self):
        """Empties activation dict (sets to None) to reset calculations."""
        if self.verbose >= 2:
            print("Clearing activations dictionary")

        for k in self._activations_squared:
            self._activations_squared[k] = None

    def _apply_activation_hooks(self):
        """Registers functions to get layer activations as forward hooks on model."""

        def get_activation_hook(name):
            def hook(module, input, output):
                # NOTE: module is the particular layer object,
                # output is a single tensor of activations (not a tuple),
                # input is the layer inputs - tuples of tensors (one tensor per input
                # channel/source?). Since we only have one input channel per layer, we
                # just need to retrieve that.
                dims = self._get_layer_norm_dims(name, input[0])

                # Due to memory requirements, we will aggregate the square of the norm
                # along the sample dimension until we are ready to calculate the
                # WANDA metric.
                # NOTE: tensor.detach() does not create a new object in memory, so
                # in-place operations on the result are captured. This is causing issues
                # with pytorch functional gradient calls, which break on in-place
                # operations on tensors. autograd.grad() does not break on these
                # operations. We avoid this issue by tracking the activations with a
                # numpy array.
                # NOTE: Instead of taking the norm and squaring it, as done in the WANDA
                # paper/code, we square the activations, and sum them, which is
                # equivalent, but saves the additional sqrt (in the norm) and squaring
                # (out of it), which cancel.
                # NOTE: We convert to float32 before squaring to try to avoid
                # underflowing/overflowing.
                # NOTE: The paper uses a sum, but the Wanda code uses the mean. It
                # doesn't really matter since the difference amounts to multiplying the
                # result by a constant. So, we align with the paper and use a sum.
                activations = input[0].detach().cpu().to(torch.float32).numpy()
                activations_squared = (activations**2).sum(axis=tuple(dims))

                if self._activations_squared[name] is None:
                    self._activations_squared[name] = activations_squared
                else:
                    self._activations_squared[name] += activations_squared

                if self.verbose >= 2 and self._model_device == "cuda":
                    print_total_gpu_memory_allocated()

            return hook

        for w_name in self._weight_dict:
            # Remove the .weight / .bias suffix
            layer_name = ".".join(w_name.split(".")[:-1])
            # Add activation hook to the layer. Map the hook results to w_name so
            # they will be easier to retrieve during processing
            layer = getattr_nested(self.model, layer_name)
            self._activations_squared[w_name] = None  # initialize to empty
            self._activation_hook_handles[w_name] = layer.register_forward_hook(
                get_activation_hook(w_name)
            )

    def _remove_activation_hooks(self):
        """Removes the activations hooks."""
        for hook_handle in self._activation_hook_handles.values():
            hook_handle.remove()
        self._activation_hook_handles.clear()

    def __del__(self):
        """Removes the activation hooks when deleting the pruner."""
        self._remove_activation_hooks()

    def _prep_weight_dict(self):
        """Prepares the weights for processing."""
        self._weight_dict = {}
        for name in self.layer_names:
            weights = getattr_nested(self.model, name)

            # Flatten weights
            num_weights = weights.numel()
            weights_flat = weights.view(num_weights, 1)

            weights_original = weights.detach().cpu().clone()
            weights_original_flat = weights_original.view(num_weights, 1)

            self._weight_dict[name] = {
                "weights": weights,
                "weights_flat": weights_flat,
                "weights_original": weights_original,
                "weights_original_flat": weights_original_flat,
                "current_state": None,
            }
        clear_memory_caches()

    def reset(self):
        """Resets the Pruner, e.g.: the RNG and the weights to the initial solution."""
        if self.verbose >= 2:
            print("Resetting pruner")

        # Reset self._grad and the other cached derivatives
        self._reset_cached_derivatives()

        # Reset the RNG
        self._rng = np.random.default_rng(self.seed)

        # Perform the initial pruning to self.density
        self._perform_init_pruning()

        self._current_loss = None

    def _perform_init_pruning(self):
        # Need to do a forward pass to cache the activations if we are to use the
        # WANDA metric for initialization
        if self.init_method.startswith("wanda"):
            self._apply_activation_hooks()
            self._reset_activations()
            self._do_batched_forward(self.X_init)
            self._remove_activation_hooks()

        total_init_pruning_time = 0
        for w_name, w_dict in self._weight_dict.items():
            if self.verbose >= 2:
                if self._cuda_is_available:
                    torch.cuda.synchronize()
                init_pruning_start = time.perf_counter()

            self._perform_init_pruning_layer(w_name, w_dict)

            if self.verbose >= 2:
                if self._cuda_is_available:
                    torch.cuda.synchronize()
                init_pruning_time = time.perf_counter() - init_pruning_start
                total_init_pruning_time += init_pruning_time
                print(
                    f"    _perform_init_pruning_layer() took: {init_pruning_time:.2f}s"
                )

        if self.verbose >= 2:
            print(f"Done with initial pruning in {total_init_pruning_time:.2f}s")

    def _perform_init_pruning_layer(self, w_name, w_dict):
        if self.verbose >= 2:
            print(f"Performing initial pruning on layer {w_name}")

            if self.verbose >= 3:
                if self._model_device == "cuda":
                    print_total_gpu_memory_allocated()
                print(f"CPU memory consumption: {psutil.virtual_memory().percent}%")

        # Shorthand
        weights_flat = w_dict["weights_flat"]
        weights_original_flat = w_dict["weights_original_flat"]

        # Number of weights - total, and the number to keep and prune
        num_weights = weights_flat.numel()
        N_keep = round(num_weights * self.density)
        N_prune = num_weights - N_keep

        if self.init_method == "random":
            # Randomly select N_prune weight indices
            init_prune = self._rng.choice(num_weights, N_prune, replace=False).tolist()
            init_keep_bitarray = bitarray(num_weights)
            # A loop is cheaper in memory than using tolist()
            for index in init_prune:
                init_keep_bitarray[index] = 1
            init_keep = [i for i, bit in enumerate(init_keep_bitarray) if not bit]

            if (self.fix_frac_prune != 0) or (self.fix_frac_keep != 0):
                print(
                    "WARNING: fix_frac_prune and fix_frac_keep are not supported with init_method=random (since there are no scores), setting them to zero"
                )
                self.fix_frac_prune = 0
                self.fix_frac_keep = 0

            fix_prune = []
            fix_keep = []
            num_fix_prune = 0
            num_fix_keep = 0

        else:
            init_method_base = self.init_method.split("_")[0]
            weight_scores = self._calc_weight_scores(
                init_method_base, w_name, self.X_init, self.y_init
            )

            if not self.init_method.endswith(
                "per_output"
            ) and not self.init_method.endswith("per_input"):
                # "Per layer" - make into an array with a single row
                weight_scores = weight_scores.reshape(1, -1)
                sort_axis = 1

            else:
                # "Per output" or "per input"

                # Reshape the scores to have the shapes of the weights
                weight_scores = weight_scores.reshape(w_dict["weights"].shape)

                # A Linear layer of dimensions (m,n) has n inputs and m outputs. Sorting
                # with axis=1 means that we sort each row separately. Each row
                # corresponds to a different output, hence sorting with axis=1 means
                # per-output.
                if self.init_method.endswith("per_output"):
                    sort_axis = 1
                elif self.init_method.endswith("per_input"):
                    sort_axis = 0

            # Indices that order weight_scores (2D) in ascending order. We first find
            # the indices within each row/column, but then convert them to indices into
            # the flattened weights (which is what we need, ultimately)
            ordering = np.argsort(weight_scores, axis=sort_axis)
            ordering = sorted_indices_2d_to_1d(ordering, axis=sort_axis)

            # We need to convert N_keep and N_prune from number of weights to number of
            # rows/columns for the per input/output case. Note: for the per layer case,
            # num_rows=1, so the below scaling will not change N_keep and N_prune, as
            # expected.
            num_rows, num_cols = weight_scores.shape
            if sort_axis == 1:
                # Number of columns to keep/prune
                N_keep //= num_rows
                N_prune = num_cols - N_keep
            elif sort_axis == 0:
                # Number of rows to keep/prune
                N_keep //= num_cols
                N_prune = num_rows - N_keep

            # Fix the lowest scoring fix_frac_prune weights (always pruned). The initial
            # solution will prune the rest of the lowest-scoring N_prune weights.
            # Note that num_fix_prune is the number of rows/columns to fix.
            num_fix_prune = (
                round(N_prune * self.fix_frac_prune) if self.fix_frac_prune != 0 else 0
            )
            # Equivalent to fix_prune = ordering[:num_fix_prune, :] if sort_axis=0,
            # and ordering[:, :num_fix_prune] if sort_axis=1
            fix_prune = (
                array_slice(ordering, sort_axis, 0, num_fix_prune).reshape(-1).tolist()
            )
            # Equivalent to init_prune = ordering[num_fix_prune:N_prune, :] if
            # sort_axis=0 and init_prune = ordering[:, num_fix_prune:N_prune] if
            # if sort_axis=1
            init_prune = (
                array_slice(ordering, sort_axis, num_fix_prune, N_prune)
                .reshape(-1)
                .tolist()
            )

            # Fix the highest scoring fix_frac_keep weights (always keep). The initial
            # solution will keep the rest of the highest-scoring N_keep weights.
            if self.fix_frac_keep != 0:
                # Note that num_fix_keep is the number of rows/columns to fix.
                num_fix_keep = round(N_keep * self.fix_frac_keep)

                # Equivalent to fix_keep = ordering[-num_fix_keep:, :] if sort_axis=0,
                # and fix_keep = ordering[:, -num_fix_keep:] if sort_axis=1
                fix_keep = (
                    array_slice(ordering, sort_axis, -num_fix_keep, None)
                    .reshape(-1)
                    .tolist()
                )
                # Equivalent to init_keep = ordering[-N_keep:-num_fix_keep, :] if
                # sort_axis=0 and init_keep = ordering[:, -N_keep:-num_fix_keep] if
                # sort_axis=1
                init_keep = (
                    array_slice(ordering, sort_axis, -N_keep, -num_fix_keep)
                    .reshape(-1)
                    .tolist()
                )
            else:
                num_fix_keep = 0
                fix_keep = []
                # Equivalent to init_keep = ordering[-N_keep:, :] if sort_axis=0, and
                # init_keep = ordering[:, -N_keep: if sort_axis=1]
                init_keep = (
                    array_slice(ordering, sort_axis, -N_keep, None).reshape(-1).tolist()
                )

            # Free up memory as soon as possible
            del ordering

            # Reset the cached derivatives, so that we don't accidentally reuse
            # them for another layer in this loop
            self._reset_cached_derivatives()

        if self.verbose >= 1:
            print(
                f"{w_name=}, {num_weights=}, {self.density=}, {num_fix_keep=}, {num_fix_prune=}"
            )

        # Apply initial and fixed states to the model by changing the weights
        # Note: apply the fixed state first so that we can clear that memory, which
        # we won't need, unlike init_state which becomes current_state below
        fix_state = State(num_weights, fix_prune, fix_keep)
        fix_state.apply_to_weights(weights_flat, weights_original_flat)
        del fix_state, fix_prune, fix_keep

        init_state = State(num_weights, init_prune, init_keep)
        init_state.apply_to_weights(weights_flat, weights_original_flat)

        if self.tabu_frac != 0:
            # Candidates = all "free" (not fixed) weights in the layer
            num_candidates = len(init_prune) + len(init_keep)

            # Number of weights to tabu
            num_tabu = round(self.tabu_frac * num_candidates)
            # Validate the layer will have enough weights remaining after tabu
            # list is filled.
            if (num_candidates - num_tabu) < self.block_size:
                if self.verbose >= 1:
                    print(
                        f"Layer {w_name} cannot support tabu list of size "
                        f"{num_tabu} (tabu fraction {self.tabu_frac} * num "
                        f"candidate weights {num_candidates}), given block size "
                        f"{self.block_size}. Switching off tabu for this layer."
                    )
                tabu_list = None
            else:
                tabu_list = deque(maxlen=num_tabu)
        else:
            tabu_list = None

        w_dict.update(
            {
                "current_state": init_state,
                "tabu_list": tabu_list,
            }
        )

        del init_prune, init_keep
        if self.verbose >= 2:
            layer_density = weights_flat.count_nonzero().item() / weights_flat.numel()
            print(f"Done with initial pruning of layer {w_name}, {layer_density=:.3f}")

    def _reset_cached_derivatives(self):
        """Initializes / resets the cached derivatives to avoid reusing stale copies."""
        self._grad = None
        self._grad_block = None
        self._grad_sample = None
        self._grad_sample_block = None
        self._hess = None
        self._hess_block = None

    def step(self, X, y, w_name=None):
        """Executes a single BCD move.

        Args:
            X (torch.Tensor): The data to use for this step.
            y (torch.Tensor): The labels to use for this step.
            w_name (str, optional): Name of weights tensor to apply the step to. If None
                the layer / weights tensor is chosen randomly (from all of them).
                Defaults to None.
        """
        # Reset the saved derivatives to avoid reusing them from previous calls to
        # step()
        self._reset_cached_derivatives()

        if w_name is None:
            # Select layer / weights randomly
            w_names = list(self._weight_dict.keys())
            w_name = self._rng.choice(w_names, 1)[0]

        # Select indices of weights for the next block
        selected_pruned, selected_kept = self._select_indices_block(w_name, X, y)
        k_effective = len(selected_pruned)

        # We want to do indices_block = selected_pruned + selected_kept, but do it
        # in place on selected_pruned to use up less memory
        selected_pruned.extend(selected_kept)
        del selected_kept
        indices_block = selected_pruned

        # Slice the weights over the candidate indices
        w_dict = self._weight_dict[w_name]
        w0 = w_dict["weights_original_flat"][indices_block].to(
            torch.float32
        )  # Original weights (on CPU)
        w = (
            w_dict["weights_flat"][indices_block].detach().cpu().to(torch.float32)
        )  # Current weights
        dw = w - w0  # Change in weights from original to previous step

        if self.verbose >= 2:
            if self.verbose == 2:
                print(f"    {w_name=} {k_effective=}")

            elif self.verbose >= 3:
                print(f"    {w_name=} {k_effective=} {indices_block=}")
                print(f'    {w_dict["tabu_list"]}')

        # Calculate the Hessian for this block and add its contribution to Q and const
        hess_block = self._calc_hess_block(X, y, w_name, indices_block).detach().cpu()
        const = 0.5 * (dw.T * hess_block @ dw).sum().item()
        Q = (w0 * hess_block @ dw).diagflat() + 0.5 * w0.T * hess_block * w0
        n = len(Q)

        # Add gradient term
        if self.grad_multiplier != 0:
            grad_block = (
                self._calc_grad_block(X, y, w_name, indices_block).detach().cpu()
            )
            Q -= (self.grad_multiplier * w0.T * grad_block).diagflat()
            const -= self.grad_multiplier * (dw.T * grad_block).sum().item()

        if self.remove_zero_rows:
            # Find rows with dL=0. Note: we do this purposefully before we add the ridge
            # term - we want to know if dL=0 without that contribution, which can be
            # non-zero even if the Hessian row/column and the gradient are all zero. The
            # ridge term promotes x_i=0 (keep) but for zero grad and Hessian, we
            # actually want x_i=1 (prune) - since we can prune those weights "for free"
            # (no change to the loss function)
            filter_zero = Q.abs().sum(dim=1) == 0
            num_zero_rows = sum(filter_zero).item()
            if self.verbose >= 1:
                print(f"    {num_zero_rows=}")

            if num_zero_rows > 0:
                # Edge case - no need to call the solver - choose k_effective of the
                # zero rows to prune (x_i=1) and keep the rest (x_i=0)
                if num_zero_rows >= k_effective:
                    # Prune k_effective rows
                    zero_indices = filter_zero.nonzero().squeeze().numpy()
                    to_prune = self._rng.choice(
                        zero_indices, k_effective, replace=False
                    ).tolist()
                    # Construct to_keep via bitarray to reduce memory overhead
                    keep_bitarray = bitarray(n)
                    for index in to_prune:
                        keep_bitarray[index] = 1
                    to_keep = [i for i, bit in enumerate(keep_bitarray) if not bit]

                    move = Move(to_prune, to_keep)

                    # Update the current state and apply the move to the weights
                    w_dict["current_state"].apply_move(move)
                    move.apply_to_weights(
                        w_dict["weights_flat"],
                        w_dict["weights_original_flat"],
                    )

                    self.dL_track.append(0.0)

                    if self.verbose >= 1:
                        print(
                            f"    Problem solved trivially since {num_zero_rows=} and {k_effective=}, skipping solver"
                        )
                    return

                # We definitely want to prune weights with dL=0, so we remove them from
                # Q and modify k_effective - the number of weights to prune out of n
                Q = Q[~filter_zero, :][:, ~filter_zero]
                k_effective -= num_zero_rows

            w0_ = w0[~filter_zero]
            dw_ = dw[~filter_zero]

        else:
            w0_ = w0
            dw_ = dw

        if self.ridge_multiplier != 0:
            # The ridge term is delta_w**2
            const += self.ridge_multiplier * (dw_**2).sum().item()
            Q += self.ridge_multiplier * (w0_ * (2 * dw_ + w0_)).diagflat()

        # The .float() is needed in case Q is float16 - it needs to be float32 for CPU.
        Q = Q.float()
        # Q typically has tiny elements. We scale it up so that the mean of the abs of
        # the non-zero elements is 1.
        scaling = 1 / Q[Q != 0].abs().mean().item()
        Q *= scaling  # Must rescale resulting energies, see below

        if self.verbose >= 2:
            # Fraction of non-zero elements before applying cut-off
            Q_density = torch.count_nonzero(Q).item() / Q.numel()

        # We also cut off any tiny elements so that the solver doesn't have to deal with
        # them. By setting it to a very small number, we avoid issues of potentially
        # setting the cutoff too large. It seems that we don't gain that much by zeroing
        # out these elements anyway, and it's not something we care to experiment with
        # currently.
        cutoff = 1e-12
        Q[Q <= cutoff] = 0

        if self.verbose >= 2:
            # Fraction of non-zero elements after applying cut-off
            Q_density_cutoff = torch.count_nonzero(Q).item() / Q.numel()
            print(
                f"    {scaling=:.3e} {cutoff=:.3e} {Q_density=:.5f} {Q_density_cutoff=:.5f}"
            )

        # We use k_effective here and not k, to maintain the same number of pruned
        # weights before and after this step.
        problem = Problem(Q, k_effective)
        best_x, _ = self.block_solver.solve(problem)
        # The result returned from the solver can be a bit off if the numbers are very
        # small. So, for now we recalculate it. We also need to remember to unscale it.
        best_E = (best_x.T @ Q @ best_x).item() / scaling

        # Note: must check feasibility before expanding the solution below, since we
        # need to check the reduced solution with the reduced problem
        if not problem.is_feasible(best_x):
            raise ValueError("block_solver returned an infeasible solution")

        # Calculate estimated change in loss
        delta_w = -(dw_.cpu() + best_x * w0_.cpu())
        delta_w_squared = (delta_w**2).sum()
        ridge_contribution = self.ridge_multiplier * delta_w_squared
        dE = const + best_E  # Actual change in energy
        dL = dE - ridge_contribution  # Just change in loss
        self.dL_track.append(dL)

        if w_dict["tabu_list"] is not None:
            self._update_tabu_list(w_dict["tabu_list"], indices_block)

        if delta_w_squared != 0:
            if self.remove_zero_rows:
                # If zero rows/columns were removed, we need to expand the solution to
                # account for them
                best_x_expanded = torch.ones(n, 1)
                best_x_expanded[~filter_zero] = best_x
                best_x = best_x_expanded

            if self.verbose >= 2:
                # Re-evaluate at delta_w
                if self.remove_zero_rows:
                    delta_w_expanded = torch.zeros(n, 1)
                    delta_w_expanded[~filter_zero] = delta_w
                else:
                    delta_w_expanded = delta_w

                dL_eval = (
                    0.5
                    * delta_w_expanded.T
                    @ hess_block.cpu().float()
                    @ delta_w_expanded
                ).item()
                if self.grad_multiplier != 0:
                    dL_eval += (
                        self.grad_multiplier
                        * (delta_w_expanded.T @ grad_block.cpu().float()).item()
                    )

                print(
                    f"    {dL=:.3f} {dE=:.3f} {ridge_contribution=:.3f} {const=:.3f} {best_E=:.3f} {dL_eval=:.3f} {delta_w_squared=:.3f}"
                )

            # We extract the indices of the zeros/ones from best_x, and then convert
            # these indices (over the block) to the indices into the full weights
            best_x_flat = best_x.view(-1)
            to_prune = [indices_block[i] for i, x_i in enumerate(best_x_flat) if x_i]
            to_keep = [indices_block[i] for i, x_i in enumerate(best_x_flat) if not x_i]
            move = Move(to_prune, to_keep)

            # Make the move - Update the current state and the weights.
            w_dict["current_state"].apply_move(move)
            move.apply_to_weights(
                w_dict["weights_flat"], w_dict["weights_original_flat"]
            )

            if self.verbose >= 3:
                print(f"{move=}")
        else:
            if self.verbose >= 2:
                print(f"WARNING: move does not change the weights, i.e., delta_w=0")

    def _update_tabu_list(self, tabu_list, indices):
        # Thanks to the deque object, we don't need to build logic around keeping the
        # tabu list a particular length - the deque will eject the oldest entries (if
        # necessary).
        tabu_list.extend(indices)

    @staticmethod
    def flatten_and_filter(l, weight_indices=None):
        """Helper - flattens and applies weight_indices to tensors, if necessary."""
        if weight_indices is None:
            return l.flatten()
        else:
            return l.flatten()[weight_indices]

    def _calc_weight_scores(self, method, weight_name, X, y, weight_indices=None):
        """Scores the weights in weight_name using method.

        Note: if weight_indices is provided (not None), then only scores for the given
        indices will be calculated and returned.
        """
        layer_weights = self._weight_dict[weight_name]["weights_original"]

        if method == "magnitude":
            # Sort weight indices based on ascending magnitude
            weight_scores = np.abs(
                self.flatten_and_filter(
                    layer_weights.detach().cpu().numpy(),
                    weight_indices,
                )
            )

        elif method == "gradient":
            # Score = |w_i * G_i|
            # This is based on the idea of using the first order term in dL to evaluate
            # the importance of the weights. Recall that dL \simeq -G_i w_i x_i is the
            # contribution to the change in loss due to weight i (ignoring the
            # constant). If the absolute value of the prefactor G_i w_i is small, then
            # the theory is that the effect of that weight on the change in loss is
            # small as well, so it should be pruned.
            if self._grad is not None:
                # grad already calculated
                grad = self._grad
                if weight_indices is not None:
                    # Select out subset of cached gradient
                    grad = self._grad[weight_indices]
            else:
                if self.verbose >= 2:
                    if self._cuda_is_available:
                        torch.cuda.synchronize()
                    grad_start = time.perf_counter()
                # Forced to recalculate:
                # Calculate gradient. An important implementation detail is that
                # calculating the partial gradient requires calculating the whole
                # thing and then selecting out only the entries for weight_indices.
                # But if we're calculating the full gradient anyway, we might as
                # well calculate it just once. So, we overrule weight_indices below
                # and pass None even though the argument passed in might not have
                # been None, cache the full gradient, and then select out
                # weight_indices if needed. For cases in which
                # selection_mode="gradient", this saves calculating the gradient
                # once for each step, since otherwise it would be caculated twice -
                # once for each keep/prune set.
                if self.max_batch_size != 1:
                    grad = self._recalc_grad_block(
                        X, y, weight_name, indices_block=None
                    )

                else:
                    # For the case max_batch_size=1, calculating the grad requires
                    # calculating the grad_sample! So, we should cache it for downstream
                    # use, which means we don't need to re-calculate the per-sample
                    # grad to estimate the Hessian later, nor calculate the mean
                    # gradient. It is large, but we store it on the CPU, where we have
                    # less strict memory requirements.
                    grad_sample = self._recalc_grad_sample_block(
                        X, y, weight_name, indices_block=None
                    )
                    self._grad_sample = grad_sample
                    grad = self._grad_sample.mean(axis=0)

                self._grad = grad
                if weight_indices is not None:
                    grad = grad[weight_indices]

                if self.verbose >= 2:
                    if self._cuda_is_available:
                        torch.cuda.synchronize()
                    grad_time = time.perf_counter() - grad_start
                    print(f"    Calculated gradient in {grad_time:.2f}s")

                # If we are not sub-setting (via weight_indices), cache the gradient
                if weight_indices is None:
                    self._grad = grad

            # NOTE: grad will have already been filtered (if weight_indices is not
            # None), so passing weight_indices into flatten_and_filter() will break,
            # because the indices will not be aligned.
            # NOTE: cast to float32 to try and avoid under and overflowing - we make
            # sure to do this only once the tensors are on the CPU.
            layer_weights = layer_weights.detach().cpu().to(torch.float32).numpy()
            grad = grad.detach().cpu().to(torch.float32).numpy()
            weight_scores = np.abs(
                self.flatten_and_filter(
                    layer_weights,
                    weight_indices,
                )
                * self.flatten_and_filter(grad)
            )

        elif method == "wanda":
            # Score = w_i * ||activation_i||_2
            # NOTE: We are accruing the squared norm across the input dimension, so
            # we must take the square root here to get the L2 norm.
            layer_weights = layer_weights.detach().cpu().to(torch.float32).numpy()
            activation_norm = np.sqrt(self._activations_squared[weight_name])
            weight_scores = np.abs(
                self.flatten_and_filter(
                    layer_weights * activation_norm,
                    weight_indices,
                )
            )

        else:
            raise ValueError(f"Invalid weight scoring method {method}")

        return weight_scores

    def _select_indices_block(self, w_name, X, y):
        """Selects indices of weights for the next block."""
        w_dict = self._weight_dict[w_name]

        # Shorter naming. Note: these are lists.
        currently_pruned = w_dict["current_state"].to_prune
        currently_kept = w_dict["current_state"].to_keep

        if self.block_size >= len(currently_pruned) + len(currently_kept):
            # Select all - no need to calculate scores!
            return currently_pruned, currently_kept

        if self.selection_method == "random":
            # Score = random
            w_idx_pruned = currently_pruned
            self._rng.shuffle(w_idx_pruned)

            w_idx_kept = currently_kept
            self._rng.shuffle(w_idx_kept)

        else:
            # Other weight scoring methods
            if self.selection_method == "wanda":
                # Make forward pass to update cached activations
                self._apply_activation_hooks()
                self._reset_activations()
                self._do_batched_forward(X)
                self._remove_activation_hooks()

            # We sort pruned weights in descending order of score. The prune set
            # contains (indices of) weights that have been pruned, and so when we
            # consider weights to include in selected_pruned, we want to start from the
            # largest score and work down, since the largest score weights in the pruned
            # set are most likely to be mis-labeled (i.e., most likely to have a
            # meaningful impact on model loss, and so might not want to be pruned).
            weight_scores_pruned = self._calc_weight_scores(
                self.selection_method, w_name, X, y, currently_pruned
            )
            # NOTE: np.argsort is always ascending, so need to flip
            w_idx_pruned = np.argsort(weight_scores_pruned, axis=0)[::-1]
            # Convert from indices into currently_pruned to indices into weights
            w_idx_pruned = [currently_pruned[index] for index in w_idx_pruned]

            # For the keep set, we want to consider smallest score first, since those
            # weights are most likely to be mis-labeled - could be pruned, but currently
            # are not pruned. Therefore, we sort in ascending order of score.
            weight_scores_kept = self._calc_weight_scores(
                self.selection_method, w_name, X, y, currently_kept
            )
            w_idx_kept = np.argsort(weight_scores_kept, axis=0)
            # Convert from indices into currently_kept to indices into weights
            w_idx_kept = [currently_kept[index] for index in w_idx_kept]

        # Select the indices for pruned/kept
        selected_pruned = self._select_indices(
            self.k, w_idx_pruned, w_dict["tabu_list"]
        )
        selected_kept = self._select_indices(
            self.block_size - self.k, w_idx_kept, w_dict["tabu_list"]
        )

        return selected_pruned, selected_kept

    def _select_indices(self, num_select, indices_sorted, tabu_list):
        """Selects num_select indices from indices_sorted given tabu."""
        if tabu_list is not None:
            # Cast to set so that membership check is faster below
            tabu_set = set(tabu_list)
        else:
            tabu_set = set()

        # Use takewhile() to avoid iterating over all the candidates - just pull out
        # the first num_select valid candidates and stop
        # Note: selected might be shorter than num_select if there are fewer candidates
        c = count()
        selected = list(
            takewhile(
                lambda index: next(c) < num_select,
                (index for index in indices_sorted if index not in tabu_set),
            )
        )

        return selected

    def _calc_grad_block(self, X, y, w_name, indices_block):
        """Returns/calculates grad_block for indices_block (caches results)."""
        # Below we try to reuse grad_block, grad, grad_sample_block, or grad_sample,
        # only resorting to recalculation if there is no choice.
        if self._grad_block is not None:
            pass  # Nothing to do - grad block already calculated

        elif self._grad is not None:
            self._grad_block = self._grad[indices_block]

        elif self._grad_sample_block is not None:
            self._grad_block = self._grad_sample_block.mean(axis=0)

        elif self._grad_sample is not None:
            self._grad_sample_block = self._grad_sample[:, indices_block]
            self._grad_block = self._grad_sample_block.mean(axis=0)

        else:
            # Forced to recalculate:
            self._grad_block = self._recalc_grad_block(X, y, w_name, indices_block)

        return self._grad_block

    def _recalc_grad_block(self, X, y, w_name, indices_block):
        """Calculates grad_block for indices_block."""
        if is_llm(self.model):
            calc_grad = calc_grad_autograd_llm
        else:
            calc_grad = calc_grad_autograd

        with torch.enable_grad():
            grad_block = calc_grad(
                self.model,
                self.loss_function,
                X,
                y,
                w_name,
                self._model_device,
                self.max_batch_size,
                indices_block,
            )

        return grad_block

    def _calc_grad_sample_block(self, X, y, w_name, indices_block):
        """Returns/calculates grad_sample_block for indices_block (caches results)."""
        if self.verbose >= 2:
            if self._cuda_is_available:
                torch.cuda.synchronize()
            grad_sample_start = time.perf_counter()

        # Below we try to reuse grad_sample_block, only resorting to recalculation if
        # there is no choice.
        if self._grad_sample_block is not None:
            pass  # Nothing to do - per-sample grad block already calculated

        elif self._grad_sample is not None:
            self._grad_sample_block = self._grad_sample[:, indices_block]

        else:
            # Forced to recalculate:
            self._grad_sample_block = self._recalc_grad_sample_block(
                X, y, w_name, indices_block
            )

        if self.verbose >= 2:
            if self._cuda_is_available:
                torch.cuda.synchronize()
            grad_sample_time = time.perf_counter() - grad_sample_start
            print(f"    _calc_grad_sample_block() took: {grad_sample_time:.2f}s")

        return self._grad_sample_block

    def _recalc_grad_sample_block(self, X, y, w_name, indices_block):
        """Calculates grad_sample_block for indices_block."""
        if is_llm(self.model):
            calc_grad_sample = calc_grad_sample_autograd_llm
        else:
            calc_grad_sample = calc_grad_sample_functional

        # torch.enable_grad(), but here we don't. We avoid introducing it here or
        # removing it there so that we don't cause issues, for now.
        grad_sample_block = calc_grad_sample(
            self.model,
            self.loss_function,
            X,
            y,
            w_name,
            self._model_device,
            self.max_batch_size,
            indices_block,
        )
        return grad_sample_block

    def _calc_hess_block(self, X, y, w_name, indices_block):
        """Returns/calculates hess_block for indices_block (caches results)."""
        if self.verbose >= 2:
            if self._cuda_is_available:
                torch.cuda.synchronize()
            hess_start = time.perf_counter()

        # Below we try to reuse hess_block or hess, only resorting to recalculation if
        # there is no choice.
        if self._hess_block is not None:
            pass  # Nothing to do - hess block already calculated

        elif self._hess is not None:
            self._hess_block = self._hess[indices_block]

        elif self.calc_hessian_method == "gradient_per_sample":
            grad_sample_block = self._calc_grad_sample_block(
                X, y, w_name, indices_block
            )
            # The gradients are often tiny. For larger models, we use float16, which has
            # a limited range. If we multiply two matrices that are float16 with small
            # numbers in them, we risk significant underflowing occurring. Since
            # grad_sample_block is on the CPU anyway, we cast it to float32, mitigating
            # the underflowing issues in the product.
            grad_sample_block = grad_sample_block.to(torch.float32)
            self._hess_block = (grad_sample_block.T @ grad_sample_block) / len(X)

        elif self.calc_hessian_method == "exact":
            self._hess_block = calc_hessian(
                self.model,
                self.loss_function,
                X,
                y,
                w_name,
                indices_block,
                self._model_device,
                self.max_batch_size,
            )

        elif self.calc_hessian_method == "exact_piecewise":
            self._hess_block = calc_hessian_piecewise(
                self.model,
                self.loss_function,
                X,
                y,
                w_name,
                indices_block,
                self._model_device,
                self.max_batch_size,
                chunk_size=1000,
            )

        elif self.calc_hessian_method == "gradient_mean":
            grad_block = self._calc_grad_block(X, y, w_name, indices_block)
            self._hess_block = torch.outer(grad_block, grad_block) / len(X)

        else:
            raise ValueError(f"Invalid calc_hessian_method={self.calc_hessian_method}")

        if self.verbose >= 2:
            if self._cuda_is_available:
                torch.cuda.synchronize()
            hess_time = time.perf_counter() - hess_start
            print(f"    _calc_hess_block() took: {hess_time:.2f}s")

        return self._hess_block
