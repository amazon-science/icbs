# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch

from .util import bitarray_from_list, is_flat


class State:
    """State keeps track of which weights should be pruned or kept."""

    def __init__(self, n, to_prune, to_keep):
        if not isinstance(to_prune, list) or not isinstance(to_keep, list):
            raise TypeError("to_prune and to_keep should be lists")

        self.n = n
        self._to_prune = bitarray_from_list(to_prune, n)
        self._to_keep = bitarray_from_list(to_keep, n)

        if any(self._to_prune & self._to_keep):
            raise ValueError("The intersection of to_prune and to_keep should be empty")

    @property
    def to_prune(self):
        return [i for i, val in enumerate(self._to_prune) if val]

    @to_prune.setter
    def to_prune(self, value):
        raise AttributeError("to_prune is read-only")

    @property
    def to_keep(self):
        return [i for i, val in enumerate(self._to_keep) if val]

    @to_keep.setter
    def to_keep(self, value):
        raise AttributeError("to_keep is read-only")

    def apply_move(self, move):
        """Applies move to this state."""
        self._to_prune[move.to_prune] = 1
        self._to_keep[move.to_keep] = 1

        self._to_prune[move.to_keep] = 0
        self._to_keep[move.to_prune] = 0

    def apply_to_weights(self, weights, weights_original):
        """Applies this state to weights, using weights_original.

        Note: the assumption is that weights and weights_original are flat tensors.
        Usually it is better to pass in a view (rather than a copy).
        """
        if not is_flat(weights) or not is_flat(weights_original):
            raise ValueError("weights and weights_original must be of dimension 1")

        with torch.inference_mode():
            # When a GPU is available, weights_original will be on the CPU while weights
            # will be on the GPU. We need to make sure that weights_original is on the
            # same device as weights so that the below will work.
            if weights_original.device != weights.device:
                weights_original = weights_original.to(weights.device)
                made_copy = True
            else:
                made_copy = False

            weights[self.to_prune] = 0
            weights[self.to_keep] = weights_original[self.to_keep]

            if made_copy:
                # Clean up the copy we made.
                del weights_original

    def __repr__(self):
        return "to_prune=" + repr(self._to_prune) + " to_keep=" + repr(self._to_keep)

    def __str__(self):
        return "to_prune=" + str(self.to_prune) + " to_keep=" + str(self.to_keep)

    def __eq__(self, other):
        return self._to_prune == other._to_prune and self._to_keep == other._to_keep
