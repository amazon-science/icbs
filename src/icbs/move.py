# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass, field

import torch

from .util import is_flat


@dataclass
class Move:
    """Class for keeping track of pruning moves."""

    to_prune: list[int] = field(default_factory=list)
    to_keep: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.to_prune, list) or not isinstance(self.to_keep, list):
            raise TypeError("to_prune and to_keep must be lists")

    def apply_to_weights(self, weights, weights_original):
        """Applies this move to weights, using weights_original.

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
