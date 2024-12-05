# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import pytest
import torch
from icbs.move import Move


def test_init_type_raises():
    with pytest.raises(TypeError):
        Move({1, 2}, [3, 4])


def test_apply_to_weights():
    move = Move([0, 1], [2, 3])
    weights = torch.Tensor([0, 4, 7, 0])
    weights_original = torch.Tensor([-5, 4, 7, -8])

    move.apply_to_weights(weights, weights_original)
    assert torch.equal(weights, torch.Tensor([0, 0, 7, -8]))
