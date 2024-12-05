# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import pytest
import torch
from icbs.move import Move
from icbs.state import State


def test_init_type_raises():
    with pytest.raises(TypeError):
        State(5, {1, 2}, [3, 4])


def test_init_intersection_raises():
    with pytest.raises(ValueError):
        State(3, [1, 2], [0, 2])


def test_apply_move():
    state = State(6, [1, 2], [3, 4])
    state.apply_move(Move([1, 2], [3, 4]))
    assert state == State(6, [1, 2], [3, 4])

    state.apply_move(Move([1], [3, 5]))
    assert state == State(6, [1, 2], [3, 4, 5])


def test_apply_to_weights():
    state = State(4, [0, 1], [2, 3])
    weights = torch.Tensor([0, 4, 7, 0])
    weights_original = torch.Tensor([-5, 4, 7, -8])

    state.apply_to_weights(weights, weights_original)
    assert torch.equal(weights, torch.Tensor([0, 0, 7, -8]))


def test_set_to_keep_raise():
    state = State(5, [0, 1], [2, 3])
    with pytest.raises(AttributeError):
        state.to_keep = [4]


def test_set_to_prune_raise():
    state = State(5, [0, 1], [2, 3])
    with pytest.raises(AttributeError):
        state.to_prune = [4]


def test_eq():
    state = State(5, [0, 1], [2, 3])
    assert state == state

    state2 = State(4, [0, 1], [2, 3])
    assert state2 != state

    state3 = State(5, [2, 3], [0, 1])
    assert state3 == state3
    assert state3 != state
