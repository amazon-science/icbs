# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from pathlib import Path

import pytest
import torch
from icbs.problem import Problem


@pytest.mark.parametrize("invalid_value", [-1, 4])
def test_init_invalid_raise(invalid_value):
    with pytest.raises(ValueError):
        Problem(torch.zeros((3, 3)), invalid_value)


def test_construct_cardinality_penalty_term():
    problem = Problem(torch.zeros((5, 5)), 3)
    Q_constraint, const = problem.construct_cardinality_penalty_term()

    eval_constraint = lambda x: x.T @ Q_constraint @ x + const

    # Feasible
    x = torch.Tensor([[0, 1, 1, 0, 1]]).T
    assert eval_constraint(x) == 0

    # Infeasible, one off
    xs = [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1]]
    for x in xs:
        x = torch.Tensor([x]).T
        assert eval_constraint(x) == 1

    # Infeasible, two off
    xs = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]]
    for x in xs:
        x = torch.Tensor([x]).T
        assert eval_constraint(x) == 4


def test_is_feasible():
    problem = Problem(torch.zeros((5, 5)), 3)

    # Feasible
    xs = [[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [1, 1, 0, 0, 1]]
    for x in xs:
        x = torch.Tensor([x]).T
        assert problem.is_feasible(x)

    # Infeasible
    xs = [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 1]]
    for x in xs:
        x = torch.Tensor([x]).T
        assert not problem.is_feasible(x)


def test_evaluate():
    Q = torch.Tensor([[2, 1], [3, 2]])
    problem = Problem(Q, 2)
    x = torch.Tensor([[1, 1]]).T
    assert problem.evaluate(x) == 8


def test_evaluate_raise():
    Q = torch.Tensor([[2, 1], [3, 2]])
    problem = Problem(Q, 2)
    x = torch.Tensor([[0, 1]]).T
    with pytest.raises(ValueError):
        problem.evaluate(x)


def test_to_qubo():
    Q = torch.Tensor([[2, 1], [1, 1]])
    problem = Problem(Q, 2)
    qubo, const = problem.to_qubo(penalty=10)
    x = torch.Tensor([[1, 1]]).T
    assert x.T @ qubo @ x + const == 5
    print(const)


@pytest.mark.parametrize("scaling", [1.0, 100.0])
def test_to_ubqp_file(scaling):
    temp_filename = Path(".temp_test_1234xcv.ubqp")
    Q = torch.Tensor([[2, 1, -3], [1, 0, 0], [-3, 0, 6.2]])
    problem = Problem(Q, 2)
    _ = problem.to_ubqp_file(penalty=0, filename=temp_filename, scaling=scaling)

    with open(temp_filename, "rt") as f:
        assert f.readline() == "3 4\n"
        assert f.readline() == f"1 1 {2.0*scaling}\n"
        assert f.readline() == f"1 2 {1.0*scaling}\n"
        assert f.readline() == f"1 3 {-3.0*scaling}\n"
        assert f.readline() == f"3 3 {6.199999809265137*scaling}\n"

    temp_filename.unlink()


def test_to_ising():
    Q_ = torch.Tensor([[2, 1], [1, 1]])
    problem = Problem(Q_, 2)
    Q, Q_const = problem.to_qubo(penalty=10)
    J, h, const = problem.to_ising(penalty=10)

    # Evaluate all possible spin / bit states in the respective Ising / QUBO - the
    # result should always be the same
    for s in [[-1, 1], [1, -1], [1, 1], [-1, -1]]:
        s = torch.Tensor([s]).T
        x = (1 + s) / 2
        E_qubo = x.T @ Q @ x + Q_const
        E_ising = s.T @ J @ s + s.T @ h + const
        assert E_qubo == E_ising
