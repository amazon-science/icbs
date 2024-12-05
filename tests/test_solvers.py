# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import pytest
import torch
from icbs.problem import Problem
from icbs.solvers import ConstrainedSASolver, ExhaustiveSolver

# Define fixtures for the solvers with their parameters, so that they can be reused


@pytest.fixture
def exhaustive_solver():
    return ExhaustiveSolver()


@pytest.fixture
def constrained_sa_solver():
    return ConstrainedSASolver()


@pytest.fixture
def get_problem():
    def _get_problem(k):
        Q = torch.Tensor([[2, 1], [1, 1]])
        problem = Problem(Q, k)

        return problem

    return _get_problem


@pytest.mark.parametrize(
    "solver_fixture", ["exhaustive_solver", "constrained_sa_solver"]
)
def test_solver_trivial1(get_problem, solver_fixture, request):
    # There is only one solution with cardinality 2
    solver = request.getfixturevalue(solver_fixture)
    best_x, best_E = solver.solve(get_problem(2))
    assert best_E == 5.0

    assert torch.equal(best_x, torch.tensor([[1], [1]]))


@pytest.mark.parametrize(
    "solver_fixture", ["exhaustive_solver", "constrained_sa_solver"]
)
def test_solver_trivial2(get_problem, solver_fixture, request):
    # Find the best solution with cardinality 1
    solver = request.getfixturevalue(solver_fixture)
    best_x, best_E = solver.solve(get_problem(1))
    assert best_E == 1.0
    assert torch.equal(best_x, torch.tensor([[0], [1]]))


@pytest.mark.parametrize(
    "solver_fixture", ["exhaustive_solver", "constrained_sa_solver"]
)
@pytest.mark.parametrize("seed", [4, 16, 523, 78, 1001])
def test_solver_consistency(solver_fixture, request, seed):
    # All solvers should return an energy that matches the solution returned, upon
    # re-evaluation
    solver = request.getfixturevalue(solver_fixture)

    torch.manual_seed(seed)
    n = 5
    k = 4

    Q = torch.randn(n, n)
    Q = Q + Q.T  # Need a symmetrical matrix
    problem = Problem(Q, k)

    best_x, best_E = solver.solve(problem)
    best_E_evaluated = problem.evaluate(best_x)

    # Note: we can't reduce the threshold here, presumably due to float representation
    # inaccuracies
    assert abs(best_E - best_E_evaluated) < 1e-4


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("seed", [42, 66, 314])
def test_solver_csa_vs_exhaustive(constrained_sa_solver, exhaustive_solver, k, seed):
    # We'd expect the constrained SA solver to solve these small problems to optimality.
    torch.manual_seed(seed)
    n = 5
    Q = torch.randn(n, n)
    Q = Q + Q.T  # Need a symmetrical matrix
    problem = Problem(Q, k)
    best_x, best_E = constrained_sa_solver.solve(problem)
    best_x2, best_E2 = exhaustive_solver.solve(problem)

    assert abs(best_E - best_E2) < 1e-5
    assert torch.equal(best_x, best_x2)
