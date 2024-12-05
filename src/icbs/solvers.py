# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import datetime
import os
import subprocess
import uuid
from abc import ABC, abstractmethod
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


class Solver(ABC):
    @abstractmethod
    def solve(self, problem):
        """Solves the optimization problem."""


class ExhaustiveSolver(Solver):
    """An exhaustive solver for QUBOs with a cardinality constraint.

    Note: This solver is exhaustive and was written in Python - it is useful mostly for
    testing purposes.
    """

    def __init__(self, verbose=0):
        self.verbose = verbose

    def solve(self, problem):
        # Unpack the problem
        n = problem.n
        k = problem.k

        best_E = np.inf
        best_x = None
        if self.verbose >= 2:
            print(f"    {n=} {k=}")

        for feasible_solution_indices in combinations(np.arange(n), k):
            # Double brackets are important here so that we get a vector that is 2D,
            # so that we can transpose it correctly below, when we do x.T
            x = torch.tensor(
                [[1.0 if i in feasible_solution_indices else 0 for i in range(n)]]
            ).T

            # Calculate energy for this feasible solution
            E = problem.evaluate(x)

            # Update best seen
            if E < best_E:
                best_E = E
                # Note: it's a tuple - immutable, so keeping a reference is fine
                best_feasible_solution_indices = feasible_solution_indices
            if self.verbose >= 2:
                print(f"        {feasible_solution_indices=} {E=:.5f} {best_E=:.5f}")

        # Recreate the best_x from the best_feasible_solution_indices
        best_x = torch.tensor(
            [[1.0 if i in best_feasible_solution_indices else 0 for i in range(n)]]
        ).T

        return best_x, best_E


class ConstrainedSASolver(Solver):
    """A simulated annealing (SA) solver for QUBOs with a cardinality constraint."""

    solver_filename = Path(__file__).parent / "sa-card"

    # Set to large number to avoid additional output
    output_interval = 10**6

    def __init__(
        self,
        beta_initial=0.001,
        beta_final=100.0,
        beta_factor=1.1,
        num_sweeps=1,
        num_restarts=1,
        verbose=0,
    ):
        """A simulated annealing (SA) solver for QUBOs with a cardinality constraint.

        Notes: This solver requires the compiled executable to be in the right place.
        Please follow the installation instructions in the main README.

        Args:
            beta_initial (float, optional): Initial temperature. Defaults to 0.001.
            beta_final (float, optional): Final temperature. Defaults to 100.0.
            beta_factor (float, optional): Temperature factor. Defaults to 1.1.
            num_sweeps (int, optional): Number of sweeps per temperature. Defaults to 1.
            num_restarts (int, optional): Number of restarts. Defaults to 1.
            verbose (int, optional): Level of verbosity - higher provides more verbose
                output. Defaults to 0.
        """
        super().__init__()
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.beta_factor = beta_factor
        self.num_sweeps = num_sweeps
        self.num_restarts = num_restarts
        self.verbose = verbose

    def solve(self, problem):
        # Temporary problem filename
        time_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")
        problem_filename = Path(f".{os.getpid()}_{time_str}_{uuid.uuid4().hex}.tmp")

        if self.verbose >= 2:
            print(f"    Solving with ConstrainedSASolver, {problem_filename=}")

        # Set the penalty to 0 since the cardinality constraint is taken into account
        # natively by this solver and should not be reflected in the UBQP file.
        const = problem.to_ubqp_file(penalty=0, filename=problem_filename)

        completed_process = subprocess.run(
            [
                self.solver_filename,
                "ubqp",
                problem_filename,
                str(problem.k),
                str(self.beta_initial),
                str(self.beta_final),
                str(self.beta_factor),
                str(self.num_sweeps),
                str(self.num_restarts),
            ],
            capture_output=True,
            text=True,
        )

        if completed_process.stderr.strip() != "":
            print(f"Solver produced an error when solving {problem_filename}:")
            print(completed_process.stderr)

        # It's easier to parse the output if we split it into lines first
        output = completed_process.stdout.splitlines()

        # Extract best_E and other meta data
        meta_split = output[-2].split()
        best_E = float(meta_split[5]) + const
        n = int(meta_split[1])  # Number of variables
        assert n == problem.n

        # Print some meta data
        if self.verbose >= 1:
            elapsed_time = float(meta_split[3])

            print(f"    {n=} {best_E=:.3f} {elapsed_time=:.3f}")

        elif self.verbose >= 3:
            print("Solver output (stdout):")
            print(completed_process.stdout)

        # Extract best_x
        solution_str = output[-1].replace(" ", "")[5:-1].split(",")
        best_x = torch.Tensor([list(int(el) for el in solution_str)]).T

        # Delete the temporary problem file
        problem_filename.unlink()

        return best_x, best_E
