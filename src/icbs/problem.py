# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch

from .util import get_square_size


class Problem:
    """A problem class for pruning problems."""

    def __init__(self, Q, k):
        """Instantiates a Problem class object for use with a solver.

        Args:
            Q (torch.Tensor): The Q matrix.
            k (int): The cardinality.
        """
        self.Q = Q
        self.k = k

        self.n = get_square_size(Q)

        if self.k > self.n or self.k < 0:
            raise ValueError(
                f"Cardinality {k=} is invalid for the number of variables {self.n}"
            )

    def to_qubo(self, penalty):
        """Returns a QUBO and constant representing this problem."""
        if penalty == 0:
            return self.Q, 0.0
        else:
            Q_constraint, const = self.construct_cardinality_penalty_term()
            return self.Q + penalty * Q_constraint, penalty * const

    def to_ising(self, penalty):
        """Returns an Ising problem and constant representing this problem."""
        Q, Q_const = self.to_qubo(penalty)

        # Take x.T Q x + Q_const, substitute x = 2(1-s), expand, and compare to
        # s.T J s + s.T h + const to find the below equations.
        J = (Q - Q.diag().diag()) / 4
        h = (Q.sum(axis=1)) / 2
        const = (Q.sum().sum() + Q.trace()) / 4 + Q_const

        return J, h, const.item()

    def to_ubqp_file(self, penalty, filename, scaling=1.0):
        """Writes the problem to filename in the UBQP format, returns the constant.

        Args:
            penalty (float): The coefficient of the penalty terms.
            filename (str): The filename to save the UBQP to.
            scaling (float, optional): Scaling factor to apply to the QUBO - the
                elements in the file will all be multiplied by this factor.
                Defaults to 1.0 (no scaling).

        Returns:
            const: The constant for this QUBO problem.
        """
        qubo, const = self.to_qubo(penalty)
        qubo = qubo.detach().numpy()
        with open(filename, "wt") as f:
            lines = []
            num_couplers = 0

            # qubo is symmetric, so we only need to loop over a triangle - each element
            # appears only once, by definition in the UBQP format.
            for i in range(self.n):
                for j in range(i, self.n):
                    if qubo[i, j] != 0:
                        # Indices for UBQP start at 1, not 0, hence the +1
                        lines.append(f"{i+1} {j+1} {qubo[i, j]*scaling}\n")

                        num_couplers += 1

            f.write(f"{self.n} {num_couplers}\n")
            f.writelines(lines)

        return const

    def construct_cardinality_penalty_term(self):
        """Constructs a cardinality constraint penalty term."""
        # We want to enforce sum(x) = k, so we add a penalty term of the form
        # (sum(x)-k)^2 = sum(xi^2) + sum_ij xi xj - 2k sum(xi) + k^2, where the sum is
        # over i \neq j. Since xi^2=xi for binary variables, we get
        # = sum_ij xi xj -2k sum(xi) + k^2, where now the sum is over ALL i and j.
        # This can also be rewritten as x.T @ (ones(n) - 2k I) @ x + k^2, hence:
        Q_constraint = torch.ones(self.n) - 2 * self.k * torch.eye(self.n)
        const = self.k**2

        return Q_constraint, const

    def evaluate(self, x):
        """Returns the objective function value for solution x.

        Args:
            x (torch.Tensor): A binary vector. Should be a 2D column vector.
        """
        if self.is_feasible(x):
            # Calculate energy for this feasible solution
            E = (x.T @ self.Q @ x).item()
        else:
            raise ValueError("x is infeasible")

        return E

    def is_feasible(self, x):
        """Returns True if x is feasible.

        Args:
            x (torch.Tensor): A binary vector. Should be a 2D column vector.
        """
        if len(x) != self.n:
            raise ValueError("x is invalid, should be of length n")

        return x.view(-1).sum().item() == self.k
