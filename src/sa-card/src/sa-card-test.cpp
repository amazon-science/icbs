// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <bitset>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <ising-instance.hpp>
#include <ising-solver-heuristics.hpp>
#include <random>
#include <vector>

using qubo_type_list = std::tuple<double, long double, std::int64_t>;

template <std::size_t N, class ValueType>
void run_constr_sa_test() {
  using value_t = ValueType;
  using assignment_t = std::bitset<N>;
  std::size_t const num_vars_max = N;
  std::size_t seed = 0;
  for (std::size_t num_vars = 8; num_vars <= num_vars_max; num_vars += 7) {
    for (std::size_t nb_min = 0; nb_min < 8; nb_min += 3) {
      for (std::size_t nb_max = nb_min;
           nb_max < std::min(num_vars, nb_min + 10); nb_max += 3) {
        for (std::size_t num_rep = 0; num_rep < 10; ++seed, ++num_rep) {
          auto inst = urandom_dense_ising<value_t>(num_vars, seed);
          SimulatedAnnealingParameters<> sa_params;
          value_t last_energy = std::numeric_limits<value_t>::lowest();
          for (std::size_t num_restarts = 1; num_restarts < 20;
               num_restarts += 5) {
            sa_params.num_restarts = num_restarts;
            sa_params.manual_num_restarts = true;
            auto const& [sol, energy] = run_constrained_sa<assignment_t>(
                inst, sa_params, nb_min, nb_max);
            REQUIRE(last_energy <= energy);  // more restarts yield better
                                             // energy, unless there is a bug
            REQUIRE(sol.count() >= nb_min);
            REQUIRE(sol.count() <= nb_max);
            REQUIRE(std::abs(energy - inst.compute_energy(sol)) < 1.e-9);
            last_energy = energy;
          }
        }
      }
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Constrained simulated annealing test",
                        "[ising][sa][constr]", qubo_type_list) {
  std::size_t constexpr N = 100;
  run_constr_sa_test<N, TestType>();
}
