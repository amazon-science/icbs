// © 2024 Amazon Web Services, Inc. or its affiliates. All Rights Reserved.
// This AWS Content is provided subject to the terms of the AWS Customer
// Agreement available at http://aws.amazon.com/agreement or other written
// agreement between Customer and either Amazon Web Services, Inc. or Amazon
// Web Services EMEA SARL or both.

#pragma once

#include <ising-instance.hpp>
#include <map>
#include <set>
#include <vector>

// Struct representing the parameters for simulated annealing.
template <class RealType = double>
struct SimulatedAnnealingParameters {
  // initial inverse temperature
  RealType initial_beta = 1.e-3;
  // factor by which beta is increased during simulated annealing
  RealType beta_factor = 1.1;
  // stop once beta is larger than or equal to final_beta (or earlier if no move
  // is accepted)
  RealType final_beta = 1.e+2;
  // number of sweeps over all spins (per temperature)
  std::uint32_t num_sweeps = 1;
  // number of restarts
  bool manual_num_restarts = false;
  std::uint32_t num_restarts = 0;
};

template <class Assignment, class ValueType, class RealType>
auto run_constrained_sa(IsingInstance<ValueType> const& inst,
                        SimulatedAnnealingParameters<RealType> const& params,
                        std::size_t min_num_ones, std::size_t max_num_ones,
                        Assignment const& init = 0) {
  static_assert(std::is_signed<ValueType>(), "ValueType must be signed.");
  using calc_type = double;
  using value_type = ValueType;

  auto const n = inst.num_vars();
  calc_type const beta0 = params.initial_beta;
  calc_type const beta_final = params.final_beta;
  calc_type const beta_fac = params.beta_factor;
  calc_type beta = beta0;
  Assignment sol = init;
  std::size_t popcnt = sol.count();
  if (popcnt < min_num_ones || popcnt > max_num_ones) {
    if (init != 0)
      throw std::runtime_error(
          "Invalid initial solution provided to constrained SA.");
    sol = 0;
    for (std::size_t i = 0; i < min_num_ones; ++i) sol[i] = 1;
    popcnt = min_num_ones;
  }
  Assignment best_sol = sol;
  value_type best_energy = inst.compute_energy(sol);
  value_type energy = best_energy;
  value_type init_energy = energy;
  Assignment init_sol = sol;
  std::size_t nrestarts = params.manual_num_restarts ? params.num_restarts : 10;
#ifdef _OPENMP
#pragma omp parallel for firstprivate(sol, energy, beta, popcnt) if (n > 10)
#endif
  for (std::size_t run = 0; run < (params.num_sweeps > 0 ? nrestarts : 0u);
       ++run) {
    std::mt19937_64 mt(run);
    std::uniform_real_distribution<calc_type> dist(0., 1.);
    sol = init_sol;
    energy = init_energy;
    popcnt = sol.count();
    auto rng = [mt, dist]() mutable { return dist(mt); };
    auto delta_energy = [&inst, &sol, n](std::size_t flip_idx) {
      value_type d = 0;
      bool new_value = !sol[flip_idx];
      for (std::size_t j = 0; j < flip_idx; ++j) {
        auto const J = inst.J(flip_idx, j);
        d += (sol[j] != new_value) ? J : -J;
      }
      for (std::size_t j = flip_idx + 1; j < n; ++j) {
        auto const J = inst.J(j, flip_idx);
        d += (sol[j] != new_value) ? J : -J;
      }
      d += new_value ? inst.h[flip_idx] : -inst.h[flip_idx];
      return 2 * d;
    };
    bool changed = true;
    std::set<std::size_t> ones, zeros;
    for (std::size_t i = 0; i < n; ++i) {
      if (sol[i])
        ones.insert(i);
      else
        zeros.insert(i);
    }
    assert(popcnt == ones.size());
    for (beta = beta0; beta < beta_final && changed; beta *= beta_fac) {
      changed = false;
      for (std::size_t sweep = 0; sweep < params.num_sweeps; ++sweep) {
        for (std::size_t i = 0; i < n; ++i) {
          // single-spin flip if cardinality constraint allows it
          if (popcnt >= min_num_ones + sol[i] &&
              popcnt + !sol[i] <= max_num_ones) {
            value_type delta = delta_energy(i);
            if (delta >= 0 || rng() < std::exp(delta * beta)) {
              energy += delta;
              if (sol[i]) {
                ones.erase(i);
                zeros.insert(i);
                popcnt--;
              } else {
                zeros.erase(i);
                ones.insert(i);
                popcnt++;
              }
              sol.flip(i);
              if (std::abs(delta) > 0) changed = true;
              value_type current_best_energy;
#ifdef _OPENMP
#pragma omp atomic read
#endif
              current_best_energy = best_energy;
              if (energy > current_best_energy) {
                // reduce errors due to finite precision
                auto energy_exact = inst.compute_energy(sol);
                energy = energy_exact;
#ifdef _OPENMP
#pragma omp critical
                {
#endif
                  if (energy > best_energy) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
                    best_energy = energy;
                    best_sol = sol;
                  }
#ifdef _OPENMP
                }
#endif
              }
            }
          } else {  // else: swap spin-up with spin-down to keep `sol` feasible
            auto it = ones.begin();
            auto set_size = ones.size();
            if (sol[i]) {
              it = zeros.begin();
              set_size = zeros.size();
            }
            if (set_size == 0) continue;
            std::advance(it, std::uniform_int_distribution<std::size_t>(
                                 0, set_size - 1)(mt));
            std::size_t const j = *it;
            value_type delta = delta_energy(i);
            sol.flip(i);
            delta += delta_energy(j);
            sol.flip(i);
            if (delta >= 0 || rng() < std::exp(delta * beta)) {
              energy += delta;
              if (sol[i]) {
                zeros.erase(j);
                zeros.insert(i);
                ones.erase(i);
                ones.insert(j);
              } else {
                zeros.erase(i);
                zeros.insert(j);
                ones.erase(j);
                ones.insert(i);
              }
              sol.flip(i);
              sol.flip(j);
              if (std::abs(delta) > 0) changed = true;
              value_type current_best_energy;
#ifdef _OPENMP
#pragma omp atomic read
#endif
              current_best_energy = best_energy;
              if (energy > current_best_energy) {
                // reduce errors due to finite precision
                auto energy_exact = inst.compute_energy(sol);
                energy = energy_exact;
#ifdef _OPENMP
#pragma omp critical
                {
#endif
                  if (energy > best_energy) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
                    best_energy = energy;
                    best_sol = sol;
                  }
#ifdef _OPENMP
                }
#endif
              }
            }
          }
        }
      }
    }
  }
  assert(popcnt == sol.count());
  assert(popcnt >= min_num_ones);
  assert(popcnt <= max_num_ones);
  return std::make_tuple(best_sol, best_energy);
}
