// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <solution.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

template <class RealType>
class ising_solver_output_handler {
 public:
  using real_t = RealType;
  using duration_t = std::chrono::duration<real_t>;
  using timepoint_t = std::chrono::steady_clock::time_point;

  // the constructor sets the output interval and the output target (e.g.,
  // std::cout)
  ising_solver_output_handler(real_t const& interval, std::ostream& out)
      : interval_(interval), out_(out) {
    last_output_elapsed_time_ = 0;
    start_time_ = std::chrono::steady_clock::now();
  }

  // outputs anytime solver information if the elapsed time since the last
  // output is at least the set interval (see constructor)
  template <class A, class V>
  void conditional_output(std::size_t n, Solution<A, V> const& sol) {
    auto now = std::chrono::steady_clock::now();
    real_t last_output;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    last_output = last_output_elapsed_time_;
    auto delta =
        std::chrono::duration_cast<duration_t>(now - start_time_).count();
    if (delta >= interval_ + last_output) {
#ifdef _OPENMP
#pragma omp critical(ising_solver_output_update)
      {
        last_output = last_output_elapsed_time_;
        now = std::chrono::steady_clock::now();
        delta =
            std::chrono::duration_cast<duration_t>(now - start_time_).count();
        if (delta >= interval_ + last_output) {
#endif
          output_(n, sol);
#ifdef _OPENMP
        }
      }
#endif
    }
  }

 private:
  template <class A, class V>
  void output_(std::size_t n, Solution<A, V> const& sol) {
    auto now = std::chrono::steady_clock::now();
    real_t delta =
        std::chrono::duration_cast<duration_t>(now - start_time_).count();
#ifdef _OPENMP
#pragma omp atomic write
#endif
    last_output_elapsed_time_ = delta;
    out_ << "\t[" << delta << "s @ n=" << n << "] ";
    V best_bound, best_value;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    best_bound = sol.best_bound;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    best_value = sol.best_solution_value;
    auto gap = best_bound - best_value;
    std::size_t tid = 0, nt = 1;
#ifdef _OPENMP
    tid = omp_get_thread_num();
    nt = omp_get_num_threads();
#endif
    out_ << "BestSol: " << best_value << " BestBound: " << best_bound
         << " Gap: " << gap << " (" << (gap * 100. / best_bound)
         << "%) [thread#" << 1u + tid << "/" << nt << "]" << std::endl;
  }
  real_t interval_;
  std::ostream& out_;
  timepoint_t start_time_;
  real_t last_output_elapsed_time_;
};
