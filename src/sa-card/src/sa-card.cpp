// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <bitset>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ising-solver-heuristics.hpp>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout
        << "Missing arguments:\n"
        << argv[0]
        << " [mc | ubqp] [filename] [cardinality] {initial_beta} {final_beta} "
           "{beta_factor} {num_sweeps} {num_restarts}\n";
    return 0;
  }
  bool verbose = true;
  using value_t = double;
  std::size_t constexpr N = 8192;
  using assignment_t = std::bitset<N>;
  std::map<std::string, std::size_t> instances;
  instances["mc"] = 0;
  instances["ubqp"] = 1;
  std::string str_type = argv[1], filename("");
  if (instances.count(str_type) == 0) {
    std::cout << "Not a valid instance type: " << str_type << std::endl;
    return 0;
  }
  std::size_t inst_id = instances[str_type];
  filename = argv[2];
  if (verbose)
    std::cout << "# Reading an instance from " << filename << std::endl;

  std::size_t k = std::stoi(argv[3]);

  SimulatedAnnealingParameters params;
  params.initial_beta = argc > 4 ? std::stof(argv[4]) : 1.e-3;
  params.final_beta = argc > 5 ? std::stof(argv[5]) : 1.e+2;
  params.beta_factor = argc > 6 ? std::stof(argv[6]) : 1.1;
  params.num_sweeps = argc > 7 ? std::stoi(argv[7]) : 1;
  params.num_restarts = argc > 8 ? std::stoi(argv[8]) : 1;
  params.manual_num_restarts = true;

  IsingInstance<value_t> inst(0);
  // energy offset and factor for conversion between different problem types
  value_t energy_offset = 0;
  double overall_factor = 1.;
  if (inst_id == 0) {
    auto res = read_mc<value_t>(filename);
    inst = std::get<0>(res);
    energy_offset = std::get<1>(res);
    if (verbose)
      std::cout << "# Energy offset to consider: " << energy_offset
                << std::endl;
    overall_factor = 0.5;
  } else if (inst_id == 1) {
    auto res = read_ubqp<value_t>(filename);
    inst = std::get<0>(res);
    energy_offset = std::get<1>(res);
    if (verbose)
      std::cout << "# Energy offset to consider: " << energy_offset
                << std::endl;
    overall_factor = -0.5;
  }
  std::cout << std::fixed << std::setprecision(2) << "n: " << inst.num_vars()
            << std::flush;
  auto t1 = std::chrono::high_resolution_clock::now();
  // Min and max number of ones are both set to k
  auto [best_solution, best_value] =
      run_constrained_sa<assignment_t>(inst, params, k, k);
  auto t2 = std::chrono::high_resolution_clock::now();
  double delta =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          .count();
  std::cout << std::fixed << std::setprecision(8) << " t: " << delta
            << " bestsol: " << overall_factor * (best_value + energy_offset)
            << " offset: " << energy_offset << std::endl;
  std::cout << "sol: [";
  for (std::size_t i = 0; i < inst.num_vars(); ++i) {
    std::cout << best_solution[i];
    if (i < inst.num_vars() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  return 0;
}
