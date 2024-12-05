// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cassert>
#include <cstring>
#include <fstream>
#include <random>
#include <tuple>
#include <vector>

// Represents an Ising instance with energy
// H(s) = - Sum_i Sum_{j < i} J_ij s_i s_j - Sum_i h_i s_i
// and s_i in {+1, -1}.
//
// Spin variables s_i are represented as Boolean variables x_i in {0,1}
// such that s_i = (-1)^x_i. This is the representation assumed by the
// `compute_energy` member function below.
template <class ValueType>
struct IsingInstance {
 public:
  using value_t = ValueType;
  using vector_t = std::vector<value_t>;
  std::vector<vector_t> Jmat;  // coupling matrix
  vector_t h;                  // fields vector
  value_t& J(std::size_t i, std::size_t j) { return Jmat[i][j]; }
  value_t const& J(std::size_t i, std::size_t j) const { return Jmat[i][j]; }
  std::size_t num_vars() const { return h.size(); }
  // construct an Ising instance with n spins
  IsingInstance(std::size_t n) : Jmat(n), h(n, 0) {
    for (std::size_t i = 0; i < n; ++i) Jmat[i].resize(n);
  }
  IsingInstance<value_t> sub_problem(std::size_t n,
                                     bool zero_field = false) const {
    assert(n <= num_vars());
    IsingInstance<value_t> sp(n);
    std::size_t N = num_vars();
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < i; ++j)
        sp.J(i, j) = sp.J(j, i) = J(N - n + i, N - n + j);
      sp.h[i] = zero_field ? value_t(0) : h[N - n + i];
    }
    return sp;
  }
  // computes the energy of a given assignment `x` with x_i in {0,1}
  // assuming a spin-to-Boolean mapping s.t. s_i = (-1)^x_i
  template <class Assignment>
  value_t compute_energy(Assignment const& x) const {
    value_t val(0);
    for (std::size_t i = 0; i < num_vars(); ++i) {
      for (std::size_t j = 0; j < i; ++j)
        val += (x[i] == x[j]) ? -J(i, j) : J(i, j);
      val += x[i] ? h[i] : -h[i];
    }
    return val;
  }
};

// Construct a dense Ising instance with couplings and fields chosen uniformly
// at random from [-100,100]. Useful for testing / benchmarking.
template <class ValueType>
IsingInstance<ValueType> urandom_dense_ising(std::size_t n,
                                             std::size_t seed = 0) {
  IsingInstance<ValueType> inst(n);
  std::mt19937_64 rnd(seed);
  std::uniform_real_distribution<double> dist(-100, 100);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j) inst.J(i, j) = dist(rnd);
    inst.h[i] = dist(rnd);
  }
  return inst;
}


// Converts an unconstrained BQP instance (given by the matrix Q) to an Ising
// instance and returns a tuple containing (1) the Ising instance and (2) the
// energy offset s.t. 0.5 * (E_Ising(x) + energy_offset) = E_BQP(x).
// Please note the extra factor of two to keep integer couplings integer during
// the conversion.
//
// The unconstrained BQP instance is
//   max -x' Q x
//   s.t. x in {0,1}^n
//
template <class ValueType>
auto ubqp_to_ising(std::size_t num_vars, std::vector<ValueType> const& Qmat) {
  using value_t = ValueType;
  IsingInstance<value_t> inst(num_vars);
  ValueType energy_offset(0);
  for (std::size_t i = 0; i < num_vars; ++i) {
    inst.h[i] = 0;
    for (std::size_t j = 0; j < i; ++j) {
      auto const Qij = Qmat[num_vars * i + j];
      inst.J(i, j) = Qij;
      inst.h[i] += -Qij;
      inst.h[j] += -Qij;
      energy_offset -= Qij;
    }
    auto const Qii = Qmat[num_vars * i + i];
    energy_offset -= Qii;
    inst.h[i] -= Qii;
  }
  return std::make_tuple(inst, energy_offset);
}

// Reads a (coupling) matrix from a file and returns the matrix (as a
// vector<value_t>) together with the number of variables as a tuple.
//
// The file format is
//
// [num_vars] [edge_count]
// [i] [j] [Q_ij]
// ....
template <class ValueType>
std::tuple<std::vector<ValueType>, std::size_t> read_matrix(
    std::string const& filename) {
  using value_t = ValueType;
  std::fstream file;
  file.open(filename, std::ios::in);
  if (!file) throw std::runtime_error("File not found.");
  std::size_t num_vars, edge_count;
  file >> num_vars;
  file >> edge_count;
  std::vector<value_t> Qmat(num_vars * num_vars);
  std::size_t idx, idx2, num_edges = 0;
  ValueType weight;
  while (file >> idx) {
    file >> idx2;
    file >> weight;
    if (idx < 1 || idx < 1 || idx > num_vars || idx2 > num_vars)
      throw std::runtime_error("Indices must be in [1, n].");
    auto const i = idx - 1;
    auto const j = idx2 - 1;
    Qmat[num_vars * i + j] = Qmat[num_vars * j + i] = weight;
    num_edges++;
  }
  if (num_edges != edge_count) std::runtime_error("File edge count incorrect.");
  file.close();
  return std::make_tuple(Qmat, num_vars);
}

// Reads an unconstrained BQP instance from a file with the format
//
// [num_vars] [edge_count]
// [i] [j] [Q_ij]
// ....
//
// such that the corresponding problem is
//   min x' Q x
//   s.t. x in {0,1}^n
//
// and returns an equivalent Ising instance and the energy offset, but
// for a maximization problem. Note: This introduces a factor of (-1), i.e.,
//   -0.5 * (E_Ising(x) + energy_offset) = E_BQP(x)
// due to the conversion between min and max.
template <class ValueType>
std::tuple<IsingInstance<ValueType>, ValueType> read_ubqp(
    std::string const& filename) {
  using value_t = ValueType;
  auto [Qmat, num_vars] = read_matrix<value_t>(filename);
  return ubqp_to_ising(num_vars, Qmat);
}

// Reads a MaxCut instance from a file with the format
// used in biqmaclib (https://biqmac.aau.at/), i.e.:
//
// [num_vars] [edge_count]
// [i] [j] [Q_ij]
// ....
//
// such that the corresponding problem is
//   max x' Q x
//   s.t. x in {-1,1}^n
//
// and returns an equivalent Ising instance and the energy offset s.t.
//   0.5 * (E_Ising(x) + energy_offset) = E_MaxCut(x)
template <class ValueType>
std::tuple<IsingInstance<ValueType>, ValueType> read_mc(
    std::string const& filename) {
  using value_t = ValueType;
  auto [Qmat, num_vars] = read_matrix<value_t>(filename);
  IsingInstance<value_t> inst(num_vars);
  ValueType energy_offset(0);
  for (std::size_t i = 0; i < num_vars; ++i) {
    inst.h[i] = 0;
    for (std::size_t j = 0; j < i; ++j) {
      auto const Qij = Qmat[num_vars * i + j];
      inst.J(i, j) = Qij;
      energy_offset += Qij;
    }
  }
  return std::make_tuple(inst, energy_offset);
}


// Reads an Ising instance from a file in the format
//
// [num_vars] [energy_offset]
// [J_21]
// ...
// [J_n1] ... [J_n(n-1)]
// [h_1] ... [h_n]
// opt-val: [optimal solution value]
//
// and returns a tuple (instance, optimal_solution_value, energy_offset)
// J_ij must be lower triangular, i.e., J_ij = 0 for j >= i.
template <class ValueType>
std::tuple<IsingInstance<ValueType>, ValueType, ValueType> read_ising(
    std::string const& filename) {
  std::fstream file;
  file.open(filename);
  if (!file) throw std::runtime_error("Reading instance failed.");
  std::size_t n;
  ValueType offset, optval;
  file >> n;
  IsingInstance<ValueType> inst(n);
  file >> offset;
  std::string zero;
  for (std::size_t i = 1; i < inst.num_vars(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      file >> inst.J(i, j);
    }
  }
  for (std::size_t i = 0; i < inst.num_vars(); ++i) {
    file >> inst.h[i];
  }
  std::string optstr;
  file >> optstr;
  if (optstr != "opt-val:")
    throw std::runtime_error("Invalid file format: Expecting 'opt-val:'.");
  file >> optval;
  file.close();
  return std::make_tuple(inst, optval, offset);
}
