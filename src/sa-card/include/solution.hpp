// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cstring>
#include <limits>
#include <sstream>

// Represents a solution to an n-variable maximization problem.
// The solution consists of
// 1. Assignment (best_solution)
// 2. Objective value (best_solution_value)
// 3. Best dual bound (best_bound)
template <class AssignmentType, class ValueType>
struct Solution {
  using assignment_t = AssignmentType;
  using value_t = ValueType;

  Solution(std::size_t n)
      : n(n),
        best_solution(0),
        best_solution_value(std::numeric_limits<value_t>::lowest()),
        best_bound(std::numeric_limits<value_t>::max()) {}
  std::size_t n;
  assignment_t best_solution;
  value_t best_solution_value;
  value_t best_bound;
};
