#pragma once

#include <vector>

namespace latva::partable {

// User-supplied start *hints* for free parameters — the optimization-side
// counterpart to the model-side `ParTable`. Indexed by 0-based free ordinal
// (matches the θ ordering): `hint[k]` is the start value the user wrote for
// the (k+1)-th free parameter via `start(v)*x` or the `v?x` shorthand. NaN
// (or an out-of-range index — an empty `hint` is the common "no hints" case)
// means "no user hint; fall back to the heuristic".
//
// `lavaanify` produces this alongside the `ParTable` (it knows the modifiers);
// `simple_start_values` / `fit` consume it. It is deliberately NOT part of the
// `ParTable`: the partable says *what to estimate*; the start hints are *where
// the search begins*, which is an estimation concern, not a model one. A fixed
// parameter's value, by contrast, IS a model fact and lives on the ParTable
// (`fixed_value`).
struct Starts {
  std::vector<double> hint;
};

}  // namespace latva::partable
