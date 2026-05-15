#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate {

// Per-parameter box constraints, in the **full θ space** (size = pt.n_free()).
//
// Sentinel: `±std::numeric_limits<double>::infinity()` on a coordinate means
// "no bound on this axis". Optimizers that don't natively support `inf` bounds
// must skip those coordinates internally. An empty `Bounds` (size 0) means
// "no bounds applied anywhere".
struct Bounds {
  Eigen::VectorXd lower;     // size n_free; -inf where unbounded.
  Eigen::VectorXd upper;     // size n_free; +inf where unbounded.

  bool empty() const noexcept { return lower.size() == 0; }
};

// Auto-derive bounds from the partable. Free *variance-diagonal* parameters
// (Ψ-diagonals and Θ-diagonals — i.e. `~~` rows with `lhs == rhs`) get
// `lower = 0`, `upper = +inf`. Every other free parameter gets `(-inf, +inf)`.
//
// Multi-group invariance models share a θ index across rows via labels or
// `eq_groups`; if any of the sharing rows is a variance diagonal, the
// shared θ index gets the lower=0 bound (the bound applies to the *parameter*,
// not the row).
//
// Applies a lower-bound of 0 to any free parameter that lives on a Ψ or Θ
// diagonal (latent or residual variance). Pure preventive bound — keeps the
// optimizer out of the negative-variance region so the implied Σ stays PD.
// Returns `NumericIssue` only on a malformed partable (e.g.
// `free[i]` out of range vs `n_free()`); a clean partable always succeeds.
post_expected<Bounds>
bounds_from_partable(const spec::LatentStructure& pt);

}  // namespace magmaan::estimate
