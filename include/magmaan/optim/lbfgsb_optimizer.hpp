#pragma once

#include <functional>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/optim/concepts.hpp"           // LsResidualFn / LsJacobianFn
#include "magmaan/optim/lbfgs_optimizer.hpp"    // LbfgsOutput shared; reuse LbfgsOptions

namespace magmaan::optim {

// Options for L-BFGS-B. We reuse `LbfgsOptions` so callers can swap
// `LbfgsOptimizer` / `LbfgsBOptimizer` without rewriting their option
// blocks; the field semantics line up cleanly:
//
//   max_iter ↔ LBFGSBParam::max_iterations
//   ftol     ↔ LBFGSBParam::delta   (paired with past = 1)
//   gtol     ↔ LBFGSBParam::epsilon (absolute projected-gradient tol)
//   history  ↔ LBFGSBParam::m
//
// One semantic shift vs `LbfgsOptimizer`: the unconstrained adapter
// disables `delta`/`past` and relies only on the gradient norm. With
// bounds, ‖Pg‖∞ legitimately plateaus once the active set is correct
// (the gradient is non-zero but its projection onto the feasible cone
// is); function-value stalling is the more useful "done" signal then,
// so we leave delta-based convergence enabled at `past = 1, delta = ftol`.
using LbfgsBOptions = LbfgsOptions;

// LBFGS-B adapter (yixuan/LBFGSpp's `LBFGSBSolver<double>`). Same
// objective contract as `LbfgsOptimizer`: callable takes `(x, grad_out)`
// and returns f(x), writing ∇f into grad_out.
//
// Implements **both** the Optimizer and BoundedOptimizer concepts. The
// unbounded overload delegates to the bounded overload with
// `lb = -∞, ub = +∞` — `LBFGSBSolver`'s projection math (`max_step_size`,
// `proj_grad_norm`) handles infinite bounds without producing NaNs, so
// the unbounded path is mathematically the same algorithm with no
// active set.
//
// Initial-point contract: an infeasible `x0` is **projected** into
// `[lb, ub]` before the solver runs. LBFGSpp does not auto-project and
// would otherwise compute a meaningless projected-gradient norm from an
// out-of-box start. Callers from `fit_bounded` already supply feasible
// starts via `simple_start_values`; the projection is defensive.
//
// Error mapping (mirrors `LbfgsOptimizer`):
//   NumericIssue          — lb / ub size mismatch (surfaced before LBFGSpp).
//   LineSearchFailed      — solver throws (line-search exhaustion or
//                           bad params); we catch and convert.
//   NonFiniteObjective    — final fval is NaN/Inf.
//   OptimizerNonConvergence — n_iter >= max_iter.
class LbfgsBOptimizer {
 public:
  static constexpr std::string_view name = "lbfgsb";

  LbfgsBOptimizer(LbfgsBOptions opts = {}) noexcept : opts_(opts) {}

  using Objective = std::function<double(const Eigen::VectorXd& /*x*/,
                                         Eigen::VectorXd&       /*grad_out*/)>;

  // Bounded minimization. `lower` / `upper` must each be the same size
  // as `x0`; use `±std::numeric_limits<double>::infinity()` per
  // coordinate to mean "no bound on this axis".
  fit_expected<LbfgsOutput>
  minimize(Objective f,
           const Eigen::VectorXd& x0,
           const Eigen::VectorXd& lower,
           const Eigen::VectorXd& upper) const;

  // Unbounded minimization — equivalent to passing all-±∞ bounds.
  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0) const;

  // LS-shape bounded overload — satisfies `LsBoundedOptimizer<>`. LBFGS-B
  // doesn't exploit LS structure on its own; the adapter folds `r` and `J_r`
  // into a scalar objective `f = ½‖r‖², g = J_rᵀ·r` and forwards to the
  // bounded scalar `minimize`. Returned `fmin` is exactly that ½‖r‖² value
  // so the result shape matches the LS-aware path on Ceres.
  fit_expected<LbfgsOutput>
  minimize_ls(LsResidualFn r_fn, LsJacobianFn J_fn,
              Eigen::Index n_resid,
              const Eigen::VectorXd& x0,
              const Eigen::VectorXd& lower,
              const Eigen::VectorXd& upper) const;

 private:
  LbfgsBOptions opts_;
};

}  // namespace magmaan::optim
