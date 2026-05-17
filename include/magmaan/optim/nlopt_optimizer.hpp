#pragma once

#ifndef MAGMAAN_WITH_NLOPT
// Stub-only header when NLopt is disabled — including this without the build
// flag is a usage error, but we don't `#error` because the test suite may
// conditionally include this header and rely on the `#ifdef` guard around the
// declarations rather than the include itself. (Mirrors ceres_optimizer.hpp.)
namespace magmaan::optim {
struct NloptOptimizer;
}  // namespace magmaan::optim
#else

#include <functional>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"   // shared LbfgsOptions / LbfgsOutput

namespace magmaan::optim {

// NLopt's SLSQP tuning maps onto the shared `LbfgsOptions` knobs, so callers
// can swap optimizer backends without rewriting their option blocks:
//
//   max_iter ↔ nlopt_set_maxeval     (evaluation budget)
//   ftol     ↔ nlopt_set_ftol_rel    (relative objective-change stop)
//   gtol     ↔ nlopt_set_xtol_rel    (relative parameter-step stop — NLopt has
//                                     no gradient-norm criterion; the step
//                                     tolerance is the closest analogue and
//                                     the one SLSQP actually honors)
//
// `history` is unused — SLSQP carries a dense BFGS update, not a limited-memory
// history.
using NloptOptions = LbfgsOptions;

// NloptOptimizer — wraps NLopt's `NLOPT_LD_SLSQP`, a gradient-based sequential
// quadratic programming method with native box-bound support. SLSQP is a
// genuinely different algorithm class from line-search L-BFGS, which makes it
// a useful independent cross-check of fitted optima. The same NLopt dependency
// also carries `AUGLAG` (augmented Lagrangian) for future nonlinear-constraint
// support — see the forward-looking section of the optimizer plan.
//
// Implements both the `Optimizer` and `BoundedOptimizer` concepts: the
// unbounded overload delegates to the bounded one with all-±∞ bounds, which
// NLopt interprets as "no bound on this axis" (±HUGE_VAL == ±infinity).
//
// Error mapping (mirrors LbfgsBOptimizer):
//   NumericIssue            — lb/ub size mismatch, or nlopt_create failure.
//   OptimizerNonConvergence — NLopt's evaluation/time budget was exhausted.
//   LineSearchFailed        — NLOPT_FAILURE / NLOPT_FORCED_STOP.
//   NonFiniteObjective      — final objective is NaN/Inf.
class NloptOptimizer {
 public:
  static constexpr std::string_view name = "nlopt-slsqp";

  NloptOptimizer(NloptOptions opts = {}) noexcept : opts_(opts) {}

  NloptOptions options() const noexcept { return opts_; }

  using Objective = std::function<double(const Eigen::VectorXd& /*x*/,
                                         Eigen::VectorXd&       /*grad_out*/)>;

  // Bounded minimization. `lower` / `upper` must each be the same size as
  // `x0`; use `±std::numeric_limits<double>::infinity()` per coordinate to
  // mean "no bound on this axis". An infeasible `x0` is projected into the box
  // before the solve.
  fit_expected<LbfgsOutput>
  minimize(Objective f,
           const Eigen::VectorXd& x0,
           const Eigen::VectorXd& lower,
           const Eigen::VectorXd& upper) const;

  // Unbounded minimization — equivalent to passing all-±∞ bounds.
  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0) const;

 private:
  NloptOptions opts_;
};

}  // namespace magmaan::optim

#endif  // MAGMAAN_WITH_NLOPT
