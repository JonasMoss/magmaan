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

// Algorithm selector kept opaque to downstream code so the NLopt header
// stays out of the public interface (it leaks into every TU otherwise).
// The mapping to `nlopt_algorithm` is done inside `nlopt_optimizer.cpp`,
// which is the only TU that needs <nlopt.h>.
enum class NloptAlgorithm {
  Slsqp,   // NLOPT_LD_SLSQP             — gradient SQP with bounds (Kraft 1988)
  Bobyqa,  // NLOPT_LN_BOBYQA            — derivative-free quadratic-model TR
                                         // (Powell 2009); requires finite bounds
  Tnewton, // NLOPT_LD_TNEWTON_PRECOND_RESTART
                                         //                — preconditioned
                                         // truncated Newton (Nash 1985)
  Var2,    // NLOPT_LD_VAR2              — Shanno-Phua full BFGS (1980)
  Lbfgs,   // NLOPT_LD_LBFGS             — NLopt's own L-BFGS
};

// NLopt's tuning maps onto the shared `LbfgsOptions` knobs, so callers
// can swap optimizer backends without rewriting their option blocks:
//
//   max_iter ↔ nlopt_set_maxeval     (evaluation budget)
//   ftol     ↔ nlopt_set_ftol_rel    (relative objective-change stop)
//   gtol     ↔ nlopt_set_xtol_rel    (relative parameter-step stop — NLopt has
//                                     no gradient-norm criterion; the step
//                                     tolerance is the closest analogue and
//                                     the one SLSQP actually honors)
//
// `history` is unused — none of the wrapped NLopt algorithms carry a
// limited-memory history (BOBYQA / TNEWTON have their own internal sizing,
// VAR2 is full BFGS, NLopt's LBFGS uses NLopt's internal default).
using NloptOptions = LbfgsOptions;

// NloptOptimizer — wraps an NLopt scalar minimization algorithm selected at
// construction. The single adapter parameterises over `nlopt_algorithm`
// rather than spawning one class per algorithm; the trampoline
// (`magmaan_nlopt_objective`) already handles both gradient-based and
// derivative-free algorithms via the `if (grad)` guard NLopt uses to signal
// "this algorithm doesn't want a gradient."
//
// Implements both the `Optimizer` and `BoundedOptimizer` concepts: the
// unbounded overload delegates to the bounded one with all-±∞ bounds, which
// NLopt interprets as "no bound on this axis" (±HUGE_VAL == ±infinity).
// Some algorithms — notably BOBYQA — require *finite* bounds; calling the
// unbounded overload on those fails inside NLopt with NLOPT_INVALID_ARGS,
// surfaced as NumericIssue.
//
// Error mapping (mirrors LbfgsBOptimizer):
//   NumericIssue            — lb/ub size mismatch, or nlopt_create failure.
//   OptimizerNonConvergence — NLopt's evaluation/time budget was exhausted.
//   LineSearchFailed        — NLOPT_FAILURE / NLOPT_FORCED_STOP.
//   NonFiniteObjective      — final objective is NaN/Inf.
class NloptOptimizer {
 public:
  static constexpr std::string_view name = "nlopt";

  // Default constructor preserves the historical name `nlopt-slsqp`: the
  // SLSQP gradient SQP was the first algorithm wired up, and is the one
  // most callers see. New algorithm-specific entry points pass their own
  // algorithm enum.
  NloptOptimizer(NloptOptions opts = {},
                 NloptAlgorithm algo = NloptAlgorithm::Slsqp) noexcept
      : opts_(opts), algo_(algo) {}

  NloptOptions   options()   const noexcept { return opts_; }
  NloptAlgorithm algorithm() const noexcept { return algo_; }

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

  // Unbounded minimization — equivalent to passing all-±∞ bounds. NLopt
  // algorithms that require finite bounds (BOBYQA) will return
  // NLOPT_INVALID_ARGS, surfaced here as NumericIssue.
  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0) const;

 private:
  NloptOptions   opts_;
  NloptAlgorithm algo_;
};

}  // namespace magmaan::optim

#endif  // MAGMAAN_WITH_NLOPT
