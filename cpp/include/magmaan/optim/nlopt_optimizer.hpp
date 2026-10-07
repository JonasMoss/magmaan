#pragma once

#include <functional>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/optim/problem.hpp"   // OptimOptions / OptimOutput

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

// Explicit controls live in OptimOptions::nlopt; see project/reference/optimizer-controls.md.
// Legacy max_iter/ftol/gtol map to maxeval/ftol_rel/xtol_rel respectively.
// In particular, legacy gtol is NOT the Luksan gradient tolerance (tolg).
// Legacy history remains unused; nlopt.vector_storage selects memory explicitly.

// NloptOptimizer — wraps an NLopt scalar minimization algorithm selected at
// construction. The single adapter parameterises over `nlopt_algorithm`
// rather than spawning one class per algorithm; the trampoline
// (`magmaan_nlopt_objective`) already handles both gradient-based and
// derivative-free algorithms via the `if (grad)` guard NLopt uses to signal
// "this algorithm doesn't want a gradient."
//
// Implements both the `Optimizer` and `BoundedOptimizer` interfaces: the
// unbounded overload delegates to the bounded one with all-±∞ bounds, which
// NLopt interprets as "no bound on this axis" (±HUGE_VAL == ±infinity).
// Some algorithms — notably BOBYQA — require *finite* bounds; calling the
// unbounded overload on those fails inside NLopt with NLOPT_INVALID_ARGS,
// surfaced as NumericIssue.
//
// Error mapping:
//   NumericIssue            — lb/ub size mismatch, or nlopt_create failure.
//   NonFiniteObjective      — final objective is NaN/Inf.
// Finite, evaluable soft exits return candidates tagged BudgetExhausted or
// LineSearchFailed (or LineSearchSalvaged after a stationary stop). Raw codes
// and resolved controls are retained in the audit; acceptance is independent.
class NloptOptimizer {
 public:
  static constexpr std::string_view name = "nlopt";

  // Default constructor preserves the historical name `nlopt-slsqp`: the
  // SLSQP gradient SQP was the first algorithm wired up, and is the one
  // most callers see. New algorithm-specific entry points pass their own
  // algorithm enum.
  NloptOptimizer(OptimOptions opts = {},
                 NloptAlgorithm algo = NloptAlgorithm::Slsqp) noexcept
      : opts_(opts), algo_(algo) {}

  OptimOptions   options()   const noexcept { return opts_; }
  NloptAlgorithm algorithm() const noexcept { return algo_; }

  using Objective = std::function<double(const Eigen::VectorXd& /*x*/,
                                         Eigen::VectorXd&       /*grad_out*/)>;

  // Bounded minimization. `lower` / `upper` must each be the same size as
  // `x0`; use `±std::numeric_limits<double>::infinity()` per coordinate to
  // mean "no bound on this axis". An infeasible `x0` is projected into the box
  // before the solve.
  fit_expected<OptimOutput>
  minimize(Objective f,
           const Eigen::VectorXd& x0,
           const Eigen::VectorXd& lower,
           const Eigen::VectorXd& upper) const;

  // Bounded scalar nonlinear program. This is currently meaningful only for
  // SLSQP: the problem must encode equality constraints by identical
  // constraint lower/upper entries. Other NLopt algorithms are rejected with a
  // NumericIssue because they do not support magmaan's nonlinear-constraint
  // contract.
  fit_expected<OptimOutput>
  minimize_constrained(const ConstrainedScalarProblem& prob,
                       const Eigen::VectorXd& x0,
                       const Eigen::VectorXd& lower,
                       const Eigen::VectorXd& upper) const;

  // Unbounded minimization — equivalent to passing all-±∞ bounds. NLopt
  // algorithms that require finite bounds (BOBYQA) will return
  // NLOPT_INVALID_ARGS, surfaced here as NumericIssue.
  fit_expected<OptimOutput>
  minimize(Objective f, const Eigen::VectorXd& x0) const;

 private:
  OptimOptions   opts_;
  NloptAlgorithm algo_;
};

}  // namespace magmaan::optim
