#pragma once

#include <functional>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"   // shared LbfgsOptions / LbfgsOutput

namespace magmaan::optim {

// Trust-region tuning reuses the shared `LbfgsOptions` struct for interface
// parity with the other optimizer adapters (so callers can swap backends
// without rewriting option blocks). CppNumericalSolvers' trust-region solver
// carries its own well-tuned radius-control defaults; the adapter does not
// forward these knobs to it — a cross-check needs no bespoke tuning — so the
// struct is currently accepted but unused by `minimize`.
using TrustRegionOptions = LbfgsOptions;

// TrustRegionOptimizer — wraps CppNumericalSolvers' Newton trust-region solver
// (`cppoptlib::solver::TrustRegionNewton`). lavaan fits ML with nlminb, itself
// a trust region; this gives magmaan an independent, genuinely trust-region
// cross-check of the line-search L-BFGS optima.
//
// The solver is a *Newton* method — it needs a Hessian. magmaan's objective
// contract supplies only value+gradient, so the adapter forms the Hessian by
// central finite differences of the analytic gradient (2·n gradient
// evaluations per Hessian; cheap at SEM parameter counts). This is the true
// objective Hessian — the correct curvature for a Newton step — and at
// convergence the optimum is fixed by the (analytic) gradient, so the FD
// Hessian does not degrade θ̂ accuracy, only the convergence path.
//
// Satisfies the `Optimizer` concept only: CppNumericalSolvers' trust region is
// unconstrained, so there is no bounded overload. Callers needing box bounds
// use L-BFGS-B or NLopt SLSQP instead.
//
// Error mapping:
//   NumericIssue       — empty parameter vector.
//   LineSearchFailed   — the solver threw (caught at the TU boundary).
//   NonFiniteObjective — the solver returned a non-finite point or value.
class TrustRegionOptimizer {
 public:
  static constexpr std::string_view name = "trust-region";

  TrustRegionOptimizer(TrustRegionOptions opts = {}) noexcept : opts_(opts) {}

  TrustRegionOptions options() const noexcept { return opts_; }

  using Objective = std::function<double(const Eigen::VectorXd& /*x*/,
                                         Eigen::VectorXd&       /*grad_out*/)>;

  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0) const;

 private:
  TrustRegionOptions opts_;
};

}  // namespace magmaan::optim
