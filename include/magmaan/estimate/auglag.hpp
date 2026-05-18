#pragma once

#include <cstdint>
#include <functional>

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"        // fit_expected, LbfgsOptions, Bounds
#include "magmaan/optim/problem.hpp"

namespace magmaan::estimate {

// Tuning for the augmented-Lagrangian outer loop.
struct AugLagOptions {
  int    max_outer  = 40;     // outer iterations before giving up
  double rho0       = 10.0;   // initial penalty weight
  double rho_grow   = 10.0;   // ρ multiplier when the violation stalls
  double rho_max    = 1e10;   // penalty ceiling
  double feas_tol   = 1e-9;   // ‖h(x)‖∞ at which the loop stops early
  double shrink     = 0.25;   // raise ρ unless ‖h‖ shrank by ≥ this factor
  double accept_tol = 1e-6;   // ‖h(x̂)‖∞ above which the system is infeasible
};

struct AugLagResult {
  Eigen::VectorXd x;                          // solution in the driven space
  double          base_fmin = 0.0;            // F(x̂) — the un-penalized objective
  int             outer_iterations = 0;
  double          constraint_violation = 0.0; // ‖h(x̂)‖∞
};

// Constraint residual h(x) and Jacobian ∂h/∂x, evaluated in the optimizer's
// driven space.
using ConstraintFn    = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
using ConstraintJacFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

// Minimize the scalar problem `base` subject to `m` nonlinear equality
// constraints h(x) == 0, by a Hestenes-Powell-Rockafellar augmented
// Lagrangian. Each outer iteration minimizes
//
//   L_ρ(x) = F(x) + λᵀh(x) + (ρ/2)‖h(x)‖²
//
// over x with L-BFGS (honoring `bounds`), then updates λ ← λ + ρ·h(x) and
// raises ρ when the constraint violation fails to shrink. The reported
// `base_fmin` is the un-penalized F(x̂) — the discrepancy / chi-square need it,
// not the penalized value. Errors (`FitError::Kind::NumericIssue`) when the
// outer loop cannot drive ‖h‖ below `accept_tol` (an infeasible system).
fit_expected<AugLagResult>
augmented_lagrangian(const optim::ScalarProblem& base, const ConstraintFn& h,
                     const ConstraintJacFn& jac, std::int32_t m,
                     const Eigen::VectorXd& x0, const Bounds& bounds,
                     LbfgsOptions opts, AugLagOptions al = {});

}  // namespace magmaan::estimate
