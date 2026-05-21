#pragma once

#include <functional>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

// The contract between objective builders and optimizers.
//
// Objective builders (`gmm::residuals`, `gmm::gp`, `estimate::ml_objective`) package a
// model + sample into one of two plain structs of closures; optimizers
// (`optim::lbfgs`, `optim::ceres_lm`) consume them. No templates, no concepts —
// the closures are `std::function`, so the optimizer never sees the builder's
// type. Composition happens at the call site.

namespace magmaan::optim {

// x ↦ whitened residual r̃(x); F = ½‖r̃‖². May fail (non-PD Σ at x, etc.).
using ResidualFn  = std::function<fit_expected<Eigen::VectorXd>(const Eigen::VectorXd&)>;
// x ↦ whitened Jacobian J̃(x) = ∂r̃/∂x.
using JacobianFn  = std::function<fit_expected<Eigen::MatrixXd>(const Eigen::VectorXd&)>;
// x ↦ whitened residual and Jacobian at the same point. Optional fast path for
// builders that can compute both from one model evaluation.
struct LsEvaluation {
  Eigen::VectorXd residual;
  Eigen::MatrixXd jacobian;
};
using LsEvaluationFn = std::function<fit_expected<LsEvaluation>(const Eigen::VectorXd&)>;
// (x, grad_out) ↦ F(x), writing ∇F into grad_out. +inf signals "x invalid".
using ObjectiveFn = std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)>;
// x ↦ full θ. The optimizer drives `x` (which may be a reduced parameter —
// profiled β, or constraint-reduced α); `expand` recovers the full vector.
using ExpandFn    = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

// LS-shape callback aliases — the single-argument residual / Jacobian closures
// the bounded optimizers' `minimize_ls` overloads take. Identical to
// `ResidualFn` / `JacobianFn`; kept under their own names for those signatures.
using LsResidualFn = ResidualFn;
using LsJacobianFn = JacobianFn;

// Least-squares-shaped problem: F(x) = ½‖r(x)‖², with the full Jacobian J(x)
// available so Levenberg–Marquardt sees the true Gauss–Newton structure.
// `n_param` is the dimension the optimizer drives (≤ dim θ when profiled or
// constraint-reduced); `expand` maps the solution back to full θ.
struct GmmProblem {
  ResidualFn   r;
  JacobianFn   J;
  LsEvaluationFn eval;
  Eigen::Index n_resid = 0;
  Eigen::Index n_param = 0;
  ExpandFn     expand;
};

// Scalar-shaped problem: F(x) and ∇F(x) only — no sum-of-squares structure.
// `estimate::ml_objective` produces this; `optim::scalarize` adapts a GmmProblem.
struct ScalarProblem {
  ObjectiveFn  f;
  Eigen::Index n_param = 0;
  ExpandFn     expand;
};

// Optimizer termination status. The error path (`fit_expected`'s unexpected
// branch) already carries hard failures — max-iter, non-finite objective,
// unrecoverable line-search abort. This enum refines the *success* path so a
// caller can tell a clean stationary stop from a salvaged or singular one.
enum class OptimStatus {
  Converged,            // clean stop: (projected) gradient norm under tolerance
  LineSearchSalvaged,   // line search aborted, but the iterate was stationary
  SingularConvergence,  // PORT singular convergence (IV(1)=7): Hessian singular
  Unknown,              // backend did not report a refined status
};

// Optimizer output. `x` is in the driven parameter space; the caller applies
// the problem's `expand` to recover full θ.
struct OptimResult {
  Eigen::VectorXd x;
  double          fmin       = 0.0;
  int             iterations = 0;
  int             f_evals    = 0;
  int             g_evals    = 0;
  // Refined success status and the (projected) gradient infinity-norm at the
  // returned point. `grad_inf_norm < 0` means the backend did not compute it.
  OptimStatus     status        = OptimStatus::Converged;
  double          grad_inf_norm = -1.0;
};

}  // namespace magmaan::optim
