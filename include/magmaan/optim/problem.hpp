#pragma once

#include <functional>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

// The contract between objective builders and optimizers.
//
// Objective builders (`gmm::residuals`, `gmm::gp`, `nt::ml_objective`) package a
// model + sample into one of two plain structs of closures; optimizers
// (`optim::lbfgs`, `optim::ceres_lm`) consume them. No templates, no concepts —
// the closures are `std::function`, so the optimizer never sees the builder's
// type. Composition happens at the call site.

namespace magmaan::optim {

// x ↦ whitened residual r̃(x); F = ½‖r̃‖². May fail (non-PD Σ at x, etc.).
using ResidualFn  = std::function<fit_expected<Eigen::VectorXd>(const Eigen::VectorXd&)>;
// x ↦ whitened Jacobian J̃(x) = ∂r̃/∂x.
using JacobianFn  = std::function<fit_expected<Eigen::MatrixXd>(const Eigen::VectorXd&)>;
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
  Eigen::Index n_resid = 0;
  Eigen::Index n_param = 0;
  ExpandFn     expand;
};

// Scalar-shaped problem: F(x) and ∇F(x) only — no sum-of-squares structure.
// `nt::ml_objective` produces this; `optim::scalarize` adapts a GmmProblem.
struct ScalarProblem {
  ObjectiveFn  f;
  Eigen::Index n_param = 0;
  ExpandFn     expand;
};

// Optimizer output. `x` is in the driven parameter space; the caller applies
// the problem's `expand` to recover full θ.
struct OptimResult {
  Eigen::VectorXd x;
  double          fmin       = 0.0;
  int             iterations = 0;
};

}  // namespace magmaan::optim
