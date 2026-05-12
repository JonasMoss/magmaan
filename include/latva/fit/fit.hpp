#pragma once

#include <limits>
#include <utility>

#include <Eigen/Core>

#include "latva/expected.hpp"
#include "latva/fit/constraints.hpp"
#include "latva/fit/lbfgs_optimizer.hpp"
#include "latva/fit/ml.hpp"
#include "latva/fit/resolve_fixed_x.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/fit/start_values.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/model/model_evaluator.hpp"
#include "latva/partable/partable.hpp"
#include "latva/partable/start_hints.hpp"

namespace latva::fit {

// Estimation results — pure data, NO back-pointer to the LatentStructure.
// (See plan: separation of concerns. Caller composes pt + Estimates.)
struct Estimates {
  Eigen::VectorXd theta;     // size = pt.n_free()
  double          fmin       = 0.0;
  int             iterations = 0;
};

// fit() — closes the parser → lavaanify → matrix_rep → evaluator → ML →
// LBFGS loop into a single call. Returns Estimates (just θ̂ and the
// optimizer's final fmin/iterations).
//
// Template on Discrepancy and Optimizer concepts so methods developers
// can swap ML for GLS, LBFGS for Ceres, without touching the core.
//
// In v0 only ML/LbfgsOptimizer are shipped. Concept refinements land in
// later phases; for now the templates accept any class with the right
// member functions (no formal `Discrepancy`/`Optimizer` concept yet —
// adding them in P9 / extensibility cleanup).
template <class D = ML, class O = LbfgsOptimizer>
fit_expected<Estimates>
fit(partable::LatentStructure         pt,         // by value — we may resolve fixed.x
    const model::MatrixRep&    rep,
    const SampleStats&         samp,
    D discrepancy = {},
    O optimizer   = {},
    partable::Starts starts = {}) {        // user start hints (from lavaanify); {} ⇒ heuristic
  // Fill in any fixed.x rows' fixed_value from the sample covariance
  // (lavaan does this implicitly during fit).
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(e.error());
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(FitError{
        FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail,
        0, 0.0});
  }
  auto ev = std::move(*ev_or);

  auto x0_or = simple_start_values(pt, rep, samp, starts);
  if (!x0_or.has_value()) return std::unexpected(x0_or.error());

  // Linear equality constraints (shared labels, explicit `a == b`, general
  // linear `a == 2*b+c` / `Σλ == k`) are enforced by reparameterizing
  // θ = θ₀ + K·α — over the reduced α (size n_alpha ≤ npar). The unconstrained
  // case is the identity reparam (θ₀ = 0, K = I, active() == false), so it
  // stays on the zero-overhead branch below.
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(FitError{
        FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail, 0, 0.0});
  }
  const EqConstraints& con = *con_or;

  // F_ML(θ) and its θ-gradient at a full parameter vector.
  auto eval_at = [&](const Eigen::VectorXd& x,
                     Eigen::VectorXd&       grad) -> double {
    auto sm = ev.sigma(x);
    if (!sm.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    auto val = discrepancy.value(samp, *sm);
    if (!val.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    auto J = ev.dsigma_dtheta(x);
    if (!J.has_value()) {
      grad.setZero();
      return *val;
    }
    // dμ/dθ — empty MatrixXd for covariance-only models; the gradient
    // call below detects that and skips the mean-term contribution.
    auto Jmu = ev.dmu_dtheta(x);
    if (!Jmu.has_value()) {
      grad.setZero();
      return *val;
    }
    auto g = discrepancy.gradient(samp, *sm, *J, *Jmu);
    if (g.has_value()) grad = *g;
    else               grad.setZero();
    return *val;
  };

  if (!con.active()) {
    auto out_or = optimizer.minimize(eval_at, *x0_or);
    if (!out_or.has_value()) return std::unexpected(out_or.error());
    return Estimates{std::move(out_or->theta_hat), out_or->fmin,
                     out_or->iterations};
  }

  // Fully constrained (n_alpha == 0): the constraints pin θ entirely — nothing
  // to optimize. Return θ₀ and its objective value directly.
  if (con.n_alpha == 0) {
    Eigen::VectorXd theta = con.expand(Eigen::VectorXd(0));
    Eigen::VectorXd scratch(theta.size());
    const double f = eval_at(theta, scratch);
    return Estimates{std::move(theta), f, 0};
  }

  // Reparameterized: optimize over α (size n_alpha < npar); θ = θ₀ + K·α,
  // ∇_α = K'·∇_θ. Start α by projecting the simple start values onto the
  // constraint surface (per-group mean for the pure-merge case).
  Eigen::VectorXd alpha0 = con.contract(*x0_or);
  auto objective_alpha = [&](const Eigen::VectorXd& a,
                             Eigen::VectorXd&       grad_a) -> double {
    const Eigen::VectorXd x = con.expand(a);
    Eigen::VectorXd grad_x(x.size());
    const double v = eval_at(x, grad_x);
    grad_a = con.reduce_gradient(grad_x);
    return v;
  };
  auto out_or = optimizer.minimize(objective_alpha, alpha0);
  if (!out_or.has_value()) return std::unexpected(out_or.error());
  return Estimates{con.expand(out_or->theta_hat), out_or->fmin,
                   out_or->iterations};
}

}  // namespace latva::fit
