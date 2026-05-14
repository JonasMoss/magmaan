#pragma once

#include <limits>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/bounds.hpp"
#include "magmaan/fit/concepts.hpp"
#include "magmaan/fit/constraints.hpp"
#include "magmaan/fit/lbfgs_optimizer.hpp"
#include "magmaan/fit/lbfgsb_optimizer.hpp"
#include "magmaan/fit/ml.hpp"
#include "magmaan/fit/resolve_fixed_x.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/fit/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/partable/partable.hpp"
#include "magmaan/partable/start_hints.hpp"

namespace magmaan::fit {

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
// can swap ML for GLS, LBFGS for Ceres, without touching the core. Both `D`
// and `O` are concept-constrained (magmaan/fit/concepts.hpp):
//   • Discrepancy<D> — value(s, m) → fit_expected<double>;
//                      gradient(s, m, J, Jmu) → fit_expected<VectorXd>.
//   • Optimizer<O>  — minimize(f, x0) → fit_expected<LbfgsOutput>.
// For bounds support see `fit_bounded` below, which constrains `O` on
// `BoundedOptimizer` instead.
template <Discrepancy D = ML, Optimizer O = LbfgsOptimizer>
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

  auto ml_cache_or = [&]() {
    if constexpr (std::same_as<std::remove_cvref_t<D>, ML>) {
      return discrepancy.prepare(samp);
    } else {
      return fit_expected<int>(0);
    }
  }();
  if (!ml_cache_or.has_value()) return std::unexpected(ml_cache_or.error());

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
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }

    if constexpr (std::same_as<std::remove_cvref_t<D>, ML>) {
      auto vg = discrepancy.value_gradient(samp, *ml_cache_or, eval->moments,
                                           eval->J_sigma, eval->J_mu);
      if (!vg.has_value()) {
        grad.setZero();
        return std::numeric_limits<double>::infinity();
      }
      grad = std::move(vg->gradient);
      return vg->value;
    } else {
      auto val = discrepancy.value(samp, eval->moments);
      if (!val.has_value()) {
        grad.setZero();
        return std::numeric_limits<double>::infinity();
      }
      auto g = discrepancy.gradient(samp, eval->moments,
                                    eval->J_sigma, eval->J_mu);
      if (g.has_value()) grad = *g;
      else               grad.setZero();
      return *val;
    }
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

// fit_bounded() — parallel entry point for **bounded** optimization.
//
// Same end-to-end pipeline as `fit<>` (parser → lavaanify → matrix_rep →
// evaluator → discrepancy → optimizer), but the optimizer call goes through
// the `BoundedOptimizer::minimize(f, x0, lower, upper)` overload. Bounds are
// auto-derived from the partable via `bounds_from_partable` if `bounds.empty()`
// (the typical caller path); pass an explicit `Bounds` to override.
//
// Two paths at compile time, picked by concept refinement:
//   • `LsDiscrepancy<D>` + `LsBoundedOptimizer<O>`  → `minimize_ls(...)` with a
//     true multi-residual cost function. ULS / GLS / WLS go through here.
//   • Otherwise                                     → scalar `minimize(...)`
//     with the per-backend wrapping (Ceres' sqrt-trick, LBFGS-B native).
//
// Equality constraints (shared labels, `a == b`, multi-group invariance) are
// supported on the LS path only and are folded in as **penalty residuals**:
// the LS problem is augmented with `√μ · (A_eq·θ − b_eq)` rows, where μ is
// chosen so the residual norm is ≲ `1e-5` at convergence (post-solve checked
// against `tol_eq` and surfaced as `NumericIssue` if exceeded). This keeps
// native per-axis bounds on θ intact and trades hard equality for a uniform
// LS mechanism — for SEM constraints (mostly axis-aligned) the violation is
// effectively at machine precision. The scalar path still errors cleanly
// with `NumericIssue` when constraints are active (ML + bounds + eq is a
// future tranche; ML rarely needs bounds since PD-Σ is a natural barrier).
template <Discrepancy D = ML, BoundedOptimizer O = LbfgsBOptimizer>
fit_expected<Estimates>
fit_bounded(partable::LatentStructure pt,
            const model::MatrixRep&   rep,
            const SampleStats&        samp,
            Bounds                    bounds,
            D discrepancy = {},
            O optimizer   = {},
            partable::Starts starts = {}) {
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

  auto ml_cache_or = [&]() {
    if constexpr (std::same_as<std::remove_cvref_t<D>, ML>) {
      return discrepancy.prepare(samp);
    } else {
      return fit_expected<int>(0);
    }
  }();
  if (!ml_cache_or.has_value()) return std::unexpected(ml_cache_or.error());

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(FitError{
        FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail, 0, 0.0});
  }
  const EqConstraints& con = *con_or;

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(FitError{
          FitError::Kind::NumericIssue,
          "fit_bounded: bounds_from_partable failed: " + b_or.error().detail,
          0, 0.0});
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != x0_or->size() ||
      bounds.upper.size() != x0_or->size()) {
    return std::unexpected(FitError{
        FitError::Kind::NumericIssue,
        "fit_bounded: bounds size mismatch (lower=" +
            std::to_string(bounds.lower.size()) +
            ", upper=" + std::to_string(bounds.upper.size()) +
            ", x0=" + std::to_string(x0_or->size()) + ")",
        0, 0.0});
  }

  if constexpr (LsDiscrepancy<D> && LsBoundedOptimizer<O>) {
    // ---- LS path: true multi-residual Gauss–Newton + optional eq-penalty. ----

    // Compute the data-residual layout once. The LS-aware discrepancy will
    // report its own `n_resid` via `residuals(samp, m₀)`; we evaluate at the
    // start to size n_data, then size the augmented residual vector.
    auto sm0 = ev.sigma(*x0_or);
    if (!sm0.has_value()) {
      return std::unexpected(FitError{
          FitError::Kind::InvalidStartValues,
          "fit_bounded (ls): start-point Σ evaluation failed: " +
              sm0.error().detail, 0, 0.0});
    }
    auto r0 = discrepancy.residuals(samp, *sm0);
    if (!r0.has_value()) {
      return std::unexpected(r0.error());
    }
    const Eigen::Index n_data = r0->size();
    const Eigen::Index n_eq   = con.active() ? con.A_eq.rows() : 0;
    const Eigen::Index n_total = n_data + n_eq;

    // Penalty weight: scale to the magnitude of F at the start so the eq
    // residuals dominate as θ drifts off the constraint surface, but stay
    // proportional to the data residuals when they're large. 1e10 multiplier
    // puts equality residuals at ~1e-5 of the data residuals at convergence,
    // i.e. ‖A_eq·θ − b_eq‖₂ ≲ √(F_final / μ_eq) ≈ 1e-5 for F_final ~ 1e0.
    const double F0 = 0.5 * r0->squaredNorm();
    const double mu_eq = n_eq > 0
        ? std::max(1.0, F0) * 1e10
        : 0.0;
    const double sqrt_mu_eq = std::sqrt(mu_eq);

    // `ev.{sigma,dsigma_dtheta,dmu_dtheta}` return `model_expected<...>`
    // with `ModelError`; the LS callbacks want `fit_expected<...>` with
    // `FitError`. Wrap on the error path — the optimizer (Ceres LM /
    // LBFGS-B adapter) reads only `has_value()`, so the detail is just
    // for diagnostics if the whole solve later bails out.
    auto wrap_model_err = [](const auto& me, const char* who) -> FitError {
      return FitError{FitError::Kind::NonPositiveDefiniteSigma,
                      std::string("fit_bounded (ls) ") + who + ": " + me.detail,
                      0, 0.0};
    };

    auto resid_fn = [&, n_data, n_total, sqrt_mu_eq, wrap_model_err](
        const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
      auto eval = ev.evaluate(x, false, false);
      if (!eval.has_value())
        return std::unexpected(wrap_model_err(eval.error(), "evaluate"));
      auto r = discrepancy.residuals(samp, eval->moments);
      if (!r.has_value()) return std::unexpected(r.error());
      if (r->size() != n_data) {
        return std::unexpected(FitError{
            FitError::Kind::NumericIssue,
            "fit_bounded (ls): residual length changed mid-fit", 0, 0.0});
      }
      Eigen::VectorXd r_aug(n_total);
      r_aug.head(n_data) = *r;
      if (n_total > n_data) {
        r_aug.tail(n_total - n_data).noalias() =
            sqrt_mu_eq * (con.A_eq * x - con.b_eq);
      }
      return r_aug;
    };

    auto jac_fn = [&, n_data, n_total, sqrt_mu_eq, wrap_model_err](
        const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
      auto eval = ev.evaluate(x, true, true);
      if (!eval.has_value())
        return std::unexpected(wrap_model_err(eval.error(), "evaluate"));
      auto Jr = discrepancy.residual_jacobian(samp, eval->moments,
                                             eval->J_sigma, eval->J_mu);
      if (!Jr.has_value()) return std::unexpected(Jr.error());
      const Eigen::Index n_free = Jr->cols();
      Eigen::MatrixXd J_aug(n_total, n_free);
      J_aug.topRows(n_data) = *Jr;
      if (n_total > n_data) {
        J_aug.bottomRows(n_total - n_data).noalias() = sqrt_mu_eq * con.A_eq;
      }
      return J_aug;
    };

    auto out_or = optimizer.minimize_ls(LsResidualFn(resid_fn),
                                        LsJacobianFn(jac_fn),
                                        n_total,
                                        *x0_or,
                                        bounds.lower,
                                        bounds.upper);
    if (!out_or.has_value()) return std::unexpected(out_or.error());

    // The LS optimizer returns ½‖r_aug‖² which includes the penalty term.
    // Strip it for the user-facing `fmin` (= ½‖r_data‖² = F_data) and
    // verify the equality residual stayed inside `tol_eq`.
    double fmin_data = out_or->fmin;
    if (n_eq > 0) {
      const Eigen::VectorXd eq_resid = con.A_eq * out_or->theta_hat - con.b_eq;
      const double eq_max = eq_resid.cwiseAbs().maxCoeff();
      constexpr double tol_eq = 1e-6;
      if (eq_max > tol_eq) {
        return std::unexpected(FitError{
            FitError::Kind::NumericIssue,
            "fit_bounded (ls): equality residual " + std::to_string(eq_max) +
                " exceeded tolerance " + std::to_string(tol_eq) +
                " (consider raising μ_eq or tightening the optimizer)",
            out_or->iterations, out_or->fmin});
      }
      fmin_data = out_or->fmin - 0.5 * mu_eq * eq_resid.squaredNorm();
      if (fmin_data < 0.0) fmin_data = 0.0;   // float-noise clamp
    }
    return Estimates{std::move(out_or->theta_hat), fmin_data,
                     out_or->iterations};
  } else {
    // ---- Scalar fallback path (non-LS discrepancy or non-LS optimizer). ----
    if (con.active()) {
      return std::unexpected(FitError{
          FitError::Kind::NumericIssue,
          "fit_bounded (scalar): equality constraints + bounds requires an "
          "LS-shape discrepancy and an LsBoundedOptimizer; ML+bounds+eq is a "
          "future tranche",
          0, 0.0});
    }

    auto eval_at = [&](const Eigen::VectorXd& x,
                       Eigen::VectorXd&       grad) -> double {
      auto eval = ev.evaluate(x, true, true);
      if (!eval.has_value()) {
        grad.setZero();
        return std::numeric_limits<double>::infinity();
      }

      if constexpr (std::same_as<std::remove_cvref_t<D>, ML>) {
        auto vg = discrepancy.value_gradient(samp, *ml_cache_or, eval->moments,
                                             eval->J_sigma, eval->J_mu);
        if (!vg.has_value()) {
          grad.setZero();
          return std::numeric_limits<double>::infinity();
        }
        grad = std::move(vg->gradient);
        return vg->value;
      } else {
        auto val = discrepancy.value(samp, eval->moments);
        if (!val.has_value()) {
          grad.setZero();
          return std::numeric_limits<double>::infinity();
        }
        auto g = discrepancy.gradient(samp, eval->moments,
                                      eval->J_sigma, eval->J_mu);
        if (g.has_value()) grad = *g;
        else               grad.setZero();
        return *val;
      }
    };

    auto out_or = optimizer.minimize(eval_at, *x0_or,
                                     bounds.lower, bounds.upper);
    if (!out_or.has_value()) return std::unexpected(out_or.error());
    return Estimates{std::move(out_or->theta_hat), out_or->fmin,
                     out_or->iterations};
  }
}

}  // namespace magmaan::fit
