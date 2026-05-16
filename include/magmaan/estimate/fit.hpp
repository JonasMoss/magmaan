#pragma once

#include <limits>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/optim/concepts.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/nt/ml.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;
using nt::ml::ML;
using optim::BoundedOptimizer;
using optim::Discrepancy;
using optim::LbfgsBOptimizer;
using optim::LbfgsOptimizer;
using optim::LsBoundedOptimizer;
using optim::LsDiscrepancy;
using optim::LsResidualFn;
using optim::LsJacobianFn;
using optim::Optimizer;

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
// and `O` are concept-constrained (magmaan/optim/concepts.hpp):
//   • Discrepancy<D> — value(s, m) → fit_expected<double>;
//                      gradient(s, m, J, Jmu) → fit_expected<VectorXd>.
//   • Optimizer<O>  — minimize(f, x0) → fit_expected<LbfgsOutput>.
// For bounds support see `fit_bounded` below, which constrains `O` on
// `BoundedOptimizer` instead.
template <Discrepancy D = ML, Optimizer O = LbfgsOptimizer>
fit_expected<Estimates>
fit(spec::LatentStructure         pt,         // by value — we may resolve fixed.x
    const model::MatrixRep&    rep,
    const SampleStats&         samp,
    D discrepancy = {},
    O optimizer   = {},
    spec::Starts starts = {}) {        // user start hints (from lavaanify); {} ⇒ heuristic
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
fit_bounded(spec::LatentStructure pt,
            const model::MatrixRep&   rep,
            const SampleStats&        samp,
            Bounds                    bounds,
            D discrepancy = {},
            O optimizer   = {},
            spec::Starts starts = {}) {
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
    // ---- LS path: true multi-residual Gauss–Newton. ----
    //
    // Equality constraints are handled by K-reparameterization (θ = θ₀ + K·α,
    // optimizing the reduced problem over α — exact constraints, no penalty);
    // unconstrained models optimize θ directly. Either way the optimizer sees
    // a plain bounded least-squares problem.

    // Evaluate at the start point to size the data residual.
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

    // Equality constraints — any kind — are handled by K-reparameterization:
    // θ = θ₀ + K·α, optimizing the reduced LS problem over α so the
    // constraints hold exactly by construction (no penalty, no
    // ill-conditioning). For pure-merge K each θ_k is a copy of one α
    // (θ_k = α_{group[k]}, θ₀ = 0), so the θ box bounds fold straight onto α
    // box bounds. For general-linear K (an SVD kernel basis that rotates the
    // parameter axes) a θ box is not an α box — α is optimized unbounded and
    // the bounds are verified on the recovered θ̂ afterwards.
    if (con.active()) {
      const Eigen::Index n_alpha = con.Kmat.cols();
      const bool pure_merge = !con.group.empty();

      auto resid_a = [&, n_data, wrap_model_err](
          const Eigen::VectorXd& a) -> fit_expected<Eigen::VectorXd> {
        const Eigen::VectorXd x = con.expand(a);
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
        return *r;
      };

      if (n_alpha == 0) {           // every parameter pinned by constraints
        auto r = resid_a(Eigen::VectorXd(0));
        if (!r.has_value()) return std::unexpected(r.error());
        return Estimates{con.expand(Eigen::VectorXd(0)),
                         0.5 * r->squaredNorm(), 0};
      }

      auto jac_a = [&, wrap_model_err](
          const Eigen::VectorXd& a) -> fit_expected<Eigen::MatrixXd> {
        const Eigen::VectorXd x = con.expand(a);
        auto eval = ev.evaluate(x, true, true);
        if (!eval.has_value())
          return std::unexpected(wrap_model_err(eval.error(), "evaluate"));
        auto Jr = discrepancy.residual_jacobian(samp, eval->moments,
                                                eval->J_sigma, eval->J_mu);
        if (!Jr.has_value()) return std::unexpected(Jr.error());
        Eigen::MatrixXd J_alpha = *Jr * con.Kmat;   // chain rule: J_r·K
        return J_alpha;
      };

      // α box bounds: folded per group for pure-merge, open for general-linear.
      constexpr double kInf = std::numeric_limits<double>::infinity();
      Eigen::VectorXd lo_a = Eigen::VectorXd::Constant(n_alpha, -kInf);
      Eigen::VectorXd hi_a = Eigen::VectorXd::Constant(n_alpha,  kInf);
      if (pure_merge) {
        for (Eigen::Index k = 0; k < bounds.lower.size(); ++k) {
          const auto g = static_cast<Eigen::Index>(con.group[
              static_cast<std::size_t>(k)]);
          lo_a(g) = std::max(lo_a(g), bounds.lower(k));
          hi_a(g) = std::min(hi_a(g), bounds.upper(k));
        }
      }

      Eigen::VectorXd alpha0 = con.contract(*x0_or);
      alpha0 = alpha0.cwiseMax(lo_a).cwiseMin(hi_a);

      auto out_a = optimizer.minimize_ls(LsResidualFn(resid_a),
                                         LsJacobianFn(jac_a),
                                         n_data, alpha0, lo_a, hi_a);
      if (!out_a.has_value()) return std::unexpected(out_a.error());
      Eigen::VectorXd theta_hat = con.expand(out_a->theta_hat);

      if (!pure_merge) {
        // General-linear α was optimized unbounded — a θ box does not map to
        // an α box. Verify the recovered θ̂ honors the bounds; a real
        // violation means the constrained optimum is Heywood, which an α-box
        // cannot represent. Fail explicitly rather than return such a fit.
        constexpr double tol_b = 1e-6;
        for (Eigen::Index k = 0; k < theta_hat.size(); ++k) {
          if (theta_hat(k) < bounds.lower(k) - tol_b ||
              theta_hat(k) > bounds.upper(k) + tol_b) {
            return std::unexpected(FitError{
                FitError::Kind::NumericIssue,
                "fit_bounded (ls): general-linear equality drove parameter " +
                    std::to_string(k) + " past its bound — the constrained "
                    "optimum appears Heywood and is not representable as an "
                    "alpha-box",
                out_a->iterations, out_a->fmin});
          }
        }
      }
      return Estimates{std::move(theta_hat), out_a->fmin, out_a->iterations};
    }

    // ---- Unconstrained: plain bounded LS over θ. ----
    auto resid_fn = [&, n_data, wrap_model_err](
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
      return std::move(*r);
    };

    auto jac_fn = [&, wrap_model_err](
        const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
      auto eval = ev.evaluate(x, true, true);
      if (!eval.has_value())
        return std::unexpected(wrap_model_err(eval.error(), "evaluate"));
      auto Jr = discrepancy.residual_jacobian(samp, eval->moments,
                                             eval->J_sigma, eval->J_mu);
      if (!Jr.has_value()) return std::unexpected(Jr.error());
      return std::move(*Jr);
    };

    auto out_or = optimizer.minimize_ls(LsResidualFn(resid_fn),
                                        LsJacobianFn(jac_fn),
                                        n_data,
                                        *x0_or,
                                        bounds.lower,
                                        bounds.upper);
    if (!out_or.has_value()) return std::unexpected(out_or.error());
    return Estimates{std::move(out_or->theta_hat), out_or->fmin,
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

}  // namespace magmaan::estimate
