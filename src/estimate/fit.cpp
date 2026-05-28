#include "magmaan/estimate/fit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/reparameterize.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/estimate/gmm/gp.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/optim/terminal_audit.hpp"
#include "magmaan/spec/partable.hpp"

#include "detail_vech.hpp"

// Convenience composers: package the data → objective → constraints →
// optimizer pipeline into a single non-template call. These are the
// canonical compositions from the refactor plan, named so callers (R
// package, tests, inference) bind to a function rather than a template.

namespace magmaan::estimate {

namespace {

FitError fit_err(FitError::Kind kind, std::string detail) {
  return FitError{kind, std::move(detail), 0, 0.0};
}

// Dispatch a scalar-objective optimization by backend. Shared by `fit_ml` and
// (via `scalarize`) the gmm/gls path. `Backend::Ceres` has no scalar entry
// point — it is a least-squares-only backend — and is rejected here.
fit_expected<optim::OptimResult>
run_scalar(const optim::ScalarProblem& prob, const Eigen::VectorXd& x0,
           const Bounds& bounds, Backend backend, OptimOptions opts) {
  switch (backend) {
    case Backend::NloptSlsqp:
      return optim::nlopt_slsqp(prob, x0, bounds, opts);
    case Backend::NloptBobyqa:
      // BOBYQA requires *finite* bounds (no ±infinity sentinels). The free
      // function enforces this at the optim layer; the dispatch is a clean
      // pass-through.
      return optim::nlopt_bobyqa(prob, x0, bounds, opts);
    case Backend::NloptTnewton:
      return optim::nlopt_tnewton(prob, x0, bounds, opts);
    case Backend::NloptVar2:
      return optim::nlopt_var2(prob, x0, bounds, opts);
    case Backend::NloptLbfgs:
      return optim::nlopt_lbfgs(prob, x0, bounds, opts);
    case Backend::Ipopt:
#ifdef MAGMAAN_WITH_IPOPT
      return optim::ipopt(prob, x0, bounds, opts);
#else
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "IPOPT backend requested but MAGMAAN_WITH_IPOPT is off"));
#endif
    case Backend::Port:
#ifdef MAGMAAN_WITH_PORT
      // PORT drmngb honors box bounds natively; an empty `bounds` is passed
      // through to the adapter, which fills ±infinity sentinels.
      return optim::port(prob, x0, bounds, opts);
#else
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "Port backend requested but MAGMAAN_WITH_PORT is off"));
#endif
    case Backend::Ceres:
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "Ceres backend applies only to the least-squares path "
          "(fit_gmm / fit_gls); it has no scalar-objective entry point"));
    case Backend::CeresBfgs:
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "CeresBfgs backend applies only to the least-squares path "
          "(fit_gmm / fit_gls); it has no scalar-objective entry point"));
    case Backend::PortNls:
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "PortNls backend applies only to the least-squares path "
          "(fit_gmm / fit_gls / fit_snlls); it sees the residual structure "
          "directly and has no scalar-objective entry point — use Port for "
          "scalar trust-region fits"));
  }
  return std::unexpected(fit_err(FitError::Kind::NumericIssue,
      "unknown optimizer backend"));
}

fit_expected<optim::OptimResult>
run_gmm(const optim::GmmProblem& prob, const Eigen::VectorXd& x0,
        const Bounds& bounds, Backend backend, OptimOptions opts) {
  if (backend == Backend::Ceres || backend == Backend::CeresBfgs) {
#ifdef MAGMAAN_WITH_CERES
    optim::CeresOptions copts;
    copts.max_iter = opts.max_iter;
    copts.ftol     = opts.ftol;
    copts.gtol     = opts.gtol;
    if (backend == Backend::CeresBfgs) {
      if (!bounds.empty()) {
        return std::unexpected(fit_err(FitError::Kind::NumericIssue,
            "CeresBfgs backend is unbounded and cannot honor box bounds"));
      }
      return optim::ceres_bfgs(prob, x0, copts);
    }
    return optim::ceres_lm(prob, x0, bounds, copts);
#else
    (void)opts;
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_gmm: Ceres backend requested but MAGMAAN_WITH_CERES is off"));
#endif
  }
  if (backend == Backend::PortNls) {
#ifdef MAGMAAN_WITH_PORT
    // PortNls drives the multi-residual problem directly through NL2SOL; no
    // scalarisation. The adapter handles ±infinity bounds internally.
    return optim::port_nls(prob, x0, bounds, opts);
#else
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_gmm: PortNls backend requested but MAGMAAN_WITH_PORT is off"));
#endif
  }
  return run_scalar(optim::scalarize(prob), x0, bounds, backend, opts);
}

fit_expected<optim::OptimResult>
run_scalar_constrained(const optim::ScalarProblem& prob,
                       const optim::ConstraintFn& h,
                       const optim::ConstraintJacFn& J_h,
                       Eigen::Index n_constraint,
                       const Eigen::VectorXd& x0, const Bounds& bounds,
                       Backend backend, OptimOptions opts, const char* who) {
  optim::ConstrainedScalarProblem cprob;
  cprob.objective         = prob;
  cprob.h                 = h;
  cprob.J_h               = J_h;
  cprob.n_constraint      = n_constraint;
  cprob.constraint_lower  = Eigen::VectorXd::Zero(n_constraint);
  cprob.constraint_upper  = Eigen::VectorXd::Zero(n_constraint);

  switch (backend) {
    case Backend::NloptSlsqp:
      return optim::nlopt_slsqp_constrained(cprob, x0, bounds, opts);
    case Backend::Ipopt:
#ifdef MAGMAAN_WITH_IPOPT
      return optim::ipopt_constrained(cprob, x0, bounds, opts);
#else
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          std::string(who) +
              ": nonlinear equality constraints require optimizer "
              "\"nlopt-slsqp\" or an IPOPT-enabled build with optimizer "
              "\"ipopt\""));
#endif
    default:
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          std::string(who) +
              ": nonlinear equality constraints require optimizer "
              "\"nlopt-slsqp\" or \"ipopt\""));
  }
  return std::unexpected(fit_err(FitError::Kind::NumericIssue,
      "unknown constrained optimizer backend"));
}

// Shared prelude — resolve fixed.x, build the evaluator, build constraints.
struct Prelude {
  model::ModelEvaluator  ev;
  EqConstraints          con;   // linear-equality affine reparameterization
  NonlinearEqConstraints nl;    // nonlinear `==` constraints
};

struct FcSemPrelude {
  model::FcSemEvaluator  ev;
  EqConstraints          con;
  NonlinearEqConstraints nl;
};

fit_expected<Prelude>
prelude(spec::LatentStructure& pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        const char* who) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(e.error());
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  if (x0.size() != static_cast<Eigen::Index>(pt.n_free())) {
    return std::unexpected(fit_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": x0 size (" + std::to_string(x0.size()) +
            ") != n_free (" + std::to_string(pt.n_free()) + ")"));
  }
  // `allow_nonlinear`: the ML / GMM composers enforce nonlinear `==` through
  // IPOPT, so `build_eq_constraints` builds only the *linear*
  // reparameterization here instead of rejecting the model. The SNLLS /
  // ordinal paths reject nonlinear constraints separately.
  auto con_or = build_eq_constraints(pt, /*allow_nonlinear=*/true);
  if (!con_or.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constraint: " + con_or.error().detail));
  }
  return Prelude{std::move(*ev_or), std::move(*con_or),
                 build_nl_constraints(pt)};
}

fit_expected<FcSemPrelude>
prelude_fcsem(spec::LatentStructure& pt, const Eigen::VectorXd& x0,
              const char* who) {
  auto ev_or = model::FcSemEvaluator::build(pt);
  if (!ev_or.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": FcSemEvaluator::build failed: " +
            ev_or.error().detail));
  }
  if (x0.size() != static_cast<Eigen::Index>(pt.n_free())) {
    return std::unexpected(fit_err(FitError::Kind::InvalidStartValues,
        std::string(who) + ": x0 size (" + std::to_string(x0.size()) +
            ") != n_free (" + std::to_string(pt.n_free()) + ")"));
  }
  auto con_or = build_eq_constraints(pt, /*allow_nonlinear=*/true);
  if (!con_or.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constraint: " + con_or.error().detail));
  }
  return FcSemPrelude{std::move(*ev_or), std::move(*con_or),
                      build_nl_constraints(pt)};
}

}  // namespace

namespace {

fit_expected<Estimates>
compose_scalar_ml(const optim::ScalarProblem& prob, const EqConstraints& con,
                  const NonlinearEqConstraints& nl,
                  const Eigen::VectorXd& x0, const Bounds& bounds,
                  Backend backend, OptimOptions opts, const char* who) {
  if (nl.active()) {
    if (!con.active()) {
      auto h_fn   = [&nl](const Eigen::VectorXd& th) { return nl.h(th); };
      auto jac_fn = [&nl](const Eigen::VectorXd& th) { return nl.jacobian(th); };
      auto r =
          run_scalar_constrained(prob, h_fn, jac_fn, nl.m(), x0, bounds,
                                 backend, opts, who);
      if (!r.has_value()) return std::unexpected(r.error());
      return Estimates{prob.expand(r->x), r->fmin, r->iterations,
                       r->f_evals, r->g_evals, r->status, r->grad_inf_norm,
                       std::move(r->audit)};
    }

    const optim::ScalarProblem prob_a = reparameterize(prob, con);
    if (con.n_alpha == 0) {
      Eigen::VectorXd theta = con.expand(Eigen::VectorXd(0));
      Eigen::VectorXd scratch(theta.size());
      const double f = prob.f(theta, scratch);
      return Estimates{std::move(theta), f, 0};
    }
    auto h_a = [&nl, &con](const Eigen::VectorXd& a) {
      return nl.h(con.expand(a));
    };
    auto jac_a = [&nl, &con](const Eigen::VectorXd& a) {
      return Eigen::MatrixXd(nl.jacobian(con.expand(a)) * con.K());
    };
    Eigen::VectorXd alpha0 = con.contract(x0);
    Bounds abounds;
    const bool pure_merge = !con.group.empty();
    if (!bounds.empty() && pure_merge) {
      abounds = fold_alpha_bounds(con, bounds);
      alpha0  = alpha0.cwiseMax(abounds.lower).cwiseMin(abounds.upper);
    }
    auto r = run_scalar_constrained(prob_a, h_a, jac_a, nl.m(), alpha0,
                                    abounds, backend, opts, who);
    if (!r.has_value()) return std::unexpected(r.error());
    Eigen::VectorXd theta_hat = prob_a.expand(r->x);
    if (!bounds.empty() && !pure_merge) {
      constexpr double tol = 1e-6;
      for (Eigen::Index k = 0; k < theta_hat.size(); ++k) {
        if (theta_hat(k) < bounds.lower(k) - tol ||
            theta_hat(k) > bounds.upper(k) + tol) {
          return std::unexpected(fit_err(FitError::Kind::NumericIssue,
              std::string(who) +
                  ": general-linear equality drove parameter " +
                  std::to_string(k) + " past its bound"));
        }
      }
    }
    return Estimates{std::move(theta_hat), r->fmin, r->iterations,
                     r->f_evals, r->g_evals, r->status, r->grad_inf_norm,
                     std::move(r->audit)};
  }

  if (!con.active()) {
    auto out = run_scalar(prob, x0, bounds, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    return Estimates{prob.expand(out->x), out->fmin, out->iterations,
                     out->f_evals, out->g_evals,
                     out->status, out->grad_inf_norm,
                     std::move(out->audit)};
  }

  const optim::ScalarProblem prob_a = reparameterize(prob, con);
  if (con.n_alpha == 0) {
    Eigen::VectorXd theta = con.expand(Eigen::VectorXd(0));
    Eigen::VectorXd scratch(theta.size());
    const double f = prob.f(theta, scratch);
    return Estimates{std::move(theta), f, 0};
  }

  Eigen::VectorXd alpha0 = con.contract(x0);
  Bounds abounds;
  const bool pure_merge = !con.group.empty();
  if (!bounds.empty() && pure_merge) {
    abounds = fold_alpha_bounds(con, bounds);
    alpha0 = alpha0.cwiseMax(abounds.lower).cwiseMin(abounds.upper);
  }
  auto out = run_scalar(prob_a, alpha0, abounds, backend, opts);
  if (!out.has_value()) return std::unexpected(out.error());
  return Estimates{prob_a.expand(out->x), out->fmin, out->iterations,
                   out->f_evals, out->g_evals,
                   out->status, out->grad_inf_norm,
                   std::move(out->audit)};
}

// Shared moment-quadratic composition: build the LS problem, fold equality
// constraints, optimize, expand. `fit_gmm` and `fit_gls` differ only in how
// the weight is produced.
fit_expected<Estimates>
compose_gmm(const model::ModelEvaluator& ev, const EqConstraints& con,
            const NonlinearEqConstraints& nl, const SampleStats& samp,
            const Eigen::VectorXd& x0, const gmm::Weight& weight,
            const Bounds& bounds, Backend backend, OptimOptions opts) {
  auto prob_or = gmm::residuals(ev, samp, x0, weight);
  if (!prob_or.has_value()) return std::unexpected(prob_or.error());
  const optim::GmmProblem prob = std::move(*prob_or);

  // Nonlinear equality constraints — scalarize and let a constrained scalar
  // optimizer drive the NLP. The LS sum-of-squares structure is not exposed to
  // the constrained optimizers in v1; exact Hessian support is tracked
  // separately.
  if (nl.active()) {
    if (!con.active()) {
      const optim::ScalarProblem sprob = optim::scalarize(prob);
      auto h_fn   = [&nl](const Eigen::VectorXd& th) { return nl.h(th); };
      auto jac_fn = [&nl](const Eigen::VectorXd& th) { return nl.jacobian(th); };
      auto r =
          run_scalar_constrained(sprob, h_fn, jac_fn, nl.m(), x0, bounds,
                                 backend, opts, "fit_gmm");
      if (!r.has_value()) return std::unexpected(r.error());
      return Estimates{prob.expand(r->x), r->fmin, r->iterations,
                       r->f_evals, r->g_evals, r->status, r->grad_inf_norm,
                       std::move(r->audit)};
    }
    // Linear + nonlinear equality constraints together: fold the linear
    // equalities into the affine α-reparameterization (θ = θ₀ + K·α), then
    // run the constrained scalar optimizer on the nonlinear constraints
    // re-expressed in α:
    // h_α(α) = h(θ₀+Kα), ∂h_α/∂α = (∂h/∂θ)·K.
    const optim::GmmProblem prob_a = reparameterize(prob, con);
    if (con.n_alpha == 0) {  // every parameter pinned by the linear system
      auto rr = prob_a.r(Eigen::VectorXd(0));
      if (!rr.has_value()) return std::unexpected(rr.error());
      return Estimates{prob_a.expand(Eigen::VectorXd(0)),
                       0.5 * rr->squaredNorm(), 0};
    }
    const optim::ScalarProblem sprob_a = optim::scalarize(prob_a);
    auto h_a = [&nl, &con](const Eigen::VectorXd& a) {
      return nl.h(con.expand(a));
    };
    auto jac_a = [&nl, &con](const Eigen::VectorXd& a) {
      return Eigen::MatrixXd(nl.jacobian(con.expand(a)) * con.K());
    };
    Eigen::VectorXd alpha0 = con.contract(x0);
    Bounds abounds;
    const bool pure_merge = !con.group.empty();
    if (!bounds.empty() && pure_merge) {
      abounds = fold_alpha_bounds(con, bounds);
      alpha0  = alpha0.cwiseMax(abounds.lower).cwiseMin(abounds.upper);
    }
    auto r = run_scalar_constrained(sprob_a, h_a, jac_a, nl.m(), alpha0,
                                    abounds, backend, opts, "fit_gmm");
    if (!r.has_value()) return std::unexpected(r.error());
    Eigen::VectorXd theta_hat = prob_a.expand(r->x);
    if (!bounds.empty() && !pure_merge) {
      // General-linear α was optimized unbounded — verify θ̂ honors the box.
      constexpr double tol = 1e-6;
      for (Eigen::Index k = 0; k < theta_hat.size(); ++k) {
        if (theta_hat(k) < bounds.lower(k) - tol ||
            theta_hat(k) > bounds.upper(k) + tol) {
          return std::unexpected(fit_err(FitError::Kind::NumericIssue,
              "fit_gmm: general-linear equality drove parameter " +
                  std::to_string(k) + " past its bound"));
        }
      }
    }
    return Estimates{std::move(theta_hat), r->fmin, r->iterations,
                     r->f_evals, r->g_evals, r->status, r->grad_inf_norm,
                     std::move(r->audit)};
  }

  if (!con.active()) {
    auto out = run_gmm(prob, x0, bounds, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    return Estimates{prob.expand(out->x), out->fmin, out->iterations,
                     out->f_evals, out->g_evals,
                     out->status, out->grad_inf_norm,
                     std::move(out->audit)};
  }

  // Constrained: optimize over the reduced α (θ = θ₀ + K·α).
  const optim::GmmProblem prob_a = reparameterize(prob, con);
  if (con.n_alpha == 0) {  // every parameter pinned by constraints
    auto r = prob_a.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    return Estimates{prob_a.expand(Eigen::VectorXd(0)),
                     0.5 * r->squaredNorm(), 0};
  }

  Eigen::VectorXd alpha0 = con.contract(x0);
  Bounds abounds;
  const bool pure_merge = !con.group.empty();
  if (!bounds.empty() && pure_merge) {
    abounds = fold_alpha_bounds(con, bounds);
    alpha0 = alpha0.cwiseMax(abounds.lower).cwiseMin(abounds.upper);
  }
  auto out = run_gmm(prob_a, alpha0, abounds, backend, opts);
  if (!out.has_value()) return std::unexpected(out.error());
  Eigen::VectorXd theta_hat = prob_a.expand(out->x);
  if (!bounds.empty() && !pure_merge) {
    // General-linear α was optimized unbounded — verify θ̂ honors the box.
    constexpr double tol = 1e-6;
    for (Eigen::Index k = 0; k < theta_hat.size(); ++k) {
      if (theta_hat(k) < bounds.lower(k) - tol ||
          theta_hat(k) > bounds.upper(k) + tol) {
        return std::unexpected(fit_err(FitError::Kind::NumericIssue,
            "fit_gmm: general-linear equality drove parameter " +
                std::to_string(k) + " past its bound"));
      }
    }
  }
  return Estimates{std::move(theta_hat), out->fmin, out->iterations,
                   out->f_evals, out->g_evals,
                   out->status, out->grad_inf_norm,
                   std::move(out->audit)};
}

}  // namespace

// Attach the L2 fit-finalization audit. Called from every public fit_* entry
// point that has a `pre->ev` / `pre->con` / `pre->nl` in scope — i.e. all
// MatrixRep paths (fit_ml/fit_gls/fit_gmm/fit_snlls/fit_snlls_gls). The
// FCSEM path uses a different evaluator and is excluded; its diagnostics
// stay default-initialized. `static` for internal linkage — the helper
// lives outside the anonymous namespaces only because both anon blocks
// (220-431, 506-543) bracket their own compose_* helpers; placing it here
// keeps it callable from all the fit_* entry points below.
static void attach_diagnostics(Estimates& est, const Prelude& pre,
                               const Bounds& bounds) {
  est.diagnostics = finalize_fit_diagnostics(est.theta, pre.ev, pre.con,
                                             pre.nl, bounds);
}

namespace {

fit_expected<Estimates>
compose_fisher_ml(const Prelude& pre, const SampleStats& samp,
                  const Eigen::VectorXd& x0, const Bounds& bounds,
                  OptimOptions opts);

}  // namespace

fit_expected<Estimates>
fit_gmm(spec::LatentStructure pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        gmm::Weight weight, Bounds bounds, Backend backend,
        OptimOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_gmm");
  if (!pre.has_value()) return std::unexpected(pre.error());
  auto est = compose_gmm(pre->ev, pre->con, pre->nl, samp, x0, weight, bounds,
                         backend, opts);
  if (!est.has_value()) return est;
  attach_diagnostics(*est, *pre, bounds);
  return est;
}

fit_expected<Estimates>
fit_gls(spec::LatentStructure pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        Bounds bounds, Backend backend, OptimOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_gls");
  if (!pre.has_value()) return std::unexpected(pre.error());
  auto W = gmm::normal_theory_weight(pre->ev, samp, x0);
  if (!W.has_value()) return std::unexpected(W.error());
  auto est = compose_gmm(pre->ev, pre->con, pre->nl, samp, x0, *W, bounds,
                         backend, opts);
  if (!est.has_value()) return est;
  attach_diagnostics(*est, *pre, bounds);
  return est;
}

fit_expected<Estimates>
fit_gls_pairwise(spec::LatentStructure pt, const model::MatrixRep& rep,
                 const data::RawData& raw,
                 const data::PairwiseSampleStats& pw,
                 const Eigen::VectorXd& x0, Bounds bounds, Backend backend,
                 OptimOptions opts) {
  // Pack pairwise stats into a SampleStats so the existing prelude / layout
  // machinery sees the same shape it expects.
  SampleStats samp;
  samp.S     = pw.S;
  samp.mean  = pw.mean;
  samp.n_obs = pw.n_obs;

  auto pre = prelude(pt, rep, samp, x0, "fit_gls_pairwise");
  if (!pre.has_value()) return std::unexpected(pre.error());

  // Detect mean structure by probing the evaluator at x0 — mirrors how
  // `gmm::normal_theory_weight` determines the G3b layout.
  auto eval0 = pre->ev.evaluate(x0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_gls_pairwise: model evaluation at x0 failed"));
  }
  if (eval0->moments.sigma.size() != samp.S.size()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_gls_pairwise: evaluator block count mismatches pw"));
  }

  // Materialize Γ_NT^pw per block.
  auto gnt_pw_or = data::gamma_nt_pairwise(raw, pw);
  if (!gnt_pw_or.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_gls_pairwise: gamma_nt_pairwise failed: " +
        gnt_pw_or.error().detail));
  }
  const auto& gnt_pw = *gnt_pw_or;

  // Build per-block W in the [μ (if any) | σ-vech] G3b layout.
  gmm::Weight W;
  W.reserve(samp.S.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::Index p = samp.S[b].rows();
    const Eigen::Index pstar = p * (p + 1) / 2;
    const bool has_means_b = (b < eval0->moments.mu.size())
                          && (eval0->moments.mu[b].size() > 0);
    const Eigen::Index block_rows = (has_means_b ? p : 0) + pstar;

    // σ-block: W_σ = (Γ_NT^pw)⁻¹.
    Eigen::LLT<Eigen::MatrixXd> llt(gnt_pw[b]);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "fit_gls_pairwise: Γ_NT^pw not positive definite in block " +
          std::to_string(b)));
    }
    Eigen::MatrixXd Wsigma =
        llt.solve(Eigen::MatrixXd::Identity(pstar, pstar));
    Wsigma = (0.5 * (Wsigma + Wsigma.transpose())).eval();

    Eigen::MatrixXd Wb = Eigen::MatrixXd::Zero(block_rows, block_rows);
    Eigen::Index off = 0;
    if (has_means_b) {
      // μ-block: keep the Σ-only convention `Ŝ^pw⁻¹` for v0. Pairwise μ ACOV
      // (Hadamard `π_jk/(π_j π_k)` on Σ̂_pw) is a v1 extension once needed.
      Eigen::LLT<Eigen::MatrixXd> s_llt(samp.S[b]);
      if (s_llt.info() != Eigen::Success) {
        return std::unexpected(fit_err(FitError::Kind::NonPositiveDefiniteSample,
            "fit_gls_pairwise: pairwise S not positive definite in block " +
            std::to_string(b)));
      }
      Wb.block(0, 0, p, p) =
          s_llt.solve(Eigen::MatrixXd::Identity(p, p));
      off = p;
    }
    Wb.block(off, off, pstar, pstar) = Wsigma;
    W.push_back(std::move(Wb));
  }

  auto est = compose_gmm(pre->ev, pre->con, pre->nl, samp, x0, W, bounds,
                         backend, opts);
  if (!est.has_value()) return est;
  attach_diagnostics(*est, *pre, bounds);
  return est;
}

fit_expected<Estimates>
fit_ml(spec::LatentStructure pt, const model::MatrixRep& rep,
       const SampleStats& samp, const Eigen::VectorXd& x0, Bounds bounds,
       Backend backend, OptimOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_ml");
  if (!pre.has_value()) return std::unexpected(pre.error());
  const model::ModelEvaluator& ev = pre->ev;

  auto obj_or = estimate::ml_objective(ev, samp);
  if (!obj_or.has_value()) return std::unexpected(obj_or.error());
  const optim::ScalarProblem prob = std::move(*obj_or);
  auto est = compose_scalar_ml(prob, pre->con, pre->nl, x0, bounds, backend,
                               opts, "fit_ml");
  if (!est.has_value()) return est;
  attach_diagnostics(*est, *pre, bounds);
  return est;
}

fit_expected<Estimates>
fit_ml_fisher(spec::LatentStructure pt, const model::MatrixRep& rep,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              Bounds bounds, OptimOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_ml_fisher");
  if (!pre.has_value()) return std::unexpected(pre.error());
  return compose_fisher_ml(*pre, samp, x0, bounds, opts);
}

fit_expected<Estimates>
fit_ml_fcsem(spec::LatentStructure pt, const SampleStats& samp,
             const Eigen::VectorXd& x0, Bounds bounds, Backend backend,
             OptimOptions opts) {
  auto pre = prelude_fcsem(pt, x0, "fit_ml_fcsem");
  if (!pre.has_value()) return std::unexpected(pre.error());
  auto obj_or = estimate::ml_objective(pre->ev, samp);
  if (!obj_or.has_value()) return std::unexpected(obj_or.error());
  const optim::ScalarProblem prob = std::move(*obj_or);
  return compose_scalar_ml(prob, pre->con, pre->nl, x0, bounds, backend, opts,
                           "fit_ml_fcsem");
}

namespace {

// Shared Golub–Pereyra composition: profile out the linear block, optimize the
// nonlinear β. `fit_snlls` and `fit_snlls_gls` differ only in the weight.
fit_expected<Estimates>
compose_snlls(const spec::LatentStructure& pt, const model::ModelEvaluator& ev,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              const gmm::Weight& weight, Backend backend, OptimOptions opts) {
  if (!pt.nl_constraints.empty()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_snlls: nonlinear equality constraints are not supported by the "
        "separable (Golub–Pereyra) least-squares path — use fit_ml or fit_gmm"));
  }
  auto base_or = gmm::residuals(ev, samp, x0, weight);
  if (!base_or.has_value()) return std::unexpected(base_or.error());

  auto gp_or = gmm::gp(*base_or, pt, ev, x0);
  if (!gp_or.has_value()) return std::unexpected(gp_or.error());
  const optim::GmmProblem& prob = gp_or->problem;

  // Covariance-only models have no nonlinear parameters — the profile solves
  // them outright, no outer optimization.
  if (prob.n_param == 0) {
    auto r = prob.r(Eigen::VectorXd(0));
    if (!r.has_value()) return std::unexpected(r.error());
    Estimates est{prob.expand(Eigen::VectorXd(0)),
                  0.5 * r->squaredNorm(), 0};
    est.n_nonlinear = gp_or->n_nonlinear;
    est.n_linear    = gp_or->n_linear;
    if (gp_or->n_alpha_solve_fast)
      est.n_alpha_solve_fast = *gp_or->n_alpha_solve_fast;
    if (gp_or->n_alpha_solve_fallback)
      est.n_alpha_solve_fallback = *gp_or->n_alpha_solve_fallback;
    return est;
  }

  // The nonlinear block (Λ, Β) carries no box bounds, so β is optimized
  // unbounded; variance non-negativity lives on the profiled-out α.
  auto out = run_gmm(prob, gp_or->beta0, Bounds{}, backend, opts);
  if (!out.has_value()) return std::unexpected(out.error());
  Estimates est{prob.expand(out->x), out->fmin, out->iterations,
                out->f_evals, out->g_evals,
                out->status, out->grad_inf_norm,
                std::move(out->audit)};
  est.n_nonlinear = gp_or->n_nonlinear;
  est.n_linear    = gp_or->n_linear;
  if (gp_or->n_alpha_solve_fast)
    est.n_alpha_solve_fast = *gp_or->n_alpha_solve_fast;
  if (gp_or->n_alpha_solve_fallback)
    est.n_alpha_solve_fallback = *gp_or->n_alpha_solve_fallback;
  return est;
}

fit_expected<double>
total_n_obs_double(const SampleStats& samp, const char* who) {
  double total = 0.0;
  for (auto n : samp.n_obs) total += static_cast<double>(n);
  if (total <= 0.0) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        std::string(who) + ": SampleStats has non-positive total n_obs"));
  }
  return total;
}

// Expected Hessian of F_ML itself, not the N/2-scaled post-fit information
// matrix. This keeps the Fisher equation local and literal:
// H_E(θ) d = -∇F_ML(θ).
fit_expected<Eigen::MatrixXd>
expected_ml_hessian_f(const model::ModelEvaluator& ev,
                      const SampleStats& samp,
                      const Eigen::VectorXd& theta) {
  using detail::vech_index;
  using detail::vech_len;

  auto eval = ev.evaluate(theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::NonPositiveDefiniteSigma,
        "fit_ml_fisher: expected Hessian evaluate failed: " +
            eval.error().detail));
  }
  const auto& moments = eval->moments;
  const auto& J       = eval->J_sigma;
  const auto& Jmu     = eval->J_mu;

  if (samp.S.size() != moments.sigma.size() ||
      samp.n_obs.size() != samp.S.size()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_ml_fisher: sample moments and implied moments have incompatible "
        "block counts"));
  }
  auto N_or = total_n_obs_double(samp, "fit_ml_fisher");
  if (!N_or.has_value()) return std::unexpected(N_or.error());
  const double n_total = *N_or;

  const std::size_t n_blocks = samp.S.size();
  const std::size_t n_free   = ev.n_free();
  std::vector<Eigen::MatrixXd> SigmaInv(n_blocks);
  std::vector<double>          weight(n_blocks, 0.0);
  std::vector<Eigen::Index>    p_dim(n_blocks, 0);
  std::vector<Eigen::Index>    vech_off(n_blocks, 0);
  std::vector<bool>            mean_used(n_blocks, false);

  Eigen::Index running = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const auto& S = samp.S[b];
    const Eigen::MatrixXd Sigma =
        0.5 * (moments.sigma[b] + moments.sigma[b].transpose());
    if (S.rows() != Sigma.rows() || S.cols() != Sigma.cols()) {
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "fit_ml_fisher: block " + std::to_string(b) +
              " S and Sigma have different shapes"));
    }
    const Eigen::Index p = Sigma.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(fit_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ml_fisher: implied Sigma is not positive definite in block " +
              std::to_string(b)));
    }
    SigmaInv[b] = llt.solve(Eigen::MatrixXd::Identity(p, p));
    weight[b]   = static_cast<double>(samp.n_obs[b]) / n_total;
    p_dim[b]    = p;
    vech_off[b] = running;
    running += vech_len(p);

    mean_used[b] = b < samp.mean.size() && b < moments.mu.size() &&
                   samp.mean[b].size() > 0 && moments.mu[b].size() > 0;
    if (mean_used[b] && samp.mean[b].size() != p) {
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "fit_ml_fisher: block " + std::to_string(b) +
              " sample mean and Sigma have different dimensions"));
    }
  }

  if (J.rows() != running ||
      J.cols() != static_cast<Eigen::Index>(n_free)) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_ml_fisher: covariance Jacobian shape does not match moments"));
  }

  const bool has_mean_rows = Jmu.size() > 0;
  std::vector<Eigen::Index> mu_off(n_blocks, 0);
  if (has_mean_rows) {
    Eigen::Index running_p = 0;
    for (std::size_t b = 0; b < n_blocks; ++b) {
      mu_off[b] = running_p;
      running_p += p_dim[b];
    }
    if (Jmu.rows() != running_p ||
        Jmu.cols() != static_cast<Eigen::Index>(n_free)) {
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "fit_ml_fisher: mean Jacobian shape does not match moments"));
    }
  }

  std::vector<std::vector<Eigen::MatrixXd>> T(
      n_free, std::vector<Eigen::MatrixXd>(n_blocks));
  Eigen::MatrixXd M;
  for (std::size_t k = 0; k < n_free; ++k) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const Eigen::Index p = p_dim[b];
      M.setZero(p, p);
      for (Eigen::Index c = 0; c < p; ++c) {
        for (Eigen::Index r = c; r < p; ++r) {
          const double v = J(vech_off[b] + vech_index(p, r, c),
                             static_cast<Eigen::Index>(k));
          M(r, c) = v;
          if (r != c) M(c, r) = v;
        }
      }
      T[k][b].noalias() = SigmaInv[b] * M;
    }
  }

  std::vector<std::vector<Eigen::VectorXd>> eta;
  if (has_mean_rows) {
    eta.assign(n_free, std::vector<Eigen::VectorXd>(n_blocks));
    for (std::size_t k = 0; k < n_free; ++k) {
      for (std::size_t b = 0; b < n_blocks; ++b) {
        if (!mean_used[b]) {
          eta[k][b] = Eigen::VectorXd::Zero(p_dim[b]);
          continue;
        }
        const Eigen::VectorXd nu_kb =
            Jmu.col(static_cast<Eigen::Index>(k))
                .segment(mu_off[b], p_dim[b]);
        eta[k][b].noalias() = SigmaInv[b] * nu_kb;
      }
    }
  }

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(n_free), static_cast<Eigen::Index>(n_free));
  for (std::size_t a = 0; a < n_free; ++a) {
    for (std::size_t b = a; b < n_free; ++b) {
      double acc = 0.0;
      for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        double per_block =
            (T[a][blk].transpose().array() * T[b][blk].array()).sum();
        if (has_mean_rows && mean_used[blk]) {
          const Eigen::VectorXd nu_a =
              Jmu.col(static_cast<Eigen::Index>(a))
                  .segment(mu_off[blk], p_dim[blk]);
          per_block += 2.0 * nu_a.dot(eta[b][blk]);
        }
        acc += weight[blk] * per_block;
      }
      H(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = acc;
      if (a != b)
        H(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) = acc;
    }
  }

  if (!H.allFinite()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_ml_fisher: expected Hessian contains non-finite entries"));
  }
  return H;
}

Eigen::VectorXd
solve_fisher_direction(const Eigen::MatrixXd& H,
                       const Eigen::VectorXd& grad) {
  Eigen::MatrixXd A = 0.5 * (H + H.transpose());
  const double scale =
      std::max(1.0, A.cwiseAbs().maxCoeff());
  Eigen::VectorXd step = Eigen::VectorXd::Zero(grad.size());

  for (double ridge : {0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2}) {
    Eigen::MatrixXd Ar = A;
    if (ridge > 0.0) {
      Ar.diagonal().array() += ridge * scale;
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Ar);
    if (ldlt.info() != Eigen::Success) continue;
    step = ldlt.solve(-grad);
    if (step.allFinite() && grad.dot(step) < 0.0) return step;
  }

  // Last-resort descent direction for rank-deficient expected information.
  // This keeps the globalized scoring loop moving while the terminal audit
  // still judges the returned point on the true ML gradient.
  return -grad;
}

fit_expected<Estimates>
compose_fisher_ml(const Prelude& pre, const SampleStats& samp,
                  const Eigen::VectorXd& x0, const Bounds& bounds,
                  OptimOptions opts) {
  const model::ModelEvaluator& ev = pre.ev;

  auto obj_or = estimate::ml_objective(ev, samp);
  if (!obj_or.has_value()) return std::unexpected(obj_or.error());
  const optim::ScalarProblem ml_prob = std::move(*obj_or);

  if (pre.nl.active()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_ml_fisher: nonlinear equality constraints are not supported by "
        "the Fisher scoring path; use fit_ml with optimizer \"nlopt-slsqp\" "
        "or \"ipopt\""));
  }

  const EqConstraints& con = pre.con;
  const bool driven_reduced = con.active();
  const bool pure_merge = !con.group.empty();
  if (driven_reduced && !bounds.empty() && !pure_merge) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_ml_fisher: box bounds with general-linear equality constraints "
        "are not supported by the Fisher scoring path"));
  }

  Eigen::VectorXd driven = driven_reduced ? con.contract(x0) : x0;
  Bounds driven_bounds;
  if (!bounds.empty()) {
    driven_bounds = driven_reduced ? fold_alpha_bounds(con, bounds) : bounds;
    driven = driven.cwiseMax(driven_bounds.lower).cwiseMin(driven_bounds.upper);
  }

  auto expand_driven = [&con, driven_reduced](const Eigen::VectorXd& z) {
    return driven_reduced ? con.expand(z) : z;
  };
  auto reduce_gradient = [&con, driven_reduced](const Eigen::VectorXd& g) {
    return driven_reduced ? con.reduce_gradient(g) : g;
  };
  auto reduce_hessian = [&con, driven_reduced](const Eigen::MatrixXd& H) {
    return driven_reduced ? Eigen::MatrixXd(con.K().transpose() * H * con.K())
                          : H;
  };
  auto inside_bounds = [&](const Eigen::VectorXd& z) {
    if (driven_bounds.empty()) return true;
    constexpr double tol = 1e-12;
    for (Eigen::Index i = 0; i < z.size(); ++i) {
      if (std::isfinite(driven_bounds.lower[i]) &&
          z[i] < driven_bounds.lower[i] - tol) return false;
      if (std::isfinite(driven_bounds.upper[i]) &&
          z[i] > driven_bounds.upper[i] + tol) return false;
    }
    return true;
  };
  auto proj_grad_inf = [&](const Eigen::VectorXd& z,
                           const Eigen::VectorXd& g) {
    if (g.size() == 0) return 0.0;
    if (driven_bounds.empty()) return g.cwiseAbs().maxCoeff();
    constexpr double kBoundTol = 1e-12;
    double m = 0.0;
    for (Eigen::Index i = 0; i < z.size(); ++i) {
      double gi = g[i];
      const bool at_lo = std::isfinite(driven_bounds.lower[i]) &&
                         (z[i] - driven_bounds.lower[i] <= kBoundTol);
      const bool at_up = std::isfinite(driven_bounds.upper[i]) &&
                         (driven_bounds.upper[i] - z[i] <= kBoundTol);
      if (at_lo && gi > 0.0) gi = 0.0;
      if (at_up && gi < 0.0) gi = 0.0;
      m = std::max(m, std::abs(gi));
    }
    return m;
  };

  Eigen::VectorXd theta = expand_driven(driven);
  Eigen::VectorXd grad_full = Eigen::VectorXd::Zero(theta.size());
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(driven.size());
  double f_cur = ml_prob.f(theta, grad_full);
  if (!std::isfinite(f_cur)) {
    return std::unexpected(fit_err(FitError::Kind::NonFiniteObjective,
        "fit_ml_fisher: F_ML(x0) is non-finite"));
  }
  grad = reduce_gradient(grad_full);

  int total_f_evals = 1;
  int total_g_evals = 1;
  int iter = 0;
  optim::OptimStatus final_status = optim::OptimStatus::BudgetExhausted;
  constexpr double armijo_c = 1e-4;
  constexpr int max_backtracks = 40;

  for (int k = 0; k < opts.max_iter; ++k) {
    if (proj_grad_inf(driven, grad) <= opts.gtol) {
      final_status = optim::OptimStatus::Converged;
      break;
    }

    auto H_or = expected_ml_hessian_f(ev, samp, theta);
    if (!H_or.has_value()) return std::unexpected(H_or.error());
    const Eigen::MatrixXd H_driven = reduce_hessian(*H_or);
    Eigen::VectorXd d = solve_fisher_direction(H_driven, grad);
    if (!d.allFinite() || d.size() != driven.size()) {
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "fit_ml_fisher: Fisher direction is non-finite"));
    }
    const double slope = grad.dot(d);
    if (!(slope < 0.0)) {
      d = -grad;
    }

    double alpha = 1.0;
    Eigen::VectorXd driven_new(driven.size());
    Eigen::VectorXd theta_new(theta.size());
    Eigen::VectorXd grad_full_new = Eigen::VectorXd::Zero(theta.size());
    Eigen::VectorXd grad_new = Eigen::VectorXd::Zero(driven.size());
    double f_new = std::numeric_limits<double>::infinity();
    bool armijo_ok = false;
    const double descent_slope = grad.dot(d);

    for (int bt = 0; bt < max_backtracks; ++bt) {
      driven_new = driven + alpha * d;
      if (!inside_bounds(driven_new)) {
        alpha *= 0.5;
        continue;
      }
      theta_new = expand_driven(driven_new);
      f_new = ml_prob.f(theta_new, grad_full_new);
      ++total_f_evals;
      ++total_g_evals;
      if (std::isfinite(f_new) &&
          f_new <= f_cur + armijo_c * alpha * descent_slope) {
        grad_new = reduce_gradient(grad_full_new);
        armijo_ok = true;
        break;
      }
      alpha *= 0.5;
    }

    ++iter;
    if (!armijo_ok) {
      final_status = optim::OptimStatus::LineSearchSalvaged;
      break;
    }

    const double df = std::abs(f_cur - f_new);
    driven = driven_new;
    theta = theta_new;
    grad = grad_new;
    f_cur = f_new;

    if (proj_grad_inf(driven, grad) <= opts.gtol) {
      final_status = optim::OptimStatus::Converged;
      break;
    }
    if (df <= opts.ftol * (std::abs(f_cur) + opts.ftol) &&
        proj_grad_inf(driven, grad) <= 100.0 * opts.gtol) {
      final_status = optim::OptimStatus::LineSearchSalvaged;
      break;
    }
  }

  Eigen::VectorXd lower, upper;
  if (driven_bounds.empty()) {
    lower = Eigen::VectorXd::Constant(driven.size(),
                                      -std::numeric_limits<double>::infinity());
    upper = Eigen::VectorXd::Constant(driven.size(),
                                       std::numeric_limits<double>::infinity());
  } else {
    lower = driven_bounds.lower;
    upper = driven_bounds.upper;
  }
  optim::ObjectiveFn driven_ml =
      [&](const Eigen::VectorXd& z, Eigen::VectorXd& g) -> double {
        const Eigen::VectorXd th = expand_driven(z);
        Eigen::VectorXd g_full(th.size());
        const double v = ml_prob.f(th, g_full);
        g = reduce_gradient(g_full);
        return v;
      };
  optim::TerminalAudit audit =
      optim::audit_terminal_iterate(driven_ml, driven, f_cur, lower, upper);
  if (final_status == optim::OptimStatus::BudgetExhausted &&
      audit.stationary) {
    final_status = optim::OptimStatus::Converged;
  }

  Estimates est{theta, f_cur, iter, total_f_evals, total_g_evals,
                final_status, audit.grad_inf_norm, std::move(audit)};
  attach_diagnostics(est, pre, bounds);
  return est;
}

// Strategy for the inner subproblem inside the IRLS outer loop. `FullGls`
// hands the reweighted GLS problem to `compose_gmm` (the full-θ path);
// `SnllsGls` profiles out the conditionally-linear block via `compose_snlls`
// (Golub–Pereyra variable projection) and the outer optimizer sees only β.
// Both reach the same fixed point on separable models; SNLLS reduces the
// inner dim and the per-iteration cost.
enum class IrlsInner { FullGls, SnllsGls };

fit_expected<SampleStats>
irls_inner_sample_stats(const model::ModelEvaluator& ev,
                        const SampleStats& samp,
                        const Eigen::VectorXd& theta) {
  auto eval = ev.evaluate(theta, false, false);
  if (!eval.has_value()) {
    return std::unexpected(
        fit_err(FitError::Kind::NonPositiveDefiniteSigma,
                "compose_irls_outer: mean-adjustment theta: " +
                    eval.error().detail));
  }

  SampleStats out = samp;
  for (std::size_t b = 0; b < out.S.size(); ++b) {
    if (b >= samp.mean.size() || b >= eval->moments.mu.size()) continue;
    if (samp.mean[b].size() == 0 || eval->moments.mu[b].size() == 0) continue;
    if (samp.mean[b].size() != eval->moments.mu[b].size()) {
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "compose_irls_outer: block " + std::to_string(b) +
              " sample mean and implied mu have different sizes"));
    }
    const Eigen::VectorXd d = samp.mean[b] - eval->moments.mu[b];
    out.S[b] = samp.S[b] + d * d.transpose();
  }
  return out;
}

// Shared IRLS-ML outer loop: refresh the expected Fisher information
// weight at each iterate, dispatch the inner step to `compose_gmm` or
// `compose_snlls` per `inner`, Armijo-backtrack on F_ML, exit on
// projected-gradient stationarity or gradient-gated stagnation. `pre` is
// the caller's prelude (evaluator + linear/nonlinear constraint shells);
// `pt` is forwarded so the SNLLS path can hand it to `compose_snlls` for
// the GP profile.
fit_expected<Estimates>
compose_irls_outer(const spec::LatentStructure& pt, const Prelude& pre,
                   const SampleStats& samp, const Eigen::VectorXd& x0,
                   const Bounds& bounds, Backend backend, OptimOptions opts,
                   IrlsOptions irls_opts, IrlsInner inner) {
  const model::ModelEvaluator& ev = pre.ev;

  auto obj_or = estimate::ml_objective(ev, samp);
  if (!obj_or.has_value()) return std::unexpected(obj_or.error());
  const optim::ScalarProblem ml_prob = std::move(*obj_or);

  if (pre.nl.active()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_ml_irls: nonlinear equality constraints are not supported by "
        "the IRLS outer loop; use fit_ml with optimizer \"nlopt-slsqp\" or "
        "\"ipopt\""));
  }
  if (inner == IrlsInner::SnllsGls && !bounds.empty()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_ml_irls_snlls: box bounds are not supported by the separable "
        "Golub-Pereyra inner path"));
  }

  const EqConstraints& con = pre.con;
  const bool driven_reduced = con.active();
  const bool pure_merge = !con.group.empty();
  if (driven_reduced && !bounds.empty() && !pure_merge) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_ml_irls: box bounds with general-linear equality constraints "
        "are not supported by the IRLS outer loop"));
  }

  Eigen::VectorXd driven = driven_reduced ? con.contract(x0) : x0;
  Bounds driven_bounds;
  if (!bounds.empty()) {
    driven_bounds = driven_reduced ? fold_alpha_bounds(con, bounds) : bounds;
    driven = driven.cwiseMax(driven_bounds.lower).cwiseMin(driven_bounds.upper);
  }

  auto expand_driven = [&con, driven_reduced](const Eigen::VectorXd& z) {
    return driven_reduced ? con.expand(z) : z;
  };
  auto reduce_gradient = [&con, driven_reduced](const Eigen::VectorXd& g) {
    return driven_reduced ? con.reduce_gradient(g) : g;
  };

  Eigen::VectorXd theta = expand_driven(driven);
  Eigen::VectorXd grad_full = Eigen::VectorXd::Zero(theta.size());
  Eigen::VectorXd grad      = Eigen::VectorXd::Zero(driven.size());
  double f_cur = ml_prob.f(theta, grad_full);
  if (!std::isfinite(f_cur)) {
    return std::unexpected(fit_err(FitError::Kind::NonFiniteObjective,
        "compose_irls_outer: F_ML(x0) is non-finite"));
  }
  grad = reduce_gradient(grad_full);

  // Projected-gradient infinity norm on F_ML, matching `audit_terminal_iterate`
  // KKT semantics: a coordinate at a finite bound contributes only when its
  // gradient pushes inward.
  auto proj_grad_inf = [&](const Eigen::VectorXd& z,
                           const Eigen::VectorXd& g) {
    if (g.size() == 0) return 0.0;
    if (driven_bounds.empty()) return g.cwiseAbs().maxCoeff();
    constexpr double kBoundTol = 1e-12;
    double m = 0.0;
    for (Eigen::Index i = 0; i < z.size(); ++i) {
      double     gi = g[i];
      const bool at_lo = std::isfinite(driven_bounds.lower[i]) &&
                         (z[i] - driven_bounds.lower[i] <= kBoundTol);
      const bool at_up = std::isfinite(driven_bounds.upper[i]) &&
                         (driven_bounds.upper[i] - z[i] <= kBoundTol);
      if (at_lo && gi > 0.0) gi = 0.0;
      if (at_up && gi < 0.0) gi = 0.0;
      m = std::max(m, std::abs(gi));
    }
    return m;
  };

  int outer_iter   = 0;
  int total_f_evals = 1;  // counted the initial value/gradient evaluation
  int total_g_evals = 1;
  int stagnation   = 0;   // consecutive outer steps with df below ftol
  // SNLLS-path telemetry — carried out of the last successful inner call.
  std::int32_t snlls_n_nonlinear         = -1;
  std::int32_t snlls_n_linear            = -1;
  std::int32_t snlls_n_alpha_fast        = -1;
  std::int32_t snlls_n_alpha_fallback    = -1;
  optim::OptimStatus final_status = optim::OptimStatus::BudgetExhausted;

  for (int k = 0; k < irls_opts.max_outer; ++k) {
    // Stationarity check on F_ML at the current iterate — the primary
    // convergence criterion. ‖proj-∇F_ML‖_∞ ≤ gtol mirrors lavaan's
    // `check.gradient = TRUE` post-fit audit, applied as a loop guard.
    if (proj_grad_inf(driven, grad) <= irls_opts.gtol) {
      final_status = optim::OptimStatus::Converged;
      break;
    }

    // Refresh the expected Fisher information weight W(θ_k) at the current
    // iterate, then solve the inner subproblem to convergence with that
    // weight frozen. `expected_information_weight` builds W from Σ(θ_k) —
    // distinct from `normal_theory_weight`, which uses S and stays constant
    // across iterations (i.e. would collapse IRLS to one-shot GLS).
    auto inner_samp_or = irls_inner_sample_stats(ev, samp, theta);
    if (!inner_samp_or.has_value()) return std::unexpected(inner_samp_or.error());
    const SampleStats& inner_samp = *inner_samp_or;

    auto W_or = gmm::expected_information_weight(ev, inner_samp, theta);
    if (!W_or.has_value()) return std::unexpected(W_or.error());

    fit_expected<Estimates> inner_or = std::unexpected(
        fit_err(FitError::Kind::NumericIssue,
                "compose_irls_outer: unreachable inner-strategy dispatch"));
    switch (inner) {  // dispatch on the inner-step strategy
      case IrlsInner::FullGls:
        inner_or = compose_gmm(ev, pre.con, pre.nl, inner_samp, theta, *W_or,
                               bounds, backend, opts);
        break;
      case IrlsInner::SnllsGls:
        // Golub–Pereyra path. Profiles α (Θ, Ψ, ν) out, optimizes only β
        // (Λ, B); `compose_snlls` returns full-θ via the profile's expand.
        // Box bounds are rejected before entering the loop; the GP β block is
        // unbounded, same as `fit_snlls_gls`.
        inner_or = compose_snlls(pt, ev, inner_samp, theta, *W_or, backend,
                                 opts);
        break;
    }
    if (!inner_or.has_value()) return std::unexpected(inner_or.error());
    const Eigen::VectorXd& theta_trial  = inner_or->theta;
    const Eigen::VectorXd  driven_trial =
        driven_reduced ? con.contract(theta_trial) : theta_trial;
    total_f_evals += inner_or->f_evals;
    total_g_evals += inner_or->g_evals;
    if (inner_or->n_nonlinear >= 0) {
      snlls_n_nonlinear      = inner_or->n_nonlinear;
      snlls_n_linear         = inner_or->n_linear;
      snlls_n_alpha_fast     = inner_or->n_alpha_solve_fast;
      snlls_n_alpha_fallback = inner_or->n_alpha_solve_fallback;
    }

    // Armijo backtracking on F_ML along d = θ_trial − θ_k. The inner step
    // is a descent direction for F_ML at θ_k by construction (the two
    // gradients agree there up to a factor of two); the line search guards
    // against overshoot away from the optimum.
    const Eigen::VectorXd d       = driven_trial - driven;
    const double          g_dot_d = grad.dot(d);
    double                alpha   = 1.0;
    Eigen::VectorXd       driven_new(driven.size());
    Eigen::VectorXd       theta_new(theta.size());
    Eigen::VectorXd       grad_full_new = Eigen::VectorXd::Zero(theta.size());
    Eigen::VectorXd       grad_new = Eigen::VectorXd::Zero(driven.size());
    double                f_new   = std::numeric_limits<double>::infinity();
    bool                  armijo_ok = false;
    // If `g·d ≥ 0` the analytic gradient at θ_k disagrees with the inner
    // step direction (numerical noise near a stationary point, or an inner
    // solver that didn't fully converge); fall back to a non-descent-aware
    // simple-decrease test so we can still accept a step that lowers F_ML.
    const double slope = (g_dot_d < 0.0) ? g_dot_d : 0.0;
    for (int bt = 0; bt < irls_opts.armijo_max_backtracks; ++bt) {
      driven_new = driven + alpha * d;
      theta_new  = expand_driven(driven_new);
      f_new      = ml_prob.f(theta_new, grad_full_new);
      ++total_f_evals;
      ++total_g_evals;
      if (std::isfinite(f_new) &&
          f_new <= f_cur + irls_opts.armijo_c * alpha * slope) {
        grad_new = reduce_gradient(grad_full_new);
        armijo_ok = true;
        break;
      }
      alpha *= 0.5;
    }

    ++outer_iter;

    if (!armijo_ok) {
      // Could not find a sufficient-decrease step at any α down to
      // 2^{-armijo_max_backtracks}. Keep the current iterate.
      final_status = optim::OptimStatus::LineSearchSalvaged;
      break;
    }

    const double df = std::abs(f_cur - f_new);
    driven = driven_new;
    theta = theta_new;
    grad  = grad_new;
    f_cur = f_new;

    if (proj_grad_inf(driven, grad) <= irls_opts.gtol) {
      final_status = optim::OptimStatus::Converged;
      break;
    }

    // Stagnation detector. The F-relative test alone fires falsely after a
    // heavily-backtracked Armijo step (tiny α ⇒ tiny df even at a non-
    // stationary iterate); pair it with the gradient norm. Only count a step
    // as stagnant when *both* the relative F-change is below ftol *and* the
    // projected gradient norm is within a relative slack of `gtol` —
    // otherwise small df reflects a heavily-damped step at a still-non-
    // stationary iterate (keep iterating, the next reweight may help) rather
    // than convergence. Five consecutive stagnant steps bail as
    // LineSearchSalvaged.
    const double rel_gtol_slack = 100.0 * irls_opts.gtol;
    if (df <= irls_opts.ftol * (std::abs(f_cur) + irls_opts.ftol) &&
        proj_grad_inf(driven, grad) <= rel_gtol_slack) {
      ++stagnation;
      if (stagnation >= 5) {
        final_status = optim::OptimStatus::LineSearchSalvaged;
        break;
      }
    } else {
      stagnation = 0;
    }
  }

  // Terminal audit on F_ML at the returned iterate, in the same driven
  // coordinate system the outer loop used (full θ when unconstrained,
  // equality-reduced α when linear constraints are active).
  Eigen::VectorXd lower, upper;
  if (driven_bounds.empty()) {
    lower = Eigen::VectorXd::Constant(driven.size(),
                                      -std::numeric_limits<double>::infinity());
    upper = Eigen::VectorXd::Constant(driven.size(),
                                       std::numeric_limits<double>::infinity());
  } else {
    lower = driven_bounds.lower;
    upper = driven_bounds.upper;
  }
  optim::ObjectiveFn driven_ml =
      [&](const Eigen::VectorXd& z, Eigen::VectorXd& g) -> double {
        const Eigen::VectorXd th = expand_driven(z);
        Eigen::VectorXd g_full(th.size());
        const double v = ml_prob.f(th, g_full);
        g = reduce_gradient(g_full);
        return v;
      };
  optim::TerminalAudit audit =
      optim::audit_terminal_iterate(driven_ml, driven, f_cur, lower, upper);
  if (final_status == optim::OptimStatus::BudgetExhausted &&
      audit.stationary) {
    final_status = optim::OptimStatus::Converged;
  }

  Estimates est{theta, f_cur, outer_iter, total_f_evals, total_g_evals,
                final_status, audit.grad_inf_norm, std::move(audit)};
  est.n_nonlinear            = snlls_n_nonlinear;
  est.n_linear               = snlls_n_linear;
  est.n_alpha_solve_fast     = snlls_n_alpha_fast;
  est.n_alpha_solve_fallback = snlls_n_alpha_fallback;
  attach_diagnostics(est, pre, bounds);
  return est;
}

}  // namespace

fit_expected<Estimates>
fit_ml_irls(spec::LatentStructure pt, const model::MatrixRep& rep,
            const SampleStats& samp, const Eigen::VectorXd& x0, Bounds bounds,
            Backend backend, OptimOptions opts, IrlsOptions irls_opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_ml_irls");
  if (!pre.has_value()) return std::unexpected(pre.error());
  return compose_irls_outer(pt, *pre, samp, x0, bounds, backend, opts,
                            irls_opts, IrlsInner::FullGls);
}

fit_expected<Estimates>
fit_ml_irls_snlls(spec::LatentStructure pt, const model::MatrixRep& rep,
                  const SampleStats& samp, const Eigen::VectorXd& x0,
                  Bounds bounds, Backend backend, OptimOptions opts,
                  IrlsOptions irls_opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_ml_irls_snlls");
  if (!pre.has_value()) return std::unexpected(pre.error());
  return compose_irls_outer(pt, *pre, samp, x0, bounds, backend, opts,
                            irls_opts, IrlsInner::SnllsGls);
}

fit_expected<Estimates>
fit_snlls(spec::LatentStructure pt, const model::MatrixRep& rep,
          const SampleStats& samp, const Eigen::VectorXd& x0,
          gmm::Weight weight, Backend backend, OptimOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_snlls");
  if (!pre.has_value()) return std::unexpected(pre.error());
  auto est = compose_snlls(pt, pre->ev, samp, x0, weight, backend, opts);
  if (!est.has_value()) return est;
  // SNLLS has no box bounds on the nonlinear block; attach_diagnostics
  // reads `bounds.empty()` correctly and reports no active bounds.
  attach_diagnostics(*est, *pre, Bounds{});
  return est;
}

fit_expected<Estimates>
fit_snlls_gls(spec::LatentStructure pt, const model::MatrixRep& rep,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              Backend backend, OptimOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_snlls_gls");
  if (!pre.has_value()) return std::unexpected(pre.error());
  auto W = gmm::normal_theory_weight(pre->ev, samp, x0);
  if (!W.has_value()) return std::unexpected(W.error());
  auto est = compose_snlls(pt, pre->ev, samp, x0, *W, backend, opts);
  if (!est.has_value()) return est;
  attach_diagnostics(*est, *pre, Bounds{});
  return est;
}

}  // namespace magmaan::estimate
