#include "magmaan/estimate/fit.hpp"

#include <string>
#include <utility>

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
#include "magmaan/spec/partable.hpp"

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
    return Estimates{prob.expand(Eigen::VectorXd(0)),
                     0.5 * r->squaredNorm(), 0};
  }

  // The nonlinear block (Λ, Β) carries no box bounds, so β is optimized
  // unbounded; variance non-negativity lives on the profiled-out α.
  auto out = run_gmm(prob, gp_or->beta0, Bounds{}, backend, opts);
  if (!out.has_value()) return std::unexpected(out.error());
  return Estimates{prob.expand(out->x), out->fmin, out->iterations,
                   out->f_evals, out->g_evals,
                   out->status, out->grad_inf_norm,
                   std::move(out->audit)};
}

}  // namespace

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
