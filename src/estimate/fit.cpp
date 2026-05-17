#include "magmaan/estimate/fit.hpp"

#include <limits>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/reparameterize.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/gmm/gp.hpp"
#include "magmaan/gmm/moment_quadratic.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/nt/ml.hpp"
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

// Fold per-θ box bounds onto the constraint-reduced α for a pure-merge
// reparameterization (each θ_k is a copy of α_{group[k]}).
Bounds fold_alpha_bounds(const EqConstraints& con, const Bounds& b) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const Eigen::Index na = con.Kmat.cols();
  Bounds out;
  out.lower = Eigen::VectorXd::Constant(na, -kInf);
  out.upper = Eigen::VectorXd::Constant(na, kInf);
  for (Eigen::Index k = 0; k < b.lower.size(); ++k) {
    const auto g = static_cast<Eigen::Index>(con.group[static_cast<std::size_t>(k)]);
    out.lower(g) = std::max(out.lower(g), b.lower(k));
    out.upper(g) = std::min(out.upper(g), b.upper(k));
  }
  return out;
}

// Dispatch a scalar-objective optimization by backend. Shared by `fit_ml` and
// (via `scalarize`) the gmm/gls path. `Backend::Ceres` has no scalar entry
// point — it is a least-squares-only backend — and is rejected here.
fit_expected<optim::OptimResult>
run_scalar(const optim::ScalarProblem& prob, const Eigen::VectorXd& x0,
           const Bounds& bounds, Backend backend, LbfgsOptions opts) {
  switch (backend) {
    case Backend::Nlopt:
#ifdef MAGMAAN_WITH_NLOPT
      return optim::nlopt_slsqp(prob, x0, bounds, opts);
#else
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "Nlopt backend requested but MAGMAAN_WITH_NLOPT is off"));
#endif
    case Backend::TrustRegion:
      if (!bounds.empty()) {
        return std::unexpected(fit_err(FitError::Kind::NumericIssue,
            "TrustRegion backend is unbounded and cannot honor box bounds — "
            "supply empty bounds, or use the L-BFGS / NLopt backend"));
      }
      return optim::trust_region(prob, x0, opts);
    case Backend::Ceres:
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "Ceres backend applies only to the least-squares path "
          "(fit_gmm / fit_gls); it has no scalar-objective entry point"));
    case Backend::Lbfgs:
      break;
  }
  return optim::lbfgs(prob, x0, bounds, opts);
}

fit_expected<optim::OptimResult>
run_gmm(const optim::GmmProblem& prob, const Eigen::VectorXd& x0,
        const Bounds& bounds, Backend backend, LbfgsOptions opts) {
  if (backend == Backend::Ceres) {
#ifdef MAGMAAN_WITH_CERES
    optim::CeresOptions copts;
    copts.max_iter = opts.max_iter;
    copts.ftol     = opts.ftol;
    copts.gtol     = opts.gtol;
    return optim::ceres_lm(prob, x0, bounds, copts);
#else
    (void)opts;
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "fit_gmm: Ceres backend requested but MAGMAAN_WITH_CERES is off"));
#endif
  }
  return run_scalar(optim::scalarize(prob), x0, bounds, backend, opts);
}

// Shared prelude — resolve fixed.x, build the evaluator, build constraints.
struct Prelude {
  model::ModelEvaluator ev;
  EqConstraints         con;
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
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        std::string(who) + ": constraint: " + con_or.error().detail));
  }
  return Prelude{std::move(*ev_or), std::move(*con_or)};
}

}  // namespace

namespace {

// Shared moment-quadratic composition: build the LS problem, fold equality
// constraints, optimize, expand. `fit_gmm` and `fit_gls` differ only in how
// the weight is produced.
fit_expected<Estimates>
compose_gmm(const model::ModelEvaluator& ev, const EqConstraints& con,
            const SampleStats& samp, const Eigen::VectorXd& x0,
            const gmm::Weight& weight, const Bounds& bounds, Backend backend,
            LbfgsOptions opts) {
  auto prob_or = gmm::residuals(ev, samp, x0, weight);
  if (!prob_or.has_value()) return std::unexpected(prob_or.error());
  const optim::GmmProblem prob = std::move(*prob_or);

  if (!con.active()) {
    auto out = run_gmm(prob, x0, bounds, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    return Estimates{prob.expand(out->x), out->fmin, out->iterations};
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
  return Estimates{std::move(theta_hat), out->fmin, out->iterations};
}

}  // namespace

fit_expected<Estimates>
fit_gmm(spec::LatentStructure pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        gmm::Weight weight, Bounds bounds, Backend backend,
        LbfgsOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_gmm");
  if (!pre.has_value()) return std::unexpected(pre.error());
  return compose_gmm(pre->ev, pre->con, samp, x0, weight, bounds, backend,
                     opts);
}

fit_expected<Estimates>
fit_gls(spec::LatentStructure pt, const model::MatrixRep& rep,
        const SampleStats& samp, const Eigen::VectorXd& x0,
        Bounds bounds, Backend backend, LbfgsOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_gls");
  if (!pre.has_value()) return std::unexpected(pre.error());
  auto W = gmm::normal_theory_weight(pre->ev, samp, x0);
  if (!W.has_value()) return std::unexpected(W.error());
  return compose_gmm(pre->ev, pre->con, samp, x0, *W, bounds, backend, opts);
}

fit_expected<Estimates>
fit_ml(spec::LatentStructure pt, const model::MatrixRep& rep,
       const SampleStats& samp, const Eigen::VectorXd& x0, Bounds bounds,
       Backend backend, LbfgsOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_ml");
  if (!pre.has_value()) return std::unexpected(pre.error());
  const model::ModelEvaluator& ev = pre->ev;
  const EqConstraints& con = pre->con;

  auto obj_or = nt::ml_objective(ev, samp);
  if (!obj_or.has_value()) return std::unexpected(obj_or.error());
  const optim::ScalarProblem prob = std::move(*obj_or);

  if (!con.active()) {
    auto out = run_scalar(prob, x0, bounds, backend, opts);
    if (!out.has_value()) return std::unexpected(out.error());
    return Estimates{prob.expand(out->x), out->fmin, out->iterations};
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
  return Estimates{prob_a.expand(out->x), out->fmin, out->iterations};
}

namespace {

// Shared Golub–Pereyra composition: profile out the linear block, optimize the
// nonlinear β. `fit_snlls` and `fit_snlls_gls` differ only in the weight.
fit_expected<Estimates>
compose_snlls(const spec::LatentStructure& pt, const model::ModelEvaluator& ev,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              const gmm::Weight& weight, Backend backend, LbfgsOptions opts) {
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
  return Estimates{prob.expand(out->x), out->fmin, out->iterations};
}

}  // namespace

fit_expected<Estimates>
fit_snlls(spec::LatentStructure pt, const model::MatrixRep& rep,
          const SampleStats& samp, const Eigen::VectorXd& x0,
          gmm::Weight weight, Backend backend, LbfgsOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_snlls");
  if (!pre.has_value()) return std::unexpected(pre.error());
  return compose_snlls(pt, pre->ev, samp, x0, weight, backend, opts);
}

fit_expected<Estimates>
fit_snlls_gls(spec::LatentStructure pt, const model::MatrixRep& rep,
              const SampleStats& samp, const Eigen::VectorXd& x0,
              Backend backend, LbfgsOptions opts) {
  auto pre = prelude(pt, rep, samp, x0, "fit_snlls_gls");
  if (!pre.has_value()) return std::unexpected(pre.error());
  auto W = gmm::normal_theory_weight(pre->ev, samp, x0);
  if (!W.has_value()) return std::unexpected(W.error());
  return compose_snlls(pt, pre->ev, samp, x0, *W, backend, opts);
}

}  // namespace magmaan::estimate
