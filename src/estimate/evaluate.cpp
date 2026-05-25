#include "magmaan/estimate/evaluate.hpp"

#include <string>
#include <utility>

#include <Eigen/Core>

#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/optimizers.hpp"        // optim::scalarize
#include "magmaan/optim/terminal_audit.hpp"

// `evaluate_at` is the no-optimizer twin of the `fit_*` composers: it runs
// the same KKT-projected-gradient audit at a θ supplied from outside (e.g.
// lavaan's `theta_hat`) instead of at the optimizer's terminal iterate.
//
// The composition reuses the same builders the fit path uses
// (`gmm::residuals` + `optim::scalarize`, or `nt::ml_objective`), so the
// objective the audit recomputes is byte-identical to what `fit_uls /
// fit_gls / fit_wls / fit_ml` would have minimised on the same model.

namespace magmaan::estimate {

namespace {

FitError fit_err(FitError::Kind kind, std::string detail) {
  return FitError{kind, std::move(detail), 0, 0.0};
}

// Shared prelude: resolve fixed.x, build the evaluator, build the
// linear-equality and nonlinear-equality constraint structures. Diagnostics
// reads these; the audit ignores them (it lives in driven coordinates and
// the caller's θ is in full coordinates already).
struct EvalPrelude {
  model::ModelEvaluator  ev;
  EqConstraints          con;
  NonlinearEqConstraints nl;
};

fit_expected<EvalPrelude>
make_prelude(spec::LatentStructure& pt, const model::MatrixRep& rep,
             const SampleStats& samp) {
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(e.error());
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::InvalidStartValues,
        "evaluate_at: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto con_or = build_eq_constraints(pt, /*allow_nonlinear=*/true);
  if (!con_or.has_value()) {
    return std::unexpected(fit_err(FitError::Kind::NumericIssue,
        "evaluate_at: constraint: " + con_or.error().detail));
  }
  return EvalPrelude{std::move(*ev_or), std::move(*con_or),
                     build_nl_constraints(pt)};
}

// Build the scalar objective for the chosen moment estimator. ULS/GLS/WLS go
// through `gmm::residuals` → `optim::scalarize`; ML uses `nt::ml_objective`
// directly. The returned `ScalarProblem` carries an `ObjectiveFn` closure
// the audit can call to recompute F and ∇F at `theta_full`.
fit_expected<optim::ScalarProblem>
build_scalar_objective(const model::ModelEvaluator& ev,
                       const SampleStats& samp,
                       const Eigen::VectorXd& theta_full,
                       Estimator estimator, const gmm::Weight& weight) {
  if (estimator == Estimator::ML) {
    return ml_objective(ev, samp);
  }
  gmm::Weight w_to_use;
  if (estimator == Estimator::ULS) {
    // Empty weight ⇒ identity.
  } else if (estimator == Estimator::GLS) {
    auto w_or = gmm::normal_theory_weight(ev, samp, theta_full);
    if (!w_or.has_value()) return std::unexpected(w_or.error());
    w_to_use = std::move(*w_or);
  } else {  // WLS
    if (weight.empty()) {
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "evaluate_at: estimator = WLS requires a non-empty `weight` (call "
          "`normal_theory_weight` for GLS, supply a Browne-1984 fourth-"
          "moment matrix for WLS, or use Estimator::ULS for unit weights)"));
    }
    w_to_use = weight;
  }
  auto gmm_or = gmm::residuals(ev, samp, theta_full, w_to_use);
  if (!gmm_or.has_value()) return std::unexpected(gmm_or.error());
  return optim::scalarize(*gmm_or);
}

}  // namespace

fit_expected<Estimates>
evaluate_at(spec::LatentStructure pt, const model::MatrixRep& rep,
            const SampleStats& samp, const Eigen::VectorXd& theta_full,
            Estimator estimator, const gmm::Weight& weight,
            Bounds bounds, optim::TerminalAuditOptions audit_opts) {
  // Up-front contract: theta_full lives in the full-θ space.
  if (theta_full.size() != static_cast<Eigen::Index>(pt.n_free())) {
    return std::unexpected(fit_err(FitError::Kind::InvalidStartValues,
        "evaluate_at: theta_full size (" + std::to_string(theta_full.size()) +
            ") != pt.n_free() (" + std::to_string(pt.n_free()) + ")"));
  }

  auto prelude_or = make_prelude(pt, rep, samp);
  if (!prelude_or.has_value()) return std::unexpected(prelude_or.error());
  auto& prelude = *prelude_or;

  auto obj_or = build_scalar_objective(prelude.ev, samp, theta_full,
                                       estimator, weight);
  if (!obj_or.has_value()) return std::unexpected(obj_or.error());
  const optim::ScalarProblem prob = std::move(*obj_or);

  // Recompute F and ∇F at theta_full. The audit will recompute internally
  // too, but caching the value here lets us pass it as `reported_f` and gives
  // a guaranteed-finite gradient slot for the caller via the audit struct.
  Eigen::VectorXd grad(theta_full.size());
  grad.setZero();
  const double f_at = prob.f(theta_full, grad);

  // Resolve bounds. Default = `variance_bounds(pt)` (lavaan's `pos.var`
  // preset): no sample-stats dependency, prevents the audit from
  // misjudging boundary variance coordinates whose gradient would push θ
  // further into the inadmissible negative-variance region.
  if (bounds.empty()) {
    auto vb_or = variance_bounds(pt);
    if (!vb_or.has_value()) {
      return std::unexpected(fit_err(FitError::Kind::NumericIssue,
          "evaluate_at: variance_bounds: " + vb_or.error().detail));
    }
    bounds = std::move(*vb_or);
  } else if (bounds.lower.size() != theta_full.size() ||
             bounds.upper.size() != theta_full.size()) {
    return std::unexpected(fit_err(FitError::Kind::InvalidStartValues,
        "evaluate_at: bounds size mismatch (lower=" +
            std::to_string(bounds.lower.size()) + ", upper=" +
            std::to_string(bounds.upper.size()) + ", expected " +
            std::to_string(theta_full.size()) + ")"));
  }

  optim::TerminalAudit audit = optim::audit_terminal_iterate(
      prob.f, theta_full, /*reported_f=*/f_at,
      bounds.lower, bounds.upper, audit_opts);

  FitDiagnostics diagnostics = finalize_fit_diagnostics(
      theta_full, prelude.ev, prelude.con, prelude.nl, bounds,
      /*snlls_profile_fallback_flag=*/false);

  // `iterations = 0`, `f_evals = 1`, `g_evals = 1` are the documented
  // "no outer optimizer ran" defaults from fit.hpp, with `f_evals`/`g_evals`
  // bumped to reflect the single objective+gradient evaluation we did do.
  // `optimizer_status = Converged` is a sentinel — `audit.advisory_status`
  // is the authoritative verdict for callers.
  Estimates out;
  out.theta            = theta_full;
  out.fmin             = f_at;
  out.iterations       = 0;
  out.f_evals          = 1;
  out.g_evals          = 1;
  out.optimizer_status = optim::OptimStatus::Converged;
  out.grad_inf_norm    = audit.grad_inf_norm;
  out.audit            = std::move(audit);
  out.diagnostics      = std::move(diagnostics);
  return out;
}

}  // namespace magmaan::estimate
