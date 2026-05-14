#pragma once

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/constraints.hpp"
#include "magmaan/fit/fit.hpp"
#include "magmaan/fit/fit_measures.hpp"
#include "magmaan/fit/lbfgs_optimizer.hpp"
#include "magmaan/fit/raw_data.hpp"
#include "magmaan/fit/resolve_fixed_x.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/fit/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/partable/partable.hpp"
#include "magmaan/partable/start_hints.hpp"

namespace magmaan::fit {

struct FIMLPattern {
  std::size_t block = 0;
  std::vector<Eigen::Index> observed;
  std::int64_t n_obs = 0;
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
};

struct FIMLCache {
  std::vector<FIMLPattern> patterns;
  std::vector<Eigen::Index> sigma_offsets;
  std::vector<Eigen::Index> mu_offsets;
  std::vector<Eigen::Index> block_p;
  std::int64_t n_total = 0;
};

struct FIMLValueGradient {
  double value = 0.0;
  Eigen::VectorXd gradient;
};

struct FIMLExtras {
  double       logl              = 0.0;
  double       unrestricted_logl = 0.0;
  double       chi2              = 0.0;
  double       aic               = 0.0;
  double       bic               = 0.0;
  double       bic2              = 0.0;
  int          npar              = 0;
  std::int64_t ntotal            = 0;
};

struct FIMLRobustMLR {
  Eigen::MatrixXd vcov;
  Eigen::VectorXd se;
  Eigen::VectorXd eigvals;
  double       chisq_scaled      = std::numeric_limits<double>::quiet_NaN();
  double       scaling_factor    = std::numeric_limits<double>::quiet_NaN();
  double       trace_ugamma      = std::numeric_limits<double>::quiet_NaN();
  double       trace_ugamma_h1   = std::numeric_limits<double>::quiet_NaN();
  double       trace_ugamma_h0   = std::numeric_limits<double>::quiet_NaN();
  int          df                = 0;
  std::int64_t ntotal            = 0;
};

// Full-information ML over raw continuous data with arbitrary observed-value
// patterns. The optimized scalar is the per-observation observed-pattern
// normal-theory deviance without saturated/H1 constants:
//
//   sum_patterns (n_r / N) * [log|Σ_oo| + tr(Σ_oo^-1 (S_r + dd'))]
//
// where d = xbar_r - μ_o and S_r is the N-divisor covariance inside the
// pattern. This is the right objective for point estimation; saturated/H1 and
// baseline likelihood accounting are handled by the post-fit helpers below.
struct FIML {
  static constexpr std::string_view name = "FIML";

  fit_expected<FIMLCache>
  prepare(const RawData& raw) const;

  fit_expected<double>
  value(const RawData& raw, const FIMLCache& cache,
        const model::ImpliedMoments& moments) const;

  fit_expected<Eigen::VectorXd>
  gradient(const RawData& raw, const FIMLCache& cache,
           const model::ImpliedMoments& moments,
           const Eigen::MatrixXd& J_sigma,
           const Eigen::MatrixXd& J_mu) const;

  fit_expected<FIMLValueGradient>
  value_gradient(const RawData& raw, const FIMLCache& cache,
                 const model::ImpliedMoments& moments,
                 const Eigen::MatrixXd& J_sigma,
                 const Eigen::MatrixXd& J_mu) const;
};

// Start-value and fixed.x helper for FIML. Means use all observed values in
// each column; covariances use pairwise observed rows with an N divisor.
fit_expected<SampleStats>
fiml_start_sample_stats(const RawData& raw);

// First public FIML fixed.x policy: observed exogenous variables may be fixed
// from complete raw data, but missing values in those variables are not yet
// supported because lavaan's conditional fixed.x likelihood accounting is a
// separate contract from the joint observed-data FIML objective.
fit_expected<void>
validate_fiml_fixed_x_missing_policy(const partable::LatentStructure& pt,
                                     const RawData& raw);

// Post-fit likelihood accounting for continuous raw-data FIML. The optimizer
// minimizes only the observed-pattern deviance without constants; this helper
// adds the normal constants and fits the saturated/H1 observed-data normal
// model so logl, unrestricted.logl, chi-square, and information criteria use
// lavaan-compatible missing-data likelihood accounting.
post_expected<FIMLExtras>
fiml_extras(partable::LatentStructure pt,
            const model::MatrixRep& rep,
            const RawData& raw,
            const Estimates& est,
            FIML discrepancy = {});

// Robust missing-data reporting for lavaan's continuous FIML MLR corner
// (`missing = "fiml", estimator = "MLR"`). The sandwich meat is built from
// observed-pattern casewise deviance gradients, and the bread is a
// finite-difference observed FIML Hessian.
post_expected<FIMLRobustMLR>
fiml_robust_mlr(partable::LatentStructure pt,
                const model::MatrixRep& rep,
                const RawData& raw,
                const Estimates& est,
                int df,
                double chisq,
                FIML discrepancy = {},
                double h_step = 1e-4);

// FIML independence/baseline chi-square for raw continuous data with missing
// values. Unlike complete-data `baseline_chi2(SampleStats)`, this evaluates the
// diagonal normal model directly over observed-value patterns and compares it
// to the FIML saturated/H1 likelihood.
post_expected<BaselineFit>
fiml_baseline_chi2(const RawData& raw,
                   FIML discrepancy = {});

template <class O = LbfgsOptimizer>
fit_expected<Estimates>
fit_fiml(partable::LatentStructure pt,
         const model::MatrixRep& rep,
         const RawData& raw,
         FIML discrepancy = {},
         O optimizer = {},
         partable::Starts starts = {}) {
  if (auto e = validate_fiml_fixed_x_missing_policy(pt, raw); !e.has_value()) {
    return std::unexpected(e.error());
  }

  auto start_samp_or = fiml_start_sample_stats(raw);
  if (!start_samp_or.has_value()) return std::unexpected(start_samp_or.error());
  const SampleStats& start_samp = *start_samp_or;

  if (auto e = resolve_fixed_x_from_sample(pt, rep, start_samp);
      !e.has_value()) {
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

  auto x0_or = simple_start_values(pt, rep, start_samp, starts);
  if (!x0_or.has_value()) return std::unexpected(x0_or.error());

  auto cache_or = discrepancy.prepare(raw);
  if (!cache_or.has_value()) return std::unexpected(cache_or.error());

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(FitError{
        FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail, 0, 0.0});
  }
  const EqConstraints& con = *con_or;

  auto eval_at = [&](const Eigen::VectorXd& x,
                     Eigen::VectorXd& grad) -> double {
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    auto vg = discrepancy.value_gradient(raw, *cache_or, eval->moments,
                                         eval->J_sigma, eval->J_mu);
    if (!vg.has_value()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    grad = std::move(vg->gradient);
    return vg->value;
  };

  if (!con.active()) {
    auto out_or = optimizer.minimize(eval_at, *x0_or);
    if (!out_or.has_value()) return std::unexpected(out_or.error());
    return Estimates{std::move(out_or->theta_hat), out_or->fmin,
                     out_or->iterations};
  }

  if (con.n_alpha == 0) {
    Eigen::VectorXd theta = con.expand(Eigen::VectorXd(0));
    Eigen::VectorXd scratch(theta.size());
    const double f = eval_at(theta, scratch);
    return Estimates{std::move(theta), f, 0};
  }

  Eigen::VectorXd alpha0 = con.contract(*x0_or);
  auto objective_alpha = [&](const Eigen::VectorXd& a,
                             Eigen::VectorXd& grad_a) -> double {
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

}  // namespace magmaan::fit
