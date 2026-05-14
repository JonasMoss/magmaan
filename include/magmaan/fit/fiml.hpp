#pragma once

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/constraints.hpp"
#include "magmaan/fit/fit.hpp"
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

// Full-information ML over raw continuous data with arbitrary observed-value
// patterns. The optimized scalar is the per-observation observed-pattern
// normal-theory deviance without saturated/H1 constants:
//
//   sum_patterns (n_r / N) * [log|Σ_oo| + tr(Σ_oo^-1 (S_r + dd'))]
//
// where d = xbar_r - μ_o and S_r is the N-divisor covariance inside the
// pattern. This is the right objective for point estimation; saturated
// likelihood constants and fit-statistic reporting are a later layer.
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

template <class O = LbfgsOptimizer>
fit_expected<Estimates>
fit_fiml(partable::LatentStructure pt,
         const model::MatrixRep& rep,
         const RawData& raw,
         FIML discrepancy = {},
         O optimizer = {},
         partable::Starts starts = {}) {
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
