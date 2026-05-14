#include "magmaan/estimate/ordinal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/fit/concepts.hpp"
#include "magmaan/fit/constraints.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/fit/start_values.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/op.hpp"

namespace magmaan::estimate {

namespace {

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

Eigen::Index vech_index(Eigen::Index p, Eigen::Index r, Eigen::Index c) noexcept {
  return c * p - (c * (c - 1)) / 2 + (r - c);
}

Eigen::Index vech_len(Eigen::Index p) noexcept {
  return p * (p + 1) / 2;
}

fit_expected<std::int64_t> total_n_obs(const data::OrdinalStats& s) {
  std::int64_t total = 0;
  for (auto n : s.n_obs) total += n;
  if (total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalStats has non-positive total n_obs"));
  }
  return total;
}

fit_expected<void> validate_stats(const data::OrdinalStats& s,
                                  const model::MatrixRep& rep,
                                  OrdinalWeightKind kind) {
  const std::size_t nb = rep.dims.size();
  if (s.R.size() != nb || s.thresholds.size() != nb ||
      s.threshold_ov.size() != nb || s.threshold_level.size() != nb ||
      s.n_obs.size() != nb || s.n_levels.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalStats block counts do not match MatrixRep"));
  }
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? s.W_dwls : s.W_wls;
  if (Ws.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalStats weight block count does not match MatrixRep"));
  }
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = rep.dims[b].n_observed;
    if (s.R[b].rows() != p || s.R[b].cols() != p) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats R dimension mismatch in block " + std::to_string(b)));
    }
    if (static_cast<Eigen::Index>(s.threshold_ov[b].size()) != s.thresholds[b].size() ||
        static_cast<Eigen::Index>(s.threshold_level[b].size()) != s.thresholds[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats threshold metadata mismatch in block " + std::to_string(b)));
    }
    const Eigen::Index mdim = s.thresholds[b].size() + p * (p - 1) / 2;
    if (Ws[b].rows() != mdim || Ws[b].cols() != mdim) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats weight dimension mismatch in block " + std::to_string(b)));
    }
    Eigen::LLT<Eigen::MatrixXd> llt(Ws[b]);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats weight matrix is not positive definite in block " +
              std::to_string(b)));
    }
  }
  return {};
}

struct ThresholdLayout {
  std::vector<std::vector<std::int32_t>> free;
  std::vector<std::vector<double>> fixed;
  std::vector<std::vector<char>> present;
};

fit_expected<ThresholdLayout>
make_threshold_layout(const spec::LatentStructure& pt,
                      const model::MatrixRep& rep,
                      const data::OrdinalStats& stats) {
  ThresholdLayout out;
  const std::size_t nb = rep.dims.size();
  out.free.resize(nb);
  out.fixed.resize(nb);
  out.present.resize(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t nth = static_cast<std::size_t>(stats.thresholds[b].size());
    out.free[b].assign(nth, 0);
    out.fixed[b].assign(nth, 0.0);
    out.present[b].assign(nth, 0);
  }

  // Match rows by row order per variable. The R helper generates rows in this
  // deterministic order (`x | t1`, `x | t2`, ...).
  std::vector<std::vector<std::int32_t>> seen;
  seen.resize(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    seen[b].assign(static_cast<std::size_t>(rep.dims[b].n_observed), 0);
  }
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] != parse::Op::Threshold || pt.group[i] <= 0) continue;
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= nb) continue;
    const std::int32_t ov = pt.lhs_var[i] >= 0
        ? pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])]
        : -1;
    if (ov < 0 || static_cast<std::size_t>(ov) >= seen[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal threshold row references a non-observed variable"));
    }
    const std::int32_t lev = ++seen[b][static_cast<std::size_t>(ov)];
    Eigen::Index pos = -1;
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      if (stats.threshold_ov[b][static_cast<std::size_t>(k)] == ov &&
          stats.threshold_level[b][static_cast<std::size_t>(k)] == lev) {
        pos = k;
        break;
      }
    }
    if (pos < 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal threshold row has no matching sample threshold"));
    }
    out.free[b][static_cast<std::size_t>(pos)] = pt.free[i];
    out.fixed[b][static_cast<std::size_t>(pos)] =
        std::isfinite(pt.fixed_value[i]) ? pt.fixed_value[i] : 0.0;
    out.present[b][static_cast<std::size_t>(pos)] = 1;
  }
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t k = 0; k < out.free[b].size(); ++k) {
      if (!out.present[b][k]) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "missing ordinal threshold row for block " + std::to_string(b)));
      }
    }
  }
  return out;
}

fit_expected<std::vector<Eigen::MatrixXd>>
weight_factors(const data::OrdinalStats& stats, OrdinalWeightKind kind) {
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  std::vector<Eigen::MatrixXd> out;
  out.reserve(Ws.size());
  for (std::size_t b = 0; b < Ws.size(); ++b) {
    Eigen::LLT<Eigen::MatrixXd> llt(Ws[b]);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal weight matrix is not positive definite in block " +
              std::to_string(b)));
    }
    out.push_back(llt.matrixL());
  }
  return out;
}

Eigen::VectorXd implied_thresholds(const ThresholdLayout& layout,
                                   const Eigen::VectorXd& theta,
                                   std::size_t b) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(layout.free[b].size()));
  for (Eigen::Index k = 0; k < out.size(); ++k) {
    const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
    out(k) = fr > 0 ? theta(fr - 1) : layout.fixed[b][static_cast<std::size_t>(k)];
  }
  return out;
}

Eigen::VectorXd corr_lower(const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = Sigma.rows();
  Eigen::VectorXd out(p * (p - 1) / 2);
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const double den = std::sqrt(std::max(1e-12, Sigma(i, i) * Sigma(j, j)));
      out(k++) = Sigma(i, j) / den;
    }
  }
  return out;
}

Eigen::MatrixXd corr_jacobian(const Eigen::MatrixXd& Sigma,
                              const Eigen::MatrixXd& J_sigma,
                              Eigen::Index sigma_off) {
  const Eigen::Index p = Sigma.rows();
  const Eigen::Index n_free = J_sigma.cols();
  Eigen::MatrixXd J(p * (p - 1) / 2, n_free);
  Eigen::Index row = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const double sii = std::max(1e-12, Sigma(i, i));
      const double sjj = std::max(1e-12, Sigma(j, j));
      const double den = std::sqrt(sii * sjj);
      const double rho = Sigma(i, j) / den;
      J.row(row) = J_sigma.row(sigma_off + vech_index(p, i, j)) / den;
      J.row(row).noalias() -=
          (0.5 * rho / sii) * J_sigma.row(sigma_off + vech_index(p, i, i));
      J.row(row).noalias() -=
          (0.5 * rho / sjj) * J_sigma.row(sigma_off + vech_index(p, j, j));
      ++row;
    }
  }
  return J;
}

fit_expected<Eigen::VectorXd>
ordinal_residuals(const data::OrdinalStats& stats,
                  const ThresholdLayout& layout,
                  const model::ImpliedMoments& moments,
                  const std::vector<Eigen::MatrixXd>& factors,
                  const Eigen::VectorXd& theta) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    n_total += stats.thresholds[b].size() + p * (p - 1) / 2;
  }
  Eigen::VectorXd out(n_total);
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index nth = stats.thresholds[b].size();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    Eigen::VectorXd d(nth + ncorr);
    d.head(nth) = implied_thresholds(layout, theta, b) - stats.thresholds[b];
    d.tail(ncorr) = corr_lower(moments.sigma[b]) - corr_lower(stats.R[b]);
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.segment(off, d.size()) = sw * (factors[b].transpose() * d);
    off += d.size();
  }
  if (!out.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "ordinal LS residuals contain non-finite values"));
  }
  return out;
}

fit_expected<Eigen::MatrixXd>
ordinal_jacobian(const data::OrdinalStats& stats,
                 const ThresholdLayout& layout,
                 const model::ImpliedMoments& moments,
                 const Eigen::MatrixXd& J_sigma,
                 const std::vector<Eigen::MatrixXd>& factors) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    n_total += stats.thresholds[b].size() + p * (p - 1) / 2;
  }
  Eigen::MatrixXd out(n_total, J_sigma.cols());
  Eigen::Index out_off = 0;
  Eigen::Index sigma_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index nth = stats.thresholds[b].size();
    const Eigen::Index ncorr = p * (p - 1) / 2;
    Eigen::MatrixXd Jb(nth + ncorr, J_sigma.cols());
    Jb.setZero();
    for (Eigen::Index k = 0; k < nth; ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0) Jb(k, fr - 1) = 1.0;
    }
    Jb.bottomRows(ncorr) = corr_jacobian(moments.sigma[b], J_sigma, sigma_off);
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.block(out_off, 0, Jb.rows(), Jb.cols()) =
        sw * (factors[b].transpose() * Jb);
    out_off += Jb.rows();
    sigma_off += vech_len(p);
  }
  return out;
}

fit::SampleStats sample_stats_for_starts(const data::OrdinalStats& stats) {
  fit::SampleStats samp;
  samp.S = stats.R;
  samp.n_obs = stats.n_obs;
  return samp;
}

void seed_threshold_starts(Eigen::VectorXd& x,
                           const ThresholdLayout& layout,
                           const data::OrdinalStats& stats) {
  for (std::size_t b = 0; b < stats.thresholds.size(); ++b) {
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0 && fr <= x.size()) x(fr - 1) = stats.thresholds[b](k);
    }
  }
}

}  // namespace

fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    optim::LbfgsBOptimizer optimizer,
                    spec::Starts starts) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());
  const ThresholdLayout& layout = *layout_or;

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto ev = std::move(*ev_or);

  fit::SampleStats samp = sample_stats_for_starts(stats);
  auto x0_or = fit::simple_start_values(pt, rep, samp, starts);
  if (!x0_or.has_value()) return std::unexpected(x0_or.error());
  Eigen::VectorXd x0 = *x0_or;
  seed_threshold_starts(x0, layout, stats);

  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: bounds size mismatch"));
  }

  auto factors_or = weight_factors(stats, weights);
  if (!factors_or.has_value()) return std::unexpected(factors_or.error());
  const auto& factors = *factors_or;

  auto con_or = fit::build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail));
  }
  const fit::EqConstraints& con = *con_or;

  auto eval0 = ev.evaluate(x0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = ordinal_residuals(stats, layout, eval0->moments, factors, x0);
  if (!r0.has_value()) return std::unexpected(r0.error());
  const Eigen::Index n_data = r0->size();
  const Eigen::Index n_eq = con.active() ? con.A_eq.rows() : 0;
  const Eigen::Index n_total = n_data + n_eq;
  const double F0 = 0.5 * r0->squaredNorm();
  const double mu_eq = n_eq > 0 ? std::max(1.0, F0) * 1e10 : 0.0;
  const double sqrt_mu_eq = std::sqrt(mu_eq);

  auto resid_fn = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::VectorXd> {
    auto eval = ev.evaluate(x, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    auto r = ordinal_residuals(stats, layout, eval->moments, factors, x);
    if (!r.has_value()) return std::unexpected(r.error());
    Eigen::VectorXd out(n_total);
    out.head(n_data) = *r;
    if (n_eq > 0) out.tail(n_eq).noalias() = sqrt_mu_eq * (con.A_eq * x - con.b_eq);
    return out;
  };

  auto jac_fn = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(x, true, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    auto J = ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma, factors);
    if (!J.has_value()) return std::unexpected(J.error());
    Eigen::MatrixXd out(n_total, J->cols());
    out.topRows(n_data) = *J;
    if (n_eq > 0) out.bottomRows(n_eq).noalias() = sqrt_mu_eq * con.A_eq;
    return out;
  };

  auto out_or = optimizer.minimize_ls(fit::LsResidualFn(resid_fn),
                                      fit::LsJacobianFn(jac_fn),
                                      n_total, x0,
                                      bounds.lower, bounds.upper);
  if (!out_or.has_value()) return std::unexpected(out_or.error());

  double fmin_data = out_or->fmin;
  if (n_eq > 0) {
    const Eigen::VectorXd eq_resid = con.A_eq * out_or->theta_hat - con.b_eq;
    const double eq_max = eq_resid.cwiseAbs().maxCoeff();
    if (eq_max > 1e-6) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: equality residual exceeded tolerance"));
    }
    fmin_data = out_or->fmin - 0.5 * mu_eq * eq_resid.squaredNorm();
    if (fmin_data < 0.0) fmin_data = 0.0;
  }
  return Estimates{std::move(out_or->theta_hat), fmin_data, out_or->iterations};
}

}  // namespace magmaan::estimate
