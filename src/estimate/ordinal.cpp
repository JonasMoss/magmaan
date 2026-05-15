#include "magmaan/estimate/ordinal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/fit/concepts.hpp"
#include "magmaan/fit/constraints.hpp"
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/fit/start_values.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/op.hpp"

#ifdef MAGMAAN_WITH_CERES
#include "magmaan/optim/ceres_optimizer.hpp"
#endif

namespace magmaan::estimate {

namespace {

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

PostError make_post_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

PostError fit_to_post(FitError e) {
  return make_post_err(PostError::Kind::NumericIssue, std::move(e.detail));
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

fit_expected<std::int64_t> total_n_obs(const data::MixedOrdinalStats& s) {
  std::int64_t total = 0;
  for (auto n : s.n_obs) total += n;
  if (total <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalStats has non-positive total n_obs"));
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

fit_expected<void> validate_stats(const data::MixedOrdinalStats& s,
                                  const model::MatrixRep& rep,
                                  OrdinalWeightKind kind) {
  const std::size_t nb = rep.dims.size();
  if (s.R.size() != nb || s.mean.size() != nb || s.ordered.size() != nb ||
      s.thresholds.size() != nb || s.threshold_ov.size() != nb ||
      s.threshold_level.size() != nb || s.moments.size() != nb ||
      s.n_obs.size() != nb || s.n_levels.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalStats block counts do not match MatrixRep"));
  }
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? s.W_dwls : s.W_wls;
  if (Ws.size() != nb || s.NACOV.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalStats weight/NACOV block count does not match MatrixRep"));
  }
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = rep.dims[b].n_observed;
    if (s.R[b].rows() != p || s.R[b].cols() != p || s.mean[b].size() != p ||
        s.ordered[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats dimension mismatch in block " + std::to_string(b)));
    }
    Eigen::Index n_cont = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (s.ordered[b][static_cast<std::size_t>(j)] == 0) ++n_cont;
    }
    const Eigen::Index mdim = s.thresholds[b].size() + 2 * n_cont + p * (p - 1) / 2;
    if (s.moments[b].size() != mdim || Ws[b].rows() != mdim || Ws[b].cols() != mdim ||
        s.NACOV[b].rows() != mdim || s.NACOV[b].cols() != mdim) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats moment/weight dimension mismatch in block " +
              std::to_string(b)));
    }
    Eigen::LLT<Eigen::MatrixXd> llt(Ws[b]);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats weight matrix is not positive definite in block " +
              std::to_string(b)));
    }
  }
  return {};
}

fit_expected<std::vector<std::vector<char>>>
ordered_indicator_layout(const spec::LatentStructure& pt,
                         const data::OrdinalStats& stats) {
  const std::size_t nb = stats.R.size();
  std::vector<std::vector<char>> out(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    out[b].assign(static_cast<std::size_t>(stats.R[b].rows()), 0);
    for (std::int32_t ov : stats.threshold_ov[b]) {
      if (ov < 0 || static_cast<std::size_t>(ov) >= out[b].size()) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "OrdinalStats threshold metadata references an invalid observed variable"));
      }
      out[b][static_cast<std::size_t>(ov)] = 1;
    }
  }
  const std::size_t ng = static_cast<std::size_t>(pt.n_groups());
  if (ng != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalStats block count does not match partable group count"));
  }
  return out;
}

fit_expected<std::vector<std::vector<char>>>
ordered_indicator_layout(const spec::LatentStructure& pt,
                         const data::MixedOrdinalStats& stats) {
  const std::size_t nb = stats.R.size();
  std::vector<std::vector<char>> out(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    out[b].assign(static_cast<std::size_t>(stats.R[b].rows()), 0);
    for (std::size_t j = 0; j < stats.ordered[b].size(); ++j) {
      out[b][j] = stats.ordered[b][j] != 0 ? 1 : 0;
    }
  }
  const std::size_t ng = static_cast<std::size_t>(pt.n_groups());
  if (ng != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "MixedOrdinalStats block count does not match partable group count"));
  }
  return out;
}

fit_expected<void>
compact_free_set(spec::LatentStructure& pt,
                 const std::vector<char>& remove_free,
                 spec::Starts* starts) {
  const std::int32_t old_n = pt.n_free();
  if (static_cast<std::int32_t>(remove_free.size()) != old_n + 1) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "ordinal delta free-set compaction received inconsistent metadata"));
  }

  std::vector<char> seen(static_cast<std::size_t>(old_n) + 1, 0);
  std::vector<std::int32_t> new_to_old;
  new_to_old.reserve(static_cast<std::size_t>(old_n));
  auto append_by_op = [&](std::int32_t group, parse::Op op, bool matching) {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.group[i] != group) continue;
      if ((pt.op[i] == op) != matching) continue;
      const std::int32_t old = pt.free[i];
      if (old <= 0 || remove_free[static_cast<std::size_t>(old)] != 0 ||
          seen[static_cast<std::size_t>(old)] != 0) {
        continue;
      }
      seen[static_cast<std::size_t>(old)] = 1;
      new_to_old.push_back(old);
    }
  };
  for (std::int32_t g = 1; g <= pt.n_groups(); ++g) {
    append_by_op(g, parse::Op::Measurement, true);
    append_by_op(g, parse::Op::Threshold, true);
    append_by_op(g, parse::Op::Threshold, false);
  }

  std::vector<std::int32_t> old_to_new(static_cast<std::size_t>(old_n) + 1, 0);
  for (std::size_t neu = 0; neu < new_to_old.size(); ++neu) {
    old_to_new[static_cast<std::size_t>(new_to_old[neu])] =
        static_cast<std::int32_t>(neu) + 1;
  }

  for (std::int32_t& fr : pt.free) {
    if (fr <= 0) continue;
    const std::int32_t neu = old_to_new[static_cast<std::size_t>(fr)];
    if (neu <= 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal delta cannot remove a response-scale variance that shares a free index"));
    }
    fr = neu;
  }

  std::vector<std::int32_t> eq_new(new_to_old.size(), 0);
  const bool have_eq =
      static_cast<std::int32_t>(pt.eq_groups.size()) == old_n;
  for (std::size_t k = 0; k < new_to_old.size(); ++k) {
    const std::int32_t old = new_to_old[k];
    eq_new[k] = have_eq ? pt.eq_groups[static_cast<std::size_t>(old - 1)]
                        : old - 1;
  }
  pt.eq_groups = std::move(eq_new);

  const std::size_t n_lin = pt.lin_constraint_d.size();
  if (n_lin > 0) {
    const std::size_t old_cols = static_cast<std::size_t>(old_n);
    if (pt.lin_constraint_R.size() != n_lin * old_cols) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "ordinal delta free-set compaction found malformed linear constraints"));
    }
    std::vector<double> R_new(n_lin * new_to_old.size(), 0.0);
    for (std::size_t r = 0; r < n_lin; ++r) {
      for (std::int32_t old = 1; old <= old_n; ++old) {
        const double v =
            pt.lin_constraint_R[r * old_cols + static_cast<std::size_t>(old - 1)];
        const std::int32_t neu = old_to_new[static_cast<std::size_t>(old)];
        if (neu > 0) {
          R_new[r * new_to_old.size() + static_cast<std::size_t>(neu - 1)] = v;
        } else if (std::abs(v) > 1e-12) {
          return std::unexpected(make_err(FitError::Kind::NumericIssue,
              "ordinal delta does not support constraints on derived response-scale variances"));
        }
      }
    }
    pt.lin_constraint_R = std::move(R_new);
  }

  if (starts != nullptr && !starts->hint.empty()) {
    std::vector<double> hint_new(new_to_old.size(),
                                 std::numeric_limits<double>::quiet_NaN());
    for (std::size_t k = 0; k < new_to_old.size(); ++k) {
      const std::size_t old = static_cast<std::size_t>(new_to_old[k] - 1);
      if (old < starts->hint.size()) hint_new[k] = starts->hint[old];
    }
    starts->hint = std::move(hint_new);
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

fit_expected<ThresholdLayout>
make_threshold_layout(const spec::LatentStructure& pt,
                      const model::MatrixRep& rep,
                      const data::MixedOrdinalStats& stats) {
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

  std::vector<std::vector<std::int32_t>> seen(nb);
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
          "mixed ordinal threshold row references a non-observed variable"));
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
          "mixed ordinal threshold row has no matching sample threshold"));
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
            "missing mixed ordinal threshold row for block " + std::to_string(b)));
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

fit_expected<std::vector<Eigen::MatrixXd>>
weight_factors(const data::MixedOrdinalStats& stats, OrdinalWeightKind kind) {
  const auto& Ws = kind == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  std::vector<Eigen::MatrixXd> out;
  out.reserve(Ws.size());
  for (std::size_t b = 0; b < Ws.size(); ++b) {
    Eigen::LLT<Eigen::MatrixXd> llt(Ws[b]);
    if (llt.info() != Eigen::Success) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "mixed ordinal weight matrix is not positive definite in block " +
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
      out(k++) = Sigma(i, j);
    }
  }
  return out;
}

Eigen::MatrixXd corr_jacobian(const Eigen::MatrixXd& Sigma,
                              const Eigen::MatrixXd& J_sigma,
                              Eigen::Index sigma_off) {
  (void)Sigma;
  const Eigen::Index p = Sigma.rows();
  const Eigen::Index n_free = J_sigma.cols();
  Eigen::MatrixXd J(p * (p - 1) / 2, n_free);
  Eigen::Index row = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      J.row(row) = J_sigma.row(sigma_off + vech_index(p, i, j));
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

Eigen::Index ordinal_moment_rows(const data::OrdinalStats& stats) {
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    n_total += stats.thresholds[b].size() + p * (p - 1) / 2;
  }
  return n_total;
}

Eigen::MatrixXd ordinal_moment_jacobian(const data::OrdinalStats& stats,
                                        const ThresholdLayout& layout,
                                        const model::ImpliedMoments& moments,
                                        const Eigen::MatrixXd& J_sigma) {
  Eigen::MatrixXd out(ordinal_moment_rows(stats), J_sigma.cols());
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
    out.block(out_off, 0, Jb.rows(), Jb.cols()) = Jb;
    out_off += Jb.rows();
    sigma_off += vech_len(p);
  }
  return out;
}

Eigen::Index mixed_moment_rows(const data::MixedOrdinalStats& stats) {
  Eigen::Index n_total = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    n_total += stats.moments[b].size();
  }
  return n_total;
}

Eigen::VectorXd mixed_model_moments(const data::MixedOrdinalStats& stats,
                                    const ThresholdLayout& layout,
                                    const model::ImpliedMoments& moments,
                                    const Eigen::VectorXd& theta,
                                    std::size_t b) {
  const Eigen::Index p = stats.R[b].rows();
  Eigen::VectorXd out(stats.moments[b].size());
  Eigen::Index k = 0;
  const Eigen::Index nth = stats.thresholds[b].size();
  out.segment(k, nth) = implied_thresholds(layout, theta, b);
  k += nth;
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
      const double mu = b < moments.mu.size() && moments.mu[b].size() == p
          ? moments.mu[b](j)
          : 0.0;
      out(k++) = -mu;
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
      out(k++) = moments.sigma[b](j, j);
    }
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out(k++) = moments.sigma[b](i, j);
    }
  }
  return out;
}

Eigen::MatrixXd mixed_moment_jacobian(const data::MixedOrdinalStats& stats,
                                      const ThresholdLayout& layout,
                                      const model::ImpliedMoments& moments,
                                      const Eigen::MatrixXd& J_sigma,
                                      const Eigen::MatrixXd& J_mu) {
  Eigen::MatrixXd out(mixed_moment_rows(stats), J_sigma.cols());
  Eigen::Index out_off = 0;
  Eigen::Index sigma_off = 0;
  Eigen::Index mu_off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index nth = stats.thresholds[b].size();
    Eigen::MatrixXd Jb(stats.moments[b].size(), J_sigma.cols());
    Jb.setZero();
    Eigen::Index row = 0;
    for (Eigen::Index k = 0; k < nth; ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0) Jb(row, fr - 1) = 1.0;
      ++row;
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
        if (J_mu.rows() > 0) Jb.row(row) = -J_mu.row(mu_off + j);
        ++row;
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (stats.ordered[b][static_cast<std::size_t>(j)] == 0) {
        Jb.row(row++) = J_sigma.row(sigma_off + vech_index(p, j, j));
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        Jb.row(row++) = J_sigma.row(sigma_off + vech_index(p, i, j));
      }
    }
    out.block(out_off, 0, Jb.rows(), Jb.cols()) = Jb;
    out_off += Jb.rows();
    sigma_off += vech_len(p);
    mu_off += p;
    (void)moments;
  }
  return out;
}

fit_expected<Eigen::VectorXd>
mixed_ordinal_residuals(const data::MixedOrdinalStats& stats,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const std::vector<Eigen::MatrixXd>& factors,
                        const Eigen::VectorXd& theta) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  Eigen::VectorXd out(mixed_moment_rows(stats));
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    Eigen::VectorXd d =
        mixed_model_moments(stats, layout, moments, theta, b) - stats.moments[b];
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.segment(off, d.size()) = sw * (factors[b].transpose() * d);
    off += d.size();
  }
  if (!out.allFinite()) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "mixed ordinal LS residuals contain non-finite values"));
  }
  return out;
}

fit_expected<Eigen::MatrixXd>
mixed_ordinal_jacobian(const data::MixedOrdinalStats& stats,
                       const ThresholdLayout& layout,
                       const model::ImpliedMoments& moments,
                       const Eigen::MatrixXd& J_sigma,
                       const Eigen::MatrixXd& J_mu,
                       const std::vector<Eigen::MatrixXd>& factors) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  Eigen::MatrixXd Jfull =
      mixed_moment_jacobian(stats, layout, moments, J_sigma, J_mu);
  Eigen::MatrixXd out(Jfull.rows(), Jfull.cols());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) /
                                static_cast<double>(*N));
    out.block(off, 0, mb, Jfull.cols()) =
        sw * (factors[b].transpose() * Jfull.block(off, 0, mb, Jfull.cols()));
    off += mb;
  }
  return out;
}

post_expected<Eigen::MatrixXd> inverse_sym_pd(const Eigen::MatrixXd& A,
                                              std::string_view what) {
  Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        std::string(what) + " is not positive definite"));
  }
  return ldlt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
}

post_expected<Eigen::MatrixXd> symmetric_sqrt_psd(const Eigen::MatrixXd& A,
                                                  std::string_view what) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
  if (es.info() != Eigen::Success) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        std::string(what) + " eigendecomposition failed"));
  }
  const double tol = 1e-10 * std::max<double>(1.0, A.cwiseAbs().maxCoeff());
  Eigen::VectorXd vals = es.eigenvalues();
  for (Eigen::Index i = 0; i < vals.size(); ++i) {
    if (vals(i) < -tol) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          std::string(what) + " is not positive semidefinite"));
    }
    vals(i) = std::sqrt(std::max(0.0, vals(i)));
  }
  return es.eigenvectors() * vals.asDiagonal() * es.eigenvectors().transpose();
}

fit::SampleStats sample_stats_for_starts(const data::OrdinalStats& stats) {
  fit::SampleStats samp;
  samp.S = stats.R;
  samp.n_obs = stats.n_obs;
  return samp;
}

fit::SampleStats sample_stats_for_starts(const data::MixedOrdinalStats& stats) {
  fit::SampleStats samp;
  samp.S = stats.R;
  samp.mean = stats.mean;
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

void seed_threshold_starts(Eigen::VectorXd& x,
                           const ThresholdLayout& layout,
                           const data::MixedOrdinalStats& stats) {
  for (std::size_t b = 0; b < stats.thresholds.size(); ++b) {
    for (Eigen::Index k = 0; k < stats.thresholds[b].size(); ++k) {
      const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
      if (fr > 0 && fr <= x.size()) x(fr - 1) = stats.thresholds[b](k);
    }
  }
}

}  // namespace

fit_expected<void>
prepare_ordinal_delta_partable(spec::LatentStructure& pt,
                                const data::OrdinalStats& stats,
                                spec::Starts* starts) {
  auto ordered_or = ordered_indicator_layout(pt, stats);
  if (!ordered_or.has_value()) return std::unexpected(ordered_or.error());
  const auto& ordered = *ordered_or;

  const std::int32_t old_n = pt.n_free();
  std::vector<char> remove_free(static_cast<std::size_t>(old_n) + 1, 0);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if ((pt.op[i] != parse::Op::Covariance &&
         pt.op[i] != parse::Op::Intercept) ||
        pt.group[i] <= 0) {
      continue;
    }
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= ordered.size()) continue;
    if (pt.lhs_var[i] < 0) continue;
    if (pt.op[i] == parse::Op::Covariance &&
        (pt.rhs_var[i] < 0 || pt.lhs_var[i] != pt.rhs_var[i])) {
      continue;
    }
    const std::int32_t ov = pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])];
    if (ov < 0 || static_cast<std::size_t>(ov) >= ordered[b].size() ||
        ordered[b][static_cast<std::size_t>(ov)] == 0) {
      continue;
    }
    if (pt.free[i] > 0) remove_free[static_cast<std::size_t>(pt.free[i])] = 1;
    pt.free[i] = 0;
    pt.fixed_value[i] = pt.op[i] == parse::Op::Covariance ? 1.0 : 0.0;
  }
  return compact_free_set(pt, remove_free, starts);
}

fit_expected<void>
prepare_mixed_ordinal_delta_partable(spec::LatentStructure& pt,
                                      const data::MixedOrdinalStats& stats,
                                      spec::Starts* starts) {
  auto ordered_or = ordered_indicator_layout(pt, stats);
  if (!ordered_or.has_value()) return std::unexpected(ordered_or.error());
  const auto& ordered = *ordered_or;

  const std::int32_t old_n = pt.n_free();
  std::vector<char> remove_free(static_cast<std::size_t>(old_n) + 1, 0);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if ((pt.op[i] != parse::Op::Covariance &&
         pt.op[i] != parse::Op::Intercept) ||
        pt.group[i] <= 0) {
      continue;
    }
    const std::size_t b = static_cast<std::size_t>(pt.group[i] - 1);
    if (b >= ordered.size()) continue;
    if (pt.lhs_var[i] < 0) continue;
    if (pt.op[i] == parse::Op::Covariance &&
        (pt.rhs_var[i] < 0 || pt.lhs_var[i] != pt.rhs_var[i])) continue;
    const std::int32_t ov = pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[i])];
    if (ov < 0 || static_cast<std::size_t>(ov) >= ordered[b].size() ||
        ordered[b][static_cast<std::size_t>(ov)] == 0) {
      continue;
    }
    if (pt.free[i] > 0) remove_free[static_cast<std::size_t>(pt.free[i])] = 1;
    pt.free[i] = 0;
    pt.fixed_value[i] = pt.op[i] == parse::Op::Covariance ? 1.0 : 0.0;
  }
  return compact_free_set(pt, remove_free, starts);
}

post_expected<OrdinalRobustResult>
robust_ordinal(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::OrdinalStats& stats,
               const Estimates& est,
               OrdinalWeightKind weights) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (stats.NACOV.size() != stats.R.size()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "OrdinalStats NACOV block count does not match MatrixRep"));
  }
  const auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double N_total = static_cast<double>(*N_or);

  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: fitted theta length does not match ordinal delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, false);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: fitted evaluation failed: " + eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      ordinal_moment_jacobian(stats, *layout_or, eval->moments, eval->J_sigma);

  auto con_or = fit::build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: constraint reparameterization has incompatible shape"));
  }
  const Eigen::MatrixXd Delta_alpha = Delta_full * K;

  const Eigen::Index total_rows = Delta_alpha.rows();
  const Eigen::Index n_alpha = Delta_alpha.cols();
  const int df = static_cast<int>(total_rows - n_alpha);
  if (df < 0) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        "robust_ordinal: model has more reduced parameters than ordinal moments"));
  }

  Eigen::MatrixXd Dtilde(total_rows, n_alpha);
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(total_rows, total_rows);
  Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(total_rows, total_rows);
  const auto& Ws = weights == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.thresholds[b].size() + p * (p - 1) / 2;
    if (stats.NACOV[b].rows() != mb || stats.NACOV[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "OrdinalStats NACOV dimension mismatch in block " + std::to_string(b)));
    }
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) / N_total);
    Dtilde.block(off, 0, mb, n_alpha) =
        sw * Delta_alpha.block(off, 0, mb, n_alpha);
    W.block(off, off, mb, mb) = Ws[b];
    Gamma.block(off, off, mb, mb) = stats.NACOV[b];
    off += mb;
  }

  Eigen::MatrixXd A = Dtilde.transpose() * W * Dtilde;
  A = 0.5 * (A + A.transpose());
  auto A_inv_or = inverse_sym_pd(A, "robust_ordinal bread");
  if (!A_inv_or.has_value()) return std::unexpected(A_inv_or.error());
  const Eigen::MatrixXd& A_inv = *A_inv_or;

  Eigen::MatrixXd B = Dtilde.transpose() * W * Gamma * W * Dtilde;
  B = 0.5 * (B + B.transpose());
  Eigen::MatrixXd V_alpha = (A_inv * B * A_inv) / N_total;
  V_alpha = 0.5 * (V_alpha + V_alpha.transpose());

  OrdinalRobustResult out;
  out.vcov = K * V_alpha * K.transpose();
  out.vcov = 0.5 * (out.vcov + out.vcov.transpose());
  out.se.resize(out.vcov.rows());
  const double diag_tol = 1e-12 * std::max<double>(1.0, out.vcov.cwiseAbs().maxCoeff());
  for (Eigen::Index i = 0; i < out.se.size(); ++i) {
    const double v = out.vcov(i, i);
    out.se(i) = v >= -diag_tol ? std::sqrt(std::max(0.0, v))
                               : std::numeric_limits<double>::quiet_NaN();
  }

  out.chisq_standard = N_total * est.fmin;
  out.df = df;

  if (df > 0) {
    Eigen::MatrixXd U = W - W * Dtilde * A_inv * Dtilde.transpose() * W;
    U = 0.5 * (U + U.transpose());
    auto sqrtG_or = symmetric_sqrt_psd(Gamma, "robust_ordinal NACOV");
    if (!sqrtG_or.has_value()) return std::unexpected(sqrtG_or.error());
    Eigen::MatrixXd M = (*sqrtG_or) * U * (*sqrtG_or);
    M = 0.5 * (M + M.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_ordinal: U-Gamma eigendecomposition failed"));
    }
    out.eigvals = es.eigenvalues().tail(df);
    for (Eigen::Index i = 0; i < out.eigvals.size(); ++i) {
      if (out.eigvals(i) < 0.0 && out.eigvals(i) > -1e-10) out.eigvals(i) = 0.0;
    }
  } else {
    out.eigvals.resize(0);
  }

  out.satorra_bentler = fit::satorra_bentler(out.chisq_standard, out.df, out.eigvals);
  out.mean_var_adjusted = fit::mean_var_adjusted(out.chisq_standard, out.df, out.eigvals);
  out.scaled_shifted = fit::scaled_shifted(out.chisq_standard, out.df, out.eigvals);
  return out;
}

post_expected<OrdinalRobustResult>
robust_mixed_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::MixedOrdinalStats& stats,
                     const Estimates& est,
                     OrdinalWeightKind weights) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  const auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double N_total = static_cast<double>(*N_or);

  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: fitted theta length does not match mixed delta partable"));
  }

  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(fit_to_post(layout_or.error()));

  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  auto eval = ev_or->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: fitted evaluation failed: " + eval.error().detail));
  }

  const Eigen::MatrixXd Delta_full =
      mixed_moment_jacobian(stats, *layout_or, eval->moments,
                            eval->J_sigma, eval->J_mu);

  auto con_or = fit::build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: constraint reparameterization has incompatible shape"));
  }
  const Eigen::MatrixXd Delta_alpha = Delta_full * K;
  const Eigen::Index total_rows = Delta_alpha.rows();
  const Eigen::Index n_alpha = Delta_alpha.cols();
  const int df = static_cast<int>(total_rows - n_alpha);
  if (df < 0) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        "robust_mixed_ordinal: model has more reduced parameters than mixed moments"));
  }

  Eigen::MatrixXd Dtilde(total_rows, n_alpha);
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(total_rows, total_rows);
  Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(total_rows, total_rows);
  const auto& Ws = weights == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    const double sw = std::sqrt(static_cast<double>(stats.n_obs[b]) / N_total);
    Dtilde.block(off, 0, mb, n_alpha) =
        sw * Delta_alpha.block(off, 0, mb, n_alpha);
    W.block(off, off, mb, mb) = Ws[b];
    Gamma.block(off, off, mb, mb) = stats.NACOV[b];
    off += mb;
  }

  Eigen::MatrixXd A = Dtilde.transpose() * W * Dtilde;
  A = 0.5 * (A + A.transpose());
  auto A_inv_or = inverse_sym_pd(A, "robust_mixed_ordinal bread");
  if (!A_inv_or.has_value()) return std::unexpected(A_inv_or.error());
  const Eigen::MatrixXd& A_inv = *A_inv_or;

  Eigen::MatrixXd B = Dtilde.transpose() * W * Gamma * W * Dtilde;
  B = 0.5 * (B + B.transpose());
  Eigen::MatrixXd V_alpha = (A_inv * B * A_inv) / N_total;
  V_alpha = 0.5 * (V_alpha + V_alpha.transpose());

  OrdinalRobustResult out;
  out.vcov = K * V_alpha * K.transpose();
  out.vcov = 0.5 * (out.vcov + out.vcov.transpose());
  out.se.resize(out.vcov.rows());
  const double diag_tol = 1e-12 * std::max<double>(1.0, out.vcov.cwiseAbs().maxCoeff());
  for (Eigen::Index i = 0; i < out.se.size(); ++i) {
    const double v = out.vcov(i, i);
    out.se(i) = v >= -diag_tol ? std::sqrt(std::max(0.0, v))
                               : std::numeric_limits<double>::quiet_NaN();
  }
  out.chisq_standard = N_total * est.fmin;
  out.df = df;
  if (df > 0) {
    Eigen::MatrixXd U = W - W * Dtilde * A_inv * Dtilde.transpose() * W;
    U = 0.5 * (U + U.transpose());
    auto sqrtG_or = symmetric_sqrt_psd(Gamma, "robust_mixed_ordinal NACOV");
    if (!sqrtG_or.has_value()) return std::unexpected(sqrtG_or.error());
    Eigen::MatrixXd M = (*sqrtG_or) * U * (*sqrtG_or);
    M = 0.5 * (M + M.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "robust_mixed_ordinal: U-Gamma eigendecomposition failed"));
    }
    out.eigvals = es.eigenvalues().tail(df);
    for (Eigen::Index i = 0; i < out.eigvals.size(); ++i) {
      if (out.eigvals(i) < 0.0 && out.eigvals(i) > -1e-10) out.eigvals(i) = 0.0;
    }
  } else {
    out.eigvals.resize(0);
  }
  out.satorra_bentler = fit::satorra_bentler(out.chisq_standard, out.df, out.eigvals);
  out.mean_var_adjusted = fit::mean_var_adjusted(out.chisq_standard, out.df, out.eigvals);
  out.scaled_shifted = fit::scaled_shifted(out.chisq_standard, out.df, out.eigvals);
  return out;
}

template <optim::LsBoundedOptimizer O>
fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    O optimizer,
                    spec::Starts starts) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, &starts); !p.has_value()) {
    return std::unexpected(p.error());
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
  return Estimates{std::move(out_or->theta_hat), 2.0 * fmin_data,
                   out_or->iterations};
}

template <optim::LsBoundedOptimizer O>
fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          Bounds bounds,
                          OrdinalWeightKind weights,
                          O optimizer,
                          spec::Starts starts) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, &starts); !p.has_value()) {
    return std::unexpected(p.error());
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
          "fit_mixed_ordinal_bounded: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }
  if (bounds.lower.size() != x0.size() || bounds.upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_mixed_ordinal_bounded: bounds size mismatch"));
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
        "fit_mixed_ordinal_bounded: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = mixed_ordinal_residuals(stats, layout, eval0->moments, factors, x0);
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
          "fit_mixed_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    auto r = mixed_ordinal_residuals(stats, layout, eval->moments, factors, x);
    if (!r.has_value()) return std::unexpected(r.error());
    Eigen::VectorXd out(n_total);
    out.head(n_data) = *r;
    if (n_eq > 0) out.tail(n_eq).noalias() = sqrt_mu_eq * (con.A_eq * x - con.b_eq);
    return out;
  };

  auto jac_fn = [&](const Eigen::VectorXd& x) -> fit_expected<Eigen::MatrixXd> {
    auto eval = ev.evaluate(x, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_err(FitError::Kind::NonPositiveDefiniteSigma,
          "fit_mixed_ordinal_bounded: evaluate failed: " + eval.error().detail));
    }
    auto J = mixed_ordinal_jacobian(stats, layout, eval->moments,
                                    eval->J_sigma, eval->J_mu, factors);
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
          "fit_mixed_ordinal_bounded: equality residual exceeded tolerance"));
    }
    fmin_data = out_or->fmin - 0.5 * mu_eq * eq_resid.squaredNorm();
    if (fmin_data < 0.0) fmin_data = 0.0;
  }
  return Estimates{std::move(out_or->theta_hat), 2.0 * fmin_data,
                   out_or->iterations};
}

template fit_expected<Estimates>
fit_ordinal_bounded<optim::LbfgsBOptimizer>(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    Bounds bounds,
    OrdinalWeightKind weights,
    optim::LbfgsBOptimizer optimizer,
    spec::Starts starts);

template fit_expected<Estimates>
fit_mixed_ordinal_bounded<optim::LbfgsBOptimizer>(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    Bounds bounds,
    OrdinalWeightKind weights,
    optim::LbfgsBOptimizer optimizer,
    spec::Starts starts);

fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    optim::LbfgsBOptimizer optimizer,
                    spec::Starts starts) {
  return fit_ordinal_bounded<optim::LbfgsBOptimizer>(
      std::move(pt), rep, stats, std::move(bounds), weights,
      std::move(optimizer), std::move(starts));
}

fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          Bounds bounds,
                          OrdinalWeightKind weights,
                          optim::LbfgsBOptimizer optimizer,
                          spec::Starts starts) {
  return fit_mixed_ordinal_bounded<optim::LbfgsBOptimizer>(
      std::move(pt), rep, stats, std::move(bounds), weights,
      std::move(optimizer), std::move(starts));
}

#ifdef MAGMAAN_WITH_CERES
template fit_expected<Estimates>
fit_ordinal_bounded<optim::CeresBoundedOptimizer>(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::OrdinalStats& stats,
    Bounds bounds,
    OrdinalWeightKind weights,
    optim::CeresBoundedOptimizer optimizer,
    spec::Starts starts);

template fit_expected<Estimates>
fit_mixed_ordinal_bounded<optim::CeresBoundedOptimizer>(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    Bounds bounds,
    OrdinalWeightKind weights,
    optim::CeresBoundedOptimizer optimizer,
    spec::Starts starts);
#endif

}  // namespace magmaan::estimate
