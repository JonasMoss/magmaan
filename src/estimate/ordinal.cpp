#include "magmaan/estimate/ordinal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SVD>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/op.hpp"

namespace magmaan::estimate {

using data::SampleStats;
using inference::ScoreCandidate;
using inference::ScoreCandidateKind;
using inference::ScoreTestResult;
using inference::ScoreTestTable;
using inference::chi2_pvalue;
using robust::MeanVarAdjustedResult;
using robust::SatorraBentlerResult;
using robust::ScaledShiftedResult;
using optim::LbfgsOptions;

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

PostError model_to_post(ModelError e) {
  return make_post_err(PostError::Kind::NumericIssue, std::move(e.detail));
}

Eigen::Index vech_index(Eigen::Index p, Eigen::Index r, Eigen::Index c) noexcept {
  return c * p - (c * (c - 1)) / 2 + (r - c);
}

Eigen::Index vech_len(Eigen::Index p) noexcept {
  return p * (p + 1) / 2;
}

bool matrix_all_finite(const Eigen::MatrixXd& M) {
  for (Eigen::Index c = 0; c < M.cols(); ++c)
    for (Eigen::Index r = 0; r < M.rows(); ++r)
      if (!std::isfinite(M(r, c))) return false;
  return true;
}

bool vector_all_finite(const Eigen::VectorXd& v) {
  for (Eigen::Index i = 0; i < v.size(); ++i)
    if (!std::isfinite(v(i))) return false;
  return true;
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
  if (Ws.size() != nb || s.NACOV.size() != nb) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "OrdinalStats weight/NACOV block count does not match MatrixRep"));
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
    if (Ws[b].rows() != mdim || Ws[b].cols() != mdim ||
        s.NACOV[b].rows() != mdim || s.NACOV[b].cols() != mdim) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats moment/weight dimension mismatch in block " +
              std::to_string(b)));
    }
    if (s.n_obs[b] <= 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats n_obs must be positive in block " + std::to_string(b)));
    }
    if (!matrix_all_finite(s.R[b]) || !vector_all_finite(s.thresholds[b]) ||
        !matrix_all_finite(Ws[b]) || !matrix_all_finite(s.NACOV[b])) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "OrdinalStats contains non-finite values in block " +
              std::to_string(b)));
    }
    for (Eigen::Index k = 0; k < s.NACOV[b].rows(); ++k) {
      if (!(s.NACOV[b](k, k) > 0.0)) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "OrdinalStats NACOV diagonal is not positive in block " +
                std::to_string(b)));
      }
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
    if (static_cast<Eigen::Index>(s.threshold_ov[b].size()) != s.thresholds[b].size() ||
        static_cast<Eigen::Index>(s.threshold_level[b].size()) != s.thresholds[b].size()) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats threshold metadata mismatch in block " +
              std::to_string(b)));
    }
    Eigen::Index n_cont = 0;
    std::vector<char> has_threshold(static_cast<std::size_t>(p), 0);
    for (Eigen::Index j = 0; j < p; ++j) {
      const std::int32_t flag = s.ordered[b][static_cast<std::size_t>(j)];
      if (flag != 0 && flag != 1) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats ordered mask must contain only 0/1 values in block " +
                std::to_string(b)));
      }
      if (flag == 0) ++n_cont;
    }
    for (std::int32_t ov : s.threshold_ov[b]) {
      if (ov < 0 || ov >= p) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats threshold metadata references an invalid observed "
            "variable in block " + std::to_string(b)));
      }
      if (s.ordered[b][static_cast<std::size_t>(ov)] == 0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats threshold metadata references a continuous variable "
            "in block " + std::to_string(b)));
      }
      has_threshold[static_cast<std::size_t>(ov)] = 1;
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (s.ordered[b][static_cast<std::size_t>(j)] != 0 &&
          has_threshold[static_cast<std::size_t>(j)] == 0) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats missing thresholds for ordered variable in block " +
                std::to_string(b)));
      }
    }
    const Eigen::Index mdim = s.thresholds[b].size() + 2 * n_cont + p * (p - 1) / 2;
    if (s.moments[b].size() != mdim || Ws[b].rows() != mdim || Ws[b].cols() != mdim ||
        s.NACOV[b].rows() != mdim || s.NACOV[b].cols() != mdim) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats moment/weight dimension mismatch in block " +
              std::to_string(b)));
    }
    if (s.n_obs[b] <= 0) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats n_obs must be positive in block " + std::to_string(b)));
    }
    if (!matrix_all_finite(s.R[b]) || !vector_all_finite(s.mean[b]) ||
        !vector_all_finite(s.thresholds[b]) ||
        !vector_all_finite(s.moments[b]) ||
        !matrix_all_finite(Ws[b]) || !matrix_all_finite(s.NACOV[b])) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "MixedOrdinalStats contains non-finite values in block " +
              std::to_string(b)));
    }
    for (Eigen::Index k = 0; k < s.NACOV[b].rows(); ++k) {
      if (!(s.NACOV[b](k, k) > 0.0)) {
        return std::unexpected(make_err(FitError::Kind::NumericIssue,
            "MixedOrdinalStats NACOV diagonal is not positive in block " +
                std::to_string(b)));
      }
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

// === Theta-parameterization helpers =========================================
// Under theta the latent-response residual variances are fixed to 1, so the
// implied total variances Σ*ᵢᵢ float; the implied moments are standardized
// before comparison with the (always unit-variance) polychoric sample moments.

// Σᵢⱼ / √(Σᵢᵢ Σⱼⱼ) over the strict lower triangle.
Eigen::VectorXd std_corr_lower(const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = Sigma.rows();
  Eigen::VectorXd sd(p);
  for (Eigen::Index i = 0; i < p; ++i) sd(i) = std::sqrt(Sigma(i, i));
  Eigen::VectorXd out(p * (p - 1) / 2);
  Eigen::Index k = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out(k++) = Sigma(i, j) / (sd(i) * sd(j));
    }
  }
  return out;
}

// ∂[Σᵢⱼ/√(ΣᵢᵢΣⱼⱼ)]/∂θ over the strict lower triangle, from ∂vech(Σ)/∂θ.
Eigen::MatrixXd std_corr_jacobian(const Eigen::MatrixXd& Sigma,
                                  const Eigen::MatrixXd& J_sigma,
                                  Eigen::Index sigma_off) {
  const Eigen::Index p = Sigma.rows();
  Eigen::MatrixXd J(p * (p - 1) / 2, J_sigma.cols());
  Eigen::Index row = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const double sij = Sigma(i, j);
      const double sii = Sigma(i, i);
      const double sjj = Sigma(j, j);
      const double inv_i = 1.0 / std::sqrt(sii);
      const double inv_j = 1.0 / std::sqrt(sjj);
      J.row(row) =
          (inv_i * inv_j) * J_sigma.row(sigma_off + vech_index(p, i, j)) -
          (0.5 * sij * inv_i / sii * inv_j) *
              J_sigma.row(sigma_off + vech_index(p, i, i)) -
          (0.5 * sij * inv_i * inv_j / sjj) *
              J_sigma.row(sigma_off + vech_index(p, j, j));
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
                  const Eigen::VectorXd& theta,
                  OrdinalParameterization param) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  const bool theta_param = param == OrdinalParameterization::Theta;
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
    const Eigen::MatrixXd& Sig = moments.sigma[b];
    Eigen::VectorXd d(nth + ncorr);
    Eigen::VectorXd it = implied_thresholds(layout, theta, b);
    if (theta_param) {
      // Standardize: implied thresholds τ_θ/√Σ*ᵢᵢ, implied correlations.
      for (Eigen::Index k = 0; k < nth; ++k) {
        const Eigen::Index ov =
            stats.threshold_ov[b][static_cast<std::size_t>(k)];
        it(k) /= std::sqrt(Sig(ov, ov));
      }
      d.head(nth) = it - stats.thresholds[b];
      d.tail(ncorr) = std_corr_lower(Sig) - corr_lower(stats.R[b]);
    } else {
      d.head(nth) = it - stats.thresholds[b];
      d.tail(ncorr) = corr_lower(Sig) - corr_lower(stats.R[b]);
    }
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
                 const std::vector<Eigen::MatrixXd>& factors,
                 const Eigen::VectorXd& theta,
                 OrdinalParameterization param) {
  auto N = total_n_obs(stats);
  if (!N.has_value()) return std::unexpected(N.error());
  const bool theta_param = param == OrdinalParameterization::Theta;
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
    const Eigen::MatrixXd& Sig = moments.sigma[b];
    Eigen::MatrixXd Jb(nth + ncorr, J_sigma.cols());
    Jb.setZero();
    if (theta_param) {
      // Threshold rows: ∂(τ_θ/√Σ*ᵢᵢ)/∂θ — a selector term on the free
      // threshold parameter plus a structural term through Σ*ᵢᵢ.
      const Eigen::VectorXd it = implied_thresholds(layout, theta, b);
      for (Eigen::Index k = 0; k < nth; ++k) {
        const Eigen::Index ov =
            stats.threshold_ov[b][static_cast<std::size_t>(k)];
        const double sii = Sig(ov, ov);
        const double inv_sd = 1.0 / std::sqrt(sii);
        const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
        if (fr > 0) Jb(k, fr - 1) += inv_sd;
        Jb.row(k) += (-0.5 * it(k) * inv_sd / sii) *
                     J_sigma.row(sigma_off + vech_index(p, ov, ov));
      }
      Jb.bottomRows(ncorr) = std_corr_jacobian(Sig, J_sigma, sigma_off);
    } else {
      for (Eigen::Index k = 0; k < nth; ++k) {
        const std::int32_t fr = layout.free[b][static_cast<std::size_t>(k)];
        if (fr > 0) Jb(k, fr - 1) = 1.0;
      }
      Jb.bottomRows(ncorr) = corr_jacobian(Sig, J_sigma, sigma_off);
    }
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

data::SampleStats sample_stats_for_starts(const data::OrdinalStats& stats) {
  data::SampleStats samp;
  samp.S = stats.R;
  samp.n_obs = stats.n_obs;
  return samp;
}

data::SampleStats sample_stats_for_starts(const data::MixedOrdinalStats& stats) {
  data::SampleStats samp;
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

OrdinalRobustResult ordinal_result_from_weighted(const WeightedRobustResult& r) {
  OrdinalRobustResult out;
  out.vcov = r.vcov;
  out.se = r.se;
  out.eigvals = r.eigvals;
  out.chisq_standard = r.chisq_standard;
  out.df = r.df;
  out.satorra_bentler = r.satorra_bentler;
  out.mean_var_adjusted = r.mean_var_adjusted;
  out.scaled_shifted = r.scaled_shifted;
  return out;
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
prepare_ordinal_partable(spec::LatentStructure& pt,
                         const data::OrdinalStats& stats,
                         OrdinalParameterization parameterization,
                         spec::Starts* starts) {
  // The prepared partable is identical for Delta and Theta: magmaan fixes the
  // ordinal-indicator residual variances and intercepts the same way for both.
  // The Delta/Theta distinction is realized in the fit objective (whether the
  // implied moments are standardized), not in the partable layout.
  (void)parameterization;
  return prepare_ordinal_delta_partable(pt, stats, starts);
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

fit_expected<void>
prepare_mixed_ordinal_partable(spec::LatentStructure& pt,
                                const data::MixedOrdinalStats& stats,
                                OrdinalParameterization parameterization,
                                spec::Starts* starts) {
  if (parameterization == OrdinalParameterization::Delta) {
    return prepare_mixed_ordinal_delta_partable(pt, stats, starts);
  }
  return std::unexpected(make_err(FitError::Kind::NumericIssue,
      "mixed ordinal theta parameterization is not supported yet; use delta"));
}

// Start-value producer for the ordinal delta path. Prepares the partable
// (delta parameterization fixes the ordinal indicator variances/intercepts —
// this is what changes n_free), seeds the structural parameters via the simple
// scheme, then overwrites the free thresholds from the sample thresholds. The
// returned vector is sized for the *prepared* partable, which is exactly what
// `fit_ordinal_bounded` rebuilds internally.
fit_expected<Eigen::VectorXd>
ordinal_start_values(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalStats& stats,
                     spec::Starts starts) {
  if (auto p = prepare_ordinal_delta_partable(pt, stats, &starts);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());

  data::SampleStats samp = sample_stats_for_starts(stats);
  auto x0_or = simple_start_values(pt, rep, samp, starts);
  if (!x0_or.has_value()) return std::unexpected(x0_or.error());
  Eigen::VectorXd x0 = std::move(*x0_or);
  seed_threshold_starts(x0, *layout_or, stats);
  return x0;
}

fit_expected<Eigen::VectorXd>
mixed_ordinal_start_values(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalStats& stats,
                           spec::Starts starts) {
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, &starts);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto layout_or = make_threshold_layout(pt, rep, stats);
  if (!layout_or.has_value()) return std::unexpected(layout_or.error());

  data::SampleStats samp = sample_stats_for_starts(stats);
  auto x0_or = simple_start_values(pt, rep, samp, starts);
  if (!x0_or.has_value()) return std::unexpected(x0_or.error());
  Eigen::VectorXd x0 = std::move(*x0_or);
  seed_threshold_starts(x0, *layout_or, stats);
  return x0;
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

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_ordinal: constraint reparameterization has incompatible shape"));
  }

  const auto& Ws = weights == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  std::vector<WeightedMomentBlock> blocks;
  blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index p = stats.R[b].rows();
    const Eigen::Index mb = stats.thresholds[b].size() + p * (p - 1) / 2;
    if (stats.NACOV[b].rows() != mb || stats.NACOV[b].cols() != mb) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "OrdinalStats NACOV dimension mismatch in block " + std::to_string(b)));
    }
    blocks.push_back(WeightedMomentBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .gamma = stats.NACOV[b],
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  auto out = robust_weighted_moments(blocks, K, est.fmin);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
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

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const Eigen::MatrixXd& K = con_or->K();
  if (K.rows() != Delta_full.cols()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "robust_mixed_ordinal: constraint reparameterization has incompatible shape"));
  }

  const auto& Ws = weights == OrdinalWeightKind::DWLS ? stats.W_dwls : stats.W_wls;
  std::vector<WeightedMomentBlock> blocks;
  blocks.reserve(stats.R.size());
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < stats.R.size(); ++b) {
    const Eigen::Index mb = stats.moments[b].size();
    blocks.push_back(WeightedMomentBlock{
        .jacobian = Delta_full.block(off, 0, mb, Delta_full.cols()),
        .weight = Ws[b],
        .gamma = stats.NACOV[b],
        .n_obs = stats.n_obs[b]});
    off += mb;
  }

  auto out = robust_weighted_moments(blocks, K, est.fmin);
  if (!out.has_value()) return std::unexpected(out.error());
  return ordinal_result_from_weighted(*out);
}

namespace {

post_expected<Eigen::MatrixXd> invert_score_spd(const Eigen::MatrixXd& A,
                                                std::string_view what) {
  Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (A + A.transpose()));
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        std::string(what) + " is not positive definite"));
  }
  return ldlt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
}

post_expected<inference::ScoreTestResult>
ordinal_score_for_direction(const inference::ScoreCandidate& candidate,
                            const Eigen::VectorXd& score_full,
                            const Eigen::MatrixXd& info_full,
                            const Eigen::MatrixXd& K_nuisance,
                            const Eigen::VectorXd& direction) {
  const Eigen::VectorXd I_d = info_full * direction;
  double score_eff = direction.dot(score_full);
  double info_eff = direction.dot(I_d);
  if (K_nuisance.cols() > 0) {
    const Eigen::MatrixXd I_aa =
        K_nuisance.transpose() * info_full * K_nuisance;
    const Eigen::VectorXd I_ab = K_nuisance.transpose() * I_d;
    const Eigen::VectorXd score_a = K_nuisance.transpose() * score_full;
    auto inv = invert_score_spd(I_aa, "ordinal score nuisance information");
    if (!inv.has_value()) return std::unexpected(inv.error());
    score_eff -= I_ab.dot((*inv) * score_a);
    info_eff -= I_ab.dot((*inv) * I_ab);
  }
  if (!(info_eff > 1e-10 * std::max<double>(1.0, std::abs(info_eff)))) {
    return std::unexpected(make_post_err(PostError::Kind::InfoMatrixSingular,
        "ordinal score efficient information is not positive"));
  }
  inference::ScoreTestResult out;
  out.candidate = candidate;
  out.score = score_eff;
  out.information = info_eff;
  out.mi = (score_eff * score_eff) / info_eff;
  out.df = 1;
  out.p_value = inference::chi2_pvalue(out.mi, 1);
  out.epc = score_eff / info_eff;
  return out;
}

post_expected<Eigen::MatrixXd> ordinal_null_space(const Eigen::MatrixXd& A,
                                                  Eigen::Index n_cols) {
  if (A.rows() == 0) return Eigen::MatrixXd::Identity(n_cols, n_cols);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  svd.setThreshold(1e-9);
  return Eigen::MatrixXd(svd.matrixV().rightCols(n_cols - svd.rank()));
}

post_expected<Eigen::VectorXd>
ordinal_release_direction(const EqConstraints& con, Eigen::Index release_row) {
  Eigen::MatrixXd A_rel(con.A_eq.rows() - 1, con.A_eq.cols());
  Eigen::Index out = 0;
  for (Eigen::Index r = 0; r < con.A_eq.rows(); ++r) {
    if (r == release_row) continue;
    A_rel.row(out++) = con.A_eq.row(r);
  }
  auto K_rel = ordinal_null_space(A_rel, con.npar);
  if (!K_rel.has_value()) return std::unexpected(K_rel.error());
  const Eigen::MatrixXd M = K_rel->transpose() * con.K();
  auto z = ordinal_null_space(M.transpose(), K_rel->cols());
  if (!z.has_value()) return std::unexpected(z.error());
  if (z->cols() != 1) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score equality release is not one-dimensional"));
  }
  Eigen::VectorXd d = (*K_rel) * z->col(0);
  const double norm = d.norm();
  if (!(norm > 0.0)) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score equality-release direction is degenerate"));
  }
  d /= norm;
  return d;
}

bool ordinal_fixed_candidate(const spec::LatentStructure& pt,
                             const model::MatrixRep& rep,
                             std::size_t row) {
  if (row >= pt.size()) return false;
  if (pt.is_constraint_row(row)) return false;
  if (pt.free[row] != 0) return false;
  if (row < pt.exo.size() && pt.exo[row] != 0) return false;
  if (row >= pt.fixed_value.size() || !std::isfinite(pt.fixed_value[row])) {
    return false;
  }
  return pt.op[row] == parse::Op::Threshold ||
         (row < rep.cell_for_row.size() && rep.cell_for_row[row].used);
}

bool ordinal_var_is_latent(const spec::LatentStructure& pt, std::int32_t v) {
  return v >= 0 && static_cast<std::size_t>(v) < pt.var_role.size() &&
         pt.var_role[static_cast<std::size_t>(v)] == spec::VarRole::Latent;
}

bool ordinal_var_is_indicator(const spec::LatentStructure& pt, std::int32_t v) {
  return v >= 0 && static_cast<std::size_t>(v) < pt.var_role.size() &&
         pt.var_role[static_cast<std::size_t>(v)] == spec::VarRole::Indicator;
}

struct OrdinalAbsentRow {
  parse::Op    op;
  std::int32_t lhs;
  std::int32_t rhs;
  std::int32_t group;
};

std::vector<OrdinalAbsentRow>
enumerate_ordinal_absent_rows(
    const spec::LatentStructure& pt,
    const inference::ModificationIndexOptions& opts) {
  std::vector<OrdinalAbsentRow> out;
  std::vector<std::int32_t> latents;
  std::vector<std::int32_t> indicators;
  for (std::int32_t v = 0; v < pt.n_vars; ++v) {
    if (ordinal_var_is_latent(pt, v)) latents.push_back(v);
    else if (ordinal_var_is_indicator(pt, v)) indicators.push_back(v);
  }

  using Key = std::array<std::int32_t, 3>;
  for (std::int32_t g = 1; g <= pt.n_groups(); ++g) {
    std::set<Key> present;
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.group[i] != g) continue;
      const std::int32_t a = pt.lhs_var[i];
      const std::int32_t b = pt.rhs_var[i];
      if (pt.op[i] == parse::Op::Measurement) {
        present.insert({0, a, b});
      } else if (pt.op[i] == parse::Op::Covariance) {
        present.insert({1, std::min(a, b), std::max(a, b)});
      } else if (pt.op[i] == parse::Op::Regression) {
        present.insert({2, a, b});
      }
    }

    if (opts.include_loadings) {
      for (const std::int32_t f : latents) {
        for (const std::int32_t x : indicators) {
          if (!present.count({0, f, x})) {
            out.push_back({parse::Op::Measurement, f, x, g});
          }
        }
      }
    }
    if (opts.include_covariances) {
      auto cov_pairs = [&](const std::vector<std::int32_t>& vs) {
        for (std::size_t i = 0; i < vs.size(); ++i) {
          for (std::size_t j = i + 1; j < vs.size(); ++j) {
            const std::int32_t a = std::min(vs[i], vs[j]);
            const std::int32_t b = std::max(vs[i], vs[j]);
            if (!present.count({1, a, b})) {
              out.push_back({parse::Op::Covariance, a, b, g});
            }
          }
        }
      };
      cov_pairs(indicators);
      cov_pairs(latents);
    }
  }
  return out;
}

spec::LatentStructure
append_ordinal_absent_rows(spec::LatentStructure pt,
                           const std::vector<OrdinalAbsentRow>& rows) {
  for (const OrdinalAbsentRow& r : rows) {
    pt.op.push_back(r.op);
    pt.lhs_var.push_back(r.lhs);
    pt.rhs_var.push_back(r.rhs);
    pt.group.push_back(r.group);
    pt.free.push_back(0);
    pt.exo.push_back(0);
    pt.fixed_value.push_back(0.0);
  }
  return pt;
}

struct OrdinalModificationIndexModel {
  spec::LatentStructure pt;
  model::MatrixRep rep;
  std::size_t original_rows = 0;
};

post_expected<OrdinalModificationIndexModel>
prepare_ordinal_modification_index_model(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const inference::ModificationIndexOptions& options) {
  if (options.candidates == inference::ScoreCandidateSet::FixedRowsOnly) {
    const std::size_t original_rows = pt.size();
    return OrdinalModificationIndexModel{std::move(pt), rep, original_rows};
  }

  const std::size_t original_rows = pt.size();
  const std::vector<OrdinalAbsentRow> absent =
      enumerate_ordinal_absent_rows(pt, options);
  pt = append_ordinal_absent_rows(std::move(pt), absent);
  auto mr = model::build_matrix_rep(pt);
  if (!mr.has_value()) return std::unexpected(model_to_post(mr.error()));
  return OrdinalModificationIndexModel{std::move(pt), std::move(*mr),
                                       original_rows};
}

void ordinal_add_free_group(spec::LatentStructure& pt, std::int32_t old_n) {
  if (static_cast<std::int32_t>(pt.eq_groups.size()) == old_n) {
    pt.eq_groups.push_back(old_n);
  } else if (!pt.eq_groups.empty()) {
    pt.eq_groups.clear();
  }
}

template <class Stats, class ResidualFn, class JacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_modification_indices_impl(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const Stats& stats,
                                  const Estimates& est,
                                  OrdinalWeightKind weights,
                                  const inference::ModificationIndexOptions& options,
                                  ResidualFn residual_fn,
                                  JacobianFn jacobian_fn,
                                  PrepareFn prepare_fn) {
  auto work = prepare_ordinal_modification_index_model(std::move(pt), rep,
                                                       options);
  if (!work.has_value()) return std::unexpected(work.error());

  if (auto v = validate_stats(stats, work->rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(work->pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != work->pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal modification indices: fitted theta length does not match delta partable"));
  }
  auto con0 = build_eq_constraints(work->pt);
  if (!con0.has_value()) return std::unexpected(con0.error());
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);

  inference::ScoreTestTable table;
  for (std::size_t row = 0; row < work->pt.size(); ++row) {
    if (!ordinal_fixed_candidate(work->pt, work->rep, row)) continue;
    spec::LatentStructure aug = work->pt;
    const double fixed_value = aug.fixed_value[row];
    const std::int32_t old_n = aug.n_free();
    aug.free[row] = old_n + 1;
    aug.fixed_value[row] = std::numeric_limits<double>::quiet_NaN();
    ordinal_add_free_group(aug, old_n);

    Eigen::VectorXd theta(est.theta.size() + 1);
    if (est.theta.size() > 0) theta.head(est.theta.size()) = est.theta;
    theta(est.theta.size()) = fixed_value;

    auto layout = make_threshold_layout(aug, work->rep, stats);
    if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
    auto factors = weight_factors(stats, weights);
    if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
    auto ev = model::ModelEvaluator::build(aug, work->rep);
    if (!ev.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal modification indices: ModelEvaluator::build failed: " +
              ev.error().detail));
    }
    auto eval = ev->evaluate(theta, true, true);
    if (!eval.has_value()) {
      return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
          "ordinal modification indices: fitted evaluation failed: " +
              eval.error().detail));
    }
    auto r = residual_fn(stats, *layout, eval->moments, *factors, theta);
    if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
    auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                         eval->J_mu, *factors);
    if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));

    const bool generated_absent = row >= work->original_rows;
    // lavaan scores generated all-ordinal absent rows on the single-counted
    // moment scale; explicit fixed partable rows retain the fixed-row scale.
    const double moment_scale =
        (generated_absent && std::is_same_v<Stats, data::OrdinalStats>)
            ? 1.0
            : 2.0;
    const Eigen::VectorXd score =
        -moment_scale * n_total * (J->transpose() * *r);
    Eigen::MatrixXd info = moment_scale * n_total * (J->transpose() * *J);
    info = 0.5 * (info + info.transpose());

    Eigen::VectorXd direction = Eigen::VectorXd::Zero(score.size());
    direction(score.size() - 1) = 1.0;
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::FixedParam;
    cand.row = row;
    cand.op = work->pt.op[row];
    cand.lhs_var = work->pt.lhs_var[row];
    cand.rhs_var = work->pt.rhs_var[row];
    cand.group = work->pt.group[row];
    Eigen::MatrixXd K_aug = Eigen::MatrixXd::Zero(score.size(), con0->K().cols());
    if (con0->K().rows() > 0) K_aug.topRows(con0->K().rows()) = con0->K();
    auto res = ordinal_score_for_direction(cand, score, info, K_aug, direction);
    if (res.has_value()) table.rows.push_back(*res);
  }
  return table;
}

template <class Stats, class ResidualFn, class JacobianFn, class PrepareFn>
post_expected<inference::ScoreTestTable>
ordinal_score_tests_impl(spec::LatentStructure pt,
                         const model::MatrixRep& rep,
                         const Stats& stats,
                         const Estimates& est,
                         OrdinalWeightKind weights,
                         ResidualFn residual_fn,
                         JacobianFn jacobian_fn,
                         PrepareFn prepare_fn) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(fit_to_post(v.error()));
  }
  if (auto p = prepare_fn(pt, stats); !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score tests: fitted theta length does not match delta partable"));
  }
  auto con = build_eq_constraints(pt);
  if (!con.has_value()) return std::unexpected(con.error());
  inference::ScoreTestTable table;
  if (!con->active()) return table;
  auto N_or = total_n_obs(stats);
  if (!N_or.has_value()) return std::unexpected(fit_to_post(N_or.error()));
  const double n_total = static_cast<double>(*N_or);

  auto layout = make_threshold_layout(pt, rep, stats);
  if (!layout.has_value()) return std::unexpected(fit_to_post(layout.error()));
  auto factors = weight_factors(stats, weights);
  if (!factors.has_value()) return std::unexpected(fit_to_post(factors.error()));
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score tests: ModelEvaluator::build failed: " + ev.error().detail));
  }
  auto eval = ev->evaluate(est.theta, true, true);
  if (!eval.has_value()) {
    return std::unexpected(make_post_err(PostError::Kind::NumericIssue,
        "ordinal score tests: fitted evaluation failed: " + eval.error().detail));
  }
  auto r = residual_fn(stats, *layout, eval->moments, *factors, est.theta);
  if (!r.has_value()) return std::unexpected(fit_to_post(r.error()));
  auto J = jacobian_fn(stats, *layout, eval->moments, eval->J_sigma,
                       eval->J_mu, *factors);
  if (!J.has_value()) return std::unexpected(fit_to_post(J.error()));
  const Eigen::VectorXd score = -n_total * (J->transpose() * *r);
  Eigen::MatrixXd info = n_total * (J->transpose() * *J);
  info = 0.5 * (info + info.transpose());

  for (Eigen::Index row = 0; row < con->A_eq.rows(); ++row) {
    auto d = ordinal_release_direction(*con, row);
    if (!d.has_value()) return std::unexpected(d.error());
    inference::ScoreCandidate cand;
    cand.kind = inference::ScoreCandidateKind::EqualityRelease;
    cand.row = static_cast<std::size_t>(row);
    cand.op = parse::Op::EqConstraint;
    auto res = ordinal_score_for_direction(cand, score, info, con->K(), *d);
    if (res.has_value()) table.rows.push_back(*res);
  }
  return table;
}

}  // namespace

post_expected<inference::ScoreTestTable>
modification_indices_ordinal(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::OrdinalStats& stats,
                             const Estimates& est,
                             OrdinalWeightKind weights) {
  inference::ModificationIndexOptions options;
  return modification_indices_ordinal(std::move(pt), rep, stats, est, weights,
                                      options);
}

post_expected<inference::ScoreTestTable>
modification_indices_ordinal(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::OrdinalStats& stats,
                             const Estimates& est,
                             OrdinalWeightKind weights,
                             const inference::ModificationIndexOptions& options) {
  auto residual_fn = [](const data::OrdinalStats& s,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const std::vector<Eigen::MatrixXd>& factors,
                        const Eigen::VectorXd& theta) {
    return ordinal_residuals(s, layout, moments, factors, theta,
                             OrdinalParameterization::Delta);
  };
  auto jacobian_fn = [](const data::OrdinalStats& s,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const Eigen::MatrixXd& J_sigma,
                        const Eigen::MatrixXd&,
                        const std::vector<Eigen::MatrixXd>& factors) {
    return ordinal_jacobian(s, layout, moments, J_sigma, factors,
                            Eigen::VectorXd{}, OrdinalParameterization::Delta);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::OrdinalStats& s) {
    return prepare_ordinal_delta_partable(p, s, nullptr);
  };
  return ordinal_modification_indices_impl(std::move(pt), rep, stats, est,
                                           weights, options, residual_fn,
                                           jacobian_fn, prepare_fn);
}

post_expected<inference::ScoreTestTable>
score_tests_ordinal(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    const Estimates& est,
                    OrdinalWeightKind weights) {
  auto residual_fn = [](const data::OrdinalStats& s,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const std::vector<Eigen::MatrixXd>& factors,
                        const Eigen::VectorXd& theta) {
    return ordinal_residuals(s, layout, moments, factors, theta,
                             OrdinalParameterization::Delta);
  };
  auto jacobian_fn = [](const data::OrdinalStats& s,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const Eigen::MatrixXd& J_sigma,
                        const Eigen::MatrixXd&,
                        const std::vector<Eigen::MatrixXd>& factors) {
    return ordinal_jacobian(s, layout, moments, J_sigma, factors,
                            Eigen::VectorXd{}, OrdinalParameterization::Delta);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::OrdinalStats& s) {
    return prepare_ordinal_delta_partable(p, s, nullptr);
  };
  return ordinal_score_tests_impl(std::move(pt), rep, stats, est, weights,
                                  residual_fn, jacobian_fn, prepare_fn);
}

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::MixedOrdinalStats& stats,
                                   const Estimates& est,
                                   OrdinalWeightKind weights) {
  inference::ModificationIndexOptions options;
  return modification_indices_mixed_ordinal(std::move(pt), rep, stats, est,
                                            weights, options);
}

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& est,
    OrdinalWeightKind weights,
    const inference::ModificationIndexOptions& options) {
  auto residual_fn = [](const data::MixedOrdinalStats& s,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const std::vector<Eigen::MatrixXd>& factors,
                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_residuals(s, layout, moments, factors, theta);
  };
  auto jacobian_fn = [](const data::MixedOrdinalStats& s,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const Eigen::MatrixXd& J_sigma,
                        const Eigen::MatrixXd& J_mu,
                        const std::vector<Eigen::MatrixXd>& factors) {
    return mixed_ordinal_jacobian(s, layout, moments, J_sigma, J_mu, factors);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::MixedOrdinalStats& s) {
    return prepare_mixed_ordinal_delta_partable(p, s, nullptr);
  };
  return ordinal_modification_indices_impl(std::move(pt), rep, stats, est,
                                           weights, options, residual_fn,
                                           jacobian_fn, prepare_fn);
}

post_expected<inference::ScoreTestTable>
score_tests_mixed_ordinal(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          const Estimates& est,
                          OrdinalWeightKind weights) {
  auto residual_fn = [](const data::MixedOrdinalStats& s,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const std::vector<Eigen::MatrixXd>& factors,
                        const Eigen::VectorXd& theta) {
    return mixed_ordinal_residuals(s, layout, moments, factors, theta);
  };
  auto jacobian_fn = [](const data::MixedOrdinalStats& s,
                        const ThresholdLayout& layout,
                        const model::ImpliedMoments& moments,
                        const Eigen::MatrixXd& J_sigma,
                        const Eigen::MatrixXd& J_mu,
                        const std::vector<Eigen::MatrixXd>& factors) {
    return mixed_ordinal_jacobian(s, layout, moments, J_sigma, J_mu, factors);
  };
  auto prepare_fn = [](spec::LatentStructure& p, const data::MixedOrdinalStats& s) {
    return prepare_mixed_ordinal_delta_partable(p, s, nullptr);
  };
  return ordinal_score_tests_impl(std::move(pt), rep, stats, est, weights,
                                  residual_fn, jacobian_fn, prepare_fn);
}

namespace {

// Dispatch a bounded least-squares ordinal problem (residual + Jacobian
// closures, equality penalty rows already folded in) to the chosen backend.
fit_expected<optim::OptimResult>
run_ordinal_ls(optim::ResidualFn r, optim::JacobianFn J, Eigen::Index n_resid,
               const Eigen::VectorXd& x0, const Bounds& bounds,
               Backend backend, LbfgsOptions opts) {
  optim::GmmProblem prob;
  prob.r       = std::move(r);
  prob.J       = std::move(J);
  prob.n_resid = n_resid;
  prob.n_param = x0.size();
  prob.expand  = [](const Eigen::VectorXd& x) { return x; };
  if (backend == Backend::Ceres) {
#ifdef MAGMAAN_WITH_CERES
    optim::CeresOptions copts;
    copts.max_iter = opts.max_iter;
    copts.ftol     = opts.ftol;
    copts.gtol     = opts.gtol;
    return optim::ceres_lm(prob, x0, bounds, copts);
#else
    (void)opts;
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "fit_ordinal_bounded: Ceres backend requested but MAGMAAN_WITH_CERES "
        "is off"));
#endif
  }
  return optim::lbfgs(optim::scalarize(prob), x0, bounds, opts);
}

}  // namespace

fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    const Eigen::VectorXd& x0,
                    Backend backend,
                    LbfgsOptions opts,
                    OrdinalParameterization parameterization) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
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

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

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

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail));
  }
  const EqConstraints& con = *con_or;

  auto eval0 = ev.evaluate(x0, false, false);
  if (!eval0.has_value()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_ordinal_bounded: start evaluation failed: " + eval0.error().detail));
  }
  auto r0 = ordinal_residuals(stats, layout, eval0->moments, factors, x0,
                              parameterization);
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
    auto r = ordinal_residuals(stats, layout, eval->moments, factors, x,
                               parameterization);
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
    auto J = ordinal_jacobian(stats, layout, eval->moments, eval->J_sigma,
                              factors, x, parameterization);
    if (!J.has_value()) return std::unexpected(J.error());
    Eigen::MatrixXd out(n_total, J->cols());
    out.topRows(n_data) = *J;
    if (n_eq > 0) out.bottomRows(n_eq).noalias() = sqrt_mu_eq * con.A_eq;
    return out;
  };

  auto out_or = run_ordinal_ls(resid_fn, jac_fn, n_total, x0, bounds,
                               backend, opts);
  if (!out_or.has_value()) return std::unexpected(out_or.error());

  double fmin_data = out_or->fmin;
  if (n_eq > 0) {
    const Eigen::VectorXd eq_resid = con.A_eq * out_or->x - con.b_eq;
    const double eq_max = eq_resid.cwiseAbs().maxCoeff();
    if (eq_max > 1e-6) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_ordinal_bounded: equality residual exceeded tolerance"));
    }
    fmin_data = out_or->fmin - 0.5 * mu_eq * eq_resid.squaredNorm();
    if (fmin_data < 0.0) fmin_data = 0.0;
  }
  return Estimates{std::move(out_or->x), 2.0 * fmin_data,
                   out_or->iterations};
}

fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          Bounds bounds,
                          OrdinalWeightKind weights,
                          const Eigen::VectorXd& x0,
                          Backend backend,
                          LbfgsOptions opts) {
  if (auto v = validate_stats(stats, rep, weights); !v.has_value()) {
    return std::unexpected(v.error());
  }
  if (auto p = prepare_mixed_ordinal_delta_partable(pt, stats, nullptr); !p.has_value()) {
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

  if (x0.size() != pt.n_free()) {
    return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
        "fit_mixed_ordinal_bounded: x0 size (" + std::to_string(x0.size()) +
            ") != prepared partable n_free (" +
            std::to_string(pt.n_free()) + ")"));
  }

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

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "constraint: " + con_or.error().detail));
  }
  const EqConstraints& con = *con_or;

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

  auto out_or = run_ordinal_ls(resid_fn, jac_fn, n_total, x0, bounds,
                               backend, opts);
  if (!out_or.has_value()) return std::unexpected(out_or.error());

  double fmin_data = out_or->fmin;
  if (n_eq > 0) {
    const Eigen::VectorXd eq_resid = con.A_eq * out_or->x - con.b_eq;
    const double eq_max = eq_resid.cwiseAbs().maxCoeff();
    if (eq_max > 1e-6) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "fit_mixed_ordinal_bounded: equality residual exceeded tolerance"));
    }
    fmin_data = out_or->fmin - 0.5 * mu_eq * eq_resid.squaredNorm();
    if (fmin_data < 0.0) fmin_data = 0.0;
  }
  return Estimates{std::move(out_or->x), 2.0 * fmin_data,
                   out_or->iterations};
}

}  // namespace magmaan::estimate
