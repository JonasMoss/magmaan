#include "magmaan/estimate/frontier/pairwise.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include "magmaan/error.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/reparameterize.hpp"

namespace magmaan::estimate::frontier {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kProbFloor = 1e-12;

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

FitError make_fit_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

PostError fit_to_post(const FitError& e) {
  return make_err(PostError::Kind::NumericIssue, e.detail);
}

bool matrix_all_finite(const Eigen::MatrixXd& M) {
  for (Eigen::Index c = 0; c < M.cols(); ++c)
    for (Eigen::Index r = 0; r < M.rows(); ++r)
      if (!std::isfinite(M(r, c))) return false;
  return true;
}

double normal_quantile(double p) noexcept {
  p = std::clamp(p, kProbFloor, 1.0 - kProbFloor);
  static constexpr double a[] = {
      -3.969683028665376e+01, 2.209460984245205e+02,
      -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00};
  static constexpr double b[] = {
      -5.447609879822406e+01, 1.615858368580409e+02,
      -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01};
  static constexpr double c[] = {
      -7.784894002430293e-03, -3.223964580411365e-01,
      -2.400758277161838e+00, -2.549732539343734e+00,
       4.374664141464968e+00,  2.938163982698783e+00};
  static constexpr double d[] = {
       7.784695709041462e-03,  3.224671290700398e-01,
       2.445134137142996e+00,  3.754408661907416e+00};
  constexpr double plow = 0.02425;
  constexpr double phigh = 1.0 - plow;
  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
            c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
             c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
          a[5]) *
         q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r +
          1.0);
}

post_expected<Eigen::VectorXd>
thresholds_for_variable(const data::OrdinalStats& stats,
                        std::size_t block,
                        std::int32_t variable,
                        std::int32_t n_levels,
                        const std::vector<Eigen::VectorXd>& implied_thresholds) {
  if (n_levels < 2) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: pair label has fewer than two levels"));
  }
  if (block >= implied_thresholds.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: implied threshold block count mismatch"));
  }
  if (implied_thresholds[block].size() != stats.thresholds[block].size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: implied threshold dimension mismatch in block " +
            std::to_string(block)));
  }

  Eigen::VectorXd out(n_levels - 1);
  std::vector<char> seen(static_cast<std::size_t>(n_levels), 0);
  for (Eigen::Index k = 0; k < implied_thresholds[block].size(); ++k) {
    if (stats.threshold_ov[block][static_cast<std::size_t>(k)] != variable) {
      continue;
    }
    const std::int32_t level =
        stats.threshold_level[block][static_cast<std::size_t>(k)];
    if (level < 1 || level >= n_levels) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite: threshold metadata has invalid level"));
    }
    if (seen[static_cast<std::size_t>(level)] != 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite: duplicate threshold metadata for pair variable"));
    }
    out(level - 1) = implied_thresholds[block](k);
    seen[static_cast<std::size_t>(level)] = 1;
  }
  for (std::int32_t level = 1; level < n_levels; ++level) {
    if (seen[static_cast<std::size_t>(level)] == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite: missing implied threshold for pair variable"));
    }
  }
  if (!out.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: implied thresholds are non-finite"));
  }
  for (Eigen::Index k = 1; k < out.size(); ++k) {
    if (!(out(k - 1) < out(k))) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite: implied thresholds must be strictly increasing"));
    }
  }
  return out;
}

struct ThresholdMap {
  std::vector<std::vector<std::int32_t>> free;
  std::vector<std::vector<double>> fixed;
};

post_expected<ThresholdMap>
make_threshold_map(const spec::LatentStructure& pt,
                   const model::MatrixRep& rep,
                   const data::OrdinalStats& stats) {
  ThresholdMap out;
  const std::size_t nb = rep.dims.size();
  out.free.resize(nb);
  out.fixed.resize(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::size_t nth = static_cast<std::size_t>(stats.thresholds[b].size());
    out.free[b].assign(nth, 0);
    out.fixed[b].assign(nth, 0.0);
  }

  std::vector<std::vector<std::int32_t>> seen(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    seen[b].assign(static_cast<std::size_t>(rep.dims[b].n_observed), 0);
  }
  for (std::size_t r = 0; r < pt.size(); ++r) {
    if (pt.op[r] != parse::Op::Threshold || pt.group[r] <= 0) continue;
    const std::size_t b = static_cast<std::size_t>(pt.group[r] - 1);
    if (b >= nb || pt.lhs_var[r] < 0) continue;
    const std::int32_t ov = pt.ov_pos[static_cast<std::size_t>(pt.lhs_var[r])];
    if (ov < 0 || static_cast<std::size_t>(ov) >= seen[b].size()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite: threshold row references invalid observed variable"));
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
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite: threshold row has no sample threshold"));
    }
    out.free[b][static_cast<std::size_t>(pos)] = pt.free[r];
    out.fixed[b][static_cast<std::size_t>(pos)] =
        std::isfinite(pt.fixed_value[r]) ? pt.fixed_value[r] : 0.0;
  }
  return out;
}

std::vector<Eigen::VectorXd>
thresholds_from_theta(const data::OrdinalStats& stats,
                      const ThresholdMap& map,
                      const Eigen::VectorXd& theta) {
  std::vector<Eigen::VectorXd> out = stats.thresholds;
  for (std::size_t b = 0; b < out.size(); ++b) {
    for (Eigen::Index k = 0; k < out[b].size(); ++k) {
      const std::int32_t fr = map.free[b][static_cast<std::size_t>(k)];
      out[b](k) = fr > 0 ? theta(fr - 1) : map.fixed[b][static_cast<std::size_t>(k)];
    }
  }
  return out;
}

std::vector<Eigen::MatrixXd>
correlations_from_moments(const model::ImpliedMoments& moments) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(moments.sigma.size());
  for (const auto& S : moments.sigma) {
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(S.rows(), S.cols());
    for (Eigen::Index j = 0; j < S.cols(); ++j) {
      for (Eigen::Index i = j + 1; i < S.rows(); ++i) {
        const double den = std::sqrt(std::max(S(i, i), kProbFloor) *
                                     std::max(S(j, j), kProbFloor));
        R(i, j) = R(j, i) = std::clamp(S(i, j) / den, -0.999999, 0.999999);
      }
    }
    out.push_back(std::move(R));
  }
  return out;
}

Eigen::MatrixXd expected_counts(double n,
                                const Eigen::VectorXd& thresholds_i,
                                const Eigen::VectorXd& thresholds_j,
                                double rho) {
  Eigen::MatrixXd out(thresholds_i.size() + 1, thresholds_j.size() + 1);
  for (Eigen::Index a = 0; a < out.rows(); ++a) {
    const double lo_i = (a == 0) ? -kInf : thresholds_i(a - 1);
    const double hi_i = (a + 1 == out.rows()) ? kInf : thresholds_i(a);
    for (Eigen::Index b = 0; b < out.cols(); ++b) {
      const double lo_j = (b == 0) ? -kInf : thresholds_j(b - 1);
      const double hi_j = (b + 1 == out.cols()) ? kInf : thresholds_j(b);
      out(a, b) = n * data::ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho);
    }
  }
  return out;
}

post_expected<void>
validate_sample(const data::PairwiseOrdinalStats& sample,
                const std::vector<Eigen::MatrixXd>& implied_correlation,
                const PairwiseOrdinalCompositeOptions& options) {
  const auto& stats = sample.stats;
  const std::size_t nb = stats.R.size();
  if (nb == 0 || sample.block_diagnostics.size() != nb ||
      stats.thresholds.size() != nb || stats.threshold_ov.size() != nb ||
      stats.threshold_level.size() != nb || stats.n_obs.size() != nb ||
      stats.n_levels.size() != nb || implied_correlation.size() != nb) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: block count mismatch"));
  }
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: invalid rho bounds"));
  }
  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = stats.R[b].rows();
    if (p < 2 || stats.R[b].cols() != p ||
        implied_correlation[b].rows() != p ||
        implied_correlation[b].cols() != p ||
        stats.threshold_ov[b].size() !=
            static_cast<std::size_t>(stats.thresholds[b].size()) ||
        stats.threshold_level[b].size() !=
            static_cast<std::size_t>(stats.thresholds[b].size()) ||
        stats.n_levels[b].size() != static_cast<std::size_t>(p) ||
        stats.n_obs[b] <= 0 ||
        !matrix_all_finite(implied_correlation[b])) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite: invalid block " + std::to_string(b)));
    }
  }
  return {};
}

double scaling_weight_for_pair(std::int64_t n_obs,
                               PairwiseCompositeWeighting weighting) {
  if (weighting == PairwiseCompositeWeighting::EqualPair) return 1.0;
  return static_cast<double>(n_obs);
}

double weighted_pair_negloglik(double negloglik,
                               std::int64_t n_obs,
                               PairwiseCompositeWeighting weighting) {
  if (weighting == PairwiseCompositeWeighting::EqualPair) {
    return negloglik / static_cast<double>(n_obs);
  }
  return negloglik;
}

void accumulate_pair(PairwiseOrdinalCompositeBlock& block,
                     PairwiseOrdinalCompositePair pair) {
  block.negloglik += pair.negloglik;
  block.weighted_negloglik += pair.weighted_negloglik;
  block.scaling_denominator += pair.scaling_weight;
  block.pairs.push_back(std::move(pair));
}

void finalize_block(PairwiseOrdinalCompositeBlock& block,
                    PairwiseCompositeScaling scaling) {
  block.objective = scaling == PairwiseCompositeScaling::AverageNegLogLik
      ? block.weighted_negloglik / block.scaling_denominator
      : block.weighted_negloglik;
}

void accumulate_block(PairwiseOrdinalCompositeResult& out,
                      PairwiseOrdinalCompositeBlock block) {
  out.negloglik += block.negloglik;
  out.weighted_negloglik += block.weighted_negloglik;
  out.scaling_denominator += block.scaling_denominator;
  out.blocks.push_back(std::move(block));
}

post_expected<void>
finalize_result(PairwiseOrdinalCompositeResult& out,
                PairwiseCompositeScaling scaling) {
  if (!(out.scaling_denominator > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: non-positive scaling denominator"));
  }
  out.objective = scaling == PairwiseCompositeScaling::AverageNegLogLik
      ? out.weighted_negloglik / out.scaling_denominator
      : out.weighted_negloglik;
  return {};
}

data::OrdinalPairJointMlOptions joint_options(
    PairwiseOrdinalCompositeOptions options) {
  data::OrdinalPairJointMlOptions out;
  out.rho_lower = options.rho_lower;
  out.rho_upper = options.rho_upper;
  out.lavaan_adjust_2x2 = options.lavaan_adjust_2x2;
  return out;
}

post_expected<Eigen::MatrixXd>
score_contributions_from_counts(const Eigen::MatrixXd& counts,
                                const Eigen::VectorXd& thresholds_i,
                                const Eigen::VectorXd& thresholds_j,
                                double rho,
                                std::string_view caller) {
  if (counts.rows() != thresholds_i.size() + 1 ||
      counts.cols() != thresholds_j.size() + 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": score count/threshold dimension mismatch"));
  }
  if (!counts.allFinite() || (counts.array() < 0.0).any()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": score counts must be finite and nonnegative"));
  }
  std::int64_t n = 0;
  for (Eigen::Index r = 0; r < counts.rows(); ++r) {
    for (Eigen::Index c = 0; c < counts.cols(); ++c) {
      const double v = counts(r, c);
      const double rounded = std::round(v);
      if (std::abs(v - rounded) > 1e-10) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            std::string(caller) + ": score counts must be integer-valued"));
      }
      n += static_cast<std::int64_t>(rounded);
    }
  }
  if (n <= 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": score counts have no observations"));
  }

  Eigen::VectorXi x_i(n);
  Eigen::VectorXi x_j(n);
  Eigen::Index row = 0;
  for (Eigen::Index r = 0; r < counts.rows(); ++r) {
    for (Eigen::Index c = 0; c < counts.cols(); ++c) {
      const auto reps = static_cast<Eigen::Index>(std::llround(counts(r, c)));
      for (Eigen::Index k = 0; k < reps; ++k) {
        x_i(row) = static_cast<int>(r);
        x_j(row) = static_cast<int>(c);
        ++row;
      }
    }
  }

  auto scores = data::ordinal_pair_scores(
      x_i, x_j, rho, thresholds_i, thresholds_j);
  if (!scores.has_value()) return std::unexpected(scores.error());

  Eigen::MatrixXd out(
      n, thresholds_i.size() + thresholds_j.size() + 1);
  out.leftCols(thresholds_i.size()) = scores->threshold_i;
  out.middleCols(thresholds_i.size(), thresholds_j.size()) =
      scores->threshold_j;
  out.col(out.cols() - 1) = scores->rho;
  return out;
}

Eigen::MatrixXd score_gamma(const Eigen::MatrixXd& scores) {
  if (scores.rows() == 0) {
    return Eigen::MatrixXd::Zero(scores.cols(), scores.cols());
  }
  return (scores.transpose() * scores) / static_cast<double>(scores.rows());
}

data::PairwiseOrdinalStats
pairwise_stats_from_saturated(const PairwiseOrdinalObservedData& data) {
  data::PairwiseOrdinalStats out;
  out.stats = data.stats;
  out.block_diagnostics.resize(data.saturated.blocks.size());
  for (std::size_t b = 0; b < data.saturated.blocks.size(); ++b) {
    auto& bd = out.block_diagnostics[b];
    bd.pair_diagnostics.reserve(data.saturated.blocks[b].pairs.size());
    for (const auto& pair : data.saturated.blocks[b].pairs) {
      data::OrdinalPairDiagnostics pd;
      pd.label = pair.label;
      pd.rho = pair.rho;
      pd.negloglik = pair.negloglik;
      pd.n_obs = pair.n_obs;
      pd.n_missing = pair.n_missing;
      pd.hit_lower = pair.hit_lower;
      pd.hit_upper = pair.hit_upper;
      pd.counts = pair.counts;
      pd.adjusted_counts = pair.adjusted_counts;
      pd.expected_counts = pair.expected_counts;
      pd.residual_counts = pair.residual_counts;
      bd.pair_diagnostics.push_back(std::move(pd));
    }
  }
  return out;
}

post_expected<PairwiseOrdinalCompositeResult>
evaluate_composite_theta(const spec::LatentStructure& pt,
                         const PairwiseOrdinalObservedData& data,
                         const model::ModelEvaluator& ev,
                         const ThresholdMap& th_map,
                         const Eigen::VectorXd& theta,
                         PairwiseOrdinalCompositeOptions options) {
  if (theta.size() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: theta size mismatch"));
  }
  auto eval = ev.evaluate(theta, false, false);
  if (!eval.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: model evaluation failed: " +
            eval.error().detail));
  }
  auto thresholds = thresholds_from_theta(data.stats, th_map, theta);
  auto R = correlations_from_moments(eval->moments);
  auto sample = pairwise_stats_from_saturated(data);
  return pairwise_ordinal_composite_objective(sample, thresholds, R, options);
}

struct ObservedPairRows {
  data::OrdinalPairLabel label;
  std::vector<std::int32_t> row_index;
  Eigen::VectorXi x_i;
  Eigen::VectorXi x_j;
};

post_expected<ObservedPairRows>
observed_pair_rows(const Eigen::MatrixXd& X,
                   const data::OrdinalPairLabel& label) {
  ObservedPairRows out;
  out.label = label;
  std::vector<int> vi;
  std::vector<int> vj;
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    const double ai = X(r, label.i);
    const double aj = X(r, label.j);
    if (!std::isfinite(ai) || !std::isfinite(aj)) continue;
    const int ci = static_cast<int>(std::llround(ai));
    const int cj = static_cast<int>(std::llround(aj));
    if (std::abs(ai - ci) > 1e-10 || std::abs(aj - cj) > 1e-10 ||
        ci < 1 || ci > label.n_levels_i || cj < 1 || cj > label.n_levels_j) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite: observed category outside declared levels"));
    }
    out.row_index.push_back(static_cast<std::int32_t>(r));
    vi.push_back(ci - 1);
    vj.push_back(cj - 1);
  }
  out.x_i.resize(static_cast<Eigen::Index>(vi.size()));
  out.x_j.resize(static_cast<Eigen::Index>(vj.size()));
  for (Eigen::Index r = 0; r < out.x_i.size(); ++r) {
    out.x_i(r) = vi[static_cast<std::size_t>(r)];
    out.x_j(r) = vj[static_cast<std::size_t>(r)];
  }
  return out;
}

post_expected<void>
validate_observed_blocks(const std::vector<Eigen::MatrixXd>& X,
                         const std::vector<std::vector<std::int32_t>>& n_levels,
                         const PairwiseOrdinalCompositeOptions& options) {
  if (X.empty() || X.size() != n_levels.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal observed joint composite: block count mismatch"));
  }
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal observed joint composite: invalid rho bounds"));
  }
  for (std::size_t b = 0; b < X.size(); ++b) {
    if (X[b].rows() < 2 || X[b].cols() < 2 ||
        n_levels[b].size() != static_cast<std::size_t>(X[b].cols())) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal observed joint composite: invalid block " +
              std::to_string(b)));
    }
    for (std::int32_t level_count : n_levels[b]) {
      if (level_count < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "pairwise ordinal observed joint composite: fewer than two levels in block " +
                std::to_string(b)));
      }
    }
  }
  return {};
}

data::OrdinalStats
stats_from_observed_blocks(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& n_levels,
    const PairwiseOrdinalCompositeResult& saturated) {
  data::OrdinalStats stats;
  const std::size_t nb = X.size();
  stats.R.resize(nb);
  stats.thresholds.resize(nb);
  stats.threshold_ov.resize(nb);
  stats.threshold_level.resize(nb);
  stats.n_obs.resize(nb);
  stats.n_levels = n_levels;
  stats.ov_names.resize(nb);

  for (std::size_t b = 0; b < nb; ++b) {
    const Eigen::Index p = X[b].cols();
    stats.R[b] = Eigen::MatrixXd::Identity(p, p);
    for (const auto& pair : saturated.blocks[b].pairs) {
      stats.R[b](pair.label.i, pair.label.j) = pair.rho;
      stats.R[b](pair.label.j, pair.label.i) = pair.rho;
    }
    Eigen::Index nth = 0;
    for (std::int32_t nl : n_levels[b]) nth += nl - 1;
    stats.thresholds[b].resize(nth);
    stats.threshold_ov[b].reserve(static_cast<std::size_t>(nth));
    stats.threshold_level[b].reserve(static_cast<std::size_t>(nth));
    Eigen::Index off = 0;
    for (Eigen::Index c = 0; c < p; ++c) {
      std::vector<std::int64_t> counts(
          static_cast<std::size_t>(n_levels[b][static_cast<std::size_t>(c)]), 0);
      std::int64_t n = 0;
      for (Eigen::Index r = 0; r < X[b].rows(); ++r) {
        const double v = X[b](r, c);
        if (!std::isfinite(v)) continue;
        const int cat = static_cast<int>(std::llround(v));
        if (cat >= 1 && cat <= n_levels[b][static_cast<std::size_t>(c)]) {
          ++counts[static_cast<std::size_t>(cat - 1)];
          ++n;
        }
      }
      std::int64_t cum = 0;
      for (std::int32_t lev = 1;
           lev < n_levels[b][static_cast<std::size_t>(c)]; ++lev) {
        cum += counts[static_cast<std::size_t>(lev - 1)];
        const double prob = n > 0
            ? static_cast<double>(cum) / static_cast<double>(n)
            : 0.5;
        stats.thresholds[b](off) = normal_quantile(prob);
        stats.threshold_ov[b].push_back(static_cast<std::int32_t>(c));
        stats.threshold_level[b].push_back(lev);
        ++off;
      }
    }
    stats.n_obs[b] = X[b].rows();
  }
  return stats;
}

post_expected<Eigen::MatrixXd>
invert_symmetric(const Eigen::MatrixXd& A, std::string_view who) {
  if (A.rows() != A.cols() || A.rows() == 0 || !A.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        std::string(who) + ": matrix is not finite square"));
  }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
  if (ldlt.info() != Eigen::Success) {
    return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
        std::string(who) + ": LDLT failed"));
  }
  const Eigen::VectorXd d = ldlt.vectorD();
  const double max_abs = d.cwiseAbs().maxCoeff();
  const double tol = std::max(1e-12, 1e-10 * max_abs);
  for (Eigen::Index k = 0; k < d.size(); ++k) {
    if (!(std::abs(d(k)) > tol)) {
      return std::unexpected(make_err(PostError::Kind::InfoMatrixSingular,
          std::string(who) + ": matrix is singular"));
    }
  }
  Eigen::MatrixXd inv = ldlt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
  return 0.5 * (inv + inv.transpose());
}

fit_expected<optim::OptimResult>
run_composite_scalar(const optim::ScalarProblem& prob,
                     const Eigen::VectorXd& x0,
                     const Bounds& bounds,
                     Backend backend,
                     optim::OptimOptions opts) {
  switch (backend) {
    case Backend::NloptSlsqp:
      return optim::nlopt_slsqp(prob, x0, bounds, opts);
    case Backend::NloptBobyqa:
      return optim::nlopt_bobyqa(prob, x0, bounds, opts);
    case Backend::NloptTnewton:
      return optim::nlopt_tnewton(prob, x0, bounds, opts);
    case Backend::NloptVar2:
      return optim::nlopt_var2(prob, x0, bounds, opts);
    case Backend::NloptLbfgs:
      return optim::nlopt_lbfgs(prob, x0, bounds, opts);
    case Backend::Port:
#ifdef MAGMAAN_WITH_PORT
      return optim::port(prob, x0, bounds, opts);
#else
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "pairwise ordinal composite: Port backend requested but MAGMAAN_WITH_PORT is off"));
#endif
    case Backend::Ipopt:
#ifdef MAGMAAN_WITH_IPOPT
      return optim::ipopt(prob, x0, bounds, opts);
#else
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "pairwise ordinal composite: IPOPT backend requested but MAGMAAN_WITH_IPOPT is off"));
#endif
    case Backend::Ceres:
    case Backend::CeresBfgs:
    case Backend::PortNls:
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "pairwise ordinal composite: requested backend is not a scalar optimizer"));
  }
  return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
      "pairwise ordinal composite: unknown optimizer backend"));
}

Eigen::VectorXd finite_diff_gradient(
    const std::function<post_expected<double>(const Eigen::VectorXd&)>& f,
    const Eigen::VectorXd& x,
    double step) {
  Eigen::VectorXd g = Eigen::VectorXd::Zero(x.size());
  for (Eigen::Index k = 0; k < x.size(); ++k) {
    const double h = step * std::max(1.0, std::abs(x(k)));
    Eigen::VectorXd xp = x;
    Eigen::VectorXd xm = x;
    xp(k) += h;
    xm(k) -= h;
    auto fp = f(xp);
    auto fm = f(xm);
    if (fp.has_value() && fm.has_value() && std::isfinite(*fp) &&
        std::isfinite(*fm)) {
      g(k) = (*fp - *fm) / (2.0 * h);
    }
  }
  return g;
}

}  // namespace

post_expected<PairwiseOrdinalCompositeResult>
pairwise_ordinal_composite_objective(
    const data::PairwiseOrdinalStats& sample,
    const std::vector<Eigen::VectorXd>& implied_thresholds,
    const std::vector<Eigen::MatrixXd>& implied_correlation,
    PairwiseOrdinalCompositeOptions options) {
  auto ok = validate_sample(sample, implied_correlation, options);
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (implied_thresholds.size() != sample.stats.thresholds.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite: implied threshold block count mismatch"));
  }

  PairwiseOrdinalCompositeResult out;
  out.blocks.reserve(sample.block_diagnostics.size());

  for (std::size_t b = 0; b < sample.block_diagnostics.size(); ++b) {
    PairwiseOrdinalCompositeBlock block;
    block.n_obs = sample.stats.n_obs[b];
    block.pairs.reserve(sample.block_diagnostics[b].pair_diagnostics.size());

    for (const auto& pd : sample.block_diagnostics[b].pair_diagnostics) {
      if (pd.n_obs <= 0 || pd.counts.rows() != pd.label.n_levels_i ||
          pd.counts.cols() != pd.label.n_levels_j) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "pairwise ordinal composite: invalid pair diagnostics in block " +
                std::to_string(b)));
      }
      if (pd.label.block != static_cast<std::int32_t>(b) ||
          pd.label.i <= pd.label.j ||
          pd.label.i >= implied_correlation[b].rows() ||
          pd.label.j < 0 ||
          pd.label.j >= implied_correlation[b].cols()) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "pairwise ordinal composite: pair label mismatch in block " +
                std::to_string(b)));
      }

      auto th_i = thresholds_for_variable(
          sample.stats, b, pd.label.i, pd.label.n_levels_i, implied_thresholds);
      if (!th_i.has_value()) return std::unexpected(th_i.error());
      auto th_j = thresholds_for_variable(
          sample.stats, b, pd.label.j, pd.label.n_levels_j, implied_thresholds);
      if (!th_j.has_value()) return std::unexpected(th_j.error());

      const double rho = implied_correlation[b](pd.label.i, pd.label.j);
      if (!std::isfinite(rho) || rho <= -1.0 || rho >= 1.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "pairwise ordinal composite: implied rho must be finite and inside (-1, 1)"));
      }
      const Eigen::MatrixXd counts = options.lavaan_adjust_2x2
          ? pd.adjusted_counts
          : pd.counts;
      auto nll = data::ordinal_pair_negloglik(counts, *th_i, *th_j, rho);
      if (!nll.has_value()) return std::unexpected(nll.error());
      const double weighted =
          weighted_pair_negloglik(*nll, pd.n_obs, options.weighting);
      const double scale_weight =
          scaling_weight_for_pair(pd.n_obs, options.weighting);
      Eigen::MatrixXd expected =
          expected_counts(counts.sum(), *th_i, *th_j, rho);
      Eigen::MatrixXd residual = counts - expected;
      auto score_or = score_contributions_from_counts(
          pd.counts, *th_i, *th_j, rho, "pairwise ordinal composite");
      if (!score_or.has_value()) return std::unexpected(score_or.error());
      Eigen::MatrixXd gamma = score_gamma(*score_or);

      PairwiseOrdinalCompositePair pair{
          .label = pd.label,
          .thresholds_i = std::move(*th_i),
          .thresholds_j = std::move(*th_j),
          .rho = rho,
          .negloglik = *nll,
          .weighted_negloglik = weighted,
          .scaling_weight = scale_weight,
          .n_obs = pd.n_obs,
          .n_missing = pd.n_missing,
          .hit_lower = rho <= options.rho_lower,
          .hit_upper = rho >= options.rho_upper,
          .counts = pd.counts,
          .adjusted_counts = counts,
          .expected_counts = std::move(expected),
          .residual_counts = std::move(residual),
          .score_contributions = std::move(*score_or),
          .score_gamma = std::move(gamma)};

      accumulate_pair(block, std::move(pair));
    }

    if (block.pairs.empty()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite: block has no pairs"));
    }
    finalize_block(block, options.scaling);
    accumulate_block(out, std::move(block));
  }

  auto done = finalize_result(out, options.scaling);
  if (!done.has_value()) return std::unexpected(done.error());
  return out;
}

post_expected<PairwiseOrdinalCompositeResult>
pairwise_ordinal_joint_composite_objective(
    const data::PairwiseOrdinalStats& sample,
    PairwiseOrdinalCompositeOptions options) {
  auto ok = validate_sample(sample, sample.stats.R, options);
  if (!ok.has_value()) return std::unexpected(ok.error());

  PairwiseOrdinalCompositeResult out;
  out.blocks.reserve(sample.block_diagnostics.size());
  const auto joint_opts = joint_options(options);

  for (std::size_t b = 0; b < sample.block_diagnostics.size(); ++b) {
    PairwiseOrdinalCompositeBlock block;
    block.n_obs = sample.stats.n_obs[b];
    block.pairs.reserve(sample.block_diagnostics[b].pair_diagnostics.size());

    for (const auto& pd : sample.block_diagnostics[b].pair_diagnostics) {
      if (pd.n_obs <= 0 || pd.counts.rows() != pd.label.n_levels_i ||
          pd.counts.cols() != pd.label.n_levels_j) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "pairwise ordinal joint composite: invalid pair diagnostics in block " +
                std::to_string(b)));
      }
      auto joint = data::fit_ordinal_pair_joint_ml(pd.counts, joint_opts);
      if (!joint.has_value()) return std::unexpected(joint.error());

      Eigen::MatrixXd expected =
          expected_counts(joint->adjusted_counts.sum(), joint->thresholds_i,
                          joint->thresholds_j, joint->rho);
      Eigen::MatrixXd residual = joint->adjusted_counts - expected;
      auto score_or = score_contributions_from_counts(
          pd.counts, joint->thresholds_i, joint->thresholds_j, joint->rho,
          "pairwise ordinal joint composite");
      if (!score_or.has_value()) return std::unexpected(score_or.error());
      Eigen::MatrixXd gamma = score_gamma(*score_or);
      const double weighted =
          weighted_pair_negloglik(joint->negloglik, pd.n_obs, options.weighting);
      const double scale_weight =
          scaling_weight_for_pair(pd.n_obs, options.weighting);

      PairwiseOrdinalCompositePair pair{
          .label = pd.label,
          .thresholds_i = std::move(joint->thresholds_i),
          .thresholds_j = std::move(joint->thresholds_j),
          .rho = joint->rho,
          .negloglik = joint->negloglik,
          .weighted_negloglik = weighted,
          .scaling_weight = scale_weight,
          .n_obs = pd.n_obs,
          .n_missing = pd.n_missing,
          .hit_lower = joint->hit_lower,
          .hit_upper = joint->hit_upper,
          .counts = pd.counts,
          .adjusted_counts = std::move(joint->adjusted_counts),
          .expected_counts = std::move(expected),
          .residual_counts = std::move(residual),
          .score_contributions = std::move(*score_or),
          .score_gamma = std::move(gamma)};

      accumulate_pair(block, std::move(pair));
    }

    if (block.pairs.empty()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal joint composite: block has no pairs"));
    }
    finalize_block(block, options.scaling);
    accumulate_block(out, std::move(block));
  }

  auto done = finalize_result(out, options.scaling);
  if (!done.has_value()) return std::unexpected(done.error());
  return out;
}

post_expected<PairwiseOrdinalCompositeResult>
pairwise_ordinal_observed_joint_composite_objective(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& n_levels,
    PairwiseOrdinalCompositeOptions options) {
  auto ok = validate_observed_blocks(X, n_levels, options);
  if (!ok.has_value()) return std::unexpected(ok.error());

  PairwiseOrdinalCompositeResult out;
  out.blocks.reserve(X.size());
  const auto joint_opts = joint_options(options);

  for (std::size_t b = 0; b < X.size(); ++b) {
    const Eigen::Index p = X[b].cols();
    PairwiseOrdinalCompositeBlock block;
    block.n_obs = static_cast<std::int64_t>(X[b].rows());
    block.pairs.reserve(static_cast<std::size_t>(p * (p - 1) / 2));

    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        auto joint = data::fit_ordinal_pair_observed_joint_ml(
            X[b].col(i), X[b].col(j),
            n_levels[b][static_cast<std::size_t>(i)],
            n_levels[b][static_cast<std::size_t>(j)],
            joint_opts);
        if (!joint.has_value()) return std::unexpected(joint.error());

        Eigen::MatrixXd expected =
            expected_counts(joint->fit.adjusted_counts.sum(),
                            joint->fit.thresholds_i,
                            joint->fit.thresholds_j,
                            joint->fit.rho);
        Eigen::MatrixXd residual = joint->fit.adjusted_counts - expected;
        auto score_or = score_contributions_from_counts(
            joint->counts, joint->fit.thresholds_i, joint->fit.thresholds_j,
            joint->fit.rho, "pairwise ordinal observed joint composite");
        if (!score_or.has_value()) return std::unexpected(score_or.error());
        Eigen::MatrixXd gamma = score_gamma(*score_or);
        const double weighted =
            weighted_pair_negloglik(joint->fit.negloglik, joint->n_obs,
                                    options.weighting);
        const double scale_weight =
            scaling_weight_for_pair(joint->n_obs, options.weighting);

        PairwiseOrdinalCompositePair pair{
            .label = data::OrdinalPairLabel{
                .block = static_cast<std::int32_t>(b),
                .i = static_cast<std::int32_t>(i),
                .j = static_cast<std::int32_t>(j),
                .n_levels_i = n_levels[b][static_cast<std::size_t>(i)],
                .n_levels_j = n_levels[b][static_cast<std::size_t>(j)]},
            .thresholds_i = std::move(joint->fit.thresholds_i),
            .thresholds_j = std::move(joint->fit.thresholds_j),
            .rho = joint->fit.rho,
            .negloglik = joint->fit.negloglik,
            .weighted_negloglik = weighted,
            .scaling_weight = scale_weight,
            .n_obs = joint->n_obs,
            .n_missing = joint->n_missing,
            .hit_lower = joint->fit.hit_lower,
            .hit_upper = joint->fit.hit_upper,
            .counts = std::move(joint->counts),
            .adjusted_counts = std::move(joint->fit.adjusted_counts),
            .expected_counts = std::move(expected),
            .residual_counts = std::move(residual),
            .score_contributions = std::move(*score_or),
            .score_gamma = std::move(gamma)};

        accumulate_pair(block, std::move(pair));
      }
    }

    if (block.pairs.empty()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal observed joint composite: block has no pairs"));
    }
    finalize_block(block, options.scaling);
    accumulate_block(out, std::move(block));
  }

  auto done = finalize_result(out, options.scaling);
  if (!done.has_value()) return std::unexpected(done.error());
  return out;
}

post_expected<PairwiseOrdinalObservedData>
pairwise_ordinal_observed_data(
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<std::vector<std::int32_t>>& n_levels,
    PairwiseOrdinalCompositeOptions options) {
  auto sat = pairwise_ordinal_observed_joint_composite_objective(
      X, n_levels, options);
  if (!sat.has_value()) return std::unexpected(sat.error());
  PairwiseOrdinalObservedData out;
  out.X = X;
  out.saturated = std::move(*sat);
  out.stats = stats_from_observed_blocks(X, n_levels, out.saturated);
  out.options = options;
  return out;
}

fit_expected<PairwiseOrdinalCompositeFit>
fit_pairwise_ordinal_composite(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const PairwiseOrdinalObservedData& data,
    const Eigen::VectorXd& x0,
    Bounds bounds,
    Backend backend,
    optim::OptimOptions opts,
    spec::Starts starts) {
  if (pt.nonlinear_eq_rows.size() > 0 || pt.has_inequality_constraints ||
      pt.has_unenforced_constraints) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        "fit_pairwise_ordinal_composite: only linear equality constraints are supported"));
  }
  if (auto p = prepare_ordinal_delta_partable(pt, data.stats, &starts);
      !p.has_value()) {
    return std::unexpected(p.error());
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_fit_err(FitError::Kind::InvalidStartValues,
        "fit_pairwise_ordinal_composite: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto map_or = make_threshold_map(pt, rep, data.stats);
  if (!map_or.has_value()) return std::unexpected(make_fit_err(
      FitError::Kind::NumericIssue, map_or.error().detail));
  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        con_or.error().detail));
  }

  Eigen::VectorXd start = x0;
  if (start.size() == 0) {
    auto x0_or = ordinal_start_values(pt, rep, data.stats, std::move(starts));
    if (!x0_or.has_value()) return std::unexpected(x0_or.error());
    start = std::move(*x0_or);
  }
  if (start.size() != pt.n_free()) {
    return std::unexpected(make_fit_err(FitError::Kind::InvalidStartValues,
        "fit_pairwise_ordinal_composite: x0 size does not match prepared partable"));
  }
  if (bounds.empty()) {
    auto b_or = bounds_from_partable(pt);
    if (!b_or.has_value()) {
      return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
          "fit_pairwise_ordinal_composite: bounds_from_partable failed: " +
              b_or.error().detail));
    }
    bounds = std::move(*b_or);
  }

  const auto objective = [&](const Eigen::VectorXd& theta)
      -> post_expected<double> {
    auto obj = evaluate_composite_theta(
        pt, data, *ev_or, *map_or, theta, data.options);
    if (!obj.has_value()) return std::unexpected(obj.error());
    return obj->weighted_negloglik;
  };

  optim::ScalarProblem prob;
  prob.n_param = start.size();
  prob.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    auto f = objective(x);
    if (!f.has_value() || !std::isfinite(*f)) {
      grad = Eigen::VectorXd::Zero(x.size());
      return std::numeric_limits<double>::infinity();
    }
    if (grad.size() == x.size()) {
      grad = finite_diff_gradient(objective, x, 1e-5);
    }
    return *f;
  };

  optim::ScalarProblem run_prob = prob;
  Eigen::VectorXd run_start = start;
  Bounds run_bounds = bounds;
  if (con_or->active()) {
    run_prob = optim::reparameterize(prob, *con_or);
    run_start = con_or->contract(start);
    const bool pure_merge = !con_or->group.empty();
    run_bounds = (!bounds.empty() && pure_merge)
        ? optim::fold_alpha_bounds(*con_or, bounds)
        : Bounds{};
  }
  auto opt = run_composite_scalar(run_prob, run_start, run_bounds, backend, opts);
  if (!opt.has_value()) return std::unexpected(opt.error());
  Eigen::VectorXd theta = run_prob.expand(opt->x);
  auto final_obj = evaluate_composite_theta(
      pt, data, *ev_or, *map_or, theta, data.options);
  if (!final_obj.has_value()) {
    return std::unexpected(make_fit_err(FitError::Kind::NumericIssue,
        final_obj.error().detail));
  }
  PairwiseOrdinalCompositeFit out;
  out.estimates = Estimates{std::move(theta), final_obj->negloglik,
                            opt->iterations, opt->f_evals, opt->g_evals,
                            opt->status, opt->grad_inf_norm, opt->audit};
  out.estimates.fmin = final_obj->weighted_negloglik;
  out.objective = std::move(*final_obj);
  out.df = -1;
  return out;
}

post_expected<PairwiseOrdinalCompositeGodambe>
pairwise_ordinal_composite_godambe(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const PairwiseOrdinalObservedData& data,
    const Estimates& est,
    double fd_step) {
  if (auto p = prepare_ordinal_delta_partable(pt, data.stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite Godambe: ModelEvaluator::build failed: " +
            ev_or.error().detail));
  }
  auto map_or = make_threshold_map(pt, rep, data.stats);
  if (!map_or.has_value()) return std::unexpected(map_or.error());
  if (est.theta.size() != pt.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "pairwise ordinal composite Godambe: theta size mismatch"));
  }

  const Eigen::Index q = est.theta.size();
  Eigen::Index n_total = 0;
  for (const auto& Xb : data.X) n_total += Xb.rows();
  auto case_nll = [&](const Eigen::VectorXd& theta)
      -> post_expected<Eigen::MatrixXd> {
    auto eval = ev_or->evaluate(theta, false, false);
    if (!eval.has_value()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "pairwise ordinal composite Godambe: evaluation failed: " +
              eval.error().detail));
    }
    auto thresholds = thresholds_from_theta(data.stats, *map_or, theta);
    auto R = correlations_from_moments(eval->moments);
    Eigen::MatrixXd rows = Eigen::MatrixXd::Zero(n_total, 1);
    Eigen::Index row_off = 0;
    for (std::size_t b = 0; b < data.X.size(); ++b) {
      for (const auto& pair : data.saturated.blocks[b].pairs) {
        const double pair_weight =
            data.options.weighting == PairwiseCompositeWeighting::EqualPair
                ? 1.0 / static_cast<double>(pair.n_obs)
                : 1.0;
        auto obs = observed_pair_rows(data.X[b], pair.label);
        if (!obs.has_value()) return std::unexpected(obs.error());
        auto th_i = thresholds_for_variable(
            data.stats, b, pair.label.i, pair.label.n_levels_i, thresholds);
        if (!th_i.has_value()) return std::unexpected(th_i.error());
        auto th_j = thresholds_for_variable(
            data.stats, b, pair.label.j, pair.label.n_levels_j, thresholds);
        if (!th_j.has_value()) return std::unexpected(th_j.error());
        const double rho = R[b](pair.label.i, pair.label.j);
        for (Eigen::Index r = 0; r < obs->x_i.size(); ++r) {
          const Eigen::Index global =
              row_off + obs->row_index[static_cast<std::size_t>(r)];
          const Eigen::Index ai = obs->x_i(r);
          const Eigen::Index aj = obs->x_j(r);
          const double lo_i = ai == 0 ? -kInf : (*th_i)(ai - 1);
          const double hi_i = ai + 1 == pair.label.n_levels_i ? kInf : (*th_i)(ai);
          const double lo_j = aj == 0 ? -kInf : (*th_j)(aj - 1);
          const double hi_j = aj + 1 == pair.label.n_levels_j ? kInf : (*th_j)(aj);
          const double pr = std::max(
              kProbFloor,
              data::ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
          rows(global, 0) -= pair_weight * std::log(pr);
        }
      }
      row_off += data.X[b].rows();
    }
    return rows;
  };

  Eigen::MatrixXd scores(n_total, q);
  for (Eigen::Index k = 0; k < q; ++k) {
    const double h = fd_step * std::max(1.0, std::abs(est.theta(k)));
    Eigen::VectorXd xp = est.theta;
    Eigen::VectorXd xm = est.theta;
    xp(k) += h;
    xm(k) -= h;
    auto vp = case_nll(xp);
    auto vm = case_nll(xm);
    if (!vp.has_value()) return std::unexpected(vp.error());
    if (!vm.has_value()) return std::unexpected(vm.error());
    scores.col(k) = (vp->col(0) - vm->col(0)) / (2.0 * h);
  }
  Eigen::MatrixXd meat =
      (scores.transpose() * scores) / static_cast<double>(n_total);

  auto total_grad = [&](const Eigen::VectorXd& theta)
      -> post_expected<Eigen::VectorXd> {
    auto v = case_nll(theta);
    if (!v.has_value()) return std::unexpected(v.error());
    Eigen::VectorXd g(q);
    for (Eigen::Index k = 0; k < q; ++k) {
      const double h = fd_step * std::max(1.0, std::abs(theta(k)));
      Eigen::VectorXd xp = theta;
      Eigen::VectorXd xm = theta;
      xp(k) += h;
      xm(k) -= h;
      auto vp = case_nll(xp);
      auto vm = case_nll(xm);
      if (!vp.has_value()) return std::unexpected(vp.error());
      if (!vm.has_value()) return std::unexpected(vm.error());
      g(k) = (vp->sum() - vm->sum()) / (2.0 * h);
    }
    return g;
  };

  Eigen::MatrixXd bread(q, q);
  for (Eigen::Index k = 0; k < q; ++k) {
    const double h = fd_step * std::max(1.0, std::abs(est.theta(k)));
    Eigen::VectorXd xp = est.theta;
    Eigen::VectorXd xm = est.theta;
    xp(k) += h;
    xm(k) -= h;
    auto gp = total_grad(xp);
    auto gm = total_grad(xm);
    if (!gp.has_value()) return std::unexpected(gp.error());
    if (!gm.has_value()) return std::unexpected(gm.error());
    bread.col(k) = (*gp - *gm) / (2.0 * h) / static_cast<double>(n_total);
  }
  bread = 0.5 * (bread + bread.transpose());
  auto con = build_eq_constraints(pt);
  if (!con.has_value()) return std::unexpected(con.error());
  const Eigen::MatrixXd K =
      con->active() ? con->Kmat : Eigen::MatrixXd::Identity(q, q);
  const Eigen::MatrixXd bread_alpha = K.transpose() * bread * K;
  const Eigen::MatrixXd meat_alpha = K.transpose() * meat * K;
  auto inv_or = invert_symmetric(bread_alpha,
                                 "pairwise ordinal composite Godambe");
  if (!inv_or.has_value()) return std::unexpected(inv_or.error());
  Eigen::MatrixXd vcov_alpha =
      (*inv_or) * meat_alpha * (*inv_or) / static_cast<double>(n_total);
  Eigen::MatrixXd vcov = K * vcov_alpha * K.transpose();
  vcov = 0.5 * (vcov + vcov.transpose());
  Eigen::MatrixXd vcov_naive =
      K * ((*inv_or) / static_cast<double>(n_total)) * K.transpose();

  PairwiseOrdinalCompositeGodambe out;
  out.bread = std::move(bread);
  out.meat = std::move(meat);
  out.vcov = std::move(vcov);
  out.vcov_naive = std::move(vcov_naive);
  out.se.resize(q);
  out.se_naive.resize(q);
  for (Eigen::Index k = 0; k < q; ++k) {
    out.se(k) = std::sqrt(std::max(0.0, out.vcov(k, k)));
    out.se_naive(k) = std::sqrt(std::max(0.0, out.vcov_naive(k, k)));
  }
  out.casewise_scores = std::move(scores);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(out.bread);
  const double smax = svd.singularValues()(0);
  const double smin = svd.singularValues()(svd.singularValues().size() - 1);
  out.condition_bread = smin > 0.0 ? smax / smin : kInf;
  return out;
}

post_expected<robust::LRSatorra2000Result>
lr_test_pairwise_ordinal_composite(
    spec::LatentStructure pt_H1,
    const model::MatrixRep& rep_H1,
    const PairwiseOrdinalObservedData& data,
    const PairwiseOrdinalCompositeFit& fit_H1,
    spec::LatentStructure pt_H0,
    const model::MatrixRep& rep_H0,
    const PairwiseOrdinalCompositeFit& fit_H0,
    robust::SatorraAMethod a_method,
    double fd_step) {
  (void)rep_H0;
  if (auto p = prepare_ordinal_delta_partable(pt_H1, data.stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  if (auto p = prepare_ordinal_delta_partable(pt_H0, data.stats, nullptr);
      !p.has_value()) {
    return std::unexpected(fit_to_post(p.error()));
  }
  auto con1 = build_eq_constraints(pt_H1);
  if (!con1.has_value()) return std::unexpected(con1.error());
  auto con0 = build_eq_constraints(pt_H0);
  if (!con0.has_value()) return std::unexpected(con0.error());
  const int df_diff = con1->n_alpha - con0->n_alpha;
  if (df_diff <= 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_pairwise_ordinal_composite: H1 must be less restricted than H0"));
  }

  Eigen::MatrixXd A_alpha;
  if (a_method == robust::SatorraAMethod::Exact) {
    auto restr = robust::restriction_alpha_from_K(*con1, *con0);
    if (!restr.has_value()) return std::unexpected(restr.error());
    A_alpha = std::move(restr->A);
  } else {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_pairwise_ordinal_composite: delta A-method is not implemented for composite likelihood"));
  }

  auto god = pairwise_ordinal_composite_godambe(
      std::move(pt_H1), rep_H1, data, fit_H1.estimates, fd_step);
  if (!god.has_value()) return std::unexpected(god.error());
  const Eigen::MatrixXd K1 =
      con1->active()
          ? con1->Kmat
          : Eigen::MatrixXd::Identity(god->bread.rows(), god->bread.cols());
  auto sd = robust::compute_satorra2000_from_sandwich(
      K1.transpose() * god->bread * K1,
      K1.transpose() * god->meat * K1,
      A_alpha);
  if (!sd.has_value()) return std::unexpected(sd.error());
  return robust::lr_test_satorra2000(
      2.0 * (fit_H0.objective.weighted_negloglik -
             fit_H1.objective.weighted_negloglik),
      *sd);
}

}  // namespace magmaan::estimate::frontier
