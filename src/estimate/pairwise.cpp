#include "magmaan/estimate/pairwise.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "magmaan/error.hpp"

namespace magmaan::estimate::frontier {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

bool matrix_all_finite(const Eigen::MatrixXd& M) {
  for (Eigen::Index c = 0; c < M.cols(); ++c)
    for (Eigen::Index r = 0; r < M.rows(); ++r)
      if (!std::isfinite(M(r, c))) return false;
  return true;
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

}  // namespace magmaan::estimate::frontier
