#include "magmaan/data/pairwise_mixed.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "magmaan/error.hpp"

namespace magmaan::data {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kProbFloor = 1.4901161193847656e-8;
constexpr double kTwoPi = 6.28318530717958647692;

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

double normal_cdf(double x) noexcept {
  if (x == kInf) return 1.0;
  if (x == -kInf) return 0.0;
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normal_pdf(double x) noexcept {
  if (!std::isfinite(x)) return 0.0;
  constexpr double inv_sqrt_2pi = 0.39894228040143267794;
  return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

post_expected<void>
validate_polyserial_inputs(const Eigen::Ref<const Eigen::VectorXi>& categories,
                           const Eigen::Ref<const Eigen::VectorXd>& u,
                           const Eigen::Ref<const Eigen::VectorXd>& thresholds,
                           std::string_view caller) {
  if (categories.size() != u.size() || categories.size() == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": category and continuous vectors must be nonempty and equal size"));
  }
  if (!u.allFinite() || !thresholds.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": non-finite input"));
  }
  for (Eigen::Index k = 1; k < thresholds.size(); ++k) {
    if (!(thresholds(k - 1) < thresholds(k))) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(caller) + ": thresholds must be strictly increasing"));
    }
  }
  const int n_levels = static_cast<int>(thresholds.size() + 1);
  for (Eigen::Index r = 0; r < categories.size(); ++r) {
    if (categories(r) < 0 || categories(r) >= n_levels) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(caller) + ": category outside threshold range"));
    }
  }
  return {};
}

post_expected<void>
validate_continuous_pair_inputs(const Eigen::Ref<const Eigen::VectorXd>& x_i,
                                const Eigen::Ref<const Eigen::VectorXd>& x_j,
                                std::string_view caller) {
  if (x_i.size() != x_j.size() || x_i.size() < 2) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": vectors must have equal size and at least 2 rows"));
  }
  if (!x_i.allFinite() || !x_j.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": non-finite input"));
  }
  return {};
}

post_expected<void>
validate_normal_pair_parameters(double var_i,
                                double var_j,
                                double cov,
                                std::string_view caller) {
  if (!std::isfinite(var_i) || !std::isfinite(var_j) || !std::isfinite(cov) ||
      var_i <= 0.0 || var_j <= 0.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": variances must be finite and positive"));
  }
  const double det = var_i * var_j - cov * cov;
  if (!std::isfinite(det) || det <= 0.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": covariance matrix must be positive definite"));
  }
  return {};
}

double polyserial_prob_unchecked(int cat,
                                 double u,
                                 double rho,
                                 const Eigen::VectorXd& th) noexcept {
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));
  const double lo = (cat == 0) ? -kInf : th(cat - 1);
  const double hi = (cat == th.size()) ? kInf : th(cat);
  return std::max(kProbFloor,
                  normal_cdf((hi - rho * u) / sd) -
                      normal_cdf((lo - rho * u) / sd));
}

double neglog_polyserial_unchecked(const Eigen::Ref<const Eigen::VectorXi>& cat,
                                   const Eigen::Ref<const Eigen::VectorXd>& u,
                                   const Eigen::Ref<const Eigen::VectorXd>& th,
                                   double rho) noexcept {
  double out = 0.0;
  for (Eigen::Index i = 0; i < cat.size(); ++i) {
    out -= std::log(polyserial_prob_unchecked(cat(i), u(i), rho, th));
  }
  return out;
}

Eigen::MatrixXd score_gamma(const Eigen::MatrixXd& scores) {
  if (scores.rows() == 0) {
    return Eigen::MatrixXd::Zero(scores.cols(), scores.cols());
  }
  return (scores.transpose() * scores) / static_cast<double>(scores.rows());
}

}  // namespace

post_expected<std::vector<MixedPairLabel>>
mixed_pair_labels(const std::vector<std::int32_t>& ordered,
                  std::int32_t n_thresholds) {
  if (ordered.empty() || n_thresholds < 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_pair_labels: ordered mask must be nonempty and threshold count nonnegative"));
  }

  Eigen::Index n_cont = 0;
  for (auto z : ordered) {
    if (z == 0) ++n_cont;
  }
  const Eigen::Index p = static_cast<Eigen::Index>(ordered.size());
  Eigen::Index moment_index = static_cast<Eigen::Index>(n_thresholds) + 2 * n_cont;
  std::vector<MixedPairLabel> out;
  out.reserve(static_cast<std::size_t>(p * (p - 1) / 2));
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      const bool oi = ordered[static_cast<std::size_t>(i)] != 0;
      const bool oj = ordered[static_cast<std::size_t>(j)] != 0;
      MixedPairKind kind = MixedPairKind::continuous_continuous;
      if (oi && oj) {
        kind = MixedPairKind::ordinal_ordinal;
      } else if (oi || oj) {
        kind = MixedPairKind::continuous_ordinal;
      }
      out.push_back(MixedPairLabel{
          .i = static_cast<std::int32_t>(i),
          .j = static_cast<std::int32_t>(j),
          .moment_index = static_cast<std::int32_t>(moment_index++),
          .kind = kind});
    }
  }
  return out;
}

post_expected<std::vector<MixedMomentLabel>>
mixed_moment_labels(const std::vector<std::int32_t>& ordered,
                    const std::vector<std::int32_t>& threshold_ov,
                    const std::vector<std::int32_t>& threshold_level) {
  if (ordered.empty() || threshold_ov.size() != threshold_level.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_moment_labels: invalid ordered mask or threshold metadata"));
  }

  const auto p = static_cast<std::int32_t>(ordered.size());
  std::vector<MixedMomentLabel> out;
  out.reserve(threshold_ov.size() + ordered.size() + ordered.size() +
              ordered.size() * (ordered.size() - 1) / 2);

  std::int32_t index = 0;
  for (std::size_t k = 0; k < threshold_ov.size(); ++k) {
    const std::int32_t ov = threshold_ov[k];
    if (ov < 0 || ov >= p ||
        ordered[static_cast<std::size_t>(ov)] == 0 || threshold_level[k] < 1) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_moment_labels: threshold metadata references an invalid ordered variable"));
    }
    out.push_back(MixedMomentLabel{
        .index = index++,
        .kind = MixedMomentKind::threshold,
        .variable = ov,
        .threshold_level = threshold_level[k]});
  }

  for (std::int32_t j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) {
      out.push_back(MixedMomentLabel{
          .index = index++,
          .kind = MixedMomentKind::continuous_mean,
          .variable = j});
    }
  }
  for (std::int32_t j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) {
      out.push_back(MixedMomentLabel{
          .index = index++,
          .kind = MixedMomentKind::continuous_variance,
          .variable = j});
    }
  }

  auto pairs_or = mixed_pair_labels(ordered, static_cast<std::int32_t>(threshold_ov.size()));
  if (!pairs_or.has_value()) return std::unexpected(pairs_or.error());
  for (const auto& pair : *pairs_or) {
    out.push_back(MixedMomentLabel{
        .index = pair.moment_index,
        .kind = MixedMomentKind::pair,
        .variable_i = pair.i,
        .variable_j = pair.j,
        .pair_kind = pair.kind});
  }
  return out;
}

post_expected<double>
continuous_pair_normal_negloglik(const Eigen::Ref<const Eigen::VectorXd>& x_i,
                                 const Eigen::Ref<const Eigen::VectorXd>& x_j,
                                 double mean_i,
                                 double mean_j,
                                 double var_i,
                                 double var_j,
                                 double cov) {
  auto ok = validate_continuous_pair_inputs(x_i, x_j,
                                            "continuous_pair_normal_negloglik");
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (!std::isfinite(mean_i) || !std::isfinite(mean_j)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "continuous_pair_normal_negloglik: means must be finite"));
  }
  ok = validate_normal_pair_parameters(var_i, var_j, cov,
                                       "continuous_pair_normal_negloglik");
  if (!ok.has_value()) return std::unexpected(ok.error());

  const double det = var_i * var_j - cov * cov;
  const double inv_ii = var_j / det;
  const double inv_jj = var_i / det;
  const double inv_ij = -cov / det;
  double quad = 0.0;
  for (Eigen::Index r = 0; r < x_i.size(); ++r) {
    const double di = x_i(r) - mean_i;
    const double dj = x_j(r) - mean_j;
    quad += inv_ii * di * di + 2.0 * inv_ij * di * dj + inv_jj * dj * dj;
  }
  return 0.5 * static_cast<double>(x_i.size()) * std::log(kTwoPi * kTwoPi * det) +
         0.5 * quad;
}

post_expected<ContinuousPairNormalResult>
fit_continuous_pair_normal_ml(const Eigen::Ref<const Eigen::VectorXd>& x_i,
                              const Eigen::Ref<const Eigen::VectorXd>& x_j) {
  auto ok = validate_continuous_pair_inputs(x_i, x_j,
                                            "fit_continuous_pair_normal_ml");
  if (!ok.has_value()) return std::unexpected(ok.error());

  ContinuousPairNormalResult out;
  out.n_obs = static_cast<std::int64_t>(x_i.size());
  out.mean_i = x_i.mean();
  out.mean_j = x_j.mean();
  for (Eigen::Index r = 0; r < x_i.size(); ++r) {
    const double di = x_i(r) - out.mean_i;
    const double dj = x_j(r) - out.mean_j;
    out.var_i += di * di;
    out.var_j += dj * dj;
    out.cov += di * dj;
  }
  const double n = static_cast<double>(x_i.size());
  out.var_i /= n;
  out.var_j /= n;
  out.cov /= n;
  ok = validate_normal_pair_parameters(out.var_i, out.var_j, out.cov,
                                       "fit_continuous_pair_normal_ml");
  if (!ok.has_value()) return std::unexpected(ok.error());
  out.rho = out.cov / std::sqrt(out.var_i * out.var_j);
  auto nll_or = continuous_pair_normal_negloglik(
      x_i, x_j, out.mean_i, out.mean_j, out.var_i, out.var_j, out.cov);
  if (!nll_or.has_value()) return std::unexpected(nll_or.error());
  out.negloglik = *nll_or;
  auto scores_or = continuous_pair_normal_scores(
      x_i, x_j, out.mean_i, out.mean_j, out.var_i, out.var_j, out.cov);
  if (!scores_or.has_value()) return std::unexpected(scores_or.error());
  out.score_contributions = std::move(scores_or->score_contributions);
  out.score_gamma = std::move(scores_or->score_gamma);
  return out;
}

post_expected<ContinuousPairNormalScores>
continuous_pair_normal_scores(const Eigen::Ref<const Eigen::VectorXd>& x_i,
                              const Eigen::Ref<const Eigen::VectorXd>& x_j,
                              double mean_i,
                              double mean_j,
                              double var_i,
                              double var_j,
                              double cov) {
  auto ok = validate_continuous_pair_inputs(x_i, x_j,
                                            "continuous_pair_normal_scores");
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (!std::isfinite(mean_i) || !std::isfinite(mean_j)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "continuous_pair_normal_scores: means must be finite"));
  }
  ok = validate_normal_pair_parameters(var_i, var_j, cov,
                                       "continuous_pair_normal_scores");
  if (!ok.has_value()) return std::unexpected(ok.error());

  const double det = var_i * var_j - cov * cov;
  const double inv_ii = var_j / det;
  const double inv_jj = var_i / det;
  const double inv_ij = -cov / det;

  ContinuousPairNormalScores out{
      .score_contributions = Eigen::MatrixXd::Zero(x_i.size(), 5),
      .score_gamma = Eigen::MatrixXd::Zero(5, 5)};
  for (Eigen::Index r = 0; r < x_i.size(); ++r) {
    const double di = x_i(r) - mean_i;
    const double dj = x_j(r) - mean_j;
    const double qi = inv_ii * di + inv_ij * dj;
    const double qj = inv_ij * di + inv_jj * dj;
    out.score_contributions(r, 0) = qi;
    out.score_contributions(r, 1) = qj;
    out.score_contributions(r, 2) = 0.5 * (qi * qi - inv_ii);
    out.score_contributions(r, 3) = 0.5 * (qj * qj - inv_jj);
    out.score_contributions(r, 4) = qi * qj - inv_ij;
  }
  out.score_gamma = score_gamma(out.score_contributions);
  return out;
}

post_expected<double>
polyserial_pair_negloglik(const Eigen::Ref<const Eigen::VectorXi>& categories,
                          const Eigen::Ref<const Eigen::VectorXd>& u,
                          const Eigen::Ref<const Eigen::VectorXd>& thresholds,
                          double rho) {
  auto ok = validate_polyserial_inputs(categories, u, thresholds,
                                       "polyserial_pair_negloglik");
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (!std::isfinite(rho) || rho <= -1.0 || rho >= 1.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "polyserial_pair_negloglik: rho must be finite and inside (-1, 1)"));
  }
  return neglog_polyserial_unchecked(categories, u, thresholds, rho);
}

post_expected<PolyserialPairMlResult>
fit_polyserial_pair_rho_ml(
    const Eigen::Ref<const Eigen::VectorXi>& categories,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds,
    PolyserialPairMlOptions options) {
  auto ok = validate_polyserial_inputs(categories, u, thresholds,
                                       "fit_polyserial_pair_rho_ml");
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper) || options.max_iter < 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_polyserial_pair_rho_ml: invalid options"));
  }

  PolyserialPairMlResult out;
  double lo = options.rho_lower;
  double hi = options.rho_upper;
  constexpr double gr = 0.6180339887498948482;
  double c = hi - gr * (hi - lo);
  double d = lo + gr * (hi - lo);
  double fc = neglog_polyserial_unchecked(categories, u, thresholds, c);
  double fd = neglog_polyserial_unchecked(categories, u, thresholds, d);
  for (int iter = 0; iter < options.max_iter; ++iter) {
    if (fc < fd) {
      hi = d;
      d = c;
      fd = fc;
      c = hi - gr * (hi - lo);
      fc = neglog_polyserial_unchecked(categories, u, thresholds, c);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + gr * (hi - lo);
      fd = neglog_polyserial_unchecked(categories, u, thresholds, d);
    }
    out.iterations = iter + 1;
  }
  out.rho = 0.5 * (lo + hi);
  out.negloglik = neglog_polyserial_unchecked(categories, u, thresholds, out.rho);
  const double width = options.rho_upper - options.rho_lower;
  out.hit_lower = std::abs(out.rho - options.rho_lower) <= 1e-8 * width;
  out.hit_upper = std::abs(out.rho - options.rho_upper) <= 1e-8 * width;
  return out;
}

post_expected<PolyserialPairScores>
polyserial_pair_scores(const Eigen::Ref<const Eigen::VectorXi>& categories,
                       const Eigen::Ref<const Eigen::VectorXd>& u,
                       double rho,
                       const Eigen::Ref<const Eigen::VectorXd>& thresholds) {
  auto ok = validate_polyserial_inputs(categories, u, thresholds,
                                       "polyserial_pair_scores");
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (!std::isfinite(rho) || rho <= -1.0 || rho >= 1.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "polyserial_pair_scores: rho must be finite and inside (-1, 1)"));
  }

  const Eigen::Index n = categories.size();
  const Eigen::Index nth = thresholds.size();
  PolyserialPairScores out{
      .rho = Eigen::VectorXd::Zero(n),
      .thresholds = Eigen::MatrixXd::Zero(n, nth),
      .score_contributions = Eigen::MatrixXd::Zero(n, nth + 1),
      .score_gamma = Eigen::MatrixXd::Zero(nth + 1, nth + 1)};
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));
  const double h = 1e-5;
  const double rp = std::min(0.999, rho + h);
  const double rm = std::max(-0.999, rho - h);
  for (Eigen::Index r = 0; r < n; ++r) {
    const int c = categories(r);
    const double lik = polyserial_prob_unchecked(c, u(r), rho, thresholds);
    const double pp = polyserial_prob_unchecked(c, u(r), rp, thresholds);
    const double pm = polyserial_prob_unchecked(c, u(r), rm, thresholds);
    out.rho(r) = (std::log(pp) - std::log(pm)) / (rp - rm);
    for (Eigen::Index a = 0; a < nth; ++a) {
      const double z = normal_pdf((thresholds(a) - rho * u(r)) / sd) / sd;
      if (c == a) out.thresholds(r, a) += z / lik;
      if (c == a + 1) out.thresholds(r, a) -= z / lik;
    }
  }
  out.score_contributions.leftCols(nth) = out.thresholds;
  out.score_contributions.col(nth) = out.rho;
  out.score_gamma = score_gamma(out.score_contributions);
  return out;
}

}  // namespace magmaan::data
