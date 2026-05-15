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

}  // namespace

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
  PolyserialPairScores out{Eigen::VectorXd::Zero(n),
                           Eigen::MatrixXd::Zero(n, nth)};
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
  return out;
}

}  // namespace magmaan::data
