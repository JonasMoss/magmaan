#include "magmaan/data/pairwise_ordinal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "magmaan/error.hpp"

namespace magmaan::data {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kProbFloor = 1.4901161193847656e-8;
constexpr double kPi = 3.14159265358979323846;

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

double bvn_cdf(double h, double k, double rho) noexcept {
  if (h == -kInf || k == -kInf) return 0.0;
  if (h == kInf) return normal_cdf(k);
  if (k == kInf) return normal_cdf(h);
  const double upper = std::min(8.0, h);
  const double lower = -8.0;
  if (upper <= lower) return 0.0;

  static constexpr double x[32] = {
      -0.9972638618494816, -0.9856115115452684, -0.9647622555875064,
      -0.9349060759377397, -0.8963211557660521, -0.8493676137325700,
      -0.7944837959679424, -0.7321821187402897, -0.6630442669302152,
      -0.5877157572407623, -0.5068999089322294, -0.4213512761306353,
      -0.3318686022821277, -0.2392873622521371, -0.1444719615827965,
      -0.0483076656877383,  0.0483076656877383,  0.1444719615827965,
       0.2392873622521371,  0.3318686022821277,  0.4213512761306353,
       0.5068999089322294,  0.5877157572407623,  0.6630442669302152,
       0.7321821187402897,  0.7944837959679424,  0.8493676137325700,
       0.8963211557660521,  0.9349060759377397,  0.9647622555875064,
       0.9856115115452684,  0.9972638618494816};
  static constexpr double w[32] = {
      0.0070186100094701, 0.0162743947309057, 0.0253920653092621,
      0.0342738629130214, 0.0428358980222267, 0.0509980592623762,
      0.0586840934785355, 0.0658222227763618, 0.0723457941088485,
      0.0781938957870703, 0.0833119242269467, 0.0876520930044038,
      0.0911738786957639, 0.0938443990808046, 0.0956387200792749,
      0.0965400885147278, 0.0965400885147278, 0.0956387200792749,
      0.0938443990808046, 0.0911738786957639, 0.0876520930044038,
      0.0833119242269467, 0.0781938957870703, 0.0723457941088485,
      0.0658222227763618, 0.0586840934785355, 0.0509980592623762,
      0.0428358980222267, 0.0342738629130214, 0.0253920653092621,
      0.0162743947309057, 0.0070186100094701};

  const double mid = 0.5 * (upper + lower);
  const double half = 0.5 * (upper - lower);
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));
  double sum = 0.0;
  for (int i = 0; i < 32; ++i) {
    const double z = mid + half * x[i];
    sum += w[i] * normal_pdf(z) * normal_cdf((k - rho * z) / sd);
  }
  return std::clamp(half * sum, 0.0, 1.0);
}

double bvn_pdf(double x, double y, double rho) noexcept {
  if (!std::isfinite(x) || !std::isfinite(y)) return 0.0;
  const double one_minus = std::max(1e-12, 1.0 - rho * rho);
  const double z = x * x - 2.0 * rho * x * y + y * y;
  constexpr double inv_2pi = 0.15915494309189533577;
  return inv_2pi / std::sqrt(one_minus) * std::exp(-0.5 * z / one_minus);
}

double floored_rect_prob(double lo_i, double hi_i, double lo_j, double hi_j,
                         double rho) noexcept {
  return std::max(kProbFloor,
                  ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
}

post_expected<void>
validate_pair_shape(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
                    std::string_view caller) {
  if (counts.rows() < 2 || counts.cols() < 2) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": counts must be at least 2x2"));
  }
  if (thresholds_i.size() != counts.rows() - 1 ||
      thresholds_j.size() != counts.cols() - 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": threshold sizes do not match counts"));
  }
  if (!counts.allFinite() || !thresholds_i.allFinite() ||
      !thresholds_j.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": non-finite input"));
  }
  if ((counts.array() < 0.0).any() || counts.sum() <= 0.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": counts must be nonnegative with positive total"));
  }
  for (Eigen::Index k = 1; k < thresholds_i.size(); ++k) {
    if (!(thresholds_i(k - 1) < thresholds_i(k))) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(caller) + ": thresholds_i must be strictly increasing"));
    }
  }
  for (Eigen::Index k = 1; k < thresholds_j.size(); ++k) {
    if (!(thresholds_j(k - 1) < thresholds_j(k))) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(caller) + ": thresholds_j must be strictly increasing"));
    }
  }
  return {};
}

Eigen::MatrixXd adjusted_polychoric_table(const Eigen::MatrixXd& counts) {
  Eigen::MatrixXd out = counts;
  if (out.rows() == 2 && out.cols() == 2) {
    std::vector<Eigen::Index> zeros;
    for (Eigen::Index r = 0; r < 2; ++r) {
      for (Eigen::Index c = 0; c < 2; ++c) {
        if (out(r, c) == 0.0) zeros.push_back(r * 2 + c);
      }
    }
    if (zeros.size() == 1) {
      const Eigen::Index z = zeros[0];
      if (z == 0 || z == 3) {
        out(0, 0) += 0.5;
        out(1, 1) += 0.5;
        out(1, 0) -= 0.5;
        out(0, 1) -= 0.5;
      } else {
        out(0, 0) -= 0.5;
        out(1, 1) -= 0.5;
        out(1, 0) += 0.5;
        out(0, 1) += 0.5;
      }
    }
  }
  return out.cwiseMax(0.0);
}

double neglog_pair_unchecked(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                             const Eigen::Ref<const Eigen::VectorXd>& th_i,
                             const Eigen::Ref<const Eigen::VectorXd>& th_j,
                             double rho) noexcept {
  double out = 0.0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -kInf : th_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? kInf : th_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double n = counts(a, b);
      if (n <= 0.0) continue;
      const double lo_j = (b == 0) ? -kInf : th_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? kInf : th_j(b);
      out -= n * std::log(floored_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
    }
  }
  return out;
}

}  // namespace

post_expected<Eigen::MatrixXd>
ordinal_pair_table(const Eigen::Ref<const Eigen::VectorXi>& x_i,
                   const Eigen::Ref<const Eigen::VectorXi>& x_j,
                   std::int32_t n_levels_i,
                   std::int32_t n_levels_j) {
  if (x_i.size() != x_j.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_table: category vectors must have equal size"));
  }
  if (x_i.size() == 0 || n_levels_i < 2 || n_levels_j < 2) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_table: empty data or fewer than two levels"));
  }
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_levels_i, n_levels_j);
  for (Eigen::Index r = 0; r < x_i.size(); ++r) {
    const int ci = x_i(r);
    const int cj = x_j(r);
    if (ci < 0 || ci >= n_levels_i || cj < 0 || cj >= n_levels_j) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal_pair_table: category outside declared level range"));
    }
    out(ci, cj) += 1.0;
  }
  return out;
}

double ordinal_bvn_rect_prob(double lo_i, double hi_i,
                             double lo_j, double hi_j,
                             double rho) noexcept {
  const double p = bvn_cdf(hi_i, hi_j, rho) - bvn_cdf(lo_i, hi_j, rho) -
                   bvn_cdf(hi_i, lo_j, rho) + bvn_cdf(lo_i, lo_j, rho);
  return std::clamp(p, 0.0, 1.0);
}

double ordinal_bvn_rect_drho(double lo_i, double hi_i,
                             double lo_j, double hi_j,
                             double rho) noexcept {
  return bvn_pdf(hi_i, hi_j, rho) - bvn_pdf(lo_i, hi_j, rho) -
         bvn_pdf(hi_i, lo_j, rho) + bvn_pdf(lo_i, lo_j, rho);
}

post_expected<double>
ordinal_pair_negloglik(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                       const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                       const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
                       double rho) {
  auto ok = validate_pair_shape(counts, thresholds_i, thresholds_j,
                                "ordinal_pair_negloglik");
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (!std::isfinite(rho) || rho <= -1.0 || rho >= 1.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_negloglik: rho must be finite and inside (-1, 1)"));
  }
  return neglog_pair_unchecked(counts, thresholds_i, thresholds_j, rho);
}

post_expected<OrdinalPairMlResult>
fit_ordinal_pair_rho_ml(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                        const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                        const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
                        OrdinalPairMlOptions options) {
  auto ok = validate_pair_shape(counts, thresholds_i, thresholds_j,
                                "fit_ordinal_pair_rho_ml");
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper) || options.max_iter < 1) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_ordinal_pair_rho_ml: invalid options"));
  }

  Eigen::MatrixXd adjusted = options.lavaan_adjust_2x2
      ? adjusted_polychoric_table(counts)
      : Eigen::MatrixXd(counts);

  OrdinalPairMlResult out;
  out.adjusted_counts = adjusted;

  if (adjusted.rows() == 2 && adjusted.cols() == 2) {
    const double ad = adjusted(0, 0) * adjusted(1, 1);
    const double bc = adjusted(0, 1) * adjusted(1, 0);
    if (std::abs(ad - bc) <= 1e-12 * std::max(1.0, std::abs(ad) + std::abs(bc))) {
      out.rho = 0.0;
      out.negloglik = neglog_pair_unchecked(adjusted, thresholds_i,
                                            thresholds_j, out.rho);
      return out;
    }
    if (thresholds_i.size() == 1 && thresholds_j.size() == 1 &&
        std::abs(thresholds_i(0)) <= 1e-12 &&
        std::abs(thresholds_j(0)) <= 1e-12) {
      out.rho = std::clamp(-std::cos(2.0 * kPi * adjusted(0, 0) /
                                     adjusted.sum()),
                           options.rho_lower, options.rho_upper);
      out.negloglik = neglog_pair_unchecked(adjusted, thresholds_i,
                                            thresholds_j, out.rho);
      out.hit_lower = out.rho == options.rho_lower;
      out.hit_upper = out.rho == options.rho_upper;
      return out;
    }
  }

  double lo = options.rho_lower;
  double hi = options.rho_upper;
  constexpr double gr = 0.6180339887498948482;
  double c = hi - gr * (hi - lo);
  double d = lo + gr * (hi - lo);
  double fc = neglog_pair_unchecked(adjusted, thresholds_i, thresholds_j, c);
  double fd = neglog_pair_unchecked(adjusted, thresholds_i, thresholds_j, d);
  for (int iter = 0; iter < options.max_iter; ++iter) {
    if (fc < fd) {
      hi = d;
      d = c;
      fd = fc;
      c = hi - gr * (hi - lo);
      fc = neglog_pair_unchecked(adjusted, thresholds_i, thresholds_j, c);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + gr * (hi - lo);
      fd = neglog_pair_unchecked(adjusted, thresholds_i, thresholds_j, d);
    }
    out.iterations = iter + 1;
  }
  out.rho = 0.5 * (lo + hi);
  out.negloglik = neglog_pair_unchecked(adjusted, thresholds_i, thresholds_j,
                                        out.rho);
  const double width = options.rho_upper - options.rho_lower;
  out.hit_lower = std::abs(out.rho - options.rho_lower) <= 1e-8 * width;
  out.hit_upper = std::abs(out.rho - options.rho_upper) <= 1e-8 * width;
  return out;
}

post_expected<OrdinalPairScores>
ordinal_pair_scores(const Eigen::Ref<const Eigen::VectorXi>& x_i,
                    const Eigen::Ref<const Eigen::VectorXi>& x_j,
                    double rho,
                    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j) {
  if (x_i.size() != x_j.size() || x_i.size() == 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_scores: category vectors must be nonempty and equal size"));
  }
  if (!std::isfinite(rho) || rho <= -1.0 || rho >= 1.0 ||
      !thresholds_i.allFinite() || !thresholds_j.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_scores: non-finite input"));
  }
  const int n_levels_i = static_cast<int>(thresholds_i.size() + 1);
  const int n_levels_j = static_cast<int>(thresholds_j.size() + 1);
  OrdinalPairScores out{
      Eigen::VectorXd::Zero(x_i.size()),
      Eigen::MatrixXd::Zero(x_i.size(), thresholds_i.size()),
      Eigen::MatrixXd::Zero(x_i.size(), thresholds_j.size())};
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));

  for (Eigen::Index r = 0; r < x_i.size(); ++r) {
    const int ci = x_i(r);
    const int cj = x_j(r);
    if (ci < 0 || ci >= n_levels_i || cj < 0 || cj >= n_levels_j) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal_pair_scores: category outside threshold range"));
    }
    const double lo_i = (ci == 0) ? -kInf : thresholds_i(ci - 1);
    const double hi_i = (ci + 1 == n_levels_i) ? kInf : thresholds_i(ci);
    const double lo_j = (cj == 0) ? -kInf : thresholds_j(cj - 1);
    const double hi_j = (cj + 1 == n_levels_j) ? kInf : thresholds_j(cj);
    const double lik = floored_rect_prob(lo_i, hi_i, lo_j, hi_j, rho);
    out.rho(r) = ordinal_bvn_rect_drho(lo_i, hi_i, lo_j, hi_j, rho) / lik;

    for (Eigen::Index a = 0; a < thresholds_i.size(); ++a) {
      const double t = thresholds_i(a);
      const double z = normal_pdf(t) *
          (normal_cdf((hi_j - rho * t) / sd) -
           normal_cdf((lo_j - rho * t) / sd));
      if (ci == a) out.threshold_i(r, a) += z / lik;
      if (ci == a + 1) out.threshold_i(r, a) -= z / lik;
    }
    for (Eigen::Index a = 0; a < thresholds_j.size(); ++a) {
      const double t = thresholds_j(a);
      const double z = normal_pdf(t) *
          (normal_cdf((hi_i - rho * t) / sd) -
           normal_cdf((lo_i - rho * t) / sd));
      if (cj == a) out.threshold_j(r, a) += z / lik;
      if (cj == a + 1) out.threshold_j(r, a) -= z / lik;
    }
  }
  return out;
}

}  // namespace magmaan::data
