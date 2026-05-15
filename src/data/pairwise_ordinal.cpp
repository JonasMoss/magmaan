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

double normal_quantile(double p) noexcept {
  constexpr double a1 = -3.969683028665376e+01;
  constexpr double a2 =  2.209460984245205e+02;
  constexpr double a3 = -2.759285104469687e+02;
  constexpr double a4 =  1.383577518672690e+02;
  constexpr double a5 = -3.066479806614716e+01;
  constexpr double a6 =  2.506628277459239e+00;
  constexpr double b1 = -5.447609879822406e+01;
  constexpr double b2 =  1.615858368580409e+02;
  constexpr double b3 = -1.556989798598866e+02;
  constexpr double b4 =  6.680131188771972e+01;
  constexpr double b5 = -1.328068155288572e+01;
  constexpr double c1 = -7.784894002430293e-03;
  constexpr double c2 = -3.223964580411365e-01;
  constexpr double c3 = -2.400758277161838e+00;
  constexpr double c4 = -2.549732539343734e+00;
  constexpr double c5 =  4.374664141464968e+00;
  constexpr double c6 =  2.938163982698783e+00;
  constexpr double d1 =  7.784695709041462e-03;
  constexpr double d2 =  3.224671290700398e-01;
  constexpr double d3 =  2.445134137142996e+00;
  constexpr double d4 =  3.754408661907416e+00;
  constexpr double plow = 0.02425;
  constexpr double phigh = 1.0 - plow;

  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
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

post_expected<Eigen::VectorXd>
marginal_threshold_start(const Eigen::VectorXd& margins,
                         double min_spacing,
                         std::string_view caller) {
  if (margins.size() < 2 || !margins.allFinite() ||
      (margins.array() <= 0.0).any()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": all marginal categories must be nonempty"));
  }
  const double total = margins.sum();
  if (!(std::isfinite(total) && total > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": nonpositive marginal total"));
  }
  Eigen::VectorXd th(margins.size() - 1);
  double cum = 0.0;
  for (Eigen::Index k = 0; k < th.size(); ++k) {
    cum += margins(k);
    const double p = cum / total;
    if (!(p > 0.0 && p < 1.0)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(caller) + ": invalid marginal cumulative probability"));
    }
    th(k) = normal_quantile(std::clamp(p, 1e-12, 1.0 - 1e-12));
    if (k > 0 && th(k) <= th(k - 1) + min_spacing) {
      th(k) = th(k - 1) + min_spacing;
    }
  }
  return th;
}

void encode_thresholds(const Eigen::VectorXd& th,
                       double min_spacing,
                       Eigen::VectorXd& x,
                       Eigen::Index offset) {
  if (th.size() == 0) return;
  x(offset) = th(0);
  for (Eigen::Index k = 1; k < th.size(); ++k) {
    const double gap = std::max(th(k) - th(k - 1) - min_spacing, 1e-6);
    x(offset + k) = std::log(gap);
  }
}

Eigen::VectorXd decode_thresholds(const Eigen::VectorXd& x,
                                  Eigen::Index offset,
                                  Eigen::Index n_thresholds,
                                  double min_spacing) {
  Eigen::VectorXd th(n_thresholds);
  if (n_thresholds == 0) return th;
  th(0) = x(offset);
  for (Eigen::Index k = 1; k < n_thresholds; ++k) {
    const double log_gap = std::clamp(x(offset + k), -30.0, 30.0);
    th(k) = th(k - 1) + min_spacing + std::exp(log_gap);
  }
  return th;
}

double encode_rho(double rho, double lower, double upper) noexcept {
  const double mid = 0.5 * (lower + upper);
  const double half = 0.5 * (upper - lower);
  const double z = std::clamp((rho - mid) / half, -0.999999, 0.999999);
  return 0.5 * std::log((1.0 + z) / (1.0 - z));
}

double decode_rho(double z, double lower, double upper) noexcept {
  const double mid = 0.5 * (lower + upper);
  const double half = 0.5 * (upper - lower);
  return mid + half * std::tanh(z);
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

post_expected<OrdinalPairObservedTable>
ordinal_pair_observed_table(const Eigen::Ref<const Eigen::VectorXd>& x_i,
                            const Eigen::Ref<const Eigen::VectorXd>& x_j,
                            std::int32_t n_levels_i,
                            std::int32_t n_levels_j) {
  if (x_i.size() != x_j.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_observed_table: category vectors must have equal size"));
  }
  if (x_i.size() == 0 || n_levels_i < 2 || n_levels_j < 2) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_observed_table: empty data or fewer than two levels"));
  }
  OrdinalPairObservedTable out{
      Eigen::MatrixXd::Zero(n_levels_i, n_levels_j), 0, 0};
  for (Eigen::Index r = 0; r < x_i.size(); ++r) {
    const double vi = x_i(r);
    const double vj = x_j(r);
    if (std::isnan(vi) || std::isnan(vj)) {
      ++out.n_missing;
      continue;
    }
    if (!std::isfinite(vi) || !std::isfinite(vj) ||
        std::floor(vi) != vi || std::floor(vj) != vj) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal_pair_observed_table: observed categories must be finite integers"));
    }
    const int ci = static_cast<int>(vi);
    const int cj = static_cast<int>(vj);
    if (ci < 1 || ci > n_levels_i || cj < 1 || cj > n_levels_j) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal_pair_observed_table: category outside declared level range"));
    }
    out.counts(ci - 1, cj - 1) += 1.0;
    ++out.n_obs;
  }
  if (out.n_obs <= 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_observed_table: no observed pairs"));
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

post_expected<OrdinalPairJointMlResult>
fit_ordinal_pair_joint_ml(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                          OrdinalPairJointMlOptions options) {
  if (counts.rows() < 2 || counts.cols() < 2 || !counts.allFinite() ||
      (counts.array() < 0.0).any() || counts.sum() <= 0.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_ordinal_pair_joint_ml: counts must be a finite nonnegative table with positive total"));
  }
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper) || options.max_iter < 1 ||
      !(std::isfinite(options.ftol) && options.ftol > 0.0) ||
      !(std::isfinite(options.gtol) && options.gtol > 0.0) ||
      !(std::isfinite(options.fd_step) && options.fd_step > 0.0) ||
      !(std::isfinite(options.min_threshold_spacing) &&
        options.min_threshold_spacing > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_ordinal_pair_joint_ml: invalid options"));
  }

  Eigen::MatrixXd adjusted = options.lavaan_adjust_2x2
      ? adjusted_polychoric_table(counts)
      : Eigen::MatrixXd(counts);
  const auto th_i_start = marginal_threshold_start(
      adjusted.rowwise().sum(), options.min_threshold_spacing,
      "fit_ordinal_pair_joint_ml");
  if (!th_i_start.has_value()) return std::unexpected(th_i_start.error());
  const auto th_j_start = marginal_threshold_start(
      adjusted.colwise().sum().transpose(), options.min_threshold_spacing,
      "fit_ordinal_pair_joint_ml");
  if (!th_j_start.has_value()) return std::unexpected(th_j_start.error());

  auto rho_start = fit_ordinal_pair_rho_ml(
      adjusted, *th_i_start, *th_j_start,
      OrdinalPairMlOptions{.rho_lower = options.rho_lower,
                           .rho_upper = options.rho_upper,
                           .max_iter = 72,
                           .lavaan_adjust_2x2 = false});
  if (!rho_start.has_value()) return std::unexpected(rho_start.error());

  const Eigen::Index n_th_i = counts.rows() - 1;
  const Eigen::Index n_th_j = counts.cols() - 1;
  const Eigen::Index npar = n_th_i + n_th_j + 1;
  Eigen::VectorXd x0(npar);
  encode_thresholds(*th_i_start, options.min_threshold_spacing, x0, 0);
  encode_thresholds(*th_j_start, options.min_threshold_spacing, x0, n_th_i);
  x0(npar - 1) = encode_rho(rho_start->rho,
                            options.rho_lower,
                            options.rho_upper);

  auto objective_value = [&](const Eigen::VectorXd& x) {
    const Eigen::VectorXd th_i = decode_thresholds(
        x, 0, n_th_i, options.min_threshold_spacing);
    const Eigen::VectorXd th_j = decode_thresholds(
        x, n_th_i, n_th_j, options.min_threshold_spacing);
    const double rho = decode_rho(x(npar - 1),
                                  options.rho_lower,
                                  options.rho_upper);
    return neglog_pair_unchecked(adjusted, th_i, th_j, rho);
  };

  auto gradient = [&](const Eigen::VectorXd& x) {
    Eigen::VectorXd grad(x.size());
    for (Eigen::Index k = 0; k < x.size(); ++k) {
      const double h = options.fd_step * std::max(1.0, std::abs(x(k)));
      Eigen::VectorXd xp = x;
      Eigen::VectorXd xm = x;
      xp(k) += h;
      xm(k) -= h;
      const double fp = objective_value(xp);
      const double fm = objective_value(xm);
      grad(k) = (fp - fm) / (2.0 * h);
    }
    return grad;
  };

  Eigen::VectorXd x = x0;
  double f = objective_value(x);
  if (!std::isfinite(f)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_ordinal_pair_joint_ml: non-finite starting objective"));
  }

  int iterations = 0;
  for (; iterations < options.max_iter; ++iterations) {
    const Eigen::VectorXd grad = gradient(x);
    if (!grad.allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "fit_ordinal_pair_joint_ml: non-finite gradient"));
    }
    const double grad_inf = grad.lpNorm<Eigen::Infinity>();
    if (grad_inf <= options.gtol) break;

    const double grad_sq = grad.squaredNorm();
    double step = 1.0;
    bool accepted = false;
    bool ftol_stop = false;
    for (int ls = 0; ls < 40; ++ls) {
      const Eigen::VectorXd candidate = x - step * grad;
      const double f_candidate = objective_value(candidate);
      if (std::isfinite(f_candidate) &&
          f_candidate < f - 1e-4 * step * grad_sq) {
        ftol_stop = std::abs(f - f_candidate) <=
            options.ftol * std::max(1.0, std::abs(f));
        x = candidate;
        f = f_candidate;
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) break;
    if (ftol_stop) {
      ++iterations;
      break;
    }
  }

  OrdinalPairJointMlResult out;
  out.thresholds_i = decode_thresholds(x, 0, n_th_i,
                                       options.min_threshold_spacing);
  out.thresholds_j = decode_thresholds(x, n_th_i, n_th_j,
                                       options.min_threshold_spacing);
  out.rho = decode_rho(x(npar - 1),
                       options.rho_lower,
                       options.rho_upper);
  out.negloglik = neglog_pair_unchecked(adjusted, out.thresholds_i,
                                        out.thresholds_j, out.rho);
  out.iterations = iterations;
  const double width = options.rho_upper - options.rho_lower;
  out.hit_lower = std::abs(out.rho - options.rho_lower) <= 1e-8 * width;
  out.hit_upper = std::abs(out.rho - options.rho_upper) <= 1e-8 * width;
  out.adjusted_counts = std::move(adjusted);
  return out;
}

post_expected<OrdinalPairObservedMlResult>
fit_ordinal_pair_observed_rho_ml(
    const Eigen::Ref<const Eigen::VectorXd>& x_i,
    const Eigen::Ref<const Eigen::VectorXd>& x_j,
    std::int32_t n_levels_i,
    std::int32_t n_levels_j,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
    OrdinalPairMlOptions options) {
  auto tab_or = ordinal_pair_observed_table(x_i, x_j, n_levels_i, n_levels_j);
  if (!tab_or.has_value()) return std::unexpected(tab_or.error());
  auto fit_or = fit_ordinal_pair_rho_ml(tab_or->counts, thresholds_i,
                                        thresholds_j, options);
  if (!fit_or.has_value()) return std::unexpected(fit_or.error());
  return OrdinalPairObservedMlResult{
      .fit = std::move(*fit_or),
      .counts = std::move(tab_or->counts),
      .n_obs = tab_or->n_obs,
      .n_missing = tab_or->n_missing};
}

post_expected<OrdinalPairObservedJointMlResult>
fit_ordinal_pair_observed_joint_ml(
    const Eigen::Ref<const Eigen::VectorXd>& x_i,
    const Eigen::Ref<const Eigen::VectorXd>& x_j,
    std::int32_t n_levels_i,
    std::int32_t n_levels_j,
    OrdinalPairJointMlOptions options) {
  auto tab_or = ordinal_pair_observed_table(x_i, x_j, n_levels_i, n_levels_j);
  if (!tab_or.has_value()) return std::unexpected(tab_or.error());
  auto fit_or = fit_ordinal_pair_joint_ml(tab_or->counts, options);
  if (!fit_or.has_value()) return std::unexpected(fit_or.error());
  return OrdinalPairObservedJointMlResult{
      .fit = std::move(*fit_or),
      .counts = std::move(tab_or->counts),
      .n_obs = tab_or->n_obs,
      .n_missing = tab_or->n_missing};
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
