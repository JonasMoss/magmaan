#include "magmaan/data/pairwise_ordinal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/LU>

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

double normal_pdf_derivative(double x) noexcept {
  if (!std::isfinite(x)) return 0.0;
  return -x * normal_pdf(x);
}

double bvn_pdf_drho(double x, double y, double rho) noexcept {
  if (!std::isfinite(x) || !std::isfinite(y)) return 0.0;
  const double one_minus = std::max(1e-12, 1.0 - rho * rho);
  const double z = x * x - 2.0 * rho * x * y + y * y;
  const double numerator =
      one_minus * (rho + x * y) - rho * z;
  return bvn_pdf(x, y, rho) * numerator / (one_minus * one_minus);
}

double bvn_cdf_dxx(double variable, double fixed, double rho) noexcept {
  if (!std::isfinite(variable)) return 0.0;
  if (fixed == -kInf) return 0.0;
  if (fixed == kInf) return normal_pdf_derivative(variable);
  const double one_minus = std::max(1e-12, 1.0 - rho * rho);
  const double sd = std::sqrt(one_minus);
  const double arg = (fixed - rho * variable) / sd;
  return normal_pdf_derivative(variable) * normal_cdf(arg) -
         rho * normal_pdf(variable) * normal_pdf(arg) / sd;
}

double bvn_cdf_dxdrho(double variable, double fixed, double rho) noexcept {
  if (!std::isfinite(variable) || !std::isfinite(fixed)) return 0.0;
  const double one_minus = std::max(1e-12, 1.0 - rho * rho);
  const double sd = std::sqrt(one_minus);
  const double arg = (fixed - rho * variable) / sd;
  return normal_pdf(variable) * normal_pdf(arg) *
         (rho * fixed - variable) / (one_minus * sd);
}

double bvn_cdf_dxdy(double x, double y, double rho) noexcept {
  return bvn_pdf(x, y, rho);
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

post_expected<OrdinalPairHWeightedResult>
h_weighted_result(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                  const Eigen::Ref<const Eigen::VectorXd>& th_i,
                  const Eigen::Ref<const Eigen::VectorXd>& th_j,
                  double rho,
                  const PolychoricHScoreOptions& h_options) {
  const double total = counts.sum();
  OrdinalPairHWeightedResult out;
  out.rho = rho;
  out.adjusted_counts = counts;
  out.probabilities = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.expected_counts = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.residual_counts = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.pearson_residuals = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.weights = Eigen::MatrixXd::Ones(counts.rows(), counts.cols());

  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -kInf : th_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? kInf : th_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -kInf : th_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? kInf : th_j(b);
      const double raw_p = ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho);
      const double p = std::max(kProbFloor, raw_p);
      const double expected = total * p;
      const double t = counts(a, b) / expected;
      auto h = eval_polychoric_h_score(t, h_options);
      if (!h.has_value()) return std::unexpected(h.error());

      out.probabilities(a, b) = p;
      out.expected_counts(a, b) = expected;
      out.residual_counts(a, b) = counts(a, b) - expected;
      out.pearson_residuals(a, b) =
          out.residual_counts(a, b) / std::sqrt(expected);
      out.weights(a, b) = t > 0.0 ? h->h / t : 1.0;
      out.objective += p * h->phi;
      if (raw_p > kProbFloor) {
        out.score += h->h * ordinal_bvn_rect_drho(lo_i, hi_i, lo_j, hi_j, rho);
      }
    }
  }
  return out;
}

double h_weighted_objective_unchecked(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    const Eigen::Ref<const Eigen::VectorXd>& th_i,
    const Eigen::Ref<const Eigen::VectorXd>& th_j,
    double rho,
    const PolychoricHScoreOptions& h_options) {
  auto result = h_weighted_result(counts, th_i, th_j, rho, h_options);
  if (!result.has_value()) return std::numeric_limits<double>::infinity();
  return result->objective;
}

bool h_score_is_ml_like(const PolychoricHScoreOptions& h_options) noexcept {
  return h_options.kind == PolychoricHScoreKind::ML ||
         (h_options.kind == PolychoricHScoreKind::WmaHardCap &&
          h_options.k == std::numeric_limits<double>::infinity());
}

post_expected<OrdinalPairJointDpdResult>
dpd_result(const Eigen::Ref<const Eigen::MatrixXd>& counts,
           const Eigen::Ref<const Eigen::VectorXd>& th_i,
           const Eigen::Ref<const Eigen::VectorXd>& th_j,
           double rho,
           double alpha) {
  const double total = counts.sum();
  OrdinalPairJointDpdResult out;
  out.thresholds_i = th_i;
  out.thresholds_j = th_j;
  out.rho = rho;
  out.adjusted_counts = counts;
  out.probabilities = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.expected_counts = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.residual_counts = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.pearson_residuals = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.weights = Eigen::MatrixXd::Ones(counts.rows(), counts.cols());

  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -kInf : th_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? kInf : th_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -kInf : th_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? kInf : th_j(b);
      const double p = std::max(
          kProbFloor, ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
      const double fhat = counts(a, b) / total;
      const double p_alpha = std::pow(p, alpha);
      const double p_1_alpha = p * p_alpha;
      const double expected = total * p;
      out.probabilities(a, b) = p;
      out.expected_counts(a, b) = expected;
      out.residual_counts(a, b) = counts(a, b) - expected;
      out.pearson_residuals(a, b) =
          out.residual_counts(a, b) / std::sqrt(expected);
      out.weights(a, b) = p_alpha;
      out.objective += p_1_alpha -
                       ((1.0 + alpha) / alpha) * fhat * p_alpha;
    }
  }
  if (!std::isfinite(out.objective)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_ordinal_pair_joint_dpd: non-finite objective"));
  }
  return out;
}

double dpd_objective_unchecked(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    const Eigen::Ref<const Eigen::VectorXd>& th_i,
    const Eigen::Ref<const Eigen::VectorXd>& th_j,
    double rho,
    double alpha) {
  auto result = dpd_result(counts, th_i, th_j, rho, alpha);
  if (!result.has_value()) return std::numeric_limits<double>::infinity();
  return result->objective;
}

post_expected<std::int64_t>
integer_total_count(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                    std::string_view caller) {
  std::int64_t n = 0;
  for (Eigen::Index r = 0; r < counts.rows(); ++r) {
    for (Eigen::Index c = 0; c < counts.cols(); ++c) {
      const double v = counts(r, c);
      const double rounded = std::round(v);
      if (std::abs(v - rounded) > 1e-10) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            std::string(caller) + ": counts must be integer-valued"));
      }
      n += static_cast<std::int64_t>(rounded);
    }
  }
  if (n <= 0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": counts have no observations"));
  }
  return n;
}

Eigen::VectorXd cell_log_score(Eigen::Index ci,
                               Eigen::Index cj,
                               const Eigen::Ref<const Eigen::VectorXd>& th_i,
                               const Eigen::Ref<const Eigen::VectorXd>& th_j,
                               double rho) {
  const Eigen::Index nth_i = th_i.size();
  const Eigen::Index nth_j = th_j.size();
  Eigen::VectorXd out = Eigen::VectorXd::Zero(nth_i + nth_j + 1);
  const Eigen::Index n_levels_i = nth_i + 1;
  const Eigen::Index n_levels_j = nth_j + 1;
  const double lo_i = (ci == 0) ? -kInf : th_i(ci - 1);
  const double hi_i = (ci + 1 == n_levels_i) ? kInf : th_i(ci);
  const double lo_j = (cj == 0) ? -kInf : th_j(cj - 1);
  const double hi_j = (cj + 1 == n_levels_j) ? kInf : th_j(cj);
  const double lik = floored_rect_prob(lo_i, hi_i, lo_j, hi_j, rho);
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));

  for (Eigen::Index a = 0; a < nth_i; ++a) {
    const double t = th_i(a);
    const double z = normal_pdf(t) *
        (normal_cdf((hi_j - rho * t) / sd) -
         normal_cdf((lo_j - rho * t) / sd));
    if (ci == a) out(a) += z / lik;
    if (ci == a + 1) out(a) -= z / lik;
  }
  for (Eigen::Index a = 0; a < nth_j; ++a) {
    const double t = th_j(a);
    const double z = normal_pdf(t) *
        (normal_cdf((hi_i - rho * t) / sd) -
         normal_cdf((lo_i - rho * t) / sd));
    if (cj == a) out(nth_i + a) += z / lik;
    if (cj == a + 1) out(nth_i + a) -= z / lik;
  }
  out(out.size() - 1) =
      ordinal_bvn_rect_drho(lo_i, hi_i, lo_j, hi_j, rho) / lik;
  return out;
}

int threshold_boundary_sign(Eigen::Index category,
                            Eigen::Index threshold_index) noexcept {
  if (category == threshold_index) return 1;
  if (category == threshold_index + 1) return -1;
  return 0;
}

Eigen::MatrixXd cell_probability_hessian(
    Eigen::Index ci,
    Eigen::Index cj,
    const Eigen::Ref<const Eigen::VectorXd>& th_i,
    const Eigen::Ref<const Eigen::VectorXd>& th_j,
    double rho) {
  const Eigen::Index nth_i = th_i.size();
  const Eigen::Index nth_j = th_j.size();
  const Eigen::Index npar = nth_i + nth_j + 1;
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(npar, npar);
  const Eigen::Index n_levels_i = nth_i + 1;
  const Eigen::Index n_levels_j = nth_j + 1;
  const double lo_i = (ci == 0) ? -kInf : th_i(ci - 1);
  const double hi_i = (ci + 1 == n_levels_i) ? kInf : th_i(ci);
  const double lo_j = (cj == 0) ? -kInf : th_j(cj - 1);
  const double hi_j = (cj + 1 == n_levels_j) ? kInf : th_j(cj);
  const Eigen::Index rho_idx = npar - 1;

  out(rho_idx, rho_idx) =
      bvn_pdf_drho(hi_i, hi_j, rho) - bvn_pdf_drho(lo_i, hi_j, rho) -
      bvn_pdf_drho(hi_i, lo_j, rho) + bvn_pdf_drho(lo_i, lo_j, rho);

  for (Eigen::Index k = 0; k < nth_i; ++k) {
    const int s = threshold_boundary_sign(ci, k);
    if (s == 0) continue;
    const double th = th_i(k);
    out(k, k) =
        static_cast<double>(s) *
        (bvn_cdf_dxx(th, hi_j, rho) - bvn_cdf_dxx(th, lo_j, rho));
    out(k, rho_idx) =
        static_cast<double>(s) *
        (bvn_cdf_dxdrho(th, hi_j, rho) -
         bvn_cdf_dxdrho(th, lo_j, rho));
    out(rho_idx, k) = out(k, rho_idx);
  }

  for (Eigen::Index k = 0; k < nth_j; ++k) {
    const int s = threshold_boundary_sign(cj, k);
    if (s == 0) continue;
    const Eigen::Index idx = nth_i + k;
    const double th = th_j(k);
    out(idx, idx) =
        static_cast<double>(s) *
        (bvn_cdf_dxx(th, hi_i, rho) - bvn_cdf_dxx(th, lo_i, rho));
    out(idx, rho_idx) =
        static_cast<double>(s) *
        (bvn_cdf_dxdrho(th, hi_i, rho) -
         bvn_cdf_dxdrho(th, lo_i, rho));
    out(rho_idx, idx) = out(idx, rho_idx);
  }

  for (Eigen::Index k = 0; k < nth_i; ++k) {
    const int si = threshold_boundary_sign(ci, k);
    if (si == 0) continue;
    for (Eigen::Index l = 0; l < nth_j; ++l) {
      const int sj = threshold_boundary_sign(cj, l);
      if (sj == 0) continue;
      const Eigen::Index idx_j = nth_i + l;
      out(k, idx_j) = static_cast<double>(si * sj) *
                      bvn_cdf_dxdy(th_i(k), th_j(l), rho);
      out(idx_j, k) = out(k, idx_j);
    }
  }
  return out;
}

post_expected<Eigen::MatrixXd>
h_weighted_bread(const Eigen::Ref<const Eigen::MatrixXd>& counts,
                 const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
                 const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
                 double rho,
                 const PolychoricHScoreOptions& h_options,
                 std::string_view caller) {
  const double total = counts.sum();
  const Eigen::Index npar = thresholds_i.size() + thresholds_j.size() + 1;
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(npar, npar);
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -kInf : thresholds_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? kInf : thresholds_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -kInf : thresholds_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? kInf : thresholds_j(b);
      const double p = std::max(
          kProbFloor, ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
      const double t = counts(a, b) / (total * p);
      auto h = eval_polychoric_h_score(t, h_options);
      if (!h.has_value()) return std::unexpected(h.error());
      if (!std::isfinite(h->h) || !std::isfinite(h->dh)) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            std::string(caller) + ": non-finite h-score derivative"));
      }
      const Eigen::VectorXd score =
          cell_log_score(a, b, thresholds_i, thresholds_j, rho);
      const Eigen::MatrixXd hessian =
          cell_probability_hessian(a, b, thresholds_i, thresholds_j, rho);
      out.noalias() += h->h * hessian;
      out.noalias() -= h->dh * t * p * (score * score.transpose());
    }
  }
  if (!out.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(caller) + ": non-finite analytic bread"));
  }
  out = 0.5 * (out + out.transpose());
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

double joint_h_weighted_gradient_inf_unchecked(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
    double rho,
    double rho_lower,
    double rho_upper,
    double min_threshold_spacing,
    double fd_step,
    const PolychoricHScoreOptions& h_options) {
  const Eigen::Index n_th_i = thresholds_i.size();
  const Eigen::Index n_th_j = thresholds_j.size();
  const Eigen::Index npar = n_th_i + n_th_j + 1;
  Eigen::VectorXd x(npar);
  encode_thresholds(thresholds_i, min_threshold_spacing, x, 0);
  encode_thresholds(thresholds_j, min_threshold_spacing, x, n_th_i);
  x(npar - 1) = encode_rho(rho, rho_lower, rho_upper);

  auto objective_value = [&](const Eigen::VectorXd& z) {
    const Eigen::VectorXd th_i = decode_thresholds(
        z, 0, n_th_i, min_threshold_spacing);
    const Eigen::VectorXd th_j = decode_thresholds(
        z, n_th_i, n_th_j, min_threshold_spacing);
    const double r = decode_rho(z(npar - 1), rho_lower, rho_upper);
    return h_weighted_objective_unchecked(counts, th_i, th_j, r, h_options);
  };

  Eigen::VectorXd grad(npar);
  for (Eigen::Index k = 0; k < npar; ++k) {
    const double h = fd_step * std::max(1.0, std::abs(x(k)));
    Eigen::VectorXd xp = x;
    Eigen::VectorXd xm = x;
    xp(k) += h;
    xm(k) -= h;
    const double fp = objective_value(xp);
    const double fm = objective_value(xm);
    grad(k) = (fp - fm) / (2.0 * h);
  }
  if (!grad.allFinite()) return std::numeric_limits<double>::infinity();
  return grad.lpNorm<Eigen::Infinity>();
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

post_expected<OrdinalPairHWeightedResult>
fit_ordinal_pair_rho_h_weighted(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
    OrdinalPairHWeightedOptions options) {
  auto ok = validate_pair_shape(counts, thresholds_i, thresholds_j,
                                "fit_ordinal_pair_rho_h_weighted");
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper) || options.max_iter < 1 ||
      !(std::isfinite(options.x_tol) && options.x_tol > 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_ordinal_pair_rho_h_weighted: invalid options"));
  }
  auto h_ok = eval_polychoric_h_score(1.0, options.h_score);
  if (!h_ok.has_value()) return std::unexpected(h_ok.error());

  if (h_score_is_ml_like(options.h_score)) {
    auto ml = fit_ordinal_pair_rho_ml(
        counts, thresholds_i, thresholds_j,
        OrdinalPairMlOptions{.rho_lower = options.rho_lower,
                             .rho_upper = options.rho_upper,
                             .max_iter = options.max_iter,
                             .lavaan_adjust_2x2 = options.lavaan_adjust_2x2});
    if (!ml.has_value()) return std::unexpected(ml.error());
    auto out = h_weighted_result(ml->adjusted_counts, thresholds_i,
                                 thresholds_j, ml->rho, options.h_score);
    if (!out.has_value()) return std::unexpected(out.error());
    out->iterations = ml->iterations;
    out->converged = true;
    out->hit_lower = ml->hit_lower;
    out->hit_upper = ml->hit_upper;
    return out;
  }

  Eigen::MatrixXd adjusted = options.lavaan_adjust_2x2
      ? adjusted_polychoric_table(counts)
      : Eigen::MatrixXd(counts);

  double lo = options.rho_lower;
  double hi = options.rho_upper;
  constexpr double gr = 0.6180339887498948482;
  double c = hi - gr * (hi - lo);
  double d = lo + gr * (hi - lo);
  double fc = h_weighted_objective_unchecked(
      adjusted, thresholds_i, thresholds_j, c, options.h_score);
  double fd = h_weighted_objective_unchecked(
      adjusted, thresholds_i, thresholds_j, d, options.h_score);
  const double width0 = hi - lo;
  const double tol = options.x_tol * std::max(1.0, width0);
  int iterations = 0;
  for (; iterations < options.max_iter && (hi - lo) > tol; ++iterations) {
    if (fc < fd) {
      hi = d;
      d = c;
      fd = fc;
      c = hi - gr * (hi - lo);
      fc = h_weighted_objective_unchecked(
          adjusted, thresholds_i, thresholds_j, c, options.h_score);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + gr * (hi - lo);
      fd = h_weighted_objective_unchecked(
          adjusted, thresholds_i, thresholds_j, d, options.h_score);
    }
  }
  auto out = h_weighted_result(adjusted, thresholds_i, thresholds_j,
                               0.5 * (lo + hi), options.h_score);
  if (!out.has_value()) return std::unexpected(out.error());
  out->iterations = iterations;
  out->converged = (hi - lo) <= tol;
  const double width = options.rho_upper - options.rho_lower;
  out->hit_lower = std::abs(out->rho - options.rho_lower) <= 1e-8 * width;
  out->hit_upper = std::abs(out->rho - options.rho_upper) <= 1e-8 * width;
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

post_expected<OrdinalPairJointHWeightedResult>
fit_ordinal_pair_joint_h_weighted(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    OrdinalPairJointHWeightedOptions options) {
  if (counts.rows() < 2 || counts.cols() < 2 || !counts.allFinite() ||
      (counts.array() < 0.0).any() || counts.sum() <= 0.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_ordinal_pair_joint_h_weighted: counts must be a finite nonnegative table with positive total"));
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
        "fit_ordinal_pair_joint_h_weighted: invalid options"));
  }
  auto h_ok = eval_polychoric_h_score(1.0, options.h_score);
  if (!h_ok.has_value()) return std::unexpected(h_ok.error());

  if (h_score_is_ml_like(options.h_score)) {
    auto ml = fit_ordinal_pair_joint_ml(
        counts, OrdinalPairJointMlOptions{
            .rho_lower = options.rho_lower,
            .rho_upper = options.rho_upper,
            .max_iter = options.max_iter,
            .ftol = options.ftol,
            .gtol = options.gtol,
            .fd_step = options.fd_step,
            .min_threshold_spacing = options.min_threshold_spacing,
            .lavaan_adjust_2x2 = options.lavaan_adjust_2x2});
    if (!ml.has_value()) return std::unexpected(ml.error());
    auto diag = h_weighted_result(ml->adjusted_counts, ml->thresholds_i,
                                  ml->thresholds_j, ml->rho, options.h_score);
    if (!diag.has_value()) return std::unexpected(diag.error());
    const double gradient_inf = joint_h_weighted_gradient_inf_unchecked(
        diag->adjusted_counts, ml->thresholds_i, ml->thresholds_j, ml->rho,
        options.rho_lower, options.rho_upper, options.min_threshold_spacing,
        options.fd_step, options.h_score);
    return OrdinalPairJointHWeightedResult{
        .thresholds_i = std::move(ml->thresholds_i),
        .thresholds_j = std::move(ml->thresholds_j),
        .rho = ml->rho,
        .objective = diag->objective,
        .gradient_inf = gradient_inf,
        .iterations = ml->iterations,
        .converged = true,
        .hit_lower = ml->hit_lower,
        .hit_upper = ml->hit_upper,
        .adjusted_counts = std::move(diag->adjusted_counts),
        .probabilities = std::move(diag->probabilities),
        .expected_counts = std::move(diag->expected_counts),
        .residual_counts = std::move(diag->residual_counts),
        .pearson_residuals = std::move(diag->pearson_residuals),
        .weights = std::move(diag->weights)};
  }

  Eigen::MatrixXd adjusted = options.lavaan_adjust_2x2
      ? adjusted_polychoric_table(counts)
      : Eigen::MatrixXd(counts);
  const auto th_i_start = marginal_threshold_start(
      adjusted.rowwise().sum(), options.min_threshold_spacing,
      "fit_ordinal_pair_joint_h_weighted");
  if (!th_i_start.has_value()) return std::unexpected(th_i_start.error());
  const auto th_j_start = marginal_threshold_start(
      adjusted.colwise().sum().transpose(), options.min_threshold_spacing,
      "fit_ordinal_pair_joint_h_weighted");
  if (!th_j_start.has_value()) return std::unexpected(th_j_start.error());

  auto rho_start = fit_ordinal_pair_rho_h_weighted(
      adjusted, *th_i_start, *th_j_start,
      OrdinalPairHWeightedOptions{
          .rho_lower = options.rho_lower,
          .rho_upper = options.rho_upper,
          .max_iter = 72,
          .x_tol = 1e-10,
          .lavaan_adjust_2x2 = false,
          .h_score = options.h_score});
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
    return h_weighted_objective_unchecked(adjusted, th_i, th_j, rho,
                                          options.h_score);
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
        "fit_ordinal_pair_joint_h_weighted: non-finite starting objective"));
  }

  int iterations = 0;
  double gradient_inf = std::numeric_limits<double>::infinity();
  bool converged = false;
  for (; iterations < options.max_iter; ++iterations) {
    const Eigen::VectorXd grad = gradient(x);
    if (!grad.allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "fit_ordinal_pair_joint_h_weighted: non-finite gradient"));
    }
    gradient_inf = grad.lpNorm<Eigen::Infinity>();
    if (gradient_inf <= options.gtol) {
      converged = true;
      break;
    }

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
      converged = true;
      break;
    }
  }

  const Eigen::VectorXd thresholds_i = decode_thresholds(
      x, 0, n_th_i, options.min_threshold_spacing);
  const Eigen::VectorXd thresholds_j = decode_thresholds(
      x, n_th_i, n_th_j, options.min_threshold_spacing);
  const double rho = decode_rho(x(npar - 1),
                                options.rho_lower,
                                options.rho_upper);
  const Eigen::VectorXd final_grad = gradient(x);
  if (final_grad.allFinite()) {
    gradient_inf = final_grad.lpNorm<Eigen::Infinity>();
    converged = converged || gradient_inf <= options.gtol;
  }
  auto diag = h_weighted_result(adjusted, thresholds_i, thresholds_j, rho,
                                options.h_score);
  if (!diag.has_value()) return std::unexpected(diag.error());

  const double width = options.rho_upper - options.rho_lower;
  return OrdinalPairJointHWeightedResult{
      .thresholds_i = thresholds_i,
      .thresholds_j = thresholds_j,
      .rho = rho,
      .objective = diag->objective,
      .gradient_inf = gradient_inf,
      .iterations = iterations,
      .converged = converged,
      .hit_lower = std::abs(rho - options.rho_lower) <= 1e-8 * width,
      .hit_upper = std::abs(rho - options.rho_upper) <= 1e-8 * width,
      .adjusted_counts = std::move(diag->adjusted_counts),
      .probabilities = std::move(diag->probabilities),
      .expected_counts = std::move(diag->expected_counts),
      .residual_counts = std::move(diag->residual_counts),
      .pearson_residuals = std::move(diag->pearson_residuals),
      .weights = std::move(diag->weights)};
}

post_expected<OrdinalPairJointDpdResult>
fit_ordinal_pair_joint_dpd(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    OrdinalPairJointDpdOptions options) {
  if (counts.rows() < 2 || counts.cols() < 2 || !counts.allFinite() ||
      (counts.array() < 0.0).any() || counts.sum() <= 0.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_ordinal_pair_joint_dpd: counts must be a finite nonnegative table with positive total"));
  }
  if (!std::isfinite(options.rho_lower) || !std::isfinite(options.rho_upper) ||
      !(options.rho_lower > -1.0) || !(options.rho_upper < 1.0) ||
      !(options.rho_lower < options.rho_upper) || options.max_iter < 1 ||
      !(std::isfinite(options.ftol) && options.ftol > 0.0) ||
      !(std::isfinite(options.gtol) && options.gtol > 0.0) ||
      !(std::isfinite(options.fd_step) && options.fd_step > 0.0) ||
      !(std::isfinite(options.min_threshold_spacing) &&
        options.min_threshold_spacing > 0.0) ||
      !(std::isfinite(options.alpha) && options.alpha >= 0.0)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fit_ordinal_pair_joint_dpd: invalid options"));
  }

  if (options.alpha == 0.0) {
    auto ml = fit_ordinal_pair_joint_ml(
        counts, OrdinalPairJointMlOptions{
            .rho_lower = options.rho_lower,
            .rho_upper = options.rho_upper,
            .max_iter = options.max_iter,
            .ftol = options.ftol,
            .gtol = options.gtol,
            .fd_step = options.fd_step,
            .min_threshold_spacing = options.min_threshold_spacing,
            .lavaan_adjust_2x2 = options.lavaan_adjust_2x2});
    if (!ml.has_value()) return std::unexpected(ml.error());
    auto diag = dpd_result(ml->adjusted_counts, ml->thresholds_i,
                           ml->thresholds_j, ml->rho, 1.0);
    if (!diag.has_value()) return std::unexpected(diag.error());
    diag->objective = ml->negloglik / ml->adjusted_counts.sum();
    diag->gradient_inf = 0.0;
    diag->iterations = ml->iterations;
    diag->converged = true;
    diag->hit_lower = ml->hit_lower;
    diag->hit_upper = ml->hit_upper;
    diag->weights.setOnes();
    return diag;
  }

  Eigen::MatrixXd adjusted = options.lavaan_adjust_2x2
      ? adjusted_polychoric_table(counts)
      : Eigen::MatrixXd(counts);
  const auto th_i_start = marginal_threshold_start(
      adjusted.rowwise().sum(), options.min_threshold_spacing,
      "fit_ordinal_pair_joint_dpd");
  if (!th_i_start.has_value()) return std::unexpected(th_i_start.error());
  const auto th_j_start = marginal_threshold_start(
      adjusted.colwise().sum().transpose(), options.min_threshold_spacing,
      "fit_ordinal_pair_joint_dpd");
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
    return dpd_objective_unchecked(adjusted, th_i, th_j, rho, options.alpha);
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
        "fit_ordinal_pair_joint_dpd: non-finite starting objective"));
  }

  int iterations = 0;
  double gradient_inf = std::numeric_limits<double>::infinity();
  bool converged = false;
  for (; iterations < options.max_iter; ++iterations) {
    const Eigen::VectorXd grad = gradient(x);
    if (!grad.allFinite()) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "fit_ordinal_pair_joint_dpd: non-finite gradient"));
    }
    gradient_inf = grad.lpNorm<Eigen::Infinity>();
    if (gradient_inf <= options.gtol) {
      converged = true;
      break;
    }

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
      converged = true;
      break;
    }
  }

  const Eigen::VectorXd thresholds_i = decode_thresholds(
      x, 0, n_th_i, options.min_threshold_spacing);
  const Eigen::VectorXd thresholds_j = decode_thresholds(
      x, n_th_i, n_th_j, options.min_threshold_spacing);
  const double rho = decode_rho(x(npar - 1),
                                options.rho_lower,
                                options.rho_upper);
  const Eigen::VectorXd final_grad = gradient(x);
  if (final_grad.allFinite()) {
    gradient_inf = final_grad.lpNorm<Eigen::Infinity>();
    converged = converged || gradient_inf <= options.gtol;
  }

  auto out = dpd_result(adjusted, thresholds_i, thresholds_j, rho,
                        options.alpha);
  if (!out.has_value()) return std::unexpected(out.error());
  const double width = options.rho_upper - options.rho_lower;
  out->gradient_inf = gradient_inf;
  out->iterations = iterations;
  out->converged = converged;
  out->hit_lower = std::abs(rho - options.rho_lower) <= 1e-8 * width;
  out->hit_upper = std::abs(rho - options.rho_upper) <= 1e-8 * width;
  return out;
}

post_expected<OrdinalPairHWeightedInfluence>
ordinal_pair_h_weighted_influence(
    const Eigen::Ref<const Eigen::MatrixXd>& counts,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_i,
    const Eigen::Ref<const Eigen::VectorXd>& thresholds_j,
    double rho,
    OrdinalPairHWeightedInfluenceOptions options) {
  auto ok = validate_pair_shape(counts, thresholds_i, thresholds_j,
                                "ordinal_pair_h_weighted_influence");
  if (!ok.has_value()) return std::unexpected(ok.error());
  if (!std::isfinite(rho) || rho <= -1.0 || rho >= 1.0) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_h_weighted_influence: invalid options"));
  }
  auto h_ok = eval_polychoric_h_score(1.0, options.h_score);
  if (!h_ok.has_value()) return std::unexpected(h_ok.error());
  auto n_or = integer_total_count(counts, "ordinal_pair_h_weighted_influence");
  if (!n_or.has_value()) return std::unexpected(n_or.error());

  const auto n = static_cast<Eigen::Index>(*n_or);
  const Eigen::Index npar = thresholds_i.size() + thresholds_j.size() + 1;
  OrdinalPairHWeightedInfluence out;
  out.n_obs = *n_or;
  out.probabilities = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.ratios = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.h_values = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.dh_values = Eigen::MatrixXd::Zero(counts.rows(), counts.cols());
  out.weights = Eigen::MatrixXd::Ones(counts.rows(), counts.cols());
  out.estimating_functions = Eigen::MatrixXd::Zero(n, npar);

  Eigen::Index row = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -kInf : thresholds_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? kInf : thresholds_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -kInf : thresholds_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? kInf : thresholds_j(b);
      const double p = std::max(
          kProbFloor, ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
      const double t = counts(a, b) / (static_cast<double>(*n_or) * p);
      auto h = eval_polychoric_h_score(t, options.h_score);
      if (!h.has_value()) return std::unexpected(h.error());
      const double weight = t > 0.0 ? h->h / t : 1.0;
      out.probabilities(a, b) = p;
      out.ratios(a, b) = t;
      out.h_values(a, b) = h->h;
      out.dh_values(a, b) = h->dh;
      out.weights(a, b) = weight;

      const auto reps = static_cast<Eigen::Index>(std::llround(counts(a, b)));
      if (reps <= 0) continue;
      const Eigen::VectorXd score =
          h->dh * cell_log_score(a, b, thresholds_i, thresholds_j, rho);
      for (Eigen::Index k = 0; k < reps; ++k) {
        out.estimating_functions.row(row) = score.transpose();
        ++row;
      }
    }
  }
  if (row != n) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_h_weighted_influence: count expansion mismatch"));
  }

  out.estimating_functions.rowwise() -=
      out.estimating_functions.colwise().mean();
  out.score_gamma =
      (out.estimating_functions.transpose() * out.estimating_functions) /
      static_cast<double>(out.n_obs);
  auto bread_or = h_weighted_bread(counts, thresholds_i, thresholds_j, rho,
                                   options.h_score,
                                   "ordinal_pair_h_weighted_influence");
  if (!bread_or.has_value()) return std::unexpected(bread_or.error());
  out.bread = std::move(*bread_or);

  Eigen::FullPivLU<Eigen::MatrixXd> lu(out.bread);
  lu.setThreshold(1e-10);
  if (lu.rank() < out.bread.rows()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_h_weighted_influence: singular bread matrix"));
  }
  const Eigen::MatrixXd bread_inv =
      lu.solve(Eigen::MatrixXd::Identity(npar, npar));
  if (!bread_inv.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_pair_h_weighted_influence: non-finite bread inverse"));
  }
  out.influence = -out.estimating_functions * bread_inv.transpose();
  out.gamma = (out.influence.transpose() * out.influence) /
              static_cast<double>(out.n_obs);
  out.gamma = 0.5 * (out.gamma + out.gamma.transpose());
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
