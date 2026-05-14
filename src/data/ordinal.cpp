#include "magmaan/data/ordinal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::data {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

double normal_cdf(double x) noexcept {
  if (x == kInf) return 1.0;
  if (x == -kInf) return 0.0;
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normal_pdf(double x) noexcept {
  constexpr double inv_sqrt_2pi = 0.39894228040143267794;
  return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

// Acklam's rational approximation. Accuracy is ample for threshold starts and
// ACOV scaling; inputs are clamped before this is called.
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

double rect_prob(double lo1, double hi1, double lo2, double hi2,
                 double rho) noexcept {
  const double p = bvn_cdf(hi1, hi2, rho) - bvn_cdf(lo1, hi2, rho) -
                   bvn_cdf(hi1, lo2, rho) + bvn_cdf(lo1, lo2, rho);
  return std::max(1e-12, p);
}

double neglog_pair(const Eigen::MatrixXi& tab,
                   const Eigen::VectorXd& th1,
                   const Eigen::VectorXd& th2,
                   double rho) noexcept {
  const int k1 = static_cast<int>(tab.rows());
  const int k2 = static_cast<int>(tab.cols());
  double out = 0.0;
  for (int a = 0; a < k1; ++a) {
    const double lo1 = (a == 0) ? -kInf : th1(a - 1);
    const double hi1 = (a + 1 == k1) ? kInf : th1(a);
    for (int b = 0; b < k2; ++b) {
      const int n = tab(a, b);
      if (n == 0) continue;
      const double lo2 = (b == 0) ? -kInf : th2(b - 1);
      const double hi2 = (b + 1 == k2) ? kInf : th2(b);
      out -= static_cast<double>(n) * std::log(rect_prob(lo1, hi1, lo2, hi2, rho));
    }
  }
  return out;
}

double estimate_polychoric(const Eigen::MatrixXi& tab,
                           const Eigen::VectorXd& th1,
                           const Eigen::VectorXd& th2) noexcept {
  double lo = -0.999;
  double hi = 0.999;
  constexpr double gr = 0.6180339887498948482;
  double c = hi - gr * (hi - lo);
  double d = lo + gr * (hi - lo);
  double fc = neglog_pair(tab, th1, th2, c);
  double fd = neglog_pair(tab, th1, th2, d);
  for (int iter = 0; iter < 72; ++iter) {
    if (fc < fd) {
      hi = d;
      d = c;
      fd = fc;
      c = hi - gr * (hi - lo);
      fc = neglog_pair(tab, th1, th2, c);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + gr * (hi - lo);
      fd = neglog_pair(tab, th1, th2, d);
    }
  }
  return 0.5 * (lo + hi);
}

}  // namespace

post_expected<OrdinalStats>
ordinal_stats_from_integer_data(const std::vector<Eigen::MatrixXd>& Xs) {
  if (Xs.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ordinal_stats_from_integer_data: no data blocks"));
  }

  OrdinalStats out;
  out.R.reserve(Xs.size());
  out.thresholds.reserve(Xs.size());
  out.threshold_ov.reserve(Xs.size());
  out.threshold_level.reserve(Xs.size());
  out.W_dwls.reserve(Xs.size());
  out.W_wls.reserve(Xs.size());
  out.n_obs.reserve(Xs.size());
  out.n_levels.reserve(Xs.size());

  for (std::size_t b = 0; b < Xs.size(); ++b) {
    const auto& X = Xs[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    if (n < 2 || p == 0) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "ordinal_stats_from_integer_data: block " + std::to_string(b) +
              " must have at least 2 rows and 1 column"));
    }
    std::vector<std::int32_t> levels(static_cast<std::size_t>(p), 0);
    std::vector<std::vector<int>> counts(static_cast<std::size_t>(p));
    for (Eigen::Index j = 0; j < p; ++j) {
      int max_level = 0;
      for (Eigen::Index i = 0; i < n; ++i) {
        const double v = X(i, j);
        if (!std::isfinite(v) || std::floor(v) != v || v < 1.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                  " has non-integer/non-positive category values"));
        }
        max_level = std::max(max_level, static_cast<int>(v));
      }
      if (max_level < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                " variable " + std::to_string(j) + " has fewer than 2 levels"));
      }
      counts[static_cast<std::size_t>(j)].assign(static_cast<std::size_t>(max_level), 0);
      for (Eigen::Index i = 0; i < n; ++i) {
        const int lvl = static_cast<int>(X(i, j));
        counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(lvl - 1)] += 1;
      }
      levels[static_cast<std::size_t>(j)] = max_level;
    }

    Eigen::Index nth = 0;
    for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::Index mdim = nth + ncorr;
    Eigen::VectorXd th(nth);
    std::vector<std::int32_t> th_ov;
    std::vector<std::int32_t> th_level;
    th_ov.reserve(static_cast<std::size_t>(nth));
    th_level.reserve(static_cast<std::size_t>(nth));

    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    Eigen::Index off = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      const int k = levels[static_cast<std::size_t>(j)];
      th_by_var[static_cast<std::size_t>(j)].resize(k - 1);
      int cum = 0;
      for (int c = 0; c < k - 1; ++c) {
        cum += counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)];
        if (cum <= 0 || cum >= n) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                  " variable " + std::to_string(j) +
                  " has an empty boundary category"));
        }
        const double prob = std::clamp(static_cast<double>(cum) / static_cast<double>(n),
                                       1e-12, 1.0 - 1e-12);
        const double z = normal_quantile(prob);
        th(off) = z;
        th_by_var[static_cast<std::size_t>(j)](c) = z;
        th_ov.push_back(static_cast<std::int32_t>(j));
        th_level.push_back(static_cast<std::int32_t>(c + 1));
        ++off;
      }
    }

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    Eigen::Index corr_off = nth;
    Eigen::VectorXd var = Eigen::VectorXd::Ones(mdim);
    for (Eigen::Index k = 0; k < nth; ++k) {
      const double q = normal_cdf(th(k));
      const double ph = std::max(1e-8, normal_pdf(th(k)));
      var(k) = std::max(1e-8, q * (1.0 - q) /
                                  (static_cast<double>(n) * ph * ph));
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        Eigen::MatrixXi tab(levels[static_cast<std::size_t>(i)],
                            levels[static_cast<std::size_t>(j)]);
        tab.setZero();
        for (Eigen::Index r = 0; r < n; ++r) {
          const int li = static_cast<int>(X(r, i)) - 1;
          const int lj = static_cast<int>(X(r, j)) - 1;
          tab(li, lj) += 1;
        }
        const double rho = estimate_polychoric(
            tab, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        R(i, j) = R(j, i) = rho;
        var(corr_off) = std::max(1e-8, (1.0 - rho * rho) * (1.0 - rho * rho) /
                                         static_cast<double>(n));
        ++corr_off;
      }
    }

    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) W(k, k) = 1.0 / var(k);
    out.R.push_back(std::move(R));
    out.thresholds.push_back(std::move(th));
    out.threshold_ov.push_back(std::move(th_ov));
    out.threshold_level.push_back(std::move(th_level));
    out.W_dwls.push_back(W);
    out.W_wls.push_back(std::move(W));
    out.n_obs.push_back(static_cast<std::int64_t>(n));
    out.n_levels.push_back(std::move(levels));
  }
  return out;
}

}  // namespace magmaan::data
