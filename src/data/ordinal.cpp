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
#include <Eigen/Eigenvalues>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

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

double bvn_pdf(double x, double y, double rho) noexcept {
  if (!std::isfinite(x) || !std::isfinite(y)) return 0.0;
  const double one_minus = std::max(1e-12, 1.0 - rho * rho);
  const double z = x * x - 2.0 * rho * x * y + y * y;
  constexpr double inv_2pi = 0.15915494309189533577;
  return inv_2pi / std::sqrt(one_minus) * std::exp(-0.5 * z / one_minus);
}

double rect_prob_raw(double lo1, double hi1, double lo2, double hi2,
                     double rho) noexcept {
  const double p = bvn_cdf(hi1, hi2, rho) - bvn_cdf(lo1, hi2, rho) -
                   bvn_cdf(hi1, lo2, rho) + bvn_cdf(lo1, lo2, rho);
  return p;
}

double rect_prob(double lo1, double hi1, double lo2, double hi2,
                 double rho) noexcept {
  return std::max(kProbFloor, rect_prob_raw(lo1, hi1, lo2, hi2, rho));
}

double rect_drho(double lo1, double hi1, double lo2, double hi2,
                 double rho) noexcept {
  return bvn_pdf(hi1, hi2, rho) - bvn_pdf(lo1, hi2, rho) -
         bvn_pdf(hi1, lo2, rho) + bvn_pdf(lo1, lo2, rho);
}

double neglog_pair(const Eigen::MatrixXd& tab,
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
      const double n = tab(a, b);
      if (n <= 0.0) continue;
      const double lo2 = (b == 0) ? -kInf : th2(b - 1);
      const double hi2 = (b + 1 == k2) ? kInf : th2(b);
      out -= n * std::log(rect_prob(lo1, hi1, lo2, hi2, rho));
    }
  }
  return out;
}

Eigen::MatrixXd adjusted_polychoric_table(const Eigen::MatrixXd& tab) {
  Eigen::MatrixXd out = tab;
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

double estimate_polychoric(const Eigen::MatrixXd& tab_in,
                           const Eigen::VectorXd& th1,
                           const Eigen::VectorXd& th2) noexcept {
  const Eigen::MatrixXd tab = adjusted_polychoric_table(tab_in);
  if (tab.rows() == 2 && tab.cols() == 2) {
    const double ad = tab(0, 0) * tab(1, 1);
    const double bc = tab(0, 1) * tab(1, 0);
    if (std::abs(ad - bc) <= 1e-12 * std::max(1.0, std::abs(ad) + std::abs(bc))) {
      return 0.0;
    }
    if (th1.size() == 1 && th2.size() == 1 &&
        std::abs(th1(0)) <= 1e-12 && std::abs(th2(0)) <= 1e-12) {
      return std::clamp(-std::cos(2.0 * kPi * tab(0, 0) / tab.sum()),
                        -0.999, 0.999);
    }
  }
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

double polyserial_prob(int cat, double u, double rho,
                       const Eigen::VectorXd& th) noexcept {
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));
  const double lo = (cat == 0) ? -kInf : th(cat - 1);
  const double hi = (cat == th.size()) ? kInf : th(cat);
  return std::max(kProbFloor,
                  normal_cdf((hi - rho * u) / sd) -
                      normal_cdf((lo - rho * u) / sd));
}

double neglog_polyserial(const Eigen::VectorXi& cat,
                         const Eigen::VectorXd& u,
                         double rho,
                         const Eigen::VectorXd& th) noexcept {
  double out = 0.0;
  for (Eigen::Index i = 0; i < cat.size(); ++i) {
    out -= std::log(polyserial_prob(cat(i), u(i), rho, th));
  }
  return out;
}

double estimate_polyserial(const Eigen::VectorXi& cat,
                           const Eigen::VectorXd& u,
                           const Eigen::VectorXd& th) noexcept {
  double lo = -0.999;
  double hi = 0.999;
  constexpr double gr = 0.6180339887498948482;
  double c = hi - gr * (hi - lo);
  double d = lo + gr * (hi - lo);
  double fc = neglog_polyserial(cat, u, c, th);
  double fd = neglog_polyserial(cat, u, d, th);
  for (int iter = 0; iter < 72; ++iter) {
    if (fc < fd) {
      hi = d;
      d = c;
      fd = fc;
      c = hi - gr * (hi - lo);
      fc = neglog_polyserial(cat, u, c, th);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + gr * (hi - lo);
      fd = neglog_polyserial(cat, u, d, th);
    }
  }
  return 0.5 * (lo + hi);
}

struct PolyserialScores {
  Eigen::VectorXd rho;
  Eigen::MatrixXd th;
};

PolyserialScores polyserial_scores(const Eigen::VectorXi& cat,
                                   const Eigen::VectorXd& u,
                                   double rho,
                                   const Eigen::VectorXd& th) {
  const Eigen::Index n = cat.size();
  const Eigen::Index nth = th.size();
  PolyserialScores out{Eigen::VectorXd::Zero(n),
                       Eigen::MatrixXd::Zero(n, nth)};
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));
  const double h = 1e-5;
  const double rp = std::min(0.999, rho + h);
  const double rm = std::max(-0.999, rho - h);
  for (Eigen::Index r = 0; r < n; ++r) {
    const int c = cat(r);
    const double lik = polyserial_prob(c, u(r), rho, th);
    const double pp = polyserial_prob(c, u(r), rp, th);
    const double pm = polyserial_prob(c, u(r), rm, th);
    out.rho(r) = (std::log(pp) - std::log(pm)) / (rp - rm);
    for (Eigen::Index a = 0; a < nth; ++a) {
      const double z = normal_pdf((th(a) - rho * u(r)) / sd) / sd;
      if (c == a) out.th(r, a) += z / lik;
      if (c == a + 1) out.th(r, a) -= z / lik;
    }
  }
  return out;
}

struct PairScores {
  Eigen::VectorXd rho;
  Eigen::MatrixXd th_i;
  Eigen::MatrixXd th_j;
};

PairScores pair_scores(const Eigen::MatrixXi& Xcat,
                       Eigen::Index i,
                       Eigen::Index j,
                       double rho,
                       const Eigen::VectorXd& th_i,
                       const Eigen::VectorXd& th_j) {
  const Eigen::Index n = Xcat.rows();
  const Eigen::Index ni = th_i.size();
  const Eigen::Index nj = th_j.size();
  PairScores out{
      Eigen::VectorXd::Zero(n),
      Eigen::MatrixXd::Zero(n, ni),
      Eigen::MatrixXd::Zero(n, nj)};
  const double sd = std::sqrt(std::max(1e-12, 1.0 - rho * rho));

  for (Eigen::Index r = 0; r < n; ++r) {
    const int ci = Xcat(r, i);
    const int cj = Xcat(r, j);
    const double lo_i = (ci == 0) ? -kInf : th_i(ci - 1);
    const double hi_i = (ci == ni) ? kInf : th_i(ci);
    const double lo_j = (cj == 0) ? -kInf : th_j(cj - 1);
    const double hi_j = (cj == nj) ? kInf : th_j(cj);
    const double lik = rect_prob(lo_i, hi_i, lo_j, hi_j, rho);
    out.rho(r) = rect_drho(lo_i, hi_i, lo_j, hi_j, rho) / lik;

    for (Eigen::Index a = 0; a < ni; ++a) {
      const double t = th_i(a);
      const double z = normal_pdf(t) *
          (normal_cdf((hi_j - rho * t) / sd) -
           normal_cdf((lo_j - rho * t) / sd));
      if (ci == a) out.th_i(r, a) += z / lik;
      if (ci == a + 1) out.th_i(r, a) -= z / lik;
    }
    for (Eigen::Index a = 0; a < nj; ++a) {
      const double t = th_j(a);
      const double z = normal_pdf(t) *
          (normal_cdf((hi_i - rho * t) / sd) -
           normal_cdf((lo_i - rho * t) / sd));
      if (cj == a) out.th_j(r, a) += z / lik;
      if (cj == a + 1) out.th_j(r, a) -= z / lik;
    }
  }
  return out;
}

Eigen::VectorXd mixed_moment_vector(const Eigen::MatrixXd& R,
                                    const Eigen::VectorXd& mean,
                                    const std::vector<std::int32_t>& ordered,
                                    const Eigen::VectorXd& thresholds) {
  const Eigen::Index p = R.rows();
  Eigen::Index n_cont = 0;
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) ++n_cont;
  }
  Eigen::VectorXd out(thresholds.size() + 2 * n_cont + p * (p - 1) / 2);
  Eigen::Index k = 0;
  out.segment(k, thresholds.size()) = thresholds;
  k += thresholds.size();
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) out(k++) = -mean(j);
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    if (ordered[static_cast<std::size_t>(j)] == 0) out(k++) = R(j, j);
  }
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) out(k++) = R(i, j);
  }
  return out;
}

post_expected<Eigen::MatrixXd> symmetric_inverse_pd(const Eigen::MatrixXd& A,
                                                    std::string what) {
  if (A.rows() != A.cols() || !A.allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not a finite square matrix"));
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      0.5 * (A + A.transpose()));
  if (es.info() != Eigen::Success || !es.eigenvalues().allFinite()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " eigendecomposition failed"));
  }
  const double max_eval = es.eigenvalues().maxCoeff();
  const double tol = 1e-10 * std::max(1.0, max_eval);
  if (es.eigenvalues().minCoeff() <= tol) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::move(what) + " is not positive definite"));
  }
  Eigen::VectorXd inv = es.eigenvalues().cwiseInverse();
  return es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
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
  out.NACOV.reserve(Xs.size());
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
    Eigen::MatrixXi Xcat(n, p);
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
        Xcat(i, j) = lvl - 1;
      }
      for (int c = 0; c < max_level; ++c) {
        if (counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)] == 0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                  " variable " + std::to_string(j) + " has an empty category"));
        }
      }
      levels[static_cast<std::size_t>(j)] = max_level;
    }

    Eigen::Index nth = 0;
    for (auto k : levels) nth += static_cast<Eigen::Index>(k - 1);
    const Eigen::Index ncorr = p * (p - 1) / 2;
    const Eigen::Index mdim = nth + ncorr;
    Eigen::VectorXd th(nth);
    std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), 0);
    std::vector<std::int32_t> th_ov;
    std::vector<std::int32_t> th_level;
    th_ov.reserve(static_cast<std::size_t>(nth));
    th_level.reserve(static_cast<std::size_t>(nth));

    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    Eigen::Index off = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      th_start[static_cast<std::size_t>(j)] = off;
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

    Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index j = 0; j < p; ++j) {
        const int c = Xcat(r, j);
        const auto& thj = th_by_var[static_cast<std::size_t>(j)];
        const double lo = (c == 0) ? -kInf : thj(c - 1);
        const double hi = (c == thj.size()) ? kInf : thj(c);
        const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
        const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
        if (c < thj.size()) SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
        if (c > 0) SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
      }
    }

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    Eigen::MatrixXd SC_COR = Eigen::MatrixXd::Zero(n, ncorr);
    Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(ncorr, nth);
    Eigen::Index corr_idx = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        Eigen::MatrixXd tab(levels[static_cast<std::size_t>(i)],
                            levels[static_cast<std::size_t>(j)]);
        tab.setZero();
        for (Eigen::Index r = 0; r < n; ++r) {
          tab(Xcat(r, i), Xcat(r, j)) += 1.0;
        }
        const double rho = estimate_polychoric(
            tab, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        R(i, j) = R(j, i) = rho;
        PairScores ps = pair_scores(
            Xcat, i, j, rho, th_by_var[static_cast<std::size_t>(i)],
            th_by_var[static_cast<std::size_t>(j)]);
        SC_COR.col(corr_idx) = ps.rho;
        const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
        const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
        A21.block(corr_idx, si, 1, ps.th_i.cols()) =
            ps.rho.transpose() * ps.th_i;
        A21.block(corr_idx, sj, 1, ps.th_j.cols()) =
            ps.rho.transpose() * ps.th_j;
        ++corr_idx;
      }
    }

    Eigen::MatrixXd SC(n, mdim);
    SC.leftCols(nth) = SC_TH;
    SC.rightCols(ncorr) = SC_COR;
    const Eigen::MatrixXd INNER = SC.transpose() * SC;

    Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
    const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
    for (Eigen::Index j = 0; j < p; ++j) {
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len = static_cast<Eigen::Index>(
          levels[static_cast<std::size_t>(j)] - 1);
      A11.block(start, start, len, len) =
          INNER_TH.block(start, start, len, len);
    }
    auto A11_inv_or = symmetric_inverse_pd(
        A11, "ordinal threshold information matrix");
    if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

    Eigen::VectorXd A22_diag(ncorr);
    for (Eigen::Index k = 0; k < ncorr; ++k) {
      A22_diag(k) = SC_COR.col(k).squaredNorm();
      if (A22_diag(k) <= 1e-12 || !std::isfinite(A22_diag(k))) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                " has a singular polychoric score block"));
      }
    }
    const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();

    Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(mdim, mdim);
    B_inv.block(0, 0, nth, nth) = *A11_inv_or;
    B_inv.block(nth, 0, ncorr, nth).noalias() =
        -A22_inv * A21 * (*A11_inv_or);
    B_inv.block(nth, nth, ncorr, ncorr) = A22_inv;

    Eigen::MatrixXd NACOV = static_cast<double>(n) *
        (B_inv * INNER * B_inv.transpose());
    NACOV = 0.5 * (NACOV + NACOV.transpose());

    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(mdim, mdim);
    for (Eigen::Index k = 0; k < mdim; ++k) {
      const double v = NACOV(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "ordinal_stats_from_integer_data: block " + std::to_string(b) +
                " has a non-positive NACOV diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    auto W_wls_or = symmetric_inverse_pd(
        NACOV, "ordinal NACOV matrix");
    if (!W_wls_or.has_value()) return std::unexpected(W_wls_or.error());

    out.R.push_back(std::move(R));
    out.thresholds.push_back(std::move(th));
    out.threshold_ov.push_back(std::move(th_ov));
    out.threshold_level.push_back(std::move(th_level));
    out.NACOV.push_back(std::move(NACOV));
    out.W_dwls.push_back(std::move(W_dwls));
    out.W_wls.push_back(std::move(*W_wls_or));
    out.n_obs.push_back(static_cast<std::int64_t>(n));
    out.n_levels.push_back(std::move(levels));
  }
  return out;
}

post_expected<MixedOrdinalStats>
mixed_ordinal_stats_from_data(const std::vector<Eigen::MatrixXd>& Xs,
                              const std::vector<std::vector<std::int32_t>>& ordered) {
  if (Xs.empty()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_from_data: no data blocks"));
  }
  if (Xs.size() != ordered.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "mixed_ordinal_stats_from_data: ordered mask block count mismatch"));
  }

  MixedOrdinalStats out;
  out.R.reserve(Xs.size());
  out.mean.reserve(Xs.size());
  out.ordered.reserve(Xs.size());
  out.thresholds.reserve(Xs.size());
  out.threshold_ov.reserve(Xs.size());
  out.threshold_level.reserve(Xs.size());
  out.moments.reserve(Xs.size());
  out.NACOV.reserve(Xs.size());
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
          "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
              " must have at least 2 rows and 1 column"));
    }
    if (ordered[b].size() != static_cast<std::size_t>(p)) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
              " ordered mask length does not match column count"));
    }
    bool have_ordered = false;
    bool have_cont = false;
    for (auto z : ordered[b]) {
      have_ordered = have_ordered || z != 0;
      have_cont = have_cont || z == 0;
    }
    if (!have_ordered || !have_cont) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
              " must contain both ordered and continuous variables"));
    }

    Eigen::MatrixXi Xcat = Eigen::MatrixXi::Zero(n, p);
    std::vector<std::int32_t> levels(static_cast<std::size_t>(p), 0);
    std::vector<std::vector<int>> counts(static_cast<std::size_t>(p));
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        for (Eigen::Index i = 0; i < n; ++i) {
          if (!std::isfinite(X(i, j))) {
            return std::unexpected(make_err(PostError::Kind::NumericIssue,
                "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                    " has non-finite continuous values"));
          }
        }
        continue;
      }
      int max_level = 0;
      for (Eigen::Index i = 0; i < n; ++i) {
        const double v = X(i, j);
        if (!std::isfinite(v) || std::floor(v) != v || v < 1.0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                  " has non-integer/non-positive category values"));
        }
        max_level = std::max(max_level, static_cast<int>(v));
      }
      if (max_level < 2) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                " variable " + std::to_string(j) + " has fewer than 2 levels"));
      }
      counts[static_cast<std::size_t>(j)].assign(static_cast<std::size_t>(max_level), 0);
      for (Eigen::Index i = 0; i < n; ++i) {
        const int lvl = static_cast<int>(X(i, j));
        counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(lvl - 1)] += 1;
        Xcat(i, j) = lvl - 1;
      }
      for (int c = 0; c < max_level; ++c) {
        if (counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)] == 0) {
          return std::unexpected(make_err(PostError::Kind::NumericIssue,
              "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                  " variable " + std::to_string(j) + " has an empty category"));
        }
      }
      levels[static_cast<std::size_t>(j)] = max_level;
    }

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd var = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) continue;
      mean(j) = X.col(j).mean();
      for (Eigen::Index i = 0; i < n; ++i) {
        const double c = X(i, j) - mean(j);
        var(j) += c * c;
      }
      var(j) /= static_cast<double>(n);
      if (var(j) <= 0.0 || !std::isfinite(var(j))) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                " has a non-positive continuous variance"));
      }
      U.col(j) = (X.col(j).array() - mean(j)) / std::sqrt(var(j));
    }

    Eigen::Index nth = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] != 0) {
        nth += static_cast<Eigen::Index>(levels[static_cast<std::size_t>(j)] - 1);
      }
    }
    Eigen::VectorXd th(nth);
    std::vector<Eigen::Index> th_start(static_cast<std::size_t>(p), -1);
    std::vector<std::int32_t> th_ov;
    std::vector<std::int32_t> th_level;
    std::vector<Eigen::VectorXd> th_by_var(static_cast<std::size_t>(p));
    Eigen::Index off = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
      th_start[static_cast<std::size_t>(j)] = off;
      const int k = levels[static_cast<std::size_t>(j)];
      th_by_var[static_cast<std::size_t>(j)].resize(k - 1);
      int cum = 0;
      for (int c = 0; c < k - 1; ++c) {
        cum += counts[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)];
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

    Eigen::MatrixXd SC_TH = Eigen::MatrixXd::Zero(n, nth);
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index j = 0; j < p; ++j) {
        if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
        const int c = Xcat(r, j);
        const auto& thj = th_by_var[static_cast<std::size_t>(j)];
        const double lo = (c == 0) ? -kInf : thj(c - 1);
        const double hi = (c == thj.size()) ? kInf : thj(c);
        const double pr = std::max(kProbFloor, normal_cdf(hi) - normal_cdf(lo));
        const Eigen::Index base = th_start[static_cast<std::size_t>(j)];
        if (c < thj.size()) SC_TH(r, base + c) += normal_pdf(thj(c)) / pr;
        if (c > 0) SC_TH(r, base + c - 1) -= normal_pdf(thj(c - 1)) / pr;
      }
    }
    const Eigen::MatrixXd INNER_TH = SC_TH.transpose() * SC_TH;
    Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(nth, nth);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) continue;
      const Eigen::Index start = th_start[static_cast<std::size_t>(j)];
      const Eigen::Index len = static_cast<Eigen::Index>(
          levels[static_cast<std::size_t>(j)] - 1);
      A11.block(start, start, len, len) =
          INNER_TH.block(start, start, len, len);
    }
    auto A11_inv_or = symmetric_inverse_pd(
        A11, "mixed ordinal threshold information matrix");
    if (!A11_inv_or.has_value()) return std::unexpected(A11_inv_or.error());

    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) R(j, j) = var(j);
    }

    std::vector<Eigen::VectorXd> assoc_scores;
    std::vector<Eigen::VectorXd> assoc_if_scale;
    Eigen::MatrixXd A21 = Eigen::MatrixXd::Zero(p * (p - 1) / 2, nth);
    Eigen::Index assoc_count = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        if (oi && oj) {
          Eigen::MatrixXd tab(levels[static_cast<std::size_t>(i)],
                              levels[static_cast<std::size_t>(j)]);
          tab.setZero();
          for (Eigen::Index r = 0; r < n; ++r) tab(Xcat(r, i), Xcat(r, j)) += 1.0;
          const double rho = estimate_polychoric(
              tab, th_by_var[static_cast<std::size_t>(i)],
              th_by_var[static_cast<std::size_t>(j)]);
          R(i, j) = R(j, i) = rho;
          PairScores ps = pair_scores(
              Xcat, i, j, rho, th_by_var[static_cast<std::size_t>(i)],
              th_by_var[static_cast<std::size_t>(j)]);
          assoc_scores.push_back(ps.rho);
          assoc_if_scale.push_back(Eigen::VectorXd::Ones(n));
          const Eigen::Index si = th_start[static_cast<std::size_t>(i)];
          const Eigen::Index sj = th_start[static_cast<std::size_t>(j)];
          A21.block(assoc_count, si, 1, ps.th_i.cols()) =
              ps.rho.transpose() * ps.th_i;
          A21.block(assoc_count, sj, 1, ps.th_j.cols()) =
              ps.rho.transpose() * ps.th_j;
        } else if (oi || oj) {
          const Eigen::Index o = oi ? i : j;
          const Eigen::Index c = oi ? j : i;
          Eigen::VectorXi cat(n);
          for (Eigen::Index r = 0; r < n; ++r) cat(r) = Xcat(r, o);
          const double rho = estimate_polyserial(
              cat, U.col(c), th_by_var[static_cast<std::size_t>(o)]);
          const double sd = std::sqrt(var(c));
          R(i, j) = R(j, i) = rho * sd;
          PolyserialScores ps = polyserial_scores(
              cat, U.col(c), rho, th_by_var[static_cast<std::size_t>(o)]);
          assoc_scores.push_back(ps.rho);
          assoc_if_scale.push_back(Eigen::VectorXd::Constant(n, sd));
          const Eigen::Index so = th_start[static_cast<std::size_t>(o)];
          A21.block(assoc_count, so, 1, ps.th.cols()) =
              ps.rho.transpose() * ps.th;
        } else {
          double cov = 0.0;
          for (Eigen::Index r = 0; r < n; ++r) {
            cov += (X(r, i) - mean(i)) * (X(r, j) - mean(j));
          }
          cov /= static_cast<double>(n);
          R(i, j) = R(j, i) = cov;
          assoc_scores.push_back(Eigen::VectorXd::Zero(n));
          assoc_if_scale.push_back(Eigen::VectorXd::Zero(n));
        }
        ++assoc_count;
      }
    }

    const Eigen::Index n_assoc = p * (p - 1) / 2;
    Eigen::MatrixXd SC_ASSOC(n, n_assoc);
    Eigen::VectorXd A22_diag(n_assoc);
    for (Eigen::Index k = 0; k < n_assoc; ++k) {
      SC_ASSOC.col(k) = assoc_scores[static_cast<std::size_t>(k)];
      A22_diag(k) = SC_ASSOC.col(k).squaredNorm();
      if (A22_diag(k) <= 1e-12 || !std::isfinite(A22_diag(k))) {
        A22_diag(k) = 1.0;
      }
    }
    const Eigen::MatrixXd A22_inv = A22_diag.cwiseInverse().asDiagonal();
    Eigen::MatrixXd B_inv = Eigen::MatrixXd::Zero(nth + n_assoc, nth + n_assoc);
    B_inv.block(0, 0, nth, nth) = *A11_inv_or;
    B_inv.block(nth, 0, n_assoc, nth).noalias() =
        -A22_inv * A21.topRows(n_assoc) * (*A11_inv_or);
    B_inv.block(nth, nth, n_assoc, n_assoc) = A22_inv;
    Eigen::MatrixXd SC(n, nth + n_assoc);
    SC.leftCols(nth) = SC_TH;
    SC.rightCols(n_assoc) = SC_ASSOC;
    Eigen::MatrixXd IF_est = static_cast<double>(n) * SC * B_inv.transpose();
    for (Eigen::Index k = 0; k < n_assoc; ++k) {
      const Eigen::Index row = nth + k;
      IF_est.col(row).array() *= assoc_if_scale[static_cast<std::size_t>(k)].array();
    }

    const Eigen::VectorXd moment = mixed_moment_vector(R, mean, ordered[b], th);
    Eigen::MatrixXd IF(n, moment.size());
    Eigen::Index pos = 0;
    IF.middleCols(pos, nth) = IF_est.leftCols(nth);
    pos += nth;
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        IF.col(pos++) = -(X.col(j).array() - mean(j)).matrix();
      }
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      if (ordered[b][static_cast<std::size_t>(j)] == 0) {
        IF.col(pos++) = (X.col(j).array() - mean(j)).square().matrix().array() - var(j);
      }
    }
    Eigen::Index assoc_pos = 0;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        const bool oi = ordered[b][static_cast<std::size_t>(i)] != 0;
        const bool oj = ordered[b][static_cast<std::size_t>(j)] != 0;
        if (oi || oj) {
          IF.col(pos++) = IF_est.col(nth + assoc_pos);
        } else {
          IF.col(pos++) =
              ((X.col(i).array() - mean(i)) * (X.col(j).array() - mean(j))).matrix().array() -
              R(i, j);
        }
        ++assoc_pos;
      }
    }
    Eigen::MatrixXd NACOV = (IF.transpose() * IF) / static_cast<double>(n);
    NACOV = 0.5 * (NACOV + NACOV.transpose());

    Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(NACOV.rows(), NACOV.cols());
    for (Eigen::Index k = 0; k < NACOV.rows(); ++k) {
      const double v = NACOV(k, k);
      if (!std::isfinite(v) || v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "mixed_ordinal_stats_from_data: block " + std::to_string(b) +
                " has a non-positive NACOV diagonal"));
      }
      W_dwls(k, k) = 1.0 / v;
    }
    auto W_wls_or = symmetric_inverse_pd(NACOV, "mixed ordinal NACOV matrix");
    if (!W_wls_or.has_value()) return std::unexpected(W_wls_or.error());

    out.R.push_back(std::move(R));
    out.mean.push_back(std::move(mean));
    out.ordered.push_back(ordered[b]);
    out.thresholds.push_back(std::move(th));
    out.threshold_ov.push_back(std::move(th_ov));
    out.threshold_level.push_back(std::move(th_level));
    out.moments.push_back(std::move(moment));
    out.NACOV.push_back(std::move(NACOV));
    out.W_dwls.push_back(std::move(W_dwls));
    out.W_wls.push_back(std::move(*W_wls_or));
    out.n_obs.push_back(static_cast<std::int64_t>(n));
    out.n_levels.push_back(std::move(levels));
  }
  return out;
}

}  // namespace magmaan::data
