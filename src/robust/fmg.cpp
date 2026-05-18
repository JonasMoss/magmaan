#include "magmaan/robust/fmg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/weighted_chisq.hpp"

namespace magmaan::robust {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

double nan() noexcept {
  return std::numeric_limits<double>::quiet_NaN();
}

double clamp01(double x) noexcept {
  if (std::isnan(x)) return x;
  return std::clamp(x, 0.0, 1.0);
}

double gamma_p_series(double a, double x) noexcept {
  if (!(a > 0.0) || x < 0.0) return nan();
  if (x == 0.0) return 0.0;
  double sum = 1.0 / a;
  double term = sum;
  constexpr int max_iter = 200;
  for (int n = 1; n < max_iter; ++n) {
    term *= x / (a + static_cast<double>(n));
    sum += term;
    if (std::abs(term) < std::abs(sum) * 1e-15) break;
  }
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

double gamma_q_cfrac(double a, double x) noexcept {
  if (!(a > 0.0) || x < 0.0) return nan();
  if (x == 0.0) return 1.0;
  constexpr double fpmin = 1e-300;
  double b = x + 1.0 - a;
  double c = 1.0 / fpmin;
  double d = 1.0 / b;
  double h = d;
  constexpr int max_iter = 200;
  for (int n = 1; n < max_iter; ++n) {
    const double an = -static_cast<double>(n) * (static_cast<double>(n) - a);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < fpmin) d = fpmin;
    c = b + an / c;
    if (std::abs(c) < fpmin) c = fpmin;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < 1e-15) break;
  }
  return h * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

double gamma_p(double a, double x) noexcept {
  if (x < a + 1.0) return gamma_p_series(a, x);
  return 1.0 - gamma_q_cfrac(a, x);
}

double gamma_q(double a, double x) noexcept {
  if (x < a + 1.0) return 1.0 - gamma_p_series(a, x);
  return gamma_q_cfrac(a, x);
}

double beta_cont_frac(double a, double b, double x) noexcept {
  constexpr int max_iter = 200;
  constexpr double eps = 3e-14;
  constexpr double fpmin = 1e-300;

  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::abs(d) < fpmin) d = fpmin;
  d = 1.0 / d;
  double h = d;

  for (int m = 1; m <= max_iter; ++m) {
    const double m2 = 2.0 * static_cast<double>(m);
    double aa = static_cast<double>(m) * (b - static_cast<double>(m)) * x /
                ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < fpmin) d = fpmin;
    c = 1.0 + aa / c;
    if (std::abs(c) < fpmin) c = fpmin;
    d = 1.0 / d;
    h *= d * c;

    aa = -(a + static_cast<double>(m)) * (qab + static_cast<double>(m)) * x /
         ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < fpmin) d = fpmin;
    c = 1.0 + aa / c;
    if (std::abs(c) < fpmin) c = fpmin;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < eps) break;
  }
  return h;
}

double regularized_beta(double a, double b, double x) noexcept {
  if (!(a > 0.0) || !(b > 0.0) || x < 0.0 || x > 1.0) return nan();
  if (x == 0.0) return 0.0;
  if (x == 1.0) return 1.0;
  const double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) -
                             std::lgamma(b) + a * std::log(x) +
                             b * std::log1p(-x));
  if (x < (a + 1.0) / (a + b + 2.0)) {
    return bt * beta_cont_frac(a, b, x) / a;
  }
  return 1.0 - bt * beta_cont_frac(b, a, 1.0 - x) / b;
}

double f_upper_tail(double x, double d1, double d2) noexcept {
  if (!(x >= 0.0) || !(d1 > 0.0) || !(d2 > 0.0)) return nan();
  if (x == 0.0) return 1.0;
  const bool d1_inf = std::isinf(d1);
  const bool d2_inf = std::isinf(d2);
  if (d1_inf && d2_inf) {
    if (x < 1.0) return 1.0;
    if (x > 1.0) return 0.0;
    return 0.5;
  }
  if (d1_inf) {
    return clamp01(gamma_p(0.5 * d2, 0.5 * d2 / x));
  }
  if (d2_inf) {
    return clamp01(gamma_q(0.5 * d1, 0.5 * d1 * x));
  }
  const double z = d2 / (d2 + d1 * x);
  return clamp01(regularized_beta(0.5 * d2, 0.5 * d1, z));
}

double chi2_equivalent(double p, int df) noexcept {
  if (df <= 0 || std::isnan(p)) return nan();
  if (p >= 1.0) return 0.0;
  if (p <= 0.0) return std::numeric_limits<double>::infinity();
  double lo = 0.0;
  double hi = static_cast<double>(df);
  while (inference::chi2_pvalue(hi, df) > p && hi < 1e12) hi *= 2.0;
  for (int it = 0; it < 120; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (inference::chi2_pvalue(mid, df) > p) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return hi;
}

Eigen::VectorXd top_lambdas(const Eigen::Ref<const Eigen::VectorXd>& eigvals,
                            int df) {
  if (df <= 0 || eigvals.size() == 0) return Eigen::VectorXd::Zero(0);
  Eigen::VectorXd sorted = eigvals;
  std::sort(sorted.data(), sorted.data() + sorted.size(), std::greater<double>());
  const Eigen::Index n = std::min<Eigen::Index>(
      static_cast<Eigen::Index>(df), sorted.size());
  return sorted.head(n);
}

Eigen::VectorXd peba_lambdas(const Eigen::Ref<const Eigen::VectorXd>& lambdas,
                             int j) {
  const Eigen::Index m = lambdas.size();
  if (m == 0 || j <= 0) return Eigen::VectorXd::Zero(0);
  const Eigen::Index jj = static_cast<Eigen::Index>(j);
  const Eigen::Index k = (m + jj - 1) / jj;
  Eigen::VectorXd col_mean(jj);
  for (Eigen::Index col = 0; col < jj; ++col) {
    const Eigen::Index start = col * k;
    const Eigen::Index len = std::min(k, m - start);
    col_mean(col) = (len > 0) ? lambdas.segment(start, len).mean() : nan();
  }
  const double global_mean = lambdas.mean();
  Eigen::VectorXd out(m);
  for (Eigen::Index i = 0; i < m; ++i) {
    const Eigen::Index col = i / k;
    out(i) = 0.5 * (col_mean(col) + global_mean);
  }
  return out;
}

Eigen::VectorXd pols_lambdas(const Eigen::Ref<const Eigen::VectorXd>& lambdas,
                             double gamma) {
  const Eigen::Index m = lambdas.size();
  if (m == 0 || !(gamma > 0.0)) return Eigen::VectorXd::Zero(0);
  if (m == 1) return lambdas;

  const double mean_x = 0.5 * (static_cast<double>(m) + 1.0);
  const double mean_y = lambdas.mean();
  double cov_num = 0.0;
  double var_num = 0.0;
  for (Eigen::Index i = 0; i < m; ++i) {
    const double x = static_cast<double>(i + 1);
    cov_num += (x - mean_x) * (lambdas(i) - mean_y);
    var_num += (x - mean_x) * (x - mean_x);
  }
  const double beta1 = (cov_num / var_num) / gamma;
  const double beta0 = mean_y - beta1 * mean_x;
  Eigen::VectorXd out(m);
  for (Eigen::Index i = 0; i < m; ++i) {
    const double x = static_cast<double>(i + 1);
    out(i) = std::max(beta0 + beta1 * x, 0.0);
  }
  return out;
}

double scaled_f_pvalue(double chi2, const Eigen::Ref<const Eigen::VectorXd>& lambdas) {
  if (lambdas.size() == 0) return nan();
  const double s1 = lambdas.sum();
  const double s2 = lambdas.array().square().sum();
  const double s3 = lambdas.array().cube().sum();
  const double denom = 2.0 * s1 * s2 * s2 - s1 * s1 * s3 + 2.0 * s2 * s3;

  double d1 = nan();
  double d2 = nan();
  double c = nan();
  if (denom > 0.0) {
    d1 = s1 * (s1 * s1 * s2 - 2.0 * s2 * s2 + 4.0 * s1 * s3) / denom;
    const double d2_denom = s3 * s1 - s2 * s2;
    d2 = (d2_denom > 0.0)
             ? (s1 * s1 * s2 + 2.0 * s2 * s2) / d2_denom + 6.0
             : std::numeric_limits<double>::infinity();
    if (d2 < 6.0) d2 = std::numeric_limits<double>::infinity();
    c = s1 * (s1 * s1 * s2 - 2.0 * s2 * s2 + 4.0 * s1 * s3) /
        (s1 * s1 * s2 - 4.0 * s2 * s2 + 6.0 * s1 * s3);
  } else {
    d1 = std::numeric_limits<double>::infinity();
    d2 = s1 * s1 / s2 + 4.0;
    c = s1 * (s1 * s1 + 2.0 * s2) / (s1 * s1 + 4.0 * s2);
  }
  return f_upper_tail(chi2 / c, d1, d2);
}

}  // namespace

FmgTestResult
fmg_test(double chi2_source,
         int df,
         const Eigen::Ref<const Eigen::VectorXd>& ugamma_eigenvalues,
         FmgOptions options) {
  FmgTestResult out;
  out.chi2_source = chi2_source;
  out.df = df;
  out.method = options.method;
  out.param = options.param;
  out.lambdas_raw = top_lambdas(ugamma_eigenvalues, df);
  out.lambdas = out.lambdas_raw;

  if (df <= 0 || !std::isfinite(chi2_source) || chi2_source < 0.0) {
    out.p_value = nan();
    out.chi2_equiv = nan();
    return out;
  }

  if (options.truncate_negative) {
    for (Eigen::Index i = 0; i < out.lambdas.size(); ++i) {
      if (out.lambdas(i) < 0.0) {
        out.lambdas(i) = 0.0;
        ++out.n_truncated;
      }
    }
  }

  switch (options.method) {
    case FmgMethod::StandardChiSquare:
      out.p_value = inference::chi2_pvalue(chi2_source, df);
      break;
    case FmgMethod::SatorraBentler:
      out.lambdas_reference = out.lambdas;
      out.p_value = inference::chi2_pvalue(
          satorra_bentler(chi2_source, df, out.lambdas).chi2_scaled, df);
      break;
    case FmgMethod::ScaledShifted: {
      out.lambdas_reference = out.lambdas;
      const auto ss = scaled_shifted(chi2_source, df, out.lambdas);
      out.p_value = inference::chi2_pvalue(ss.chi2_adj, ss.df);
      break;
    }
    case FmgMethod::ScaledF:
      out.lambdas_reference = out.lambdas;
      out.p_value = scaled_f_pvalue(chi2_source, out.lambdas);
      break;
    case FmgMethod::All:
      out.lambdas_reference = out.lambdas;
      out.p_value = imhof_upper(out.lambdas, chi2_source);
      break;
    case FmgMethod::PenalizedAll:
      if (out.lambdas.size() > 0) {
        out.lambdas_reference =
            0.5 * (out.lambdas.array() + out.lambdas.mean()).matrix();
        out.p_value = imhof_upper(out.lambdas_reference, chi2_source);
      } else {
        out.p_value = nan();
      }
      break;
    case FmgMethod::Peba:
      out.lambdas_reference =
          peba_lambdas(out.lambdas, static_cast<int>(std::ceil(options.param)));
      out.p_value = (out.lambdas_reference.size() > 0)
                        ? imhof_upper(out.lambdas_reference, chi2_source)
                        : nan();
      break;
    case FmgMethod::Pols:
      out.lambdas_reference = pols_lambdas(out.lambdas, options.param);
      out.p_value = (out.lambdas_reference.size() > 0)
                        ? imhof_upper(out.lambdas_reference, chi2_source)
                        : nan();
      break;
  }

  out.p_value = clamp01(out.p_value);
  out.chi2_equiv = chi2_equivalent(out.p_value, df);
  return out;
}

post_expected<FmgTestResult>
fmg_test_from_reduced_matrix(double chi2_source,
                             int df,
                             const Eigen::Ref<const Eigen::MatrixXd>& M,
                             FmgOptions options) {
  auto ev_or = ugamma_eigenvalues(M);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "fmg_test_from_reduced_matrix: ugamma_eigenvalues failed: " +
            ev_or.error().detail));
  }
  return fmg_test(chi2_source, df, *ev_or, options);
}

post_expected<FmgTestResult>
lr_test_fmg(double T_diff,
            const SatorraDiffResult& sd,
            FmgOptions options) {
  if (options.method == FmgMethod::StandardChiSquare ||
      options.method == FmgMethod::SatorraBentler ||
      options.method == FmgMethod::ScaledShifted ||
      options.method == FmgMethod::ScaledF) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "lr_test_fmg: nested FMG supports All, PenalizedAll, Peba, or Pols"));
  }
  return fmg_test(T_diff, static_cast<int>(sd.eigenvalues.size()),
                  sd.eigenvalues, options);
}

}  // namespace magmaan::robust
