#include "magmaan/robust/frontier/fmg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/weighted_chisq.hpp"

#include "../detail_distribution_math.hpp"

namespace magmaan::robust::frontier {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

using detail::clamp01;
using detail::f_upper_tail;
using detail::nan;

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

// EBA-j weights: sort into j equal blocks (size ceil(m/j), the last short) and
// replace every eigenvalue with its block average. EBA1 reproduces SB; EBAd
// (j = m) reproduces the raw spectrum used by `All`.
Eigen::VectorXd eba_lambdas(const Eigen::Ref<const Eigen::VectorXd>& lambdas,
                            int j) {
  const Eigen::Index m = lambdas.size();
  if (m == 0 || j <= 0) return Eigen::VectorXd::Zero(0);
  const Eigen::Index jj = static_cast<Eigen::Index>(j);
  const Eigen::Index k = (m + jj - 1) / jj;
  Eigen::VectorXd out(m);
  for (Eigen::Index i = 0; i < m; ++i) {
    const Eigen::Index col = i / k;
    const Eigen::Index start = col * k;
    const Eigen::Index len = std::min(k, m - start);
    out(i) = (len > 0) ? lambdas.segment(start, len).mean() : nan();
  }
  return out;
}

// pEBA-j weights: the EBA-j block averages, each pulled halfway toward the
// global mean (the penalization that counteracts eigenvalue under/overshoot).
Eigen::VectorXd peba_lambdas(const Eigen::Ref<const Eigen::VectorXd>& lambdas,
                             int j) {
  Eigen::VectorXd block = eba_lambdas(lambdas, j);
  if (block.size() == 0) return block;
  const double global_mean = lambdas.mean();
  return (0.5 * (block.array() + global_mean)).matrix();
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
    case FmgMethod::Eba:
      out.lambdas_reference =
          eba_lambdas(out.lambdas, static_cast<int>(std::ceil(options.param)));
      out.p_value = (out.lambdas_reference.size() > 0)
                        ? imhof_upper(out.lambdas_reference, chi2_source)
                        : nan();
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

}  // namespace magmaan::robust::frontier
