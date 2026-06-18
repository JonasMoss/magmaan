#include "magmaan/measures/fit_measures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/model/fcsem_evaluator.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::measures {

using estimate::build_eq_constraints;
using estimate::resolve_fixed_x_from_sample;
using inference::noncentral_chisq_cdf;

using data::SampleStats;
using estimate::Estimates;

namespace {

constexpr double two_pi = 6.283185307179586476925286766559;

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

// log|A| via the Cholesky factor; assumes `llt` succeeded.
double log_det_from_llt(const Eigen::LLT<Eigen::MatrixXd>& llt) noexcept {
  double s = 0.0;
  const auto L = llt.matrixL();
  for (Eigen::Index i = 0; i < L.rows(); ++i) s += std::log(L(i, i));
  return 2.0 * s;
}

// Bisect a monotone `f` to a zero on [lo, hi]; caller guarantees opposite
// signs at the endpoints. ~200 halvings → ULP convergence. Cheap relative
// to each `f` call (a noncentral-χ² CDF evaluation).
template <class F>
double bisect_zero(F&& f, double lo, double hi) noexcept {
  double flo = f(lo);
  if (flo == 0.0) return lo;
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi) break;        // adjacent doubles — done
    const double fm = f(mid);
    if (fm == 0.0) return mid;
    if ((fm < 0.0) == (flo < 0.0)) { lo = mid; flo = fm; }
    else                           { hi = mid; }
  }
  return 0.5 * (lo + hi);
}

double chisq_upper_tail(double x, double df) noexcept {
  if (!(df > 0.0) || !std::isfinite(x)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double cdf = noncentral_chisq_cdf(x, df, 0.0);
  return std::isfinite(cdf) ? 1.0 - cdf
                            : std::numeric_limits<double>::quiet_NaN();
}

double lavaan_cfi(double x2, double df, double x2_null, double df_null,
                  double c_hat, double c_hat_null, bool robust) noexcept {
  if (!std::isfinite(x2) || !std::isfinite(df) ||
      !std::isfinite(x2_null) || !std::isfinite(df_null) ||
      !std::isfinite(c_hat) || !std::isfinite(c_hat_null)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double adj = robust ? c_hat : 1.0;
  const double adj_null = robust ? c_hat_null : 1.0;
  const double t1 = std::max(0.0, x2 - adj * df);
  const double t2 = std::max({x2 - adj * df,
                              x2_null - adj_null * df_null,
                              0.0});
  if (t1 == 0.0 && t2 == 0.0) return 1.0;
  return (t2 > 0.0) ? 1.0 - t1 / t2
                    : std::numeric_limits<double>::quiet_NaN();
}

double lavaan_tli(double x2, double df, double x2_null, double df_null,
                  double c_hat, double c_hat_null, bool robust) noexcept {
  if (!std::isfinite(x2) || !std::isfinite(df) ||
      !std::isfinite(x2_null) || !std::isfinite(df_null) ||
      !std::isfinite(c_hat) || !std::isfinite(c_hat_null)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (df <= 0.0) return 1.0;
  const double adj = robust ? c_hat : 1.0;
  const double adj_null = robust ? c_hat_null : 1.0;
  const double t1 = (x2 - adj * df) * df_null;
  const double t2 = (x2_null - adj_null * df_null) * df;
  return (std::abs(t2) > 0.0) ? 1.0 - t1 / t2
                              : std::numeric_limits<double>::quiet_NaN();
}

FitMeasures lavaan_rmsea_family(double x2, double df, std::int64_t n_total,
                                std::size_t n_groups, double c_hat,
                                double close_h0, double notclose_h0) noexcept {
  FitMeasures out;
  out.rmsea_close_h0 = close_h0;
  out.rmsea_notclose_h0 = notclose_h0;
  const double G = static_cast<double>(std::max<std::size_t>(1, n_groups));
  const double sqrtG = std::sqrt(G);
  if (df > 0.0 && n_total > 0 && std::isfinite(x2) &&
      std::isfinite(c_hat) && c_hat > 0.0) {
    const double n = static_cast<double>(n_total);
    out.rmsea = std::sqrt(std::max((x2 / n) / df - c_hat / n, 0.0)) * sqrtG;

    out.rmsea_ci_lower = 0.0;
    out.rmsea_ci_upper = 0.0;
    if (df >= 1.0 && x2 >= 0.0) {
      const double cdf0 = noncentral_chisq_cdf(x2, df, 0.0);
      const double scale = c_hat / (n * df);
      if (cdf0 >= 0.95) {
        const double lam_l = bisect_zero(
            [&](double lam) { return noncentral_chisq_cdf(x2, df, lam) - 0.95; },
            0.0, x2);
        if (std::isfinite(lam_l) && lam_l > 0.0)
          out.rmsea_ci_lower = std::sqrt(lam_l * scale) * sqrtG;
      }
      const double n_rmsea = std::max(n, 4.0 * x2);
      if (cdf0 >= 0.05 &&
          noncentral_chisq_cdf(x2, df, n_rmsea) <= 0.05) {
        const double lam_u = bisect_zero(
            [&](double lam) { return noncentral_chisq_cdf(x2, df, lam) - 0.05; },
            0.0, n_rmsea);
        if (std::isfinite(lam_u) && lam_u > 0.0)
          out.rmsea_ci_upper = std::sqrt(lam_u * scale) * sqrtG;
      }
    }

    const double ncp_close = n * df * close_h0 * close_h0 / (G * c_hat);
    const double cdf_close = noncentral_chisq_cdf(x2, df, ncp_close);
    out.rmsea_pvalue = std::isfinite(cdf_close)
                           ? 1.0 - cdf_close
                           : std::numeric_limits<double>::quiet_NaN();
    const double ncp_notclose = n * df * notclose_h0 * notclose_h0 /
                                (G * c_hat);
    out.rmsea_notclose_pvalue =
        noncentral_chisq_cdf(x2, df, ncp_notclose);
  } else {
    out.rmsea = 0.0;
    out.rmsea_ci_lower = 0.0;
    out.rmsea_ci_upper = 0.0;
    out.rmsea_pvalue = std::numeric_limits<double>::quiet_NaN();
    out.rmsea_notclose_pvalue = std::numeric_limits<double>::quiet_NaN();
  }
  return out;
}

post_expected<FitExtras>
fit_extras_from_implied(const spec::LatentStructure& pt,
                        const SampleStats& samp,
                        const model::ImpliedMoments& sm,
                        std::string_view label) {
  if (sm.sigma.size() != samp.S.size() || samp.n_obs.size() != samp.S.size()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        std::string(label) +
            ": SampleStats and implied moments have different block counts"));
  }

  auto con_or = build_eq_constraints(pt);
  if (!con_or.has_value()) return std::unexpected(con_or.error());
  const int npar = static_cast<int>(con_or->n_alpha);

  FitExtras out;
  out.npar = npar;

  std::int64_t N_total = 0;
  for (auto n : samp.n_obs) N_total += n;
  out.ntotal = N_total;

  // Fixed.x exogenous observed variables — lavaan reports `logl` (and hence
  // AIC/BIC) conditional on these, subtracting the saturated marginal logl of
  // the fixed.x block from both H0 and H1.
  std::vector<Eigen::Index> exo_idx;
  {
    std::unordered_set<std::int32_t> exo_vars;
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.exo[i] != 1) continue;
      if (i < pt.lhs_var.size() && pt.lhs_var[i] >= 0)
        exo_vars.insert(pt.lhs_var[i]);
      if (i < pt.rhs_var.size() && pt.rhs_var[i] >= 0)
        exo_vars.insert(pt.rhs_var[i]);
    }
    std::unordered_set<Eigen::Index> seen;
    for (std::int32_t v : exo_vars) {
      if (v < 0 || static_cast<std::size_t>(v) >= pt.ov_pos.size()) continue;
      const std::int32_t pos = pt.ov_pos[static_cast<std::size_t>(v)];
      if (pos < 0) continue;
      const Eigen::Index idx = static_cast<Eigen::Index>(pos);
      if (seen.insert(idx).second) exo_idx.push_back(idx);
    }
    std::sort(exo_idx.begin(), exo_idx.end());
  }

  double logl = 0.0;
  double logl_unres = 0.0;
  double srmr_acc = 0.0;

  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd S = 0.5 * (samp.S[b] + samp.S[b].transpose());
    const Eigen::Index p = S.rows();
    if (p == 0) continue;
    if (S.cols() != p || sm.sigma[b].rows() != p || sm.sigma[b].cols() != p) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(label) + ": block " + std::to_string(b) +
              " S and implied covariance dimensions differ"));
    }
    const double n_b = static_cast<double>(samp.n_obs[b]);

    const Eigen::MatrixXd Sigma =
        0.5 * (sm.sigma[b] + sm.sigma[b].transpose());
    Eigen::LLT<Eigen::MatrixXd> llt_sig(Sigma);
    if (llt_sig.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(label) + ": implied Sigma for block " +
              std::to_string(b) + " is not positive definite at theta-hat"));
    }
    const double log_det_sigma = log_det_from_llt(llt_sig);
    const Eigen::MatrixXd Sigma_inv =
        llt_sig.solve(Eigen::MatrixXd::Identity(p, p));

    Eigen::LLT<Eigen::MatrixXd> llt_S(S);
    if (llt_S.info() != Eigen::Success) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string(label) + ": sample S for block " + std::to_string(b) +
              " is not positive definite"));
    }
    const double log_det_S = log_det_from_llt(llt_S);
    const double tr_S_Sinv = S.cwiseProduct(Sigma_inv).sum();

    const bool has_means_b =
        (sm.mu.size() > b && sm.mu[b].size() == p) &&
        (samp.mean.size() > b && samp.mean[b].size() == p);
    double mahal = 0.0;
    Eigen::VectorXd mean_res;
    if (has_means_b) {
      const Eigen::VectorXd d = samp.mean[b] - sm.mu[b];
      mahal = d.dot(Sigma_inv * d);
      mean_res.resize(p);
      for (Eigen::Index i = 0; i < p; ++i)
        mean_res(i) = d(i) / std::sqrt(S(i, i));
    }

    logl += -0.5 * n_b * (static_cast<double>(p) * std::log(two_pi) +
                          log_det_sigma + tr_S_Sinv + mahal);
    logl_unres += -0.5 * n_b * (static_cast<double>(p) * std::log(two_pi) +
                                log_det_S + static_cast<double>(p));

    if (!exo_idx.empty() && exo_idx.back() < p) {
      const Eigen::Index px = static_cast<Eigen::Index>(exo_idx.size());
      Eigen::MatrixXd Sxx(px, px);
      for (Eigen::Index r = 0; r < px; ++r)
        for (Eigen::Index c = 0; c < px; ++c)
          Sxx(r, c) = S(exo_idx[static_cast<std::size_t>(r)],
                        exo_idx[static_cast<std::size_t>(c)]);
      Eigen::LLT<Eigen::MatrixXd> llt_xx(Sxx);
      if (llt_xx.info() != Eigen::Success) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            std::string(label) + ": fixed.x exogenous sub-block for block " +
                std::to_string(b) + " is not positive definite"));
      }
      const double marg =
          -0.5 * n_b *
          (static_cast<double>(px) * std::log(two_pi) +
           log_det_from_llt(llt_xx) + static_cast<double>(px));
      logl -= marg;
      logl_unres -= marg;
    }

    double sum_sq = 0.0;
    for (Eigen::Index c = 0; c < p; ++c) {
      const double dc = (S(c, c) - Sigma(c, c)) / S(c, c);
      sum_sq += dc * dc;
      for (Eigen::Index r = c + 1; r < p; ++r) {
        const double rij =
            (S(r, c) - Sigma(r, c)) / std::sqrt(S(r, r) * S(c, c));
        sum_sq += rij * rij;
      }
    }
    double pstar = static_cast<double>(p) * static_cast<double>(p + 1) / 2.0;
    if (has_means_b) {
      sum_sq += mean_res.squaredNorm();
      pstar += static_cast<double>(p);
    }
    const double srmr_b = (pstar > 0.0) ? std::sqrt(sum_sq / pstar) : 0.0;
    if (N_total > 0) srmr_acc += (n_b / static_cast<double>(N_total)) * srmr_b;
  }

  out.logl = logl;
  out.unrestricted_logl = logl_unres;
  out.srmr = srmr_acc;
  out.aic = -2.0 * logl + 2.0 * static_cast<double>(npar);
  out.bic = (N_total > 0)
                ? -2.0 * logl +
                      static_cast<double>(npar) *
                          std::log(static_cast<double>(N_total))
                : std::numeric_limits<double>::quiet_NaN();
  out.bic2 = (N_total > 0)
                 ? -2.0 * logl +
                       static_cast<double>(npar) *
                           std::log((static_cast<double>(N_total) + 2.0) /
                                    24.0)
                 : std::numeric_limits<double>::quiet_NaN();
  return out;
}

}  // namespace

BaselineFit baseline_chi2(const SampleStats& samp) noexcept {
  BaselineFit out;
  double total = 0.0;
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const auto& S = samp.S[b];
    const auto p = S.rows();
    // log|S| via LLT — falls back to "skip block" if S is non-PD (the
    // optimizer would already have flagged this).
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() != Eigen::Success) continue;
    double log_det_S = 0.0;
    for (Eigen::Index i = 0; i < p; ++i)
      log_det_S += std::log(llt.matrixL()(i, i));
    log_det_S *= 2.0;
    double log_det_diag = 0.0;
    for (Eigen::Index i = 0; i < p; ++i) log_det_diag += std::log(S(i, i));
    const double F_b = log_det_diag - log_det_S;
    total += static_cast<double>(samp.n_obs[b]) * F_b;
    out.df += static_cast<int>(p) * (static_cast<int>(p) - 1) / 2;
  }
  out.chi2 = total;
  return out;
}

BaselineFit baseline_chi2(const spec::LatentStructure& pt,
                          const SampleStats& samp) noexcept {
  BaselineFit out = baseline_chi2(samp);

  std::vector<Eigen::Index> exo_idx;
  {
    std::unordered_set<std::int32_t> exo_vars;
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.exo[i] != 1) continue;
      if (i < pt.lhs_var.size() && pt.lhs_var[i] >= 0) exo_vars.insert(pt.lhs_var[i]);
      if (i < pt.rhs_var.size() && pt.rhs_var[i] >= 0) exo_vars.insert(pt.rhs_var[i]);
    }
    std::unordered_set<Eigen::Index> seen;
    for (std::int32_t v : exo_vars) {
      if (v < 0 || static_cast<std::size_t>(v) >= pt.ov_pos.size()) continue;
      const std::int32_t pos = pt.ov_pos[static_cast<std::size_t>(v)];
      if (pos < 0) continue;
      const Eigen::Index idx = static_cast<Eigen::Index>(pos);
      if (seen.insert(idx).second) exo_idx.push_back(idx);
    }
    std::sort(exo_idx.begin(), exo_idx.end());
  }
  if (exo_idx.size() < 2) return out;

  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const auto& S = samp.S[b];
    const Eigen::Index p = S.rows();
    if (exo_idx.back() >= p) continue;
    const Eigen::Index px = static_cast<Eigen::Index>(exo_idx.size());
    Eigen::MatrixXd Sxx(px, px);
    for (Eigen::Index r = 0; r < px; ++r) {
      for (Eigen::Index c = 0; c < px; ++c) {
        Sxx(r, c) = S(exo_idx[static_cast<std::size_t>(r)],
                      exo_idx[static_cast<std::size_t>(c)]);
      }
    }
    Eigen::LLT<Eigen::MatrixXd> llt(Sxx);
    if (llt.info() != Eigen::Success) continue;
    double log_det_Sxx = 0.0;
    for (Eigen::Index i = 0; i < px; ++i)
      log_det_Sxx += std::log(llt.matrixL()(i, i));
    log_det_Sxx *= 2.0;
    double log_det_diag = 0.0;
    for (Eigen::Index i = 0; i < px; ++i) log_det_diag += std::log(Sxx(i, i));
    const double F_b = log_det_diag - log_det_Sxx;
    out.chi2 -= static_cast<double>(samp.n_obs[b]) * F_b;
  }
  const int px = static_cast<int>(exo_idx.size());
  out.df -= px * (px - 1) / 2;
  if (out.df < 0) out.df = 0;
  return out;
}

FitMeasures fit_measures(double             chi2_user,
                         int                df_user,
                         const BaselineFit& baseline,
                         std::int64_t        N_total,
                         std::size_t         n_groups) noexcept {
  FitMeasures out;
  const double T_u  = chi2_user;
  const double T_b  = baseline.chi2;
  const double df_u = static_cast<double>(df_user);
  const double df_b = static_cast<double>(baseline.df);

  // CFI = 1 − max(0, T_u − df_u) / max(0, T_b − df_b)
  const double num_u = std::max(0.0, T_u - df_u);
  const double num_b = std::max(0.0, T_b - df_b);
  out.cfi = (num_b > 0.0) ? std::max(0.0, 1.0 - num_u / num_b) : 1.0;

  // TLI: (T_b/df_b − T_u/df_u) / (T_b/df_b − 1). lavaan reports 1 for
  // saturated user models (df_u = 0); otherwise the ratio needs positive
  // baseline and user degrees of freedom.
  if (df_u == 0.0) {
    out.tli = 1.0;
  } else if (df_b > 0 && df_u > 0) {
    const double ratio_b = T_b / df_b;
    const double ratio_u = T_u / df_u;
    const double denom   = ratio_b - 1.0;
    out.tli = (std::abs(denom) > 0.0) ? (ratio_b - ratio_u) / denom
                                      : std::numeric_limits<double>::quiet_NaN();
  } else {
    out.tli = std::numeric_limits<double>::quiet_NaN();
  }

  // RMSEA = √(max(0, (T_u − df_u) / (df_u · N))) · √G. The √G multi-group
  // correction is lavaan's convention (Steiger; `lav_fit_rmsea`'s `* sqrt(G)`)
  // — with one group it's the textbook formula.
  const double sqrtG =
      std::sqrt(static_cast<double>(std::max<std::size_t>(1, n_groups)));
  if (df_u > 0 && N_total > 0) {
    const double num = std::max(0.0, T_u - df_u);
    out.rmsea = std::sqrt(num / (df_u * static_cast<double>(N_total))) * sqrtG;
  } else {
    out.rmsea = 0.0;
  }

  // RMSEA 90% confidence interval (lavaan `lav_fit_rmsea_ci`, c.hat = 1):
  // λ_L solves the noncentral-χ²(df_u, λ) CDF at T_u == 0.95 (else 0);
  // λ_U solves it == 0.05 (else 0); bound = √(λ/(df_u·N)) · √G.
  out.rmsea_ci_lower = 0.0;
  out.rmsea_ci_upper = 0.0;
  if (df_u >= 1.0 && N_total > 0 && std::isfinite(T_u) && T_u >= 0.0) {
    const double scale = 1.0 / (df_u * static_cast<double>(N_total));
    const double cdf0  = noncentral_chisq_cdf(T_u, df_u, 0.0);  // central χ²(df_u) at T_u
    if (cdf0 >= 0.95) {  // else lower.lambda(0) < 0  ⇒  lower bound 0
      const double lam_l = bisect_zero(
          [&](double lam) { return noncentral_chisq_cdf(T_u, df_u, lam) - 0.95; },
          0.0, T_u);
      if (std::isfinite(lam_l) && lam_l > 0.0)
        out.rmsea_ci_lower = std::sqrt(lam_l * scale) * sqrtG;
    }
    const double n_rmsea = std::max(static_cast<double>(N_total), 4.0 * T_u);
    // else upper.lambda(0) < 0 (tiny T_u) or upper.lambda(N.RMSEA) > 0  ⇒  0
    if (cdf0 >= 0.05 && noncentral_chisq_cdf(T_u, df_u, n_rmsea) <= 0.05) {
      const double lam_u = bisect_zero(
          [&](double lam) { return noncentral_chisq_cdf(T_u, df_u, lam) - 0.05; },
          0.0, n_rmsea);
      if (std::isfinite(lam_u) && lam_u > 0.0)
        out.rmsea_ci_upper = std::sqrt(lam_u * scale) * sqrtG;
    }
  }
  out.rmsea_pvalue = std::numeric_limits<double>::quiet_NaN();
  out.rmsea_notclose_pvalue = std::numeric_limits<double>::quiet_NaN();
  if (df_u > 0.0 && N_total > 0 && std::isfinite(T_u) && T_u >= 0.0) {
    const double G = static_cast<double>(std::max<std::size_t>(1, n_groups));
    const double ncp_close =
        (static_cast<double>(N_total) * df_u * out.rmsea_close_h0 *
         out.rmsea_close_h0) / G;
    const double cdf_close = noncentral_chisq_cdf(T_u, df_u, ncp_close);
    out.rmsea_pvalue = std::isfinite(cdf_close) ? 1.0 - cdf_close
                                                : std::numeric_limits<double>::quiet_NaN();

    const double ncp_notclose =
        (static_cast<double>(N_total) * df_u * out.rmsea_notclose_h0 *
         out.rmsea_notclose_h0) / G;
    out.rmsea_notclose_pvalue =
        noncentral_chisq_cdf(T_u, df_u, ncp_notclose);
  }
  return out;
}

FitMeasures fit_measures(double             chi2_user,
                         int                df_user,
                         const BaselineFit& baseline,
                         const SampleStats& samp) noexcept {
  std::int64_t N_total = 0;
  for (auto n : samp.n_obs) N_total += n;
  return fit_measures(chi2_user, df_user, baseline, N_total, samp.S.size());
}

RobustFitMeasures
robust_fit_measures(const RobustFitMeasureInputs& in) noexcept {
  RobustFitMeasures out;
  const double df = static_cast<double>(in.df);
  const double df_b = static_cast<double>(in.baseline_df);
  const double c = in.scaling_factor;
  const double c_b = in.baseline_scaling_factor;

  out.chisq_scaled = in.chi2_scaled;
  out.df_scaled = in.df;
  out.pvalue_scaled = chisq_upper_tail(in.chi2_scaled, df);
  out.chisq_scaling_factor = c;

  out.baseline_chisq_scaled = in.baseline_chi2_scaled;
  out.baseline_df_scaled = in.baseline_df;
  out.baseline_pvalue_scaled =
      chisq_upper_tail(in.baseline_chi2_scaled, df_b);
  out.baseline_chisq_scaling_factor = c_b;

  out.cfi_scaled = lavaan_cfi(in.chi2_scaled, df,
                              in.baseline_chi2_scaled, df_b,
                              1.0, 1.0, false);
  out.tli_scaled = lavaan_tli(in.chi2_scaled, df,
                              in.baseline_chi2_scaled, df_b,
                              1.0, 1.0, false);
  out.cfi_robust = lavaan_cfi(in.chi2, df, in.baseline_chi2, df_b,
                              c, c_b, true);
  out.tli_robust = lavaan_tli(in.chi2, df, in.baseline_chi2, df_b,
                              c, c_b, true);

  const double scaled_df = (df > 0.0 && std::isfinite(c) && c > 0.0)
                               ? df * c
                               : std::numeric_limits<double>::quiet_NaN();
  const FitMeasures rmsea_scaled =
      lavaan_rmsea_family(in.chi2, scaled_df, in.n_total, in.n_groups,
                          1.0, in.rmsea_close_h0, in.rmsea_notclose_h0);
  out.rmsea_scaled = rmsea_scaled.rmsea;
  out.rmsea_ci_lower_scaled = rmsea_scaled.rmsea_ci_lower;
  out.rmsea_ci_upper_scaled = rmsea_scaled.rmsea_ci_upper;
  out.rmsea_pvalue_scaled = rmsea_scaled.rmsea_pvalue;
  out.rmsea_notclose_pvalue_scaled =
      rmsea_scaled.rmsea_notclose_pvalue;

  const FitMeasures rmsea_robust_value =
      lavaan_rmsea_family(in.chi2, df, in.n_total, in.n_groups,
                          c, in.rmsea_close_h0, in.rmsea_notclose_h0);
  const FitMeasures rmsea_robust_tail =
      lavaan_rmsea_family(in.chi2_scaled, df, in.n_total, in.n_groups,
                          c, in.rmsea_close_h0, in.rmsea_notclose_h0);
  out.rmsea_robust = rmsea_robust_value.rmsea;
  out.rmsea_ci_lower_robust = rmsea_robust_tail.rmsea_ci_lower;
  out.rmsea_ci_upper_robust = rmsea_robust_tail.rmsea_ci_upper;
  out.rmsea_pvalue_robust = rmsea_robust_tail.rmsea_pvalue;
  out.rmsea_notclose_pvalue_robust =
      rmsea_robust_tail.rmsea_notclose_pvalue;
  return out;
}

post_expected<FitExtras>
fit_extras(spec::LatentStructure        pt,
           const model::MatrixRep&   rep,
           const SampleStats&        samp,
           const Estimates&          est) {
  // Mirror fit() / information_expected: resolve fixed.x from the sample,
  // then rebuild the evaluator. Without the fixed.x fill, ModelEvaluator
  // would see NA fixed_values as 0 and the implied Σ would collapse on
  // path-style models.
  if (auto e = resolve_fixed_x_from_sample(pt, rep, samp); !e.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "resolve_fixed_x_from_sample failed: " + e.error().detail));
  }
  auto ev_or = model::ModelEvaluator::build(pt, rep);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ModelEvaluator::build failed: " + ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  if (static_cast<std::size_t>(est.theta.size()) != ev.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "Estimates.theta size " + std::to_string(est.theta.size()) +
            " ≠ evaluator n_free " + std::to_string(ev.n_free())));
  }
  if (samp.S.size() != ev.n_blocks()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "SampleStats and evaluator have different block counts"));
  }

  auto sm_or = ev.sigma(est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "ev.sigma(θ̂) failed: " + sm_or.error().detail));
  }
  return fit_extras_from_implied(pt, samp, *sm_or, "fit_extras");
}

post_expected<FitExtras>
fit_extras_fcsem(const spec::LatentStructure& pt,
                 const SampleStats&           samp,
                 const Estimates&             est) {
  auto ev_or = model::FcSemEvaluator::build(pt);
  if (!ev_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "FcSemEvaluator::build failed: " + ev_or.error().detail));
  }
  const auto& ev = *ev_or;
  if (static_cast<std::size_t>(est.theta.size()) != ev.n_free()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "Estimates.theta size " + std::to_string(est.theta.size()) +
            " != evaluator n_free " + std::to_string(ev.n_free())));
  }
  if (samp.S.size() != ev.n_blocks()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "SampleStats and evaluator have different block counts"));
  }

  auto sm_or = ev.sigma(samp, est.theta);
  if (!sm_or.has_value()) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "FcSemEvaluator::sigma failed: " + sm_or.error().detail));
  }
  return fit_extras_from_implied(pt, samp, *sm_or, "fit_extras_fcsem");
}

}  // namespace magmaan::measures
