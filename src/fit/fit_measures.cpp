#include "latva/fit/fit_measures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace latva::fit {

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

FitMeasures fit_measures(const Inference&   user,
                         const BaselineFit& baseline,
                         const SampleStats& samp) noexcept {
  FitMeasures out;
  const double T_u  = user.chi2;
  const double T_b  = baseline.chi2;
  const double df_u = user.df;
  const double df_b = baseline.df;

  // CFI = 1 − max(0, T_u − df_u) / max(0, T_b − df_b)
  const double num_u = std::max(0.0, T_u - df_u);
  const double num_b = std::max(0.0, T_b - df_b);
  out.cfi = (num_b > 0.0) ? std::max(0.0, 1.0 - num_u / num_b) : 1.0;

  // TLI: (T_b/df_b − T_u/df_u) / (T_b/df_b − 1). Undefined when df_b = 0
  // or df_u = 0; conventional return value is NaN (or 1 for a perfect
  // fit at the saturated case). We return NaN to flag the singular case.
  if (df_b > 0 && df_u > 0) {
    const double ratio_b = T_b / df_b;
    const double ratio_u = T_u / df_u;
    const double denom   = ratio_b - 1.0;
    out.tli = (std::abs(denom) > 0.0) ? (ratio_b - ratio_u) / denom
                                      : std::numeric_limits<double>::quiet_NaN();
  } else {
    out.tli = std::numeric_limits<double>::quiet_NaN();
  }

  // RMSEA = √(max(0, (T_u − df_u) / (df_u · N))). For multi-group we use
  // the total N (matches lavaan's default; some authors prefer Σ_b √(…)).
  std::int64_t N_total = 0;
  for (auto n : samp.n_obs) N_total += n;
  if (df_u > 0 && N_total > 0) {
    const double num = std::max(0.0, T_u - df_u);
    out.rmsea = std::sqrt(num / (df_u * static_cast<double>(N_total)));
  } else {
    out.rmsea = 0.0;
  }
  return out;
}

}  // namespace latva::fit
