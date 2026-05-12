#pragma once

#include "latva/fit/inference.hpp"
#include "latva/fit/sample_stats.hpp"

namespace latva::fit {

// Closed-form baseline (independence) model fit: every off-diagonal of
// every block's Σ is fixed at 0, with diagonal entries free (so the
// model-implied variance of each indicator just equals the sample
// variance). For meanstructure, intercepts are also free (saturated for
// means).
//
// Per block, the baseline F_b reduces to  log|diag(S_b)| − log|S_b|.
// T_baseline = Σ_b n_b · F_b, df_baseline = Σ_b p_b(p_b−1)/2.
struct BaselineFit {
  double chi2 = 0.0;
  int    df   = 0;
};

BaselineFit baseline_chi2(const SampleStats& samp) noexcept;

// Standard practical fit indices — closed-form functions of T_user,
// df_user, T_baseline, df_baseline, and N.
//
//   CFI   = 1 − max(0, T_u − df_u) / max(0, T_b − df_b)
//   TLI   = (T_b/df_b − T_u/df_u) / (T_b/df_b − 1)
//   RMSEA = √(max(0, (T_u − df_u) / (df_u · N)))
//
// SRMR isn't included here (it needs S vs Σ̂ residuals).
struct FitMeasures {
  double cfi   = 0.0;
  double tli   = 0.0;
  double rmsea = 0.0;
};

FitMeasures fit_measures(const Inference&  user,
                         const BaselineFit& baseline,
                         const SampleStats& samp) noexcept;

}  // namespace latva::fit
