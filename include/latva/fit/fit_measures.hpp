#pragma once

#include <cstdint>

#include "latva/fit/inference.hpp"      // Inference, Estimates (via fit.hpp)
#include "latva/fit/sample_stats.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/partable/partable.hpp"

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
// SRMR and the log-likelihood-based information criteria need the implied
// Σ̂(θ̂), not just T/df — see `fit_extras` below.
struct FitMeasures {
  double cfi   = 0.0;
  double tli   = 0.0;
  double rmsea = 0.0;
};

FitMeasures fit_measures(const Inference&  user,
                         const BaselineFit& baseline,
                         const SampleStats& samp) noexcept;

// Fit indices that need the implied moments Σ̂(θ̂) (and the sample
// S / means / n) — the log-likelihood-based information criteria and SRMR.
// Pure data, owned by the caller (mirrors `Inference`).
//
//   logl              = Σ_b −n_b/2·[ p_b·log 2π + log|Σ̂_b| + tr(S_b Σ̂_b⁻¹)
//                                     + (m̄_b−μ̂_b)ᵀΣ̂_b⁻¹(m̄_b−μ̂_b) ]
//   unrestricted_logl = Σ_b −n_b/2·[ p_b·log 2π + log|S_b| + p_b ]   (saturated)
//                       — equivalently `logl + chi2/2`.
//   aic  = −2·logl + 2·npar
//   bic  = −2·logl + npar·log(N)
//   bic2 = −2·logl + npar·log((N+2)/24)            (sample-size-adjusted BIC)
//   srmr = Σ_g (n_g/N)·√( Σ vech-residuals² / pstar )   (Bentler type:
//          residuals standardized by the *sample* SDs — off-diagonal
//          (s_ij−σ̂_ij)/√(s_ii s_jj), diagonal (s_ii−σ̂_ii)/s_ii; pstar =
//          p(p+1)/2, plus the p mean residuals (m̄_i−μ̂_i)/√(s_ii) and
//          pstar += p when the model has mean structure). 0 at saturation.
//   npar = #free parameters after equality merging (build_eq_constraints).
//   ntotal = Σ_b n_b.
//
// When the model has `fixed.x` exogenous *observed* variables (a `~` on
// observed covariates), `logl` / `unrestricted_logl` / `aic` / `bic` / `bic2`
// are *conditional* on those variables — the saturated marginal logl of the
// fixed.x block is subtracted (lavaan's convention). For pure CFA and for
// SEM with only latent exogenous, there's nothing to subtract.
//
// `pt` is taken by value because `fit_extras` resolves fixed.x `fixed_value`s
// from `samp` internally — symmetric with `fit()` / `ExpectedInfoSE::compute`.
// Returns `PostError::Kind::NumericIssue` if any implied Σ̂_b is non-PD at θ̂.
struct FitExtras {
  double    logl              = 0.0;
  double    unrestricted_logl = 0.0;
  double    aic               = 0.0;
  double    bic               = 0.0;
  double    bic2              = 0.0;
  double    srmr              = 0.0;
  int       npar              = 0;
  std::int64_t ntotal         = 0;
};

post_expected<FitExtras>
fit_extras(partable::LatentStructure        pt,
           const model::MatrixRep&   rep,
           const SampleStats&        samp,
           const Estimates&          est);

}  // namespace latva::fit
