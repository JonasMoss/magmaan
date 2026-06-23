#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

namespace magmaan::estimate {

using data::SampleStats;
using inference::ScoreCandidate;
using inference::ScoreCandidateKind;
using inference::ScoreTestResult;
using inference::ScoreTestTable;
using inference::chi2_pvalue;
using robust::MeanVarAdjustedResult;
using robust::SatorraBentlerResult;
using robust::ScaledShiftedResult;

enum class OrdinalWeightKind {
  DWLS,
  WLS,
  ULS,  // identity weight (no NACOV needed for the fit); robust = ULSMV
};

enum class OrdinalParameterization {
  Delta,
  Theta,
};

struct OrdinalRobustResult {
  Eigen::MatrixXd vcov;              // full free-parameter covariance
  Eigen::VectorXd se;                // sqrt(diag(vcov))
  Eigen::VectorXd eigvals;           // nonzero U-Gamma eigenvalues
  double chisq_standard = 0.0;       // N * F_min
  int df = 0;
  robust::SatorraBentlerResult satorra_bentler;
  robust::MeanVarAdjustedResult mean_var_adjusted;
  robust::ScaledShiftedResult scaled_shifted;
};

struct OrdinalFitMeasures {
  measures::BaselineFit baseline;
  measures::FitMeasures indices;
  double srmr = 0.0;
  double crmr = 0.0;
};

struct OrdinalCatmlDwlsRmsea {
  double xx3 = 0.0;
  int df3 = 0;
  double c_hat3 = 0.0;
  double xx3_scaled = 0.0;
  double rmsea_robust = 0.0;
};

// Misspecification-aware inference for a correlation/standardized RMR fit index
// (CRMR by default; SRMR with `srmr_denominator`). The criterion is the sum of
// squared polychoric-correlation residuals G evaluated at the DWLS estimator; the
// reported CI is the absolute-fit companion to `ordinal_dwls_profile_rmsea` and
// propagates the sampling variability of the estimated weight (the γ channel)
// through the joint NACOV of the extended (u, γ) first-stage vector. `stat = N·G`.
// `point = sqrt(G/k)`; `point_bias_corrected` subtracts the signed bias trace.
// The trace uses the Gauss-Newton Q_G = Dφᵀ V₀ Dφ; the residual·curvature term
// (∂²r/∂x², O(‖r‖)) is omitted, so the correction is exact at the null and a
// leading-order approximation off it (the CI width and exact-fit p-value, driven
// by the full chain-rule gradient g_G, are unaffected).
// With `estimated_weight = false` the γ channel is dropped (fixed-weight
// comparator = current practice). Multi-group: criteria pool as sample-size-
// weighted sums over the block-diagonal per-group Γ_x. For G>1 `point` is the
// root of the pooled mean-square, sqrt(Σ_b (n_b/N)‖r_b‖²/k); by Jensen this is
// ≥ fit_measures' CRMR (lavaan's per-group Σ_b (n_b/N)·sqrt(‖r_b‖²/k)), the two
// agreeing only at one group. The bias-corrected point and CI live on this same
// pooled quadratic-form (N·G) scale.
struct OrdinalCrmrInference {
  double point = 0.0;                 // sqrt(G / k)
  double point_bias_corrected = 0.0;  // sqrt(max(N·G − tr(QΓ),0)/(N·k))
  double exact_fit_pvalue = 0.0;      // Pr(Σλⱼχ²₁ > N·G)
  double ci_lower = 0.0;              // misspec normal-theory interval, mapped
  double ci_upper = 0.0;
  double bias_trace = 0.0;            // tr(Q_G Γ_x) (signed)
  double grad_var = 0.0;             // g_Gᵀ Γ_x g_G (per-unit; Var(N·G)=N·grad_var)
  double stat = 0.0;                  // N · G
  int k = 0;                          // denominator (off-diag, or vech for srmr)
  int spectrum_size = 0;
  bool fixed_weight = false;
  bool srmr_denominator = false;
  Eigen::VectorXd eigvals;            // positive Q_G Γ_x spectrum
  std::vector<std::string> warnings;
};

// Misspecification-aware inference for RMSEA, the absolute-fit companion whose
// metric IS the estimated weight (so its γ channel is large, unlike CRMR). The
// criterion is the DWLS discrepancy F = rᵀ W r itself; since θ̂ minimizes it, the
// envelope theorem makes the gradient the bare profile score
// g_F = (−2 W r, −r²/γ²) with no estimator-projection term. `point` is the
// bias-corrected RMSEA (= ordinal_dwls_profile_rmsea's rmsea); the CI propagates
// the estimated weight through `grad_var = g_Fᵀ Γ_x g_F` (Var(N·F)=N·grad_var).
// `estimated_weight=false` zeros the γ channel (fixed-weight comparator).
// Multi-group: criteria pool as sample-size-weighted sums over the block-diagonal per-group Γ_x.
struct OrdinalRmseaInference {
  double point = 0.0;            // sqrt(max(F − tr(QΓ)/N,0)·G/df)
  double ci_lower = 0.0;         // misspec normal-theory interval on F₀, mapped
  double ci_upper = 0.0;
  double exact_fit_pvalue = 0.0; // Pr(Σλⱼχ²₁ > N·F)
  double bias_trace = 0.0;       // tr(Q Γ_x) (signed)
  double grad_var = 0.0;         // g_Fᵀ Γ_x g_F (per-unit; Var(N·F)=N·grad_var)
  double fmin = 0.0;             // F = full DWLS discrepancy
  double stat = 0.0;             // N · F
  int df = 0;
  int spectrum_size = 0;
  bool fixed_weight = false;
  Eigen::VectorXd eigvals;       // positive Q Γ_x spectrum
  std::vector<std::string> warnings;
};

// Estimated-weight misspecification inference for the incremental fit indices
// CFI and TLI, the program's first two-model (user-vs-baseline) statistics. The
// noncentralities δ_u = T_u − tr(Q_u Γ_x) and δ_b = T_b − tr(Q_b Γ_x) share the
// one data-only joint NACOV Γ_x, so their joint law is a single bilinear form:
// Var(T_u)=N gᵤᵀΓ_x gᵤ, Var(T_b)=N g_bᵀΓ_x g_b, Cov=N gᵤᵀΓ_x g_b, with the
// envelope-score gradients g = (−2Wd, −d²/γ²) of each model. CFI = 1 − δ_u/δ_b
// and TLI = 1 − (Q̄_b/Q̄_u)·δ_u/δ_b are then a ratio delta-method; the TLI
// interval is the CFI interval scaled by Q̄_b/Q̄_u (Q̄ = signed bias trace, the
// generalized df). `estimated_weight=false` zeros the γ channel (fixed-weight
// comparator). See docs/research/notes/cfi_tli_misspec_inference.tex.
// Multi-group: criteria pool as sample-size-weighted sums over the block-diagonal per-group Γ_x.
struct OrdinalIncrementalFitInference {
  double cfi = 0.0;               // 1 − δ_u/δ_b, clipped to [0,1]
  double cfi_ci_lower = 0.0;      // misspec normal-theory interval, clipped
  double cfi_ci_upper = 0.0;
  double tli = 0.0;               // 1 − (Q̄_b/Q̄_u)·δ_u/δ_b
  double tli_ci_lower = 0.0;
  double tli_ci_upper = 0.0;
  double delta_user = 0.0;        // δ_u = T_u − Q̄_u
  double delta_baseline = 0.0;    // δ_b = T_b − Q̄_b
  double stat_user = 0.0;         // T_u = N·F_u
  double stat_baseline = 0.0;     // T_b = N·F_b
  double gendf_user = 0.0;        // Q̄_u = tr(Q_u Γ_x) (signed)
  double gendf_baseline = 0.0;    // Q̄_b = tr(Q_b Γ_x) (signed)
  double var_user = 0.0;          // V_uu = N gᵤᵀΓ_x gᵤ = Var(T_u)
  double var_baseline = 0.0;      // V_bb = Var(T_b)
  double cov_user_baseline = 0.0; // V_ub = Cov(T_u, T_b)
  double var_cfi = 0.0;           // delta-method Var(ĈFI)
  double var_tli = 0.0;           // = (Q̄_b/Q̄_u)² · var_cfi
  int df_user = 0;                // nominal df (reference)
  int df_baseline = 0;
  bool fixed_weight = false;
  std::vector<std::string> warnings;
};

// Consolidated estimated-weight misspecification fit table for an all-ordinal
// DWLS fit: the reportable RMSEA, CRMR, SRMR, CFI and TLI with their confidence
// intervals, bundled into one post-fit result (the single C++/R surface). SRMR
// is the CRMR statistic rescaled to the vech denominator. The per-index
// diagnostics (bias traces, gradient variances, spectra) stay on the individual
// `OrdinalRmseaInference`/`OrdinalCrmrInference`/`OrdinalIncrementalFitInference`
// results. `conf_level`/`fixed_weight` record the options. Multi-group via the block-diagonal per-group Γ_x.
struct OrdinalMisspecFitMeasures {
  double conf_level = 0.90;
  bool fixed_weight = false;
  double rmsea = 0.0;            // bias-corrected profile RMSEA
  double rmsea_ci_lower = 0.0;
  double rmsea_ci_upper = 0.0;
  double rmsea_pvalue = 0.0;     // exact-fit mixture tail Pr(Σλⱼχ²₁ > N·F)
  double crmr = 0.0;             // sqrt(G/ncorr)
  double crmr_ci_lower = 0.0;
  double crmr_ci_upper = 0.0;
  double srmr = 0.0;             // crmr · sqrt(ncorr/vech)
  double srmr_ci_lower = 0.0;
  double srmr_ci_upper = 0.0;
  double crmr_pvalue = 0.0;      // shared CRMR/SRMR exact-fit mixture tail
  double cfi = 0.0;
  double cfi_ci_lower = 0.0;
  double cfi_ci_upper = 0.0;
  double tli = 0.0;
  double tli_ci_lower = 0.0;
  double tli_ci_upper = 0.0;
  double stat_user = 0.0;        // T_u = N·F
  double stat_baseline = 0.0;    // T_b
  int df_user = 0;
  int df_baseline = 0;
  std::vector<std::string> warnings;
};

fit_expected<void>
prepare_ordinal_delta_partable(spec::LatentStructure& pt,
                                const data::OrdinalStats& stats,
                                spec::Starts* starts = nullptr);

fit_expected<void>
prepare_ordinal_delta_partable(spec::LatentStructure& pt,
                                const data::OrdinalMoments& moments,
                                spec::Starts* starts = nullptr);

fit_expected<void>
prepare_ordinal_partable(spec::LatentStructure& pt,
                         const data::OrdinalStats& stats,
                         OrdinalParameterization parameterization,
                         spec::Starts* starts = nullptr);

fit_expected<void>
prepare_ordinal_partable(spec::LatentStructure& pt,
                         const data::OrdinalMoments& moments,
                         OrdinalParameterization parameterization,
                         spec::Starts* starts = nullptr);

fit_expected<void>
prepare_mixed_ordinal_delta_partable(spec::LatentStructure& pt,
                                      const data::MixedOrdinalStats& stats,
                                      spec::Starts* starts = nullptr);

fit_expected<void>
prepare_mixed_ordinal_partable(spec::LatentStructure& pt,
                                const data::MixedOrdinalStats& stats,
                                OrdinalParameterization parameterization,
                                spec::Starts* starts = nullptr);

post_expected<OrdinalRobustResult>
robust_ordinal(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::OrdinalStats& stats,
               const Estimates& est,
               OrdinalWeightKind weights,
               OrdinalParameterization parameterization =
                   OrdinalParameterization::Delta,
               robust::Information bread = robust::Information::Expected);

// Infinitesimal-jackknife (misspecification-robust, "regime = ij") covariance
// for an all-ordinal moment-quadratic fit. Observed-Hessian bread with an IJ
// meat Σ_i v_i v_iᵀ, v_i = W_b g_i + corr_i, that -- unlike the fixed-weight
// observed-bread sandwich -- carries the influence of the ESTIMATED weight
// Ŵ (the cov(Γ̂) term). DWLS carries the diagonal `diag(NACOV)^{-1}` channel;
// WLS carries the dense `NACOV^{-1}` channel. That term is leading-order under
// misspecification and identically zero under the null or for ULS (W = I,
// fixed); the channel-1 piece (v_i = W_b g_i) reproduces the observed-bread
// sandwich exactly. Requires `stats.moment_influence`; DWLS/WLS also require
// `stats.int_data`, complete for ordinary stats or missing-coded (`-1`) for
// pairwise-overlap observed stats, so the estimated-weight influence is not
// silently approximated.
post_expected<OrdinalRobustResult>
robust_ordinal_ij(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::OrdinalStats& stats,
                  const Estimates& est,
                  OrdinalWeightKind weights,
                  OrdinalParameterization parameterization =
                      OrdinalParameterization::Delta);

// Per-case one-step misspecification-robust ("complete-sandwich") parameter
// influences for an all-ordinal DWLS/WLS/ULS(MV) fit: the categorical analogue
// of `continuous_ls_casewise_influence_ij` and the casewise dual of
// `robust_ordinal_ij`. Builds the same per-case IJ blocks (with the
// estimated-weight `IF(Ŵ)` correction for DWLS/WLS) and observed bread, then
// returns the per-case rows via `casewise_influence_from_ij_blocks` rather than
// summing them into the SE meat. `Σ_i c_i c_iᵀ` reproduces `robust_ordinal_ij`'s
// vcov exactly; `influence_naive` drops the weight correction (the fixed-weight
// influence semfindr/Pek-MacCallum use). ULS has a fixed weight ⇒ no correction.
// Same `stats.moment_influence` / `stats.int_data` requirements as
// `robust_ordinal_ij`. Frontier.
post_expected<CasewiseInfluenceIJ>
ordinal_casewise_influence_ij(spec::LatentStructure pt,
                              const model::MatrixRep& rep,
                              const data::OrdinalStats& stats,
                              const Estimates& est,
                              OrdinalWeightKind weights,
                              OrdinalParameterization parameterization =
                                  OrdinalParameterization::Delta);

post_expected<OrdinalRobustResult>
robust_ordinal(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::OrdinalMoments& moments,
               data::OrdinalGammaCache& gamma_cache,
               const Estimates& est,
               data::OrdinalWeightPlan plan);

post_expected<OrdinalRobustResult>
robust_mixed_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::MixedOrdinalStats& stats,
                     const Estimates& est,
                     OrdinalWeightKind weights,
                     OrdinalParameterization parameterization =
                         OrdinalParameterization::Delta,
                     robust::Information bread = robust::Information::Expected);

// Mixed continuous/ordinal infinitesimal-jackknife covariance. ULS is a
// fixed-weight identity sandwich and only needs `stats.moment_influence`, so it
// also supports pairwise-overlap observed mixed stats. DWLS includes the
// diagonal estimated-weight influence and WLS includes the dense estimated-
// weight influence for ordinary complete-data ML/polyserial mixed stats, which
// requires `stats.moment_influence` and `stats.raw_data`.
post_expected<OrdinalRobustResult>
robust_mixed_ordinal_ij(spec::LatentStructure pt,
                        const model::MatrixRep& rep,
                        const data::MixedOrdinalStats& stats,
                        const Estimates& est,
                        OrdinalWeightKind weights,
                        OrdinalParameterization parameterization =
                            OrdinalParameterization::Delta);

post_expected<OrdinalRobustResult>
robust_mixed_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::MixedOrdinalMoments& moments,
                     data::OrdinalGammaCache& gamma_cache,
                     const Estimates& est,
                     data::OrdinalWeightPlan plan);

post_expected<robust::LRSatorra2000Result>
lr_test_satorra2000_ordinal(
    spec::LatentStructure pt_H1,
    const model::MatrixRep& rep_H1,
    const data::OrdinalStats& stats,
    const Estimates& est_H1,
    spec::LatentStructure pt_H0,
    const model::MatrixRep& rep_H0,
    const Estimates& est_H0,
    OrdinalWeightKind weights,
    double T_H0,
    double T_H1,
    int df_H0,
    int df_H1,
    robust::SatorraAMethod a_method = robust::SatorraAMethod::Exact,
    OrdinalParameterization parameterization = OrdinalParameterization::Delta);

post_expected<robust::LRSatorra2000Result>
lr_test_satorra2000_mixed_ordinal(
    spec::LatentStructure pt_H1,
    const model::MatrixRep& rep_H1,
    const data::MixedOrdinalStats& stats,
    const Estimates& est_H1,
    spec::LatentStructure pt_H0,
    const model::MatrixRep& rep_H0,
    const Estimates& est_H0,
    OrdinalWeightKind weights,
    double T_H0,
    double T_H1,
    int df_H0,
    int df_H1,
    robust::SatorraAMethod a_method = robust::SatorraAMethod::Exact,
    OrdinalParameterization parameterization = OrdinalParameterization::Delta);

post_expected<inference::ScoreTestTable>
modification_indices_ordinal(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::OrdinalStats& stats,
                             const Estimates& est,
                             OrdinalWeightKind weights,
                             OrdinalParameterization parameterization =
                                 OrdinalParameterization::Delta);

post_expected<inference::ScoreTestTable>
modification_indices_ordinal(spec::LatentStructure pt,
                             const model::MatrixRep& rep,
                             const data::OrdinalStats& stats,
                             const Estimates& est,
                             OrdinalWeightKind weights,
                             const inference::ModificationIndexOptions& options,
                             OrdinalParameterization parameterization =
                                 OrdinalParameterization::Delta);

post_expected<inference::ScoreTestTable>
score_tests_ordinal(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    const Estimates& est,
                    OrdinalWeightKind weights,
                    OrdinalParameterization parameterization =
                        OrdinalParameterization::Delta);

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::MixedOrdinalStats& stats,
                                   const Estimates& est,
                                   OrdinalWeightKind weights,
                                   OrdinalParameterization parameterization =
                                       OrdinalParameterization::Delta);

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::MixedOrdinalStats& stats,
                                   const Estimates& est,
                                   OrdinalWeightKind weights,
                                   const inference::ModificationIndexOptions& options,
                                   OrdinalParameterization parameterization =
                                       OrdinalParameterization::Delta);

post_expected<inference::ScoreTestTable>
score_tests_mixed_ordinal(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          const Estimates& est,
                          OrdinalWeightKind weights,
                          OrdinalParameterization parameterization =
                              OrdinalParameterization::Delta);

// ── Robust (generalized / SB-scaled) ordinal score tests ────────────────────
// Frontier surface (lavaan implements no robust score test): every candidate
// row carries the ordinary `mi` plus `mi_scaled = mi / c` with the
// per-direction scaling `c = gᵀB1g / gᵀA1g`, where A1 = Σ_b (n_b/N)·Δ_bᵀW_bΔ_b
// and B1 = Σ_b (n_b/N)·Δ_bᵀW_bΓ̂_bW_bΔ_b in the [thresholds ; associations]
// moment metric — W the estimation weight (ULS identity / DWLS diagonal /
// full WLS, the same blocks `robust_ordinal` uses) and Γ̂ = `stats.NACOV` (the
// polychoric/polyserial moment ACOV, which must be populated). Under WLS the
// meat collapses onto the bread (W = Γ̂⁻¹ ⇒ B1 = A1, c ≡ 1), the exact
// reduction-to-ordinary baseline. `mi` keeps the lavaan-matched row-type scale
// convention of the non-robust sweep (`mi_scaled` inherits it: c carries no
// moment-scale factor). Mixed stats support DWLS/WLS only, like the
// non-robust mixed sweep. Single group only (v1).
namespace frontier {

// Experimental all-ordinal second-stage weights over an already-built
// threshold + polychoric `OrdinalStats` object. This is intended for research
// comparisons: callers can build pairwise/listwise ordinal moments and Gamma
// once, then refit the SEM with several stage-2 weights without re-estimating
// thresholds, polychorics, or the observed Gamma.
enum class OrdinalStage2Weight {
  Uls,   // identity
  Dwls,  // diag(Gamma_observed)^-1
  Wls,   // Gamma_observed^-1 (ADF/full WLS)
  Nt,    // normal-theory/GLS-like association block
  Dls,   // ((1-a) Gamma_NT + a Gamma_observed)^-1
};

struct OrdinalStage2DlsOptions {
  double a = 0.5;
};

post_expected<std::vector<Eigen::MatrixXd>>
ordinal_stage2_weight_blocks(const data::OrdinalStats& stats,
                             OrdinalStage2Weight kind,
                             OrdinalStage2DlsOptions dls = {});

// Return a copy whose `W_wls` slot is replaced by the selected full stage-2
// weight. ULS/DWLS/WLS are copied through their existing materialized weights;
// NT/DLS are experimental full-weight paths consumed by the ordinary WLS
// fitting/sandwich machinery.
post_expected<data::OrdinalStats>
ordinal_stats_with_stage2_weight(const data::OrdinalStats& stats,
                                 OrdinalStage2Weight kind,
                                 OrdinalStage2DlsOptions dls = {});

// `estimated_weight = true` routes the per-direction robust scaling through the
// complete (Hall-Inoue infinitesimal-jackknife) sandwich, which carries the
// data-dependent-weight IF(Ŵ) meat term — beyond lavaan, which scales MI only by
// the global SB scalar. DWLS/WLS only (ULS has a fixed weight); requires the
// per-case `moment_influence` and `int_data` carried by the fit's OrdinalStats.
// Not yet implemented for mixed-ordinal (errors if requested there).
post_expected<inference::ScoreTestTable>
modification_indices_ordinal_robust(spec::LatentStructure pt,
                                    const model::MatrixRep& rep,
                                    const data::OrdinalStats& stats,
                                    const Estimates& est,
                                    OrdinalWeightKind weights,
                                    const inference::ModificationIndexOptions&
                                        options = {},
                                    OrdinalParameterization parameterization =
                                        OrdinalParameterization::Delta,
                                    bool estimated_weight = false);

post_expected<inference::ScoreTestTable>
score_tests_ordinal_robust(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::OrdinalStats& stats,
                           const Estimates& est,
                           OrdinalWeightKind weights,
                           OrdinalParameterization parameterization =
                               OrdinalParameterization::Delta,
                           bool estimated_weight = false);

post_expected<inference::ScoreTestTable>
modification_indices_mixed_ordinal_robust(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    const Estimates& est,
    OrdinalWeightKind weights,
    const inference::ModificationIndexOptions& options = {},
    OrdinalParameterization parameterization = OrdinalParameterization::Delta,
    bool estimated_weight = false);

post_expected<inference::ScoreTestTable>
score_tests_mixed_ordinal_robust(spec::LatentStructure pt,
                                 const model::MatrixRep& rep,
                                 const data::MixedOrdinalStats& stats,
                                 const Estimates& est,
                                 OrdinalWeightKind weights,
                                 OrdinalParameterization parameterization =
                                     OrdinalParameterization::Delta,
                                 bool estimated_weight = false);

}  // namespace frontier

// Start-value producers for ordinal LS. They run the partable preparation step
// internally, so the returned vector is sized for the *prepared* partable —
// exactly what the matching `fit_*_ordinal_bounded` rebuilds. Pass the result
// straight in.
fit_expected<Eigen::VectorXd>
ordinal_start_values(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalStats& stats,
                     spec::Starts starts = {});

fit_expected<Eigen::VectorXd>
ordinal_start_values(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalMoments& moments,
                     spec::Starts starts = {});

fit_expected<Eigen::VectorXd>
mixed_ordinal_start_values(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalStats& stats,
                           spec::Starts starts = {});

fit_expected<Eigen::VectorXd>
mixed_ordinal_start_values(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalMoments& moments,
                           spec::Starts starts = {});

// Ordinal DWLS / WLS fit. The model is fitted as a bounded least-squares
// problem in the thresholds + polychoric correlations; `backend` selects the
// optimizer (NLopt L-BFGS by default, or Ceres LM when MAGMAAN_WITH_CERES is
// enabled). `opts` tunes the optimizer; Ceres reads max_iter / ftol / gtol
// from it.
//
// `parameterization` selects the lavaan-compatible ordinal parameterization.
// `Delta` (the default) compares the implied latent-response covariances
// directly to the polychoric correlations; `Theta` fixes the latent-response
// residual variances to 1, lets the total variances float, and compares the
// *standardized* implied moments — Σ*ᵢⱼ/√(Σ*ᵢᵢΣ*ⱼⱼ) and thresholds rescaled by
// √Σ*ᵢᵢ. The two are reparameterizations of one model (same fmin / χ² / df).
fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalStats& stats,
                    Bounds bounds,
                    OrdinalWeightKind weights,
                    const Eigen::VectorXd& x0,
                    Backend backend = Backend::NloptLbfgs,
                    optim::OptimOptions opts = {},
                    OrdinalParameterization parameterization =
                        OrdinalParameterization::Delta);

// Cache-aware all-ordinal LS over moment metadata plus an explicit Gamma cache.
// `FitOnly` keeps ULS/DWLS cheap: ULS uses identity weights and DWLS asks only
// for diagonal Gamma. `FitPlusInference` may materialize the full cache pieces
// needed by a later cache-aware robust_ordinal() call.
fit_expected<Estimates>
fit_ordinal_bounded(spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const data::OrdinalMoments& moments,
                    data::OrdinalGammaCache* gamma_cache,
                    Bounds bounds,
                    data::OrdinalWeightPlan plan,
                    const Eigen::VectorXd& x0,
                    Backend backend = Backend::NloptLbfgs,
                    optim::OptimOptions opts = {});

// Golub-Pereyra profiled all-ordinal delta LS over the cache-aware threshold
// design. Thresholds are eliminated first through the affine model
// tau_b = c_b + H_b * gamma — covering free thresholds, equality-label merges
// (including cross-group threshold invariance), and threshold-only linear
// equality constraints — then conditionally linear covariance parameters are
// profiled by the SNLLS engine. The threshold normal equations are joint
// across groups with n_b/N block weights. ULS uses identity weights, DWLS
// uses diagonal Gamma weights, and WLS uses the Schur-complement profiled
// weight from the full Gamma cache.
fit_expected<Estimates>
fit_ordinal_snlls(spec::LatentStructure pt,
                  const model::MatrixRep& rep,
                  const data::OrdinalMoments& moments,
                  data::OrdinalGammaCache* gamma_cache,
                  data::OrdinalWeightPlan plan,
                  const Eigen::VectorXd& x0,
                  Backend backend = Backend::NloptLbfgs,
                  optim::OptimOptions opts = {});

// Golub-Pereyra profiled all-ordinal delta LS over the full threshold +
// correlation moment objective. Thresholds remain in the LS moment stack and
// enter the GP linear block with the other conditionally-linear parameters,
// so linear threshold constraints can be handled by the GP constraint basis.
fit_expected<Estimates>
fit_ordinal_snlls_full_thresholds(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const data::OrdinalMoments& moments,
                                  data::OrdinalGammaCache* gamma_cache,
                                  data::OrdinalWeightPlan plan,
                                  const Eigen::VectorXd& x0,
                                  Backend backend = Backend::NloptLbfgs,
                                  optim::OptimOptions opts = {});

fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalStats& stats,
                          Bounds bounds,
                          OrdinalWeightKind weights,
                          const Eigen::VectorXd& x0,
                          Backend backend = Backend::NloptLbfgs,
                          optim::OptimOptions opts = {},
                          OrdinalParameterization parameterization =
                              OrdinalParameterization::Delta);

fit_expected<Estimates>
fit_mixed_ordinal_bounded(spec::LatentStructure pt,
                          const model::MatrixRep& rep,
                          const data::MixedOrdinalMoments& moments,
                          data::OrdinalGammaCache* gamma_cache,
                          Bounds bounds,
                          data::OrdinalWeightPlan plan,
                          const Eigen::VectorXd& x0,
                          Backend backend = Backend::NloptLbfgs,
                          optim::OptimOptions opts = {});

// Golub-Pereyra profiled mixed continuous/ordinal LS over the full mixed
// threshold + continuous mean/variance + association moment objective. Under
// delta the thresholds, means, variances, and covariances are conditionally
// linear; under theta the standardized covariance moments leave only the
// thresholds in the Golub-Pereyra linear block. This entry point consumes the
// materialized mixed stats/weights; lazy mixed full-Gamma construction
// remains a later slice.
fit_expected<Estimates>
fit_mixed_ordinal_snlls_full_thresholds(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats,
    OrdinalWeightKind weights,
    const Eigen::VectorXd& x0,
    Backend backend = Backend::NloptLbfgs,
    optim::OptimOptions opts = {},
    OrdinalParameterization parameterization = OrdinalParameterization::Delta);

fit_expected<Estimates>
fit_mixed_ordinal_snlls_full_thresholds(
    spec::LatentStructure pt,
    const model::MatrixRep& rep,
    const data::MixedOrdinalMoments& moments,
    data::OrdinalGammaCache* gamma_cache,
    data::OrdinalWeightPlan plan,
    const Eigen::VectorXd& x0,
    Backend backend = Backend::NloptLbfgs,
    optim::OptimOptions opts = {});

post_expected<OrdinalFitMeasures>
fit_measures_ordinal(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::OrdinalStats& stats,
                     const Estimates& est,
                     OrdinalWeightKind weights,
                     OrdinalParameterization parameterization =
                         OrdinalParameterization::Delta);

post_expected<OrdinalCatmlDwlsRmsea>
catml_dwls_rmsea_ordinal(spec::LatentStructure pt,
                         const model::MatrixRep& rep,
                         const data::OrdinalStats& stats,
                         const Estimates& est,
                         OrdinalParameterization parameterization =
                             OrdinalParameterization::Delta);

// Estimated-weight (γ-channel-aware) misspecification inference for CRMR/SRMR.
// `estimated_weight=true` propagates the polychoric-weight sampling variability
// through the joint NACOV of (u, γ); `false` is the fixed-weight comparator.
// `srmr_denominator=true` reports the SRMR (vech) scaling instead of CRMR.
// Multi-group (criteria pool over groups; CRMR/SRMR requires a common variable
// count across groups). Requires `stats.moment_influence` and `stats.int_data`.
post_expected<OrdinalCrmrInference>
ordinal_crmr_misspec_inference(spec::LatentStructure pt,
                               const model::MatrixRep& rep,
                               const data::OrdinalStats& stats,
                               const Estimates& est,
                               OrdinalParameterization parameterization =
                                   OrdinalParameterization::Delta,
                               bool estimated_weight = true,
                               bool srmr_denominator = false,
                               double conf_level = 0.90,
                               double eig_tol = 1e-10);

// Mixed continuous/ordinal counterpart of `ordinal_crmr_misspec_inference`.
// The association residuals use the same standardization convention as
// `mixed_ordinal_srmr`: continuous-continuous and polyserial rows are divided by
// their observed standard-deviation scale, while ordinal-ordinal rows are
// already correlations. `srmr_denominator=false` reports the off-diagonal CRMR
// denominator; `true` reports the full-vech SRMR denominator.
post_expected<OrdinalCrmrInference>
mixed_ordinal_crmr_misspec_inference(spec::LatentStructure pt,
                                     const model::MatrixRep& rep,
                                     const data::MixedOrdinalStats& stats,
                                     const Estimates& est,
                                     OrdinalParameterization parameterization =
                                         OrdinalParameterization::Delta,
                                     bool estimated_weight = true,
                                     bool srmr_denominator = false,
                                     double conf_level = 0.90,
                                     double eig_tol = 1e-10);

// Estimated-weight (γ-channel-aware) misspecification confidence interval for
// RMSEA. Reuses `ordinal_dwls_profile_rmsea` for the Hessian/Γ_x/bias/spectrum
// and adds the envelope-score gradient variance for the interval.
// `estimated_weight=false` is the fixed-weight comparator. Multi-group via the block-diagonal per-group Γ_x.
post_expected<OrdinalRmseaInference>
ordinal_rmsea_misspec_inference(spec::LatentStructure pt,
                                const model::MatrixRep& rep,
                                const data::OrdinalStats& stats,
                                const Estimates& est,
                                OrdinalParameterization parameterization =
                                    OrdinalParameterization::Delta,
                                bool estimated_weight = true,
                                double conf_level = 0.90,
                                double eig_tol = 1e-10);

// Mixed continuous/ordinal counterpart of `ordinal_rmsea_misspec_inference`.
// Reuses `mixed_ordinal_dwls_profile_rmsea` for the profile Hessian, signed
// trace, spectrum, and joint NACOV of `(u, gamma)`, then adds the same envelope-
// score normal-theory CI for the DWLS discrepancy. This is the first mixed slice
// of the misspec fit-measure bundle: no mixed CRMR/SRMR or CFI/TLI convention is
// implied here.
post_expected<OrdinalRmseaInference>
mixed_ordinal_rmsea_misspec_inference(spec::LatentStructure pt,
                                      const model::MatrixRep& rep,
                                      const data::MixedOrdinalStats& stats,
                                      const Estimates& est,
                                      OrdinalParameterization parameterization =
                                          OrdinalParameterization::Delta,
                                      bool estimated_weight = true,
                                      double conf_level = 0.90,
                                      double eig_tol = 1e-10);

// Estimated-weight (γ-channel-aware) misspecification inference for the
// incremental fit indices CFI and TLI. Runs the user model through
// `ordinal_dwls_profile_rmsea` and the analytic independence baseline (free
// thresholds, zero correlations) through `weighted_moment_profile_rmsea_-
// estimated_weight` with the SAME Γ_x, then forms the ratio delta-method of the
// two noncentralities (see OrdinalIncrementalFitInference).
// `estimated_weight=false` is the fixed-weight comparator. Multi-group via the block-diagonal per-group Γ_x.
post_expected<OrdinalIncrementalFitInference>
ordinal_cfi_tli_misspec_inference(spec::LatentStructure pt,
                                  const model::MatrixRep& rep,
                                  const data::OrdinalStats& stats,
                                  const Estimates& est,
                                  OrdinalParameterization parameterization =
                                      OrdinalParameterization::Delta,
                                  bool estimated_weight = true,
                                  double conf_level = 0.90,
                                  double eig_tol = 1e-10);

// Consolidated estimated-weight misspecification fit table (RMSEA + CRMR/SRMR +
// CFI/TLI with CIs) for an all-ordinal DWLS fit. Delegates to the per-index
// inference entry points (each computing the shared profile/Γ_x internally), so
// the bundled values match them exactly; SRMR is the CRMR result rescaled to the
// vech denominator. `estimated_weight=false` is the fixed-weight comparator.
// Multi-group via the block-diagonal per-group Γ_x.
post_expected<OrdinalMisspecFitMeasures>
ordinal_fit_measures_misspec_inference(spec::LatentStructure pt,
                                       const model::MatrixRep& rep,
                                       const data::OrdinalStats& stats,
                                       const Estimates& est,
                                       OrdinalParameterization parameterization =
                                           OrdinalParameterization::Delta,
                                       bool estimated_weight = true,
                                       double conf_level = 0.90,
                                       double eig_tol = 1e-10);

// Fixed-misspecification estimated-weight (categorical DWLS) profile-RMSEA for
// an all-ordinal fit. The first-stage object is the EXTENDED moment vector
// x = (u, γ) with u = thresholds+polychorics and γ = diag(Γ̂_u) the diagonal
// NACOV that DWLS inverts. The joint NACOV Γ_x is the cross-product of the
// stacked per-case influence rows [g_i | IF_i(γ)]; IF_i(γ) combines the
// data-direct Γ̂ influence and the κ-movement channel exactly as
// `robust_ordinal_ij` builds the estimated-weight correction. The result routes
// through `weighted_moment_profile_rmsea_estimated_weight`, so the residual-
// driven γ channel reshapes the reference law under fixed misspecification and
// is dormant at exact fit. Requires `stats.moment_influence` and
// `stats.int_data` (complete, or `-1`-coded for pairwise-overlap stats).
post_expected<WeightedProfileRMSEAResult>
ordinal_dwls_profile_rmsea(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::OrdinalStats& stats,
                           const Estimates& est,
                           OrdinalParameterization parameterization =
                               OrdinalParameterization::Delta,
                           double eig_tol = 1e-10);

// Nested estimated-weight (categorical DWLS) profile-LRT for two all-ordinal
// models fit to the SAME data (one `stats`). `*_H1` is the less restricted
// model, `*_H0` the more restricted. Both share the data-only joint NACOV Γ_x,
// so the nested profile contrast Q_H0 − Q_H1 is compared over a common
// first-stage space; the result reports nominal `df_diff` separately from the
// positive profile `spectrum_size`.
post_expected<WeightedProfileLRTResult>
ordinal_dwls_profile_lrt(spec::LatentStructure pt_H1,
                         const model::MatrixRep& rep_H1,
                         const data::OrdinalStats& stats,
                         const Estimates& est_H1,
                         spec::LatentStructure pt_H0,
                         const model::MatrixRep& rep_H0,
                         const Estimates& est_H0,
                         OrdinalParameterization parameterization =
                             OrdinalParameterization::Delta,
                         double eig_tol = 1e-10);

// Fixed-misspecification estimated-weight profile-RMSEA for a mixed
// continuous/ordinal DWLS fit. The first-stage object is the extended mixed
// moment vector x = (u, gamma), where gamma = diag(Gamma_hat_u). The joint
// NACOV Gamma_x is the cross-product of [g_i | IF_i(gamma)], using the same
// mixed Gamma diagonal influence channels as `robust_mixed_ordinal_ij`.
post_expected<WeightedProfileRMSEAResult>
mixed_ordinal_dwls_profile_rmsea(spec::LatentStructure pt,
                                 const model::MatrixRep& rep,
                                 const data::MixedOrdinalStats& stats,
                                 const Estimates& est,
                                 OrdinalParameterization parameterization =
                                     OrdinalParameterization::Delta,
                                 double eig_tol = 1e-10);

// Nested estimated-weight profile-LRT for two mixed continuous/ordinal DWLS
// models fit to the SAME mixed stats.
post_expected<WeightedProfileLRTResult>
mixed_ordinal_dwls_profile_lrt(spec::LatentStructure pt_H1,
                               const model::MatrixRep& rep_H1,
                               const data::MixedOrdinalStats& stats,
                               const Estimates& est_H1,
                               spec::LatentStructure pt_H0,
                               const model::MatrixRep& rep_H0,
                               const Estimates& est_H0,
                               OrdinalParameterization parameterization =
                                   OrdinalParameterization::Delta,
                               double eig_tol = 1e-10);

post_expected<OrdinalFitMeasures>
fit_measures_mixed_ordinal(spec::LatentStructure pt,
                           const model::MatrixRep& rep,
                           const data::MixedOrdinalStats& stats,
                           const Estimates& est,
                           OrdinalWeightKind weights,
                           OrdinalParameterization parameterization =
                               OrdinalParameterization::Delta);

}  // namespace magmaan::estimate
