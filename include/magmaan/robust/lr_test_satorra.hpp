#pragma once

#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/robust/satorra2000.hpp"
#include "magmaan/robust/weighted_chisq.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::robust {

using estimate::EqConstraints;

enum class SatorraAMethod {
  Exact,
  Delta
};

enum class SatorraMomentConvention {
  Magmaan,
  Lavaan
};

struct Satorra2000Options {
  SatorraAMethod   a_method    = SatorraAMethod::Exact;
  GammaSource      gamma       = GammaSource::Empirical;
  GammaComputation computation = GammaComputation::Streaming;
};

struct LRSatorraBentlerDiffResult {
  double T_diff   = 0.0;
  int    df_diff  = 0;
  double scale_c  = 0.0;
  double T_scaled = 0.0;
  double p_value  = 0.0;

  double c_H0     = 0.0;
  double c_H1     = 0.0;
  double c_hybrid = std::numeric_limits<double>::quiet_NaN();

  std::vector<std::string> warnings;
};

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2001(double T_H0, double T_H1,
                            int df_H0, int df_H1,
                            double c_H0, double c_H1);

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2010(double T_H0, double T_H1,
                            int df_H0, int df_H1,
                            double c_H0, double c_M10);

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2001_from_data(
    const spec::LatentStructure& pt_H1,
    const model::MatrixRep&      rep_H1,
    const Eigen::VectorXd&       theta_H1_full,
    const spec::LatentStructure& pt_H0,
    const model::MatrixRep&      rep_H0,
    const Eigen::VectorXd&       theta_H0_full,
    const data::RawData&         raw,
    double                       T_H0,
    double                       T_H1,
    int                          df_H0,
    int                          df_H1,
    GammaSource                  gamma = GammaSource::Empirical);

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2010_from_data(
    const spec::LatentStructure& pt_H1,
    const model::MatrixRep&      rep_H1,
    const Eigen::VectorXd&       theta_H0_full,
    const spec::LatentStructure& pt_H0,
    const model::MatrixRep&      rep_H0,
    const Eigen::VectorXd&       theta_H0_for_H0,
    const data::RawData&         raw,
    double                       T_H0,
    double                       T_H1,
    int                          df_H0,
    int                          df_H1,
    GammaSource                  gamma = GammaSource::Empirical);

// ── Scalar SB2001 / SB2010 difference tests for missing data (FIML / ML2S) ──
// These are the two-constant scalar statistics (NOT the FMG-able spectrum):
// c_d = (df0·c0 − df1·c1)/m for 2001 (c0, c1 the single-model GOF scalings at
// their own MLEs), and c_d = (df0·c0 − df1·c_M10)/m for 2010 (c_M10 = the H1
// structure's GOF scaling evaluated at the H0 estimates, the positivity fix).
// They mirror the complete-data `_from_data` engines and are intended as
// baseline comparison columns (the lavaan SB2001 statistic and its positivity
// fix). 2010 requires same-parameter nesting (theta_H0 injectable into H1).
post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2001_fiml_from_data(
    const spec::LatentStructure& pt_H1, const model::MatrixRep& rep_H1,
    const Eigen::VectorXd& theta_H1_full,
    const spec::LatentStructure& pt_H0, const model::MatrixRep& rep_H0,
    const Eigen::VectorXd& theta_H0_full,
    const data::RawData& raw,
    double T_H0, double T_H1, int df_H0, int df_H1, double h_step = 1e-4);

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2001_ml2s_from_data(
    const spec::LatentStructure& pt_H1, const model::MatrixRep& rep_H1,
    const Eigen::VectorXd& theta_H1_full,
    const spec::LatentStructure& pt_H0, const model::MatrixRep& rep_H0,
    const Eigen::VectorXd& theta_H0_full,
    const data::RawData& raw,
    double T_H0, double T_H1, int df_H0, int df_H1, double h_step = 1e-4);

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2010_fiml_from_data(
    const spec::LatentStructure& pt_H1, const model::MatrixRep& rep_H1,
    const Eigen::VectorXd& theta_H0_full,
    const spec::LatentStructure& pt_H0, const model::MatrixRep& rep_H0,
    const Eigen::VectorXd& theta_H0_for_H0,
    const data::RawData& raw,
    double T_H0, double T_H1, int df_H0, int df_H1, double h_step = 1e-4);

post_expected<LRSatorraBentlerDiffResult>
lr_test_satorra_bentler2010_ml2s_from_data(
    const spec::LatentStructure& pt_H1, const model::MatrixRep& rep_H1,
    const Eigen::VectorXd& theta_H0_full,
    const spec::LatentStructure& pt_H0, const model::MatrixRep& rep_H0,
    const Eigen::VectorXd& theta_H0_for_H0,
    const data::RawData& raw,
    double T_H0, double T_H1, int df_H0, int df_H1, double h_step = 1e-4);

// ============================================================================
// Satorra-2000 nested likelihood-ratio test for two SEM fits H1 ⊃ H0.
//
// Three reported p-values, all computed from the m generalised eigenvalues
// `λ = eigvals(S, C)` of the m × m reduced problem (m = df_H0 − df_H1):
//
//   • p_scaled    — Satorra-Bentler-style mean correction.
//                   T_scaled = T_diff / ĉ ,  ĉ = (Σ λ) / m ,
//                   p_scaled = Pr( χ²(m) > T_scaled ).
//   • p_adjusted  — Yuan-Bentler / Asparouhov-Muthén mean-and-variance
//                   adjustment.  d̂₀ = (Σ λ)² / (Σ λ²) (real-valued df) and
//                   T_adjusted = T_diff · d̂₀ / (Σ λ).
//                   p_adjusted = Pr( χ²(d̂₀) > T_adjusted ).
//   • p_mixture   — exact tail of the asymptotic Σⱼ λⱼ χ²₁ⱼ distribution
//                   via `weighted_chisq_upper`.
//   • p_unscaled is also reported for reference (the naïve χ²(m) p-value of
//     T_diff with no λ correction).
// ============================================================================

struct LRSatorra2000Result {
  double           T_diff;
  int              df_diff;        // = m
  double           p_unscaled;

  Eigen::VectorXd  eigenvalues;    // length m, ascending

  double           scale_c;        // ĉ = Σλ/m
  double           T_scaled;
  double           p_scaled;

  double           adjust_d0;      // d̂₀ = (Σλ)² / Σλ²  (real)
  double           T_adjusted;
  double           p_adjusted;

  ScaledShiftedResult scaled_shifted;
  double              p_scaled_shifted;

  double           p_mixture;      // weighted-χ² tail of T_diff

  std::vector<std::string> warnings;
};

// Low-level entry: caller has already run `compute_satorra2000` and knows
// T_diff (= χ²_H0 − χ²_H1) and df_diff (= m).  Wraps eigvals → p-values.
post_expected<LRSatorra2000Result>
lr_test_satorra2000(double                  T_diff,
                    const SatorraDiffResult& sd);

// High-level entry: takes the inputs for the H1 fit plus per-group raw data
// and the two K matrices, builds Pi_alpha / Σ̂_g / SatorraGroup, derives
// A_alpha from (K_H1, K_H0), and runs the full pipeline.
//
// `T_H0`, `T_H1`, `df_H0`, `df_H1` come from the caller's two fits — for ML
// these are `N · F_ML(θ̂)` at each model, with `df` adjusted for the
// constraint rank.
//
// `weight_per_group` is `n_g / N_total` (matches the ML pooled objective
// `F = Σ_g f_g · F_g`); passing this in (rather than recomputing from n_per_group)
// lets the caller match the exact convention used in the χ² statistic.
post_expected<LRSatorra2000Result>
lr_test_satorra2000_from_data(
    const spec::LatentStructure&     pt,
    const model::MatrixRep&              rep,
    const Eigen::VectorXd&               theta_H1_full,   // size npar
    const EqConstraints&                 K_H1,
    const EqConstraints&                 K_H0,
    const std::vector<Eigen::MatrixXd>&  X_per_group,
    const std::vector<Eigen::VectorXd>&  mean_per_group,
    const std::vector<std::int32_t>&     n_per_group,
    const std::vector<double>&           weight_per_group,
    double                               T_H0,
    double                               T_H1,
    int                                  df_H0,
    int                                  df_H1,
    GammaSource                          gamma = GammaSource::Empirical);

post_expected<LRSatorra2000Result>
lr_test_satorra2000_from_data(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&              rep_H1,
    const Eigen::VectorXd&               theta_H1_full,
    const EqConstraints&                 K_H1,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&              rep_H0,
    const Eigen::VectorXd&               theta_H0_full,
    const EqConstraints&                 K_H0,
    const std::vector<Eigen::MatrixXd>&  X_per_group,
    const std::vector<Eigen::VectorXd>&  mean_per_group,
    const std::vector<std::int32_t>&     n_per_group,
    const std::vector<double>&           weight_per_group,
    double                               T_H0,
    double                               T_H1,
    int                                  df_H0,
    int                                  df_H1,
    Satorra2000Options                   options);

post_expected<SatorraDiffResult>
compute_fiml_satorra2000(
    const Eigen::Ref<const Eigen::MatrixXd>& Delta1_alpha,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& Gamma,
    const Eigen::Ref<const Eigen::MatrixXd>& A_alpha);

post_expected<SatorraDiffResult>
compute_satorra2000_from_sandwich(
    const Eigen::Ref<const Eigen::MatrixXd>& A1,
    const Eigen::Ref<const Eigen::MatrixXd>& B1,
    const Eigen::Ref<const Eigen::MatrixXd>& A_alpha);

post_expected<LRSatorra2000Result>
lr_test_satorra2000_fiml_from_data(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&          rep_H1,
    const Eigen::VectorXd&           theta_H1_full,
    const EqConstraints&             K_H1,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&          rep_H0,
    const Eigen::VectorXd&           theta_H0_full,
    const EqConstraints&             K_H0,
    const data::RawData&             raw,
    double                           T_H0,
    double                           T_H1,
    int                              df_H0,
    int                              df_H1,
    GammaSource                      gamma = GammaSource::Empirical,
    SatorraAMethod                   a_method = SatorraAMethod::Exact,
    double                           h_step = 1e-4,
    const estimate::fiml::SaturatedMoments* sm_precomputed = nullptr,
    SatorraMomentConvention          convention = SatorraMomentConvention::Magmaan);

post_expected<LRSatorra2000Result>
lr_test_satorra2000_ml2s_from_data(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&          rep_H1,
    const Eigen::VectorXd&           theta_H1_full,
    const EqConstraints&             K_H1,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&          rep_H0,
    const Eigen::VectorXd&           theta_H0_full,
    const EqConstraints&             K_H0,
    const data::RawData&             raw,
    double                           T_H0,
    double                           T_H1,
    int                              df_H0,
    int                              df_H1,
    GammaSource                      gamma = GammaSource::Empirical,
    SatorraAMethod                   a_method = SatorraAMethod::Exact,
    double                           h_step = 1e-4,
    const estimate::fiml::SaturatedMoments* sm_precomputed = nullptr,
    estimate::fiml::TwoStageWeight    kind = estimate::fiml::TwoStageWeight::Nt,
    estimate::fiml::TwoStageDlsOptions dls = {},
    SatorraMomentConvention          convention = SatorraMomentConvention::Magmaan);

// ============================================================================
// Satorra-Bentler "method 2001" difference-spectrum nested tests for missing
// data: the U_D = U0 − U1 estimator (vs the restriction-map "method 2000"
// above). Each model's residual projector is built against the common saturated
// eta-space meat side, differenced, and the top df_H0 − df_H1 eigenvalues of
// (U0 − U1)·Γ are read out through the same LRSatorra2000Result family
// (SB / mean-var / scaled-shifted / mixture); the eigenvalues also feed the
// FMG/pEBA tail transforms. No restriction matrix or EqConstraints are needed:
// the spectrum comes from the two single-model fits.
// (Satorra & Bentler 2001 p.510; semTests::ugamma_nested(m0, m1, "2001").)
//
// FIML: model projectors use observed FIML information as bread, with
// V = saturated information (sm.H) and Γ = saturated ACOV (sm.acov) on the meat
// side. ML2S: V = two-stage NT weight, Γ = two-stage moment ACOV.
post_expected<LRSatorra2000Result>
lr_test_satorra2001_fiml_from_data(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&          rep_H1,
    const Eigen::VectorXd&           theta_H1_full,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&          rep_H0,
    const Eigen::VectorXd&           theta_H0_full,
    const data::RawData&             raw,
    double                           T_H0,
    double                           T_H1,
    int                              df_H0,
    int                              df_H1,
    double                           h_step = 1e-4,
    const estimate::fiml::SaturatedMoments* sm_precomputed = nullptr);

post_expected<LRSatorra2000Result>
lr_test_satorra2001_ml2s_from_data(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&          rep_H1,
    const Eigen::VectorXd&           theta_H1_full,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&          rep_H0,
    const Eigen::VectorXd&           theta_H0_full,
    const data::RawData&             raw,
    double                           T_H0,
    double                           T_H1,
    int                              df_H0,
    int                              df_H1,
    double                           h_step = 1e-4,
    const estimate::fiml::SaturatedMoments* sm_precomputed = nullptr,
    estimate::fiml::TwoStageWeight    kind = estimate::fiml::TwoStageWeight::Nt,
    estimate::fiml::TwoStageDlsOptions dls = {});

post_expected<LRSatorra2000Result>
lr_test_satorra2000_fiml_from_data(
    const spec::LatentStructure&     pt_H1,
    const model::MatrixRep&          rep_H1,
    const Eigen::VectorXd&           theta_H1_full,
    const EqConstraints&             K_H1,
    const spec::LatentStructure&     pt_H0,
    const model::MatrixRep&          rep_H0,
    const Eigen::VectorXd&           theta_H0_full,
    const EqConstraints&             K_H0,
    const data::RawData&             raw,
    double                           T_H0,
    double                           T_H1,
    int                              df_H0,
    int                              df_H1,
    const estimate::fiml::FIMLPack&  pack,
    const estimate::fiml::FIMLH1&    h1,
    GammaSource                      gamma = GammaSource::Empirical,
    SatorraAMethod                   a_method = SatorraAMethod::Exact,
    double                           h_step = 1e-4,
    SatorraMomentConvention          convention = SatorraMomentConvention::Magmaan);

}  // namespace magmaan::robust
