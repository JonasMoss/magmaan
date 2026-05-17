#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/robust/satorra2000.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::robust {

using estimate::EqConstraints;

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
//                   via `imhof_upper`.
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

  double           p_mixture;      // Imhof tail of T_diff

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

}  // namespace magmaan::robust
