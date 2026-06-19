#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate {

using data::RawData;
using data::SampleStats;
using robust::MeanVarAdjustedResult;
using robust::SatorraBentlerResult;
using robust::ScaledShiftedResult;

struct WeightedMomentBlock {
  Eigen::MatrixXd jacobian;  // model moments wrt full free theta
  Eigen::MatrixXd weight;    // estimator weight for this moment block
  Eigen::MatrixXd gamma;     // NACOV/meat for this moment block
  std::int64_t n_obs = 0;
};

struct WeightedRobustResult {
  Eigen::MatrixXd vcov;
  Eigen::VectorXd se;
  Eigen::VectorXd eigvals;
  double chisq_standard = 0.0;
  int df = 0;
  robust::SatorraBentlerResult satorra_bentler;
  robust::MeanVarAdjustedResult mean_var_adjusted;
  robust::ScaledShiftedResult scaled_shifted;
};

struct WeightedMomentIJBlock {
  Eigen::MatrixXd jacobian;          // model moments wrt full free theta
  Eigen::MatrixXd weight;            // fixed estimator weight W_b
  Eigen::MatrixXd moment_influence;  // n_b × m_b rows g_i
  Eigen::MatrixXd weight_correction; // optional n_b × m_b rows from IF(W_hat)
  std::int64_t n_obs = 0;
};

// Misspecification-robust ("observed-Hessian") bread for a moment-quadratic
// fit, in the K-reduced parameter space. Central-differences the per-unit
// moment-LS gradient
//   g(θ) = Σ_b (n_b/N)·Δ_b(θ)ᵀ W_b d_b(θ),   d_b(θ) = σ_b(θ) − s_b
// supplied by `grad_at` (full free-θ, length q = theta_hat.size()), forms the
// θ-space Hessian by finite difference, then reduces it to H_α = Kᵀ·H_θ·K
// (n_alpha × n_alpha) so it can be passed to `robust_weighted_moments` as
// `bread_override`. It reduces to the Gauss-Newton bread Δ'WΔ as the residual
// → 0 (correct specification). `h_rel` sets the relative central-difference
// step h_k = h_rel·max(1, |θ_k|).
post_expected<Eigen::MatrixXd>
observed_moment_bread_fd(
    const std::function<post_expected<Eigen::VectorXd>(const Eigen::VectorXd&)>&
        grad_at,
    const Eigen::VectorXd& theta_hat,
    const Eigen::MatrixXd& K,
    double h_rel = 1e-4);

// `bread_override`, when set, replaces the Gauss-Newton bread A = D̃ᵀWD̃ with
// the supplied n_alpha × n_alpha matrix (e.g. the observed-Hessian bread from
// `observed_moment_bread_fd`) in both the sandwich vcov and the U-factor. The
// meat (Γ/NACOV) is unchanged. Default (nullopt) ⇒ the existing expected /
// Gauss-Newton path, byte-for-byte.
post_expected<WeightedRobustResult>
robust_weighted_moments(const std::vector<WeightedMomentBlock>& blocks,
                        const Eigen::MatrixXd& K,
                        double fmin,
                        const std::optional<Eigen::MatrixXd>& bread_override =
                            std::nullopt);

// Infinitesimal-jackknife parameter covariance for a moment-quadratic fit with
// observed-Hessian bread. Each block contributes casewise rows
//   v_i = g_i W_b + correction_i,
// where `correction_i` carries the leading influence of an estimated weight
// (empty for fixed weights). The function returns vcov/se/chisq/df; scaled-test
// eigenvalues are intentionally left empty because estimated-weight corrections
// require estimator-specific test-statistic work.
post_expected<WeightedRobustResult>
robust_weighted_moment_ij(const std::vector<WeightedMomentIJBlock>& blocks,
                          const Eigen::MatrixXd& K,
                          double fmin,
                          const Eigen::MatrixXd& observed_bread);

// Parameter-space sandwich {A1, B1} in the moment metric, the LS counterpart
// of `robust::param_space_sandwich`: per-unit bread A1 = Σ_b (n_b/N)·Δ_bᵀW_bΔ_b
// and meat B1 = Σ_b (n_b/N)·Δ_bᵀW_bΓ̂_bW_bΔ_b, with W the ESTIMATION weight
// (not Γ_NT⁻¹). Always full θ-space (K_con empty): the robust score-test caller
// projects equality constraints itself. No inversion is performed, so a
// singular A1 is fine. The per-direction robust scaling c = gᵀB1g / gᵀA1g is
// NOT invariant to the scale of W (B1 carries W twice); callers must pass the
// same W the score/information evaluation used.
post_expected<robust::ParamSpaceSandwich>
weighted_param_space_sandwich(const std::vector<WeightedMomentBlock>& blocks);

// Moment-metric sandwich for a continuous moment-quadratic (ULS/GLS/WLS/DWLS)
// fit, evaluated at `est.theta` (which may belong to an augmented partable —
// the robust modification-index path calls this at the freed-candidate null
// point). `weight` follows the `robust_continuous_ls` convention (empty ⇒ ULS
// identity). The three overloads differ in the Γ̂ source: per-block supplied,
// empirical from complete-data raw, or model-implied Γ_NT — blockdiag(M_b,
// gamma_nt(M_b)) with M_b = Σ̂_b(θ) (Structured) or S_b (Unstructured). The
// Γ_NT meat with the GLS weight reproduces B1 = A1 (c ≡ 1) exactly.
post_expected<robust::ParamSpaceSandwich>
continuous_ls_param_space_sandwich(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::SampleStats& samp,
                                   const Estimates& est,
                                   const gmm::Weight& weight,
                                   const std::vector<Eigen::MatrixXd>& gamma);

post_expected<robust::ParamSpaceSandwich>
continuous_ls_param_space_sandwich(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::SampleStats& samp,
                                   const Estimates& est,
                                   const gmm::Weight& weight,
                                   const data::RawData& raw);

post_expected<robust::ParamSpaceSandwich>
continuous_ls_param_space_sandwich(spec::LatentStructure pt,
                                   const model::MatrixRep& rep,
                                   const data::SampleStats& samp,
                                   const Estimates& est,
                                   const gmm::Weight& weight,
                                   robust::WeightMoments nt_moments);

// Moment-quadratic least-squares objective F(θ) = ½·r̃(θ)ᵀr̃(θ). The `weight`
// is the only thing that distinguishes the estimators: empty ⇒ ULS; a
// normal-theory weight (`gmm::normal_theory_weight`) ⇒ GLS; a caller-supplied
// weight ⇒ WLS / DWLS.
post_expected<double>
evaluate_ls_objective(spec::LatentStructure pt,
                      const model::MatrixRep& rep,
                      const data::SampleStats& samp,
                      const Eigen::VectorXd& theta,
                      const gmm::Weight& weight);

// Continuous-LS reference χ². An empty `weight` ⇒ ULS, reported via Browne's
// residual-based normal-theory statistic; a non-empty `weight` ⇒ GLS / WLS,
// reported as 2N·fmin.
post_expected<double>
continuous_ls_chisq(data::SampleStats samp,
                    spec::LatentStructure pt,
                    const model::MatrixRep& rep,
                    const Estimates& est,
                    const gmm::Weight& weight);

// Non-robust (information-inverse) standard errors for a continuous
// moment-quadratic fit: the npar × npar LS information `Σ_b n_b·Δ_bᵀ W_b Δ_b`,
// with `Δ_b` the model-moment Jacobian and `W_b` the estimator weight (empty
// ⇒ ULS identity). Its inverse — through `inference::vcov`, which folds in any
// equality constraints — is the `se = "standard"` covariance, the non-sandwich
// counterpart to `robust_continuous_ls`.
post_expected<Eigen::MatrixXd>
ls_information(spec::LatentStructure pt,
               const model::MatrixRep& rep,
               const data::SampleStats& samp,
               const Estimates& est,
               const gmm::Weight& weight);

// Robust (sandwich) inference for a continuous moment-quadratic fit. `weight`
// follows the same convention as above (empty ⇒ ULS identity weight). `gamma`
// supplies the per-block moment NACOV directly, or `raw` supplies the raw data
// from which it is estimated.
post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     const gmm::Weight& weight,
                     const std::vector<Eigen::MatrixXd>& gamma,
                     robust::Information bread = robust::Information::Expected);

post_expected<WeightedRobustResult>
robust_continuous_ls(spec::LatentStructure pt,
                     const model::MatrixRep& rep,
                     const data::SampleStats& samp,
                     const Estimates& est,
                     const gmm::Weight& weight,
                     const data::RawData& raw,
                     robust::Information bread = robust::Information::Expected);

// Complete-data infinitesimal-jackknife covariance for continuous LS with a
// fixed second-stage weight. Empty `weight` is ULS; non-empty weights are treated
// as caller-fixed. Estimated-weight corrections for GLS/WLS/DLS are separate
// Hall-Inoue adapters and are not included here.
post_expected<WeightedRobustResult>
robust_continuous_ls_fixed_weight_ij(spec::LatentStructure pt,
                                     const model::MatrixRep& rep,
                                     const data::SampleStats& samp,
                                     const Estimates& est,
                                     const gmm::Weight& weight,
                                     const data::RawData& raw);

}  // namespace magmaan::estimate
