#pragma once

#include <cmath>
#include <limits>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/fit.hpp"          // Estimates
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/fcsem_evaluator.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::inference {

using data::RawData;
using data::SampleStats;
using estimate::Estimates;

// =============================================================================
// Information / vcov / SE — orthogonal primitives, chained at the call site.
//
//   info = information_<method>(pt, rep, samp, est)   one of three methods
//   vcov = vcov(info, pt)                              inverts + constraint K
//   se   = se(vcov)                                    sqrt(diag(.))
//
// chi² and df are *not* derived from the information matrix and don't live
// here — see `chi2_stat` / `df_stat` further down.
//
// `pt` is taken by value in each `information_*` because we resolve fixed.x
// `fixed_value`s from `samp` internally — symmetric with `fit()`. The caller's
// LatentStructure is untouched.
// =============================================================================

// Expected-information matrix for the ML discrepancy:
//
//   I[a, b] = Σ_blocks (n_b/2) · [ tr(Σ_b⁻¹ ∂Σ_b/∂θ_a Σ_b⁻¹ ∂Σ_b/∂θ_b)
//                                   + 2 · ν_a' Σ_b⁻¹ ν_b ]
//
// Mean-structure contribution is the ν' Σ⁻¹ ν term; empty for covariance-only
// models.
post_expected<Eigen::MatrixXd>
information_expected(spec::LatentStructure       pt,
                     const model::MatrixRep&         rep,
                     const SampleStats&              samp,
                     const Estimates&                est);

// Native FC-SEM counterpart. Uses the sample-backed FcSemEvaluator and its
// numerical covariance Jacobian; covariance-only in the current tranche.
post_expected<Eigen::MatrixXd>
information_expected_fcsem(spec::LatentStructure       pt,
                           const SampleStats&          samp,
                           const Estimates&            est,
                           double                      rel_step = 1e-6);

// Observed-information matrix via central-difference Hessian of the analytic
// ML gradient.
//
//   H[:, k] ≈ (∇F_ML(θ̂ + h e_k) − ∇F_ML(θ̂ − h e_k)) / (2h)
//
// then symmetrized; observed info = (N/2) · H. `h_step` defaults to 1e-4 —
// keeps central-difference truncation and roundoff balanced at v0 model sizes
// (≤ ~30 free params, p ≤ 10).
post_expected<Eigen::MatrixXd>
information_observed_fd(spec::LatentStructure       pt,
                        const model::MatrixRep&         rep,
                        const SampleStats&              samp,
                        const Estimates&                est,
                        double                          h_step = 1e-4);

// Observed-information matrix via the closed-form ML Hessian:
//
//   H = H1 + H2,
//   H1[a,b] = -tr(Σ⁻¹ M_b Σ⁻¹ M_a) + 2 · tr(Σ⁻¹ M_b · Σ⁻¹ S Σ⁻¹ · M_a)
//   H2[a,b] = tr(G · ∂²Σ/∂θ_a ∂θ_b),  G = Σ⁻¹ − Σ⁻¹ S Σ⁻¹
//
// ∂²Σ/∂θ² derived case-by-case on (mat_a, mat_b), including the reduced-form
// Λ/Ψ/Β cross-terms. Mean structures add the analytic Hessian terms for
// μ = ν + Λ(I−B)⁻¹α, including Λ-α, Λ-B, α-B, and B-B interactions.
post_expected<Eigen::MatrixXd>
information_observed_analytic(spec::LatentStructure       pt,
                              const model::MatrixRep&         rep,
                              const SampleStats&              samp,
                              const Estimates&                est);

// Cross-products information matrix (parameter-level outer product of
// casewise ML scores):
//
//   I_XP = Σ_i s_i s_iᵀ ,    s_i = ∂ℓ_i/∂θ |_{θ̂}
//
// Mplus calls the SE built from this method "MLF". Computed as
// `(Z_c · WΔ)ᵀ · (Z_c · WΔ)` where Z_c is the block-stacked matrix of
// centred per-case moment contributions (μ-rows on top of σ-vech rows when
// the model has mean structure — the G3b layout shared with
// `robust::casewise_contributions`) and WΔ is the block-stacked
// score-weight Jacobian W_b · Δ_b = Γ_NT(Σ̂_b)⁻¹ · ∂vech(Σ_b)/∂θ, plus the
// μ-segment Σ̂_b⁻¹ · ∂μ_b/∂θ when means are modelled. Per-block scaling
// (n_b/N) falls out of stacking — each row of Z_c contributes only through
// its block's WΔ slice. Under MVN, I_XP → expected information as N → ∞.
//
// Errors when Σ̂_b is non-PD at θ̂ (`InfoMatrixSingular`).
post_expected<Eigen::MatrixXd>
information_cross_products(spec::LatentStructure       pt,
                           const model::MatrixRep&         rep,
                           const SampleStats&              samp,
                           const RawData&                  raw,
                           const Estimates&                est);

// Parameter covariance matrix:
//   * no constraints: vcov = info⁻¹
//   * linear equality (shared labels / cross-group invariance / general
//     linear `==`): vcov = K · (Kᵀ I K)⁻¹ · Kᵀ, K the reparameterization
//     basis from `build_eq_constraints(pt)`.
//   * nonlinear equality (`a == b*c`): vcov = Z · (Zᵀ I Z)⁻¹ · Zᵀ, Z an
//     orthonormal basis of the null space of the constraint Jacobian
//     H = ∂h/∂θ at θ̂. This needs θ̂ — pass `est.theta` as `theta`; a model
//     with nonlinear constraints errors when `theta` is omitted.
// Returns `PostError::InfoMatrixSingular` if the (reduced) information matrix
// isn't invertible.
post_expected<Eigen::MatrixXd>
vcov(const Eigen::MatrixXd&            info,
     const spec::LatentStructure&  pt,
     const Eigen::VectorXd&        theta = {});

// Standard errors: √diag(vcov), NaN on a negative diagonal entry. Never fails.
Eigen::VectorXd se(const Eigen::MatrixXd& vcov) noexcept;

// =============================================================================
// Test-statistic / model-dimension primitives — independent of any Hessian.
// =============================================================================

// Model fit test statistic: chi² = N_total · F_ML(θ̂). Trivial closed form;
// inlined so callers don't pay for an out-of-line call. `est.fmin` is the
// final discrepancy value; `samp.n_obs` gives the per-block sample sizes.
inline double chi2_stat(const SampleStats& samp,
                        const Estimates&   est) noexcept {
  double n_total = 0.0;
  for (auto n : samp.n_obs) n_total += static_cast<double>(n);
  return n_total * est.fmin;
}

// Degrees of freedom: Σ_b p_b(p_b+1)/2 (+ Σ_b p_b if the model has mean
// structure) − fixed_x_moments − n_free + constraint.rank + nonlinear-`==`
// rank. The linear part is a pure function of `(pt, samp)`; each independent
// nonlinear `==` constraint adds rank(H(θ̂)) — pass `est.theta` as `theta`
// when the model carries nonlinear equality constraints (it errors when
// omitted).
//
// Returns `PostError::NumericIssue` if `pt` carries unenforced or infeasible
// constraints (propagated from `build_eq_constraints`).
post_expected<int>
df_stat(const spec::LatentStructure& pt,
        const SampleStats&               samp,
        const Eigen::VectorXd&           theta = {});

// =============================================================================
// Other tests / utilities — unchanged in spirit, just no longer dependent on
// the (now removed) Inference struct.
// =============================================================================

// Wald test for the linear restriction `R · θ = q`:
//   W = (Rθ̂ − q)' · (R · vcov · Rᵀ)⁻¹ · (Rθ̂ − q)
// distributed as χ²(k) under H₀, where k = R.rows() (caller supplies
// full-row-rank restrictions).
struct WaldTestResult {
  double chi2 = 0.0;
  int    df   = 0;
};

post_expected<WaldTestResult>
wald_test(const Eigen::MatrixXd& R, const Eigen::VectorXd& q,
          const Estimates& est, const Eigen::MatrixXd& vcov);

// Upper-tail χ²(df) p-value: P(X > chi2). Returns NaN when df ≤ 0 or
// chi2 < 0; returns 1 when chi2 == 0. Hand-rolled regularized upper
// incomplete gamma (series + continued-fraction switch at x ≈ a + 1)
// so we don't drag boost::math in.
double chi2_pvalue(double chi2, int df) noexcept;

// Noncentral χ²(df, ncp) CDF: P(X ≤ x), X ~ χ²(df, ncp). A Poisson(ncp/2)-
// weighted mixture of central χ²(df+2j) CDFs, summed outward from the Poisson
// mode (log-space weights) so it stays accurate for large ncp where the j = 0
// term would underflow. `ncp == 0` ⇒ the central χ²(df) CDF (`= 1 −
// chi2_pvalue(x, df)` for integer df). Returns NaN for df ≤ 0 / ncp < 0 /
// non-finite inputs; result clamped to [0, 1]. (Equivalent to R's
// `pchisq(x, df, ncp)`.) Used for the RMSEA confidence interval.
double noncentral_chisq_cdf(double x, double df, double ncp) noexcept;

// Reweighted least-squares (RLS) chi² — Browne's quadratic-form residual
// test in its model-based normal-theory form. Per block:
//
//   F_RLS_b = ½·tr((Σ̂_b⁻¹·(S_b − Σ̂_b))²)
//   T_RLS   = Σ_b n_b · F_RLS_b
//
// Asymptotically equivalent to T_ML = N·F_ML; numerically differs in finite
// samples. Matches lavaan's `test = "browne.residual.nt.model"`.
post_expected<double>
rls_chi2(const SampleStats&            samp,
         const model::ImpliedMoments&  implied);

post_expected<double>
rls_chi2(spec::LatentStructure       pt,
         const model::MatrixRep&     rep,
         const SampleStats&          samp,
         const Eigen::VectorXd&      theta);

// Browne's residual-based normal-theory test — full quadratic form with
// model-space projected out. Matches lavaan's `test = "browne.residual.nt"`.
// See the .cpp for the full derivation.
post_expected<double>
browne_residual_nt(spec::LatentStructure        pt,
                   const model::MatrixRep&   rep,
                   const SampleStats&        samp,
                   const Estimates&          est);

// Browne's residual-based ADF test — same model-space projection as
// `browne_residual_nt`, but the block weight is the empirical fourth-moment
// ACOV from complete raw data instead of the normal-theory ACOV.
post_expected<double>
browne_residual_adf(spec::LatentStructure        pt,
                    const model::MatrixRep&   rep,
                    const SampleStats&        samp,
                    const RawData&            raw,
                    const Estimates&          est);

// Per-parameter z-test: z_k = θ̂_k / SE_k, p_k = P(χ²(1) > z_k²). Convenience
// view of what `Estimates` and a separately-computed `se` vector already carry.
struct ZTestResult {
  Eigen::VectorXd z;
  Eigen::VectorXd p_value;
};

inline ZTestResult z_test(const Estimates& est,
                          const Eigen::VectorXd& se_vec) noexcept {
  ZTestResult out;
  const Eigen::Index n = est.theta.size();
  out.z.resize(n);
  out.p_value.resize(n);
  for (Eigen::Index k = 0; k < n; ++k) {
    const double s = (k < se_vec.size())
                         ? se_vec(k)
                         : std::numeric_limits<double>::quiet_NaN();
    if (!(s > 0.0)) {
      out.z(k)       = std::numeric_limits<double>::quiet_NaN();
      out.p_value(k) = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    out.z(k)       = est.theta(k) / s;
    out.p_value(k) = chi2_pvalue(out.z(k) * out.z(k), 1);
  }
  return out;
}

}  // namespace magmaan::inference
