#pragma once

#include <limits>
#include <string_view>

#include <Eigen/Core>

#include "latva/expected.hpp"
#include "latva/fit/fit.hpp"          // Estimates
#include "latva/fit/sample_stats.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/partable/partable.hpp"

namespace latva::fit {

// Post-fit inference for a single estimate. Pure data, owned by caller.
//
//   info : (n_free × n_free) expected Fisher information at θ̂
//   vcov : info⁻¹ — parameter covariance matrix
//   se   : √diag(vcov) — standard errors, one per free parameter
//   chi2 : N · F_ML(θ̂) — model fit test statistic (lavaan's `likelihood
//          = "normal"` default; Wishart variant would be (N−1) · F_ML)
//   df   : Σ_b p_b(p_b + 1)/2 − n_free
struct Inference {
  Eigen::MatrixXd info;
  Eigen::MatrixXd vcov;
  Eigen::VectorXd se;
  double          chi2 = 0.0;
  int             df   = 0;
};

// Expected-information SEs for the ML discrepancy.
//
//   I[a, b] = Σ_blocks (n_b/2) · tr(Σ_b⁻¹ ∂Σ_b/∂θ_a Σ_b⁻¹ ∂Σ_b/∂θ_b)
//
// `compute()` rebuilds a ModelEvaluator from (pt, rep) — symmetric with the
// `fit()` calling pattern. Build is cheap relative to the inference math.
struct ExpectedInfoSE {
  static constexpr std::string_view name = "expected";

  // `pt` is taken by value because we resolve fixed.x `fixed_value`s from
  // `samp` internally — symmetric with `fit()`. The caller's LatentStructure is untouched.
  post_expected<Inference>
  compute(partable::LatentStructure        pt,
          const model::MatrixRep&   rep,
          const SampleStats&        samp,
          const Estimates&          est) const;
};

// Observed-information SEs via finite-difference Hessian of the analytic
// ML gradient. Works for every model the optimizer can fit — pure CFA,
// CFA+structural, path. The Hessian columns are
//
//   H[:, k] ≈ (∇F_ML(θ̂ + h e_k) − ∇F_ML(θ̂ − h e_k)) / (2h)
//
// then symmetrized; observed info = (N/2) · H. `h_step` defaults to 1e-4
// which keeps central-difference truncation and roundoff balanced at v0
// model sizes (≤ ~30 free params, p ≤ 10).
struct FdObservedInfoSE {
  static constexpr std::string_view name = "observed.fd";

  double h_step = 1e-4;

  post_expected<Inference>
  compute(partable::LatentStructure        pt,
          const model::MatrixRep&   rep,
          const SampleStats&        samp,
          const Estimates&          est) const;
};

// Observed-information SEs via the closed-form ML Hessian.
//
//   H = H1 + H2,
//   H1[a,b] = -tr(Σ⁻¹ M_b Σ⁻¹ M_a) + 2 · tr(Σ⁻¹ M_b · Σ⁻¹ S Σ⁻¹ · M_a)
//   H2[a,b] = tr(G · ∂²Σ/∂θ_a ∂θ_b),  G = Σ⁻¹ − Σ⁻¹ S Σ⁻¹
//
// ∂²Σ/∂θ² is derived case-by-case on (mat_a, mat_b). v0 implements the
// Pure-CFA cases ((Λ,Λ), (Λ,Ψ); others are zero); Reduced-form B params
// surface as `PostError::NumericIssue` until the (·,B) cases land.
struct AnalyticObservedInfoSE {
  static constexpr std::string_view name = "observed.analytic";

  post_expected<Inference>
  compute(partable::LatentStructure        pt,
          const model::MatrixRep&   rep,
          const SampleStats&        samp,
          const Estimates&          est) const;
};

// Result of a likelihood-ratio test comparing two nested ML fits.
//   T = T_restricted − T_unrestricted   (= n · ΔF_ML)
//   df = df_restricted − df_unrestricted
// Caller must ensure the two fits are nested; the function doesn't
// verify nesting. P-values aren't computed here — once the codebase has
// an incomplete-gamma routine, a free `chi2_pvalue(T, df)` can layer on.
struct LRTestResult {
  double chi2_diff = 0.0;
  int    df_diff   = 0;
};

inline LRTestResult lr_test(const Inference& restricted,
                            const Inference& unrestricted) noexcept {
  return LRTestResult{restricted.chi2 - unrestricted.chi2,
                      restricted.df   - unrestricted.df};
}

// Wald test for the linear restriction `R · θ = q`. Test statistic
//   W = (Rθ̂ − q)' · (R · vcov · Rᵀ)⁻¹ · (Rθ̂ − q)
// distributed as χ²(k) under H₀, where k = rank(R) (taken as R.rows() —
// caller must supply restrictions with full row rank).
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

// Reweighted least-squares (RLS) chi² statistic — Browne's quadratic-form
// residual test in its model-based normal-theory form. Per block:
//
//   F_RLS_b = ½·tr((Σ̂_b⁻¹·(S_b − Σ̂_b))²)
//   T_RLS   = Σ_b n_b · F_RLS_b
//
// Derivation: T_RLS = N · vech(S−Σ̂)' Γ_NT(Σ̂)⁻¹ vech(S−Σ̂) with
// Γ_NT(Σ̂) = 2·D⁺(Σ̂⊗Σ̂)D⁺ᵀ, simplifies to the trace form above for
// symmetric residuals. Asymptotically equivalent to T_ML = N·F_ML;
// numerically differs in finite samples. Matches lavaan's
// `test = "browne.residual.nt.model"` (the RLS / model-based variant —
// the unsuffixed `browne.residual.nt` is the model-projected version
// that uses W = S⁻¹ and subtracts the Δ-space projection, see
// `browne_residual_nt` below).
//
// Returns `PostError::NumericIssue` if any Σ̂_b is non-PD. Caller is
// responsible for computing `Σ̂` from `Estimates::theta` via the
// evaluator (this function takes the implied moments directly to stay
// independent of the SE-method machinery).
post_expected<double>
rls_chi2(const SampleStats&            samp,
         const model::ImpliedMoments&  implied);

// Browne's residual-based normal-theory test — the full quadratic form
// with model-space projected out:
//
//   T_RES = N · res' · U · res
//   U     = Γ⁻¹ − Γ⁻¹ Δ (Δ' Γ⁻¹ Δ)⁻¹ Δ' Γ⁻¹
//   res   = vech(S − Σ̂)
//   Γ     = Γ_NT(S) = 2·D⁺(S⊗S)D⁺ᵀ  (uses S, not Σ̂ — lavaan default)
//   Δ     = ∂vech(σ)/∂θ at θ̂        (from `ModelEvaluator::dsigma_dtheta`)
//
// Equivalently: T_RES = N · (res'Γ⁻¹res − b'A⁻¹b) where A = Δ'Γ⁻¹Δ,
// b = Δ'Γ⁻¹res. Implementation avoids forming the (p* × p*) Γ matrix
// explicitly — uses the identity that for symmetric M, Γ_NT⁻¹·vech(M)
// = vech(W·M·W) with the diagonal halved (W = S⁻¹).
//
// Matches lavaan's `test = "browne.residual.nt"` (default residual-based
// variant). Equal to T_ML asymptotically, distinct in finite samples
// (77.9 vs 85.3 on 3F Holzinger vs the RLS 81.4).
//
// Multi-group: joint computation across blocks — `Γ_NT` is block-diag
// with per-block weight `n_b/N_total` (so `STAT = N_total · (...)`
// collapses to `Σ_b n_b · (term1_b − term2_b)` when params are
// configural, while correctly threading cross-group equality labels
// through the joint `Δ'Γ⁻¹Δ` solve).
//
// Mean structure: stacked layout `(m̄_b − μ̂_b ; vech(S_b − Σ̂_b))` per
// block, with the μ-part of `Γ_NT` equal to `Σ_b` (so `Γ⁻¹·x_μ = Σ_b⁻¹·x_μ`,
// no diagonal halving).
post_expected<double>
browne_residual_nt(partable::LatentStructure        pt,
                   const model::MatrixRep&   rep,
                   const SampleStats&        samp,
                   const Estimates&          est);

// Per-parameter z-test: z_k = θ̂_k / SE_k, p_k = P(χ²(1) > z_k²).
// Convenience view of what `Estimates` and `Inference` already carry.
struct ZTestResult {
  Eigen::VectorXd z;
  Eigen::VectorXd p_value;
};

inline ZTestResult z_test(const Estimates& est,
                          const Inference& inf) noexcept {
  ZTestResult out;
  const Eigen::Index n = est.theta.size();
  out.z.resize(n);
  out.p_value.resize(n);
  for (Eigen::Index k = 0; k < n; ++k) {
    const double s = inf.se(k);
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

}  // namespace latva::fit
