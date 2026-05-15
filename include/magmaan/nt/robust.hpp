#pragma once

#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"          // Estimates
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/nt/weighted_chisq.hpp"
#include "magmaan/nt/restriction.hpp"
#include "magmaan/nt/satorra2000.hpp"
#include "magmaan/nt/lr_test_satorra.hpp"

namespace magmaan::nt::robust {

using data::RawData;
using data::SampleStats;
using estimate::Estimates;
using estimate::EqConstraints;
using estimate::build_eq_constraints;
using estimate::resolve_fixed_x_from_sample;

// ============================================================================
// Robust normal-theory inference — Satorra-Bentler / Yuan-Bentler / eigenvalues
// of UΓ.
//
// Mathematical setup (single block for clarity; multi-block is block-diagonal):
//
//   U = W − W·Δ·(Δᵀ·W·Δ)⁻¹·Δᵀ·W                          (p* × p*)
//   W = Γ_NT⁻¹                                            (the NT weight)
//   Δ = ∂vech(σ)/∂θ at θ̂                                   (p* × q)
//
// `U` has rank `df = p* − q` (q-dim null space is range(Δ)). Its nonzero
// eigenvalues coincide with the eigenvalues of U·Γ where Γ is any
// fourth-moment-like ACOV of vech(S) — sample, normal-theory, or the
// Browne-unbiased correction.
//
// Rather than form `U·Γ` directly (p* × p*, dense, non-symmetric), we factor
//
//   W = L·Lᵀ                       (Cholesky-style)
//   A = Lᵀ·Δ                       (p* × q)
//   N = orthonormal basis of ker(Aᵀ)   (p* × df)
//   B = L·N                        (p* × df)        ⇒  U = B·Bᵀ
//
// Then for any Γ (symmetric, PSD):
//
//   M_Γ = Bᵀ·Γ·B                   (df × df, symmetric)
//   eigvals(U·Γ) = eigvals(B·Bᵀ·Γ) = eigvals(M_Γ)
//                                 (matching the df nonzero eigenvalues)
//
// `SelfAdjointEigenSolver(M_Γ, EigenvaluesOnly)` returns the spectrum in
// O(df³). The Γ matrix itself is *never* materialised when its structure
// supports a cheaper "reduced" computation:
//
//   • sample Γ̂   : reduce via casewise contributions Z_c (N × p*) directly,
//                 (Z_c·B)ᵀ(Z_c·B) / N — never form the p* × p* matrix.
//   • Γ_NT       : apply x ↦ Γ_NT·x as an operator (cov-only: vech(2·Σ·H·Σ));
//                 fill M_Γ column by column.
//   • unbiased Γ_u: linear combination of the already-reduced sample and NT
//                  M matrices plus a rank-1 term from vech(S) (Browne's
//                  distribution-free correction).
//
// Sanity check baked in: when Γ = Γ_NT, U·Γ_NT is the orthogonal projector
// onto range(B), so its nonzero eigenvalues are exactly `df` ones (within
// numerical noise). `reduced_gamma_nt` + `ugamma_eigenvalues` should yield
// `eigvals ≈ (1, …, 1)`. Any non-trivial drift means something is
// transposed, scaled, or packed wrong upstream.
// ============================================================================

// ── The information-matrix vocabulary ───────────────────────────────────────
// Three orthogonal knobs, shared by the standard-error path (`robust_se`,
// and conceptually the existing naive-SE classes) and the UΓ test-statistic
// path (`build_u_factor` + `reduced_gamma_*`). Mirrors lavaan's `information`
// / `h1.information` but *without* lavaan's `first.order` corner: there is no
// "K-as-bread" mode — for ML the gradient-outer-product `K` is not a distinct
// quantity, it equals `Δᵀ·W·Γ̂·W·Δ` (the empirical meat below), so we expose
// it only as a meat choice, never inverted-as-a-bread.

// The "bread" — the q × q curvature matrix that gets inverted (for SEs:
// `vcov = bread⁻¹` naive, `bread⁻¹·meat·bread⁻¹` robust; for the test:
// the `E_inv` factor projecting the model space out of `U`).
//   Expected = J = Δᵀ·W·Δ      — the GLS/Fisher form (W = Γ_NT(·)⁻¹). For
//              the test this keeps `U` a rank-`df` orthogonal projector.
//              (Also ≈ lavaan's `observed.information = "h1"` for the ML
//              discrepancy, so we don't expose that as a separate value.)
//   Observed = H = H1 + H2      — the actual ML Hessian; the H2 term makes
//              it ≠ J. For the test, `U` is no longer idempotent — the MLR
//              convention; lavaan's `information = "observed",
//              observed.information = "hessian"`.
enum class Information { Expected, Observed };

// Which moments build the normal-theory weight  W = Γ_NT(M)⁻¹  (and feed
// the model-implied meat).  Mirrors lavaan's `h1.information`:
//   Structured   = M = Σ̂ (model-implied) — the default; classic SB / MLM /
//                  `h1.information = "structured"`.
//   Unstructured = M = S  (sample)        — `h1.information = "unstructured"`;
//                  the weight `browne_residual_nt` uses.
enum class WeightMoments { Structured, Unstructured };

// The "meat" — the ACOV estimate of vech(S).
//   ModelImplied   = Γ_NT(M):  SE sandwich collapses to bread⁻¹ (= the naive
//                    SE); for the test, `UΓ` is a rank-`df` projector (all
//                    eigenvalues 1, no χ² scaling).
//   Empirical      = Γ̂ (sample fourth-moment ACOV; needs raw data): SE = the
//                    full sandwich (robust SE); for the test, `UΓ` carries
//                    the Satorra-Bentler-family χ² scaling.
//   BrowneUnbiased = Browne's distribution-free correction (single-block).
enum class ScoreCovariance { ModelImplied, Empirical, BrowneUnbiased };

// The full spec. `build_u_factor` consumes `bread` + `moments` (the meat
// for the test is chosen by which `reduced_gamma_*` you call, so `cov` is
// ignored there). `robust_se` consumes all three.
struct InferenceSpec {
  Information      bread   = Information::Expected;
  WeightMoments   moments = WeightMoments::Structured;
  ScoreCovariance cov     = ScoreCovariance::ModelImplied;
};

// Stored U-factor for one (model, θ̂, spec) triple. Carries the per-block
// Cholesky factors of the weight matrix so the meat reductions can reuse
// them. All three Γ flavors take a `const UFactor&` and produce their own
// M matrix; `ugamma_eigenvalues(M)` then returns the spectrum.
//
// Two shapes, per `spec.bread`:
//   ProjectionExpected — `U = B·Bᵀ` is the rank-`df` orthogonal projector
//     (`B = L_Γ⁻ᵀ·N`). `reduced_gamma_*` return `M = BᵀΓB` (df × df), so
//     `ugamma_eigenvalues` gives `df` eigenvalues (all 1 for `Γ = Γ_NT`).
//   ObservedHessian — `U = L_W·(I − A·H_obs⁻¹·Aᵀ)·L_Wᵀ` is *not* idempotent
//     (`L_W = L_Γ⁻ᵀ`, `A = L_Γ⁻¹·Δ`). `reduced_gamma_*` return the
//     symmetric `M = R̃ᵀ·(I − A·H_obs⁻¹·Aᵀ)·R̃` where `R̃·R̃ᵀ = L_Wᵀ·Γ·L_W`
//     (= `I` for `Γ = Γ_NT`, matching moments), so `ugamma_eigenvalues`
//     gives up to `p*` eigenvalues — `df` (= p*−q) stays the χ²-reference df.
struct UFactor {
  enum class Kind { ProjectionExpected, ObservedHessian };
  Kind            kind  = Kind::ProjectionExpected;
  Eigen::MatrixXd B;                       // ProjectionExpected: total_rows × df  (= L_Γ⁻ᵀ·N)
  Eigen::MatrixXd A;                       // ObservedHessian: total_rows × q      (= L_Γ⁻¹·Δ)
  Eigen::MatrixXd H_obs_inv;               // ObservedHessian: q × q               (inverse observed Hessian)
  Eigen::Index    df         = 0;          // (Σ_b p_b·has_means + p*) − rank(Δ)
  Eigen::Index    pstar      = 0;          // Σ_b p*_b — σ-only total (unchanged)
  Eigen::Index    total_rows = 0;          // Σ_b ((has_means ? p_b : 0) + p*_b)
  bool            has_means  = false;      // whether μ-rows are stacked into B/A
  WeightMoments   moments = WeightMoments::Structured;  // which moments built W

  // Per-block geometry, mirrors `SampleStats` ordering. Stacked-block layout
  // (G3b, mirroring `browne_residual_nt`'s `[μ_b; vech(Σ_b)]` ordering): each
  // block occupies a contiguous slice of size `(has_means ? p : 0) + pstar`
  // in B/A; the μ-segment (when present) sits on top of the σ-segment.
  struct Block {
    Eigen::Index    p          = 0;        // n_observed in block
    Eigen::Index    pstar      = 0;        // p(p+1)/2
    Eigen::Index    row_offset = 0;        // start row of THIS BLOCK'S σ-vech segment in B/A
    Eigen::Index    mu_off     = -1;       // start row of μ-segment (= row_offset − p when has_means; -1 otherwise)
    Eigen::Index    n_obs      = 0;        // sample size of this block (the natural Γ̂ divisor)
    // Cholesky factor L_Γ of the σ-moments-weight matrix Γ_NT_cov(M_b) where
    // M_b is Σ̂_b (structured) or S_b (unstructured). `solve()` against this
    // gives the L_Γ⁻¹ / L_Γ⁻ᵀ actions on the σ-segment.
    Eigen::LLT<Eigen::MatrixXd> llt_gamma_nt;
    // Cholesky factor of M_b itself — the μ-block of Γ_NT(M_b) is just M_b
    // (no factor of 2, no off-diag halving) in the stacked `[μ; vech(Σ)]`
    // moment convention. Only populated when `uf.has_means` is true.
    Eigen::LLT<Eigen::MatrixXd> llt_M;
    Eigen::MatrixXd             Sigma_hat;  // per-block Σ̂ (for the NT meat)
    Eigen::MatrixXd             S;          // per-block sample cov (unstructured meat)
  };
  std::vector<Block> blocks;
};

// Build the U-factor from (pt, rep, samp, est, spec). `spec.cov` is ignored
// — the meat for the test is the `reduced_gamma_*` choice. Internally:
//   1. Run `prepare_evaluator` (resolves fixed.x, rebuilds evaluator).
//   2. For each block b: pick M_b = Σ̂_b (structured) or S_b (unstructured)
//      per `spec.moments`; form Γ_NT(M_b), LLT-factor it → L_Γ,b.
//   3. Compute A_b = L_Γ,b⁻¹·Δ_b via triangular solve. Stack as A.
//   4. `spec.bread == Expected` (`kind = ProjectionExpected`):
//        HouseholderQR of A → trailing (p* − rank(A)) columns of Q are an
//        orthonormal basis N of ker(Aᵀ); B_b = L_Γ,b⁻ᵀ·N_b. `U = B·Bᵀ`.
//      `spec.bread == Observed` (`kind = ObservedHessian`): compute the
//        observed Hessian H_obs (via `information_observed_analytic`, FD
//        fallback) in total-N per-unit units, project it through any equality
//        constraints, then store `A` and `H_obs⁻¹`.
//        `U = L_Γ⁻ᵀ·(I − A·H_obs⁻¹·Aᵀ)·L_Γ⁻¹` — not idempotent.
//
// Multi-block: both `Expected` and `Observed` stack A/B per block. Mean
// structure (`~1`): when `dmu_dtheta(theta)` is non-empty, each block's
// stacked slice is `[μ_b; vech(Σ_b)]` (μ-rows on top, σ-rows below) and the
// per-block Γ_NT block-diagonalises as
// `[M_b 0; 0 Γ_NT_cov(M_b)]` (G3b). Mirrors `browne_residual_nt`'s layout.
//
// `PostError::InfoMatrixSingular` if Γ_NT(M_b) or M_b is non-PD, Δ is
// rank-deficient (Expected), or H_obs is non-invertible (Observed).
post_expected<UFactor>
build_u_factor(spec::LatentStructure        pt,
               const model::MatrixRep&   rep,
               const SampleStats&        samp,
               const Estimates&          est,
               InferenceSpec             spec = {});

// ============================================================================
// Three Γ-reduction functions — each returns M_Γ = Bᵀ·Γ·B (df × df, sym).
// ============================================================================

// (a) Sample Γ̂ — reduced from casewise contributions, never forms Γ̂.
//
// `Zc` is the (N_total × C) matrix of centred casewise contributions,
// where C depends on whether the UFactor was built with mean structure:
//   • cov-only UFactor  (uf.has_means == false): C = Σ_b p_b*.
//                       z_i,b = vech((x_i − m̄_b)(x_i − m̄_b)ᵀ) − vech(S_b)
//   • means UFactor     (uf.has_means == true):  C = uf.total_rows.
//                       z_i,b = [x_i − m̄_b; vech((x_i − m̄_b)(...)ᵀ) − vech(S_b)]
//                       per block, with the μ-segment on top of the σ-segment
//                       in the block's contiguous column slice (G3b layout).
// Each row of Zc carries nonzero entries only in its block's column slice.
// Use `casewise_contributions(raw, samp, include_means)` to build it correctly
// (pass `include_means=true` whenever `uf.has_means` is true).
//
// Reduces the *block-diagonal* empirical ACOV Γ̂ = diag_b(Γ̂_b), with each
// Γ̂_b = (Zc_b)ᵀ(Zc_b) / denom_b:
//
//   M = Σ_b (Zc_b·B_b)ᵀ·(Zc_b·B_b) / denom_b           (df × df, symmetric)
//
// `denom` is the per-block divisor vector — length `n_blocks` (the natural
// choice is each block's n_b: `fit$nobs`), or length 1 to use one divisor
// for every block. For a single block this is the usual `(Zc·B)ᵀ(Zc·B)/n`
// (pass `n−1` for a divisor-correcting variant). NB: passing a single
// pooled `N_total` for a multi-block model is *wrong* — it averages the
// block contributions instead of summing them; supply the per-block vector.
//
// The convenience `double` overload uses one `denom` for every block.
post_expected<Eigen::MatrixXd>
reduced_gamma_sample(const UFactor&                            uf,
                     const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
                     const Eigen::Ref<const Eigen::VectorXd>&  denom);
post_expected<Eigen::MatrixXd>
reduced_gamma_sample(const UFactor&                            uf,
                     const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
                     double                                    denom);

// Streaming variant for memory-bounded N. Sums rank-1 outer products
// (Bᵀ·z_i)(Bᵀ·z_i)ᵀ over rows, divides by `denom` at the end. Use when
// holding the full `Zc` matrix in memory is undesirable; the loop only
// allocates a df-vector at a time.
post_expected<Eigen::MatrixXd>
reduced_gamma_sample_streaming(const UFactor&                            uf,
                               const std::vector<Eigen::VectorXd>&       zc_rows,
                               double                                    denom);

// Helper: build the stacked Zc matrix from RawData + SampleStats. Each row
// of Zc corresponds to one observation; the row carries that observation's
// block contribution in the appropriate column slice (zeros elsewhere — so
// cross-block summation in `Zcᵀ·Zc` is automatic and correct).
//
// `include_means = false` (default, cov-only): output shape
//   (Σ_b n_b) × (Σ_b p_b*); each row stores
//   vech((x_i − m̄_b)(x_i − m̄_b)ᵀ) − vech(S_b) at block b's σ-vech slice.
//
// `include_means = true` (G3b): output shape (Σ_b n_b) × Σ_b (p_b + p_b*);
//   per block the column layout is [μ-cols | σ-vech cols] (μ on top), with
//   the per-row μ-segment carrying `x_i − m̄_b` and the σ-segment carrying
//   the centred outer-product residual. Matches `UFactor::Block::mu_off`
//   and `row_offset` once the UFactor has been built — column offsets here
//   are computed inside, callers don't need to reconstruct them.
//
// Row order: all rows from block 0 first, then block 1, etc.
post_expected<Eigen::MatrixXd>
casewise_contributions(const RawData& raw, const SampleStats& samp,
                       bool include_means = false);

// (b) Normal-theory Γ_NT — operator-only, never forms Γ_NT. The "meat"
// moments match the bread: if `uf.moments == Structured` it uses Σ̂_b, if
// `Unstructured` it uses S_b.
//
// For each column b_j of B (df columns total) and each block, applies the
// operator
//   vech(H) ↦ vech(2·M_b·H·M_b)         (M_b = Σ̂_b or S_b; no diag halving
//                                        — the "2" is outside the vech and
//                                        absorbs the Γ_NT symmetry scaling)
// to the block-slice of b_j, then dots with Bᵀ to fill M column by column.
//
// Sanity: returned M has eigenvalues exactly `(1, …, 1)` (within ~1e-10)
// when paired with `build_u_factor` on the same fit/spec — `U·Γ_NT` is a
// rank-`df` projector when the bread and meat moments agree (which they do
// here by construction).
post_expected<Eigen::MatrixXd>
reduced_gamma_nt(const UFactor& uf);

// (c) Browne's unbiased Γ_u (distribution-free, complete-data, simple
// setting only). Closed form:
//
//   Bᵀ·Γ_u·B = a · M_sample  −  b · M_nt  +  b · 2/(N−1) · (Bᵀs)(Bᵀs)ᵀ
//   a = N·(N−1) / ((N−2)·(N−3))
//   b = N       / ((N−2)·(N−3))
//   s = vech(S) stacked per block
//
// `N` here is the single block's sample size. The reduced-matrix shorthand
// errors for multi-block models because the per-block sample pieces have
// already been summed. Use `reduced_gamma_unbiased_casewise()` for grouped
// data so the per-block Browne coefficients can be applied before stitching.
post_expected<Eigen::MatrixXd>
reduced_gamma_unbiased(const UFactor&            uf,
                       const SampleStats&        samp,
                       const Eigen::MatrixXd&    M_sample,
                       const Eigen::MatrixXd&    M_nt);

// Multi-block Browne-unbiased correction. This overload consumes the raw
// stacked casewise contributions so it can apply Browne's finite-sample
// coefficients per block before summing into the common reduced basis. The
// reduced-matrix overload above remains the single-block shorthand.
post_expected<Eigen::MatrixXd>
reduced_gamma_unbiased_casewise(
    const UFactor&                            uf,
    const SampleStats&                        samp,
    const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
    const Eigen::Ref<const Eigen::VectorXd>&  denom);
post_expected<Eigen::MatrixXd>
reduced_gamma_unbiased_casewise(
    const UFactor&                            uf,
    const SampleStats&                        samp,
    const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
    double                                    denom);

// ============================================================================
// Eigenvalues + robust statistics
// ============================================================================

// `Eigen::SelfAdjointEigenSolver(M, ComputeEigenvaluesOnly)` wrapped with
// the post-error pattern. Symmetrises `M` before solving (kills tiny
// upstream asymmetry from floating-point matmul). Returns `df` ascending
// real eigenvalues.
post_expected<Eigen::VectorXd>
ugamma_eigenvalues(const Eigen::Ref<const Eigen::MatrixXd>& M);

// Satorra-Bentler scaled chi²: matches mean of the asymptotic mixture
// distribution of T_ML.
//
//   c = mean(λ) = Σλ / df            (the "scaling correction factor")
//   T_SB = T_ML / c
//
// Under multivariate normality, c → 1 and T_SB → T_ML. Lavaan exposes
// this as `test = "satorra.bentler"`.
struct SatorraBentlerResult {
  double chi2_scaled = 0.0;
  double scale_c     = 0.0;   // c = Σλ / df
  int    df          = 0;     // unchanged from T_ML
};
SatorraBentlerResult
satorra_bentler(double                                    t_ml,
                int                                       df,
                const Eigen::Ref<const Eigen::VectorXd>&  eigvals) noexcept;

// Satterthwaite mean-and-variance-adjusted chi² (Asparouhov-Muthén 2010,
// lavaan `test = "mean.var.adjusted"`):
//
//   c_mean = Σλ / df,    c_var = Σλ² / df
//   df_adj  = df · c_mean² / c_var  =  (Σλ)² / Σλ²
//   T_adj   = T_ML · c_mean / c_var
//
// Matches both first two moments of the asymptotic mixture distribution.
// The df is non-integer; reference distribution is χ²(df_adj). Equivalent
// to the "Yuan-Bentler T_3" stat under the standard convention.
struct MeanVarAdjustedResult {
  double chi2_adj = 0.0;
  double df_adj   = 0.0;       // non-integer Satterthwaite df
};
MeanVarAdjustedResult
mean_var_adjusted(double                                    t_ml,
                  int                                       df,
                  const Eigen::Ref<const Eigen::VectorXd>&  eigvals) noexcept;

// Satorra-Bentler scaled-and-shifted chi² (Satorra-Bentler 2010, lavaan
// `test = "scaled.shifted"`):
//
//   a       = √(df / Σλ²)
//   b       = df − a · Σλ
//   T_adj   = T_ML · a + b      (linear transform — both scale and shift)
//
// The df stays at `df` (referenced against χ²(df)). Combines mean and
// variance corrections by a different functional form than the
// Satterthwaite df-adjustment; chosen by some methodologists because the
// reference distribution stays integer-df.
struct ScaledShiftedResult {
  double chi2_adj = 0.0;
  int    df       = 0;
  double scale_a  = 0.0;       // T_adj = T_ML·a + b
  double shift_b  = 0.0;
};
ScaledShiftedResult
scaled_shifted(double                                    t_ml,
               int                                       df,
               const Eigen::Ref<const Eigen::VectorXd>&  eigvals) noexcept;

// ============================================================================
// Robust ("sandwich") standard errors
// ============================================================================
//
//   vcov_robust = (1/N) · A1⁻¹ · B1 · A1⁻¹
//   A1 = Σ_b (n_b/N)·Δ_bᵀW_bΔ_b      (spec.bread == Information::Expected; the
//                                     per-unit GLS/Fisher form, W_b = Γ_NT(M_b)⁻¹)
//      = H = H1 + H2 / N              (spec.bread == Information::Observed — the
//                                     MLR convention, projected through K)
//   B1 = Σ_b (n_b/N)·Δ_bᵀW_bΓ̂_bW_bΔ_b   (the per-unit score variance; with the
//                                     `gamma_hat` overload it's (WΔ)ᵀ·Γ̂·(WΔ),
//                                     single-block; with the `RawData` overload
//                                     it's (Z_c·WΔ)ᵀ·(Z_c·WΔ)/N_total with WΔ
//                                     stacked block-diagonal — never forms the
//                                     p* × p* matrix)
//   M_b   = Σ̂_b (Structured) or S_b (Unstructured).
//
// Under multivariate normality `Γ̂_b → Γ_NT(M_b)` ⇒ `B1 → A1` ⇒ `vcov_robust →
// (1/N)·A1⁻¹` = the naive expected `vcov` (so `information_expected` ∘ `vcov`
// is the `{Expected, Structured, ModelImplied}` corner of this). For a single group
// A1/B1 are just Δᵀ W Δ / Δᵀ W Γ̂ W Δ.
//
// With linear equality constraints (shared labels, `a == b`) the SE path
// reparameterizes θ = K·α: Δ → Δ·K, vcov_α = (1/N)·A1⁻¹·B1·A1⁻¹ (n_alpha²),
// vcov = K·vcov_α·Kᵀ (npar²) — `se.size() == n_free` (the expanded form, with
// constrained-equal parameters sharing an SE), mirroring the lavaan vcov shape.
//
// lavaan mapping: `{Expected, Structured, Empirical}` ≡ `se = "robust.sem"`
// (the SB / MLM SE — multi-group via the `RawData` overload); `{Observed,
// Structured, Empirical}` ≡ `se = "robust.huber.white"` (MLR).
// `spec.cov == BrowneUnbiased` errors for now.
struct RobustSeResult {
  Eigen::MatrixXd vcov;       // n_free × n_free  — Cov(θ̂) (robust)
  Eigen::VectorXd se;         // n_free            — √diag(vcov)
};

// Primitive: caller supplies Γ̂ directly (p* × p*, e.g. from `empirical_gamma`
// or lavaan's NACOV).  meat = (WΔ)ᵀ·Γ̂·(WΔ). Single-block only (for multi-group
// the per-block weighting is implicit — use the raw-data overload).
post_expected<RobustSeResult>
robust_se(spec::LatentStructure        pt,
          const model::MatrixRep&   rep,
          const SampleStats&        samp,
          const Estimates&          est,
          const Eigen::MatrixXd&    gamma_hat,
          InferenceSpec             spec = {Information::Expected,
                                            WeightMoments::Structured,
                                            ScoreCovariance::Empirical});

// Convenience: derives Γ̂ from raw data via `casewise_contributions` — never
// materialises the p* × p* matrix (meat = (Z_c·WΔ)ᵀ·(Z_c·WΔ)/N_total). Works
// multi-group for both `Expected` and `Observed` bread; the per-block n_b/N
// weighting falls out of the block-diagonal Z_c · WΔ.
post_expected<RobustSeResult>
robust_se(spec::LatentStructure        pt,
          const model::MatrixRep&   rep,
          const SampleStats&        samp,
          const Estimates&          est,
          const RawData&            raw,
          InferenceSpec             spec = {Information::Expected,
                                            WeightMoments::Structured,
                                            ScoreCovariance::Empirical});

}  // namespace magmaan::nt::robust
