#pragma once

#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"          // Estimates
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/data/pairwise_cov.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/robust/weighted_chisq.hpp"
#include "magmaan/robust/restriction.hpp"
#include "magmaan/robust/satorra2000.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"

namespace magmaan::robust {

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
//   Pairwise     = use `data::gamma_nt_pairwise(raw, pw)[b]` directly as the
//                  bread's Γ (not `Γ_NT(Σ̂_b)`). Requires raw data + a
//                  `PairwiseSampleStats` — the build_u_factor overload that
//                  takes those arguments must be used. With matching
//                  `cov = ModelImplied` meat (`reduced_gamma_nt_pairwise`)
//                  the sandwich SE collapses to `(1/N)·A⁻¹` (the naive
//                  expected vcov on a Γ_NT^pw weight); with `cov =
//                  Empirical` meat (`Ψ̂'Ψ̂/n`) it becomes the pairwise +
//                  non-normal robust SE — the principled SE for a
//                  pairwise SEM fit.
enum class WeightMoments { Structured, Unstructured, Pairwise };

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

// Overload for `spec.moments == Pairwise`. Builds the per-block bread Γ
// from `data::gamma_nt_pairwise(raw, pw)[b]` instead of `gamma_nt(Σ̂_b)` /
// `gamma_nt(S_b)`. `raw` must carry the missingness mask; `pw` must match
// `raw`'s block layout (typically `data::pairwise_sample_stats(raw)`).
// `spec.moments` is required to be `Pairwise` here — the other two values
// route to the existing 5-arg overload.
//
// The μ-block of the bread (when `has_means`) keeps the existing
// `LLT(Σ̂_b)` convention — same compromise as `fit_gls_pairwise`'s μ-block
// weight. The pairwise μ ACOV is still in `docs/backlog/speculative.md`.
post_expected<UFactor>
build_u_factor(spec::LatentStructure                pt,
               const model::MatrixRep&              rep,
               const SampleStats&                   samp,
               const Estimates&                     est,
               const RawData&                       raw,
               const data::PairwiseSampleStats&     pw,
               InferenceSpec                        spec);

// Both-breads pair. Runs `build_u_factor`'s shared phase once (Δ-stacking,
// per-block Γ_NT Cholesky, `A = L_Γ⁻¹·Δ` per-block solve) and only re-runs
// the bread-specific tail twice — QR-of-A for Expected, observed Hessian
// compute-and-invert for Observed. Equivalent to two single-bread
// `build_u_factor` calls (same `moments`) but skips one full shared pass.
// Used by `robust_test_moments_both_breads` for the χ² adjustments path.
struct UFactorPair {
  UFactor expected;
  UFactor observed;
};

post_expected<UFactorPair>
build_u_factor_pair(spec::LatentStructure        pt,
                    const model::MatrixRep&   rep,
                    const SampleStats&        samp,
                    const Estimates&          est,
                    WeightMoments             moments = WeightMoments::Structured);

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

// Caller-supplied empirical Γ̂ counterpart. `gamma_hat` is the stacked
// block-diagonal moment covariance in the same column layout as Zc:
// cov-only ⇒ Σ_b p_b*; means ⇒ Σ_b (p_b + p_b*). Each diagonal block must
// already use the intended divisor/scaling, e.g. Γ̂_b = Zc_bᵀZc_b / n_b for
// the usual multi-block robust test. For a single block this is simply
// `crossprod(Zc) / n`.
//
// This is algebraically the same reduction as `reduced_gamma_sample()`:
//   BᵀΓ̂B == (ZcB)ᵀ(ZcB)/n
// but lets callers reuse a Γ̂ matrix they already materialized for sandwich
// SEs or benchmarking.
post_expected<Eigen::MatrixXd>
reduced_gamma_sample_from_gamma(
    const UFactor&                            uf,
    const Eigen::Ref<const Eigen::MatrixXd>&  gamma_hat);

// Fast robust-test path for SB / MV-adj / scaled-shifted adjustments.
// Returns the trace moments needed by those adjustments for both
// bread=Expected and bread=Observed without returning the full reduced
// df × df matrix to the caller. Matrix-returning reducers above remain the
// diagnostic / spectrum path.
struct RobustTestMomentsBothBreads {
  WeightedChiSquareMoments expected;  // bread = Expected
  WeightedChiSquareMoments observed;  // bread = Observed
};

post_expected<RobustTestMomentsBothBreads>
robust_test_moments_both_breads(
    spec::LatentStructure                     pt,
    const model::MatrixRep&                   rep,
    const SampleStats&                        samp,
    const Estimates&                          est,
    const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
    const Eigen::Ref<const Eigen::VectorXd>&  denom,
    WeightMoments                             moments = WeightMoments::Structured);
post_expected<RobustTestMomentsBothBreads>
robust_test_moments_both_breads(
    spec::LatentStructure                     pt,
    const model::MatrixRep&                   rep,
    const SampleStats&                        samp,
    const Estimates&                          est,
    const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
    double                                    denom,
    WeightMoments                             moments = WeightMoments::Structured);
post_expected<RobustTestMomentsBothBreads>
robust_test_moments_both_breads_from_gamma(
    spec::LatentStructure                     pt,
    const model::MatrixRep&                   rep,
    const SampleStats&                        samp,
    const Estimates&                          est,
    const Eigen::Ref<const Eigen::MatrixXd>&  gamma_hat,
    WeightMoments                             moments = WeightMoments::Structured);

// Reference/materialized counterpart to `reduced_gamma_sample()`.
// Forms each empirical Γ̂_b = Zc_bᵀZc_b / denom_b explicitly before reducing
// with B_bᵀΓ̂_bB_b. This is intentionally less memory-frugal; it exists for
// diagnostics, benchmarks, and parity checks of the reduced casewise algebra.
post_expected<Eigen::MatrixXd>
reduced_gamma_sample_materialized(
    const UFactor&                            uf,
    const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
    const Eigen::Ref<const Eigen::VectorXd>&  denom);
post_expected<Eigen::MatrixXd>
reduced_gamma_sample_materialized(
    const UFactor&                            uf,
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

// Tiled complete-data projection helpers. These avoid materializing the full
// N × p* (or N × total_rows with means) casewise contribution matrix. They are
// currently defined for the expected-bread projection U-factor, where
// Y = Zc·B and M = YᵀY / denom.
post_expected<Eigen::MatrixXd>
casewise_projected_rows_tiled(const UFactor&            uf,
                              const RawData&            raw,
                              const SampleStats&        samp,
                              Eigen::Index              tile_rows = 128);
post_expected<Eigen::MatrixXd>
reduced_gamma_sample_tiled(const UFactor&                           uf,
                           const RawData&                           raw,
                           const SampleStats&                       samp,
                           const Eigen::Ref<const Eigen::VectorXd>& denom,
                           Eigen::Index                             tile_rows = 128);
post_expected<Eigen::MatrixXd>
reduced_gamma_sample_tiled(const UFactor&     uf,
                           const RawData&     raw,
                           const SampleStats& samp,
                           double             denom,
                           Eigen::Index       tile_rows = 128);

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

// Van-Praag-style pairwise casewise contributions Ψ̂ for incomplete continuous
// data. Per-block layout mirrors `casewise_contributions(raw, samp,
// include_means)`: rows stacked block-by-block, columns block-stacked into
// `[μ-cols | σ-vech cols]` per block (μ-cols omitted when `include_means =
// false`, the cov-only layout). For row `t` in block `b`:
//
//   σ-segment, component a = (j, ℓ):
//     ψ̂_ta = (A_t(j,ℓ) / π̂_(j,ℓ)) · ( q_ta − Ŝ_(j,ℓ) )
//     q_ta = (x_tj − m̄^(j,ℓ)_j)(x_tℓ − m̄^(j,ℓ)_ℓ)        if A_t(j,ℓ) = 1
//          = 0                                              if A_t(j,ℓ) = 0
//
//   μ-segment, variable j (only when `include_means = true`):
//     ψ̂_tj_μ = (R_tj / π̂_j) · ( x_tj − m̄_j )            if R_tj = 1
//            = 0                                            if R_tj = 0
//
// The σ-side uses pair-specific Van Praag means inside each overlap set
// (the same means that `pairwise_sample_stats` uses to build `Ŝ`); the μ
// side uses the marginal column mean and the marginal availability
// `π̂_j = π̂_(j,j) = pi_hat(j, j)` already in `PairwiseSampleStats`. On
// complete data both segments reduce to the complete-data forms produced
// by `casewise_contributions(raw, samp, include_means)`.
//
// Ψ̂ plugs directly into `reduced_gamma_sample(uf, Ψ̂, n_b)` and the
// downstream U-Gamma / SB / mean-var / scaled-shifted machinery — the
// projection algebra is identical to the complete-data case, only the
// influence-function rows change.
post_expected<Eigen::MatrixXd>
pairwise_casewise_contributions(const RawData& raw,
                                const data::PairwiseSampleStats& pw,
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

// (b'') Sample-moment normal-theory Bᵀ·Γ_NT(S)·B — same projection as
// `reduced_gamma_nt(uf)` but with the *sample* covariance S in the Γ_NT
// operator instead of the bread's structured moments Σ̂. This is the NT term
// the Du-Bentler unbiased Γ requires: Γ_u is a distribution-free estimator of
// the population fourth-moment ACOV, so its normal-theory piece is Γ_NT(S),
// not Γ_NT(Σ̂). When the U-factor is built `Structured`, `reduced_gamma_nt(uf)`
// returns ≈ I (the bread orthonormalises against Γ_NT(Σ̂)); feeding that as the
// unbiasing `M_nt` silently makes the correction model-dependent and breaks
// parity with lavaan/semTests off the (rare) S = Σ̂ point. Only the
// `ProjectionExpected` U-factor is supported.
post_expected<Eigen::MatrixXd>
reduced_gamma_nt_sample(const UFactor& uf);

// (b') Pairwise normal-theory Γ_NT^pw — the missing-data analogue of
// `reduced_gamma_nt(uf)`, computed via the pattern-grouped expectation
// identity without materialising the p* × p* `Γ_NT^pw` matrix. For each
// distinct missingness pattern k in `raw.mask[b]`, builds the row-scaled
// `B̃_k = diag(a_k/π̂)·B` (σ-rows masked-and-rescaled by per-pair
// availability), applies the existing `Γ_NT(Σ̂_b)` operator column-by-
// column, and accumulates `(n_k/n) · B̃_k'·(Γ_NT·B̃_k)`. Operator path:
// O(K·df·p³ + K·df²·p*) per block, where K is the # distinct patterns.
//
// Used as the meat for SB-style chi-square scaling on a pairwise fit
// where you want the model-implied (not empirical) meat. Paired with a
// `WeightMoments::Pairwise` UFactor produces the bread-meat collapse —
// `U·Γ_NT^pw` is a rank-df projector, eigenvalues ≈ (1, …, 1). Paired
// with a `WeightMoments::Structured` UFactor gives the SB correction
// that accounts for the pairwise ACOV mismatch.
//
// μ-block treatment matches `data::gamma_nt_pairwise`: cov-only on the
// σ-segment via the pattern-grouped operator; μ-segment uses the
// existing complete-data operator (`Σ̂_b` on the μ residual). Full
// pairwise μ ACOV stays in `docs/backlog/speculative.md`.
post_expected<Eigen::MatrixXd>
reduced_gamma_nt_pairwise(const UFactor&                       uf,
                          const RawData&                       raw,
                          const data::PairwiseSampleStats&     pw);

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

// Zc-overload: caller supplies casewise contributions directly. Same
// `(Z_c·WΔ)ᵀ·(Z_c·WΔ)/N_total` meat as the raw-data overload, but skips the
// internal `casewise_contributions` rebuild — useful in simulation loops
// where the same Zc feeds multiple sandwich/eigvalue calls. Zc must match
// the layout from `casewise_contributions(raw, samp, include_means)`:
// cov-only ⇒ `(Σ_b n_b) × (Σ_b p_b*)`; means ⇒ same N with per-block
// [μ-cols | σ-vech cols] (G3b layout). `n_total` is the divisor used in the
// meat (the natural choice is `sum(samp.n_obs)`).
post_expected<RobustSeResult>
robust_se(spec::LatentStructure                     pt,
          const model::MatrixRep&                   rep,
          const SampleStats&                        samp,
          const Estimates&                          est,
          const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
          double                                    n_total,
          InferenceSpec                             spec = {Information::Expected,
                                                            WeightMoments::Structured,
                                                            ScoreCovariance::Empirical});

// Both-breads pair. Runs `robust_setup` once and produces the sandwich SE
// vectors for both `bread = Expected` and `bread = Observed` against the
// same meat — avoids re-stacking Δ, re-Choleskying Γ_NT(M_b), and rebuilding
// WΔ on the second call. The Observed Hessian is the only bread-specific
// cost. `moments` and `cov` apply to both members of the pair (this is the
// typical use case: same Γ̂, different bread). Three overloads to mirror
// the single-bread `robust_se`: caller-supplied Γ̂, raw data, or pre-computed
// casewise contributions Zc.
struct RobustSeBothBreads {
  RobustSeResult expected;   // bread = Expected
  RobustSeResult observed;   // bread = Observed
};

post_expected<RobustSeBothBreads>
robust_se_both_breads(spec::LatentStructure        pt,
                      const model::MatrixRep&   rep,
                      const SampleStats&        samp,
                      const Estimates&          est,
                      const Eigen::MatrixXd&    gamma_hat,
                      WeightMoments             moments = WeightMoments::Structured,
                      ScoreCovariance           cov     = ScoreCovariance::Empirical);

post_expected<RobustSeBothBreads>
robust_se_both_breads(spec::LatentStructure        pt,
                      const model::MatrixRep&   rep,
                      const SampleStats&        samp,
                      const Estimates&          est,
                      const RawData&            raw,
                      WeightMoments             moments = WeightMoments::Structured,
                      ScoreCovariance           cov     = ScoreCovariance::Empirical);

post_expected<RobustSeBothBreads>
robust_se_both_breads(spec::LatentStructure                     pt,
                      const model::MatrixRep&                   rep,
                      const SampleStats&                        samp,
                      const Estimates&                          est,
                      const Eigen::Ref<const Eigen::MatrixXd>&  Zc,
                      double                                    n_total,
                      WeightMoments                             moments = WeightMoments::Structured,
                      ScoreCovariance                           cov     = ScoreCovariance::Empirical);

// ── Parameter-space sandwich terms (for robust score / modification-index tests)
//
// Surfaces the same {bread, meat} that `robust_se` assembles, but as raw q×q
// matrices in θ-space, so a caller can contract a custom efficient-score
// direction g against them: the robust generalized/SB-scaled score statistic is
//   mi_robust = mi_NT · (gᵀA1 g)/(gᵀB1 g)   (== mi_NT / c, c = gᵀB1g / gᵀA1g).
// Both A1 and B1 are PER-UNIT (averaged over N), so c is dimensionless and → 1
// under normality. The meat equals the bread EXACTLY when Γ̂ = Γ_NT (since
// `WΓ_NTW = W`), which is what the `InferenceSpec`-only overload returns.
//
//   A1 = bread (Expected: Δ'WΔ; Observed: H_obs/N), per `spec.bread`.
//   B1 = meat  (Δ'WΓ̂WΔ; model-implied Γ_NT for the no-Γ̂ overload).
//   K_con = npar × q reparameterization (empty ⇒ identity / θ-space).
//
// `reparam_constraints = false` keeps A1/B1 in full npar θ-space even when the
// model has active equality constraints — required by the score-test path,
// where the nuisance projection lives in θ-space (the released constraint's K).
struct ParamSpaceSandwich {
  Eigen::MatrixXd A1;     // q × q  bread
  Eigen::MatrixXd B1;     // q × q  meat
  Eigen::MatrixXd K_con;  // npar × q (empty ⇒ identity)
  Eigen::Index    q = 0;
};

// Model-implied (Γ_NT) meat: B1 = Δ'WΔ. For the Expected bread A1 == B1 (c ≡ 1),
// the exact reduction-to-NT baseline; for the Observed bread B1 = Δ'WΔ ≠ H_obs.
post_expected<ParamSpaceSandwich>
param_space_sandwich(spec::LatentStructure pt, const model::MatrixRep& rep,
                     const SampleStats& samp, const Estimates& est,
                     InferenceSpec spec = {}, bool reparam_constraints = true);

// Empirical meat from raw data (via `casewise_contributions`).
post_expected<ParamSpaceSandwich>
param_space_sandwich(spec::LatentStructure pt, const model::MatrixRep& rep,
                     const SampleStats& samp, const Estimates& est,
                     const RawData& raw, InferenceSpec spec = {},
                     bool reparam_constraints = true);

// Empirical meat from a caller-supplied Γ̂ (σ-only ⇒ pstar; means ⇒ total_rows).
post_expected<ParamSpaceSandwich>
param_space_sandwich(spec::LatentStructure pt, const model::MatrixRep& rep,
                     const SampleStats& samp, const Estimates& est,
                     const Eigen::MatrixXd& gamma_hat, InferenceSpec spec = {},
                     bool reparam_constraints = true);

}  // namespace magmaan::robust
