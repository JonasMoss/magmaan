#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/data/sample_stats.hpp"

namespace magmaan::data {

// Per-block raw observations matrix — what FIML and the robust ADF /
// Satorra-Bentler machinery need that `SampleStats` alone can't supply.
//
//   X[b] : (n_b × p_b) per-block raw data. Each row is one observation;
//          columns follow the same variable ordering as `SampleStats::S[b]`.
//
// Caller-owned, never mutated by the library. NT consumers (`fit`,
// `information_expected`, `browne_residual_nt`, ...) ignore RawData and continue
// to consume `SampleStats` directly. The two are linked through
// `sample_stats_from_raw(raw)`, which produces moment summaries that
// match what lavaan does internally with `likelihood = "normal"` (N-divisor
// cov, sample mean as the m̄ estimate).
//
// Missingness is represented by `mask` and consumed by FIML. Complete-data
// moment helpers such as `sample_stats_from_raw()` still assert mask is empty
// and treat X as fully observed.
struct RawData {
  // Per-block raw data. X[b] is (n_b × p_b).
  std::vector<Eigen::MatrixXd> X;

  // Optional per-block missingness mask (1 = observed, 0 = missing).
  // Same shape as X[b] when present.
  std::vector<Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>> mask;
};

// Derive `SampleStats` from raw data with the lavaan `likelihood = "normal"`
// moment convention: per block,
//
//   m̄_b   = (1/n_b)   · Σ_i x_i
//   S_b   = (1/n_b)   · Σ_i (x_i − m̄_b)(x_i − m̄_b)ᵀ    -- N-divisor, NOT N−1
//   n_b   = number of rows in X[b]
//
// The N-divisor convention matches lavaan's internal `sample.cov` when
// `likelihood = "normal"`, so our existing post-fit tests stay
// consistent. Callers that already have moments computed with R's
// `cov()` (N−1 divisor) should rescale by `(N−1)/N` before constructing
// `SampleStats` directly — or use this function to get the moments
// straight from raw observations.
//
// Returns `PostError::NumericIssue` if any block has fewer than 2 rows
// (covariance undefined) or zero columns.
post_expected<SampleStats>
sample_stats_from_raw(const RawData& raw);

// Empirical fourth-moment ACOV of vech(S) — the building block for ADF
// estimation, sandwich SE, and Satorra-Bentler scaling. For a single
// block of raw data X (n × p),
//
//   d_i  = vech((x_i − m̄)(x_i − m̄)ᵀ)
//   Γ̂   = (1/n) · Σ_i (d_i − vech(S_N)) (d_i − vech(S_N))ᵀ
//
// where S_N is the N-divisor sample covariance. Result is (p* × p*) with
// p* = p(p+1)/2; vech ordering is lower-triangle column-major (same as
// `dsigma_dtheta` and `browne_residual_nt`).
//
// Under multivariate normality `Γ̂ → Γ_NT = 2 D⁺(Σ ⊗ Σ) D⁺ᵀ` as `n → ∞`;
// the two diverge when the data has heavier tails or kurtosis ≠ 3, which
// is exactly the regime robust SE / Satorra-Bentler scaling addresses.
//
// Returns `PostError::NumericIssue` if `X` has fewer than 2 rows.
post_expected<Eigen::MatrixXd>
empirical_gamma(const Eigen::Ref<const Eigen::MatrixXd>& X);

// Normal-theory ACOV of vech(S) computed from Σ alone:
//
//   Γ_NT[ (i,j), (k,l) ] = σ_ik · σ_jl + σ_il · σ_jk     (i ≤ j, k ≤ l)
//
// Equivalent to `2 · D⁺(Σ ⊗ Σ) D⁺ᵀ` but built directly in (p* × p*)
// without forming the (p² × p*) duplication-matrix pseudoinverse. The
// counterpart to `empirical_gamma(X)`: under multivariate-normal data
// the two converge as `n → ∞`. Used for sandwich SE and Satorra-Bentler
// scaling as the model-implied weight matrix.
//
// Returns `PostError::NumericIssue` if `Sigma` is not square.
post_expected<Eigen::MatrixXd>
gamma_nt(const Eigen::Ref<const Eigen::MatrixXd>& Sigma);

}  // namespace magmaan::data
