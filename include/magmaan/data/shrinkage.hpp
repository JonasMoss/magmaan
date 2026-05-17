#pragma once

#include <cstdint>
#include <vector>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::data {

// Covariance shrinkage as an explicit sample-moment transformation.
//
// `shrink_sample_stats` takes a `SampleStats` and returns a shrunk copy plus
// per-block repair diagnostics. It is an opt-in transformation: a fit that
// wants shrinkage calls this first and passes the result to `fit_*`. Nothing
// here changes a default — passing the raw `SampleStats` straight to a fitter
// is unaffected.
//
// Shrinkage replaces the sample covariance `S` with a convex combination
// `S' = (1 - s)·S + s·T` of `S` and a structured, well-conditioned target
// `T`. It trades a little bias for a large variance reduction and lifts the
// smallest eigenvalue away from zero — useful for small-N / near-singular `S`.

enum class CovarianceShrinkageKind : std::uint8_t {
  None,                 // S' = S (diagnostics only)
  Ridge,                // T = I            (best on correlation-scale input)
  IdentityTarget,       // T = (tr S / p)·I (Ledoit-Wolf scaled identity)
  DiagonalTarget,       // T = diag(S)      (shrinks covariances toward 0)
  ConstantCorrelation,  // T has the average sample correlation off-diagonal
};

struct CovarianceShrinkageOptions {
  CovarianceShrinkageKind kind = CovarianceShrinkageKind::None;

  // Shrinkage intensity `s ∈ [0, 1]` (clamped). Used unless
  // `estimate_intensity` is set. `s = 0` ⇒ no change, `s = 1` ⇒ the target.
  double intensity = 0.0;

  // Estimate the intensity from the data instead of using `intensity`: the
  // normal-theory Ledoit-Wolf / Schäfer-Strimmer optimal value. Supported for
  // `DiagonalTarget` only; any other kind returns an explicit error.
  bool estimate_intensity = false;
};

// Per-block repair diagnostics — what the shrinkage did to the conditioning.
struct CovarianceRepairResult {
  double raw_min_eigen = 0.0;   // smallest eigenvalue of the original S
  double min_eigen     = 0.0;   // smallest eigenvalue of the shrunk S'
  double intensity     = 0.0;   // shrinkage intensity actually applied
  bool   shrunk        = false; // true when a non-zero intensity was applied
};

struct ShrunkSampleStats {
  SampleStats stats;                                  // shrunk sample moments
  std::vector<CovarianceRepairResult> block_diagnostics;
};

// Apply covariance shrinkage to every block of `samp`. Means and `n_obs` pass
// through unchanged. Returns `NumericIssue` on a malformed input or an
// unsupported `estimate_intensity` combination.
post_expected<ShrunkSampleStats>
shrink_sample_stats(const SampleStats& samp, CovarianceShrinkageOptions opts);

}  // namespace magmaan::data
