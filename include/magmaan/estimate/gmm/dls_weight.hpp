#pragma once

#include <Eigen/Core>
#include <vector>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::estimate {

// Browne's distributionally-weighted least squares (DLS) weight builder.
//
// DLS interpolates between the normal-theory GLS weight and the
// asymptotically-distribution-free (ADF / WLS) weight. The two estimators
// differ only in the moment weight `W = Γ⁻¹`, where Γ is the asymptotic
// covariance of the sample moments. DLS mixes the two Γ matrices,
//
//   Γ_DLS = (1 - a)·Γ_NT + a·Γ_ADF,   W_DLS = Γ_DLS⁻¹,
//
// with `a ∈ [0, 1]`: `a = 0` recovers GLS, `a = 1` recovers WLS/ADF.
// Mixing the Γ matrices (rather than the weights) keeps both endpoints exact.
//
// `dls_weight` is an explicit builder: it returns a `gmm::Weight` that the
// caller passes to `fit_gmm` / the LS modification-index and robust-sandwich
// surfaces, exactly like a hand-built WLS weight. It does not change any
// default. Following Browne, the mix applies to the covariance moments only;
// the mean-structure block uses the normal-theory weight regardless of `a`.
struct DlsWeightOptions {
  double a = 0.5;   // mixing scalar in [0, 1]
};

// Empirical-Bayes DLS scalar selection.
//
// Du and Wu (2024) motivate choosing the DLS mixing scalar from the observed
// departure of the empirical fourth-moment Γ_ADF from Γ_NT. This lightweight
// builder treats that departure as a noisy random-effects signal: the finite-
// sample noise of Γ_ADF is estimated from casewise fourth-moment products, and
// the single DLS scalar is the reliability ratio
//
//   a_EB = max(0, ||Γ_ADF - Γ_NT||² - noise) / ||Γ_ADF - Γ_NT||².
//
// Thus near-normal samples shrink toward GLS (`a ≈ 0`), while stable
// non-normal fourth-moment departures move toward ADF/WLS (`a ≈ 1`). The
// scalar is intentionally diagnostic and explicit; no lavaan-compatible
// default is changed.
struct EmpiricalBayesDlsOptions {
  double min_a = 0.0;
  double max_a = 1.0;
};

struct EmpiricalBayesDlsMixingScalar {
  double a = 0.0;
  double observed_delta_norm2 = 0.0;
  double estimated_noise_norm2 = 0.0;
  double signal_norm2 = 0.0;
  std::vector<double> block_observed_delta_norm2;
  std::vector<double> block_estimated_noise_norm2;
};

// Build the per-block DLS weight over the `[mean ; vech(cov)]` moment layout,
// aligned with `gmm::residuals` / `gmm::normal_theory_weight`. `ev` + `theta0`
// supply the moment layout (block dimensions, mean structure); `samp` supplies
// the normal-theory Γ and `raw` the ADF Γ. `raw` must be complete (no
// missingness) and dimensionally consistent with `samp`.
fit_expected<gmm::Weight>
dls_weight(const model::ModelEvaluator& ev, const data::SampleStats& samp,
           const data::RawData& raw, const Eigen::VectorXd& theta0,
           DlsWeightOptions opts = {});

// Estimate the empirical-Bayes DLS mixing scalar from complete raw data and
// sample covariances. The scalar is shared across blocks, matching
// `DlsWeightOptions::a`.
fit_expected<EmpiricalBayesDlsMixingScalar>
empirical_bayes_dls_mixing_scalar(const data::SampleStats& samp,
                                  const data::RawData& raw,
                                  EmpiricalBayesDlsOptions opts = {});

// Convenience builder: estimate `a_EB` and call `dls_weight(...)` with that
// scalar. Use `empirical_bayes_dls_mixing_scalar(...)` first when callers want
// to report the scalar and its diagnostics.
fit_expected<gmm::Weight>
empirical_bayes_dls_weight(const model::ModelEvaluator& ev,
                           const data::SampleStats& samp,
                           const data::RawData& raw,
                           const Eigen::VectorXd& theta0,
                           EmpiricalBayesDlsOptions opts = {});

}  // namespace magmaan::estimate
