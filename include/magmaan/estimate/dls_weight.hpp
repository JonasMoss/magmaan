#pragma once

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/gmm/moment_quadratic.hpp"
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

// Build the per-block DLS weight over the `[mean ; vech(cov)]` moment layout,
// aligned with `gmm::residuals` / `gmm::normal_theory_weight`. `ev` + `theta0`
// supply the moment layout (block dimensions, mean structure); `samp` supplies
// the normal-theory Γ and `raw` the ADF Γ. `raw` must be complete (no
// missingness) and dimensionally consistent with `samp`.
fit_expected<gmm::Weight>
dls_weight(const model::ModelEvaluator& ev, const data::SampleStats& samp,
           const data::RawData& raw, const Eigen::VectorXd& theta0,
           DlsWeightOptions opts = {});

}  // namespace magmaan::estimate
