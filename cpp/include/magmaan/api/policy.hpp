#pragma once

// magmaan's inference policy: the one set of choices the ordinary-user
// package applies by default (project/design/r-interface-vision.md). Every
// component is either computed under the policy or carries a reason; no
// component is ever replaced by a result under another convention.

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/frontier/multiinfo_penalty.hpp"
#include "magmaan/robust/prepared_ntml.hpp"

namespace magmaan::api {

enum class InferenceReason {
  Available,
  NotConverged,      // the fit did not pass its convergence verdict
  Saturated,         // zero degrees of freedom: no global test exists
  UnsupportedModel,  // outside the policy's model class (e.g. fixed x)
  NumericFailure,    // singular information or a failed decomposition
  NotNested,         // the null is not contained in the alternative
  UnsupportedNesting, // nesting through moments requires an unsupported correspondence
  BoundaryNesting,   // no regular interior nested-test reference
  Penalized,         // a penalized (barrier) estimate: no validated sampling contract
  Inapplicable,      // the component does not exist for this estimator (an LR test without a likelihood)
};

std::string_view reason_name(InferenceReason reason) noexcept;

// The fit-level facts that gate inference. `policy_fit_state()` derives them
// from the estimates; bindings that rebuild estimates without their
// diagnostics supply them directly.
struct PolicyFitState {
  bool converged = true;
  bool psd_boundary = false;
  // magmaan's own convergence check when a non-native acceptance rule (a
  // compatibility preset) decided `converged`; unset when magmaan's check
  // decided or did not run. Inference follows `converged` either way.
  std::optional<bool> native_converged = {};
  // The estimate maximizes a penalized likelihood with a positive weight.
  // Its covariance and tests need a penalized-estimating-equation and
  // sampling-law contract that is not validated, and inference for the
  // unpenalized estimator does not apply, so every component is unavailable.
  bool penalized = false;
};

PolicyFitState policy_fit_state(const estimate::Estimates& estimates);
// A barrier fit. Weight zero is the unpenalized criterion.
PolicyFitState policy_fit_state(const estimate::frontier::PenalizedFit& fit);

// Why a penalized estimate has no policy inference; shared by every binding.
inline constexpr std::string_view penalized_detail =
    "the estimate maximizes a penalized likelihood (barrier); inference for it is "
    "not validated, and inference for the unpenalized estimator does not apply";

// The selected acceptance rule and magmaan's own check disagree. Nothing is
// recomputed for it: callers report the disagreement next to results that
// follow the selected rule.
bool verdict_disagreement(const PolicyFitState& state) noexcept;

// One test statistic with its calibrations: a global test against the
// saturated model, or a nested test of a null against an alternative.
struct PolicyTest {
  InferenceReason reason = InferenceReason::Available;
  std::string detail;
  double statistic = std::numeric_limits<double>::quiet_NaN();
  int df = 0;
  // Mean eigenvalue of the UGamma spectrum: the SB scaling divisor.
  double sb_scale = std::numeric_limits<double>::quiet_NaN();
  double p_sb = std::numeric_limits<double>::quiet_NaN();
  double p_peba4 = std::numeric_limits<double>::quiet_NaN();
  int peba_blocks = 0; // actual nonempty PEBA4 blocks; zero when unavailable
  Eigen::VectorXd eigenvalues;  // ascending, length df
  // What the statistic is when the slot name does not say it: "fit_function"
  // for a least-squares global test, whose score statistic equals n F.
  std::string label;
};

struct PolicyInference {
  InferenceReason covariance_reason = InferenceReason::Available;
  std::string covariance_detail;
  Eigen::MatrixXd covariance;  // free parameters, total-sample scale
  PolicyTest score, lr;
  // The fit is a PSD-constrained estimate on the cone boundary. Inference is
  // still computed there: when the population is interior, the PSD and
  // ordinary estimators coincide with probability tending to one, so the
  // regular limits apply. Callers report that assumption.
  bool psd_boundary = false;
  bool verdict_disagreement = false;  // see verdict_disagreement()
};

// Nested tests of `null` against `alternative`, which must be fits to one
// prepared dataset, with the null dropping, fixing or constraining alternative
// paths (or nesting through an interior moment parameterization). The
// likelihood-ratio statistic is the difference of the two fit statistics;
// the score uses H1 at the embedded null, whose restriction directions are
// projected against the lifted null nuisance tangent, so it keeps its
// meaning at a PSD boundary null. Both use observed geometry: the score
// projects with the observed information at the null fit and keeps the
// expected metric on the projected directions (the FIML recipe), and the
// Satorra-2000 LR spectrum reduces through the observed information at the
// alternative. Each is calibrated with SB and PEBA4 from its spectrum.
struct PolicyNested {
  PolicyTest score, lr;
  bool psd_boundary = false;  // either fit is a PSD estimate on the boundary
  bool verdict_disagreement = false;  // for either fit
};

// All components unavailable for one reason.
PolicyInference policy_unavailable(InferenceReason reason, std::string detail);

// Complete-data normal-theory ML.
//
// Parameter covariance: the sandwich V (sum_i s_i s_i') V with V the inverse
// observed information and s_i the exact casewise likelihood scores at the
// fitted point (robust::frontier::ntml_score_sandwich).
//
// Global tests against the saturated model: the score statistic and the
// likelihood-ratio statistic, each calibrated with SB (mean scaling) and
// PEBA4 from the UGamma spectrum. Both use the shared expected-information
// geometry; under the global null the observed and expected information
// differ by O_p(n^-1/2), and the SB and PEBA4 evidence behind the policy is
// for this geometry.
// A fully specified model has an empty parameter covariance and tests every
// saturated mean/covariance direction; a saturated model has no global test.
PolicyInference policy_inference_ml(robust::frontier::NTMLFit& fit,
                                    const PolicyFitState& state);

// All-ordinal DWLS, delta or theta parameterization, one or more groups.
//
// Parameter covariance: the infinitesimal-jackknife sandwich
// (estimate::robust_ordinal_ij): observed bread, Stage-1 threshold and
// polychoric influence, and the influence of the estimated diagonal weight,
// which is leading order when the model is misspecified and vanishes at
// exact fit.
//
// Global test against the saturated model: the fit-function statistic n F
// with the fixed-weight UGamma spectrum (estimate::robust_ordinal), calibrated
// with SB and PEBA4. The objective is quadratic in the saturated moments and
// the weight influence vanishes under the global null, so the global score
// statistic equals n F; it is reported once, in `score`, labelled
// "fit_function". DWLS has no likelihood, so `lr` is Inapplicable. Only plain
// DWLS fits (weight diag(NACOV)^-1) qualify; a different weight fails the IJ
// recipe check and is reported as UnsupportedModel.
PolicyInference policy_inference_dwls(spec::LatentStructure pt,
                                      const model::MatrixRep& rep,
                                      const data::OrdinalStats& stats,
                                      const estimate::Estimates& estimates,
                                      estimate::OrdinalParameterization parameterization,
                                      const PolicyFitState& state,
                                      const std::vector<std::int8_t>* row_user = nullptr);

// Nested tests for two all-ordinal DWLS fits to one dataset (`stats`), with
// `null` restricting `alternative`. Both fits must be plain DWLS.
//
// The `lr` slot holds the fit-function difference T = n (F_null - F_alt),
// labelled "fit_function_difference" (DWLS has no likelihood), with the
// estimated-weight profile reference law (estimate::ordinal_dwls_profile_lrt):
// each model's profile at its own estimate over the extended first-stage
// moments (thresholds, polychorics and the diagonal of the DWLS weight), so the
// reference stays valid when the larger model is misspecified (evidence 13).
// At exact fit the weight channel is dormant and the spectrum is the
// fixed-weight Satorra-2000 one. `eigenvalues` is the positive profile
// spectrum (values below 1e-8 of the largest count as zero), padded with
// zeros to at least the restriction df; SB scales by its
// trace over the restriction df (mean matching over every term, which a top-df
// truncation would lose) and PEBA4 uses the whole spectrum. No nested DWLS
// score test is derived, so `score` is UnsupportedModel. Nesting is verified
// with the shared restriction embedding.
PolicyNested policy_nested_dwls(spec::LatentStructure null_pt,
                                const model::MatrixRep& null_rep,
                                const estimate::Estimates& null_estimates,
                                const PolicyFitState& null_state,
                                spec::LatentStructure alternative_pt,
                                const model::MatrixRep& alternative_rep,
                                const estimate::Estimates& alternative_estimates,
                                const PolicyFitState& alternative_state,
                                const data::OrdinalStats& stats,
                                estimate::OrdinalParameterization parameterization);

PolicyNested policy_nested_ml(std::shared_ptr<robust::frontier::NTMLFit> null,
                              const PolicyFitState& null_state,
                              std::shared_ptr<robust::frontier::NTMLFit> alternative,
                              const PolicyFitState& alternative_state);

}  // namespace magmaan::api
