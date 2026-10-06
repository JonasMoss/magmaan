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
#include <vector>

#include <Eigen/Core>

#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/fiml.hpp"
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
  EquivalentModels, // equal moment manifolds: no positive-df comparison
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

class Fit;
struct PolicyFitIndex {
  std::string index;
  double estimate = std::numeric_limits<double>::quiet_NaN();
  InferenceReason reason = InferenceReason::Available;
  std::string detail;
};
struct PolicyFitDiscrepancy {
  double discrepancy = std::numeric_limits<double>::quiet_NaN();
  double trace = std::numeric_limits<double>::quiet_NaN();
  double corrected = std::numeric_limits<double>::quiet_NaN();
  int df = 0;
};
struct PolicyFitMeasures {
  std::vector<PolicyFitIndex> indices;
  PolicyFitDiscrepancy user, baseline;
  std::int64_t ntotal = 0;
  std::size_t n_groups = 0;
  double residual_uncorrected = std::numeric_limits<double>::quiet_NaN();
  double residual_trace = std::numeric_limits<double>::quiet_NaN();
};
// Point estimates only. Natural pooled discrepancy, N divisor, Steiger's G
// factor for RMSEA, nominal-df truncated TLI. Independence is unconstrained
// across groups and uses the same data treatment and estimator.
PolicyFitMeasures policy_fit_measures(const Fit& fit);
PolicyFitMeasures policy_fit_measures_unavailable(InferenceReason reason,
    std::string detail, bool ordinal = false, bool likelihood = true);
PolicyFitMeasures policy_fit_measures(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::Estimates& est, const PolicyFitState& state, bool fiml = false, bool uls = false);
PolicyFitMeasures policy_fit_measures(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::OrdinalStats& stats,
    const estimate::Estimates& est, const PolicyFitState& state,
    estimate::OrdinalParameterization parameterization = estimate::OrdinalParameterization::Delta);
PolicyFitMeasures policy_fit_measures(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::MixedOrdinalStats& stats,
    const estimate::Estimates& est, const PolicyFitState& state,
    estimate::OrdinalParameterization parameterization = estimate::OrdinalParameterization::Delta);

PolicyFitMeasures policy_fit_measures_two_stage(spec::LatentStructure pt,
    const model::MatrixRep& rep, const estimate::Estimates& est,
    const estimate::SaturatedMoments& stage1, const PolicyFitState& state);

// One test statistic with its calibrations: a global test against the
// saturated model, or a nested test of a null against an alternative.
struct PolicyTest {
  InferenceReason reason = InferenceReason::Available;
  std::string detail;
  double statistic = std::numeric_limits<double>::quiet_NaN();
  int df = 0;
  // Mean eigenvalue of the UGamma spectrum: the SB scaling divisor.
  double sb_scale = std::numeric_limits<double>::quiet_NaN();
  // Default reporting law; SB remains a comparator for ML/FIML.
  // DWLS selects All for global and nested tests; SB/PEBA4 remain comparators.
  std::string reference = "peba4";
  double p_all = std::numeric_limits<double>::quiet_NaN();
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

// Owning evaluation-point snapshots. Inputs cannot be edited or combined with
// another fit after construction; lazily retained ingredients include failures.
// As with NTMLFit, a snapshot is used from one thread at a time.
class FimlPolicyFit {
 public:
  FimlPolicyFit(spec::LatentStructure pt, model::MatrixRep rep, data::RawData raw,
      estimate::fiml::FIMLPack pack, estimate::Estimates estimates);
  struct Impl;
  const std::shared_ptr<Impl> impl;
};
class DwlsPolicyFit {
 public:
  DwlsPolicyFit(spec::LatentStructure pt, model::MatrixRep rep,
      data::OrdinalStats stats, estimate::Estimates estimates,
      estimate::OrdinalParameterization parameterization,
      std::vector<std::int8_t> row_user = {});
  struct Impl;
  const std::shared_ptr<Impl> impl;
};
PolicyInference policy_inference_fiml(FimlPolicyFit& fit, const PolicyFitState& state);
PolicyNested policy_nested_fiml(FimlPolicyFit& null, const PolicyFitState& null_state,
    FimlPolicyFit& alternative, const PolicyFitState& alternative_state);
PolicyInference policy_inference_dwls(DwlsPolicyFit& fit, const PolicyFitState& state);
PolicyNested policy_nested_dwls(DwlsPolicyFit& null, const PolicyFitState& null_state,
    DwlsPolicyFit& alternative, const PolicyFitState& alternative_state);
class MixedDwlsPolicyFit {
 public:
  MixedDwlsPolicyFit(spec::LatentStructure pt, model::MatrixRep rep,
      data::MixedOrdinalStats stats, estimate::Estimates estimates,
      estimate::OrdinalParameterization parameterization,
      std::vector<std::int8_t> row_user = {});
  struct Impl;
  const std::shared_ptr<Impl> impl;
};
PolicyInference policy_inference_dwls(MixedDwlsPolicyFit& fit, const PolicyFitState& state);
PolicyNested policy_nested_dwls(MixedDwlsPolicyFit& null, const PolicyFitState& null_state,
    MixedDwlsPolicyFit& alternative, const PolicyFitState& alternative_state);
// Number of expensive evaluation-point builds, for reuse diagnostics.
std::size_t policy_ingredient_builds(const FimlPolicyFit& fit);
std::size_t policy_ingredient_builds(const DwlsPolicyFit& fit);
std::size_t policy_ingredient_builds(const MixedDwlsPolicyFit& fit);

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

// Observed-data FIML. Score sensitivity is observed at H0, its metric is
// pattern-conditional expected information, and its meat consists of direct
// casewise likelihood scores at H0. Global LR uses saturated observed H1.
PolicyInference policy_inference_fiml(spec::LatentStructure pt,
    const model::MatrixRep& rep, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack, const estimate::Estimates& estimates,
    const PolicyFitState& state);

// Nested LR uses observed bread and empirical casewise score meat at the
// larger fit, with the exact restriction map; no transported H1 influence.
PolicyNested policy_nested_fiml(spec::LatentStructure null_pt,
    const model::MatrixRep& null_rep, const estimate::Estimates& null_estimates,
    const PolicyFitState& null_state, spec::LatentStructure alternative_pt,
    const model::MatrixRep& alternative_rep,
    const estimate::Estimates& alternative_estimates,
    const PolicyFitState& alternative_state, const data::RawData& raw,
    const estimate::fiml::FIMLPack& pack);

// All-ordinal DWLS, delta or theta parameterization, one or more groups.
//
// Parameter covariance: the infinitesimal-jackknife sandwich
// (estimate::robust_ordinal_ij, Exact first stage): observed bread, threshold and
// polychoric influence, and the influence of the estimated diagonal weight,
// which is leading order when the model is misspecified and vanishes at
// exact fit.
//
// Global test against the saturated model: the fit-function statistic n F
// with UGamma from exact sampling rows and unchanged OPG fitting weights, calibrated
// with the exact weighted chi-square All tail on every positive sample
// eigenvalue (decision study 05; limited validation, confirmation pending).
// The objective is quadratic in the saturated moments and
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
// parameter-space estimated-weight IJ law at the larger fit: observed Hessian
// H, IJ meat B, and exact restriction map A, with df_diff eigenvalues of
// (A H^-1 A')^-1 A H^-1 B H^-1 A'. SB and PEBA4 calibrate this r-term spectrum.
// Validation is limited; fresh-seed confirmation is pending. The separately
// fitted ordinal_dwls_profile_lrt remains an explicitly named lab comparator.
// At exact fit the weight channel is dormant, giving fixed-weight Satorra-2000.
// No nested DWLS score test is derived, so `score` is UnsupportedModel.
// Parameter nesting uses the shared restriction embedding; failed NotNested
// embeddings fall back to an implied-moment embedding and null tangent T.
// The same sandwich law uses a row basis A annihilating T. Moment nesting has
// limited numerical validation; calibration is pending. Zero-df equivalence
// reports EquivalentModels.
PolicyNested policy_nested_dwls(spec::LatentStructure null_pt,
                                const model::MatrixRep& null_rep,
                                const estimate::Estimates& null_estimates,
                                const PolicyFitState& null_state,
                                spec::LatentStructure alternative_pt,
                                const model::MatrixRep& alternative_rep,
                                const estimate::Estimates& alternative_estimates,
                                const PolicyFitState& alternative_state,
                                const data::OrdinalStats& stats,
                                estimate::OrdinalParameterization parameterization,
                                const std::vector<std::int8_t>* null_row_user = nullptr,
                                const std::vector<std::int8_t>* alternative_row_user = nullptr);

// Mixed DWLS uses exact empirical first-stage sampling rows for the global
// spectrum, retaining OPG NACOV fitting weights. Covariance and nested law
// include estimated-weight influence. Limited validation; calibration pending.
PolicyInference policy_inference_dwls(spec::LatentStructure pt,
                                      const model::MatrixRep& rep,
                                      const data::MixedOrdinalStats& stats,
                                      const estimate::Estimates& estimates,
                                      estimate::OrdinalParameterization parameterization,
                                      const PolicyFitState& state,
                                      const std::vector<std::int8_t>* row_user = nullptr);
PolicyNested policy_nested_dwls(spec::LatentStructure null_pt,
                                const model::MatrixRep& null_rep,
                                const estimate::Estimates& null_estimates,
                                const PolicyFitState& null_state,
                                spec::LatentStructure alternative_pt,
                                const model::MatrixRep& alternative_rep,
                                const estimate::Estimates& alternative_estimates,
                                const PolicyFitState& alternative_state,
                                const data::MixedOrdinalStats& stats,
                                estimate::OrdinalParameterization parameterization,
                                const std::vector<std::int8_t>* null_row_user = nullptr,
                                const std::vector<std::int8_t>* alternative_row_user = nullptr);

namespace frontier {
// Local moment embedding and null tangent in the alternative's equality-
// reduced coordinates. Weighted residuals use the same sample and DWLS weight.
// A is an orthonormal row basis annihilating tangent; embedding reproduces
// the null fit's implied moments. This is a numerical local inclusion witness.
struct MomentNestedTangent {
  Eigen::VectorXd embedding;
  Eigen::MatrixXd tangent;
  Eigen::MatrixXd A;
};
post_expected<MomentNestedTangent> moment_nested_tangent(
    spec::LatentStructure null_pt, const model::MatrixRep& null_rep,
    const estimate::Estimates& null_estimates,
    spec::LatentStructure alternative_pt, const model::MatrixRep& alternative_rep,
    const estimate::Estimates& alternative_estimates, const data::OrdinalStats& stats,
    estimate::OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* null_row_user = nullptr,
    const std::vector<std::int8_t>* alternative_row_user = nullptr);
post_expected<MomentNestedTangent> moment_nested_tangent(
    spec::LatentStructure null_pt, const model::MatrixRep& null_rep,
    const estimate::Estimates& null_estimates,
    spec::LatentStructure alternative_pt, const model::MatrixRep& alternative_rep,
    const estimate::Estimates& alternative_estimates, const data::MixedOrdinalStats& stats,
    estimate::OrdinalParameterization parameterization,
    const std::vector<std::int8_t>* null_row_user = nullptr,
    const std::vector<std::int8_t>* alternative_row_user = nullptr);
} // namespace frontier

PolicyNested policy_nested_ml(std::shared_ptr<robust::frontier::NTMLFit> null,
                              const PolicyFitState& null_state,
                              std::shared_ptr<robust::frontier::NTMLFit> alternative,
                              const PolicyFitState& alternative_state);

}  // namespace magmaan::api
