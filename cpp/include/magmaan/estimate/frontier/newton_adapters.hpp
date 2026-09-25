#pragma once

#include "magmaan/estimate/frontier/newton_accuracy.hpp"
#include "magmaan/estimate/frontier/multiinfo_penalty.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/twolevel.hpp"

namespace magmaan::estimate::frontier {

// Numerical curvature is differentiated from the actual objective gradient.
// Both h and h/2 central differences must be evaluable; steps may shrink near
// a domain boundary, but no one-sided or Gauss-Newton fallback is substituted.
struct NewtonDifferenceOptions {
  double relative_step = 1e-4;
  double max_relative_error = 1e-3;
  int max_shrink = 12;
  Eigen::VectorXd parameter_scales; // empty => max(1, abs(theta_i))
};
struct NewtonAdapterOptions {
  NewtonAccuracyOptions accuracy;
  NewtonDifferenceOptions differences;
  StationarityDomain domain = StationarityDomain::Ambient;
  Bounds bounds; // empty means unbounded; never inferred from the estimator
  double active_bound_tol = 1e-6;
  // LS adapters only; otherwise rejected. Replaces the analytic Hessian of
  // the moment-quadratic adapters by J'WJ, which cannot detect a saddle.
  bool gauss_newton = false;
};

// Low-level derivative adapter for an ORIGINAL full-theta objective. The
// native objective and gradient are multiplied by native_to_total; objective
// in the returned record is total/n_obs. Closures are consumed synchronously.
NewtonDerivatives evaluate_newton_objective(
    const optim::ScalarProblem& problem, const Eigen::VectorXd& theta,
    double n_obs, double native_to_total,
    NewtonObjectiveKind kind = NewtonObjectiveKind::Supplied,
    NewtonDifferenceOptions options = {});

// Analytic derivatives of the moment-quadratic objective (ULS, GLS, WLS, DWLS,
// fixed-weight GMM) at theta: the exact Hessian and, as the sandwich metric,
// the normal-theory variance of the total gradient. `ev` must carry the
// resolved (fixed.x) partable. Fit finalization and the LS adapters share it.
NewtonDerivatives evaluate_newton_moment_quadratic(
    const model::ModelEvaluator& ev, const SampleStats& sample,
    const Eigen::VectorXd& theta, const gmm::Weight& weight = {});

// Each adapter returns owning artifacts without fitting or changing a fit's
// stored verdict. Build/input errors are expected errors; numerical curvature
// failures remain inspectable artifacts with non-Available status. Likelihood
// adapters measure the step with the observed information and the
// moment-quadratic adapters with the normal-theory sandwich, both in
// standard-error units; finite-difference and Gauss-Newton curvature keep the
// objective's own Hessian as metric. No penalty is omitted from a penalized
// objective.
fit_expected<NewtonAudit> audit_newton_objective(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const optim::ScalarProblem& problem, const Eigen::VectorXd& theta,
    double n_obs, double native_to_total, NewtonAdapterOptions options = {});

// Fixed-weight full-model LS. Empty weight = ULS; full/diagonal supplied
// weights = WLS/DWLS/GMM. For fitted-weight GMM pass the FINAL frozen weight,
// exactly as in its estimating equation, not a function of perturbed theta.
fit_expected<NewtonAudit> audit_newton_gmm(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const SampleStats& sample, const Eigen::VectorXd& theta,
    const gmm::Weight& weight = {}, NewtonAdapterOptions options = {});
// Reconstruct the final expected-information weight once, then hold it fixed
// throughout differentiation. This does not audit the outer weight iteration.
fit_expected<NewtonAudit> audit_newton_gmm_fitted_weight(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const SampleStats& sample, const Eigen::VectorXd& theta,
    GmmFittedWeightKind kind = GmmFittedWeightKind::ExpectedInformation,
    NewtonAdapterOptions options = {});

// Reuse supplied Stage-1 moments without running EM. Nt audits Stage-2 ML;
// all other kinds audit the corresponding fixed-weight moment quadratic.
// This does not establish Stage-1 convergence or sampling accuracy.
fit_expected<NewtonAudit> audit_newton_ml2s(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const fiml::SaturatedMoments& stage1, const Eigen::VectorXd& theta,
    fiml::TwoStageWeight kind = fiml::TwoStageWeight::Nt,
    fiml::TwoStageDlsOptions dls = {}, NewtonAdapterOptions options = {});

fit_expected<NewtonAudit> audit_newton_uls(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const SampleStats& sample, const Eigen::VectorXd& theta,
    NewtonAdapterOptions options = {});
fit_expected<NewtonAudit> audit_newton_gls(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const SampleStats& sample, const Eigen::VectorXd& theta,
    NewtonAdapterOptions options = {});
fit_expected<NewtonAudit> audit_newton_wls(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const SampleStats& sample, const Eigen::VectorXd& theta,
    const gmm::Weight& weight, NewtonAdapterOptions options = {});
// SNLLS uses the expanded theta and the original LS weight; this checks all
// eliminated coordinates too. GLS-SNLLS uses audit_newton_gls at expanded theta.
fit_expected<NewtonAudit> audit_newton_snlls(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const SampleStats& sample, const Eigen::VectorXd& theta,
    const gmm::Weight& weight = {}, NewtonAdapterOptions options = {});

fit_expected<NewtonAudit> audit_newton_fiml(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::RawData& raw, const fiml::FIMLPack& pack,
    const Eigen::VectorXd& theta, NewtonAdapterOptions options = {});
fit_expected<NewtonAudit> audit_newton_ordinal(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::OrdinalStats& stats, const Eigen::VectorXd& theta,
    OrdinalWeightKind weights = OrdinalWeightKind::DWLS,
    OrdinalParameterization parameterization = OrdinalParameterization::Delta,
    NewtonAdapterOptions options = {});
fit_expected<NewtonAudit> audit_newton_mixed_ordinal(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::MixedOrdinalStats& stats, const Eigen::VectorXd& theta,
    OrdinalWeightKind weights = OrdinalWeightKind::DWLS,
    OrdinalParameterization parameterization = OrdinalParameterization::Delta,
    NewtonAdapterOptions options = {});
fit_expected<NewtonAudit> audit_newton_catml(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::OrdinalStats& stats, const Eigen::VectorXd& theta,
    NewtonAdapterOptions options = {});
fit_expected<NewtonAudit> audit_newton_twolevel(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::ClusterSampleStats& stats, const Eigen::VectorXd& theta,
    NewtonAdapterOptions options = {});
fit_expected<NewtonAudit> audit_newton_penalized_ml(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const SampleStats& sample, const Eigen::VectorXd& theta,
    MultiInfoPenaltyOptions penalty = {}, NewtonAdapterOptions options = {});
fit_expected<NewtonAudit> audit_newton_penalized_fiml(
    spec::LatentStructure pt, const model::MatrixRep& rep,
    const data::RawData& raw, const fiml::FIMLPack& pack,
    const Eigen::VectorXd& theta, MultiInfoPenaltyOptions penalty = {},
    NewtonAdapterOptions options = {});

} // namespace magmaan::estimate::frontier
