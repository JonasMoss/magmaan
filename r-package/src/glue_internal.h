// Private types and declarations shared by the topical Rcpp fitting glue.
// Definitions live in glue_util.cpp; topic-only helpers stay in their own TU.
#pragma once

#include "internal.h"
#include "ntml_snapshot.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "magmaan/estimate/coordinates.hpp"
#include "magmaan/estimate/configured_ml.hpp"
#include "magmaan/estimate/resolve_fixed_x.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/diagnostics.hpp"
#include "magmaan/estimate/evaluate.hpp"
#include "magmaan/estimate/frontier/ml_psd_fallback.hpp"
#include "magmaan/estimate/frontier/multiinfo_penalty.hpp"
#include "magmaan/estimate/frontier/sphere.hpp"
#include "magmaan/estimate/frontier/newton_accuracy.hpp"
#include "magmaan/estimate/frontier/newton_adapters.hpp"
#include "magmaan/estimate/frontier/convergence.hpp"
#include "magmaan/estimate/frontier/ml2s_audit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/frontier/shrinkage.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/measures/effects.hpp"
#include "magmaan/measures/factor_scores.hpp"
#include "magmaan/measures/composite_weights.hpp"
#include "magmaan/measures/standardized.hpp"
#include "magmaan/measures/residuals.hpp"
#include "magmaan/measures/reliability.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/twolevel.hpp"
#include "magmaan/estimate/frontier/rbm.hpp"
#include "magmaan/estimate/frontier/sam.hpp"
#include "magmaan/estimate/frontier/pairwise.hpp"
#include "magmaan/estimate/frontier/communality.hpp"
#include "magmaan/estimate/frontier/noniterative_cfa.hpp"
#include "magmaan/robust/frontier/noniterative_inference.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/gmm/dls_weight.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/inference/score.hpp"

namespace magmaanr::fitglue {

using FimlPack = magmaan::estimate::fiml::FIMLPack;

using FimlH1 = magmaan::estimate::fiml::FIMLH1;

using SaturatedMoments = magmaan::estimate::fiml::SaturatedMoments;

// The fit's fixed-weight recipe. fit_model() and estimate() record it as
// fit$moment_weight ("custom" for a supplied W) and, for DLS, fit$stage2_dls_a.
// The computational label cannot carry it: DWLS, DLS and supplied-W fits all
// fit as "WLS". Fits without the record (the direct lab fitters) fall back to
// the label's recipe; the C++ recipe guard then refuses a fitting weight that
// recipe does not rebuild.
struct RecordedMomentWeight {
  magmaan::estimate::gmm::FixedWeightKind kind =
      magmaan::estimate::gmm::FixedWeightKind::Uls;
  bool supplied = false;
  double dls_a = 0.5;
};

magmaan::estimate::FittingOptions fitting_options_from(const Rcpp::List& value);
Rcpp::List fitting_report_to_r(const magmaan::estimate::FittingReport& report);
std::string start_name_from_arg(Rcpp::Nullable<Rcpp::String> start,
                                const char* caller,
                                const char* default_name);
magmaan::estimate::StartMethod start_method(const std::string& name);
const char* start_method_name(magmaan::estimate::StartMethod method);
magmaan::estimate::StartTransport start_transport_from_arg(const std::string& mode);
Rcpp::List start_result_to_r(const magmaan::estimate::StartValues& value);
Eigen::VectorXd start_values_or_stop(Ctx& ctx,
    const magmaan::spec::Starts& starts, const std::string& default_name = "fabin3",
    std::string* applied_policy = nullptr, std::string* fallback_reason = nullptr,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue, bool normalize = false);
Eigen::VectorXd ordinal_starts_or_stop(const Ctx& ctx,
                                       const magmaan::data::OrdinalStats& stats,
                                       const magmaan::spec::Starts& starts,
                                       Rcpp::Nullable<Rcpp::List> control = R_NilValue);
Eigen::VectorXd mixed_ordinal_starts_or_stop(
    const Ctx& ctx, const magmaan::data::MixedOrdinalStats& stats,
    const magmaan::spec::Starts& starts);
Rcpp::List names_list(const std::vector<std::vector<std::string>>& nn);
Rcpp::List matrix_blocks_to_r(const std::vector<Eigen::MatrixXd>& blocks,
                              const std::vector<std::vector<std::string>>& names,
                              bool square_names);
Rcpp::List vector_blocks_to_r(const std::vector<Eigen::VectorXd>& blocks,
                              const std::vector<std::vector<std::string>>& names);
Rcpp::NumericVector residual_rms_to_r(const magmaan::measures::ResidualRms& r);
Rcpp::List residual_summary_to_r(
    const std::vector<magmaan::measures::ResidualSummary>& summ);
Rcpp::List standardized_residuals_to_r(
    const magmaan::measures::StandardizedResiduals& r,
    const std::vector<std::vector<std::string>>& ov_names);
const char* fit_check_to_r(magmaan::estimate::FitCheck check);
int common_converged_value(const magmaan::estimate::Estimates& est);
Rcpp::LogicalVector common_converged_to_r(
    const magmaan::estimate::Estimates& est);
Rcpp::List common_verdict_to_r(const magmaan::estimate::FitDiagnostics& d);
const char* optim_status_to_r(magmaan::optim::OptimStatus status);
const char* score_candidate_kind_str(
    magmaan::inference::ScoreCandidateKind kind);
magmaan::inference::ScoreInformation score_information_from(
    const std::string& information);
magmaan::inference::ScoreCandidateSet score_candidates_from(
    const std::string& candidates);
magmaan::inference::ModificationIndexOptions modification_options_from(
    const std::string& information, const std::string& candidates,
    bool include_loadings, bool include_covariances);
magmaan::estimate::OrdinalWeightKind ordinal_weight_from_estimator(
    const std::string& estimator, const char* call);
magmaan::measures::frontier::reliability::OmegaTarget
omega_target_from_string(const std::string& target, const char* call);
magmaan::estimate::frontier::OrdinalStage2Weight
ordinal_stage2_weight_from_string(const std::string& s);
std::string ordinal_weight_for_postfit(Rcpp::List fit,
                                       const std::string& estimator);
std::string ordinal_weight_key_from_arg(const std::string& weight,
                                        Rcpp::List fit,
                                        const std::string& estimator);
Rcpp::List stats_from_fit_or_arg(Rcpp::List fit, SEXP arg,
                                 const char* field, const char* call);
Rcpp::List audit_to_r(const magmaan::optim::TerminalAudit& a);
Rcpp::List covariance_blocks_to_r(
    const std::vector<magmaan::estimate::CovarianceBlockDiagnostics>& blocks);
Rcpp::List admissibility_to_r(
    const magmaan::estimate::AdmissibilityDiagnostics& d);
Rcpp::List geometric_stationarity_to_r(
    const magmaan::estimate::GeometricStationarityDiagnostics& d);
Rcpp::List newton_accuracy_to_r(
    const magmaan::estimate::NewtonAccuracyDiagnostics& a);
SEXP retained_ls_weights_to_r(
    const magmaan::estimate::frontier::NewtonDerivatives& d);
Rcpp::List input_errors_to_r(
    const magmaan::estimate::frontier::NewtonInputErrorBounds& e);
Rcpp::List distance_interval_to_r(
    const magmaan::estimate::frontier::NewtonDistanceInterval& x);
Rcpp::List verified_assessment_to_r(
    const magmaan::estimate::frontier::ConvergenceAssessment& a);
Rcpp::List diagnostics_to_r(const magmaan::estimate::FitDiagnostics& d);
Rcpp::List fit_result(Ctx& ctx,
                      const magmaan::estimate::Estimates& est,
                      const magmaan::spec::Starts* starts,
                      const char* estimator);
magmaan::estimate::Bounds bounds_from_nullable(Rcpp::Nullable<Rcpp::List> bounds);
std::vector<Eigen::MatrixXd> wls_dense_from_arg(SEXP W, std::size_t n_blocks);
magmaan::estimate::gmm::Weight wls_from_arg(SEXP W, std::size_t n_blocks);
Rcpp::List ordinal_stats_to_r(const magmaan::data::OrdinalStats& s);
Rcpp::List mixed_ordinal_stats_to_r(const magmaan::data::MixedOrdinalStats& s);
magmaan::data::RawData fiml_raw_from_arg(const lvm::MatrixRep& rep, SEXP raw_data);
magmaan::data::RawData complete_raw_from_arg(const lvm::MatrixRep& rep,
                                             SEXP raw_data);
Rcpp::List fiml_raw_to_r(const magmaan::data::RawData& raw,
                         const std::vector<std::vector<std::string>>& ov_names,
                         const std::vector<std::string>& group_labels);
Rcpp::XPtr<FimlPack> fiml_pack_xptr(FimlPack pack);
Rcpp::XPtr<FimlH1> fiml_h1_xptr(FimlH1 h1);
const FimlPack* fiml_pack_ptr_from_fit(Rcpp::List fit);
const FimlH1* fiml_h1_ptr_from_fit(Rcpp::List fit);
const FimlPack& fiml_pack_for_fit(Rcpp::List fit,
                                  const magmaan::data::RawData& raw,
                                  std::unique_ptr<FimlPack>& owned);
const FimlH1& fiml_h1_for_fit(Rcpp::List fit,
                              const magmaan::data::RawData& raw,
                              const FimlPack& pack,
                              std::unique_ptr<FimlH1>& owned);
const SaturatedMoments& fiml_saturated_for_fit(
    Rcpp::List fit, const magmaan::data::RawData& raw, const FimlPack& pack,
    const FimlH1& h1, std::unique_ptr<SaturatedMoments>& owned);
Rcpp::List fiml_fit_result(Ctx& ctx,
                           const magmaan::data::RawData& raw,
                           const magmaan::estimate::Estimates& est,
                           const magmaan::spec::Starts* starts);
magmaan::data::OrdinalStats ordinal_stats_from_arg(Rcpp::List x);
void attach_ordinal_parameter_values(Rcpp::List& out, const Ctx& ctx,
                                     const magmaan::estimate::Estimates& est,
                                     const char* parameterization);
Rcpp::List ordinal_fit_result(Ctx& ctx,
                              const magmaan::data::OrdinalStats& stats,
                              const magmaan::estimate::Estimates& est,
                              const magmaan::spec::Starts* starts,
                              const char* estimator,
                              const char* parameterization = "delta");
Rcpp::List mixed_ordinal_fit_result(
    Ctx& ctx,
    const magmaan::data::MixedOrdinalStats& stats,
    const magmaan::estimate::Estimates& est,
    const magmaan::spec::Starts* starts,
    const char* estimator,
    const char* parameterization = "delta");
magmaan::estimate::frontier::MultiInfoPenaltyOptions multiinfo_options_from(
    double eta, Rcpp::Nullable<Rcpp::NumericVector> weight,
    const std::string& target);
Rcpp::List multiinfo_penalty_to_r(
    const Ctx& ctx, const magmaan::estimate::frontier::PenalizedFit& fit);
bool is_ml2s_estimator_label(const std::string& estimator);
bool ml2s_weight_needs_raw_ij(magmaan::estimate::fiml::TwoStageWeight kind);
magmaan::estimate::fiml::TwoStageDlsOptions ml2s_dls_options_from_fit(
    Rcpp::List fit);
void ml2s_recorded_stage2(Rcpp::List fit, SEXP stage2_weight_arg,
                          SEXP dls_a_arg, const char* call,
                          magmaan::estimate::fiml::TwoStageWeight& kind,
                          magmaan::estimate::fiml::TwoStageDlsOptions& dls);
SEXP fitting_weight_arg(Rcpp::List fit, SEXP weight, std::size_t n_blocks,
                        const char* call);
magmaan::estimate::gmm::Weight continuous_ls_weight(
    Rcpp::List fit, const Ctx& ctx, const magmaan::estimate::Estimates& est,
    const std::string& estimator, SEXP weight, const char* call);
RecordedMomentWeight recorded_moment_weight(Rcpp::List fit,
                                            const std::string& estimator);
magmaan::estimate::ContinuousLsIJWeightMode continuous_ij_mode_for_fit(
    Rcpp::List fit, const std::string& estimator,
    magmaan::estimate::gmm::FixedWeightOptions* dls_opts = nullptr);
magmaan::inference::frontier::ScoreFlipMultiplier
score_flip_multiplier_from_string(const std::string& multiplier);
magmaan::inference::frontier::ScoreFlipSensitivity
score_flip_sensitivity_from_string(const std::string& sensitivity);
magmaan::inference::frontier::GlobalScoreFlipOptions::Metric
global_score_metric_from_string(const std::string& metric);

}  // namespace magmaanr::fitglue
