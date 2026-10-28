#include "magmaan/api/fit_measure_cache.hpp"
#include "glue_internal.h"
#include "magmaan/api/policy.hpp"
#include "magmaan/api/frontier/fit_index_intervals.hpp"
#include "magmaan/api/conventions.hpp"
// [[Rcpp::depends(RcppEigen)]]
using namespace magmaanr;
using namespace magmaanr::fitglue;

// The R fit-owned cache validates the complete portable fit before attaching
// this transient handle. It never survives serialization or a process change.
namespace {
magmaan::api::FitMeasureCache* fit_measure_cache(Rcpp::List fit) {
  SEXP native = fit.attr("fit_measure_native");
  if (Rf_isNull(native)) return nullptr;
  return Rcpp::XPtr<magmaan::api::FitMeasureCache>(native).get();
}
}

// [[Rcpp::export]]
SEXP fit_measure_cache_impl() {
  return Rcpp::XPtr<magmaan::api::FitMeasureCache>(new magmaan::api::FitMeasureCache(), true);
}

// [[Rcpp::export]]
Rcpp::DataFrame policy_fit_measures_impl(Rcpp::List fit, Rcpp::LogicalVector state) {
  using namespace magmaan::api;
  PolicyFitState s;
  s.converged = state.size() > 0 && state[0] == TRUE;
  s.penalized = state.size() > 3 && state[3] == TRUE;
  const bool ordinal = fit.containsElementNamed("ordinal") && Rcpp::as<bool>(fit["ordinal"]);
  const bool mixed = fit.containsElementNamed("mixed_ordinal") && Rcpp::as<bool>(fit["mixed_ordinal"]);
  const std::string estimator = fit.containsElementNamed("estimator") ? Rcpp::as<std::string>(fit["estimator"]) : "ML";
  const bool uls = estimator == "ULS" && !ordinal && !mixed;
  const bool likelihood = estimator == "ML" || estimator == "FIML";
  PolicyFitMeasures out;
  if (s.penalized || !s.converged) {
    out = policy_fit_measures_unavailable(s.penalized ? InferenceReason::Penalized : InferenceReason::NotConverged,
        s.penalized ? std::string(penalized_detail) : "the fit did not pass its convergence verdict", ordinal || mixed, likelihood);
  } else if ((ordinal || mixed) && (estimator == "DWLS" || estimator == "RDWLS")) {
    Ctx ctx = ctx_from_fit(fit);
    auto est = est_from_fit(fit);
    const auto parameterization = ordinal_parameterization_from_string(
        fit.containsElementNamed("parameterization") ? Rcpp::as<std::string>(fit["parameterization"]) : "delta");
    if (mixed) out = policy_fit_measures(ctx.pt, ctx.rep,
        mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(fit, R_NilValue, "mixed_ordinal_stats", "policy_fit_measures")), est, s, parameterization);
    else out = policy_fit_measures(ctx.pt, ctx.rep,
        ordinal_stats_from_arg(stats_from_fit_or_arg(fit, R_NilValue, "ordinal_stats", "policy_fit_measures")), est, s, parameterization);
  } else if (estimator == "ML2S" && !ordinal && !mixed) {
    const std::string weight = fit.containsElementNamed("stage2_weight") ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
    if (weight != "nt") out = policy_fit_measures_unavailable(InferenceReason::UnsupportedModel,
        "missing ML2S non-NT natural-discrepancy profile correction", false, false);
    else {
      Ctx ctx = ctx_from_fit(fit);
      auto est = est_from_fit(fit);
      auto raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
      auto sm = magmaan::estimate::fiml::saturated_em_moments(raw);
      if (!sm) out = policy_fit_measures_unavailable(InferenceReason::NumericFailure, sm.error().detail, false, false);
      else out = policy_fit_measures_two_stage(ctx.pt,ctx.rep,est,*sm,s);
    }
  } else if ((likelihood || uls) && !fit.containsElementNamed("nclusters")) {
    Ctx ctx = ctx_from_fit(fit);
    auto est = est_from_fit(fit);
    if (!fit.containsElementNamed("raw_data") || Rf_isNull(fit["raw_data"]))
      out = policy_fit_measures_unavailable(InferenceReason::UnsupportedModel, "raw sampling rows are required", false, likelihood);
    else {
      auto raw = estimator == "FIML" ? fiml_raw_from_arg(ctx.rep, fit["raw_data"]) : complete_raw_from_arg(ctx.rep, fit["raw_data"]);
      out = policy_fit_measures(ctx.pt, ctx.rep, raw, est, s, estimator == "FIML", uls, fit_measure_cache(fit));
    }
  } else out = policy_fit_measures_unavailable(InferenceReason::UnsupportedModel,
      estimator == "GLS" || estimator == "WLS" ? "missing natural-discrepancy estimated-weight profile trace for this estimator" : "missing natural-discrepancy profile correction or sampling rows for this estimator", ordinal || mixed, likelihood);
  Rcpp::CharacterVector index(out.indices.size()), reason(out.indices.size());
  Rcpp::NumericVector estimate(out.indices.size());
  for (std::size_t i = 0; i < out.indices.size(); ++i) {
    const auto& value = out.indices[i];
    index[i] = value.index;
    estimate[i] = value.reason == InferenceReason::Available ? value.estimate : NA_REAL;
    reason[i] = value.reason == InferenceReason::Available ? Rcpp::String(NA_STRING) :
        Rcpp::String(std::string(reason_name(value.reason)) + ": " + value.detail);
  }
  auto result = Rcpp::DataFrame::create(Rcpp::_["index"] = index, Rcpp::_["estimate"] = estimate, Rcpp::_["reason"] = reason);
  auto details = [](const PolicyFitDiscrepancy& d) {
    return Rcpp::List::create(Rcpp::_["discrepancy"] = d.discrepancy, Rcpp::_["trace"] = d.trace,
        Rcpp::_["corrected"] = d.corrected, Rcpp::_["df"] = d.df);
  };
  result.attr("details") = Rcpp::List::create(Rcpp::_["user"] = details(out.user), Rcpp::_["baseline"] = details(out.baseline),
      Rcpp::_["ntotal"] = static_cast<double>(out.ntotal), Rcpp::_["ngroups"] = static_cast<double>(out.n_groups),
      Rcpp::_["srmr_uncorrected"] = out.residual_uncorrected, Rcpp::_["srmr_trace"] = out.residual_trace);
  return result;
}

// [[Rcpp::export]]
Rcpp::DataFrame convention_fit_measures_impl(Rcpp::List fit, std::string convention,
    Rcpp::LogicalVector state) {
  using namespace magmaan::api;
  LavaanConvention c = LavaanConvention::ML;
  bool found = false;
  for (const auto candidate : {LavaanConvention::ML, LavaanConvention::MLM, LavaanConvention::MLR,
      LavaanConvention::DWLS, LavaanConvention::WLSMV, LavaanConvention::WLSM,
      LavaanConvention::ULS, LavaanConvention::ULSMV, LavaanConvention::WLS})
    if (convention_name(candidate) == convention) { c = candidate; found = true; }
  if (!found) Rcpp::stop("unknown lavaan convention: %s", convention);
  PolicyFitState s;
  s.converged = state.size() > 0 && state[0] == TRUE;
  s.penalized = state.size() > 3 && state[3] == TRUE;
  auto ctx = ctx_from_fit(fit);
  auto est = est_from_fit(fit);
  const std::string estimator = Rcpp::as<std::string>(fit["estimator"]);
  const bool ordinal = fit.containsElementNamed("ordinal") && Rcpp::as<bool>(fit["ordinal"]);
  const bool mixed = fit.containsElementNamed("mixed_ordinal") && Rcpp::as<bool>(fit["mixed_ordinal"]);
  Result<ConventionFitMeasures> out = std::unexpected(make_error(ErrorStage::UnsupportedCombination,
      "this lavaan convention is not checked for the fitted model"));
  if (ordinal && !mixed) {
    const auto w = ordinal_weight_from_estimator(estimator, "convention_fit_measures");
    const auto p = ordinal_parameterization_from_string(Rcpp::as<std::string>(fit["parameterization"]));
    out = convention_fit_measures(ctx.pt, ctx.rep,
        ordinal_stats_from_arg(stats_from_fit_or_arg(fit, R_NilValue, "ordinal_stats", "convention_fit_measures")),
        est, w, p, c, s);
  } else if ((estimator == "ML" || estimator == "FIML") && !mixed &&
      !fit.containsElementNamed("nclusters") && fit.containsElementNamed("raw_data")) {
    auto raw = estimator == "FIML" ? fiml_raw_from_arg(ctx.rep, fit["raw_data"]) :
        complete_raw_from_arg(ctx.rep, fit["raw_data"]);
    out = convention_fit_measures(ctx.pt, ctx.rep, raw, est, c, s, estimator == "FIML", fit_measure_cache(fit));
  }
  if (!out) Rcpp::stop("convention_fit_measures(): %s", out.error().detail);
  Rcpp::CharacterVector index(out->indices.size()), reason(out->indices.size());
  Rcpp::NumericVector estimate(out->indices.size());
  for (std::size_t i = 0; i < out->indices.size(); ++i) {
    const auto& x = out->indices[i]; index[i] = x.index;
    estimate[i] = x.reason == InferenceReason::Available && std::isfinite(x.estimate) ? x.estimate : NA_REAL;
    reason[i] = x.reason != InferenceReason::Available ?
        Rcpp::String(std::string(reason_name(x.reason)) + ": " + x.detail) :
        std::isfinite(x.estimate) ? Rcpp::String(NA_STRING) : Rcpp::String("inapplicable");
  }
  auto result = Rcpp::DataFrame::create(Rcpp::_["index"] = index,
      Rcpp::_["estimate"] = estimate, Rcpp::_["reason"] = reason);
  result.attr("lavaan_compat") = out->convention;
  return result;
}

// [[Rcpp::export]]
Rcpp::DataFrame fit_index_intervals_misspec_impl(Rcpp::List fit, double level,
    Rcpp::LogicalVector state) {
  using namespace magmaan::api;
  using namespace magmaan::api::frontier;
  PolicyFitState s;
  s.converged = state.size() > 0 && state[0] == TRUE;
  s.penalized = state.size() > 3 && state[3] == TRUE;
  const std::string estimator = Rcpp::as<std::string>(fit["estimator"]);
  const bool ordinal = fit.containsElementNamed("ordinal") && Rcpp::as<bool>(fit["ordinal"]);
  const bool mixed = fit.containsElementNamed("mixed_ordinal") && Rcpp::as<bool>(fit["mixed_ordinal"]);
  FitIndexIntervals out;
  if (s.penalized || !s.converged)
    out = fit_index_intervals_misspec_unavailable(s.penalized ? InferenceReason::Penalized : InferenceReason::NotConverged,
        s.penalized ? std::string(penalized_detail) : "the fit did not pass its convergence verdict");
  else if (ordinal || mixed || fit.containsElementNamed("association") || fit.containsElementNamed("nclusters") ||
      (estimator != "ML" && estimator != "FIML") ||
      !fit.containsElementNamed("raw_data") || Rf_isNull(fit["raw_data"]))
    out = fit_index_intervals_misspec_unavailable(InferenceReason::UnsupportedModel,
        "candidate intervals support only continuous ML/FIML with raw observations");
  else {
    auto ctx = ctx_from_fit(fit);
    auto est = est_from_fit(fit);
    auto raw = estimator == "FIML" ? fiml_raw_from_arg(ctx.rep,fit["raw_data"]) : complete_raw_from_arg(ctx.rep,fit["raw_data"]);
    out = fit_index_intervals_misspec(ctx.pt,ctx.rep,raw,est,s,estimator == "FIML",level,fit_measure_cache(fit));
  }
  const auto n = out.indices.size();
  Rcpp::CharacterVector index(n), variant(n), reason(n);
  Rcpp::NumericVector estimate(n), lower(n), upper(n), v1(n), v2(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto& row = out.indices[i];
    index[i] = row.index; variant[i] = row.variant;
    estimate[i] = std::isfinite(row.estimate) ? row.estimate : NA_REAL;
    lower[i] = row.reason == InferenceReason::Available ? row.lower : NA_REAL;
    upper[i] = row.reason == InferenceReason::Available ? row.upper : NA_REAL;
    v1[i] = std::isfinite(row.v1) ? row.v1 : NA_REAL;
    v2[i] = std::isfinite(row.v2) ? row.v2 : NA_REAL;
    reason[i] = row.reason == InferenceReason::Available ? Rcpp::String(NA_STRING) :
        Rcpp::String(std::string(reason_name(row.reason))+": "+row.detail);
  }
  auto result = Rcpp::DataFrame::create(Rcpp::_["index"]=index,Rcpp::_["variant"]=variant,
      Rcpp::_["estimate"]=estimate,Rcpp::_["lower"]=lower,Rcpp::_["upper"]=upper,
      Rcpp::_["V1"]=v1,Rcpp::_["V2"]=v2,Rcpp::_["reason"]=reason);
  result.attr("details") = Rcpp::List::create(Rcpp::_["F_hat"]=out.casewise_discrepancy,
      Rcpp::_["F_fit"]=out.points.user.discrepancy,Rcpp::_["F_tilde"]=out.points.user.corrected,
      Rcpp::_["takeuchi_trace"]=out.points.user.trace,Rcpp::_["quadratic_trace"]=out.quadratic_trace,
      Rcpp::_["quadratic_squared_trace"]=out.quadratic_squared_trace,
      Rcpp::_["ntotal"]=static_cast<double>(out.points.ntotal),Rcpp::_["level"]=level);
  return result;
}
