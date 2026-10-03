#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

#include "prepared.h"
#include "score_primitives.h"
#include "magmaan/api/policy.hpp"
#include "magmaan/api/conventions.hpp"

namespace {

Rcpp::List policy_test_list(const magmaan::api::PolicyTest& t);
Rcpp::List policy_inference_list(const magmaan::api::PolicyInference& out);
static magmaan::api::PolicyFitState policy_state_from(const Rcpp::LogicalVector& s);
static magmaan::api::LavaanConvention lavaan_convention_from(const std::string& name);
static Rcpp::List convention_test_list(const magmaan::api::ConventionTest& t);

Rcpp::List policy_test_list(const magmaan::api::PolicyTest& t) {
  return Rcpp::List::create(
      Rcpp::_["available"] = t.reason == magmaan::api::InferenceReason::Available,
      Rcpp::_["reason"] = std::string(magmaan::api::reason_name(t.reason)),
      Rcpp::_["detail"] = t.detail,
      Rcpp::_["statistic"] = t.statistic, Rcpp::_["df"] = t.df,
      Rcpp::_["sb_scale"] = t.sb_scale, Rcpp::_["p_sb"] = t.p_sb,
      Rcpp::_["p_peba4"] = t.p_peba4, Rcpp::_["peba_blocks"] = t.peba_blocks,
      Rcpp::_["eigenvalues"] = Rcpp::wrap(t.eigenvalues),
      Rcpp::_["label"] = t.label);
}

Rcpp::List policy_inference_list(const magmaan::api::PolicyInference& out) {
  using magmaan::api::InferenceReason;
  const bool has_cov = out.covariance_reason == InferenceReason::Available;
  return Rcpp::List::create(
      Rcpp::_["covariance"] = has_cov ? Rcpp::RObject(Rcpp::wrap(out.covariance)) : Rcpp::RObject(R_NilValue),
      Rcpp::_["covariance_available"] = has_cov,
      Rcpp::_["covariance_reason"] = std::string(magmaan::api::reason_name(out.covariance_reason)),
      Rcpp::_["covariance_detail"] = out.covariance_detail,
      Rcpp::_["score"] = policy_test_list(out.score),
      Rcpp::_["lr"] = policy_test_list(out.lr),
      Rcpp::_["psd_boundary"] = out.psd_boundary,
      Rcpp::_["verdict_disagreement"] = out.verdict_disagreement);
}

// policy_inference_impl() — mirrors api::policy_inference_ml() on a prepared
// inference context. The fit state comes from the R fit because the context
// rebuilds estimates without their convergence diagnostics.
//
// The R fit state: converged by the selected rule, a PSD boundary estimate,
// magmaan's own check (NA when it decided or did not run), and a penalized
// estimate. A penalized fit needs no context: R passes NULL, because nothing
// is computed for it.
static magmaan::api::PolicyFitState policy_state_from(const Rcpp::LogicalVector& s) {
  magmaan::api::PolicyFitState out;
  out.converged = s.size() > 0 && s[0] == TRUE;
  out.psd_boundary = s.size() > 1 && s[1] == TRUE;
  if (s.size() > 2 && !Rcpp::LogicalVector::is_na(s[2])) out.native_converged = s[2] == TRUE;
  out.penalized = s.size() > 3 && s[3] == TRUE;
  return out;
}

static magmaan::api::LavaanConvention lavaan_convention_from(const std::string& name) {
  using C = magmaan::api::LavaanConvention;
  for (const auto c : {C::ML, C::MLM, C::MLR, C::DWLS, C::WLSMV, C::ULS, C::ULSMV, C::WLS})
    if (magmaan::api::convention_name(c) == name) return c;
  Rcpp::stop("unknown lavaan inference convention: %s", name);
  return C::ML;
}

static Rcpp::List convention_test_list(const magmaan::api::ConventionTest& t) {
  return Rcpp::List::create(
      Rcpp::_["available"] = t.reason == magmaan::api::InferenceReason::Available,
      Rcpp::_["reason"] = std::string(magmaan::api::reason_name(t.reason)),
      Rcpp::_["detail"] = t.detail, Rcpp::_["method"] = t.method,
      Rcpp::_["statistic"] = t.statistic, Rcpp::_["unscaled_statistic"] = t.unscaled_statistic,
      Rcpp::_["df"] = t.df, Rcpp::_["pvalue"] = t.p_value,
      Rcpp::_["scale"] = t.scale, Rcpp::_["shift"] = t.shift);
}

}  // namespace

// [[Rcpp::export]]
SEXP prepared_model_impl(SEXP partable, std::string kind,
                         Rcpp::Nullable<Rcpp::List> schema = R_NilValue) {
  return prepared::model(partable, kind, schema);
}

// [[Rcpp::export]]
SEXP prepared_data_impl(SEXP model, SEXP X, std::string kind, Rcpp::List ordered) {
  return prepared::dataset(model, X, kind, ordered);
}

// [[Rcpp::export]]
Rcpp::List prepared_weight_impl(SEXP data, std::string method, SEXP W, bool full,
                                SEXP model, double dls_a = 0.5) {
  return prepared::weight(data, method, W, full, model, dls_a);
}

// [[Rcpp::export]]
Rcpp::List prepared_estimate_impl(SEXP model, SEXP data, SEXP weight,
                                 std::string estimator,
                                 Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List> control = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List> bounds = R_NilValue,
                                 std::string covariance = "unrestricted",
                                 std::string barrier_target = "joint", double barrier_weight = 0.25,
                                 SEXP start_hints = R_NilValue) {
  return prepared::fit(model, data, weight, estimator, optimizer, control, bounds,
                       covariance, barrier_target, barrier_weight, start_hints);
}

// [[Rcpp::export]]
Rcpp::List prepare_inference_impl(Rcpp::List fit, SEXP raw, SEXP shared_data = R_NilValue) {
  return score_bindings::prepare(fit, raw, shared_data);
}

// [[Rcpp::export]]
Rcpp::List score_rows_impl(SEXP context, std::string space) {
  return score_bindings::rows(context, space);
}

// [[Rcpp::export]]
Rcpp::List score_components_impl(SEXP context, SEXP H1, std::string sensitivity, std::string metric) {
  return score_bindings::components(context,H1,sensitivity,metric);
}

// [[Rcpp::export]]
Rcpp::List project_scores_impl(SEXP components, bool retain_rows, bool center) {
  return score_bindings::project(components,retain_rows,center);
}

// [[Rcpp::export]]
Rcpp::List score_quadratic_impl(Rcpp::NumericVector score, Rcpp::NumericMatrix metric, SEXP meat) {
  return score_bindings::quadratic(score,metric,meat);
}

// [[Rcpp::export]]
Rcpp::List score_reference_impl(SEXP projected, bool spectrum) {
  return score_bindings::reference(projected,spectrum);
}

// [[Rcpp::export]]
Rcpp::List resample_scores_impl(SEXP projected, int n_flips, double seed,
                               std::string multiplier, double two_point_skewness) {
  return score_bindings::resample(projected,n_flips,seed,multiplier,two_point_skewness);
}

// [[Rcpp::export]]
Rcpp::List score_sandwich_impl(SEXP projected) {
  const auto& score = score_bindings::get<magmaan::inference::frontier::ProjectedScore>(
      projected,"magmaan_projected_score");
  auto statistic = magmaan::inference::frontier::score_sandwich(score);
  if (!statistic) stop_post(statistic.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = *statistic, Rcpp::_["df"] = score.score.size());
}

// [[Rcpp::export]]
Rcpp::NumericMatrix inference_information_impl(SEXP context, std::string type) {
  const auto& c = score_bindings::get<score_bindings::Context>(context,"magmaan_inference_context");
  if (c.estimator == "ML2S") Rcpp::stop("inference_information(): use the ML2S Stage-1/Stage-2 covariance interface");
  magmaan::post_expected<Eigen::MatrixXd> out;
  if (type == "expected" && c.ntml) {
    auto info = magmaan::robust::frontier::ntml_information(*c.ntml);
    if (!info) stop_post(info.error());
    return Rcpp::wrap(**info);
  }
  if (type == "expected") {
    out = c.estimator == "FIML" ? magmaan::estimate::fiml::fiml_expected_information(
        c.ctx.pt,c.ctx.rep,c.raw,c.estimates,c.pack) : magmaan::inference::information_expected(
        c.ctx.pt,c.ctx.rep,c.ctx.samp,c.estimates);
  } else if (type == "observed") {
    out = c.estimator == "FIML" ? magmaan::estimate::fiml::fiml_observed_information(
        c.ctx.pt,c.ctx.rep,c.raw,c.estimates,c.pack) : magmaan::inference::information_observed_analytic(
        c.ctx.pt,c.ctx.rep,c.ctx.samp,c.estimates);
  } else Rcpp::stop("inference_information(): unknown type");
  if (!out) stop_post(out.error());
  return Rcpp::wrap(*out);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix parameter_covariance_impl(SEXP context, Rcpp::NumericMatrix information, SEXP meat) {
  const auto& c = score_bindings::get<score_bindings::Context>(context,"magmaan_inference_context");
  auto out = magmaan::inference::vcov(Rcpp::as<Eigen::MatrixXd>(information),c.ctx.pt,c.estimates.theta);
  if (!out) stop_post(out.error());
  const bool model = TYPEOF(meat) == STRSXP &&
      Rcpp::as<std::string>(meat) == "model";
  if (!model) {
    Eigen::MatrixXd B;
    if (Rf_isNull(meat)) {
      if (c.raw.X.empty())
        Rcpp::stop("parameter_covariance(): empirical meat requires retained raw data");
      if (c.estimator == "ML") {
        // Covariance-only ML profiles its mean; use the core moment scores.
        auto rows = magmaan::inference::casewise_scores(
            c.ctx.pt, c.ctx.rep, c.ctx.samp, c.raw, c.estimates);
        if (!rows) stop_post(rows.error());
        B = rows->transpose() * *rows;
      } else {
        const Rcpp::List scores = score_bindings::rows(context, "parameter");
        const Eigen::MatrixXd rows = Rcpp::as<Eigen::MatrixXd>(scores["rows"]);
        B = rows.transpose() * rows;
      }
    } else {
      if (TYPEOF(meat) == STRSXP)
        Rcpp::stop("parameter_covariance(): named meat must be 'model'");
      B = Rcpp::as<Eigen::MatrixXd>(meat);
    }
    if (B.rows() != out->rows() || B.cols() != out->cols() || !B.allFinite() || !B.isApprox(B.transpose()))
      Rcpp::stop("parameter_covariance(): meat must be finite, symmetric and match information");
    *out = (*out * B * *out).eval();
  }
  return Rcpp::wrap(*out);
}

// [[Rcpp::export]]
Rcpp::List inference_snapshot_impl(SEXP context) {
  score_bindings::get<score_bindings::Context>(context,"magmaan_inference_context");
  return Rcpp::List(R_ExternalPtrProtected(context));
}

// [[Rcpp::export]]
SEXP score_components_matrix_impl(Rcpp::NumericVector score, Rcpp::NumericMatrix rows,
    Rcpp::NumericMatrix sensitivity, Rcpp::NumericMatrix metric,
    Rcpp::NumericMatrix nuisance, Rcpp::NumericMatrix directions, bool influence_rows) {
  magmaan::inference::frontier::ScoreComponents c;
  c.score = Rcpp::as<Eigen::VectorXd>(score);
  c.rows = Rcpp::as<Eigen::MatrixXd>(rows);
  c.sensitivity = Rcpp::as<Eigen::MatrixXd>(sensitivity);
  c.metric = Rcpp::as<Eigen::MatrixXd>(metric);
  c.nuisance = Rcpp::as<Eigen::MatrixXd>(nuisance);
  c.directions = Rcpp::as<Eigen::MatrixXd>(directions);
  c.influence_rows = influence_rows;
  const auto n = c.score.size();
  if (n == 0 || !c.score.allFinite() || c.rows.cols() != n || c.rows.rows() == 0 ||
      !c.rows.allFinite() || c.metric.rows() != n || c.metric.cols() != n ||
      c.sensitivity.rows() != n || c.sensitivity.cols() != n ||
      !c.metric.allFinite() || !c.sensitivity.allFinite() ||
      !c.metric.isApprox(c.metric.transpose()) || !c.sensitivity.isApprox(c.sensitivity.transpose()) ||
      c.nuisance.rows() != n || c.directions.rows() != n || c.directions.cols() == 0 ||
      !c.nuisance.allFinite() || !c.directions.allFinite())
    Rcpp::stop("score_components_from_matrices(): incompatible or non-finite ingredients");
  if (!influence_rows && (c.rows.colwise().sum().transpose() - c.score).norm() >
      1e-9 * std::max(1.0, c.score.norm()))
    Rcpp::stop("score_components_from_matrices(): likelihood rows must sum to the observed score");
  return score_bindings::handle(std::move(c),"magmaan_score_components");
}

// [[Rcpp::export]]
SEXP prepare_ntml_data_impl(Rcpp::List fit, SEXP raw, std::string storage) {
  auto ctx = ctx_from_fit(fit);
  auto method = magmaan::robust::frontier::ContributionStorage::Auto;
  if (storage == "casewise") method = magmaan::robust::frontier::ContributionStorage::Casewise;
  else if (storage == "tiled") method = magmaan::robust::frontier::ContributionStorage::Tiled;
  else if (storage != "auto") Rcpp::stop("unknown contribution storage");
  auto data = magmaan::robust::frontier::prepare_ntml_data(
      complete_raw_from_arg(ctx.rep,raw),ctx.meanstructure,method);
  if (!data) stop_post(data.error());
  return score_bindings::handle(*data,"magmaan_ntml_data");
}

// [[Rcpp::export]]
SEXP prepare_ntml_hypothesis_impl(SEXP null_context, SEXP alternative_context) {
  auto& a = score_bindings::get<score_bindings::Context>(null_context,"magmaan_inference_context");
  auto& b = score_bindings::get<score_bindings::Context>(alternative_context,"magmaan_inference_context");
  auto h = magmaan::robust::frontier::prepare_ntml_hypothesis(a.ntml,b.ntml);
  if (!h) stop_post(h.error());
  return score_bindings::handle(*h,"magmaan_ntml_hypothesis");
}

// [[Rcpp::export]]
Rcpp::List ntml_quadratic_impl(SEXP object, bool hypothesis, bool score,
                               bool observed = false) {
  magmaan::post_expected<std::shared_ptr<magmaan::robust::frontier::NTMLQuadratic>> q;
  if (hypothesis) {
    auto& h = score_bindings::get<std::shared_ptr<magmaan::robust::frontier::NTMLHypothesis>>(object,"magmaan_ntml_hypothesis");
    q = magmaan::robust::frontier::ntml_quadratic(*h,score,
        observed ? magmaan::robust::Information::Observed
                 : magmaan::robust::Information::Expected);
  } else {
    if (observed) Rcpp::stop("inference_quadratic(): geometry = \"observed\" applies to nested hypotheses");
    auto& c = score_bindings::get<score_bindings::Context>(object,"magmaan_inference_context");
    if (!c.ntml) Rcpp::stop("inference_quadratic(): shared geometry requires an interior random-X continuous ML fit with affine constraints");
    q = magmaan::robust::frontier::ntml_quadratic(*c.ntml,score);
  }
  if (!q) stop_post(q.error());
  return Rcpp::List::create(Rcpp::_["statistic"]=(**q).statistic,Rcpp::_["df"]=(**q).df,
      Rcpp::_["native"]=score_bindings::handle(*q,"magmaan_ntml_quadratic"));
}

// [[Rcpp::export]]
Rcpp::List ntml_reference_impl(SEXP object, bool spectrum) {
  auto& q = *score_bindings::get<std::shared_ptr<magmaan::robust::frontier::NTMLQuadratic>>(object,"magmaan_ntml_quadratic");
  Rcpp::List out=Rcpp::List::create(Rcpp::_["statistic"]=q.statistic,Rcpp::_["df"]=q.df);
  if (spectrum) {
    auto values=magmaan::robust::frontier::ntml_spectrum(q);
    if (!values) stop_post(values.error()); out["eigenvalues"]=Rcpp::wrap(**values);
  } else out["mean_scale"]=(q.reduced ? q.reduced->trace() : q.rows.squaredNorm())/q.df;
  return out;
}

// Casewise rows of a shared-geometry quadratic, one row per observation in
// group-block order. For a score quadratic the statistic is the squared norm of
// their column sums; for either kind their crossproduct is the reduced matrix
// whose eigenvalues are the reference spectrum.
// [[Rcpp::export]]
Rcpp::NumericMatrix ntml_rows_impl(SEXP object) {
  auto& q = *score_bindings::get<std::shared_ptr<magmaan::robust::frontier::NTMLQuadratic>>(object,"magmaan_ntml_quadratic");
  if (q.rows.size() == 0)
    Rcpp::stop("inference_rows(): this quadratic kept no casewise rows (tiled large-N storage)");
  return Rcpp::wrap(q.rows);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix ntml_covariance_impl(SEXP context, bool robust) {
  auto& c=score_bindings::get<score_bindings::Context>(context,"magmaan_inference_context");
  if (!c.ntml) Rcpp::stop("inference_covariance(): shared geometry requires continuous ML");
  auto v = robust ? magmaan::robust::frontier::ntml_score_sandwich(*c.ntml, magmaan::robust::Information::Observed)
                  : magmaan::robust::frontier::ntml_covariance(*c.ntml, false);
  if (!v) stop_post(v.error()); return Rcpp::wrap(**v);
}

// [[Rcpp::export]]
Rcpp::List convention_inference_impl(Rcpp::List fit, SEXP context,
    std::string convention, Rcpp::LogicalVector state) {
  using namespace magmaan::api;
  const auto c = lavaan_convention_from(convention);
  const auto fit_state = policy_state_from(state);
  ConventionInference out;
  if (fit_state.penalized) {
    out = convention_unavailable(c, InferenceReason::Penalized, std::string(penalized_detail), fit_state);
  } else if (!fit_state.converged) {
    out = convention_unavailable(c, InferenceReason::NotConverged, "the fit did not pass its convergence verdict", fit_state);
  } else if (fit.containsElementNamed("ordinal") && Rcpp::as<bool>(fit["ordinal"])) {
    auto ctx = ctx_from_fit(fit);
    const auto est = est_from_fit(fit);
    const auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(fit, R_NilValue,
        "ordinal_stats", "convention_inference"));
    const std::string estimator = Rcpp::as<std::string>(fit["estimator"]);
    const std::string parameterization = Rcpp::as<std::string>(fit["parameterization"]);
    out = lavaan_inference_ordinal(std::move(ctx.pt), ctx.rep, stats, est,
        ordinal_weight_from_estimator(estimator, "convention_inference"),
        ordinal_parameterization_from_string(parameterization), c, fit_state);
  } else if (!Rf_isNull(context)) {
    auto& ctx = score_bindings::get<score_bindings::Context>(context, "magmaan_inference_context");
    if (ctx.estimator == "ML" && ctx.ntml) out = lavaan_inference_ml(*ctx.ntml, c, fit_state);
    else out = convention_unavailable(c, InferenceReason::UnsupportedModel,
        "this lavaan convention is not checked for the fitted model", fit_state);
  } else {
    out = convention_unavailable(c, InferenceReason::UnsupportedModel,
        "this lavaan convention is not checked for the fitted model", fit_state);
  }
  const bool has_cov = out.covariance_reason == InferenceReason::Available;
  return Rcpp::List::create(Rcpp::_["convention"] = out.convention,
      Rcpp::_["covariance"] = has_cov ? Rcpp::RObject(Rcpp::wrap(out.covariance)) : Rcpp::RObject(R_NilValue),
      Rcpp::_["covariance_available"] = has_cov,
      Rcpp::_["covariance_reason"] = std::string(reason_name(out.covariance_reason)),
      Rcpp::_["covariance_detail"] = out.covariance_detail,
      Rcpp::_["test"] = convention_test_list(out.test),
      Rcpp::_["psd_boundary"] = out.psd_boundary,
      Rcpp::_["verdict_disagreement"] = out.verdict_disagreement);
}

// [[Rcpp::export]]
Rcpp::List convention_nested_impl(SEXP null_context, SEXP alternative_context,
    std::string convention, Rcpp::LogicalVector null_state, Rcpp::LogicalVector alternative_state) {
  using namespace magmaan::api;
  const auto c = lavaan_convention_from(convention);
  const auto s0 = policy_state_from(null_state), s1 = policy_state_from(alternative_state);
  ConventionTest out;
  if (s0.penalized || s1.penalized) {
    out.reason = InferenceReason::Penalized; out.detail = penalized_detail;
  } else if (!s0.converged || !s1.converged) {
    out.reason = InferenceReason::NotConverged; out.detail = "a fit did not pass its convergence verdict";
  } else if (Rf_isNull(null_context) || Rf_isNull(alternative_context)) {
    out.reason = InferenceReason::UnsupportedModel; out.detail = "nested lavaan conventions cover complete-data ML so far";
  } else {
    auto& a = score_bindings::get<score_bindings::Context>(null_context, "magmaan_inference_context");
    auto& b = score_bindings::get<score_bindings::Context>(alternative_context, "magmaan_inference_context");
    if (a.estimator == "ML" && b.estimator == "ML" && a.ntml && b.ntml)
      out = lavaan_nested_ml(a.ntml, s0, b.ntml, s1, c);
    else { out.reason = InferenceReason::UnsupportedModel; out.detail = "nested lavaan conventions cover complete-data ML so far"; }
  }
  return Rcpp::List::create(Rcpp::_["test"] = convention_test_list(out),
      Rcpp::_["psd_boundary"] = s0.psd_boundary || s1.psd_boundary,
      Rcpp::_["verdict_disagreement"] = verdict_disagreement(s0) || verdict_disagreement(s1));
}

// [[Rcpp::export]]
Rcpp::List policy_inference_impl(SEXP context, Rcpp::LogicalVector state) {
  using magmaan::api::InferenceReason;
  const auto fit_state = policy_state_from(state);
  magmaan::api::PolicyInference out;
  if (fit_state.penalized) {
    out = magmaan::api::policy_unavailable(InferenceReason::Penalized,
                                           std::string(magmaan::api::penalized_detail));
  } else if (auto& c = score_bindings::get<score_bindings::Context>(
                 context, "magmaan_inference_context");
             c.estimator != "ML" && c.estimator != "FIML") {
    out = magmaan::api::policy_unavailable(InferenceReason::UnsupportedModel,
        "the inference policy covers single-level ML and FIML");
  } else if (c.estimator == "FIML") {
    out = magmaan::api::policy_inference_fiml(c.ctx.pt, c.ctx.rep, c.raw, c.pack, c.estimates, fit_state);
  } else if (!c.ntml) {
    out = magmaan::api::policy_unavailable(InferenceReason::UnsupportedModel,
        "the inference policy requires random x, affine equality constraints and no active bounds");
  } else {
    out = magmaan::api::policy_inference_ml(*c.ntml, fit_state);
  }
  out.verdict_disagreement = magmaan::api::verdict_disagreement(fit_state);
  return policy_inference_list(out);
}

// policy_inference_dwls_impl() — mirrors api::policy_inference_dwls() for an
// all-ordinal DWLS fit, from the fit's partable, estimates and retained
// ordinal statistics. Only plain DWLS (weight diag(NACOV)^-1) qualifies;
// Stage-2 NT/DLS, supplied-weight, ULS and WLS fits are unavailable.
//
// [[Rcpp::export]]
Rcpp::List policy_inference_dwls_impl(Rcpp::List fit, Rcpp::LogicalVector state) {
  using magmaan::api::InferenceReason;
  const auto fit_state = policy_state_from(state);
  auto text = [&](const char* name) {
    return fit.containsElementNamed(name) && !Rf_isNull(fit[name])
        ? Rcpp::as<std::string>(fit[name]) : std::string();
  };
  const bool ordinal = fit.containsElementNamed("ordinal") && Rcpp::as<bool>(fit["ordinal"]);
  const std::string computational = text("ordinal_computational_weight");
  const std::string recipe = text("moment_weight");
  magmaan::api::PolicyInference out;
  if (fit_state.penalized) {
    out = magmaan::api::policy_unavailable(InferenceReason::Penalized,
                                           std::string(magmaan::api::penalized_detail));
  } else if (!ordinal || text("estimator") != "DWLS" ||
             (!computational.empty() && computational != "DWLS") ||
             (!recipe.empty() && recipe != "dwls")) {
    out = magmaan::api::policy_unavailable(InferenceReason::UnsupportedModel,
        "the categorical inference policy covers plain all-ordinal DWLS fits");
  } else {
    Ctx ctx = ctx_from_fit(fit);
    const magmaan::estimate::Estimates est = est_from_fit(fit);
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "ordinal_stats", "policy_inference"));
    const std::string parameterization = fit.containsElementNamed("parameterization")
        ? Rcpp::as<std::string>(fit["parameterization"])
        : ordinal_parameterization_attr(fit["partable"]);
    out = magmaan::api::policy_inference_dwls(std::move(ctx.pt), ctx.rep, stats, est,
        ordinal_parameterization_from_string(parameterization), fit_state);
  }
  out.verdict_disagreement = magmaan::api::verdict_disagreement(fit_state);
  return policy_inference_list(out);
}

// policy_nested_dwls_impl() — mirrors api::policy_nested_dwls() for two plain
// all-ordinal DWLS fits to the same ordinal statistics (the caller checks that
// they are the same observations).
//
// [[Rcpp::export]]
Rcpp::List policy_nested_dwls_impl(Rcpp::List fit_H1, Rcpp::List fit_H0,
                                   Rcpp::LogicalVector null_state,
                                   Rcpp::LogicalVector alternative_state) {
  const auto null_fit = policy_state_from(null_state);
  const auto alternative_fit = policy_state_from(alternative_state);
  auto text = [](Rcpp::List fit, const char* name) {
    return fit.containsElementNamed(name) && !Rf_isNull(fit[name])
        ? Rcpp::as<std::string>(fit[name]) : std::string();
  };
  for (Rcpp::List fit : {fit_H1, fit_H0}) {
    const bool ordinal = fit.containsElementNamed("ordinal") && Rcpp::as<bool>(fit["ordinal"]);
    const std::string computational = text(fit, "ordinal_computational_weight");
    const std::string recipe = text(fit, "moment_weight");
    if (!ordinal || text(fit, "estimator") != "DWLS" ||
        (!computational.empty() && computational != "DWLS") ||
        (!recipe.empty() && recipe != "dwls")) {
      Rcpp::stop("the categorical nested policy covers plain all-ordinal DWLS fits");
    }
  }
  auto parameterization_of = [&](Rcpp::List fit) {
    return fit.containsElementNamed("parameterization")
        ? Rcpp::as<std::string>(fit["parameterization"])
        : ordinal_parameterization_attr(fit["partable"]);
  };
  const std::string parameterization = parameterization_of(fit_H1);
  if (parameterization_of(fit_H0) != parameterization)
    Rcpp::stop("the two fits use different ordinal parameterizations");
  Ctx c1 = ctx_from_fit(fit_H1);
  Ctx c0 = ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates e1 = est_from_fit(fit_H1);
  const magmaan::estimate::Estimates e0 = est_from_fit(fit_H0);
  auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
      fit_H1, R_NilValue, "ordinal_stats", "policy_nested"));
  auto out = magmaan::api::policy_nested_dwls(std::move(c0.pt), c0.rep, e0, null_fit,
      std::move(c1.pt), c1.rep, e1, alternative_fit, stats,
      ordinal_parameterization_from_string(parameterization));
  return Rcpp::List::create(Rcpp::_["score"] = policy_test_list(out.score),
                            Rcpp::_["lr"] = policy_test_list(out.lr),
                            Rcpp::_["psd_boundary"] = out.psd_boundary,
                            Rcpp::_["verdict_disagreement"] = out.verdict_disagreement);
}

// policy_nested_impl() — mirrors api::policy_nested_ml() on two prepared
// inference contexts that share one prepared dataset.
//
// [[Rcpp::export]]
Rcpp::List policy_nested_impl(SEXP null_context, SEXP alternative_context,
                              Rcpp::LogicalVector null_state,
                              Rcpp::LogicalVector alternative_state) {
  using magmaan::api::InferenceReason;
  const auto null_fit = policy_state_from(null_state);
  const auto alternative_fit = policy_state_from(alternative_state);
  magmaan::api::PolicyNested out;
  auto unavailable = [&](InferenceReason reason, const std::string& detail) {
    for (auto* t : {&out.score, &out.lr}) {
      t->reason = reason;
      t->detail = detail;
    }
  };
  if (null_fit.penalized || alternative_fit.penalized) {
    unavailable(InferenceReason::Penalized, std::string(magmaan::api::penalized_detail));
  } else {
    auto& a = score_bindings::get<score_bindings::Context>(null_context,"magmaan_inference_context");
    auto& b = score_bindings::get<score_bindings::Context>(alternative_context,"magmaan_inference_context");
    if (a.estimator == "FIML" && b.estimator == "FIML") {
      out = magmaan::api::policy_nested_fiml(a.ctx.pt, a.ctx.rep, a.estimates, null_fit,
          b.ctx.pt, b.ctx.rep, b.estimates, alternative_fit, a.raw, a.pack);
    } else if (a.estimator != "ML" || b.estimator != "ML" || !a.ntml || !b.ntml) {
      unavailable(InferenceReason::UnsupportedModel,
                  "nested policy tests cover complete-data ML with random x, affine equality "
                  "constraints and no active bounds");
    } else {
      out = magmaan::api::policy_nested_ml(a.ntml, null_fit, b.ntml, alternative_fit);
    }
  }
  out.verdict_disagreement = magmaan::api::verdict_disagreement(null_fit) ||
                             magmaan::api::verdict_disagreement(alternative_fit);
  return Rcpp::List::create(Rcpp::_["score"] = policy_test_list(out.score),
                            Rcpp::_["lr"] = policy_test_list(out.lr),
                            Rcpp::_["psd_boundary"] = out.psd_boundary,
                            Rcpp::_["verdict_disagreement"] = out.verdict_disagreement);
}

// [[Rcpp::export]]
Rcpp::List inference_reuse_impl(SEXP context) {
  auto& c=score_bindings::get<score_bindings::Context>(context,"magmaan_inference_context");
  if (!c.ntml) Rcpp::stop("reuse counters currently require continuous ML");
  const auto& f=*c.ntml;
  return Rcpp::List::create(Rcpp::_["geometry_builds"]=static_cast<double>(f.geometry_builds),
    Rcpp::_["u_builds"]=static_cast<double>(f.u_builds),
    Rcpp::_["information_builds"]=static_cast<double>(f.information_builds),
    Rcpp::_["score_spectrum_builds"]=static_cast<double>(f.score ? f.score->spectrum_builds : 0),
    Rcpp::_["lr_spectrum_builds"]=static_cast<double>(f.lr ? f.lr->spectrum_builds : 0),
    Rcpp::_["contribution_builds"]=static_cast<double>(f.data->contribution_builds),
    Rcpp::_["projection_passes"]=static_cast<double>(f.data->projection_passes),
    Rcpp::_["storage"]=f.data->storage == magmaan::robust::frontier::ContributionStorage::Tiled ? "tiled" : "casewise");
}

// Internal test hook; not exported from the package namespace.
// [[Rcpp::export]]
double prepared_structure_count_impl() {
  return static_cast<double>(prepared::structural_preparations);
}
