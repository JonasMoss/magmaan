#pragma once
#include "magmaan/robust/prepared_ntml.hpp"
#include "magmaan/api/policy.hpp"

namespace score_bindings {
using namespace magmaan;
using namespace magmaan::inference::frontier;
struct Context {
  Ctx ctx;
  estimate::Estimates estimates;
  data::RawData raw;
  estimate::fiml::FIMLPack pack;
  std::string estimator;
  std::shared_ptr<robust::frontier::NTMLFit> ntml;
  std::shared_ptr<api::FimlPolicyFit> fiml_policy;
};

template<class T> T& get(SEXP ptr, const char* tag) {
  if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install(tag) ||
      !R_ExternalPtrAddr(ptr))
    Rcpp::stop("magmaan: invalid or restored %s; prepare it again", tag);
  return *Rcpp::XPtr<T>(ptr);
}
template<class T> SEXP handle(T value, const char* tag) {
  return Rcpp::XPtr<T>(new T(std::move(value)), true, Rf_install(tag));
}
Rcpp::List prepare(Rcpp::List fit, SEXP raw, SEXP shared_data = R_NilValue) {
  Context c;
  c.ctx = ctx_from_fit(fit);
  c.estimates = est_from_fit(fit);
  c.estimator = fit.containsElementNamed("estimator") ? Rcpp::as<std::string>(fit["estimator"]) : "ML";
  if (c.estimator != "ML" && c.estimator != "FIML" && c.estimator != "ML2S")
    Rcpp::stop("prepare_inference(): score adapters currently support ML, FIML and fixed-NT ML2S");
  c.raw = c.estimator == "ML" ? complete_raw_from_arg(c.ctx.rep, raw) : fiml_raw_from_arg(c.ctx.rep, raw);
  std::shared_ptr<robust::frontier::NTMLData> shared;
  if (c.estimator == "ML") {
    if (!Rf_isNull(shared_data)) shared = get<std::shared_ptr<robust::frontier::NTMLData>>(shared_data,"magmaan_ntml_data");
    else {
      auto d=robust::frontier::prepare_ntml_data(c.raw,c.ctx.meanstructure);
      if (!d) stop_post(d.error()); shared=*d;
    }
    c.pack=shared->pack;
  } else {
    std::unique_ptr<FimlPack> owned;
    // Serialized fits retain raw data but their external cache pointers are
    // cleared by R. Rebuild the snapshot cache through the existing fallback.
    Rcpp::List cache_fit(Rf_shallow_duplicate(fit));
    if (cache_fit.containsElementNamed("fiml_pack")) {
      SEXP pointer = cache_fit["fiml_pack"];
      if (TYPEOF(pointer) == EXTPTRSXP && R_ExternalPtrAddr(pointer) == nullptr)
        cache_fit["fiml_pack"] = R_NilValue;
    }
    c.pack = fiml_pack_for_fit(cache_fit, c.raw, owned);
  }
  if (c.estimator == "ML") {
    const auto& sample = c.pack.start_stats;
    if (sample.n_obs != c.ctx.samp.n_obs || sample.S.size() != c.ctx.samp.S.size())
      Rcpp::stop("prepare_inference(): raw data do not match the fitted sample");
    for (std::size_t b = 0; b < sample.S.size(); ++b) {
      if (!sample.S[b].isApprox(c.ctx.samp.S[b],1e-10) ||
          (!c.ctx.samp.mean.empty() && !sample.mean[b].isApprox(c.ctx.samp.mean[b],1e-10)))
        Rcpp::stop("prepare_inference(): raw moments do not match the fitted sample");
    }
  }

  if (c.estimator == "ML") {
    std::shared_ptr<robust::frontier::NTMLData> data=shared;
    if (!Rf_isNull(shared_data)) {
      data = get<std::shared_ptr<robust::frontier::NTMLData>>(shared_data,"magmaan_ntml_data");
      if (data->raw.X.size() != c.raw.X.size()) Rcpp::stop("inference data layout mismatch");
      for (std::size_t b=0;b<c.raw.X.size();++b)
        if (data->raw.X[b].rows()!=c.raw.X[b].rows() || data->raw.X[b].cols()!=c.raw.X[b].cols() ||
            !(data->raw.X[b].array()==c.raw.X[b].array()).all()) Rcpp::stop("inference data observations differ");
    }
    auto f = robust::frontier::prepare_ntml_fit(data,c.ctx.pt,c.ctx.rep,c.estimates);
    if (f) c.ntml = std::move(*f);
    else if (!Rf_isNull(shared_data)) stop_post(f.error());
  } else if (!Rf_isNull(shared_data)) Rcpp::stop("shared NTML data require an ML fit");

  Rcpp::List out = Rcpp::List::create(Rcpp::_["theta"] = Rcpp::wrap(c.estimates.theta),
      Rcpp::_["estimator"] = c.estimator);
  Rcpp::List snapshot = cache_fit_context(fit,c.ctx);
  if (c.estimator == "ML") {
    snapshot["raw_data"] = raw;
    SEXP native=handle(c.ntml,"magmaan_ntml_fit");
    Rcpp::List keys=Rcpp::List::create(snapshot["partable"],snapshot["S"],snapshot["nobs"],
        snapshot["sample_mean"],snapshot["theta"],snapshot["fmin"],snapshot["raw_data"]);
    R_SetExternalPtrProtected(native,keys);
    snapshot.attr("magmaan_ntml") = native;
  }
  out["fit"] = snapshot;
  out["original_fit"] = fit;
  out["raw"] = raw;
  Rcpp::RObject ptr(handle(std::move(c), "magmaan_inference_context"));
  // Preserve the Stage-1/H1 handles and the R snapshot used to build this context.
  R_SetExternalPtrProtected(ptr, snapshot);
  out["native"] = ptr;
  return out;
}
Rcpp::List rows(SEXP ptr, std::string space) {
  const auto& c = get<Context>(ptr, "magmaan_inference_context");
  if (c.estimator == "ML2S") Rcpp::stop("scores(): ML2S uses Stage-1 influence contributions; use score_components()");
  post_expected<Eigen::MatrixXd> value;
  if (space == "parameter") {
    value = estimate::fiml::fiml_casewise_deviance_scores(c.ctx.pt,c.ctx.rep,c.raw,c.pack,c.estimates);
  } else if (space == "saturated") {
    auto ev = model::ModelEvaluator::build(c.ctx.pt,c.ctx.rep);
    if (!ev) stop_model(ev.error());
    auto moments = ev->evaluate(c.estimates.theta, false, false);
    if (!moments) stop_model(moments.error());
    value = estimate::fiml::fiml_saturated_casewise_deviance_scores(
        c.raw,c.pack,moments->moments,c.ctx.meanstructure);
  } else Rcpp::stop("scores(): unknown coordinate space");
  if (!value) stop_post(value.error());
  *value *= -0.5;
  Eigen::VectorXd sum = value->colwise().sum().transpose();
  return Rcpp::List::create(Rcpp::_["score"] = Rcpp::wrap(sum), Rcpp::_["rows"] = Rcpp::wrap(*value),
                           Rcpp::_["space"] = space, Rcpp::_["normalization"] = "sum-loglikelihood");
}
Rcpp::List components(SEXP ptr, SEXP partable, std::string sensitivity, std::string metric) {
  const auto& c = get<Context>(ptr, "magmaan_inference_context");
  Rcpp::List fit(R_ExternalPtrProtected(ptr));
  ScoreGeometryOptions options;
  options.sensitivity = score_flip_sensitivity_from_string(sensitivity);
  options.metric = global_score_metric_from_string(metric);
  post_expected<ScoreComponents> out;
  if (!Rf_isNull(partable)) {
    if (c.estimator == "ML2S" || metric != "expected")
      Rcpp::stop("score_components(): nested scores support ML/FIML with the expected metric");
    Rcpp::List sample = Rcpp::List::create(Rcpp::_["S"] = fit["S"],Rcpp::_["nobs"] = fit["nobs"],
        Rcpp::_["mean"] = fit.containsElementNamed("sample_mean") ? SEXP(fit["sample_mean"]) : R_NilValue);
    auto h1 = ctx_from_partable_sample_stats(partable,sample,"score_components H1");
    out = nested_score_components(h1.pt,h1.rep,c.ctx.pt,c.ctx.rep,
        c.estimator == "ML" ? &c.ctx.samp : nullptr,c.raw,
        c.estimator == "ML" ? nullptr : &c.pack,c.estimates,options.sensitivity);
  } else if (c.estimator == "ML2S") {
    if (fit.containsElementNamed("stage1_regularization") && !Rf_isNull(fit["stage1_regularization"]))
      Rcpp::stop("score_components(): regularized ML2S Stage 1 is unsupported");
    const auto weight = fit.containsElementNamed("stage2_weight") ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
    if (weight != "nt" && weight != "NT") Rcpp::stop("score_components(): only fixed-NT ML2S is supported");
    std::unique_ptr<FimlH1> owned_h1;
    const auto& h1 = fiml_h1_for_fit(fit,c.raw,c.pack,owned_h1);
    std::unique_ptr<SaturatedMoments> owned_sm;
    const auto& sm = fiml_saturated_for_fit(fit,c.raw,c.pack,h1,owned_sm);
    out = global_score_components_ml2s(c.ctx.pt,c.ctx.rep,c.raw,c.pack,h1,sm,c.estimates,options);
  } else out = global_score_components(c.ctx.pt,c.ctx.rep,c.raw,c.pack,c.estimates,options);
  if (!out) stop_post(out.error());
  Rcpp::List ans = Rcpp::List::create(Rcpp::_["score"] = Rcpp::wrap(out->score),
    Rcpp::_["metric"] = Rcpp::wrap(out->metric), Rcpp::_["sensitivity"] = Rcpp::wrap(out->sensitivity),
    Rcpp::_["nuisance"] = Rcpp::wrap(out->nuisance), Rcpp::_["directions"] = Rcpp::wrap(out->directions),
    Rcpp::_["rows"] = Rcpp::wrap(out->rows), Rcpp::_["row_stratum"] = Rcpp::wrap(out->row_stratum),
    Rcpp::_["nobs"] = Rcpp::wrap(out->n_obs), Rcpp::_["influence_rows"] = out->influence_rows,
    Rcpp::_["sensitivity_shrinkage"] = out->sensitivity_shrinkage);
  ans["native"] = handle(std::move(*out), "magmaan_score_components");
  return ans;
}
Rcpp::List projected_result(ProjectedScore value) {
  Rcpp::List out = Rcpp::List::create(Rcpp::_["statistic"] = value.statistic,
    Rcpp::_["df"] = value.score.size(), Rcpp::_["score"] = Rcpp::wrap(value.score),
    Rcpp::_["metric"] = Rcpp::wrap(value.metric), Rcpp::_["meat"] = Rcpp::wrap(value.meat),
    Rcpp::_["projection"] = Rcpp::wrap(value.projection),Rcpp::_["nobs"] = value.n_obs,
    Rcpp::_["influence_rows"] = value.influence_rows,Rcpp::_["retained_rows"] = value.rows.rows() > 0);
  out["native"] = handle(std::move(value), "magmaan_projected_score");
  return out;
}
Rcpp::List project(SEXP ptr, bool retain, bool center) {
  auto out = project_scores(get<ScoreComponents>(ptr,"magmaan_score_components"),retain,center);
  if (!out) stop_post(out.error());
  return projected_result(std::move(*out));
}
Rcpp::List quadratic(Rcpp::NumericVector score, Rcpp::NumericMatrix metric, SEXP meat) {
  Eigen::MatrixXd B;
  if (!Rf_isNull(meat)) B = Rcpp::as<Eigen::MatrixXd>(meat);
  auto out = score_quadratic(Rcpp::as<Eigen::VectorXd>(score),Rcpp::as<Eigen::MatrixXd>(metric),B);
  if (!out) stop_post(out.error());
  return projected_result(std::move(*out));
}
Rcpp::List reference(SEXP ptr, bool spectrum) {
  const auto& s = get<ProjectedScore>(ptr,"magmaan_projected_score");
  Rcpp::List out = Rcpp::List::create(Rcpp::_["statistic"] = s.statistic,Rcpp::_["df"] = s.score.size());
  if (spectrum) {
    auto value = score_spectrum(s);
    if (!value) stop_post(value.error());
    out["eigenvalues"] = Rcpp::wrap(*value);
    out["mean_scale"] = value->sum()/static_cast<double>(value->size());
  } else {
    auto value = score_mean_scale(s);
    if (!value) stop_post(value.error());
    out["mean_scale"] = *value;
  }
  return out;
}
Rcpp::List resample(SEXP ptr, int n, double seed, std::string multiplier, double skew) {
  if (!std::isfinite(seed) || seed < 0 || seed != std::floor(seed) || seed > 9007199254740991.0)
    Rcpp::stop("resample_scores(): invalid seed");
  auto out = resample_scores(get<ProjectedScore>(ptr,"magmaan_projected_score"),n,
      static_cast<std::uint64_t>(seed),score_flip_multiplier_from_string(multiplier),skew);
  if (!out) stop_post(out.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = out->statistic,Rcpp::_["p_value"] = out->p_value,
    Rcpp::_["mc_se"] = out->mc_se,Rcpp::_["n_flips"] = out->n_flips);
}
}
