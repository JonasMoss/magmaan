// Binding-owned immutable handles. Included by fit_prepared.cpp to share the result
// converters; SEM computations remain in the core estimators.
#pragma once
#include "magmaan/spec/lin_constraints.hpp"

namespace prepared {
using namespace magmaan;
// Internal trace counts binding-owned structural preparation, not numerical workspaces.
inline std::size_t structural_preparations = 0;
struct Model {
  Ctx ctx;
  spec::Starts starts;
  std::string kind;
  std::string parameterization;
  std::vector<std::vector<std::int32_t>> levels;
  std::optional<FitError> association_error;
};
struct Data {
  std::string kind;
  data::RawData raw;
  data::SampleStats sample;
  data::OrdinalStats ordinal;
  data::MixedOrdinalStats mixed;
  std::optional<estimate::fiml::FIMLPack> pack;
  std::vector<std::vector<std::string>> names;
  bool meanstructure = false;
};
struct Weight {
  std::string method;
  data::OrdinalStats ordinal;
  data::MixedOrdinalStats mixed;
  estimate::gmm::Weight continuous;
  data::OrdinalGammaCache cache;
};

template<class T> T& get(SEXP ptr, const char* tag) {
  if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install(tag) ||
      !R_ExternalPtrAddr(ptr))
    Rcpp::stop("magmaan: invalid or restored %s handle; prepare it again in this R process", tag);
  return *Rcpp::XPtr<T>(ptr);
}
template<class T> SEXP handle(T value, const char* tag) {
  return Rcpp::XPtr<T>(new T(std::move(value)), true, Rf_install(tag));
}

template<class Moments, class Stats> void copy_moments(const Moments& m, Stats& s) {
  s.R = m.R; s.thresholds = m.thresholds;
  s.threshold_ov = m.threshold_ov; s.threshold_level = m.threshold_level;
  s.n_obs = m.n_obs; s.n_levels = m.n_levels; s.ov_names = m.ov_names;
  s.NACOV.resize(m.R.size()); s.W_dwls.resize(m.R.size()); s.W_wls.resize(m.R.size());
}
void copy_mixed(const data::MixedOrdinalMoments& m, data::MixedOrdinalStats& s) {
  copy_moments(m, s);
  s.mean = m.mean; s.ordered = m.ordered; s.moments = m.moments;
}

template<class Stats> void copy_weights(data::OrdinalGammaCache cache, Stats& s) {
  auto ok = data::ordinal_gamma_cache_ensure_dwls_weights(cache);
  if (!ok) stop_post(ok.error());
  for (std::size_t b = 0; b < cache.blocks.size(); ++b) {
    s.W_dwls[b] = cache.blocks[b].w_dwls;
    if (cache.blocks[b].has_full) s.NACOV[b] = cache.blocks[b].gamma;
  }
}

SEXP model(SEXP partable, std::string kind, Rcpp::Nullable<Rcpp::List> schema) {
  auto parsed = partable_from_arg(partable, "prepare_model");
  Model m;
  m.ctx.pt = std::move(parsed.structure); m.ctx.names = std::move(parsed.names);
  m.ctx.pt.group_equal = group_equal_attr(partable);
  m.starts = std::move(parsed.starts); m.kind = kind;
  m.parameterization = ordinal_parameterization_attr(partable);
  if (kind == "ordinal") {
    auto valid = estimate::validate_ordinal_association_model(m.ctx.pt, &m.ctx.names.row_user);
    if (!valid) m.association_error = valid.error();
    auto s = ordinal_stats_from_arg(Rcpp::List(schema.get()));
    m.levels = s.n_levels;
    auto ok = estimate::prepare_ordinal_partable(m.ctx.pt, s,
        ordinal_parameterization_from_string(m.parameterization), &m.starts, &m.ctx.names.row_user);
    if (!ok) stop_fit(ok.error());
    // Schema preparation can compact automatic intercept coordinates. Resolve
    // ordered affine rows against the final free map used by configured fits.
    spec::resolve_lin_constraints(m.ctx.pt, m.ctx.names);
  } else if (kind == "mixed") {
    auto s = mixed_ordinal_stats_from_arg(Rcpp::List(schema.get()));
    m.levels = s.n_levels;
    auto ok = estimate::prepare_mixed_ordinal_delta_partable(m.ctx.pt, s, &m.starts);
    if (!ok) stop_fit(ok.error());
  }
  ++structural_preparations;
  auto rep = model::build_matrix_rep(m.ctx.pt, &m.ctx.names);
  if (!rep) stop_model(rep.error());
  m.ctx.rep = std::move(*rep);
  if (m.ctx.rep.ov_names.empty()) Rcpp::stop("magmaan: empty model");
  m.ctx.ov_names = m.ctx.rep.ov_names[0];
  m.ctx.meanstructure = has_meanstructure(m.ctx.pt);
  return handle(std::move(m), "magmaan_prepared_model");
}

SEXP dataset(SEXP model_ptr, SEXP X, std::string kind, Rcpp::List ordered) {
  const auto& m = get<Model>(model_ptr, "magmaan_prepared_model");
  Data d; d.kind = kind; d.names = m.ctx.rep.ov_names;
  d.meanstructure = m.ctx.meanstructure;
  if (kind == "moments" && TYPEOF(X) == VECSXP && Rcpp::List(X).containsElementNamed("S")) {
    Rcpp::List ss(X);
    if (!ss.containsElementNamed("nobs")) Rcpp::stop("magmaan: sample statistics need nobs");
    SEXP mean = ss.containsElementNamed("mean") ? SEXP(ss["mean"]) : R_NilValue;
    d.sample = ctx_from_parts(m.ctx.pt, m.ctx.names, ss["S"], ss["nobs"], mean,
                              true, &m.ctx.rep).samp;
    return handle(std::move(d), "magmaan_prepared_data");
  }
  d.raw = fiml_raw_from_arg(m.ctx.rep, X);
  if (kind == "raw") {
    if (!m.ctx.meanstructure) Rcpp::stop("magmaan: prepare_model(..., meanstructure = TRUE) for FIML");
    auto ok = estimate::fiml::validate_fiml_fixed_x_missing_policy(m.ctx.pt, d.raw);
    if (!ok) stop_fit(ok.error());
    auto pack = estimate::fiml::fiml_pack(d.raw);
    if (!pack) stop_fit(pack.error());
    d.sample = pack->start_stats; d.pack = std::move(*pack);
  } else {
    for (const auto& x : d.raw.X)
      if (!x.allFinite()) Rcpp::stop("magmaan: non-finite values in prepared data");
    d.raw.mask.clear();
    if (kind == "moments") {
      auto ss = data::sample_stats_from_raw(d.raw);
      if (!ss) stop_post(ss.error());
      d.sample = std::move(*ss);
    } else {
      const auto plan = data::ordinal_weight_plan(data::OrdinalWorkspacePurpose::FitOnly,
                                                 data::OrdinalEstimatorKind::ULS);
      if (kind == "ordinal") {
        auto ws = data::ordinal_workspace_from_integer_data(d.raw.X, plan);
        if (!ws) stop_post(ws.error());
        copy_moments(ws->moments, d.ordinal);
        d.ordinal.ov_names = d.names;
        if (d.ordinal.n_levels != m.levels) Rcpp::stop("magmaan: ordinal category schema changed; prepare a new model");
      } else if (kind == "mixed") {
        auto masks = Rcpp::as<std::vector<std::vector<std::int32_t>>>(ordered);
        auto ws = data::mixed_ordinal_workspace_from_data(d.raw.X, masks, plan);
        if (!ws) stop_post(ws.error());
        copy_mixed(ws->moments, d.mixed); d.mixed.ov_names = d.names;
        if (d.mixed.n_levels != m.levels) Rcpp::stop("magmaan: ordinal category schema changed; prepare a new model");
      } else Rcpp::stop("magmaan: unsupported data kind");
    }
  }
  if (!d.meanstructure) d.sample.mean.clear();
  return handle(std::move(d), "magmaan_prepared_data");
}

Rcpp::List weight(SEXP data_ptr, std::string method, SEXP W, bool full,
                  SEXP model_ptr, double dls_a) {
  const auto& d = get<Data>(data_ptr, "magmaan_prepared_data");
  Weight w; w.method = method;
  // Dense per-block W for the R-visible return. Stays empty on the ordinal /
  // mixed paths, which is what `Rcpp::wrap` saw before too (an empty list).
  std::vector<Eigen::MatrixXd> W_dense;
  if (method != "DWLS" && method != "WLS" && method != "ULS" && method != "GLS" && method != "DLS")
    Rcpp::stop("magmaan: weight method must be ULS, GLS, DWLS, WLS or DLS");
  if (d.kind == "ordinal" || d.kind == "mixed") {
    if (d.kind == "mixed" && (method == "GLS" || method == "DLS" || !Rf_isNull(W)))
      Rcpp::stop("magmaan: mixed fixed-weight expansion is deferred");
    if (!Rf_isNull(W) && method != "WLS" && method != "DWLS")
      Rcpp::stop("magmaan: custom ordinal W requires WLS or DWLS");
    const auto plan = data::ordinal_weight_plan(data::OrdinalWorkspacePurpose::FitOnly,
                                               data::OrdinalEstimatorKind::DWLS);
    if (d.kind == "ordinal") {
      if (method == "ULS" && !full) {
        w.ordinal = d.ordinal;
      } else if (full || !Rf_isNull(W) || method == "WLS" || method == "GLS" || method == "DLS") {
        auto s = data::ordinal_stats_from_integer_data(d.raw.X, method == "WLS" && Rf_isNull(W));
        if (!s) stop_post(s.error());
        w.ordinal = std::move(*s);
        w.cache = data::ordinal_gamma_cache_from_stats(w.ordinal);
      } else {
        auto ws = data::ordinal_workspace_from_integer_data(d.raw.X, plan);
        if (!ws) stop_post(ws.error());
        w.ordinal = d.ordinal; copy_weights(ws->gamma_cache, w.ordinal);
        w.cache = std::move(ws->gamma_cache);
      }
      if (!Rf_isNull(W)) {
        auto blocks = wls_dense_from_arg(W, w.ordinal.R.size());
        if (method == "DWLS") w.ordinal.W_dwls = std::move(blocks);
        else w.ordinal.W_wls = std::move(blocks);
        w.cache = data::ordinal_gamma_cache_from_stats(w.ordinal);
      } else if (method == "GLS" || method == "DLS") {
        auto s = estimate::frontier::ordinal_stats_with_stage2_weight(w.ordinal,
            method == "GLS" ? estimate::gmm::FixedWeightKind::Nt : estimate::gmm::FixedWeightKind::Dls,
            {.a = dls_a});
        if (!s) stop_post(s.error());
        w.ordinal = std::move(*s);
        w.cache = data::ordinal_gamma_cache_from_stats(w.ordinal);
      }
      w.ordinal.ov_names = d.names;
    } else {
      if (method == "ULS" && !full) {
        w.mixed = d.mixed;
      } else if (full || method == "WLS") {
        auto s = data::mixed_ordinal_stats_from_data(d.raw.X, d.mixed.ordered, method == "WLS");
        if (!s) stop_post(s.error());
        w.mixed = std::move(*s);
        w.cache = data::ordinal_gamma_cache_from_stats(w.mixed);
      } else {
        auto ws = data::mixed_ordinal_workspace_from_data(d.raw.X, d.mixed.ordered, plan);
        if (!ws) stop_post(ws.error());
        w.mixed = d.mixed; copy_weights(ws->gamma_cache, w.mixed);
        w.cache = std::move(ws->gamma_cache);
      }
      w.mixed.ov_names = d.names;
    }
  } else if (d.kind == "moments") {
    if (!Rf_isNull(W)) {
      if (method != "WLS" && method != "DWLS") Rcpp::stop("magmaan: custom W requires WLS or DWLS");
      W_dense = wls_dense_from_arg(W, d.sample.S.size());
    } else {
      if (d.raw.X.empty() && method != "ULS" && method != "GLS")
        Rcpp::stop("magmaan: empirical weights require raw data or explicit W");
      const auto& model = get<Model>(model_ptr, "magmaan_prepared_model");
      auto evaluator = lvm::ModelEvaluator::build(model.ctx.pt, model.ctx.rep);
      if (!evaluator) stop_model(evaluator.error());
      auto x0 = estimate::simple_start_values(model.ctx.pt, model.ctx.rep, d.sample);
      if (!x0) stop_fit(x0.error());
      auto metric = estimate::gmm::fixed_moment_weight(*evaluator, d.sample, *x0,
          ordinal_stage2_weight_from_string(method), d.raw.X.empty() ? nullptr : &d.raw,
          {.a = dls_a});
      if (!metric) stop_fit(metric.error());
      w.continuous = std::move(*metric);
      for (const auto& block : w.continuous) W_dense.push_back(block.to_dense());
    }
    if (W_dense.size() != d.sample.S.size()) Rcpp::stop("magmaan: weight block count mismatch");
    for (std::size_t b = 0; b < W_dense.size(); ++b) {
      const auto p = d.sample.S[b].rows();
      const auto q = p * (p + 1) / 2 + (d.meanstructure ? p : 0);
      if (W_dense[b].rows() != q || W_dense[b].cols() != q || !W_dense[b].allFinite() ||
          !W_dense[b].isApprox(W_dense[b].transpose()))
        Rcpp::stop("magmaan: W must be finite, symmetric and match the moment dimensions");
      Eigen::LLT<Eigen::MatrixXd> llt(W_dense[b]);
      if (llt.info() != Eigen::Success) Rcpp::stop("magmaan: W must be positive definite");
    }
    if (!Rf_isNull(W)) w.continuous = dense_weight_or_stop(W_dense, "magmaan: prepare_weight W");
  } else Rcpp::stop("magmaan: weights require moment data");
  // Keep the dataset alive and reject accidentally reusing its weight elsewhere.
  Rcpp::List out = Rcpp::List::create(Rcpp::_["W"] = Rcpp::wrap(W_dense));
  if (d.kind == "ordinal") out["stats"] = ordinal_stats_to_r(w.ordinal);
  if (d.kind == "mixed") out["stats"] = mixed_ordinal_stats_to_r(w.mixed);
  Rcpp::RObject ptr(handle(std::move(w), "magmaan_prepared_weight"));
  R_SetExternalPtrProtected(ptr, data_ptr);
  out["native"] = ptr;
  return out;
}

Rcpp::List fit(SEXP model_ptr, SEXP data_ptr, SEXP weight_ptr, std::string method,
               Rcpp::Nullable<Rcpp::String> optimizer, Rcpp::Nullable<Rcpp::List> control,
               Rcpp::Nullable<Rcpp::List> bounds, std::string covariance,
               std::string target, double penalty_weight, SEXP start_hints) {
  const auto& m = get<Model>(model_ptr, "magmaan_prepared_model");
  const auto& d = get<Data>(data_ptr, "magmaan_prepared_data");
  if (d.names != m.ctx.rep.ov_names || d.meanstructure != m.ctx.meanstructure)
    Rcpp::stop("magmaan: model/data schemas differ");
  const Weight* w = nullptr;
  if (!Rf_isNull(weight_ptr)) {
    w = &get<Weight>(weight_ptr, "magmaan_prepared_weight");
    if (R_ExternalPtrProtected(weight_ptr) != data_ptr) Rcpp::stop("magmaan: weight belongs to another dataset");
    if (w->method != method) Rcpp::stop("magmaan: estimator and weight method disagree");
  }
  auto starts = m.starts;
  if (!Rf_isNull(start_hints)) {
    // Ordinal preparation can renumber free coordinates across groups. Apply
    // row-aligned hints to the prepared free map, rather than the R projection's.
    auto rows = Rcpp::as<std::vector<double>>(start_hints);
    if (rows.size() != m.ctx.pt.size()) Rcpp::stop("magmaan: start hint row count mismatch");
    starts.hint.assign(m.ctx.pt.n_free(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < rows.size(); ++i)
      if (m.ctx.pt.free[i] > 0 && std::isfinite(rows[i]))
        starts.hint[m.ctx.pt.free[i] - 1] = rows[i];
  }
  Ctx ctx = m.ctx; ctx.samp = d.sample;
  if (covariance != "unrestricted" && covariance != "psd" && covariance != "barrier")
    Rcpp::stop("magmaan: invalid covariance policy");
  if (covariance != "unrestricted" && bounds.isNotNull())
    Rcpp::stop("magmaan: covariance policy does not accept additional bounds");
  if (d.kind == "mixed" && covariance == "barrier")
    Rcpp::stop("magmaan: mixed/polyserial barrier fitting is deferred");
  Rcpp::List ctl = control.isNotNull() ? Rcpp::List(control.get()) : Rcpp::List::create();
  if (ctl.containsElementNamed("fitting_options")) {
    check_optim_control_names(ctl, {"fitting_options", "start"});
    if (optimizer.isNotNull()) Rcpp::stop("select optimizer through fitting options");
    auto options = fitting_options_from(Rcpp::as<Rcpp::List>(ctl["fitting_options"]));
    Eigen::VectorXd explicit_start;
    if (ctl.containsElementNamed("start")) explicit_start = Rcpp::as<Eigen::VectorXd>(ctl["start"]);
    if (d.kind == "ordinal") {
      const auto& stats = w ? w->ordinal : d.ordinal;
      ctx.samp.S = stats.R; ctx.samp.n_obs = stats.n_obs; ctx.meanstructure = false;
      auto x0 = ordinal_starts_or_stop(ctx, stats, starts);
      starts.hint.resize(ctx.pt.n_free(), std::numeric_limits<double>::quiet_NaN());
      for (std::size_t i = 0; i < ctx.pt.size(); ++i)
        if (ctx.pt.op[i] == parse::Op::Threshold && ctx.pt.free[i] > 0 &&
            !std::isfinite(starts.hint[ctx.pt.free[i] - 1]))
          starts.hint[ctx.pt.free[i] - 1] = x0(ctx.pt.free[i] - 1);
      auto e = estimate::fit_ordinal_configured(ctx.pt, ctx.rep, stats,
          options, starts, explicit_start, bounds_from_nullable(bounds),
          estimate::OrdinalWeightKind::DWLS,
          ordinal_parameterization_from_string(m.parameterization), &ctx.names.row_user);
      if (!e) stop_fit(e.error());
      auto out = ordinal_fit_result(ctx, stats, *e, &starts, "DWLS", m.parameterization.c_str());
      out["ordinal_computational_weight"] = "DWLS";
      return out;
    }
    if (d.kind == "raw") {
      auto h1 = estimate::lavaan_fiml_h1(d.raw, *d.pack);
      if (!h1) stop_fit(h1.error());
      auto e = estimate::fit_fiml_configured(ctx.pt, ctx.rep, d.raw, *d.pack, *h1,
          options, starts, explicit_start);
      if (!e) stop_fit(e.error());
      auto out = fiml_fit_result(ctx, d.raw, *e, &starts);
      out["fiml_pack"] = fiml_pack_xptr(*d.pack);
      out["fiml_h1"] = fiml_h1_xptr(std::move(*h1));
      return out;
    }
    auto e = estimate::fit_ml_configured(ctx.pt, ctx.rep, ctx.samp, options, starts,
        explicit_start, bounds_from_nullable(bounds));
    if (!e) stop_fit(e.error());
    return fit_result(ctx, *e, &starts, "ML");
  }
  const auto backend = optimizer.isNull() && covariance == "psd" ? estimate::Backend::NloptSlsqp :
      optimizer.isNull() && covariance == "barrier" ? estimate::Backend::Port : backend_from_optimizer_arg(optimizer);
  const auto opts = d.kind == "moments" && method == "ML" && covariance == "psd"
      ? optim_opts_from(control, estimate::frontier::ml_psd_optim_options())
      : d.kind == "moments" && (covariance == "barrier" || method == "ML")
      ? optim_opts_from(control, estimate::ml_optim_options()) : optim_opts_from(control);
  auto penalty_options = multiinfo_options_from(1.25, R_NilValue, target);
  penalty_options.weight = penalty_weight;
  std::optional<estimate::frontier::PenalizedFit> penalty;
  auto decorate = [&](Rcpp::List out) {
    out["covariance_policy"] = covariance;
    if (penalty) {
      out["penalty"] = multiinfo_penalty_to_r(ctx, *penalty);
      out["penalty_inference"] = "not_validated";
    }
    return out;
  };
  const auto bnd = bounds_from_nullable(bounds);
  auto plan = data::ordinal_weight_plan(data::OrdinalWorkspacePurpose::FitOnly,
      method == "ULS" ? data::OrdinalEstimatorKind::ULS :
      method == "DWLS" ? data::OrdinalEstimatorKind::DWLS : data::OrdinalEstimatorKind::WLS,
      m.parameterization == "theta" ? data::OrdinalMomentParameterization::Theta :
                                      data::OrdinalMomentParameterization::Delta);
  auto cache = w ? w->cache : data::OrdinalGammaCache{};
  fit_expected<estimate::Estimates> e;
  if (d.kind == "ordinal") {
    const auto& s = w ? w->ordinal : d.ordinal;
    if (s.n_levels != m.levels) Rcpp::stop("magmaan: model/data category schemas differ");
    if (method == "ML" && m.association_error) stop_fit(*m.association_error);
    if (method == "ML" && (w || bounds.isNotNull()))
      Rcpp::stop("magmaan: ordinal association ML uses no LS weight or bounds");
    if (method != "ML" && method != "ULS" && !w) Rcpp::stop("magmaan: ordinal LS requires a prepared weight");
    ctx.samp.S = s.R; ctx.samp.n_obs = s.n_obs; ctx.meanstructure = false;
    auto x0 = ordinal_starts_or_stop(ctx, s, starts);
    // Schema-only preparation leaves empirical threshold hints empty. Refresh
    // them locally, matching fresh augmentation without changing the structure.
    starts.hint.resize(ctx.pt.n_free(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < ctx.pt.size(); ++i)
      if (ctx.pt.op[i] == parse::Op::Threshold && ctx.pt.free[i] > 0 &&
          !std::isfinite(starts.hint[ctx.pt.free[i] - 1]))
        starts.hint[ctx.pt.free[i] - 1] = x0(ctx.pt.free[i] - 1);
    const auto parameterization = m.parameterization == "theta" ? estimate::OrdinalParameterization::Theta : estimate::OrdinalParameterization::Delta;
    const auto weights = method == "ML" ? estimate::OrdinalWeightKind::DWLS :
        method == "GLS" || method == "DLS" ? estimate::OrdinalWeightKind::WLS :
        ordinal_weight_from_estimator(method, "prepared covariance fit");
    if (covariance == "barrier") {
      auto fit = estimate::frontier::fit_ordinal_multiinfo(ctx.pt, ctx.rep, s, x0,
          method == "ML", weights, parameterization, penalty_options, backend, opts);
      if (!fit) stop_fit(fit.error());
      penalty = std::move(*fit); e = penalty->estimates;
    } else if (covariance == "psd") {
      e = method == "ML" ? estimate::frontier::fit_ml_psd(ctx.pt, ctx.rep, s, x0, backend, opts)
          : estimate::frontier::fit_ordinal_psd(ctx.pt, ctx.rep, s, {}, weights, x0, backend, opts, parameterization);
    } else e = method == "ML"
        ? estimate::frontier::fit_ml(ctx.pt, ctx.rep, s, x0, backend, opts)
        // Default fits replay fit_model()'s equality-aware composition.
        // Explicit lean weights retain the cache-aware estimation-only path;
        // the full-stats overload requires dense Gamma even for DWLS/ULS.
        : std::all_of(s.NACOV.begin(), s.NACOV.end(),
            [](const auto& gamma) { return gamma.size() > 0; })
        ? estimate::fit_ordinal_bounded(ctx.pt, ctx.rep, s, bnd, weights, x0, backend, opts, parameterization)
        : estimate::fit_ordinal_bounded(ctx.pt, ctx.rep,
            data::ordinal_moments_from_stats(s), &cache, bnd, plan, x0, backend, opts);
    if (!e) stop_fit(e.error());
    const auto label = method == "GLS" || method == "DLS" ? "WLS" : method.c_str();
    auto out = ordinal_fit_result(ctx, s, *e, &starts, label, m.parameterization.c_str());
    if (method != "ML") out["ordinal_computational_weight"] =
        method == "GLS" || method == "DLS" ? "WLS" : method;
    if (e->association) {
      Rcpp::List composition = out["composition"];
      composition["algorithm"] = std::string(estimate::backend_name(backend));
      out["composition"] = composition;
    }
    return decorate(out);
  }
  if (d.kind == "mixed") {
    const auto& s = w ? w->mixed : d.mixed;
    if (s.n_levels != m.levels) Rcpp::stop("magmaan: model/data category schemas differ");
    if (!w) Rcpp::stop("magmaan: mixed LS requires a prepared weight");
    ctx.samp.S = s.R; ctx.samp.mean = s.mean; ctx.samp.n_obs = s.n_obs; ctx.meanstructure = true;
    auto x0 = mixed_ordinal_starts_or_stop(ctx, s, starts);
    starts.hint.resize(ctx.pt.n_free(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < ctx.pt.size(); ++i)
      if (ctx.pt.op[i] == parse::Op::Threshold && ctx.pt.free[i] > 0 &&
          !std::isfinite(starts.hint[ctx.pt.free[i] - 1]))
        starts.hint[ctx.pt.free[i] - 1] = x0(ctx.pt.free[i] - 1);
    if (covariance == "psd") e = estimate::frontier::fit_mixed_ordinal_psd(
        ctx.pt, ctx.rep, s, {}, ordinal_weight_from_estimator(method, "prepared mixed PSD"),
        x0, backend, opts, m.parameterization == "theta" ? estimate::OrdinalParameterization::Theta : estimate::OrdinalParameterization::Delta);
    else e = estimate::fit_mixed_ordinal_bounded(ctx.pt, ctx.rep,
        data::mixed_ordinal_moments_from_stats(s), &cache, bnd, plan, x0, backend, opts);
    if (!e) stop_fit(e.error());
    return decorate(mixed_ordinal_fit_result(ctx, s, *e, &starts, method.c_str(), m.parameterization.c_str()));
  }
  std::string start_policy = d.kind == "moments" && covariance == "barrier" ? "scaled-fabin" :
      d.kind == "moments" && method == "ML" ? (covariance == "psd" ? "scaled-fabin" : "layered") :
      d.kind == "moments" && method == "GLS" && covariance == "unrestricted" ? "layered" : "fabin3";
  std::string fallback = "none";
  const auto x0 = start_values_or_stop(ctx, starts, start_policy, &start_policy, &fallback, control,
      d.kind == "moments" && method == "ML" && covariance != "barrier");
  if (d.kind == "raw") {
    if (method != "FIML" || bounds.isNotNull()) Rcpp::stop("magmaan: raw data supports FIML without bounds");
    if (covariance == "barrier") {
      auto fit = estimate::fiml::frontier::fit_fiml_multiinfo(ctx.pt, ctx.rep, d.raw, x0,
          *d.pack, penalty_options, {}, backend, opts);
      if (!fit) stop_fit(fit.error());
      penalty = std::move(*fit); e = penalty->estimates;
    } else if (covariance == "psd") e = estimate::fiml::frontier::fit_fiml_psd(ctx.pt, ctx.rep, d.raw, x0, *d.pack, backend, opts);
    else e = estimate::fit_fiml(ctx.pt, ctx.rep, d.raw, x0, *d.pack,
                          fiml_backend_from_optimizer_arg(optimizer), opts);
    if (!e) stop_fit(e.error());
    Rcpp::List out = fiml_fit_result(ctx, d.raw, *e, &starts);
    out["fiml_pack"] = fiml_pack_xptr(*d.pack);
    return decorate(out);
  }
  if (covariance != "unrestricted") {
    estimate::gmm::Weight metric = w ? w->continuous : estimate::gmm::Weight{};
    if (method == "GLS" && !w) {
      auto evaluator = lvm::ModelEvaluator::build(ctx.pt, ctx.rep);
      if (!evaluator) stop_model(evaluator.error());
      auto W = estimate::gmm::normal_theory_weight(*evaluator, ctx.samp, x0);
      if (!W) stop_fit(W.error());
      metric = std::move(*W);
    }
    if (covariance == "barrier") {
      auto fit = method == "ML" ? estimate::frontier::fit_ml_multiinfo(ctx.pt, ctx.rep, ctx.samp, x0, penalty_options, {}, backend, opts)
          : estimate::frontier::fit_gmm_multiinfo(ctx.pt, ctx.rep, ctx.samp, x0, metric, penalty_options, backend, opts);
      if (!fit) stop_fit(fit.error());
      penalty = std::move(*fit); e = penalty->estimates;
    } else e = method == "ML" ? estimate::frontier::fit_ml_psd(ctx.pt, ctx.rep, ctx.samp, x0, backend, opts)
        : estimate::frontier::fit_gmm_psd(ctx.pt, ctx.rep, ctx.samp, x0, metric, backend, opts);
  } else if (method == "ML") e = estimate::fit_ml(ctx.pt, ctx.rep, ctx.samp, x0, bnd, backend, opts);
  else if (method == "GLS" && !w) e = estimate::fit_gls(ctx.pt, ctx.rep, ctx.samp, x0, bnd, backend, opts);
  else if (method == "ULS" || method == "WLS" || method == "DWLS" || method == "GLS" || method == "DLS")
    e = estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0, w ? w->continuous : estimate::gmm::Weight{}, bnd, backend, opts);
  else Rcpp::stop("magmaan: unsupported prepared estimator");
  if (!e) stop_fit(e.error());
  auto out = fit_result(ctx, *e, &starts,
      method == "DWLS" || method == "DLS" ? "WLS" : method.c_str());
  if (method == "ML" && covariance != "barrier") {
    out["ml_start_policy"] = start_policy;
    out["ml_start_fallback_reason"] = fallback;
  }
  return decorate(out);
}
} // namespace prepared
