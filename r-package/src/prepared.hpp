// Binding-owned immutable handles. Included by fit.cpp to share its result
// converters; SEM computations remain in the core estimators.
#pragma once

namespace prepared {
using namespace magmaan;
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
    auto ok = estimate::prepare_ordinal_delta_partable(m.ctx.pt, s, &m.starts);
    if (!ok) stop_fit(ok.error());
  } else if (kind == "mixed") {
    auto s = mixed_ordinal_stats_from_arg(Rcpp::List(schema.get()));
    m.levels = s.n_levels;
    auto ok = estimate::prepare_mixed_ordinal_delta_partable(m.ctx.pt, s, &m.starts);
    if (!ok) stop_fit(ok.error());
  }
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

Rcpp::List weight(SEXP data_ptr, std::string method, SEXP W, bool full) {
  const auto& d = get<Data>(data_ptr, "magmaan_prepared_data");
  Weight w; w.method = method;
  // Dense per-block W for the R-visible return. Stays empty on the ordinal /
  // mixed paths, which is what `Rcpp::wrap` saw before too (an empty list).
  std::vector<Eigen::MatrixXd> W_dense;
  if (method != "DWLS" && method != "WLS" && method != "ULS") Rcpp::stop("magmaan: weight method must be ULS, DWLS or WLS");
  if (d.kind == "ordinal" || d.kind == "mixed") {
    if (!Rf_isNull(W)) Rcpp::stop("magmaan: custom ordinal weights are not supported here");
    const auto plan = data::ordinal_weight_plan(data::OrdinalWorkspacePurpose::FitOnly,
                                               data::OrdinalEstimatorKind::DWLS);
    if (d.kind == "ordinal") {
      if (method == "ULS" && !full) {
        w.ordinal = d.ordinal;
      } else if (full || method == "WLS") {
        auto s = data::ordinal_stats_from_integer_data(d.raw.X, method == "WLS");
        if (!s) stop_post(s.error());
        w.ordinal = std::move(*s);
        w.cache = data::ordinal_gamma_cache_from_stats(w.ordinal);
      } else {
        auto ws = data::ordinal_workspace_from_integer_data(d.raw.X, plan);
        if (!ws) stop_post(ws.error());
        w.ordinal = d.ordinal; copy_weights(ws->gamma_cache, w.ordinal);
        w.cache = std::move(ws->gamma_cache);
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
    // Two forms are kept deliberately. `dense` is what the validation below
    // checks and what R gets back, exactly as before. `w.continuous` carries the
    // *structured* gmm::BlockWeight the fit consumes — Identity for ULS and
    // Diagonal for DWLS instead of a q x q matrix, which is the point of the
    // BlockWeight retype. Materializing `dense` is not a regression: this
    // function already returned dense blocks to R.
    std::vector<Eigen::MatrixXd> dense;
    std::vector<Eigen::VectorXd> dwls_diag;  // non-empty ⇒ derived DWLS path
    if (method == "ULS") {
      if (!Rf_isNull(W)) Rcpp::stop("magmaan: ULS does not accept a custom W");
      for (const auto& S : d.sample.S) {
        const auto p = S.rows();
        const auto size = p * (p + 1) / 2 + (d.meanstructure ? p : 0);
        dense.emplace_back(Eigen::MatrixXd::Identity(size, size));
      }
    } else if (!Rf_isNull(W)) dense = wls_dense_from_arg(W, d.sample.S.size());
    else {
      if (d.raw.X.empty()) Rcpp::stop("magmaan: empirical weights require raw data or explicit W");
      for (const auto& X : d.raw.X) {
        auto gamma = d.meanstructure ? data::empirical_gamma_with_means(X) : data::empirical_gamma(X);
        if (!gamma) stop_post(gamma.error());
        if (method == "DWLS") {
          if (!gamma->diagonal().allFinite() || (gamma->diagonal().array() <= 0).any())
            Rcpp::stop("magmaan: non-positive Gamma diagonal");
          Eigen::VectorXd dinv = gamma->diagonal().cwiseInverse();
          dense.emplace_back(dinv.asDiagonal());
          dwls_diag.push_back(std::move(dinv));
        } else {
          Eigen::LLT<Eigen::MatrixXd> llt(*gamma);
          if (llt.info() != Eigen::Success) Rcpp::stop("magmaan: empirical Gamma is not positive definite");
          dense.push_back(llt.solve(Eigen::MatrixXd::Identity(gamma->rows(), gamma->cols())));
        }
      }
    }
    if (dense.size() != d.sample.S.size()) Rcpp::stop("magmaan: weight block count mismatch");
    for (std::size_t b = 0; b < dense.size(); ++b) {
      const auto p = d.sample.S[b].rows();
      const auto size = p * (p + 1) / 2 + (d.meanstructure ? p : 0);
      const auto& Wb = dense[b];
      if (Wb.rows() != size || Wb.cols() != size || !Wb.allFinite() || !Wb.isApprox(Wb.transpose()))
        Rcpp::stop("magmaan: W must be finite, symmetric and match the moment dimensions");
      // gmm::dense_weight would accept a positive *semi*definite block. This
      // surface has always required positive definite, so keep the stricter
      // check here rather than silently inheriting the looser one.
      Eigen::LLT<Eigen::MatrixXd> llt(Wb);
      if (llt.info() != Eigen::Success) Rcpp::stop("magmaan: W must be positive definite");
    }
    if (method == "ULS") {
      for (const auto& Wb : dense)
        w.continuous.push_back(estimate::gmm::BlockWeight::identity(Wb.rows()));
    } else if (!dwls_diag.empty()) {
      for (const auto& dg : dwls_diag)
        w.continuous.push_back(estimate::gmm::BlockWeight::diagonal(dg));
    } else {
      // User-supplied W, or an empirical dense Gamma inverse.
      w.continuous =
          magmaanr::dense_weight_or_stop(dense, "magmaan: prepare_weight W");
    }
    W_dense = std::move(dense);
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
               std::string target, double penalty_weight) {
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
  Ctx ctx = m.ctx; ctx.samp = d.sample;
  if (covariance != "unrestricted" && covariance != "psd" && covariance != "barrier")
    Rcpp::stop("magmaan: invalid covariance policy");
  if (covariance != "unrestricted" && bounds.isNotNull())
    Rcpp::stop("magmaan: covariance policy does not accept additional bounds");
  if (d.kind == "mixed" && covariance != "unrestricted")
    Rcpp::stop("magmaan: prepared mixed covariance composition is deferred");
  const auto backend = optimizer.isNull() && covariance == "psd" ? estimate::Backend::NloptSlsqp :
      optimizer.isNull() && covariance == "barrier" ? estimate::Backend::Port : backend_from_optimizer_arg(optimizer);
  const auto opts = covariance == "barrier" ? optim_opts_from(control, estimate::ml_optim_options()) : optim_opts_from(control);
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
    auto x0 = ordinal_starts_or_stop(ctx, s, m.starts);
    const auto parameterization = m.parameterization == "theta" ? estimate::OrdinalParameterization::Theta : estimate::OrdinalParameterization::Delta;
    const auto weights = method == "ML" ? estimate::OrdinalWeightKind::DWLS : ordinal_weight_from_estimator(method, "prepared covariance fit");
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
        : estimate::fit_ordinal_bounded(ctx.pt, ctx.rep,
            data::ordinal_moments_from_stats(s), &cache, bnd, plan, x0, backend, opts);
    if (!e) stop_fit(e.error());
    auto out = ordinal_fit_result(ctx, s, *e, &m.starts, method.c_str(), m.parameterization.c_str());
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
    auto x0 = mixed_ordinal_starts_or_stop(ctx, s, m.starts);
    e = estimate::fit_mixed_ordinal_bounded(ctx.pt, ctx.rep,
        data::mixed_ordinal_moments_from_stats(s), &cache, bnd, plan, x0, backend, opts);
    if (!e) stop_fit(e.error());
    return mixed_ordinal_fit_result(ctx, s, *e, &m.starts, method.c_str(), m.parameterization.c_str());
  }
  const auto x0 = start_values_or_stop(ctx, m.starts, "fabin3", nullptr, nullptr, control);
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
    Rcpp::List out = fiml_fit_result(ctx, d.raw, *e, &m.starts);
    out["fiml_pack"] = fiml_pack_xptr(*d.pack);
    return decorate(out);
  }
  if (covariance != "unrestricted") {
    estimate::gmm::Weight metric = w ? w->continuous : estimate::gmm::Weight{};
    if (method == "GLS") {
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
  else if (method == "GLS") e = estimate::fit_gls(ctx.pt, ctx.rep, ctx.samp, x0, bnd, backend, opts);
  else if (method == "ULS" || method == "WLS" || method == "DWLS")
    e = estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0, w ? w->continuous : estimate::gmm::Weight{}, bnd, backend, opts);
  else Rcpp::stop("magmaan: unsupported prepared estimator");
  if (!e) stop_fit(e.error());
  return decorate(fit_result(ctx, *e, &m.starts, method.c_str()));
}
} // namespace prepared
