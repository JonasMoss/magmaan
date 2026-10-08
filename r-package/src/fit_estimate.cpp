#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

namespace {

Rcpp::List dims_list(const std::vector<lvm::BlockDims>& dd);
Rcpp::DataFrame cells_df(const std::vector<lvm::Cell>& cells);
Rcpp::DataFrame structural_cells_df(const std::vector<lvm::StructuralCell>& sc);
Rcpp::List snlls_fit_result(Ctx& ctx,
                            const magmaan::estimate::Estimates& est,
                            const magmaan::spec::Starts* starts,
                            const char* estimator,
                            const char* backend);
Rcpp::List bounds_to_r(const magmaan::estimate::Bounds& bounds);
std::string bounds_preset_key(Rcpp::Nullable<Rcpp::String> preset);
magmaan::estimate::Bounds twolevel_bounds_from_nullable(
    Rcpp::Nullable<Rcpp::List> bounds,
    Rcpp::Nullable<Rcpp::String> preset,
    const magmaan::spec::LatentStructure& pt,
    const magmaan::model::MatrixRep& rep,
    const magmaan::data::ClusterSampleStats& cs);
inline magmaan::optim::TerminalAuditOptions
audit_opts_from(Rcpp::Nullable<Rcpp::List> audit_options);

Rcpp::List dims_list(const std::vector<lvm::BlockDims>& dd) {
  Rcpp::List out(static_cast<R_xlen_t>(dd.size()));
  for (std::size_t b = 0; b < dd.size(); ++b)
    out[static_cast<R_xlen_t>(b)] = Rcpp::List::create(Rcpp::_["n_observed"] = static_cast<int>(dd[b].n_observed),
                                                       Rcpp::_["n_latent"]   = static_cast<int>(dd[b].n_latent));
  return out;
}

Rcpp::DataFrame cells_df(const std::vector<lvm::Cell>& cells) {
  const R_xlen_t m = static_cast<R_xlen_t>(cells.size());
  Rcpp::CharacterVector mat(m);
  Rcpp::IntegerVector row(m), col(m), block(m);
  Rcpp::LogicalVector used(m);
  for (R_xlen_t i = 0; i < m; ++i) {
    const lvm::Cell& c = cells[static_cast<std::size_t>(i)];
    mat[i]   = std::string(lvm::to_string(c.mat));
    row[i]   = c.row;
    col[i]   = c.col;
    block[i] = c.block;
    used[i]  = c.used;
  }
  Rcpp::List l = Rcpp::List::create(Rcpp::_["mat"] = mat, Rcpp::_["row"] = row, Rcpp::_["col"] = col,
                                    Rcpp::_["block"] = block, Rcpp::_["used"] = used);
  l.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -static_cast<int>(m));
  l.attr("class") = "data.frame";
  return Rcpp::DataFrame(l);
}

Rcpp::DataFrame structural_cells_df(const std::vector<lvm::StructuralCell>& sc) {
  const R_xlen_t m = static_cast<R_xlen_t>(sc.size());
  Rcpp::CharacterVector mat(m);
  Rcpp::IntegerVector row(m), col(m), block(m);
  Rcpp::NumericVector value(m);
  for (R_xlen_t i = 0; i < m; ++i) {
    const lvm::StructuralCell& c = sc[static_cast<std::size_t>(i)];
    mat[i]   = std::string(lvm::to_string(c.mat));
    row[i]   = c.row;
    col[i]   = c.col;
    block[i] = c.block;
    value[i] = c.value;
  }
  Rcpp::List l = Rcpp::List::create(Rcpp::_["mat"] = mat, Rcpp::_["row"] = row, Rcpp::_["col"] = col,
                                    Rcpp::_["block"] = block, Rcpp::_["value"] = value);
  l.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -static_cast<int>(m));
  l.attr("class") = "data.frame";
  return Rcpp::DataFrame(l);
}

Rcpp::List snlls_fit_result(Ctx& ctx,
                            const magmaan::estimate::Estimates& est,
                            const magmaan::spec::Starts* starts,
                            const char* estimator,
                            const char* backend) {
  Rcpp::List out = fit_result(ctx, est, starts, estimator);
  out["backend"] = backend;
  out["snlls_compatible"] = true;
  // SNLLS profile shape: the outer optimizer drives n_nonlinear (β block);
  // n_linear (α block) is profiled out in closed form. The sentinel `-1` in
  // Estimates would mean "no separable split applied" — it should never reach
  // this path, so map negatives to NA so callers can guard cleanly anyway.
  out["n_nonlinear"] = est.n_nonlinear >= 0
      ? Rcpp::IntegerVector::create(est.n_nonlinear)
      : Rcpp::IntegerVector::create(NA_INTEGER);
  out["n_linear"] = est.n_linear >= 0
      ? Rcpp::IntegerVector::create(est.n_linear)
      : Rcpp::IntegerVector::create(NA_INTEGER);
  // SNLLS inner-solve telemetry: count of `profile_at()` cache misses that
  // took the fast Cholesky-on-normal-equations path vs the rank-revealing
  // QR fallback. Same sentinel semantics as `n_nonlinear` / `n_linear`.
  out["n_alpha_solve_fast"] = est.n_alpha_solve_fast >= 0
      ? Rcpp::IntegerVector::create(est.n_alpha_solve_fast)
      : Rcpp::IntegerVector::create(NA_INTEGER);
  out["n_alpha_solve_fallback"] = est.n_alpha_solve_fallback >= 0
      ? Rcpp::IntegerVector::create(est.n_alpha_solve_fallback)
      : Rcpp::IntegerVector::create(NA_INTEGER);
  return out;
}

Rcpp::List bounds_to_r(const magmaan::estimate::Bounds& bounds) {
  return Rcpp::List::create(
      Rcpp::_["lower"] = Rcpp::wrap(bounds.lower),
      Rcpp::_["upper"] = Rcpp::wrap(bounds.upper));
}

std::string bounds_preset_key(Rcpp::Nullable<Rcpp::String> preset) {
  if (preset.isNull()) return "none";
  std::string key = Rcpp::as<std::string>(preset.get());
  const auto first = key.find_first_not_of(" \t\r\n");
  const auto last = key.find_last_not_of(" \t\r\n");
  key = (first == std::string::npos) ? std::string{} :
      key.substr(first, last - first + 1);
  for (char& ch : key) {
    if (ch == '_') ch = '-';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return key;
}

magmaan::estimate::Bounds twolevel_bounds_from_nullable(
    Rcpp::Nullable<Rcpp::List> bounds,
    Rcpp::Nullable<Rcpp::String> preset,
    const magmaan::spec::LatentStructure& pt,
    const magmaan::model::MatrixRep& rep,
    const magmaan::data::ClusterSampleStats& cs) {
  magmaan::estimate::Bounds explicit_bounds = bounds_from_nullable(bounds);
  const std::string key = bounds_preset_key(preset);
  if (key.empty() || key == "none" || key == "no" || key == "false" ||
      key == "unbounded") {
    return explicit_bounds;
  }
  if (!explicit_bounds.empty()) {
    Rcpp::stop("magmaan: fit_twolevel(): pass either explicit `bounds` or a "
               "named bounds preset, not both");
  }

  if (key == "pos.var" || key == "pos-var" || key == "variance" ||
      key == "variance-bounds") {
    auto b_or = magmaan::estimate::variance_bounds(pt);
    if (!b_or.has_value()) stop_post(b_or.error());
    return *b_or;
  }
  if (key != "standard" && key != "wide" && key != "default" &&
      key != "loading" && key != "loadings") {
    Rcpp::stop("magmaan: fit_twolevel(): unsupported bounds preset '%s'",
               key.c_str());
  }

  auto samp_or = magmaan::estimate::twolevel::twolevel_h1_sample_stats(cs, rep);
  if (!samp_or.has_value()) stop_fit(samp_or.error());

  magmaan::post_expected<magmaan::estimate::Bounds> b_or;
  if (key == "standard") {
    b_or = magmaan::estimate::standard_bounds(pt, *samp_or);
  } else if (key == "wide" || key == "default") {
    b_or = magmaan::estimate::wide_bounds(pt, *samp_or);
  } else if (key == "loading" || key == "loadings") {
    b_or = magmaan::estimate::loading_bounds(pt, *samp_or);
  }
  if (!b_or.has_value()) stop_post(b_or.error());
  return *b_or;
}

inline magmaan::optim::TerminalAuditOptions
audit_opts_from(Rcpp::Nullable<Rcpp::List> audit_options) {
  magmaan::optim::TerminalAuditOptions o;  // struct defaults: Absolute, 1e-3
  if (audit_options.isNotNull()) {
    Rcpp::List l(audit_options.get());
    if (l.containsElementNamed("stationarity_mode")) {
      const std::string m = Rcpp::as<std::string>(l["stationarity_mode"]);
      if      (m == "absolute") o.stationarity_mode =
          magmaan::optim::TerminalAuditOptions::StationarityMode::Absolute;
      else if (m == "relative") o.stationarity_mode =
          magmaan::optim::TerminalAuditOptions::StationarityMode::Relative;
      else Rcpp::stop("magmaan: audit_options$stationarity_mode must be "
                      "\"absolute\" or \"relative\" (got \"%s\")", m);
    }
    if (l.containsElementNamed("absolute_tol"))
      o.absolute_tol = Rcpp::as<double>(l["absolute_tol"]);
    if (l.containsElementNamed("stationarity_tol"))
      o.stationarity_tol = Rcpp::as<double>(l["stationarity_tol"]);
    if (l.containsElementNamed("active_bound_tol"))
      o.active_bound_tol = Rcpp::as<double>(l["active_bound_tol"]);
    if (l.containsElementNamed("f_consistency_rel"))
      o.f_consistency_rel = Rcpp::as<double>(l["f_consistency_rel"]);
  }
  return o;
}

}  // namespace

// fit_fit() declaration — the historical entry point lives as a thin alias of
// fit_ml_impl() for the `estimate_fit` magmaan_core slot. Definition is below
// fit_ml_impl so the alias can call it.
Rcpp::List fit_ml_impl(SEXP partable, Rcpp::List sample_stats,
                       Rcpp::Nullable<Rcpp::String> optimizer,
                       Rcpp::Nullable<Rcpp::List>   control,
                       Rcpp::Nullable<Rcpp::List>   bounds);


// =============================================================================

// model_matrix_rep() — mirrors build_matrix_rep(pt). A function of the partable
// alone (no fit / data / θ̂): the LISREL layout that downstream evaluation uses.
// `partable` a partable data.frame. Returns list(form = "PureCFA"|"Reduced",
// dims = list(list(n_observed, n_latent), ...) per block, ov_names = list(<chr>,
// ...) per block — the observed-variable order everything else (Σ̂ columns, the
// u-factor's B rows, casewise vech) is in — lv_names = list(<chr>, ...) per block
// (extended: includes phantom latents in Reduced form), cells = data.frame(mat,
// row, col, block, used) one row per partable row, structural_cells = data.frame(
// mat, row, col, block, value)).
//
// [[Rcpp::export]]
Rcpp::List model_matrix_rep(SEXP partable) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "model_matrix_rep");
  auto rep_or = lvm::build_matrix_rep(parsed.structure, &parsed.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  const lvm::MatrixRep& rep = *rep_or;
  return Rcpp::List::create(
      Rcpp::_["form"]             = std::string(rep.form == lvm::RepForm::PureCFA ? "PureCFA" : "Reduced"),
      Rcpp::_["dims"]             = dims_list(rep.dims),
      Rcpp::_["ov_names"]         = names_list(rep.ov_names),
      Rcpp::_["lv_names"]         = names_list(rep.lv_names),
      Rcpp::_["cells"]            = cells_df(rep.cell_for_row),
      Rcpp::_["structural_cells"] = structural_cells_df(rep.structural_cells));
}

// [[Rcpp::export]]
Rcpp::List bounds_variance_impl(SEXP partable) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "bounds_variance");
  auto b_or = magmaan::estimate::variance_bounds(parsed.structure);
  if (!b_or.has_value()) stop_post(b_or.error());
  return bounds_to_r(*b_or);
}

// [[Rcpp::export]]
Rcpp::List bounds_standard_impl(SEXP partable, Rcpp::List sample_stats) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "bounds_standard");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  auto b_or = magmaan::estimate::standard_bounds(ctx.pt, ctx.samp);
  if (!b_or.has_value()) stop_post(b_or.error());
  return bounds_to_r(*b_or);
}

// [[Rcpp::export]]
Rcpp::List bounds_wide_impl(SEXP partable, Rcpp::List sample_stats) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "bounds_wide");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  auto b_or = magmaan::estimate::wide_bounds(ctx.pt, ctx.samp);
  if (!b_or.has_value()) stop_post(b_or.error());
  return bounds_to_r(*b_or);
}

// [[Rcpp::export]]
Rcpp::List bounds_loading_impl(SEXP partable, Rcpp::List sample_stats) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "bounds_loading");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  auto b_or = magmaan::estimate::loading_bounds(ctx.pt, ctx.samp);
  if (!b_or.has_value()) stop_post(b_or.error());
  return bounds_to_r(*b_or);
}

// [[Rcpp::export]]
Rcpp::List fit_fit(SEXP partable, Rcpp::List sample_stats,
                   Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                   Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                   Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  return fit_ml_impl(partable, sample_stats, optimizer, control, bounds);
}

// fit_ml() — public ML spelling. Threads through an `optimizer` string and
// optional `control` list. The default start is the layered moment start
// (control$start = "scaled-fabin" gives the former default). Backends:
//   "nlopt-lbfgs" (default), "ipopt", "port", "nlopt-slsqp",
//   "nlopt-tnewton", "nlopt-var2"  (any scalar-shape backend)
// "ceres" / "ceres-bfgs" are rejected — Ceres applies to the LS path only.
// "nlopt-bobyqa" requires finite bounds, supplied via `bounds`.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_impl(SEXP partable, Rcpp::List sample_stats,
                       Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                       Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                       Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_ml");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  reuse_identification(ctx, partable);
  if (control.isNotNull()) {
    Rcpp::List ctl(control.get());
    if (ctl.containsElementNamed("fitting_options")) {
      check_optim_control_names(ctl, {"fitting_options", "start"});
      if (optimizer.isNotNull()) Rcpp::stop("select optimizer through fitting options");
      auto options = fitting_options_from(Rcpp::as<Rcpp::List>(ctl["fitting_options"]));
      Eigen::VectorXd explicit_start;
      if (ctl.containsElementNamed("start")) explicit_start = Rcpp::as<Eigen::VectorXd>(ctl["start"]);
      auto est = magmaan::estimate::fit_ml_configured(ctx.pt, ctx.rep, ctx.samp,
          options, starts, explicit_start, bounds_from_nullable(bounds));
      if (!est) stop_fit(est.error());
      return fit_result(ctx, *est, &starts, "ML");
    }
  }
  std::string start_policy = "layered";
  std::string start_fallback_reason = "none";

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, start_policy, &start_policy, &start_fallback_reason, control, true);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_ml(ctx.pt, ctx.rep, ctx.samp, x0,
      bounds_from_nullable(bounds), backend, optim_opts_from(control, magmaan::estimate::ml_optim_options()));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  Rcpp::List out = fit_result(ctx, est, &starts, "ML");
  out["ml_start_policy"] = start_policy;
  out["ml_start_fallback_reason"] = start_fallback_reason;
  return out;
}

// Frozen complete-continuous weight selection, shared with prepared fitting.
// [[Rcpp::export]]
Rcpp::List fixed_moment_weight_impl(SEXP partable, Rcpp::List sample_stats,
                                    std::string method, SEXP raw_data = R_NilValue,
                                    double dls_a = 0.5) {
  auto parsed = partable_from_arg(partable, "fixed_moment_weight");
  auto starts = std::move(parsed.starts);
  auto ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names), sample_stats);
  auto evaluator = lvm::ModelEvaluator::build(ctx.pt, ctx.rep);
  if (!evaluator) stop_model(evaluator.error());
  std::optional<magmaan::data::RawData> raw;
  if (!Rf_isNull(raw_data)) {
    raw = fiml_raw_from_arg(ctx.rep, raw_data); raw->mask.clear();
    for (const auto& X : raw->X) if (!X.allFinite()) Rcpp::stop("magmaan: fixed weights require complete observations");
  }
  auto weight = magmaan::estimate::gmm::fixed_moment_weight(*evaluator, ctx.samp,
      start_values_or_stop(ctx, starts, "simple"), ordinal_stage2_weight_from_string(method),
      raw ? &*raw : nullptr, {.a = dls_a});
  if (!weight) stop_fit(weight.error());
  std::vector<Eigen::MatrixXd> blocks;
  for (const auto& block : *weight) blocks.push_back(block.to_dense());
  Rcpp::List weights = Rcpp::wrap(blocks);
  weights.attr("cancels_measurement_units") =
      std::all_of(weight->begin(), weight->end(), [](const auto& block) {
        return block.cancels_measurement_units();
      });
  magmaan::estimate::frontier::NewtonDerivatives retained;
  retained.ls_weight=std::move(*weight);
  return Rcpp::List::create(Rcpp::_["W"] = weights,
      Rcpp::_["retained_ls_weights"]=retained_ls_weights_to_r(retained));
}

// Shared non-mixed moment discrepancy plus the model barrier.
// [[Rcpp::export]]
Rcpp::List fit_moments_barrier_impl(
    SEXP partable, Rcpp::List sample_stats, std::string estimator = "ML",
    SEXP W = R_NilValue, std::string target = "joint", double weight = 0.25,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue) {
  auto parsed = partable_from_arg(partable, "fit_moments_barrier");
  auto starts = std::move(parsed.starts);
  auto ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names), sample_stats);
  reuse_identification(ctx, partable);
  const auto x0 = start_values_or_stop(ctx, starts, "scaled-fabin", nullptr, nullptr, control);
  auto options = multiinfo_options_from(1.25, R_NilValue, target);
  options.weight = weight;
  const auto backend = optimizer.isNull() ? magmaan::estimate::Backend::Port : backend_from_optimizer_arg(optimizer);
  magmaan::fit_expected<magmaan::estimate::frontier::PenalizedFit> fit;
  if (estimator == "ML") {
    if (!Rf_isNull(W)) Rcpp::stop("magmaan: ML barrier uses no LS weight");
    fit = magmaan::estimate::frontier::fit_ml_multiinfo(ctx.pt, ctx.rep, ctx.samp, x0, options,
        {}, backend, optim_opts_from(control, magmaan::estimate::ml_optim_options()));
  } else {
    magmaan::estimate::gmm::Weight w;
    if (estimator == "WLS") {
      if (Rf_isNull(W)) Rcpp::stop("magmaan: WLS barrier requires W");
      w = wls_from_arg(W, ctx.samp.S.size());
    } else if (estimator == "GLS") {
      if (!Rf_isNull(W)) Rcpp::stop("magmaan: GLS builds its own weight");
      auto evaluator = lvm::ModelEvaluator::build(ctx.pt, ctx.rep);
      if (!evaluator) stop_model(evaluator.error());
      auto metric = magmaan::estimate::gmm::normal_theory_weight(*evaluator, ctx.samp, x0);
      if (!metric) stop_fit(metric.error());
      w = std::move(*metric);
    } else if (estimator != "ULS" || !Rf_isNull(W)) Rcpp::stop("magmaan: invalid barrier discrepancy/weight");
    fit = magmaan::estimate::frontier::fit_gmm_multiinfo(ctx.pt, ctx.rep, ctx.samp, x0,
        std::move(w), options, backend, optim_opts_from(control, magmaan::estimate::ml_optim_options()));
  }
  if (!fit) stop_fit(fit.error());
  auto out = fit_result(ctx, fit->estimates, &starts, estimator.c_str());
  out["penalty"] = multiinfo_penalty_to_r(ctx, *fit);
  out["covariance_policy"] = "barrier";
  out["penalty_inference"] = "not_validated";
  return out;
}

// fit_twolevel() — two-level (multilevel) normal-theory ML over clustered raw
// data. `data` holds the observed columns (named); `cluster_id` is a (per-row)
// cluster index; `group_id` (optional) is a per-row 0-based, contiguous group
// index for multi-group models (NULL ⇒ single group). Returns theta-hat,
// observed-info SEs, the LRT chi-square (vs the saturated H1), df, a fitted
// partable, and basic diagnostics.
//
// DIRECT-CORE BYPASS (v1 contract): this builds matrix_rep + cluster sample
// stats + start values and drives estimate::twolevel directly; it does *not*
// route through api::fit / sem.cpp.
//
// [[Rcpp::export]]
Rcpp::List fit_twolevel_impl(SEXP partable, Rcpp::NumericMatrix data,
                             Rcpp::IntegerVector cluster_id,
                             Rcpp::Nullable<Rcpp::IntegerVector> group_id = R_NilValue,
                             Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                             Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                             Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue,
                             Rcpp::Nullable<Rcpp::String> bounds_preset = R_NilValue) {
  namespace tl = magmaan::estimate::twolevel;
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_twolevel");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  magmaan::spec::LatentStructure pt = std::move(parsed.structure);
  magmaan::spec::LatentNames names = std::move(parsed.names);

  auto rep_or = magmaan::model::build_matrix_rep(pt, &names);
  if (!rep_or.has_value()) {
    Rcpp::stop("magmaan: fit_twolevel(): matrix_rep failed: %s",
               rep_or.error().detail.c_str());
  }
  const magmaan::model::MatrixRep rep = std::move(*rep_or);
  if (rep.ov_names.empty()) Rcpp::stop("magmaan: fit_twolevel(): empty model");

  // v1 (shared observed set): every block — within/between, all groups — carries
  // the same p observed variables in the same order, so a single column
  // permutation to the block-0 within order serves every group.
  for (std::size_t b = 1; b < rep.ov_names.size(); ++b) {
    if (rep.ov_names[b] != rep.ov_names[0]) {
      Rcpp::stop("magmaan: fit_twolevel(): v1 requires a shared observed set "
                 "with identical ordering across level/group blocks");
    }
  }

  // Order data columns to the within-block observed order (shared variable set).
  const std::vector<int> perm = perm_for_cols(data, rep.ov_names[0], "data");
  const int n = data.nrow();
  const int p = static_cast<int>(perm.size());
  Eigen::MatrixXd X(n, p);
  for (int r = 0; r < n; ++r)
    for (int k = 0; k < p; ++k)
      X(r, k) = data(r, perm[static_cast<std::size_t>(k)]);

  if (cluster_id.size() != n) {
    Rcpp::stop("magmaan: fit_twolevel(): cluster_id length (%d) != nrow(data) (%d)",
               static_cast<int>(cluster_id.size()), n);
  }
  std::vector<std::int32_t> cid(cluster_id.begin(), cluster_id.end());

  // Per-row group index. NULL ⇒ a single group (all rows). When supplied, the
  // values are taken as 0-based, contiguous group indices that line up with the
  // model's per-group level blocks (no first-appearance remap) so the index is
  // unambiguous and matches lavaan's fixed group ordering.
  std::vector<int> grow(static_cast<std::size_t>(n), 0);
  int ng = 1;
  if (group_id.isNotNull()) {
    Rcpp::IntegerVector gv(group_id.get());
    if (gv.size() != n) {
      Rcpp::stop("magmaan: fit_twolevel(): group_id length (%d) != nrow(data) (%d)",
                 static_cast<int>(gv.size()), n);
    }
    int gmax = -1;
    for (int r = 0; r < n; ++r) {
      const int g = gv[r];
      if (g < 0) Rcpp::stop("magmaan: fit_twolevel(): group_id must be 0-based");
      grow[static_cast<std::size_t>(r)] = g;
      if (g > gmax) gmax = g;
    }
    ng = gmax + 1;
    std::vector<char> seen(static_cast<std::size_t>(ng), 0);
    for (int r = 0; r < n; ++r)
      seen[static_cast<std::size_t>(grow[static_cast<std::size_t>(r)])] = 1;
    for (int g = 0; g < ng; ++g)
      if (!seen[static_cast<std::size_t>(g)])
        Rcpp::stop("magmaan: fit_twolevel(): group_id has an empty group index %d", g);
  }

  std::vector<std::int32_t> cols(static_cast<std::size_t>(p));
  for (int k = 0; k < p; ++k) cols[static_cast<std::size_t>(k)] = k;

  // Build per-group two-level sufficient statistics and stitch them into one
  // ClusterSampleStats. The per-group sufficient statistics are independent (no
  // cross-group coupling), so calling the single-group cluster_sample_stats once
  // per group subset is numerically identical to one multi-group pass.
  //
  // TODO(stream-A multi-group core): once data::cluster_sample_stats gains its
  // group selector (and data_from_cluster accepts multiple groups), replace this
  // per-group stitch with that single canonical multi-group call. The downstream
  // estimation (objective/H1/information) already loops over cs.groups, so only
  // this construction step changes.
  magmaan::data::ClusterSampleStats cs;
  cs.within_ov_index = cols;
  cs.between_ov_index = cols;
  cs.groups.reserve(static_cast<std::size_t>(ng));
  for (int g = 0; g < ng; ++g) {
    std::vector<int> rows;
    rows.reserve(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r)
      if (grow[static_cast<std::size_t>(r)] == g) rows.push_back(r);
    const int ng_rows = static_cast<int>(rows.size());
    Eigen::MatrixXd Xg(ng_rows, p);
    std::vector<std::int32_t> cidg(static_cast<std::size_t>(ng_rows));
    for (int i = 0; i < ng_rows; ++i) {
      Xg.row(i) = X.row(rows[static_cast<std::size_t>(i)]);
      cidg[static_cast<std::size_t>(i)] =
          cid[static_cast<std::size_t>(rows[static_cast<std::size_t>(i)])];
    }
    auto cg_or = magmaan::data::cluster_sample_stats(Xg, cidg, cols, cols);
    if (!cg_or.has_value()) {
      Rcpp::stop("magmaan: fit_twolevel(): cluster statistics failed: %s",
                 cg_or.error().detail.c_str());
    }
    cs.groups.push_back(std::move(cg_or->groups.front()));
  }

  // Multi-group two-level needs per-group within/between level blocks in the rep.
  const auto pairs = magmaan::model::level_block_pairs(rep);
  if (static_cast<int>(pairs.size()) < ng) {
    Rcpp::stop("magmaan: fit_twolevel(): data has %d group(s) but the model has "
               "%d level-block group(s); multi-group two-level requires a "
               "grouped two-level model", ng, static_cast<int>(pairs.size()));
  }

  auto x0_or = tl::twolevel_start_values(pt, rep, cs, starts);
  if (!x0_or.has_value()) stop_fit(x0_or.error());
  const Eigen::VectorXd x0 = std::move(*x0_or);

  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  const magmaan::estimate::Bounds fit_bounds =
      twolevel_bounds_from_nullable(bounds, bounds_preset, pt, rep, cs);
  auto est_or = tl::fit_ml_twolevel(pt, rep, cs, x0, fit_bounds, backend,
                                    optim_opts_from(control));
  if (!est_or.has_value()) stop_fit(est_or.error());
  const magmaan::estimate::Estimates est = std::move(*est_or);

  // Observed-information SEs.
  Rcpp::NumericVector se(est.theta.size(), NA_REAL);
  auto ev_or = magmaan::model::ModelEvaluator::build(pt, rep);
  if (ev_or.has_value()) {
    const magmaan::model::ModelEvaluator ev = std::move(*ev_or);
    auto info_or = tl::twolevel_information(ev, cs, est.theta, /*expected=*/false);
    if (info_or.has_value()) {
      const Eigen::MatrixXd vcov = info_or->inverse();
      for (Eigen::Index k = 0; k < est.theta.size(); ++k) {
        const double v = vcov(k, k);
        se[k] = (v > 0.0) ? std::sqrt(v) : NA_REAL;
      }
    }
  }

  // LRT chi-square vs the saturated H1 (chi2 = F_model - F_H1 = 2*fmin - F_H1).
  double chisq = NA_REAL;
  int df = NA_INTEGER;
  auto h1_or = tl::twolevel_h1_moments(cs);
  if (h1_or.has_value()) {
    chisq = 2.0 * est.fmin - h1_or->value;
    // Saturated free parameters, summed over groups: per group Σ_W and Σ_B are
    // each unstructured p_g(p_g+1)/2 and μ_B adds p_g, i.e. p_g(p_g+1) + p_g.
    // df = Σ_g [ p_g(p_g+1) + p_g ] − q.  Single group reduces to p(p+1)+p.
    long psat = 0;
    for (const auto& gst : cs.groups) {
      const long pg = static_cast<long>(gst.p_within);
      psat += pg * (pg + 1L) + pg;
    }
    df = static_cast<int>(psat - static_cast<long>(pt.n_free()));
  }

  Rcpp::NumericVector theta(est.theta.size());
  for (Eigen::Index k = 0; k < est.theta.size(); ++k) theta[k] = est.theta[k];

  // Per-group level-1 sizes (within) and cluster counts (between), for printing.
  Rcpp::IntegerVector nobs_out(ng), nclusters_out(ng);
  long ntotal = 0, ncl_total = 0;
  for (int g = 0; g < ng; ++g) {
    const auto& gst = cs.groups[static_cast<std::size_t>(g)];
    nobs_out[g]      = static_cast<int>(gst.n_within);
    nclusters_out[g] = static_cast<int>(gst.n_clusters);
    ntotal    += gst.n_within;
    ncl_total += gst.n_clusters;
  }

  using magmaan::optim::OptimStatus;
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["converged"]    = common_converged_to_r(est),
      Rcpp::_["verdict"] = common_verdict_to_r(est.diagnostics),
      Rcpp::_["estimator"]    = "ML",
      Rcpp::_["fmin"]         = est.fmin,
      Rcpp::_["iterations"]   = est.iterations,
      Rcpp::_["f_evals"]      = est.f_evals,
      Rcpp::_["g_evals"]      = est.g_evals,
      Rcpp::_["npar"]         = static_cast<int>(pt.n_free()),
      Rcpp::_["ngroups"]      = ng,
      Rcpp::_["ntotal"]       = static_cast<int>(ntotal),
      Rcpp::_["nclusters"]    = static_cast<int>(ncl_total),
      Rcpp::_["group_var"]    = names.group_var,
      Rcpp::_["group_labels"] = Rcpp::wrap(names.group_labels),
      Rcpp::_["theta"]        = theta,
      Rcpp::_["se"]           = se,
      Rcpp::_["ov_names"]     = Rcpp::wrap(rep.ov_names[0]),
      Rcpp::_["partable"]     = partable_df(pt, names, est, &starts),
      Rcpp::_["nobs"]         = nobs_out,
      Rcpp::_["nclusters_by_group"] = nclusters_out,
      Rcpp::_["meanstructure"] = true);
  out["chisq"] = chisq;
  out["df"]    = df;
  out["level"] = 2;
  out["optimizer_status"] = optim_status_to_r(est.optimizer_status);
  out["grad_norm"]        = est.grad_inf_norm;
  out["audit"]            = audit_to_r(est.audit);
  out["diagnostics"]      = diagnostics_to_r(est.diagnostics);
  return out;
}

// fit_ml_fisher() — normal-theory ML via local Fisher scoring. This path uses
// `control = list(max_iter, ftol, gtol)` for the scoring loop; there is no
// optimizer backend because the Fisher step plus Armijo globalization is the
// optimizer.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_fisher_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::List> control = R_NilValue,
                              Rcpp::Nullable<Rcpp::List> bounds  = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ml_fisher");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  auto e_or = magmaan::estimate::fit_ml_fisher(
      ctx.pt, ctx.rep, ctx.samp, x0, bounds_from_nullable(bounds),
      optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ML-Fisher");
}

// fit_ml_fisher_snlls() — local Fisher scoring with a Schur-complement solve
// over the same beta/alpha split used by the SNLLS research paths.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_fisher_snlls_impl(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> bounds  = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ml_fisher_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  auto e_or = magmaan::estimate::fit_ml_fisher_snlls(
      ctx.pt, ctx.rep, ctx.samp, x0, bounds_from_nullable(bounds),
      optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ML-Fisher-SNLLS");
}

// fit_ml_irls() — normal-theory ML via iteratively reweighted GLS (Fisher
// scoring). Same ML objective as fit_ml(), different algorithm: at iterate θ_k
// build the expected Fisher information weight W(θ_k) = ½D'(Σ(θ_k)⁻¹⊗Σ(θ_k)⁻¹)D,
// solve the inner GLS subproblem, accept a damped step via Armijo on F_ML.
// Mean structures adjust the frozen inner covariance target by the current
// mean residual d_k d_k' so the inner score matches the ML score up to scale.
//
// `optimizer` names the *inner* LS solver. NULL defaults to "port-nls" (the
// natural LS-shape Gauss-Newton trust region for the inner subproblem);
// "nlopt-lbfgs" and "ceres" also work but throw away the residual structure.
// `control` accepts both the inner solver's max_iter / ftol / gtol and the
// outer-loop irls_max_outer / irls_ftol / irls_gtol / irls_armijo_c knobs.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_irls_impl(SEXP partable, Rcpp::List sample_stats,
                            Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                            Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                            Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ml_irls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::PortNls
                         : backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_ml_irls(ctx.pt, ctx.rep, ctx.samp, x0,
      bounds_from_nullable(bounds), backend, optim_opts_from(control),
      irls_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ML");
}

// fit_ml_irls_snlls() — same as fit_ml_irls but each outer iterate's inner
// GLS subproblem is solved by Golub–Pereyra variable projection (β = Λ, B
// optimized; α = Θ, Ψ, ν closed-form). Rejects box bounds, nonlinear
// constraints, and the non-separable models gmm::gp_compatible rejects.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_irls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                                  Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                  Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                  Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ml_irls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  const magmaan::estimate::Backend backend =
      optimizer.isNull() ? magmaan::estimate::Backend::PortNls
                         : backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_ml_irls_snlls(ctx.pt, ctx.rep, ctx.samp,
      x0, bounds_from_nullable(bounds), backend, optim_opts_from(control),
      irls_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  // Use snlls_fit_result so the n_nonlinear / n_linear columns surface on
  // the SNLLS speed-survey path the same way fit_snlls_gls's do.
  return snlls_fit_result(ctx, est, &starts, "ML-IRLS-SNLLS",
                          std::string(magmaan::estimate::backend_name(backend)).c_str());
}

// fit_uls() — composes fit_gmm(pt, rep, samp, x0, {}, bounds, backend).
// `optimizer` selects the backend (default "nlopt-lbfgs"); "ceres" / "ceres-bfgs"
// dispatch to the Ceres LS path, "port-nls" to PORT NL2SOL, etc.
//
// [[Rcpp::export]]
Rcpp::List fit_uls_impl(SEXP partable, Rcpp::List sample_stats,
                        Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0,
      {}, bounds_from_nullable(bounds), backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "ULS");
}

// fit_gls() — composes fit_gls(pt, rep, samp, x0, bounds, backend). The
// default start is the layered moment start.
//
// [[Rcpp::export]]
Rcpp::List fit_gls_impl(SEXP partable, Rcpp::List sample_stats,
                        Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "layered", nullptr, nullptr, control);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_gls(ctx.pt, ctx.rep, ctx.samp, x0,
      bounds_from_nullable(bounds), backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "GLS");
}

// fit_wls() — composes fit_gmm(pt, rep, samp, x0, W, bounds, backend).
//
// [[Rcpp::export]]
Rcpp::List fit_wls_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                        Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                        Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::estimate::gmm::Weight wls = wls_from_arg(W, ctx.samp.S.size());
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_gmm(ctx.pt, ctx.rep, ctx.samp, x0,
      std::move(wls), bounds_from_nullable(bounds),
      backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fit_result(ctx, est, &starts, "WLS");
}

// [[Rcpp::export]]
Rcpp::List evaluate_at_impl(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::NumericVector theta, std::string estimator,
    Rcpp::Nullable<Rcpp::RObject> W = R_NilValue,
    Rcpp::Nullable<Rcpp::List>    bounds = R_NilValue,
    Rcpp::Nullable<Rcpp::List>    audit_options = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "evaluate_at");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);

  magmaan::estimate::Estimator est_enum;
  if      (estimator == "ULS") est_enum = magmaan::estimate::Estimator::ULS;
  else if (estimator == "GLS") est_enum = magmaan::estimate::Estimator::GLS;
  else if (estimator == "WLS") est_enum = magmaan::estimate::Estimator::WLS;
  else if (estimator == "ML")  est_enum = magmaan::estimate::Estimator::ML;
  else Rcpp::stop("magmaan: evaluate_at: estimator must be one of "
                  "\"ULS\", \"GLS\", \"WLS\", \"ML\" (got \"%s\")", estimator);

  magmaan::estimate::gmm::Weight wls;
  if (est_enum == magmaan::estimate::Estimator::WLS) {
    if (W.isNull())
      Rcpp::stop("magmaan: evaluate_at: estimator = \"WLS\" requires a "
                 "non-NULL W weight matrix (or list of matrices for "
                 "multi-group)");
    wls = wls_from_arg(Rcpp::RObject(W.get()), ctx.samp.S.size());
  }

  const Eigen::VectorXd theta_vec = Rcpp::as<Eigen::VectorXd>(theta);
  auto e_or = magmaan::estimate::evaluate_at(
      ctx.pt, ctx.rep, ctx.samp, theta_vec, est_enum, wls,
      bounds_from_nullable(bounds), audit_opts_from(audit_options));
  if (!e_or.has_value()) stop_fit(e_or.error());
  Rcpp::List out = fit_result(ctx, *e_or, &starts, estimator.c_str());
  // Optional methods-development artifacts; the core owns every calculation.
  // Recomputing explicitly requested artifacts leaves the stored verdict intact.
  const auto audit_flag=[&](const char* name) {
    if(audit_options.isNull()) return false;
    Rcpp::List settings(audit_options.get());
    return settings.containsElementNamed(name) && Rcpp::as<bool>(settings[name]);
  };
  if (audit_flag("retain_newton_artifacts") || audit_flag("verified_newton")) {
    auto weight = wls;
    if (est_enum == magmaan::estimate::Estimator::GLS) {
      auto ev = magmaan::model::ModelEvaluator::build(ctx.pt, ctx.rep);
      if (!ev) stop_model(ev.error());
      auto nt = magmaan::estimate::gmm::normal_theory_weight(*ev, ctx.samp, theta_vec);
      if (!nt) stop_fit(nt.error());
      weight = std::move(*nt);
    }
    magmaan::estimate::frontier::NewtonAdapterOptions opts;
    opts.active_bound_tol = audit_opts_from(audit_options).active_bound_tol;
    opts.bounds = bounds_from_nullable(bounds);
    if (opts.bounds.empty()) {
      auto preset = magmaan::estimate::variance_bounds(ctx.pt);
      if (!preset) stop_post(preset.error());
      opts.bounds = std::move(*preset);
    }
    auto audit = est_enum == magmaan::estimate::Estimator::ML
        ? magmaan::fit_expected<magmaan::estimate::frontier::NewtonAudit>(
            magmaan::estimate::frontier::audit_newton_derivatives(ctx.pt, ctx.rep,
                magmaan::estimate::frontier::evaluate_newton_ml(ctx.pt, ctx.rep, ctx.samp, theta_vec),
                opts.domain, opts.accuracy, opts.bounds, opts.active_bound_tol))
        : magmaan::estimate::frontier::audit_newton_gmm(
            ctx.pt, ctx.rep, ctx.samp, theta_vec, weight, opts);
    if (!audit) stop_fit(audit.error());
    auto& a = *audit;
    out["newton_audit"] = Rcpp::List::create(
        Rcpp::_["diagnostics"] = newton_accuracy_to_r(a.diagnostics),
        Rcpp::_["gradient"] = Rcpp::wrap(a.geometry.reduced_gradient),
        Rcpp::_["hessian"] = Rcpp::wrap(a.geometry.reduced_hessian),
        Rcpp::_["metric"] = Rcpp::wrap(a.geometry.reduced_metric),
        Rcpp::_["metric_factor"] = Rcpp::wrap(a.geometry.reduced_metric_factor),
        Rcpp::_["metric_score_residual"] = Rcpp::wrap(a.derivatives.metric_score_residual),
        Rcpp::_["curvature_status"] = std::string(magmaan::estimate::to_string(a.system.status)),
        Rcpp::_["curvature_condition"] = a.system.condition,
        Rcpp::_["curvature_coordinate_map"] = Rcpp::wrap(a.system.coordinate_map),
        Rcpp::_["curvature_equilibrated_hessian"] = Rcpp::wrap(a.system.equilibrated_hessian),
        Rcpp::_["curvature_jacobian_condition"] = a.system.jacobian_condition,
        Rcpp::_["curvature_factor_residual"] = a.system.jacobian_factor_residual,
        Rcpp::_["ls_curvature_correction"] = Rcpp::wrap(a.derivatives.ls_curvature_correction),
        Rcpp::_["whitened_jacobian"] = Rcpp::wrap(a.derivatives.whitened_jacobian),
        Rcpp::_["whitened_residual"] = Rcpp::wrap(a.derivatives.whitened_residual),
        Rcpp::_["n_obs"] = a.derivatives.n_obs,
        Rcpp::_["newton_step"] = Rcpp::wrap(a.solution.step),
        Rcpp::_["factor_status"] = std::string(magmaan::estimate::to_string(a.metric_factor_system.status)),
        Rcpp::_["factor_condition"] = a.metric_factor_system.condition,
        Rcpp::_["factor_residual"] = a.metric_factor_system.factor_residual,
        Rcpp::_["factor_rank"] = static_cast<int>(a.metric_factor_system.rank),
        Rcpp::_["factor_scale"] = Rcpp::wrap(a.metric_factor_system.scale),
        Rcpp::_["equilibrated_factor"] = Rcpp::wrap(a.metric_factor_system.equilibrated_factor),
        Rcpp::_["curvature_scale"] = Rcpp::wrap(a.system.scale),
        Rcpp::_["detail"] = a.derivatives.detail);
    Rcpp::List artifacts(out["newton_audit"]);
    artifacts["retained_ls_weights"] = retained_ls_weights_to_r(a.derivatives);
    artifacts["derivative_basis"] = Rcpp::wrap(
        (a.geometry.equality_basis * a.geometry.tangent_basis).eval());
    auto interval_to_r = [](const magmaan::estimate::frontier::NewtonDistanceInterval& x) {
      return Rcpp::List::create(
          Rcpp::_["status"] = std::string(magmaan::estimate::to_string(x.status)),
          Rcpp::_["decision"] = std::string(magmaan::estimate::frontier::to_string(x.decision)),
          Rcpp::_["distance"] = x.distance, Rcpp::_["lower"] = x.lower,
          Rcpp::_["upper"] = x.upper, Rcpp::_["error_bound"] = x.error_bound,
          Rcpp::_["rank_margin"] = x.rank_margin,
          Rcpp::_["factor_error_bound"] = x.factor_error_bound,
          Rcpp::_["orthogonality_error_bound"] = x.orthogonality_error_bound);
    };
    const bool ls = a.derivatives.metric_kind == magmaan::estimate::NewtonMetricKind::Sandwich;
    auto interval = [&](double matrix_error, double vector_error) {
      if (a.box.applied || a.geometry.domain != magmaan::estimate::StationarityDomain::Ambient) {
        magmaan::estimate::frontier::NewtonDistanceInterval unavailable;
        unavailable.status = magmaan::estimate::NewtonAccuracyStatus::Unsupported;
        return unavailable;
      }
      return ls ? magmaan::estimate::frontier::newton_metric_distance_interval(
          a.metric_factor_system, a.derivatives.metric_score_residual, matrix_error, vector_error)
          : magmaan::estimate::frontier::newton_hessian_distance_interval(
              a.system, a.geometry.reduced_gradient, matrix_error, vector_error);
    };
    artifacts["distance_interval_retained_inputs"] = interval_to_r(interval(0, 0));
    artifacts["interval_input_scope"] = ls
        ? "column-scaled retained factor and score residual; construction errors excluded"
        : "retained equilibrated Hessian and scaled score; construction errors excluded";
    Rcpp::List settings(audit_options.get());
    if (settings.containsElementNamed("interval_input_errors")) {
      Rcpp::List errors(settings["interval_input_errors"]);
      if (!errors.containsElementNamed("matrix") || !errors.containsElementNamed("vector"))
        Rcpp::stop("interval_input_errors needs matrix and vector norm bounds");
      artifacts["distance_interval_conditional"] = interval_to_r(interval(
          Rcpp::as<double>(errors["matrix"]), Rcpp::as<double>(errors["vector"])));
    }
    const bool verified=settings.containsElementNamed("verified_newton") && Rcpp::as<bool>(settings["verified_newton"]);
    if (verified || (settings.containsElementNamed("derive_interval_input_errors") &&
        Rcpp::as<bool>(settings["derive_interval_input_errors"]))) {
      const auto errors = magmaan::estimate::frontier::newton_input_error_bounds(
          ctx.pt, ctx.rep, ctx.samp, theta_vec, a, est_enum);
      a.input_errors=errors;
      artifacts["derived_interval_input_errors"] = Rcpp::List::create(
          Rcpp::_["status"] = std::string(magmaan::estimate::to_string(errors.status)),
          Rcpp::_["matrix"] = errors.matrix, Rcpp::_["vector"] = errors.vector,
          Rcpp::_["curvature"] = errors.curvature,
          Rcpp::_["curvature_lower_bound"] = errors.curvature_lower_bound,
          Rcpp::_["detail"] = errors.detail);
      if (errors.status == magmaan::estimate::NewtonAccuracyStatus::Available) {
        artifacts["distance_interval_derived_inputs"] = interval_to_r(
            magmaan::estimate::frontier::newton_input_distance_interval(a, errors));
      }
    }
    if(verified) {
      const double reported=settings.containsElementNamed("reported_objective")
          ? Rcpp::as<double>(settings["reported_objective"]) : e_or->fmin;
      auto report=magmaan::estimate::frontier::audit_convergence(ctx.pt,ctx.rep,a,{}, {},reported);
      if(!report) stop_fit(report.error());
      auto policy=magmaan::estimate::frontier::newton_convergence_policy();
      policy.require_verified_inputs=true; policy.require_objective_consistency=true;
      auto assessment=verified_assessment_to_r(magmaan::estimate::frontier::assess_convergence(*report,policy));
      out["verified_convergence"]=assessment;
      out["converged_compatibility"]=out["converged"];
      out["converged"]=assessment["converged"];
    }
    out["newton_audit"] = artifacts;
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::List fit_uls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  if (bounds.isNotNull()) {
    Rcpp::stop("SNLLS does not support bounds; use an ordinary LS fit with "
               "bounds, or explicitly request an unbounded SNLLS fit");
  }
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_snlls(ctx.pt, ctx.rep, ctx.samp, x0,
      {}, backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "ULS-SNLLS",
                          std::string(magmaan::estimate::backend_name(backend)).c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_gls_snlls_impl(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_gls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  if (bounds.isNotNull()) {
    Rcpp::stop("SNLLS does not support bounds; use an ordinary LS fit with "
               "bounds, or explicitly request an unbounded SNLLS fit");
  }
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_snlls_gls(ctx.pt, ctx.rep, ctx.samp, x0,
      backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "GLS-SNLLS",
                          std::string(magmaan::estimate::backend_name(backend)).c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_wls_snlls_impl(SEXP partable, Rcpp::List sample_stats, SEXP W,
                              Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                              Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_snlls");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  magmaan::estimate::gmm::Weight wls = wls_from_arg(W, ctx.samp.S.size());
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  if (bounds.isNotNull()) {
    Rcpp::stop("SNLLS does not support bounds; use an ordinary LS fit with "
               "bounds, or explicitly request an unbounded SNLLS fit");
  }
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto e_or = magmaan::estimate::fit_snlls(ctx.pt, ctx.rep, ctx.samp, x0,
      std::move(wls), backend, optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return snlls_fit_result(ctx, est, &starts, "WLS-SNLLS",
                          std::string(magmaan::estimate::backend_name(backend)).c_str());
}

// The flat per-Backend shim explosion (fit_uls_ceres_impl, fit_uls_snlls_ceres_impl,
// fit_gls_ceres_impl, fit_gls_snlls_ceres_impl, fit_gls_snlls_ceres_bfgs_impl,
// fit_wls_ceres_impl, fit_wls_snlls_ceres_impl) lived here through Phase 3 and
// was retired in Phase 4. Callers now pass `optimizer = "ceres"` / `"ceres-bfgs"`
// to the unified per-family entries above; the C++ Backend enum is the
// single dispatch surface.

// fit_start_values() — returns a theta-ordered start vector (length npar).
// `sample_stats` is as in fit_fit(); values are used verbatim. The default
// matches complete-data ML: the layered start. "scaled-fabin" is the former
// default (auto-transported FABIN3); explicit simple/fabin3 retain their native
// meaning; transport can be selected independently.
//
// [[Rcpp::export]]
Rcpp::NumericVector fit_start_values(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::Nullable<Rcpp::String> start = R_NilValue,
    Rcpp::Nullable<Rcpp::String> transport = R_NilValue) {
  namespace es = magmaan::estimate;
  auto parsed = partable_from_arg(partable, "fit_start_values");
  auto starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names), sample_stats);
  const auto name = start_name_from_arg(start, "fit_start_values", "layered");
  es::StartPolicy policy{start_method(name), name == "scaled-fabin"
      ? es::StartTransport::AutoStdLv : es::StartTransport::Native};
  if (transport.isNotNull()) {
    const auto mode = Rcpp::as<std::string>(transport.get());
    policy.transport = start_transport_from_arg(mode);
  }
  auto value = es::start_values(ctx.pt, ctx.rep, ctx.samp, policy, starts);
  if (!value) stop_fit(value.error());
  Rcpp::NumericVector out = Rcpp::wrap(value->theta);
  out.attr("start_method") = start_method_name(value->method);
  out.attr("start_transport") = value->branch == es::MlStartBranch::TransportedStdLv
      ? "std-lv-to-marker" : "native";
  out.attr("start_fallback_reason") = es::start_transport_reason(value->fallback_reason);
  out.attr("start_notes") = Rcpp::wrap(value->notes);
  return out;
}

// fit_coordinate_map() — the optimizer coordinates a fit would use from
// `start` (theta order): parameter units (free parameters), and the reduced
// coordinates' units, S-weighted information diagonal, center and scale for
// `scaling` ("none", "sample_units", "information"). Diagnostic only.
//
// [[Rcpp::export]]
Rcpp::List fit_coordinate_map(SEXP partable, Rcpp::List sample_stats,
                              Rcpp::NumericVector start,
                              std::string scaling = "sample_units",
                              bool center_locations = true) {
  namespace es = magmaan::estimate;
  using magmaan::optim::CoordinateScaling;
  auto parsed = partable_from_arg(partable, "fit_coordinate_map");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names), sample_stats);
  if (auto e = es::resolve_fixed_x_from_sample(ctx.pt, ctx.rep, ctx.samp); !e) stop_fit(e.error());
  auto ev = lvm::ModelEvaluator::build(ctx.pt, ctx.rep);
  if (!ev) stop_model(ev.error());
  auto con = es::build_eq_constraints(ctx.pt, /*allow_nonlinear=*/true);
  if (!con) stop_post(con.error());
  const CoordinateScaling kind = scaling == "none" ? CoordinateScaling::None
      : scaling == "sample_units" ? CoordinateScaling::SampleUnits
      : scaling == "information" ? CoordinateScaling::Information
      : (Rcpp::stop("scaling must be \"none\", \"sample_units\" or \"information\""),
         CoordinateScaling::None);
  const Eigen::VectorXd theta = Rcpp::as<Eigen::VectorXd>(start);
  if (theta.size() != ctx.pt.n_free()) Rcpp::stop("start must have one value per free parameter");
  const Eigen::VectorXd alpha = con->contract(theta);
  auto units = es::parameter_units(ctx.pt, ctx.rep, ctx.samp);
  if (!units) stop_fit(units.error());
  auto reduced = es::reduced_units(*units, *con);
  if (!reduced) stop_fit(reduced.error());
  auto map = es::coordinate_map(kind, center_locations, ctx.pt, ctx.rep, *ev, *con, ctx.samp, alpha);
  if (!map) stop_fit(map.error());
  auto info = es::reduced_information_diagonal(*ev, *con, ctx.samp, alpha);
  return Rcpp::List::create(
      Rcpp::_["kind"] = es::coordinate_scaling_name(map->kind),
      Rcpp::_["units"] = Rcpp::wrap(*units),
      Rcpp::_["reduced_units"] = Rcpp::wrap(*reduced),
      Rcpp::_["information"] = info ? Rcpp::wrap(*info) : Rcpp::wrap(Eigen::VectorXd(alpha.size()).setConstant(NA_REAL)),
      Rcpp::_["center"] = Rcpp::wrap(map->center),
      Rcpp::_["scale"] = Rcpp::wrap(map->scale),
      Rcpp::_["alpha"] = Rcpp::wrap(alpha));
}
