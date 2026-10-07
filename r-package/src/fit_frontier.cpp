#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

namespace {

struct FcSemCtx {
  magmaan::spec::LatentStructure pt;
  magmaan::spec::LatentNames names;
  magmaan::spec::Starts starts;
  magmaan::data::SampleStats samp;
  std::vector<std::string> ov_names;
};

enum class PsdGmmKind { Uls, Gls, Wls };

std::vector<std::string>
fcsem_observed_names(const magmaan::spec::LatentStructure& pt,
                     const magmaan::spec::LatentNames& names);
FcSemCtx fcsem_model_from_syntax(const std::string& syntax);
magmaan::data::SampleStats fcsem_sample_stats_from_arg(
    Rcpp::List sample_stats, const std::vector<std::string>& ov_names);
FcSemCtx fcsem_ctx_from_syntax_sample_stats(const std::string& syntax,
                                            Rcpp::List sample_stats);
FcSemCtx fcsem_ctx_from_fit(Rcpp::List fit);
Rcpp::DataFrame fcsem_partable_df(
    const magmaan::spec::LatentStructure& pt,
    const magmaan::spec::LatentNames& names,
    const magmaan::spec::Starts& starts,
    const magmaan::estimate::Estimates* est = nullptr);
Rcpp::List fcsem_fit_result(FcSemCtx& ctx,
                            const magmaan::estimate::Estimates& est,
                            const std::string& syntax);
Rcpp::DataFrame fcsem_standardized_rows_df(
    const std::vector<magmaan::measures::standardize::FcSemStandardizedRow>& rows);
magmaan::estimate::frontier::RBMOptions
rbm_options_from(Rcpp::Nullable<Rcpp::String> optimizer,
                 Rcpp::Nullable<Rcpp::List> control);
std::string rbm_method_key(std::string method);
Rcpp::List rbm_metadata_to_r(
    const magmaan::estimate::frontier::RBMResult& r,
    const std::string& method,
    const char* base_estimator);
SEXP weight_to_r(const magmaan::estimate::gmm::Weight& W);
magmaan::estimate::frontier::SphereOptions sphere_options_from(
    const std::string& metric, double pin_weight, double pole_tol,
    bool polish, std::string start = "canonical",
    const magmaan::spec::Starts* starts = nullptr,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue);
Rcpp::List gauge_report_to_r(const Ctx& ctx,
                             const magmaan::estimate::frontier::SphereFit& fit,
                             const std::string& metric);
Rcpp::List frontier_fit_gmm_psd_common(
    SEXP partable, Rcpp::List sample_stats, SEXP W,
    Rcpp::Nullable<Rcpp::String> optimizer,
    Rcpp::Nullable<Rcpp::List> control,
    double start_eigen_floor, double feasibility_tol,
    PsdGmmKind kind, const char* call, const char* estimator);
magmaan::estimate::frontier::SamMethod
sam_method_from_string(std::string x);
magmaan::estimate::frontier::SamMapping
sam_mapping_from_string(std::string x);
magmaan::estimate::frontier::SamSe
sam_se_from_string(std::string x);
const char* sam_method_to_string(magmaan::estimate::frontier::SamMethod x);
const char* sam_mapping_to_string(magmaan::estimate::frontier::SamMapping x);
const char* sam_se_to_string(magmaan::estimate::frontier::SamSe x);
std::vector<std::string>
names_for_indices(const std::vector<std::int32_t>& idx,
                  const std::vector<std::string>& names);
Rcpp::IntegerVector
one_based_indices(const std::vector<std::int32_t>& idx,
                  const std::vector<std::string>& names);
Rcpp::NumericMatrix
matrix_to_r(const Eigen::MatrixXd& x,
            const std::vector<std::string>& row_names = {},
            const std::vector<std::string>& col_names = {});
Rcpp::NumericVector
vector_to_r(const Eigen::VectorXd& x,
            const std::vector<std::string>& names = {});
Rcpp::List sam_estimates_to_r(const magmaan::estimate::Estimates& est);
Rcpp::List
sample_stats_to_r(const magmaan::data::SampleStats& samp,
                  const std::vector<std::vector<std::string>>& names_by_block);
Rcpp::List
sam_measurement_block_to_r(
    const magmaan::estimate::frontier::SamMeasurementBlock& block,
    const std::vector<std::string>& latent_names_full,
    const std::vector<std::string>& ov_names_full);
Rcpp::List
sam_result_to_r(Ctx& ctx,
                const magmaan::estimate::frontier::SamResult& sam,
                magmaan::estimate::frontier::SamMethod method,
                magmaan::estimate::frontier::SamMapping mapping,
                magmaan::estimate::frontier::SamSe se_method,
                bool lambda_correction,
                int alpha_correction);

std::vector<std::string>
fcsem_observed_names(const magmaan::spec::LatentStructure& pt,
                     const magmaan::spec::LatentNames& names) {
  std::vector<std::string> out;
  out.reserve(pt.ov_order.size());
  for (std::int32_t id : pt.ov_order) {
    if (id < 0 || static_cast<std::size_t>(id) >= names.var_name.size()) {
      Rcpp::stop("magmaan: native FC-SEM observed-variable inventory is invalid");
    }
    out.push_back(names.var_name[static_cast<std::size_t>(id)]);
  }
  return out;
}

FcSemCtx fcsem_model_from_syntax(const std::string& syntax) {
  auto flat = magmaan::parse::Parser::parse(syntax);
  if (!flat.has_value()) {
    const auto& e = flat.error();
    Rcpp::stop("magmaan parse error at %u:%u (bytes %u..%u): %s",
               e.span.line, e.span.col, e.span.begin, e.span.end, e.detail);
  }
  magmaan::spec::BuildOptions opts;
  opts.composite_mode = magmaan::spec::CompositeMode::FcSem;
  magmaan::spec::Starts starts;
  magmaan::spec::LatentNames names;
  auto pt_or = magmaan::spec::build(*flat, opts, &starts, &names);
  if (!pt_or.has_value()) {
    Rcpp::stop("magmaan lavaanify error: %s", pt_or.error().detail);
  }
  if (pt_or->composite_mode != magmaan::spec::CompositeMode::FcSem ||
      pt_or->composite_blocks.empty()) {
    Rcpp::stop("magmaan: native FC-SEM requires at least one `<~` composite");
  }
  if (pt_or->n_groups() != 1) {
    Rcpp::stop("magmaan: native FC-SEM R frontier currently supports one group");
  }

  FcSemCtx ctx;
  ctx.pt = std::move(*pt_or);
  ctx.names = std::move(names);
  ctx.starts = std::move(starts);
  ctx.ov_names = fcsem_observed_names(ctx.pt, ctx.names);
  return ctx;
}

magmaan::data::SampleStats fcsem_sample_stats_from_arg(
    Rcpp::List sample_stats, const std::vector<std::string>& ov_names) {
  if (!sample_stats.containsElementNamed("S") ||
      !sample_stats.containsElementNamed("nobs")) {
    Rcpp::stop("magmaan: native FC-SEM sample_stats must contain $S and $nobs");
  }
  Rcpp::List Sl = TYPEOF(sample_stats["S"]) == VECSXP
      ? Rcpp::List(sample_stats["S"])
      : Rcpp::List::create(Rcpp::NumericMatrix(sample_stats["S"]));
  Rcpp::IntegerVector nobs = Rcpp::as<Rcpp::IntegerVector>(sample_stats["nobs"]);
  if (Sl.size() != 1 || nobs.size() != 1) {
    Rcpp::stop("magmaan: native FC-SEM R frontier currently supports one group");
  }
  const int nb = nobs[0];
  if (nb == NA_INTEGER || nb <= 0) {
    Rcpp::stop("magmaan: native FC-SEM nobs must be a positive integer");
  }

  Rcpp::NumericMatrix S0(Sl[0]);
  const std::vector<int> perm = perm_for_cols(S0, ov_names, "S");
  magmaan::data::SampleStats out;
  out.S.push_back(reorder_cov(S0, perm));
  validate_finite_matrix(out.S.back(), "sample covariance", 0);
  out.n_obs.push_back(static_cast<std::int64_t>(nb));
  return out;
}

FcSemCtx fcsem_ctx_from_syntax_sample_stats(const std::string& syntax,
                                            Rcpp::List sample_stats) {
  FcSemCtx ctx = fcsem_model_from_syntax(syntax);
  ctx.samp = fcsem_sample_stats_from_arg(sample_stats, ctx.ov_names);
  return ctx;
}

FcSemCtx fcsem_ctx_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("fcsem") || !Rcpp::as<bool>(fit["fcsem"]) ||
      !fit.containsElementNamed("syntax")) {
    Rcpp::stop("magmaan: expected a native FC-SEM fit object");
  }
  Rcpp::List ss = Rcpp::List::create(
      Rcpp::_["S"] = fit["S"],
      Rcpp::_["nobs"] = fit["nobs"]);
  return fcsem_ctx_from_syntax_sample_stats(Rcpp::as<std::string>(fit["syntax"]),
                                            ss);
}

Rcpp::DataFrame fcsem_partable_df(
    const magmaan::spec::LatentStructure& pt,
    const magmaan::spec::LatentNames& names,
    const magmaan::spec::Starts& starts,
    const magmaan::estimate::Estimates* est ) {
  const magmaan::compat::lavaan::LavaanParTable native =
      magmaan::compat::lavaan::to_lavaan_partable(pt, names, starts);
  Rcpp::DataFrame out = partable_df_from_lavaan(native, est);
  Rf_setAttrib(out, Rf_install("magmaan.fcsem"), Rf_ScalarLogical(1));
  return out;
}

Rcpp::List fcsem_fit_result(FcSemCtx& ctx,
                            const magmaan::estimate::Estimates& est,
                            const std::string& syntax) {
  Rcpp::NumericMatrix S0 = Rcpp::wrap(ctx.samp.S[0]);
  Rcpp::CharacterVector nm = Rcpp::wrap(ctx.ov_names);
  S0.attr("dimnames") = Rcpp::List::create(nm, nm);
  Rcpp::List S_out = Rcpp::List::create(S0);
  Rcpp::IntegerVector nobs_out =
      Rcpp::IntegerVector::create(static_cast<int>(ctx.samp.n_obs[0]));

  using magmaan::optim::OptimStatus;
  const char* opt_status = optim_status_to_r(est.optimizer_status);

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["converged"]     = common_converged_to_r(est),
      Rcpp::_["verdict"] = common_verdict_to_r(est.diagnostics),
      Rcpp::_["estimator"]     = "FCSEM-ML",
      Rcpp::_["fmin"]          = est.fmin,
      Rcpp::_["iterations"]    = est.iterations,
      Rcpp::_["f_evals"]       = est.f_evals,
      Rcpp::_["g_evals"]       = est.g_evals,
      Rcpp::_["npar"]          = static_cast<int>(ctx.pt.n_free()),
      Rcpp::_["ngroups"]       = 1,
      Rcpp::_["ntotal"]        = static_cast<int>(ctx.samp.n_obs[0]),
      Rcpp::_["group_var"]     = "",
      Rcpp::_["group_labels"]  = Rcpp::CharacterVector::create(),
      Rcpp::_["theta"]         = Rcpp::wrap(est.theta),
      Rcpp::_["ov_names"]      = Rcpp::wrap(ctx.ov_names),
      Rcpp::_["partable"]      = fcsem_partable_df(ctx.pt, ctx.names,
                                                   ctx.starts, &est),
      Rcpp::_["S"]             = S_out,
      Rcpp::_["nobs"]          = nobs_out,
      Rcpp::_["sample_mean"]   = R_NilValue,
      Rcpp::_["meanstructure"] = false,
      Rcpp::_["syntax"]        = syntax,
      Rcpp::_["fcsem"]         = true);
  out["optimizer_status"] = opt_status;
  out["grad_norm"] = est.grad_inf_norm;
  out["audit"] = audit_to_r(est.audit);
  // FCSEM fits do not run the L2 finalization audit (different evaluator
  // type); diagnostics is default-constructed and surfaces as the "audit
  // did not run" schema slot.
  out["diagnostics"] = diagnostics_to_r(est.diagnostics);
  return out;
}

Rcpp::DataFrame fcsem_standardized_rows_df(
    const std::vector<magmaan::measures::standardize::FcSemStandardizedRow>& rows) {
  const R_xlen_t n = static_cast<R_xlen_t>(rows.size());
  Rcpp::IntegerVector row(n), group(n), freev(n);
  Rcpp::CharacterVector lhs(n), op(n), rhs(n);
  Rcpp::NumericVector est(n), se(n), std_lv(n), std_lv_se(n), std_all(n),
      std_all_se(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& r = rows[static_cast<std::size_t>(i)];
    row[i] = static_cast<int>(r.row) + 1;
    lhs[i] = r.lhs;
    op[i] = std::string(magmaan::parse::to_string(r.op));
    rhs[i] = r.rhs;
    group[i] = r.group;
    freev[i] = r.free;
    est[i] = r.est;
    se[i] = r.se;
    std_lv[i] = r.std_lv;
    std_lv_se[i] = r.std_lv_se;
    std_all[i] = r.std_all;
    std_all_se[i] = r.std_all_se;
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["row"] = row,
      Rcpp::_["lhs"] = lhs,
      Rcpp::_["op"] = op,
      Rcpp::_["rhs"] = rhs,
      Rcpp::_["group"] = group,
      Rcpp::_["free"] = freev,
      Rcpp::_["est"] = est,
      Rcpp::_["se"] = se,
      Rcpp::_["std.lv"] = std_lv,
      Rcpp::_["std.lv.se"] = std_lv_se,
      Rcpp::_["std.all"] = std_all,
      Rcpp::_["std.all.se"] = std_all_se,
      Rcpp::_["stringsAsFactors"] = false);
}

magmaan::estimate::frontier::RBMOptions
rbm_options_from(Rcpp::Nullable<Rcpp::String> optimizer,
                 Rcpp::Nullable<Rcpp::List> control) {
  magmaan::estimate::frontier::RBMOptions opts;
  if (optimizer.isNotNull()) {
    opts.backend = backend_from_optimizer_arg(optimizer);
  }
  opts.optim = optim_opts_from(control);
  if (control.isNotNull()) {
    Rcpp::List l(control.get());
    if (l.containsElementNamed("fd_rel_step"))
      opts.fd_rel_step = Rcpp::as<double>(l["fd_rel_step"]);
    if (l.containsElementNamed("fd_abs_step"))
      opts.fd_abs_step = Rcpp::as<double>(l["fd_abs_step"]);
    if (l.containsElementNamed("check_admissibility"))
      opts.check_admissibility = Rcpp::as<bool>(l["check_admissibility"]);
    if (l.containsElementNamed("admissibility_tol"))
      opts.admissibility_tol = Rcpp::as<double>(l["admissibility_tol"]);
  }
  return opts;
}

std::string rbm_method_key(std::string method) {
  for (char& ch : method) {
    if (ch == '-') ch = '_';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (method == "explicit" || method == "erb" || method == "erbm") {
    return "explicit";
  }
  if (method == "implicit" || method == "irb" || method == "irbm") {
    return "implicit";
  }
  Rcpp::stop("magmaan: RBM method must be 'explicit' or 'implicit'");
}

Rcpp::List rbm_metadata_to_r(
    const magmaan::estimate::frontier::RBMResult& r,
    const std::string& method,
    const char* base_estimator) {
  return Rcpp::List::create(
      Rcpp::_["method"] = method,
      Rcpp::_["base_estimator"] = base_estimator,
      Rcpp::_["correction"] = Rcpp::wrap(r.correction),
      Rcpp::_["adjustment"] = Rcpp::wrap(r.adjustment),
      Rcpp::_["information"] = Rcpp::wrap(r.information),
      Rcpp::_["meat"] = Rcpp::wrap(r.meat),
      Rcpp::_["information_reduced"] = Rcpp::wrap(r.information_reduced),
      Rcpp::_["meat_reduced"] = Rcpp::wrap(r.meat_reduced),
      Rcpp::_["trace"] = r.trace_term,
      Rcpp::_["penalty"] = r.penalty,
      Rcpp::_["penalty_per_observation"] = r.penalty_per_observation,
      Rcpp::_["penalized_fmin"] = r.penalized_fmin,
      Rcpp::_["admissible"] = r.admissible,
      Rcpp::_["bounds_satisfied"] = r.bounds_satisfied,
      Rcpp::_["sigma_pd"] = r.sigma_pd,
      Rcpp::_["warnings"] = Rcpp::wrap(r.warnings));
}

SEXP weight_to_r(const magmaan::estimate::gmm::Weight& W) {
  return weight_to_r_dense(W);
}

// `start`: "canonical" (the identification-invariant sphere start) or
// "user". Explicit start values in the model, or an explicit start policy in
// `control`, are written in the user's identification, so they select the user
// start.
magmaan::estimate::frontier::SphereOptions sphere_options_from(
    const std::string& metric, double pin_weight, double pole_tol,
    bool polish, std::string start ,
    const magmaan::spec::Starts* starts ,
    Rcpp::Nullable<Rcpp::List> control ) {
  magmaan::estimate::frontier::SphereOptions out;
  if (start != "canonical" && start != "user") {
    Rcpp::stop("magmaan: `start` must be \"canonical\" or \"user\"");
  }
  if (starts) {
    for (double h : starts->hint)
      if (std::isfinite(h)) start = "user";
  }
  if (control.isNotNull()) {
    Rcpp::List ctl(control.get());
    if (ctl.containsElementNamed("start") && !Rf_isNull(ctl["start"])) start = "user";
    if(ctl.containsElementNamed("verified_newton")) out.verified_newton=Rcpp::as<bool>(ctl["verified_newton"]);
  }
  out.start = start == "canonical"
                  ? magmaan::estimate::frontier::SphereStart::Canonical
                  : magmaan::estimate::frontier::SphereStart::User;
  if (metric == "unit_free") {
    out.metric = magmaan::estimate::frontier::SphereMetric::UnitFree;
  } else if (metric == "raw") {
    out.metric = magmaan::estimate::frontier::SphereMetric::Raw;
  } else {
    Rcpp::stop("magmaan: `metric` must be \"unit_free\" or \"raw\"");
  }
  out.pin_weight = pin_weight;
  out.pole_tol = pole_tol;
  out.polish = polish;
  return out;
}

// The sphere report: gauge units, latents left in the user chart, and the
// sphere-chart solution as a partable of the internal (gauge-free) model.
Rcpp::List gauge_report_to_r(const Ctx& ctx,
                             const magmaan::estimate::frontier::SphereFit& fit,
                             const std::string& metric) {
  const auto& report = fit.report;
  const auto& plan = report.plan;
  auto name_of = [&ctx](std::int32_t v) -> std::string {
    return (v >= 0 && static_cast<std::size_t>(v) < ctx.names.var_name.size())
               ? ctx.names.var_name[static_cast<std::size_t>(v)]
               : std::string();
  };
  const R_xlen_t nu = static_cast<R_xlen_t>(plan.units.size());
  Rcpp::CharacterVector latent(nu), kind(nu), blocks(nu);
  Rcpp::IntegerVector dim(nu);
  Rcpp::NumericVector level(nu);
  Rcpp::LogicalVector singular(nu);
  for (R_xlen_t u = 0; u < nu; ++u) {
    const auto& unit = plan.units[static_cast<std::size_t>(u)];
    latent[u] = name_of(unit.latent);
    kind[u] = unit.kind == magmaan::estimate::frontier::GaugeKind::Affine
                  ? "affine" : "linear";
    std::string b;
    for (std::size_t k = 0; k < unit.blocks.size(); ++k) {
      if (k) b += ",";
      b += std::to_string(unit.blocks[k]);
    }
    blocks[u] = b;
    dim[u] = static_cast<int>(unit.basis.cols());
    level[u] = u < report.scales.direction_level.size()
                   ? report.scales.direction_level(u) : NA_REAL;
    singular[u] = std::find(report.scales.singular_units.begin(),
                            report.scales.singular_units.end(),
                            static_cast<std::int32_t>(u)) !=
                  report.scales.singular_units.end();
  }
  Rcpp::DataFrame units = Rcpp::DataFrame::create(
      Rcpp::_["latent"] = latent, Rcpp::_["blocks"] = blocks,
      Rcpp::_["kind"] = kind, Rcpp::_["span_dim"] = dim,
      Rcpp::_["direction_level"] = level, Rcpp::_["singular"] = singular,
      Rcpp::_["stringsAsFactors"] = false);

  const R_xlen_t np = static_cast<R_xlen_t>(plan.passthrough.size());
  Rcpp::CharacterVector pl(np), reason(np);
  Rcpp::IntegerVector pb(np);
  for (R_xlen_t k = 0; k < np; ++k) {
    const auto& p = plan.passthrough[static_cast<std::size_t>(k)];
    pl[k] = name_of(p.latent);
    pb[k] = p.block;
    reason[k] = p.reason;
  }
  Rcpp::DataFrame passthrough = Rcpp::DataFrame::create(
      Rcpp::_["latent"] = pl, Rcpp::_["block"] = pb, Rcpp::_["reason"] = reason,
      Rcpp::_["stringsAsFactors"] = false);

  magmaan::estimate::Estimates internal;
  internal.theta = report.internal_theta;
  Rcpp::DataFrame sphere_partable =
      partable_df(report.internal_pt, ctx.names, internal, nullptr);

  const auto& a = report.native_audit;
  const auto& v = report.native_verdict;
  auto check = [](const magmaan::estimate::frontier::ConvergenceCheck& c) {
    return Rcpp::List::create(Rcpp::_["status"] = fit_check_to_r(c.status),
        Rcpp::_["required"] = c.required, Rcpp::_["reason"] = c.reason);
  };
  Rcpp::LogicalVector native_converged(1);
  native_converged[0] = v.status == magmaan::estimate::FitCheck::Unchecked
      ? NA_LOGICAL : v.status == magmaan::estimate::FitCheck::Passed;
  const auto& computations = a.computations;
  Rcpp::List native_audit = Rcpp::List::create(
      Rcpp::_["status"] = fit_check_to_r(v.status),
      Rcpp::_["converged"] = native_converged,
      Rcpp::_["domain"] = v.domain == magmaan::estimate::StationarityDomain::Psd ? "psd" : "ambient",
      Rcpp::_["first_order_metric"] = "sphere_product_euclidean",
      Rcpp::_["objective"] = check(v.objective),
      Rcpp::_["objective_consistency"] = check(v.objective_consistency),
      Rcpp::_["feasibility"] = check(v.feasibility),
      Rcpp::_["first_order"] = check(v.first_order),
      Rcpp::_["newton"] = check(v.newton),
      Rcpp::_["fmin"] = a.evidence.objective.recomputed,
      Rcpp::_["reported_fmin"] = a.evidence.objective.reported,
      Rcpp::_["first_order_residual"] = a.evidence.geometric_stationarity.ambient_residual_l2,
      Rcpp::_["newton_accuracy"] = newton_accuracy_to_r(a.evidence.newton_accuracy),
      Rcpp::_["point"] = Rcpp::wrap(computations.derivatives.theta),
      Rcpp::_["gradient"] = Rcpp::wrap(computations.derivatives.gradient),
      Rcpp::_["hessian"] = Rcpp::wrap(computations.derivatives.hessian),
      Rcpp::_["tangent_basis"] = Rcpp::wrap(computations.geometry.tangent_basis),
      Rcpp::_["reduced_gradient"] = Rcpp::wrap(computations.geometry.reduced_gradient),
      Rcpp::_["reduced_hessian"] = Rcpp::wrap(computations.geometry.reduced_hessian),
      Rcpp::_["detail"] = a.detail);
  auto legacy_policy=magmaan::estimate::frontier::newton_convergence_policy();
  legacy_policy.require_objective_consistency=true;
  native_audit["compatibility_assessment"]=verified_assessment_to_r(
      magmaan::estimate::frontier::assess_convergence(a,legacy_policy));
  native_audit["n_obs"] = computations.derivatives.n_obs;
  native_audit["retained_ls_weights"] = retained_ls_weights_to_r(computations.derivatives);
  auto system = [](const magmaan::estimate::frontier::NewtonSystem& s) {
    return Rcpp::List::create(
        Rcpp::_["status"] = std::string(magmaan::estimate::to_string(s.status)),
        Rcpp::_["condition"] = s.condition,
        Rcpp::_["coordinate_map"] = Rcpp::wrap(s.coordinate_map),
        Rcpp::_["equilibrated_hessian"] = Rcpp::wrap(s.equilibrated_hessian),
        Rcpp::_["jacobian_condition"] = s.jacobian_condition,
        Rcpp::_["jacobian_factor_residual"] = s.jacobian_factor_residual);
  };
  native_audit["curvature_system"] = system(computations.system);
  native_audit["ls_curvature_correction"] = Rcpp::wrap(computations.derivatives.ls_curvature_correction);
  native_audit["whitened_jacobian"] = Rcpp::wrap(computations.derivatives.whitened_jacobian);
  native_audit["newton_step"] = Rcpp::wrap(computations.solution.step);
  native_audit["accuracy_metric_system"] = system(computations.metric_system);
  const auto& factor_system = computations.metric_factor_system;
  native_audit["accuracy_metric_factor_system"] = Rcpp::List::create(
      Rcpp::_["status"] = std::string(magmaan::estimate::to_string(factor_system.status)),
      Rcpp::_["condition"] = factor_system.condition,
      Rcpp::_["factor_residual"] = factor_system.factor_residual,
      Rcpp::_["rank"] = static_cast<int>(factor_system.rank));
  native_audit["reduced_metric_factor"] = Rcpp::wrap(computations.geometry.reduced_metric_factor);
  native_audit["metric_score_residual"] = Rcpp::wrap(computations.derivatives.metric_score_residual);
  native_audit["reduced_metric"] = Rcpp::wrap(computations.geometry.reduced_metric);
  native_audit["equilibrated_factor"]=Rcpp::wrap(computations.metric_factor_system.equilibrated_factor);
  native_audit["factor_scale"]=Rcpp::wrap(computations.metric_factor_system.scale);
  native_audit["curvature_scale"]=Rcpp::wrap(computations.system.scale);
  if(computations.input_errors) {
    native_audit["derived_interval_input_errors"]=input_errors_to_r(*computations.input_errors);
    native_audit["distance_interval_derived_inputs"]=distance_interval_to_r(
        magmaan::estimate::frontier::newton_input_distance_interval(computations,*computations.input_errors));
  }
  if(a.input_map) {
    const auto& map=*a.input_map;
    Rcpp::List mapped_units(map.spheres.size());
    for(std::size_t k=0;k<map.spheres.size();++k) {
      const auto& unit=map.spheres[k];
      Rcpp::IntegerMatrix parameters(unit.units.size(),unit.parameters.size());
      for(std::size_t member=0;member<unit.parameters.size();++member)
        for(Eigen::Index j=0;j<unit.units.size();++j) parameters(j,member)=unit.parameters[member][j];
      mapped_units[k]=Rcpp::List::create(Rcpp::_["basis"]=Rcpp::wrap(unit.basis),
          Rcpp::_["units"]=Rcpp::wrap(unit.units),Rcpp::_["parameters"]=parameters,Rcpp::_["offset"]=unit.offset);
    }
    native_audit["input_map"]=Rcpp::List::create(Rcpp::_["offset"]=Rcpp::wrap(map.offset),
        Rcpp::_["rest_basis"]=Rcpp::wrap(map.rest_basis),Rcpp::_["rounded_point"]=Rcpp::wrap(map.rounded_point),
        Rcpp::_["spheres"]=mapped_units);
  }


  return Rcpp::List::create(
      Rcpp::_["chart"] = "sphere",
      Rcpp::_["metric"] = metric,
      Rcpp::_["user_chart"] = fit.user_chart,
      Rcpp::_["units"] = units,
      Rcpp::_["passthrough"] = passthrough,
      Rcpp::_["sphere_partable"] = sphere_partable,
      Rcpp::_["fmin_sphere"] = report.fmin_internal,
      Rcpp::_["pin_residual"] = report.pin_residual,
      Rcpp::_["residual"] = Rcpp::List::create(
          Rcpp::_["fixed_rows"] = report.residual.fixed_rows,
          Rcpp::_["linear_constraints"] = report.residual.linear_constraints),
      Rcpp::_["optimizer_status"] = optim_status_to_r(report.optimizer_status),
      Rcpp::_["driven_stationary"] = report.driven_audit.stationary,
      Rcpp::_["native_audit"] = native_audit,
      Rcpp::_["iterations"] = report.iterations,
      Rcpp::_["start"] = report.start_used,
      Rcpp::_["driven_scaled"] = report.driven_scaled,
      Rcpp::_["polish"] = Rcpp::List::create(
          Rcpp::_["polished"] = report.polished,
          Rcpp::_["iterations"] = report.polish_iterations,
          Rcpp::_["shift"] = report.polish_shift,
          Rcpp::_["error"] = report.polish_error));
}

Rcpp::List frontier_fit_gmm_psd_common(
    SEXP partable, Rcpp::List sample_stats, SEXP W,
    Rcpp::Nullable<Rcpp::String> optimizer,
    Rcpp::Nullable<Rcpp::List> control,
    double start_eigen_floor, double feasibility_tol,
    PsdGmmKind kind, const char* call, const char* estimator) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, call);
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(
      std::move(parsed.structure), std::move(parsed.names), sample_stats);
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  const magmaan::estimate::Backend backend =
      optimizer.isNull()
          ? magmaan::estimate::Backend::NloptSlsqp
          : backend_from_optimizer_arg(optimizer);
  magmaan::estimate::frontier::PsdFitOptions psd_opts;
  psd_opts.start_eigen_floor = start_eigen_floor;
  psd_opts.feasibility_tol = feasibility_tol;

  magmaan::fit_expected<magmaan::estimate::Estimates> fit_or;
  if (kind == PsdGmmKind::Gls) {
    fit_or = magmaan::estimate::frontier::fit_gls_psd(
        ctx.pt, ctx.rep, ctx.samp, x0, backend, optim_opts_from(control),
        psd_opts);
  } else {
    magmaan::estimate::gmm::Weight weight;
    if (kind == PsdGmmKind::Wls) {
      weight = wls_from_arg(W, ctx.samp.S.size());
    }
    fit_or = magmaan::estimate::frontier::fit_gmm_psd(
        ctx.pt, ctx.rep, ctx.samp, x0, std::move(weight), backend,
        optim_opts_from(control), psd_opts);
  }
  if (!fit_or.has_value()) stop_fit(fit_or.error());
  const magmaan::estimate::Estimates est = std::move(*fit_or);
  return fit_result(ctx, est, &starts, estimator);
}

magmaan::estimate::frontier::SamMethod
sam_method_from_string(std::string x) {
  for (char& ch : x) {
    if (ch == '_') ch = '-';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (x == "local") return magmaan::estimate::frontier::SamMethod::Local;
  if (x == "global") return magmaan::estimate::frontier::SamMethod::Global;
  Rcpp::stop("magmaan: frontier_sam(): unsupported method '%s' "
             "(accepted: local, global)", x.c_str());
  return magmaan::estimate::frontier::SamMethod::Local;
}

magmaan::estimate::frontier::SamMapping
sam_mapping_from_string(std::string x) {
  for (char& ch : x) {
    if (ch == '_') ch = '-';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (x == "ml") return magmaan::estimate::frontier::SamMapping::ML;
  if (x == "gls") return magmaan::estimate::frontier::SamMapping::GLS;
  if (x == "uls") return magmaan::estimate::frontier::SamMapping::ULS;
  Rcpp::stop("magmaan: frontier_sam(): unsupported mapping '%s' "
             "(accepted: ml, gls, uls)", x.c_str());
  return magmaan::estimate::frontier::SamMapping::ML;
}

magmaan::estimate::frontier::SamSe
sam_se_from_string(std::string x) {
  for (char& ch : x) {
    if (ch == '_') ch = '-';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (x == "none") return magmaan::estimate::frontier::SamSe::None;
  if (x == "standard") return magmaan::estimate::frontier::SamSe::Standard;
  if (x == "twostep") return magmaan::estimate::frontier::SamSe::Twostep;
  if (x == "twostep.robust" || x == "twostep-robust" || x == "robust")
    return magmaan::estimate::frontier::SamSe::TwostepRobust;
  Rcpp::stop("magmaan: frontier_sam(): unsupported se '%s' "
             "(accepted: none, standard, twostep, twostep.robust)", x.c_str());
  return magmaan::estimate::frontier::SamSe::Twostep;
}

const char* sam_method_to_string(magmaan::estimate::frontier::SamMethod x) {
  using magmaan::estimate::frontier::SamMethod;
  return x == SamMethod::Global ? "global" : "local";
}

const char* sam_mapping_to_string(magmaan::estimate::frontier::SamMapping x) {
  using magmaan::estimate::frontier::SamMapping;
  if (x == SamMapping::GLS) return "gls";
  if (x == SamMapping::ULS) return "uls";
  return "ml";
}

const char* sam_se_to_string(magmaan::estimate::frontier::SamSe x) {
  using magmaan::estimate::frontier::SamSe;
  if (x == SamSe::None) return "none";
  if (x == SamSe::Standard) return "standard";
  if (x == SamSe::TwostepRobust) return "twostep.robust";
  return "twostep";
}

std::vector<std::string>
names_for_indices(const std::vector<std::int32_t>& idx,
                  const std::vector<std::string>& names) {
  std::vector<std::string> out;
  out.reserve(idx.size());
  for (std::int32_t id : idx) {
    if (id >= 0 && static_cast<std::size_t>(id) < names.size()) {
      out.push_back(names[static_cast<std::size_t>(id)]);
    } else {
      out.push_back(std::to_string(id + 1));
    }
  }
  return out;
}

Rcpp::IntegerVector
one_based_indices(const std::vector<std::int32_t>& idx,
                  const std::vector<std::string>& names) {
  Rcpp::IntegerVector out(static_cast<R_xlen_t>(idx.size()));
  for (R_xlen_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<int>(idx[static_cast<std::size_t>(i)] + 1);
  const std::vector<std::string> nm = names_for_indices(idx, names);
  if (!nm.empty()) out.attr("names") = Rcpp::wrap(nm);
  return out;
}

Rcpp::NumericMatrix
matrix_to_r(const Eigen::MatrixXd& x,
            const std::vector<std::string>& row_names ,
            const std::vector<std::string>& col_names ) {
  Rcpp::NumericMatrix out = Rcpp::wrap(x);
  if (static_cast<Eigen::Index>(row_names.size()) == x.rows() &&
      static_cast<Eigen::Index>(col_names.size()) == x.cols()) {
    out.attr("dimnames") = Rcpp::List::create(Rcpp::wrap(row_names),
                                              Rcpp::wrap(col_names));
  }
  return out;
}

Rcpp::NumericVector
vector_to_r(const Eigen::VectorXd& x,
            const std::vector<std::string>& names ) {
  Rcpp::NumericVector out = Rcpp::wrap(x);
  if (static_cast<Eigen::Index>(names.size()) == x.size()) {
    out.attr("names") = Rcpp::wrap(names);
  }
  return out;
}

Rcpp::List sam_estimates_to_r(const magmaan::estimate::Estimates& est) {
  return Rcpp::List::create(
      Rcpp::_["converged"] = common_converged_to_r(est),
      Rcpp::_["verdict"] = common_verdict_to_r(est.diagnostics),
      Rcpp::_["theta"] = Rcpp::wrap(est.theta),
      Rcpp::_["fmin"] = est.fmin,
      Rcpp::_["iterations"] = est.iterations,
      Rcpp::_["f_evals"] = est.f_evals,
      Rcpp::_["g_evals"] = est.g_evals,
      Rcpp::_["optimizer_status"] = optim_status_to_r(est.optimizer_status),
      Rcpp::_["grad_norm"] = est.grad_inf_norm);
}

Rcpp::List
sample_stats_to_r(const magmaan::data::SampleStats& samp,
                  const std::vector<std::vector<std::string>>& names_by_block) {
  const R_xlen_t nb = static_cast<R_xlen_t>(samp.S.size());
  Rcpp::List S_out(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::vector<std::string> nm =
        static_cast<std::size_t>(b) < names_by_block.size()
            ? names_by_block[static_cast<std::size_t>(b)]
            : std::vector<std::string>{};
    S_out[b] = matrix_to_r(samp.S[static_cast<std::size_t>(b)], nm, nm);
  }
  Rcpp::RObject mean_out = R_NilValue;
  if (!samp.mean.empty()) {
    Rcpp::List M_out(nb);
    for (R_xlen_t b = 0; b < nb; ++b) {
      const std::vector<std::string> nm =
          static_cast<std::size_t>(b) < names_by_block.size()
              ? names_by_block[static_cast<std::size_t>(b)]
              : std::vector<std::string>{};
      M_out[b] = vector_to_r(samp.mean[static_cast<std::size_t>(b)], nm);
    }
    mean_out = M_out;
  }
  Rcpp::IntegerVector nobs(static_cast<R_xlen_t>(samp.n_obs.size()));
  for (R_xlen_t b = 0; b < nobs.size(); ++b)
    nobs[b] = static_cast<int>(samp.n_obs[static_cast<std::size_t>(b)]);
  return Rcpp::List::create(Rcpp::_["S"] = S_out,
                            Rcpp::_["mean"] = mean_out,
                            Rcpp::_["nobs"] = nobs);
}

Rcpp::List
sam_measurement_block_to_r(
    const magmaan::estimate::frontier::SamMeasurementBlock& block,
    const std::vector<std::string>& latent_names_full,
    const std::vector<std::string>& ov_names_full) {
  const std::vector<std::string> lat_names =
      names_for_indices(block.latents, latent_names_full);
  const std::vector<std::string> ind_names =
      names_for_indices(block.indicators, ov_names_full);
  return Rcpp::List::create(
      Rcpp::_["latents"] = one_based_indices(block.latents, latent_names_full),
      Rcpp::_["indicators"] =
          one_based_indices(block.indicators, ov_names_full),
      Rcpp::_["latent_names"] = Rcpp::wrap(lat_names),
      Rcpp::_["indicator_names"] = Rcpp::wrap(ind_names),
      Rcpp::_["estimates"] = sam_estimates_to_r(block.estimates),
      Rcpp::_["Lambda"] = matrix_to_r(block.Lambda, ind_names, lat_names),
      Rcpp::_["Theta"] = matrix_to_r(block.Theta, ind_names, ind_names),
      Rcpp::_["M"] = matrix_to_r(block.M, lat_names, ind_names),
      Rcpp::_["Nu"] = block.Nu.size() > 0
          ? static_cast<SEXP>(vector_to_r(block.Nu, ind_names))
          : R_NilValue,
      Rcpp::_["vcov"] = block.vcov.size() > 0
          ? static_cast<SEXP>(Rcpp::wrap(block.vcov))
          : R_NilValue);
}

Rcpp::List
sam_result_to_r(Ctx& ctx,
                const magmaan::estimate::frontier::SamResult& sam,
                magmaan::estimate::frontier::SamMethod method,
                magmaan::estimate::frontier::SamMapping mapping,
                magmaan::estimate::frontier::SamSe se_method,
                bool lambda_correction,
                int alpha_correction) {
  magmaan::estimate::Estimates joint = sam.structural;
  joint.theta = sam.theta;
  Rcpp::List out = fit_result(ctx, joint, nullptr, "SAM");
  out["vcov"] = sam.vcov.size() > 0
      ? static_cast<SEXP>(Rcpp::wrap(sam.vcov))
      : R_NilValue;
  out["se"] = sam.se.size() > 0
      ? static_cast<SEXP>(Rcpp::wrap(sam.se))
      : R_NilValue;

  const std::vector<std::string> latent_names =
      (!ctx.rep.lv_names.empty() &&
       ctx.rep.lv_names[0].size() == static_cast<std::size_t>(sam.VETA.rows()))
          ? ctx.rep.lv_names[0]
          : std::vector<std::string>{};
  const std::vector<std::string> ov_names =
      !ctx.rep.ov_names.empty() ? ctx.rep.ov_names[0]
                                : std::vector<std::string>{};
  std::vector<std::vector<std::string>> latent_samp_names =
      sam.structural_rep.ov_names;

  Rcpp::List measurement(static_cast<R_xlen_t>(sam.measurement.size()));
  for (R_xlen_t i = 0; i < measurement.size(); ++i) {
    measurement[i] = sam_measurement_block_to_r(
        sam.measurement[static_cast<std::size_t>(i)], latent_names, ov_names);
  }

  out["sam"] = Rcpp::List::create(
      Rcpp::_["method"] = sam_method_to_string(method),
      Rcpp::_["mapping"] = sam_mapping_to_string(mapping),
      Rcpp::_["se_method"] = sam_se_to_string(se_method),
      Rcpp::_["lambda_correction"] = lambda_correction,
      Rcpp::_["alpha_correction"] = alpha_correction,
      Rcpp::_["VETA"] = matrix_to_r(sam.VETA, latent_names, latent_names),
      Rcpp::_["EETA"] = sam.EETA.size() > 0
          ? static_cast<SEXP>(vector_to_r(sam.EETA, latent_names))
          : R_NilValue,
      Rcpp::_["mapping_matrix"] = matrix_to_r(sam.mapping, latent_names, ov_names),
      Rcpp::_["reliability"] = Rcpp::wrap(sam.reliability),
      Rcpp::_["lambda_star"] = sam.lambda_star,
      Rcpp::_["latent_samp"] =
          sample_stats_to_r(sam.latent_samp, latent_samp_names),
      Rcpp::_["structural"] = sam_estimates_to_r(sam.structural),
      Rcpp::_["measurement"] = measurement);
  return out;
}

}  // namespace

// Complete-data ML with PSD primitive LISREL covariance matrices. The
// Cholesky lift is internal to C++; R receives the ordinary partable-shaped
// estimate and the same post-fit diagnostics as fit_ml_impl().
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_ml_psd_impl(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    double start_eigen_floor = 1e-6,
    double feasibility_tol = 1e-6,
    bool diagonal_preconditioning = true) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "frontier_fit_ml_psd");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(
      std::move(parsed.structure), std::move(parsed.names), sample_stats);
  std::string start_policy = "scaled-fabin";
  std::string start_fallback_reason = "none";

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, start_policy, &start_policy, &start_fallback_reason, control, true);
  const magmaan::estimate::Backend backend =
      optimizer.isNull()
          ? magmaan::estimate::Backend::NloptSlsqp
          : backend_from_optimizer_arg(optimizer);
  magmaan::estimate::frontier::PsdFitOptions psd_opts;
  psd_opts.start_eigen_floor = start_eigen_floor;
  psd_opts.feasibility_tol = feasibility_tol;
  psd_opts.diagonal_preconditioning = diagonal_preconditioning;
  auto e_or = magmaan::estimate::frontier::fit_ml_psd(
      ctx.pt, ctx.rep, ctx.samp, x0, backend, optim_opts_from(control, magmaan::estimate::frontier::ml_psd_optim_options()),
      psd_opts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  Rcpp::List out = fit_result(ctx, est, &starts, "ML");
  out["ml_start_policy"] = start_policy;
  out["ml_start_fallback_reason"] = start_fallback_reason;
  out["psd_preconditioning"] = diagonal_preconditioning ? "diagonal" : "none";
  return out;
}

// [[Rcpp::export]]
Rcpp::List frontier_fit_ml_psd_fallback_impl(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::Nullable<Rcpp::String> ordinary_optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::String> psd_optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> ordinary_control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> psd_control = R_NilValue,
    double start_eigen_floor = 1e-6, double feasibility_tol = 1e-6,
    bool diagonal_preconditioning = true) {
  namespace ef = magmaan::estimate::frontier;
  auto parsed = partable_from_arg(partable, "frontier_fit_ml_psd_fallback");
  auto starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(
      std::move(parsed.structure), std::move(parsed.names), sample_stats);
  // Pinned apart from fit_ml's layered default: with a layered ordinary stage
  // the recovery loses (decisions/01, lane psd-ml, second run, rule E).
  std::string start_policy = "scaled-fabin";
  std::string start_fallback_reason = "none";

  if (psd_control.isNotNull() && (Rcpp::List(psd_control.get()).containsElementNamed("start") ||
      Rcpp::List(psd_control.get()).containsElementNamed("start_transport")))
    Rcpp::stop("set the initial start policy in ordinary_control; PSD uses the ordinary estimates or that original start");
  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, start_policy, &start_policy, &start_fallback_reason, ordinary_control, true);
  ef::MlPsdFallbackOptions options;
  options.ordinary_backend = backend_from_optimizer_arg(ordinary_optimizer);
  if (psd_optimizer.isNotNull())
    options.psd_backend = backend_from_optimizer_arg(psd_optimizer);
  options.ordinary = optim_opts_from(ordinary_control, options.ordinary);
  options.psd = optim_opts_from(psd_control, options.psd);
  options.covariance.start_eigen_floor = start_eigen_floor;
  options.covariance.feasibility_tol = feasibility_tol;
  options.covariance.diagonal_preconditioning = diagonal_preconditioning;
  const auto result = ef::fit_ml_psd_fallback(ctx.pt, ctx.rep, ctx.samp, x0, options);
  auto attempt = [&](const magmaan::fit_expected<magmaan::estimate::Estimates>& value,
                     bool psd) {
    Rcpp::List row = Rcpp::List::create(
        Rcpp::_["fit"] = R_NilValue, Rcpp::_["error"] = R_NilValue,
        Rcpp::_["ml_start_policy"] = psd && result.warm_start_used ? "ordinary-estimates" : start_policy,
        Rcpp::_["ml_start_fallback_reason"] = psd && result.warm_start_used ? "none" : start_fallback_reason);
    if (value.has_value()) {
      Rcpp::List fit = fit_result(ctx, *value, &starts, "ML");
      if (psd && result.warm_start_used) {
        auto warm = magmaan::estimate::explicit_start_values(ctx.pt, result.ordinary->theta);
        if (!warm) stop_fit(warm.error());
        fit["start"] = start_result_to_r(*warm);
      }
      fit["ml_start_policy"] = psd && result.warm_start_used ? "ordinary-estimates" : start_policy;
      fit["ml_start_fallback_reason"] = psd && result.warm_start_used ? "none" : start_fallback_reason;
      if (psd) fit["psd_preconditioning"] = diagonal_preconditioning ? "diagonal" : "none";
      row["fit"] = fit;
    } else {
      const auto& error = value.error();
      row["error"] = Rcpp::List::create(
          Rcpp::_["kind"] = fit_error_kind(error.kind),
          Rcpp::_["detail"] = error.detail,
          Rcpp::_["iterations"] = error.iterations,
          Rcpp::_["f_value"] = error.f_value);
    }
    return row;
  };
  const char* reason = "none";
  switch (result.reason) {
    case ef::PsdFallbackReason::None: break;
    case ef::PsdFallbackReason::OrdinaryError: reason = "ordinary-error"; break;
    case ef::PsdFallbackReason::OrdinaryRejected: reason = "ordinary-rejected"; break;
    case ef::PsdFallbackReason::OrdinaryInadmissible: reason = "ordinary-inadmissible"; break;
  }
  Rcpp::List ordinary = attempt(result.ordinary, false);
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["fit"] = R_NilValue,
      Rcpp::_["converged"] = result.accepted_fit() != nullptr,
      Rcpp::_["ordinary"] = ordinary,
      Rcpp::_["psd"] = R_NilValue,
      Rcpp::_["fallback_used"] = result.psd.has_value(),
      Rcpp::_["fallback_reason"] = reason,
      Rcpp::_["warm_start_used"] = result.warm_start_used);
  if (result.psd.has_value()) {
    Rcpp::List psd = attempt(*result.psd, true);
    out["psd"] = psd;
    if (result.accepted_fit()) out["fit"] = psd["fit"];
  } else if (result.accepted_fit()) {
    out["fit"] = ordinary["fit"];
  }
  return out;
}

// Sphere-chart estimation (frontier). The optimizer walks unit-norm loading
// directions; the result is translated to and finalized in the user's chart,
// so it is the ordinary estimate whenever that estimate exists. When the user
// chart does not contain the fitted point, the return value is a bare list
// with `user_chart = FALSE` and the `gauge` report; the R wrapper turns that
// into a classed condition.
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_sphere_impl(
    SEXP partable, Rcpp::List sample_stats, std::string estimator = "ML",
    bool psd = false, SEXP W = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue,
    std::string metric = "unit_free", double pin_weight = 1.0,
    double pole_tol = 1e-6, double start_eigen_floor = 1e-6,
    double feasibility_tol = 1e-6, bool diagonal_preconditioning = false,
    bool polish = true, std::string start = "canonical") {
  namespace fr = magmaan::estimate::frontier;
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "frontier_fit_sphere");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(
      std::move(parsed.structure), std::move(parsed.names), sample_stats);
  const fr::SphereOptions sopts = sphere_options_from(
      metric, pin_weight, pole_tol, polish, start, &starts, control);

  Eigen::VectorXd x0;
  std::string start_policy = "scaled-fabin";
  std::string start_fallback_reason = "none";
  if (estimator == "ML") {
    if (control.isNotNull()) {
      Rcpp::List ctl(control.get());
      if (ctl.containsElementNamed("start"))
        start_policy = start_name_from_arg(
            Rcpp::Nullable<Rcpp::String>(ctl["start"]), "fit_ml", "scaled-fabin");
    }
    x0 = start_values_or_stop(ctx, starts, start_policy, &start_policy, &start_fallback_reason);
  } else {
    x0 = start_values_or_stop(ctx, starts);
  }

  magmaan::fit_expected<fr::SphereFit> r;
  if (psd) {
    if (estimator != "ML") {
      Rcpp::stop("magmaan: frontier_fit_sphere(psd = TRUE) supports estimator = \"ML\" only");
    }
    if (bounds.isNotNull()) {
      Rcpp::stop("magmaan: frontier_fit_sphere(psd = TRUE) does not take `bounds`");
    }
    const magmaan::estimate::Backend backend =
        optimizer.isNull() ? magmaan::estimate::Backend::NloptSlsqp
                           : backend_from_optimizer_arg(optimizer);
    fr::PsdFitOptions psd_opts;
    psd_opts.start_eigen_floor = start_eigen_floor;
    psd_opts.feasibility_tol = feasibility_tol;
    psd_opts.diagonal_preconditioning = diagonal_preconditioning;
    r = fr::fit_ml_psd_sphere(
        ctx.pt, ctx.rep, ctx.samp, x0, backend,
        optim_opts_from(control, fr::ml_psd_optim_options()), psd_opts, sopts);
  } else {
    const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
    const magmaan::estimate::Bounds b = bounds_from_nullable(bounds);
    if (estimator == "ML") {
      r = fr::fit_ml_sphere(ctx.pt, ctx.rep, ctx.samp, x0, b, backend,
          optim_opts_from(control, magmaan::estimate::ml_optim_options()), sopts);
    } else if (estimator == "ULS") {
      r = fr::fit_gmm_sphere(ctx.pt, ctx.rep, ctx.samp, x0, {}, b, backend,
                             optim_opts_from(control), sopts);
    } else if (estimator == "GLS") {
      r = fr::fit_gls_sphere(ctx.pt, ctx.rep, ctx.samp, x0, b, backend,
                             optim_opts_from(control), sopts);
    } else if (estimator == "WLS") {
      if (Rf_isNull(W)) Rcpp::stop("magmaan: continuous WLS requires explicit `W`");
      r = fr::fit_gmm_sphere(ctx.pt, ctx.rep, ctx.samp, x0,
                             wls_from_arg(W, ctx.samp.S.size()), b, backend,
                             optim_opts_from(control), sopts);
    } else {
      Rcpp::stop("magmaan: frontier_fit_sphere() supports estimator = ML, ULS, "
                 "GLS, WLS (complete data) or FIML");
    }
  }
  if (!r.has_value()) stop_fit(r.error());
  if (!r->user_chart) {
    return Rcpp::List::create(Rcpp::_["user_chart"] = false,
                              Rcpp::_["gauge"] = gauge_report_to_r(ctx, *r, metric));
  }
  Rcpp::List out = fit_result(ctx, r->estimates, &starts, estimator.c_str());
  out["gauge"] = gauge_report_to_r(ctx, *r, metric);
  if(sopts.verified_newton) {
    auto assessment=verified_assessment_to_r(r->report.native_verdict);
    out["verified_convergence"]=assessment;
    out["converged_compatibility"]=out["converged"];
    out["converged"]=assessment["converged"];
  }

  if (estimator == "ML") {
    out["ml_start_policy"] = start_policy;
    out["ml_start_fallback_reason"] = start_fallback_reason;
  }
  if (psd) out["psd_preconditioning"] = diagonal_preconditioning ? "diagonal" : "none";
  return out;
}

// [[Rcpp::export]]
Rcpp::List frontier_fit_fiml_sphere_impl(
    SEXP partable, SEXP raw_data,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    std::string metric = "unit_free", double pin_weight = 1.0,
    double pole_tol = 1e-6, bool polish = true, std::string start = "canonical") {
  namespace fr = magmaan::estimate::frontier;
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "frontier_fit_sphere");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.names = std::move(parsed.names);
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  if (ctx.rep.ov_names.empty() || ctx.rep.ov_names[0].empty())
    Rcpp::stop("magmaan: model has no observed variables");

  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, raw_data);
  auto pack_or = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack_or.has_value()) stop_fit(pack_or.error());
  ctx.samp = pack_or->start_stats;
  ctx.ov_names = ctx.rep.ov_names[0];
  ctx.meanstructure = has_meanstructure(ctx.pt);
  if (!ctx.meanstructure) ctx.samp.mean.clear();

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts);
  const magmaan::estimate::Backend backend =
      fiml_backend_from_optimizer_arg(optimizer);
  auto r = fr::fit_fiml_sphere(ctx.pt, ctx.rep, raw, x0, backend,
                               optim_opts_from(control),
                               sphere_options_from(metric, pin_weight, pole_tol,
                                                   polish, start, &starts, control));
  if (!r.has_value()) stop_fit(r.error());
  if (!r->user_chart) {
    return Rcpp::List::create(Rcpp::_["user_chart"] = false,
                              Rcpp::_["gauge"] = gauge_report_to_r(ctx, *r, metric));
  }
  Rcpp::List out = fiml_fit_result(ctx, raw, r->estimates, &starts);
  out["gauge"] = gauge_report_to_r(ctx, *r, metric);
  if(control.isNotNull() && Rcpp::List(control).containsElementNamed("verified_newton") &&
      Rcpp::as<bool>(Rcpp::List(control)["verified_newton"])) {
    auto assessment=verified_assessment_to_r(r->report.native_verdict);
    out["verified_convergence"]=assessment;
    out["converged_compatibility"]=out["converged"];
    out["converged"]=assessment["converged"];
  }
  auto h1_or = magmaan::estimate::fiml::fiml_h1_moments(
      raw, *pack_or, fiml_h1_opts_from(control));
  if (!h1_or.has_value()) stop_fit(h1_or.error());
  out["fiml_h1"] = fiml_h1_xptr(std::move(*h1_or));
  out["fiml_pack"] = fiml_pack_xptr(std::move(*pack_or));
  return out;
}

// Re-express a fitted partable (with an `est` column) under the
// identification of another partable of the same model. The source may be a
// sphere report's gauge-free `sphere_partable`.
//
// [[Rcpp::export]]
Rcpp::List frontier_reidentify_impl(SEXP from_partable, SEXP to_partable,
                                    double pole_tol = 1e-6) {
  namespace fr = magmaan::estimate::frontier;
  Rcpp::DataFrame from_df(from_partable);
  if (!from_df.containsElementNamed("est") || !from_df.containsElementNamed("free")) {
    Rcpp::stop("magmaan: frontier_reidentify() needs a fitted partable with "
               "`free` and `est` columns");
  }
  magmaan::compat::lavaan::ParsedLavaanParTable from =
      partable_from_arg(from_partable, "frontier_reidentify");
  magmaan::compat::lavaan::ParsedLavaanParTable to =
      partable_from_arg(to_partable, "frontier_reidentify");
  Rcpp::IntegerVector free = from_df["free"];
  Rcpp::NumericVector est = from_df["est"];
  Eigen::VectorXd theta = Eigen::VectorXd::Constant(
      from.structure.n_free(), std::numeric_limits<double>::quiet_NaN());
  for (R_xlen_t i = 0; i < free.size(); ++i) {
    const int f = free[i];
    if (f > 0 && f <= theta.size()) theta(f - 1) = est[i];
  }
  if (!theta.allFinite()) {
    Rcpp::stop("magmaan: frontier_reidentify(): the source partable has "
               "missing estimates");
  }
  auto r = fr::reidentify(from.structure, theta, to.structure, pole_tol);
  if (!r.has_value()) stop_post(r.error());
  magmaan::estimate::Estimates out_est;
  out_est.theta = r->theta;
  return Rcpp::List::create(
      Rcpp::_["partable"] = partable_df(to.structure, to.names, out_est, &to.starts),
      Rcpp::_["theta"] = Rcpp::wrap(r->theta),
      Rcpp::_["direction_level"] = Rcpp::wrap(r->direction_level),
      Rcpp::_["residual"] = Rcpp::List::create(
          Rcpp::_["fixed_rows"] = r->residual.fixed_rows,
          Rcpp::_["linear_constraints"] = r->residual.linear_constraints));
}

// Complete-data ML plus the frontier multi-information (complete-data
// correlation log-determinant) penalty. `fmin` is the UNPENALIZED ½F at the
// penalized estimate; `penalty` carries λ, P(θ̃), and per-equation terms.
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_ml_multiinfo_impl(
    SEXP partable, Rcpp::List sample_stats, double eta = 1.25,
    Rcpp::Nullable<Rcpp::NumericVector> weight = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue,
    std::string target = "joint") {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "frontier_fit_ml_multiinfo");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(
      std::move(parsed.structure), std::move(parsed.names), sample_stats);
  std::string start_policy = "scaled-fabin";
  std::string start_fallback_reason = "none";

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, start_policy, &start_policy, &start_fallback_reason, control);
  const magmaan::estimate::Backend backend = backend_from_optimizer_arg(optimizer);
  auto r = magmaan::estimate::frontier::fit_ml_multiinfo(
      ctx.pt, ctx.rep, ctx.samp, x0, multiinfo_options_from(eta, weight, target),
      bounds_from_nullable(bounds), backend,
      optim_opts_from(control, magmaan::estimate::ml_optim_options()));
  if (!r.has_value()) stop_fit(r.error());
  Rcpp::List out = fit_result(ctx, r->estimates, &starts, "ML");
  out["ml_start_policy"] = start_policy;
  out["ml_start_fallback_reason"] = start_fallback_reason;
  out["penalty"] = multiinfo_penalty_to_r(ctx, *r);
  return out;
}

// Continuous ULS over PSD primitive LISREL covariance matrices.
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_uls_psd_impl(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    double start_eigen_floor = 1e-6,
    double feasibility_tol = 1e-6) {
  return frontier_fit_gmm_psd_common(
      partable, sample_stats, R_NilValue, optimizer, control,
      start_eigen_floor, feasibility_tol, PsdGmmKind::Uls,
      "frontier_fit_uls_psd", "ULS");
}

// Normal-theory GLS over PSD primitive LISREL covariance matrices.
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_gls_psd_impl(
    SEXP partable, Rcpp::List sample_stats,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    double start_eigen_floor = 1e-6,
    double feasibility_tol = 1e-6) {
  return frontier_fit_gmm_psd_common(
      partable, sample_stats, R_NilValue, optimizer, control,
      start_eigen_floor, feasibility_tol, PsdGmmKind::Gls,
      "frontier_fit_gls_psd", "GLS");
}

// Caller-fixed WLS/ADF over PSD primitive LISREL covariance matrices.
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_wls_psd_impl(
    SEXP partable, Rcpp::List sample_stats, SEXP W,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    double start_eigen_floor = 1e-6,
    double feasibility_tol = 1e-6) {
  return frontier_fit_gmm_psd_common(
      partable, sample_stats, W, optimizer, control,
      start_eigen_floor, feasibility_tol, PsdGmmKind::Wls,
      "frontier_fit_wls_psd", "WLS");
}

// [[Rcpp::export]]
Rcpp::List frontier_sam_impl(
    SEXP partable,
    Rcpp::List sample_stats,
    SEXP raw_data = R_NilValue,
    std::string method = "local",
    std::string mapping = "ml",
    std::string se = "twostep",
    bool lambda_correction = true,
    int alpha_correction = 0,
    bool meanstructure = false,
    Rcpp::Nullable<Rcpp::String> mm_optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::String> struc_optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> mm_control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> struc_control = R_NilValue) {
  Ctx ctx = ctx_from_partable_sample_stats(partable, sample_stats,
                                           "frontier_sam");
  magmaan::estimate::frontier::SamOptions opts;
  opts.method = sam_method_from_string(method);
  opts.local.mapping = sam_mapping_from_string(mapping);
  opts.se = sam_se_from_string(se);
  opts.local.lambda_correction = lambda_correction;
  opts.local.alpha_correction = alpha_correction;
  opts.meanstructure = meanstructure;
  opts.mm_backend = backend_from_optimizer_arg(mm_optimizer);
  opts.struc_backend = struc_optimizer.isNull()
      ? opts.mm_backend
      : backend_from_optimizer_arg(struc_optimizer);
  opts.mm_control = optim_opts_from(mm_control);
  opts.struc_control = struc_control.isNull()
      ? opts.mm_control
      : optim_opts_from(struc_control);

  auto sam_or = [&]() {
    if (Rf_isNull(raw_data)) {
      return magmaan::estimate::frontier::fit_sam(
          ctx.pt, ctx.rep, ctx.names, ctx.samp, opts);
    }
    magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
    return magmaan::estimate::frontier::fit_sam(
        ctx.pt, ctx.rep, ctx.names, raw, opts);
  }();
  if (!sam_or.has_value()) stop_fit(sam_or.error());
  return sam_result_to_r(ctx, *sam_or, opts.method, opts.local.mapping,
                         opts.se, opts.local.lambda_correction,
                         opts.local.alpha_correction);
}

// [[Rcpp::export]]
Rcpp::List frontier_rbm_impl(
    Rcpp::List fit,
    SEXP raw_data = R_NilValue,
    SEXP weight = R_NilValue,
    SEXP stage2_weight = R_NilValue,
    SEXP dls_a = R_NilValue,
    std::string method = "explicit",
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue,
    bool estimated_weight = true) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  const std::string method_key = rbm_method_key(std::move(method));
  magmaan::estimate::frontier::RBMOptions opts =
      rbm_options_from(optimizer, control);
  opts.estimated_weight = estimated_weight;
  const magmaan::estimate::Bounds b = bounds_from_nullable(bounds);

  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "ordinal_stats", "frontier_rbm"));
    const std::string parameterization_name =
        fit.containsElementNamed("parameterization")
            ? Rcpp::as<std::string>(fit["parameterization"])
            : ordinal_parameterization_attr(fit["partable"]);
    const auto parameterization =
        ordinal_parameterization_from_string(parameterization_name);
    const auto ow = ordinal_weight_from_estimator(
        ordinal_weight_for_postfit(fit, estimator), "frontier_rbm");
    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_ordinal(
                  ctx.pt, ctx.rep, stats, est, ow, parameterization, b, opts)
            : magmaan::estimate::frontier::rbm_implicit_ordinal(
                  ctx.pt, ctx.rep, stats, est, ow, parameterization, b, opts);
    if (!rbm.has_value()) stop_fit(rbm.error());
    Rcpp::List out = ordinal_fit_result(
        ctx, stats, rbm->estimates, nullptr, "RBM-ORDINAL",
        parameterization_name.c_str());
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, estimator.c_str());
    return out;
  }

  if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "mixed_ordinal_stats", "frontier_rbm"));
    const std::string parameterization_name =
        fit.containsElementNamed("parameterization")
            ? Rcpp::as<std::string>(fit["parameterization"])
            : ordinal_parameterization_attr(fit["partable"]);
    const auto parameterization =
        ordinal_parameterization_from_string(parameterization_name);
    const auto ow = ordinal_weight_from_estimator(
        ordinal_weight_for_postfit(fit, estimator), "frontier_rbm");
    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_mixed_ordinal(
                  ctx.pt, ctx.rep, stats, est, ow, parameterization, b, opts)
            : magmaan::estimate::frontier::rbm_implicit_mixed_ordinal(
                  ctx.pt, ctx.rep, stats, est, ow, parameterization, b, opts);
    if (!rbm.has_value()) stop_fit(rbm.error());
    Rcpp::List out = mixed_ordinal_fit_result(
        ctx, stats, rbm->estimates, nullptr, "RBM-MIXED-ORDINAL",
        parameterization_name.c_str());
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, estimator.c_str());
    return out;
  }

  if (fit.containsElementNamed("stage1")) {
    SEXP rd = raw_data;
    if (Rf_isNull(rd) && fit.containsElementNamed("raw_data")) {
      rd = fit["raw_data"];
    }
    if (Rf_isNull(rd)) {
      Rcpp::stop("magmaan: frontier_rbm() needs raw_data for ML2S fits");
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, rd);
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
    std::unique_ptr<FimlH1> owned_h1;
    const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);
    magmaan::estimate::fiml::TwoStageWeight kind;
    magmaan::estimate::fiml::TwoStageDlsOptions dls;
    ml2s_recorded_stage2(fit, stage2_weight, dls_a, "frontier_rbm()", kind,
                         dls);
    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_two_stage(
                  ctx.pt, ctx.rep, raw, pack, h1, est, kind, dls, b, opts)
            : magmaan::estimate::frontier::rbm_implicit_two_stage(
                  ctx.pt, ctx.rep, raw, pack, h1, est, kind, dls, b, opts);
    if (!rbm.has_value()) stop_fit(rbm.error());
    Rcpp::List out = fit_result(ctx, rbm->estimates, nullptr, "RBM-ML2S");
    out["stage1"] = fit["stage1"];
    out["stage2_weight"] = stage2_weight;
    out["stage2_dls_a"] = dls_a;
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, "ML2S");
    return out;
  }

  const bool is_fiml = fit.containsElementNamed("fiml") &&
                       Rcpp::as<bool>(fit["fiml"]);
  if (is_fiml) {
    SEXP rd = raw_data;
    if (Rf_isNull(rd)) {
      if (!fit.containsElementNamed("raw_data")) {
        Rcpp::stop("magmaan: frontier_rbm() needs raw_data or a FIML fit with $raw_data");
      }
      rd = fit["raw_data"];
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, rd);
    auto pack = magmaan::estimate::fiml::fiml_pack(raw);
    if (!pack.has_value()) stop_fit(pack.error());

    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_fiml(
                  ctx.pt, ctx.rep, raw, *pack, est, b, opts)
            : magmaan::estimate::frontier::rbm_implicit_fiml(
                  ctx.pt, ctx.rep, raw, *pack, est, b, opts);
    if (!rbm.has_value()) stop_fit(rbm.error());

    Rcpp::List out = fit_result(ctx, rbm->estimates, nullptr, "RBM-FIML");
    out["fiml"] = true;
    out["raw_data"] =
        fiml_raw_to_r(raw, ctx.rep.ov_names, ctx.names.group_labels);
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, "FIML");
    return out;
  }

  if (estimator == "ULS" || estimator == "GLS" || estimator == "WLS") {
    if (Rf_isNull(raw_data)) {
      Rcpp::stop("magmaan: frontier_rbm() needs raw_data for continuous LS fits");
    }
    magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
    auto w = continuous_ls_weight(fit, ctx, est, estimator, weight, "RBM");
    magmaan::estimate::gmm::FixedWeightOptions dls_opts;
    const auto mode = estimated_weight
        ? continuous_ij_mode_for_fit(fit, estimator, &dls_opts)
        : magmaan::estimate::ContinuousLsIJWeightMode::Fixed;
    magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
        method_key == "explicit"
            ? magmaan::estimate::frontier::rbm_explicit_continuous_ls(
                  ctx.pt, ctx.rep, ctx.samp, est, w, raw, mode, dls_opts, b,
                  opts)
            : magmaan::estimate::frontier::rbm_implicit_continuous_ls(
                  ctx.pt, ctx.rep, ctx.samp, est, w, raw, mode, dls_opts, b,
                  opts);
    if (!rbm.has_value()) stop_fit(rbm.error());
    Rcpp::List out = fit_result(ctx, rbm->estimates, nullptr,
                                ("RBM-" + estimator).c_str());
    out["rbm"] = rbm_metadata_to_r(*rbm, method_key, estimator.c_str());
    return out;
  }

  if (Rf_isNull(raw_data)) {
    Rcpp::stop("magmaan: frontier_rbm() needs raw_data for complete-data ML fits");
  }
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
  magmaan::fit_expected<magmaan::estimate::frontier::RBMResult> rbm =
      method_key == "explicit"
          ? magmaan::estimate::frontier::rbm_explicit_ml(
                ctx.pt, ctx.rep, ctx.samp, raw, est, b, opts)
          : magmaan::estimate::frontier::rbm_implicit_ml(
                ctx.pt, ctx.rep, ctx.samp, raw, est, b, opts);
  if (!rbm.has_value()) stop_fit(rbm.error());

  Rcpp::List out = fit_result(ctx, rbm->estimates, nullptr, "RBM-ML");
  out["rbm"] = rbm_metadata_to_r(*rbm, method_key, "ML");
  return out;
}

// fcsem_model_spec() — build the native, non-folded FC-SEM partable directly
// from syntax. This is the R frontier analogue of api::frontier::model_spec().
//
// [[Rcpp::export]]
Rcpp::List fcsem_model_spec_impl(std::string syntax) {
  FcSemCtx ctx = fcsem_model_from_syntax(syntax);
  return Rcpp::List::create(
      Rcpp::_["syntax"]   = syntax,
      Rcpp::_["partable"] = fcsem_partable_df(ctx.pt, ctx.names, ctx.starts),
      Rcpp::_["ov_names"] = Rcpp::wrap(ctx.ov_names));
}

// fit_ml_fcsem() — native FC-SEM ML, using covariance-only sample statistics.
// Starts come from simple_fcsem_start_values(); optimization currently uses
// the same optimizer control list as the ordinary R ML bridge.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_fcsem_impl(std::string syntax, Rcpp::List sample_stats,
                             Rcpp::Nullable<Rcpp::List> control = R_NilValue) {
  FcSemCtx ctx = fcsem_ctx_from_syntax_sample_stats(syntax, sample_stats);
  auto x0_or = magmaan::estimate::simple_fcsem_start_values(ctx.pt, ctx.samp);
  if (!x0_or.has_value()) stop_fit(x0_or.error());
  auto e_or = magmaan::estimate::fit_ml_fcsem(
      ctx.pt, ctx.samp, *x0_or, {}, magmaan::estimate::Backend::NloptLbfgs,
      optim_opts_from(control));
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return fcsem_fit_result(ctx, est, syntax);
}

// [[Rcpp::export]]
Rcpp::List fcsem_standard_errors_impl(Rcpp::List fit) {
  FcSemCtx ctx = fcsem_ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto info_or =
      magmaan::inference::information_expected_fcsem(ctx.pt, ctx.samp, est);
  if (!info_or.has_value()) stop_post(info_or.error());
  auto vcov_or = magmaan::inference::vcov(*info_or, ctx.pt, est.theta);
  if (!vcov_or.has_value()) stop_post(vcov_or.error());
  return Rcpp::List::create(
      Rcpp::_["information"] = Rcpp::wrap(*info_or),
      Rcpp::_["vcov"]        = Rcpp::wrap(*vcov_or),
      Rcpp::_["se"]          = Rcpp::wrap(magmaan::inference::se(*vcov_or)));
}

// [[Rcpp::export]]
Rcpp::List fcsem_fit_measures_impl(Rcpp::List fit) {
  FcSemCtx ctx = fcsem_ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const double chi2 = magmaan::inference::chi2_stat(ctx.samp, est);
  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  const magmaan::measures::BaselineFit bl =
      magmaan::measures::baseline_chi2(ctx.samp);
  const magmaan::measures::FitMeasures fm =
      magmaan::measures::fit_measures(chi2, *df_or, bl, ctx.samp);
  auto fx_or = magmaan::measures::fit_extras_fcsem(ctx.pt, ctx.samp, est);
  if (!fx_or.has_value()) stop_post(fx_or.error());
  return Rcpp::List::create(
      Rcpp::_["chisq"]                  = chi2,
      Rcpp::_["df"]                     = *df_or,
      Rcpp::_["baseline.chisq"]         = bl.chi2,
      Rcpp::_["baseline.df"]            = bl.df,
      Rcpp::_["cfi"]                    = fm.cfi,
      Rcpp::_["tli"]                    = fm.tli,
      Rcpp::_["rmsea"]                  = fm.rmsea,
      Rcpp::_["rmsea.ci.lower"]         = fm.rmsea_ci_lower,
      Rcpp::_["rmsea.ci.upper"]         = fm.rmsea_ci_upper,
      Rcpp::_["rmsea.pvalue"]           = fm.rmsea_pvalue,
      Rcpp::_["rmsea.close.h0"]         = fm.rmsea_close_h0,
      Rcpp::_["rmsea.notclose.pvalue"]  = fm.rmsea_notclose_pvalue,
      Rcpp::_["rmsea.notclose.h0"]      = fm.rmsea_notclose_h0,
      Rcpp::_["srmr"]                   = fx_or->srmr,
      Rcpp::_["logl"]                   = fx_or->logl,
      Rcpp::_["unrestricted.logl"]      = fx_or->unrestricted_logl,
      Rcpp::_["aic"]                    = fx_or->aic,
      Rcpp::_["bic"]                    = fx_or->bic,
      Rcpp::_["bic2"]                   = fx_or->bic2,
      Rcpp::_["npar"]                   = fx_or->npar,
      Rcpp::_["ntotal"]                 = static_cast<double>(fx_or->ntotal));
}

// [[Rcpp::export]]
Rcpp::DataFrame fcsem_standardized_rows_impl(Rcpp::List fit,
                                             Rcpp::NumericMatrix vcov) {
  FcSemCtx ctx = fcsem_ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto rows_or = magmaan::measures::standardize::standardized_rows_fcsem(
      ctx.pt, ctx.names, ctx.samp, est, vcov_m);
  if (!rows_or.has_value()) stop_post(rows_or.error());
  return fcsem_standardized_rows_df(*rows_or);
}

// [[Rcpp::export]]
SEXP frontier_dls_weight_impl(Rcpp::List fit, SEXP raw_data,
                              double dls_a = 0.5) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
  auto ev_or = magmaan::model::ModelEvaluator::build(ctx.pt, ctx.rep);
  if (!ev_or.has_value()) stop_model(ev_or.error());
  magmaan::estimate::frontier::DlsWeightOptions opts;
  opts.a = dls_a;
  auto w_or = magmaan::estimate::frontier::dls_weight(
      *ev_or, ctx.samp, raw, est.theta, opts);
  if (!w_or.has_value()) stop_fit(w_or.error());
  return weight_to_r(*w_or);
}
