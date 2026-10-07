#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

namespace {

magmaan::estimate::fiml::Stage1RegularizationTarget
stage1_regularization_target_from_string(const std::string& target);
magmaan::estimate::fiml::Stage1RegularizationOptions
stage1_regularization_options_from(SEXP arg);
Rcpp::List saturated_moments_to_r(
    const magmaan::estimate::fiml::SaturatedMoments& out,
    bool include_information = true,
    const magmaan::estimate::fiml::FIMLH1* endpoint = nullptr);
Rcpp::List stage1_regularization_diagnostics_to_r(
    const magmaan::estimate::fiml::Stage1RegularizedMoments& r,
    const magmaan::estimate::fiml::Stage1RegularizationOptions& opts);

magmaan::estimate::fiml::Stage1RegularizationTarget
stage1_regularization_target_from_string(const std::string& target) {
  using T = magmaan::estimate::fiml::Stage1RegularizationTarget;
  if (target == "diagonal") return T::Diagonal;
  if (target == "scaled_identity" || target == "scaled.identity")
    return T::ScaledIdentity;
  if (target == "identity") return T::Identity;
  Rcpp::stop("magmaan: stage1_regularization$target must be diagonal, "
             "scaled_identity, or identity");
}

magmaan::estimate::fiml::Stage1RegularizationOptions
stage1_regularization_options_from(SEXP arg) {
  using Opt = magmaan::estimate::fiml::Stage1RegularizationOptions;
  Opt opts;
  if (Rf_isNull(arg)) return opts;

  opts.enabled = true;
  opts.condition_max = 1e6;
  if (Rf_isLogical(arg) && Rf_length(arg) == 1) {
    const int flag = LOGICAL(arg)[0];
    opts.enabled = flag != 0 && flag != NA_LOGICAL;
    return opts;
  }
  if (TYPEOF(arg) != VECSXP) {
    Rcpp::stop("magmaan: stage1_regularization must be NULL, TRUE/FALSE, or a list");
  }

  Rcpp::List l(arg);
  if (l.containsElementNamed("enabled") && !Rf_isNull(l["enabled"])) {
    opts.enabled = Rcpp::as<bool>(l["enabled"]);
  }
  if (!opts.enabled) return opts;
  if (l.containsElementNamed("target") && !Rf_isNull(l["target"])) {
    opts.target = stage1_regularization_target_from_string(
        Rcpp::as<std::string>(l["target"]));
  }
  if (l.containsElementNamed("intensity") && !Rf_isNull(l["intensity"])) {
    const double x = Rcpp::as<double>(l["intensity"]);
    opts.intensity = std::isfinite(x)
        ? x
        : std::numeric_limits<double>::quiet_NaN();
  }
  if (l.containsElementNamed("condition_max") &&
      !Rf_isNull(l["condition_max"])) {
    const double x = Rcpp::as<double>(l["condition_max"]);
    opts.condition_max = (std::isfinite(x) || std::isinf(x))
        ? x
        : std::numeric_limits<double>::infinity();
  }
  if (l.containsElementNamed("min_eigenvalue") &&
      !Rf_isNull(l["min_eigenvalue"])) {
    opts.min_eigenvalue = Rcpp::as<double>(l["min_eigenvalue"]);
  }
  if (l.containsElementNamed("jacobian_step") &&
      !Rf_isNull(l["jacobian_step"])) {
    opts.jacobian_step = Rcpp::as<double>(l["jacobian_step"]);
  }
  return opts;
}

Rcpp::List saturated_moments_to_r(
    const magmaan::estimate::fiml::SaturatedMoments& out,
    bool include_information,
    const magmaan::estimate::fiml::FIMLH1* endpoint) {
  const R_xlen_t nb = static_cast<R_xlen_t>(out.mean.size());
  Rcpp::List mean_out(nb), cov_out(nb);
  Rcpp::IntegerVector nobs(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::size_t bi = static_cast<std::size_t>(b);
    mean_out[b] = Rcpp::wrap(out.mean[bi]);
    cov_out[b]  = Rcpp::wrap(out.cov[bi]);
    nobs[b]     = static_cast<int>(out.n_obs[bi]);
  }

  Rcpp::List ans = Rcpp::List::create(
      Rcpp::Named("mean") = mean_out,
      Rcpp::Named("cov")  = cov_out,
      Rcpp::Named("n_obs") = nobs,
      Rcpp::Named("warnings") = Rcpp::wrap(out.warnings),
      Rcpp::Named("acov") = Rcpp::wrap(out.acov));
  if (include_information) {
    ans["H"] = Rcpp::wrap(out.H);
    ans["J"] = Rcpp::wrap(out.J);
  }
  ans["raw_H"] = Rcpp::wrap(out.raw_H);
  ans["raw_gradient"] = Rcpp::wrap(out.raw_gradient);
  ans["raw_hessian_analytic"] = out.raw_hessian_analytic;
  ans["information_repaired"] = out.information_repaired;
  ans["information_ridge"] = out.information_ridge;
  ans["information_min_eigen"] = out.information_min_eigen;
  ans["solver_recorded"] = out.solver_recorded;
  const auto& o = out.solver_options;
  ans["solver_options"] = Rcpp::List::create(
      Rcpp::_["h1_em_max_iter"] = o.max_iter,
      Rcpp::_["h1_em_param_tol"] = o.parameter_tol,
      Rcpp::_["h1_em_objective_tol"] = o.objective_tol,
      Rcpp::_["h1_em_cov_floor"] = o.covariance_floor,
      Rcpp::_["h1_em_cov_warn"] = o.covariance_warn,
      Rcpp::_["h1_em_error_on_nonconvergence"] = o.error_on_nonconvergence);
  Rcpp::List blocks(out.solver_blocks.size());
  using Stop = magmaan::estimate::fiml::H1StopReason;
  for (std::size_t b = 0; b < out.solver_blocks.size(); ++b) {
    const auto& x = out.solver_blocks[b];
    blocks[b] = Rcpp::List::create(
        Rcpp::_["stop"] = x.stop == Stop::Direct ? "direct" :
            x.stop == Stop::ParameterTolerance ? "parameter_tolerance" : "iteration_limit",
        Rcpp::_["iterations"] = x.iterations,
        Rcpp::_["parameter_change"] = x.parameter_change,
        Rcpp::_["objective_change"] = x.objective_change,
        Rcpp::_["objective_converged"] = x.objective_converged,
        Rcpp::_["covariance_repairs"] = x.covariance_repairs,
        Rcpp::_["max_covariance_ridge"] = x.max_covariance_ridge,
        Rcpp::_["min_covariance_eigen"] = x.min_covariance_eigen);
  }
  ans["solver_blocks"] = blocks;
  if (endpoint) ans["endpoint_value"] = endpoint->value;
  return ans;
}

// Optional numerical provenance is absent on legacy serialized Stage-1 lists.
// Preserve that absence; never invent a successful solver history.
bool saturated_audit_from_list(Rcpp::List st, SaturatedMoments& out) {
  if (!magmaanr::saturated_from_list(st, out)) return false;
  if (st.containsElementNamed("raw_H")) out.raw_H = Rcpp::as<Eigen::MatrixXd>(st["raw_H"]);
  if (st.containsElementNamed("raw_gradient")) out.raw_gradient = Rcpp::as<Eigen::VectorXd>(st["raw_gradient"]);
  if (st.containsElementNamed("raw_hessian_analytic")) out.raw_hessian_analytic = Rcpp::as<bool>(st["raw_hessian_analytic"]);
  if (st.containsElementNamed("information_repaired")) out.information_repaired = Rcpp::as<bool>(st["information_repaired"]);
  if (st.containsElementNamed("information_ridge")) out.information_ridge = Rcpp::as<double>(st["information_ridge"]);
  if (st.containsElementNamed("information_min_eigen")) out.information_min_eigen = Rcpp::as<double>(st["information_min_eigen"]);
  if (st.containsElementNamed("solver_recorded")) out.solver_recorded = Rcpp::as<bool>(st["solver_recorded"]);
  if (st.containsElementNamed("solver_options")) out.solver_options = fiml_h1_opts_from(Rcpp::List(st["solver_options"]));
  if (st.containsElementNamed("solver_blocks")) {
    Rcpp::List blocks(st["solver_blocks"]);
    using Stop = magmaan::estimate::fiml::H1StopReason;
    for (R_xlen_t b = 0; b < blocks.size(); ++b) {
      Rcpp::List x(blocks[b]);
      magmaan::estimate::fiml::H1BlockDiagnostics d;
      const auto s = Rcpp::as<std::string>(x["stop"]);
      if (s != "direct" && s != "parameter_tolerance" && s != "iteration_limit")
        Rcpp::stop("invalid retained Stage-1 stop reason");
      d.stop = s == "direct" ? Stop::Direct : s == "parameter_tolerance" ? Stop::ParameterTolerance : Stop::IterationLimit;
      d.iterations = Rcpp::as<int>(x["iterations"]);
      d.parameter_change = Rcpp::as<double>(x["parameter_change"]);
      d.objective_change = Rcpp::as<double>(x["objective_change"]);
      d.objective_converged = Rcpp::as<bool>(x["objective_converged"]);
      d.covariance_repairs = Rcpp::as<int>(x["covariance_repairs"]);
      d.max_covariance_ridge = Rcpp::as<double>(x["max_covariance_ridge"]);
      d.min_covariance_eigen = Rcpp::as<double>(x["min_covariance_eigen"]);
      out.solver_blocks.push_back(d);
    }
  }
  return true;
}

Rcpp::List stage1_regularization_diagnostics_to_r(
    const magmaan::estimate::fiml::Stage1RegularizedMoments& r,
    const magmaan::estimate::fiml::Stage1RegularizationOptions& opts) {
  const R_xlen_t nb = static_cast<R_xlen_t>(r.block_diagnostics.size());
  Rcpp::CharacterVector target(nb);
  Rcpp::NumericVector raw_min(nb), raw_max(nb), raw_cond(nb);
  Rcpp::NumericVector min_eig(nb), max_eig(nb), cond(nb), intensity(nb);
  Rcpp::LogicalVector applied(nb);
  bool any = false;
  for (R_xlen_t b = 0; b < nb; ++b) {
    const auto& d = r.block_diagnostics[static_cast<std::size_t>(b)];
    target[b] = d.target;
    raw_min[b] = d.raw_min_eigen;
    raw_max[b] = d.raw_max_eigen;
    raw_cond[b] = d.raw_condition;
    min_eig[b] = d.min_eigen;
    max_eig[b] = d.max_eigen;
    cond[b] = d.condition;
    intensity[b] = d.intensity;
    applied[b] = d.applied;
    any = any || d.applied;
  }
  Rcpp::List blocks = Rcpp::List::create(
      Rcpp::_["target"] = target,
      Rcpp::_["raw_min_eigen"] = raw_min,
      Rcpp::_["raw_max_eigen"] = raw_max,
      Rcpp::_["raw_condition"] = raw_cond,
      Rcpp::_["min_eigen"] = min_eig,
      Rcpp::_["max_eigen"] = max_eig,
      Rcpp::_["condition"] = cond,
      Rcpp::_["intensity"] = intensity,
      Rcpp::_["applied"] = applied);
  const std::string first_target =
      target.size() > 0 ? Rcpp::as<std::string>(target[0]) : "";
  return Rcpp::List::create(
      Rcpp::_["enabled"] = opts.enabled,
      Rcpp::_["target"] = first_target,
      Rcpp::_["condition_max"] = opts.condition_max,
      Rcpp::_["min_eigenvalue"] = opts.min_eigenvalue,
      Rcpp::_["fixed_intensity"] =
          std::isfinite(opts.intensity) ? opts.intensity : NA_REAL,
      Rcpp::_["jacobian_step"] = opts.jacobian_step,
      Rcpp::_["applied"] = any,
      Rcpp::_["blocks"] = blocks);
}

}  // namespace

// fit_fiml() — mirrors estimate::fit_fiml(pt, rep, raw, FIML{}).
// `raw_data` is a magmaan_fiml_data object from df_to_fiml_data(), or a list
// with $X and optional $mask. Missing values are retained in $X and represented
// by $mask; columns are reordered to the model's observed-variable order.
//
// [[Rcpp::export]]
Rcpp::List fit_fiml_impl(SEXP partable, SEXP raw_data,
                         Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                         Rcpp::Nullable<Rcpp::List> control = R_NilValue) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_fiml");
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
  if (auto e = magmaan::estimate::fiml::validate_fiml_fixed_x_missing_policy(
          ctx.pt, raw); !e.has_value()) {
    stop_fit(e.error());
  }
  auto pack_or = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack_or.has_value()) stop_fit(pack_or.error());
  ctx.samp = pack_or->start_stats;
  ctx.ov_names = ctx.rep.ov_names[0];
  ctx.meanstructure = has_meanstructure(ctx.pt);
  if (!ctx.meanstructure) ctx.samp.mean.clear();

  Rcpp::List ctl = control.isNotNull() ? Rcpp::List(control.get()) : Rcpp::List::create();
  auto h1_or = ctl.containsElementNamed("fitting_options")
      ? magmaan::estimate::lavaan_fiml_h1(raw, *pack_or)
      : magmaan::estimate::fiml::fiml_h1_moments(raw, *pack_or, fiml_h1_opts_from(control));
  if (!h1_or) stop_fit(h1_or.error());
  magmaan::fit_expected<magmaan::estimate::Estimates> e_or;
  if (ctl.containsElementNamed("fitting_options")) {
    check_optim_control_names(ctl, {"fitting_options", "start"});
    if (optimizer.isNotNull()) Rcpp::stop("select optimizer through fitting options");
    auto options = fitting_options_from(Rcpp::as<Rcpp::List>(ctl["fitting_options"]));
    Eigen::VectorXd explicit_start;
    if (ctl.containsElementNamed("start")) explicit_start = Rcpp::as<Eigen::VectorXd>(ctl["start"]);
    e_or = magmaan::estimate::fit_fiml_configured(ctx.pt, ctx.rep, raw,
        *pack_or, *h1_or, options, starts, explicit_start);
  } else {
    const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
    e_or = magmaan::estimate::fit_fiml(ctx.pt, ctx.rep, raw, x0, *pack_or,
        fiml_backend_from_optimizer_arg(optimizer), optim_opts_from(control));
  }
  if (!e_or) stop_fit(e_or.error());
  Rcpp::List out = fiml_fit_result(ctx, raw, *e_or, &starts);
  out["fiml_h1"] = fiml_h1_xptr(std::move(*h1_or));
  out["fiml_pack"] = fiml_pack_xptr(std::move(*pack_or));
  return out;
}

// Raw-data FIML with PSD primitive LISREL covariance matrices. The Cholesky
// lift is internal; the returned fit retains the ordinary partable parameters.
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_fiml_psd_impl(
    SEXP partable, SEXP raw_data,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    double start_eigen_floor = 1e-6,
    double feasibility_tol = 1e-6) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "frontier_fit_fiml_psd");
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
  if (auto e = magmaan::estimate::fiml::validate_fiml_fixed_x_missing_policy(
          ctx.pt, raw); !e.has_value()) {
    stop_fit(e.error());
  }
  auto pack_or = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack_or.has_value()) stop_fit(pack_or.error());
  ctx.samp = pack_or->start_stats;
  ctx.ov_names = ctx.rep.ov_names[0];
  ctx.meanstructure = has_meanstructure(ctx.pt);
  if (!ctx.meanstructure) ctx.samp.mean.clear();

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  const magmaan::estimate::Backend backend =
      optimizer.isNull()
          ? magmaan::estimate::Backend::NloptSlsqp
          : fiml_backend_from_optimizer_arg(optimizer);
  magmaan::estimate::frontier::PsdFitOptions psd_opts;
  psd_opts.start_eigen_floor = start_eigen_floor;
  psd_opts.feasibility_tol = feasibility_tol;
  auto e_or = magmaan::estimate::fiml::frontier::fit_fiml_psd(
      ctx.pt, ctx.rep, raw, x0, *pack_or, backend, optim_opts_from(control),
      psd_opts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  Rcpp::List out = fiml_fit_result(ctx, raw, est, &starts);
  out["fiml_pack"] = fiml_pack_xptr(std::move(*pack_or));
  out["covariance_policy"] = "psd";
  return out;
}

// Casewise FIML plus the multi-information penalty (N = number of cases).
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_fiml_multiinfo_impl(
    SEXP partable, SEXP raw_data, double eta = 1.25,
    Rcpp::Nullable<Rcpp::NumericVector> weight = R_NilValue,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue,
    std::string target = "joint") {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "frontier_fit_fiml_multiinfo");
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
  if (auto e = magmaan::estimate::fiml::validate_fiml_fixed_x_missing_policy(
          ctx.pt, raw); !e.has_value()) {
    stop_fit(e.error());
  }
  auto pack_or = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack_or.has_value()) stop_fit(pack_or.error());
  ctx.samp = pack_or->start_stats;
  ctx.ov_names = ctx.rep.ov_names[0];
  ctx.meanstructure = has_meanstructure(ctx.pt);
  if (!ctx.meanstructure) ctx.samp.mean.clear();

  const Eigen::VectorXd x0 = start_values_or_stop(ctx, starts, "fabin3", nullptr, nullptr, control);
  const magmaan::estimate::Backend backend =
      optimizer.isNull()
          ? magmaan::estimate::Backend::NloptLbfgs
          : fiml_backend_from_optimizer_arg(optimizer);
  auto r = magmaan::estimate::fiml::frontier::fit_fiml_multiinfo(
      ctx.pt, ctx.rep, raw, x0, *pack_or,
      multiinfo_options_from(eta, weight, target),
      bounds_from_nullable(bounds), backend, optim_opts_from(control));
  if (!r.has_value()) stop_fit(r.error());
  Rcpp::List out = fiml_fit_result(ctx, raw, r->estimates, &starts);
  out["fiml_pack"] = fiml_pack_xptr(std::move(*pack_or));
  out["penalty"] = multiinfo_penalty_to_r(ctx, *r);
  return out;
}

// saturated_em_moments_impl() — Stage-1 of the Savalei-Bentler (2009) two-stage
// missing-data path. Takes raw data only (no `partable`/spec needed because the
// saturated model has no structural restrictions) and returns the per-block EM
// mean and covariance plus the block-diagonal saturated information `H`,
// score-covariance `J`, and sandwich `ACOV = H^{-1} J H^{-1}`. See the C++
// `SaturatedMoments` doc comment for the η = (μ, vech(Σ)) layout convention.
//
// [[Rcpp::export]]
Rcpp::List saturated_em_moments_impl(
    SEXP raw_data, double h_step = 1e-4,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue) {
  SEXP X_arg = raw_data;
  SEXP mask_arg = R_NilValue;
  if (TYPEOF(raw_data) == VECSXP) {
    Rcpp::List rd(raw_data);
    if (rd.containsElementNamed("X")) {
      X_arg = rd["X"];
      if (rd.containsElementNamed("mask")) mask_arg = rd["mask"];
    }
  }

  const std::size_t n_blocks = TYPEOF(X_arg) == VECSXP
      ? static_cast<std::size_t>(Rcpp::List(X_arg).size())
      : 1u;
  if (n_blocks == 0) Rcpp::stop("magmaan: saturated_em_moments needs at least one data block");

  magmaan::data::RawData raw;
  raw.X.reserve(n_blocks);
  bool any_missing = false;
  std::vector<Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>> masks;
  masks.reserve(n_blocks);

  for (std::size_t b = 0; b < n_blocks; ++b) {
    Rcpp::NumericMatrix Xb = block_matrix(X_arg, b, n_blocks, "data$X");
    const int n = Xb.nrow();
    const int p = Xb.ncol();
    Eigen::MatrixXd X(n, p);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);

    Rcpp::LogicalMatrix Mb;
    const bool has_mask = !Rf_isNull(mask_arg);
    if (has_mask) {
      Mb = block_mask_matrix(mask_arg, b, n_blocks, "data$mask");
      if (Mb.nrow() != n || Mb.ncol() != p)
        Rcpp::stop("magmaan: data$mask block %d has shape %dx%d but data$X has %dx%d",
                   static_cast<int>(b + 1), Mb.nrow(), Mb.ncol(), n, p);
    }

    for (int r = 0; r < n; ++r) {
      for (int k = 0; k < p; ++k) {
        const double x = Xb(r, k);
        const bool observed = has_mask
            ? (Mb(r, k) != NA_LOGICAL && Mb(r, k) != 0)
            : std::isfinite(x);
        if (observed && !std::isfinite(x)) {
          Rcpp::stop("magmaan: data$mask marks a non-finite value as observed "
                     "in block %d, row %d", static_cast<int>(b + 1), r + 1);
        }
        M(r, k) = static_cast<std::uint8_t>(observed ? 1 : 0);
        X(r, k) = observed ? x : std::numeric_limits<double>::quiet_NaN();
        if (!observed) any_missing = true;
      }
    }
    raw.X.push_back(std::move(X));
    masks.push_back(std::move(M));
  }
  if (any_missing) raw.mask = std::move(masks);

  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack) stop_fit(pack.error());
  auto endpoint = magmaan::estimate::fiml::fiml_h1_moments(raw, *pack, fiml_h1_opts_from(control));
  if (!endpoint) stop_fit(endpoint.error());
  // Analytic curvature does not use the retained legacy FD step argument.
  if (!(h_step > 0.0)) Rcpp::stop("saturated_em_moments: h_step must be > 0");
  auto out_or = magmaan::estimate::fiml::saturated_em_moments(raw, *pack, *endpoint);
  if (!out_or.has_value()) stop_post(out_or.error());
  return saturated_moments_to_r(*out_or, true, &*endpoint);
}

// regularize_saturated_stage1_impl() — frontier ML2S Stage-1 conditioning.
// Regularizes the saturated EM covariance used as Stage-2 input and propagates
// the same transformation through Stage-1 ACOV. The raw input list is preserved
// by the R wrapper as `$stage1_raw`; this helper returns only the transformed
// Stage-1 list and diagnostics.
//
// [[Rcpp::export]]
Rcpp::List regularize_saturated_stage1_impl(Rcpp::List stage1,
                                            SEXP regularization = R_NilValue) {
  SaturatedMoments sm;
  if (!magmaanr::saturated_from_list(stage1, sm)) {
    Rcpp::stop("magmaan: regularize_saturated_stage1_impl needs a Stage-1 "
               "list with mean/cov/n_obs/acov");
  }
  auto opts = stage1_regularization_options_from(regularization);
  auto r_or = magmaan::estimate::fiml::regularize_saturated_stage1(sm, opts);
  if (!r_or.has_value()) stop_post(r_or.error());
  bool any_applied = false;
  for (const auto& d : r_or->block_diagnostics) any_applied = any_applied || d.applied;
  return Rcpp::List::create(
      Rcpp::_["stage1"] = saturated_moments_to_r(r_or->moments,
                                                 /*include_information=*/!any_applied),
      Rcpp::_["diagnostics"] =
          stage1_regularization_diagnostics_to_r(*r_or, opts));
}

// infer_fiml_observed_vcov() inverts the analytic observed FIML information
// for a fit carrying its retained raw-data/pack state.
//
// [[Rcpp::export]]
Rcpp::List infer_fiml_observed_vcov(Rcpp::List fit) {
  if (!fit.containsElementNamed("raw_data")) {
    Rcpp::stop("magmaan: infer_fiml_observed_vcov() requires a FIML fit with $raw_data");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  auto info_or = magmaan::estimate::fiml::fiml_observed_information(
      ctx.pt, ctx.rep, raw, est, pack);
  if (!info_or.has_value()) stop_post(info_or.error());
  auto vcov_or = magmaan::inference::vcov(*info_or, ctx.pt, est.theta);
  if (!vcov_or.has_value()) stop_post(vcov_or.error());
  return Rcpp::List::create(
      Rcpp::_["information"] = Rcpp::wrap(*info_or),
      Rcpp::_["vcov"] = Rcpp::wrap(*vcov_or),
      Rcpp::_["se"] = Rcpp::wrap(magmaan::inference::se(*vcov_or)));
}

// infer_fiml_information_vcov() compares the three FIML information
// conventions exposed by lavaan: expected Fisher, observed H1, and the full
// observed Hessian. Each convention is paired with both its model-based
// covariance and the same empirical-score sandwich covariance. The latter is
// V_model (Σ_i s_i s_i') V_model, which also preserves the equality-constraint
// tangent projection already folded into V_model.
//
// [[Rcpp::export]]
Rcpp::List infer_fiml_information_vcov(Rcpp::List fit) {
  if (!fit.containsElementNamed("raw_data")) {
    Rcpp::stop(
        "magmaan: infer_fiml_information_vcov() requires a FIML fit "
        "with $raw_data");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);

  auto scores_or = magmaan::estimate::fiml::fiml_casewise_deviance_scores(
      ctx.pt, ctx.rep, raw, pack, est);
  if (!scores_or.has_value()) stop_post(scores_or.error());
  Eigen::MatrixXd meat =
      0.25 * (scores_or->transpose() * (*scores_or));
  meat = 0.5 * (meat + meat.transpose()).eval();

  auto expected_or = magmaan::estimate::fiml::fiml_expected_information(
      ctx.pt, ctx.rep, raw, est, pack);
  if (!expected_or.has_value()) stop_post(expected_or.error());
  auto observed_h1_or =
      magmaan::estimate::fiml::fiml_observed_h1_information(
          ctx.pt, ctx.rep, raw, est, pack);
  if (!observed_h1_or.has_value()) stop_post(observed_h1_or.error());
  auto observed_hessian_or =
      magmaan::estimate::fiml::fiml_observed_information(
          ctx.pt, ctx.rep, raw, est, pack);
  if (!observed_hessian_or.has_value()) stop_post(observed_hessian_or.error());

  const auto summarize = [&](const Eigen::MatrixXd& information) {
    auto model_vcov_or =
        magmaan::inference::vcov(information, ctx.pt, est.theta);
    if (!model_vcov_or.has_value()) {
      const auto& error = model_vcov_or.error();
      return Rcpp::List::create(
          Rcpp::_["ok"] = false,
          Rcpp::_["error"] =
              std::string(post_error_kind(error.kind)) + ": " + error.detail,
          Rcpp::_["information"] = Rcpp::wrap(information));
    }
    Eigen::MatrixXd sandwich_vcov =
        (*model_vcov_or) * meat * (*model_vcov_or);
    sandwich_vcov =
        0.5 * (sandwich_vcov + sandwich_vcov.transpose()).eval();
    return Rcpp::List::create(
        Rcpp::_["ok"] = true,
        Rcpp::_["error"] = R_NilValue,
        Rcpp::_["information"] = Rcpp::wrap(information),
        Rcpp::_["vcov_model"] = Rcpp::wrap(*model_vcov_or),
        Rcpp::_["se_model"] =
            Rcpp::wrap(magmaan::inference::se(*model_vcov_or)),
        Rcpp::_["vcov_sandwich"] = Rcpp::wrap(sandwich_vcov),
        Rcpp::_["se_sandwich"] =
            Rcpp::wrap(magmaan::inference::se(sandwich_vcov)));
  };

  return Rcpp::List::create(
      Rcpp::_["expected"] = summarize(*expected_or),
      Rcpp::_["observed_h1"] = summarize(*observed_h1_or),
      Rcpp::_["observed_hessian"] = summarize(*observed_hessian_or),
      Rcpp::_["score_crossproducts"] = Rcpp::wrap(meat));
}

// estimate_fiml_robust_mlr() — mirrors estimate::fiml::fiml_robust_mlr().
// Computes observed-pattern sandwich SEs plus Yuan-Bentler/Mplus scaled-test
// traces for a FIML fit carrying $raw_data.
//
// [[Rcpp::export]]
Rcpp::List estimate_fiml_robust_mlr(Rcpp::List fit, double h_step = 1e-4) {
  if (!fit.containsElementNamed("raw_data")) {
    Rcpp::stop("magmaan: estimate_fiml_robust_mlr() requires a FIML fit with $raw_data");
  }
  if (!(h_step > 0.0)) {
    Rcpp::stop("magmaan: estimate_fiml_robust_mlr() requires h_step > 0");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);
  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  auto extras_or = magmaan::estimate::fiml::fiml_extras(
      ctx.pt, ctx.rep, raw, est, pack, h1);
  if (!extras_or.has_value()) stop_post(extras_or.error());
  auto r_or = magmaan::estimate::fiml::fiml_robust_mlr(
      ctx.pt, ctx.rep, raw, est, *df_or, extras_or->chi2, pack, h1);
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
      Rcpp::_["se"] = Rcpp::wrap(r_or->se),
      Rcpp::_["eigvals"] = Rcpp::wrap(r_or->eigvals),
      Rcpp::_["chisq_scaled"] = r_or->chisq_scaled,
      Rcpp::_["scaling_factor"] = r_or->scaling_factor,
      Rcpp::_["trace_ugamma"] = r_or->trace_ugamma,
      Rcpp::_["trace_ugamma_h1"] = r_or->trace_ugamma_h1,
      Rcpp::_["trace_ugamma_h0"] = r_or->trace_ugamma_h0,
      Rcpp::_["df"] = r_or->df,
      Rcpp::_["ntotal"] = static_cast<double>(r_or->ntotal),
      Rcpp::_["chisq"] = extras_or->chi2);
}

// fiml_fit_measures_impl() — FIML-specific standard and optional robust/scaled
// global fit measures. Standard measures use the FIML model-vs-saturated LRT
// and FIML independence baseline; robust = TRUE adds the corrected XX3/c.hat3
// user and baseline reductions used by lavaan-style robust CFI/RMSEA.
//
// [[Rcpp::export]]
Rcpp::List fiml_fit_measures_impl(Rcpp::List fit, bool robust = true) {
  if (!fit.containsElementNamed("raw_data")) {
    Rcpp::stop("magmaan: fiml_fit_measures_impl() requires a FIML fit with $raw_data");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);

  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  auto extras_or = magmaan::estimate::fiml::fiml_extras(
      ctx.pt, ctx.rep, raw, est, pack, h1);
  if (!extras_or.has_value()) stop_post(extras_or.error());
  auto baseline_or = magmaan::estimate::fiml::fiml_baseline_chi2(
      ctx.pt, raw, pack, h1);
  if (!baseline_or.has_value()) stop_post(baseline_or.error());

  const auto& fx = *extras_or;
  const auto& bl = *baseline_or;
  const magmaan::measures::FitMeasures fm =
      magmaan::measures::fit_measures(
          fx.chi2, *df_or, bl, pack.cache.n_total, pack.cache.block_p.size());

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["chisq"] = fx.chi2,
      Rcpp::_["df"] = *df_or,
      Rcpp::_["pvalue"] = magmaan::inference::chi2_pvalue(fx.chi2, *df_or),
      Rcpp::_["baseline.chisq"] = bl.chi2,
      Rcpp::_["baseline.df"] = bl.df,
      Rcpp::_["cfi"] = fm.cfi,
      Rcpp::_["tli"] = fm.tli,
      Rcpp::_["rmsea"] = fm.rmsea,
      Rcpp::_["rmsea.ci.lower"] = fm.rmsea_ci_lower,
      Rcpp::_["rmsea.ci.upper"] = fm.rmsea_ci_upper,
      Rcpp::_["rmsea.pvalue"] = fm.rmsea_pvalue,
      Rcpp::_["rmsea.close.h0"] = fm.rmsea_close_h0,
      Rcpp::_["rmsea.notclose.pvalue"] = fm.rmsea_notclose_pvalue,
      Rcpp::_["rmsea.notclose.h0"] = fm.rmsea_notclose_h0,
      Rcpp::_["srmr"] = fx.srmr,
      Rcpp::_["logl"] = fx.logl,
      Rcpp::_["unrestricted.logl"] = fx.unrestricted_logl,
      Rcpp::_["aic"] = fx.aic,
      Rcpp::_["bic"] = fx.bic,
      Rcpp::_["bic2"] = fx.bic2,
      Rcpp::_["npar"] = fx.npar,
      Rcpp::_["ntotal"] = static_cast<double>(fx.ntotal));

  if (robust && *df_or > 0) {
    std::unique_ptr<SaturatedMoments> owned_sm;
    const SaturatedMoments& sm =
        fiml_saturated_for_fit(fit, raw, pack, h1, owned_sm);
    auto r_or = magmaan::estimate::fiml::fiml_corrected_fit_measures(
        ctx.pt, ctx.rep, raw, est, *df_or, pack, h1, sm);
    if (!r_or.has_value()) stop_post(r_or.error());
    const auto& r = *r_or;
    const auto& rf = r.indices;
    out["XX3"] = r.xx3;
    out["df3"] = r.df3;
    out["c.hat3"] = r.c_hat3;
    out["XX3.scaled"] = r.xx3_scaled;
    out["baseline.XX3"] = r.xx3_null;
    out["baseline.df3"] = r.df3_null;
    out["baseline.c.hat3"] = r.c_hat3_null;
    out["baseline.XX3.scaled"] = r.xx3_null_scaled;
    out["chisq.scaled"] = rf.chisq_scaled;
    out["df.scaled"] = rf.df_scaled;
    out["pvalue.scaled"] = rf.pvalue_scaled;
    out["chisq.scaling.factor"] = rf.chisq_scaling_factor;
    out["baseline.chisq.scaled"] = rf.baseline_chisq_scaled;
    out["baseline.df.scaled"] = rf.baseline_df_scaled;
    out["baseline.pvalue.scaled"] = rf.baseline_pvalue_scaled;
    out["baseline.chisq.scaling.factor"] =
        rf.baseline_chisq_scaling_factor;
    out["cfi.scaled"] = rf.cfi_scaled;
    out["tli.scaled"] = rf.tli_scaled;
    out["cfi.robust"] = rf.cfi_robust;
    out["tli.robust"] = rf.tli_robust;
    out["rmsea.scaled"] = rf.rmsea_scaled;
    out["rmsea.ci.lower.scaled"] = rf.rmsea_ci_lower_scaled;
    out["rmsea.ci.upper.scaled"] = rf.rmsea_ci_upper_scaled;
    out["rmsea.pvalue.scaled"] = rf.rmsea_pvalue_scaled;
    out["rmsea.notclose.pvalue.scaled"] =
        rf.rmsea_notclose_pvalue_scaled;
    out["rmsea.robust"] = rf.rmsea_robust;
    out["rmsea.ci.lower.robust"] = rf.rmsea_ci_lower_robust;
    out["rmsea.ci.upper.robust"] = rf.rmsea_ci_upper_robust;
    out["rmsea.pvalue.robust"] = rf.rmsea_pvalue_robust;
    out["rmsea.notclose.pvalue.robust"] =
        rf.rmsea_notclose_pvalue_robust;
  }

  return out;
}

// infer_ml2s_casewise_influence_ij_fit() — per-case one-step misspecification-
// robust ("complete-sandwich") parameter influences for a two-stage (ML2S) fit:
// the missing-data member of the estimated-weight case-influence family, the
// casewise dual of estimate_two_stage_em_ml_inference()'s observed-bread vcov.
// Returns the N_total x n_free `influence` matrix (column-Gram = the ML2S IJ
// vcov) and its fixed-weight `naive` counterpart. Reuses the Stage-1 EM
// pack/H1 the fit carries. The NT Stage-2 weight (lavaan robust.two.stage)
// treats the weight as fixed, so its correction is zero (complete == naive);
// the non-NT weights (DWLS/ADF/DLS) carry the live data-dependent-weight term.
// Beyond lavaan/semfindr.
//
// [[Rcpp::export]]
Rcpp::List infer_ml2s_casewise_influence_ij_fit(
    Rcpp::List fit, SEXP raw_data, SEXP stage2_weight = R_NilValue,
    SEXP dls_a = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, raw_data);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);
  magmaan::estimate::fiml::TwoStageWeight kind;
  magmaan::estimate::fiml::TwoStageDlsOptions dls;
  ml2s_recorded_stage2(fit, stage2_weight, dls_a,
                       "infer_ml2s_casewise_influence_ij_fit()", kind, dls);
  auto r_or = magmaan::estimate::fiml::two_stage_casewise_influence_ij(
      ctx.pt, ctx.rep, raw, est, pack, h1, kind, dls);
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::Named("influence") = Rcpp::wrap(r_or->influence),
      Rcpp::Named("influence_naive") = Rcpp::wrap(r_or->influence_naive),
      Rcpp::Named("n_total") = static_cast<double>(r_or->n_total));
}

// estimate_two_stage_em_ml_inference() — mirrors
// estimate::fiml::two_stage_em_ml_inference(). Takes the Stage-2 ML fit on EM
// moments plus the original raw data used for Stage 1.
//
// [[Rcpp::export]]
Rcpp::List estimate_two_stage_em_ml_inference(Rcpp::List fit, SEXP raw_data,
                                              double h_step = 1e-4,
                                              std::string stage2_weight = "nt",
                                              double dls_a = 0.5) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, raw_data);
  // Reuse the Stage-1 saturated moments the fit already carries; only fall back
  // to a from-scratch EM (and its pack) when no usable $stage1 is present.
  SaturatedMoments sm;
  std::unique_ptr<SaturatedMoments> owned_sm;
  const SaturatedMoments* sm_ptr = &sm;
  if (!magmaanr::saturated_from_stage1(fit, sm)) {
    auto sm_or = magmaan::estimate::fiml::saturated_em_moments(raw, h_step);
    if (!sm_or.has_value()) stop_post(sm_or.error());
    owned_sm = std::make_unique<SaturatedMoments>(std::move(*sm_or));
    sm_ptr = owned_sm.get();
  }
  const auto kind = magmaanr::two_stage_weight_from_arg(stage2_weight);
  magmaan::estimate::fiml::TwoStageDlsOptions dls;
  dls.a = dls_a;
  auto r_or = magmaan::estimate::fiml::two_stage_em_ml_inference(
      ctx.pt, ctx.rep, est, *sm_ptr, kind, dls,
      magmaan::estimate::fiml::TwoStageBread::Expected);
  if (!r_or.has_value()) stop_post(r_or.error());
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
      Rcpp::_["se"] = Rcpp::wrap(r_or->se),
      Rcpp::_["eigvals"] = Rcpp::wrap(r_or->eigvals),
      Rcpp::_["chisq"] = r_or->chisq,
      Rcpp::_["chisq_scaled"] = r_or->chisq_scaled,
      Rcpp::_["scaling_factor"] = r_or->scaling_factor,
      Rcpp::_["trace_ugamma"] = r_or->trace_ugamma,
      Rcpp::_["df"] = r_or->df,
      Rcpp::_["ntotal"] = static_cast<double>(r_or->ntotal));
  out["chisq.scaled"] = r_or->chisq_scaled;
  out["chisq.scaling.factor"] = r_or->scaling_factor;

  if (r_or->df > 0) {
    auto fm_or = magmaan::estimate::fiml::two_stage_fit_measures(
        ctx.pt, *sm_ptr, *r_or, kind, dls);
    if (!fm_or.has_value()) stop_post(fm_or.error());
    const auto& bl = fm_or->baseline;
    const auto& fm = fm_or->indices;
    out["baseline_chisq"] = bl.chi2;
    out["baseline.chisq"] = bl.chi2;
    out["baseline_df"] = bl.df;
    out["baseline.df"] = bl.df;
    out["baseline_chisq_scaled"] = fm.baseline_chisq_scaled;
    out["baseline.chisq.scaled"] = fm.baseline_chisq_scaled;
    out["baseline_scaling_factor"] = fm.baseline_chisq_scaling_factor;
    out["baseline.chisq.scaling.factor"] = fm.baseline_chisq_scaling_factor;
    out["baseline_pvalue_scaled"] = fm.baseline_pvalue_scaled;
    out["baseline.pvalue.scaled"] = fm.baseline_pvalue_scaled;
    out["cfi_scaled"] = fm.cfi_scaled;
    out["cfi.scaled"] = fm.cfi_scaled;
    out["tli_scaled"] = fm.tli_scaled;
    out["tli.scaled"] = fm.tli_scaled;
    out["cfi_robust"] = fm.cfi_robust;
    out["cfi.robust"] = fm.cfi_robust;
    out["tli_robust"] = fm.tli_robust;
    out["tli.robust"] = fm.tli_robust;
    out["rmsea_scaled"] = fm.rmsea_scaled;
    out["rmsea.scaled"] = fm.rmsea_scaled;
    out["rmsea_ci_lower_scaled"] = fm.rmsea_ci_lower_scaled;
    out["rmsea.ci.lower.scaled"] = fm.rmsea_ci_lower_scaled;
    out["rmsea_ci_upper_scaled"] = fm.rmsea_ci_upper_scaled;
    out["rmsea.ci.upper.scaled"] = fm.rmsea_ci_upper_scaled;
    out["rmsea_pvalue_scaled"] = fm.rmsea_pvalue_scaled;
    out["rmsea.pvalue.scaled"] = fm.rmsea_pvalue_scaled;
    out["rmsea_notclose_pvalue_scaled"] = fm.rmsea_notclose_pvalue_scaled;
    out["rmsea.notclose.pvalue.scaled"] = fm.rmsea_notclose_pvalue_scaled;
    out["rmsea_robust"] = fm.rmsea_robust;
    out["rmsea.robust"] = fm.rmsea_robust;
    out["rmsea_ci_lower_robust"] = fm.rmsea_ci_lower_robust;
    out["rmsea.ci.lower.robust"] = fm.rmsea_ci_lower_robust;
    out["rmsea_ci_upper_robust"] = fm.rmsea_ci_upper_robust;
    out["rmsea.ci.upper.robust"] = fm.rmsea_ci_upper_robust;
    out["rmsea_pvalue_robust"] = fm.rmsea_pvalue_robust;
    out["rmsea.pvalue.robust"] = fm.rmsea_pvalue_robust;
    out["rmsea_notclose_pvalue_robust"] = fm.rmsea_notclose_pvalue_robust;
    out["rmsea.notclose.pvalue.robust"] = fm.rmsea_notclose_pvalue_robust;
  }
  return out;
}

// Equation-level frontier diagnostic for identifying the Stage-1 and Stage-2
// information conventions used by historical ML2S scaled tests.
// [[Rcpp::export]]
Rcpp::List frontier_ml2s_information_choices_impl(Rcpp::List fit,
                                                   SEXP raw_data = R_NilValue,
                                                   double eigen_tol = 1e-9) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  SaturatedMoments sm;
  if (!magmaanr::saturated_from_stage1(fit, sm)) {
    Rcpp::stop("frontier_ml2s_information_choices(): fit must carry a usable "
               "$stage1 saturated-moment object");
  }
  SEXP rd = raw_data;
  if (Rf_isNull(rd) && fit.containsElementNamed("raw_data")) {
    rd = fit["raw_data"];
  }
  if (Rf_isNull(rd)) {
    Rcpp::stop("frontier_ml2s_information_choices(): raw_data is required, "
               "either explicitly or in fit$raw_data");
  }
  const magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, rd);
  auto pack_or = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack_or.has_value()) stop_fit(pack_or.error());
  auto r_or = magmaan::estimate::fiml::frontier::two_stage_information_choices(
      ctx.pt, ctx.rep, raw, est, sm, *pack_or, eigen_tol);
  if (!r_or.has_value()) stop_post(r_or.error());

  const R_xlen_t n = static_cast<R_xlen_t>(r_or->choices.size());
  Rcpp::CharacterVector name(n), stage1_name(n), stage1_bread_point(n),
      stage1_bread_kind(n), stage1_meat_point(n), stage2_name(n);
  Rcpp::NumericVector trace(n), scale(n), chisq_scaled(n), min_stage1_h(n),
      min_h(n), min_u(n);
  Rcpp::IntegerVector neg_stage1_h(n), neg_h(n), neg_u(n), rank_u(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& row = r_or->choices[static_cast<std::size_t>(i)];
    name[i] = row.name;
    stage1_name[i] = row.stage1_information;
    stage1_bread_point[i] = row.stage1_bread_point;
    stage1_bread_kind[i] = row.stage1_bread_kind;
    stage1_meat_point[i] = row.stage1_meat_point;
    stage2_name[i] = row.stage2_information;
    trace[i] = row.trace_ugamma;
    scale[i] = row.scaling_factor;
    chisq_scaled[i] = row.chisq_scaled;
    min_stage1_h[i] = row.min_stage1_information_eigenvalue;
    min_h[i] = row.min_information_eigenvalue;
    min_u[i] = row.min_projector_eigenvalue;
    neg_stage1_h[i] =
        static_cast<int>(row.stage1_information_negative_eigenvalues);
    neg_h[i] = static_cast<int>(row.information_negative_eigenvalues);
    neg_u[i] = static_cast<int>(row.projector_negative_eigenvalues);
    rank_u[i] = static_cast<int>(row.projector_rank);
  }
  return Rcpp::List::create(
      Rcpp::_["choices"] = Rcpp::DataFrame::create(
          Rcpp::_["information"] = name,
          Rcpp::_["stage1_information"] = stage1_name,
          Rcpp::_["stage1_bread_point"] = stage1_bread_point,
          Rcpp::_["stage1_bread_kind"] = stage1_bread_kind,
          Rcpp::_["stage1_meat_point"] = stage1_meat_point,
          Rcpp::_["stage2_information"] = stage2_name,
          Rcpp::_["trace_ugamma"] = trace,
          Rcpp::_["scaling_factor"] = scale,
          Rcpp::_["chisq_scaled"] = chisq_scaled,
          Rcpp::_["min_stage1_information_eigenvalue"] = min_stage1_h,
          Rcpp::_["min_information_eigenvalue"] = min_h,
          Rcpp::_["min_projector_eigenvalue"] = min_u,
          Rcpp::_["stage1_information_negative_eigenvalues"] =
              neg_stage1_h,
          Rcpp::_["information_negative_eigenvalues"] = neg_h,
          Rcpp::_["projector_negative_eigenvalues"] = neg_u,
          Rcpp::_["projector_rank"] = rank_u),
      Rcpp::_["chisq"] = r_or->chisq,
      Rcpp::_["df"] = r_or->df,
      Rcpp::_["delta_rank"] = static_cast<int>(r_or->delta_rank),
      Rcpp::_["saturated_expected_observed_max_abs"] =
          r_or->saturated_expected_observed_max_abs,
      Rcpp::_["stage1_expected_observed_max_abs"] =
          r_or->stage1_expected_observed_max_abs);
}

// Equation-37 FIML diagnostic crossing the canonical residual-information and
// saturated-moment sandwich choices.
// [[Rcpp::export]]
Rcpp::List frontier_fiml_information_choices_impl(Rcpp::List fit,
                                                  SEXP raw_data = R_NilValue,
                                                  double eigen_tol = 1e-9) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  SEXP rd = raw_data;
  if (Rf_isNull(rd) && fit.containsElementNamed("raw_data")) {
    rd = fit["raw_data"];
  }
  if (Rf_isNull(rd)) {
    Rcpp::stop("frontier_fiml_information_choices(): raw_data is required, "
               "either explicitly or in fit$raw_data");
  }
  const magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, rd);
  std::unique_ptr<FimlPack> owned_pack;
  std::unique_ptr<FimlH1> owned_h1;
  std::unique_ptr<SaturatedMoments> owned_saturated;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);
  const SaturatedMoments& sm =
      fiml_saturated_for_fit(fit, raw, pack, h1, owned_saturated);
  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  auto extras_or = magmaan::estimate::fiml::fiml_extras(
      ctx.pt, ctx.rep, raw, est, pack, h1);
  if (!extras_or.has_value()) stop_post(extras_or.error());
  auto r_or = magmaan::estimate::fiml::frontier::fiml_information_choices(
      ctx.pt, ctx.rep, raw, est, sm, pack,
      extras_or->chi2, *df_or, eigen_tol);
  if (!r_or.has_value()) stop_post(r_or.error());

  const R_xlen_t n = static_cast<R_xlen_t>(r_or->choices.size());
  Rcpp::CharacterVector name(n), residual(n), bread_point(n), bread_kind(n),
      meat_point(n);
  Rcpp::NumericVector trace(n), scale(n), chisq_scaled(n), min_u(n),
      min_bread(n);
  Rcpp::IntegerVector neg_u(n), neg_bread(n), rank_u(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& row = r_or->choices[static_cast<std::size_t>(i)];
    name[i] = row.name;
    residual[i] = row.residual_information;
    bread_point[i] = row.omega_bread_point;
    bread_kind[i] = row.omega_bread_kind;
    meat_point[i] = row.omega_meat_point;
    trace[i] = row.trace_ugamma;
    scale[i] = row.scaling_factor;
    chisq_scaled[i] = row.chisq_scaled;
    min_u[i] = row.min_residual_information_eigenvalue;
    min_bread[i] = row.min_omega_bread_eigenvalue;
    neg_u[i] =
        static_cast<int>(row.residual_information_negative_eigenvalues);
    neg_bread[i] = static_cast<int>(row.omega_bread_negative_eigenvalues);
    rank_u[i] = static_cast<int>(row.residual_rank);
  }
  return Rcpp::List::create(
      Rcpp::_["choices"] = Rcpp::DataFrame::create(
          Rcpp::_["information"] = name,
          Rcpp::_["residual_information"] = residual,
          Rcpp::_["omega_bread_point"] = bread_point,
          Rcpp::_["omega_bread_kind"] = bread_kind,
          Rcpp::_["omega_meat_point"] = meat_point,
          Rcpp::_["trace_ugamma"] = trace,
          Rcpp::_["scaling_factor"] = scale,
          Rcpp::_["chisq_scaled"] = chisq_scaled,
          Rcpp::_["min_residual_information_eigenvalue"] = min_u,
          Rcpp::_["min_omega_bread_eigenvalue"] = min_bread,
          Rcpp::_["residual_information_negative_eigenvalues"] = neg_u,
          Rcpp::_["omega_bread_negative_eigenvalues"] = neg_bread,
          Rcpp::_["residual_rank"] = rank_u),
      Rcpp::_["chisq"] = r_or->chisq,
      Rcpp::_["df"] = r_or->df,
      Rcpp::_["delta_rank"] = static_cast<int>(r_or->delta_rank));
}

// two_stage_stage2_weight_blocks_impl() — mirrors
// estimate::fiml::two_stage_stage2_weight_blocks(). Builds the per-block Stage-2
// weight (Nt / Dwls / Adf / Dls) from a Stage-1 saturated-moments list, returned
// as a list of per-block matrices suitable for `fit_wls(..., W = .)`.
//
// [[Rcpp::export]]
Rcpp::List two_stage_stage2_weight_blocks_impl(Rcpp::List stage1,
                                               std::string stage2_weight = "nt",
                                               double dls_a = 0.5) {
  SaturatedMoments sm;
  if (!magmaanr::saturated_from_list(stage1, sm)) {
    Rcpp::stop("magmaan: two_stage_stage2_weight_blocks needs a Stage-1 list "
               "with mean/cov/n_obs/acov");
  }
  const auto kind = magmaanr::two_stage_weight_from_arg(stage2_weight);
  magmaan::estimate::fiml::TwoStageDlsOptions dls;
  dls.a = dls_a;
  auto w_or = magmaan::estimate::fiml::two_stage_stage2_weight_blocks(
      sm, kind, dls);
  if (!w_or.has_value()) stop_post(w_or.error());
  Rcpp::List out(static_cast<R_xlen_t>(w_or->size()));
  for (std::size_t b = 0; b < w_or->size(); ++b) {
    out[static_cast<R_xlen_t>(b)] = Rcpp::wrap((*w_or)[b]);
  }
  out.attr("cancels_measurement_units") =
      kind == magmaan::estimate::fiml::TwoStageWeight::Nt ||
      kind == magmaan::estimate::fiml::TwoStageWeight::Dls;
  return out;
}

// infer_fiml_fmg_spectrum() — mirrors estimate::fiml::fiml_ugamma_spectrum().
// First-principles missing-data UΓ spectrum for FMG goodness-of-fit tests: the
// df nonzero eigenvalues of U·Γ_mis built from the saturated H1 information and
// the saturated-moment ACOV. Biased gamma only (the Du-Bentler unbiased gamma
// is undefined under FIML — no $unbiased key), ML/LRT base statistic.
//
// [[Rcpp::export]]
Rcpp::List infer_fiml_fmg_spectrum(Rcpp::List fit, double h_step = 1e-4) {
  if (!fit.containsElementNamed("raw_data")) {
    Rcpp::stop("magmaan: infer_fiml_fmg_spectrum() requires a FIML fit with $raw_data");
  }
  if (!(h_step > 0.0)) {
    Rcpp::stop("magmaan: infer_fiml_fmg_spectrum() requires h_step > 0");
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1& h1 = fiml_h1_for_fit(fit, raw, pack, owned_h1);
  std::unique_ptr<SaturatedMoments> owned_sm;
  const SaturatedMoments& sm = fiml_saturated_for_fit(fit, raw, pack, h1, owned_sm);
  auto df_or = magmaan::inference::df_stat(ctx.pt, ctx.samp, est.theta);
  if (!df_or.has_value()) stop_post(df_or.error());
  auto extras_or = magmaan::estimate::fiml::fiml_extras(
      ctx.pt, ctx.rep, raw, est, pack, h1);
  if (!extras_or.has_value()) stop_post(extras_or.error());
  auto sp_or = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      ctx.pt, ctx.rep, raw, est, *df_or, extras_or->chi2, pack, h1, sm);
  if (!sp_or.has_value()) stop_post(sp_or.error());
  return Rcpp::List::create(
      Rcpp::_["biased"] = Rcpp::wrap(sp_or->eigvals),
      Rcpp::_["chi2_lrt"] = sp_or->chi2_lrt,
      Rcpp::_["df"] = sp_or->df,
      Rcpp::_["trace_xcheck"] = sp_or->trace_xcheck);
}

// Owning pattern artifacts permit checking the observed-data objective without
// borrowing fit-time external pointers; serialized fits can rebuild the pack.
// [[Rcpp::export]]
Rcpp::List frontier_fiml_newton_audit_impl(
    Rcpp::List fit, Rcpp::Nullable<Rcpp::NumericVector> theta = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const auto est = est_from_fit(fit);
  if (!fit.containsElementNamed("raw_data") ||
      Rcpp::as<std::string>(fit["estimator"]) != "FIML")
    Rcpp::stop("requires a direct FIML fit with raw_data");
  auto raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack) stop_fit(pack.error());
  const Eigen::VectorXd point = theta.isNull() ? est.theta
      : Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(theta.get()));
  auto audit = magmaan::estimate::frontier::audit_newton_fiml(
      ctx.pt, ctx.rep, raw, *pack, point);
  if (!audit) stop_fit(audit.error());
  Rcpp::List patterns(pack->cache.patterns.size());
  for (std::size_t i = 0; i < pack->cache.patterns.size(); ++i) {
    const auto& p = pack->cache.patterns[i];
    Rcpp::IntegerVector observed(p.observed.size());
    for (std::size_t j = 0; j < p.observed.size(); ++j)
      observed[j] = static_cast<int>(p.observed[j] + 1);
    patterns[i] = Rcpp::List::create(
        Rcpp::_["block"] = static_cast<int>(p.block + 1),
        Rcpp::_["observed"] = observed,
        Rcpp::_["n_obs"] = static_cast<double>(p.n_obs),
        Rcpp::_["mean"] = Rcpp::wrap(p.mean),
        Rcpp::_["cov"] = Rcpp::wrap(p.cov));
  }
  const auto& a = *audit;
  magmaan::estimate::Estimates at; at.theta = point;
  const auto native = magmaan::compat::lavaan::to_lavaan_partable(
      ctx.pt, ctx.names, magmaan::spec::Starts{});
  // This complete-moment certificate entry point rejects the FIML objective
  // kind explicitly; pairwise start statistics never stand in for its data.
  const auto source = magmaan::estimate::frontier::newton_input_error_bounds(
      ctx.pt, ctx.rep, pack->start_stats, point, a, magmaan::estimate::Estimator::ML);
  return Rcpp::List::create(
      Rcpp::_["diagnostics"] = newton_accuracy_to_r(a.diagnostics),
      Rcpp::_["objective"] = a.derivatives.objective,
      Rcpp::_["n_obs"] = a.derivatives.n_obs,
      Rcpp::_["gradient"] = Rcpp::wrap(a.derivatives.gradient),
      Rcpp::_["hessian"] = Rcpp::wrap(a.derivatives.hessian),
      Rcpp::_["derivative_basis"] = Rcpp::wrap(
          (a.geometry.equality_basis * a.geometry.tangent_basis).eval()),
      Rcpp::_["theta"] = Rcpp::wrap(point),
      Rcpp::_["partable"] = partable_df_from_lavaan(native, &at),
      Rcpp::_["patterns"] = patterns,
      Rcpp::_["construction_status"] = std::string(magmaan::estimate::to_string(source.status)),
      Rcpp::_["construction_detail"] = source.detail);
}

// Original two-stage objectives, with historical evidence kept distinct from
// a deliberately supplied Stage-1 point. No EM or optimizer is run here.
// [[Rcpp::export]]
Rcpp::List frontier_ml2s_convergence_audit_impl(
    Rcpp::List fit, Rcpp::Nullable<Rcpp::NumericVector> theta = R_NilValue,
    Rcpp::Nullable<Rcpp::List> stage1_point = R_NilValue) {
  namespace af = magmaan::estimate::frontier;
  namespace ff = magmaan::estimate::fiml;
  if (!is_ml2s_estimator_label(Rcpp::as<std::string>(fit["estimator"])) ||
      !fit.containsElementNamed("raw_data") || !fit.containsElementNamed("stage1"))
    Rcpp::stop("requires an ordinary ML2S fit with raw_data and stage1");
  if (fit.containsElementNamed("covariance_policy") &&
      Rcpp::as<std::string>(fit["covariance_policy"]) != "unrestricted")
    Rcpp::stop("ML2S composition audit supports ordinary Stage 2 only");
  Ctx ctx = ctx_from_fit(fit);
  auto raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
  auto pack = ff::fiml_pack(raw);
  if (!pack) stop_fit(pack.error());
  Rcpp::List source = fit.containsElementNamed("stage1_raw")
      ? Rcpp::List(fit["stage1_raw"]) : Rcpp::List(fit["stage1"]);
  ff::SaturatedMoments sm;
  if (!saturated_audit_from_list(source, sm)) Rcpp::stop("Stage-1 moments/ACOV missing");
  ff::FIMLH1 endpoint;
  endpoint.mu = sm.mean; endpoint.sigma = sm.cov;
  endpoint.solver_recorded = sm.solver_recorded;
  endpoint.solver_options = sm.solver_options;
  endpoint.solver_blocks = sm.solver_blocks;
  bool value_recorded = source.containsElementNamed("endpoint_value") && stage1_point.isNull();
  if (stage1_point.isNotNull()) {
    ff::SaturatedMoments at;
    if (!magmaanr::saturated_target_from_list(Rcpp::List(stage1_point.get()), at))
      Rcpp::stop("Stage-1 point needs mean/cov/n_obs");
    endpoint.mu = at.mean; endpoint.sigma = at.cov;
    endpoint.solver_recorded = false; endpoint.solver_blocks.clear();
    auto evaluated = ff::saturated_em_moments(raw, *pack, endpoint);
    if (!evaluated) stop_post(evaluated.error());
    sm = std::move(*evaluated);
  }
  if (value_recorded) endpoint.value = Rcpp::as<double>(source["endpoint_value"]);
  else {
    magmaan::model::ImpliedMoments m; m.mu = endpoint.mu; m.sigma = endpoint.sigma;
    auto v = ff::FIML{}.value(raw, pack->cache, m);
    if (!v) stop_fit(v.error());
    endpoint.value = *v;
  }
  const auto kind = magmaanr::two_stage_weight_from_arg(Rcpp::as<std::string>(fit["stage2_weight"]));
  const auto dls = ml2s_dls_options_from_fit(fit);
  af::Ml2sAuditOptions opts;
  std::optional<af::Ml2sStage2Input> recorded;
  Rcpp::List snapshot;
  if (fit.containsElementNamed("stage2_input") && !Rf_isNull(fit["stage2_input"])) {
    snapshot = Rcpp::List(fit["stage2_input"]);
    if (Rcpp::as<std::string>(snapshot["covariance_policy"]) != "ordinary")
      Rcpp::stop("ML2S composition audit supports ordinary Stage 2 only");
    af::Ml2sStage2Input input;
    input.kind = magmaanr::two_stage_weight_from_arg(Rcpp::as<std::string>(snapshot["stage2_weight"]));
    input.dls.a = Rcpp::as<double>(snapshot["dls_a"]);
    Rcpp::List captured(snapshot["moments"]);
    const bool has_acov = magmaanr::saturated_from_list(captured, input.moments);
    if (!has_acov && !magmaanr::saturated_target_from_list(captured, input.moments))
      Rcpp::stop("invalid Stage-2 input snapshot");
    opts.transformation = stage1_regularization_options_from(snapshot["transformation"]);
    if (snapshot.containsElementNamed("bounds") && !Rf_isNull(snapshot["bounds"]))
      opts.stage2.bounds = bounds_from_nullable(Rcpp::List(snapshot["bounds"]));
    if (has_acov || input.kind == ff::TwoStageWeight::Nt || input.kind == ff::TwoStageWeight::Uls)
      recorded = std::move(input);
  } else if (fit.containsElementNamed("stage1_raw")) {
    Rcpp::stop("transformed legacy fit lacks its transformation/input record");
  }
  const auto est = est_from_fit(fit);
  const Eigen::VectorXd point = theta.isNull() ? est.theta
      : Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(theta.get()));
  auto r = af::audit_ml2s(ctx.pt, ctx.rep, raw, *pack, endpoint, point, kind, dls, opts,
      recorded, theta.isNull() ? std::optional<double>(est.fmin) : std::nullopt, &sm);
  if (!r) stop_fit(r.error());
  if (!value_recorded) r->stage1.evidence.objective.reported_available = false;
  // Also verify the captured caller-unit sample and actual supplied W, beyond
  // the core recipe/moment identity. A missing record cannot be reconstructed
  // from today's producer and called historical evidence.
  auto equal = [](const auto& a, const auto& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() && (a.array() == b.array()).all();
  };
  if (recorded) {
    bool same = ctx.samp.n_obs == recorded->moments.n_obs &&
        ctx.samp.S.size() == recorded->moments.cov.size() && ctx.samp.mean.size() == recorded->moments.mean.size();
    for (std::size_t b = 0; same && b < ctx.samp.S.size(); ++b)
      same = equal(ctx.samp.S[b], recorded->moments.cov[b]) && equal(ctx.samp.mean[b], recorded->moments.mean[b]);
    if (kind != ff::TwoStageWeight::Nt) {
      if (!snapshot.containsElementNamed("weight_blocks") || Rf_isNull(snapshot["weight_blocks"]))
        r->handoff = {magmaan::estimate::FitCheck::Unchecked, true, "actual Stage-2 weight record missing"};
      else {
        const auto& weight_source = recorded->kind == ff::TwoStageWeight::Uls
            ? r->stage2_input.moments : recorded->moments;
        auto w = ff::two_stage_stage2_weight_blocks(weight_source, recorded->kind, recorded->dls);
        if (!w) stop_post(w.error());
        Rcpp::List supplied(snapshot["weight_blocks"]);
        same = same && supplied.size() == static_cast<R_xlen_t>(w->size());
        for (R_xlen_t b = 0; same && b < supplied.size(); ++b)
          same = equal(Rcpp::as<Eigen::MatrixXd>(supplied[b]), (*w)[static_cast<std::size_t>(b)]);
      }
    }
    if (!same) r->handoff = {magmaan::estimate::FitCheck::Failed, true, "captured sample or supplied weight differs from fit/recipe"};
  }
  af::Ml2sConvergencePolicy policy;
  policy.stage1 = af::newton_convergence_policy(); policy.stage2 = af::newton_convergence_policy();
  policy.stage1.require_objective_consistency = true;
  policy.stage2.require_objective_consistency = theta.isNull();
  policy.require_solver_stop = true;
  const auto assessment = af::assess_convergence(*r, policy);
  auto check = [](const af::ConvergenceCheck& x) {
    return Rcpp::List::create(Rcpp::_["status"] = fit_check_to_r(x.status), Rcpp::_["reason"] = x.reason);
  };
  const auto& a = r->stage2.computations;
  magmaan::estimate::Estimates at; at.theta = point;
  const auto pt = magmaan::compat::lavaan::to_lavaan_partable(ctx.pt, ctx.names, magmaan::spec::Starts{});
  return Rcpp::List::create(
      Rcpp::_["status"] = fit_check_to_r(assessment.status),
      Rcpp::_["stage1_assessment"] = verified_assessment_to_r(assessment.stage1),
      Rcpp::_["handoff"] = check(assessment.handoff), Rcpp::_["solver_stop"] = check(assessment.solver_stop),
      Rcpp::_["stage2_assessment"] = verified_assessment_to_r(assessment.stage2),
      Rcpp::_["source"] = saturated_moments_to_r(r->source),
      Rcpp::_["stage2_input"] = saturated_moments_to_r(r->stage2_input.moments),
      Rcpp::_["stage1"] = Rcpp::List::create(
          Rcpp::_["value_recorded"] = value_recorded,
          Rcpp::_["objective"] = r->stage1.derivatives.objective,
          Rcpp::_["theta"] = Rcpp::wrap(r->stage1.derivatives.theta),
          Rcpp::_["gradient"] = Rcpp::wrap(r->stage1.derivatives.gradient),
          Rcpp::_["hessian"] = Rcpp::wrap(r->stage1.derivatives.hessian),
          Rcpp::_["diagnostics"] = newton_accuracy_to_r(r->stage1.evidence.newton_accuracy)),
      Rcpp::_["stage2"] = Rcpp::List::create(
          Rcpp::_["objective"] = a.derivatives.objective,
          Rcpp::_["theta"] = Rcpp::wrap(point),
          Rcpp::_["gradient"] = Rcpp::wrap(a.derivatives.gradient),
          Rcpp::_["hessian"] = Rcpp::wrap(a.derivatives.hessian),
          Rcpp::_["metric_factor"] = Rcpp::wrap(a.derivatives.metric_factor),
          Rcpp::_["derivative_basis"] = Rcpp::wrap((a.geometry.equality_basis * a.geometry.tangent_basis).eval()),
          Rcpp::_["retained_ls_weights"] = retained_ls_weights_to_r(a.derivatives),
          Rcpp::_["diagnostics"] = newton_accuracy_to_r(a.diagnostics)),
      Rcpp::_["partable"] = partable_df_from_lavaan(pt, &at));
}
