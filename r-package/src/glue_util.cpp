#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

namespace magmaanr::fitglue {

magmaan::estimate::FittingOptions fitting_options_from(const Rcpp::List& value) {
  check_optim_control_names(value, {"preset", "starts", "optimizer", "convergence", "marker"});
  magmaan::estimate::FittingOptions out;
  auto read = [&](const char* name, std::optional<std::string>& target) {
    if (!value.containsElementNamed(name)) return;
    SEXP x = value[name];
    if (TYPEOF(x) != STRSXP || Rf_length(x) != 1 || STRING_ELT(x, 0) == NA_STRING)
      Rcpp::stop("options$%s must be one nonmissing string", name);
    target = Rcpp::as<std::string>(x);
  };
  read("preset", out.preset); read("starts", out.starts);
  read("optimizer", out.optimizer); read("convergence", out.convergence); read("marker", out.marker);
  return out;
}

Rcpp::List fitting_report_to_r(const magmaan::estimate::FittingReport& report) {
  const auto& setup = report.setup;
  Rcpp::List request = Rcpp::List::create();
  auto add = [&](const char* name, const std::optional<std::string>& value) {
    if (value) request[name] = *value;
  };
  add("preset", setup.requested.preset); add("starts", setup.requested.starts);
  add("optimizer", setup.requested.optimizer); add("convergence", setup.requested.convergence); add("marker", setup.requested.marker);
  Rcpp::List attempts(report.attempts.size());
  for (std::size_t j = 0; j < report.attempts.size(); ++j) {
    const auto& a = report.attempts[j];
    const auto& c = a.controls;
    Rcpp::List port = Rcpp::List::create();
    auto real = [&](const char* name, const std::optional<double>& v) { if (v) port[name] = *v; };
    real("rel_f_tol", c.port.rel_f_tol); real("abs_f_tol", c.port.abs_f_tol);
    real("x_tol", c.port.x_tol); real("false_conv_tol", c.port.false_conv_tol);
    real("step_min", c.port.step_min); real("step_max", c.port.step_max);
    if (c.port.max_eval) port["max_eval"] = *c.port.max_eval;
    if (c.port.max_iter) port["max_iter"] = *c.port.max_iter;
    attempts[j] = Rcpp::List::create(
        Rcpp::_["start"] = Rcpp::wrap(a.start),
        Rcpp::_["optimizer_start"] = Rcpp::wrap(a.optimizer_start),
        Rcpp::_["parameter_scale"] = Rcpp::wrap(a.parameter_scale),
        Rcpp::_["port_scale"] = Rcpp::wrap(a.port_scale),
        Rcpp::_["simple_start"] = a.simple_start,
        Rcpp::_["standardized"] = a.standardized,
        Rcpp::_["accepted"] = a.accepted,
        Rcpp::_["raw_status"] = a.raw_status,
        Rcpp::_["iterations"] = a.iterations,
        Rcpp::_["fmin"] = a.fmin,
        Rcpp::_["gradient_max"] = a.gradient_max,
        Rcpp::_["error"] = a.error,
        Rcpp::_["controls"] = Rcpp::List::create(
            Rcpp::_["max_iter"] = c.max_iter, Rcpp::_["ftol"] = c.ftol,
            Rcpp::_["gtol"] = c.gtol, Rcpp::_["port"] = port,
            Rcpp::_["normalize_sample"] = c.normalize_sample,
            Rcpp::_["coordinate_scaling"] = magmaan::estimate::coordinate_scaling_name(c.coordinate_scaling),
            Rcpp::_["center_locations"] = c.center_locations));
  }
  return Rcpp::List::create(
      Rcpp::_["marker_switch"] = Rcpp::DataFrame::create(
          Rcpp::_["lv"] = Rcpp::CharacterVector(), Rcpp::_["old"] = Rcpp::CharacterVector(),
          Rcpp::_["new"] = Rcpp::CharacterVector(), Rcpp::_["r_old"] = Rcpp::NumericVector(),
          Rcpp::_["r_new"] = Rcpp::NumericVector(), Rcpp::_["reverted"] = Rcpp::LogicalVector()),
      Rcpp::_["requested"] = request,
      Rcpp::_["effective"] = Rcpp::List::create(
          Rcpp::_["starts"] = setup.starts, Rcpp::_["optimizer"] = setup.optimizer,
          Rcpp::_["convergence"] = setup.convergence, Rcpp::_["marker"] = setup.marker),
      Rcpp::_["modified_preset"] = setup.modified_preset,
      Rcpp::_["selected_attempt"] = report.attempts.empty() ? NA_INTEGER : static_cast<int>(report.selected_attempt + 1),
      Rcpp::_["attempts"] = attempts);
}

std::string start_name_from_arg(Rcpp::Nullable<Rcpp::String> start,
                                const char* caller,
                                const char* default_name) {
  if (start.isNull()) return default_name;
  std::string name = Rcpp::as<std::string>(start.get());
  for (char& ch : name) {
    if (ch == '_') ch = '-';
    else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (name.empty() || name == "default") return default_name;
  if (name == "lavaan" || name == "fabin") {
    return "fabin3";
  }
  if (name == "scaled-fabin" || name == "simple" || name == "fabin2" || name == "fabin3" ||
      name == "layered" ||
      name == "guttman" || name == "guttman1952" ||
      name == "bentler" || name == "bentler1982" ||
      name == "jamesstein" || name == "james-stein" || name == "js") {
    return name;
  }
  Rcpp::stop("magmaan: %s(): unsupported start '%s' "
             "(accepted: default, scaled-fabin, simple, fabin2, fabin3, layered, guttman, bentler1982, jamesstein)",
             caller, name.c_str());
  return default_name;
}

magmaan::estimate::StartMethod start_method(const std::string& name) {
  using M = magmaan::estimate::StartMethod;
  if (name == "simple") return M::Simple;
  if (name == "fabin2") return M::Fabin2;
  if (name == "guttman" || name == "guttman1952") return M::Guttman;
  if (name == "bentler" || name == "bentler1982") return M::Bentler1982;
  if (name == "jamesstein" || name == "james-stein" || name == "js") return M::JamesStein;
  if (name == "layered") return M::Layered;
  return M::Fabin3;
}

const char* start_method_name(magmaan::estimate::StartMethod method) {
  using M = magmaan::estimate::StartMethod;
  switch (method) {
    case M::Simple: return "simple";
    case M::Fabin2: return "fabin2";
    case M::Fabin3: return "fabin3";
    case M::Guttman: return "guttman";
    case M::Bentler1982: return "bentler1982";
    case M::JamesStein: return "jamesstein";
    case M::Layered: return "layered";
  }
  return "unknown";
}

magmaan::estimate::StartTransport start_transport_from_arg(const std::string& mode) {
  using T = magmaan::estimate::StartTransport;
  if (mode == "auto") return T::AutoStdLv;
  if (mode == "native") return T::Native;
  if (mode == "required") return T::RequireStdLv;
  Rcpp::stop("start transport must be auto, native or required");
  return T::Native;
}

Rcpp::List start_result_to_r(const magmaan::estimate::StartValues& value) {
  using T = magmaan::estimate::StartTransport;
  return Rcpp::List::create(
      Rcpp::_["theta"] = Rcpp::wrap(value.theta),
      Rcpp::_["method"] = value.explicit_vector ? "explicit" : start_method_name(value.method),
      Rcpp::_["requested_transport"] = value.requested_transport == T::AutoStdLv ? "auto" :
          value.requested_transport == T::RequireStdLv ? "required" : "native",
      Rcpp::_["transport"] = value.branch == magmaan::estimate::StartBranch::TransportedStdLv
          ? "std-lv-to-marker" : "native",
      Rcpp::_["fallback_reason"] = magmaan::estimate::start_transport_reason(value.fallback_reason));
}

// Retain the vector supplied to fit, before any optimizer projection or profiling.
Eigen::VectorXd start_values_or_stop(Ctx& ctx,
    const magmaan::spec::Starts& starts, const std::string& default_name ,
    std::string* applied_policy , std::string* fallback_reason ,
    Rcpp::Nullable<Rcpp::List> control , bool normalize ) {
  namespace es = magmaan::estimate;
  std::string name = default_name;
  Rcpp::List ctl = control.isNotNull() ? Rcpp::List(control.get()) : Rcpp::List::create();
  SEXP input = ctl.containsElementNamed("start") ? static_cast<SEXP>(ctl["start"]) : R_NilValue;
  const bool explicit_vector = !Rf_isNull(input) && (TYPEOF(input) == REALSXP || TYPEOF(input) == INTSXP);
  if (!explicit_vector)
    name = start_name_from_arg(Rcpp::Nullable<Rcpp::String>(input), "start policy", default_name.c_str());
  es::StartPolicy policy{start_method(name), name == "scaled-fabin"
      ? es::StartTransport::AutoStdLv : es::StartTransport::Native};
  if (ctl.containsElementNamed("start_transport"))
    policy.transport = start_transport_from_arg(Rcpp::as<std::string>(ctl["start_transport"]));
  if (explicit_vector && ctl.containsElementNamed("start_transport") &&
      policy.transport != es::StartTransport::Native)
    Rcpp::stop("explicit start vectors are already in target coordinates; use native transport");
  if (ctl.containsElementNamed("normalize_sample"))
    normalize = normalize && Rcpp::as<bool>(ctl["normalize_sample"]);
  auto value = explicit_vector
      ? es::explicit_start_values(ctx.pt, Rcpp::as<Eigen::VectorXd>(input))
      : normalize ? es::normalized_ml_start_values(ctx.pt, ctx.rep, ctx.samp, policy, starts)
                  : es::start_values(ctx.pt, ctx.rep, ctx.samp, policy, starts);
  if (!value) stop_fit(value.error());
  if (applied_policy) *applied_policy = explicit_vector ? "explicit" :
      (name == "scaled-fabin" && policy.transport != es::StartTransport::Native
       ? (value->branch == es::StartBranch::TransportedStdLv
          ? "transported-std-lv-fabin" : "native-fabin-fallback") : name);
  if (fallback_reason) *fallback_reason = es::start_transport_reason(value->fallback_reason);
  ctx.start_values = std::move(*value);
  return ctx.start_values->theta;
}

Eigen::VectorXd ordinal_starts_or_stop(const Ctx& ctx,
                                       const magmaan::data::OrdinalStats& stats,
                                       const magmaan::spec::Starts& starts,
                                       Rcpp::Nullable<Rcpp::List> control) {
  // Numeric starts are already in the prepared free coordinates. In
  // particular, nested-null refits must not fall back to the old row hints.
  if (control.isNotNull()) {
    Rcpp::List ctl(control.get());
    if (ctl.containsElementNamed("start") && Rf_isNumeric(ctl["start"])) {
      auto x = Rcpp::as<Eigen::VectorXd>(ctl["start"]);
      if (x.size() != ctx.pt.n_free() || !x.allFinite())
        Rcpp::stop("magmaan: ordinal numeric start must be finite and match the prepared free parameters");
      return x;
    }
  }
  auto x = magmaan::estimate::ordinal_start_values(ctx.pt, ctx.rep, stats,
                                                   starts);
  if (!x.has_value()) stop_fit(x.error());
  return std::move(*x);
}

Eigen::VectorXd mixed_ordinal_starts_or_stop(
    const Ctx& ctx, const magmaan::data::MixedOrdinalStats& stats,
    const magmaan::spec::Starts& starts) {
  auto x = magmaan::estimate::mixed_ordinal_start_values(ctx.pt, ctx.rep, stats,
                                                         starts);
  if (!x.has_value()) stop_fit(x.error());
  return std::move(*x);
}

Rcpp::List names_list(const std::vector<std::vector<std::string>>& nn) {
  Rcpp::List out(static_cast<R_xlen_t>(nn.size()));
  for (std::size_t b = 0; b < nn.size(); ++b) out[static_cast<R_xlen_t>(b)] = Rcpp::wrap(nn[b]);
  return out;
}

Rcpp::List matrix_blocks_to_r(const std::vector<Eigen::MatrixXd>& blocks,
                              const std::vector<std::vector<std::string>>& names,
                              bool square_names) {
  Rcpp::List out(static_cast<R_xlen_t>(blocks.size()));
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    Rcpp::NumericMatrix M = Rcpp::wrap(blocks[b]);
    if (b < names.size()) {
      Rcpp::CharacterVector nm = Rcpp::wrap(names[b]);
      M.attr("dimnames") = square_names
          ? Rcpp::List::create(nm, nm)
          : Rcpp::List::create(R_NilValue, nm);
    }
    out[static_cast<R_xlen_t>(b)] = M;
  }
  return out;
}

Rcpp::List vector_blocks_to_r(const std::vector<Eigen::VectorXd>& blocks,
                              const std::vector<std::vector<std::string>>& names) {
  Rcpp::List out(static_cast<R_xlen_t>(blocks.size()));
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    Rcpp::NumericVector v = Rcpp::wrap(blocks[b]);
    if (b < names.size() && v.size() == static_cast<R_xlen_t>(names[b].size())) {
      v.attr("names") = Rcpp::wrap(names[b]);
    }
    out[static_cast<R_xlen_t>(b)] = v;
  }
  return out;
}

// One RMS residual section -> the 11-row column of lavaan's $summary table.
// Non-finite inferential fields map to R's NA (lavaan reports NA there).
Rcpp::NumericVector residual_rms_to_r(const magmaan::measures::ResidualRms& r) {
  auto na = [](double x) { return std::isfinite(x) ? x : NA_REAL; };
  return Rcpp::NumericVector::create(
      na(r.srmr), na(r.srmr_se), na(r.srmr_exactfit_z),
      na(r.srmr_exactfit_pvalue), na(r.usrmr), na(r.usrmr_se),
      na(r.usrmr_ci_lower), na(r.usrmr_ci_upper), na(r.usrmr_closefit_h0),
      na(r.usrmr_closefit_z), na(r.usrmr_closefit_pvalue));
}

// Per-block residual summary -> list of data frames shaped like
// lavResiduals(fit)$summary: rows are the SRMR-family statistics, columns are
// cov (and mean/total when the block carries a mean structure).
Rcpp::List residual_summary_to_r(
    const std::vector<magmaan::measures::ResidualSummary>& summ) {
  const Rcpp::CharacterVector row_names = Rcpp::CharacterVector::create(
      "srmr", "srmr.se", "srmr.exactfit.z", "srmr.exactfit.pvalue", "usrmr",
      "usrmr.se", "usrmr.ci.lower", "usrmr.ci.upper", "usrmr.closefit.h0.value",
      "usrmr.closefit.z", "usrmr.closefit.pvalue");
  Rcpp::List out(static_cast<R_xlen_t>(summ.size()));
  for (std::size_t b = 0; b < summ.size(); ++b) {
    const auto& s = summ[b];
    Rcpp::List df =
        s.has_mean
            ? Rcpp::List::create(Rcpp::_["cov"] = residual_rms_to_r(s.cov),
                                 Rcpp::_["mean"] = residual_rms_to_r(s.mean),
                                 Rcpp::_["total"] = residual_rms_to_r(s.total))
            : Rcpp::List::create(Rcpp::_["cov"] = residual_rms_to_r(s.cov));
    df.attr("class") = "data.frame";
    df.attr("row.names") = row_names;
    out[static_cast<R_xlen_t>(b)] = df;
  }
  return out;
}

// A measures::StandardizedResiduals -> the R list shared by the NT and the
// estimated-weight residual bindings (raw/cor/se/z matrices, SRMR, $summary).
Rcpp::List standardized_residuals_to_r(
    const magmaan::measures::StandardizedResiduals& r,
    const std::vector<std::vector<std::string>>& ov_names) {
  return Rcpp::List::create(
      Rcpp::_["cov_raw"] = matrix_blocks_to_r(r.cov_raw, ov_names, true),
      Rcpp::_["cov_cor"] = matrix_blocks_to_r(r.cov_cor, ov_names, true),
      Rcpp::_["cov_se"] = matrix_blocks_to_r(r.cov_se, ov_names, true),
      Rcpp::_["cov_z"] = matrix_blocks_to_r(r.cov_z, ov_names, true),
      Rcpp::_["mean_raw"] = vector_blocks_to_r(r.mean_raw, ov_names),
      Rcpp::_["mean_cor"] = vector_blocks_to_r(r.mean_cor, ov_names),
      Rcpp::_["mean_se"] = vector_blocks_to_r(r.mean_se, ov_names),
      Rcpp::_["mean_z"] = vector_blocks_to_r(r.mean_z, ov_names),
      Rcpp::_["srmr"] = r.srmr,
      Rcpp::_["summary"] = residual_summary_to_r(r.summary));
}

const char* fit_check_to_r(magmaan::estimate::FitCheck check) {
  using magmaan::estimate::FitCheck;
  return check == FitCheck::Passed ? "passed"
       : check == FitCheck::Failed ? "failed" : "unchecked";
}

int common_converged_value(const magmaan::estimate::Estimates& est) {
  using magmaan::estimate::FitCheck;
  const auto status = magmaan::estimate::fit_verdict(est).status;
  return status == FitCheck::Unchecked ? NA_LOGICAL
       : status == FitCheck::Passed;
}

Rcpp::LogicalVector common_converged_to_r(
    const magmaan::estimate::Estimates& est) {
  return Rcpp::LogicalVector::create(common_converged_value(est));
}

Rcpp::List common_verdict_to_r(const magmaan::estimate::FitDiagnostics& d) {
  const auto v = magmaan::estimate::common_fit_verdict(d);
  const auto& o = d.objective;
  return Rcpp::List::create(
      Rcpp::_["status"] = fit_check_to_r(v.status),
      Rcpp::_["objective"] = fit_check_to_r(v.objective),
      Rcpp::_["stationarity"] = fit_check_to_r(v.stationarity),
      Rcpp::_["domain"] = v.domain == magmaan::estimate::StationarityDomain::Psd
          ? "psd" : "ambient",
      Rcpp::_["criterion"] =
          v.criterion == magmaan::estimate::StationarityCriterion::Newton
          ? "newton" : "first_order",
      Rcpp::_["identification"] = fit_check_to_r(v.identification),
      Rcpp::_["objective_multiplier"] = o.multiplier,
      Rcpp::_["objective_recomputed"] = o.recomputed,
      Rcpp::_["objective_reported"] = o.reported,
      Rcpp::_["objective_consistency_tolerance"] = o.consistency_tolerance);
}

const char* optim_status_to_r(magmaan::optim::OptimStatus status) {
  using magmaan::optim::OptimStatus;
  return status == OptimStatus::Converged           ? "converged"
       : status == OptimStatus::LineSearchSalvaged ? "line_search_salvaged"
       : status == OptimStatus::LineSearchFailed   ? "line_search_failed"
       : status == OptimStatus::SingularConvergence ? "singular_convergence"
       : status == OptimStatus::NoisyObjective     ? "noisy_objective"
       : status == OptimStatus::FalseConvergence   ? "false_convergence"
       : status == OptimStatus::BudgetExhausted    ? "budget_exhausted"
                                                    : "unknown";
}

const char* score_candidate_kind_str(
    magmaan::inference::ScoreCandidateKind kind) {
  using K = magmaan::inference::ScoreCandidateKind;
  switch (kind) {
  case K::FixedParam: return "fixed";
  case K::EqualityRelease: return "equality_release";
  }
  return "unknown";
}

magmaan::inference::ScoreInformation score_information_from(
    const std::string& information) {
  if (information == "observed" || information == "Observed" ||
      information == "OBSERVED") {
    return magmaan::inference::ScoreInformation::Observed;
  }
  if (information == "expected" || information == "Expected" ||
      information == "EXPECTED" || information.empty()) {
    return magmaan::inference::ScoreInformation::Expected;
  }
  Rcpp::stop("magmaan: score information must be 'expected' or 'observed' "
             "(got '%s')", information);
}

magmaan::inference::ScoreCandidateSet score_candidates_from(
    const std::string& candidates) {
  if (candidates == "all" || candidates == "absent" ||
      candidates == "with_absent") {
    return magmaan::inference::ScoreCandidateSet::WithAbsentRows;
  }
  if (candidates == "fixed" || candidates == "fixed_rows" ||
      candidates == "fixed_rows_only" || candidates.empty()) {
    return magmaan::inference::ScoreCandidateSet::FixedRowsOnly;
  }
  Rcpp::stop("magmaan: score candidates must be 'fixed' or 'all' (got '%s')",
             candidates);
}

magmaan::inference::ModificationIndexOptions modification_options_from(
    const std::string& information, const std::string& candidates,
    bool include_loadings, bool include_covariances) {
  magmaan::inference::ModificationIndexOptions opts;
  opts.information = score_information_from(information);
  opts.candidates = score_candidates_from(candidates);
  opts.include_loadings = include_loadings;
  opts.include_covariances = include_covariances;
  return opts;
}

magmaan::estimate::OrdinalWeightKind ordinal_weight_from_estimator(
    const std::string& estimator, const char* call) {
  if (estimator == "ULS") return magmaan::estimate::OrdinalWeightKind::ULS;
  if (estimator == "DWLS") return magmaan::estimate::OrdinalWeightKind::DWLS;
  if (estimator == "WLS") return magmaan::estimate::OrdinalWeightKind::WLS;
  Rcpp::stop("magmaan: %s requires an ordinal ULS/DWLS/WLS fit", call);
}

magmaan::measures::frontier::reliability::OmegaTarget
omega_target_from_string(const std::string& target, const char* call) {
  namespace rel = magmaan::measures::frontier::reliability;
  if (target == "total") return rel::OmegaTarget::Total;
  if (target == "hierarchical") return rel::OmegaTarget::Hierarchical;
  Rcpp::stop("magmaan: %s target must be 'total' or 'hierarchical'", call);
}

magmaan::estimate::frontier::OrdinalStage2Weight
ordinal_stage2_weight_from_string(const std::string& s) {
  std::string key = s;
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  if (key == "uls") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Uls;
  }
  if (key == "dwls") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Dwls;
  }
  if (key == "wls" || key == "adf") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Wls;
  }
  if (key == "nt" || key == "gls") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Nt;
  }
  if (key == "dls") {
    return magmaan::estimate::frontier::OrdinalStage2Weight::Dls;
  }
  Rcpp::stop("magmaan: ordinal stage2_weight must be one of "
             "'uls', 'dwls', 'wls'/'adf', 'nt'/'gls', or 'dls' (got '%s')",
             s);
}

std::string ordinal_weight_for_postfit(Rcpp::List fit,
                                       const std::string& estimator) {
  if (fit.containsElementNamed("ordinal_computational_weight")) {
    return Rcpp::as<std::string>(fit["ordinal_computational_weight"]);
  }
  return estimator;
}

std::string ordinal_weight_key_from_arg(const std::string& weight,
                                        Rcpp::List fit,
                                        const std::string& estimator) {
  std::string key = weight;
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  if (key.empty() || key == "fit") return ordinal_weight_for_postfit(fit, estimator);
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char ch) { return std::toupper(ch); });
  return key;
}

Rcpp::List stats_from_fit_or_arg(Rcpp::List fit, SEXP arg,
                                 const char* field, const char* call) {
  if (!Rf_isNull(arg)) {
    if (TYPEOF(arg) != VECSXP) {
      Rcpp::stop("magmaan: %s requires `%s` as a categorical stats list",
                 call, field);
    }
    return Rcpp::List(arg);
  }
  if (!fit.containsElementNamed(field)) {
    Rcpp::stop("magmaan: %s requires `%s` (pass the data object used for fitting)",
               call, field);
  }
  return Rcpp::List(SEXP(fit[field]));
}

// Convert a TerminalAudit (L1, driven coordinates) to an R sub-list. Surfaced
// as `fit$audit` — same mapping for OptimStatus as the existing `optimizer_status`
// string. `active_set` carries {-1, 0, +1} per driven coordinate, distinct
// from L2's `active_bounds_full` (which indexes the expanded θ).
Rcpp::List audit_to_r(const magmaan::optim::TerminalAudit& a) {
  const char* advisory = optim_status_to_r(a.advisory_status);
  Rcpp::List controls = Rcpp::List::create();
  if (a.nlopt_controls) {
    const auto& c = *a.nlopt_controls;
    auto real = [&](const char* key, const std::optional<double>& value) { if (value) controls[key] = *value; };
    real("ftol_rel", c.ftol_rel); real("ftol_abs", c.ftol_abs);
    real("xtol_rel", c.xtol_rel); real("xtol_abs", c.xtol_abs);
    real("tolg", c.tolg); real("constraint_tol", c.constraint_tol);
    if (c.max_eval) controls["max_eval"] = *c.max_eval;
    if (c.vector_storage) controls["vector_storage"] = *c.vector_storage;
  }
  Rcpp::IntegerVector active(static_cast<R_xlen_t>(a.active_set.size()));
  for (std::size_t i = 0; i < a.active_set.size(); ++i)
    active[static_cast<R_xlen_t>(i)] = static_cast<int>(a.active_set[i]);
  Rcpp::List port_endpoint;
  if (a.port_endpoint) {
    const auto& e = *a.port_endpoint;
    port_endpoint = Rcpp::List::create(
        Rcpp::_["stored_objective"] = e.stored_objective,
        Rcpp::_["returned_x_objective"] = e.returned_x_objective,
        Rcpp::_["best_point_substituted"] = e.best_point_substituted);
  }
  return Rcpp::List::create(
      Rcpp::_["port_endpoint"] = port_endpoint,
      Rcpp::_["stationary"]       = a.stationary,
      Rcpp::_["raw_backend_status"] = a.raw_backend_status,
      Rcpp::_["nlopt_controls"] = controls,
      Rcpp::_["backend_gradient_max"] = a.backend_gradient_max,
      Rcpp::_["grad_inf_norm"]    = a.grad_inf_norm,
      Rcpp::_["raw_grad_inf_norm"] = a.raw_grad_inf_norm,
      Rcpp::_["grad_scaled_inf"]  = a.grad_scaled_inf,
      Rcpp::_["stationarity_rhs"] = a.stationarity_rhs,
      Rcpp::_["f_recomputed"]     = a.f_recomputed,
      Rcpp::_["f_consistent"]     = a.f_consistent,
      Rcpp::_["f_finite"]         = a.f_finite,
      Rcpp::_["constrained"]       = a.constrained,
      Rcpp::_["constraint_violation_inf"] =
          a.constraint_violation_inf,
      Rcpp::_["constraint_jacobian_rank"] =
          a.constraint_jacobian_rank,
      Rcpp::_["active_set"]       = active,
      Rcpp::_["advisory_status"]  = advisory);
}

// Convert FitDiagnostics (L2, expanded θ) to an R sub-list. Surfaced as
// `fit$diagnostics`. Active-bound indices are converted 0-based → 1-based at
// the R boundary so they index `theta`/`partable` rows directly in R.
Rcpp::List covariance_blocks_to_r(
    const std::vector<magmaan::estimate::CovarianceBlockDiagnostics>& blocks) {
  Rcpp::List out(static_cast<R_xlen_t>(blocks.size()));
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& d = blocks[b];
    Rcpp::IntegerVector rows(d.covariance_rows.begin(),
                             d.covariance_rows.end());
    Rcpp::IntegerVector negative(d.negative_variance_rows.begin(),
                                 d.negative_variance_rows.end());
    Rcpp::IntegerVector correlation(d.invalid_correlation_rows.begin(),
                                    d.invalid_correlation_rows.end());
    for (R_xlen_t i = 0; i < rows.size(); ++i) rows[i] += 1;
    for (R_xlen_t i = 0; i < negative.size(); ++i) negative[i] += 1;
    for (R_xlen_t i = 0; i < correlation.size(); ++i) correlation[i] += 1;
    out[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["block"] = d.block + 1,
        Rcpp::_["min_eigenvalue"] = d.min_eigenvalue,
        Rcpp::_["finite"] = d.finite,
        Rcpp::_["psd"] = d.psd,
        Rcpp::_["positive_definite"] = d.positive_definite,
        Rcpp::_["covariance_rows"] = rows,
        Rcpp::_["negative_variance_rows"] = negative,
        Rcpp::_["invalid_correlation_rows"] = correlation);
  }
  return out;
}

Rcpp::List admissibility_to_r(
    const magmaan::estimate::AdmissibilityDiagnostics& d) {
  return Rcpp::List::create(
      Rcpp::_["checked"] = d.checked,
      Rcpp::_["covariance_matrices_psd"] = d.covariance_matrices_psd,
      Rcpp::_["implied_sigma_pd"] = d.implied_sigma_pd,
      Rcpp::_["admissible"] = d.admissible,
      Rcpp::_["theta"] = covariance_blocks_to_r(d.theta_blocks),
      Rcpp::_["psi"] = covariance_blocks_to_r(d.psi_blocks));
}

Rcpp::List geometric_stationarity_to_r(
    const magmaan::estimate::GeometricStationarityDiagnostics& d) {
  return Rcpp::List::create(
      Rcpp::_["checked"] = d.checked,
      Rcpp::_["metric"] = "model_frobenius",
      Rcpp::_["gradient_finite"] = d.gradient_finite,
      Rcpp::_["feasible"] = d.feasible,
      Rcpp::_["covariance_feasible"] = d.covariance_feasible,
      Rcpp::_["ambient_stationary"] = d.ambient_stationary,
      Rcpp::_["ambient_residual_inf"] = d.ambient_residual_inf,
      Rcpp::_["ambient_residual_l2"] = d.ambient_residual_l2,
      Rcpp::_["cone_stationary"] = d.cone_stationary,
      Rcpp::_["cone_residual_inf"] = d.cone_residual_inf,
      Rcpp::_["cone_residual_l2"] = d.cone_residual_l2,
      Rcpp::_["raw_gradient_inf"] = d.raw_gradient_inf,
      Rcpp::_["stationarity_tol"] = d.stationarity_tol,
      Rcpp::_["covariance_eigen_tol"] = d.covariance_eigen_tol,
      Rcpp::_["covariance_active_blocks"] = d.covariance_active_blocks,
      Rcpp::_["covariance_nullity"] = d.covariance_nullity,
      Rcpp::_["ambient_projection_converged"] =
          d.ambient_projection_converged,
      Rcpp::_["cone_projection_converged"] =
          d.cone_projection_converged,
      Rcpp::_["ambient_projection_iterations"] =
          d.ambient_projection_iterations,
      Rcpp::_["cone_projection_iterations"] =
          d.cone_projection_iterations);
}

SEXP retained_ls_weights_to_r(const magmaan::estimate::frontier::NewtonDerivatives& d) {
  if(!d.ls_weight) return R_NilValue;
  Rcpp::List out(d.ls_weight->size());
  using Kind=magmaan::estimate::gmm::BlockWeight::Kind;
  for(std::size_t k=0;k<d.ls_weight->size();++k) {
    const auto& w=(*d.ls_weight)[k];
    const char* kind=w.kind()==Kind::Identity ? "identity" :
        w.kind()==Kind::Diagonal ? "diagonal" : w.kind()==Kind::Dense ? "dense_factor" : "normal_theory_root";
    out[k]=Rcpp::List::create(Rcpp::_["kind"]=kind,
        Rcpp::_["diagonal"]=Rcpp::wrap(w.diagonal_values()),
        Rcpp::_["factor"]=Rcpp::wrap(w.dense_factor()),
        Rcpp::_["root"]=Rcpp::wrap(w.normal_theory_root()),
        Rcpp::_["has_means"]=w.has_means());
  }
  return out;
}

Rcpp::List newton_accuracy_to_r(
    const magmaan::estimate::NewtonAccuracyDiagnostics& a) {
  auto num = [](double x) { return std::isfinite(x) ? x : NA_REAL; };
  return Rcpp::List::create(
      Rcpp::_["checked"] = a.checked,
      Rcpp::_["status"] = std::string(magmaan::estimate::to_string(a.status)),
      Rcpp::_["objective"] = std::string(magmaan::estimate::to_string(a.objective)),
      Rcpp::_["curvature"] = std::string(magmaan::estimate::to_string(a.curvature)),
      Rcpp::_["metric"] = std::string(magmaan::estimate::to_string(a.metric)),
      Rcpp::_["distance"] = num(a.distance),
      Rcpp::_["passed"] = a.passed,
      Rcpp::_["budget"] = a.budget,
      Rcpp::_["covariance_interior"] = a.covariance_interior,
      Rcpp::_["predicted_gain"] = num(a.predicted_gain),
      Rcpp::_["max_step"] = num(a.max_step),
      Rcpp::_["condition"] = num(a.condition),
      Rcpp::_["solve_residual"] = num(a.solve_residual),
      Rcpp::_["n_reduced"] = a.n_reduced,
      Rcpp::_["psd_domain"] = a.psd_domain,
      Rcpp::_["unit_normalized"] = a.unit_normalized,
      Rcpp::_["box_constrained"] = a.box_constrained,
      Rcpp::_["null_directions"] = a.null_directions,
      Rcpp::_["constrained_directions"] = a.constrained_directions,
      Rcpp::_["min_multiplier"] = num(a.min_multiplier));
}

Rcpp::List input_errors_to_r(const magmaan::estimate::frontier::NewtonInputErrorBounds& e) {
  return Rcpp::List::create(Rcpp::_["status"]=std::string(magmaan::estimate::to_string(e.status)),
      Rcpp::_["matrix"]=e.matrix,Rcpp::_["vector"]=e.vector,Rcpp::_["curvature"]=e.curvature,
      Rcpp::_["curvature_lower_bound"]=e.curvature_lower_bound,Rcpp::_["detail"]=e.detail);
}
Rcpp::List distance_interval_to_r(const magmaan::estimate::frontier::NewtonDistanceInterval& x) {
  return Rcpp::List::create(Rcpp::_["status"]=std::string(magmaan::estimate::to_string(x.status)),
      Rcpp::_["decision"]=std::string(magmaan::estimate::frontier::to_string(x.decision)),
      Rcpp::_["distance"]=x.distance,Rcpp::_["lower"]=x.lower,Rcpp::_["upper"]=x.upper,
      Rcpp::_["rank_margin"]=x.rank_margin,Rcpp::_["factor_error_bound"]=x.factor_error_bound);
}
Rcpp::List verified_assessment_to_r(const magmaan::estimate::frontier::ConvergenceAssessment& a) {
  auto check=[](const magmaan::estimate::frontier::ConvergenceCheck& x) {
    return Rcpp::List::create(Rcpp::_["status"]=fit_check_to_r(x.status),Rcpp::_["reason"]=x.reason);
  };
  Rcpp::LogicalVector passed(1);
  passed[0]=a.status==magmaan::estimate::FitCheck::Unchecked ? NA_LOGICAL : a.status==magmaan::estimate::FitCheck::Passed;
  return Rcpp::List::create(Rcpp::_["status"]=fit_check_to_r(a.status),Rcpp::_["converged"]=passed,
      Rcpp::_["objective"]=check(a.objective),Rcpp::_["objective_consistency"]=check(a.objective_consistency),
      Rcpp::_["feasibility"]=check(a.feasibility),Rcpp::_["newton"]=check(a.newton),
      Rcpp::_["identification"]=check(a.identification));
}

Rcpp::List identification_to_r(
    const magmaan::estimate::IdentificationReport& r,
    const std::vector<std::string>* labels) {
  namespace fr = magmaan::estimate::frontier;
  Rcpp::NumericMatrix directions = Rcpp::wrap(r.null_directions);
  const std::vector<std::string> names =
      labels != nullptr &&
              static_cast<Eigen::Index>(labels->size()) == r.null_directions.rows()
          ? *labels
          : std::vector<std::string>{};
  if (!names.empty() && r.null_directions.rows() > 0)
    directions.attr("dimnames") = Rcpp::List::create(Rcpp::wrap(names), R_NilValue);
  Rcpp::IntegerVector rank(1);
  rank[0] = r.rank < 0 ? NA_INTEGER : r.rank;
  return Rcpp::List::create(
      Rcpp::_["status"] = std::string(magmaan::estimate::to_string(r.status)),
      Rcpp::_["reason"] = std::string(magmaan::estimate::to_string(r.reason)),
      Rcpp::_["map"] = std::string(magmaan::estimate::to_string(r.map)),
      Rcpp::_["n_parameters"] = r.n_parameters,
      Rcpp::_["n_moments"] = r.n_moments,
      Rcpp::_["counting_rule"] = r.counting_rule,
      Rcpp::_["rank"] = rank,
      Rcpp::_["n_points"] = r.n_points,
      Rcpp::_["null_tolerance"] = r.null_tolerance,
      Rcpp::_["identified_tolerance"] = r.identified_tolerance,
      Rcpp::_["min_relative_singular_values"] =
          Rcpp::wrap(r.min_relative_singular_values),
      Rcpp::_["smallest_singular_values"] = Rcpp::wrap(r.smallest_singular_values),
      Rcpp::_["null_directions"] = directions,
      Rcpp::_["null_direction_text"] =
          Rcpp::wrap(fr::describe_null_directions(r, names)),
      Rcpp::_["directions_at_estimate"] = r.directions_at_estimate,
      Rcpp::_["direction_types"] = Rcpp::wrap(r.direction_types),
      Rcpp::_["direction_factors"] = Rcpp::wrap(r.direction_factors),
      Rcpp::_["suggested_fixes"] = Rcpp::wrap(r.suggested_fixes),
      Rcpp::_["gauge_dimension"] = r.gauge_dimension,
      Rcpp::_["deficit_dimension"] = r.deficit_dimension);
}

Rcpp::List diagnostics_to_r(const magmaan::estimate::FitDiagnostics& d,
                            const std::vector<std::string>* labels) {
  Rcpp::LogicalVector sigma_pd(static_cast<R_xlen_t>(d.sigma_pd_per_block.size()));
  for (std::size_t b = 0; b < d.sigma_pd_per_block.size(); ++b)
    sigma_pd[static_cast<R_xlen_t>(b)] = d.sigma_pd_per_block[b];

  Rcpp::IntegerVector at_lo(d.active_bounds_full.at_lower.begin(),
                            d.active_bounds_full.at_lower.end());
  Rcpp::IntegerVector at_up(d.active_bounds_full.at_upper.begin(),
                            d.active_bounds_full.at_upper.end());
  for (R_xlen_t i = 0; i < at_lo.size(); ++i) at_lo[i] += 1;
  for (R_xlen_t i = 0; i < at_up.size(); ++i) at_up[i] += 1;

  return Rcpp::List::create(
      Rcpp::_["sigma_pd_per_block"]     = sigma_pd,
      Rcpp::_["sigma_pd_all"]           = d.sigma_pd_all,
      Rcpp::_["observed_variance_ratio"] = d.observed_variance_ratio,
      Rcpp::_["numerical_scaling_message"] = d.numerical_scaling_message,
      Rcpp::_["lin_eq_residual_inf"]    = d.lin_eq_residual_inf,
      Rcpp::_["lin_eq_satisfied"]       = d.lin_eq_satisfied,
      Rcpp::_["nl_eq_residual"]         = Rcpp::wrap(d.nl_eq_residual),
      Rcpp::_["nl_eq_residual_inf"]     = d.nl_eq_residual_inf,
      Rcpp::_["nl_eq_satisfied"]        = d.nl_eq_satisfied,
      Rcpp::_["active_bounds_lower"]    = at_lo,
      Rcpp::_["active_bounds_upper"]    = at_up,
      Rcpp::_["admissibility"]          = admissibility_to_r(d.admissibility),
      Rcpp::_["geometric_stationarity"] =
          geometric_stationarity_to_r(d.geometric_stationarity),
      Rcpp::_["newton_accuracy"] = newton_accuracy_to_r(d.newton_accuracy),
      Rcpp::_["identification"] = identification_to_r(d.identification, labels),
      Rcpp::_["verdict"] = common_verdict_to_r(d),
      Rcpp::_["snlls_profile_fallback"] = d.snlls_profile_fallback);
}

Rcpp::List fit_result(Ctx& ctx,
                      const magmaan::estimate::Estimates& est,
                      const magmaan::spec::Starts* starts,
                      const char* estimator) {
  const std::size_t nb = ctx.samp.S.size();
  Rcpp::List S_out(static_cast<R_xlen_t>(nb));
  for (std::size_t b = 0; b < nb; ++b) {
    Rcpp::NumericMatrix Sb = Rcpp::wrap(ctx.samp.S[b]);
    Rcpp::CharacterVector nm = Rcpp::wrap(ctx.rep.ov_names[b]);
    Sb.attr("dimnames") = Rcpp::List::create(nm, nm);
    S_out[static_cast<R_xlen_t>(b)] = Sb;
  }
  // Keep the means protected after the temporary list leaves its block;
  // constructing the partable/verdict can trigger GC before out owns them.
  Rcpp::RObject mean_out = R_NilValue;
  if (!ctx.samp.mean.empty()) {
    Rcpp::List Ml(static_cast<R_xlen_t>(nb));
    for (std::size_t b = 0; b < nb; ++b)
      Ml[static_cast<R_xlen_t>(b)] = Rcpp::wrap(ctx.samp.mean[b]);
    mean_out = Ml;
  }
  Rcpp::IntegerVector nobs_out(static_cast<R_xlen_t>(nb));
  std::int64_t ntotal = 0;
  for (std::size_t b = 0; b < nb; ++b) {
    nobs_out[static_cast<R_xlen_t>(b)] = static_cast<int>(ctx.samp.n_obs[b]);
    ntotal += ctx.samp.n_obs[b];
  }

  // Backend termination remains diagnostic. Convergence is the common
  // full-model verdict, with NA for paths whose checks are not yet wired.
  using magmaan::optim::OptimStatus;
  const char* opt_status = optim_status_to_r(est.optimizer_status);

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["converged"]     = common_converged_to_r(est),
      Rcpp::_["verdict"] = common_verdict_to_r(est.diagnostics),
      Rcpp::_["estimator"]     = estimator,
      Rcpp::_["fmin"]          = est.fmin,
      Rcpp::_["iterations"]    = est.iterations,
      Rcpp::_["f_evals"]       = est.f_evals,
      Rcpp::_["coordinate_scaling"] =
          magmaan::estimate::coordinate_scaling_name(est.coordinate_scaling),
      Rcpp::_["g_evals"]       = est.g_evals,
      Rcpp::_["npar"]          = static_cast<int>(ctx.pt.n_free()),
      Rcpp::_["ngroups"]       = static_cast<int>(nb),
      Rcpp::_["ntotal"]        = static_cast<int>(ntotal),
      Rcpp::_["group_var"]     = ctx.names.group_var,
      Rcpp::_["group_labels"]  = Rcpp::wrap(ctx.names.group_labels),
      Rcpp::_["theta"]         = Rcpp::wrap(est.theta),
      Rcpp::_["ov_names"]      = Rcpp::wrap(ctx.ov_names),
      Rcpp::_["partable"]      = partable_df(ctx.pt, ctx.names, est, starts),
      Rcpp::_["S"]             = S_out,
      Rcpp::_["nobs"]          = nobs_out,
      Rcpp::_["sample_mean"]   = mean_out,
      Rcpp::_["meanstructure"] = ctx.meanstructure);
  out["sample_normalized"] = est.sample_normalized;
  out["optimizer_status"] = opt_status;
  out["grad_norm"]        = est.grad_inf_norm;
  out["audit"]            = audit_to_r(est.audit);
  const std::vector<std::string> labels =
      magmaan::estimate::frontier::free_parameter_labels(ctx.pt, ctx.names);
  out["diagnostics"]      = diagnostics_to_r(est.diagnostics, &labels);
  if (est.fitting) {
    out["fitting"] = fitting_report_to_r(*est.fitting);
    Rcpp::List verdict = common_verdict_to_r(est.diagnostics);
    const auto v = magmaan::estimate::fit_verdict(est);
    verdict["status"] = fit_check_to_r(v.status);
    verdict["stationarity"] = fit_check_to_r(v.stationarity);
    verdict["objective"] = fit_check_to_r(v.objective);
    verdict["identification"] = fit_check_to_r(v.identification);
    verdict["policy"] = est.fitting->setup.convergence;
    if (est.selected_verdict) verdict["criterion"] = "optimizer_gradient";
    out["verdict"] = verdict;
    if (!est.fitting->attempts.empty()) out["start"] = Rcpp::List::create(
        Rcpp::_["theta"] = Rcpp::wrap(est.fitting->attempts.front().start),
        Rcpp::_["method"] = est.fitting->explicit_start ? std::string("explicit")
                                                         : est.fitting->setup.starts);
  }
  if (est.substituted_backend)
    out["optimizer_substituted"] =
        std::string(magmaan::estimate::backend_name(*est.substituted_backend));
  if (ctx.start_values) out["start"] = start_result_to_r(*ctx.start_values);
  return out;
}

magmaan::estimate::Bounds bounds_from_nullable(Rcpp::Nullable<Rcpp::List> bounds) {
  magmaan::estimate::Bounds out;
  if (bounds.isNull()) return out;
  Rcpp::List b(bounds.get());
  if (!b.containsElementNamed("lower") || !b.containsElementNamed("upper"))
    Rcpp::stop("magmaan: bounds must be a list with $lower and $upper");
  out.lower = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(b["lower"]));
  out.upper = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(b["upper"]));
  return out;
}

// Parse R's matrix / list-of-matrices weight argument into dense blocks.
// `prepare_weight` needs this dense form (it validates it and hands it back to
// R); everything else wants the structured Weight from wls_from_arg below.
std::vector<Eigen::MatrixXd> wls_dense_from_arg(SEXP W, std::size_t n_blocks) {
  std::vector<Eigen::MatrixXd> weights;
  weights.reserve(n_blocks);
  if (Rf_isMatrix(W)) {
    if (n_blocks != 1)
      Rcpp::stop("magmaan: WLS weights must be a list of %d matrices for a %d-group model",
                 static_cast<int>(n_blocks), static_cast<int>(n_blocks));
    weights.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(W)));
  } else if (TYPEOF(W) == VECSXP) {
    Rcpp::List Wl(W);
    if (static_cast<std::size_t>(Wl.size()) != n_blocks)
      Rcpp::stop("magmaan: WLS weights list has length %d but the model has %d group(s)",
                 static_cast<int>(Wl.size()), static_cast<int>(n_blocks));
    for (R_xlen_t b = 0; b < Wl.size(); ++b)
      weights.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Wl[b])));
  } else {
    Rcpp::stop("magmaan: WLS weights must be a matrix or a list of matrices");
  }
  return weights;
}

// R supplies a genuinely dense Gamma-hat inverse, so Dense is the right
// BlockWeight kind here; see internal.h for why the conversion is shared.
magmaan::estimate::gmm::Weight wls_from_arg(SEXP W, std::size_t n_blocks) {
  auto weight = dense_weight_or_stop(wls_dense_from_arg(W, n_blocks),
                                     "magmaan: WLS weights");
  Rcpp::RObject source(W);
  if (source.hasAttribute("cancels_measurement_units") &&
      Rcpp::as<bool>(source.attr("cancels_measurement_units")))
    for (auto& block : weight) block.set_cancels_measurement_units();
  return weight;
}

Rcpp::List ordinal_stats_to_r(const magmaan::data::OrdinalStats& s) {
  const R_xlen_t nb = static_cast<R_xlen_t>(s.R.size());
  Rcpp::List R(nb), thresholds(nb), threshold_ov(nb), threshold_level(nb),
      moments(nb), NACOV(nb), W_dwls(nb), W_wls(nb), moment_influence(nb),
      int_data(nb), moment_bread(nb), n_levels(nb);
  Rcpp::IntegerVector nobs(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::size_t bi = static_cast<std::size_t>(b);
    R[b] = Rcpp::wrap(s.R[bi]);
    thresholds[b] = Rcpp::wrap(s.thresholds[bi]);
    const Eigen::Index p = s.R[bi].rows();
    const Eigen::Index nth = s.thresholds[bi].size();
    Eigen::VectorXd mb(nth + p * (p - 1) / 2);
    mb.head(nth) = s.thresholds[bi];
    Eigen::Index pos = nth;
    for (Eigen::Index j = 0; j < p; ++j) {
      for (Eigen::Index i = j + 1; i < p; ++i) {
        mb(pos++) = s.R[bi](i, j);
      }
    }
    moments[b] = Rcpp::wrap(mb);
    Rcpp::IntegerVector ov(static_cast<R_xlen_t>(s.threshold_ov[bi].size()));
    Rcpp::IntegerVector lev(static_cast<R_xlen_t>(s.threshold_level[bi].size()));
    for (R_xlen_t k = 0; k < ov.size(); ++k) {
      ov[k] = s.threshold_ov[bi][static_cast<std::size_t>(k)] + 1;
      lev[k] = s.threshold_level[bi][static_cast<std::size_t>(k)];
    }
    threshold_ov[b] = ov;
    threshold_level[b] = lev;
    NACOV[b] = bi < s.NACOV.size()
        ? Rcpp::wrap(s.NACOV[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    W_dwls[b] = bi < s.W_dwls.size()
        ? Rcpp::wrap(s.W_dwls[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    W_wls[b] = bi < s.W_wls.size()
        ? Rcpp::wrap(s.W_wls[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    moment_influence[b] = bi < s.moment_influence.size()
        ? Rcpp::wrap(s.moment_influence[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    int_data[b] = bi < s.int_data.size()
        ? Rcpp::wrap(s.int_data[bi])
        : Rcpp::wrap(Eigen::MatrixXi(0, 0));
    moment_bread[b] = bi < s.moment_bread.size()
        ? Rcpp::wrap(s.moment_bread[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    nobs[b] = static_cast<int>(s.n_obs[bi]);
    n_levels[b] = Rcpp::wrap(s.n_levels[bi]);
  }
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["R"] = R,
      Rcpp::_["thresholds"] = thresholds,
      Rcpp::_["threshold_ov"] = threshold_ov,
      Rcpp::_["threshold_level"] = threshold_level,
      Rcpp::_["moments"] = moments,
      Rcpp::_["NACOV"] = NACOV,
      Rcpp::_["W_dwls"] = W_dwls,
      Rcpp::_["W_wls"] = W_wls,
      Rcpp::_["moment_influence"] = moment_influence,
      Rcpp::_["int_data"] = int_data,
      Rcpp::_["moment_bread"] = moment_bread,
      Rcpp::_["nobs"] = nobs,
      Rcpp::_["n_levels"] = n_levels);
  if (!s.pairwise_gamma.empty()) {
    Rcpp::List support_i(static_cast<R_xlen_t>(s.moment_support_i.size()));
    Rcpp::List support_j(static_cast<R_xlen_t>(s.moment_support_j.size()));
    Rcpp::List moment_nobs(static_cast<R_xlen_t>(s.moment_n_obs.size()));
    Rcpp::List overlap(static_cast<R_xlen_t>(s.moment_overlap_n_obs.size()));
    for (R_xlen_t b = 0; b < support_i.size(); ++b) {
      const auto bi = static_cast<std::size_t>(b);
      Rcpp::IntegerVector si(static_cast<R_xlen_t>(s.moment_support_i[bi].size()));
      Rcpp::IntegerVector sj(static_cast<R_xlen_t>(s.moment_support_j[bi].size()));
      Rcpp::NumericVector mn(static_cast<R_xlen_t>(s.moment_n_obs[bi].size()));
      for (R_xlen_t k = 0; k < si.size(); ++k) {
        si[k] = s.moment_support_i[bi][static_cast<std::size_t>(k)] + 1;
        sj[k] = s.moment_support_j[bi][static_cast<std::size_t>(k)] >= 0
                    ? s.moment_support_j[bi][static_cast<std::size_t>(k)] + 1
                    : NA_INTEGER;
        mn[k] = static_cast<double>(s.moment_n_obs[bi][static_cast<std::size_t>(k)]);
      }
      support_i[b] = si;
      support_j[b] = sj;
      moment_nobs[b] = mn;
      overlap[b] = Rcpp::wrap(s.moment_overlap_n_obs[bi].cast<double>());
    }
    out.attr("pd_gamma") = s.pairwise_gamma;
    out.attr("moment_support_i") = support_i;
    out.attr("moment_support_j") = support_j;
    out.attr("moment_nobs") = moment_nobs;
    out.attr("moment_overlap_nobs") = overlap;
  }
  out.attr("class") = Rcpp::CharacterVector::create("magmaan_ordinal_data", "list");
  return out;
}

Rcpp::List mixed_ordinal_stats_to_r(const magmaan::data::MixedOrdinalStats& s) {
  const R_xlen_t nb = static_cast<R_xlen_t>(s.R.size());
  Rcpp::List R(nb), mean(nb), ordered_mask(nb), thresholds(nb), threshold_ov(nb),
      threshold_level(nb), moments(nb), NACOV(nb), W_dwls(nb), W_wls(nb),
      moment_influence(nb), gamma_diag_influence(nb),
      gamma_full_influence(nb), raw_data(nb), n_levels(nb);
  Rcpp::IntegerVector nobs(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::size_t bi = static_cast<std::size_t>(b);
    R[b] = Rcpp::wrap(s.R[bi]);
    mean[b] = Rcpp::wrap(s.mean[bi]);
    ordered_mask[b] = Rcpp::wrap(s.ordered[bi]);
    thresholds[b] = Rcpp::wrap(s.thresholds[bi]);
    moments[b] = Rcpp::wrap(s.moments[bi]);
    Rcpp::IntegerVector ov(static_cast<R_xlen_t>(s.threshold_ov[bi].size()));
    Rcpp::IntegerVector lev(static_cast<R_xlen_t>(s.threshold_level[bi].size()));
    for (R_xlen_t k = 0; k < ov.size(); ++k) {
      ov[k] = s.threshold_ov[bi][static_cast<std::size_t>(k)] + 1;
      lev[k] = s.threshold_level[bi][static_cast<std::size_t>(k)];
    }
    threshold_ov[b] = ov;
    threshold_level[b] = lev;
    NACOV[b] = Rcpp::wrap(s.NACOV[bi]);
    W_dwls[b] = Rcpp::wrap(s.W_dwls[bi]);
    W_wls[b] = Rcpp::wrap(s.W_wls[bi]);
    moment_influence[b] = bi < s.moment_influence.size()
        ? Rcpp::wrap(s.moment_influence[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    gamma_diag_influence[b] = bi < s.gamma_diag_influence.size()
        ? Rcpp::wrap(s.gamma_diag_influence[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    gamma_full_influence[b] = bi < s.gamma_full_influence.size()
        ? Rcpp::wrap(s.gamma_full_influence[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    raw_data[b] = bi < s.raw_data.size()
        ? Rcpp::wrap(s.raw_data[bi])
        : Rcpp::wrap(Eigen::MatrixXd(0, 0));
    nobs[b] = static_cast<int>(s.n_obs[bi]);
    n_levels[b] = Rcpp::wrap(s.n_levels[bi]);
  }
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["R"] = R,
      Rcpp::_["mean"] = mean,
      Rcpp::_["ordered_mask"] = ordered_mask,
      Rcpp::_["thresholds"] = thresholds,
      Rcpp::_["threshold_ov"] = threshold_ov,
      Rcpp::_["threshold_level"] = threshold_level,
      Rcpp::_["moments"] = moments,
      Rcpp::_["NACOV"] = NACOV,
      Rcpp::_["W_dwls"] = W_dwls,
      Rcpp::_["W_wls"] = W_wls,
      Rcpp::_["moment_influence"] = moment_influence,
      Rcpp::_["gamma_diag_influence"] = gamma_diag_influence,
      Rcpp::_["gamma_full_influence"] = gamma_full_influence,
      Rcpp::_["raw_data"] = raw_data,
      Rcpp::_["nobs"] = nobs,
      Rcpp::_["n_levels"] = n_levels);
  out.attr("class") = Rcpp::CharacterVector::create("magmaan_mixed_ordinal_data", "list");
  return out;
}

magmaan::data::RawData fiml_raw_from_arg(const lvm::MatrixRep& rep, SEXP raw_data) {
  SEXP X_arg = raw_data;
  SEXP mask_arg = R_NilValue;
  if (TYPEOF(raw_data) == VECSXP) {
    Rcpp::List rd(raw_data);
    if (rd.containsElementNamed("X")) {
      X_arg = rd["X"];
      if (rd.containsElementNamed("mask")) mask_arg = rd["mask"];
    }
  }

  const std::size_t n_blocks = rep.dims.size();
  magmaan::data::RawData raw;
  raw.X.reserve(n_blocks);
  raw.mask.reserve(n_blocks);

  for (std::size_t b = 0; b < n_blocks; ++b) {
    Rcpp::NumericMatrix Xb = block_matrix(X_arg, b, n_blocks, "raw_data$X");
    const std::vector<int> perm = perm_for_cols(Xb, rep.ov_names[b], "raw_data$X");
    const int n = Xb.nrow();
    const int p = static_cast<int>(perm.size());

    Rcpp::LogicalMatrix Mb;
    const bool has_mask = !Rf_isNull(mask_arg);
    if (has_mask) {
      Mb = block_mask_matrix(mask_arg, b, n_blocks, "raw_data$mask");
      if (Mb.nrow() != n || Mb.ncol() != Xb.ncol())
        Rcpp::stop("magmaan: raw_data$mask block %d has shape %dx%d but raw_data$X has %dx%d",
                   static_cast<int>(b + 1), Mb.nrow(), Mb.ncol(), Xb.nrow(), Xb.ncol());
    }

    Eigen::MatrixXd X(n, p);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);
    for (int r = 0; r < n; ++r) {
      for (int k = 0; k < p; ++k) {
        const int src = perm[static_cast<std::size_t>(k)];
        const double x = Xb(r, src);
        bool observed = has_mask
            ? (Mb(r, src) != NA_LOGICAL && Mb(r, src) != 0)
            : std::isfinite(x);
        if (observed && !std::isfinite(x)) {
          Rcpp::stop("magmaan: raw_data$mask marks a non-finite value as observed "
                     "in block %d, row %d", static_cast<int>(b + 1), r + 1);
        }
        M(r, k) = static_cast<std::uint8_t>(observed ? 1 : 0);
        X(r, k) = observed ? x : std::numeric_limits<double>::quiet_NaN();
      }
    }

    raw.X.push_back(std::move(X));
    raw.mask.push_back(std::move(M));
  }
  return raw;
}

magmaan::data::RawData complete_raw_from_arg(const lvm::MatrixRep& rep,
                                             SEXP raw_data) {
  SEXP X_arg = raw_data;
  if (TYPEOF(raw_data) == VECSXP) {
    Rcpp::List rd(raw_data);
    if (rd.containsElementNamed("X")) X_arg = rd["X"];
  }

  const std::size_t n_blocks = rep.dims.size();
  magmaan::data::RawData raw;
  raw.X.reserve(n_blocks);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    Rcpp::NumericMatrix Xb = block_matrix(X_arg, b, n_blocks, "raw_data$X");
    const std::vector<int> perm = perm_for_cols(Xb, rep.ov_names[b], "raw_data$X");
    Eigen::MatrixXd X = reorder_data_cols(Xb, perm);
    validate_finite_matrix(X, "raw data", b);
    raw.X.push_back(std::move(X));
  }
  return raw;
}

Rcpp::List fiml_raw_to_r(const magmaan::data::RawData& raw,
                         const std::vector<std::vector<std::string>>& ov_names,
                         const std::vector<std::string>& group_labels) {
  const R_xlen_t nb = static_cast<R_xlen_t>(raw.X.size());
  Rcpp::List X_out(nb), M_out(nb);
  Rcpp::IntegerVector nobs(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::size_t bi = static_cast<std::size_t>(b);
    Rcpp::NumericMatrix Xb = Rcpp::wrap(raw.X[bi]);
    Rcpp::LogicalMatrix Mb(raw.mask[bi].rows(), raw.mask[bi].cols());
    for (R_xlen_t r = 0; r < Mb.nrow(); ++r)
      for (R_xlen_t c = 0; c < Mb.ncol(); ++c)
        Mb(r, c) = raw.mask[bi](r, c) != 0;
    Rcpp::CharacterVector nm = Rcpp::wrap(ov_names[bi]);
    Xb.attr("dimnames") = Rcpp::List::create(R_NilValue, nm);
    Mb.attr("dimnames") = Rcpp::List::create(R_NilValue, nm);
    X_out[b] = Xb;
    M_out[b] = Mb;
    nobs[b] = static_cast<int>(raw.X[bi].rows());
  }
  if (!group_labels.empty() && static_cast<R_xlen_t>(group_labels.size()) == nb) {
    Rcpp::CharacterVector gl = Rcpp::wrap(group_labels);
    X_out.attr("names") = gl;
    M_out.attr("names") = gl;
  }
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["X"] = X_out,
      Rcpp::_["mask"] = M_out,
      Rcpp::_["ov_names"] = names_list(ov_names),
      Rcpp::_["group_labels"] = Rcpp::wrap(group_labels),
      Rcpp::_["nobs"] = nobs);
  out.attr("class") = Rcpp::CharacterVector::create("magmaan_fiml_data", "list");
  return out;
}

Rcpp::XPtr<FimlPack> fiml_pack_xptr(FimlPack pack) {
  Rcpp::XPtr<FimlPack> xp(new FimlPack(std::move(pack)), true);
  xp.attr("class") =
      Rcpp::CharacterVector::create("magmaan_fiml_pack", "externalptr");
  return xp;
}

Rcpp::XPtr<FimlH1> fiml_h1_xptr(FimlH1 h1) {
  Rcpp::XPtr<FimlH1> xp(new FimlH1(std::move(h1)), true);
  xp.attr("class") =
      Rcpp::CharacterVector::create("magmaan_fiml_h1", "externalptr");
  return xp;
}

const FimlPack* fiml_pack_ptr_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("fiml_pack")) return nullptr;
  SEXP xp = fit["fiml_pack"];
  if (Rf_isNull(xp)) return nullptr;
  if (TYPEOF(xp) != EXTPTRSXP) {
    Rcpp::stop("magmaan: fit$fiml_pack is not an external pointer");
  }
  void* addr = R_ExternalPtrAddr(xp);
  if (addr == nullptr) Rcpp::stop("magmaan: fit$fiml_pack is null");
  return static_cast<const FimlPack*>(addr);
}

const FimlH1* fiml_h1_ptr_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("fiml_h1")) return nullptr;
  SEXP xp = fit["fiml_h1"];
  if (Rf_isNull(xp)) return nullptr;
  if (TYPEOF(xp) != EXTPTRSXP) {
    Rcpp::stop("magmaan: fit$fiml_h1 is not an external pointer");
  }
  void* addr = R_ExternalPtrAddr(xp);
  if (addr == nullptr) Rcpp::stop("magmaan: fit$fiml_h1 is null");
  return static_cast<const FimlH1*>(addr);
}

const FimlPack& fiml_pack_for_fit(Rcpp::List fit,
                                  const magmaan::data::RawData& raw,
                                  std::unique_ptr<FimlPack>& owned) {
  if (const FimlPack* pack = fiml_pack_ptr_from_fit(fit)) return *pack;
  auto pack_or = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack_or.has_value()) stop_fit(pack_or.error());
  owned = std::make_unique<FimlPack>(std::move(*pack_or));
  return *owned;
}

const FimlH1& fiml_h1_for_fit(Rcpp::List fit,
                              const magmaan::data::RawData& raw,
                              const FimlPack& pack,
                              std::unique_ptr<FimlH1>& owned) {
  if (const FimlH1* h1 = fiml_h1_ptr_from_fit(fit)) return *h1;
  auto h1_or = magmaan::estimate::fiml::fiml_h1_moments(raw, pack);
  if (!h1_or.has_value()) stop_fit(h1_or.error());
  owned = std::make_unique<FimlH1>(std::move(*h1_or));
  return *owned;
}

// Shared Stage-1 saturated FIML moments (the EM moments + H/J/acov = Γ_mis):
// reuse the fit$stage1 reconstruction when present (ML2S, via
// magmaanr::saturated_from_stage1), otherwise compute once. Mirrors
// fiml_pack_for_fit / fiml_h1_for_fit so FMG, two-stage SB, and the nested LRT
// all consume one saturated build instead of three.
const SaturatedMoments& fiml_saturated_for_fit(
    Rcpp::List fit, const magmaan::data::RawData& raw, const FimlPack& pack,
    const FimlH1& h1, std::unique_ptr<SaturatedMoments>& owned) {
  owned = std::make_unique<SaturatedMoments>();
  if (magmaanr::saturated_from_stage1_with_information(fit, *owned))
    return *owned;
  auto sm_or = magmaan::estimate::fiml::saturated_em_moments(raw, pack, h1);
  if (!sm_or.has_value()) stop_post(sm_or.error());
  *owned = std::move(*sm_or);
  return *owned;
}

Rcpp::List fiml_fit_result(Ctx& ctx,
                           const magmaan::data::RawData& raw,
                           const magmaan::estimate::Estimates& est,
                           const magmaan::spec::Starts* starts) {
  Rcpp::List out = fit_result(ctx, est, starts, "FIML");
  out["fiml"] = true;
  out["raw_data"] = fiml_raw_to_r(raw, ctx.rep.ov_names, ctx.names.group_labels);
  return out;
}

magmaan::data::OrdinalStats ordinal_stats_from_arg(Rcpp::List x) {
  const char* what = "ordinal_stats";
  for (const char* nm : {"R", "thresholds", "threshold_ov", "threshold_level",
                         "NACOV", "W_dwls", "W_wls", "nobs", "n_levels"}) {
    if (!x.containsElementNamed(nm)) Rcpp::stop("magmaan: %s is missing $%s", what, nm);
  }
  Rcpp::List Rl(x["R"]), thl(x["thresholds"]), ovl(x["threshold_ov"]),
      levl(x["threshold_level"]), NAl(x["NACOV"]),
      Wdl(x["W_dwls"]), Wfl(x["W_wls"]),
      nlevl(x["n_levels"]);
  Rcpp::IntegerVector nobs(x["nobs"]);
  // The estimated-weight (Hall-Inoue) score-test path needs the per-case
  // influence + integer data; read them through when present (the high-level
  // DWLS path stores them).
  const bool has_mi = x.containsElementNamed("moment_influence");
  Rcpp::List mil = has_mi ? Rcpp::List(x["moment_influence"]) : Rcpp::List();
  const bool has_id = x.containsElementNamed("int_data");
  Rcpp::List idl = has_id ? Rcpp::List(x["int_data"]) : Rcpp::List();
  const R_xlen_t nb = Rl.size();
  magmaan::data::OrdinalStats out;
  out.R.reserve(static_cast<std::size_t>(nb));
  out.thresholds.reserve(static_cast<std::size_t>(nb));
  out.threshold_ov.reserve(static_cast<std::size_t>(nb));
  out.threshold_level.reserve(static_cast<std::size_t>(nb));
  out.NACOV.reserve(static_cast<std::size_t>(nb));
  out.W_dwls.reserve(static_cast<std::size_t>(nb));
  out.W_wls.reserve(static_cast<std::size_t>(nb));
  out.n_obs.reserve(static_cast<std::size_t>(nb));
  out.n_levels.reserve(static_cast<std::size_t>(nb));
  for (R_xlen_t b = 0; b < nb; ++b) {
    out.R.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Rl[b])));
    out.thresholds.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(thl[b])));
    Rcpp::IntegerVector ov(ovl[b]), lev(levl[b]);
    std::vector<std::int32_t> ov0(static_cast<std::size_t>(ov.size()));
    std::vector<std::int32_t> lev0(static_cast<std::size_t>(lev.size()));
    for (R_xlen_t k = 0; k < ov.size(); ++k) {
      ov0[static_cast<std::size_t>(k)] = ov[k] - 1;
      lev0[static_cast<std::size_t>(k)] = lev[k];
    }
    out.threshold_ov.push_back(std::move(ov0));
    out.threshold_level.push_back(std::move(lev0));
    out.NACOV.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(NAl[b])));
    out.W_dwls.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Wdl[b])));
    out.W_wls.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Wfl[b])));
    if (has_mi)
      out.moment_influence.push_back(
          Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(mil[b])));
    if (has_id)
      out.int_data.push_back(
          Rcpp::as<Eigen::MatrixXi>(Rcpp::IntegerMatrix(idl[b])));
    out.n_obs.push_back(static_cast<std::int64_t>(nobs[b]));
    out.n_levels.push_back(Rcpp::as<std::vector<std::int32_t>>(Rcpp::IntegerVector(nlevl[b])));
  }
  if (x.containsElementNamed("sampling_moment_influence")) {
    Rcpp::List rows(x["sampling_moment_influence"]);
    if (rows.size() != nb) Rcpp::stop("sampling_moment_influence has incompatible block count");
    for (R_xlen_t b = 0; b < nb; ++b)
      out.sampling_moment_influence.push_back(Rcpp::as<Eigen::MatrixXd>(rows[b]));
  }
  return out;
}

void attach_ordinal_parameter_values(Rcpp::List& out, const Ctx& ctx,
                                     const magmaan::estimate::Estimates& est,
                                     const char* parameterization) {
  auto values = magmaan::estimate::ordinal_parameter_values(
      ctx.pt, ctx.rep, est.theta,
      ordinal_parameterization_from_string(parameterization));
  if (!values) stop_post(values.error());
  Rcpp::DataFrame pt = Rcpp::as<Rcpp::DataFrame>(out["partable"]);
  Rcpp::NumericVector estimates = pt["est"];
  for (Eigen::Index i = 0; i < values->size(); ++i) estimates[i] = (*values)(i);
  pt["est"] = estimates;
  out["partable"] = pt;
}

Rcpp::List ordinal_fit_result(Ctx& ctx,
                              const magmaan::data::OrdinalStats& stats,
                              const magmaan::estimate::Estimates& est,
                              const magmaan::spec::Starts* starts,
                              const char* estimator,
                              const char* parameterization ) {
  Rcpp::List out = fit_result(ctx, est, starts, estimator);
  attach_ordinal_parameter_values(out, ctx, est, parameterization);
  out["ordinal"] = true;
  out["parameterization"] = parameterization;
  // Carry the group.equal families on the partable so the nested ordinal LR
  // test (which rebuilds each structure via ctx_from_fit) re-applies the
  // Wu-Estabrook release; from_lavaan_partable would otherwise drop them.
  if (!ctx.pt.group_equal.empty()) {
    Rcpp::DataFrame pt_out = Rcpp::as<Rcpp::DataFrame>(out["partable"]);
    stamp_group_equal_attr(pt_out, ctx.pt.group_equal);
    out["partable"] = pt_out;
  }
  Rcpp::List stats_r = ordinal_stats_to_r(stats);
  out["ordinal_stats"] = stats_r;
  out["thresholds"] = stats_r["thresholds"];
  out["polychoric"] = stats_r["R"];
  if (est.association) {
    const auto& a = *est.association;
    out["association"] = Rcpp::List::create(
        Rcpp::_["n_moments"] = a.n_moments,
        Rcpp::_["n_coordinates"] = a.n_coordinates,
        Rcpp::_["rank"] = a.rank, Rcpp::_["df"] = a.df);
    out["npar_active"] = a.n_coordinates;
    out["df"] = a.df;
    out["stage1_policy"] = "unchanged_polychoric";
    out["covariance_policy"] = "unrestricted";
    out["composition"] = Rcpp::List::create(
        Rcpp::_["moment_source"] = "polychoric",
        Rcpp::_["moment_target"] = "correlation",
        Rcpp::_["discrepancy"] = "ML",
        Rcpp::_["covariance_domain"] = "unrestricted",
        Rcpp::_["model_penalty"] = "none",
        Rcpp::_["thresholds"] = "saturated_stage1",
        Rcpp::_["inference"] = "not_validated");
  }
  return out;
}

Rcpp::List mixed_ordinal_fit_result(
    Ctx& ctx,
    const magmaan::data::MixedOrdinalStats& stats,
    const magmaan::estimate::Estimates& est,
    const magmaan::spec::Starts* starts,
    const char* estimator,
    const char* parameterization ) {
  Rcpp::List out = fit_result(ctx, est, starts, estimator);
  attach_ordinal_parameter_values(out, ctx, est, parameterization);
  out["mixed_ordinal"] = true;
  out["parameterization"] = parameterization;
  out["mixed_ordinal_stats"] = mixed_ordinal_stats_to_r(stats);
  return out;
}

magmaan::estimate::frontier::MultiInfoPenaltyOptions multiinfo_options_from(
    double eta, Rcpp::Nullable<Rcpp::NumericVector> weight,
    const std::string& target) {
  magmaan::estimate::frontier::MultiInfoPenaltyOptions out;
  out.eta = eta;
  if (target == "joint") {
    out.target = magmaan::estimate::frontier::PenaltyTarget::Joint;
  } else if (target == "determinacy") {
    out.target = magmaan::estimate::frontier::PenaltyTarget::Determinacy;
  } else {
    Rcpp::stop("magmaan: `target` must be \"joint\" or \"determinacy\"");
  }
  if (weight.isNotNull()) {
    Rcpp::NumericVector w(weight.get());
    if (w.size() != 1) Rcpp::stop("magmaan: `weight` must be NULL or a single number");
    out.weight = w[0];
  }
  return out;
}

// Per-variable terms keyed by the extended latent / observed names, plus the
// scalar pieces of the penalized fit. Joint: log(1 − R²_i) per equation.
// Determinacy: log(1 − ρ²_j) per genuine latent, ρ²_j its factor-score
// determinacy, plus per-block TC(η) and I(η; y).
Rcpp::List multiinfo_penalty_to_r(
    const Ctx& ctx, const magmaan::estimate::frontier::PenalizedFit& fit) {
  const auto& report = fit.penalty;
  const bool determinacy =
      report.target == magmaan::estimate::frontier::PenaltyTarget::Determinacy;
  const R_xlen_t nt = static_cast<R_xlen_t>(report.terms.size());
  Rcpp::IntegerVector block(nt);
  Rcpp::CharacterVector kind(nt), variable(nt);
  Rcpp::NumericVector log_term(nt), r2(nt);
  for (R_xlen_t i = 0; i < nt; ++i) {
    const auto& t = report.terms[static_cast<std::size_t>(i)];
    const auto b = static_cast<std::size_t>(t.block);
    const auto j = static_cast<std::size_t>(t.index);
    block[i] = t.block + 1;
    kind[i] = t.latent ? "latent" : "observed";
    const auto& names = t.latent ? ctx.rep.lv_names : ctx.rep.ov_names;
    variable[i] = b < names.size() && j < names[b].size() ? names[b][j] : "";
    log_term[i] = t.log_one_minus_r2;
    r2[i] = 1.0 - std::exp(t.log_one_minus_r2);
  }
  Rcpp::DataFrame terms = Rcpp::DataFrame::create(
      Rcpp::_["block"] = block, Rcpp::_["kind"] = kind,
      Rcpp::_["variable"] = variable,
      Rcpp::_[determinacy ? "log_one_minus_rho2" : "log_one_minus_r2"] = log_term,
      Rcpp::_[determinacy ? "rho2" : "r2"] = r2,
      Rcpp::_["stringsAsFactors"] = false);
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["type"] = determinacy ? "determinacy" : "multiinfo",
      Rcpp::_["weight"] = fit.weight,
      Rcpp::_["value"] = report.value,
      Rcpp::_["penalized_fmin"] = fit.penalized_fmin,
      Rcpp::_["n_total"] = fit.n_total,
      Rcpp::_["recursive"] = report.recursive,
      Rcpp::_["start_repaired"] = fit.start_repaired,
      Rcpp::_["block_value"] = Rcpp::wrap(report.block_value),
      Rcpp::_["residual_log_det_corr"] = Rcpp::wrap(report.residual_log_det_corr),
      Rcpp::_["terms"] = terms);
  if (determinacy) {
    out["total_correlation"] = Rcpp::wrap(report.total_correlation);
    out["mutual_information"] = Rcpp::wrap(report.mutual_information);
  }
  return out;
}

bool is_ml2s_estimator_label(const std::string& estimator) {
  return estimator == "ML2S" || estimator.rfind("ML2S_", 0) == 0;
}

bool ml2s_weight_needs_raw_ij(magmaan::estimate::fiml::TwoStageWeight kind) {
  using magmaan::estimate::fiml::TwoStageWeight;
  return kind == TwoStageWeight::Dwls || kind == TwoStageWeight::Adf ||
         kind == TwoStageWeight::Dls;
}

magmaan::estimate::fiml::TwoStageDlsOptions ml2s_dls_options_from_fit(
    Rcpp::List fit) {
  magmaan::estimate::fiml::TwoStageDlsOptions dls;
  if (fit.containsElementNamed("stage2_dls_a")) {
    dls.a = Rcpp::as<double>(fit["stage2_dls_a"]);
  }
  return dls;
}

// The recorded Stage-2 weight of an ML2S fit (fit$stage2_weight and
// fit$stage2_dls_a). An explicit `stage2_weight` or `dls_a` argument must agree
// with the record, because a different weight's influence does not describe
// the fit's estimate; fits without a record use the argument (default NT).
void ml2s_recorded_stage2(Rcpp::List fit, SEXP stage2_weight_arg,
                          SEXP dls_a_arg, const char* call,
                          magmaan::estimate::fiml::TwoStageWeight& kind,
                          magmaan::estimate::fiml::TwoStageDlsOptions& dls) {
  const bool recorded = fit.containsElementNamed("stage2_weight") &&
                        !Rf_isNull(fit["stage2_weight"]);
  const std::string record =
      recorded ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
  kind = magmaanr::two_stage_weight_from_arg(record);
  if (!Rf_isNull(stage2_weight_arg)) {
    const auto given = magmaanr::two_stage_weight_from_arg(
        Rcpp::as<std::string>(stage2_weight_arg));
    if (recorded && given != kind) {
      Rcpp::stop("magmaan: %s: stage2_weight differs from the fit's recorded "
                 "Stage-2 weight '%s'; omit it", call, record.c_str());
    }
    kind = given;
  }
  dls = ml2s_dls_options_from_fit(fit);
  if (!Rf_isNull(dls_a_arg)) {
    const double given = Rcpp::as<double>(dls_a_arg);
    if (fit.containsElementNamed("stage2_dls_a") &&
        !Rf_isNull(fit["stage2_dls_a"]) &&
        kind == magmaan::estimate::fiml::TwoStageWeight::Dls &&
        given != dls.a) {
      Rcpp::stop("magmaan: %s: dls_a differs from the fit's recorded "
                 "Stage-2 DLS weight a = %g; omit it", call, dls.a);
    }
    dls.a = given;
  }
}

// The fitting weight of a continuous WLS-computed fit. fit_model() and
// estimate() record it in fit$W (DWLS, ADF, DLS and supplied weights). An
// explicit `weight` must be that record or a common positive multiple of it,
// which has the same minimizer (the weight-scale transport checks use one):
// score tests and sandwiches evaluated with any other weight do not describe
// this fit's estimate.
SEXP fitting_weight_arg(Rcpp::List fit, SEXP weight, std::size_t n_blocks,
                        const char* call) {
  if (!fit.containsElementNamed("W") || Rf_isNull(fit["W"])) return weight;
  if (Rf_isNull(weight)) return fit["W"];
  const auto given = wls_dense_from_arg(weight, n_blocks);
  const auto recorded = wls_dense_from_arg(fit["W"], n_blocks);
  double cross = 0.0;
  double norm2 = 0.0;
  bool same_shape = true;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    same_shape = same_shape && given[b].rows() == recorded[b].rows() &&
                 given[b].cols() == recorded[b].cols();
    if (!same_shape) break;
    cross += (given[b].array() * recorded[b].array()).sum();
    norm2 += recorded[b].squaredNorm();
  }
  const double scale = norm2 > 0.0 ? cross / norm2 : 0.0;
  bool proportional = same_shape && scale > 0.0;
  for (std::size_t b = 0; proportional && b < n_blocks; ++b) {
    const Eigen::MatrixXd expected = scale * recorded[b];
    proportional = (given[b] - expected).norm() <=
                   1e-8 * std::max(1.0, expected.norm());
  }
  if (!proportional) {
    Rcpp::stop("magmaan: %s: `weight` is not the fitting weight recorded in "
               "fit$W (or a positive multiple of it); omit `weight`", call);
  }
  return weight;
}

// Estimation weight for a continuous LS fit (empty for ULS; ML carries none and
// is handled by the caller via the weight-free overloads). WLS-computed fits
// (WLS, DWLS, DLS, supplied W) use the recorded fit$W unless the fit predates
// that record, in which case the caller passes `weight`.
magmaan::estimate::gmm::Weight continuous_ls_weight(
    Rcpp::List fit, const Ctx& ctx, const magmaan::estimate::Estimates& est,
    const std::string& estimator, SEXP weight, const char* call) {
  if (estimator == "ULS") return magmaan::estimate::gmm::Weight{};
  if (estimator == "GLS") {
    auto ev_or = magmaan::model::ModelEvaluator::build(ctx.pt, ctx.rep);
    if (!ev_or.has_value()) stop_model(ev_or.error());
    auto w_or =
        magmaan::estimate::gmm::normal_theory_weight(*ev_or, ctx.samp, est.theta);
    if (!w_or.has_value()) stop_fit(w_or.error());
    return *w_or;
  }
  if (estimator == "WLS") {
    const std::size_t n_blocks = ctx.samp.S.size();
    SEXP w = fitting_weight_arg(fit, weight, n_blocks, call);
    if (Rf_isNull(w))
      Rcpp::stop("magmaan: WLS %s need the fitting weight; the fit records "
                 "none in fit$W, so pass `weight`", call);
    return wls_from_arg(w, n_blocks);
  }
  Rcpp::stop("magmaan: robust %s are not exposed for estimator '%s'", call,
             estimator.c_str());
}

RecordedMomentWeight recorded_moment_weight(Rcpp::List fit,
                                            const std::string& estimator) {
  using Kind = magmaan::estimate::gmm::FixedWeightKind;
  std::string key;
  if (fit.containsElementNamed("moment_weight") &&
      !Rf_isNull(fit["moment_weight"])) {
    key = Rcpp::as<std::string>(fit["moment_weight"]);
  } else if (fit.containsElementNamed("composition") &&
             TYPEOF(fit["composition"]) == VECSXP) {
    Rcpp::List composition = fit["composition"];
    if (composition.containsElementNamed("weight") &&
        !Rf_isNull(composition["weight"])) {
      key = Rcpp::as<std::string>(composition["weight"]);
    }
  }
  if (key.empty()) {
    if (estimator == "ULS") key = "uls";
    else if (estimator == "GLS") key = "nt";
    else if (estimator == "WLS") key = "adf";
  }
  RecordedMomentWeight out;
  if (key == "uls") out.kind = Kind::Uls;
  else if (key == "nt") out.kind = Kind::Nt;
  else if (key == "dwls") out.kind = Kind::Dwls;
  else if (key == "adf" || key == "wls") out.kind = Kind::Wls;
  else if (key == "dls") out.kind = Kind::Dls;
  else if (key == "custom") {
    out.kind = Kind::Wls;
    out.supplied = true;
  } else {
    Rcpp::stop("magmaan: the fit records an unknown moment weight '%s'",
               key.c_str());
  }
  if (fit.containsElementNamed("stage2_dls_a") &&
      !Rf_isNull(fit["stage2_dls_a"])) {
    out.dls_a = Rcpp::as<double>(fit["stage2_dls_a"]);
  }
  return out;
}

// The estimated-weight IJ mode for the fit's recorded recipe (and its DLS
// mixing weight). A supplied weight has no recipe, so the C++ resolver refuses
// it with UnsupportedInference.
magmaan::estimate::ContinuousLsIJWeightMode continuous_ij_mode_for_fit(
    Rcpp::List fit, const std::string& estimator,
    magmaan::estimate::gmm::FixedWeightOptions* dls_opts) {
  const RecordedMomentWeight recipe = recorded_moment_weight(fit, estimator);
  auto mode_or =
      magmaan::estimate::continuous_ls_ij_mode_for(recipe.kind, recipe.supplied);
  if (!mode_or.has_value()) stop_post(mode_or.error());
  if (dls_opts != nullptr) dls_opts->a = recipe.dls_a;
  return *mode_or;
}

magmaan::inference::frontier::ScoreFlipMultiplier
score_flip_multiplier_from_string(const std::string& multiplier) {
  using Multiplier = magmaan::inference::frontier::ScoreFlipMultiplier;
  if (multiplier == "rademacher") return Multiplier::Rademacher;
  if (multiplier == "mammen") return Multiplier::Mammen;
  if (multiplier == "two-point") return Multiplier::TwoPoint;
  if (multiplier == "gaussian") return Multiplier::Gaussian;
  if (multiplier == "centered-exponential") {
    return Multiplier::CenteredExponential;
  }
  Rcpp::stop("magmaan: score_flip_test multiplier must be 'rademacher', "
             "'mammen', 'two-point', 'gaussian', or "
             "'centered-exponential'");
  return Multiplier::Rademacher;
}

magmaan::inference::frontier::ScoreFlipSensitivity
score_flip_sensitivity_from_string(const std::string& sensitivity) {
  using Sensitivity =
      magmaan::inference::frontier::ScoreFlipSensitivity;
  if (sensitivity == "expected") {
    return Sensitivity::ExpectedInformation;
  }
  if (sensitivity == "observed") {
    return Sensitivity::ObservedInformation;
  }
  if (sensitivity == "observed-shrink-light") {
    return Sensitivity::ObservedInformationLightShrinkage;
  }
  if (sensitivity == "observed-shrink-sqrt") {
    return Sensitivity::ObservedInformationSqrtShrinkage;
  }
  if (sensitivity == "observed-h1") {
    return Sensitivity::SaturatedObservedInformation;
  }
  Rcpp::stop("magmaan: score_flip_test sensitivity must be 'expected', "
             "'observed', 'observed-shrink-light', "
             "'observed-shrink-sqrt', or 'observed-h1'");
  return Sensitivity::ExpectedInformation;
}

magmaan::inference::frontier::GlobalScoreFlipOptions::Metric
global_score_metric_from_string(const std::string& metric) {
  using Metric =
      magmaan::inference::frontier::GlobalScoreFlipOptions::Metric;
  if (metric == "expected") return Metric::ExpectedInformation;
  if (metric == "observed") return Metric::ObservedInformation;
  if (metric == "observed-h1") {
    return Metric::SaturatedObservedInformation;
  }
  Rcpp::stop("magmaan: global_score_flip_test metric must be 'expected', "
             "'observed', or 'observed-h1'");
  return Metric::ExpectedInformation;
}

}  // namespace magmaanr::fitglue
