#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

namespace {

Rcpp::DataFrame noniter_mi_table_df(
    const magmaan::robust::frontier::NonIterativeModificationIndexTable& tab,
    const magmaan::spec::LatentNames& names);
std::string noniter_lower(std::string s);
magmaan::estimate::frontier::NonIterativeEstimator noniter_which(const std::string& s);
const char* noniter_map_name(magmaan::estimate::frontier::NonIterativeEstimator which);
bool noniter_estimator_auto(const std::string& estimator);
bool noniter_has_recorded_map(const Rcpp::List& fit);
magmaan::estimate::frontier::NonIterativeEstimator
noniter_recorded_map(const Rcpp::List& fit);
void noniter_require_recorded_map(const Rcpp::List& fit,
                                  magmaan::estimate::frontier::NonIterativeEstimator which,
                                  const char* role);
bool noniter_is_restricted_fit(const Rcpp::List& fit);
magmaan::estimate::frontier::NonIterativeEstimator
noniter_which_for_fit(const Rcpp::List& fit, const std::string& estimator);
magmaan::estimate::frontier::CommunalityMethod communality_which(const std::string& s);
magmaan::estimate::frontier::CommunalityMethod
noniter_comm_for_fit(const Rcpp::List& fit);
magmaan::estimate::frontier::CompositeWeight composite_which(const std::string& s);
magmaan::estimate::frontier::AdmissibilityPolicy
admissibility_which(const std::string& s);
magmaan::estimate::frontier::AdmissibilityConfig
admissibility_config(const std::string& policy, double margin, double beta0,
                     double rate);
magmaan::estimate::frontier::AdmissibilityConfig
noniter_admissibility_for_fit(const Rcpp::List& fit);
magmaan::estimate::frontier::ScoreConditioningPolicy
score_conditioning_which(const std::string& s);
magmaan::estimate::frontier::ScoreConditioningConfig
score_conditioning_config(const std::string& policy, double floor0,
                          double rate);
magmaan::estimate::frontier::HConditioningPolicy
h_conditioning_which(const std::string& s);
magmaan::estimate::frontier::HConditioningConfig
h_conditioning_config(const std::string& policy, double floor0, double rate);
magmaan::estimate::frontier::HConditioningConfig
noniter_h_conditioning_for_fit(const Rcpp::List& fit);
void noniter_require_raw_h_conditioning(const Rcpp::List& fit);
magmaan::estimate::frontier::ScoreConditioningConfig
noniter_score_conditioning_for_fit(const Rcpp::List& fit);
bool same_score_conditioning(
    const magmaan::estimate::frontier::ScoreConditioningConfig& lhs,
    const magmaan::estimate::frontier::ScoreConditioningConfig& rhs);
Rcpp::DataFrame score_conditioning_diagnostics_df(
    const std::vector<magmaan::estimate::frontier::ScoreConditioningDiagnostics>&
        diagnostics);
Rcpp::DataFrame h_conditioning_diagnostics_df(
    const std::vector<magmaan::estimate::frontier::HConditioningDiagnostics>&
        diagnostics);
magmaan::estimate::frontier::CompositeWeight
noniter_comp_for_fit(const Rcpp::List& fit);
magmaan::estimate::frontier::CompositeWeight
noniter_resolve_composite(
    magmaan::estimate::frontier::NonIterativeEstimator which,
    magmaan::estimate::frontier::CompositeWeight composite);
magmaan::robust::frontier::Discrepancy noniter_disc(const std::string& s);
magmaan::post_expected<magmaan::robust::frontier::NonIterativeInference>
noniter_inference_dispatch(Ctx& ctx, const magmaan::estimate::Estimates& est,
                           magmaan::estimate::frontier::NonIterativeEstimator which,
                           magmaan::robust::frontier::Discrepancy disc,
                           const std::string& gamma, SEXP data,
                           bool restricted = false,
                           magmaan::estimate::frontier::CommunalityMethod comm =
                               magmaan::estimate::frontier::CommunalityMethod::TriadWls,
                           magmaan::estimate::frontier::CompositeWeight composite =
                               magmaan::estimate::frontier::CompositeWeight::EstimatorDefault,
                           magmaan::estimate::frontier::AdmissibilityConfig admissibility = {},
                           magmaan::estimate::frontier::ScoreConditioningConfig
                               score_conditioning = {});
magmaan::post_expected<magmaan::robust::frontier::NonIterativeSE>
noniter_se_dispatch(Ctx& ctx, const magmaan::estimate::Estimates& est,
                    magmaan::estimate::frontier::NonIterativeEstimator which,
                    const std::string& gamma, SEXP data,
                    bool restricted = false,
                    magmaan::estimate::frontier::CommunalityMethod comm =
                        magmaan::estimate::frontier::CommunalityMethod::TriadWls,
                    magmaan::estimate::frontier::CompositeWeight composite =
                        magmaan::estimate::frontier::CompositeWeight::EstimatorDefault,
                    magmaan::estimate::frontier::AdmissibilityConfig admissibility = {},
                    magmaan::estimate::frontier::ScoreConditioningConfig
                        score_conditioning = {});
Rcpp::List wrap_noniter_inference(const magmaan::robust::frontier::NonIterativeInference& inf);
Rcpp::List wrap_noniter_se(const magmaan::robust::frontier::NonIterativeSE& se);
Rcpp::List wrap_noniter_diff(const magmaan::robust::frontier::NonIterativeDiffTest& d);
magmaan::post_expected<magmaan::robust::frontier::GroupedNonIterativeInference>
noniter_grouped_dispatch(Ctx& ctx, const magmaan::estimate::Estimates& est,
                         magmaan::estimate::frontier::NonIterativeEstimator which,
                         magmaan::robust::frontier::Discrepancy disc,
                         const std::string& gamma, SEXP data,
                         bool restricted,
                         magmaan::estimate::frontier::CommunalityMethod comm =
                             magmaan::estimate::frontier::CommunalityMethod::TriadWls,
                         magmaan::estimate::frontier::CompositeWeight composite =
                             magmaan::estimate::frontier::CompositeWeight::EstimatorDefault,
                         magmaan::estimate::frontier::AdmissibilityConfig admissibility = {},
                         magmaan::estimate::frontier::ScoreConditioningConfig
                             score_conditioning = {});
Rcpp::List
wrap_noniter_grouped(const magmaan::robust::frontier::GroupedNonIterativeInference& inf);

Rcpp::DataFrame noniter_mi_table_df(
    const magmaan::robust::frontier::NonIterativeModificationIndexTable& tab,
    const magmaan::spec::LatentNames& names) {
  const R_xlen_t n = static_cast<R_xlen_t>(tab.rows.size());
  Rcpp::CharacterVector kind(n), op(n), lhs(n), rhs(n);
  Rcpp::IntegerVector row(n), group(n);
  Rcpp::NumericVector score_raw(n), var_raw(n), z_raw(n), mi_raw(n), p_raw(n),
      epc_raw(n), drop_raw(n), p_drop_raw(n), score_resid(n), var_resid(n),
      z_resid(n), mi_resid(n), p_resid(n), epc_resid(n), drop_resid(n),
      p_drop_resid(n), signature_norm(n), residualized_norm(n);

  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& r = tab.rows[static_cast<std::size_t>(i)];
    const auto& c = r.candidate;
    kind[i] = score_candidate_kind_str(c.kind);
    row[i] = static_cast<int>(c.row) + 1;
    op[i] = std::string(magmaan::parse::to_string(c.op));
    group[i] = c.group;
    if (c.lhs_var >= 0 &&
        static_cast<std::size_t>(c.lhs_var) < names.var_name.size()) {
      lhs[i] = names.var_name[static_cast<std::size_t>(c.lhs_var)];
    } else if (c.row < names.row_lhs.size()) {
      lhs[i] = names.row_lhs[c.row];
    } else {
      lhs[i] = "";
    }
    if (c.rhs_var >= 0 &&
        static_cast<std::size_t>(c.rhs_var) < names.var_name.size()) {
      rhs[i] = names.var_name[static_cast<std::size_t>(c.rhs_var)];
    } else if (c.row < names.row_rhs.size()) {
      rhs[i] = names.row_rhs[c.row];
    } else {
      rhs[i] = "";
    }
    score_raw[i] = r.score_raw;
    var_raw[i] = r.var_raw;
    z_raw[i] = r.z_raw;
    mi_raw[i] = r.mi_raw;
    p_raw[i] = r.p_raw;
    epc_raw[i] = r.epc_raw;
    drop_raw[i] = r.drop_raw;
    p_drop_raw[i] = r.p_drop_raw;
    score_resid[i] = r.score_resid;
    var_resid[i] = r.var_resid;
    z_resid[i] = r.z_resid;
    mi_resid[i] = r.mi_resid;
    p_resid[i] = r.p_resid;
    epc_resid[i] = r.epc_resid;
    drop_resid[i] = r.drop_resid;
    p_drop_resid[i] = r.p_drop_resid;
    signature_norm[i] = r.signature_norm;
    residualized_norm[i] = r.residualized_norm;
  }

  Rcpp::DataFrame out = Rcpp::DataFrame::create(
      Rcpp::_["kind"] = kind,
      Rcpp::_["row"] = row,
      Rcpp::_["lhs"] = lhs,
      Rcpp::_["op"] = op,
      Rcpp::_["rhs"] = rhs,
      Rcpp::_["group"] = group,
      Rcpp::_["score.raw"] = score_raw,
      Rcpp::_["var.raw"] = var_raw,
      Rcpp::_["z.raw"] = z_raw,
      Rcpp::_["mi.raw"] = mi_raw,
      Rcpp::_["pvalue.raw"] = p_raw,
      Rcpp::_["epc.raw"] = epc_raw,
      Rcpp::_["drop.raw"] = drop_raw,
      Rcpp::_["pvalue.drop.raw"] = p_drop_raw,
      Rcpp::_["score.resid"] = score_resid,
      Rcpp::_["var.resid"] = var_resid,
      Rcpp::_["z.resid"] = z_resid,
      Rcpp::_["mi.resid"] = mi_resid,
      Rcpp::_["pvalue.resid"] = p_resid,
      Rcpp::_["epc.resid"] = epc_resid,
      Rcpp::_["drop.resid"] = drop_resid,
      Rcpp::_["pvalue.drop.resid"] = p_drop_resid,
      Rcpp::_["signature.norm"] = signature_norm,
      Rcpp::_["residualized.norm"] = residualized_norm,
      Rcpp::_["stringsAsFactors"] = false);
  out.attr("warnings") = Rcpp::wrap(tab.warnings);
  return out;
}

std::string noniter_lower(std::string s) {
  for (char& c : s)
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
  return s;
}

magmaan::estimate::frontier::NonIterativeEstimator noniter_which(const std::string& s) {
  const std::string k = noniter_lower(s);
  if (k == "guttman_lavaan" || k == "guttman-lavaan" || k == "guttman_ar" ||
      k == "guttman-ar" || k == "guttman_spearman" || k == "spearman" ||
      k == "guttman" || k == "guttman1952")
    return magmaan::estimate::frontier::NonIterativeEstimator::GuttmanLavaan;
  if (k == "guttman_aligned" || k == "guttman-aligned" ||
      k == "guttman_gls_aligned" || k == "guttman-gls-aligned" ||
      k == "aligned")
    return magmaan::estimate::frontier::NonIterativeEstimator::GuttmanAligned;
  Rcpp::stop("magmaan: unknown non-iterative estimator '%s' "
             "(accepted: guttman_lavaan, guttman_aligned)",
             s.c_str());
}

const char* noniter_map_name(magmaan::estimate::frontier::NonIterativeEstimator which) {
  using K = magmaan::estimate::frontier::NonIterativeEstimator;
  switch (which) {
    case K::GuttmanLavaan:
      return "guttman_lavaan";
    case K::GuttmanAligned:
      return "guttman_aligned";
  }
  return "guttman_lavaan";
}

bool noniter_estimator_auto(const std::string& estimator) {
  const std::string k = noniter_lower(estimator);
  return k == "auto" || k == "";
}

bool noniter_has_recorded_map(const Rcpp::List& fit) {
  return fit.containsElementNamed("noniterative_map");
}

magmaan::estimate::frontier::NonIterativeEstimator
noniter_recorded_map(const Rcpp::List& fit) {
  return noniter_which(Rcpp::as<std::string>(fit["noniterative_map"]));
}

void noniter_require_recorded_map(const Rcpp::List& fit,
                                  magmaan::estimate::frontier::NonIterativeEstimator which,
                                  const char* role) {
  if (!noniter_has_recorded_map(fit)) return;
  const auto recorded = noniter_recorded_map(fit);
  if (recorded != which) {
    Rcpp::stop("noniterative_cfa_pseudo_lrt: %s fit was produced by '%s', "
               "but inference requested '%s'",
               role, noniter_map_name(recorded), noniter_map_name(which));
  }
}

bool noniter_is_restricted_fit(const Rcpp::List& fit) {
  if (!fit.containsElementNamed("estimator")) return false;
  return noniter_lower(Rcpp::as<std::string>(fit["estimator"])) ==
         "noniterative_restricted";
}

magmaan::estimate::frontier::NonIterativeEstimator
noniter_which_for_fit(const Rcpp::List& fit, const std::string& estimator) {
  if (!noniter_estimator_auto(estimator)) return noniter_which(estimator);
  if (noniter_has_recorded_map(fit)) {
    return noniter_recorded_map(fit);
  }
  if (noniter_is_restricted_fit(fit)) {
    return magmaan::estimate::frontier::NonIterativeEstimator::GuttmanAligned;
  }
  return magmaan::estimate::frontier::NonIterativeEstimator::GuttmanLavaan;
}

magmaan::estimate::frontier::CommunalityMethod communality_which(const std::string& s) {
  const std::string k = noniter_lower(s);
  namespace ef = magmaan::estimate::frontier;
  if (k == "triad_mean" || k == "triad-mean" || k == "ar" ||
      k == "average_ratio" || k == "average-ratio")
    return ef::CommunalityMethod::TriadMean;
  if (k == "triad_pooled" || k == "triad-pooled" || k == "rs" ||
      k == "ratio_of_sums" || k == "ratio-of-sums")
    return ef::CommunalityMethod::TriadPooled;
  if (k == "triad_ls" || k == "triad-ls" || k == "ilm")
    return ef::CommunalityMethod::TriadLeastSquares;
  if (k == "extended_triad_ls" || k == "extended-triad-ls" ||
      k == "anchor_triad_ls" || k == "anchor-triad-ls" ||
      k == "anchor_ilm" || k == "anchor-ilm")
    return ef::CommunalityMethod::ExtendedTriadLeastSquares;
  if (k == "triad_wls" || k == "triad-wls" || k == "gmm_block" ||
      k == "gmm-block")
    return ef::CommunalityMethod::TriadWls;
  if (k == "triad_wls_joint" || k == "triad-wls-joint" || k == "gmm_full" ||
      k == "gmm-full")
    return ef::CommunalityMethod::TriadWlsJoint;
  Rcpp::stop("magmaan: unknown Guttman H method '%s' "
             "(accepted: triad_mean, triad_pooled, triad_ls, "
             "extended_triad_ls, triad_wls, triad_wls_joint)",
             s.c_str());
}

magmaan::estimate::frontier::CommunalityMethod
noniter_comm_for_fit(const Rcpp::List& fit) {
  namespace ef = magmaan::estimate::frontier;
  if (fit.containsElementNamed("communality")) {
    return communality_which(Rcpp::as<std::string>(fit["communality"]));
  }
  return ef::CommunalityMethod::TriadWls;
}

magmaan::estimate::frontier::CompositeWeight composite_which(const std::string& s) {
  const std::string k = noniter_lower(s);
  namespace ef = magmaan::estimate::frontier;
  if (k == "auto" || k == "default" || k == "")
    return ef::CompositeWeight::EstimatorDefault;
  if (k == "unit" || k == "incidence" || k == "z")
    return ef::CompositeWeight::Unit;
  if (k == "standardized" || k == "standardised" || k == "std" ||
      k == "correlation")
    return ef::CompositeWeight::Standardized;
  Rcpp::stop("magmaan: unknown Guttman composite weight '%s' "
             "(accepted: auto, unit, standardized)",
             s.c_str());
}

magmaan::estimate::frontier::AdmissibilityPolicy
admissibility_which(const std::string& s) {
  const std::string k = noniter_lower(s);
  namespace ef = magmaan::estimate::frontier;
  if (k == "raw" || k == "none" || k == "off" || k == "")
    return ef::AdmissibilityPolicy::Raw;
  if (k == "hard" || k == "clip" || k == "box")
    return ef::AdmissibilityPolicy::Hard;
  if (k == "soft" || k == "softclip")
    return ef::AdmissibilityPolicy::Soft;
  Rcpp::stop("magmaan: unknown communality admissibility policy '%s' "
             "(accepted: raw, hard, soft)", s.c_str());
}

magmaan::estimate::frontier::AdmissibilityConfig
admissibility_config(const std::string& policy, double margin, double beta0,
                     double rate) {
  return magmaan::estimate::frontier::AdmissibilityConfig{
      admissibility_which(policy), margin, beta0, rate};
}

magmaan::estimate::frontier::AdmissibilityConfig
noniter_admissibility_for_fit(const Rcpp::List& fit) {
  const std::string policy = fit.containsElementNamed("admissibility")
                                 ? Rcpp::as<std::string>(fit["admissibility"])
                                 : "raw";
  const double margin = fit.containsElementNamed("margin")
                            ? Rcpp::as<double>(fit["margin"])
                            : 1e-4;
  const double beta0 = fit.containsElementNamed("beta0")
                           ? Rcpp::as<double>(fit["beta0"])
                           : 1.0;
  const double rate = fit.containsElementNamed("rate")
                          ? Rcpp::as<double>(fit["rate"])
                          : 0.5;
  return admissibility_config(policy, margin, beta0, rate);
}

magmaan::estimate::frontier::ScoreConditioningPolicy
score_conditioning_which(const std::string& s) {
  const std::string k = noniter_lower(s);
  namespace ef = magmaan::estimate::frontier;
  if (k == "raw" || k == "none" || k == "off" || k == "")
    return ef::ScoreConditioningPolicy::Raw;
  if (k == "hard") return ef::ScoreConditioningPolicy::Hard;
  if (k == "soft" || k == "smooth") return ef::ScoreConditioningPolicy::Soft;
  Rcpp::stop("magmaan: unknown score conditioning policy '%s' "
             "(accepted: raw, hard, soft)", s.c_str());
}

magmaan::estimate::frontier::ScoreConditioningConfig
score_conditioning_config(const std::string& policy, double floor0,
                          double rate) {
  return magmaan::estimate::frontier::ScoreConditioningConfig{
      score_conditioning_which(policy), floor0, rate};
}

magmaan::estimate::frontier::HConditioningPolicy
h_conditioning_which(const std::string& s) {
  const std::string k = noniter_lower(s);
  namespace ef = magmaan::estimate::frontier;
  if (k == "raw" || k == "none" || k == "off" || k == "")
    return ef::HConditioningPolicy::Raw;
  if (k == "hard") return ef::HConditioningPolicy::Hard;
  if (k == "soft" || k == "smooth") return ef::HConditioningPolicy::Soft;
  Rcpp::stop("magmaan: unknown H conditioning policy '%s' "
             "(accepted: raw, hard, soft)", s.c_str());
}

magmaan::estimate::frontier::HConditioningConfig
h_conditioning_config(const std::string& policy, double floor0, double rate) {
  return magmaan::estimate::frontier::HConditioningConfig{
      h_conditioning_which(policy), floor0, rate};
}

magmaan::estimate::frontier::HConditioningConfig
noniter_h_conditioning_for_fit(const Rcpp::List& fit) {
  const std::string policy = fit.containsElementNamed("h_conditioning")
                                 ? Rcpp::as<std::string>(fit["h_conditioning"])
                                 : "raw";
  const double floor0 = fit.containsElementNamed("h_floor0")
                            ? Rcpp::as<double>(fit["h_floor0"])
                            : 1.0;
  const double rate = fit.containsElementNamed("h_rate")
                          ? Rcpp::as<double>(fit["h_rate"])
                          : 0.5;
  return h_conditioning_config(policy, floor0, rate);
}

void noniter_require_raw_h_conditioning(const Rcpp::List& fit) {
  if (noniter_h_conditioning_for_fit(fit).policy !=
      magmaan::estimate::frontier::HConditioningPolicy::Raw)
    Rcpp::stop("non-raw H conditioning is a point-estimation feasibility "
               "prototype; post-fit inference is not yet supported");
}

magmaan::estimate::frontier::ScoreConditioningConfig
noniter_score_conditioning_for_fit(const Rcpp::List& fit) {
  const std::string policy = fit.containsElementNamed("score_conditioning")
                                 ? Rcpp::as<std::string>(fit["score_conditioning"])
                                 : "raw";
  const double floor0 = fit.containsElementNamed("score_floor0")
                            ? Rcpp::as<double>(fit["score_floor0"])
                            : 1.0;
  const double rate = fit.containsElementNamed("score_rate")
                          ? Rcpp::as<double>(fit["score_rate"])
                          : 0.5;
  return score_conditioning_config(policy, floor0, rate);
}

bool same_score_conditioning(
    const magmaan::estimate::frontier::ScoreConditioningConfig& lhs,
    const magmaan::estimate::frontier::ScoreConditioningConfig& rhs) {
  return lhs.policy == rhs.policy && lhs.floor0 == rhs.floor0 &&
         lhs.rate_exp == rhs.rate_exp;
}

Rcpp::DataFrame score_conditioning_diagnostics_df(
    const std::vector<magmaan::estimate::frontier::ScoreConditioningDiagnostics>&
        diagnostics) {
  const R_xlen_t n = static_cast<R_xlen_t>(diagnostics.size());
  Rcpp::IntegerVector block(n);
  Rcpp::NumericVector target_floor(n), raw_min(n), repaired_min(n),
      raw_normalized_min(n), repaired_normalized_min(n), shrinkage(n),
      min_score_variance(n), min_abs_marker(n);
  Rcpp::LogicalVector hard_violation(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& d = diagnostics[static_cast<std::size_t>(i)];
    block[i] = static_cast<int>(i) + 1;
    target_floor[i] = d.target_floor;
    raw_min[i] = d.raw_min_eigenvalue;
    repaired_min[i] = d.repaired_min_eigenvalue;
    raw_normalized_min[i] = d.raw_normalized_min_eigenvalue;
    repaired_normalized_min[i] = d.repaired_normalized_min_eigenvalue;
    shrinkage[i] = d.shrinkage;
    hard_violation[i] = d.hard_violation;
    min_score_variance[i] = d.min_score_variance;
    min_abs_marker[i] = d.min_abs_marker;
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["block"] = block,
      Rcpp::_["target_floor"] = target_floor,
      Rcpp::_["raw_min_eigenvalue"] = raw_min,
      Rcpp::_["repaired_min_eigenvalue"] = repaired_min,
      Rcpp::_["raw_normalized_min_eigenvalue"] = raw_normalized_min,
      Rcpp::_["repaired_normalized_min_eigenvalue"] = repaired_normalized_min,
      Rcpp::_["shrinkage"] = shrinkage,
      Rcpp::_["hard_violation"] = hard_violation,
      Rcpp::_["min_score_variance"] = min_score_variance,
      Rcpp::_["min_abs_marker"] = min_abs_marker);
}

Rcpp::DataFrame h_conditioning_diagnostics_df(
    const std::vector<magmaan::estimate::frontier::HConditioningDiagnostics>&
        diagnostics) {
  const R_xlen_t n = static_cast<R_xlen_t>(diagnostics.size());
  Rcpp::IntegerVector block(n);
  Rcpp::NumericVector target_floor(n), raw_min(n), repaired_min(n),
      raw_normalized_min(n), repaired_normalized_min(n), shrinkage(n),
      min_h_variance(n);
  Rcpp::LogicalVector hard_violation(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& d = diagnostics[static_cast<std::size_t>(i)];
    block[i] = static_cast<int>(i) + 1;
    target_floor[i] = d.target_floor;
    raw_min[i] = d.raw_min_eigenvalue;
    repaired_min[i] = d.repaired_min_eigenvalue;
    raw_normalized_min[i] = d.raw_normalized_min_eigenvalue;
    repaired_normalized_min[i] = d.repaired_normalized_min_eigenvalue;
    shrinkage[i] = d.shrinkage;
    hard_violation[i] = d.hard_violation;
    min_h_variance[i] = d.min_h_variance;
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["block"] = block,
      Rcpp::_["target_floor"] = target_floor,
      Rcpp::_["raw_min_eigenvalue"] = raw_min,
      Rcpp::_["repaired_min_eigenvalue"] = repaired_min,
      Rcpp::_["raw_normalized_min_eigenvalue"] = raw_normalized_min,
      Rcpp::_["repaired_normalized_min_eigenvalue"] = repaired_normalized_min,
      Rcpp::_["shrinkage"] = shrinkage,
      Rcpp::_["hard_violation"] = hard_violation,
      Rcpp::_["min_h_variance"] = min_h_variance);
}

magmaan::estimate::frontier::CompositeWeight
noniter_comp_for_fit(const Rcpp::List& fit) {
  namespace ef = magmaan::estimate::frontier;
  if (fit.containsElementNamed("composite")) {
    return composite_which(Rcpp::as<std::string>(fit["composite"]));
  }
  return ef::CompositeWeight::EstimatorDefault;
}

magmaan::estimate::frontier::CompositeWeight
noniter_resolve_composite(
    magmaan::estimate::frontier::NonIterativeEstimator which,
    magmaan::estimate::frontier::CompositeWeight composite) {
  return magmaan::estimate::frontier::resolve_composite_weight(which, composite);
}

magmaan::robust::frontier::Discrepancy noniter_disc(const std::string& s) {
  const std::string k = noniter_lower(s);
  if (k == "uls") return magmaan::robust::frontier::Discrepancy::ULS;
  if (k == "ntml" || k == "ml") return magmaan::robust::frontier::Discrepancy::NTML;
  Rcpp::stop("magmaan: unknown discrepancy '%s' (accepted: uls, ntml)", s.c_str());
}

magmaan::post_expected<magmaan::robust::frontier::NonIterativeInference>
noniter_inference_dispatch(Ctx& ctx, const magmaan::estimate::Estimates& est,
                           magmaan::estimate::frontier::NonIterativeEstimator which,
                           magmaan::robust::frontier::Discrepancy disc,
                           const std::string& gamma, SEXP data,
                           bool restricted ,
                           magmaan::estimate::frontier::CommunalityMethod comm ,
                           magmaan::estimate::frontier::CompositeWeight composite ,
                           magmaan::estimate::frontier::AdmissibilityConfig admissibility ,
                           magmaan::estimate::frontier::ScoreConditioningConfig
                               score_conditioning ) {
  namespace rf = magmaan::robust::frontier;
  const std::string g = noniter_lower(gamma);
  if (g == "nt" || g == "normal" || g == "normal.theory" || g == "normaltheory")
    return restricted
        ? rf::noniterative_inference_restricted_nt(
              ctx.pt, ctx.rep, ctx.samp, est.theta, which, disc, comm, composite,
              admissibility, score_conditioning)
        : rf::noniterative_inference_nt(
              ctx.pt, ctx.rep, ctx.samp, est.theta, which, disc, composite,
              admissibility, score_conditioning);
  if (g == "empirical" || g == "adf" || g == "sandwich") {
    if (Rf_isNull(data)) Rcpp::stop("magmaan: empirical Gamma requires raw `data`");
    magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, data);
    return restricted
        ? rf::noniterative_inference_restricted_empirical(
              ctx.pt, ctx.rep, ctx.samp, raw, est.theta, which, disc, comm,
              composite, admissibility, score_conditioning)
        : rf::noniterative_inference_empirical(
              ctx.pt, ctx.rep, ctx.samp, raw, est.theta, which, disc, composite,
              admissibility, score_conditioning);
  }
  Rcpp::stop("magmaan: unknown gamma '%s' (accepted: nt, empirical)", gamma.c_str());
}

magmaan::post_expected<magmaan::robust::frontier::NonIterativeSE>
noniter_se_dispatch(Ctx& ctx, const magmaan::estimate::Estimates& est,
                    magmaan::estimate::frontier::NonIterativeEstimator which,
                    const std::string& gamma, SEXP data,
                    bool restricted ,
                    magmaan::estimate::frontier::CommunalityMethod comm ,
                    magmaan::estimate::frontier::CompositeWeight composite ,
                    magmaan::estimate::frontier::AdmissibilityConfig admissibility ,
                    magmaan::estimate::frontier::ScoreConditioningConfig
                        score_conditioning ) {
  namespace rf = magmaan::robust::frontier;
  const std::string g = noniter_lower(gamma);
  if (g == "nt" || g == "normal" || g == "normal.theory" || g == "normaltheory")
    return restricted
        ? rf::noniterative_se_grouped_restricted_nt(
              ctx.pt, ctx.rep, ctx.samp, est.theta, which, comm, composite,
              admissibility, score_conditioning)
        : rf::noniterative_se_grouped_nt(
              ctx.pt, ctx.rep, ctx.samp, est.theta, which, composite,
              admissibility, score_conditioning);
  if (g == "empirical" || g == "adf" || g == "sandwich") {
    if (Rf_isNull(data)) Rcpp::stop("magmaan: empirical Gamma requires raw `data`");
    magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, data);
    return restricted
        ? rf::noniterative_se_grouped_restricted_empirical(
              ctx.pt, ctx.rep, ctx.samp, raw, est.theta, which, comm, composite,
              admissibility, score_conditioning)
        : rf::noniterative_se_grouped_empirical(
              ctx.pt, ctx.rep, ctx.samp, raw, est.theta, which, composite,
              admissibility, score_conditioning);
  }
  Rcpp::stop("magmaan: unknown gamma '%s' (accepted: nt, empirical)", gamma.c_str());
}

Rcpp::List wrap_noniter_inference(const magmaan::robust::frontier::NonIterativeInference& inf) {
  return Rcpp::List::create(
      Rcpp::_["se"] = Rcpp::wrap(inf.se),
      Rcpp::_["vcov"] = Rcpp::wrap(inf.Omega),
      Rcpp::_["T"] = inf.T_gof,
      Rcpp::_["df"] = inf.df,
      Rcpp::_["scale_c"] = inf.scale_c,
      Rcpp::_["p_scaled"] = inf.p_scaled,
      Rcpp::_["p_meanvar"] = inf.p_meanvar,
      Rcpp::_["p_scaled_shifted"] = inf.p_scaled_shifted,
      Rcpp::_["p_mixture"] = inf.p_mixture,
      Rcpp::_["rls_check"] = inf.rls_check,
      Rcpp::_["eigenvalues"] = Rcpp::wrap(inf.gof_eigenvalues),
      Rcpp::_["warnings"] = Rcpp::wrap(inf.warnings));
}

Rcpp::List wrap_noniter_se(const magmaan::robust::frontier::NonIterativeSE& se) {
  return Rcpp::List::create(
      Rcpp::_["se"] = Rcpp::wrap(se.se),
      Rcpp::_["vcov"] = Rcpp::wrap(se.Omega),
      Rcpp::_["theta_hat"] = Rcpp::wrap(se.theta_hat),
      Rcpp::_["block_of_param"] = Rcpp::wrap(se.block_of_param),
      Rcpp::_["warnings"] = Rcpp::wrap(se.warnings));
}

Rcpp::List wrap_noniter_diff(const magmaan::robust::frontier::NonIterativeDiffTest& d) {
  return Rcpp::List::create(
      Rcpp::_["T_diff"] = d.T_d,
      Rcpp::_["df_diff"] = d.df_d,
      Rcpp::_["T_d"] = d.T_d,
      Rcpp::_["df_d"] = d.df_d,
      Rcpp::_["p_scaled"] = d.p_scaled,
      Rcpp::_["p_adjusted"] = d.p_adjusted,
      Rcpp::_["p_scaled_shifted"] = d.p_scaled_shifted,
      Rcpp::_["p_mixture"] = d.p_mixture,
      Rcpp::_["eigenvalues"] = Rcpp::wrap(d.eigenvalues),
      Rcpp::_["warnings"] = Rcpp::wrap(d.warnings));
}

magmaan::post_expected<magmaan::robust::frontier::GroupedNonIterativeInference>
noniter_grouped_dispatch(Ctx& ctx, const magmaan::estimate::Estimates& est,
                         magmaan::estimate::frontier::NonIterativeEstimator which,
                         magmaan::robust::frontier::Discrepancy disc,
                         const std::string& gamma, SEXP data,
                         bool restricted,
                         magmaan::estimate::frontier::CommunalityMethod comm,
                         magmaan::estimate::frontier::CompositeWeight composite,
                         magmaan::estimate::frontier::AdmissibilityConfig admissibility,
                         magmaan::estimate::frontier::ScoreConditioningConfig
                             score_conditioning) {
  namespace rf = magmaan::robust::frontier;
  const std::string g = noniter_lower(gamma);
  if (g == "nt" || g == "normal" || g == "normal.theory" || g == "normaltheory")
    return restricted
        ? rf::noniterative_inference_grouped_restricted_nt(
              ctx.pt, ctx.rep, ctx.samp, est.theta, which, disc, comm, composite,
              admissibility, score_conditioning)
        : rf::noniterative_inference_grouped_nt(
              ctx.pt, ctx.rep, ctx.samp, est.theta, which, disc, composite,
              admissibility, score_conditioning);
  if (g == "empirical" || g == "adf" || g == "sandwich") {
    if (Rf_isNull(data)) Rcpp::stop("magmaan: empirical Gamma requires raw `data`");
    magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, data);
    return restricted
        ? rf::noniterative_inference_grouped_restricted_empirical(
              ctx.pt, ctx.rep, ctx.samp, raw, est.theta, which, disc, comm,
              composite, admissibility, score_conditioning)
        : rf::noniterative_inference_grouped_empirical(
              ctx.pt, ctx.rep, ctx.samp, raw, est.theta, which, disc, composite,
              admissibility, score_conditioning);
  }
  Rcpp::stop("magmaan: unknown gamma '%s' (accepted: nt, empirical)", gamma.c_str());
}

Rcpp::List
wrap_noniter_grouped(const magmaan::robust::frontier::GroupedNonIterativeInference& inf) {
  return Rcpp::List::create(
      Rcpp::_["se"] = Rcpp::wrap(inf.se),
      Rcpp::_["vcov"] = Rcpp::wrap(inf.Omega),
      Rcpp::_["T"] = inf.T_gof,
      Rcpp::_["df"] = inf.df,
      Rcpp::_["scale_c"] = inf.scale_c,
      Rcpp::_["p_scaled"] = inf.p_scaled,
      Rcpp::_["p_meanvar"] = inf.p_meanvar,
      Rcpp::_["p_scaled_shifted"] = inf.p_scaled_shifted,
      Rcpp::_["p_mixture"] = inf.p_mixture,
      Rcpp::_["rls_check"] = inf.rls_check,
      Rcpp::_["eigenvalues"] = Rcpp::wrap(inf.gof_eigenvalues),
      Rcpp::_["block_of_param"] = Rcpp::wrap(inf.block_of_param),
      Rcpp::_["warnings"] = Rcpp::wrap(inf.warnings));
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List frontier_guttman_h_impl(Rcpp::NumericMatrix S,
                                   Rcpp::IntegerVector blocks,
                                   std::string method = "triad_ls") {
  if (S.nrow() != S.ncol())
    Rcpp::stop("magmaan: frontier_guttman_h() requires a square covariance matrix");
  if (blocks.size() != S.nrow())
    Rcpp::stop("magmaan: frontier_guttman_h() block vector length must match S");

  std::vector<std::int32_t> block_of;
  block_of.reserve(static_cast<std::size_t>(blocks.size()));
  for (R_xlen_t i = 0; i < blocks.size(); ++i) {
    const int b = blocks[i];
    if (b == NA_INTEGER || b < 1)
      Rcpp::stop("magmaan: frontier_guttman_h() blocks must be positive integers");
    block_of.push_back(static_cast<std::int32_t>(b - 1));
  }

  const Eigen::MatrixXd Smat = Rcpp::as<Eigen::MatrixXd>(S);
  const auto which = communality_which(method);
  auto out = magmaan::estimate::frontier::estimate_h_communalities(
      Smat, block_of, which);
  if (!out.has_value()) stop_fit(out.error());

  return Rcpp::List::create(
      Rcpp::_["method"] = magmaan::estimate::frontier::communality_method_name(which),
      Rcpp::_["h2"] = Rcpp::wrap(out->h2),
      Rcpp::_["h_diag"] = Rcpp::wrap(out->h_diag),
      Rcpp::_["H"] = Rcpp::wrap(out->H));
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_fit_impl(SEXP partable, Rcpp::List sample_stats,
                                     std::string estimator = "guttman_lavaan",
                                     std::string composite = "auto",
                                     std::string admissibility = "raw",
                                     double margin = 1e-4,
                                     double beta0 = 1.0,
                                     double rate = 0.5,
                                     std::string score_conditioning = "raw",
                                     double score_floor0 = 1.0,
                                     double score_rate = 0.5,
                                     std::string h_conditioning = "raw",
                                     double h_floor0 = 1.0,
                                     double h_rate = 0.5) {
  auto parsed = partable_from_arg(partable, "noniterative_cfa_fit");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const auto which = noniter_which(estimator);
  const auto comp = composite_which(composite);
  const auto admiss = admissibility_config(
      admissibility, margin, beta0, rate);
  const auto score = score_conditioning_config(
      score_conditioning, score_floor0, score_rate);
  const auto h = h_conditioning_config(h_conditioning, h_floor0, h_rate);
  auto fit = magmaan::estimate::frontier::fit_noniterative_cfa(
      ctx.pt, ctx.rep, ctx.samp, which, comp, admiss, score, h);
  if (!fit.has_value()) stop_fit(fit.error());
  magmaan::estimate::Estimates est;
  est.theta = std::move(fit->theta);
  est.fmin = 0.0;
  Rcpp::List out = fit_result(ctx, est, &starts, "noniterative");
  out["noniterative_map"] = noniter_map_name(which);
  out["composite"] =
      magmaan::estimate::frontier::composite_weight_name(
          noniter_resolve_composite(which, comp));
  out["admissibility"] = magmaan::estimate::frontier::admissibility_policy_name(
      admiss.policy);
  out["margin"] = margin;
  out["beta0"] = beta0;
  out["rate"] = rate;
  out["n_h2_clamped"] = Rcpp::wrap(fit->n_h2_clamped);
  out["score_conditioning"] =
      magmaan::estimate::frontier::score_conditioning_policy_name(score.policy);
  out["score_floor0"] = score_floor0;
  out["score_rate"] = score_rate;
  out["score_conditioning_diagnostics"] =
      score_conditioning_diagnostics_df(fit->score_conditioning_diagnostics);
  out["h_conditioning"] =
      magmaan::estimate::frontier::h_conditioning_policy_name(h.policy);
  out["h_floor0"] = h_floor0;
  out["h_rate"] = h_rate;
  out["h_conditioning_diagnostics"] =
      h_conditioning_diagnostics_df(fit->h_conditioning_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_metric_fit_impl(SEXP partable, Rcpp::List sample_stats,
                                            std::string estimator = "guttman_aligned",
                                            std::string composite = "auto",
                                            std::string admissibility = "raw",
                                            double margin = 1e-4,
                                            double beta0 = 1.0,
                                            double rate = 0.5,
                                            std::string score_conditioning = "raw",
                                            double score_floor0 = 1.0,
                                            double score_rate = 0.5,
                                            std::string h_conditioning = "raw",
                                            double h_floor0 = 1.0,
                                            double h_rate = 0.5) {
  auto parsed = partable_from_arg(partable, "noniterative_cfa_metric_fit");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const auto which = noniter_which(estimator);
  const auto comp = composite_which(composite);
  const auto admiss = admissibility_config(
      admissibility, margin, beta0, rate);
  const auto score = score_conditioning_config(
      score_conditioning, score_floor0, score_rate);
  const auto h = h_conditioning_config(h_conditioning, h_floor0, h_rate);
  auto fit = magmaan::estimate::frontier::fit_noniterative_cfa_metric(
      ctx.pt, ctx.rep, ctx.samp, which, comp, admiss, score, h);
  if (!fit.has_value()) stop_fit(fit.error());
  magmaan::estimate::Estimates est;
  est.theta = std::move(fit->theta);
  est.fmin = 0.0;
  Rcpp::List out = fit_result(ctx, est, &starts, "noniterative_metric");
  out["noniterative_map"] = noniter_map_name(which);
  out["composite"] =
      magmaan::estimate::frontier::composite_weight_name(
          noniter_resolve_composite(which, comp));
  out["admissibility"] = magmaan::estimate::frontier::admissibility_policy_name(
      admiss.policy);
  out["margin"] = margin;
  out["beta0"] = beta0;
  out["rate"] = rate;
  out["n_h2_clamped"] = Rcpp::wrap(fit->n_h2_clamped);
  out["score_conditioning"] =
      magmaan::estimate::frontier::score_conditioning_policy_name(score.policy);
  out["score_floor0"] = score_floor0;
  out["score_rate"] = score_rate;
  out["score_conditioning_diagnostics"] =
      score_conditioning_diagnostics_df(fit->score_conditioning_diagnostics);
  out["h_conditioning"] =
      magmaan::estimate::frontier::h_conditioning_policy_name(h.policy);
  out["h_floor0"] = h_floor0;
  out["h_rate"] = h_rate;
  out["h_conditioning_diagnostics"] =
      h_conditioning_diagnostics_df(fit->h_conditioning_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_restricted_fit_impl(SEXP partable, Rcpp::List sample_stats,
                                                std::string estimator = "guttman_aligned",
                                                std::string communality = "triad_wls",
                                                std::string composite = "auto",
                                                std::string admissibility = "raw",
                                                double margin = 1e-4,
                                                double beta0 = 1.0,
                                                double rate = 0.5,
                                                std::string score_conditioning = "raw",
                                                double score_floor0 = 1.0,
                                                double score_rate = 0.5,
                                                std::string h_conditioning = "raw",
                                                double h_floor0 = 1.0,
                                                double h_rate = 0.5) {
  auto parsed = partable_from_arg(partable, "noniterative_cfa_restricted_fit");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure),
                                  std::move(parsed.names), sample_stats);
  const auto which = noniter_which(estimator);
  const auto comm = communality_which(communality);
  const auto comp = composite_which(composite);
  const auto admiss = admissibility_config(
      admissibility, margin, beta0, rate);
  const auto score = score_conditioning_config(
      score_conditioning, score_floor0, score_rate);
  const auto h = h_conditioning_config(h_conditioning, h_floor0, h_rate);
  auto fit = magmaan::estimate::frontier::fit_noniterative_cfa_restricted(
      ctx.pt, ctx.rep, ctx.samp, which, comm, comp, admiss, score, h);
  if (!fit.has_value()) stop_fit(fit.error());
  magmaan::estimate::Estimates est;
  est.theta = std::move(fit->theta);
  est.fmin = 0.0;
  Rcpp::List out = fit_result(ctx, est, &starts, "noniterative_restricted");
  out["noniterative_map"] = noniter_map_name(which);
  out["communality"] =
      magmaan::estimate::frontier::communality_method_name(comm);
  out["composite"] =
      magmaan::estimate::frontier::composite_weight_name(
          noniter_resolve_composite(which, comp));
  out["admissibility"] = magmaan::estimate::frontier::admissibility_policy_name(
      admiss.policy);
  out["margin"] = margin;
  out["beta0"] = beta0;
  out["rate"] = rate;
  out["n_h2_clamped"] = Rcpp::wrap(fit->n_h2_clamped);
  out["score_conditioning"] =
      magmaan::estimate::frontier::score_conditioning_policy_name(score.policy);
  out["score_floor0"] = score_floor0;
  out["score_rate"] = score_rate;
  out["score_conditioning_diagnostics"] =
      score_conditioning_diagnostics_df(fit->score_conditioning_diagnostics);
  out["h_conditioning"] =
      magmaan::estimate::frontier::h_conditioning_policy_name(h.policy);
  out["h_floor0"] = h_floor0;
  out["h_rate"] = h_rate;
  out["h_conditioning_diagnostics"] =
      h_conditioning_diagnostics_df(fit->h_conditioning_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_inference_impl(Rcpp::List fit, std::string estimator = "auto",
                                           std::string discrepancy = "uls",
                                           std::string gamma = "nt", SEXP data = R_NilValue) {
  noniter_require_raw_h_conditioning(fit);
  Ctx ctx = ctx_from_fit(fit);
  const auto est = est_from_fit(fit);
  auto inf = noniter_inference_dispatch(ctx, est, noniter_which_for_fit(fit, estimator),
                                        noniter_disc(discrepancy), gamma, data,
                                        noniter_is_restricted_fit(fit),
                                        noniter_comm_for_fit(fit),
                                        noniter_comp_for_fit(fit),
                                        noniter_admissibility_for_fit(fit),
                                        noniter_score_conditioning_for_fit(fit));
  if (!inf.has_value()) stop_post(inf.error());
  return wrap_noniter_inference(*inf);
}

// [[Rcpp::export]]
Rcpp::DataFrame noniterative_cfa_modindices_impl(
    Rcpp::List fit, std::string estimator = "auto",
    std::string discrepancy = "uls", std::string gamma = "nt",
    SEXP data = R_NilValue, std::string candidates = "all",
    bool include_loadings = true, bool include_covariances = true) {
  noniter_require_raw_h_conditioning(fit);
  Ctx ctx = ctx_from_fit(fit);
  if (ctx.samp.S.size() != 1) {
    Rcpp::stop("magmaan: noniterative_cfa_modification_indices() currently "
               "supports single-group covariance-only fits");
  }
  const auto est = est_from_fit(fit);
  const auto opts = modification_options_from(
      "expected", candidates, include_loadings, include_covariances);
  auto inf = noniter_inference_dispatch(ctx, est, noniter_which_for_fit(fit, estimator),
                                        noniter_disc(discrepancy), gamma, data,
                                        noniter_is_restricted_fit(fit),
                                        noniter_comm_for_fit(fit),
                                        noniter_comp_for_fit(fit),
                                        noniter_admissibility_for_fit(fit),
                                        noniter_score_conditioning_for_fit(fit));
  if (!inf.has_value()) stop_post(inf.error());
  auto tab = magmaan::robust::frontier::noniterative_modification_indices(
      ctx.pt, ctx.rep, ctx.samp, est.theta, *inf, opts);
  if (!tab.has_value()) stop_post(tab.error());
  return noniter_mi_table_df(*tab, ctx.names);
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_se_impl(Rcpp::List fit, std::string estimator = "auto",
                                    std::string gamma = "nt",
                                    SEXP data = R_NilValue) {
  noniter_require_raw_h_conditioning(fit);
  Ctx ctx = ctx_from_fit(fit);
  const auto est = est_from_fit(fit);
  auto se = noniter_se_dispatch(ctx, est, noniter_which_for_fit(fit, estimator),
                                gamma, data, noniter_is_restricted_fit(fit),
                                noniter_comm_for_fit(fit),
                                noniter_comp_for_fit(fit),
                                noniter_admissibility_for_fit(fit),
                                noniter_score_conditioning_for_fit(fit));
  if (!se.has_value()) stop_post(se.error());
  return wrap_noniter_se(*se);
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_wald_impl(Rcpp::List fit, Rcpp::NumericMatrix R,
                                      Rcpp::NumericVector q, std::string estimator = "auto",
                                      std::string discrepancy = "uls",
                                      std::string gamma = "nt", SEXP data = R_NilValue) {
  noniter_require_raw_h_conditioning(fit);
  (void)discrepancy;
  Ctx ctx = ctx_from_fit(fit);
  const auto est = est_from_fit(fit);
  auto se = noniter_se_dispatch(ctx, est, noniter_which_for_fit(fit, estimator),
                                gamma, data, noniter_is_restricted_fit(fit),
                                noniter_comm_for_fit(fit),
                                noniter_comp_for_fit(fit),
                                noniter_admissibility_for_fit(fit),
                                noniter_score_conditioning_for_fit(fit));
  if (!se.has_value()) stop_post(se.error());
  const Eigen::MatrixXd Rm = Rcpp::as<Eigen::MatrixXd>(R);
  const Eigen::VectorXd qv = Rcpp::as<Eigen::VectorXd>(q);
  auto w = magmaan::robust::frontier::noniterative_wald(est.theta, *se, Rm, qv);
  if (!w.has_value()) stop_post(w.error());
  return Rcpp::List::create(Rcpp::_["chi2"] = w->chi2, Rcpp::_["df"] = w->df);
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_difference_impl(Rcpp::List fit0, Rcpp::List fit1, int df_d,
                                            std::string estimator = "auto",
                                            std::string discrepancy = "uls",
                                            std::string gamma = "nt",
                                            SEXP data0 = R_NilValue, SEXP data1 = R_NilValue) {
  noniter_require_raw_h_conditioning(fit0);
  noniter_require_raw_h_conditioning(fit1);
  Ctx c0 = ctx_from_fit(fit0);
  const auto e0 = est_from_fit(fit0);
  Ctx c1 = ctx_from_fit(fit1);
  const auto e1 = est_from_fit(fit1);
  const auto which0 = noniter_which_for_fit(fit0, estimator);
  const auto which1 = noniter_which_for_fit(fit1, estimator);
  const auto score0 = noniter_score_conditioning_for_fit(fit0);
  const auto score1 = noniter_score_conditioning_for_fit(fit1);
  if (!same_score_conditioning(score0, score1))
    Rcpp::stop("noniterative_cfa_difference_test: fit0/fit1 score "
               "conditioning configurations differ");
  const auto disc = noniter_disc(discrepancy);
  auto inf0 = noniter_grouped_dispatch(
      c0, e0, which0, disc, gamma, data0, noniter_is_restricted_fit(fit0),
      noniter_comm_for_fit(fit0), noniter_comp_for_fit(fit0),
      noniter_admissibility_for_fit(fit0),
      score0);
  if (!inf0.has_value()) stop_post(inf0.error());
  auto inf1 = noniter_grouped_dispatch(
      c1, e1, which1, disc, gamma, data1, noniter_is_restricted_fit(fit1),
      noniter_comm_for_fit(fit1), noniter_comp_for_fit(fit1),
      noniter_admissibility_for_fit(fit1),
      score1);
  if (!inf1.has_value()) stop_post(inf1.error());
  auto d = magmaan::robust::frontier::noniterative_difference_test(*inf0, *inf1, df_d);
  if (!d.has_value()) stop_post(d.error());
  return wrap_noniter_diff(*d);
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_grouped_inference_impl(Rcpp::List fit,
                                                   std::string estimator = "auto",
                                                   std::string discrepancy = "uls",
                                                   std::string gamma = "nt",
                                                   SEXP data = R_NilValue) {
  noniter_require_raw_h_conditioning(fit);
  Ctx ctx = ctx_from_fit(fit);
  const auto est = est_from_fit(fit);
  auto inf = noniter_grouped_dispatch(ctx, est, noniter_which_for_fit(fit, estimator),
                                      noniter_disc(discrepancy), gamma, data,
                                      noniter_is_restricted_fit(fit),
                                      noniter_comm_for_fit(fit),
                                      noniter_comp_for_fit(fit),
                                      noniter_admissibility_for_fit(fit),
                                      noniter_score_conditioning_for_fit(fit));
  if (!inf.has_value()) stop_post(inf.error());
  return wrap_noniter_grouped(*inf);
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_pseudo_lrt_impl(Rcpp::List fit_H1, Rcpp::List fit_H0,
                                            std::string estimator = "auto",
                                            std::string discrepancy = "uls",
                                            std::string gamma = "nt",
                                            SEXP data = R_NilValue) {
  noniter_require_raw_h_conditioning(fit_H1);
  noniter_require_raw_h_conditioning(fit_H0);
  Ctx ctx1 = ctx_from_fit(fit_H1);
  const auto est1 = est_from_fit(fit_H1);
  Ctx ctx0 = ctx_from_fit(fit_H0);
  const auto est0 = est_from_fit(fit_H0);
  const auto which1 = noniter_which_for_fit(fit_H1, estimator);
  const auto which0 = noniter_which_for_fit(fit_H0, estimator);
  const auto comm1 = noniter_comm_for_fit(fit_H1);
  const auto comm0 = noniter_comm_for_fit(fit_H0);
  const auto comp1 = noniter_comp_for_fit(fit_H1);
  const auto comp0 = noniter_comp_for_fit(fit_H0);
  const auto comp1r = noniter_resolve_composite(which1, comp1);
  const auto comp0r = noniter_resolve_composite(which0, comp0);
  const auto score1 = noniter_score_conditioning_for_fit(fit_H1);
  const auto score0 = noniter_score_conditioning_for_fit(fit_H0);
  noniter_require_recorded_map(fit_H1, which1, "H1");
  noniter_require_recorded_map(fit_H0, which0, "H0");
  if (which0 != which1) {
    Rcpp::stop("noniterative_cfa_pseudo_lrt: H1/H0 non-iterative maps differ "
               "('%s' vs '%s')",
               noniter_map_name(which1), noniter_map_name(which0));
  }
  if (comm0 != comm1) {
    Rcpp::stop("noniterative_cfa_pseudo_lrt: H1/H0 communality methods differ "
               "('%s' vs '%s')",
               magmaan::estimate::frontier::communality_method_name(comm1),
               magmaan::estimate::frontier::communality_method_name(comm0));
  }
  if (comp0r != comp1r) {
    Rcpp::stop("noniterative_cfa_pseudo_lrt: H1/H0 composite weights differ "
               "('%s' vs '%s')",
               magmaan::estimate::frontier::composite_weight_name(comp1r),
               magmaan::estimate::frontier::composite_weight_name(comp0r));
  }
  if (!same_score_conditioning(score0, score1))
    Rcpp::stop("noniterative_cfa_pseudo_lrt: H1/H0 score conditioning "
               "configurations differ");
  const auto disc = noniter_disc(discrepancy);
  auto inf1 = noniter_grouped_dispatch(
      ctx1, est1, which1, disc, gamma, data, noniter_is_restricted_fit(fit_H1),
      comm1, comp1, noniter_admissibility_for_fit(fit_H1), score1);
  if (!inf1.has_value()) stop_post(inf1.error());
  auto inf0 = noniter_grouped_dispatch(
      ctx0, est0, which0, disc, gamma, data, noniter_is_restricted_fit(fit_H0),
      comm0, comp0, noniter_admissibility_for_fit(fit_H0), score0);
  if (!inf0.has_value()) stop_post(inf0.error());

  const int df_d = inf0->df - inf1->df;
  if (df_d < 1) {
    Rcpp::stop("noniterative_cfa_pseudo_lrt: H0 df (%d) must exceed H1 df (%d)",
               inf0->df, inf1->df);
  }
  auto d = magmaan::robust::frontier::noniterative_difference_test(*inf0, *inf1, df_d);
  if (!d.has_value()) stop_post(d.error());
  return wrap_noniter_diff(*d);
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_constrained_impl(Rcpp::List fit,
                                             std::string estimator = "auto",
                                             std::string discrepancy = "uls",
                                             std::string gamma = "nt",
                                             SEXP data = R_NilValue) {
  noniter_require_raw_h_conditioning(fit);
  (void)discrepancy;
  Ctx ctx = ctx_from_fit(fit);
  const auto est = est_from_fit(fit);
  auto se = noniter_se_dispatch(ctx, est, noniter_which_for_fit(fit, estimator),
                                gamma, data, noniter_is_restricted_fit(fit),
                                noniter_comm_for_fit(fit),
                                noniter_comp_for_fit(fit),
                                noniter_admissibility_for_fit(fit),
                                noniter_score_conditioning_for_fit(fit));
  if (!se.has_value()) stop_post(se.error());
  auto con = magmaan::robust::frontier::noniterative_constrained_fit(ctx.pt, *se);
  if (!con.has_value()) stop_post(con.error());
  return Rcpp::List::create(
      Rcpp::_["theta_hat"] = Rcpp::wrap(con->theta_hat),
      Rcpp::_["theta_tilde"] = Rcpp::wrap(con->theta_tilde),
      Rcpp::_["vcov"] = Rcpp::wrap(con->Omega),
      Rcpp::_["vcov_constrained"] = Rcpp::wrap(con->Omega_tilde),
      Rcpp::_["se_constrained"] = Rcpp::wrap(con->se_constrained),
      Rcpp::_["W"] = con->W,
      Rcpp::_["k"] = con->k,
      Rcpp::_["p_wald"] = con->p_wald,
      Rcpp::_["warnings"] = Rcpp::wrap(con->warnings));
}

// [[Rcpp::export]]
Rcpp::List noniterative_cfa_scalar_impl(Rcpp::List fit, int ref_group = 1,
                                        std::string estimator = "auto",
                                        std::string discrepancy = "uls",
                                        std::string gamma = "nt",
                                        SEXP data = R_NilValue) {
  noniter_require_raw_h_conditioning(fit);
  (void)discrepancy;
  Ctx ctx = ctx_from_fit(fit);
  const auto est = est_from_fit(fit);
  auto se = noniter_se_dispatch(ctx, est, noniter_which_for_fit(fit, estimator),
                                gamma, data, noniter_is_restricted_fit(fit),
                                noniter_comm_for_fit(fit),
                                noniter_comp_for_fit(fit),
                                noniter_admissibility_for_fit(fit),
                                noniter_score_conditioning_for_fit(fit));
  if (!se.has_value()) stop_post(se.error());
  if (ref_group < 1) Rcpp::stop("magmaan: ref_group is 1-based (>= 1)");
  auto sc = magmaan::robust::frontier::noniterative_scalar_invariance(
      ctx.pt, ctx.rep, *se, static_cast<std::size_t>(ref_group - 1));
  if (!sc.has_value()) stop_post(sc.error());
  const std::size_t ng = sc->alpha.size();
  Rcpp::List alpha(ng), alpha_se(ng), alpha_cov(ng);
  Rcpp::IntegerVector groups(ng);
  for (std::size_t i = 0; i < ng; ++i) {
    alpha[static_cast<R_xlen_t>(i)] = Rcpp::wrap(sc->alpha[i]);
    alpha_se[static_cast<R_xlen_t>(i)] = Rcpp::wrap(sc->alpha_se[i]);
    alpha_cov[static_cast<R_xlen_t>(i)] = Rcpp::wrap(sc->alpha_cov[i]);
    groups[static_cast<R_xlen_t>(i)] = static_cast<int>(sc->groups[i]) + 1;  // 1-based
  }
  return Rcpp::List::create(
      Rcpp::_["ref_group"] = static_cast<int>(sc->ref_group) + 1,
      Rcpp::_["groups"] = groups,
      Rcpp::_["nu"] = Rcpp::wrap(sc->nu),
      Rcpp::_["alpha"] = alpha,
      Rcpp::_["alpha_se"] = alpha_se,
      Rcpp::_["alpha_cov"] = alpha_cov,
      Rcpp::_["d"] = Rcpp::wrap(sc->d_stacked),
      Rcpp::_["W"] = sc->W,
      Rcpp::_["df"] = sc->df,
      Rcpp::_["rank"] = sc->rank,
      Rcpp::_["p_value"] = sc->p_value,
      Rcpp::_["warnings"] = Rcpp::wrap(sc->warnings));
}
