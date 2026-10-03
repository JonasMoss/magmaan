#include "glue_internal.h"
#include "gamma_arg.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

namespace {

// Two-stage (ML2S) robust MI/release inputs from the fit: the Stage-1 EM
// moments, the recorded Stage-2 weight and DLS mixing weight, and, for an
// estimated DWLS/ADF/DLS weight, the retained raw data with its FIML pack and
// H1. The meat is the Stage-1 saturated-moment covariance, so the complete-data
// `weight`, `bread`, `moments`, `cov` and `data` choices do not apply.
struct Ml2sScoreInputs {
  SaturatedMoments sm;
  magmaan::inference::frontier::Ml2sScoreOptions opts;
  std::unique_ptr<magmaan::data::RawData> raw;
  std::unique_ptr<FimlPack> owned_pack;
  std::unique_ptr<FimlH1> owned_h1;
};

Rcpp::DataFrame score_table_df(
    const magmaan::inference::ScoreTestTable& tab,
    const magmaan::spec::LatentNames& names);
magmaan::estimate::gmm::Weight ml2s_stage2_weight_for_fit(Rcpp::List fit);
Rcpp::DataFrame label_ml2s_score_table(Rcpp::DataFrame df);
void validate_fiml_robust_score_options(
    const std::string& bread, const std::string& moments,
    const std::string& cov, SEXP weight, bool estimated_weight);
void prepare_ml2s_score_inputs(Rcpp::List fit, const Ctx& ctx, SEXP raw_arg,
                               SEXP weight, const std::string& bread,
                               const std::string& moments,
                               const std::string& cov, bool estimated_weight,
                               Ml2sScoreInputs& in);
const FimlPack& fiml_robust_score_pack(
    Rcpp::List fit, SEXP raw_arg, const magmaan::data::RawData& raw,
    std::unique_ptr<FimlPack>& owned);
SEXP fiml_robust_score_data(Rcpp::List fit, SEXP raw_arg);
magmaan::inference::frontier::ScoreFlipCalibration
score_flip_calibration_from_string(const std::string& calibration,
                                   int n_flips);
magmaan::inference::frontier::ScoreFlipMultiplierStudentization
score_flip_multiplier_studentization_from_string(
    const std::string& studentization);
const char* score_flip_sensitivity_string(
    magmaan::inference::frontier::ScoreFlipSensitivity sensitivity);
const char* global_score_metric_string(
    magmaan::inference::frontier::GlobalScoreFlipOptions::Metric metric);
Rcpp::List score_flip_result_to_r(
    const magmaan::inference::frontier::ScoreFlipTestResult& out);
Rcpp::List global_score_flip_result_to_r(
    const magmaan::inference::frontier::GlobalScoreFlipTestResult& out);

Rcpp::DataFrame score_table_df(
    const magmaan::inference::ScoreTestTable& tab,
    const magmaan::spec::LatentNames& names) {
  const R_xlen_t n = static_cast<R_xlen_t>(tab.rows.size());
  Rcpp::CharacterVector kind(n), op(n), lhs(n), rhs(n);
  Rcpp::IntegerVector row(n), group(n), df(n);
  Rcpp::NumericVector score(n), information(n), mi(n), pvalue(n), epc(n),
      epc_lv(n), epc_all(n), v_eff(n), mi_scaled(n), scaling_factor(n);
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
    score[i] = r.score;
    information[i] = r.information;
    mi[i] = r.mi;
    df[i] = r.df;
    pvalue[i] = r.p_value;
    epc[i] = r.epc;
    epc_lv[i] = r.epc_lv;
    epc_all[i] = r.epc_all;
    v_eff[i] = r.v_eff;
    mi_scaled[i] = (r.mi_scaled == 0.0 && r.scaling_factor == 1.0 &&
                    r.v_eff == 0.0)
        ? r.mi
        : r.mi_scaled;
    scaling_factor[i] = r.scaling_factor;
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["kind"] = kind,
      Rcpp::_["row"] = row,
      Rcpp::_["lhs"] = lhs,
      Rcpp::_["op"] = op,
      Rcpp::_["rhs"] = rhs,
      Rcpp::_["group"] = group,
      Rcpp::_["score"] = score,
      Rcpp::_["information"] = information,
      Rcpp::_["mi"] = mi,
      Rcpp::_["df"] = df,
      Rcpp::_["pvalue"] = pvalue,
      Rcpp::_["epc"] = epc,
      Rcpp::_["epc.lv"] = epc_lv,
      Rcpp::_["epc.all"] = epc_all,
      Rcpp::_["v.eff"] = v_eff,
      Rcpp::_["mi.scaled"] = mi_scaled,
      Rcpp::_["scaling.factor"] = scaling_factor,
      Rcpp::_["stringsAsFactors"] = false);
}

// The fitted Stage-2 weight of a weighted ML2S fit, rebuilt from Stage 1 as
// the fitter built it, for the naive (complete-data-form) score sweep.
magmaan::estimate::gmm::Weight ml2s_stage2_weight_for_fit(Rcpp::List fit) {
  SaturatedMoments sm;
  if (!magmaanr::saturated_from_stage1(fit, sm)) {
    Rcpp::stop("magmaan: ML2S modification indices require fit$stage1");
  }
  const std::string stage2_weight = fit.containsElementNamed("stage2_weight")
      ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
  auto w_or = magmaan::estimate::fiml::two_stage_stage2_weight_structured(
      sm, magmaanr::two_stage_weight_from_arg(stage2_weight),
      ml2s_dls_options_from_fit(fit));
  if (!w_or.has_value()) stop_post(w_or.error());
  return std::move(*w_or);
}

// ML2S score tables report `mi` as the naive Stage-2 statistic: Stage-1 EM
// moments treated as complete-data moments. Robust tables add `mi.scaled`,
// whose meat carries the Stage-1 missing-data uncertainty.
Rcpp::DataFrame label_ml2s_score_table(Rcpp::DataFrame df) {
  df.attr("mi_type") = "naive_stage2";
  return df;
}

void validate_fiml_robust_score_options(
    const std::string& bread, const std::string& moments,
    const std::string& cov, SEXP weight, bool estimated_weight) {
  if (bread != "observed") {
    Rcpp::stop("magmaan: FIML robust MI/release requires bread='observed'; "
               "expected-information bread is unsupported");
  }
  if (moments != "structured" || cov != "empirical") {
    Rcpp::stop("magmaan: FIML robust MI/release uses observed-pattern casewise "
               "score covariance; only moments='structured', cov='empirical' "
               "are supported");
  }
  if (!Rf_isNull(weight) || estimated_weight) {
    Rcpp::stop("magmaan: FIML robust MI/release has no second-stage weight; "
               "weight and estimated_weight are unsupported");
  }
}

void prepare_ml2s_score_inputs(Rcpp::List fit, const Ctx& ctx, SEXP raw_arg,
                               SEXP weight, const std::string& bread,
                               const std::string& moments,
                               const std::string& cov, bool estimated_weight,
                               Ml2sScoreInputs& in) {
  if (!Rf_isNull(weight)) {
    Rcpp::stop("magmaan: ML2S robust MI/release builds its Stage-2 weight "
               "from Stage 1; omit `weight`");
  }
  if (!Rf_isNull(raw_arg)) {
    Rcpp::stop("magmaan: ML2S robust MI/release uses the fit's Stage-1 "
               "moments and retained data; omit `data`");
  }
  if (bread != "expected" || moments != "structured" || cov != "empirical") {
    Rcpp::stop("magmaan: ML2S robust MI/release uses the expected Stage-2 "
               "bread and the Stage-1 moment covariance as meat; only "
               "bread='expected', moments='structured', cov='empirical' are "
               "supported");
  }
  if (!magmaanr::saturated_from_stage1(fit, in.sm)) {
    Rcpp::stop("magmaan: ML2S robust MI/release requires fit$stage1");
  }
  const std::string stage2_weight = fit.containsElementNamed("stage2_weight")
      ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
  in.opts.weight = magmaanr::two_stage_weight_from_arg(stage2_weight);
  in.opts.dls = ml2s_dls_options_from_fit(fit);
  in.opts.robust.estimated_weight = estimated_weight;
  if (estimated_weight && ml2s_weight_needs_raw_ij(in.opts.weight)) {
    if (!fit.containsElementNamed("raw_data") || Rf_isNull(fit["raw_data"])) {
      Rcpp::stop("magmaan: estimated_weight ML2S robust MI/release requires "
                 "an ML2S fit carrying $raw_data");
    }
    in.raw = std::make_unique<magmaan::data::RawData>(
        fiml_raw_from_arg(ctx.rep, fit["raw_data"]));
    const FimlPack& pack = fiml_pack_for_fit(fit, *in.raw, in.owned_pack);
    in.opts.robust.raw = in.raw.get();
    in.opts.robust.pack = &pack;
    in.opts.robust.h1 = &fiml_h1_for_fit(fit, *in.raw, pack, in.owned_h1);
  }
}

// A caller-supplied data block must not borrow the fitted missingness cache:
// its patterns and sufficient statistics can differ even at the same dimensions.
const FimlPack& fiml_robust_score_pack(
    Rcpp::List fit, SEXP raw_arg, const magmaan::data::RawData& raw,
    std::unique_ptr<FimlPack>& owned) {
  if (Rf_isNull(raw_arg)) return fiml_pack_for_fit(fit, raw, owned);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack.has_value()) stop_fit(pack.error());
  owned = std::make_unique<FimlPack>(std::move(*pack));
  return *owned;
}

SEXP fiml_robust_score_data(Rcpp::List fit, SEXP raw_arg) {
  if (!Rf_isNull(raw_arg)) return raw_arg;
  if (fit.containsElementNamed("raw_data") && !Rf_isNull(fit["raw_data"])) {
    return fit["raw_data"];
  }
  Rcpp::stop("magmaan: FIML robust MI/release requires fit$raw_data or data=");
}

magmaan::inference::frontier::ScoreFlipCalibration
score_flip_calibration_from_string(const std::string& calibration,
                                   int n_flips) {
  using Calibration =
      magmaan::inference::frontier::ScoreFlipCalibration;
  Calibration calibration_kind = Calibration::All;
  if (calibration == "asymptotic") {
    calibration_kind = Calibration::AsymptoticOnly;
  } else if (calibration == "effective") {
    calibration_kind = Calibration::Effective;
  } else if (calibration == "effective-standardized") {
    calibration_kind = Calibration::EffectiveStandardized;
  } else if (calibration != "all") {
    Rcpp::stop("magmaan: score_flip_test calibration must be 'asymptotic', "
               "'effective', 'effective-standardized', or 'all'");
  }
  if (calibration_kind != Calibration::AsymptoticOnly && n_flips < 1) {
    Rcpp::stop("magmaan: score_flip_test n_flips must be positive");
  }
  return calibration_kind;
}

magmaan::inference::frontier::ScoreFlipMultiplierStudentization
score_flip_multiplier_studentization_from_string(
    const std::string& studentization) {
  using Studentization =
      magmaan::inference::frontier::ScoreFlipMultiplierStudentization;
  if (studentization == "none") return Studentization::None;
  if (studentization == "weighted-meat") {
    return Studentization::WeightedMeat;
  }
  Rcpp::stop("magmaan: score_flip_test multiplier studentization must be "
             "'none' or 'weighted-meat'");
  return Studentization::None;
}

const char* score_flip_sensitivity_string(
    magmaan::inference::frontier::ScoreFlipSensitivity sensitivity) {
  using Sensitivity =
      magmaan::inference::frontier::ScoreFlipSensitivity;
  switch (sensitivity) {
    case Sensitivity::ExpectedInformation:
      return "expected";
    case Sensitivity::ObservedInformation:
      return "observed";
    case Sensitivity::ObservedInformationLightShrinkage:
      return "observed-shrink-light";
    case Sensitivity::ObservedInformationSqrtShrinkage:
      return "observed-shrink-sqrt";
    case Sensitivity::SaturatedObservedInformation:
      return "observed-h1";
  }
  return "expected";
}

const char* global_score_metric_string(
    magmaan::inference::frontier::GlobalScoreFlipOptions::Metric metric) {
  using Metric =
      magmaan::inference::frontier::GlobalScoreFlipOptions::Metric;
  switch (metric) {
    case Metric::ExpectedInformation:
      return "expected";
    case Metric::ObservedInformation:
      return "observed";
    case Metric::SaturatedObservedInformation:
      return "observed-h1";
  }
  return "expected";
}

Rcpp::List score_flip_result_to_r(
    const magmaan::inference::frontier::ScoreFlipTestResult& out) {
  return Rcpp::List::create(
      Rcpp::_ ["df"] = out.df,
      Rcpp::_ ["statistic_basic"] = out.statistic_basic,
      Rcpp::_ ["statistic_effective"] = out.statistic_effective,
      Rcpp::_ ["statistic_standardized"] = out.statistic_standardized,
      Rcpp::_ ["p_basic"] = out.p_basic,
      Rcpp::_ ["p_effective"] = out.p_effective,
      Rcpp::_ ["p_standardized"] = out.p_standardized,
      Rcpp::_ ["p_value"] = out.p_value,
      Rcpp::_ ["mc_se_basic"] = out.mc_se_basic,
      Rcpp::_ ["mc_se_effective"] = out.mc_se_effective,
      Rcpp::_ ["mc_se_standardized"] = out.mc_se_standardized,
      Rcpp::_ ["statistic_multiplier_studentized"] =
          out.statistic_multiplier_studentized,
      Rcpp::_ ["p_multiplier_studentized"] =
          out.p_multiplier_studentized,
      Rcpp::_ ["mc_se_multiplier_studentized"] =
          out.mc_se_multiplier_studentized,
      Rcpp::_ ["multiplier_studentized_min_eigenvalue"] =
          out.multiplier_studentized_min_eigenvalue,
      Rcpp::_ ["multiplier_studentized_max_condition"] =
          out.multiplier_studentized_max_condition,
      Rcpp::_ ["p_chisq"] = out.p_chisq,
      Rcpp::_ ["scaling_factor"] = out.scaling_factor,
      Rcpp::_ ["statistic_mean_scaled"] = out.statistic_mean_scaled,
      Rcpp::_ ["p_mean_scaled"] = out.p_mean_scaled,
      Rcpp::_ ["p_mixture"] = out.p_mixture,
      Rcpp::_ ["statistic_sandwich"] = out.statistic_sandwich,
      Rcpp::_ ["p_sandwich"] = out.p_sandwich,
      Rcpp::_ ["sandwich_available"] = out.sandwich_available,
      Rcpp::_ ["sandwich_min_eigenvalue"] = out.sandwich_min_eigenvalue,
      Rcpp::_ ["sandwich_condition"] = out.sandwich_condition,
      Rcpp::_ ["eigenvalues"] = Rcpp::wrap(out.eigvals),
      Rcpp::_ ["n_flips"] = out.n_flips,
      Rcpp::_ ["seed"] = static_cast<double>(out.seed),
      Rcpp::_ ["sensitivity"] =
          score_flip_sensitivity_string(out.sensitivity),
      Rcpp::_ ["nuisance_stationarity_norm"] = out.nuisance_stationarity_norm,
      Rcpp::_ ["min_variance_eigenvalue"] = out.min_variance_eigenvalue,
      Rcpp::_ ["max_variance_condition"] = out.max_variance_condition,
      Rcpp::_ ["mean_variance_relative_shift"] = out.mean_variance_relative_shift,
      Rcpp::_ ["max_variance_relative_shift"] = out.max_variance_relative_shift,
      Rcpp::_ ["setup_seconds"] = out.setup_seconds,
      Rcpp::_ ["resampling_score_seconds"] = out.resampling_score_seconds,
      Rcpp::_ ["resampling_standardization_seconds"] =
          out.resampling_standardization_seconds,
      Rcpp::_ ["asymptotic_seconds"] = out.asymptotic_seconds,
      Rcpp::_ ["total_seconds"] = out.total_seconds);
}

Rcpp::List global_score_flip_result_to_r(
    const magmaan::inference::frontier::GlobalScoreFlipTestResult& out) {
  Rcpp::List result = score_flip_result_to_r(out.flip);
  result["metric"] = global_score_metric_string(out.metric);
  result["sensitivity_shrinkage"] = out.sensitivity_shrinkage;
  result["n_obs"] = out.n_obs;
  result["projected_score"] = Rcpp::wrap(out.projected_score);
  result["projected_metric"] = Rcpp::wrap(out.projected_metric);
  result["projected_meat"] = Rcpp::wrap(out.projected_meat);
  result["saturated_moment_dim"] = out.saturated_moment_dim;
  result["tangent_rank"] = out.tangent_rank;
  result["tangent_min_singular_value"] = out.tangent_min_singular_value;
  result["tangent_condition"] = out.tangent_condition;
  return result;
}

}  // namespace

// infer_information_expected() — mirrors information_expected(pt, rep, samp, est).
// Returns the (n_free × n_free) expected Fisher information matrix at θ̂.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_information_expected(Rcpp::List fit) {
  if (auto cached=magmaanr::ntml_snapshot(fit)) {
    auto info=magmaan::robust::frontier::ntml_information(*cached);
    if (!info) stop_post(info.error()); return Rcpp::wrap(**info);
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r = magmaan::inference::information_expected(ctx.pt, ctx.rep, ctx.samp, est);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_information_observed_fd() — mirrors information_observed_fd(...).
// Observed-information matrix via central-difference Hessian of the analytic
// ML gradient.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_information_observed_fd(Rcpp::List fit,
                                                  double h_step = 1e-4) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r = magmaan::inference::information_observed_fd(ctx.pt, ctx.rep, ctx.samp, est, h_step);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_information_observed_analytic() — mirrors information_observed_analytic(...).
// Closed-form observed-information matrix; rejects mean-structure models (use
// the FD variant there).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_information_observed_analytic(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r = magmaan::inference::information_observed_analytic(ctx.pt, ctx.rep, ctx.samp, est);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// frontier_newton_accuracy_impl() — opt-in local accuracy diagnostic for a
// complete-data ML fit (estimate::frontier::newton_accuracy_ml). Never errors
// on numerical failure: the status field says why no distance is available.
//
// [[Rcpp::export]]
Rcpp::List frontier_newton_accuracy_impl(Rcpp::List fit, double budget,
                                        bool psd) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::estimate::frontier::NewtonAccuracyOptions opts;
  opts.budget = budget;
  const auto a = psd
      ? magmaan::estimate::frontier::newton_accuracy_ml_psd(
            ctx.pt, ctx.rep, ctx.samp, est, opts)
      : magmaan::estimate::frontier::newton_accuracy_ml(
            ctx.pt, ctx.rep, ctx.samp, est, opts);
  return newton_accuracy_to_r(a);
}

// infer_information_cross_products() — mirrors information_cross_products(...).
// Parameter-level outer-product-of-scores Σᵢ sᵢsᵢᵀ at θ̂. Needs raw data
// (per-case moment contributions). Mplus calls the SE built from this method
// "MLF".
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_information_cross_products(Rcpp::List fit,
                                                     SEXP raw_data) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = complete_raw_from_arg(ctx.rep, raw_data);
  auto r = magmaan::inference::information_cross_products(
      ctx.pt, ctx.rep, ctx.samp, raw, est);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_vcov() — mirrors vcov(info, pt). Inverts the information matrix,
// applying the constraint projection K·(KᵀIK)⁻¹·Kᵀ when shared labels /
// invariance / general-linear equalities are active. Takes `fit` so the
// partable comes along (constraints live on pt). The `info` matrix is the
// only numerical input — caller chooses which information variant.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_vcov(Rcpp::NumericMatrix info, Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const Eigen::MatrixXd info_m = Rcpp::as<Eigen::MatrixXd>(info);
  // θ̂ comes along so the constraint Jacobian H(θ̂) can be evaluated when the
  // model carries nonlinear equality constraints; ignored otherwise.
  const Eigen::VectorXd theta = Rcpp::as<Eigen::VectorXd>(fit["theta"]);
  auto r = magmaan::inference::vcov(info_m, ctx.pt, theta);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_vcov_partable() — primitive form of infer_vcov(): the information
// matrix plus the model partable, without requiring a fit list.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_vcov_partable(Rcpp::NumericMatrix info, SEXP partable) {
  auto parsed = partable_from_arg(partable, "infer_vcov_partable");
  const Eigen::MatrixXd info_m = Rcpp::as<Eigen::MatrixXd>(info);
  auto r = magmaan::inference::vcov(info_m, parsed.structure);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_se() — mirrors se(vcov). Returns √diag(vcov); NaN for negative
// diagonal entries (Heywood case). Never errors.
//
// [[Rcpp::export]]
Rcpp::NumericVector infer_se(Rcpp::NumericMatrix vcov) {
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  return Rcpp::wrap(magmaan::inference::se(vcov_m));
}

// compute_defined_impl() — mirrors measures::effects::compute_defined(flat, pt,
// names, est, vcov). `syntax` must be the original model syntax because the
// lavaan-shaped partable only carries the projected `:=` rows, not their parsed
// expression trees.
//
// [[Rcpp::export]]
Rcpp::DataFrame compute_defined_impl(std::string syntax,
                                     Rcpp::List fit,
                                     Rcpp::NumericMatrix vcov) {
  // Exposed for ordinal/mixed fits too: ctx_from_fit() rebuilds the prepared
  // partable, which lines up with the reduced est/vcov, and the delta-method
  // value/SE are parameterization-agnostic (mirrors api::compute_defined).
  auto flat_or = magmaan::parse::Parser::parse(syntax);
  if (!flat_or.has_value()) {
    const auto& e = flat_or.error();
    Rcpp::stop("magmaan parse error at %u:%u (bytes %u..%u): %s",
               e.span.line, e.span.col, e.span.begin, e.span.end, e.detail);
  }
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto defs_or = magmaan::measures::effects::compute_defined(
      *flat_or, ctx.pt, ctx.names, est, vcov_m);
  if (!defs_or.has_value()) stop_post(defs_or.error());

  const R_xlen_t n = static_cast<R_xlen_t>(defs_or->entries.size());
  Rcpp::CharacterVector lhs(n), op(n), rhs(n);
  Rcpp::NumericVector est_out(n), se(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const auto& entry = defs_or->entries[static_cast<std::size_t>(i)];
    lhs[i] = entry.name;
    op[i] = ":=";
    rhs[i] = "";
    est_out[i] = entry.value;
    se[i] = entry.se;
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["lhs"] = lhs,
      Rcpp::_["op"] = op,
      Rcpp::_["rhs"] = rhs,
      Rcpp::_["est"] = est_out,
      Rcpp::_["se"] = se,
      Rcpp::_["stringsAsFactors"] = false);
}

// infer_chi2_stat() — mirrors chi2_stat(samp, est). Returns 2·N_total·fmin =
// N·F (the GOF χ²). This primitive does not need a fit object: sample_stats
// supplies nobs and fmin is the optimizer's minimised objective ½·F (fit$fmin
// when called after fit_fit()); see project/design/numerical-conventions.md.
//
// [[Rcpp::export]]
double infer_chi2_stat(Rcpp::List sample_stats, double fmin) {
  if (!sample_stats.containsElementNamed("nobs"))
    Rcpp::stop("magmaan: sample_stats must contain $nobs");
  magmaan::data::SampleStats samp;
  Rcpp::IntegerVector nv = Rcpp::as<Rcpp::IntegerVector>(sample_stats["nobs"]);
  for (R_xlen_t i = 0; i < nv.size(); ++i)
    samp.n_obs.push_back(static_cast<std::int64_t>(nv[i]));
  magmaan::estimate::Estimates est;
  est.fmin = fmin;
  return magmaan::inference::chi2_stat(samp, est);
}

// infer_df_stat() — mirrors df_stat(pt, samp). Returns Σ_b p_b(p_b+1)/2 (+ means)
// − fixed_x − n_free + constraint.rank. Pure function of the model and the
// data dimensions; doesn't depend on θ̂. Errors on unenforced constraints.
//
// [[Rcpp::export]]
int infer_df_stat(SEXP partable, Rcpp::List sample_stats) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "infer_df_stat");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto r = magmaan::inference::df_stat(ctx.pt, ctx.samp);
  if (!r.has_value()) stop_post(r.error());
  return *r;
}

// infer_baseline() — mirrors baseline_chi2(samp). Takes sample stats directly.
//
// [[Rcpp::export]]
Rcpp::List infer_baseline(Rcpp::List sample_stats) {
  if (!sample_stats.containsElementNamed("S") || !sample_stats.containsElementNamed("nobs"))
    Rcpp::stop("magmaan: sample_stats must contain $S and $nobs");
  magmaan::data::SampleStats samp;
  Rcpp::List Sl(sample_stats["S"]);
  Rcpp::IntegerVector nv = Rcpp::as<Rcpp::IntegerVector>(sample_stats["nobs"]);
  if (Sl.size() != nv.size())
    Rcpp::stop("magmaan: sample_stats$S and sample_stats$nobs must have the same length");
  for (R_xlen_t b = 0; b < Sl.size(); ++b) {
    samp.S.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Sl[b])));
    samp.n_obs.push_back(static_cast<std::int64_t>(nv[b]));
  }
  const magmaan::measures::BaselineFit bl = magmaan::measures::baseline_chi2(samp);
  return Rcpp::List::create(Rcpp::_["chi2"] = bl.chi2, Rcpp::_["df"] = bl.df);
}

// infer_baseline_fit() — partable-aware baseline_chi2(pt, samp). Applies the
// fixed.x exogenous correction: lavaan's independence/baseline model frees the
// exogenous (co)variances, so baseline.df drops by px(px-1)/2 and baseline.chisq
// loses the exo-block fit. A no-op when there are < 2 exogenous variables, so it
// is safe for every fit; only fixed.x models change (matching lavaan).
//
// [[Rcpp::export]]
Rcpp::List infer_baseline_fit(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::measures::BaselineFit bl =
      magmaan::measures::baseline_chi2(ctx.pt, ctx.samp);
  return Rcpp::List::create(Rcpp::_["chi2"] = bl.chi2, Rcpp::_["df"] = bl.df);
}

// inference_modification_indices() — mirrors inference::modification_indices()
// for ML/FIML/ML2S/continuous LS fit objects. WLS-computed fits use the
// fitting weight recorded in fit$W; fits without one need `weight`.
//
// [[Rcpp::export]]
Rcpp::DataFrame inference_modification_indices(
    Rcpp::List fit, SEXP weight = R_NilValue,
    std::string information = "expected", std::string candidates = "fixed",
    bool include_loadings = true, bool include_covariances = true,
    double h_step = 1e-4) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  const auto opts = modification_options_from(
      information, candidates, include_loadings, include_covariances);
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;
  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, weight, "ordinal_stats", "ordinal modification indices"));
    out = magmaan::estimate::modification_indices_ordinal(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "ordinal modification indices"),
        opts, ordinal_parameterization_from_string(
                  fit.containsElementNamed("parameterization")
                      ? Rcpp::as<std::string>(fit["parameterization"])
                      : ordinal_parameterization_attr(fit["partable"])));
  } else if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, weight, "mixed_ordinal_stats",
        "mixed ordinal modification indices"));
    out = magmaan::estimate::modification_indices_mixed_ordinal(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "mixed ordinal modification indices"),
        opts, ordinal_parameterization_from_string(
                  fit.containsElementNamed("parameterization")
                      ? Rcpp::as<std::string>(fit["parameterization"])
                      : ordinal_parameterization_attr(fit["partable"])));
  } else if (estimator == "FIML") {
    if (!fit.containsElementNamed("raw_data")) {
      Rcpp::stop("magmaan: FIML modification indices require fit$raw_data");
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
    out = magmaan::inference::modification_indices_fiml(
        ctx.pt, ctx.rep, raw, est, opts, pack, magmaan::estimate::fiml::FIML{},
        h_step);
  } else if (estimator == "ULS") {
    out = magmaan::inference::modification_indices(
        ctx.pt, ctx.rep, ctx.samp, est, magmaan::estimate::gmm::Weight{}, opts);
  } else if (estimator == "GLS") {
    auto ev_or = magmaan::model::ModelEvaluator::build(ctx.pt, ctx.rep);
    if (!ev_or.has_value()) stop_model(ev_or.error());
    auto w_or = magmaan::estimate::gmm::normal_theory_weight(
        *ev_or, ctx.samp, est.theta);
    if (!w_or.has_value()) stop_fit(w_or.error());
    out = magmaan::inference::modification_indices(
        ctx.pt, ctx.rep, ctx.samp, est, *w_or, opts);
  } else if (estimator == "WLS") {
    auto w = continuous_ls_weight(fit, ctx, est, estimator, weight,
                                  "modification indices");
    out = magmaan::inference::modification_indices(
        ctx.pt, ctx.rep, ctx.samp, est, w, opts);
  } else if (estimator == "ML" || estimator.empty()) {
    out = magmaan::inference::modification_indices(
        ctx.pt, ctx.rep, ctx.samp, est, opts);
  } else if (is_ml2s_estimator_label(estimator)) {
    // ML2S: fit$S/sample_mean are the Stage-1 EM moments the Stage-2 fit used,
    // so ctx.samp carries them and the sweep is the naive Stage-2 score test
    // on them (ML for NT, the moment quadratic with the Stage-2 weight
    // otherwise). Stage-1 uncertainty enters only modification_indices_robust.
    if (estimator == "ML2S") {
      out = magmaan::inference::modification_indices(
          ctx.pt, ctx.rep, ctx.samp, est, opts);
    } else {
      out = magmaan::inference::modification_indices(
          ctx.pt, ctx.rep, ctx.samp, est, ml2s_stage2_weight_for_fit(fit),
          opts);
    }
    if (!out.has_value()) stop_post(out.error());
    return label_ml2s_score_table(score_table_df(*out, ctx.names));
  } else {
    Rcpp::stop("magmaan: modification indices are not yet exposed for estimator '%s'",
               estimator);
  }
  if (!out.has_value()) stop_post(out.error());
  return score_table_df(*out, ctx.names);
}

// inference_score_tests() — mirrors inference::score_tests() for ML/FIML/ML2S
// and continuous LS fit objects. WLS-computed fits use the recorded fit$W.
//
// [[Rcpp::export]]
Rcpp::DataFrame inference_score_tests(Rcpp::List fit, SEXP weight = R_NilValue,
                                      double h_step = 1e-4) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;
  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, weight, "ordinal_stats", "ordinal score tests"));
    out = magmaan::estimate::score_tests_ordinal(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "ordinal score tests"),
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])));
  } else if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, weight, "mixed_ordinal_stats", "mixed ordinal score tests"));
    out = magmaan::estimate::score_tests_mixed_ordinal(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "mixed ordinal score tests"),
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])));
  } else if (estimator == "FIML") {
    if (!fit.containsElementNamed("raw_data")) {
      Rcpp::stop("magmaan: FIML score tests require fit$raw_data");
    }
    magmaan::data::RawData raw = fiml_raw_from_arg(ctx.rep, fit["raw_data"]);
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit, raw, owned_pack);
    out = magmaan::inference::score_tests_fiml(
        ctx.pt, ctx.rep, raw, est, pack, magmaan::estimate::fiml::FIML{},
        h_step);
  } else if (estimator == "ULS") {
    out = magmaan::inference::score_tests(
        ctx.pt, ctx.rep, ctx.samp, est, magmaan::estimate::gmm::Weight{});
  } else if (estimator == "GLS") {
    auto ev_or = magmaan::model::ModelEvaluator::build(ctx.pt, ctx.rep);
    if (!ev_or.has_value()) stop_model(ev_or.error());
    auto w_or = magmaan::estimate::gmm::normal_theory_weight(
        *ev_or, ctx.samp, est.theta);
    if (!w_or.has_value()) stop_fit(w_or.error());
    out = magmaan::inference::score_tests(ctx.pt, ctx.rep, ctx.samp, est,
                                          *w_or);
  } else if (estimator == "WLS") {
    auto w = continuous_ls_weight(fit, ctx, est, estimator, weight,
                                  "score tests");
    out = magmaan::inference::score_tests(ctx.pt, ctx.rep, ctx.samp, est, w);
  } else if (estimator == "ML" || estimator.empty()) {
    out = magmaan::inference::score_tests(ctx.pt, ctx.rep, ctx.samp, est);
  } else if (is_ml2s_estimator_label(estimator)) {
    // Naive Stage-2 release tests on the Stage-1 EM moments; see
    // inference_modification_indices().
    out = estimator == "ML2S"
        ? magmaan::inference::score_tests(ctx.pt, ctx.rep, ctx.samp, est)
        : magmaan::inference::score_tests(ctx.pt, ctx.rep, ctx.samp, est,
                                          ml2s_stage2_weight_for_fit(fit));
    if (!out.has_value()) stop_post(out.error());
    return label_ml2s_score_table(score_table_df(*out, ctx.names));
  } else {
    Rcpp::stop("magmaan: score tests are not yet exposed for estimator '%s'",
               estimator);
  }
  if (!out.has_value()) stop_post(out.error());
  return score_table_df(*out, ctx.names);
}

// [[Rcpp::export]]
Rcpp::DataFrame inference_modification_indices_robust(
    Rcpp::List fit, SEXP raw = R_NilValue, SEXP weight = R_NilValue,
    std::string bread = "observed", std::string moments = "structured",
    std::string cov = "empirical", std::string information = "expected",
    std::string candidates = "fixed", bool include_loadings = true,
    bool include_covariances = true, bool estimated_weight = true,
    SEXP gamma = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  const auto base = modification_options_from(
      information, candidates, include_loadings, include_covariances);
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;

  validate_score_gamma_request(gamma, estimated_weight, estimator, cov);

  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "ordinal_stats",
        "ordinal robust modification indices"));
    replace_score_nacov(stats, gamma);
    out = magmaan::estimate::frontier::modification_indices_ordinal_robust(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "ordinal robust modification indices"),
        base,
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])),
        estimated_weight);
  } else if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "mixed_ordinal_stats",
        "mixed ordinal robust modification indices"));
    replace_score_nacov(stats, gamma);
    out = magmaan::estimate::frontier::modification_indices_mixed_ordinal_robust(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "mixed ordinal robust modification indices"),
        base,
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])),
        estimated_weight);
  } else if (estimator == "FIML") {
    validate_fiml_robust_score_options(bread, moments, cov, weight,
                                     estimated_weight);
    if (information != "observed") {
      Rcpp::stop("magmaan: FIML robust modification indices require "
                 "information='observed'; expected-statistic information "
                 "is unsupported");
    }
    const auto rd = fiml_raw_from_arg(ctx.rep, fiml_robust_score_data(fit, raw));
    std::unique_ptr<FimlPack> owned_pack;
    const auto& pack = fiml_robust_score_pack(fit, raw, rd, owned_pack);
    out = magmaan::inference::frontier::modification_indices_fiml_robust(
        ctx.pt, ctx.rep, rd, est, pack, base);
  } else if (is_ml2s_estimator_label(estimator)) {
    Ml2sScoreInputs in;
    prepare_ml2s_score_inputs(fit, ctx, raw, weight, bread, moments, cov,
                              estimated_weight, in);
    in.opts.base = base;
    out = magmaan::inference::frontier::modification_indices_ml2s(
        ctx.pt, ctx.rep, in.sm, est, in.opts);
    if (!out.has_value()) stop_post(out.error());
    return label_ml2s_score_table(score_table_df(*out, ctx.names));
  } else {
    const bool is_ml = (estimator == "ML" || estimator.empty());
    magmaan::inference::frontier::RobustScoreOptions opts;
    opts.base = base;
    if (estimated_weight) {
      // The complete (estimated-weight) sandwich needs the fitting data and an
      // estimated second-stage weight; ML has none.
      if (is_ml) {
        Rcpp::stop("magmaan: estimated_weight robust modification indices need "
                   "an estimated second-stage weight (GLS, WLS, DWLS or DLS, "
                   "or an ordinal fit), not ML");
      }
      if (Rf_isNull(raw)) {
        Rcpp::stop("magmaan: estimated_weight robust modification indices "
                   "require the fitting data; pass data=");
      }
      opts.spec = spec_from(bread, moments, cov);
      opts.estimated_weight = true;
      opts.ij_weight_mode =
          continuous_ij_mode_for_fit(fit, estimator, &opts.dls_opts);
      magmaan::data::RawData rd = complete_raw_from_arg(ctx.rep, raw);
      auto w = continuous_ls_weight(fit, ctx, est, estimator, weight,
                                    "modification indices");
      out = magmaan::inference::frontier::modification_indices_robust(
          ctx.pt, ctx.rep, ctx.samp, rd, est, w, opts);
    } else {
      const bool model_implied = (cov == "model_implied");
      opts.spec = spec_from(bread, moments, cov);
      if (!Rf_isNull(gamma)) {
        const auto blocks = supplied_gamma_blocks(gamma, continuous_gamma_dimensions(ctx));
        if (is_ml) {
          const auto full = supplied_ml_gamma(blocks, ctx);
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, full, est, opts);
        } else {
          const auto w = continuous_ls_weight(fit, ctx, est, estimator, weight,
                                              "supplied-Gamma score inference");
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, blocks, est, w, opts);
        }
      } else if (model_implied) {
        if (is_ml) {
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, est, opts);
        } else {
          auto w = continuous_ls_weight(fit, ctx, est, estimator, weight,
                                        "modification indices");
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, est, w, opts);
        }
      } else {
        if (Rf_isNull(raw)) {
          Rcpp::stop("magmaan: robust modification indices with cov='%s' "
                     "require the fitting data; pass data= (or use "
                     "cov='model_implied')", cov.c_str());
        }
        magmaan::data::RawData rd = complete_raw_from_arg(ctx.rep, raw);
        if (is_ml) {
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, rd, est, opts);
        } else {
          auto w = continuous_ls_weight(fit, ctx, est, estimator, weight,
                                        "modification indices");
          out = magmaan::inference::frontier::modification_indices_robust(
              ctx.pt, ctx.rep, ctx.samp, rd, est, w, opts);
        }
      }
    }
  }
  if (!out.has_value()) stop_post(out.error());
  return score_table_df(*out, ctx.names);
}

// [[Rcpp::export]]
Rcpp::DataFrame inference_score_tests_robust(
    Rcpp::List fit, SEXP raw = R_NilValue, SEXP weight = R_NilValue,
    std::string bread = "observed", std::string moments = "structured",
    std::string cov = "empirical", bool estimated_weight = true,
    SEXP gamma = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"])
      : "";
  const bool is_ordinal_fit = fit.containsElementNamed("ordinal") &&
                              Rcpp::as<bool>(fit["ordinal"]);
  const bool is_mixed_ordinal_fit =
      fit.containsElementNamed("mixed_ordinal") &&
      Rcpp::as<bool>(fit["mixed_ordinal"]);
  magmaan::post_expected<magmaan::inference::ScoreTestTable> out;

  validate_score_gamma_request(gamma, estimated_weight, estimator, cov);

  if (is_ordinal_fit) {
    auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "ordinal_stats", "ordinal robust score tests"));
    replace_score_nacov(stats, gamma);
    out = magmaan::estimate::frontier::score_tests_ordinal_robust(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "ordinal robust score tests"),
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])),
        estimated_weight);
  } else if (is_mixed_ordinal_fit) {
    auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(
        fit, R_NilValue, "mixed_ordinal_stats",
        "mixed ordinal robust score tests"));
    replace_score_nacov(stats, gamma);
    out = magmaan::estimate::frontier::score_tests_mixed_ordinal_robust(
        ctx.pt, ctx.rep, stats, est,
        ordinal_weight_from_estimator(
            ordinal_weight_for_postfit(fit, estimator),
            "mixed ordinal robust score tests"),
        ordinal_parameterization_from_string(
            fit.containsElementNamed("parameterization")
                ? Rcpp::as<std::string>(fit["parameterization"])
                : ordinal_parameterization_attr(fit["partable"])),
        estimated_weight);
  } else if (estimator == "FIML") {
    validate_fiml_robust_score_options(bread, moments, cov, weight,
                                     estimated_weight);
    const auto rd = fiml_raw_from_arg(ctx.rep, fiml_robust_score_data(fit, raw));
    std::unique_ptr<FimlPack> owned_pack;
    const auto& pack = fiml_robust_score_pack(fit, raw, rd, owned_pack);
    out = magmaan::inference::frontier::score_tests_fiml_robust(
        ctx.pt, ctx.rep, rd, est, pack);
  } else if (is_ml2s_estimator_label(estimator)) {
    Ml2sScoreInputs in;
    prepare_ml2s_score_inputs(fit, ctx, raw, weight, bread, moments, cov,
                              estimated_weight, in);
    out = magmaan::inference::frontier::score_tests_ml2s(
        ctx.pt, ctx.rep, in.sm, est, in.opts);
    if (!out.has_value()) stop_post(out.error());
    return label_ml2s_score_table(score_table_df(*out, ctx.names));
  } else {
    const bool is_ml = (estimator == "ML" || estimator.empty());
    magmaan::inference::frontier::RobustScoreOptions opts;
    if (estimated_weight) {
      if (is_ml) {
        Rcpp::stop("magmaan: estimated_weight robust score tests need an "
                   "estimated second-stage weight (GLS, WLS, DWLS or DLS, or "
                   "an ordinal fit), not ML");
      }
      if (Rf_isNull(raw)) {
        Rcpp::stop("magmaan: estimated_weight robust score tests require the "
                   "fitting data; pass data=");
      }
      opts.spec = spec_from(bread, moments, cov);
      opts.estimated_weight = true;
      opts.ij_weight_mode =
          continuous_ij_mode_for_fit(fit, estimator, &opts.dls_opts);
      magmaan::data::RawData rd = complete_raw_from_arg(ctx.rep, raw);
      auto w = continuous_ls_weight(fit, ctx, est, estimator, weight, "score tests");
      out = magmaan::inference::frontier::score_tests_robust(
          ctx.pt, ctx.rep, ctx.samp, rd, est, w, opts);
    } else {
      const bool model_implied = (cov == "model_implied");
      opts.spec = spec_from(bread, moments, cov);
      if (!Rf_isNull(gamma)) {
        const auto blocks = supplied_gamma_blocks(gamma, continuous_gamma_dimensions(ctx));
        if (is_ml) {
          const auto full = supplied_ml_gamma(blocks, ctx);
          out = magmaan::inference::frontier::score_tests_robust(
              ctx.pt, ctx.rep, ctx.samp, full, est, opts);
        } else {
          const auto w = continuous_ls_weight(fit, ctx, est, estimator, weight,
                                              "supplied-Gamma score inference");
          out = magmaan::inference::frontier::score_tests_robust(
              ctx.pt, ctx.rep, ctx.samp, blocks, est, w, opts);
        }
      } else if (model_implied) {
        if (is_ml) {
          Rcpp::stop("magmaan: ML robust score tests require the fitting data "
                     "(cov='empirical'); pass data=");
        }
        auto w = continuous_ls_weight(fit, ctx, est, estimator, weight, "score tests");
        out = magmaan::inference::frontier::score_tests_robust(
            ctx.pt, ctx.rep, ctx.samp, est, w, opts);
      } else {
        if (Rf_isNull(raw)) {
          Rcpp::stop("magmaan: robust score tests with cov='%s' require the "
                     "fitting data; pass data= (or use cov='model_implied')",
                     cov.c_str());
        }
        magmaan::data::RawData rd = complete_raw_from_arg(ctx.rep, raw);
        if (is_ml) {
          out = magmaan::inference::frontier::score_tests_robust(
              ctx.pt, ctx.rep, ctx.samp, rd, est, opts);
        } else {
          auto w =
              continuous_ls_weight(fit, ctx, est, estimator, weight, "score tests");
          out = magmaan::inference::frontier::score_tests_robust(
              ctx.pt, ctx.rep, ctx.samp, rd, est, w, opts);
        }
      }
    }
  }
  if (!out.has_value()) stop_post(out.error());
  return score_table_df(*out, ctx.names);
}

// [[Rcpp::export]]
Rcpp::List inference_score_flip_test(Rcpp::List fit_H1, Rcpp::List fit_H0,
                                     SEXP raw, int n_flips = 999,
                                     double seed = 1.0,
                                     std::string calibration = "effective",
                                     std::string multiplier = "rademacher",
                                     double two_point_skewness = 1.0,
                                     bool center_multiplier_scores = false,
                                     std::string multiplier_studentization =
                                         "none",
                                     std::string sensitivity = "observed") {
  const auto calibration_kind =
      score_flip_calibration_from_string(calibration, n_flips);
  const auto multiplier_kind =
      score_flip_multiplier_from_string(multiplier);
  const auto multiplier_studentization_kind =
      score_flip_multiplier_studentization_from_string(
          multiplier_studentization);
  const auto sensitivity_kind =
      score_flip_sensitivity_from_string(sensitivity);
  if (!std::isfinite(seed) || seed < 0.0 || seed > 9007199254740991.0) {
    Rcpp::stop("magmaan: score_flip_test seed must be an integer in [0, 2^53-1]");
  }
  Ctx h1 = ctx_from_fit(fit_H1);
  Ctx h0 = ctx_from_fit(fit_H0);
  const std::string est1 = fit_H1.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit_H1["estimator"]) : "ML";
  const std::string est0 = fit_H0.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit_H0["estimator"]) : "ML";
  if (est1 != est0 || (est1 != "ML" && est1 != "FIML")) {
    Rcpp::stop("magmaan: score_flip_test requires a matching ML or FIML fit pair");
  }
  magmaan::inference::frontier::ScoreFlipOptions options;
  options.n_flips = n_flips;
  options.seed = static_cast<std::uint64_t>(seed);
  options.calibration = calibration_kind;
  options.multiplier = multiplier_kind;
  options.two_point_skewness = two_point_skewness;
  options.center_multiplier_scores = center_multiplier_scores;
  options.multiplier_studentization =
      multiplier_studentization_kind;
  options.sensitivity = sensitivity_kind;
  magmaan::post_expected<magmaan::inference::frontier::ScoreFlipTestResult> out;
  if (est1 == "FIML") {
    magmaan::data::RawData rd = fiml_raw_from_arg(h1.rep, raw);
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit_H0, rd, owned_pack);
    out = magmaan::inference::frontier::score_flip_test(
        h1.pt, h1.rep, h0.pt, h0.rep, rd, pack, est_from_fit(fit_H0), options);
  } else {
    magmaan::data::RawData rd = complete_raw_from_arg(h1.rep, raw);
    out = magmaan::inference::frontier::score_flip_test(
        h1.pt, h1.rep, h0.pt, h0.rep, h0.samp, rd, est_from_fit(fit_H0), options);
  }
  if (!out.has_value()) stop_post(out.error());
  return score_flip_result_to_r(*out);
}

// Score/flip calibration needs H0 estimates but only H1's model tangent. This
// overload accepts H1's partable directly so simulations do not fit an unused
// unrestricted model merely to reconstruct its structure.
// [[Rcpp::export]]
Rcpp::List inference_score_flip_test_model(
    SEXP partable_H1, Rcpp::List fit_H0, SEXP raw, int n_flips = 999,
    double seed = 1.0, std::string calibration = "effective",
    std::string multiplier = "rademacher",
    double two_point_skewness = 1.0,
    bool center_multiplier_scores = false,
    std::string multiplier_studentization = "none",
    std::string sensitivity = "observed") {
  const auto calibration_kind =
      score_flip_calibration_from_string(calibration, n_flips);
  const auto multiplier_kind =
      score_flip_multiplier_from_string(multiplier);
  const auto multiplier_studentization_kind =
      score_flip_multiplier_studentization_from_string(
          multiplier_studentization);
  const auto sensitivity_kind =
      score_flip_sensitivity_from_string(sensitivity);
  if (!std::isfinite(seed) || seed < 0.0 || seed > 9007199254740991.0) {
    Rcpp::stop("magmaan: score_flip_test seed must be an integer in [0, 2^53-1]");
  }
  Ctx h0 = ctx_from_fit(fit_H0);
  Rcpp::List sample_stats = Rcpp::List::create(
      Rcpp::_ ["S"] = fit_H0["S"],
      Rcpp::_ ["nobs"] = fit_H0["nobs"],
      Rcpp::_ ["mean"] = fit_H0.containsElementNamed("sample_mean")
          ? SEXP(fit_H0["sample_mean"]) : R_NilValue);
  Ctx h1 = ctx_from_partable_sample_stats(
      partable_H1, sample_stats, "inference_score_flip_test_model");
  const std::string estimator = fit_H0.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit_H0["estimator"]) : "ML";
  if (estimator != "ML" && estimator != "FIML") {
    Rcpp::stop("magmaan: score_flip_test requires an ML or FIML null fit");
  }
  magmaan::inference::frontier::ScoreFlipOptions options;
  options.n_flips = n_flips;
  options.seed = static_cast<std::uint64_t>(seed);
  options.calibration = calibration_kind;
  options.multiplier = multiplier_kind;
  options.two_point_skewness = two_point_skewness;
  options.center_multiplier_scores = center_multiplier_scores;
  options.multiplier_studentization =
      multiplier_studentization_kind;
  options.sensitivity = sensitivity_kind;
  magmaan::post_expected<magmaan::inference::frontier::ScoreFlipTestResult> out;
  if (estimator == "FIML") {
    magmaan::data::RawData rd = fiml_raw_from_arg(h1.rep, raw);
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit_H0, rd, owned_pack);
    out = magmaan::inference::frontier::score_flip_test(
        h1.pt, h1.rep, h0.pt, h0.rep, rd, pack, est_from_fit(fit_H0), options);
  } else {
    magmaan::data::RawData rd = complete_raw_from_arg(h1.rep, raw);
    out = magmaan::inference::frontier::score_flip_test(
        h1.pt, h1.rep, h0.pt, h0.rep, h0.samp, rd,
        est_from_fit(fit_H0), options);
  }
  if (!out.has_value()) stop_post(out.error());
  return score_flip_result_to_r(*out);
}

// [[Rcpp::export]]
Rcpp::List inference_global_score_flip_test(
    Rcpp::List fit, SEXP raw, int n_flips = 999, double seed = 1.0,
    std::string multiplier = "rademacher",
    double two_point_skewness = 1.0,
    bool center_multiplier_scores = false,
    std::string multiplier_studentization = "none",
    std::string sensitivity = "auto",
    std::string metric = "expected") {
  if (n_flips < 1) {
    Rcpp::stop("magmaan: global_score_flip_test n_flips must be positive");
  }
  if (!std::isfinite(seed) || seed < 0.0 || seed > 9007199254740991.0) {
    Rcpp::stop("magmaan: global_score_flip_test seed must be an integer in [0, 2^53-1]");
  }
  Ctx ctx = ctx_from_fit(fit);
  const std::string estimator = fit.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit["estimator"]) : "ML";
  if (estimator != "ML" && estimator != "FIML" && estimator != "ML2S") {
    Rcpp::stop("magmaan: global_score_flip_test requires an ML, FIML, or "
               "normal-theory ML2S fit");
  }
  if (sensitivity == "auto") sensitivity = estimator == "FIML" ? "observed" : "expected";
  magmaan::data::RawData rd = estimator == "FIML" || estimator == "ML2S"
      ? fiml_raw_from_arg(ctx.rep, raw)
      : complete_raw_from_arg(ctx.rep, raw);
  magmaan::inference::frontier::GlobalScoreFlipOptions options;
  options.resampling.n_flips = n_flips;
  options.resampling.seed = static_cast<std::uint64_t>(seed);
  options.resampling.multiplier =
      score_flip_multiplier_from_string(multiplier);
  options.resampling.two_point_skewness = two_point_skewness;
  options.resampling.center_multiplier_scores = center_multiplier_scores;
  options.resampling.multiplier_studentization =
      score_flip_multiplier_studentization_from_string(
          multiplier_studentization);
  options.resampling.sensitivity =
      score_flip_sensitivity_from_string(sensitivity);
  options.metric = global_score_metric_from_string(metric);
  magmaan::post_expected<
      magmaan::inference::frontier::GlobalScoreFlipTestResult> out;
  if (estimator == "ML2S") {
    if (fit.containsElementNamed("stage1_regularization") &&
        !Rf_isNull(fit["stage1_regularization"])) {
      Rcpp::stop("magmaan: global_score_flip_test does not support regularized "
                 "Stage-1 ML2S fits");
    }
    const std::string stage2_weight = fit.containsElementNamed("stage2_weight")
        ? Rcpp::as<std::string>(fit["stage2_weight"]) : "nt";
    if (stage2_weight != "nt" && stage2_weight != "NT") {
      Rcpp::stop("magmaan: global_score_flip_test supports only the fixed "
                 "normal-theory ML2S Stage-2 weight");
    }
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit, rd, owned_pack);
    std::unique_ptr<FimlH1> owned_h1;
    const FimlH1& h1 = fiml_h1_for_fit(fit, rd, pack, owned_h1);
    std::unique_ptr<SaturatedMoments> owned_sm;
    const SaturatedMoments& sm =
        fiml_saturated_for_fit(fit, rd, pack, h1, owned_sm);
    out = magmaan::inference::frontier::global_score_flip_test_ml2s(
        std::move(ctx.pt), ctx.rep, rd, pack, h1, sm, est_from_fit(fit),
        options);
  } else if (estimator == "FIML") {
    std::unique_ptr<FimlPack> owned_pack;
    const FimlPack& pack = fiml_pack_for_fit(fit, rd, owned_pack);
    out = magmaan::inference::frontier::global_score_flip_test(
        std::move(ctx.pt), ctx.rep, rd, pack, est_from_fit(fit), options);
  } else {
    out = magmaan::inference::frontier::global_score_flip_test(
        std::move(ctx.pt), ctx.rep, ctx.samp, rd, est_from_fit(fit), options);
  }
  if (!out.has_value()) stop_post(out.error());
  return global_score_flip_result_to_r(*out);
}

// infer_z_test() — mirrors z_test(est, se). `se` is the SE vector from
// infer_se(infer_vcov(info, fit)).
//
// [[Rcpp::export]]
Rcpp::List infer_z_test(Rcpp::List fit, Rcpp::NumericVector se) {
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::VectorXd se_v = Rcpp::as<Eigen::VectorXd>(se);
  const magmaan::inference::ZTestResult zt = magmaan::inference::z_test(est, se_v);
  return Rcpp::List::create(Rcpp::_["z"] = Rcpp::wrap(zt.z),
                            Rcpp::_["pvalue"] = Rcpp::wrap(zt.p_value));
}

// infer_z_test_theta() — primitive form of infer_z_test(): estimate vector plus
// SE vector, without requiring a fit list.
//
// [[Rcpp::export]]
Rcpp::List infer_z_test_theta(Rcpp::NumericVector theta, Rcpp::NumericVector se) {
  const magmaan::estimate::Estimates est = est_from_theta(theta);
  const Eigen::VectorXd se_v = Rcpp::as<Eigen::VectorXd>(se);
  const magmaan::inference::ZTestResult zt = magmaan::inference::z_test(est, se_v);
  return Rcpp::List::create(Rcpp::_["z"] = Rcpp::wrap(zt.z),
                            Rcpp::_["pvalue"] = Rcpp::wrap(zt.p_value));
}

// infer_chi2_pvalue() — mirrors chi2_pvalue(chi2, df), vectorized over `chi2`
// (recycles `df` if length 1); NA/NaN in -> NA out. Also the LR-test primitive
// (subtract test statistics / dfs of two nested fits, call this).
//
// [[Rcpp::export]]
Rcpp::NumericVector infer_chi2_pvalue(Rcpp::NumericVector chi2, Rcpp::IntegerVector df) {
  if (df.size() == 0) Rcpp::stop("magmaan: df must have length >= 1");
  const R_xlen_t n = chi2.size();
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const double c = chi2[i];
    const int d = df[df.size() == 1 ? 0 : (i % df.size())];
    out[i] = (ISNA(c) || d == NA_INTEGER) ? NA_REAL : magmaan::inference::chi2_pvalue(c, d);
  }
  return out;
}

// infer_wald_test() — mirrors wald_test(R, q, est, vcov). `R` is k x npar
// (columns indexed by partable$free); `q` defaults to zeros; `vcov` is the
// parameter covariance matrix (from infer_vcov()).
//
// [[Rcpp::export]]
Rcpp::List infer_wald_test(Rcpp::List fit, Rcpp::NumericMatrix R,
                           Rcpp::NumericMatrix vcov,
                           Rcpp::Nullable<Rcpp::NumericVector> q = R_NilValue) {
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  Eigen::MatrixXd Rmat = Rcpp::as<Eigen::MatrixXd>(R);
  {
    Rcpp::List pt_df(fit["partable"]);
    Rcpp::IntegerVector freev(pt_df["free"]);
    int npar = 0;
    for (R_xlen_t i = 0; i < freev.size(); ++i) if (freev[i] > npar) npar = freev[i];
    if (static_cast<int>(Rmat.cols()) != npar)
      Rcpp::stop("magmaan: R must have %d columns (one per free parameter)", npar);
  }
  Eigen::VectorXd qv = q.isNotNull() ? Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(q.get()))
                                     : Eigen::VectorXd::Zero(Rmat.rows());
  if (qv.size() != Rmat.rows())
    Rcpp::stop("magmaan: q must have length %d (= nrow(R))", static_cast<int>(Rmat.rows()));
  Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto w_or = magmaan::inference::wald_test(Rmat, qv, est, vcov_m);
  if (!w_or.has_value()) stop_post(w_or.error());
  return Rcpp::List::create(Rcpp::_["chi2"] = w_or->chi2, Rcpp::_["df"] = w_or->df,
                            Rcpp::_["pvalue"] = magmaan::inference::chi2_pvalue(w_or->chi2, w_or->df));
}

// infer_wald_test_theta() — primitive form of infer_wald_test(): R/q, theta,
// and parameter vcov. R must have one column per element of theta.
//
// [[Rcpp::export]]
Rcpp::List infer_wald_test_theta(Rcpp::NumericVector theta,
                                 Rcpp::NumericMatrix R,
                                 Rcpp::NumericMatrix vcov,
                                 Rcpp::Nullable<Rcpp::NumericVector> q = R_NilValue) {
  const magmaan::estimate::Estimates est = est_from_theta(theta);
  Eigen::MatrixXd Rmat = Rcpp::as<Eigen::MatrixXd>(R);
  if (Rmat.cols() != est.theta.size())
    Rcpp::stop("magmaan: R must have %d columns (one per free parameter)",
               static_cast<int>(est.theta.size()));
  Eigen::VectorXd qv = q.isNotNull() ? Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(q.get()))
                                     : Eigen::VectorXd::Zero(Rmat.rows());
  if (qv.size() != Rmat.rows())
    Rcpp::stop("magmaan: q must have length %d (= nrow(R))", static_cast<int>(Rmat.rows()));
  Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  auto w_or = magmaan::inference::wald_test(Rmat, qv, est, vcov_m);
  if (!w_or.has_value()) stop_post(w_or.error());
  return Rcpp::List::create(Rcpp::_["chi2"] = w_or->chi2, Rcpp::_["df"] = w_or->df,
                            Rcpp::_["pvalue"] = magmaan::inference::chi2_pvalue(w_or->chi2, w_or->df));
}

// infer_browne_residual_nt() — mirrors browne_residual_nt(pt, rep, samp, est).
// Returns just the statistic (the model df is infer_df_stat(); the p-value
// is infer_chi2_pvalue(statistic, df)).
//
// [[Rcpp::export]]
Rcpp::List infer_browne_residual_nt(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto s_or = magmaan::inference::browne_residual_nt(ctx.pt, ctx.rep, ctx.samp, est);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = *s_or);
}

// infer_rls_chi2() — lavaan's `test = "browne.residual.nt.model"`, correct for
// mean structures. Needs the model Jacobian, so it goes through the structure
// rather than pre-built moments.
//
// `implied` is accepted and ignored. Every caller passes `model_implied(fit)`,
// i.e. exactly the moments this recomputes from the fit's own θ̂, so dropping
// it from the signature would churn call sites for no behavioural gain.
//
// [[Rcpp::export]]
Rcpp::List infer_rls_chi2(Rcpp::List fit, Rcpp::List implied) {
  (void)implied;
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto s_or =
      magmaan::inference::rls_chi2(ctx.pt, ctx.rep, ctx.samp, est.theta);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = *s_or);
}

// infer_nt_moment_quadratic() — the unprojected normal-theory moment quadratic
// N·r'Γ(Σ̂)⁻¹r, with the mean block included whenever the fit supplies implied
// means. This is NOT a lavaan test statistic: it omits the model-space
// projection that makes `infer_rls_chi2()` above χ²(df). Retained because the
// projection is sometimes supplied elsewhere (e.g. under an eigenvalue-spectrum
// correction).
//
// [[Rcpp::export]]
Rcpp::List infer_nt_moment_quadratic(Rcpp::List fit, Rcpp::List implied) {
  Ctx ctx = ctx_from_fit(fit);
  lvm::ImpliedMoments im;
  Rcpp::List sig(implied["sigma"]);
  for (R_xlen_t b = 0; b < sig.size(); ++b)
    im.sigma.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(sig[b])));
  if (implied.containsElementNamed("mu") && !Rf_isNull(implied["mu"])) {
    Rcpp::List m(implied["mu"]);
    for (R_xlen_t b = 0; b < m.size(); ++b)
      im.mu.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(m[b])));
  }
  auto s_or = magmaan::inference::frontier::nt_moment_quadratic(ctx.samp, im);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(
      Rcpp::_["statistic"] = s_or->statistic,
      Rcpp::_["mean"] = s_or->mean,
      Rcpp::_["covariance"] = s_or->covariance);
}

// infer_nt_moment_quadratic_sample() — primitive form of
// infer_nt_moment_quadratic(): sample moments plus model-implied moments,
// without requiring a fit list. Jacobian-free, so like its fit-based sibling it
// is the unprojected quadratic and not a lavaan test statistic. There is no
// moments-only form of the RLS statistic, which needs the model Jacobian.
//
// [[Rcpp::export]]
Rcpp::List infer_nt_moment_quadratic_sample(Rcpp::List sample_stats,
                                            Rcpp::List implied) {
  if (!sample_stats.containsElementNamed("S") || !sample_stats.containsElementNamed("nobs"))
    Rcpp::stop("magmaan: `sample_stats` must be a list with $S and $nobs");
  magmaan::data::SampleStats samp;
  Rcpp::List Sl = TYPEOF(sample_stats["S"]) == VECSXP
      ? Rcpp::List(sample_stats["S"])
      : Rcpp::List::create(Rcpp::NumericMatrix(sample_stats["S"]));
  Rcpp::IntegerVector nv = Rcpp::as<Rcpp::IntegerVector>(sample_stats["nobs"]);
  if (Sl.size() != nv.size())
    Rcpp::stop("magmaan: sample_stats$S and sample_stats$nobs must have the same length");
  for (R_xlen_t b = 0; b < Sl.size(); ++b) {
    samp.S.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Sl[b])));
    samp.n_obs.push_back(static_cast<std::int64_t>(nv[b]));
  }
  if (sample_stats.containsElementNamed("mean") && !Rf_isNull(sample_stats["mean"])) {
    Rcpp::List Ml = TYPEOF(sample_stats["mean"]) == VECSXP
        ? Rcpp::List(sample_stats["mean"])
        : Rcpp::List::create(Rcpp::NumericVector(sample_stats["mean"]));
    if (Ml.size() != Sl.size())
      Rcpp::stop("magmaan: sample_stats$mean and sample_stats$S must have the same length");
    for (R_xlen_t b = 0; b < Ml.size(); ++b)
      samp.mean.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(Ml[b])));
  }

  lvm::ImpliedMoments im;
  Rcpp::List sig(implied["sigma"]);
  for (R_xlen_t b = 0; b < sig.size(); ++b)
    im.sigma.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(sig[b])));
  if (implied.containsElementNamed("mu") && !Rf_isNull(implied["mu"])) {
    Rcpp::List m(implied["mu"]);
    for (R_xlen_t b = 0; b < m.size(); ++b)
      im.mu.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(m[b])));
  }
  auto s_or = magmaan::inference::frontier::nt_moment_quadratic(samp, im);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(
      Rcpp::_["statistic"] = s_or->statistic,
      Rcpp::_["mean"] = s_or->mean,
      Rcpp::_["covariance"] = s_or->covariance);
}
