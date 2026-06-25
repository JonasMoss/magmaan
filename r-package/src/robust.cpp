// Rcpp glue for magmaan's robust normal-theory inference: the U-factor at θ̂, the
// three Γ-meat reductions (M_Γ = BᵀΓB), the eigenvalues of UΓ, the
// Satorra-Bentler / mean-var-adjusted / scaled-shifted χ² built from them, the
// raw-data / standalone-Γ helpers, and the sandwich SEs. Composition-first: one
// thin wrapper per C++ entry point, including build_u_factor() so the U-factor
// is a first-class (transparent base-R list) object you build once and reduce /
// eigensolve several ways. Shared plumbing lives in internal.hpp.

#include "internal.hpp"

#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/frontier/fmg.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/ordinal.hpp"

#include <algorithm>

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;

namespace {

// ---- InferenceSpec enums <-> strings ---------------------------------------
// info_from_string / moments_from_string / cov_from_string / spec_from now live
// in internal.hpp (namespace magmaanr), shared with fit.cpp's robust score glue.

const char* moments_to_string(magmaan::robust::WeightMoments m) {
  switch (m) {
    case magmaan::robust::WeightMoments::Structured:   return "structured";
    case magmaan::robust::WeightMoments::Unstructured: return "unstructured";
    case magmaan::robust::WeightMoments::Pairwise:     return "pairwise";
  }
  return "structured";
}
const char* ufactor_kind_to_string(magmaan::robust::UFactor::Kind k) {
  return k == magmaan::robust::UFactor::Kind::ProjectionExpected ? "ProjectionExpected" : "ObservedHessian";
}

Rcpp::List chisq_moments_to_list(
    const magmaan::robust::WeightedChiSquareMoments& m) {
  return Rcpp::List::create(Rcpp::_["df"] = m.df,
                            Rcpp::_["trace"] = m.trace,
                            Rcpp::_["trace_sq"] = m.trace_sq);
}

Rcpp::List test_moments_pair_to_list(
    const magmaan::robust::RobustTestMomentsBothBreads& r) {
  return Rcpp::List::create(
      Rcpp::_["expected"] = chisq_moments_to_list(r.expected),
      Rcpp::_["observed"] = chisq_moments_to_list(r.observed));
}

magmaan::estimate::OrdinalWeightKind ordinal_weight_from_string(const std::string& s) {
  if (s == "DWLS" || s == "dwls") return magmaan::estimate::OrdinalWeightKind::DWLS;
  if (s == "WLS" || s == "wls") return magmaan::estimate::OrdinalWeightKind::WLS;
  if (s == "ULS" || s == "uls") return magmaan::estimate::OrdinalWeightKind::ULS;
  Rcpp::stop("magmaan: `weight` must be 'DWLS', 'WLS' or 'ULS' (got '%s')", s);
}

magmaan::data::OrdinalStats ordinal_stats_from_arg(Rcpp::List x) {
  const char* what = "ordinal_stats";
  for (const char* nm : {"R", "thresholds", "threshold_ov", "threshold_level",
                         "NACOV", "W_dwls", "W_wls", "nobs", "n_levels"}) {
    if (!x.containsElementNamed(nm)) Rcpp::stop("magmaan: %s is missing $%s", what, nm);
  }
  Rcpp::List Rl(x["R"]), thl(x["thresholds"]), ovl(x["threshold_ov"]),
      levl(x["threshold_level"]), NAl(x["NACOV"]),
      Wdl(x["W_dwls"]), Wfl(x["W_wls"]), nlevl(x["n_levels"]);
  Rcpp::IntegerVector nobs(x["nobs"]);
  const bool has_mi = x.containsElementNamed("moment_influence");
  Rcpp::List mil = has_mi ? Rcpp::List(x["moment_influence"]) : Rcpp::List();
  const bool has_id = x.containsElementNamed("int_data");
  Rcpp::List idl = has_id ? Rcpp::List(x["int_data"]) : Rcpp::List();
  const bool has_mb = x.containsElementNamed("moment_bread");
  Rcpp::List mbl = has_mb ? Rcpp::List(x["moment_bread"]) : Rcpp::List();
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
    if (has_mb)
      out.moment_bread.push_back(
          Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(mbl[b])));
    out.n_obs.push_back(static_cast<std::int64_t>(nobs[b]));
    out.n_levels.push_back(Rcpp::as<std::vector<std::int32_t>>(Rcpp::IntegerVector(nlevl[b])));
  }
  return out;
}

[[maybe_unused]] Rcpp::List
satorra_bentler_to_list(const magmaan::robust::SatorraBentlerResult& r) {
  return Rcpp::List::create(Rcpp::_["chi2_scaled"] = r.chi2_scaled,
                            Rcpp::_["scale_c"] = r.scale_c,
                            Rcpp::_["df"] = r.df);
}

[[maybe_unused]] Rcpp::List
mean_var_to_list(const magmaan::robust::MeanVarAdjustedResult& r) {
  return Rcpp::List::create(Rcpp::_["chi2_adj"] = r.chi2_adj,
                            Rcpp::_["df_adj"] = r.df_adj);
}

[[maybe_unused]] Rcpp::List
scaled_shifted_to_list(const magmaan::robust::ScaledShiftedResult& r) {
  return Rcpp::List::create(Rcpp::_["chi2_adj"] = r.chi2_adj,
                            Rcpp::_["df"] = r.df,
                            Rcpp::_["scale_a"] = r.scale_a,
                            Rcpp::_["shift_b"] = r.shift_b);
}

Rcpp::CharacterVector warnings_to_r(const std::vector<std::string>& warnings) {
  Rcpp::CharacterVector out(static_cast<R_xlen_t>(warnings.size()));
  for (std::size_t k = 0; k < warnings.size(); ++k) {
    out[static_cast<R_xlen_t>(k)] = warnings[k];
  }
  return out;
}

Rcpp::List profile_rmsea_to_list(
    const magmaan::estimate::WeightedProfileRMSEAResult& r) {
  return Rcpp::List::create(
      Rcpp::_["profile_hessian"] = Rcpp::wrap(r.profile_hessian),
      Rcpp::_["gamma"] = Rcpp::wrap(r.gamma),
      Rcpp::_["eigvals"] = Rcpp::wrap(r.eigvals),
      Rcpp::_["profile_pencil_eigvals"] = Rcpp::wrap(r.profile_pencil_eigvals),
      Rcpp::_["fmin"] = r.fmin,
      Rcpp::_["chisq_standard"] = r.chisq_standard,
      Rcpp::_["bias_trace"] = r.bias_trace,
      Rcpp::_["bias_trace_sq"] = r.bias_trace_sq,
      Rcpp::_["trace_signed"] = r.trace_signed,
      Rcpp::_["negative_trace_abs"] = r.negative_trace_abs,
      Rcpp::_["profile_pencil_max_abs_gap"] = r.profile_pencil_max_abs_gap,
      Rcpp::_["rmsea"] = r.rmsea,
      Rcpp::_["rmsea_positive_trace"] = r.rmsea_positive_trace,
      Rcpp::_["rmsea_df"] = r.rmsea_df,
      Rcpp::_["df"] = r.df,
      Rcpp::_["spectrum_size"] = r.spectrum_size,
      Rcpp::_["negative_spectrum_size"] = r.negative_spectrum_size,
      Rcpp::_["spectrum_rank"] = r.spectrum_rank,
      Rcpp::_["profile_pencil_residual_dim"] = r.profile_pencil_residual_dim,
      Rcpp::_["profile_pencil_positive_count"] =
          r.profile_pencil_positive_count,
      Rcpp::_["profile_pencil_negative_count"] =
          r.profile_pencil_negative_count,
      Rcpp::_["profile_pencil_rank"] = r.profile_pencil_rank,
      Rcpp::_["ntotal"] = static_cast<double>(r.ntotal),
      Rcpp::_["n_groups"] = static_cast<int>(r.n_groups),
      Rcpp::_["warnings"] = warnings_to_r(r.warnings));
}

Rcpp::List profile_lrt_to_list(
    const magmaan::estimate::WeightedProfileLRTResult& r) {
  return Rcpp::List::create(
      Rcpp::_["profile_hessian"] = Rcpp::wrap(r.profile_hessian),
      Rcpp::_["gamma"] = Rcpp::wrap(r.gamma),
      Rcpp::_["eigvals"] = Rcpp::wrap(r.eigvals),
      Rcpp::_["fmin_diff"] = r.fmin_diff,
      Rcpp::_["T_diff"] = r.T_diff,
      Rcpp::_["bias_trace"] = r.bias_trace,
      Rcpp::_["bias_trace_sq"] = r.bias_trace_sq,
      Rcpp::_["trace_signed"] = r.trace_signed,
      Rcpp::_["negative_trace_abs"] = r.negative_trace_abs,
      Rcpp::_["scale_c"] = r.scale_c,
      Rcpp::_["T_scaled"] = r.T_scaled,
      Rcpp::_["p_unscaled"] = r.p_unscaled,
      Rcpp::_["p_scaled"] = r.p_scaled,
      Rcpp::_["T_adjusted"] = r.T_adjusted,
      Rcpp::_["adjust_df"] = r.adjust_df,
      Rcpp::_["p_adjusted"] = r.p_adjusted,
      Rcpp::_["scaled_shifted"] = scaled_shifted_to_list(r.scaled_shifted),
      Rcpp::_["p_scaled_shifted"] = r.p_scaled_shifted,
      Rcpp::_["p_mixture"] = r.p_mixture,
      Rcpp::_["df_diff"] = r.df_diff,
      Rcpp::_["spectrum_size"] = r.spectrum_size,
      Rcpp::_["negative_spectrum_size"] = r.negative_spectrum_size,
      Rcpp::_["spectrum_rank"] = r.spectrum_rank,
      Rcpp::_["ntotal"] = static_cast<double>(r.ntotal),
      Rcpp::_["n_groups"] = static_cast<int>(r.n_groups),
      Rcpp::_["warnings"] = warnings_to_r(r.warnings));
}

Rcpp::List rmsea_inference_to_list(
    const magmaan::estimate::OrdinalRmseaInference& r) {
  return Rcpp::List::create(
      Rcpp::_["rmsea"] = r.point,
      Rcpp::_["rmsea.ci.lower"] = r.ci_lower,
      Rcpp::_["rmsea.ci.upper"] = r.ci_upper,
      Rcpp::_["rmsea.pvalue"] = r.exact_fit_pvalue,
      Rcpp::_["bias_trace"] = r.bias_trace,
      Rcpp::_["grad_var"] = r.grad_var,
      Rcpp::_["fmin"] = r.fmin,
      Rcpp::_["chisq_standard"] = r.stat,
      Rcpp::_["df"] = r.df,
      Rcpp::_["spectrum_size"] = r.spectrum_size,
      Rcpp::_["fixed_weight"] = r.fixed_weight,
      Rcpp::_["eigvals"] = Rcpp::wrap(r.eigvals),
      Rcpp::_["warnings"] = warnings_to_r(r.warnings));
}

Rcpp::List crmr_inference_to_list(
    const magmaan::estimate::OrdinalCrmrInference& r) {
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["index"] = r.srmr_denominator ? "srmr" : "crmr",
      Rcpp::_["estimate"] = r.point,
      Rcpp::_["estimate.bias.corrected"] = r.point_bias_corrected,
      Rcpp::_["ci.lower"] = r.ci_lower,
      Rcpp::_["ci.upper"] = r.ci_upper,
      Rcpp::_["pvalue"] = r.exact_fit_pvalue,
      Rcpp::_["bias_trace"] = r.bias_trace,
      Rcpp::_["grad_var"] = r.grad_var,
      Rcpp::_["stat"] = r.stat,
      Rcpp::_["k"] = r.k,
      Rcpp::_["spectrum_size"] = r.spectrum_size,
      Rcpp::_["fixed_weight"] = r.fixed_weight,
      Rcpp::_["srmr_denominator"] = r.srmr_denominator,
      Rcpp::_["eigvals"] = Rcpp::wrap(r.eigvals),
      Rcpp::_["warnings"] = warnings_to_r(r.warnings));
  if (r.srmr_denominator) {
    out["srmr"] = r.point;
    out["srmr.ci.lower"] = r.ci_lower;
    out["srmr.ci.upper"] = r.ci_upper;
    out["srmr.pvalue"] = r.exact_fit_pvalue;
  } else {
    out["crmr"] = r.point;
    out["crmr.ci.lower"] = r.ci_lower;
    out["crmr.ci.upper"] = r.ci_upper;
    out["crmr.pvalue"] = r.exact_fit_pvalue;
  }
  return out;
}

Rcpp::List incremental_inference_to_list(
    const magmaan::estimate::OrdinalIncrementalFitInference& r) {
  return Rcpp::List::create(
      Rcpp::_["cfi"] = r.cfi,
      Rcpp::_["cfi.ci.lower"] = r.cfi_ci_lower,
      Rcpp::_["cfi.ci.upper"] = r.cfi_ci_upper,
      Rcpp::_["tli"] = r.tli,
      Rcpp::_["tli.ci.lower"] = r.tli_ci_lower,
      Rcpp::_["tli.ci.upper"] = r.tli_ci_upper,
      Rcpp::_["delta_user"] = r.delta_user,
      Rcpp::_["delta_baseline"] = r.delta_baseline,
      Rcpp::_["stat_user"] = r.stat_user,
      Rcpp::_["stat_baseline"] = r.stat_baseline,
      Rcpp::_["gendf_user"] = r.gendf_user,
      Rcpp::_["gendf_baseline"] = r.gendf_baseline,
      Rcpp::_["var_user"] = r.var_user,
      Rcpp::_["var_baseline"] = r.var_baseline,
      Rcpp::_["cov_user_baseline"] = r.cov_user_baseline,
      Rcpp::_["var_cfi"] = r.var_cfi,
      Rcpp::_["var_tli"] = r.var_tli,
      Rcpp::_["df_user"] = r.df_user,
      Rcpp::_["df_baseline"] = r.df_baseline,
      Rcpp::_["fixed_weight"] = r.fixed_weight,
      Rcpp::_["warnings"] = warnings_to_r(r.warnings));
}

Rcpp::List misspec_fit_measures_to_list(
    const magmaan::estimate::OrdinalMisspecFitMeasures& r) {
  return Rcpp::List::create(
      Rcpp::_["rmsea"] = r.rmsea,
      Rcpp::_["rmsea.ci.lower"] = r.rmsea_ci_lower,
      Rcpp::_["rmsea.ci.upper"] = r.rmsea_ci_upper,
      Rcpp::_["rmsea.pvalue"] = r.rmsea_pvalue,
      Rcpp::_["crmr"] = r.crmr,
      Rcpp::_["crmr.ci.lower"] = r.crmr_ci_lower,
      Rcpp::_["crmr.ci.upper"] = r.crmr_ci_upper,
      Rcpp::_["crmr.pvalue"] = r.crmr_pvalue,
      Rcpp::_["srmr"] = r.srmr,
      Rcpp::_["srmr.ci.lower"] = r.srmr_ci_lower,
      Rcpp::_["srmr.ci.upper"] = r.srmr_ci_upper,
      Rcpp::_["cfi"] = r.cfi,
      Rcpp::_["cfi.ci.lower"] = r.cfi_ci_lower,
      Rcpp::_["cfi.ci.upper"] = r.cfi_ci_upper,
      Rcpp::_["tli"] = r.tli,
      Rcpp::_["tli.ci.lower"] = r.tli_ci_lower,
      Rcpp::_["tli.ci.upper"] = r.tli_ci_upper,
      Rcpp::_["chisq"] = r.stat_user,
      Rcpp::_["df"] = r.df_user,
      Rcpp::_["baseline.chisq"] = r.stat_baseline,
      Rcpp::_["baseline.df"] = r.df_baseline,
      Rcpp::_["conf.level"] = r.conf_level,
      Rcpp::_["estimated.weight"] = !r.fixed_weight,
      Rcpp::_["warnings"] = warnings_to_r(r.warnings));
}

std::string ordinal_fit_weight(Rcpp::List fit, const char* caller) {
  if (fit.containsElementNamed("ordinal_computational_weight")) {
    return Rcpp::as<std::string>(fit["ordinal_computational_weight"]);
  }
  if (fit.containsElementNamed("estimator")) {
    return Rcpp::as<std::string>(fit["estimator"]);
  }
  Rcpp::stop("magmaan: %s requires a DWLS ordinal fit carrying $estimator",
             caller);
}

void require_dwls_fit(Rcpp::List fit, const char* caller) {
  const std::string weight = ordinal_fit_weight(fit, caller);
  if (!(weight == "DWLS" || weight == "dwls")) {
    Rcpp::stop("magmaan: %s is defined for DWLS fits only (got '%s')",
               caller, weight);
  }
}

// ---- UFactor <-> R list ----------------------------------------------------

Rcpp::List ufactor_to_list(const magmaan::robust::UFactor& uf) {
  Rcpp::List blocks(static_cast<R_xlen_t>(uf.blocks.size()));
  for (std::size_t b = 0; b < uf.blocks.size(); ++b) {
    const magmaan::robust::UFactor::Block& blk = uf.blocks[b];
    blocks[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["p"]          = static_cast<int>(blk.p),
        Rcpp::_["pstar"]      = static_cast<int>(blk.pstar),
        Rcpp::_["row_offset"] = static_cast<int>(blk.row_offset),
        Rcpp::_["mu_off"]     = static_cast<int>(blk.mu_off),
        Rcpp::_["n_obs"]      = static_cast<int>(blk.n_obs),
        Rcpp::_["Sigma_hat"]  = Rcpp::wrap(blk.Sigma_hat),
        Rcpp::_["S"]          = Rcpp::wrap(blk.S));
  }
  const bool proj = (uf.kind == magmaan::robust::UFactor::Kind::ProjectionExpected);
  return Rcpp::List::create(
      Rcpp::_["kind"]       = std::string(ufactor_kind_to_string(uf.kind)),
      Rcpp::_["B"]          = proj ? static_cast<SEXP>(Rcpp::wrap(uf.B)) : R_NilValue,
      Rcpp::_["A"]          = proj ? R_NilValue : static_cast<SEXP>(Rcpp::wrap(uf.A)),
      Rcpp::_["H_obs_inv"]  = proj ? R_NilValue : static_cast<SEXP>(Rcpp::wrap(uf.H_obs_inv)),
      Rcpp::_["df"]         = static_cast<int>(uf.df),
      Rcpp::_["pstar"]      = static_cast<int>(uf.pstar),
      Rcpp::_["total_rows"] = static_cast<int>(uf.total_rows),
      Rcpp::_["has_means"]  = uf.has_means,
      Rcpp::_["moments"]    = std::string(moments_to_string(uf.moments)),
      Rcpp::_["blocks"]     = blocks);
}

magmaan::robust::UFactor ufactor_from_list(Rcpp::List ul) {
  auto need = [&](const char* nm) {
    if (!ul.containsElementNamed(nm))
      Rcpp::stop("magmaan: not a u-factor list (missing '%s') — pass the result of infer_build_u_factor()", nm);
  };
  need("kind"); need("df"); need("pstar"); need("moments"); need("blocks");
  magmaan::robust::UFactor uf;
  const std::string kind = Rcpp::as<std::string>(ul["kind"]);
  if (kind == "ProjectionExpected") {
    uf.kind = magmaan::robust::UFactor::Kind::ProjectionExpected;
    need("B");
    uf.B = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(ul["B"]));
  } else if (kind == "ObservedHessian") {
    uf.kind = magmaan::robust::UFactor::Kind::ObservedHessian;
    need("A"); need("H_obs_inv");
    uf.A = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(ul["A"]));
    uf.H_obs_inv = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(ul["H_obs_inv"]));
  } else {
    Rcpp::stop("magmaan: u-factor `kind` must be 'ProjectionExpected' or 'ObservedHessian' (got '%s')", kind);
  }
  uf.df    = static_cast<Eigen::Index>(Rcpp::as<int>(ul["df"]));
  uf.pstar = static_cast<Eigen::Index>(Rcpp::as<int>(ul["pstar"]));
  uf.moments = moments_from_string(Rcpp::as<std::string>(ul["moments"]));
  uf.has_means = ul.containsElementNamed("has_means")
                     ? Rcpp::as<bool>(ul["has_means"]) : false;
  uf.total_rows = ul.containsElementNamed("total_rows")
                      ? static_cast<Eigen::Index>(Rcpp::as<int>(ul["total_rows"]))
                      : uf.pstar;   // cov-only legacy lists pre-date this field

  Rcpp::List blocks(ul["blocks"]);
  uf.blocks.resize(static_cast<std::size_t>(blocks.size()));
  for (R_xlen_t b = 0; b < blocks.size(); ++b) {
    Rcpp::List bl(blocks[b]);
    magmaan::robust::UFactor::Block& blk = uf.blocks[static_cast<std::size_t>(b)];
    blk.p          = static_cast<Eigen::Index>(Rcpp::as<int>(bl["p"]));
    blk.pstar      = static_cast<Eigen::Index>(Rcpp::as<int>(bl["pstar"]));
    blk.row_offset = static_cast<Eigen::Index>(Rcpp::as<int>(bl["row_offset"]));
    blk.mu_off     = bl.containsElementNamed("mu_off")
                         ? static_cast<Eigen::Index>(Rcpp::as<int>(bl["mu_off"]))
                         : Eigen::Index{-1};
    blk.n_obs      = bl.containsElementNamed("n_obs")
                         ? static_cast<Eigen::Index>(Rcpp::as<int>(bl["n_obs"])) : 0;
    blk.Sigma_hat  = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(bl["Sigma_hat"]));
    blk.S          = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(bl["S"]));
    // Re-derive the per-block weight Cholesky from the stored moments — and,
    // when the U-factor carries mean structure, the μ-block's M_b Cholesky too
    // (used by `apply_L_inv_block` on the μ-rows; mirrors build_u_factor()).
    const Eigen::MatrixXd& M =
        (uf.moments == magmaan::robust::WeightMoments::Structured) ? blk.Sigma_hat : blk.S;
    auto g_or = magmaan::data::gamma_nt(M);
    if (!g_or.has_value()) stop_post(g_or.error());
    blk.llt_gamma_nt.compute(*g_or);
    if (blk.llt_gamma_nt.info() != Eigen::Success)
      Rcpp::stop("magmaan: u-factor block %d weight matrix is not positive-definite",
                 static_cast<int>(b));
    if (uf.has_means) {
      blk.llt_M.compute(M);
      if (blk.llt_M.info() != Eigen::Success)
        Rcpp::stop("magmaan: u-factor block %d M_b (μ-Cholesky) not positive-definite",
                   static_cast<int>(b));
    }
  }
  return uf;
}

// ---- raw data from an R argument -------------------------------------------

// Build a (possibly multi-block) RawData from an R raw-data argument: `X` is a
// matrix (single group) or a list of per-group matrices; each block's columns
// are reordered to that block's model variable order (by colnames if present,
// else assumed already in order). Reuses the internal.hpp column-permute
// helpers.
magmaan::data::RawData raw_from_arg(const lvm::MatrixRep& rep, SEXP X) {
  const std::size_t n_blocks = rep.dims.size();
  magmaan::data::RawData raw;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    Rcpp::NumericMatrix Xb = block_matrix(X, b, n_blocks, "X");
    raw.X.push_back(reorder_data_cols(Xb, perm_for_cols(Xb, rep.ov_names[b], "X")));
  }
  return raw;
}

}  // namespace

// =============================================================================
// The UΓ-eigenvalue core
// =============================================================================

// infer_build_u_factor() — mirrors build_u_factor(pt, rep, samp, est, {bread,
// moments}). Returns the u-factor as a transparent list (kind, B|A|H_obs_inv,
// df, pstar, moments, blocks). `cov` is not an argument — build_u_factor()
// ignores it; the meat is chosen by which infer_reduced_gamma_*() you call.
//
// [[Rcpp::export]]
Rcpp::List infer_build_u_factor(Rcpp::List fit, std::string bread = "expected",
                                std::string moments = "structured") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto uf_or = magmaan::robust::build_u_factor(ctx.pt, ctx.rep, ctx.samp, est, spec_from(bread, moments));
  if (!uf_or.has_value()) stop_post(uf_or.error());
  return ufactor_to_list(*uf_or);
}

// infer_build_u_factor_parts() — primitive form of infer_build_u_factor():
// partable + sample_stats + theta, without requiring a fit list.
//
// [[Rcpp::export]]
Rcpp::List infer_build_u_factor_parts(SEXP partable, Rcpp::List sample_stats,
                                      Rcpp::NumericVector theta,
                                      std::string bread = "expected",
                                      std::string moments = "structured") {
  Ctx ctx = ctx_from_partable_sample_stats(partable, sample_stats,
                                           "infer_build_u_factor_parts");
  const magmaan::estimate::Estimates est = est_from_theta(theta);
  auto uf_or = magmaan::robust::build_u_factor(ctx.pt, ctx.rep, ctx.samp, est,
                                                   spec_from(bread, moments));
  if (!uf_or.has_value()) stop_post(uf_or.error());
  return ufactor_to_list(*uf_or);
}

// infer_reduced_gamma_nt() — mirrors reduced_gamma_nt(uf). Operator-only NT meat
// reduction; its eigenvalues are ~ all-1 (the projector sanity check).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_reduced_gamma_nt(Rcpp::List uf) {
  magmaan::robust::UFactor u = ufactor_from_list(uf);
  auto m_or = magmaan::robust::reduced_gamma_nt(u);
  if (!m_or.has_value()) stop_post(m_or.error());
  return Rcpp::wrap(*m_or);
}

// infer_build_u_factor_pairwise() — sibling of infer_build_u_factor for the
// WeightMoments::Pairwise bread. `X` is the raw data (matrix or list of
// per-group matrices); `mask` is the optional logical mask. The fit's
// sample_stats (Ŝ_pw, mean, nobs) plus θ̂ come from `fit`. Returns the
// same transparent UFactor list as infer_build_u_factor().
//
// [[Rcpp::export]]
Rcpp::List infer_build_u_factor_pairwise(Rcpp::List fit, SEXP X,
                                         SEXP mask = R_NilValue,
                                         std::string bread = "expected") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = raw_from_data_args(X, mask);
  auto pw_or = magmaan::data::pairwise_sample_stats(raw);
  if (!pw_or.has_value()) stop_post(pw_or.error());
  magmaan::robust::InferenceSpec spec = spec_from(bread, "pairwise");
  auto uf_or = magmaan::robust::build_u_factor(ctx.pt, ctx.rep, ctx.samp, est,
                                                raw, *pw_or, spec);
  if (!uf_or.has_value()) stop_post(uf_or.error());
  return ufactor_to_list(*uf_or);
}

// infer_reduced_gamma_nt_pairwise() — mirrors reduced_gamma_nt_pairwise(uf,
// raw, pw). Operator-only NT^pw meat reduction; eigenvalues are all ~1 when
// paired with a Pairwise-bread UFactor (the bread-meat collapse), and
// carry the SB-style scaling when paired with a Structured / Unstructured
// bread.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_reduced_gamma_nt_pairwise(Rcpp::List uf, SEXP X,
                                                    SEXP mask = R_NilValue) {
  magmaan::robust::UFactor u = ufactor_from_list(uf);
  magmaan::data::RawData raw = raw_from_data_args(X, mask);
  auto pw_or = magmaan::data::pairwise_sample_stats(raw);
  if (!pw_or.has_value()) stop_post(pw_or.error());
  auto m_or = magmaan::robust::reduced_gamma_nt_pairwise(u, raw, *pw_or);
  if (!m_or.has_value()) stop_post(m_or.error());
  return Rcpp::wrap(*m_or);
}

// infer_reduced_gamma_sample() — mirrors reduced_gamma_sample(uf, Zc, denom).
// `Zc` is the N x p* matrix from infer_casewise_contributions(); `denom` is the
// per-block divisor — a length-ngroups vector (conventionally `fit$nobs`; pass
// `fit$nobs - 1` for the divisor-corrected variant), or a single number to use
// the same divisor for every block. The multi-group SB meat is Σ_b B_bᵀΓ̂_bB_b
// with Γ̂_b = (Zc_b)ᵀ(Zc_b)/n_b — so for >1 block supply the per-block vector
// (a single pooled N would under-weight by ~ngroups).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_reduced_gamma_sample(Rcpp::List uf, Rcpp::NumericMatrix Zc,
                                               Rcpp::NumericVector denom) {
  magmaan::robust::UFactor u = ufactor_from_list(uf);
  Eigen::MatrixXd Zceig = Rcpp::as<Eigen::MatrixXd>(Zc);
  Eigen::VectorXd d = Rcpp::as<Eigen::VectorXd>(denom);
  auto m_or = magmaan::robust::reduced_gamma_sample(u, Zceig, d);
  if (!m_or.has_value()) stop_post(m_or.error());
  return Rcpp::wrap(*m_or);
}

// infer_reduced_gamma_sample_from_gamma() — same robust-test reduction as
// infer_reduced_gamma_sample(), but the caller supplies the already-materialized
// empirical Γ̂ matrix in the stacked moment layout. For multi-block models each
// diagonal block must already carry the intended divisor/scaling.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_reduced_gamma_sample_from_gamma(
    Rcpp::List uf, Rcpp::NumericMatrix gamma_hat) {
  magmaan::robust::UFactor u = ufactor_from_list(uf);
  Eigen::Map<const Eigen::MatrixXd> G(gamma_hat.begin(), gamma_hat.nrow(),
                                      gamma_hat.ncol());
  auto m_or = magmaan::robust::reduced_gamma_sample_from_gamma(u, G);
  if (!m_or.has_value()) stop_post(m_or.error());
  return Rcpp::wrap(*m_or);
}

// infer_robust_test_moments_both_breads_zc() — fit-level fast path for robust
// test adjustments. Builds expected/observed U-factors in C++ and returns only
// df, tr(M), and tr(M^2) for each bread, avoiding UFactor list roundtrips and
// full reduced-M matrices at the R boundary.
//
// [[Rcpp::export]]
Rcpp::List infer_robust_test_moments_both_breads_zc(
    Rcpp::List fit, Rcpp::NumericMatrix Zc, Rcpp::NumericVector denom,
    std::string moments = "structured") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  Eigen::Map<const Eigen::MatrixXd> Zc_m(Zc.begin(), Zc.nrow(), Zc.ncol());
  const Eigen::VectorXd d = Rcpp::as<Eigen::VectorXd>(denom);
  auto r_or = magmaan::robust::robust_test_moments_both_breads(
      ctx.pt, ctx.rep, ctx.samp, est, Zc_m, d, moments_from_string(moments));
  if (!r_or.has_value()) stop_post(r_or.error());
  return test_moments_pair_to_list(*r_or);
}

// [[Rcpp::export]]
Rcpp::List infer_robust_test_moments_both_breads_gamma(
    Rcpp::List fit, Rcpp::NumericMatrix gamma_hat,
    std::string moments = "structured") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  Eigen::Map<const Eigen::MatrixXd> G(gamma_hat.begin(), gamma_hat.nrow(),
                                      gamma_hat.ncol());
  auto r_or = magmaan::robust::robust_test_moments_both_breads_from_gamma(
      ctx.pt, ctx.rep, ctx.samp, est, G, moments_from_string(moments));
  if (!r_or.has_value()) stop_post(r_or.error());
  return test_moments_pair_to_list(*r_or);
}

// infer_reduced_gamma_sample_materialized() — same target as
// infer_reduced_gamma_sample(), but forms Γ̂ explicitly before reducing.
// This is a reference/benchmark path, not the memory-frugal production route.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_reduced_gamma_sample_materialized(
    Rcpp::List uf, Rcpp::NumericMatrix Zc, Rcpp::NumericVector denom) {
  magmaan::robust::UFactor u = ufactor_from_list(uf);
  Eigen::MatrixXd Zceig = Rcpp::as<Eigen::MatrixXd>(Zc);
  Eigen::VectorXd d = Rcpp::as<Eigen::VectorXd>(denom);
  auto m_or = magmaan::robust::reduced_gamma_sample_materialized(u, Zceig, d);
  if (!m_or.has_value()) stop_post(m_or.error());
  return Rcpp::wrap(*m_or);
}

// infer_reduced_gamma_unbiased() — mirrors reduced_gamma_unbiased(uf, samp,
// M_sample, M_nt). Browne's distribution-free correction; single-block only.
// `n` is the total sample size (the sample cov it needs for the rank-1 term is
// already stored in `uf`); `M_sample`/`M_nt` are infer_reduced_gamma_sample/_nt()
// results for the same `uf`.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_reduced_gamma_unbiased(Rcpp::List uf, int n,
                                                 Rcpp::NumericMatrix M_sample,
                                                 Rcpp::NumericMatrix M_nt) {
  magmaan::robust::UFactor u = ufactor_from_list(uf);
  if (u.blocks.empty()) Rcpp::stop("magmaan: u-factor has no blocks");
  magmaan::data::SampleStats samp;            // single-block; multi-block reduced_gamma_unbiased() errors
  samp.S = {u.blocks[0].S};
  samp.n_obs = {static_cast<std::int64_t>(n)};
  Eigen::MatrixXd Ms = Rcpp::as<Eigen::MatrixXd>(M_sample);
  Eigen::MatrixXd Mn = Rcpp::as<Eigen::MatrixXd>(M_nt);
  auto m_or = magmaan::robust::reduced_gamma_unbiased(u, samp, Ms, Mn);
  if (!m_or.has_value()) stop_post(m_or.error());
  return Rcpp::wrap(*m_or);
}

// infer_ugamma_eigenvalues() — mirrors ugamma_eigenvalues(M). Symmetrises `M`,
// returns the ascending real eigenvalues. The research deliverable.
//
// [[Rcpp::export]]
Rcpp::NumericVector infer_ugamma_eigenvalues(Rcpp::NumericMatrix M) {
  Eigen::MatrixXd Meig = Rcpp::as<Eigen::MatrixXd>(M);
  auto ev_or = magmaan::robust::ugamma_eigenvalues(Meig);
  if (!ev_or.has_value()) stop_post(ev_or.error());
  return Rcpp::wrap(*ev_or);
}

// infer_fmg_ugamma_spectra() — fused single-model FMG spectra path. Mirrors the
// R composition in fmg_pvalues(), but keeps UFactor construction, Zc reduction,
// optional Browne-unbiased correction, and eigensolves inside C++ so the hot
// path does not roundtrip the transparent UFactor list through R.
//
// [[Rcpp::export]]
Rcpp::List infer_fmg_ugamma_spectra(Rcpp::List fit, SEXP X,
                                    bool need_unbiased = false) {
  Ctx ctx = ctx_from_fit(fit);

  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = raw_from_arg(ctx.rep, X);
  auto ss_or = magmaan::data::sample_stats_from_raw(raw);
  if (!ss_or.has_value()) stop_post(ss_or.error());
  if (ss_or->n_obs.size() != ctx.rep.dims.size()) {
    Rcpp::stop("magmaan: infer_fmg_ugamma_spectra() raw-data block count does not match fit");
  }
  Eigen::VectorXd denom(static_cast<Eigen::Index>(ss_or->n_obs.size()));
  for (std::size_t b = 0; b < ss_or->n_obs.size(); ++b) {
    denom(static_cast<Eigen::Index>(b)) = static_cast<double>(ss_or->n_obs[b]);
  }

  auto uf_or = magmaan::robust::build_u_factor(
      ctx.pt, ctx.rep, ctx.samp, est,
      magmaan::robust::InferenceSpec{
          magmaan::robust::Information::Expected,
          magmaan::robust::WeightMoments::Structured,
          magmaan::robust::ScoreCovariance::ModelImplied});
  if (!uf_or.has_value()) stop_post(uf_or.error());

  const Eigen::Index n = raw.X[0].rows();
  Eigen::VectorXd ev_biased;
  Eigen::VectorXd ev_unbiased;
  Eigen::MatrixXd M;
  // The row-space shortcut (n×n Gram instead of the df×df reduced Γ̂) is only a
  // win for the biased spectrum. The unbiased correction needs the full df×df
  // M and a non-identity Bᵀ·Γ_NT(S)·B, so fall back to the full path whenever
  // the unbiased spectrum is requested. For multi-group data the empirical
  // meat uses per-block denominators n_g, so the unscaled row-space Gram is not
  // equivalent unless rows are block-scaled; keep that optimization single-block.
  const bool row_space =
      (ctx.rep.dims.size() == 1) && (n < uf_or->df) && !need_unbiased;
  if (row_space) {
    auto Y_or = magmaan::robust::casewise_projected_rows_tiled(
        *uf_or, raw, *ss_or);
    if (!Y_or.has_value()) stop_post(Y_or.error());
    Eigen::MatrixXd row_M = (*Y_or * Y_or->transpose()) / denom(0);
    auto ev_row_or = magmaan::robust::ugamma_eigenvalues(row_M);
    if (!ev_row_or.has_value()) stop_post(ev_row_or.error());
    ev_biased = Eigen::VectorXd::Zero(uf_or->df);
    ev_biased.tail(ev_row_or->size()) = *ev_row_or;
    std::sort(ev_biased.data(), ev_biased.data() + ev_biased.size());
  } else {
    auto M_or = magmaan::robust::reduced_gamma_sample_tiled(
        *uf_or, raw, *ss_or, denom);
    if (!M_or.has_value()) stop_post(M_or.error());
    M = std::move(*M_or);
    auto ev_or = magmaan::robust::ugamma_eigenvalues(M);
    if (!ev_or.has_value()) stop_post(ev_or.error());
    ev_biased = std::move(*ev_or);
  }

  Rcpp::List out = Rcpp::List::create(Rcpp::_["biased"] = Rcpp::wrap(ev_biased));
  if (need_unbiased) {
    auto Zc_or = magmaan::robust::casewise_contributions(
        raw, *ss_or, uf_or->has_means);
    if (!Zc_or.has_value()) stop_post(Zc_or.error());
    auto Mu_or = magmaan::robust::reduced_gamma_unbiased_casewise(
        *uf_or, *ss_or, *Zc_or, denom);
    if (!Mu_or.has_value()) stop_post(Mu_or.error());
    auto evu_or = magmaan::robust::ugamma_eigenvalues(*Mu_or);
    if (!evu_or.has_value()) stop_post(evu_or.error());
    ev_unbiased = std::move(*evu_or);
    out["unbiased"] = Rcpp::wrap(ev_unbiased);
  }
  return out;
}

// infer_satorra_bentler() — mirrors satorra_bentler(t_ml, df, eigvals).
// c = mean(eigvals); T_SB = t_ml / c.
//
// [[Rcpp::export]]
Rcpp::List infer_satorra_bentler(double t_ml, int df, Rcpp::NumericVector eigvals) {
  const Eigen::VectorXd ev = Rcpp::as<Eigen::VectorXd>(eigvals);
  const magmaan::robust::SatorraBentlerResult r = magmaan::robust::satorra_bentler(t_ml, df, ev);
  return Rcpp::List::create(Rcpp::_["chi2_scaled"] = r.chi2_scaled,
                            Rcpp::_["scale_c"] = r.scale_c, Rcpp::_["df"] = r.df);
}

// infer_mean_var_adjusted() — mirrors mean_var_adjusted(t_ml, df, eigvals).
// df_adj = (Σλ)²/Σλ²; T_adj = t_ml · Σλ/Σλ². (Satterthwaite / Yuan-Bentler T3.)
//
// [[Rcpp::export]]
Rcpp::List infer_mean_var_adjusted(double t_ml, int df, Rcpp::NumericVector eigvals) {
  const Eigen::VectorXd ev = Rcpp::as<Eigen::VectorXd>(eigvals);
  const magmaan::robust::MeanVarAdjustedResult r = magmaan::robust::mean_var_adjusted(t_ml, df, ev);
  return Rcpp::List::create(Rcpp::_["chi2_adj"] = r.chi2_adj, Rcpp::_["df_adj"] = r.df_adj);
}

// infer_scaled_shifted() — mirrors scaled_shifted(t_ml, df, eigvals).
// a = √(df/Σλ²); b = df − a·Σλ; T_adj = t_ml·a + b (df stays integer).
//
// [[Rcpp::export]]
Rcpp::List infer_scaled_shifted(double t_ml, int df, Rcpp::NumericVector eigvals) {
  const Eigen::VectorXd ev = Rcpp::as<Eigen::VectorXd>(eigvals);
  const magmaan::robust::ScaledShiftedResult r = magmaan::robust::scaled_shifted(t_ml, df, ev);
  return Rcpp::List::create(Rcpp::_["chi2_adj"] = r.chi2_adj, Rcpp::_["df"] = r.df,
                            Rcpp::_["scale_a"] = r.scale_a, Rcpp::_["shift_b"] = r.shift_b);
}

// =============================================================================
// FMG eigenvalue tests (Foldnes-Moss-Gronneberg): SB / SS / scaled-F / EBAd /
// penalized-all / pEBA / pOLS — the single-model goodness-of-fit p-values.
// =============================================================================

namespace {

magmaan::robust::frontier::FmgMethod fmg_method_from_string(const std::string& s) {
  using M = magmaan::robust::frontier::FmgMethod;
  if (s == "standard" || s == "ml" || s == "chisq")  return M::StandardChiSquare;
  if (s == "sb" || s == "satorra_bentler")           return M::SatorraBentler;
  if (s == "ss" || s == "scaled_shifted")            return M::ScaledShifted;
  if (s == "mv" || s == "mean_var_adjusted")         return M::MeanVarAdjusted;
  if (s == "scaled_f" || s == "f")                   return M::ScaledF;
  if (s == "all" || s == "ebad")                     return M::All;
  if (s == "penalized_all" || s == "pall")           return M::PenalizedAll;
  if (s == "eba")                                    return M::Eba;
  if (s == "peba")                                   return M::Peba;
  if (s == "pols")                                   return M::Pols;
  Rcpp::stop("magmaan: `method` must be one of 'standard','sb','ss','mv',"
             "'scaled_f','all','penalized_all','eba','peba','pols' (got '%s')",
             s.c_str());
}

const char* fmg_method_to_string(magmaan::robust::frontier::FmgMethod m) {
  using M = magmaan::robust::frontier::FmgMethod;
  switch (m) {
    case M::StandardChiSquare: return "standard";
    case M::SatorraBentler:    return "sb";
    case M::ScaledShifted:     return "ss";
    case M::MeanVarAdjusted:   return "mv";
    case M::ScaledF:           return "scaled_f";
    case M::All:               return "all";
    case M::PenalizedAll:      return "penalized_all";
    case M::Eba:               return "eba";
    case M::Peba:              return "peba";
    case M::Pols:              return "pols";
  }
  return "peba";
}

}  // namespace

// infer_fmg_test() — mirrors robust::frontier::fmg_test(chi2_source, df,
// ugamma_eigenvalues, {method, param, truncate_negative}). The single-model FMG
// goodness-of-fit p-value: take the source χ² (`chi2_source` = T_ML from
// infer_df_stat() or T_RLS from infer_browne_residual_nt()), its `df`, and the
// UΓ eigenvalues `eigvals` (from infer_ugamma_eigenvalues()); apply the chosen
// eigenvalue transform; return the Imhof tail p-value. `method` selects the
// transform; `param` is the pEBA block count j (method="peba", ceil-rounded) or
// the pOLS shrinkage γ (method="pols"), and is ignored by the others. The
// p-value branches (all/penalized_all/peba/pols) call imhof_upper() — magmaan's
// own CompQuadForm-equivalent — so nothing is recomputed in R.
//
// [[Rcpp::export]]
Rcpp::List infer_fmg_test(double chi2_source, int df, Rcpp::NumericVector eigvals,
                          std::string method = "peba", double param = 4.0,
                          bool truncate_negative = true) {
  const Eigen::VectorXd ev = Rcpp::as<Eigen::VectorXd>(eigvals);
  magmaan::robust::frontier::FmgOptions opt;
  opt.method = fmg_method_from_string(method);
  opt.param = param;
  opt.truncate_negative = truncate_negative;
  const magmaan::robust::frontier::FmgTestResult r =
      magmaan::robust::frontier::fmg_test(chi2_source, df, ev, opt);
  return Rcpp::List::create(
      Rcpp::_["p_value"]           = r.p_value,
      Rcpp::_["chi2_source"]       = r.chi2_source,
      Rcpp::_["df"]                = r.df,
      Rcpp::_["chi2_equiv"]        = r.chi2_equiv,
      Rcpp::_["method"]            = std::string(fmg_method_to_string(r.method)),
      Rcpp::_["param"]             = r.param,
      Rcpp::_["lambdas_raw"]       = Rcpp::wrap(r.lambdas_raw),
      Rcpp::_["lambdas"]           = Rcpp::wrap(r.lambdas),
      Rcpp::_["lambdas_reference"] = Rcpp::wrap(r.lambdas_reference),
      Rcpp::_["n_truncated"]       = r.n_truncated);
}

// =============================================================================
// Raw-data + standalone-Γ helpers
// =============================================================================

// infer_casewise_contributions() — mirrors casewise_contributions(raw,
// sample_stats_from_raw(raw)). Logically a function of the partable (it needs
// the observed-variable ordering from build_matrix_rep) plus the raw data — no
// fit required. `partable` a partable data.frame; `X` an n x p raw-data matrix
// (single group) or a list of per-group matrices (multi-group), reordered to the
// model's ov order by colnames if present. Returns the N x p* centred casewise
// vech contributions matrix (vech = lower-tri column-major; columns aligned with
// a u-factor's B rows; colMeans are 0 to machine precision; multi-group: rows
// stacked group-by-group, each row carrying only its group's vech slice).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_casewise_contributions(SEXP partable, SEXP X) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "infer_casewise_contributions");
  auto rep_or = lvm::build_matrix_rep(parsed.structure, &parsed.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  if (rep_or->ov_names.empty() || rep_or->ov_names[0].empty())
    Rcpp::stop("magmaan: model has no observed variables");
  magmaan::data::RawData raw = raw_from_arg(*rep_or, X);
  auto ss_or = magmaan::data::sample_stats_from_raw(raw);
  if (!ss_or.has_value()) stop_post(ss_or.error());
  auto zc_or = magmaan::robust::casewise_contributions(raw, *ss_or);
  if (!zc_or.has_value()) stop_post(zc_or.error());
  return Rcpp::wrap(*zc_or);
}

// data_sample_stats_from_raw() — mirrors sample_stats_from_raw(RawData{...}).
// `X` a raw-data matrix (single group) or a list of per-group matrices. Returns
// the {S, mean, nobs} bundle (the canonical fit_fit() `sample_stats` arg),
// uniformly per-group: list(S = list of <p x p, N-divisor> covariances, mean =
// list of <length-p> mean vectors, nobs = <int vector>). For one group these
// are length-1 lists / a length-1 vector.
//
// [[Rcpp::export]]
Rcpp::List data_sample_stats_from_raw(SEXP X) {
  magmaan::data::RawData raw;
  if (TYPEOF(X) == VECSXP) {
    Rcpp::List Xl(X);
    for (R_xlen_t b = 0; b < Xl.size(); ++b)
      raw.X.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Xl[b])));
  } else {
    raw.X.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(X)));
  }
  auto ss_or = magmaan::data::sample_stats_from_raw(raw);
  if (!ss_or.has_value()) stop_post(ss_or.error());
  const magmaan::data::SampleStats& ss = *ss_or;
  const std::size_t nb = ss.S.size();
  const bool has_means = !ss.mean.empty();
  Rcpp::List Sl(static_cast<R_xlen_t>(nb)), Ml(static_cast<R_xlen_t>(nb));
  Rcpp::IntegerVector nv(static_cast<R_xlen_t>(nb));
  for (std::size_t b = 0; b < nb; ++b) {
    Sl[static_cast<R_xlen_t>(b)] = Rcpp::wrap(ss.S[b]);
    if (has_means) Ml[static_cast<R_xlen_t>(b)] = Rcpp::wrap(ss.mean[b]);
    nv[static_cast<R_xlen_t>(b)] = static_cast<int>(ss.n_obs[b]);
  }
  return Rcpp::List::create(Rcpp::_["S"]    = Sl,
                            Rcpp::_["mean"] = has_means ? static_cast<SEXP>(Ml) : R_NilValue,
                            Rcpp::_["nobs"] = nv);
}

// data_pairwise_sample_stats() — mirrors data::pairwise_sample_stats(raw).
// Takes raw data (matrix or list of per-group matrices) and an optional
// missingness mask (1 = observed). Returns the per-block bundle of marginal
// column means, pairwise N-divisor covariance, pairwise availability π̂ =
// n_(j,ℓ)/n, overlap counts n_(j,ℓ), and block sample sizes.
//
// [[Rcpp::export]]
Rcpp::List data_pairwise_sample_stats(SEXP X, SEXP mask = R_NilValue) {
  magmaan::data::RawData raw = raw_from_data_args(X, mask);
  auto pw_or = magmaan::data::pairwise_sample_stats(raw);
  if (!pw_or.has_value()) stop_post(pw_or.error());
  const magmaan::data::PairwiseSampleStats& pw = *pw_or;

  const R_xlen_t nb = static_cast<R_xlen_t>(pw.S.size());
  Rcpp::List Sl(nb), Ml(nb), Pi(nb), Np(nb);
  Rcpp::IntegerVector nv(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    const std::size_t bi = static_cast<std::size_t>(b);
    Sl[b] = Rcpp::wrap(pw.S[bi]);
    Ml[b] = Rcpp::wrap(pw.mean[bi]);
    Pi[b] = Rcpp::wrap(pw.pi_hat[bi]);
    Np[b] = Rcpp::wrap(pw.n_pair[bi]);
    nv[b] = static_cast<int>(pw.n_obs[bi]);
  }
  return Rcpp::List::create(
      Rcpp::_["S"]      = Sl,
      Rcpp::_["mean"]   = Ml,
      Rcpp::_["pi_hat"] = Pi,
      Rcpp::_["n_pair"] = Np,
      Rcpp::_["nobs"]   = nv);
}

// data_gamma_nt_pairwise() — mirrors data::gamma_nt_pairwise(raw, pw).
// Returns one p*×p* matrix per block. Useful for inspecting the materialized
// Γ_NT^pw weight directly (e.g. to compose `fit_wls` by hand for the
// efficiency study) and for symmetry/PD checks.
//
// [[Rcpp::export]]
Rcpp::List data_gamma_nt_pairwise(SEXP X, SEXP mask = R_NilValue) {
  magmaan::data::RawData raw = raw_from_data_args(X, mask);
  auto pw_or = magmaan::data::pairwise_sample_stats(raw);
  if (!pw_or.has_value()) stop_post(pw_or.error());
  auto g_or = magmaan::data::gamma_nt_pairwise(raw, *pw_or);
  if (!g_or.has_value()) stop_post(g_or.error());
  const R_xlen_t nb = static_cast<R_xlen_t>(g_or->size());
  Rcpp::List out(nb);
  for (R_xlen_t b = 0; b < nb; ++b) {
    out[b] = Rcpp::wrap((*g_or)[static_cast<std::size_t>(b)]);
  }
  return out;
}

// infer_pairwise_casewise_contributions() — mirrors
// robust::pairwise_casewise_contributions(raw, pairwise_sample_stats(raw)).
// Returns the Van-Praag influence matrix Ψ̂ (N x Σ_b p_b*) ready to be passed
// into infer_reduced_gamma_sample(uf, Psi, nobs) for the same U-Gamma
// machinery as the complete-data case.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_pairwise_casewise_contributions(
    SEXP X, SEXP mask = R_NilValue, bool include_means = false) {
  magmaan::data::RawData raw = raw_from_data_args(X, mask);
  auto pw_or = magmaan::data::pairwise_sample_stats(raw);
  if (!pw_or.has_value()) stop_post(pw_or.error());
  auto psi_or = magmaan::robust::pairwise_casewise_contributions(
      raw, *pw_or, include_means);
  if (!psi_or.has_value()) stop_post(psi_or.error());
  return Rcpp::wrap(*psi_or);
}

// infer_empirical_gamma() — mirrors empirical_gamma(X). Returns the p* x p*
// fourth-moment ACOV of vech(S) (vech = lower-tri column-major).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_empirical_gamma(Rcpp::NumericMatrix X) {
  Eigen::MatrixXd Xeig = Rcpp::as<Eigen::MatrixXd>(X);
  auto g_or = magmaan::data::empirical_gamma(Xeig);
  if (!g_or.has_value()) stop_post(g_or.error());
  return Rcpp::wrap(*g_or);
}

// infer_gamma_nt() — mirrors gamma_nt(Sigma). Returns the p* x p* normal-theory
// ACOV Γ_NT = 2 D⁺(Σ⊗Σ)D⁺ᵀ (same vech order).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_gamma_nt(Rcpp::NumericMatrix Sigma) {
  Eigen::MatrixXd Seig = Rcpp::as<Eigen::MatrixXd>(Sigma);
  auto g_or = magmaan::data::gamma_nt(Seig);
  if (!g_or.has_value()) stop_post(g_or.error());
  return Rcpp::wrap(*g_or);
}

// infer_empirical_gamma_with_means() — mirrors empirical_gamma_with_means(X).
// Returns the (p + p*) × (p + p*) stacked NACOV for the moment vector
// [m̄ ; vech(S)] — Browne 1984 with meanstructure. The vech-vech sub-block
// equals infer_empirical_gamma(X) bit-for-bit; the cross-block carries the
// empirical third moments that vanish only under normality.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_empirical_gamma_with_means(Rcpp::NumericMatrix X) {
  Eigen::MatrixXd Xeig = Rcpp::as<Eigen::MatrixXd>(X);
  auto g_or = magmaan::data::empirical_gamma_with_means(Xeig);
  if (!g_or.has_value()) stop_post(g_or.error());
  return Rcpp::wrap(*g_or);
}

// =============================================================================
// Robust ordinal DWLS/WLS reporting
// =============================================================================

// infer_ordinal_robust() — post-fit categorical sandwich SEs and SB-family
// scaled tests for the delta all-ordinal LS path. `ordinal_stats` is the
// list returned by data_ordinal_stats_from_df()/data_ordinal_stats_from_raw_impl().
// Empty `weight` means reuse fit$ordinal_computational_weight when present,
// otherwise fit$estimator ("ULS", "DWLS", or "WLS").
//
// [[Rcpp::export]]
Rcpp::List infer_ordinal_robust(Rcpp::List fit, Rcpp::List ordinal_stats,
                                std::string weight = "",
                                std::string bread = "expected") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  if (weight.empty()) {
    if (fit.containsElementNamed("ordinal_computational_weight")) {
      weight = Rcpp::as<std::string>(fit["ordinal_computational_weight"]);
    } else {
      if (!fit.containsElementNamed("estimator")) {
        Rcpp::stop("magmaan: infer_ordinal_robust() needs `weight` when "
                   "fit$estimator is absent");
      }
      weight = Rcpp::as<std::string>(fit["estimator"]);
    }
  }
  const std::string parameterization_name = fit.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit["parameterization"])
      : "delta";
  auto r_or = magmaan::estimate::robust_ordinal(
      ctx.pt, ctx.rep, stats, est, ordinal_weight_from_string(weight),
      ordinal_parameterization_from_string(parameterization_name),
      info_from_string(bread));
  if (!r_or.has_value()) stop_post(r_or.error());
  const magmaan::estimate::OrdinalRobustResult& r = *r_or;
  return Rcpp::List::create(
      Rcpp::_["vcov"] = Rcpp::wrap(r.vcov),
      Rcpp::_["se"] = Rcpp::wrap(r.se),
      Rcpp::_["df"] = r.df,
      Rcpp::_["eigvals"] = Rcpp::wrap(r.eigvals),
      Rcpp::_["chisq_standard"] = r.chisq_standard,
      Rcpp::_["satorra_bentler"] = satorra_bentler_to_list(r.satorra_bentler),
      Rcpp::_["mean_var_adjusted"] = mean_var_to_list(r.mean_var_adjusted),
      Rcpp::_["scaled_shifted"] = scaled_shifted_to_list(r.scaled_shifted));
}

// infer_ordinal_robust_ij() — infinitesimal-jackknife ("regime = ij")
// misspecification-robust covariance for the delta all-ordinal LS path. Unlike
// infer_ordinal_robust(bread="observed"), this carries the influence of the
// estimated weight Ŵ = diag(NACOV)⁻¹ (the cov(Γ̂) term), so it needs the
// per-case influence functions `ordinal_stats$moment_influence`; DWLS also
// needs complete `ordinal_stats$int_data`. ULS/DWLS only.
//
// [[Rcpp::export]]
Rcpp::List infer_ordinal_robust_ij(Rcpp::List fit, Rcpp::List ordinal_stats,
                                   std::string weight = "") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  if (weight.empty()) {
    if (fit.containsElementNamed("ordinal_computational_weight")) {
      weight = Rcpp::as<std::string>(fit["ordinal_computational_weight"]);
    } else {
      if (!fit.containsElementNamed("estimator")) {
        Rcpp::stop("magmaan: infer_ordinal_robust_ij() needs `weight` when "
                   "fit$estimator is absent");
      }
      weight = Rcpp::as<std::string>(fit["estimator"]);
    }
  }
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::robust_ordinal_ij(
      ctx.pt, ctx.rep, stats, est, ordinal_weight_from_string(weight),
      ordinal_parameterization_from_string(parameterization_name));
  if (!r_or.has_value()) stop_post(r_or.error());
  const magmaan::estimate::OrdinalRobustResult& r = *r_or;
  return Rcpp::List::create(
      Rcpp::_["vcov"] = Rcpp::wrap(r.vcov),
      Rcpp::_["se"] = Rcpp::wrap(r.se),
      Rcpp::_["df"] = r.df,
      Rcpp::_["chisq_standard"] = r.chisq_standard);
}

// infer_ordinal_casewise_influence_ij_fit() — per-case one-step
// misspecification-robust ("complete-sandwich") parameter influences for an
// all-ordinal DWLS/WLS/ULS(MV) fit: the categorical dual of semfindr's
// est_change_raw_approx, the ordinal counterpart of
// infer_casewise_influence_ij_fit. Returns the N_total x n_free `influence`
// matrix (column-Gram = the ordinal estimated-weight IJ vcov) and its
// fixed-weight `naive` counterpart. Same `ordinal_stats$moment_influence` /
// `int_data` requirements as infer_ordinal_robust_ij(). Beyond lavaan/semfindr.
//
// [[Rcpp::export]]
Rcpp::List infer_ordinal_casewise_influence_ij_fit(
    Rcpp::List fit, Rcpp::List ordinal_stats, std::string weight = "") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  if (weight.empty()) {
    if (fit.containsElementNamed("ordinal_computational_weight")) {
      weight = Rcpp::as<std::string>(fit["ordinal_computational_weight"]);
    } else {
      if (!fit.containsElementNamed("estimator")) {
        Rcpp::stop("magmaan: infer_ordinal_casewise_influence_ij_fit() needs "
                   "`weight` when fit$estimator is absent");
      }
      weight = Rcpp::as<std::string>(fit["estimator"]);
    }
  }
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::ordinal_casewise_influence_ij(
      ctx.pt, ctx.rep, stats, est, ordinal_weight_from_string(weight),
      ordinal_parameterization_from_string(parameterization_name));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(
      Rcpp::Named("influence") = Rcpp::wrap(r_or->influence),
      Rcpp::Named("influence_naive") = Rcpp::wrap(r_or->influence_naive),
      Rcpp::Named("n_total") = static_cast<double>(r_or->n_total));
}

// [[Rcpp::export]]
Rcpp::List infer_mixed_ordinal_robust(Rcpp::List fit, Rcpp::List mixed_stats,
                                      std::string weight = "",
                                      std::string bread = "expected") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(mixed_stats);
  if (weight.empty()) {
    if (!fit.containsElementNamed("estimator"))
      Rcpp::stop("magmaan: infer_mixed_ordinal_robust() needs `weight` when fit$estimator is absent");
    weight = Rcpp::as<std::string>(fit["estimator"]);
  }
  const std::string parameterization_name = fit.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit["parameterization"])
      : "delta";
  auto r_or = magmaan::estimate::robust_mixed_ordinal(
      ctx.pt, ctx.rep, stats, est, ordinal_weight_from_string(weight),
      ordinal_parameterization_from_string(parameterization_name),
      info_from_string(bread));
  if (!r_or.has_value()) stop_post(r_or.error());
  const magmaan::estimate::OrdinalRobustResult& r = *r_or;
  return Rcpp::List::create(
      Rcpp::_["vcov"] = Rcpp::wrap(r.vcov),
      Rcpp::_["se"] = Rcpp::wrap(r.se),
      Rcpp::_["df"] = r.df,
      Rcpp::_["eigvals"] = Rcpp::wrap(r.eigvals),
      Rcpp::_["chisq_standard"] = r.chisq_standard,
      Rcpp::_["satorra_bentler"] = satorra_bentler_to_list(r.satorra_bentler),
      Rcpp::_["mean_var_adjusted"] = mean_var_to_list(r.mean_var_adjusted),
      Rcpp::_["scaled_shifted"] = scaled_shifted_to_list(r.scaled_shifted));
}

// [[Rcpp::export]]
Rcpp::List infer_mixed_ordinal_robust_ij(Rcpp::List fit,
                                         Rcpp::List mixed_stats,
                                         std::string weight = "") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::MixedOrdinalStats stats =
      mixed_ordinal_stats_from_arg(mixed_stats);
  if (weight.empty()) {
    if (!fit.containsElementNamed("estimator")) {
      Rcpp::stop("magmaan: infer_mixed_ordinal_robust_ij() needs `weight` "
                 "when fit$estimator is absent");
    }
    weight = Rcpp::as<std::string>(fit["estimator"]);
  }
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::robust_mixed_ordinal_ij(
      ctx.pt, ctx.rep, stats, est, ordinal_weight_from_string(weight),
      ordinal_parameterization_from_string(parameterization_name));
  if (!r_or.has_value()) stop_post(r_or.error());
  const magmaan::estimate::OrdinalRobustResult& r = *r_or;
  return Rcpp::List::create(
      Rcpp::_["vcov"] = Rcpp::wrap(r.vcov),
      Rcpp::_["se"] = Rcpp::wrap(r.se),
      Rcpp::_["df"] = r.df,
      Rcpp::_["chisq_standard"] = r.chisq_standard);
}

// infer_ordinal_profile_rmsea() — fixed-misspecification estimated-weight
// profile RMSEA for all-ordinal DWLS fits. Requires `ordinal_stats` with
// `moment_influence` and `int_data`, which the high-level DWLS path stores.
//
// [[Rcpp::export]]
Rcpp::List infer_ordinal_profile_rmsea(Rcpp::List fit,
                                       Rcpp::List ordinal_stats,
                                       double eig_tol = 1e-10) {
  require_dwls_fit(fit, "infer_ordinal_profile_rmsea()");
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::ordinal_dwls_profile_rmsea(
      ctx.pt, ctx.rep, stats, est,
      ordinal_parameterization_from_string(parameterization_name), eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return profile_rmsea_to_list(*r_or);
}

// infer_ordinal_profile_lrt() — nested estimated-weight profile LRT for two
// all-ordinal DWLS fits to the same `ordinal_stats`.
//
// [[Rcpp::export]]
Rcpp::List infer_ordinal_profile_lrt(Rcpp::List fit_H1,
                                     Rcpp::List fit_H0,
                                     Rcpp::List ordinal_stats,
                                     double eig_tol = 1e-10) {
  require_dwls_fit(fit_H1, "infer_ordinal_profile_lrt()");
  require_dwls_fit(fit_H0, "infer_ordinal_profile_lrt()");
  Ctx ctx1 = ctx_from_fit(fit_H1);
  Ctx ctx0 = ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est1 = est_from_fit(fit_H1);
  const magmaan::estimate::Estimates est0 = est_from_fit(fit_H0);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  const std::string par1 = fit_H1.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit_H1["parameterization"])
      : "delta";
  const std::string par0 = fit_H0.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit_H0["parameterization"])
      : "delta";
  if (par1 != par0) {
    Rcpp::stop("magmaan: infer_ordinal_profile_lrt() requires H1/H0 to use "
               "the same ordinal parameterization");
  }
  auto r_or = magmaan::estimate::ordinal_dwls_profile_lrt(
      std::move(ctx1.pt), ctx1.rep, stats, est1,
      std::move(ctx0.pt), ctx0.rep, est0,
      ordinal_parameterization_from_string(par1), eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return profile_lrt_to_list(*r_or);
}

// infer_ml_profile_lrt() — complete-data normal-theory ML nested profile LRT:
// the misspecification-robust ("observed-Hessian profile bread") difference test
// for two ML fits sharing the same observed data. Unlike the expected-info
// Satorra-2000 restriction map, the profile contrast lives in the first-stage
// moment space, so H1/H0 need not share npar. `X_per_group` is the per-group raw
// data in the model's ov order (the empirical Gamma is built from it).
//
// [[Rcpp::export]]
Rcpp::List infer_ml_profile_lrt(Rcpp::List fit_H1,
                                Rcpp::List fit_H0,
                                Rcpp::List X_per_group,
                                double eig_tol = 1e-10) {
  Ctx ctx1 = ctx_from_fit(fit_H1);
  Ctx ctx0 = ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est1 = est_from_fit(fit_H1);
  const magmaan::estimate::Estimates est0 = est_from_fit(fit_H0);

  const std::size_t G = ctx1.samp.S.size();
  if (static_cast<std::size_t>(X_per_group.size()) != G) {
    Rcpp::stop("infer_ml_profile_lrt: X_per_group has length %d but the model "
               "has %d group(s)",
               static_cast<int>(X_per_group.size()), static_cast<int>(G));
  }
  magmaan::data::RawData raw;
  raw.X.reserve(G);
  for (std::size_t g = 0; g < G; ++g) {
    raw.X.emplace_back(
        Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(X_per_group[g])));
  }

  auto r_or = magmaan::estimate::ml_profile_lrt(
      std::move(ctx1.pt), ctx1.rep, ctx1.samp, est1,
      std::move(ctx0.pt), ctx0.rep, est0,
      raw, eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return profile_lrt_to_list(*r_or);
}

// infer_ordinal_fit_measures_misspec() — consolidated estimated-weight
// misspecification fit table (RMSEA + CRMR/SRMR + CFI/TLI with CIs) for an
// all-ordinal DWLS fit. Requires `ordinal_stats` with `moment_influence` and
// `int_data`, which the high-level DWLS path stores. `estimated_weight = FALSE`
// is the fixed-weight comparator.
//
// [[Rcpp::export]]
Rcpp::List infer_ordinal_fit_measures_misspec(Rcpp::List fit,
                                              Rcpp::List ordinal_stats,
                                              bool estimated_weight = true,
                                              double conf_level = 0.90,
                                              double eig_tol = 1e-10) {
  require_dwls_fit(fit, "infer_ordinal_fit_measures_misspec()");
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::ordinal_fit_measures_misspec_inference(
      ctx.pt, ctx.rep, stats, est,
      ordinal_parameterization_from_string(parameterization_name),
      estimated_weight, conf_level, eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return misspec_fit_measures_to_list(*r_or);
}

// infer_mixed_ordinal_rmsea_misspec() -- mixed continuous/ordinal DWLS RMSEA
// misspecification CI over the estimated-weight profile-RMSEA path.
//
// [[Rcpp::export]]
Rcpp::List infer_mixed_ordinal_rmsea_misspec(Rcpp::List fit,
                                             Rcpp::List mixed_stats,
                                             bool estimated_weight = true,
                                             double conf_level = 0.90,
                                             double eig_tol = 1e-10) {
  require_dwls_fit(fit, "infer_mixed_ordinal_rmsea_misspec()");
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::MixedOrdinalStats stats =
      mixed_ordinal_stats_from_arg(mixed_stats);
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::mixed_ordinal_rmsea_misspec_inference(
      ctx.pt, ctx.rep, stats, est,
      ordinal_parameterization_from_string(parameterization_name),
      estimated_weight, conf_level, eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return rmsea_inference_to_list(*r_or);
}

// infer_mixed_ordinal_crmr_misspec() — mixed continuous/ordinal DWLS
// CRMR/SRMR misspecification CI over standardized association residuals.
// `srmr_denominator = TRUE` switches from the off-diagonal CRMR denominator to
// the full-vech SRMR denominator.
//
// [[Rcpp::export]]
Rcpp::List infer_mixed_ordinal_crmr_misspec(Rcpp::List fit,
                                            Rcpp::List mixed_stats,
                                            bool estimated_weight = true,
                                            bool srmr_denominator = false,
                                            double conf_level = 0.90,
                                            double eig_tol = 1e-10) {
  require_dwls_fit(fit, "infer_mixed_ordinal_crmr_misspec()");
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::MixedOrdinalStats stats =
      mixed_ordinal_stats_from_arg(mixed_stats);
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::mixed_ordinal_crmr_misspec_inference(
      ctx.pt, ctx.rep, stats, est,
      ordinal_parameterization_from_string(parameterization_name),
      estimated_weight, srmr_denominator, conf_level, eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return crmr_inference_to_list(*r_or);
}

// infer_mixed_ordinal_cfi_tli_misspec() -- mixed continuous/ordinal DWLS CFI/TLI
// misspecification CI using the mixed independence baseline with free marginal
// rows and zero association rows.
//
// [[Rcpp::export]]
Rcpp::List infer_mixed_ordinal_cfi_tli_misspec(Rcpp::List fit,
                                               Rcpp::List mixed_stats,
                                               bool estimated_weight = true,
                                               double conf_level = 0.90,
                                               double eig_tol = 1e-10) {
  require_dwls_fit(fit, "infer_mixed_ordinal_cfi_tli_misspec()");
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::MixedOrdinalStats stats =
      mixed_ordinal_stats_from_arg(mixed_stats);
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::mixed_ordinal_cfi_tli_misspec_inference(
      ctx.pt, ctx.rep, stats, est,
      ordinal_parameterization_from_string(parameterization_name),
      estimated_weight, conf_level, eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return incremental_inference_to_list(*r_or);
}

// infer_mixed_ordinal_fit_measures_misspec() -- consolidated mixed DWLS
// misspecification fit table (RMSEA + CRMR/SRMR + CFI/TLI with CIs).
//
// [[Rcpp::export]]
Rcpp::List infer_mixed_ordinal_fit_measures_misspec(
    Rcpp::List fit,
    Rcpp::List mixed_stats,
    bool estimated_weight = true,
    double conf_level = 0.90,
    double eig_tol = 1e-10) {
  require_dwls_fit(fit, "infer_mixed_ordinal_fit_measures_misspec()");
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::MixedOrdinalStats stats =
      mixed_ordinal_stats_from_arg(mixed_stats);
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::mixed_ordinal_fit_measures_misspec_inference(
      ctx.pt, ctx.rep, stats, est,
      ordinal_parameterization_from_string(parameterization_name),
      estimated_weight, conf_level, eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return misspec_fit_measures_to_list(*r_or);
}

// [[Rcpp::export]]
Rcpp::List infer_mixed_ordinal_profile_rmsea(Rcpp::List fit,
                                             Rcpp::List mixed_stats,
                                             double eig_tol = 1e-10) {
  require_dwls_fit(fit, "infer_mixed_ordinal_profile_rmsea()");
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::MixedOrdinalStats stats =
      mixed_ordinal_stats_from_arg(mixed_stats);
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto r_or = magmaan::estimate::mixed_ordinal_dwls_profile_rmsea(
      ctx.pt, ctx.rep, stats, est,
      ordinal_parameterization_from_string(parameterization_name), eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return profile_rmsea_to_list(*r_or);
}

// [[Rcpp::export]]
Rcpp::List infer_mixed_ordinal_profile_lrt(Rcpp::List fit_H1,
                                           Rcpp::List fit_H0,
                                           Rcpp::List mixed_stats,
                                           double eig_tol = 1e-10) {
  require_dwls_fit(fit_H1, "infer_mixed_ordinal_profile_lrt()");
  require_dwls_fit(fit_H0, "infer_mixed_ordinal_profile_lrt()");
  Ctx ctx1 = ctx_from_fit(fit_H1);
  Ctx ctx0 = ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est1 = est_from_fit(fit_H1);
  const magmaan::estimate::Estimates est0 = est_from_fit(fit_H0);
  magmaan::data::MixedOrdinalStats stats =
      mixed_ordinal_stats_from_arg(mixed_stats);
  const std::string par1 = fit_H1.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit_H1["parameterization"])
      : "delta";
  const std::string par0 = fit_H0.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit_H0["parameterization"])
      : "delta";
  if (par1 != par0) {
    Rcpp::stop("magmaan: infer_mixed_ordinal_profile_lrt() requires H1/H0 to "
               "use the same ordinal parameterization");
  }
  auto r_or = magmaan::estimate::mixed_ordinal_dwls_profile_lrt(
      std::move(ctx1.pt), ctx1.rep, stats, est1,
      std::move(ctx0.pt), ctx0.rep, est0,
      ordinal_parameterization_from_string(par1), eig_tol);
  if (!r_or.has_value()) stop_post(r_or.error());
  return profile_lrt_to_list(*r_or);
}

// =============================================================================
// Robust ("sandwich") standard errors
// =============================================================================

// infer_robust_se() — mirrors robust_se(pt, rep, samp, est, gamma_hat, {bread,
// moments, cov}). `gamma_hat` a p* x p* matrix. `cov` in {"model_implied",
// "empirical","browne_unbiased"} (the last errors for now). Single-block only
// (for multi-group the per-block weighting is implicit — use infer_robust_se_raw()).
//
// [[Rcpp::export]]
Rcpp::List infer_robust_se(Rcpp::List fit, Rcpp::NumericMatrix gamma_hat,
                           std::string bread = "expected", std::string moments = "structured",
                           std::string cov = "empirical") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  Eigen::MatrixXd G = Rcpp::as<Eigen::MatrixXd>(gamma_hat);
  auto r_or = magmaan::robust::robust_se(ctx.pt, ctx.rep, ctx.samp, est, G, spec_from(bread, moments, cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
                            Rcpp::_["se"]   = Rcpp::wrap(r_or->se));
}

// infer_robust_se_parts() — primitive form of infer_robust_se(): partable +
// sample_stats + theta + Gamma, without requiring a fit list.
//
// [[Rcpp::export]]
Rcpp::List infer_robust_se_parts(SEXP partable, Rcpp::List sample_stats,
                                 Rcpp::NumericVector theta,
                                 Rcpp::NumericMatrix gamma_hat,
                                 std::string bread = "expected",
                                 std::string moments = "structured",
                                 std::string cov = "empirical") {
  Ctx ctx = ctx_from_partable_sample_stats(partable, sample_stats,
                                           "infer_robust_se_parts");
  const magmaan::estimate::Estimates est = est_from_theta(theta);
  Eigen::MatrixXd G = Rcpp::as<Eigen::MatrixXd>(gamma_hat);
  auto r_or = magmaan::robust::robust_se(ctx.pt, ctx.rep, ctx.samp, est, G,
                                             spec_from(bread, moments, cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
                            Rcpp::_["se"]   = Rcpp::wrap(r_or->se));
}

// infer_robust_se_raw() — mirrors robust_se(pt, rep, samp, est, raw, {bread,
// moments, cov}). Derives Γ̂ from raw data `X` (a matrix, or a list of per-group
// matrices) via casewise contributions — never forms the p* x p* matrix. Works
// multi-group for the `Expected` bread (≙ lavaan `se = "robust.sem"` / MLM);
// the `Observed` bread (MLR) is single-block only.
//
// [[Rcpp::export]]
Rcpp::List infer_robust_se_raw(Rcpp::List fit, SEXP X,
                               std::string bread = "expected", std::string moments = "structured",
                               std::string cov = "empirical") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = raw_from_arg(ctx.rep, X);
  auto r_or = magmaan::robust::robust_se(ctx.pt, ctx.rep, ctx.samp, est, raw, spec_from(bread, moments, cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
                            Rcpp::_["se"]   = Rcpp::wrap(r_or->se));
}

// infer_casewise_scores_fit() — the (N_total x n_free) per-case ML score matrix
// (row i = d loglik_i / d theta at theta-hat), from inference::casewise_scores.
// Continuous ML/ULS/GLS; needs the fitting raw data. Feeds the one-step
// (no-refit) case-influence approximation est_change_*_approx in R.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_casewise_scores_fit(Rcpp::List fit, SEXP X) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = raw_from_arg(ctx.rep, X);
  auto s_or = magmaan::inference::casewise_scores(ctx.pt, ctx.rep, ctx.samp, raw, est);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::wrap(*s_or);
}

// infer_robust_se_raw_parts() — primitive form of infer_robust_se_raw():
// partable + sample_stats + theta + raw data, without requiring a fit list.
//
// [[Rcpp::export]]
Rcpp::List infer_robust_se_raw_parts(SEXP partable, Rcpp::List sample_stats,
                                     Rcpp::NumericVector theta, SEXP X,
                                     std::string bread = "expected",
                                     std::string moments = "structured",
                                     std::string cov = "empirical") {
  Ctx ctx = ctx_from_partable_sample_stats(partable, sample_stats,
                                           "infer_robust_se_raw_parts");
  const magmaan::estimate::Estimates est = est_from_theta(theta);
  magmaan::data::RawData raw = raw_from_arg(ctx.rep, X);
  auto r_or = magmaan::robust::robust_se(ctx.pt, ctx.rep, ctx.samp, est, raw,
                                             spec_from(bread, moments, cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
                            Rcpp::_["se"]   = Rcpp::wrap(r_or->se));
}

// infer_robust_se_zc() — Zc-overload of robust_se. Caller supplies the
// (N_total × C) casewise contributions matrix from
// `infer_casewise_contributions()` plus the total sample size; we skip the
// internal `casewise_contributions` rebuild AND any Γ̂ materialisation. Use
// this in simulation hot loops where the same Zc feeds multiple sandwich
// (bread = expected vs observed) and χ² eigvalue calls.
//
// [[Rcpp::export]]
Rcpp::List infer_robust_se_zc(Rcpp::List fit, Rcpp::NumericMatrix Zc,
                              double n_total,
                              std::string bread = "expected",
                              std::string moments = "structured",
                              std::string cov = "empirical") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  // Zero-copy view of the R matrix — avoids a per-call O(N·p*) memcpy that's
  // very visible in simulation hot loops where the same Zc feeds back-to-back
  // bread=expected / bread=observed calls.
  Eigen::Map<const Eigen::MatrixXd> Zc_m(Zc.begin(), Zc.nrow(), Zc.ncol());
  auto r_or = magmaan::robust::robust_se(ctx.pt, ctx.rep, ctx.samp, est,
                                             Zc_m, n_total,
                                             spec_from(bread, moments, cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
                            Rcpp::_["se"]   = Rcpp::wrap(r_or->se));
}

// --- robust_se_both_breads — three overloads ----------------------------------
// Run the shared robust_se setup (Δ-stacking, Γ_NT Cholesky, WΔ) ONCE and
// return both bread=expected and bread=observed sandwich SE vectors. Avoids
// recomputing the Δ pipeline on the second bread. Returns a 2-level list:
//   list(expected = list(vcov, se), observed = list(vcov, se)).

namespace {
Rcpp::List pair_to_list(const magmaan::robust::RobustSeBothBreads& r) {
  return Rcpp::List::create(
      Rcpp::_["expected"] = Rcpp::List::create(
          Rcpp::_["vcov"] = Rcpp::wrap(r.expected.vcov),
          Rcpp::_["se"]   = Rcpp::wrap(r.expected.se)),
      Rcpp::_["observed"] = Rcpp::List::create(
          Rcpp::_["vcov"] = Rcpp::wrap(r.observed.vcov),
          Rcpp::_["se"]   = Rcpp::wrap(r.observed.se)));
}
}  // namespace

// [[Rcpp::export]]
Rcpp::List infer_robust_se_both_breads(Rcpp::List fit,
                                       Rcpp::NumericMatrix gamma_hat,
                                       std::string moments = "structured",
                                       std::string cov = "empirical") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::MatrixXd G = Rcpp::as<Eigen::MatrixXd>(gamma_hat);
  auto r_or = magmaan::robust::robust_se_both_breads(
      ctx.pt, ctx.rep, ctx.samp, est, G,
      moments_from_string(moments), cov_from_string(cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return pair_to_list(*r_or);
}

// [[Rcpp::export]]
Rcpp::List infer_robust_se_both_breads_raw(Rcpp::List fit, SEXP X,
                                           std::string moments = "structured",
                                           std::string cov = "empirical") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  magmaan::data::RawData raw = raw_from_arg(ctx.rep, X);
  auto r_or = magmaan::robust::robust_se_both_breads(
      ctx.pt, ctx.rep, ctx.samp, est, raw,
      moments_from_string(moments), cov_from_string(cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return pair_to_list(*r_or);
}

// [[Rcpp::export]]
Rcpp::List infer_robust_se_both_breads_zc(Rcpp::List fit,
                                          Rcpp::NumericMatrix Zc,
                                          double n_total,
                                          std::string moments = "structured",
                                          std::string cov = "empirical") {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  Eigen::Map<const Eigen::MatrixXd> Zc_m(Zc.begin(), Zc.nrow(), Zc.ncol());
  auto r_or = magmaan::robust::robust_se_both_breads(
      ctx.pt, ctx.rep, ctx.samp, est, Zc_m, n_total,
      moments_from_string(moments), cov_from_string(cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return pair_to_list(*r_or);
}
