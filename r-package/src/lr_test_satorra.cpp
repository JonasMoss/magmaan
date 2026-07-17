// R bindings for the Satorra-2000 nested-model χ² difference test.
//
// `infer_lr_test_satorra2000` is the single C++ entry point: it takes both
// fit objects (the R list returned by `fit_fit`), the raw data per group
// (reordered to the model's variable order), the chi²/df pair for each fit,
// and an optional GammaSource flag.  Returns an R list mirroring
// `LRSatorra2000Result` field-for-field.

// [[Rcpp::depends(RcppEigen)]]

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Rcpp.h>
#include <RcppEigen.h>

#include "internal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/weighted_inference.hpp"

namespace {

magmaan::robust::GammaSource parse_gamma(const std::string& s) {
  if (s == "NT" || s == "nt") return magmaan::robust::GammaSource::NT;
  if (s == "unbiased" || s == "UG" || s == "ug" ||
      s == "browne" || s == "du_bentler") {
    return magmaan::robust::GammaSource::Unbiased;
  }
  return magmaan::robust::GammaSource::Empirical;
}

magmaan::robust::SatorraAMethod parse_a_method(const std::string& s) {
  if (s == "delta" || s == "Delta") return magmaan::robust::SatorraAMethod::Delta;
  return magmaan::robust::SatorraAMethod::Exact;
}

magmaan::robust::SatorraMomentConvention parse_moment_convention(
    const std::string& s) {
  if (s == "magmaan" || s == "Magmaan") {
    return magmaan::robust::SatorraMomentConvention::Magmaan;
  }
  if (s == "lavaan" || s == "Lavaan") {
    return magmaan::robust::SatorraMomentConvention::Lavaan;
  }
  Rcpp::stop("magmaan: nested Satorra-2000 `convention` must be 'magmaan' "
             "or 'lavaan' (got '%s')", s);
}

magmaan::estimate::fiml::Stage1RegularizationTarget
h1_reference_covariance_target_from_string(const std::string& target) {
  using T = magmaan::estimate::fiml::Stage1RegularizationTarget;
  if (target == "diagonal") return T::Diagonal;
  if (target == "scaled_identity" || target == "scaled.identity")
    return T::ScaledIdentity;
  if (target == "identity") return T::Identity;
  Rcpp::stop("magmaan: h1_reference_regularization covariance target must be "
             "diagonal, scaled_identity, or identity");
}

bool list_has_nonnull(Rcpp::List x, const char* name) {
  return x.containsElementNamed(name) && !Rf_isNull(x[name]);
}

void apply_h1_reference_common_options(
    Rcpp::List l,
    magmaan::robust::H1ReferenceRegularizationOptions& opts) {
  if (list_has_nonnull(l, "condition_max")) {
    const double x = Rcpp::as<double>(l["condition_max"]);
    const double val = (std::isfinite(x) || std::isinf(x))
        ? x
        : std::numeric_limits<double>::infinity();
    opts.covariance_condition_max = val;
    opts.information_condition_max = val;
  }
  if (list_has_nonnull(l, "min_eigenvalue")) {
    const double x = Rcpp::as<double>(l["min_eigenvalue"]);
    opts.covariance_min_eigenvalue = x;
    opts.information_min_eigenvalue = x;
  }
}

void apply_h1_reference_covariance_options(
    Rcpp::List l,
    magmaan::robust::H1ReferenceRegularizationOptions& opts) {
  if (list_has_nonnull(l, "condition_max")) {
    const double x = Rcpp::as<double>(l["condition_max"]);
    opts.covariance_condition_max = (std::isfinite(x) || std::isinf(x))
        ? x
        : std::numeric_limits<double>::infinity();
  }
  if (list_has_nonnull(l, "min_eigenvalue")) {
    opts.covariance_min_eigenvalue = Rcpp::as<double>(l["min_eigenvalue"]);
  }
  if (list_has_nonnull(l, "target")) {
    opts.covariance_target = h1_reference_covariance_target_from_string(
        Rcpp::as<std::string>(l["target"]));
  }
  if (list_has_nonnull(l, "intensity")) {
    const double x = Rcpp::as<double>(l["intensity"]);
    opts.covariance_intensity = std::isfinite(x)
        ? x
        : std::numeric_limits<double>::quiet_NaN();
  }
  if (list_has_nonnull(l, "jacobian_step")) {
    opts.covariance_jacobian_step = Rcpp::as<double>(l["jacobian_step"]);
  }
}

void apply_h1_reference_information_options(
    Rcpp::List l,
    magmaan::robust::H1ReferenceRegularizationOptions& opts) {
  if (list_has_nonnull(l, "condition_max")) {
    const double x = Rcpp::as<double>(l["condition_max"]);
    opts.information_condition_max = (std::isfinite(x) || std::isinf(x))
        ? x
        : std::numeric_limits<double>::infinity();
  }
  if (list_has_nonnull(l, "min_eigenvalue")) {
    opts.information_min_eigenvalue = Rcpp::as<double>(l["min_eigenvalue"]);
  }
}

magmaan::robust::H1ReferenceRegularizationOptions
h1_reference_regularization_options_from(SEXP arg) {
  magmaan::robust::H1ReferenceRegularizationOptions opts;
  if (Rf_isNull(arg)) return opts;

  opts.enabled = true;
  opts.covariance_condition_max = 1e6;
  opts.information_condition_max = 1e6;
  if (Rf_isLogical(arg) && Rf_length(arg) == 1) {
    const int flag = LOGICAL(arg)[0];
    opts.enabled = flag != 0 && flag != NA_LOGICAL;
    return opts;
  }
  if (TYPEOF(arg) != VECSXP) {
    Rcpp::stop("magmaan: h1_reference_regularization must be NULL, TRUE/FALSE, or a list");
  }

  Rcpp::List l(arg);
  if (list_has_nonnull(l, "enabled")) {
    opts.enabled = Rcpp::as<bool>(l["enabled"]);
  }
  if (!opts.enabled) return opts;
  apply_h1_reference_common_options(l, opts);
  apply_h1_reference_covariance_options(l, opts);
  apply_h1_reference_information_options(l, opts);
  if (list_has_nonnull(l, "covariance")) {
    if (TYPEOF(l["covariance"]) != VECSXP) {
      Rcpp::stop("magmaan: h1_reference_regularization$covariance must be a list");
    }
    apply_h1_reference_covariance_options(Rcpp::List(l["covariance"]), opts);
  }
  if (list_has_nonnull(l, "information")) {
    if (TYPEOF(l["information"]) != VECSXP) {
      Rcpp::stop("magmaan: h1_reference_regularization$information must be a list");
    }
    apply_h1_reference_information_options(Rcpp::List(l["information"]), opts);
  }
  return opts;
}

magmaan::robust::GammaComputation parse_gamma_computation(const std::string& s) {
  if (s == "materialized" || s == "full" || s == "explicit") {
    return magmaan::robust::GammaComputation::Materialized;
  }
  if (s == "dense") {
    return magmaan::robust::GammaComputation::Dense;
  }
  return magmaan::robust::GammaComputation::Streaming;
}

magmaan::estimate::OrdinalWeightKind parse_ordinal_weight(const std::string& s) {
  if (s == "DWLS" || s == "dwls") return magmaan::estimate::OrdinalWeightKind::DWLS;
  if (s == "WLS" || s == "wls") return magmaan::estimate::OrdinalWeightKind::WLS;
  if (s == "ULS" || s == "uls") return magmaan::estimate::OrdinalWeightKind::ULS;
  Rcpp::stop("magmaan: ordinal nested test `weight` must be 'DWLS', 'WLS', or 'ULS' (got '%s')", s);
}

using FimlPack = magmaan::estimate::fiml::FIMLPack;
using FimlH1 = magmaan::estimate::fiml::FIMLH1;

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

magmaan::estimate::gmm::Weight continuous_ls_weight_from_arg(
    const magmaanr::Ctx& ctx,
    const magmaan::estimate::Estimates& est,
    const std::string& estimator,
    SEXP weight) {
  if (estimator == "ULS") {
    if (!Rf_isNull(weight)) {
      Rcpp::stop("magmaan: continuous ULS nested tests do not accept `weight`");
    }
    return {};
  }
  if (estimator == "GLS") {
    if (!Rf_isNull(weight)) {
      Rcpp::stop("magmaan: continuous GLS nested tests rebuild the fitted "
                 "normal-theory weight; omit `weight`");
    }
    auto ev_or = magmaan::model::ModelEvaluator::build(ctx.pt, ctx.rep);
    if (!ev_or.has_value()) magmaanr::stop_model(ev_or.error());
    auto w_or = magmaan::estimate::gmm::normal_theory_weight(
        *ev_or, ctx.samp, est.theta);
    if (!w_or.has_value()) magmaanr::stop_fit(w_or.error());
    return std::move(*w_or);
  }
  if (estimator != "WLS") {
    Rcpp::stop("magmaan: continuous-LS nested tests require ULS, GLS, or WLS "
               "fits (got estimator '%s')", estimator);
  }
  if (Rf_isNull(weight)) {
    Rcpp::stop("magmaan: continuous WLS nested tests require the explicit "
               "fitting `weight`");
  }
  magmaan::estimate::gmm::Weight out;
  const std::size_t n_blocks = ctx.samp.S.size();
  out.reserve(n_blocks);
  if (Rf_isMatrix(weight)) {
    if (n_blocks != 1) {
      Rcpp::stop("magmaan: WLS weights must be a list of %d matrices for a "
                 "multi-group model", static_cast<int>(n_blocks));
    }
    out.push_back(
        Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(weight)));
  } else if (TYPEOF(weight) == VECSXP) {
    Rcpp::List weights(weight);
    if (static_cast<std::size_t>(weights.size()) != n_blocks) {
      Rcpp::stop("magmaan: WLS weights list has length %d but the model has "
                 "%d group(s)", static_cast<int>(weights.size()),
                 static_cast<int>(n_blocks));
    }
    for (R_xlen_t b = 0; b < weights.size(); ++b) {
      out.push_back(Rcpp::as<Eigen::MatrixXd>(
          Rcpp::NumericMatrix(weights[b])));
    }
  } else {
    Rcpp::stop("magmaan: WLS `weight` must be a matrix or list of matrices");
  }
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
      Wdl(x["W_dwls"]), Wfl(x["W_wls"]), nlevl(x["n_levels"]);
  Rcpp::IntegerVector nobs(x["nobs"]);
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
    out.n_obs.push_back(static_cast<std::int64_t>(nobs[b]));
    out.n_levels.push_back(Rcpp::as<std::vector<std::int32_t>>(Rcpp::IntegerVector(nlevl[b])));
  }
  return out;
}

magmaan::data::RawData raw_from_group_list(Rcpp::List X_per_group,
                                           std::size_t G,
                                           const char* caller) {
  if (static_cast<std::size_t>(X_per_group.size()) != G) {
    Rcpp::stop("%s: X_per_group has length %d but the model has %d group(s)",
               caller, static_cast<int>(X_per_group.size()),
               static_cast<int>(G));
  }
  magmaan::data::RawData raw;
  raw.X.reserve(G);
  for (std::size_t g = 0; g < G; ++g) {
    raw.X.emplace_back(
        Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(X_per_group[g])));
  }
  return raw;
}

magmaan::data::RawData fiml_raw_from_fit_arg(const lvm::MatrixRep& rep,
                                             SEXP raw_data) {
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
    Rcpp::NumericMatrix Xb =
        magmaanr::block_matrix(X_arg, b, n_blocks, "raw_data$X");
    const std::vector<int> perm =
        magmaanr::perm_for_cols(Xb, rep.ov_names[b], "raw_data$X");
    const int n = Xb.nrow();
    const int p = static_cast<int>(perm.size());

    Rcpp::LogicalMatrix Mb;
    const bool has_mask = !Rf_isNull(mask_arg);
    if (has_mask) {
      Mb = magmaanr::block_mask_matrix(mask_arg, b, n_blocks, "raw_data$mask");
      if (Mb.nrow() != n || Mb.ncol() != Xb.ncol()) {
        Rcpp::stop("magmaan: raw_data$mask block %d has shape %dx%d but "
                   "raw_data$X has %dx%d",
                   static_cast<int>(b + 1), Mb.nrow(), Mb.ncol(),
                   Xb.nrow(), Xb.ncol());
      }
    }

    Eigen::MatrixXd X(n, p);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);
    for (int r = 0; r < n; ++r) {
      for (int k = 0; k < p; ++k) {
        const int src = perm[static_cast<std::size_t>(k)];
        const double x = Xb(r, src);
        const bool observed = has_mask
            ? (Mb(r, src) != NA_LOGICAL && Mb(r, src) != 0)
            : std::isfinite(x);
        if (observed && !std::isfinite(x)) {
          Rcpp::stop("magmaan: raw_data$mask marks a non-finite value as "
                     "observed in block %d, row %d",
                     static_cast<int>(b + 1), r + 1);
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

Rcpp::List sb_diff_to_list(const magmaan::robust::LRSatorraBentlerDiffResult& r) {
  Rcpp::CharacterVector warns(static_cast<R_xlen_t>(r.warnings.size()));
  for (std::size_t k = 0; k < r.warnings.size(); ++k) {
    warns[static_cast<R_xlen_t>(k)] = r.warnings[k];
  }
  return Rcpp::List::create(
      Rcpp::_["T_diff"]   = r.T_diff,
      Rcpp::_["df_diff"]  = r.df_diff,
      Rcpp::_["scale_c"]  = r.scale_c,
      Rcpp::_["T_scaled"] = r.T_scaled,
      Rcpp::_["p_value"]  = r.p_value,
      Rcpp::_["c_H0"]     = r.c_H0,
      Rcpp::_["c_H1"]     = r.c_H1,
      Rcpp::_["c_hybrid"] = r.c_hybrid,
      Rcpp::_["warnings"] = warns);
}

Rcpp::List h1_reference_regularization_to_list(
    const magmaan::robust::H1ReferenceRegularizationDiagnostics& d) {
  const R_xlen_t nb = static_cast<R_xlen_t>(d.blocks.size());
  Rcpp::CharacterVector component(nb), target(nb);
  Rcpp::NumericVector raw_min(nb), raw_max(nb), raw_cond(nb);
  Rcpp::NumericVector min_eig(nb), max_eig(nb), cond(nb);
  Rcpp::NumericVector floor(nb), intensity(nb);
  Rcpp::LogicalVector applied(nb);
  for (R_xlen_t i = 0; i < nb; ++i) {
    const auto& b = d.blocks[static_cast<std::size_t>(i)];
    component[i] = b.component;
    target[i] = b.target;
    raw_min[i] = b.raw_min_eigen;
    raw_max[i] = b.raw_max_eigen;
    raw_cond[i] = b.raw_condition;
    min_eig[i] = b.min_eigen;
    max_eig[i] = b.max_eigen;
    cond[i] = b.condition;
    floor[i] = b.floor;
    intensity[i] = b.intensity;
    applied[i] = b.applied;
  }
  return Rcpp::List::create(
      Rcpp::_["enabled"] = d.enabled,
      Rcpp::_["applied"] = d.applied,
      Rcpp::_["component"] = d.component,
      Rcpp::_["condition_max"] = d.condition_max,
      Rcpp::_["min_eigenvalue"] = d.min_eigenvalue,
      Rcpp::_["blocks"] = Rcpp::List::create(
          Rcpp::_["component"] = component,
          Rcpp::_["target"] = target,
          Rcpp::_["raw_min_eigen"] = raw_min,
          Rcpp::_["raw_max_eigen"] = raw_max,
          Rcpp::_["raw_condition"] = raw_cond,
          Rcpp::_["min_eigen"] = min_eig,
          Rcpp::_["max_eigen"] = max_eig,
          Rcpp::_["condition"] = cond,
          Rcpp::_["floor"] = floor,
          Rcpp::_["intensity"] = intensity,
          Rcpp::_["applied"] = applied));
}

Rcpp::List satorra2000_to_list(const magmaan::robust::LRSatorra2000Result& r) {
  Rcpp::CharacterVector warns(static_cast<R_xlen_t>(r.warnings.size()));
  for (std::size_t k = 0; k < r.warnings.size(); ++k) {
    warns[static_cast<R_xlen_t>(k)] = r.warnings[k];
  }
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["T_diff"]      = r.T_diff,
      Rcpp::_["df_diff"]     = r.df_diff,
      Rcpp::_["p_unscaled"]  = r.p_unscaled,
      Rcpp::_["eigenvalues"] = Rcpp::wrap(r.eigenvalues),
      Rcpp::_["eigenvalues_unbiased"] =
          Rcpp::wrap(r.eigenvalues_unbiased),
      Rcpp::_["scale_c"]     = r.scale_c,
      Rcpp::_["T_scaled"]    = r.T_scaled,
      Rcpp::_["p_scaled"]    = r.p_scaled,
      Rcpp::_["adjust_d0"]   = r.adjust_d0,
      Rcpp::_["T_adjusted"]  = r.T_adjusted,
      Rcpp::_["p_adjusted"]  = r.p_adjusted,
      Rcpp::_["scaled_shifted"] = Rcpp::List::create(
          Rcpp::_["chi2_adj"] = r.scaled_shifted.chi2_adj,
          Rcpp::_["df"]       = r.scaled_shifted.df,
          Rcpp::_["scale_a"]  = r.scaled_shifted.scale_a,
          Rcpp::_["shift_b"]  = r.scaled_shifted.shift_b,
          Rcpp::_["pvalue"]   = r.p_scaled_shifted),
      Rcpp::_["p_scaled_shifted"] = r.p_scaled_shifted,
      Rcpp::_["p_mixture"]   = r.p_mixture,
      Rcpp::_["warnings"]    = warns);
  if (r.h1_reference_regularization.enabled) {
    out["h1_reference_regularization"] =
        h1_reference_regularization_to_list(r.h1_reference_regularization);
  }
  return out;
}

void validate_same_fiml_raw(const magmaan::data::RawData& h1,
                            const magmaan::data::RawData& h0,
                            const char* caller) {
  if (h1.X.size() != h0.X.size()) {
    Rcpp::stop("%s: H0/H1 raw_data have different block counts", caller);
  }
  const bool h1_mask = !h1.mask.empty();
  const bool h0_mask = !h0.mask.empty();
  if (h1_mask != h0_mask) {
    Rcpp::stop("%s: H0/H1 raw_data mask presence differs", caller);
  }
  for (std::size_t b = 0; b < h1.X.size(); ++b) {
    if (h1.X[b].rows() != h0.X[b].rows() || h1.X[b].cols() != h0.X[b].cols()) {
      Rcpp::stop("%s: H0/H1 raw_data block %d shapes differ",
                 caller, static_cast<int>(b + 1));
    }
    if (h1_mask) {
      if (h1.mask[b].rows() != h0.mask[b].rows() ||
          h1.mask[b].cols() != h0.mask[b].cols()) {
        Rcpp::stop("%s: H0/H1 raw_data mask block %d shapes differ",
                   caller, static_cast<int>(b + 1));
      }
      for (Eigen::Index r = 0; r < h1.mask[b].rows(); ++r) {
        for (Eigen::Index c = 0; c < h1.mask[b].cols(); ++c) {
          if (h1.mask[b](r, c) != h0.mask[b](r, c)) {
            Rcpp::stop("%s: H0/H1 raw_data masks differ at block %d, row %d, column %d",
                       caller, static_cast<int>(b + 1),
                       static_cast<int>(r + 1), static_cast<int>(c + 1));
          }
          if (h1.mask[b](r, c) != 0 &&
              h1.X[b](r, c) != h0.X[b](r, c)) {
            Rcpp::stop("%s: H0/H1 raw_data observed values differ at block %d, row %d, column %d",
                       caller, static_cast<int>(b + 1),
                       static_cast<int>(r + 1), static_cast<int>(c + 1));
          }
        }
      }
    } else if (!h1.X[b].isApprox(h0.X[b], 0.0)) {
      Rcpp::stop("%s: H0/H1 raw_data values differ in block %d",
                 caller, static_cast<int>(b + 1));
    }
  }
}

}  // namespace

// infer_lr_test_satorra2000() — Satorra (2000) scaled nested-model LR test.
//
// `fit_H1` / `fit_H0` are the two fit lists returned by `fit_fit()` — they
// must share the same lavaanified partable shape, differing only in
// constraint rows (== / shared labels / linear-equality `R·θ = d`).
//
// `X_per_group` is a list of n_g × p_g raw-data matrices, one per group,
// columns already aligned to the model's observed-variable order.  Build it
// from the user-supplied data frame via the same column reorder
// `infer_casewise_contributions` uses.
//
// `T_H1`, `df_H1`, `T_H0`, `df_H0` are the χ² statistics and degrees of
// freedom returned by `infer_information_expected()` on each fit (we take them
// pre-computed to avoid a redundant SE pass inside the test).
//
// `gamma` is `"empirical"` (default — empirical Γ̂ from the casewise outer
// products, matches lavaan's `estimator = "MLR"` / `"MLM"`), `"unbiased"`
// (Browne/Du-Bentler finite-sample correction), `"both"` (paired empirical and
// unbiased spectra from one streaming pass), or `"NT"` (normal-theory Γ_NT, a
// sanity-check path where all λⱼ → 1 and `T_scaled == T_diff`).
//
// [[Rcpp::export]]
Rcpp::List infer_lr_test_satorra2000(Rcpp::List           fit_H1,
                                     Rcpp::List           fit_H0,
                                     Rcpp::List           X_per_group,
                                     double               T_H1,
                                     int                  df_H1,
                                     double               T_H0,
                                     int                  df_H0,
                                     std::string          gamma = "empirical",
                                     std::string          a_method = "exact",
                                     std::string          computation = "streaming") {
  // ── Build the H1 context (pt, rep, samp) and pull θ̂_H1 ─────────────────
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);

  // ── Re-derive both K matrices from each fit's partable ────────────────
  auto K_H1_or = magmaan::estimate::build_eq_constraints(ctx_H1.pt);
  if (!K_H1_or.has_value()) magmaanr::stop_post(K_H1_or.error());
  auto K_H0_or = magmaan::estimate::build_eq_constraints(ctx_H0.pt);
  if (!K_H0_or.has_value()) magmaanr::stop_post(K_H0_or.error());

  // ── Per-group raw data, means, n_g, weights ───────────────────────────
  const std::size_t G = ctx_H1.samp.S.size();
  if (static_cast<std::size_t>(X_per_group.size()) != G) {
    Rcpp::stop("infer_lr_test_satorra2000: X_per_group has length %d but the "
               "model has %d group(s)", static_cast<int>(X_per_group.size()),
               static_cast<int>(G));
  }
  std::vector<Eigen::MatrixXd>  Xs;     Xs.reserve(G);
  std::vector<Eigen::VectorXd>  means;  means.reserve(G);
  std::vector<std::int32_t>     n_g;    n_g.reserve(G);
  std::int64_t N_total = 0;
  for (std::size_t g = 0; g < G; ++g) {
    Eigen::MatrixXd X = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(X_per_group[g]));
    Eigen::VectorXd m = X.colwise().mean();
    n_g.push_back(static_cast<std::int32_t>(X.rows()));
    N_total += X.rows();
    Xs.emplace_back(std::move(X));
    means.emplace_back(std::move(m));
  }
  std::vector<double> w_g(G);
  for (std::size_t g = 0; g < G; ++g) {
    w_g[g] = static_cast<double>(n_g[g]) / static_cast<double>(N_total);
  }

  // ── Run the orchestrator ──────────────────────────────────────────────
  auto r_or = magmaan::robust::lr_test_satorra2000_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H1.theta,
      *K_H1_or,
      ctx_H0.pt, ctx_H0.rep, est_H0.theta,
      *K_H0_or,
      Xs, means, n_g, w_g,
      T_H0, T_H1, df_H0, df_H1,
      magmaan::robust::Satorra2000Options{
          .a_method = parse_a_method(a_method),
          .gamma = parse_gamma(gamma),
          .computation = parse_gamma_computation(computation),
          .also_unbiased = gamma == "both"});
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  const magmaan::robust::LRSatorra2000Result& r = *r_or;

  return satorra2000_to_list(r);
}

// infer_continuous_ls_lr_test_satorra2000() — estimator-specific
// Satorra-2000 restriction-map test for continuous ULS/GLS/WLS fits.
//
// [[Rcpp::export]]
Rcpp::List infer_continuous_ls_lr_test_satorra2000(
    Rcpp::List fit_H1,
    Rcpp::List fit_H0,
    Rcpp::List X_per_group,
    double T_H1,
    int df_H1,
    double T_H0,
    int df_H0,
    SEXP weight = R_NilValue,
    std::string gamma = "empirical",
    std::string a_method = "exact") {
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 =
      magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 =
      magmaanr::est_from_fit(fit_H0);
  const std::string estimator_H1 = fit_H1.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit_H1["estimator"])
      : "";
  const std::string estimator_H0 = fit_H0.containsElementNamed("estimator")
      ? Rcpp::as<std::string>(fit_H0["estimator"])
      : "";
  if (estimator_H1 != estimator_H0) {
    Rcpp::stop("infer_continuous_ls_lr_test_satorra2000: H1/H0 estimators "
               "differ ('%s' versus '%s')", estimator_H1, estimator_H0);
  }
  if (gamma == "unbiased" || gamma == "UG" || gamma == "ug") {
    Rcpp::stop("infer_continuous_ls_lr_test_satorra2000: unbiased Gamma is "
               "available only for normal-theory ML");
  }
  if (ctx_H1.samp.S.size() != ctx_H0.samp.S.size() ||
      ctx_H1.samp.n_obs != ctx_H0.samp.n_obs) {
    Rcpp::stop("infer_continuous_ls_lr_test_satorra2000: H1/H0 sample "
               "statistics have different block structure");
  }
  for (std::size_t b = 0; b < ctx_H1.samp.S.size(); ++b) {
    if (!ctx_H1.samp.S[b].isApprox(ctx_H0.samp.S[b], 1e-12)) {
      Rcpp::stop("infer_continuous_ls_lr_test_satorra2000: H1/H0 sample "
                 "covariances differ in block %d", static_cast<int>(b + 1));
    }
  }

  magmaan::data::RawData raw = raw_from_group_list(
      X_per_group, ctx_H1.samp.S.size(),
      "infer_continuous_ls_lr_test_satorra2000");
  magmaan::estimate::gmm::Weight w = continuous_ls_weight_from_arg(
      ctx_H1, est_H1, estimator_H1, weight);
  auto r_or = magmaan::estimate::lr_test_satorra2000_continuous_ls(
      std::move(ctx_H1.pt), ctx_H1.rep, ctx_H1.samp, est_H1,
      std::move(ctx_H0.pt), ctx_H0.rep, est_H0,
      w, raw, T_H0, T_H1, df_H0, df_H1,
      parse_gamma(gamma), parse_a_method(a_method));
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return satorra2000_to_list(*r_or);
}

// infer_fiml_lr_test_satorra2000() — FIML/missing-data restriction-map nested
// LR test. Uses fit_H1$raw_data as the saturated observed-pattern H1 source and
// validates fit_H0$raw_data shape/mask/value compatibility.
//
// [[Rcpp::export]]
Rcpp::List infer_fiml_lr_test_satorra2000(Rcpp::List  fit_H1,
                                          Rcpp::List  fit_H0,
                                          std::string gamma = "empirical",
                                          std::string a_method = "exact",
                                          double      h_step = 1e-4,
                                          std::string ud_method = "2000",
                                          std::string convention = "magmaan",
                                          std::string computation = "streaming",
                                          SEXP h1_reference_regularization = R_NilValue) {
  if (!fit_H1.containsElementNamed("raw_data") ||
      !fit_H0.containsElementNamed("raw_data")) {
    Rcpp::stop("infer_fiml_lr_test_satorra2000: both FIML fits must carry $raw_data");
  }
  if (ud_method != "2000" && ud_method != "2001") {
    Rcpp::stop("infer_fiml_lr_test_satorra2000: ud_method must be '2000' or '2001'");
  }
  const auto h1_ref_opts =
      h1_reference_regularization_options_from(h1_reference_regularization);
  const auto computation_kind = parse_gamma_computation(computation);
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);

  magmaan::data::RawData raw_H1 =
      fiml_raw_from_fit_arg(ctx_H1.rep, fit_H1["raw_data"]);
  magmaan::data::RawData raw_H0 =
      fiml_raw_from_fit_arg(ctx_H0.rep, fit_H0["raw_data"]);
  validate_same_fiml_raw(raw_H1, raw_H0, "infer_fiml_lr_test_satorra2000");

  std::unique_ptr<FimlPack> owned_pack;
  const FimlPack* pack = fiml_pack_ptr_from_fit(fit_H1);
  if (!pack) pack = fiml_pack_ptr_from_fit(fit_H0);
  if (!pack) {
    auto pack_or = magmaan::estimate::fiml::fiml_pack(raw_H1);
    if (!pack_or.has_value()) magmaanr::stop_fit(pack_or.error());
    owned_pack = std::make_unique<FimlPack>(std::move(*pack_or));
    pack = owned_pack.get();
  }

  std::unique_ptr<FimlH1> owned_h1;
  const FimlH1* h1 = fiml_h1_ptr_from_fit(fit_H1);
  if (!h1) h1 = fiml_h1_ptr_from_fit(fit_H0);
  if (!h1) {
    auto h1_or = magmaan::estimate::fiml::fiml_h1_moments(raw_H1, *pack);
    if (!h1_or.has_value()) magmaanr::stop_fit(h1_or.error());
    owned_h1 = std::make_unique<FimlH1>(std::move(*h1_or));
    h1 = owned_h1.get();
  }

  // One saturated build for the whole nested test: H0 and H1 share it (same
  // data). Reuse a fit's $stage1 (e.g. stamped by the caller or carried by an
  // ML2S fit) so neither the df/T path below nor the difference-spectrum driver
  // rebuilds the EM + observed information again.
  magmaan::estimate::fiml::SaturatedMoments sm;
  if (!magmaanr::saturated_from_stage1_with_information(fit_H1, sm) &&
      !magmaanr::saturated_from_stage1_with_information(fit_H0, sm)) {
    auto sm_or = magmaan::estimate::fiml::saturated_em_moments(
        raw_H1, *pack, *h1);
    if (!sm_or.has_value()) magmaanr::stop_post(sm_or.error());
    sm = std::move(*sm_or);
  }

  auto df_H1_or = magmaan::inference::df_stat(ctx_H1.pt, ctx_H1.samp, est_H1.theta);
  if (!df_H1_or.has_value()) magmaanr::stop_post(df_H1_or.error());
  auto df_H0_or = magmaan::inference::df_stat(ctx_H0.pt, ctx_H0.samp, est_H0.theta);
  if (!df_H0_or.has_value()) magmaanr::stop_post(df_H0_or.error());
  auto fx_H1_or = magmaan::estimate::fiml::fiml_extras(
      ctx_H1.pt, ctx_H1.rep, raw_H1, est_H1, *pack, *h1);
  if (!fx_H1_or.has_value()) magmaanr::stop_post(fx_H1_or.error());
  auto fx_H0_or = magmaan::estimate::fiml::fiml_extras(
      ctx_H0.pt, ctx_H0.rep, raw_H1, est_H0, *pack, *h1);
  if (!fx_H0_or.has_value()) magmaanr::stop_post(fx_H0_or.error());

  if (ud_method == "2001") {
    auto r_or = magmaan::robust::lr_test_satorra2001_fiml_from_data(
        ctx_H1.pt, ctx_H1.rep, est_H1.theta,
        ctx_H0.pt, ctx_H0.rep, est_H0.theta,
        raw_H1, fx_H0_or->chi2, fx_H1_or->chi2, *df_H0_or, *df_H1_or, h_step,
        &sm, h1_ref_opts);
    if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
    return satorra2000_to_list(*r_or);
  }

  auto K_H1_or = magmaan::estimate::build_eq_constraints(
      ctx_H1.pt, /*allow_nonlinear=*/true);
  if (!K_H1_or.has_value()) magmaanr::stop_post(K_H1_or.error());
  auto K_H0_or = magmaan::estimate::build_eq_constraints(
      ctx_H0.pt, /*allow_nonlinear=*/true);
  if (!K_H0_or.has_value()) magmaanr::stop_post(K_H0_or.error());

  auto r_or = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H1.theta, *K_H1_or,
      ctx_H0.pt, ctx_H0.rep, est_H0.theta, *K_H0_or,
      raw_H1, fx_H0_or->chi2, fx_H1_or->chi2, *df_H0_or, *df_H1_or,
      *pack, *h1, parse_gamma(gamma), parse_a_method(a_method), h_step, &sm,
      parse_moment_convention(convention), h1_ref_opts, computation_kind);
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return satorra2000_to_list(*r_or);
}

// infer_ml2s_lr_test_satorra2000() — two-stage ML restriction-map nested LR
// test. Uses the Stage-1 saturated EM moments as the Stage-2 sample moments:
// T_diff is the complete-data ML chi-square difference on those moments, while
// the difference spectrum uses the selected Stage-2 weight + EM-ACOV meat.
//
// [[Rcpp::export]]
Rcpp::List infer_ml2s_lr_test_satorra2000(Rcpp::List  fit_H1,
                                          Rcpp::List  fit_H0,
                                          std::string gamma = "empirical",
                                          std::string a_method = "exact",
                                          double      h_step = 1e-4,
                                          std::string ud_method = "2000",
                                          std::string stage2_weight = "nt",
                                          double      dls_a = 0.5,
                                          std::string convention = "magmaan") {
  if (!fit_H1.containsElementNamed("raw_data") ||
      !fit_H0.containsElementNamed("raw_data")) {
    Rcpp::stop("infer_ml2s_lr_test_satorra2000: both ML2S fits must carry $raw_data");
  }
  if (ud_method != "2000" && ud_method != "2001") {
    Rcpp::stop("infer_ml2s_lr_test_satorra2000: ud_method must be '2000' or '2001'");
  }
  const auto kind = magmaanr::two_stage_weight_from_arg(stage2_weight);
  magmaan::estimate::fiml::TwoStageDlsOptions dls;
  dls.a = dls_a;
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);

  magmaan::data::RawData raw_H1 =
      fiml_raw_from_fit_arg(ctx_H1.rep, fit_H1["raw_data"]);
  magmaan::data::RawData raw_H0 =
      fiml_raw_from_fit_arg(ctx_H0.rep, fit_H0["raw_data"]);
  validate_same_fiml_raw(raw_H1, raw_H0, "infer_ml2s_lr_test_satorra2000");

  // One saturated build for the whole nested test: H0 and H1 share it (same
  // data), and each ML2S fit already carries it as $stage1. Reuse it for the
  // df/T statistics here and pass it down so the difference-spectrum driver
  // does not rebuild the EM + observed information again.
  magmaan::estimate::fiml::SaturatedMoments sm;
  if (!magmaanr::saturated_from_stage1(fit_H1, sm) &&
      !magmaanr::saturated_from_stage1(fit_H0, sm)) {
    auto sm_or = magmaan::estimate::fiml::saturated_em_moments(raw_H1, h_step);
    if (!sm_or.has_value()) magmaanr::stop_post(sm_or.error());
    sm = std::move(*sm_or);
  }
  magmaan::data::SampleStats samp;
  samp.S = sm.cov;
  samp.mean = sm.mean;
  samp.n_obs = sm.n_obs;

  auto df_H1_or = magmaan::inference::df_stat(ctx_H1.pt, samp, est_H1.theta);
  if (!df_H1_or.has_value()) magmaanr::stop_post(df_H1_or.error());
  auto df_H0_or = magmaan::inference::df_stat(ctx_H0.pt, samp, est_H0.theta);
  if (!df_H0_or.has_value()) magmaanr::stop_post(df_H0_or.error());
  const double T_H1 = magmaan::inference::chi2_stat(samp, est_H1);
  const double T_H0 = magmaan::inference::chi2_stat(samp, est_H0);

  if (ud_method == "2001") {
    auto r_or = magmaan::robust::lr_test_satorra2001_ml2s_from_data(
        ctx_H1.pt, ctx_H1.rep, est_H1.theta,
        ctx_H0.pt, ctx_H0.rep, est_H0.theta,
        raw_H1, T_H0, T_H1, *df_H0_or, *df_H1_or, h_step, &sm, kind, dls);
    if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
    return satorra2000_to_list(*r_or);
  }

  auto K_H1_or = magmaan::estimate::build_eq_constraints(
      ctx_H1.pt, /*allow_nonlinear=*/true);
  if (!K_H1_or.has_value()) magmaanr::stop_post(K_H1_or.error());
  auto K_H0_or = magmaan::estimate::build_eq_constraints(
      ctx_H0.pt, /*allow_nonlinear=*/true);
  if (!K_H0_or.has_value()) magmaanr::stop_post(K_H0_or.error());

  auto r_or = magmaan::robust::lr_test_satorra2000_ml2s_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H1.theta, *K_H1_or,
      ctx_H0.pt, ctx_H0.rep, est_H0.theta, *K_H0_or,
      raw_H1, T_H0, T_H1, *df_H0_or, *df_H1_or,
      parse_gamma(gamma), parse_a_method(a_method), h_step, &sm, kind, dls,
      parse_moment_convention(convention));
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return satorra2000_to_list(*r_or);
}

// infer_ordinal_lr_test_satorra2000() — polychoric ordinal LS restriction-map
// nested test. The caller supplies the same OrdinalStats object used for the
// two fits, matching robust_ordinal()/fmg_tests_ordinal().
//
// [[Rcpp::export]]
Rcpp::List infer_ordinal_lr_test_satorra2000(Rcpp::List  fit_H1,
                                             Rcpp::List  fit_H0,
                                             Rcpp::List  ordinal_stats,
                                             double      T_H1,
                                             int         df_H1,
                                             double      T_H0,
                                             int         df_H0,
                                             std::string weight = "",
                                             std::string a_method = "exact") {
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);

  if (weight.empty()) {
    if (!fit_H1.containsElementNamed("estimator")) {
      Rcpp::stop("infer_ordinal_lr_test_satorra2000: `weight` is required when fit_H1$estimator is absent");
    }
    weight = Rcpp::as<std::string>(fit_H1["estimator"]);
  }
  const std::string p1 = fit_H1.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit_H1["parameterization"])
      : "delta";
  const std::string p0 = fit_H0.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit_H0["parameterization"])
      : "delta";
  if (p1 != p0) {
    Rcpp::stop("infer_ordinal_lr_test_satorra2000: H1/H0 parameterizations differ");
  }

  auto r_or = magmaan::estimate::lr_test_satorra2000_ordinal(
      ctx_H1.pt, ctx_H1.rep, stats, est_H1,
      ctx_H0.pt, ctx_H0.rep, est_H0,
      parse_ordinal_weight(weight),
      T_H0, T_H1, df_H0, df_H1,
      parse_a_method(a_method),
      magmaanr::ordinal_parameterization_from_string(p1));
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return satorra2000_to_list(*r_or);
}

// infer_mixed_ordinal_lr_test_satorra2000() — polyserial/polychoric mixed
// ordinal LS restriction-map nested test. The caller supplies the same
// MixedOrdinalStats object used for the two fits.
//
// [[Rcpp::export]]
Rcpp::List infer_mixed_ordinal_lr_test_satorra2000(Rcpp::List  fit_H1,
                                                   Rcpp::List  fit_H0,
                                                   Rcpp::List  mixed_stats,
                                                   double      T_H1,
                                                   int         df_H1,
                                                   double      T_H0,
                                                   int         df_H0,
                                                   std::string weight = "",
                                                   std::string a_method = "exact") {
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);
  magmaan::data::MixedOrdinalStats stats =
      magmaanr::mixed_ordinal_stats_from_arg(mixed_stats);

  if (weight.empty()) {
    if (!fit_H1.containsElementNamed("estimator")) {
      Rcpp::stop("infer_mixed_ordinal_lr_test_satorra2000: `weight` is required when fit_H1$estimator is absent");
    }
    weight = Rcpp::as<std::string>(fit_H1["estimator"]);
  }
  const std::string p1 = fit_H1.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit_H1["parameterization"])
      : "delta";
  const std::string p0 = fit_H0.containsElementNamed("parameterization")
      ? Rcpp::as<std::string>(fit_H0["parameterization"])
      : "delta";
  if (p1 != p0) {
    Rcpp::stop("infer_mixed_ordinal_lr_test_satorra2000: H1/H0 parameterizations differ");
  }

  auto r_or = magmaan::estimate::lr_test_satorra2000_mixed_ordinal(
      ctx_H1.pt, ctx_H1.rep, stats, est_H1,
      ctx_H0.pt, ctx_H0.rep, est_H0,
      parse_ordinal_weight(weight),
      T_H0, T_H1, df_H0, df_H1,
      parse_a_method(a_method),
      magmaanr::ordinal_parameterization_from_string(p1));
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return satorra2000_to_list(*r_or);
}

// FIML/ML2S scalar SB2001 / SB2010 difference tests (baseline columns; not the
// FMG-able spectrum). Reconstruct both missing-data fits, recover the FIML LRT
// (FIML) or two-stage ML (ML2S) chi-squares and df, then call the core scalar
// engines. SB2010 injects the H0 estimates into H1 for the M10 scale, so both
// fits must share the parameter layout (same-`==`-constrained nesting).
namespace {

struct FimlTD { double T_H0; double T_H1; int df_H0; int df_H1; };

FimlTD fiml_td(const magmaanr::Ctx& ctx_H1, const magmaan::estimate::Estimates& est_H1,
               const magmaan::data::RawData& raw_H1,
               const magmaanr::Ctx& ctx_H0, const magmaan::estimate::Estimates& est_H0,
               const magmaan::data::RawData& raw_H0, const char* who) {
  auto df_H1_or = magmaan::inference::df_stat(ctx_H1.pt, ctx_H1.samp, est_H1.theta);
  if (!df_H1_or.has_value()) magmaanr::stop_post(df_H1_or.error());
  auto df_H0_or = magmaan::inference::df_stat(ctx_H0.pt, ctx_H0.samp, est_H0.theta);
  if (!df_H0_or.has_value()) magmaanr::stop_post(df_H0_or.error());
  auto fx_H1_or = magmaan::estimate::fiml::fiml_extras(ctx_H1.pt, ctx_H1.rep, raw_H1, est_H1);
  if (!fx_H1_or.has_value()) magmaanr::stop_post(fx_H1_or.error());
  auto fx_H0_or = magmaan::estimate::fiml::fiml_extras(ctx_H0.pt, ctx_H0.rep, raw_H0, est_H0);
  if (!fx_H0_or.has_value()) magmaanr::stop_post(fx_H0_or.error());
  (void)who;
  return FimlTD{fx_H0_or->chi2, fx_H1_or->chi2, *df_H0_or, *df_H1_or};
}

FimlTD ml2s_td(const magmaanr::Ctx& ctx_H1, const magmaan::estimate::Estimates& est_H1,
               const magmaanr::Ctx& ctx_H0, const magmaan::estimate::Estimates& est_H0,
               const magmaan::data::RawData& raw_H1, double h_step) {
  auto sm_or = magmaan::estimate::fiml::saturated_em_moments(raw_H1, h_step);
  if (!sm_or.has_value()) magmaanr::stop_post(sm_or.error());
  magmaan::data::SampleStats samp;
  samp.S = sm_or->cov; samp.mean = sm_or->mean; samp.n_obs = sm_or->n_obs;
  auto df_H1_or = magmaan::inference::df_stat(ctx_H1.pt, samp, est_H1.theta);
  if (!df_H1_or.has_value()) magmaanr::stop_post(df_H1_or.error());
  auto df_H0_or = magmaan::inference::df_stat(ctx_H0.pt, samp, est_H0.theta);
  if (!df_H0_or.has_value()) magmaanr::stop_post(df_H0_or.error());
  return FimlTD{magmaan::inference::chi2_stat(samp, est_H0),
                magmaan::inference::chi2_stat(samp, est_H1), *df_H0_or, *df_H1_or};
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List infer_fiml_lr_test_satorra_bentler2001(Rcpp::List fit_H1, Rcpp::List fit_H0,
                                                  double h_step = 1e-4) {
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);
  magmaan::data::RawData raw_H1 = fiml_raw_from_fit_arg(ctx_H1.rep, fit_H1["raw_data"]);
  magmaan::data::RawData raw_H0 = fiml_raw_from_fit_arg(ctx_H0.rep, fit_H0["raw_data"]);
  validate_same_fiml_raw(raw_H1, raw_H0, "infer_fiml_lr_test_satorra_bentler2001");
  FimlTD td = fiml_td(ctx_H1, est_H1, raw_H1, ctx_H0, est_H0, raw_H0, "fiml_sb2001");
  auto r_or = magmaan::robust::lr_test_satorra_bentler2001_fiml_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H1.theta, ctx_H0.pt, ctx_H0.rep, est_H0.theta,
      raw_H1, td.T_H0, td.T_H1, td.df_H0, td.df_H1, h_step);
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return sb_diff_to_list(*r_or);
}

// [[Rcpp::export]]
Rcpp::List infer_fiml_lr_test_satorra_bentler2010(Rcpp::List fit_H1, Rcpp::List fit_H0,
                                                  double h_step = 1e-4) {
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);
  magmaan::data::RawData raw_H1 = fiml_raw_from_fit_arg(ctx_H1.rep, fit_H1["raw_data"]);
  magmaan::data::RawData raw_H0 = fiml_raw_from_fit_arg(ctx_H0.rep, fit_H0["raw_data"]);
  validate_same_fiml_raw(raw_H1, raw_H0, "infer_fiml_lr_test_satorra_bentler2010");
  FimlTD td = fiml_td(ctx_H1, est_H1, raw_H1, ctx_H0, est_H0, raw_H0, "fiml_sb2010");
  auto r_or = magmaan::robust::lr_test_satorra_bentler2010_fiml_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H0.theta, ctx_H0.pt, ctx_H0.rep, est_H0.theta,
      raw_H1, td.T_H0, td.T_H1, td.df_H0, td.df_H1, h_step);
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return sb_diff_to_list(*r_or);
}

// [[Rcpp::export]]
Rcpp::List infer_ml2s_lr_test_satorra_bentler2001(Rcpp::List fit_H1, Rcpp::List fit_H0,
                                                  double h_step = 1e-4) {
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);
  magmaan::data::RawData raw_H1 = fiml_raw_from_fit_arg(ctx_H1.rep, fit_H1["raw_data"]);
  magmaan::data::RawData raw_H0 = fiml_raw_from_fit_arg(ctx_H0.rep, fit_H0["raw_data"]);
  validate_same_fiml_raw(raw_H1, raw_H0, "infer_ml2s_lr_test_satorra_bentler2001");
  FimlTD td = ml2s_td(ctx_H1, est_H1, ctx_H0, est_H0, raw_H1, h_step);
  auto r_or = magmaan::robust::lr_test_satorra_bentler2001_ml2s_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H1.theta, ctx_H0.pt, ctx_H0.rep, est_H0.theta,
      raw_H1, td.T_H0, td.T_H1, td.df_H0, td.df_H1, h_step);
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return sb_diff_to_list(*r_or);
}

// [[Rcpp::export]]
Rcpp::List infer_ml2s_lr_test_satorra_bentler2010(Rcpp::List fit_H1, Rcpp::List fit_H0,
                                                  double h_step = 1e-4) {
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);
  magmaan::data::RawData raw_H1 = fiml_raw_from_fit_arg(ctx_H1.rep, fit_H1["raw_data"]);
  magmaan::data::RawData raw_H0 = fiml_raw_from_fit_arg(ctx_H0.rep, fit_H0["raw_data"]);
  validate_same_fiml_raw(raw_H1, raw_H0, "infer_ml2s_lr_test_satorra_bentler2010");
  FimlTD td = ml2s_td(ctx_H1, est_H1, ctx_H0, est_H0, raw_H1, h_step);
  auto r_or = magmaan::robust::lr_test_satorra_bentler2010_ml2s_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H0.theta, ctx_H0.pt, ctx_H0.rep, est_H0.theta,
      raw_H1, td.T_H0, td.T_H1, td.df_H0, td.df_H1, h_step);
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return sb_diff_to_list(*r_or);
}

// infer_lr_test_satorra_bentler2001() — lavaan-compatible SB2001 scaled
// nested-model LR approximation.
//
// [[Rcpp::export]]
Rcpp::List infer_lr_test_satorra_bentler2001(Rcpp::List  fit_H1,
                                             Rcpp::List  fit_H0,
                                             Rcpp::List  X_per_group,
                                             double      T_H1,
                                             int         df_H1,
                                             double      T_H0,
                                             int         df_H0,
                                             std::string gamma = "empirical") {
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);
  magmaan::data::RawData raw = raw_from_group_list(
      X_per_group, ctx_H1.samp.S.size(), "infer_lr_test_satorra_bentler2001");

  auto r_or = magmaan::robust::lr_test_satorra_bentler2001_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H1.theta,
      ctx_H0.pt, ctx_H0.rep, est_H0.theta,
      raw, T_H0, T_H1, df_H0, df_H1, parse_gamma(gamma));
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return sb_diff_to_list(*r_or);
}

// infer_lr_test_satorra_bentler2010() — lavaan-compatible SB2010 scaled
// nested-model LR approximation.
//
// [[Rcpp::export]]
Rcpp::List infer_lr_test_satorra_bentler2010(Rcpp::List  fit_H1,
                                             Rcpp::List  fit_H0,
                                             Rcpp::List  X_per_group,
                                             double      T_H1,
                                             int         df_H1,
                                             double      T_H0,
                                             int         df_H0,
                                             std::string gamma = "empirical") {
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  magmaanr::Ctx ctx_H0 = magmaanr::ctx_from_fit(fit_H0);
  const magmaan::estimate::Estimates est_H0 = magmaanr::est_from_fit(fit_H0);
  magmaan::data::RawData raw = raw_from_group_list(
      X_per_group, ctx_H1.samp.S.size(), "infer_lr_test_satorra_bentler2010");

  auto r_or = magmaan::robust::lr_test_satorra_bentler2010_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H0.theta,
      ctx_H0.pt, ctx_H0.rep, est_H0.theta,
      raw, T_H0, T_H1, df_H0, df_H1, parse_gamma(gamma));
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  return sb_diff_to_list(*r_or);
}
