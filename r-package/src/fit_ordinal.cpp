#include "glue_internal.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;
using namespace magmaanr::fitglue;

namespace {

std::string ordinal_stage2_label(const std::string& s);
magmaan::data::PolychoricHScoreKind h_score_kind_from(std::string kind);
magmaan::data::PairwiseOrdinalHWeightedStatsOptions
h_weighted_options_from(std::string h_kind, double k, double a, double b,
                        double lambda);
magmaan::data::PairwiseOrdinalDpdStatsOptions
ordinal_dpd_options_from(double alpha);
magmaan::data::PolyserialPairDpdOptions
polyserial_dpd_options_from(double alpha);
magmaan::data::MixedOrdinalHuberResidualOptions
huber_residual_options_from(std::string clip, double k);
magmaan::data::PairwiseOrdinalHuberResidualStatsOptions
ordinal_huber_residual_options_from(std::string clip, double k);
Rcpp::List pairwise_ordinal_diagnostics_to_r(
    const std::vector<magmaan::data::PairwiseOrdinalBlockDiagnostics>& d);
Rcpp::List mixed_polyserial_dpd_diagnostics_to_r(
    const std::vector<magmaan::data::MixedOrdinalPolyserialDpdBlockDiagnostics>& d);
Rcpp::List mixed_huber_residual_diagnostics_to_r(
    const std::vector<magmaan::data::MixedOrdinalHuberResidualBlockDiagnostics>& d);
std::vector<Eigen::MatrixXd> matrix_blocks_from_arg(SEXP X);
std::vector<std::vector<std::int32_t>>
n_levels_from_arg(Rcpp::List n_levels);
Ctx pairwise_ordinal_ctx_from_partable(SEXP partable,
                                       const char* caller,
                                       const magmaan::data::OrdinalStats& stats,
                                       magmaan::spec::Starts& starts);
Rcpp::List pairwise_objective_to_r(
    const magmaan::estimate::frontier::PairwiseOrdinalCompositeResult& obj);
Rcpp::List pairwise_godambe_to_r(
    const magmaan::estimate::frontier::PairwiseOrdinalCompositeGodambe& g);
Rcpp::List pairwise_lr_to_r(const magmaan::robust::LRSatorra2000Result& r);
magmaan::data::frontier::CovarianceShrinkageKind
shrinkage_kind_from_string(const std::string& kind);
magmaan::data::OrdinalPairwiseGammaKind ordinal_pairwise_gamma_from_string(
    const std::string& gamma);
std::vector<std::vector<std::int32_t>>
mixed_ordered_mask_from_arg(SEXP ordered_mask, std::size_t n_blocks);

std::string ordinal_stage2_label(const std::string& s) {
  std::string key = s;
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  if (key == "gls") return "GLS";
  if (key == "adf") return "ADF";
  auto kind = ordinal_stage2_weight_from_string(s);
  switch (kind) {
    case magmaan::estimate::frontier::OrdinalStage2Weight::Uls:
      return "ULS";
    case magmaan::estimate::frontier::OrdinalStage2Weight::Dwls:
      return "DWLS";
    case magmaan::estimate::frontier::OrdinalStage2Weight::Wls:
      return "WLS";
    case magmaan::estimate::frontier::OrdinalStage2Weight::Nt:
      return "NT";
    case magmaan::estimate::frontier::OrdinalStage2Weight::Dls:
      return "DLS";
  }
  return "UNKNOWN";
}

magmaan::data::PolychoricHScoreKind h_score_kind_from(std::string kind) {
  if (kind == "ml") return magmaan::data::PolychoricHScoreKind::ML;
  if (kind == "wma_hard_cap") return magmaan::data::PolychoricHScoreKind::WmaHardCap;
  if (kind == "smooth_cap") return magmaan::data::PolychoricHScoreKind::SmoothCap;
  if (kind == "exp_cap") return magmaan::data::PolychoricHScoreKind::ExpCap;
  Rcpp::stop("magmaan: h_kind must be one of 'ml', 'wma_hard_cap', 'smooth_cap', or 'exp_cap'");
}

magmaan::data::PairwiseOrdinalHWeightedStatsOptions
h_weighted_options_from(std::string h_kind, double k, double a, double b,
                        double lambda) {
  magmaan::data::PairwiseOrdinalHWeightedStatsOptions options;
  options.rho.h_score.kind = h_score_kind_from(h_kind);
  options.rho.h_score.k = k;
  options.rho.h_score.a = a;
  options.rho.h_score.b = b;
  options.rho.h_score.lambda = lambda;
  return options;
}

magmaan::data::PairwiseOrdinalDpdStatsOptions
ordinal_dpd_options_from(double alpha) {
  magmaan::data::PairwiseOrdinalDpdStatsOptions options;
  options.alpha = alpha;
  return options;
}

magmaan::data::PolyserialPairDpdOptions
polyserial_dpd_options_from(double alpha) {
  magmaan::data::PolyserialPairDpdOptions options;
  options.alpha = alpha;
  return options;
}

magmaan::data::MixedOrdinalHuberResidualOptions
huber_residual_options_from(std::string clip, double k) {
  magmaan::data::MixedOrdinalHuberResidualOptions options;
  if (clip == "none") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::None;
  } else if (clip == "hard_huber") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::HardHuber;
  } else if (clip == "pseudo_huber") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::PseudoHuber;
  } else if (clip == "tukey_biweight") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::TukeyBiweight;
  } else {
    Rcpp::stop("magmaan: unknown Huber residual clip kind '%s'", clip);
  }
  options.clip.k = k;
  return options;
}

magmaan::data::PairwiseOrdinalHuberResidualStatsOptions
ordinal_huber_residual_options_from(std::string clip, double k) {
  magmaan::data::PairwiseOrdinalHuberResidualStatsOptions options;
  if (clip == "none") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::None;
  } else if (clip == "hard_huber") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::HardHuber;
  } else if (clip == "pseudo_huber") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::PseudoHuber;
  } else if (clip == "tukey_biweight") {
    options.clip.kind = magmaan::data::HuberResidualClipKind::TukeyBiweight;
  } else {
    Rcpp::stop("magmaan: unknown Huber residual clip kind '%s'", clip);
  }
  options.clip.k = k;
  return options;
}

Rcpp::List pairwise_ordinal_diagnostics_to_r(
    const std::vector<magmaan::data::PairwiseOrdinalBlockDiagnostics>& d) {
  Rcpp::List out(static_cast<R_xlen_t>(d.size()));
  for (std::size_t b = 0; b < d.size(); ++b) {
    out[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["moment_influence"] = Rcpp::wrap(d[b].moment_influence),
        Rcpp::_["gamma"] = Rcpp::wrap(d[b].gamma),
        Rcpp::_["min_eigen_r"] = d[b].min_eigen_r,
        Rcpp::_["raw_min_eigen_r"] = d[b].raw_min_eigen_r,
        Rcpp::_["r_repair_applied"] = d[b].r_repair_applied);
  }
  return out;
}

Rcpp::List mixed_polyserial_dpd_diagnostics_to_r(
    const std::vector<magmaan::data::MixedOrdinalPolyserialDpdBlockDiagnostics>& d) {
  Rcpp::List out(static_cast<R_xlen_t>(d.size()));
  for (std::size_t b = 0; b < d.size(); ++b) {
    const auto& block = d[b];
    Rcpp::DataFrame pairs;
    {
      const R_xlen_t n = static_cast<R_xlen_t>(block.dpd_pairs.size());
      Rcpp::IntegerVector i(n), j(n), moment_index(n);
      for (R_xlen_t r = 0; r < n; ++r) {
        const auto& pair = block.dpd_pairs[static_cast<std::size_t>(r)];
        i[r] = pair.i + 1;
        j[r] = pair.j + 1;
        moment_index[r] = pair.moment_index + 1;
      }
      pairs = Rcpp::DataFrame::create(Rcpp::_["i"] = i,
                                      Rcpp::_["j"] = j,
                                      Rcpp::_["moment_index"] = moment_index,
                                      Rcpp::_["stringsAsFactors"] = false);
    }
    Rcpp::NumericVector rho(static_cast<R_xlen_t>(block.dpd_fits.size()));
    Rcpp::NumericVector objective(static_cast<R_xlen_t>(block.dpd_fits.size()));
    for (R_xlen_t r = 0; r < rho.size(); ++r) {
      const auto& fit = block.dpd_fits[static_cast<std::size_t>(r)];
      rho[r] = fit.rho;
      objective[r] = fit.objective;
    }
    out[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["pairs"] = pairs,
        Rcpp::_["rho"] = rho,
        Rcpp::_["objective"] = objective,
        Rcpp::_["moment_influence"] = Rcpp::wrap(block.moment_influence),
        Rcpp::_["gamma"] = Rcpp::wrap(block.gamma));
  }
  return out;
}

Rcpp::List mixed_huber_residual_diagnostics_to_r(
    const std::vector<magmaan::data::MixedOrdinalHuberResidualBlockDiagnostics>& d) {
  Rcpp::List out(static_cast<R_xlen_t>(d.size()));
  for (std::size_t b = 0; b < d.size(); ++b) {
    const auto& block = d[b];
    const R_xlen_t n = static_cast<R_xlen_t>(block.robust_pairs.size());
    Rcpp::IntegerVector i(n), j(n), moment_index(n);
    Rcpp::CharacterVector kind(n);
    for (R_xlen_t r = 0; r < n; ++r) {
      const auto& pair = block.robust_pairs[static_cast<std::size_t>(r)];
      i[r] = pair.i + 1;
      j[r] = pair.j + 1;
      moment_index[r] = pair.moment_index + 1;
      kind[r] = pair.kind == magmaan::data::MixedPairKind::ordinal_ordinal
          ? "ordinal_ordinal"
          : (pair.kind == magmaan::data::MixedPairKind::continuous_ordinal
                 ? "continuous_ordinal"
                 : "continuous_continuous");
    }
    Rcpp::DataFrame pairs = Rcpp::DataFrame::create(
        Rcpp::_["i"] = i,
        Rcpp::_["j"] = j,
        Rcpp::_["moment_index"] = moment_index,
        Rcpp::_["kind"] = kind,
        Rcpp::_["stringsAsFactors"] = false);
    out[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["pairs"] = pairs,
        Rcpp::_["rho"] = Rcpp::wrap(block.rho),
        Rcpp::_["objective"] = Rcpp::wrap(block.objective),
        Rcpp::_["moment_influence"] = Rcpp::wrap(block.moment_influence),
        Rcpp::_["gamma"] = Rcpp::wrap(block.gamma),
        Rcpp::_["min_eigen_r"] = block.min_eigen_r,
        Rcpp::_["raw_min_eigen_r"] = block.raw_min_eigen_r,
        Rcpp::_["r_repair_applied"] = block.r_repair_applied);
  }
  return out;
}

std::vector<Eigen::MatrixXd> matrix_blocks_from_arg(SEXP X) {
  std::vector<Eigen::MatrixXd> blocks;
  if (Rf_isMatrix(X)) {
    blocks.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(X)));
    return blocks;
  }
  if (TYPEOF(X) != VECSXP) {
    Rcpp::stop("magmaan: X must be a matrix or list of matrices");
  }
  Rcpp::List xl(X);
  blocks.reserve(static_cast<std::size_t>(xl.size()));
  for (R_xlen_t b = 0; b < xl.size(); ++b) {
    blocks.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(xl[b])));
  }
  return blocks;
}

std::vector<std::vector<std::int32_t>>
n_levels_from_arg(Rcpp::List n_levels) {
  std::vector<std::vector<std::int32_t>> out;
  out.reserve(static_cast<std::size_t>(n_levels.size()));
  for (R_xlen_t b = 0; b < n_levels.size(); ++b) {
    Rcpp::IntegerVector lev(n_levels[b]);
    out.emplace_back(Rcpp::as<std::vector<std::int32_t>>(lev));
  }
  return out;
}

Ctx pairwise_ordinal_ctx_from_partable(SEXP partable,
                                       const char* caller,
                                       const magmaan::data::OrdinalStats& stats,
                                       magmaan::spec::Starts& starts) {
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, caller);
  starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty()
      ? std::vector<std::string>{}
      : ctx.rep.ov_names[0];
  ctx.meanstructure = false;
  return ctx;
}

Rcpp::List pairwise_objective_to_r(
    const magmaan::estimate::frontier::PairwiseOrdinalCompositeResult& obj) {
  int n_pairs = 0;
  std::int64_t n_obs = 0;
  for (const auto& block : obj.blocks) {
    n_pairs += static_cast<int>(block.pairs.size());
    n_obs += block.n_obs;
  }
  return Rcpp::List::create(
      Rcpp::_["negloglik"] = obj.negloglik,
      Rcpp::_["weighted_negloglik"] = obj.weighted_negloglik,
      Rcpp::_["n_pairs"] = n_pairs,
      Rcpp::_["nobs_total"] = static_cast<double>(n_obs),
      Rcpp::_["df"] = obj.df);
}

Rcpp::List pairwise_godambe_to_r(
    const magmaan::estimate::frontier::PairwiseOrdinalCompositeGodambe& g) {
  return Rcpp::List::create(
      Rcpp::_["bread"] = Rcpp::wrap(g.bread),
      Rcpp::_["meat"] = Rcpp::wrap(g.meat),
      Rcpp::_["vcov"] = Rcpp::wrap(g.vcov),
      Rcpp::_["vcov_naive"] = Rcpp::wrap(g.vcov_naive),
      Rcpp::_["se"] = Rcpp::wrap(g.se),
      Rcpp::_["se_naive"] = Rcpp::wrap(g.se_naive),
      Rcpp::_["casewise_scores"] = Rcpp::wrap(g.casewise_scores),
      Rcpp::_["condition_bread"] = g.condition_bread);
}

Rcpp::List pairwise_lr_to_r(const magmaan::robust::LRSatorra2000Result& r) {
  Rcpp::CharacterVector warns(static_cast<R_xlen_t>(r.warnings.size()));
  for (R_xlen_t i = 0; i < warns.size(); ++i) {
    warns[i] = r.warnings[static_cast<std::size_t>(i)];
  }
  return Rcpp::List::create(
      Rcpp::_["T_diff"] = r.T_diff,
      Rcpp::_["df_diff"] = r.df_diff,
      Rcpp::_["p_unscaled"] = r.p_unscaled,
      Rcpp::_["eigenvalues"] = Rcpp::wrap(r.eigenvalues),
      Rcpp::_["scale_c"] = r.scale_c,
      Rcpp::_["T_scaled"] = r.T_scaled,
      Rcpp::_["p_scaled"] = r.p_scaled,
      Rcpp::_["adjust_d0"] = r.adjust_d0,
      Rcpp::_["T_adjusted"] = r.T_adjusted,
      Rcpp::_["p_adjusted"] = r.p_adjusted,
      Rcpp::_["p_mixture"] = r.p_mixture,
      Rcpp::_["warnings"] = warns);
}

magmaan::data::frontier::CovarianceShrinkageKind
shrinkage_kind_from_string(const std::string& kind) {
  using K = magmaan::data::frontier::CovarianceShrinkageKind;
  if (kind == "none") return K::None;
  if (kind == "ridge") return K::Ridge;
  if (kind == "identity") return K::IdentityTarget;
  if (kind == "diagonal") return K::DiagonalTarget;
  if (kind == "constant_correlation") return K::ConstantCorrelation;
  Rcpp::stop("magmaan: shrinkage kind must be none, ridge, identity, diagonal, or constant_correlation");
}

magmaan::data::OrdinalPairwiseGammaKind ordinal_pairwise_gamma_from_string(
    const std::string& gamma) {
  if (gamma == "overlap") return magmaan::data::OrdinalPairwiseGammaKind::Overlap;
  if (gamma == "nominal") return magmaan::data::OrdinalPairwiseGammaKind::Nominal;
  Rcpp::stop("magmaan: pd_gamma must be \"overlap\" or \"nominal\" (got \"%s\")",
             gamma);
}

std::vector<std::vector<std::int32_t>>
mixed_ordered_mask_from_arg(SEXP ordered_mask, std::size_t n_blocks) {
  std::vector<std::vector<std::int32_t>> ordered;
  ordered.reserve(n_blocks);
  if (Rf_isMatrix(ordered_mask)) {
    Rcpp::IntegerMatrix M(ordered_mask);
    if (n_blocks != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    std::vector<std::int32_t> row(static_cast<std::size_t>(M.ncol()));
    for (R_xlen_t j = 0; j < M.ncol(); ++j) row[static_cast<std::size_t>(j)] = M(0, j);
    ordered.push_back(std::move(row));
  } else if (TYPEOF(ordered_mask) == VECSXP) {
    Rcpp::List L(ordered_mask);
    if (static_cast<std::size_t>(L.size()) != n_blocks)
      Rcpp::stop("magmaan: ordered_mask block count does not match X");
    for (R_xlen_t b = 0; b < L.size(); ++b) {
      Rcpp::IntegerVector v(L[b]);
      ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
    }
  } else {
    Rcpp::IntegerVector v(ordered_mask);
    if (n_blocks != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
  }
  return ordered;
}

}  // namespace

// Select fitting weights while retaining ordinal moments and sampling Gamma.
// [[Rcpp::export]]
Rcpp::List ordinal_fixed_weight_stats_impl(Rcpp::List ordinal_stats,
                                           std::string method, SEXP W = R_NilValue,
                                           double dls_a = 0.5) {
  auto stats = ordinal_stats_from_arg(ordinal_stats);
  if (Rf_isNull(W)) {
    auto selected = magmaan::estimate::frontier::ordinal_stats_with_stage2_weight(stats,
        ordinal_stage2_weight_from_string(method), {.a = dls_a});
    if (!selected) stop_post(selected.error());
    stats = std::move(*selected);
  } else if (method == "DWLS") stats.W_dwls = wls_dense_from_arg(W, stats.R.size());
  else if (method == "WLS") stats.W_wls = wls_dense_from_arg(W, stats.R.size());
  else Rcpp::stop("magmaan: supplied W requires WLS or DWLS");
  // Preserve the exact R Stage-1 object; only the consumed weight slot changes.
  Rcpp::List out = Rcpp::clone(ordinal_stats);
  if (!Rf_isNull(W) && method == "DWLS") out["W_dwls"] = Rcpp::wrap(stats.W_dwls);
  else out["W_wls"] = Rcpp::wrap(stats.W_wls);
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_from_raw_impl(SEXP X, bool full_wls_weight = true) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::ordinal_stats_from_integer_data(blocks,
                                                               full_wls_weight);
  if (!out_or.has_value()) stop_post(out_or.error());
  return ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_observed_from_raw_impl(
    SEXP X, std::string pd_gamma = "overlap", bool full_wls_weight = true) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::ordinal_stats_from_observed_integer_data(
      blocks, ordinal_pairwise_gamma_from_string(pd_gamma), full_wls_weight);
  if (!out_or.has_value()) stop_post(out_or.error());
  return ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List ordinal_stage2_weight_blocks_impl(Rcpp::List ordinal_stats,
                                             std::string stage2_weight = "dwls",
                                             double dls_a = 0.5) {
  auto stats = ordinal_stats_from_arg(ordinal_stats);
  magmaan::estimate::frontier::OrdinalStage2DlsOptions dls;
  dls.a = dls_a;
  auto W_or = magmaan::estimate::frontier::ordinal_stage2_weight_blocks(
      stats, ordinal_stage2_weight_from_string(stage2_weight), dls);
  if (!W_or.has_value()) stop_post(W_or.error());
  Rcpp::List out(static_cast<R_xlen_t>(W_or->size()));
  for (R_xlen_t b = 0; b < out.size(); ++b) {
    out[b] = Rcpp::wrap((*W_or)[static_cast<std::size_t>(b)]);
  }
  out.attr("stage2_weight") = ordinal_stage2_label(stage2_weight);
  out.attr("dls_a") = dls_a;
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_h_weighted_from_raw_impl(
    SEXP X, std::string h_kind = "wma_hard_cap",
    double k = 1.5, double a = 1.6, double b = 2.2, double lambda = 0.2) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      blocks, h_weighted_options_from(h_kind, k, a, b, lambda));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "h_weighted";
  out["diagnostics"] = pairwise_ordinal_diagnostics_to_r(out_or->block_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_dpd_from_raw_impl(SEXP X, double alpha = 0.3) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::pairwise_ordinal_stats_dpd_from_integer_data(
      blocks, ordinal_dpd_options_from(alpha));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "dpd";
  out["alpha"] = alpha;
  out["diagnostics"] = pairwise_ordinal_diagnostics_to_r(out_or->block_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_ordinal_stats_huber_residual_from_raw_impl(
    SEXP X, std::string clip = "hard_huber", double k = 1.345) {
  auto blocks = matrix_blocks_from_arg(X);
  auto out_or = magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
      blocks, ordinal_huber_residual_options_from(clip, k));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "huber_residual";
  out["clip"] = clip;
  out["k"] = k;
  out["diagnostics"] = pairwise_ordinal_diagnostics_to_r(out_or->block_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_from_raw_impl(SEXP X, SEXP ordered_mask,
                                                  bool full_wls_weight = true) {
  auto blocks = matrix_blocks_from_arg(X);
  auto ordered = mixed_ordered_mask_from_arg(ordered_mask, blocks.size());
  auto out_or = magmaan::data::mixed_ordinal_stats_from_data(blocks, ordered,
                                                            full_wls_weight);
  if (!out_or.has_value()) stop_post(out_or.error());
  return mixed_ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_observed_from_raw_impl(
    SEXP X, SEXP ordered_mask, bool full_wls_weight = true) {
  auto blocks = matrix_blocks_from_arg(X);
  auto ordered = mixed_ordered_mask_from_arg(ordered_mask, blocks.size());
  auto out_or = magmaan::data::mixed_ordinal_stats_from_observed_data(
      blocks, ordered, full_wls_weight);
  if (!out_or.has_value()) stop_post(out_or.error());
  return mixed_ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_hybrid_fiml_from_raw_impl(
    SEXP X, SEXP ordered_mask, bool full_wls_weight = true,
    double h_step = 1e-4) {
  auto blocks = matrix_blocks_from_arg(X);
  auto ordered = mixed_ordered_mask_from_arg(ordered_mask, blocks.size());
  auto out_or =
      magmaan::estimate::fiml::mixed_ordinal_stats_hybrid_fiml_from_observed_data(
          blocks, ordered, full_wls_weight, h_step);
  if (!out_or.has_value()) stop_post(out_or.error());
  return mixed_ordinal_stats_to_r(*out_or);
}

// [[Rcpp::export]]
Rcpp::List data_shrink_mixed_ordinal_stats_impl(
    Rcpp::List mixed_stats, std::string kind = "diagonal",
    double intensity = 0.0, bool estimate_intensity = false) {
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(mixed_stats);
  magmaan::data::frontier::CovarianceShrinkageOptions opts;
  opts.kind = shrinkage_kind_from_string(kind);
  opts.intensity = intensity;
  opts.estimate_intensity = estimate_intensity;
  auto out_or = magmaan::data::frontier::shrink_mixed_ordinal_stats(stats, opts);
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = mixed_ordinal_stats_to_r(out_or->stats);
  Rcpp::List diag(static_cast<R_xlen_t>(out_or->block_diagnostics.size()));
  for (std::size_t b = 0; b < out_or->block_diagnostics.size(); ++b) {
    const auto& d = out_or->block_diagnostics[b];
    diag[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["raw_min_eigen"] = d.raw_min_eigen,
        Rcpp::_["min_eigen"] = d.min_eigen,
        Rcpp::_["intensity"] = d.intensity,
        Rcpp::_["shrunk"] = d.shrunk);
  }
  out["shrinkage"] = diag;
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_polyserial_dpd_from_raw_impl(
    SEXP X, SEXP ordered_mask, double alpha = 0.3) {
  auto blocks = matrix_blocks_from_arg(X);
  std::vector<std::vector<std::int32_t>> ordered;
  ordered.reserve(blocks.size());
  if (Rf_isMatrix(ordered_mask)) {
    Rcpp::IntegerMatrix M(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    std::vector<std::int32_t> row(static_cast<std::size_t>(M.ncol()));
    for (R_xlen_t j = 0; j < M.ncol(); ++j) row[static_cast<std::size_t>(j)] = M(0, j);
    ordered.push_back(std::move(row));
  } else if (TYPEOF(ordered_mask) == VECSXP) {
    Rcpp::List L(ordered_mask);
    if (static_cast<std::size_t>(L.size()) != blocks.size())
      Rcpp::stop("magmaan: ordered_mask block count does not match X");
    for (R_xlen_t b = 0; b < L.size(); ++b) {
      Rcpp::IntegerVector v(L[b]);
      ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
    }
  } else {
    Rcpp::IntegerVector v(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
  }
  auto out_or = magmaan::data::mixed_ordinal_stats_polyserial_dpd_from_data(
      blocks, ordered, polyserial_dpd_options_from(alpha));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = mixed_ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "polyserial_dpd";
  out["alpha"] = alpha;
  out["diagnostics"] = mixed_polyserial_dpd_diagnostics_to_r(
      out_or->block_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List data_mixed_ordinal_stats_huber_residual_from_raw_impl(
    SEXP X, SEXP ordered_mask, std::string clip = "hard_huber",
    double k = 1.345) {
  auto blocks = matrix_blocks_from_arg(X);
  std::vector<std::vector<std::int32_t>> ordered;
  ordered.reserve(blocks.size());
  if (Rf_isMatrix(ordered_mask)) {
    Rcpp::IntegerMatrix M(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    std::vector<std::int32_t> row(static_cast<std::size_t>(M.ncol()));
    for (R_xlen_t j = 0; j < M.ncol(); ++j) row[static_cast<std::size_t>(j)] = M(0, j);
    ordered.push_back(std::move(row));
  } else if (TYPEOF(ordered_mask) == VECSXP) {
    Rcpp::List L(ordered_mask);
    if (static_cast<std::size_t>(L.size()) != blocks.size())
      Rcpp::stop("magmaan: ordered_mask block count does not match X");
    for (R_xlen_t b = 0; b < L.size(); ++b) {
      Rcpp::IntegerVector v(L[b]);
      ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
    }
  } else {
    Rcpp::IntegerVector v(ordered_mask);
    if (blocks.size() != 1)
      Rcpp::stop("magmaan: ordered_mask must be a list for multi-group data");
    ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(v));
  }
  auto out_or = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      blocks, ordered, huber_residual_options_from(clip, k));
  if (!out_or.has_value()) stop_post(out_or.error());
  Rcpp::List out = mixed_ordinal_stats_to_r(out_or->stats);
  out["robust_method"] = "huber_residual";
  out["clip"] = clip;
  out["k"] = k;
  out["diagnostics"] = mixed_huber_residual_diagnostics_to_r(
      out_or->block_diagnostics);
  return out;
}

// [[Rcpp::export]]
Rcpp::List fit_dwls_ordinal_impl(SEXP partable, Rcpp::List ordinal_stats,
                                 Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                 Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_dwls_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  // Re-attach the group.equal families stamped by lavaan_lavaanify so the
  // ordinal prep applies the fit-time Wu-Estabrook release (lost on round-trip).
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  auto prep_or = magmaan::estimate::prepare_ordinal_delta_partable(ctx.pt, stats, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{} : ctx.rep.ov_names[0];
  ctx.meanstructure = false;
  const Eigen::VectorXd x0 = ordinal_starts_or_stop(ctx, stats, starts);
  auto e_or = magmaan::estimate::fit_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::DWLS, x0,
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "DWLS",
                            parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_uls_ordinal_impl(SEXP partable, Rcpp::List ordinal_stats,
                                Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_uls_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  // Re-attach the group.equal families stamped by lavaan_lavaanify so the
  // ordinal prep applies the fit-time Wu-Estabrook release (lost on round-trip).
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  auto prep_or = magmaan::estimate::prepare_ordinal_delta_partable(ctx.pt, stats, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{} : ctx.rep.ov_names[0];
  ctx.meanstructure = false;
  const Eigen::VectorXd x0 = ordinal_starts_or_stop(ctx, stats, starts);
  auto e_or = magmaan::estimate::fit_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::ULS, x0,
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "ULS",
                            parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_wls_ordinal_impl(SEXP partable, Rcpp::List ordinal_stats,
                                Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  // Re-attach the group.equal families stamped by lavaan_lavaanify so the
  // ordinal prep applies the fit-time Wu-Estabrook release (lost on round-trip).
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  auto prep_or = magmaan::estimate::prepare_ordinal_delta_partable(ctx.pt, stats, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{} : ctx.rep.ov_names[0];
  ctx.meanstructure = false;
  const Eigen::VectorXd x0 = ordinal_starts_or_stop(ctx, stats, starts);
  auto e_or = magmaan::estimate::fit_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::WLS, x0,
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return ordinal_fit_result(ctx, stats, est, &starts, "WLS",
                            parameterization_name.c_str());
}

// All-ordinal ULS/DWLS/WLS with PSD primitive LISREL covariance matrices.
// The polychoric matrix and fixed stage-2 weights are passed through unchanged.
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_ordinal_psd_impl(
    SEXP partable, Rcpp::List ordinal_stats,
    std::string estimator = "DWLS",
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue,
    double start_eigen_floor = 1e-6,
    double feasibility_tol = 1e-6) {
  const std::string parameterization_name =
      ordinal_parameterization_attr(partable);
  const auto parameterization =
      ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "frontier_fit_ordinal_psd");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  auto prep_or = magmaan::estimate::prepare_ordinal_partable(
      ctx.pt, stats, parameterization, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty()
      ? std::vector<std::string>{}
      : ctx.rep.ov_names[0];
  ctx.meanstructure = false;
  const Eigen::VectorXd x0 = ordinal_starts_or_stop(ctx, stats, starts);
  const magmaan::estimate::Backend backend =
      optimizer.isNull()
          ? magmaan::estimate::Backend::NloptSlsqp
          : backend_from_optimizer_arg(optimizer);
  magmaan::estimate::frontier::PsdFitOptions psd_opts;
  psd_opts.start_eigen_floor = start_eigen_floor;
  psd_opts.feasibility_tol = feasibility_tol;
  auto fit_or = magmaan::estimate::frontier::fit_ordinal_psd(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      ordinal_weight_from_estimator(estimator, "frontier_fit_ordinal_psd"),
      x0, backend, optim_opts_from(control), parameterization, psd_opts);
  if (!fit_or.has_value()) stop_fit(fit_or.error());
  const magmaan::estimate::Estimates est = std::move(*fit_or);
  Rcpp::List out = ordinal_fit_result(
      ctx, stats, est, &starts, estimator.c_str(),
      parameterization_name.c_str());
  out["covariance_policy"] = "psd";
  return out;
}

// All-ordinal ML/LS plus the same model barrier, preserving Stage-1 objects.
// [[Rcpp::export]]
Rcpp::List fit_ordinal_barrier_impl(
    SEXP partable, Rcpp::List ordinal_stats, std::string estimator = "ML",
    std::string target = "joint", double weight = 0.25,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue) {
  auto parsed = partable_from_arg(partable, "fit_ordinal_barrier");
  auto starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure); ctx.names = std::move(parsed.names);
  ctx.pt.group_equal = group_equal_attr(partable);
  const auto parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  const auto stats = ordinal_stats_from_arg(ordinal_stats);
  const bool ml = estimator == "ML";
  if (ml) {
    auto valid = magmaan::estimate::validate_ordinal_association_model(ctx.pt, &ctx.names.row_user);
    if (!valid) stop_fit(valid.error());
  }
  auto prepared = magmaan::estimate::prepare_ordinal_partable(ctx.pt, stats,
      ml ? magmaan::estimate::OrdinalParameterization::Delta : parameterization,
      &starts, &ctx.names.row_user);
  if (!prepared) stop_fit(prepared.error());
  auto rep = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep) stop_model(rep.error());
  ctx.rep = std::move(*rep); ctx.ov_names = ctx.rep.ov_names[0];
  ctx.samp.S = stats.R; ctx.samp.n_obs = stats.n_obs; ctx.meanstructure = has_meanstructure(ctx.pt);
  auto options = multiinfo_options_from(1.25, R_NilValue, target); options.weight = weight;
  const auto backend = optimizer.isNull() ? magmaan::estimate::Backend::Port : backend_from_optimizer_arg(optimizer);
  const auto weights = ml ? magmaan::estimate::OrdinalWeightKind::DWLS : ordinal_weight_from_estimator(estimator, "ordinal barrier");
  auto fit = magmaan::estimate::frontier::fit_ordinal_multiinfo(ctx.pt, ctx.rep, stats,
      ordinal_starts_or_stop(ctx, stats, starts), ml, weights, parameterization,
      options, backend, optim_opts_from(control), &ctx.names.row_user);
  if (!fit) stop_fit(fit.error());
  auto out = ordinal_fit_result(ctx, stats, fit->estimates, &starts, estimator.c_str(), parameterization_name.c_str());
  out["ordinal_stats"] = ordinal_stats; out["thresholds"] = ordinal_stats["thresholds"]; out["polychoric"] = ordinal_stats["R"];
  out["penalty"] = multiinfo_penalty_to_r(ctx, *fit);
  out["covariance_policy"] = "barrier"; out["penalty_inference"] = "not_validated";
  return out;
}

// ML discrepancy on Stage-1 polychoric correlations, ordinary or PSD.
//
// [[Rcpp::export]]
Rcpp::List fit_ml_ordinal_impl(
    SEXP partable, Rcpp::List ordinal_stats,
    bool psd = false,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    double start_eigen_floor = 1e-6,
    double feasibility_tol = 1e-6) {
  const std::string parameterization_name =
      ordinal_parameterization_attr(partable);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ml (ordinal)");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  auto valid = magmaan::estimate::validate_ordinal_association_model(
      ctx.pt, &ctx.names.row_user);
  if (!valid) stop_fit(valid.error());
  auto prep_or = magmaan::estimate::prepare_ordinal_delta_partable(
      ctx.pt, stats, &starts, &ctx.names.row_user);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty()
      ? std::vector<std::string>{}
      : ctx.rep.ov_names[0];
  ctx.meanstructure = false;
  const Eigen::VectorXd x0 = ordinal_starts_or_stop(ctx, stats, starts);
  const magmaan::estimate::Backend backend =
      optimizer.isNull()
          ? (psd ? magmaan::estimate::Backend::NloptSlsqp
                 : magmaan::estimate::Backend::NloptLbfgs)
          : backend_from_optimizer_arg(optimizer);
  magmaan::estimate::frontier::PsdFitOptions psd_opts;
  psd_opts.start_eigen_floor = start_eigen_floor;
  psd_opts.feasibility_tol = feasibility_tol;
  auto fit_or = psd
      ? magmaan::estimate::frontier::fit_ml_psd(
          ctx.pt, ctx.rep, stats, x0, backend, optim_opts_from(control),
          psd_opts, &ctx.names.row_user)
      : magmaan::estimate::frontier::fit_ml(
          ctx.pt, ctx.rep, stats, x0, backend, optim_opts_from(control), &ctx.names.row_user);
  if (!fit_or.has_value()) stop_fit(fit_or.error());
  const magmaan::estimate::Estimates est = std::move(*fit_or);
  Rcpp::List out = ordinal_fit_result(
      ctx, stats, est, &starts, "ML", parameterization_name.c_str());
  // Preserve the supplied Stage-1 objects, including R names and metadata.
  out["ordinal_stats"] = ordinal_stats;
  out["thresholds"] = ordinal_stats["thresholds"];
  out["polychoric"] = ordinal_stats["R"];
  out["covariance_policy"] = psd ? "psd" : "unrestricted";
  Rcpp::List composition = out["composition"];
  composition["covariance_domain"] = psd ? "psd" : "unrestricted";
  composition["algorithm"] = std::string(magmaan::estimate::backend_name(backend));
  out["composition"] = composition;
  return out;
}

// [[Rcpp::export]]
Rcpp::List fit_ordinal_stage2_impl(SEXP partable, Rcpp::List ordinal_stats,
                                   std::string stage2_weight = "dwls",
                                   double dls_a = 0.5,
                                   Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                   Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                   Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "fit_ordinal_stage2");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);

  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(ordinal_stats);
  magmaan::estimate::frontier::OrdinalStage2DlsOptions dls;
  dls.a = dls_a;
  auto weighted_or = magmaan::estimate::frontier::ordinal_stats_with_stage2_weight(
      stats, ordinal_stage2_weight_from_string(stage2_weight), dls);
  if (!weighted_or.has_value()) stop_post(weighted_or.error());
  magmaan::data::OrdinalStats weighted = std::move(*weighted_or);

  auto prep_or = magmaan::estimate::prepare_ordinal_delta_partable(
      ctx.pt, weighted, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = weighted.R;
  ctx.samp.n_obs = weighted.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{}
                                          : ctx.rep.ov_names[0];
  ctx.meanstructure = false;

  const Eigen::VectorXd x0 = ordinal_starts_or_stop(ctx, weighted, starts);
  auto e_or = magmaan::estimate::fit_ordinal_bounded(
      ctx.pt, ctx.rep, weighted, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::WLS, x0,
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);

  const std::string label = ordinal_stage2_label(stage2_weight);
  Rcpp::List out = ordinal_fit_result(ctx, weighted, est, &starts,
                                      label.c_str(),
                                      parameterization_name.c_str());
  out["stage2_weight"] = label;
  out["stage2_dls_a"] = dls_a;
  out["ordinal_computational_weight"] = "WLS";
  return out;
}

// [[Rcpp::export]]
Rcpp::List frontier_pairwise_ordinal_composite_nested_impl(
    SEXP partable_H1,
    SEXP partable_H0,
    SEXP X,
    Rcpp::List n_levels,
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    double fd_step = 1e-5) {
  auto blocks = matrix_blocks_from_arg(X);
  auto levels = n_levels_from_arg(n_levels);
  auto data_or =
      magmaan::estimate::frontier::pairwise_ordinal_observed_data(blocks,
                                                                  levels);
  if (!data_or.has_value()) stop_post(data_or.error());

  magmaan::spec::Starts starts1;
  magmaan::spec::Starts starts0;
  Ctx ctx1 = pairwise_ordinal_ctx_from_partable(
      partable_H1, "frontier_pairwise_ordinal_composite_nested",
      data_or->stats, starts1);
  Ctx ctx0 = pairwise_ordinal_ctx_from_partable(
      partable_H0, "frontier_pairwise_ordinal_composite_nested",
      data_or->stats, starts0);

  const magmaan::estimate::Backend backend =
      backend_from_optimizer_arg(optimizer);
  const magmaan::optim::OptimOptions opts = optim_opts_from(control);

  auto fit1_or = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      ctx1.pt, ctx1.rep, *data_or, {}, {}, backend, opts, starts1);
  if (!fit1_or.has_value()) stop_fit(fit1_or.error());
  auto fit0_or = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      ctx0.pt, ctx0.rep, *data_or, {}, {}, backend, opts, starts0);
  if (!fit0_or.has_value()) stop_fit(fit0_or.error());

  auto god1_or = magmaan::estimate::frontier::pairwise_ordinal_composite_godambe(
      ctx1.pt, ctx1.rep, *data_or, fit1_or->estimates, fd_step);
  if (!god1_or.has_value()) stop_post(god1_or.error());
  auto god0_or = magmaan::estimate::frontier::pairwise_ordinal_composite_godambe(
      ctx0.pt, ctx0.rep, *data_or, fit0_or->estimates, fd_step);
  if (!god0_or.has_value()) stop_post(god0_or.error());
  auto lr_or = magmaan::estimate::frontier::lr_test_pairwise_ordinal_composite(
      ctx1.pt, ctx1.rep, *data_or, *fit1_or, ctx0.pt, ctx0.rep, *fit0_or,
      magmaan::robust::SatorraAMethod::Exact, fd_step);
  if (!lr_or.has_value()) stop_post(lr_or.error());

  Rcpp::List h1 = ordinal_fit_result(ctx1, data_or->stats, fit1_or->estimates,
                                     &starts1, "PAIRWISE-ORDINAL-COMPOSITE",
                                     "delta");
  h1["pairwise_composite"] = true;
  h1["objective"] = pairwise_objective_to_r(fit1_or->objective);
  h1["godambe"] = pairwise_godambe_to_r(*god1_or);

  Rcpp::List h0 = ordinal_fit_result(ctx0, data_or->stats, fit0_or->estimates,
                                     &starts0, "PAIRWISE-ORDINAL-COMPOSITE",
                                     "delta");
  h0["pairwise_composite"] = true;
  h0["objective"] = pairwise_objective_to_r(fit0_or->objective);
  h0["godambe"] = pairwise_godambe_to_r(*god0_or);

  return Rcpp::List::create(
      Rcpp::_["h1"] = h1,
      Rcpp::_["h0"] = h0,
      Rcpp::_["lr"] = pairwise_lr_to_r(*lr_or),
      Rcpp::_["stats"] = ordinal_stats_to_r(data_or->stats),
      Rcpp::_["saturated"] = pairwise_objective_to_r(data_or->saturated));
}

// [[Rcpp::export]]
Rcpp::List fit_dwls_mixed_ordinal_impl(SEXP partable, Rcpp::List mixed_stats,
                                       Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                       Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                       Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_dwls_mixed_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.names = std::move(parsed.names);
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(mixed_stats);
  auto prep_or = magmaan::estimate::prepare_mixed_ordinal_delta_partable(ctx.pt, stats, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.mean = stats.mean;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{} : ctx.rep.ov_names[0];
  ctx.meanstructure = true;
  const Eigen::VectorXd x0 = mixed_ordinal_starts_or_stop(ctx, stats, starts);
  auto e_or = magmaan::estimate::fit_mixed_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::DWLS, x0,
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return mixed_ordinal_fit_result(ctx, stats, est, &starts, "DWLS",
                                  parameterization_name.c_str());
}

// [[Rcpp::export]]
Rcpp::List fit_wls_mixed_ordinal_impl(SEXP partable, Rcpp::List mixed_stats,
                                      Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
                                      Rcpp::Nullable<Rcpp::List>   control   = R_NilValue,
                                      Rcpp::Nullable<Rcpp::List>   bounds    = R_NilValue) {
  const std::string parameterization_name = ordinal_parameterization_attr(partable);
  const auto parameterization = ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_wls_mixed_ordinal");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.names = std::move(parsed.names);
  magmaan::data::MixedOrdinalStats stats = mixed_ordinal_stats_from_arg(mixed_stats);
  auto prep_or = magmaan::estimate::prepare_mixed_ordinal_delta_partable(ctx.pt, stats, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.mean = stats.mean;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty() ? std::vector<std::string>{} : ctx.rep.ov_names[0];
  ctx.meanstructure = true;
  const Eigen::VectorXd x0 = mixed_ordinal_starts_or_stop(ctx, stats, starts);
  auto e_or = magmaan::estimate::fit_mixed_ordinal_bounded(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      magmaan::estimate::OrdinalWeightKind::WLS, x0,
      backend_from_optimizer_arg(optimizer), optim_opts_from(control),
      parameterization);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);
  return mixed_ordinal_fit_result(ctx, stats, est, &starts, "WLS",
                                  parameterization_name.c_str());
}

// Mixed continuous/ordinal ULS/DWLS/WLS with PSD primitive LISREL covariance
// matrices. Stage-1 mixed moments and fixed weights are passed through
// unchanged.
//
// [[Rcpp::export]]
Rcpp::List frontier_fit_mixed_ordinal_psd_impl(
    SEXP partable, Rcpp::List mixed_stats,
    std::string estimator = "DWLS",
    Rcpp::Nullable<Rcpp::String> optimizer = R_NilValue,
    Rcpp::Nullable<Rcpp::List> control = R_NilValue,
    Rcpp::Nullable<Rcpp::List> bounds = R_NilValue,
    double start_eigen_floor = 1e-6,
    double feasibility_tol = 1e-6) {
  const std::string parameterization_name =
      ordinal_parameterization_attr(partable);
  const auto parameterization =
      ordinal_parameterization_from_string(parameterization_name);
  magmaan::compat::lavaan::ParsedLavaanParTable parsed =
      partable_from_arg(partable, "frontier_fit_mixed_ordinal_psd");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx;
  ctx.pt = std::move(parsed.structure);
  ctx.pt.group_equal = group_equal_attr(partable);
  ctx.names = std::move(parsed.names);
  magmaan::data::MixedOrdinalStats stats =
      mixed_ordinal_stats_from_arg(mixed_stats);
  auto prep_or = magmaan::estimate::prepare_mixed_ordinal_partable(
      ctx.pt, stats, parameterization, &starts);
  if (!prep_or.has_value()) stop_fit(prep_or.error());
  auto rep_or = lvm::build_matrix_rep(ctx.pt, &ctx.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  ctx.rep = std::move(*rep_or);
  ctx.samp.S = stats.R;
  ctx.samp.mean = stats.mean;
  ctx.samp.n_obs = stats.n_obs;
  ctx.ov_names = ctx.rep.ov_names.empty()
      ? std::vector<std::string>{}
      : ctx.rep.ov_names[0];
  ctx.meanstructure = true;
  const Eigen::VectorXd x0 =
      mixed_ordinal_starts_or_stop(ctx, stats, starts);
  const magmaan::estimate::Backend backend =
      optimizer.isNull()
          ? magmaan::estimate::Backend::NloptSlsqp
          : backend_from_optimizer_arg(optimizer);
  magmaan::estimate::frontier::PsdFitOptions psd_opts;
  psd_opts.start_eigen_floor = start_eigen_floor;
  psd_opts.feasibility_tol = feasibility_tol;
  auto fit_or = magmaan::estimate::frontier::fit_mixed_ordinal_psd(
      ctx.pt, ctx.rep, stats, bounds_from_nullable(bounds),
      ordinal_weight_from_estimator(
          estimator, "frontier_fit_mixed_ordinal_psd"),
      x0, backend, optim_opts_from(control), parameterization, psd_opts);
  if (!fit_or.has_value()) stop_fit(fit_or.error());
  Rcpp::List out = mixed_ordinal_fit_result(
      ctx, stats, *fit_or, &starts, estimator.c_str(),
      parameterization_name.c_str());
  out["covariance_policy"] = "psd";
  return out;
}

// ordinal_catml_dwls_rmsea_impl() — lavaan-compatible categorical robust RMSEA
// ingredients for an all-ordinal DWLS/WLSMV fit. This evaluates the CATML
// normal-theory correlation discrepancy at the existing ordinal estimates; it
// does not re-optimize.
//
// [[Rcpp::export]]
Rcpp::List ordinal_catml_dwls_rmsea_impl(Rcpp::List fit,
                                         SEXP ordinal_stats = R_NilValue) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  Rcpp::List stats_r = stats_from_fit_or_arg(
      fit, ordinal_stats, "ordinal_stats", "ordinal_catml_dwls_rmsea_impl");
  magmaan::data::OrdinalStats stats = ordinal_stats_from_arg(stats_r);
  const std::string parameterization_name =
      fit.containsElementNamed("parameterization")
          ? Rcpp::as<std::string>(fit["parameterization"])
          : "delta";
  auto out_or = magmaan::estimate::catml_dwls_rmsea_ordinal(
      ctx.pt, ctx.rep, stats, est,
      ordinal_parameterization_from_string(parameterization_name));
  if (!out_or.has_value()) stop_post(out_or.error());
  const auto& out = *out_or;
  return Rcpp::List::create(
      Rcpp::_["XX3"] = out.xx3,
      Rcpp::_["df3"] = out.df3,
      Rcpp::_["c.hat3"] = out.c_hat3,
      Rcpp::_["XX3.scaled"] = out.xx3_scaled,
      Rcpp::_["rmsea.robust"] = out.rmsea_robust);
}
