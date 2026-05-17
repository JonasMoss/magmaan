// R bindings for the Satorra-2000 nested-model χ² difference test.
//
// `infer_lr_test_satorra2000` is the single C++ entry point: it takes both
// fit objects (the R list returned by `fit_fit`), the raw data per group
// (reordered to the model's variable order), the chi²/df pair for each fit,
// and an optional GammaSource flag.  Returns an R list mirroring
// `LRSatorra2000Result` field-for-field.

// [[Rcpp::depends(RcppEigen)]]

#include <string>
#include <vector>

#include <Rcpp.h>
#include <RcppEigen.h>

#include "internal.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/robust/robust.hpp"


namespace {

magmaan::nt::robust::GammaSource parse_gamma(const std::string& s) {
  if (s == "NT" || s == "nt") return magmaan::nt::robust::GammaSource::NT;
  return magmaan::nt::robust::GammaSource::Empirical;
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
// products, matches lavaan's `estimator = "MLR"` / `"MLM"`) or `"NT"`
// (normal-theory Γ_NT, a sanity-check path where all λⱼ → 1 and
// `T_scaled == T_diff`).
//
// [[Rcpp::export]]
Rcpp::List infer_lr_test_satorra2000(Rcpp::List           fit_H1,
                                     Rcpp::List           fit_H0,
                                     Rcpp::List           X_per_group,
                                     double               T_H1,
                                     int                  df_H1,
                                     double               T_H0,
                                     int                  df_H0,
                                     std::string          gamma = "empirical") {
  // ── Build the H1 context (pt, rep, samp) and pull θ̂_H1 ─────────────────
  magmaanr::Ctx ctx_H1 = magmaanr::ctx_from_fit(fit_H1);
  const magmaan::estimate::Estimates est_H1 = magmaanr::est_from_fit(fit_H1);

  // ── Re-derive both K matrices from each fit's partable ────────────────
  auto K_H1_or = magmaan::estimate::build_eq_constraints(ctx_H1.pt);
  if (!K_H1_or.has_value()) magmaanr::stop_post(K_H1_or.error());
  auto parsed_H0 = magmaanr::partable_from_arg(fit_H0["partable"],
                                              "infer_lr_test_satorra2000");
  auto K_H0_or = magmaan::estimate::build_eq_constraints(parsed_H0.structure);
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
  auto r_or = magmaan::nt::robust::lr_test_satorra2000_from_data(
      ctx_H1.pt, ctx_H1.rep, est_H1.theta,
      *K_H1_or, *K_H0_or,
      Xs, means, n_g, w_g,
      T_H0, T_H1, df_H0, df_H1,
      parse_gamma(gamma));
  if (!r_or.has_value()) magmaanr::stop_post(r_or.error());
  const magmaan::nt::robust::LRSatorra2000Result& r = *r_or;

  Rcpp::CharacterVector warns(static_cast<R_xlen_t>(r.warnings.size()));
  for (std::size_t k = 0; k < r.warnings.size(); ++k) {
    warns[static_cast<R_xlen_t>(k)] = r.warnings[k];
  }
  return Rcpp::List::create(
      Rcpp::_["T_diff"]      = r.T_diff,
      Rcpp::_["df_diff"]     = r.df_diff,
      Rcpp::_["p_unscaled"]  = r.p_unscaled,
      Rcpp::_["eigenvalues"] = Rcpp::wrap(r.eigenvalues),
      Rcpp::_["scale_c"]     = r.scale_c,
      Rcpp::_["T_scaled"]    = r.T_scaled,
      Rcpp::_["p_scaled"]    = r.p_scaled,
      Rcpp::_["adjust_d0"]   = r.adjust_d0,
      Rcpp::_["T_adjusted"]  = r.T_adjusted,
      Rcpp::_["p_adjusted"]  = r.p_adjusted,
      Rcpp::_["p_mixture"]   = r.p_mixture,
      Rcpp::_["warnings"]    = warns);
}
