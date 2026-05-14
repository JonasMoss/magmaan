// Rcpp glue for magmaan's fitting + post-fit-inference layer. Composition-first:
// one thin wrapper per C++ entry point, no convenience bundling. fit_fit()
// mirrors fit() and returns a "fit object" (a transparent base-R list carrying
// the partable / sample stats / theta needed downstream); the other functions
// each take a fit object (and, where the C++ signature does, the upstream
// results — an se result, a baseline result, implied moments) and mirror one
// C++ function. Errors -> Rcpp::stop with magmaan's error kind + detail. Eigen
// <-> R via RcppEigen. Shared plumbing lives in internal.hpp.

#include "internal.hpp"

#include "magmaan/estimate/start_values.hpp"
#include "magmaan/nt/ml.hpp"
#include "magmaan/nt/measures.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace magmaanr;

namespace {

Rcpp::List names_list(const std::vector<std::vector<std::string>>& nn) {
  Rcpp::List out(static_cast<R_xlen_t>(nn.size()));
  for (std::size_t b = 0; b < nn.size(); ++b) out[static_cast<R_xlen_t>(b)] = Rcpp::wrap(nn[b]);
  return out;
}
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

}  // namespace

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
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "model_matrix_rep");
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

// =============================================================================

// fit_fit() — mirrors fit(pt, rep, samp, ML{}, LbfgsOptimizer{lbfgs}). `partable`
// is a partable data.frame (e.g. from lavaan_lavaanify(), possibly hand-edited).
// `sample_stats` is the {S, nobs, mean} bundle — exactly what
// data_sample_stats_from_raw() returns, or a hand-built list(S = , nobs = ):
//   $S    a covariance matrix (single group) or a list of per-group ones;
//   $nobs the matching sample size (scalar or per-group vector);
//   $mean (optional) NULL, a vector, or a list of vectors — used only when the
//         model has mean structure. Values used verbatim (no rescaling).
// `lbfgs` is a named list read for max_iter/ftol/gtol/history. Returns a fit
// object: a transparent list carrying everything the infer_information_* /
// infer_chi2_stat / infer_df_stat / infer_*_test / infer_baseline / measures_fit /
// model_implied / robust functions need to recompose, plus convenient views
// (theta, partable+est). Always uniform:
// $S / $sample_mean are per-group lists (length ngroups ≥ 1); $nobs a vector.
//
// [[Rcpp::export]]
Rcpp::List fit_fit(SEXP partable, Rcpp::List sample_stats,
                     Rcpp::Nullable<Rcpp::List> lbfgs = R_NilValue) {
  // `==` / shared-label equality rows are enforced by fit() (reparam θ = K·α);
  // `<` / `>` rows and arbitrary-expression `==` rows make fit() error with a
  // clear message; `:=` rows are ignored during the fit (post-fit quantities).
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_fit");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);

  auto e_or = magmaan::estimate::fit<magmaan::nt::ml::ML, magmaan::optim::LbfgsOptimizer>(
      ctx.pt, ctx.rep, ctx.samp, magmaan::nt::ml::ML{},
      magmaan::optim::LbfgsOptimizer{lbfgs_opts_from(lbfgs)}, starts);
  if (!e_or.has_value()) stop_fit(e_or.error());
  const magmaan::estimate::Estimates est = std::move(*e_or);

  const std::size_t nb = ctx.samp.S.size();
  Rcpp::List S_out(static_cast<R_xlen_t>(nb));
  for (std::size_t b = 0; b < nb; ++b) {
    Rcpp::NumericMatrix Sb = Rcpp::wrap(ctx.samp.S[b]);
    Rcpp::CharacterVector nm = Rcpp::wrap(ctx.rep.ov_names[b]);
    Sb.attr("dimnames") = Rcpp::List::create(nm, nm);
    S_out[static_cast<R_xlen_t>(b)] = Sb;
  }
  SEXP mean_out = R_NilValue;
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

  return Rcpp::List::create(
      Rcpp::_["converged"]     = true,                  // we stop() on failure
      Rcpp::_["fmin"]          = est.fmin,
      Rcpp::_["iterations"]    = est.iterations,
      Rcpp::_["npar"]          = static_cast<int>(ctx.pt.n_free()),
      Rcpp::_["ngroups"]       = static_cast<int>(nb),
      Rcpp::_["ntotal"]        = static_cast<int>(ntotal),
      Rcpp::_["group_var"]     = ctx.names.group_var,
      Rcpp::_["group_labels"]  = Rcpp::wrap(ctx.names.group_labels),
      Rcpp::_["theta"]         = Rcpp::wrap(est.theta),
      Rcpp::_["ov_names"]      = Rcpp::wrap(ctx.ov_names),
      Rcpp::_["partable"]      = partable_df(ctx.pt, ctx.names, est, &starts),  // lavaanify cols + est
      Rcpp::_["S"]             = S_out,
      Rcpp::_["nobs"]          = nobs_out,
      Rcpp::_["sample_mean"]   = mean_out,
      Rcpp::_["meanstructure"] = ctx.meanstructure);
}

// fit_start_values() — mirrors simple_start_values(pt, rep, samp). Returns the
// theta-ordered start vector (length npar). `sample_stats` as in fit_fit();
// values used verbatim.
//
// [[Rcpp::export]]
Rcpp::NumericVector fit_start_values(SEXP partable, Rcpp::List sample_stats) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "fit_start_values");
  magmaan::spec::Starts starts = std::move(parsed.starts);
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto sv_or = magmaan::estimate::simple_start_values(ctx.pt, ctx.rep, ctx.samp, starts);
  if (!sv_or.has_value()) stop_fit(sv_or.error());
  return Rcpp::wrap(*sv_or);
}

// model_implied() — mirrors ModelEvaluator::build(pt, rep).sigma(est.theta).
// Returns list(sigma = list of per-block p x p matrices, mu = list of per-block
// vectors — empty unless the model has mean structure).
//
// [[Rcpp::export]]
Rcpp::List model_implied(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto ev_or = lvm::ModelEvaluator::build(ctx.pt, ctx.rep);
  if (!ev_or.has_value()) stop_model(ev_or.error());
  const lvm::ModelEvaluator ev = std::move(*ev_or);
  auto im_or = ev.sigma(est.theta);
  if (!im_or.has_value()) stop_model(im_or.error());
  const lvm::ImpliedMoments& im = *im_or;

  Rcpp::List sigma(static_cast<R_xlen_t>(im.sigma.size()));
  for (std::size_t b = 0; b < im.sigma.size(); ++b)
    sigma[static_cast<R_xlen_t>(b)] = Rcpp::wrap(im.sigma[b]);
  Rcpp::List mu(static_cast<R_xlen_t>(im.mu.size()));
  for (std::size_t b = 0; b < im.mu.size(); ++b)
    mu[static_cast<R_xlen_t>(b)] = Rcpp::wrap(im.mu[b]);
  return Rcpp::List::create(Rcpp::_["sigma"] = sigma, Rcpp::_["mu"] = mu);
}

// infer_information_expected() — mirrors information_expected(pt, rep, samp, est).
// Returns the (n_free × n_free) expected Fisher information matrix at θ̂.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix infer_information_expected(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto r = magmaan::nt::infer::information_expected(ctx.pt, ctx.rep, ctx.samp, est);
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
  auto r = magmaan::nt::infer::information_observed_fd(ctx.pt, ctx.rep, ctx.samp, est, h_step);
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
  auto r = magmaan::nt::infer::information_observed_analytic(ctx.pt, ctx.rep, ctx.samp, est);
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
  auto r = magmaan::nt::infer::vcov(info_m, ctx.pt);
  if (!r.has_value()) stop_post(r.error());
  return Rcpp::wrap(*r);
}

// infer_se() — mirrors se(vcov). Returns √diag(vcov); NaN for negative
// diagonal entries (Heywood case). Never errors.
//
// [[Rcpp::export]]
Rcpp::NumericVector infer_se(Rcpp::NumericMatrix vcov) {
  const Eigen::MatrixXd vcov_m = Rcpp::as<Eigen::MatrixXd>(vcov);
  return Rcpp::wrap(magmaan::nt::infer::se(vcov_m));
}

// infer_chi2_stat() — mirrors chi2_stat(samp, est). Returns N_total · F_ML(θ̂).
// This primitive does not need a fit object: sample_stats supplies nobs and
// fmin is the optimizer's final discrepancy value (fit$fmin when called after
// fit_fit()).
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
  return magmaan::nt::infer::chi2_stat(samp, est);
}

// infer_df_stat() — mirrors df_stat(pt, samp). Returns Σ_b p_b(p_b+1)/2 (+ means)
// − fixed_x − n_free + constraint.rank. Pure function of the model and the
// data dimensions; doesn't depend on θ̂. Errors on unenforced constraints.
//
// [[Rcpp::export]]
int infer_df_stat(SEXP partable, Rcpp::List sample_stats) {
  magmaan::lavaan::ParsedLavaanParTable parsed = partable_from_arg(partable, "infer_df_stat");
  Ctx ctx = ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                                  sample_stats);
  auto r = magmaan::nt::infer::df_stat(ctx.pt, ctx.samp);
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
  const magmaan::nt::measures::BaselineFit bl = magmaan::nt::measures::baseline_chi2(samp);
  return Rcpp::List::create(Rcpp::_["chi2"] = bl.chi2, Rcpp::_["df"] = bl.df);
}

// measures_fit() — mirrors fit_measures(chi2, df, baseline, samp) plus
// fit_extras(pt, rep, samp, est) (the logl-based information criteria + SRMR).
// `chi2` and `df` are scalars from infer_chi2_stat() / infer_df_stat() (or any
// equivalent statistic / dof); `baseline` is a infer_baseline() result.
//
// [[Rcpp::export]]
Rcpp::List measures_fit(Rcpp::List fit, double chi2, int df,
                              Rcpp::List baseline) {
  Ctx ctx = ctx_from_fit(fit);
  magmaan::nt::measures::BaselineFit bl;
  bl.chi2 = Rcpp::as<double>(baseline["chi2"]);
  bl.df   = Rcpp::as<int>(baseline["df"]);
  const magmaan::nt::measures::FitMeasures fm = magmaan::nt::measures::fit_measures(chi2, df, bl, ctx.samp);
  const magmaan::estimate::Estimates   est = est_from_fit(fit);
  auto fx = magmaan::nt::measures::fit_extras(ctx.pt, ctx.rep, ctx.samp, est);
  const bool have = fx.has_value();
  return Rcpp::List::create(
      Rcpp::_["cfi"]               = fm.cfi,
      Rcpp::_["tli"]               = fm.tli,
      Rcpp::_["rmsea"]             = fm.rmsea,
      Rcpp::_["rmsea.ci.lower"]    = fm.rmsea_ci_lower,
      Rcpp::_["rmsea.ci.upper"]    = fm.rmsea_ci_upper,
      Rcpp::_["srmr"]              = have ? fx->srmr              : NA_REAL,
      Rcpp::_["logl"]              = have ? fx->logl              : NA_REAL,
      Rcpp::_["unrestricted.logl"] = have ? fx->unrestricted_logl : NA_REAL,
      Rcpp::_["aic"]               = have ? fx->aic               : NA_REAL,
      Rcpp::_["bic"]               = have ? fx->bic               : NA_REAL,
      Rcpp::_["bic2"]              = have ? fx->bic2              : NA_REAL,
      Rcpp::_["npar"]              = have ? fx->npar              : NA_INTEGER,
      Rcpp::_["ntotal"]            = have ? static_cast<double>(fx->ntotal) : NA_REAL);
}

// infer_z_test() — mirrors z_test(est, se). `se` is the SE vector from
// infer_se(infer_vcov(info, fit)).
//
// [[Rcpp::export]]
Rcpp::List infer_z_test(Rcpp::List fit, Rcpp::NumericVector se) {
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  const Eigen::VectorXd se_v = Rcpp::as<Eigen::VectorXd>(se);
  const magmaan::nt::infer::ZTestResult zt = magmaan::nt::infer::z_test(est, se_v);
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
    out[i] = (ISNA(c) || d == NA_INTEGER) ? NA_REAL : magmaan::nt::infer::chi2_pvalue(c, d);
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
  auto w_or = magmaan::nt::infer::wald_test(Rmat, qv, est, vcov_m);
  if (!w_or.has_value()) stop_post(w_or.error());
  return Rcpp::List::create(Rcpp::_["chi2"] = w_or->chi2, Rcpp::_["df"] = w_or->df,
                            Rcpp::_["pvalue"] = magmaan::nt::infer::chi2_pvalue(w_or->chi2, w_or->df));
}

// infer_browne_residual_nt() — mirrors browne_residual_nt(pt, rep, samp, est).
// Returns just the statistic (the model df is infer_df_stat(); the p-value
// is infer_chi2_pvalue(statistic, df)).
//
// [[Rcpp::export]]
Rcpp::List infer_browne_residual_nt(Rcpp::List fit) {
  Ctx ctx = ctx_from_fit(fit);
  const magmaan::estimate::Estimates est = est_from_fit(fit);
  auto s_or = magmaan::nt::infer::browne_residual_nt(ctx.pt, ctx.rep, ctx.samp, est);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = *s_or);
}

// infer_rls_chi2() — mirrors rls_chi2(samp, implied). `implied` is a
// model_implied() result. Returns just the statistic.
//
// [[Rcpp::export]]
Rcpp::List infer_rls_chi2(Rcpp::List fit, Rcpp::List implied) {
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
  auto s_or = magmaan::nt::infer::rls_chi2(ctx.samp, im);
  if (!s_or.has_value()) stop_post(s_or.error());
  return Rcpp::List::create(Rcpp::_["statistic"] = *s_or);
}
