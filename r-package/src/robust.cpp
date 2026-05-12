// Rcpp glue for latva's robust normal-theory inference: the U-factor at θ̂, the
// three Γ-meat reductions (M_Γ = BᵀΓB), the eigenvalues of UΓ, the
// Satorra-Bentler / mean-var-adjusted / scaled-shifted χ² built from them, the
// raw-data / standalone-Γ helpers, and the sandwich SEs. Composition-first: one
// thin wrapper per C++ entry point, including build_u_factor() so the U-factor
// is a first-class (transparent base-R list) object you build once and reduce /
// eigensolve several ways. Shared plumbing lives in internal.hpp.

#include "internal.hpp"

#include "latva/fit/robust.hpp"
#include "latva/fit/raw_data.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace latvar;

namespace {

// ---- InferenceSpec enums <-> strings ---------------------------------------

lvf::Information info_from_string(const std::string& s) {
  if (s == "expected") return lvf::Information::Expected;
  if (s == "observed") return lvf::Information::Observed;
  Rcpp::stop("latva: `bread` must be 'expected' or 'observed' (got '%s')", s);
}
lvf::WeightMoments moments_from_string(const std::string& s) {
  if (s == "structured")   return lvf::WeightMoments::Structured;
  if (s == "unstructured") return lvf::WeightMoments::Unstructured;
  Rcpp::stop("latva: `moments` must be 'structured' or 'unstructured' (got '%s')", s);
}
lvf::ScoreCovariance cov_from_string(const std::string& s) {
  if (s == "model_implied")   return lvf::ScoreCovariance::ModelImplied;
  if (s == "empirical")       return lvf::ScoreCovariance::Empirical;
  if (s == "browne_unbiased") return lvf::ScoreCovariance::BrowneUnbiased;
  Rcpp::stop("latva: `cov` must be 'model_implied', 'empirical', or 'browne_unbiased' (got '%s')", s);
}
const char* moments_to_string(lvf::WeightMoments m) {
  return m == lvf::WeightMoments::Structured ? "structured" : "unstructured";
}
const char* ufactor_kind_to_string(lvf::UFactor::Kind k) {
  return k == lvf::UFactor::Kind::ProjectionExpected ? "ProjectionExpected" : "ObservedHessian";
}

lvf::InferenceSpec spec_from(const std::string& bread, const std::string& moments) {
  lvf::InferenceSpec s;
  s.bread = info_from_string(bread);
  s.moments = moments_from_string(moments);
  return s;
}
lvf::InferenceSpec spec_from(const std::string& bread, const std::string& moments,
                             const std::string& cov) {
  lvf::InferenceSpec s = spec_from(bread, moments);
  s.cov = cov_from_string(cov);
  return s;
}

// ---- UFactor <-> R list ----------------------------------------------------

Rcpp::List ufactor_to_list(const lvf::UFactor& uf) {
  Rcpp::List blocks(static_cast<R_xlen_t>(uf.blocks.size()));
  for (std::size_t b = 0; b < uf.blocks.size(); ++b) {
    const lvf::UFactor::Block& blk = uf.blocks[b];
    blocks[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["p"]          = static_cast<int>(blk.p),
        Rcpp::_["pstar"]      = static_cast<int>(blk.pstar),
        Rcpp::_["row_offset"] = static_cast<int>(blk.row_offset),
        Rcpp::_["n_obs"]      = static_cast<int>(blk.n_obs),
        Rcpp::_["Sigma_hat"]  = Rcpp::wrap(blk.Sigma_hat),
        Rcpp::_["S"]          = Rcpp::wrap(blk.S));
  }
  const bool proj = (uf.kind == lvf::UFactor::Kind::ProjectionExpected);
  return Rcpp::List::create(
      Rcpp::_["kind"]      = std::string(ufactor_kind_to_string(uf.kind)),
      Rcpp::_["B"]         = proj ? static_cast<SEXP>(Rcpp::wrap(uf.B)) : R_NilValue,
      Rcpp::_["A"]         = proj ? R_NilValue : static_cast<SEXP>(Rcpp::wrap(uf.A)),
      Rcpp::_["H_obs_inv"] = proj ? R_NilValue : static_cast<SEXP>(Rcpp::wrap(uf.H_obs_inv)),
      Rcpp::_["df"]        = static_cast<int>(uf.df),
      Rcpp::_["pstar"]     = static_cast<int>(uf.pstar),
      Rcpp::_["moments"]   = std::string(moments_to_string(uf.moments)),
      Rcpp::_["blocks"]    = blocks);
}

lvf::UFactor ufactor_from_list(Rcpp::List ul) {
  auto need = [&](const char* nm) {
    if (!ul.containsElementNamed(nm))
      Rcpp::stop("latva: not a u-factor list (missing '%s') — pass the result of latva_build_u_factor()", nm);
  };
  need("kind"); need("df"); need("pstar"); need("moments"); need("blocks");
  lvf::UFactor uf;
  const std::string kind = Rcpp::as<std::string>(ul["kind"]);
  if (kind == "ProjectionExpected") {
    uf.kind = lvf::UFactor::Kind::ProjectionExpected;
    need("B");
    uf.B = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(ul["B"]));
  } else if (kind == "ObservedHessian") {
    uf.kind = lvf::UFactor::Kind::ObservedHessian;
    need("A"); need("H_obs_inv");
    uf.A = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(ul["A"]));
    uf.H_obs_inv = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(ul["H_obs_inv"]));
  } else {
    Rcpp::stop("latva: u-factor `kind` must be 'ProjectionExpected' or 'ObservedHessian' (got '%s')", kind);
  }
  uf.df    = static_cast<Eigen::Index>(Rcpp::as<int>(ul["df"]));
  uf.pstar = static_cast<Eigen::Index>(Rcpp::as<int>(ul["pstar"]));
  uf.moments = moments_from_string(Rcpp::as<std::string>(ul["moments"]));

  Rcpp::List blocks(ul["blocks"]);
  uf.blocks.resize(static_cast<std::size_t>(blocks.size()));
  for (R_xlen_t b = 0; b < blocks.size(); ++b) {
    Rcpp::List bl(blocks[b]);
    lvf::UFactor::Block& blk = uf.blocks[static_cast<std::size_t>(b)];
    blk.p          = static_cast<Eigen::Index>(Rcpp::as<int>(bl["p"]));
    blk.pstar      = static_cast<Eigen::Index>(Rcpp::as<int>(bl["pstar"]));
    blk.row_offset = static_cast<Eigen::Index>(Rcpp::as<int>(bl["row_offset"]));
    blk.n_obs      = bl.containsElementNamed("n_obs")
                         ? static_cast<Eigen::Index>(Rcpp::as<int>(bl["n_obs"])) : 0;
    blk.Sigma_hat  = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(bl["Sigma_hat"]));
    blk.S          = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(bl["S"]));
    // Re-derive the per-block weight Cholesky from the stored moments.
    const Eigen::MatrixXd& M =
        (uf.moments == lvf::WeightMoments::Structured) ? blk.Sigma_hat : blk.S;
    auto g_or = lvf::gamma_nt(M);
    if (!g_or.has_value()) stop_post(g_or.error());
    blk.llt_gamma_nt.compute(*g_or);
    if (blk.llt_gamma_nt.info() != Eigen::Success)
      Rcpp::stop("latva: u-factor block %d weight matrix is not positive-definite",
                 static_cast<int>(b));
  }
  return uf;
}

// ---- raw data from an R argument -------------------------------------------

// Build a (possibly multi-block) RawData from an R raw-data argument: `X` is a
// matrix (single group) or a list of per-group matrices; each block's columns
// are reordered to that block's model variable order (by colnames if present,
// else assumed already in order). Reuses the internal.hpp column-permute
// helpers.
lvf::RawData raw_from_arg(const lvm::MatrixRep& rep, SEXP X) {
  const std::size_t n_blocks = rep.dims.size();
  lvf::RawData raw;
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

// latva_build_u_factor() — mirrors build_u_factor(pt, rep, samp, est, {bread,
// moments}). Returns the u-factor as a transparent list (kind, B|A|H_obs_inv,
// df, pstar, moments, blocks). `cov` is not an argument — build_u_factor()
// ignores it; the meat is chosen by which latva_reduced_gamma_*() you call.
//
// [[Rcpp::export]]
Rcpp::List latva_build_u_factor(Rcpp::List fit, std::string bread = "expected",
                                std::string moments = "structured") {
  Ctx ctx = ctx_from_fit(fit);
  const lvf::Estimates est = est_from_fit(fit);
  auto uf_or = lvf::build_u_factor(ctx.pt, ctx.rep, ctx.samp, est, spec_from(bread, moments));
  if (!uf_or.has_value()) stop_post(uf_or.error());
  return ufactor_to_list(*uf_or);
}

// latva_reduced_gamma_nt() — mirrors reduced_gamma_nt(uf). Operator-only NT meat
// reduction; its eigenvalues are ~ all-1 (the projector sanity check).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix latva_reduced_gamma_nt(Rcpp::List uf) {
  lvf::UFactor u = ufactor_from_list(uf);
  auto m_or = lvf::reduced_gamma_nt(u);
  if (!m_or.has_value()) stop_post(m_or.error());
  return Rcpp::wrap(*m_or);
}

// latva_reduced_gamma_sample() — mirrors reduced_gamma_sample(uf, Zc, denom).
// `Zc` is the N x p* matrix from latva_casewise_contributions(); `denom` is the
// per-block divisor — a length-ngroups vector (conventionally `fit$nobs`; pass
// `fit$nobs - 1` for the divisor-corrected variant), or a single number to use
// the same divisor for every block. The multi-group SB meat is Σ_b B_bᵀΓ̂_bB_b
// with Γ̂_b = (Zc_b)ᵀ(Zc_b)/n_b — so for >1 block supply the per-block vector
// (a single pooled N would under-weight by ~ngroups).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix latva_reduced_gamma_sample(Rcpp::List uf, Rcpp::NumericMatrix Zc,
                                               Rcpp::NumericVector denom) {
  lvf::UFactor u = ufactor_from_list(uf);
  Eigen::MatrixXd Zceig = Rcpp::as<Eigen::MatrixXd>(Zc);
  Eigen::VectorXd d = Rcpp::as<Eigen::VectorXd>(denom);
  auto m_or = lvf::reduced_gamma_sample(u, Zceig, d);
  if (!m_or.has_value()) stop_post(m_or.error());
  return Rcpp::wrap(*m_or);
}

// latva_reduced_gamma_unbiased() — mirrors reduced_gamma_unbiased(uf, samp,
// M_sample, M_nt). Browne's distribution-free correction; single-block only.
// `n` is the total sample size (the sample cov it needs for the rank-1 term is
// already stored in `uf`); `M_sample`/`M_nt` are latva_reduced_gamma_sample/_nt()
// results for the same `uf`.
//
// [[Rcpp::export]]
Rcpp::NumericMatrix latva_reduced_gamma_unbiased(Rcpp::List uf, int n,
                                                 Rcpp::NumericMatrix M_sample,
                                                 Rcpp::NumericMatrix M_nt) {
  lvf::UFactor u = ufactor_from_list(uf);
  if (u.blocks.empty()) Rcpp::stop("latva: u-factor has no blocks");
  lvf::SampleStats samp;            // single-block; multi-block reduced_gamma_unbiased() errors
  samp.S = {u.blocks[0].S};
  samp.n_obs = {static_cast<std::int64_t>(n)};
  Eigen::MatrixXd Ms = Rcpp::as<Eigen::MatrixXd>(M_sample);
  Eigen::MatrixXd Mn = Rcpp::as<Eigen::MatrixXd>(M_nt);
  auto m_or = lvf::reduced_gamma_unbiased(u, samp, Ms, Mn);
  if (!m_or.has_value()) stop_post(m_or.error());
  return Rcpp::wrap(*m_or);
}

// latva_ugamma_eigenvalues() — mirrors ugamma_eigenvalues(M). Symmetrises `M`,
// returns the ascending real eigenvalues. The research deliverable.
//
// [[Rcpp::export]]
Rcpp::NumericVector latva_ugamma_eigenvalues(Rcpp::NumericMatrix M) {
  Eigen::MatrixXd Meig = Rcpp::as<Eigen::MatrixXd>(M);
  auto ev_or = lvf::ugamma_eigenvalues(Meig);
  if (!ev_or.has_value()) stop_post(ev_or.error());
  return Rcpp::wrap(*ev_or);
}

// latva_satorra_bentler() — mirrors satorra_bentler(t_ml, df, eigvals).
// c = mean(eigvals); T_SB = t_ml / c.
//
// [[Rcpp::export]]
Rcpp::List latva_satorra_bentler(double t_ml, int df, Rcpp::NumericVector eigvals) {
  const Eigen::VectorXd ev = Rcpp::as<Eigen::VectorXd>(eigvals);
  const lvf::SatorraBentlerResult r = lvf::satorra_bentler(t_ml, df, ev);
  return Rcpp::List::create(Rcpp::_["chi2_scaled"] = r.chi2_scaled,
                            Rcpp::_["scale_c"] = r.scale_c, Rcpp::_["df"] = r.df);
}

// latva_mean_var_adjusted() — mirrors mean_var_adjusted(t_ml, df, eigvals).
// df_adj = (Σλ)²/Σλ²; T_adj = t_ml · Σλ/Σλ². (Satterthwaite / Yuan-Bentler T3.)
//
// [[Rcpp::export]]
Rcpp::List latva_mean_var_adjusted(double t_ml, int df, Rcpp::NumericVector eigvals) {
  const Eigen::VectorXd ev = Rcpp::as<Eigen::VectorXd>(eigvals);
  const lvf::MeanVarAdjustedResult r = lvf::mean_var_adjusted(t_ml, df, ev);
  return Rcpp::List::create(Rcpp::_["chi2_adj"] = r.chi2_adj, Rcpp::_["df_adj"] = r.df_adj);
}

// latva_scaled_shifted() — mirrors scaled_shifted(t_ml, df, eigvals).
// a = √(df/Σλ²); b = df − a·Σλ; T_adj = t_ml·a + b (df stays integer).
//
// [[Rcpp::export]]
Rcpp::List latva_scaled_shifted(double t_ml, int df, Rcpp::NumericVector eigvals) {
  const Eigen::VectorXd ev = Rcpp::as<Eigen::VectorXd>(eigvals);
  const lvf::ScaledShiftedResult r = lvf::scaled_shifted(t_ml, df, ev);
  return Rcpp::List::create(Rcpp::_["chi2_adj"] = r.chi2_adj, Rcpp::_["df"] = r.df,
                            Rcpp::_["scale_a"] = r.scale_a, Rcpp::_["shift_b"] = r.shift_b);
}

// =============================================================================
// Raw-data + standalone-Γ helpers
// =============================================================================

// latva_casewise_contributions() — mirrors casewise_contributions(raw,
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
Rcpp::NumericMatrix latva_casewise_contributions(SEXP partable, SEXP X) {
  lvp::ParsedLavaanParTable parsed = partable_from_arg(partable, "latva_casewise_contributions");
  auto rep_or = lvm::build_matrix_rep(parsed.structure, &parsed.names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  if (rep_or->ov_names.empty() || rep_or->ov_names[0].empty())
    Rcpp::stop("latva: model has no observed variables");
  lvf::RawData raw = raw_from_arg(*rep_or, X);
  auto ss_or = lvf::sample_stats_from_raw(raw);
  if (!ss_or.has_value()) stop_post(ss_or.error());
  auto zc_or = lvf::casewise_contributions(raw, *ss_or);
  if (!zc_or.has_value()) stop_post(zc_or.error());
  return Rcpp::wrap(*zc_or);
}

// latva_sample_stats_from_raw() — mirrors sample_stats_from_raw(RawData{...}).
// `X` a raw-data matrix (single group) or a list of per-group matrices. Returns
// the {S, mean, nobs} bundle (the canonical latva_fit() `sample_stats` arg),
// uniformly per-group: list(S = list of <p x p, N-divisor> covariances, mean =
// list of <length-p> mean vectors, nobs = <int vector>). For one group these
// are length-1 lists / a length-1 vector.
//
// [[Rcpp::export]]
Rcpp::List latva_sample_stats_from_raw(SEXP X) {
  lvf::RawData raw;
  if (TYPEOF(X) == VECSXP) {
    Rcpp::List Xl(X);
    for (R_xlen_t b = 0; b < Xl.size(); ++b)
      raw.X.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Xl[b])));
  } else {
    raw.X.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(X)));
  }
  auto ss_or = lvf::sample_stats_from_raw(raw);
  if (!ss_or.has_value()) stop_post(ss_or.error());
  const lvf::SampleStats& ss = *ss_or;
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

// latva_empirical_gamma() — mirrors empirical_gamma(X). Returns the p* x p*
// fourth-moment ACOV of vech(S) (vech = lower-tri column-major).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix latva_empirical_gamma(Rcpp::NumericMatrix X) {
  Eigen::MatrixXd Xeig = Rcpp::as<Eigen::MatrixXd>(X);
  auto g_or = lvf::empirical_gamma(Xeig);
  if (!g_or.has_value()) stop_post(g_or.error());
  return Rcpp::wrap(*g_or);
}

// latva_gamma_nt() — mirrors gamma_nt(Sigma). Returns the p* x p* normal-theory
// ACOV Γ_NT = 2 D⁺(Σ⊗Σ)D⁺ᵀ (same vech order).
//
// [[Rcpp::export]]
Rcpp::NumericMatrix latva_gamma_nt(Rcpp::NumericMatrix Sigma) {
  Eigen::MatrixXd Seig = Rcpp::as<Eigen::MatrixXd>(Sigma);
  auto g_or = lvf::gamma_nt(Seig);
  if (!g_or.has_value()) stop_post(g_or.error());
  return Rcpp::wrap(*g_or);
}

// =============================================================================
// Robust ("sandwich") standard errors
// =============================================================================

// latva_robust_se() — mirrors robust_se(pt, rep, samp, est, gamma_hat, {bread,
// moments, cov}). `gamma_hat` a p* x p* matrix. `cov` in {"model_implied",
// "empirical","browne_unbiased"} (the last errors for now). Single-block only
// (for multi-group the per-block weighting is implicit — use latva_robust_se_raw()).
//
// [[Rcpp::export]]
Rcpp::List latva_robust_se(Rcpp::List fit, Rcpp::NumericMatrix gamma_hat,
                           std::string bread = "expected", std::string moments = "structured",
                           std::string cov = "empirical") {
  Ctx ctx = ctx_from_fit(fit);
  const lvf::Estimates est = est_from_fit(fit);
  Eigen::MatrixXd G = Rcpp::as<Eigen::MatrixXd>(gamma_hat);
  auto r_or = lvf::robust_se(ctx.pt, ctx.rep, ctx.samp, est, G, spec_from(bread, moments, cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
                            Rcpp::_["se"]   = Rcpp::wrap(r_or->se));
}

// latva_robust_se_raw() — mirrors robust_se(pt, rep, samp, est, raw, {bread,
// moments, cov}). Derives Γ̂ from raw data `X` (a matrix, or a list of per-group
// matrices) via casewise contributions — never forms the p* x p* matrix. Works
// multi-group for the `Expected` bread (≙ lavaan `se = "robust.sem"` / MLM);
// the `Observed` bread (MLR) is single-block only.
//
// [[Rcpp::export]]
Rcpp::List latva_robust_se_raw(Rcpp::List fit, SEXP X,
                               std::string bread = "expected", std::string moments = "structured",
                               std::string cov = "empirical") {
  Ctx ctx = ctx_from_fit(fit);
  const lvf::Estimates est = est_from_fit(fit);
  lvf::RawData raw = raw_from_arg(ctx.rep, X);
  auto r_or = lvf::robust_se(ctx.pt, ctx.rep, ctx.samp, est, raw, spec_from(bread, moments, cov));
  if (!r_or.has_value()) stop_post(r_or.error());
  return Rcpp::List::create(Rcpp::_["vcov"] = Rcpp::wrap(r_or->vcov),
                            Rcpp::_["se"]   = Rcpp::wrap(r_or->se));
}
