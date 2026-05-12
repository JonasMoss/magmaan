// Shared internal plumbing for the latva R bindings. Header-only helpers in
// namespace `latvar` — no // [[Rcpp::export]] here. Included by fit.cpp and
// robust.cpp; bindings.cpp (parse/lavaanify only) does not need it.

#pragma once

#include <RcppEigen.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/parse/op.hpp"
#include "latva/partable/partable.hpp"
#include "latva/partable/start_hints.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/model/model_evaluator.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/fit/fit.hpp"              // Estimates
#include "latva/fit/inference.hpp"        // Inference
#include "latva/fit/lbfgs_optimizer.hpp"  // LbfgsOptions

namespace lv  = latva;
namespace lvf = latva::fit;
namespace lvm = latva::model;
namespace lvp = latva::partable;

namespace latvar {

// ---- error -> R error -------------------------------------------------------

inline const char* model_error_kind(lv::ModelError::Kind k) {
  using K = lv::ModelError::Kind;
  switch (k) {
    case K::UnsupportedRowKind:  return "UnsupportedRowKind";
    case K::UnknownVariable:     return "UnknownVariable";
    case K::NonPositiveDefinite: return "NonPositiveDefinite";
    case K::EmptyMatrix:         return "EmptyMatrix";
    case K::EigenAssertion:      return "EigenAssertion";
  }
  return "Unknown";
}
inline const char* fit_error_kind(lv::FitError::Kind k) {
  using K = lv::FitError::Kind;
  switch (k) {
    case K::OptimizerNonConvergence:  return "OptimizerNonConvergence";
    case K::NonFiniteObjective:       return "NonFiniteObjective";
    case K::NonPositiveDefiniteSigma: return "NonPositiveDefiniteSigma";
    case K::NonPositiveDefiniteSample:return "NonPositiveDefiniteSample";
    case K::LineSearchFailed:         return "LineSearchFailed";
    case K::InvalidStartValues:       return "InvalidStartValues";
    case K::NumericIssue:             return "NumericIssue";
  }
  return "Unknown";
}
inline const char* post_error_kind(lv::PostError::Kind k) {
  using K = lv::PostError::Kind;
  switch (k) {
    case K::InfoMatrixSingular: return "InfoMatrixSingular";
    case K::BootstrapFailed:    return "BootstrapFailed";
    case K::NumericIssue:       return "NumericIssue";
  }
  return "Unknown";
}

[[noreturn]] inline void stop_model(const lv::ModelError& e) {
  Rcpp::stop("latva model error [%s]: %s", model_error_kind(e.kind), e.detail);
}
[[noreturn]] inline void stop_fit(const lv::FitError& e) {
  Rcpp::stop("latva fit error [%s] after %d iters, f=%g: %s", fit_error_kind(e.kind),
             e.iterations, e.f_value, e.detail);
}
[[noreturn]] inline void stop_post(const lv::PostError& e) {
  Rcpp::stop("latva inference error [%s]: %s", post_error_kind(e.kind), e.detail);
}

// ---- partable <-> data.frame ------------------------------------------------

inline lv::parse::Op op_from_string(const std::string& s) {
  using O = lv::parse::Op;
  if (s == "=~") return O::Measurement;
  if (s == "~")  return O::Regression;
  if (s == "~~") return O::Covariance;
  if (s == "~1") return O::Intercept;
  if (s == ":=") return O::DefineParam;
  if (s == "==") return O::EqConstraint;
  if (s == "<")  return O::LtConstraint;
  if (s == ">")  return O::GtConstraint;
  Rcpp::stop("latva: unrecognized operator '%s' in partable data.frame", s);
}

// Group identity (n_groups, group_var, group_labels) rides on the partable
// data.frame as attributes (`latva.group_var` / `latva.group_labels`) — it's
// table-level, not per-row. `attach_group_attrs` sets them; `read_group_attrs`
// reads them back (tolerating absence — `n_groups()` still works off `group`).
inline void attach_group_attrs(SEXP df, const lvp::ParTable& pt) {
  Rf_setAttrib(df, Rf_install("latva.group_var"),
               Rf_mkString(pt.group_var.c_str()));
  Rcpp::CharacterVector gl(static_cast<R_xlen_t>(pt.group_labels.size()));
  for (std::size_t i = 0; i < pt.group_labels.size(); ++i)
    gl[static_cast<R_xlen_t>(i)] = pt.group_labels[i];
  Rf_setAttrib(df, Rf_install("latva.group_labels"), gl);
}
inline void read_group_attrs(SEXP df, lvp::ParTable& pt) {
  SEXP gv = Rf_getAttrib(df, Rf_install("latva.group_var"));
  if (!Rf_isNull(gv) && TYPEOF(gv) == STRSXP && Rf_length(gv) >= 1)
    pt.group_var = Rcpp::as<std::string>(STRING_ELT(gv, 0));
  SEXP gl = Rf_getAttrib(df, Rf_install("latva.group_labels"));
  if (!Rf_isNull(gl) && TYPEOF(gl) == STRSXP)
    pt.group_labels = Rcpp::as<std::vector<std::string>>(gl);
}

// `out_starts` (optional): if non-null, receives the user start *hints* read
// off the `ustart` column for free rows (sized n_free, NaN where the column is
// NA). For fixed rows the `ustart` column is the parameter's `fixed_value`.
inline lvp::ParTable parse_partable_df(Rcpp::DataFrame df,
                                       lvp::Starts* out_starts = nullptr) {
  auto col = [&](const char* nm) -> SEXP {
    if (!df.containsElementNamed(nm))
      Rcpp::stop("latva: partable data.frame is missing required column '%s'", nm);
    return df[nm];
  };
  Rcpp::IntegerVector id(col("id")), user(col("user")), block(col("block")), group(col("group")),
      freev(col("free")), exo(col("exo"));
  Rcpp::NumericVector ustart(col("ustart"));
  std::vector<std::string> lhs   = Rcpp::as<std::vector<std::string>>(col("lhs"));
  std::vector<std::string> opstr = Rcpp::as<std::vector<std::string>>(col("op"));
  std::vector<std::string> rhs   = Rcpp::as<std::vector<std::string>>(col("rhs"));
  std::vector<std::string> label = Rcpp::as<std::vector<std::string>>(col("label"));
  std::vector<std::string> plab  = Rcpp::as<std::vector<std::string>>(col("plabel"));

  const std::size_t m = static_cast<std::size_t>(id.size());
  lvp::ParTable pt;
  pt.id.resize(m); pt.user.resize(m); pt.lhs.resize(m); pt.op.resize(m); pt.rhs.resize(m);
  pt.block.resize(m); pt.group.resize(m); pt.free.resize(m); pt.exo.resize(m);
  pt.fixed_value.resize(m); pt.label.resize(m); pt.plabel.resize(m);
  std::int32_t n_free = 0;
  for (std::size_t i = 0; i < m; ++i) {
    const R_xlen_t ri = static_cast<R_xlen_t>(i);
    pt.id[i]     = id[ri];
    pt.user[i]   = static_cast<std::int8_t>(user[ri]);
    pt.lhs[i]    = lhs[i];
    pt.op[i]     = op_from_string(opstr[i]);
    pt.rhs[i]    = rhs[i];
    pt.block[i]  = block[ri];
    pt.group[i]  = group[ri];
    pt.free[i]   = freev[ri];
    pt.exo[i]    = static_cast<std::int8_t>(exo[ri]);
    pt.label[i]  = label[i];
    pt.plabel[i] = plab[i];
    if (pt.free[i] > n_free) n_free = pt.free[i];
    // The `ustart` column is split: a fixed row's value is `fixed_value`; a
    // free row's value (if any) is a start hint, kept on `out_starts`.
    if (pt.free[i] == 0) pt.fixed_value[i] = ustart[ri];   // NaN passes through
    else                 pt.fixed_value[i] = std::numeric_limits<double>::quiet_NaN();
  }
  if (out_starts) {
    out_starts->hint.assign(static_cast<std::size_t>(n_free),
                            std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < m; ++i) {
      if (pt.free[i] <= 0) continue;
      const double u = ustart[static_cast<R_xlen_t>(i)];
      if (std::isfinite(u)) out_starts->hint[static_cast<std::size_t>(pt.free[i] - 1)] = u;
    }
  }
  read_group_attrs(df, pt);
  return pt;
}

inline bool has_meanstructure(const lvp::ParTable& pt) {
  for (lv::parse::Op op : pt.op)
    if (op == lv::parse::Op::Intercept) return true;
  return false;
}
inline bool has_constraint_rows(const lvp::ParTable& pt) {
  using O = lv::parse::Op;
  for (lv::parse::Op op : pt.op)
    if (op == O::EqConstraint || op == O::DefineParam || op == O::LtConstraint ||
        op == O::GtConstraint)
      return true;
  return false;
}

// ParTable + fitted estimates -> data.frame (lavaanify columns + est).
// `starts` (optional): the user start hints, so the reconstructed `ustart`
// column carries free-row hints (not just fixed-row values). NaN ⇒ R NA.
inline Rcpp::DataFrame partable_df(const lvp::ParTable& pt, const lvf::Estimates& est,
                                   const lvp::Starts* starts = nullptr) {
  const R_xlen_t nrow = static_cast<R_xlen_t>(pt.size());
  Rcpp::IntegerVector id(nrow), user(nrow), block(nrow), group(nrow), freev(nrow), exo(nrow);
  Rcpp::CharacterVector lhs(nrow), op(nrow), rhs(nrow), label(nrow), plabel(nrow);
  Rcpp::NumericVector ustart(nrow), est_c(nrow);
  for (R_xlen_t i = 0; i < nrow; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    id[i]     = pt.id[k];
    user[i]   = pt.user[k];
    block[i]  = pt.block[k];
    group[i]  = pt.group[k];
    freev[i]  = pt.free[k];
    exo[i]    = pt.exo[k];
    lhs[i]    = pt.lhs[k];
    op[i]     = std::string(lv::parse::to_string(pt.op[k]));
    rhs[i]    = pt.rhs[k];
    label[i]  = pt.label[k];
    plabel[i] = pt.plabel[k];
    const std::int32_t f = pt.free[k];
    if (f <= 0) {
      ustart[i] = pt.fixed_value[k];
      est_c[i]  = pt.fixed_value[k];
    } else {
      const std::size_t fi = static_cast<std::size_t>(f - 1);
      ustart[i] = (starts && fi < starts->hint.size())
                      ? starts->hint[fi]
                      : std::numeric_limits<double>::quiet_NaN();
      est_c[i]  = est.theta(static_cast<Eigen::Index>(f) - 1);
    }
  }
  Rcpp::List cols = Rcpp::List::create(
      Rcpp::_["id"] = id, Rcpp::_["lhs"] = lhs, Rcpp::_["op"] = op, Rcpp::_["rhs"] = rhs,
      Rcpp::_["user"] = user, Rcpp::_["block"] = block, Rcpp::_["group"] = group,
      Rcpp::_["free"] = freev, Rcpp::_["exo"] = exo, Rcpp::_["ustart"] = ustart,
      Rcpp::_["label"] = label, Rcpp::_["plabel"] = plabel, Rcpp::_["est"] = est_c);
  cols.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -static_cast<int>(nrow));
  cols.attr("class") = "data.frame";
  Rcpp::DataFrame out(cols);
  attach_group_attrs(out, pt);
  return out;
}

inline lvf::LbfgsOptions lbfgs_opts_from(Rcpp::Nullable<Rcpp::List> lbfgs) {
  lvf::LbfgsOptions o;  // struct defaults
  if (lbfgs.isNotNull()) {
    Rcpp::List l(lbfgs.get());
    if (l.containsElementNamed("max_iter")) o.max_iter = Rcpp::as<int>(l["max_iter"]);
    if (l.containsElementNamed("ftol"))     o.ftol     = Rcpp::as<double>(l["ftol"]);
    if (l.containsElementNamed("gtol"))     o.gtol     = Rcpp::as<double>(l["gtol"]);
    if (l.containsElementNamed("history"))  o.history  = Rcpp::as<int>(l["history"]);
  }
  return o;
}

// ---- model context ----------------------------------------------------------

struct Ctx {
  lvp::ParTable            pt;
  lvm::MatrixRep           rep;
  lvf::SampleStats         samp;
  std::vector<std::string> ov_names;     // observed-variable order latva uses (block 0)
  bool                     meanstructure = false;
};

// Permutation taking the model's observed-variable order `ov` to `M`'s
// column order: `perm[k]` is the column of `M` for the k-th model variable.
// If `M` has column names, match by name (error on a missing model variable);
// otherwise `M` must already be in the model's order. Works for both a square
// covariance and a rectangular raw-data matrix (rows are ignored).
inline std::vector<int> perm_for_cols(Rcpp::NumericMatrix M,
                                      const std::vector<std::string>& ov,
                                      const char* what) {
  const int p = static_cast<int>(ov.size());
  std::vector<int> perm(static_cast<std::size_t>(p));
  Rcpp::RObject dn_obj = M.attr("dimnames");
  if (!dn_obj.isNULL()) {
    Rcpp::List dn(dn_obj);
    SEXP cn_sexp = (dn.size() >= 2) ? static_cast<SEXP>(dn[1]) : R_NilValue;
    if (!Rf_isNull(cn_sexp)) {
      Rcpp::CharacterVector cn(cn_sexp);
      if (static_cast<int>(cn.size()) == M.ncol()) {
        std::unordered_map<std::string, int> idx;
        for (int j = 0; j < cn.size(); ++j) idx[Rcpp::as<std::string>(cn[j])] = j;
        for (int k = 0; k < p; ++k) {
          auto it = idx.find(ov[static_cast<std::size_t>(k)]);
          if (it == idx.end())
            Rcpp::stop("latva: %s has no column named '%s' (a model observed variable)",
                       what, ov[static_cast<std::size_t>(k)]);
          perm[static_cast<std::size_t>(k)] = it->second;
        }
        return perm;
      }
    }
  }
  if (M.ncol() != p)
    Rcpp::stop("latva: %s has %d columns but the model has %d observed variables; "
               "give it column names to match by name", what, M.ncol(), p);
  for (int k = 0; k < p; ++k) perm[static_cast<std::size_t>(k)] = k;
  return perm;
}

inline Eigen::MatrixXd reorder_cov(Rcpp::NumericMatrix S, const std::vector<int>& perm) {
  if (S.nrow() != S.ncol())
    Rcpp::stop("latva: a sample covariance must be square (got %dx%d)", S.nrow(), S.ncol());
  const int p = static_cast<int>(perm.size());
  Eigen::MatrixXd out(p, p);
  for (int a = 0; a < p; ++a)
    for (int b = 0; b < p; ++b)
      out(a, b) = S(perm[static_cast<std::size_t>(a)], perm[static_cast<std::size_t>(b)]);
  return out;
}
inline Eigen::MatrixXd reorder_data_cols(Rcpp::NumericMatrix X, const std::vector<int>& perm) {
  const int p = static_cast<int>(perm.size());
  const int n = X.nrow();
  Eigen::MatrixXd out(n, p);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < p; ++k)
      out(i, k) = X(i, perm[static_cast<std::size_t>(k)]);
  return out;
}
inline Eigen::VectorXd reorder_vec(Rcpp::NumericVector v, const std::vector<int>& perm) {
  const int p = static_cast<int>(perm.size());
  if (static_cast<int>(v.size()) != p)
    Rcpp::stop("latva: a sample-mean vector has length %d but the block has %d variables",
               static_cast<int>(v.size()), p);
  Eigen::VectorXd out(p);
  for (int k = 0; k < p; ++k) out(k) = v[perm[static_cast<std::size_t>(k)]];
  return out;
}

// Pull the b-th per-block matrix out of `M`: when the model has a single
// block, `M` is a matrix; otherwise `M` is a length-n_blocks list of matrices.
// (Used for both per-group covariances and per-group raw-data matrices.)
inline Rcpp::NumericMatrix block_matrix(SEXP M, std::size_t b, std::size_t n_blocks,
                                        const char* what) {
  if (Rf_isMatrix(M)) {
    if (n_blocks != 1)
      Rcpp::stop("latva: the model has %d groups; pass a list of %d per-group %s matrices",
                 static_cast<int>(n_blocks), static_cast<int>(n_blocks), what);
    return Rcpp::NumericMatrix(M);
  }
  if (TYPEOF(M) != VECSXP)
    Rcpp::stop("latva: %s must be a matrix (single-group) or a list of per-group matrices "
               "(multi-group)", what);
  Rcpp::List Ml(M);
  if (static_cast<std::size_t>(Ml.size()) != n_blocks)
    Rcpp::stop("latva: %s is a list of %d matrices but the model has %d groups",
               what, static_cast<int>(Ml.size()), static_cast<int>(n_blocks));
  return Rcpp::NumericMatrix(Ml[static_cast<R_xlen_t>(b)]);
}

// build_matrix_rep + a (possibly multi-block) SampleStats from raw R parts.
// One uniform path — single-group is just n_blocks == 1.
//   S            : a covariance matrix (1 group) or a list of them.
//   nobs         : a scalar (1 group) or a length-n_groups numeric/integer vector.
//   sample_mean  : R_NilValue, a length-p vector (1 group), or a list of them.
//   reorder      : true → permute S / mean to the model's variable order by
//                  column name (the `make_ctx` entry point — user-supplied
//                  data); false → take them as already in model order (the
//                  fit-object entry point).
inline Ctx ctx_from_parts(lvp::ParTable pt, SEXP S, SEXP nobs, SEXP sample_mean,
                          bool reorder) {
  auto rep_or = lvm::build_matrix_rep(pt);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  lvm::MatrixRep rep = std::move(*rep_or);
  if (rep.ov_names.empty() || rep.ov_names[0].empty())
    Rcpp::stop("latva: model has no observed variables");
  const std::size_t n_blocks = rep.dims.size();

  Rcpp::IntegerVector nv = Rcpp::as<Rcpp::IntegerVector>(nobs);
  if (static_cast<std::size_t>(nv.size()) != n_blocks)
    Rcpp::stop("latva: nobs has length %d but the model has %d group(s)",
               static_cast<int>(nv.size()), static_cast<int>(n_blocks));

  const bool has_means = !Rf_isNull(sample_mean);
  Rcpp::List sml;
  if (has_means) {
    sml = (TYPEOF(sample_mean) == VECSXP)
              ? Rcpp::List(sample_mean)
              : Rcpp::List::create(Rcpp::NumericVector(sample_mean));
    if (static_cast<std::size_t>(sml.size()) != n_blocks)
      Rcpp::stop("latva: sample mean has %d entries but the model has %d group(s)",
                 static_cast<int>(sml.size()), static_cast<int>(n_blocks));
  }

  lvf::SampleStats samp;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    Rcpp::NumericMatrix Sb = block_matrix(S, b, n_blocks, "S");
    if (reorder) {
      const std::vector<int> perm = perm_for_cols(Sb, rep.ov_names[b], "S");
      samp.S.push_back(reorder_cov(Sb, perm));
      if (has_means)
        samp.mean.push_back(
            reorder_vec(Rcpp::NumericVector(sml[static_cast<R_xlen_t>(b)]), perm));
    } else {
      if (Sb.nrow() != Sb.ncol())
        Rcpp::stop("latva: a sample covariance must be square (got %dx%d)",
                   Sb.nrow(), Sb.ncol());
      samp.S.push_back(Rcpp::as<Eigen::MatrixXd>(Sb));
      if (has_means)
        samp.mean.push_back(Rcpp::as<Eigen::VectorXd>(
            Rcpp::NumericVector(sml[static_cast<R_xlen_t>(b)])));
    }
    samp.n_obs.push_back(static_cast<std::int64_t>(nv[static_cast<R_xlen_t>(b)]));
  }

  Ctx ctx;
  ctx.pt = std::move(pt);
  ctx.rep = std::move(rep);
  ctx.samp = std::move(samp);
  ctx.ov_names = ctx.rep.ov_names[0];
  ctx.meanstructure = has_meanstructure(ctx.pt);
  // The sample mean only enters the discrepancy / df when the model has `~1`
  // rows; drop it otherwise so a `sample_stats` bundle (which always carries a
  // mean) doesn't accidentally make a cov-only model count `p` extra moments.
  if (!ctx.meanstructure) ctx.samp.mean.clear();
  return ctx;
}

// User-supplied data: S/nobs/mean are reordered to the model's variable order.
inline Ctx make_ctx(lvp::ParTable pt, SEXP S, SEXP n, SEXP sample_mean) {
  return ctx_from_parts(std::move(pt), S, n, sample_mean, /*reorder=*/true);
}

inline lvp::ParTable partable_from_arg(SEXP partable, const char* fn,
                                       lvp::Starts* out_starts = nullptr) {
  if (TYPEOF(partable) == STRSXP)
    Rcpp::stop("latva: %s() takes a partable data.frame (e.g. from latva_lavaanify()), "
               "not a model-syntax string — call latva_lavaanify() first", fn);
  if (!Rf_inherits(partable, "data.frame"))
    Rcpp::stop("latva: %s(): `partable` must be a data.frame (e.g. from latva_lavaanify())", fn);
  return parse_partable_df(Rcpp::DataFrame(partable), out_starts);
}

// Pull the {S, nobs, mean} sample-stats bundle out of `latva_fit`'s `arg` —
// either the bundle `list(S = …, nobs = …, mean = …|NULL)` (the canonical
// form; `latva_sample_stats_from_raw()` returns exactly this) or, lenient
// for the hand-built single-group case, a bare covariance matrix is *not*
// accepted (nobs is required) — the caller passes a `list(S = , nobs = )`.
inline Ctx ctx_from_sample_stats(lvp::ParTable pt, Rcpp::List ss) {
  if (!ss.containsElementNamed("S") || !ss.containsElementNamed("nobs"))
    Rcpp::stop("latva: `sample_stats` must be a list with $S (a covariance matrix "
               "or list of them) and $nobs (a per-group n vector) — e.g. the result "
               "of latva_sample_stats_from_raw(), or list(S = , nobs = )");
  SEXP sm = ss.containsElementNamed("mean") ? SEXP(ss["mean"]) : R_NilValue;
  return ctx_from_parts(std::move(pt), ss["S"], ss["nobs"], sm, /*reorder=*/true);
}

// Rebuild a Ctx from a fit object. `fit$S` is a list of per-group covariances
// (length ≥ 1), already in the model's variable order; `fit$nobs` is the
// per-group n vector; `fit$sample_mean` is R_NilValue or a list of vectors.
inline Ctx ctx_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("partable") || !fit.containsElementNamed("S") ||
      !fit.containsElementNamed("nobs"))
    Rcpp::stop("latva: not a fit object (need partable/S/nobs) — pass the result of latva_fit()");
  SEXP sm = fit.containsElementNamed("sample_mean") ? SEXP(fit["sample_mean"]) : R_NilValue;
  return ctx_from_parts(parse_partable_df(Rcpp::DataFrame(fit["partable"])),
                        fit["S"], fit["nobs"], sm, /*reorder=*/false);
}

inline lvf::Estimates est_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("theta"))
    Rcpp::stop("latva: not a fit object (missing theta) — pass the result of latva_fit()");
  lvf::Estimates e;
  e.theta = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(fit["theta"]));
  e.fmin = fit.containsElementNamed("fmin") ? Rcpp::as<double>(fit["fmin"])
                                            : std::numeric_limits<double>::quiet_NaN();
  e.iterations = fit.containsElementNamed("iterations") ? Rcpp::as<int>(fit["iterations"]) : 0;
  return e;
}

inline Rcpp::List inference_to_list(const lvf::Inference& inf) {
  return Rcpp::List::create(
      Rcpp::_["se"]   = Rcpp::wrap(inf.se),
      Rcpp::_["vcov"] = Rcpp::wrap(inf.vcov),
      Rcpp::_["info"] = Rcpp::wrap(inf.info),
      Rcpp::_["chi2"] = inf.chi2,
      Rcpp::_["df"]   = inf.df);
}

}  // namespace latvar
