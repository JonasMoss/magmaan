// Shared internal plumbing for the magmaan R bindings. Header-only helpers in
// namespace `magmaanr` — no // [[Rcpp::export]] here. Included by fit.cpp and
// robust.cpp; bindings.cpp (parse/lavaanify only) does not need it.

#pragma once

#include <RcppEigen.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"
#include "magmaan/spec/build.hpp"     // compute_eq_groups
#include "magmaan/compat/lavaan/composite_fold.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"   // LavaanParTable / to_/from_lavaan_partable
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/fit.hpp"              // Estimates
#include "magmaan/estimate/fiml.hpp"             // SaturatedMoments
#include "magmaan/estimate/backend_strings.hpp"  // backend_from_string
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/inference/inference.hpp"        // information_*, vcov, se, chi2_stat, df_stat
#include "magmaan/robust/robust.hpp"              // InferenceSpec (bread/moments/cov)
#include "magmaan/robust/weighted_inference.hpp"   // WeightedProfileLRTResult, ScaledShiftedResult
#include "magmaan/optim/problem.hpp"  // OptimOptions

namespace lv  = magmaan;
namespace lvm = magmaan::model;

namespace magmaanr {

inline magmaan::estimate::OrdinalParameterization
ordinal_parameterization_from_string(const std::string& s) {
  if (s == "theta" || s == "Theta" || s == "THETA") {
    return magmaan::estimate::OrdinalParameterization::Theta;
  }
  if (s == "delta" || s == "Delta" || s == "DELTA" || s.empty()) {
    return magmaan::estimate::OrdinalParameterization::Delta;
  }
  Rcpp::stop("magmaan: ordinal parameterization must be 'delta' or 'theta' (got '%s')", s);
}

inline std::string ordinal_parameterization_attr(SEXP x) {
  SEXP attr = Rf_getAttrib(x, Rf_install("magmaan.parameterization"));
  if (attr == R_NilValue || Rf_length(attr) < 1) return "delta";
  return Rcpp::as<std::string>(Rcpp::CharacterVector(attr)[0]);
}

// Recover the `group.equal` families (lavaan_lavaanify stamps them as enum
// indices on the partable df) so the ordinal fit can re-apply the fit-time
// Wu-Estabrook release that `from_lavaan_partable` cannot reconstruct. Empty
// when absent — the common single-group / no-invariance case.
inline std::vector<magmaan::spec::GroupEqual> group_equal_attr(SEXP x) {
  SEXP attr = Rf_getAttrib(x, Rf_install("magmaan.group_equal"));
  if (attr == R_NilValue || Rf_length(attr) < 1) return {};
  Rcpp::IntegerVector idx(attr);
  std::vector<magmaan::spec::GroupEqual> out;
  out.reserve(static_cast<std::size_t>(idx.size()));
  for (R_xlen_t i = 0; i < idx.size(); ++i)
    out.push_back(static_cast<magmaan::spec::GroupEqual>(idx[i]));
  return out;
}

// Stamp the `group.equal` families as enum indices on a partable df, matching
// what lavaan_lavaanify writes, so a fit result carries them forward to the
// nested ordinal LR test (which rebuilds each structure from its partable).
inline void stamp_group_equal_attr(
    SEXP target, const std::vector<magmaan::spec::GroupEqual>& ge) {
  if (ge.empty()) return;
  Rcpp::IntegerVector v(static_cast<R_xlen_t>(ge.size()));
  for (std::size_t i = 0; i < ge.size(); ++i)
    v[static_cast<R_xlen_t>(i)] = static_cast<int>(ge[i]);
  Rf_setAttrib(target, Rf_install("magmaan.group_equal"), v);
}

// ---- robust InferenceSpec enums <-> strings (shared by fit.cpp / robust.cpp) -

inline magmaan::robust::Information info_from_string(const std::string& s) {
  if (s == "expected") return magmaan::robust::Information::Expected;
  if (s == "observed") return magmaan::robust::Information::Observed;
  Rcpp::stop("magmaan: `bread` must be 'expected' or 'observed' (got '%s')", s);
}
inline magmaan::robust::WeightMoments moments_from_string(const std::string& s) {
  if (s == "structured")   return magmaan::robust::WeightMoments::Structured;
  if (s == "unstructured") return magmaan::robust::WeightMoments::Unstructured;
  if (s == "pairwise")     return magmaan::robust::WeightMoments::Pairwise;
  Rcpp::stop("magmaan: `moments` must be 'structured', 'unstructured', or 'pairwise' (got '%s')", s);
}
inline magmaan::robust::ScoreCovariance cov_from_string(const std::string& s) {
  if (s == "model_implied")   return magmaan::robust::ScoreCovariance::ModelImplied;
  if (s == "empirical")       return magmaan::robust::ScoreCovariance::Empirical;
  if (s == "browne_unbiased") return magmaan::robust::ScoreCovariance::BrowneUnbiased;
  Rcpp::stop("magmaan: `cov` must be 'model_implied', 'empirical', or 'browne_unbiased' (got '%s')", s);
}
inline magmaan::robust::InferenceSpec spec_from(const std::string& bread,
                                                const std::string& moments) {
  magmaan::robust::InferenceSpec s;
  s.bread = info_from_string(bread);
  s.moments = moments_from_string(moments);
  return s;
}
inline magmaan::robust::InferenceSpec spec_from(const std::string& bread,
                                                const std::string& moments,
                                                const std::string& cov) {
  magmaan::robust::InferenceSpec s = spec_from(bread, moments);
  s.cov = cov_from_string(cov);
  return s;
}

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
  Rcpp::stop("magmaan model error [%s]: %s", model_error_kind(e.kind), e.detail);
}
[[noreturn]] inline void stop_fit(const lv::FitError& e) {
  Rcpp::stop("magmaan fit error [%s] after %d iters, f=%g: %s", fit_error_kind(e.kind),
             e.iterations, e.f_value, e.detail);
}
[[noreturn]] inline void stop_post(const lv::PostError& e) {
  Rcpp::stop("magmaan inference error [%s]: %s", post_error_kind(e.kind), e.detail);
}

// ---- partable <-> data.frame ------------------------------------------------

inline lv::parse::Op op_from_string(const std::string& s) {
  using O = lv::parse::Op;
  if (s == "=~") return O::Measurement;
  if (s == "<~") return O::Composite;
  if (s == "~")  return O::Regression;
  if (s == "~~") return O::Covariance;
  if (s == "|")  return O::Threshold;
  if (s == "~*~") return O::ResponseScale;
  if (s == "~1") return O::Intercept;
  if (s == ":=") return O::DefineParam;
  if (s == "==") return O::EqConstraint;
  if (s == "<")  return O::LtConstraint;
  if (s == ">")  return O::GtConstraint;
  Rcpp::stop("magmaan: unrecognized operator '%s' in partable data.frame", s);
}

// Group identity (n_groups, group_var, group_labels) rides on the partable
// data.frame as attributes (`magmaan.group_var` / `magmaan.group_labels`) — it's
// table-level, not per-row. `attach_group_attrs` sets them; `read_group_attrs`
// reads them back (tolerating absence — `n_groups()` still works off `group`).
inline void attach_group_attrs(SEXP df, const std::string& group_var,
                               const std::vector<std::string>& group_labels) {
  Rf_setAttrib(df, Rf_install("magmaan.group_var"), Rf_mkString(group_var.c_str()));
  Rcpp::CharacterVector gl(static_cast<R_xlen_t>(group_labels.size()));
  for (std::size_t i = 0; i < group_labels.size(); ++i)
    gl[static_cast<R_xlen_t>(i)] = group_labels[i];
  Rf_setAttrib(df, Rf_install("magmaan.group_labels"), gl);
}
inline void read_group_attrs(SEXP df, std::string& group_var,
                             std::vector<std::string>& group_labels) {
  SEXP gv = Rf_getAttrib(df, Rf_install("magmaan.group_var"));
  if (!Rf_isNull(gv) && TYPEOF(gv) == STRSXP && Rf_length(gv) >= 1)
    group_var = Rcpp::as<std::string>(STRING_ELT(gv, 0));
  SEXP gl = Rf_getAttrib(df, Rf_install("magmaan.group_labels"));
  if (!Rf_isNull(gl) && TYPEOF(gl) == STRSXP)
    group_labels = Rcpp::as<std::vector<std::string>>(gl);
}

constexpr const char* expanded_partable_attr = "magmaan.expanded_partable";
constexpr const char* native_fcsem_partable_attr = "magmaan.fcsem";

inline bool is_true_attr(SEXP x, const char* name) {
  SEXP attr = Rf_getAttrib(x, Rf_install(name));
  return !Rf_isNull(attr) && Rf_length(attr) > 0 && Rf_asLogical(attr) == TRUE;
}

inline bool has_composite_rows(Rcpp::DataFrame df) {
  if (!df.containsElementNamed("op")) return false;
  std::vector<std::string> opstr =
      Rcpp::as<std::vector<std::string>>(static_cast<SEXP>(df["op"]));
  for (const std::string& op : opstr)
    if (op == "<~") return true;
  return false;
}

inline std::vector<magmaan::spec::CompositeInfo>
composite_info_from_folded_df(Rcpp::DataFrame df) {
  std::vector<magmaan::spec::CompositeInfo> out;
  if (!df.containsElementNamed("lhs") || !df.containsElementNamed("op") ||
      !df.containsElementNamed("rhs")) {
    return out;
  }

  std::vector<std::string> lhs =
      Rcpp::as<std::vector<std::string>>(static_cast<SEXP>(df["lhs"]));
  std::vector<std::string> op =
      Rcpp::as<std::vector<std::string>>(static_cast<SEXP>(df["op"]));
  std::vector<std::string> rhs =
      Rcpp::as<std::vector<std::string>>(static_cast<SEXP>(df["rhs"]));

  auto find_composite = [&](const std::string& name) -> std::size_t {
    for (std::size_t i = 0; i < out.size(); ++i)
      if (out[i].composite == name) return i;
    out.push_back(magmaan::spec::CompositeInfo{});
    out.back().composite = name;
    return out.size() - 1;
  };
  for (std::size_t i = 0; i < op.size(); ++i) {
    if (op[i] != "<~") continue;
    const std::size_t k = find_composite(lhs[i]);
    bool seen = false;
    for (const std::string& ind : out[k].indicators) {
      if (ind == rhs[i]) { seen = true; break; }
    }
    if (!seen) out[k].indicators.push_back(rhs[i]);
  }
  for (auto& ci : out) {
    for (std::size_t j = 1; j < ci.indicators.size(); ++j) {
      ci.excrescent.push_back(".exc." + ci.composite + "." + std::to_string(j));
    }
  }
  return out;
}

// Parse a (possibly hand-edited) lavaan-shaped partable data.frame back into
// the in-memory model triple: read the columns into a `LavaanParTable`, then
// `from_lavaan_partable` re-derives the variable inventory + equality groups +
// `LatentNames`. The data.frame's `ustart` column is split inside
// `from_lavaan_partable` (a fixed row's value → `fixed_value`; a free row's
// value → a start hint on `.starts`).
inline magmaan::compat::lavaan::ParsedLavaanParTable parse_partable_df(Rcpp::DataFrame df) {
  std::vector<magmaan::spec::CompositeInfo> composites =
      composite_info_from_folded_df(df);
  SEXP expanded = Rf_getAttrib(df, Rf_install(expanded_partable_attr));
  if (!Rf_isNull(expanded)) {
    if (!Rf_isFrame(expanded)) {
      Rcpp::stop("magmaan: `%s` attribute must be a data.frame",
                 expanded_partable_attr);
    }
    df = Rcpp::DataFrame(expanded);
  } else if (has_composite_rows(df)) {
    Rcpp::stop("magmaan: folded composite (`<~`) partables need the `%s` "
               "attribute; rebuild the partable with lavaan_lavaanify() or "
               "use the original magmaan fit object",
               expanded_partable_attr);
  }

  auto col = [&](const char* nm) -> SEXP {
    if (!df.containsElementNamed(nm))
      Rcpp::stop("magmaan: partable data.frame is missing required column '%s'", nm);
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
  magmaan::compat::lavaan::LavaanParTable lvpt;
  lvpt.id.resize(m); lvpt.user.resize(m); lvpt.lhs.resize(m); lvpt.op.resize(m);
  lvpt.rhs.resize(m); lvpt.block.resize(m); lvpt.group.resize(m); lvpt.free.resize(m);
  lvpt.exo.resize(m); lvpt.ustart.resize(m); lvpt.label.resize(m); lvpt.plabel.resize(m);
  for (std::size_t i = 0; i < m; ++i) {
    const R_xlen_t ri = static_cast<R_xlen_t>(i);
    lvpt.id[i]     = id[ri];
    lvpt.user[i]   = static_cast<std::int8_t>(user[ri]);
    lvpt.lhs[i]    = lhs[i];
    lvpt.op[i]     = op_from_string(opstr[i]);
    lvpt.rhs[i]    = rhs[i];
    lvpt.block[i]  = block[ri];
    lvpt.group[i]  = group[ri];
    lvpt.free[i]   = freev[ri];
    lvpt.exo[i]    = static_cast<std::int8_t>(exo[ri]);
    lvpt.ustart[i] = ustart[ri];   // NaN passes through
    lvpt.label[i]  = label[i];
    lvpt.plabel[i] = plab[i];
  }
  read_group_attrs(df, lvpt.group_var, lvpt.group_labels);
  auto parsed = magmaan::compat::lavaan::from_lavaan_partable(lvpt);
  if (!composites.empty()) parsed.names.composites = std::move(composites);
  return parsed;
}

inline bool has_meanstructure(const magmaan::spec::LatentStructure& pt) {
  for (lv::parse::Op op : pt.op)
    if (op == lv::parse::Op::Intercept) return true;
  return false;
}
inline bool has_constraint_rows(const magmaan::spec::LatentStructure& pt) {
  using O = lv::parse::Op;
  for (lv::parse::Op op : pt.op)
    if (op == O::EqConstraint || op == O::DefineParam || op == O::LtConstraint ||
        op == O::GtConstraint)
      return true;
  return false;
}

inline Rcpp::DataFrame partable_df_from_lavaan(
    const magmaan::compat::lavaan::LavaanParTable& pt,
    const magmaan::estimate::Estimates* est = nullptr) {
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
    ustart[i] = pt.ustart[k];
    const std::int32_t f = pt.free[k];
    est_c[i]  = (est == nullptr || f <= 0) ? pt.ustart[k]
                                           : est->theta(static_cast<Eigen::Index>(f) - 1);
  }
  Rcpp::List cols = Rcpp::List::create(
      Rcpp::_["id"] = id, Rcpp::_["lhs"] = lhs, Rcpp::_["op"] = op, Rcpp::_["rhs"] = rhs,
      Rcpp::_["user"] = user, Rcpp::_["block"] = block, Rcpp::_["group"] = group,
      Rcpp::_["free"] = freev, Rcpp::_["exo"] = exo, Rcpp::_["ustart"] = ustart,
      Rcpp::_["label"] = label, Rcpp::_["plabel"] = plabel, Rcpp::_["est"] = est_c);
  cols.attr("row.names") = Rcpp::IntegerVector::create(NA_INTEGER, -static_cast<int>(nrow));
  cols.attr("class") = "data.frame";
  Rcpp::DataFrame out(cols);
  attach_group_attrs(out, pt.group_var, pt.group_labels);
  return out;
}

// The model triple + fitted estimates -> data.frame (lavaanify columns + est).
// `starts` (optional): the user start hints, so the reconstructed `ustart`
// column carries free-row hints (not just fixed-row values). NaN ⇒ R NA.
inline Rcpp::DataFrame partable_df(const magmaan::spec::LatentStructure& structure,
                                   const magmaan::spec::LatentNames& names,
                                   const magmaan::estimate::Estimates& est,
                                   const magmaan::spec::Starts* starts = nullptr) {
  const magmaan::compat::lavaan::LavaanParTable expanded =
      magmaan::compat::lavaan::to_lavaan_partable(
          structure, names, starts ? *starts : magmaan::spec::Starts{});
  Rcpp::DataFrame expanded_df = partable_df_from_lavaan(expanded, &est);
  if (names.composites.empty()) return expanded_df;

  const magmaan::compat::lavaan::LavaanParTable folded =
      magmaan::compat::lavaan::fold_composites(expanded, names.composites);
  Rcpp::DataFrame folded_df = partable_df_from_lavaan(folded, &est);
  Rf_setAttrib(folded_df, Rf_install(expanded_partable_attr), expanded_df);
  return folded_df;
}

inline magmaan::optim::OptimOptions optim_opts_from(Rcpp::Nullable<Rcpp::List> control) {
  magmaan::optim::OptimOptions o;  // struct defaults
  if (control.isNotNull()) {
    Rcpp::List l(control.get());
    if (l.containsElementNamed("max_iter")) o.max_iter = Rcpp::as<int>(l["max_iter"]);
    if (l.containsElementNamed("ftol"))     o.ftol     = Rcpp::as<double>(l["ftol"]);
    if (l.containsElementNamed("gtol"))     o.gtol     = Rcpp::as<double>(l["gtol"]);
    if (l.containsElementNamed("history"))  o.history  = Rcpp::as<int>(l["history"]);
  }
  return o;
}

// Pulls IRLS outer-loop knobs out of the same `control` list `optim_opts_from`
// parses for the inner solver. The `irls_` prefix keeps the inner-solver
// `max_iter` / `ftol` / `gtol` namespace clean — the two layers share a
// control list but not field names.
inline magmaan::estimate::IrlsOptions
irls_opts_from(Rcpp::Nullable<Rcpp::List> control) {
  magmaan::estimate::IrlsOptions o;  // struct defaults
  if (control.isNotNull()) {
    Rcpp::List l(control.get());
    if (l.containsElementNamed("irls_max_outer"))
      o.max_outer = Rcpp::as<int>(l["irls_max_outer"]);
    if (l.containsElementNamed("irls_ftol"))
      o.ftol = Rcpp::as<double>(l["irls_ftol"]);
    if (l.containsElementNamed("irls_gtol"))
      o.gtol = Rcpp::as<double>(l["irls_gtol"]);
    if (l.containsElementNamed("irls_armijo_c"))
      o.armijo_c = Rcpp::as<double>(l["irls_armijo_c"]);
    if (l.containsElementNamed("irls_armijo_max_backtracks"))
      o.armijo_max_backtracks =
          Rcpp::as<int>(l["irls_armijo_max_backtracks"]);
  }
  return o;
}

// Parse the user-facing `optimizer = "..."` string into the C++ Backend
// enum. Empty / NULL ⇒ default "nlopt-lbfgs". Unknown strings are surfaced as a
// magmaan-style Rcpp::stop with the accepted-names list — same wording as
// the C++ backend_from_string error, so users typing `optimizer = "ceres-lm"`
// get a clear "did you mean..." correction.
inline magmaan::estimate::Backend
backend_from_optimizer_arg(Rcpp::Nullable<Rcpp::String> optimizer) {
  if (optimizer.isNull()) return magmaan::estimate::Backend::NloptLbfgs;
  const std::string name = Rcpp::as<std::string>(optimizer.get());
  if (name.empty()) return magmaan::estimate::Backend::NloptLbfgs;
  auto b_or = magmaan::estimate::backend_from_string(name);
  if (!b_or.has_value()) Rcpp::stop("magmaan: " + b_or.error().detail);
  return *b_or;
}

// ---- model context ----------------------------------------------------------

struct Ctx {
  magmaan::spec::LatentStructure     pt;
  magmaan::spec::LatentNames         names;
  lvm::MatrixRep           rep;
  magmaan::data::SampleStats         samp;
  std::vector<std::string> ov_names;     // observed-variable order magmaan uses (block 0)
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
            Rcpp::stop("magmaan: %s has no column named '%s' (a model observed variable)",
                       what, ov[static_cast<std::size_t>(k)]);
          perm[static_cast<std::size_t>(k)] = it->second;
        }
        return perm;
      }
    }
  }
  if (M.ncol() != p)
    Rcpp::stop("magmaan: %s has %d columns but the model has %d observed variables; "
               "give it column names to match by name", what, M.ncol(), p);
  for (int k = 0; k < p; ++k) perm[static_cast<std::size_t>(k)] = k;
  return perm;
}

inline Eigen::MatrixXd reorder_cov(Rcpp::NumericMatrix S, const std::vector<int>& perm) {
  if (S.nrow() != S.ncol())
    Rcpp::stop("magmaan: a sample covariance must be square (got %dx%d)", S.nrow(), S.ncol());
  const int p = static_cast<int>(perm.size());
  if (S.nrow() != p)
    Rcpp::stop("magmaan: a sample covariance is %dx%d but the block has %d variables",
               S.nrow(), S.ncol(), p);
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
    Rcpp::stop("magmaan: a sample-mean vector has length %d but the block has %d variables",
               static_cast<int>(v.size()), p);
  Eigen::VectorXd out(p);
  for (int k = 0; k < p; ++k) out(k) = v[perm[static_cast<std::size_t>(k)]];
  return out;
}

inline void validate_finite_matrix(const Eigen::MatrixXd& M, const char* what,
                                   std::size_t block) {
  for (Eigen::Index c = 0; c < M.cols(); ++c) {
    for (Eigen::Index r = 0; r < M.rows(); ++r) {
      if (!std::isfinite(M(r, c)))
        Rcpp::stop("magmaan: %s for block %d contains a non-finite value at [%d,%d]",
                   what, static_cast<int>(block), static_cast<int>(r + 1),
                   static_cast<int>(c + 1));
    }
  }
}

inline void validate_finite_vector(const Eigen::VectorXd& v, const char* what,
                                   std::size_t block) {
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    if (!std::isfinite(v(i)))
      Rcpp::stop("magmaan: %s for block %d contains a non-finite value at [%d]",
                 what, static_cast<int>(block), static_cast<int>(i + 1));
  }
}

// Pull the b-th per-block matrix out of `M`: when the model has a single
// block, `M` is a matrix; otherwise `M` is a length-n_blocks list of matrices.
// (Used for both per-group covariances and per-group raw-data matrices.)
inline Rcpp::NumericMatrix block_matrix(SEXP M, std::size_t b, std::size_t n_blocks,
                                        const char* what) {
  if (Rf_isMatrix(M)) {
    if (n_blocks != 1)
      Rcpp::stop("magmaan: the model has %d groups; pass a list of %d per-group %s matrices",
                 static_cast<int>(n_blocks), static_cast<int>(n_blocks), what);
    return Rcpp::NumericMatrix(M);
  }
  if (TYPEOF(M) != VECSXP)
    Rcpp::stop("magmaan: %s must be a matrix (single-group) or a list of per-group matrices "
               "(multi-group)", what);
  Rcpp::List Ml(M);
  if (static_cast<std::size_t>(Ml.size()) != n_blocks)
    Rcpp::stop("magmaan: %s is a list of %d matrices but the model has %d groups",
               what, static_cast<int>(Ml.size()), static_cast<int>(n_blocks));
  return Rcpp::NumericMatrix(Ml[static_cast<R_xlen_t>(b)]);
}

// Counterpart to `block_matrix` for boolean masks; same single-vs-multi-group
// shape rules. Used by raw-data converters that need to thread missingness
// indicators through alongside `X`.
inline Rcpp::LogicalMatrix block_mask_matrix(SEXP M, std::size_t b, std::size_t n_blocks,
                                             const char* what) {
  if (Rf_isMatrix(M)) {
    if (n_blocks != 1)
      Rcpp::stop("magmaan: the model has %d groups; pass a list of %d per-group %s matrices",
                 static_cast<int>(n_blocks), static_cast<int>(n_blocks), what);
    return Rcpp::LogicalMatrix(M);
  }
  if (TYPEOF(M) != VECSXP)
    Rcpp::stop("magmaan: %s must be a logical matrix (single-group) or a list of "
               "per-group logical matrices", what);
  Rcpp::List Ml(M);
  if (static_cast<std::size_t>(Ml.size()) != n_blocks)
    Rcpp::stop("magmaan: %s is a list of %d matrices but the model has %d groups",
               what, static_cast<int>(Ml.size()), static_cast<int>(n_blocks));
  return Rcpp::LogicalMatrix(Ml[static_cast<R_xlen_t>(b)]);
}

// Build a (possibly multi-block) RawData from a plain `X` / optional `mask`
// argument pair — no model/partable needed. Used by the methods-developer
// surfaces (saturated EM, pairwise sample stats, pairwise GLS) where the
// estimator has no structural restrictions and the caller may not have a
// partable on hand. `X_arg` is a NumericMatrix (single block) or a list of
// per-group NumericMatrix; `mask_arg` is optional (NilValue ⇒ auto-detect
// NA via std::isfinite).
inline magmaan::data::RawData
raw_from_data_args(SEXP X_arg, SEXP mask_arg) {
  const std::size_t n_blocks = TYPEOF(X_arg) == VECSXP
      ? static_cast<std::size_t>(Rcpp::List(X_arg).size())
      : 1u;
  if (n_blocks == 0) Rcpp::stop("magmaan: data has no blocks");

  magmaan::data::RawData raw;
  raw.X.reserve(n_blocks);
  bool any_missing = false;
  std::vector<Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>> masks;
  masks.reserve(n_blocks);
  const bool has_mask = !Rf_isNull(mask_arg);

  for (std::size_t b = 0; b < n_blocks; ++b) {
    Rcpp::NumericMatrix Xb = block_matrix(X_arg, b, n_blocks, "data$X");
    const int n = Xb.nrow();
    const int p = Xb.ncol();
    Eigen::MatrixXd X(n, p);
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);

    Rcpp::LogicalMatrix Mb;
    if (has_mask) {
      Mb = block_mask_matrix(mask_arg, b, n_blocks, "data$mask");
      if (Mb.nrow() != n || Mb.ncol() != p)
        Rcpp::stop("magmaan: data$mask block %d has shape %dx%d but data$X has %dx%d",
                   static_cast<int>(b + 1), Mb.nrow(), Mb.ncol(), n, p);
    }
    for (int r = 0; r < n; ++r) {
      for (int k = 0; k < p; ++k) {
        const double x = Xb(r, k);
        const bool observed = has_mask
            ? (Mb(r, k) != NA_LOGICAL && Mb(r, k) != 0)
            : std::isfinite(x);
        if (observed && !std::isfinite(x)) {
          Rcpp::stop("magmaan: data$mask marks a non-finite value as observed "
                     "in block %d, row %d", static_cast<int>(b + 1), r + 1);
        }
        M(r, k) = static_cast<std::uint8_t>(observed ? 1 : 0);
        X(r, k) = observed ? x : std::numeric_limits<double>::quiet_NaN();
        if (!observed) any_missing = true;
      }
    }
    raw.X.push_back(std::move(X));
    masks.push_back(std::move(M));
  }
  if (any_missing) raw.mask = std::move(masks);
  return raw;
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
inline Ctx ctx_from_parts(magmaan::spec::LatentStructure pt, magmaan::spec::LatentNames names,
                          SEXP S, SEXP nobs, SEXP sample_mean, bool reorder) {
  auto rep_or = lvm::build_matrix_rep(pt, &names);
  if (!rep_or.has_value()) stop_model(rep_or.error());
  lvm::MatrixRep rep = std::move(*rep_or);
  if (rep.ov_names.empty() || rep.ov_names[0].empty())
    Rcpp::stop("magmaan: model has no observed variables");
  const std::size_t n_blocks = rep.dims.size();

  Rcpp::IntegerVector nv = Rcpp::as<Rcpp::IntegerVector>(nobs);
  if (static_cast<std::size_t>(nv.size()) != n_blocks)
    Rcpp::stop("magmaan: nobs has length %d but the model has %d group(s)",
               static_cast<int>(nv.size()), static_cast<int>(n_blocks));

  const bool has_means = !Rf_isNull(sample_mean);
  Rcpp::List sml;
  if (has_means) {
    sml = (TYPEOF(sample_mean) == VECSXP)
              ? Rcpp::List(sample_mean)
              : Rcpp::List::create(Rcpp::NumericVector(sample_mean));
    if (static_cast<std::size_t>(sml.size()) != n_blocks)
      Rcpp::stop("magmaan: sample mean has %d entries but the model has %d group(s)",
                 static_cast<int>(sml.size()), static_cast<int>(n_blocks));
  }

  magmaan::data::SampleStats samp;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const int nb = nv[static_cast<R_xlen_t>(b)];
    if (nb == NA_INTEGER || nb <= 0)
      Rcpp::stop("magmaan: nobs for block %d must be a positive integer",
                 static_cast<int>(b));

    Rcpp::NumericMatrix Sb = block_matrix(S, b, n_blocks, "S");
    const int p = static_cast<int>(rep.ov_names[b].size());
    if (reorder) {
      const std::vector<int> perm = perm_for_cols(Sb, rep.ov_names[b], "S");
      samp.S.push_back(reorder_cov(Sb, perm));
      if (has_means) {
        SEXP mb = sml[static_cast<R_xlen_t>(b)];
        if (Rf_isNull(mb))
          Rcpp::stop("magmaan: sample mean for block %d is NULL; omit `mean` entirely "
                     "when means are unavailable", static_cast<int>(b));
        samp.mean.push_back(reorder_vec(Rcpp::NumericVector(mb), perm));
      }
    } else {
      if (Sb.nrow() != Sb.ncol())
        Rcpp::stop("magmaan: a sample covariance must be square (got %dx%d)",
                   Sb.nrow(), Sb.ncol());
      if (Sb.nrow() != p)
        Rcpp::stop("magmaan: a sample covariance for block %d is %dx%d but "
                   "the model block has %d observed variables",
                   static_cast<int>(b), Sb.nrow(), Sb.ncol(), p);
      samp.S.push_back(Rcpp::as<Eigen::MatrixXd>(Sb));
      if (has_means) {
        SEXP mb = sml[static_cast<R_xlen_t>(b)];
        if (Rf_isNull(mb))
          Rcpp::stop("magmaan: sample mean for block %d is NULL; omit `mean` entirely "
                     "when means are unavailable", static_cast<int>(b));
        Rcpp::NumericVector mv(mb);
        if (mv.size() != p)
          Rcpp::stop("magmaan: a sample-mean vector for block %d has length %d "
                     "but the block has %d variables",
                     static_cast<int>(b), static_cast<int>(mv.size()), p);
        samp.mean.push_back(Rcpp::as<Eigen::VectorXd>(mv));
      }
    }
    validate_finite_matrix(samp.S.back(), "sample covariance", b);
    if (has_means) validate_finite_vector(samp.mean.back(), "sample mean", b);
    samp.n_obs.push_back(static_cast<std::int64_t>(nb));
  }

  Ctx ctx;
  ctx.pt = std::move(pt);
  ctx.names = std::move(names);
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
inline Ctx make_ctx(magmaan::spec::LatentStructure pt, magmaan::spec::LatentNames names,
                    SEXP S, SEXP n, SEXP sample_mean) {
  return ctx_from_parts(std::move(pt), std::move(names), S, n, sample_mean,
                        /*reorder=*/true);
}

inline magmaan::compat::lavaan::ParsedLavaanParTable partable_from_arg(SEXP partable, const char* fn) {
  if (TYPEOF(partable) == STRSXP)
    Rcpp::stop("magmaan: %s() takes a partable data.frame (e.g. from lavaan_lavaanify()), "
               "not a model-syntax string — call lavaan_lavaanify() first", fn);
  if (!Rf_inherits(partable, "data.frame"))
    Rcpp::stop("magmaan: %s(): `partable` must be a data.frame (e.g. from lavaan_lavaanify())", fn);
  if (is_true_attr(partable, native_fcsem_partable_attr)) {
    Rcpp::stop("magmaan: %s(): ordinary SEM helpers do not accept native FC-SEM "
               "partables; use fit_ml_fcsem() or magmaan_fcsem()", fn);
  }
  return parse_partable_df(Rcpp::DataFrame(partable));
}

// Pull the {S, nobs, mean} sample-stats bundle out of `fit_fit`'s `arg` —
// either the bundle `list(S = …, nobs = …, mean = …|NULL)` (the canonical
// form; `data_sample_stats_from_raw()` returns exactly this) or, lenient
// for the hand-built single-group case, a bare covariance matrix is *not*
// accepted (nobs is required) — the caller passes a `list(S = , nobs = )`.
inline Ctx ctx_from_sample_stats(magmaan::spec::LatentStructure pt, magmaan::spec::LatentNames names,
                                 Rcpp::List ss) {
  if (!ss.containsElementNamed("S") || !ss.containsElementNamed("nobs"))
    Rcpp::stop("magmaan: `sample_stats` must be a list with $S (a covariance matrix "
               "or list of them) and $nobs (a per-group n vector) — e.g. the result "
               "of data_sample_stats_from_raw(), or list(S = , nobs = )");
  SEXP sm = ss.containsElementNamed("mean") ? SEXP(ss["mean"]) : R_NilValue;
  return ctx_from_parts(std::move(pt), std::move(names), ss["S"], ss["nobs"], sm,
                        /*reorder=*/true);
}

inline Ctx ctx_from_partable_sample_stats(SEXP partable, Rcpp::List sample_stats,
                                          const char* fn) {
  auto parsed = partable_from_arg(partable, fn);
  return ctx_from_sample_stats(std::move(parsed.structure), std::move(parsed.names),
                               sample_stats);
}

// Rebuild a Ctx from a fit object. `fit$S` is a list of per-group covariances
// (length ≥ 1), already in the model's variable order; `fit$nobs` is the
// per-group n vector; `fit$sample_mean` is R_NilValue or a list of vectors.
inline Ctx ctx_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("partable") || !fit.containsElementNamed("S") ||
      !fit.containsElementNamed("nobs"))
    Rcpp::stop("magmaan: not a fit object (need partable/S/nobs) — pass the result of fit_fit()");
  SEXP sm = fit.containsElementNamed("sample_mean") ? SEXP(fit["sample_mean"]) : R_NilValue;
  SEXP partable = fit["partable"];
  if (is_true_attr(partable, native_fcsem_partable_attr)) {
    Rcpp::stop("magmaan: ordinary SEM post-fit helpers do not accept native "
               "FC-SEM fits; use fcsem_standard_errors(), fcsem_fit_measures(), "
               "or fcsem_standardized_rows()");
  }
  auto parsed = parse_partable_df(Rcpp::DataFrame(partable));
  Ctx ctx = ctx_from_parts(std::move(parsed.structure), std::move(parsed.names),
                           fit["S"], fit["nobs"], sm, /*reorder=*/false);
  // `from_lavaan_partable` cannot recover group.equal; re-attach it so the
  // nested ordinal LR test's prepare_ordinal re-prep applies the same release
  // the fit did (otherwise a released H0 collapses back to the pinned scale).
  ctx.pt.group_equal = group_equal_attr(partable);
  return ctx;
}

inline magmaan::estimate::Estimates est_from_fit(Rcpp::List fit) {
  if (!fit.containsElementNamed("theta"))
    Rcpp::stop("magmaan: not a fit object (missing theta) — pass the result of fit_fit()");
  magmaan::estimate::Estimates e;
  e.theta = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(fit["theta"]));
  e.fmin = fit.containsElementNamed("fmin") ? Rcpp::as<double>(fit["fmin"])
                                            : std::numeric_limits<double>::quiet_NaN();
  e.iterations = fit.containsElementNamed("iterations") ? Rcpp::as<int>(fit["iterations"]) : 0;
  return e;
}

inline magmaan::estimate::Estimates est_from_theta(Rcpp::NumericVector theta) {
  magmaan::estimate::Estimates e;
  e.theta = Rcpp::as<Eigen::VectorXd>(theta);
  e.fmin = std::numeric_limits<double>::quiet_NaN();
  e.iterations = 0;
  return e;
}

inline magmaan::data::MixedOrdinalStats mixed_ordinal_stats_from_arg(Rcpp::List x) {
  const char* what = "mixed_ordinal_stats";
  for (const char* nm : {"R", "mean", "ordered_mask", "thresholds",
                         "threshold_ov", "threshold_level", "moments",
                         "NACOV", "W_dwls", "W_wls", "nobs", "n_levels"}) {
    if (!x.containsElementNamed(nm)) Rcpp::stop("magmaan: %s is missing $%s", what, nm);
  }
  Rcpp::List Rl(x["R"]), meanl(x["mean"]), ordl(x["ordered_mask"]),
      thl(x["thresholds"]), ovl(x["threshold_ov"]), levl(x["threshold_level"]),
      moml(x["moments"]), NAl(x["NACOV"]), Wdl(x["W_dwls"]),
      Wfl(x["W_wls"]), nlevl(x["n_levels"]);
  const bool has_mi = x.containsElementNamed("moment_influence");
  const bool has_raw = x.containsElementNamed("raw_data");
  Rcpp::List mil = has_mi ? Rcpp::List(x["moment_influence"]) : Rcpp::List();
  bool use_gdi = x.containsElementNamed("gamma_diag_influence");
  bool use_gfi = x.containsElementNamed("gamma_full_influence");
  Rcpp::List gdil = use_gdi ? Rcpp::List(x["gamma_diag_influence"]) : Rcpp::List();
  Rcpp::List gfil = use_gfi ? Rcpp::List(x["gamma_full_influence"]) : Rcpp::List();
  Rcpp::List rawl = has_raw ? Rcpp::List(x["raw_data"]) : Rcpp::List();
  Rcpp::IntegerVector nobs(x["nobs"]);
  const R_xlen_t nb = Rl.size();
  if (has_mi && mil.size() != nb) {
    Rcpp::stop("magmaan: %s$moment_influence has length %d but R has length %d",
               what, static_cast<int>(mil.size()), static_cast<int>(nb));
  }
  if (has_raw && rawl.size() != nb) {
    Rcpp::stop("magmaan: %s$raw_data has length %d but R has length %d",
               what, static_cast<int>(rawl.size()), static_cast<int>(nb));
  }
  if (use_gdi && gdil.size() != nb) {
    Rcpp::stop("magmaan: %s$gamma_diag_influence has length %d but R has length %d",
               what, static_cast<int>(gdil.size()), static_cast<int>(nb));
  }
  if (use_gfi && gfil.size() != nb) {
    Rcpp::stop("magmaan: %s$gamma_full_influence has length %d but R has length %d",
               what, static_cast<int>(gfil.size()), static_cast<int>(nb));
  }
  auto use_precomputed_influence = [what, nb](Rcpp::List xs, const char* nm) {
    bool all_empty = true;
    bool all_nonempty = true;
    for (R_xlen_t b = 0; b < nb; ++b) {
      Rcpp::NumericMatrix xb(xs[b]);
      const bool empty = xb.nrow() == 0 && xb.ncol() == 0;
      const bool nonempty = xb.nrow() > 0 && xb.ncol() > 0;
      all_empty = all_empty && empty;
      all_nonempty = all_nonempty && nonempty;
    }
    if (all_empty) return false;
    if (!all_nonempty) {
      Rcpp::stop("magmaan: %s$%s must be non-empty for every block or 0x0 for every block",
                 what, nm);
    }
    return true;
  };
  if (use_gdi) use_gdi = use_precomputed_influence(gdil, "gamma_diag_influence");
  if (use_gfi) use_gfi = use_precomputed_influence(gfil, "gamma_full_influence");
  magmaan::data::MixedOrdinalStats out;
  out.R.reserve(static_cast<std::size_t>(nb));
  out.mean.reserve(static_cast<std::size_t>(nb));
  out.ordered.reserve(static_cast<std::size_t>(nb));
  out.thresholds.reserve(static_cast<std::size_t>(nb));
  out.threshold_ov.reserve(static_cast<std::size_t>(nb));
  out.threshold_level.reserve(static_cast<std::size_t>(nb));
  out.moments.reserve(static_cast<std::size_t>(nb));
  out.NACOV.reserve(static_cast<std::size_t>(nb));
  out.W_dwls.reserve(static_cast<std::size_t>(nb));
  out.W_wls.reserve(static_cast<std::size_t>(nb));
  out.n_obs.reserve(static_cast<std::size_t>(nb));
  out.n_levels.reserve(static_cast<std::size_t>(nb));
  if (has_mi) out.moment_influence.reserve(static_cast<std::size_t>(nb));
  if (use_gdi) out.gamma_diag_influence.reserve(static_cast<std::size_t>(nb));
  if (use_gfi) out.gamma_full_influence.reserve(static_cast<std::size_t>(nb));
  if (has_raw) out.raw_data.reserve(static_cast<std::size_t>(nb));
  for (R_xlen_t b = 0; b < nb; ++b) {
    out.R.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Rl[b])));
    out.mean.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(meanl[b])));
    Rcpp::IntegerVector ord(ordl[b]);
    out.ordered.push_back(Rcpp::as<std::vector<std::int32_t>>(ord));
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
    out.moments.push_back(Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(moml[b])));
    out.NACOV.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(NAl[b])));
    out.W_dwls.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Wdl[b])));
    out.W_wls.push_back(Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(Wfl[b])));
    out.n_obs.push_back(static_cast<std::int64_t>(nobs[b]));
    out.n_levels.push_back(Rcpp::as<std::vector<std::int32_t>>(Rcpp::IntegerVector(nlevl[b])));
    if (has_mi) {
      out.moment_influence.push_back(
          Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(mil[b])));
    }
    if (use_gdi) {
      out.gamma_diag_influence.push_back(
          Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(gdil[b])));
    }
    if (use_gfi) {
      out.gamma_full_influence.push_back(
          Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(gfil[b])));
    }
    if (has_raw) {
      out.raw_data.push_back(
          Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(rawl[b])));
    }
  }
  return out;
}

// Rebuild a Stage-1 SaturatedMoments from the serializable `fit$stage1` list
// (mean/cov/n_obs/H/J/acov) that fit_ml2s() stamps on every ML2S fit. Returns
// false when the fit carries no usable stage1 (e.g. a plain FIML fit), so the
// caller can fall back to computing it. Reusing this is bit-identical to a
// recompute (the EM is deterministic) while skipping the EM + observed-
// information rebuild. Shared by fit.cpp and lr_test_satorra.cpp so two-stage
// SB, the FMG spectrum, and the nested LRT all consume one saturated build.
// Reconstruct SaturatedMoments from a Stage-1 list carrying mean/cov/n_obs/acov
// (the shape returned by saturated_em_moments_impl); H/J are optional.
inline bool saturated_from_list(Rcpp::List st,
                                magmaan::estimate::fiml::SaturatedMoments& out) {
  if (!st.containsElementNamed("mean") || !st.containsElementNamed("cov") ||
      !st.containsElementNamed("n_obs") || !st.containsElementNamed("acov") ||
      Rf_isNull(st["mean"]) || Rf_isNull(st["cov"]) ||
      Rf_isNull(st["n_obs"]) || Rf_isNull(st["acov"])) {
    return false;
  }
  Rcpp::List mean_l(st["mean"]);
  Rcpp::List cov_l(st["cov"]);
  Rcpp::IntegerVector nobs(st["n_obs"]);
  const R_xlen_t B = mean_l.size();
  if (cov_l.size() != B || nobs.size() != B) return false;
  out.mean.clear();
  out.cov.clear();
  out.n_obs.clear();
  for (R_xlen_t b = 0; b < B; ++b) {
    out.mean.push_back(
        Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(mean_l[b])));
    out.cov.push_back(
        Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(cov_l[b])));
    out.n_obs.push_back(static_cast<std::int64_t>(nobs[b]));
  }
  out.acov = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(st["acov"]));
  if (st.containsElementNamed("H") && !Rf_isNull(st["H"]))
    out.H = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(st["H"]));
  if (st.containsElementNamed("J") && !Rf_isNull(st["J"]))
    out.J = Rcpp::as<Eigen::MatrixXd>(Rcpp::NumericMatrix(st["J"]));
  return true;
}

inline bool saturated_from_stage1(Rcpp::List fit,
                                  magmaan::estimate::fiml::SaturatedMoments& out) {
  if (!fit.containsElementNamed("stage1")) return false;
  SEXP s1 = fit["stage1"];
  if (Rf_isNull(s1)) return false;
  return saturated_from_list(Rcpp::List(s1), out);
}

// Map an R stage2_weight string to the TwoStageWeight enum. "wls" is an alias
// for the full ADF weight.
inline magmaan::estimate::fiml::TwoStageWeight
two_stage_weight_from_arg(const std::string& s) {
  if (s == "nt" || s == "NT") return magmaan::estimate::fiml::TwoStageWeight::Nt;
  if (s == "uls" || s == "ULS")
    return magmaan::estimate::fiml::TwoStageWeight::Uls;
  if (s == "dwls" || s == "DWLS")
    return magmaan::estimate::fiml::TwoStageWeight::Dwls;
  if (s == "adf" || s == "ADF" || s == "wls" || s == "WLS")
    return magmaan::estimate::fiml::TwoStageWeight::Adf;
  if (s == "dls" || s == "DLS")
    return magmaan::estimate::fiml::TwoStageWeight::Dls;
  Rcpp::stop("magmaan: unknown stage2_weight '" + s +
             "' (expected nt, uls, dwls, adf, dls)");
}

// ---- profile-LRT result serializers (shared by robust.cpp + fit.cpp) -------
// The misspecification-robust profile-LRT bindings live in two glue files
// (robust.cpp: ML/ordinal/mixed; fit.cpp: continuous-LS/FIML/ML2S, co-located
// with their fit-local weight/pack helpers), so the WeightedProfileLRTResult
// serializer must be shared rather than file-local.
[[maybe_unused]] inline Rcpp::List
scaled_shifted_to_list(const magmaan::robust::ScaledShiftedResult& r) {
  return Rcpp::List::create(Rcpp::_["chi2_adj"] = r.chi2_adj,
                            Rcpp::_["df"] = r.df,
                            Rcpp::_["scale_a"] = r.scale_a,
                            Rcpp::_["shift_b"] = r.shift_b);
}

inline Rcpp::CharacterVector
warnings_to_r(const std::vector<std::string>& warnings) {
  Rcpp::CharacterVector out(static_cast<R_xlen_t>(warnings.size()));
  for (std::size_t k = 0; k < warnings.size(); ++k) {
    out[static_cast<R_xlen_t>(k)] = warnings[k];
  }
  return out;
}

inline Rcpp::List
profile_lrt_to_list(const magmaan::estimate::WeightedProfileLRTResult& r) {
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

}  // namespace magmaanr
