// Thin Rcpp glue over magmaan's C++ API. One // [[Rcpp::export]] function per
// thing we want to poke at from R. Returns plain base-R objects (data.frame /
// list / vectors) — no S3 classes, no print methods. Errors are surfaced as
// ordinary R errors via Rcpp::stop, carrying magmaan's error kind + detail.

#include <Rcpp.h>

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <variant>

#include "magmaan/version.hpp"
#include "magmaan/error.hpp"
#include "magmaan/parse/op.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"

namespace {

const char* parse_error_kind(magmaan::ParseError::Kind k) {
  using K = magmaan::ParseError::Kind;
  switch (k) {
    case K::UnexpectedChar:      return "UnexpectedChar";
    case K::UnknownOperator:     return "UnknownOperator";
    case K::UnsupportedOperator: return "UnsupportedOperator";
    case K::UnterminatedString:  return "UnterminatedString";
    case K::MalformedNumber:     return "MalformedNumber";
    case K::ExpectedLhs:         return "ExpectedLhs";
    case K::ExpectedOperator:    return "ExpectedOperator";
    case K::ExpectedRhsTerm:     return "ExpectedRhsTerm";
    case K::ModifierEvalFailed:  return "ModifierEvalFailed";
    case K::GroupVecMismatch:    return "GroupVecMismatch";
  }
  return "Unknown";
}

const char* partable_error_kind(magmaan::PartableError::Kind k) {
  using K = magmaan::PartableError::Kind;
  switch (k) {
    case K::BadGroupSpec:             return "BadGroupSpec";
    case K::UnknownLabelInConstraint: return "UnknownLabelInConstraint";
    case K::InconsistentModifiers:    return "InconsistentModifiers";
    case K::EmptyModel:               return "EmptyModel";
  }
  return "Unknown";
}

const char* constraint_kind_str(magmaan::parse::ConstraintKind k) {
  using K = magmaan::parse::ConstraintKind;
  switch (k) {
    case K::Eq:     return "==";
    case K::Lt:     return "<";
    case K::Gt:     return ">";
    case K::Define: return ":=";
  }
  return "?";
}

[[noreturn]] void stop_parse(const magmaan::ParseError& e) {
  Rcpp::stop("magmaan parse error [%s] at %u:%u (bytes %u..%u): %s",
             parse_error_kind(e.kind), e.span.line, e.span.col,
             e.span.begin, e.span.end, e.detail);
}

std::string sv2s(std::string_view sv) { return std::string(sv); }

// One modifier -> a small list describing the variant.
Rcpp::List describe_modifier(const magmaan::parse::Modifier& m) {
  using namespace magmaan::parse;
  return std::visit(
      [](const auto& v) -> Rcpp::List {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, FixedValue>) {
          return Rcpp::List::create(Rcpp::_["kind"] = "fixed",
                                    Rcpp::_["value"] = v.value);
        } else if constexpr (std::is_same_v<T, Label>) {
          return Rcpp::List::create(Rcpp::_["kind"] = "label",
                                    Rcpp::_["text"] = sv2s(v.text));
        } else if constexpr (std::is_same_v<T, StartValue>) {
          return Rcpp::List::create(Rcpp::_["kind"] = "start",
                                    Rcpp::_["value"] = v.value);
        } else if constexpr (std::is_same_v<T, Free>) {
          return Rcpp::List::create(Rcpp::_["kind"] = "free");
        } else {  // GroupVec
          Rcpp::List atoms(v.per_group.size());
          for (std::size_t i = 0; i < v.per_group.size(); ++i) {
            atoms[i] = std::visit(
                [](const auto& a) -> Rcpp::List {
                  using A = std::decay_t<decltype(a)>;
                  if constexpr (std::is_same_v<A, FixedValue>) {
                    return Rcpp::List::create(Rcpp::_["kind"] = "fixed",
                                              Rcpp::_["value"] = a.value);
                  } else if constexpr (std::is_same_v<A, Label>) {
                    return Rcpp::List::create(Rcpp::_["kind"] = "label",
                                              Rcpp::_["text"] = sv2s(a.text));
                  } else if constexpr (std::is_same_v<A, StartValue>) {
                    return Rcpp::List::create(Rcpp::_["kind"] = "start",
                                              Rcpp::_["value"] = a.value);
                  } else {  // Free
                    return Rcpp::List::create(Rcpp::_["kind"] = "free");
                  }
                },
                v.per_group[i]);
          }
          return Rcpp::List::create(Rcpp::_["kind"] = "groupvec",
                                    Rcpp::_["atoms"] = atoms);
        }
      },
      m);
}

}  // namespace

// [[Rcpp::export]]
std::string version() {
  return std::string(magmaan::version());
}

// [[Rcpp::export]]
Rcpp::List parse_parse(std::string syntax) {
  auto r = magmaan::parse::Parser::parse(syntax);
  if (!r.has_value()) stop_parse(r.error());
  const magmaan::parse::FlatPartable& fp = *r;

  const R_xlen_t n = static_cast<R_xlen_t>(fp.rows.size());
  Rcpp::CharacterVector lhs(n), op(n), rhs(n);
  Rcpp::IntegerVector block(n), mod_idx(n), line(n), col(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const magmaan::parse::FlatRow& row = fp.rows[static_cast<std::size_t>(i)];
    lhs[i]     = sv2s(row.lhs);
    op[i]      = sv2s(magmaan::parse::to_string(row.op));
    rhs[i]     = sv2s(row.rhs);
    block[i]   = static_cast<int>(row.block);
    mod_idx[i] = static_cast<int>(row.mod_idx);
    line[i]    = static_cast<int>(row.span.line);
    col[i]     = static_cast<int>(row.span.col);
  }
  Rcpp::DataFrame rows = Rcpp::DataFrame::create(
      Rcpp::_["lhs"] = lhs, Rcpp::_["op"] = op, Rcpp::_["rhs"] = rhs,
      Rcpp::_["block"] = block, Rcpp::_["mod_idx"] = mod_idx,
      Rcpp::_["line"] = line, Rcpp::_["col"] = col,
      Rcpp::_["stringsAsFactors"] = false);

  // mods[0] is a sentinel; expose 1..n-1.
  const std::size_t n_mods = fp.mods.empty() ? 0 : fp.mods.size() - 1;
  Rcpp::List mods(n_mods);
  for (std::size_t i = 1; i < fp.mods.size(); ++i) {
    mods[i - 1] = describe_modifier(fp.mods[i]);
  }

  Rcpp::List constraints(fp.constraints.size());
  for (std::size_t i = 0; i < fp.constraints.size(); ++i) {
    const magmaan::parse::Constraint& c = fp.constraints[i];
    constraints[i] = Rcpp::List::create(
        Rcpp::_["kind"] = constraint_kind_str(c.kind),
        Rcpp::_["name"] = sv2s(c.name));
  }

  return Rcpp::List::create(Rcpp::_["rows"] = rows,
                            Rcpp::_["mods"] = mods,
                            Rcpp::_["constraints"] = constraints,
                            Rcpp::_["source"] = sv2s(fp.source()));
}

// [[Rcpp::export]]
Rcpp::DataFrame lavaan_lavaanify(std::string syntax,
                                bool auto_var = true,
                                bool auto_cov_lv_x = true,
                                bool auto_cov_y = false,
                                bool auto_fix_first = true,
                                bool std_lv = false,
                                bool effect_coding = false,
                                bool fixed_x = true,
                                bool meanstructure = false,
                                int n_groups = 1,
                                std::string group_var = "",
                                Rcpp::Nullable<Rcpp::CharacterVector> group_labels = R_NilValue) {
  auto p = magmaan::parse::Parser::parse(syntax);
  if (!p.has_value()) stop_parse(p.error());

  magmaan::spec::LavaanifyOptions opts;
  opts.auto_var       = auto_var;
  opts.auto_cov_lv_x  = auto_cov_lv_x;
  opts.auto_cov_y     = auto_cov_y;
  opts.auto_fix_first = auto_fix_first;
  opts.std_lv         = std_lv;        // when true, forces auto.fix.first off (lavaan parity)
  opts.effect_coding  = effect_coding; // free all loadings + LV var; adds `Σλ == #indicators`
  opts.fixed_x        = fixed_x;
  opts.meanstructure  = meanstructure;
  opts.n_groups       = n_groups;
  opts.group_var      = group_var;
  if (group_labels.isNotNull())
    opts.group_labels = Rcpp::as<std::vector<std::string>>(group_labels.get());

  magmaan::spec::Starts starts;
  magmaan::spec::LatentNames names;
  auto pt_or = magmaan::spec::lavaanify(*p, opts, &starts, &names);
  if (!pt_or.has_value()) {
    Rcpp::stop("magmaan lavaanify error [%s]: %s",
               partable_error_kind(pt_or.error().kind), pt_or.error().detail);
  }
  const magmaan::compat::lavaan::LavaanParTable pt =
      magmaan::compat::lavaan::to_lavaan_partable(*pt_or, names, starts);

  const R_xlen_t n = static_cast<R_xlen_t>(pt.size());
  Rcpp::IntegerVector id(n), user(n), block(n), group(n), free(n), exo(n);
  Rcpp::CharacterVector lhs(n), op(n), rhs(n), label(n), plabel(n);
  Rcpp::NumericVector ustart(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    id[i]     = pt.id[k];
    user[i]   = pt.user[k];
    block[i]  = pt.block[k];
    group[i]  = pt.group[k];
    free[i]   = pt.free[k];
    exo[i]    = pt.exo[k];
    lhs[i]    = pt.lhs[k];
    op[i]     = sv2s(magmaan::parse::to_string(pt.op[k]));
    rhs[i]    = pt.rhs[k];
    label[i]  = pt.label[k];
    plabel[i] = pt.plabel[k];
    ustart[i] = pt.ustart[k];   // NaN → R NaN
  }

  Rcpp::List cols = Rcpp::List::create(
      Rcpp::_["id"] = id, Rcpp::_["lhs"] = lhs, Rcpp::_["op"] = op,
      Rcpp::_["rhs"] = rhs, Rcpp::_["user"] = user, Rcpp::_["block"] = block,
      Rcpp::_["group"] = group, Rcpp::_["free"] = free, Rcpp::_["exo"] = exo,
      Rcpp::_["ustart"] = ustart, Rcpp::_["label"] = label,
      Rcpp::_["plabel"] = plabel);

  // Append methods-developer extension columns under their map keys.
  for (const auto& kv : pt.extra_real) {
    cols[kv.first] = Rcpp::NumericVector(kv.second.begin(), kv.second.end());
  }
  for (const auto& kv : pt.extra_int) {
    cols[kv.first] = Rcpp::IntegerVector(kv.second.begin(), kv.second.end());
  }
  for (const auto& kv : pt.extra_str) {
    Rcpp::CharacterVector cv(static_cast<R_xlen_t>(kv.second.size()));
    for (std::size_t j = 0; j < kv.second.size(); ++j) cv[j] = kv.second[j];
    cols[kv.first] = cv;
  }

  // Make it a data.frame (variable column count, so we can't use
  // DataFrame::create). All columns are int/dbl/chr — no factor conversion,
  // so stringsAsFactors is moot. "compact" row names = c(NA, -n) => 1..n.
  cols.attr("row.names") =
      Rcpp::IntegerVector::create(NA_INTEGER, static_cast<int>(-n));
  cols.attr("class") = "data.frame";
  Rcpp::DataFrame df(cols);
  // Group identity rides as data.frame attributes (table-level metadata,
  // mirrored back into LatentNames.group_var / .group_labels by parse_partable_df).
  Rf_setAttrib(df, Rf_install("magmaan.group_var"), Rf_mkString(pt.group_var.c_str()));
  Rcpp::CharacterVector gl(static_cast<R_xlen_t>(pt.group_labels.size()));
  for (std::size_t j = 0; j < pt.group_labels.size(); ++j)
    gl[static_cast<R_xlen_t>(j)] = pt.group_labels[j];
  Rf_setAttrib(df, Rf_install("magmaan.group_labels"), gl);
  return df;
}
