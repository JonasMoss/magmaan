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
#include "magmaan/parse/eqs_parser.hpp"
#include "magmaan/compat/eqs/model.hpp"
#include "magmaan/compat/mplus/model.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/compat/lavaan/composite_fold.hpp"
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
    case K::RejectedConstruct:   return "RejectedConstruct";
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
    case K::CompositeTooFewIndicators:return "CompositeTooFewIndicators";
    case K::CompositeOverlap:         return "CompositeOverlap";
    case K::UnidentifiedComposite:    return "UnidentifiedComposite";
    case K::CompositeModeRequired:    return "CompositeModeRequired";
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

void attach_group_attrs(SEXP df, const std::string& group_var,
                        const std::vector<std::string>& group_labels) {
  Rf_setAttrib(df, Rf_install("magmaan.group_var"),
               Rf_mkString(group_var.c_str()));
  Rcpp::CharacterVector gl(static_cast<R_xlen_t>(group_labels.size()));
  for (std::size_t i = 0; i < group_labels.size(); ++i)
    gl[static_cast<R_xlen_t>(i)] = group_labels[i];
  Rf_setAttrib(df, Rf_install("magmaan.group_labels"), gl);
}

Rcpp::DataFrame lavaan_partable_df(
    const magmaan::compat::lavaan::LavaanParTable& pt) {
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
    ustart[i] = pt.ustart[k];   // NaN -> R NaN
  }

  Rcpp::List cols = Rcpp::List::create(
      Rcpp::_["id"] = id, Rcpp::_["lhs"] = lhs, Rcpp::_["op"] = op,
      Rcpp::_["rhs"] = rhs, Rcpp::_["user"] = user, Rcpp::_["block"] = block,
      Rcpp::_["group"] = group, Rcpp::_["free"] = free, Rcpp::_["exo"] = exo,
      Rcpp::_["ustart"] = ustart, Rcpp::_["label"] = label,
      Rcpp::_["plabel"] = plabel);

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

  cols.attr("row.names") =
      Rcpp::IntegerVector::create(NA_INTEGER, static_cast<int>(-n));
  cols.attr("class") = "data.frame";
  Rcpp::DataFrame df(cols);
  attach_group_attrs(df, pt.group_var, pt.group_labels);
  return df;
}

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
        } else if constexpr (std::is_same_v<T, EqualRef>) {
          return Rcpp::List::create(Rcpp::_["kind"] = "equal",
                                    Rcpp::_["target"] = sv2s(v.text));
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
                  } else if constexpr (std::is_same_v<A, EqualRef>) {
                    return Rcpp::List::create(Rcpp::_["kind"] = "equal",
                                              Rcpp::_["target"] = sv2s(a.text));
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

// Map a lavaan `group.equal` family string onto the magmaan `GroupEqual` enum.
// The accepted vocabulary mirrors lavaan's: any other string is a hard error
// rather than the C++ golden's silent skip, since this is the user-facing edge.
magmaan::spec::GroupEqual group_equal_from_string(const std::string& s) {
  using GE = magmaan::spec::GroupEqual;
  if (s == "loadings")             return GE::Loadings;
  if (s == "thresholds")           return GE::Thresholds;
  if (s == "intercepts")           return GE::Intercepts;
  if (s == "means")                return GE::Means;
  if (s == "residuals")            return GE::Residuals;
  if (s == "residual.covariances") return GE::ResidualCovariances;
  if (s == "lv.variances")         return GE::LvVariances;
  if (s == "lv.covariances")       return GE::LvCovariances;
  if (s == "regressions")          return GE::Regressions;
  Rcpp::stop("magmaan: unknown group.equal family '%s' (expected one of "
             "loadings, thresholds, intercepts, means, residuals, "
             "residual.covariances, lv.variances, lv.covariances, regressions)",
             s);
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
Rcpp::List eqs_model_impl(std::string syntax,
                         Rcpp::Nullable<Rcpp::CharacterVector> observed_names = R_NilValue) {
  std::vector<std::string> columns;
  if (observed_names.isNotNull())
    columns = Rcpp::as<std::vector<std::string>>(observed_names.get());
  auto flat = magmaan::parse::EqsParser::parse(syntax, columns);
  if (!flat) stop_parse(flat.error());
  magmaan::spec::LatentNames names;
  magmaan::spec::Starts starts;
  auto model = magmaan::spec::build(*flat, magmaan::compat::eqs::build_options(), &starts, &names);
  if (!model) Rcpp::stop("magmaan EQS model error: %s", model.error().detail);
  auto pt = magmaan::compat::lavaan::to_lavaan_partable(*model, names, starts);
  return Rcpp::List::create(
      Rcpp::_["partable"] = lavaan_partable_df(pt),
      Rcpp::_["syntax"] = magmaan::compat::eqs::to_lavaan_syntax(*flat));
}

// [[Rcpp::export]]
Rcpp::DataFrame lavaan_lavaanify(std::string syntax,
                                bool auto_var = true,
                                bool auto_cov_lv_x = true,
                                bool auto_cov_y = false,
                                bool orthogonal = false,
                                bool auto_fix_first = true,
                                bool auto_fix_single = true,
                                bool std_lv = false,
                                bool effect_coding = false,
                                bool fixed_x = true,
                                bool meanstructure = false,
                                bool int_ov_free = true,
                                bool int_lv_free = false,
                                int n_groups = 1,
                                std::string group_var = "",
                                Rcpp::Nullable<Rcpp::CharacterVector> group_labels = R_NilValue,
                                Rcpp::Nullable<Rcpp::CharacterVector> group_equal = R_NilValue,
                                Rcpp::Nullable<Rcpp::CharacterVector> group_partial = R_NilValue,
                                Rcpp::Nullable<Rcpp::CharacterVector> marker = R_NilValue) {
  auto p = magmaan::parse::Parser::parse(syntax);
  if (!p.has_value()) stop_parse(p.error());

  magmaan::spec::BuildOptions opts;
  opts.composite_mode = magmaan::spec::CompositeMode::HenselerOgasawara;
  opts.auto_var       = auto_var;
  opts.auto_cov_lv_x  = auto_cov_lv_x;
  opts.auto_cov_y     = auto_cov_y;
  opts.orthogonal      = orthogonal;    // fix auto latent covariances at 0 (lavaan orthogonal=)
  if (marker.isNotNull()) {
    Rcpp::CharacterVector values(marker.get());
    if (!values.hasAttribute("names")) Rcpp::stop("marker must be a named character vector");
    Rcpp::CharacterVector keys = values.names();
    for (int i = 0; i < values.size(); ++i) {
      if (keys[i] == NA_STRING || values[i] == NA_STRING || Rcpp::as<std::string>(keys[i]).empty())
        Rcpp::stop("marker requires nonmissing latent and indicator names");
      auto name = Rcpp::as<std::string>(keys[i]);
      if (opts.marker.contains(name)) Rcpp::stop("marker requires unique latent names");
      opts.marker[name] = Rcpp::as<std::string>(values[i]);
    }
  }
  opts.auto_fix_first  = auto_fix_first;
  opts.auto_fix_single = auto_fix_single;
  opts.std_lv          = std_lv;        // when true, forces auto.fix.first off (lavaan parity)
  opts.effect_coding   = effect_coding; // free all loadings + LV var; adds `Σλ == #indicators`
  opts.fixed_x         = fixed_x;
  opts.meanstructure   = meanstructure;
  opts.int_ov_free     = int_ov_free;
  opts.int_lv_free     = int_lv_free;
  opts.n_groups        = n_groups;
  opts.group_var       = group_var;
  if (group_labels.isNotNull())
    opts.group_labels = Rcpp::as<std::vector<std::string>>(group_labels.get());
  if (group_equal.isNotNull()) {
    for (const auto& s : Rcpp::as<std::vector<std::string>>(group_equal.get()))
      opts.group_equal.push_back(group_equal_from_string(s));
  }
  if (group_partial.isNotNull())
    opts.group_partial = Rcpp::as<std::vector<std::string>>(group_partial.get());

  magmaan::spec::Starts starts;
  magmaan::spec::LatentNames names;
  auto pt_or = magmaan::spec::build(*p, opts, &starts, &names);
  if (!pt_or.has_value()) {
    Rcpp::stop("magmaan lavaanify error [%s]: %s",
               partable_error_kind(pt_or.error().kind), pt_or.error().detail);
  }
  const magmaan::compat::lavaan::LavaanParTable pt =
      magmaan::compat::lavaan::to_lavaan_partable(*pt_or, names, starts);
  // Stamp the resolved `group.equal` families as enum indices on the returned
  // df. `from_lavaan_partable` cannot recover them (a lavaan partable has no
  // such column), but the fit-time Wu-Estabrook ordinal release keys off
  // `LatentStructure::group_equal`, so the ordinal fit binding reads this
  // attribute back. Indices are produced and consumed within the same build,
  // so the enum order is always consistent.
  auto stamp_group_equal = [&](SEXP target) {
    if (opts.group_equal.empty()) return;
    Rcpp::IntegerVector ge(static_cast<R_xlen_t>(opts.group_equal.size()));
    for (std::size_t i = 0; i < opts.group_equal.size(); ++i)
      ge[static_cast<R_xlen_t>(i)] = static_cast<int>(opts.group_equal[i]);
    Rf_setAttrib(target, Rf_install("magmaan.group_equal"), ge);
  };
  Rcpp::DataFrame expanded = lavaan_partable_df(pt);
  stamp_group_equal(expanded);
  if (names.composites.empty()) return expanded;

  const magmaan::compat::lavaan::LavaanParTable folded =
      magmaan::compat::lavaan::fold_composites(pt, names.composites);
  Rcpp::DataFrame df = lavaan_partable_df(folded);
  stamp_group_equal(df);
  Rf_setAttrib(df, Rf_install("magmaan.expanded_partable"), expanded);
  return df;
}

// [[Rcpp::export]]
Rcpp::List mplus_model_impl(std::string source) {
  auto parsed = magmaan::parse::MplusParser::parse(source);
  if (!parsed) stop_parse(parsed.error());
  const auto options = magmaan::compat::mplus::build_options(parsed->input);
  magmaan::spec::LatentNames names;
  magmaan::spec::Starts starts;
  auto model = magmaan::spec::build(parsed->flat, options, &starts, &names);
  if (!model) Rcpp::stop("magmaan Mplus model error: %s", model.error().detail);
  magmaan::compat::mplus::apply_provenance(*parsed,*model,names);
  auto pt = magmaan::compat::lavaan::to_lavaan_partable(*model, names, starts);
  const auto n = parsed->notes.size();
  Rcpp::CharacterVector klass(n), rule(n), message(n);
  Rcpp::IntegerVector line(n), col(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto& note = parsed->notes[i];
    klass[i] = note.klass == magmaan::parse::MplusClass::Reported ? "reported" :
        note.klass == magmaan::parse::MplusClass::DataDescription ? "data_description" : "schema";
    rule[i] = note.rule; message[i] = note.message;
    line[i] = note.span.line; col[i] = note.span.col;
  }
  Rcpp::CharacterVector group_label(parsed->input.groups.size()), group_code(parsed->input.groups.size());
  for (std::size_t i=0;i<parsed->input.groups.size();++i) {
    group_label[i]=parsed->input.groups[i].label;group_code[i]=parsed->input.groups[i].code;
  }
  const auto& plan=parsed->input.data_plan;
  Rcpp::List files(plan.files.size()), format(plan.format.size()), missing(plan.missing.size());
  for(std::size_t i=0;i<plan.files.size();++i) files[i]=Rcpp::List::create(
      Rcpp::_["path"]=plan.files[i].path,Rcpp::_["label"]=plan.files[i].label);
  for(std::size_t i=0;i<plan.format.size();++i) format[i]=Rcpp::List::create(
      Rcpp::_["kind"]=static_cast<int>(plan.format[i].kind),
      Rcpp::_["width"]=plan.format[i].width,Rcpp::_["decimals"]=plan.format[i].decimals);
  for(std::size_t i=0;i<plan.missing.size();++i) missing[i]=Rcpp::List::create(
      Rcpp::_["variables"]=plan.missing[i].variables,Rcpp::_["values"]=plan.missing[i].values);
  auto data_plan=Rcpp::List::create(Rcpp::_["files"]=files,Rcpp::_["format"]=format,
      Rcpp::_["matrix_type"]=plan.matrix_type,Rcpp::_["means"]=plan.means,
      Rcpp::_["standard_deviations"]=plan.standard_deviations,Rcpp::_["listwise"]=plan.listwise,
      Rcpp::_["file_groups"]=plan.file_groups,Rcpp::_["n_groups"]=plan.n_groups,
      Rcpp::_["n_observations"]=plan.n_observations,Rcpp::_["missing_symbol"]=plan.missing_symbol,
      Rcpp::_["missing"]=missing,Rcpp::_["names"]=parsed->input.names,Rcpp::_["analysis"]=parsed->input.analysis);
  for(std::size_t i=0;i<parsed->input.groups.size();++i)
    if(parsed->input.groups[i].code.empty()) group_code[i]=parsed->input.groups[i].label;
  return Rcpp::List::create(
      Rcpp::_["ordered"] = parsed->input.categorical,
      Rcpp::_["parameterization"] = parsed->input.parameterization == "THETA" ? "theta" : "delta",
      Rcpp::_["data_plan"] = data_plan,
      Rcpp::_["partable"] = lavaan_partable_df(pt),
      Rcpp::_["syntax"] = magmaan::compat::mplus::to_lavaan_syntax(parsed->flat),
      Rcpp::_["group_var"] = options.group_var,
      Rcpp::_["group_labels"] = group_code,
      Rcpp::_["groups"] = Rcpp::DataFrame::create(Rcpp::_["label"] = group_label, Rcpp::_["code"] = group_code),
      Rcpp::_["meanstructure"] = options.meanstructure,
      Rcpp::_["fixed_x"] = options.fixed_x,
      Rcpp::_["observed_x"] = parsed->input.observed_x,
      Rcpp::_["notes"] = Rcpp::DataFrame::create(Rcpp::_["class"] = klass,
          Rcpp::_["rule"] = rule, Rcpp::_["line"] = line, Rcpp::_["col"] = col,
          Rcpp::_["message"] = message));
}

// [[Rcpp::export]]
Rcpp::DataFrame mplus_ordinal_partable_impl(std::string source, Rcpp::List category_counts) {
  std::vector<std::vector<std::int32_t>> counts;
  for(SEXP block:category_counts) counts.push_back(Rcpp::as<std::vector<std::int32_t>>(block));
  auto model=magmaan::compat::mplus::prepare_ordinal_model(source,counts);
  if(!model) Rcpp::stop("magmaan Mplus ordinal error: %s",model.error().detail);
  auto pt=magmaan::compat::lavaan::to_lavaan_partable(model->structure,model->names,model->starts);
  auto out=lavaan_partable_df(pt);
  Rcpp::List preparation(pt.ordinal_preparation.size());
  for(std::size_t b=0;b<pt.ordinal_preparation.size();++b)
    preparation[b]=Rcpp::IntegerVector(pt.ordinal_preparation[b].begin(),pt.ordinal_preparation[b].end());
  out.attr("magmaan.ordinal_preparation")=preparation;
  return out;
}
