#include "glue_internal.h"
#include "magmaan/api/policy_mi.hpp"
#include <iomanip>
#include <sstream>

using namespace magmaanr;
using namespace magmaanr::fitglue;
namespace {
std::string affine_expression(const Eigen::MatrixXd& a, Eigen::Index row,
    const std::vector<std::string>& labels) {
  std::ostringstream out;
  out << std::setprecision(17);
  bool first = true;
  for (Eigen::Index j = 0; j < a.cols(); ++j) {
    if (a(row, j) == 0) continue;
    if (!first) out << " + ";
    out << '(' << a(row, j) << ")*" << labels[j];
    first = false;
  }
  return first ? "0" : out.str();
}
magmaan::api::PolicyFitState mi_state(Rcpp::LogicalVector state) {
  magmaan::api::PolicyFitState out;
  out.converged = state.size() > 0 && state[0] == TRUE;
  out.psd_boundary = state.size() > 1 && state[1] == TRUE;
  out.penalized = state.size() > 3 && state[3] == TRUE;
  return out;
}
}

// [[Rcpp::export]]
Rcpp::DataFrame policy_modification_indices_impl(Rcpp::List fit,
    Rcpp::LogicalVector state, SEXP raw = R_NilValue, bool releases = true) {
  using namespace magmaan;
  api::PolicyModificationIndices out;
  const auto s = mi_state(state);
  const auto estimator = Rcpp::as<std::string>(fit["estimator"]);
  if (s.penalized) out = {api::InferenceReason::Penalized, std::string(api::penalized_detail), {}};
  else if (!s.converged) out = {api::InferenceReason::NotConverged, "the fit did not pass its convergence verdict", {}};
  else if ((estimator != "ML" && estimator != "FIML" && estimator != "DWLS") ||
      fit.containsElementNamed("nclusters") || fit.containsElementNamed("association"))
    out = {api::InferenceReason::UnsupportedModel, "the modification-index policy covers single-level ML, FIML and ordinal or mixed DWLS", {}};
  else {
    auto ctx = ctx_from_fit(fit);
    const auto est = est_from_fit(fit);
    api::PolicyModificationOptions opts;
    opts.releases = releases;
    if (estimator == "DWLS") {
      const auto parameterization = ordinal_parameterization_from_string(
          fit.containsElementNamed("parameterization") ? Rcpp::as<std::string>(fit["parameterization"]) : "delta");
      if (fit.containsElementNamed("mixed_ordinal") && Rcpp::as<bool>(fit["mixed_ordinal"])) {
        auto stats = mixed_ordinal_stats_from_arg(stats_from_fit_or_arg(fit, R_NilValue,
            "mixed_ordinal_stats", "policy_modification_indices"));
        out = api::policy_modification_indices(ctx.pt, ctx.rep, stats, est, parameterization, s, opts, &ctx.names.row_user);
      } else if (fit.containsElementNamed("ordinal") && Rcpp::as<bool>(fit["ordinal"])) {
        auto stats = ordinal_stats_from_arg(stats_from_fit_or_arg(fit, R_NilValue,
            "ordinal_stats", "policy_modification_indices"));
        out = api::policy_modification_indices(ctx.pt, ctx.rep, stats, est, parameterization, s, opts);
      } else out = {api::InferenceReason::UnsupportedModel, "continuous DWLS is outside the ordinary policy", {}};
    } else if (estimator == "FIML") {
      auto rd = fiml_raw_from_arg(ctx.rep, Rf_isNull(raw) ? fit["raw_data"] : raw);
      auto pack = estimate::fiml::fiml_pack(rd);
      if (!pack) out = {api::InferenceReason::NumericFailure, pack.error().detail, {}};
      else out = api::policy_modification_indices(ctx.pt, ctx.rep, rd, *pack, est, s, opts);
    } else {
      auto rd = complete_raw_from_arg(ctx.rep, Rf_isNull(raw) ? fit["raw_data"] : raw);
      out = api::policy_modification_indices(ctx.pt, ctx.rep, ctx.samp, rd, est, s, opts);
    }
  }
  const bool available = out.reason == api::InferenceReason::Available;
  const R_xlen_t n = available ? out.table.rows.size() : 1;
  Rcpp::CharacterVector kind(n), lhs(n), op(n), rhs(n), test(n, "score"), reason(n);
  Rcpp::IntegerVector group(n), df(n, 1), row(n);
  Rcpp::NumericVector statistic(n, NA_REAL), pvalue(n, NA_REAL), epc(n, NA_REAL), lv(n, NA_REAL), all(n, NA_REAL);
  if (!available) {
    kind[0] = "fixed"; lhs[0] = ""; op[0] = ""; rhs[0] = "";
    reason[0] = std::string(api::reason_name(out.reason));
  } else {
    auto ctx = ctx_from_fit(fit);
    auto con = estimate::build_eq_constraints(ctx.pt);
    if (!con) stop_post(con.error());
    std::vector<std::string> labels(ctx.pt.n_free());
    for (std::size_t j = 0; j < ctx.pt.size(); ++j)
      if (ctx.pt.free[j] > 0) labels[ctx.pt.free[j]-1] = ctx.names.row_plabel[j];
    for (R_xlen_t i = 0; i < n; ++i) {
      const auto& r = out.table.rows[i]; const auto& c = r.candidate;
      const bool release = c.kind == inference::ScoreCandidateKind::EqualityRelease;
      kind[i] = release ? "release" : "fixed";
      op[i] = std::string(parse::to_string(c.op)); group[i] = c.group;
      row[i] = c.row + 1;
      if (release) {
        lhs[i] = affine_expression(con->A_eq, c.row, labels);
        std::ostringstream value; value << std::setprecision(17) << con->b_eq[c.row];
        rhs[i] = value.str();
      } else {
        lhs[i] = c.lhs_var >= 0 ? ctx.names.var_name[c.lhs_var] : "";
        rhs[i] = c.rhs_var >= 0 ? ctx.names.var_name[c.rhs_var] : "";
      }
      statistic[i] = r.mi_scaled; pvalue[i] = r.p_value;
      epc[i] = r.epc; lv[i] = r.epc_lv; all[i] = r.epc_all;
      reason[i] = std::string(api::reason_name(out.row_reasons[i]));
    }
  }
  auto result = Rcpp::DataFrame::create(Rcpp::_["kind"] = kind, Rcpp::_["lhs"] = lhs,
      Rcpp::_["op"] = op, Rcpp::_["rhs"] = rhs, Rcpp::_["group"] = group,
      Rcpp::_["test"] = test, Rcpp::_["statistic"] = statistic, Rcpp::_["df"] = df,
      Rcpp::_["pvalue"] = pvalue, Rcpp::_["epc"] = epc, Rcpp::_["sepc.lv"] = lv,
      Rcpp::_["sepc.all"] = all, Rcpp::_["reason"] = reason);
  result.attr("candidate_row") = row;
  result.attr("detail") = out.detail;
  return result;
}

// Build H1 from the same model triple, freeing one fixed parameter or removing
// precisely one row of the native affine restriction system. Reprojecting all
// retained restrictions avoids releasing an entire shared-label family.
// [[Rcpp::export]]
Rcpp::DataFrame policy_mi_alternative_impl(Rcpp::List fit, std::string kind,
    int row, std::string lhs, std::string op, std::string rhs, int group) {
  using namespace magmaan;
  auto ctx = ctx_from_fit(fit);
  auto con = estimate::build_eq_constraints(ctx.pt);
  if (!con) stop_post(con.error());
  auto table = compat::lavaan::to_lavaan_partable(ctx.pt, ctx.names, {});
  // Work on the public projection: source constraints and labels are replaced
  // with the retained native affine system, without changing formula rows.
  Rcpp::DataFrame base = partable_df_from_lavaan(table);
  Rcpp::Function subset = Rcpp::Environment::base_env()["["];
  Rcpp::LogicalVector keep(table.size());
  for (std::size_t i = 0; i < table.size(); ++i)
    keep[i] = table.op[i] != parse::Op::EqConstraint && table.op[i] != parse::Op::DefineParam;
  base = subset(base, keep, R_MissingArg, Rcpp::Named("drop") = false);
  auto parsed = parse_partable_df(base);
  auto& pt = parsed.structure; auto& names = parsed.names;
  std::vector<std::string> labels(pt.n_free());
  for (std::size_t i = 0; i < pt.size(); ++i) {
    names.row_label[i] = pt.free[i] > 0 ? "mi_p" + std::to_string(pt.free[i]) : "";
    if (pt.free[i] > 0) labels[pt.free[i]-1] = names.row_label[i];
  }
  if (kind == "fixed") {
    const auto op_code = op == "=~" ? parse::Op::Measurement : op == "~~" ? parse::Op::Covariance : parse::Op::Regression;
    const auto find_var = [&](const std::string& name) {
      return static_cast<std::int32_t>(std::find(names.var_name.begin(), names.var_name.end(), name) - names.var_name.begin());
    };
    const auto a = find_var(lhs), b = find_var(rhs);
    std::size_t index = pt.size();
    for (std::size_t i = 0; i < pt.size(); ++i)
      if (pt.op[i] == op_code && pt.group[i] == group &&
          ((pt.lhs_var[i] == a && pt.rhs_var[i] == b) ||
           (op_code == parse::Op::Covariance && pt.lhs_var[i] == b && pt.rhs_var[i] == a))) { index = i; break; }
    const int free = pt.n_free() + 1;
    if (index == pt.size()) {
      pt.op.push_back(op_code); pt.lhs_var.push_back(a); pt.rhs_var.push_back(b);
      pt.group.push_back(group); pt.free.push_back(free); pt.exo.push_back(0); pt.fixed_value.push_back(0);
      names.row_lhs.push_back(lhs); names.row_rhs.push_back(rhs); names.row_label.push_back("mi_p" + std::to_string(free));
      names.row_plabel.push_back(".p" + std::to_string(pt.size()) + "."); names.row_user.push_back(1);
    } else {
      if (pt.free[index] != 0) Rcpp::stop("candidate is already free");
      pt.free[index] = free; names.row_label[index] = "mi_p" + std::to_string(free);
    }
  } else if (kind != "release" || row < 1 || row > con->A_eq.rows()) Rcpp::stop("invalid equality-release candidate");
  for (Eigen::Index k = 0; k < con->A_eq.rows(); ++k) {
    if (kind == "release" && k == row - 1) continue;
    pt.op.push_back(parse::Op::EqConstraint); pt.lhs_var.push_back(-1); pt.rhs_var.push_back(-1);
    pt.group.push_back(0); pt.free.push_back(0); pt.exo.push_back(0); pt.fixed_value.push_back(0);
    names.row_lhs.push_back(affine_expression(con->A_eq, k, labels));
    std::ostringstream value; value << std::setprecision(17) << con->b_eq[k];
    names.row_rhs.push_back(value.str()); names.row_label.push_back(""); names.row_plabel.push_back(""); names.row_user.push_back(1);
  }
  return partable_df_from_lavaan(compat::lavaan::to_lavaan_partable(pt, names, {}));
}
