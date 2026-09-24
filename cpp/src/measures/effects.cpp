#include "magmaan/measures/effects.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/estimate/expr_eval.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/parse/flat_partable.hpp"

namespace magmaan::measures::effects {

using estimate::Estimates;

// The forward-mode AD expression evaluator is shared with the nonlinear
// equality-constraint path — see `magmaan/estimate/expr_eval.hpp`. A `:=` row
// already evaluated is itself an `ADValue`, so a later `:=` that refers to it
// reuses the stored value/gradient (chained references) and the delta-method
// SE of the composite falls out of the accumulated gradient.
using estimate::ADValue;
using estimate::collect_params;
using estimate::DefinedMap;
using estimate::eval_expr;
using estimate::FixedMap;
using estimate::LabelMap;
using estimate::Scope;

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

}  // namespace

post_expected<DefinedParams>
compute_defined(const parse::FlatPartable& flat,
                const spec::LatentStructure&  pt,
                const spec::LatentNames&      names,
                const Estimates&           est,
                const Eigen::MatrixXd&     vcov) {
  const std::size_t n_free = static_cast<std::size_t>(est.theta.size());
  if (vcov.rows() != static_cast<Eigen::Index>(n_free) ||
      vcov.cols() != static_cast<Eigen::Index>(n_free)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "vcov shape doesn't match Estimates.theta size"));
  }

  // Resolution maps: each non-constraint row exposes its user label (if any)
  // and its `.pN.` plabel as aliases for the same θ-index (free row) or fixed
  // value (fixed/exo row). First writer wins for labels — same-label rows are
  // equality-constrained, so any matching index is acceptable; plabels are
  // unique. Keys are string_views into `names.row_label` / `names.row_plabel`,
  // which outlive this call.
  LabelMap label_to_free;
  FixedMap label_to_fixed;
  const std::size_t nrow = pt.size();
  for (std::size_t i = 0; i < nrow; ++i) {
    if (pt.is_constraint_row(i)) continue;
    auto add_alias = [&](std::string_view key) {
      if (key.empty()) return;
      if (pt.free[i] > 0) {
        label_to_free.try_emplace(key, pt.free[i]);
      } else if (std::isfinite(pt.fixed_value[i])) {
        label_to_fixed.try_emplace(key, pt.fixed_value[i]);
      }
    };
    if (i < names.row_label.size())  add_alias(names.row_label[i]);
    if (i < names.row_plabel.size()) add_alias(names.row_plabel[i]);
  }

  // Gather the `:=` rows in source order, plus a name → source-index map.
  std::vector<const parse::Constraint*> defs;
  std::unordered_map<std::string_view, std::size_t> def_index;  // name → defs idx
  for (const auto& c : flat.constraints) {
    if (c.kind != parse::ConstraintKind::Define) continue;
    def_index.try_emplace(c.name, defs.size());  // first definition keeps its slot
    defs.push_back(&c);
  }
  if (defs.empty()) return DefinedParams{};

  // Dependency order: a `:=` whose RHS mentions another `:=` name depends on
  // it (lavaan does the same via lav_graph_order_adj_mat). Kahn's algorithm;
  // a remaining node after the sweep means a cycle (including a self-reference
  // like `a := a + 1`).
  const std::size_t n = defs.size();
  std::vector<std::vector<std::size_t>> dependents(n);  // i → defs that depend on i
  std::vector<std::size_t> in_degree(n, 0);
  {
    std::vector<std::string_view> params;
    for (std::size_t i = 0; i < n; ++i) {
      params.clear();
      collect_params(defs[i]->rhs, params);
      std::unordered_set<std::size_t> deps;  // dedup edges into i
      for (std::string_view nm : params) {
        if (auto it = def_index.find(nm); it != def_index.end()) deps.insert(it->second);
      }
      for (std::size_t d : deps) {
        dependents[d].push_back(i);
        ++in_degree[i];
      }
    }
  }
  std::vector<std::size_t> order;
  order.reserve(n);
  {
    std::vector<std::size_t> ready;
    for (std::size_t i = 0; i < n; ++i) if (in_degree[i] == 0) ready.push_back(i);
    while (!ready.empty()) {
      const std::size_t i = ready.back();
      ready.pop_back();
      order.push_back(i);
      for (std::size_t j : dependents[i]) {
        if (--in_degree[j] == 0) ready.push_back(j);
      }
    }
  }
  if (order.size() != n) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "circular `:=` parameter definitions (a definition references "
        "itself, directly or transitively)"));
  }

  // Evaluate in dependency order; store each result so later rows can chain.
  std::vector<ADValue> result(n);
  DefinedMap defined;
  defined.reserve(n);
  for (std::size_t i : order) {
    Scope sc{n_free, est.theta, label_to_free, label_to_fixed, defined};
    auto ad_or = eval_expr(defs[i]->rhs, sc);
    if (!ad_or.has_value()) return std::unexpected(ad_or.error());
    defined.insert_or_assign(defs[i]->name, *ad_or);
    result[i] = std::move(*ad_or);
  }

  // Emit in source order (matches lavaan's parameterEstimates() layout).
  DefinedParams out;
  out.entries.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const ADValue& ad = result[i];
    const double var = ad.dv.dot(vcov * ad.dv);  // delta-method variance
    DefinedParam dp;
    dp.name  = std::string(defs[i]->name);
    dp.value = ad.v;
    dp.se    = (var > 0.0) ? std::sqrt(var) : 0.0;  // clamp tiny numerical negatives
    out.entries.push_back(std::move(dp));
  }
  return out;
}

}  // namespace magmaan::measures::effects
