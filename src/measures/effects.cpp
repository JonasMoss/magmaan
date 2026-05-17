#include "magmaan/measures/effects.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/parse/flat_partable.hpp"

namespace magmaan::measures::effects {

using estimate::Estimates;

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

// Forward-mode AD: each subexpression carries a scalar value and a gradient
// vector (size n_free) in θ-space. A `:=` row that has already been evaluated
// is itself an ADValue — so a later `:=` referring to it just reuses the
// stored value/gradient (chained references), and the delta-method SE of the
// composite falls out of the accumulated gradient.
struct ADValue {
  double          v = 0.0;
  Eigen::VectorXd dv;  // size n_free
};

using LabelMap   = std::unordered_map<std::string_view, std::int32_t>;
using FixedMap   = std::unordered_map<std::string_view, double>;
using DefinedMap = std::unordered_map<std::string_view, ADValue>;

// Resolution context for one `eval()` pass. References, not copies — the
// underlying maps outlive the call. `defined` holds every `:=` row that has
// already been evaluated in dependency order (so all of the current row's
// `:=` references are present).
struct Scope {
  std::size_t            n_free = 0;
  const Eigen::VectorXd& theta;
  const LabelMap&        label_to_free;   // user labels + `.pN.` plabels of free rows  → 1-based free idx
  const FixedMap&        label_to_fixed;  // user labels + `.pN.` plabels of fixed rows → fixed value
  const DefinedMap&      defined;         // earlier `:=` names → their AD value (θ-space)
};

post_expected<ADValue> eval(const parse::Expr& e, const Scope& sc);

ADValue zero_ad(std::size_t n_free) {
  return ADValue{0.0, Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_free))};
}

post_expected<ADValue> eval_num(const parse::Num& n, const Scope& sc) {
  ADValue out = zero_ad(sc.n_free);
  out.v = n.value;
  return out;
}

post_expected<ADValue> eval_param(const parse::Param& p, const Scope& sc) {
  // A `:=` name shadows a same-named row label inside another `:=` (lavaan
  // does the same — it strips def names out of the label-resolution list).
  if (auto it = sc.defined.find(p.text); it != sc.defined.end()) {
    return it->second;
  }
  if (auto it = sc.label_to_free.find(p.text); it != sc.label_to_free.end()) {
    const auto k = it->second;  // 1-based free index
    if (k < 1 || static_cast<std::size_t>(k) > sc.n_free) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string("`:=` references '") + std::string(p.text) +
              "' with free index " + std::to_string(k) + " out of range"));
    }
    ADValue out = zero_ad(sc.n_free);
    out.v = sc.theta(k - 1);
    out.dv(k - 1) = 1.0;
    return out;
  }
  if (auto it = sc.label_to_fixed.find(p.text); it != sc.label_to_fixed.end()) {
    ADValue out = zero_ad(sc.n_free);
    out.v = it->second;  // ∂(fixed)/∂θ = 0
    return out;
  }
  return std::unexpected(make_err(PostError::Kind::NumericIssue,
      std::string("`:=` references unknown identifier '") + std::string(p.text) +
          "'; not a labeled row, a `.pN.` plabel, or another `:=` row"));
}

post_expected<ADValue> eval_bin(const parse::BinNode& b, const Scope& sc) {
  auto lhs_or = eval(*b.lhs, sc);
  if (!lhs_or.has_value()) return std::unexpected(lhs_or.error());
  auto rhs_or = eval(*b.rhs, sc);
  if (!rhs_or.has_value()) return std::unexpected(rhs_or.error());
  const ADValue& a = *lhs_or;
  const ADValue& c = *rhs_or;
  ADValue out;
  out.dv.resize(static_cast<Eigen::Index>(sc.n_free));
  switch (b.op) {
    case parse::BinOp::Add:
      out.v  = a.v + c.v;
      out.dv = a.dv + c.dv;
      break;
    case parse::BinOp::Sub:
      out.v  = a.v - c.v;
      out.dv = a.dv - c.dv;
      break;
    case parse::BinOp::Mul:
      out.v  = a.v * c.v;
      out.dv = c.v * a.dv + a.v * c.dv;
      break;
    case parse::BinOp::Div:
      if (c.v == 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "`:=` division by zero"));
      }
      out.v  = a.v / c.v;
      out.dv = (a.dv * c.v - a.v * c.dv) / (c.v * c.v);  // (a/c)' = (a'c − ac')/c²
      break;
    case parse::BinOp::Pow:
      // (a^c)' = a^c · (c'·log|a| + c·a'/a). With c constant w.r.t. θ the
      // log|a| term drops out (c.dv == 0); with variable c and a ≤ 0 the
      // log is undefined.
      if (c.dv.cwiseAbs().maxCoeff() != 0.0 && a.v <= 0.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "`:=` a^b with non-constant exponent requires a > 0"));
      }
      out.v = std::pow(a.v, c.v);
      if (a.v == 0.0 && c.v < 1.0) {
        return std::unexpected(make_err(PostError::Kind::NumericIssue,
            "`:=` 0^b with b < 1 has unbounded gradient"));
      }
      out.dv = c.v * std::pow(a.v, c.v - 1.0) * a.dv;
      if (c.dv.cwiseAbs().maxCoeff() != 0.0) {
        out.dv += out.v * std::log(a.v) * c.dv;
      }
      break;
  }
  return out;
}

post_expected<ADValue> eval_un(const parse::UnNode& u, const Scope& sc) {
  auto arg_or = eval(*u.arg, sc);
  if (!arg_or.has_value()) return std::unexpected(arg_or.error());
  ADValue out;
  if (u.op == parse::UnOp::Neg) {
    out.v  = -arg_or->v;
    out.dv = -arg_or->dv;
  } else {  // Pos
    out = *arg_or;
  }
  return out;
}

post_expected<ADValue> eval(const parse::Expr& e, const Scope& sc) {
  return std::visit(
      [&](auto&& v) -> post_expected<ADValue> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::Num>) {
          return eval_num(v, sc);
        } else if constexpr (std::is_same_v<T, parse::Param>) {
          return eval_param(v, sc);
        } else if constexpr (std::is_same_v<T, parse::BinNode>) {
          return eval_bin(v, sc);
        } else if constexpr (std::is_same_v<T, parse::UnNode>) {
          return eval_un(v, sc);
        }
      },
      static_cast<const parse::Expr::variant&>(e));
}

// Collect every `Param` identifier appearing in an expression (with
// repeats — the caller dedups).
void collect_params(const parse::Expr& e, std::vector<std::string_view>& out) {
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::Param>) {
          out.push_back(v.text);
        } else if constexpr (std::is_same_v<T, parse::BinNode>) {
          collect_params(*v.lhs, out);
          collect_params(*v.rhs, out);
        } else if constexpr (std::is_same_v<T, parse::UnNode>) {
          collect_params(*v.arg, out);
        }
      },
      static_cast<const parse::Expr::variant&>(e));
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
    auto ad_or = eval(defs[i]->rhs, sc);
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
