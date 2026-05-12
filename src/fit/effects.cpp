#include "latva/fit/effects.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/parse/flat_partable.hpp"

namespace latva::fit {

namespace {

PostError make_err(PostError::Kind k, std::string detail) {
  return PostError{k, std::move(detail)};
}

// Forward-mode AD: each subexpression carries a scalar value and a
// gradient vector (size n_free). Param lookups consult `label_to_free`
// to convert an identifier into the right θ-index; unknown names
// surface as a `NumericIssue`.
struct ADValue {
  double          v   = 0.0;
  Eigen::VectorXd dv;  // size n_free
};

using LabelMap = std::unordered_map<std::string_view, std::int32_t>;
using FixedMap = std::unordered_map<std::string_view, double>;

post_expected<ADValue>
eval(const parse::Expr& e, std::size_t n_free,
     const Eigen::VectorXd& theta,
     const LabelMap& label_to_free,
     const FixedMap& label_to_fixed);

post_expected<ADValue>
eval_num(const parse::Num& n, std::size_t n_free) {
  ADValue out;
  out.v  = n.value;
  out.dv = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_free));
  return out;
}

post_expected<ADValue>
eval_param(const parse::Param& p, std::size_t n_free,
           const Eigen::VectorXd& theta,
           const LabelMap& label_to_free,
           const FixedMap& label_to_fixed) {
  ADValue out;
  out.dv = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_free));
  if (auto it = label_to_free.find(p.text); it != label_to_free.end()) {
    const auto k = it->second;       // 1-based free index
    if (k < 1 || static_cast<std::size_t>(k) > n_free) {
      return std::unexpected(make_err(PostError::Kind::NumericIssue,
          std::string("`:=` references label '") + std::string(p.text) +
              "' with free index " + std::to_string(k) + " out of range"));
    }
    out.v = theta(k - 1);
    out.dv(k - 1) = 1.0;
    return out;
  }
  if (auto it = label_to_fixed.find(p.text); it != label_to_fixed.end()) {
    out.v = it->second;
    // ∂(fixed)/∂θ = 0 — dv already zero.
    return out;
  }
  return std::unexpected(make_err(PostError::Kind::NumericIssue,
      std::string("`:=` references unknown label '") + std::string(p.text) +
          "'; not present as a labeled row in the partable"));
}

post_expected<ADValue>
eval_bin(const parse::BinNode& b, std::size_t n_free,
         const Eigen::VectorXd& theta,
         const LabelMap& label_to_free,
         const FixedMap& label_to_fixed) {
  auto lhs_or = eval(*b.lhs, n_free, theta, label_to_free, label_to_fixed);
  if (!lhs_or.has_value()) return std::unexpected(lhs_or.error());
  auto rhs_or = eval(*b.rhs, n_free, theta, label_to_free, label_to_fixed);
  if (!rhs_or.has_value()) return std::unexpected(rhs_or.error());
  const auto& a = *lhs_or;
  const auto& c = *rhs_or;
  ADValue out;
  out.dv.resize(static_cast<Eigen::Index>(n_free));
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
      // (a/c)' = (a'c − a c') / c²
      out.dv = (a.dv * c.v - a.v * c.dv) / (c.v * c.v);
      break;
    case parse::BinOp::Pow:
      // (a^c)' = a^c · (c'·log|a| + c·a'/a). For c constant w.r.t. θ,
      // the log|a| term disappears since c.dv = 0. For variable c and
      // a ≤ 0 the log is undefined; flag.
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

post_expected<ADValue>
eval_un(const parse::UnNode& u, std::size_t n_free,
        const Eigen::VectorXd& theta,
        const LabelMap& label_to_free,
        const FixedMap& label_to_fixed) {
  auto arg_or = eval(*u.arg, n_free, theta, label_to_free, label_to_fixed);
  if (!arg_or.has_value()) return std::unexpected(arg_or.error());
  ADValue out;
  if (u.op == parse::UnOp::Neg) {
    out.v  = -arg_or->v;
    out.dv = -arg_or->dv;
  } else {  // Pos
    out  = *arg_or;
  }
  return out;
}

post_expected<ADValue>
eval(const parse::Expr& e, std::size_t n_free,
     const Eigen::VectorXd& theta,
     const LabelMap& label_to_free,
     const FixedMap& label_to_fixed) {
  return std::visit(
      [&](auto&& v) -> post_expected<ADValue> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, parse::Num>) {
          return eval_num(v, n_free);
        } else if constexpr (std::is_same_v<T, parse::Param>) {
          return eval_param(v, n_free, theta, label_to_free, label_to_fixed);
        } else if constexpr (std::is_same_v<T, parse::BinNode>) {
          return eval_bin(v, n_free, theta, label_to_free, label_to_fixed);
        } else if constexpr (std::is_same_v<T, parse::UnNode>) {
          return eval_un(v, n_free, theta, label_to_free, label_to_fixed);
        }
      },
      static_cast<const parse::Expr::variant&>(e));
}

}  // namespace

post_expected<DefinedParams>
compute_defined(const parse::FlatPartable& flat,
                const partable::LatentStructure&  pt,
                const Estimates&           est,
                const Eigen::MatrixXd&     vcov) {
  const std::size_t n_free = static_cast<std::size_t>(est.theta.size());
  if (vcov.rows() != static_cast<Eigen::Index>(n_free) ||
      vcov.cols() != static_cast<Eigen::Index>(n_free)) {
    return std::unexpected(make_err(PostError::Kind::NumericIssue,
        "vcov shape doesn't match Estimates.theta size"));
  }

  // Build label → free-θ-index and label → fixed-value maps from pt.
  // First-row-wins for both — same-label rows are equality-constrained by
  // convention, so any matching index is acceptable. Constraint rows are
  // excluded (their labels name defined params, not free θ).
  LabelMap label_to_free;
  FixedMap label_to_fixed;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const bool is_constraint =
        (pt.op[i] == parse::Op::EqConstraint ||
         pt.op[i] == parse::Op::LtConstraint ||
         pt.op[i] == parse::Op::GtConstraint ||
         pt.op[i] == parse::Op::DefineParam);
    if (is_constraint || pt.label[i].empty()) continue;
    if (pt.free[i] > 0) {
      label_to_free.try_emplace(pt.label[i], pt.free[i]);
    } else if (std::isfinite(pt.fixed_value[i])) {
      label_to_fixed.try_emplace(pt.label[i], pt.fixed_value[i]);
    }
  }

  DefinedParams out;
  for (const auto& c : flat.constraints) {
    if (c.kind != parse::ConstraintKind::Define) continue;
    auto ad_or = eval(c.rhs, n_free, est.theta, label_to_free, label_to_fixed);
    if (!ad_or.has_value()) return std::unexpected(ad_or.error());
    const auto& ad = *ad_or;
    // SE via delta method: se² = grad' · vcov · grad. Clamp tiny
    // numerical negatives to zero before sqrt.
    const double var = ad.dv.dot(vcov * ad.dv);
    DefinedParam dp;
    dp.name  = std::string(c.name);
    dp.value = ad.v;
    dp.se    = (var > 0.0) ? std::sqrt(var) : 0.0;
    out.entries.push_back(std::move(dp));
  }
  return out;
}

}  // namespace latva::fit
