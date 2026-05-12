#include "latva/fit/constraints.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "latva/error.hpp"
#include "latva/expected.hpp"
#include "latva/parse/op.hpp"

namespace latva::fit {

namespace {

PostError err(std::string detail) {
  return PostError{PostError::Kind::NumericIssue, std::move(detail)};
}

// Canonical Expr text uses no whitespace; a single identifier therefore
// contains no operator/paren characters. (`a`, `.p3.` are identifiers;
// `2*b`, `(a+b)`, `-1` are not.)
bool is_bare_identifier(const std::string& s) noexcept {
  if (s.empty()) return false;
  for (char c : s) {
    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^' ||
        c == '(' || c == ')' || c == ' ' || c == '\t') {
      return false;
    }
  }
  return true;
}

}  // namespace

Eigen::VectorXd
EqConstraints::expand(const Eigen::Ref<const Eigen::VectorXd>& alpha) const {
  Eigen::VectorXd theta(static_cast<Eigen::Index>(npar));
  for (std::int32_t k = 0; k < npar; ++k) {
    theta(k) = alpha(group[static_cast<std::size_t>(k)]);
  }
  return theta;
}

Eigen::VectorXd
EqConstraints::contract_mean(const Eigen::Ref<const Eigen::VectorXd>& theta_full) const {
  Eigen::VectorXd alpha = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_alpha));
  std::vector<std::int32_t> cnt(static_cast<std::size_t>(n_alpha), 0);
  for (std::int32_t k = 0; k < npar; ++k) {
    const std::int32_t g = group[static_cast<std::size_t>(k)];
    alpha(g) += theta_full(k);
    ++cnt[static_cast<std::size_t>(g)];
  }
  for (std::int32_t g = 0; g < n_alpha; ++g) {
    if (cnt[static_cast<std::size_t>(g)] > 0) {
      alpha(g) /= static_cast<double>(cnt[static_cast<std::size_t>(g)]);
    }
  }
  return alpha;
}

Eigen::VectorXd
EqConstraints::reduce_gradient(const Eigen::Ref<const Eigen::VectorXd>& grad_full) const {
  Eigen::VectorXd grad_a = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_alpha));
  for (std::int32_t k = 0; k < npar; ++k) {
    grad_a(group[static_cast<std::size_t>(k)]) += grad_full(k);
  }
  return grad_a;
}

Eigen::MatrixXd EqConstraints::K() const {
  Eigen::MatrixXd Kmat = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(npar),
                                               static_cast<Eigen::Index>(n_alpha));
  for (std::int32_t k = 0; k < npar; ++k) {
    Kmat(k, group[static_cast<std::size_t>(k)]) = 1.0;
  }
  return Kmat;
}

post_expected<EqConstraints>
build_eq_constraints(const partable::LatentStructure& pt) {
  const std::int32_t npar = pt.n_free();
  EqConstraints out;
  out.npar = npar;
  out.group.resize(static_cast<std::size_t>(std::max<std::int32_t>(npar, 0)));
  for (std::int32_t k = 0; k < npar; ++k) {
    out.group[static_cast<std::size_t>(k)] = k;   // each free param its own group
  }
  out.n_alpha = npar;
  out.rank = 0;
  if (npar == 0) return out;

  // `.pN.` plabel → 1-based free index, and `label` → 1-based free index,
  // restricted to free (free > 0), non-constraint rows.
  std::unordered_map<std::string, std::int32_t> plabel_to_free;
  std::unordered_map<std::string, std::int32_t> label_to_free;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    if (!pt.plabel[i].empty()) plabel_to_free.try_emplace(pt.plabel[i], pt.free[i]);
    if (!pt.label[i].empty())  label_to_free.try_emplace(pt.label[i], pt.free[i]);
  }
  auto resolve = [&](const std::string& tok, const char* side)
      -> post_expected<std::int32_t> {
    if (!is_bare_identifier(tok)) {
      return std::unexpected(err(
          std::string("constraint ") + side + " '" + tok +
          "' is an expression; only `a == b` / shared-label equality is enforced "
          "(arbitrary linear/nonlinear constraints are not yet implemented)"));
    }
    if (auto it = plabel_to_free.find(tok); it != plabel_to_free.end()) return it->second;
    if (auto it = label_to_free.find(tok);  it != label_to_free.end())  return it->second;
    return std::unexpected(err(
        std::string("constraint ") + side + " '" + tok +
        "' does not name a free parameter (fixed/defined parameter or unknown label)"));
  };

  // Union-find over 1-based free indices (index 0 unused).
  std::vector<std::int32_t> parent(static_cast<std::size_t>(npar) + 1);
  for (std::int32_t k = 0; k <= npar; ++k) parent[static_cast<std::size_t>(k)] = k;
  auto find = [&parent](std::int32_t x) -> std::int32_t {
    while (parent[static_cast<std::size_t>(x)] != x) {
      parent[static_cast<std::size_t>(x)] =
          parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
      x = parent[static_cast<std::size_t>(x)];
    }
    return x;
  };
  auto unite = [&](std::int32_t a, std::int32_t b) {
    a = find(a);
    b = find(b);
    if (a != b) {
      // Keep the smaller index as the root so the eventual group numbering
      // follows free-index order.
      parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
    }
  };

  for (std::size_t i = 0; i < pt.size(); ++i) {
    const parse::Op op = pt.op[i];
    if (op == parse::Op::LtConstraint || op == parse::Op::GtConstraint) {
      return std::unexpected(err(
          "inequality constraints (`<` / `>`) are not yet enforced by fit()"));
    }
    if (op != parse::Op::EqConstraint) continue;
    auto li = resolve(pt.lhs[i], "lhs");
    if (!li.has_value()) return std::unexpected(li.error());
    auto ri = resolve(pt.rhs[i], "rhs");
    if (!ri.has_value()) return std::unexpected(ri.error());
    unite(*li, *ri);
  }

  // Compact the union-find roots into contiguous 0-based group indices,
  // ascending in free-index order.
  std::unordered_map<std::int32_t, std::int32_t> root_to_group;
  std::int32_t next_group = 0;
  for (std::int32_t k = 1; k <= npar; ++k) {
    const std::int32_t r = find(k);
    auto [it, inserted] = root_to_group.try_emplace(r, next_group);
    if (inserted) ++next_group;
    out.group[static_cast<std::size_t>(k - 1)] = it->second;
  }
  out.n_alpha = next_group;
  out.rank = npar - next_group;
  return out;
}

}  // namespace latva::fit
