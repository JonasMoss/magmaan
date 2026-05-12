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
  // `<` / `>` rows and arbitrary-expression `==` rows are flagged by lavaanify
  // (or `compute_eq_groups`); those phases aren't enforced yet.
  if (pt.has_unenforced_constraints) {
    bool has_ineq = false;
    for (parse::Op op : pt.op)
      if (op == parse::Op::LtConstraint || op == parse::Op::GtConstraint) has_ineq = true;
    return std::unexpected(err(has_ineq
        ? "inequality constraints (`<` / `>`) are not yet enforced by fit()"
        : "this constraint kind (arbitrary linear / nonlinear `==`) is not yet "
          "enforced by fit(); only `a == b` / shared-label equality is supported"));
  }

  const std::int32_t npar = pt.n_free();
  EqConstraints out;
  out.npar = npar;
  out.group.assign(static_cast<std::size_t>(std::max<std::int32_t>(npar, 0)), 0);
  if (npar == 0) { out.n_alpha = 0; out.rank = 0; return out; }

  // `pt.eq_groups` is the precomputed merged-parameter partition (identity if
  // empty). Re-derive the group count / rank from it; `pt.eq_groups` already
  // numbers groups contiguously in free-index order, but re-compact defensively
  // in case it was edited by hand.
  std::unordered_map<std::int32_t, std::int32_t> compact;
  std::int32_t next_group = 0;
  const bool have = static_cast<std::int32_t>(pt.eq_groups.size()) == npar;
  for (std::int32_t k = 0; k < npar; ++k) {
    const std::int32_t raw = have ? pt.eq_groups[static_cast<std::size_t>(k)] : k;
    auto [it, inserted] = compact.try_emplace(raw, next_group);
    if (inserted) ++next_group;
    out.group[static_cast<std::size_t>(k)] = it->second;
  }
  out.n_alpha = next_group;
  out.rank = npar - next_group;
  return out;
}

}  // namespace latva::fit
