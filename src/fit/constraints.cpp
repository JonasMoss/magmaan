#include "latva/fit/constraints.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/QR>
#include <Eigen/SVD>

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
  if (n_alpha == 0) return theta0;
  return theta0 + Kmat * alpha;
}

Eigen::VectorXd
EqConstraints::contract(const Eigen::Ref<const Eigen::VectorXd>& theta_full) const {
  if (n_alpha == 0) return Eigen::VectorXd(0);
  // Least-squares solve of K·α = θ − θ₀. For the 0/1 K (pure-merge case) KᵀK is
  // diagonal with entries = group sizes, so this is the per-group mean of θ.
  return Kmat.colPivHouseholderQr().solve(theta_full - theta0);
}

Eigen::VectorXd
EqConstraints::reduce_gradient(const Eigen::Ref<const Eigen::VectorXd>& grad_full) const {
  return Kmat.transpose() * grad_full;
}

namespace {

// Compact a possibly-non-contiguous group-id vector into 0-based contiguous ids
// in first-appearance order, into `out` (size npar). Returns the group count.
std::int32_t compact_groups(const std::vector<std::int32_t>& eq_groups,
                            std::int32_t npar,
                            std::vector<std::int32_t>& out) {
  out.assign(static_cast<std::size_t>(npar < 0 ? 0 : npar), 0);
  const bool have = static_cast<std::int32_t>(eq_groups.size()) == npar;
  std::unordered_map<std::int32_t, std::int32_t> remap;
  std::int32_t next = 0;
  for (std::int32_t k = 0; k < npar; ++k) {
    const std::int32_t raw = have ? eq_groups[static_cast<std::size_t>(k)] : k;
    auto [it, inserted] = remap.try_emplace(raw, next);
    if (inserted) ++next;
    out[static_cast<std::size_t>(k)] = it->second;
  }
  return next;
}

}  // namespace

post_expected<EqConstraints>
build_eq_constraints(const partable::LatentStructure& pt) {
  // `<` / `>` rows and genuinely-nonlinear `==` expressions are flagged by
  // lavaanify (`compute_eq_groups` + `resolve_lin_constraints`); those can't be
  // reparameterized away.
  if (pt.has_unenforced_constraints) {
    bool has_ineq = false;
    for (parse::Op op : pt.op)
      if (op == parse::Op::LtConstraint || op == parse::Op::GtConstraint) has_ineq = true;
    return std::unexpected(err(has_ineq
        ? "inequality constraints (`<` / `>`) are not yet enforced by fit()"
        : "this constraint kind (a nonlinear `==` expression) is not yet "
          "enforced by fit(); only linear equality is supported"));
  }

  const std::int32_t npar = pt.n_free();
  EqConstraints out;
  out.npar = npar;
  if (npar <= 0) {
    out.n_alpha = 0; out.rank = 0;
    out.theta0  = Eigen::VectorXd::Zero(0);
    out.Kmat    = Eigen::MatrixXd(0, 0);
    return out;
  }
  const std::size_t P = static_cast<std::size_t>(npar);

  // The merge partition from `eq_groups` (contiguous, first-appearance order).
  std::vector<std::int32_t> group;
  const std::int32_t n_groups = compact_groups(pt.eq_groups, npar, group);

  const std::size_t n_lin = pt.lin_constraint_d.size();   // # general-linear rows

  if (n_lin == 0) {
    // ---- Pure-merge path: bit-identical to the P9-phase-1 representation. ----
    out.group   = std::move(group);
    out.n_alpha = n_groups;
    out.rank    = npar - n_groups;
    out.Kmat    = Eigen::MatrixXd::Zero(npar, n_groups);
    for (std::int32_t k = 0; k < npar; ++k)
      out.Kmat(k, out.group[static_cast<std::size_t>(k)]) = 1.0;
    out.theta0  = Eigen::VectorXd::Zero(npar);
    return out;
  }

  // ---- General-linear path: stack the merge rows + the lin_constraint rows. ----
  // Merge rows: for each group, anchor = its first member; for every other
  // member k, the row e_anchor − e_k (= 0).
  std::vector<std::int32_t> first_of(static_cast<std::size_t>(n_groups), -1);
  for (std::int32_t k = 0; k < npar; ++k) {
    const std::int32_t g = group[static_cast<std::size_t>(k)];
    if (first_of[static_cast<std::size_t>(g)] < 0) first_of[static_cast<std::size_t>(g)] = k;
  }
  const std::int32_t n_merge = npar - n_groups;
  const Eigen::Index m = static_cast<Eigen::Index>(n_merge) + static_cast<Eigen::Index>(n_lin);

  Eigen::MatrixXd R_full = Eigen::MatrixXd::Zero(m, npar);
  Eigen::VectorXd d_full = Eigen::VectorXd::Zero(m);
  Eigen::Index r_idx = 0;
  for (std::int32_t k = 0; k < npar; ++k) {
    const std::int32_t f = first_of[static_cast<std::size_t>(group[static_cast<std::size_t>(k)])];
    if (f == k) continue;
    R_full(r_idx, f) = 1.0;
    R_full(r_idx, k) = -1.0;
    ++r_idx;
  }
  for (std::size_t i = 0; i < n_lin; ++i) {
    for (std::int32_t c = 0; c < npar; ++c)
      R_full(r_idx, c) = pt.lin_constraint_R[i * P + static_cast<std::size_t>(c)];
    d_full(r_idx) = pt.lin_constraint_d[i];
    ++r_idx;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(R_full, Eigen::ComputeFullU | Eigen::ComputeFullV);
  svd.setThreshold(1e-9);   // user constraint coefficients are O(1)-ish; this
                            // separates a genuine constraint from float noise.
  const Eigen::Index rank_r = svd.rank();
  const Eigen::Index na     = static_cast<Eigen::Index>(npar) - rank_r;

  const Eigen::VectorXd theta0 = svd.solve(d_full);            // min-norm LS solution
  const double resid = (R_full * theta0 - d_full).norm();
  if (resid > 1e-8 * std::max(1.0, d_full.norm() + R_full.norm())) {
    return std::unexpected(err("infeasible linear-equality constraint system "
                               "(residual " + std::to_string(resid) + ")"));
  }

  out.n_alpha = static_cast<std::int32_t>(na);
  out.rank    = npar - out.n_alpha;
  out.Kmat    = svd.matrixV().rightCols(na);                   // npar × na, orthonormal
  out.theta0  = theta0;
  // `group` left empty — the reparam is no longer a pure parameter merge.
  return out;
}

}  // namespace latva::fit
