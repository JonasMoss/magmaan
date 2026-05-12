#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "latva/expected.hpp"
#include "latva/partable/partable.hpp"

namespace latva::fit {

// Linear equality constraints, reduced to a "merge groups of free
// parameters" reparameterization.
//
// v1 (P9 phase 1) handles only θ_i = θ_j constraints — the `==` rows
// `lavaanify` synthesizes from a shared label (`f =~ x1 + a*x2 + a*x3`,
// cross-group equality for measurement invariance) and explicit `a == b`
// statements. `build_eq_constraints` returns a PostError (with a clear
// message) for `<` / `>` rows or for an `==` side that isn't a bare
// identifier — arbitrary linear (`a == 2*b + c`) and nonlinear constraints
// are later phases.
//
// Reparameterization:
//   θ_full (size npar)  =  K · α,   α (size n_alpha)        n_alpha = npar − rank
//   θ_full[k]           =  α[group[k]]                       (so K is 0/1)
//   K (npar × n_alpha)  =  group-membership matrix
//   rank                =  npar − n_alpha                    (# independent equality
//                          constraints; df increases by `rank` vs. the
//                          unconstrained model)
//
// The unconstrained case is the identity reparameterization: group[k] = k,
// n_alpha = npar, rank = 0, active() == false.
struct EqConstraints {
  std::vector<std::int32_t> group;          // size npar; group[k] ∈ [0, n_alpha)
  std::int32_t              n_alpha = 0;    // # reduced parameters (= # equality groups)
  std::int32_t              npar    = 0;
  std::int32_t              rank    = 0;    // npar − n_alpha

  bool active() const noexcept { return rank > 0; }

  // θ_full (size npar) from α (size n_alpha): θ[k] = α[group[k]].
  Eigen::VectorXd expand(const Eigen::Ref<const Eigen::VectorXd>& alpha) const;
  // α (size n_alpha) from θ_full (size npar): α[g] = mean of θ over group g.
  Eigen::VectorXd contract_mean(const Eigen::Ref<const Eigen::VectorXd>& theta_full) const;
  // ∇_α (size n_alpha) from ∇_θ (size npar): ∇_α[g] = Σ over group g.
  Eigen::VectorXd reduce_gradient(const Eigen::Ref<const Eigen::VectorXd>& grad_full) const;
  // The npar × n_alpha 0/1 expansion matrix K (for vcov projection / inspection).
  Eigen::MatrixXd K() const;
};

// Build the equality-constraint reparameterization from a LatentStructure. Scans
// rows with `op == EqConstraint`; resolves each side (a single identifier —
// a `.pN.` plabel for the auto-synthesized rows, or a row `label` for an
// explicit `a == b`) to a 1-based free θ index and unions the two indices.
//
// Returns `PostError::Kind::NumericIssue` on:
//   - any `<` / `>` row present (inequalities not yet enforced),
//   - an `==` side that isn't a bare identifier (arbitrary linear/nonlinear
//     constraint expressions not yet supported),
//   - an `==` side that doesn't name a free parameter (fixed param, defined
//     param, or unknown label).
post_expected<EqConstraints>
build_eq_constraints(const partable::LatentStructure& pt);

}  // namespace latva::fit
