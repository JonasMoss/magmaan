#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate {

// Linear equality constraints, reduced to an *affine reparameterization* of the
// free parameter vector:
//
//   θ_full (size npar)  =  θ₀  +  K · α,    α (size n_alpha = npar − rank)
//   K (npar × n_alpha)  =  basis of ker(R_full), the stacked constraint
//                          Jacobian
//   θ₀ (size npar)      =  a particular (min-norm) solution of R_full · θ = d
//   rank                =  npar − n_alpha   (# independent equality constraints;
//                          df increases by `rank` vs. the unconstrained model)
//
// `R_full` stacks (a) the "parameter merge" rows derived from
// `LatentStructure.eq_groups` — `θ_i − θ_j = 0` for each member of a shared-label
// / bare-`a==b` / cross-group-invariance group — and (b) the general-linear rows
// in `LatentStructure.lin_constraint_R` / `lin_constraint_d` (`a == 2*b+c`,
// `Σλ == k`, …), filled by `resolve_lin_constraints`.
//
// Two cases:
//   * No general-linear rows (`lin_constraint_d` empty): `θ₀ = 0`, `K` is the
//     0/1 group-membership matrix (`K[k, group[k]] = 1`), and `group` /
//     `n_alpha` / `rank` describe it directly — bit-identical to the P9-phase-1
//     representation. (`group` is then a convenience field; the transform
//     methods always go through `Kmat`.)
//   * With general-linear rows: `K` is a direct sum of orthonormal bases, one
//     per connected constraint component. This preserves separable blocks such
//     as loading-only and intercept-only effect-coding constraints while still
//     describing the same affine surface. `θ₀ ≠ 0` in general, and `group` is
//     left empty (the reparam is no longer a pure merge).
//
// The unconstrained case is the identity reparameterization: `group[k] = k`,
// `K = I`, `θ₀ = 0`, `n_alpha = npar`, `rank = 0`, `active() == false`.
struct EqConstraints {
  Eigen::VectorXd           theta0;          // size npar; the particular solution
  Eigen::MatrixXd           Kmat;            // npar × n_alpha; kernel basis
  // Affine constraint system, equivalent to "θ lies on the manifold θ₀ + K·α":
  //   A_eq · θ = b_eq      (A_eq is `rank × npar`, full row rank;
  //                         A_eq orthonormal-by-rows in the general path,
  //                         a 0/±1 merge-row matrix in the pure-merge path.)
  // Every fit path enforces equality constraints by the K-reparameterization,
  // not by penalizing this system; `A_eq` / `b_eq` describe the constraint
  // surface for inspection and feed the equality-release score tests. Empty
  // (0 rows) when `rank == 0`.
  Eigen::MatrixXd           A_eq;
  Eigen::VectorXd           b_eq;
  std::vector<std::int32_t> group;           // size npar (pure-merge case only; else empty)
  std::int32_t              n_alpha = 0;     // # reduced parameters
  std::int32_t              npar    = 0;
  std::int32_t              rank    = 0;     // npar − n_alpha

  bool active() const noexcept { return rank > 0; }

  // θ_full (size npar) from α (size n_alpha):  θ = θ₀ + K·α.
  Eigen::VectorXd expand(const Eigen::Ref<const Eigen::VectorXd>& alpha) const;
  // α (size n_alpha) from a full θ:  least-squares solve of K·α = θ − θ₀.
  // (For the 0/1 K this is the per-group mean — KᵀK is diagonal.)
  Eigen::VectorXd contract(const Eigen::Ref<const Eigen::VectorXd>& theta_full) const;
  // ∇_α (size n_alpha) from ∇_θ (size npar):  ∇_α = Kᵀ·∇_θ.
  Eigen::VectorXd reduce_gradient(const Eigen::Ref<const Eigen::VectorXd>& grad_full) const;
  // The npar × n_alpha expansion matrix K (for vcov projection / inspection).
  const Eigen::MatrixXd& K() const noexcept { return Kmat; }
};

// Build the affine reparameterization from a `LatentStructure`. Combines the
// `eq_groups` merge partition with the general-linear `lin_constraint_R` /
// `lin_constraint_d` rows (both precomputed by `lavaanify` /
// `compute_eq_groups` + `resolve_lin_constraints`).
//
// Returns `PostError::Kind::NumericIssue` (with a kind-specific message) on any
// constraint it cannot reduce to the affine reparameterization:
//   - `pt.has_inequality_constraints` — a `<` / `>` row;
//   - `pt.has_unenforced_constraints` — a malformed `==` row;
//   - a non-empty `pt.nonlinear_eq_rows` — a nonlinear `==` expression
//     (`a == b*c`) — *unless* `allow_nonlinear` is set, in which case the
//     nonlinear rows are simply skipped here (the caller enforces them via a
//     constrained optimizer path; see `nl_constraints.hpp`);
//   - an infeasible linear-equality system (`R_full · θ = d` has no solution).
post_expected<EqConstraints>
build_eq_constraints(const spec::LatentStructure& pt,
                     bool allow_nonlinear = false);

}  // namespace magmaan::estimate
