#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/spec/partable.hpp"

namespace magmaan::estimate {

// The model's nonlinear equality constraints, `h(θ) == 0`, evaluated against
// the free parameter vector θ. Built from `LatentStructure.nl_constraints` —
// the name-free expression trees `resolve_lin_constraints` compiled — so it
// needs only the structure, no parameter names, at fit time.
//
// `h(θ)` is the m-vector of constraint residuals (each `lhs − rhs` of an `==`
// row); `jacobian(θ)` is the `m × npar` matrix `∂h/∂θ`, both evaluated by
// forward-mode automatic differentiation over the node pools.
//
// These are constraints with *clean* asymptotics: the constrained estimate is
// an interior point of the lower-dimensional manifold `{θ : h(θ) = 0}`. The
// augmented-Lagrangian fit path drives `h(θ̂) → 0`; the constrained vcov / df
// project with the Jacobian `H = ∂h/∂θ` evaluated at θ̂.
struct NonlinearEqConstraints {
  std::vector<spec::NlConstraint> rows;   // one compiled tree per nonlinear `==`
  std::int32_t                    npar = 0;

  bool active() const noexcept { return !rows.empty(); }
  std::int32_t m() const noexcept {
    return static_cast<std::int32_t>(rows.size());
  }

  // h(θ): size m(). Entry j is h_j(θ); the constraint is h_j(θ) == 0.
  Eigen::VectorXd h(const Eigen::Ref<const Eigen::VectorXd>& theta) const;
  // ∂h/∂θ: m() × npar.
  Eigen::MatrixXd jacobian(const Eigen::Ref<const Eigen::VectorXd>& theta) const;
};

// Build from a lavaanified structure (copies the compiled constraint trees).
NonlinearEqConstraints build_nl_constraints(const spec::LatentStructure& pt);

}  // namespace magmaan::estimate
