#pragma once

// A nonlinear reparameterization theta = g(u) as a closure transformer: the
// nonlinear sibling of the affine `reparameterize(problem, EqConstraints)`.
// The optimizer drives u; the wrapped problem is evaluated at g(u) and its
// gradient / Jacobian pulled back through dg/du. An optional residual-form
// penalty r_pen(u) is appended (scalar form adds 0.5 * ||r_pen||^2), which is
// how a chart pins directions the objective does not see (for example the
// radius of a sphere-normalized loading vector).

#include <functional>

#include <Eigen/Core>

#include "magmaan/optim/problem.hpp"

namespace magmaan::optim {

struct ParameterMap {
  Eigen::Index n_param = 0;                                   // dim u
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> expand;    // u -> theta
  std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> jacobian;  // dtheta/du
  // Optional penalty residual and its Jacobian (rows x n_param). Both empty
  // or both set.
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> penalty_residual;
  std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> penalty_jacobian;

  bool has_penalty() const noexcept {
    return static_cast<bool>(penalty_residual);
  }
};

// F_u(u) = F(g(u)) + 0.5 ||r_pen(u)||^2;  grad = G(u)' grad F + J_pen' r_pen.
ScalarProblem reparameterize(const ScalarProblem& prob, const ParameterMap& map);

// r_u(u) = [r(g(u)); r_pen(u)];  J_u = [J(g(u)) G(u); J_pen(u)].
GmmProblem reparameterize(const GmmProblem& prob, const ParameterMap& map);

}  // namespace magmaan::optim
