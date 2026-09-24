#include "magmaan/estimate/frontier/newton_accuracy.hpp"

#include <algorithm>
#include <cmath>

namespace magmaan::estimate::frontier {

NewtonBoxSolution solve_newton_box(
    const NewtonSystem& system, const Eigen::VectorXd& gradient,
    const Eigen::MatrixXd& normals, const Eigen::VectorXd& lower,
    NewtonAccuracyOptions opts) {
  NewtonBoxSolution r;
  r.applied = true; r.normals = normals; r.lower = lower;
  auto& out = r.solution;
  out.status = system.status; out.condition = system.condition;
  if (out.status != NewtonAccuracyStatus::Available) return r;
  out.status = NewtonAccuracyStatus::Unavailable;
  const Eigen::Index n = gradient.size(), m = lower.size();
  if (system.scale.size() != n || normals.cols() != n || normals.rows() != m ||
      !gradient.allFinite() || !normals.allFinite() || !lower.allFinite() ||
      (lower.array() > 0).any() || opts.box_max_iter < 1 ||
      !std::isfinite(opts.box_tolerance) || opts.box_tolerance <= 0) {
    r.detail = "invalid quadratic dimensions, controls, or infeasible zero correction";
    return r;
  }
  if (n == 0 || m == 0) {
    out = solve_newton_system(system, gradient);
    r.multipliers = Eigen::VectorXd::Zero(m);
    return r;
  }
  const auto& C = system.equilibrated_hessian;
  const Eigen::VectorXd g = system.scale.asDiagonal() * gradient;
  Eigen::MatrixXd A = normals * system.scale.asDiagonal();
  if (!A.allFinite() || !g.allFinite()) {
    r.detail = "nonfinite equilibrated quadratic or bounds";
    return r;
  }
  Eigen::VectorXd b = lower, row_scale = Eigen::VectorXd::Ones(m);
  for (Eigen::Index i = 0; i < m; ++i) {
    const double norm = A.row(i).stableNorm();
    if (norm > 0) { row_scale[i] = norm; A.row(i) /= norm; b[i] /= norm; }
  }
  if (!A.allFinite() || !b.allFinite()) {
    r.detail = "nonfinite normalized bounds";
    return r;
  }
  Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd multipliers = Eigen::VectorXd::Zero(m);
  std::vector<Eigen::Index> working;
  bool solved = false;
  const double tol = opts.box_tolerance;
  for (int iter = 0; iter < opts.box_max_iter; ++iter) {
    r.iterations = iter + 1;
    const Eigen::VectorXd q = C * y + g;
    const Eigen::VectorXd inverse_q = system.factorization.solve(q);
    Eigen::VectorXd direction = -inverse_q;
    Eigen::VectorXd lambda;
    if (!working.empty()) {
      Eigen::MatrixXd W(static_cast<Eigen::Index>(working.size()), n);
      for (std::size_t j = 0; j < working.size(); ++j) W.row(static_cast<Eigen::Index>(j)) = A.row(working[j]);
      const Eigen::MatrixXd inverse_W = system.factorization.solve(W.transpose());
      Eigen::LLT<Eigen::MatrixXd> schur(W * inverse_W);
      if (schur.info() != Eigen::Success) {
        r.detail = "dependent or numerically singular box working set";
        break;
      }
      lambda = schur.solve(W * inverse_q);
      direction += inverse_W * lambda;
    }
    if (!direction.allFinite() || !lambda.allFinite()) break;
    if (direction.norm() <= tol * (1 + y.norm())) {
      Eigen::Index drop = -1;
      if (lambda.size() && lambda.minCoeff(&drop) < -tol * (1 + q.norm())) {
        working.erase(working.begin() + drop);
        continue;
      }
      multipliers.setZero();
      for (std::size_t j = 0; j < working.size(); ++j)
        multipliers[working[j]] = lambda[static_cast<Eigen::Index>(j)];
      solved = true;
      break;
    }
    double alpha = 1;
    Eigen::Index blocking = -1;
    for (Eigen::Index i = 0; i < m; ++i) {
      if (std::find(working.begin(), working.end(), i) != working.end()) continue;
      const double rate = A.row(i).dot(direction);
      if (rate >= -tol * (1 + direction.norm())) continue;
      const double allowed = std::max(0.0, A.row(i).dot(y) - b[i]) / -rate;
      if (allowed < alpha) { alpha = allowed; blocking = i; }
    }
    y += alpha * direction;
    if (blocking >= 0) working.push_back(blocking);
  }
  r.working_set = std::move(working);
  out.step = system.scale.asDiagonal() * y;
  if (!solved) {
    if (r.detail.empty()) r.detail = "box quadratic solve did not converge";
    return r;
  }
  r.multipliers = multipliers.array() / row_scale.array();
  const Eigen::VectorXd slack = A * y - b;
  const Eigen::VectorXd dual = C * y + g - A.transpose() * multipliers;
  r.primal_residual = std::max(0.0, -slack.minCoeff()) / (1 + b.norm() + y.norm());
  r.dual_residual = std::max(dual.norm() / (1 + g.norm() + (C*y).norm()),
      std::max(0.0, -multipliers.minCoeff()) / (1 + multipliers.norm()));
  r.complementarity_residual = (multipliers.array() * slack.array()).matrix().norm() /
      (1 + std::abs(g.dot(y)) + std::abs(y.dot(C*y)));
  out.solve_residual = std::max({r.primal_residual, r.dual_residual, r.complementarity_residual});
  const double gain = -g.dot(y) - .5 * y.dot(C*y);
  if (!r.multipliers.allFinite() || !std::isfinite(r.primal_residual) ||
      !std::isfinite(r.dual_residual) || !std::isfinite(r.complementarity_residual) ||
      !out.step.allFinite() || !std::isfinite(gain) || !std::isfinite(out.solve_residual) || gain < 0) {
    r.detail = "invalid constrained correction or negative predicted gain";
    return r;
  }
  out.predicted_gain = gain; out.distance = std::sqrt(2 * gain);
  out.status = NewtonAccuracyStatus::Available;
  return r;
}
} // namespace magmaan::estimate::frontier
