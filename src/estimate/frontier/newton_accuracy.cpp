#include "magmaan/estimate/frontier/newton_accuracy.hpp"

#include <cmath>
#include <numeric>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::estimate {

std::string_view to_string(NewtonAccuracyStatus s) noexcept {
  switch (s) {
    case NewtonAccuracyStatus::Available: return "available";
    case NewtonAccuracyStatus::Unavailable: return "unavailable";
    case NewtonAccuracyStatus::Unsupported: return "unsupported";
    case NewtonAccuracyStatus::NonpositiveCurvature: return "nonpositive_curvature";
    case NewtonAccuracyStatus::IllConditioned: return "ill_conditioned";
    case NewtonAccuracyStatus::SolveUnreliable: return "solve_unreliable";
  }
  return "unavailable";
}

}  // namespace magmaan::estimate

namespace magmaan::estimate::frontier {

NewtonAccuracyDiagnostics
newton_accuracy_from(const Eigen::VectorXd& G, const Eigen::MatrixXd& I,
                     NewtonAccuracyOptions opts) {
  NewtonAccuracyDiagnostics a;
  a.checked = true;
  a.budget = opts.budget;
  a.n_reduced = static_cast<std::int32_t>(G.size());
  if (I.rows() != G.size() || I.cols() != G.size() || !G.allFinite() ||
      !I.allFinite()) {
    return a;
  }
  if (G.size() == 0) {
    a.status = NewtonAccuracyStatus::Available;
    a.distance = a.predicted_gain = a.max_step = a.solve_residual = 0.0;
    a.condition = 1.0;
    a.passed = true;
    return a;
  }
  if ((I.diagonal().array() <= 0.0).any()) {
    a.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return a;
  }
  // Diagonal equilibration makes the conditioning guard insensitive to
  // diagonal changes of units. No inverse is formed.
  const Eigen::VectorXd scale = I.diagonal().array().sqrt().inverse();
  Eigen::MatrixXd C = scale.asDiagonal() * I * scale.asDiagonal();
  C = (0.5 * (C + C.transpose())).eval();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C, Eigen::EigenvaluesOnly);
  if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() <= 0.0) {
    a.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return a;
  }
  a.condition = eig.eigenvalues().maxCoeff() / eig.eigenvalues().minCoeff();
  if (a.condition > opts.max_condition) {
    a.status = NewtonAccuracyStatus::IllConditioned;
    return a;
  }
  Eigen::LLT<Eigen::MatrixXd> chol(C);
  if (chol.info() != Eigen::Success) {
    a.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return a;
  }
  const Eigen::VectorXd b = scale.asDiagonal() * G;
  const Eigen::VectorXd y = chol.solve(b);
  a.solve_residual =
      (C * y - b).norm() / (C.norm() * y.norm() + b.norm() + 1e-300);
  if (a.solve_residual > opts.max_solve_residual) {
    a.status = NewtonAccuracyStatus::SolveUnreliable;
    return a;
  }
  const double d2 = b.dot(y);
  if (!(d2 >= 0.0)) {
    a.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return a;
  }
  a.status = NewtonAccuracyStatus::Available;
  a.distance = std::sqrt(d2);
  a.predicted_gain = 0.5 * d2;
  a.max_step = (scale.asDiagonal() * y).cwiseAbs().maxCoeff();
  a.passed = a.distance <= opts.budget;
  return a;
}

namespace {

bool covariance_blocks_interior(const model::ModelEvaluator& ev,
                                const Eigen::VectorXd& theta, double tol) {
  auto mat = ev.assembled(theta);
  if (!mat) return false;
  for (const auto& b : mat->blocks) {
    for (const Eigen::MatrixXd* M : {&b.Psi, &b.Theta}) {
      if (M->size() == 0) continue;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(*M, Eigen::EigenvaluesOnly);
      if (es.info() != Eigen::Success) return false;
      const double top = es.eigenvalues().cwiseAbs().maxCoeff();
      if (es.eigenvalues().minCoeff() <= tol * top) return false;
    }
  }
  return true;
}

}  // namespace

NewtonAccuracyDiagnostics
newton_accuracy_ml(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                   const SampleStats& samp, const Estimates& est,
                   NewtonAccuracyOptions opts) {
  NewtonAccuracyDiagnostics fail;
  fail.checked = true;
  fail.budget = opts.budget;
  if (build_nl_constraints(pt).active()) {
    fail.status = NewtonAccuracyStatus::Unsupported;
    return fail;
  }
  auto con = build_eq_constraints(pt);
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!con || !ev) return fail;
  auto obj = ml_objective(*ev, samp);
  if (!obj) return fail;
  Eigen::VectorXd gradient;
  const double f = obj->f(est.theta, gradient);
  if (!std::isfinite(f) || !gradient.allFinite()) return fail;
  auto info = inference::information_observed_analytic(pt, rep, samp, est);
  if (!info) return fail;
  const double n = std::accumulate(samp.n_obs.begin(), samp.n_obs.end(), 0.0);
  const Eigen::VectorXd G = n * con->reduce_gradient(gradient);
  const Eigen::MatrixXd I = con->K().transpose() * (*info) * con->K();
  NewtonAccuracyDiagnostics a = newton_accuracy_from(G, I, opts);
  a.covariance_interior =
      covariance_blocks_interior(*ev, est.theta, opts.interior_eigen_tol);
  return a;
}

}  // namespace magmaan::estimate::frontier
