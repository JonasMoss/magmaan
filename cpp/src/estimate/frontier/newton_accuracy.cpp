#include "magmaan/estimate/frontier/newton_accuracy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "magmaan/estimate/constraints.hpp"
#include "../detail_coordinates.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
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

std::string_view to_string(NewtonObjectiveKind k) noexcept {
  switch (k) {
    case NewtonObjectiveKind::CompleteDataMl: return "complete_data_ml";
    case NewtonObjectiveKind::LeastSquares: return "least_squares";
    case NewtonObjectiveKind::Fiml: return "fiml";
    case NewtonObjectiveKind::OrdinalLeastSquares: return "ordinal_least_squares";
    case NewtonObjectiveKind::MixedOrdinalLeastSquares: return "mixed_ordinal_least_squares";
    case NewtonObjectiveKind::CatMl: return "catml";
    case NewtonObjectiveKind::TwoLevelMl: return "twolevel_ml";
    case NewtonObjectiveKind::PenalizedMl: return "penalized_ml";
    case NewtonObjectiveKind::PenalizedFiml: return "penalized_fiml";
    case NewtonObjectiveKind::Supplied: return "supplied";
  }
  return "supplied";
}

std::string_view to_string(NewtonCurvatureKind k) noexcept {
  switch (k) {
    case NewtonCurvatureKind::AnalyticObserved: return "analytic";
    case NewtonCurvatureKind::GradientDifference: return "gradient_difference";
    case NewtonCurvatureKind::GaussNewton: return "gauss_newton";
    case NewtonCurvatureKind::Supplied: return "supplied";
  }
  return "supplied";
}

std::string_view to_string(NewtonMetricKind k) noexcept {
  switch (k) {
    case NewtonMetricKind::Hessian: return "hessian";
    case NewtonMetricKind::Sandwich: return "sandwich";
  }
  return "hessian";
}

bool newton_objective_requires_pd_sigma(NewtonObjectiveKind k) noexcept {
  switch (k) {
    case NewtonObjectiveKind::LeastSquares:
    case NewtonObjectiveKind::OrdinalLeastSquares:
    case NewtonObjectiveKind::MixedOrdinalLeastSquares:
      return false;
    default:
      return true;
  }
}

}  // namespace magmaan::estimate

namespace magmaan::estimate::frontier {

NewtonSystem prepare_newton_system(const Eigen::MatrixXd& H) {
  NewtonSystem out;
  if (H.rows() != H.cols() || !H.allFinite()) return out;
  if (H.rows() == 0) {
    out.status = NewtonAccuracyStatus::Available;
    out.condition = 1.0;
    return out;
  }
  if ((H.diagonal().array() <= 0.0).any()) {
    out.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return out;
  }
  out.scale = H.diagonal().array().sqrt().inverse();
  out.equilibrated_hessian = out.scale.asDiagonal() * H * out.scale.asDiagonal();
  auto& C = out.equilibrated_hessian;
  C = (0.5 * (C + C.transpose())).eval();
  if (!C.allFinite()) return out;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C, Eigen::EigenvaluesOnly);
  if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() <= 0.0) {
    out.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return out;
  }
  out.condition = eig.eigenvalues().maxCoeff() / eig.eigenvalues().minCoeff();
  out.factorization.compute(C);
  out.status = out.factorization.info() == Eigen::Success
      ? NewtonAccuracyStatus::Available : NewtonAccuracyStatus::NonpositiveCurvature;
  return out;
}

NewtonSolution solve_newton_system(const NewtonSystem& system,
                                   const Eigen::VectorXd& gradient) {
  NewtonSolution out;
  out.condition = system.condition;
  out.status = system.status;
  if (out.status != NewtonAccuracyStatus::Available) return out;
  out.status = NewtonAccuracyStatus::Unavailable;
  if (gradient.size() != system.scale.size() || !gradient.allFinite()) return out;
  if (gradient.size() == 0) {
    out.status = NewtonAccuracyStatus::Available;
    out.distance = out.predicted_gain = out.solve_residual = 0.0;
    return out;
  }
  const Eigen::VectorXd b = system.scale.asDiagonal() * gradient;
  const Eigen::VectorXd y = system.factorization.solve(b);
  const auto& C = system.equilibrated_hessian;
  out.step = -(system.scale.asDiagonal() * y);
  out.solve_residual =
      (C * y - b).norm() / (C.norm() * y.norm() + b.norm() + 1e-300);
  const double d2 = b.dot(y);
  if (!out.step.allFinite() || !std::isfinite(out.solve_residual) ||
      !std::isfinite(d2)) return out;
  if (d2 < 0.0) {
    out.status = NewtonAccuracyStatus::NonpositiveCurvature;
    return out;
  }
  out.status = NewtonAccuracyStatus::Available;
  out.distance = std::sqrt(d2);
  out.predicted_gain = 0.5 * d2;
  return out;
}

NewtonAccuracyDiagnostics assess_newton_accuracy(
    const NewtonSolution& solution, NewtonAccuracyOptions opts) {
  NewtonAccuracyDiagnostics a;
  a.checked = true;
  a.budget = opts.budget;
  a.status = solution.status;
  a.n_reduced = static_cast<std::int32_t>(solution.step.size());
  a.condition = solution.condition;
  a.solve_residual = solution.solve_residual;
  if (!std::isfinite(opts.budget) || opts.budget < 0.0 ||
      !std::isfinite(opts.max_condition) || opts.max_condition < 1.0 ||
      !std::isfinite(opts.max_solve_residual) || opts.max_solve_residual < 0.0) {
    a.status = NewtonAccuracyStatus::Unavailable;
    return a;
  }
  if (a.status != NewtonAccuracyStatus::Available) return a;
  if (!std::isfinite(a.condition) || !std::isfinite(a.solve_residual) ||
      !std::isfinite(solution.distance) || !std::isfinite(solution.predicted_gain) ||
      !solution.step.allFinite()) {
    a.status = NewtonAccuracyStatus::Unavailable;
    return a;
  }
  a.distance = solution.distance;
  a.predicted_gain = solution.predicted_gain;
  a.max_step = solution.step.size() ? solution.step.cwiseAbs().maxCoeff() : 0.0;
  if (a.condition > opts.max_condition) {
    a.status = NewtonAccuracyStatus::IllConditioned;
    return a;
  }
  if (a.solve_residual > opts.max_solve_residual) {
    a.status = NewtonAccuracyStatus::SolveUnreliable;
    return a;
  }
  a.passed = a.distance <= opts.budget;
  return a;
}

NewtonAccuracyDiagnostics
newton_accuracy_from(const Eigen::VectorXd& G, const Eigen::MatrixXd& I,
                     NewtonAccuracyOptions opts) {
  if (I.rows() != G.size() || I.cols() != G.size() || !G.allFinite()) {
    auto out = assess_newton_accuracy(NewtonSolution{}, opts);
    out.n_reduced = static_cast<std::int32_t>(G.size());
    return out;
  }
  auto out = assess_newton_accuracy(
      solve_newton_system(prepare_newton_system(I), G), opts);
  out.n_reduced = static_cast<std::int32_t>(G.size());
  return out;
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

NewtonDerivatives evaluate_newton_ml(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const SampleStats& samp, const Eigen::VectorXd& theta) {
  NewtonDerivatives out;
  out.objective_kind = NewtonObjectiveKind::CompleteDataMl;
  out.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
  out.theta = theta;
  if (theta.size() != pt.n_free() || !theta.allFinite()) return out;
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!ev) return out;
  auto obj = ml_objective(*ev, samp);
  if (!obj) return out;
  out.n_obs = std::accumulate(samp.n_obs.begin(), samp.n_obs.end(), 0.0);
  if (!(out.n_obs > 0.0) || !std::isfinite(out.n_obs)) return out;
  out.objective = obj->f(theta, out.gradient);
  out.gradient *= out.n_obs;
  out.native_to_total = out.n_obs;
  if (!std::isfinite(out.objective) || !out.gradient.allFinite()) return out;
  if (theta.size() == 0) {  // a fully specified model: no direction to move
    out.gradient.resize(0);
    out.hessian.resize(0, 0);
    out.status = NewtonAccuracyStatus::Available;
    return out;
  }
  Estimates est;
  est.theta = theta;
  auto info = inference::information_observed_analytic(pt, rep, samp, est);
  if (!info) return out;
  out.hessian = std::move(*info);
  if (!out.gradient.allFinite() || !out.hessian.allFinite()) return out;
  out.status = NewtonAccuracyStatus::Available;
  return out;
}

NewtonAudit audit_newton_ml(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const SampleStats& samp, const Eigen::VectorXd& theta,
    StationarityDomain domain, NewtonAccuracyOptions opts) {
  auto con = build_eq_constraints(pt);
  const bool single_level = rep.block_info.empty() ||
      (rep.block_info.size() == 1 && rep.block_info[0].role == model::BlockLevel::Single);
  const bool normalize = domain == StationarityDomain::Psd && rep.dims.size() == 1 &&
      single_level && con && !con->active() && !build_nl_constraints(pt).active();
  if (!normalize)
    return audit_newton_derivatives(pt, rep,
        evaluate_newton_ml(pt, rep, samp, theta), domain, opts);

  auto variables = driven::variable_units(pt, rep, samp);
  auto units = parameter_units(pt, rep, samp);
  auto unavailable = [&](const char* detail) {
    NewtonDerivatives d;
    d.theta = theta;
    d.objective_kind = NewtonObjectiveKind::CompleteDataMl;
    d.curvature_kind = NewtonCurvatureKind::AnalyticObserved;
    d.detail = detail;
    return audit_newton_derivatives(pt, rep, std::move(d), domain, opts);
  };
  if (!variables || !units || units->size() != theta.size() ||
      !units->allFinite() || (units->array() <= 0.0).any())
    return unavailable("PSD accuracy normalization requires finite positive sample/parameter units");

  auto normalized_pt = pt;
  auto normalized_rep = rep;
  auto normalized_sample = samp;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto& c = rep.cell_for_row[i];
    if (!c.used) continue;
    const double u = driven::matrix_cell_unit(*variables, c.mat, c.row, c.col,
                                              static_cast<std::size_t>(c.block));
    if (!(u > 0.0) || !std::isfinite(u))
      return unavailable("PSD accuracy normalization encountered invalid matrix-cell units");
    if (pt.free[i] > 0) {
      // A parameter shared across cells cannot acquire two different units.
      const double assigned = (*units)(pt.free[i] - 1);
      if (std::abs(u / assigned - 1.0) > 1e-12)
        return unavailable("PSD accuracy normalization needs consistent units for shared parameters");
    } else {
      normalized_pt.fixed_value[i] /= u;
    }
  }
  for (auto& c : normalized_rep.structural_cells) {
    const double u = driven::matrix_cell_unit(*variables, c.mat, c.row, c.col,
                                              static_cast<std::size_t>(c.block));
    if (!(u > 0.0) || !std::isfinite(u))
      return unavailable("PSD accuracy normalization encountered invalid structural-cell units");
    c.value /= u;
  }
  const auto& sd = variables->observed[0];
  if (normalized_sample.S[0].cols() != sd.size() ||
      (!normalized_sample.mean.empty() &&
       (normalized_sample.mean.size() != 1 || normalized_sample.mean[0].size() != sd.size())))
    return unavailable("PSD accuracy normalization requires compatible sample dimensions");
  normalized_sample.S[0].array() /= (sd * sd.transpose()).array();
  if (!normalized_sample.mean.empty())
    normalized_sample.mean[0].array() /= sd.array();
  const Eigen::VectorXd normalized_theta = theta.cwiseQuotient(*units);
  auto out = audit_newton_derivatives(normalized_pt, normalized_rep,
      evaluate_newton_ml(normalized_pt, normalized_rep, normalized_sample, normalized_theta),
      domain, opts);
  out.unit_normalized = true;

  // Keep reusable derivative artifacts in the caller's coordinates. The
  // reduced solve stays normalized; its full step is still B * solution.step.
  const auto inverse = units->cwiseInverse().eval();
  out.derivatives.theta = theta;
  if (out.derivatives.gradient.size() == units->size())
    out.derivatives.gradient.array() *= inverse.array();
  if (out.derivatives.hessian.rows() == units->size() &&
      out.derivatives.hessian.cols() == units->size())
    out.derivatives.hessian = inverse.asDiagonal() * out.derivatives.hessian * inverse.asDiagonal();
  if (out.geometry.equality_basis.rows() == units->size())
    out.geometry.equality_basis = units->asDiagonal() * out.geometry.equality_basis;
  if (out.geometry.curvature_correction.rows() == units->size() &&
      out.geometry.curvature_correction.cols() == units->size())
    out.geometry.curvature_correction = inverse.asDiagonal() * out.geometry.curvature_correction * inverse.asDiagonal();
  out.diagnostics = assess_newton_accuracy(out, opts);
  return out;
}

NewtonAudit audit_newton_derivatives(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    NewtonDerivatives derivatives, StationarityDomain domain,
    NewtonAccuracyOptions opts, const Bounds& bounds, double active_bound_tol) {
  NewtonAudit out;
  out.options = opts;
  out.bounds = bounds;
  out.active_bound_tol = active_bound_tol;
  out.derivatives = std::move(derivatives);
  bool valid_bounds = bounds.lower.size() == bounds.upper.size() &&
      std::isfinite(active_bound_tol) && active_bound_tol >= 0;
  bool has_bounds = false;
  std::vector<Eigen::Index> fixed_bounds;
  std::vector<bool> psd_diagonal(static_cast<std::size_t>(out.derivatives.theta.size()), false);
  if (domain == StationarityDomain::Psd) {
    for (std::size_t i = 0; i < pt.size(); ++i) {
      if (pt.free[i] <= 0 || pt.free[i] > out.derivatives.theta.size()) continue;
      const auto& c = rep.cell_for_row[i];
      if (c.used && c.row == c.col && (c.mat == model::MatId::Theta || c.mat == model::MatId::Psi))
        psd_diagonal[static_cast<std::size_t>(pt.free[i] - 1)] = true;
    }
  }
  if (!bounds.empty()) {
    const auto& theta = out.derivatives.theta;
    valid_bounds = valid_bounds && bounds.lower.size() == theta.size() &&
        !bounds.lower.array().isNaN().any() && !bounds.upper.array().isNaN().any();
    if (valid_bounds) {
      valid_bounds = !(bounds.lower.array() > bounds.upper.array()).any() &&
          !(theta.array() < bounds.lower.array()).any() && !(theta.array() > bounds.upper.array()).any();
      for (Eigen::Index i = 0; i < theta.size(); ++i) {
        has_bounds = has_bounds || std::isfinite(bounds.lower[i]) || std::isfinite(bounds.upper[i]);
        if (std::isfinite(bounds.lower[i]) && bounds.lower[i] == bounds.upper[i])
          fixed_bounds.push_back(i);
      }
    }
  }
  if (fixed_bounds.empty()) {
    out.geometry = prepare_newton_geometry(pt, rep, out.derivatives, domain, opts.interior_eigen_tol);
  } else {
    auto prepared = out.derivatives;
    prepared.fixed_coordinates.insert(prepared.fixed_coordinates.end(), fixed_bounds.begin(), fixed_bounds.end());
    out.geometry = prepare_newton_geometry(pt, rep, prepared, domain, opts.interior_eigen_tol);
  }
  if (!valid_bounds) {
    out.geometry.status = NewtonAccuracyStatus::Unavailable;
    out.derivatives.detail = "Newton audit: invalid bounds or point outside the supplied box";
  }
  if (out.geometry.status == NewtonAccuracyStatus::Available) {
    out.system = prepare_newton_system(out.geometry.reduced_hessian);
    if (!has_bounds) {
      out.solution = solve_newton_system(out.system, out.geometry.reduced_gradient);
    } else {
      const Eigen::MatrixXd B = out.geometry.equality_basis * out.geometry.tangent_basis;
      std::vector<Eigen::VectorXd> rows;
      std::vector<double> rhs;
      bool active_nonredundant = false;
      const auto& theta = out.derivatives.theta;
      for (Eigen::Index i = 0; i < theta.size(); ++i) {
        const bool redundant_lower = domain == StationarityDomain::Psd &&
            psd_diagonal[static_cast<std::size_t>(i)] && bounds.lower[i] <= 0;
        const bool fixed = bounds.lower[i] == bounds.upper[i];
        if (fixed) active_nonredundant = true;
        if (std::isfinite(bounds.lower[i]) && !redundant_lower && !fixed) {
          rows.emplace_back(B.row(i).transpose()); rhs.push_back(bounds.lower[i] - theta[i]);
          active_nonredundant = active_nonredundant || theta[i] - bounds.lower[i] <= active_bound_tol;
        }
        if (std::isfinite(bounds.upper[i]) && !fixed) {
          rows.emplace_back(-B.row(i).transpose()); rhs.push_back(theta[i] - bounds.upper[i]);
          active_nonredundant = active_nonredundant || bounds.upper[i] - theta[i] <= active_bound_tol;
        }
      }
      Eigen::MatrixXd A(static_cast<Eigen::Index>(rows.size()), B.cols());
      Eigen::VectorXd b(static_cast<Eigen::Index>(rhs.size()));
      for (std::size_t j = 0; j < rows.size(); ++j) {
        A.row(static_cast<Eigen::Index>(j)) = rows[j].transpose();
        b[static_cast<Eigen::Index>(j)] = rhs[j];
      }
      const bool psd_face = domain == StationarityDomain::Psd && !out.geometry.covariance_interior;
      if (psd_face) {
        auto candidate = solve_newton_system(out.system, out.geometry.reduced_gradient);
        const bool feasible_step = candidate.status == NewtonAccuracyStatus::Available &&
            ((A * candidate.step - b).array() >= 0).all();
        if (active_nonredundant || (candidate.status == NewtonAccuracyStatus::Available && !feasible_step)) {
          out.geometry.status = NewtonAccuracyStatus::Unsupported;
          out.solution.status = NewtonAccuracyStatus::Unsupported;
          out.derivatives.detail = "Newton audit: interacting box and singular PSD constraints need joint multiplier geometry";
        } else {
          out.box.applied = true; out.box.normals = A; out.box.lower = b;
          out.box.multipliers = Eigen::VectorXd::Zero(b.size());
          out.box.solution = candidate; out.solution = std::move(candidate);
        }
      } else {
        out.box = solve_newton_box(out.system, out.geometry.reduced_gradient, A, b, opts);
        out.solution = out.box.solution;
      }
    }
  } else {
    out.solution.status = out.geometry.status;
  }
  // Sandwich metric: measure the step s by d^2 = (H s)' Omega^{-1} (H s). For
  // an unconstrained step H s = -G exactly; a box step keeps its own H s. The
  // Hessian still decides curvature. A singular gradient variance leaves no
  // standard-error scale, which fails like an ill-conditioned Hessian.
  if (out.derivatives.metric_kind == NewtonMetricKind::Sandwich &&
      out.solution.status == NewtonAccuracyStatus::Available) {
    const Eigen::VectorXd y = out.box.applied
        ? Eigen::VectorXd(out.geometry.reduced_hessian * out.solution.step)
        : Eigen::VectorXd(-out.geometry.reduced_gradient);
    out.metric_system = prepare_newton_system(out.geometry.reduced_metric);
    const auto m = solve_newton_system(out.metric_system, y);
    if (m.status != NewtonAccuracyStatus::Available) {
      out.solution.status = NewtonAccuracyStatus::IllConditioned;
    } else {
      out.solution.distance = m.distance;
      out.solution.condition = std::max(out.solution.condition, m.condition);
      out.solution.solve_residual =
          std::max(out.solution.solve_residual, m.solve_residual);
    }
  }
  out.diagnostics = assess_newton_accuracy(out, opts);
  return out;
}

NewtonAccuracyDiagnostics assess_newton_accuracy(const NewtonAudit& audit) {
  return assess_newton_accuracy(audit, audit.options);
}

NewtonAccuracyDiagnostics assess_newton_accuracy(
    const NewtonAudit& audit, NewtonAccuracyOptions opts) {
  auto out = assess_newton_accuracy(audit.solution, opts);
  out.objective = audit.derivatives.objective_kind;
  out.curvature = audit.derivatives.curvature_kind;
  out.metric = audit.derivatives.metric_kind;
  out.n_reduced = static_cast<std::int32_t>(audit.geometry.reduced_gradient.size());
  out.psd_domain = audit.geometry.domain == StationarityDomain::Psd;
  out.unit_normalized = audit.unit_normalized;
  out.box_constrained = audit.box.applied;
  out.covariance_interior = audit.geometry.covariance_interior;
  out.null_directions = audit.geometry.null_directions;
  out.constrained_directions = audit.geometry.constrained_directions;
  out.min_multiplier = audit.geometry.min_multiplier;
  return out;
}

NewtonAccuracyDiagnostics
newton_accuracy_ml(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                   const SampleStats& samp, const Estimates& est,
                   NewtonAccuracyOptions opts) {
  return audit_newton_ml(pt, rep, samp, est.theta, StationarityDomain::Ambient, opts).diagnostics;
}

NewtonAccuracyDiagnostics
newton_accuracy_ml_psd(const spec::LatentStructure& pt, const model::MatrixRep& rep,
                       const SampleStats& samp, const Estimates& est,
                       NewtonAccuracyOptions opts) {
  return audit_newton_ml(pt, rep, samp, est.theta, StationarityDomain::Psd, opts).diagnostics;
}

}  // namespace magmaan::estimate::frontier

namespace magmaan::estimate::frontier {

namespace {

// ⟨N, dC/dθ_k⟩_F for every free θ_k that lands in block `block` of `mat`.
Eigen::VectorXd block_adjoint(const Eigen::MatrixXd& N, model::MatId mat,
                              std::int32_t block,
                              const spec::LatentStructure& pt,
                              const model::MatrixRep& rep, Eigen::Index npar) {
  Eigen::VectorXd out = Eigen::VectorXd::Zero(npar);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.free[i] <= 0) continue;
    const model::Cell& cell = rep.cell_for_row[i];
    if (!cell.used || cell.mat != mat || cell.block != block) continue;
    const Eigen::Index k = pt.free[i] - 1;
    if (k < 0 || k >= npar || cell.row < 0 || cell.col < 0 ||
        cell.row >= N.rows() || cell.col >= N.cols()) continue;
    out(k) += cell.row == cell.col ? N(cell.row, cell.col)
                                   : 2.0 * N(cell.row, cell.col);
  }
  return out;
}

// Frobenius-orthonormal basis element of the symmetric q x q matrices.
Eigen::MatrixXd sym_basis(Eigen::Index q, Eigen::Index i, Eigen::Index j) {
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(q, q);
  if (i == j) {
    E(i, i) = 1.0;
  } else {
    E(i, j) = E(j, i) = 1.0 / std::sqrt(2.0);
  }
  return E;
}

struct FaceComponent {
  model::MatId mat = model::MatId::Psi;
  std::int32_t block = 0;
  Eigen::Index dim = 0;               // full block dimension
  std::vector<Eigen::Index> idx;      // component rows within the block
  Eigen::MatrixXd U;                  // |idx| x q numerical null basis
  Eigen::MatrixXd P;                  // pseudo-inverse on the range
  Eigen::MatrixXd Uc;                 // held null directions
  Eigen::VectorXd omega_c;            // their multipliers
};

// Embed a component-local matrix into its full block.
Eigen::MatrixXd embed(const FaceComponent& f, const Eigen::MatrixXd& local) {
  Eigen::MatrixXd full = Eigen::MatrixXd::Zero(f.dim, f.dim);
  for (std::size_t a = 0; a < f.idx.size(); ++a)
    for (std::size_t b = 0; b < f.idx.size(); ++b)
      full(f.idx[a], f.idx[b]) =
          local(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b));
  return full;
}

// Rows d/dθ of the basis components of W' C W, W a component-local basis.
void append_face_rows(std::vector<Eigen::VectorXd>& rows,
                      const FaceComponent& f, const Eigen::MatrixXd& W,
                      const spec::LatentStructure& pt,
                      const model::MatrixRep& rep, Eigen::Index npar) {
  const Eigen::Index q = W.cols();
  for (Eigen::Index j = 0; j < q; ++j)
    for (Eigen::Index i = 0; i <= j; ++i) {
      const Eigen::MatrixXd N = W * sym_basis(q, i, j) * W.transpose();
      rows.push_back(block_adjoint(embed(f, N), f.mat, f.block, pt, rep, npar));
    }
}

Eigen::MatrixXd stack_rows(const std::vector<Eigen::VectorXd>& rows,
                           Eigen::Index npar) {
  Eigen::MatrixXd A(static_cast<Eigen::Index>(rows.size()), npar);
  for (std::size_t r = 0; r < rows.size(); ++r)
    A.row(static_cast<Eigen::Index>(r)) = rows[r].transpose();
  return A;
}

}  // namespace

NewtonGeometry prepare_newton_geometry(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const NewtonDerivatives& derivatives, StationarityDomain domain,
    double interior_eigen_tol) {
  NewtonGeometry fail;
  fail.domain = domain;
  fail.interior_eigen_tol = interior_eigen_tol;
  if (build_nl_constraints(pt).active()) {
    fail.status = NewtonAccuracyStatus::Unsupported;
    return fail;
  }
  if (derivatives.status == NewtonAccuracyStatus::Unsupported) {
    fail.status = NewtonAccuracyStatus::Unsupported;
    return fail;
  }
  if (derivatives.status != NewtonAccuracyStatus::Available ||
      derivatives.theta.size() != pt.n_free() ||
      derivatives.gradient.size() != pt.n_free() ||
      derivatives.hessian.rows() != pt.n_free() ||
      derivatives.hessian.cols() != pt.n_free() ||
      !derivatives.theta.allFinite() || !derivatives.gradient.allFinite() ||
      !derivatives.hessian.allFinite() || !std::isfinite(interior_eigen_tol) ||
      interior_eigen_tol < 0.0) return fail;
  const bool sandwich = derivatives.metric_kind == NewtonMetricKind::Sandwich;
  if (sandwich && (derivatives.metric.rows() != pt.n_free() ||
                   derivatives.metric.cols() != pt.n_free() ||
                   !derivatives.metric.allFinite())) return fail;
  auto con = build_eq_constraints(pt);
  auto ev = model::ModelEvaluator::build(pt, rep);
  if (!con || !ev) return fail;
  const auto& theta = derivatives.theta;
  const Eigen::Index npar = theta.size();
  Eigen::MatrixXd K = con->K();
  if (!derivatives.fixed_coordinates.empty()) {
    Eigen::MatrixXd A(derivatives.fixed_coordinates.size(), K.cols());
    for (std::size_t j = 0; j < derivatives.fixed_coordinates.size(); ++j) {
      const Eigen::Index k = derivatives.fixed_coordinates[j];
      if (k < 0 || k >= npar) return fail;
      A.row(static_cast<Eigen::Index>(j)) = K.row(k);
    }
    if (K.cols() > 0) {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
      const Eigen::Index rank = svd.rank();
      K = (K * svd.matrixV().rightCols(K.cols() - rank)).eval();
    }
  }
  const Eigen::VectorXd G = K.transpose() * derivatives.gradient;
  const Eigen::MatrixXd I = K.transpose() * derivatives.hessian * K;
  NewtonGeometry out = fail;
  out.equality_basis = K;
  out.tangent_basis = Eigen::MatrixXd::Identity(K.cols(), K.cols());
  out.curvature_correction = Eigen::MatrixXd::Zero(npar, npar);
  if (domain == StationarityDomain::Ambient) {
    out.reduced_gradient = G;
    out.reduced_hessian = I;
    if (sandwich) out.reduced_metric = K.transpose() * derivatives.metric * K;
    out.covariance_interior = covariance_blocks_interior(*ev, theta, interior_eigen_tol);
    out.status = NewtonAccuracyStatus::Available;
    return out;
  }
  auto mats = ev->assembled(theta);
  if (!mats) return fail;

  // Singular components of every primitive block.
  std::vector<FaceComponent> faces;
  std::vector<Eigen::VectorXd> forced;  // fixed-zero-variance rows: dC_rj = 0
  bool infeasible = false;
  for (std::size_t b = 0; b < mats->blocks.size(); ++b) {
    for (const model::MatId mat : {model::MatId::Theta, model::MatId::Psi}) {
      const Eigen::MatrixXd& raw = mat == model::MatId::Theta
          ? mats->blocks[b].Theta : mats->blocks[b].Psi;
      if (raw.size() == 0) continue;
      const Eigen::MatrixXd C = 0.5 * (raw + raw.transpose());
      const Eigen::Index dim = C.rows();
      const auto block = static_cast<std::int32_t>(b);
      const double tol = interior_eigen_tol *
          std::max(1.0, C.cwiseAbs().maxCoeff());
      // Structural pattern: free cells and fixed nonzero entries.
      Eigen::MatrixXi freecell = Eigen::MatrixXi::Zero(dim, dim);
      for (std::size_t i = 0; i < pt.size(); ++i) {
        if (pt.free[i] <= 0) continue;
        const model::Cell& cell = rep.cell_for_row[i];
        if (!cell.used || cell.mat != mat || cell.block != block) continue;
        freecell(cell.row, cell.col) = freecell(cell.col, cell.row) = 1;
      }
      std::vector<bool> active(static_cast<std::size_t>(dim), false);
      for (Eigen::Index r = 0; r < dim; ++r) {
        const bool zero_variance = freecell(r, r) == 0 && C(r, r) == 0.0;
        if (!zero_variance) {
          active[static_cast<std::size_t>(r)] = true;
          continue;
        }
        // A fixed zero variance forces its row to zero: hold free cells there.
        for (Eigen::Index c = 0; c < dim; ++c) {
          if (c == r || freecell(r, c) == 0) continue;
          Eigen::MatrixXd N = Eigen::MatrixXd::Zero(dim, dim);
          N(r, c) = N(c, r) = 0.5;
          forced.push_back(block_adjoint(N, mat, block, pt, rep, npar));
        }
      }
      // Connected components over structurally nonzero off-diagonals.
      std::vector<Eigen::Index> parent(static_cast<std::size_t>(dim));
      std::iota(parent.begin(), parent.end(), Eigen::Index{0});
      auto find = [&parent](Eigen::Index x) {
        while (parent[static_cast<std::size_t>(x)] != x)
          x = parent[static_cast<std::size_t>(x)] =
              parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
        return x;
      };
      for (Eigen::Index r = 0; r < dim; ++r)
        for (Eigen::Index c = 0; c < r; ++c)
          if (active[static_cast<std::size_t>(r)] &&
              active[static_cast<std::size_t>(c)] &&
              (freecell(r, c) != 0 || C(r, c) != 0.0))
            parent[static_cast<std::size_t>(find(r))] = find(c);
      std::vector<std::vector<Eigen::Index>> groups(static_cast<std::size_t>(dim));
      for (Eigen::Index r = 0; r < dim; ++r)
        if (active[static_cast<std::size_t>(r)])
          groups[static_cast<std::size_t>(find(r))].push_back(r);
      for (auto& idx : groups) {
        if (idx.empty()) continue;
        const auto m = static_cast<Eigen::Index>(idx.size());
        Eigen::MatrixXd Cs(m, m);
        for (Eigen::Index a = 0; a < m; ++a)
          for (Eigen::Index c = 0; c < m; ++c)
            Cs(a, c) = C(idx[static_cast<std::size_t>(a)], idx[static_cast<std::size_t>(c)]);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Cs);
        if (es.info() != Eigen::Success) return fail;
        const Eigen::VectorXd& lam = es.eigenvalues();
        if (lam.minCoeff() < -tol) infeasible = true;
        std::vector<Eigen::Index> nul, rng;
        for (Eigen::Index j = 0; j < m; ++j) (lam(j) <= tol ? nul : rng).push_back(j);
        if (nul.empty()) continue;
        FaceComponent fc;
        fc.mat = mat;
        fc.block = block;
        fc.dim = dim;
        fc.idx = idx;
        fc.U.resize(m, static_cast<Eigen::Index>(nul.size()));
        for (std::size_t j = 0; j < nul.size(); ++j)
          fc.U.col(static_cast<Eigen::Index>(j)) = es.eigenvectors().col(nul[j]);
        fc.P = Eigen::MatrixXd::Zero(m, m);
        for (const Eigen::Index j : rng)
          fc.P += es.eigenvectors().col(j) * es.eigenvectors().col(j).transpose() / lam(j);
        faces.push_back(std::move(fc));
      }
    }
  }
  if (infeasible) return fail;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(npar, npar);
  std::vector<Eigen::VectorXd> held = forced;
  std::int32_t null_dirs = 0, held_dirs = 0;
  double min_multiplier = std::numeric_limits<double>::quiet_NaN();
  if (!faces.empty()) {
    // Multipliers: least squares of the reduced gradient on all face rows.
    std::vector<Eigen::VectorXd> rows = forced;
    std::vector<std::size_t> start;
    for (const auto& fc : faces) {
      start.push_back(rows.size());
      append_face_rows(rows, fc, fc.U, pt, rep, npar);
    }
    const Eigen::MatrixXd A = stack_rows(rows, npar) * K;
    const Eigen::VectorXd lambda =
        A.transpose().completeOrthogonalDecomposition().solve(G);
    std::vector<Eigen::VectorXd> omegas;
    double scale = 0.0;
    for (std::size_t c = 0; c < faces.size(); ++c) {
      auto& fc = faces[c];
      const Eigen::Index q = fc.U.cols();
      Eigen::MatrixXd L = Eigen::MatrixXd::Zero(q, q);
      std::size_t r = start[c];
      for (Eigen::Index j = 0; j < q; ++j)
        for (Eigen::Index i = 0; i <= j; ++i)
          L += lambda(static_cast<Eigen::Index>(r++)) * sym_basis(q, i, j);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L);
      if (es.info() != Eigen::Success) return fail;
      omegas.push_back(es.eigenvalues());
      scale = std::max(scale, es.eigenvalues().cwiseAbs().maxCoeff());
      fc.Uc = fc.U * es.eigenvectors();  // rotate into multiplier eigenbasis
    }
    const double omega_tol = 1e-10 * std::max(scale, 1e-300);
    for (std::size_t c = 0; c < faces.size(); ++c) {
      auto& fc = faces[c];
      const Eigen::VectorXd& w = omegas[c];
      null_dirs += static_cast<std::int32_t>(w.size());
      min_multiplier = std::isnan(min_multiplier) ? w.minCoeff()
                                                  : std::min(min_multiplier, w.minCoeff());
      std::vector<Eigen::Index> keep;
      for (Eigen::Index j = 0; j < w.size(); ++j)
        if (w(j) > omega_tol) keep.push_back(j);
      Eigen::MatrixXd Uc(fc.Uc.rows(), static_cast<Eigen::Index>(keep.size()));
      Eigen::VectorXd wc(static_cast<Eigen::Index>(keep.size()));
      for (std::size_t j = 0; j < keep.size(); ++j) {
        Uc.col(static_cast<Eigen::Index>(j)) = fc.Uc.col(keep[j]);
        wc(static_cast<Eigen::Index>(j)) = w(keep[j]);
      }
      fc.Uc = Uc;
      fc.omega_c = wc;
      held_dirs += static_cast<std::int32_t>(keep.size());
      if (keep.empty()) continue;
      append_face_rows(held, fc, fc.Uc, pt, rep, npar);
      // Curvature of the rank-constrained set: 2 tr(M D_k P D_l).
      const Eigen::MatrixXd M = fc.Uc * fc.omega_c.asDiagonal() * fc.Uc.transpose();
      const auto m = static_cast<Eigen::Index>(fc.idx.size());
      std::vector<std::pair<Eigen::Index, Eigen::MatrixXd>> D;
      for (std::size_t i = 0; i < pt.size(); ++i) {
        if (pt.free[i] <= 0) continue;
        const model::Cell& cell = rep.cell_for_row[i];
        if (!cell.used || cell.mat != fc.mat || cell.block != fc.block) continue;
        const auto a = std::find(fc.idx.begin(), fc.idx.end(), Eigen::Index{cell.row});
        const auto bb = std::find(fc.idx.begin(), fc.idx.end(), Eigen::Index{cell.col});
        if (a == fc.idx.end() || bb == fc.idx.end()) continue;
        const Eigen::Index ra = a - fc.idx.begin(), cb = bb - fc.idx.begin();
        Eigen::MatrixXd Dk = Eigen::MatrixXd::Zero(m, m);
        Dk(ra, cb) = 1.0;
        Dk(cb, ra) = 1.0;
        D.emplace_back(pt.free[i] - 1, Dk);
      }
      for (const auto& [k, Dk] : D) {
        const Eigen::MatrixXd T = M * Dk * fc.P;
        for (const auto& [l, Dl] : D) Q(k, l) += 2.0 * (T * Dl).trace();
      }
    }
  }

  const Eigen::MatrixXd H = I + K.transpose() * Q * K;
  Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(K.cols(), K.cols());
  if (!held.empty()) {
    const Eigen::MatrixXd A = stack_rows(held, npar) * K;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::VectorXd& sv = svd.singularValues();
    const double cut = sv.size() ? 1e-10 * std::max(sv(0), 1e-300) : 0.0;
    Eigen::Index rank = 0;
    while (rank < sv.size() && sv(rank) > cut) ++rank;
    Z = svd.matrixV().rightCols(K.cols() - rank);
  }
  out.reduced_gradient = Z.transpose() * G;
  out.reduced_hessian = Z.transpose() * H * Z;
  if (sandwich) {
    out.reduced_metric = Z.transpose() * (K.transpose() * derivatives.metric * K) * Z;
  }
  out.tangent_basis = std::move(Z);
  out.curvature_correction = std::move(Q);
  out.status = NewtonAccuracyStatus::Available;
  out.null_directions = null_dirs;
  out.constrained_directions = held_dirs;
  out.min_multiplier = min_multiplier;
  out.covariance_interior = null_dirs == 0;
  return out;
}

}  // namespace magmaan::estimate::frontier
