#pragma once

#include <optional>

#include "magmaan/estimate/frontier/convergence_policy.hpp"
#include "magmaan/estimate/frontier/newton_adapters.hpp"

namespace magmaan::estimate::frontier {

struct ConvergenceRequest {
  bool newton = false;
  StationarityDomain domain = StationarityDomain::Ambient;
  Bounds bounds; // explicit; empty means unbounded, never inferred
  DiagnosticsOptions diagnostics;
  GeometricStationarityOptions geometry;
  NewtonAccuracyOptions curvature;
  NewtonDifferenceOptions differences;
};

// Owns the point, total gradient, original Hessian (when requested), geometry,
// factorization and solution. No objective closures or evaluator references
// escape collection. Feasibility/geometry preparation settings remain attached.
struct ConvergenceReport {
  ConvergenceRequest request;
  NewtonAudit computations;
  FitDiagnostics evidence;
};

// Original full-coordinate scalar objective; normalization is identical to
// evaluate_newton_objective. reported_value, if supplied, is in native units.
fit_expected<ConvergenceReport> audit_convergence(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    const optim::ScalarProblem& problem, const Eigen::VectorXd& theta,
    double n_obs, double native_to_total, ConvergenceRequest request = {},
    std::optional<double> reported_value = std::nullopt);

// Analytic ML curvature when requested, with fixed.x resolved from the sample.
fit_expected<ConvergenceReport> audit_convergence_ml(
    spec::LatentStructure pt, const model::MatrixRep& rep, const SampleStats& sample,
    const Eigen::VectorXd& theta, ConvergenceRequest request = {},
    std::optional<double> reported_value = std::nullopt);

// Compose any retained estimator adapter with common feasibility/first-order
// evidence, without reevaluating its objective or Hessian. Supply the exact
// prepared model/parameter order used by the adapter (including fixed.x and
// ordinal parameterization). Domain and bounds come from the retained audit;
// reported_value uses its recorded total/N objective units.
fit_expected<ConvergenceReport> audit_convergence(
    const spec::LatentStructure& pt, const model::MatrixRep& rep,
    NewtonAudit audit, DiagnosticsOptions diagnostics = {},
    GeometricStationarityOptions geometry = {},
    std::optional<double> reported_value = std::nullopt);

// Pure reassessment: reuses the retained Newton solution, including when a
// previous condition/solve threshold rejected it. Does not refit or factorize.
ConvergenceAssessment assess_convergence(
    const ConvergenceReport& report, ConvergencePolicy policy = {});

} // namespace magmaan::estimate::frontier
