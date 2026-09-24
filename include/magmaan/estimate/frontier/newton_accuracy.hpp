#pragma once

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"  // Estimates
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/spec/partable.hpp"

// Local accuracy diagnostic for a returned complete-data ML fit. The
// complete-data ML fit paths attach it to `FitDiagnostics::newton_accuracy`
// and the common verdict uses it; this entry point recomputes it for any
// fitted parameter vector, with other options if wanted.
//
// With G the total score (N times the gradient of the per-observation
// half-discrepancy) and I the total observed information, both reduced by the
// linear-equality basis K, the Newton distance is
//
//   d = sqrt(G' I^{-1} G).
//
// Under the local quadratic approximation d is the largest predicted Newton
// correction of any linear contrast, measured in its information-based
// standard errors, and d^2 / 2 is the predicted remaining improvement of the
// total negative log likelihood. It is a local accuracy approximation, not a
// bound on the distance to an optimum, and it says nothing about global
// optimality. The accepted budget is d <= .01 (docs/research/interior-newton-
// audit.md): one hundredth of a standard error in every linear contrast.
//
// The diagnostic applies at regular interior points of the fitting domain.
// Ordinary fits with improper estimates are eligible, since they are interior
// to the ambient domain. At a PSD-boundary solution of a PSD fit the Newton
// step is infeasible and d is not an accuracy statement. `covariance_interior`
// reports whether every primitive Psi and Theta block is positive definite, so
// callers can route boundary fits to the cone stationarity check instead.
//
// The conditioning and solve guards are numerical safeguards, not
// identification tests. This function never errors: failures are statuses.

namespace magmaan::estimate::frontier {

// The result types live in estimate/diagnostics.hpp, since the common fit
// verdict consumes them. These aliases keep the frontier spelling.
using estimate::NewtonAccuracyDiagnostics;
using estimate::NewtonAccuracyOptions;
using estimate::NewtonAccuracyStatus;
using estimate::to_string;

// Core assessment from a total score and total information that are already
// in the same (reduced) coordinates. Exposed for tests and research code.
NewtonAccuracyDiagnostics
newton_accuracy_from(const Eigen::VectorXd& total_score,
                     const Eigen::MatrixXd& total_information,
                     NewtonAccuracyOptions opts = {});

// Full-model diagnostic for complete-data normal-theory ML at `est.theta`.
NewtonAccuracyDiagnostics
newton_accuracy_ml(const spec::LatentStructure& pt,
                   const model::MatrixRep& rep,
                   const SampleStats& samp,
                   const Estimates& est,
                   NewtonAccuracyOptions opts = {});

}  // namespace magmaan::estimate::frontier
