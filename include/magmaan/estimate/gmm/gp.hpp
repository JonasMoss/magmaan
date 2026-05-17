#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/spec/partable.hpp"

// Golub–Pereyra profiling (separable nonlinear least squares). A
// moment-quadratic problem whose model has conditionally-linear parameters
// (residual variances Θ, latent (co)variances Ψ, intercepts ν, latent means α)
// can have those eliminated analytically, leaving an outer problem in the
// nonlinear parameters (loadings Λ, structural coefficients Β) alone.
//
// `gp` transforms a θ-space GmmProblem into the profiled β-space one — a
// closure transform, parallel to `estimate::reparameterize`.

namespace magmaan::estimate::gmm {

// True iff the model + start point can be profiled: it has conditionally-
// linear free parameters and no equality constraint mixes a nonlinear with a
// linear parameter.
bool gp_compatible(const spec::LatentStructure& pt,
                   const model::ModelEvaluator& ev,
                   const Eigen::VectorXd& theta0);

// The profiled problem plus the β start point. `problem` drives only the
// nonlinear β; `problem.expand` recovers full θ (β plus the profiled α̂(β)).
struct GpProblem {
  optim::GmmProblem problem;
  Eigen::VectorXd   beta0;
};

// Profile the linear parameters out of `base` (the θ-space GmmProblem from
// `gmm::residuals`). Fails (NumericIssue) when the model is not separable —
// see `gp_compatible`.
fit_expected<GpProblem>
gp(const optim::GmmProblem& base, const spec::LatentStructure& pt,
   const model::ModelEvaluator& ev, const Eigen::VectorXd& theta0);

}  // namespace magmaan::estimate::gmm
