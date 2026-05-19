#pragma once

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/optim/ceres_optimizer.hpp"   // CeresOptions — stub-safe without MAGMAAN_WITH_CERES
#include "magmaan/optim/lbfgs_optimizer.hpp"   // LbfgsOptions / LbfgsOutput
#include "magmaan/optim/problem.hpp"

// Optimizers as free functions, called directly — no `fit()` wrapper, no
// template, no concept. Backends:
//
//   • lbfgs        — L-BFGS-B scalar minimizer (replaces LbfgsOptimizer +
//                    LbfgsBOptimizer; an empty `Bounds` means unbounded).
//   • trust_region — CppNumericalSolvers Newton trust-region cross-check
//                    (unbounded scalar minimizer).
//   • ceres_lm     — Ceres Levenberg–Marquardt least-squares minimizer
//                    (only when MAGMAAN_WITH_CERES is set).
//   • ceres_bfgs   — Ceres line-search dense BFGS least-squares minimizer
//                    (unbounded only, only when MAGMAAN_WITH_CERES is set).
//   • nlopt_slsqp  — NLopt SLSQP scalar cross-check (only when
//                    MAGMAAN_WITH_NLOPT is set).
//
// To run a scalar optimizer on a least-squares problem, `scalarize` it first:
// `lbfgs(scalarize(gmm_problem), x0, bounds)`.

namespace magmaan::optim {

using estimate::Bounds;

// L-BFGS-B scalar minimization. An empty `bounds` (the default) is unbounded;
// otherwise `bounds.lower`/`upper` must each match `x0`'s size, with
// ±infinity per coordinate meaning "no bound on that axis".
fit_expected<OptimResult>
lbfgs(const ScalarProblem& prob, const Eigen::VectorXd& x0,
      const Bounds& bounds = {}, LbfgsOptions opts = {});

// ½‖r(x)‖² scalar adapter: gradient is J(x)ᵀ·r(x). Lets any scalar optimizer
// run on a least-squares problem; carries `n_param` and `expand` through.
ScalarProblem scalarize(const GmmProblem& prob);

// CppNumericalSolvers Newton trust-region minimization — an independent
// cross-check of the L-BFGS / SLSQP optima (lavaan's nlminb is itself a trust
// region). Unbounded only: the trust-region solver has no box-constraint
// support, so pass an unconstrained `ScalarProblem`.
fit_expected<OptimResult>
trust_region(const ScalarProblem& prob, const Eigen::VectorXd& x0,
             LbfgsOptions opts = {});

#ifdef MAGMAAN_WITH_CERES
// Ceres Levenberg–Marquardt least-squares minimization — feeds the optimizer
// the true multi-residual cost so JᵀJ is the natural Gauss–Newton normal
// matrix. An empty `bounds` (the default) is unbounded.
fit_expected<OptimResult>
ceres_lm(const GmmProblem& prob, const Eigen::VectorXd& x0,
         const Bounds& bounds = {}, CeresOptions opts = {});

// Ceres dense BFGS line-search minimization over the scalarized least-squares
// objective. This is unbounded: Ceres' line-search minimizer does not support
// box constraints. Intended for small research / benchmark comparisons against
// dense quasi-Newton implementations.
fit_expected<OptimResult>
ceres_bfgs(const GmmProblem& prob, const Eigen::VectorXd& x0,
           CeresOptions opts = {});
#endif

#ifdef MAGMAAN_WITH_NLOPT
// NLopt SLSQP scalar minimization — sequential quadratic programming with box
// bounds, a different algorithm class from line-search L-BFGS. Used to
// cross-check fitted optima; an empty `bounds` (the default) is unbounded.
fit_expected<OptimResult>
nlopt_slsqp(const ScalarProblem& prob, const Eigen::VectorXd& x0,
            const Bounds& bounds = {}, LbfgsOptions opts = {});
#endif

}  // namespace magmaan::optim
