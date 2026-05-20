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
//   • port         — PORT drmngb model-Hessian trust region with bounds
//                    (the algorithm behind R's `nlminb`; TOMS 611).
//                    Only when MAGMAAN_WITH_PORT is set (default ON).
//   • ceres_lm     — Ceres Levenberg–Marquardt least-squares minimizer
//                    (only when MAGMAAN_WITH_CERES is set).
//   • ceres_bfgs   — Ceres line-search dense BFGS least-squares minimizer
//                    (unbounded only, only when MAGMAAN_WITH_CERES is set).
//   • nlopt_slsqp  — NLopt SLSQP scalar cross-check (only when
//                    MAGMAAN_WITH_NLOPT is set).
//   • nlopt_bobyqa — NLopt BOBYQA (Powell derivative-free TR). Requires
//                    finite bounds (only when MAGMAAN_WITH_NLOPT).
//   • nlopt_tnewton— NLopt LD_TNEWTON_PRECOND_RESTART (Nash truncated Newton).
//                    Only when MAGMAAN_WITH_NLOPT.
//   • nlopt_var2   — NLopt LD_VAR2 (Shanno-Phua full BFGS variable-metric).
//                    Only when MAGMAAN_WITH_NLOPT.
//   • nlopt_lbfgs  — NLopt's own LBFGS — sanity-check against LBFGS++.
//                    Only when MAGMAAN_WITH_NLOPT.
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

#ifdef MAGMAAN_WITH_PORT
// PORT `drmngb` scalar minimization with simple bounds — Dennis-Gay-Welsch
// model-Hessian trust region (TOMS 611), the same algorithm R's `nlminb` runs.
// Empty `bounds` (the default) means unbounded; otherwise `bounds.lower` /
// `bounds.upper` must each match `x0`'s size, with ±infinity per coordinate
// translated by the adapter into PORT's ±1e308 sentinels.
fit_expected<OptimResult>
port(const ScalarProblem& prob, const Eigen::VectorXd& x0,
     const Bounds& bounds = {}, LbfgsOptions opts = {});

// PORT `drn2gb` LS-shape minimization with simple bounds — NL2SOL adaptive
// trust region (TOMS 573 Dennis-Gay-Welsch), the same algorithm R's `nls`
// runs. Drives the multi-residual `GmmProblem` directly (not via
// `scalarize`), so NL2SOL sees the true residual structure and can build
// its Gauss-Newton-plus-secant model Hessian. Empty `bounds` ⇒ unbounded.
fit_expected<OptimResult>
port_nls(const GmmProblem& prob, const Eigen::VectorXd& x0,
         const Bounds& bounds = {}, LbfgsOptions opts = {});
#endif

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
// bounds (Kraft 1988), a different algorithm class from line-search L-BFGS.
// Used to cross-check fitted optima; an empty `bounds` (the default) is
// unbounded.
fit_expected<OptimResult>
nlopt_slsqp(const ScalarProblem& prob, const Eigen::VectorXd& x0,
            const Bounds& bounds = {}, LbfgsOptions opts = {});

// NLopt BOBYQA — Powell 2009 derivative-free quadratic-model trust region.
// Requires *finite* bounds (no ±infinity sentinels). Useful when the gradient
// is unreliable near a Heywood boundary or saddle.
fit_expected<OptimResult>
nlopt_bobyqa(const ScalarProblem& prob, const Eigen::VectorXd& x0,
             const Bounds& bounds, LbfgsOptions opts = {});

// NLopt LD_TNEWTON_PRECOND_RESTART — Nash 1985 preconditioned truncated Newton
// with restart. Explicit second-order via Hessian-vector products + CG inner
// solve; a fundamentally different curvature scheme from L-BFGS.
fit_expected<OptimResult>
nlopt_tnewton(const ScalarProblem& prob, const Eigen::VectorXd& x0,
              const Bounds& bounds = {}, LbfgsOptions opts = {});

// NLopt LD_VAR2 — Shanno-Phua 1980 full (dense) BFGS variable-metric. The
// non-limited-memory counterpart to L-BFGS; at SEM-sized n (10–100) the full
// Hessian update may outperform L-BFGS's history truncation.
fit_expected<OptimResult>
nlopt_var2(const ScalarProblem& prob, const Eigen::VectorXd& x0,
           const Bounds& bounds = {}, LbfgsOptions opts = {});

// NLopt LD_LBFGS — NLopt's own L-BFGS. Sanity-check against LBFGS++ (the
// default `Lbfgs` backend); the two should agree on most problems but
// differ in line-search and step-control implementation details.
fit_expected<OptimResult>
nlopt_lbfgs(const ScalarProblem& prob, const Eigen::VectorXd& x0,
            const Bounds& bounds = {}, LbfgsOptions opts = {});
#endif

}  // namespace magmaan::optim
