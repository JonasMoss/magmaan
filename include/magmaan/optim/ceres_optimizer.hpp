#pragma once

#ifndef MAGMAAN_WITH_CERES
// Stub-only header when Ceres is disabled — including this without the build
// flag is a usage error, but we don't `#error` because the test suite may
// conditionally include this header and rely on the `#ifdef` guard around the
// declarations rather than the include itself.
namespace magmaan::optim {
struct CeresOptimizer;
struct CeresBoundedOptimizer;
}  // namespace magmaan::optim
#else

#include <functional>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/optim/problem.hpp"           // LsResidualFn / LsJacobianFn
#include "magmaan/optim/lbfgs_optimizer.hpp"   // shared LbfgsOutput return type

namespace magmaan::optim {

// Tuning knobs for both Ceres backends. The defaults match Ceres' own
// `Solver::Options` defaults, with a tighter `max_iter` than LBFGS++'s 1000
// because trust-region usually converges in far fewer iterations.
struct CeresOptions {
  int    max_iter = 500;
  double ftol     = 1e-10;   // function tolerance
  double gtol     = 1e-7;    // gradient tolerance (max-norm)
  double ptol     = 1e-8;    // parameter tolerance (relative step)
  bool   verbose  = false;   // route Ceres stdout to FitError detail
};

// CeresOptimizer — uses Ceres' `GradientProblemSolver` (general-purpose
// unconstrained nonlinear minimization with value + gradient). Drop-in for
// `LbfgsOptimizer`: same `minimize(Objective, x0)` signature, same return
// shape. No bounds — use `CeresBoundedOptimizer` when bounds are needed.
class CeresOptimizer {
public:
  static constexpr std::string_view name = "ceres-gradient";

  CeresOptimizer(CeresOptions opts = {}) noexcept : opts_(opts) {}

  CeresOptions options() const noexcept { return opts_; }

  using Objective = std::function<double(const Eigen::VectorXd& /*x*/,
                                         Eigen::VectorXd&       /*grad_out*/)>;

  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0) const;

private:
  CeresOptions opts_;
};

// CeresBoundedOptimizer — uses Ceres' `Problem` API with per-parameter
// `SetParameterLowerBound`/`SetParameterUpperBound`. The unbounded overload
// delegates to the GradientProblemSolver path internally, so the same
// optimizer satisfies both the `Optimizer` and `BoundedOptimizer` concepts —
// callers pick at compile time via `fit<>` (unbounded) vs `fit_bounded<>`
// (bounded).
//
// Bounded path: two cost-function shapes are supported.
//
//   1) **LS-shape (`minimize_ls`)** — preferred when the discrepancy is
//      sum-of-squares (ULS / future GLS / WLS). The caller supplies `r(θ)`
//      and `J_r(θ)`; we wrap them in a true multi-residual
//      `ceres::CostFunction`, so `JᵀJ` is the natural Gauss–Newton normal
//      matrix and Levenberg–Marquardt converges cleanly. Returns
//      `fmin = ½‖r‖²` (Ceres' `final_cost` is already ½‖r‖²).
//
//   2) **Scalar fallback (`minimize(f, x0, lo, hi)`)** — for any
//      value+gradient discrepancy (ML); wraps F as a single residual
//      `r₀ = √(2F + ε)` (ε ≈ 1e-30 regularizer so the chain-rule denominator
//      stays finite at the optimum). The gradient maps via
//      `∂r₀/∂θ = (1/r₀)·∇F`. This works for any backend but masks the LS
//      structure when the discrepancy actually has one — known to stall on
//      ULS, which is exactly why the LS path above exists.
class CeresBoundedOptimizer {
public:
  static constexpr std::string_view name = "ceres-bounded";

  CeresBoundedOptimizer(CeresOptions opts = {}) noexcept : opts_(opts) {}

  CeresOptions options() const noexcept { return opts_; }

  using Objective = std::function<double(const Eigen::VectorXd& /*x*/,
                                         Eigen::VectorXd&       /*grad_out*/)>;

  // Unbounded overload — required to satisfy `Optimizer<>`.
  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0) const;

  // Bounded overload — required to satisfy `BoundedOptimizer<>`.
  // `lower` / `upper` must have the same size as `x0`; per-coordinate
  // ±std::numeric_limits<double>::infinity() means "no bound on this axis".
  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0,
           const Eigen::VectorXd& lower,
           const Eigen::VectorXd& upper) const;

  // LS-shape bounded overload — required to satisfy `LsBoundedOptimizer<>`.
  // `n_resid` is the size of `r_fn`'s output (and `J_fn`'s row count); the
  // cost function is sized once at construction so Ceres can validate shapes.
  // `lower` / `upper` are box bounds on θ (same sentinel as above).
  fit_expected<LbfgsOutput>
  minimize_ls(LsResidualFn r_fn, LsJacobianFn J_fn,
              Eigen::Index n_resid,
              const Eigen::VectorXd& x0,
              const Eigen::VectorXd& lower,
              const Eigen::VectorXd& upper) const;

private:
  CeresOptions opts_;
};

}  // namespace magmaan::optim

#endif  // MAGMAAN_WITH_CERES
