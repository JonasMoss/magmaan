#pragma once

// Concepts that constrain the two extension points on `fit<D, O>`:
//
//   • Discrepancy<D>       — F(θ) and ∇_θ F(θ); see ML / ULS.
//   • Optimizer<O>         — minimize(f, x0); see LbfgsOptimizer, CeresOptimizer.
//   • BoundedOptimizer<O>  — minimize(f, x0, lb, ub); see CeresBoundedOptimizer.
//
// `BoundedOptimizer` refines `Optimizer` (a bounded optimizer must also work
// without bounds). `fit<D, O>` uses `Optimizer`; the parallel `fit_bounded<D, O>`
// uses `BoundedOptimizer`.
//
// Two LS-shape refinements for discrepancies whose F has a sum-of-squares form:
//
//   • LsDiscrepancy<D>          — also exposes r(θ), J_r(θ) with F = ½||r||².
//   • LsBoundedOptimizer<O>     — also accepts (r, J_r, n_resid, x0, lb, ub).
//
// When both refinements are satisfied at `fit_bounded<D, O>`, the call dispatches
// through `minimize_ls(...)`, giving Ceres' Levenberg–Marquardt the full Jacobian
// (rather than the rank-1 sqrt-trick wrap of a scalar F). For non-LS or non-LS
// optimizer combinations the call falls back to the scalar `minimize(...)` path.

#include <concepts>
#include <functional>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/fit/lbfgs_optimizer.hpp"      // LbfgsOutput is the shared optimizer return type
#include "magmaan/fit/sample_stats.hpp"
#include "magmaan/model/model_evaluator.hpp"

namespace magmaan::fit {

// ============================================================================
// Discrepancy
// ============================================================================
//
// A Discrepancy reduces (sample moments, implied moments) to a non-negative
// scalar F(θ) and its θ-gradient via the model Jacobians supplied by
// ModelEvaluator. The shape contract for J and Jmu matches what fit() already
// hands ML in `include/magmaan/fit/fit.hpp`:
//   J   = ∂vech(Σ)/∂θ   — (Σ_b vech_len(p_b)) × n_free.
//   Jmu = ∂μ/∂θ         — (Σ_b p_b) × n_free; empty for covariance-only models
//                         (the discrepancy detects via Jmu.size() == 0).
//
// Multi-group / mean-structure block weighting is internal to the discrepancy;
// it sees a flat (SampleStats, ImpliedMoments) pair.
template <class D>
concept Discrepancy = requires(const D&                     d,
                               const SampleStats&           s,
                               const model::ImpliedMoments& m,
                               const Eigen::MatrixXd&       J,
                               const Eigen::MatrixXd&       Jmu) {
  // F(θ): scalar; may fail (e.g. non-PD Σ → FitError::NonPositiveDefiniteSigma).
  { d.value(s, m) } -> std::same_as<fit_expected<double>>;

  // ∇_θ F(θ): VectorXd of length n_free. `Jmu` is mentioned explicitly so the
  // four-argument form fit.hpp calls is type-checked, even though ML supplies
  // a default value for it.
  { d.gradient(s, m, J, Jmu) } -> std::same_as<fit_expected<Eigen::VectorXd>>;
};

// ============================================================================
// Optimizer / BoundedOptimizer
// ============================================================================
//
// The objective shape `double(const VectorXd& x, VectorXd& grad)` is the same
// one `fit()` constructs via lambda from a `Discrepancy::value`/`gradient` pair.
// Returns reuse `LbfgsOutput` (theta_hat / fmin / iterations) — all optimizer
// backends share this universal output struct rather than each defining its
// own, keeping the `Optimizer` concept's return type single-valued.
template <class O>
concept Optimizer = requires(
    const O&                                                       o,
    std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)> f,
    const Eigen::VectorXd&                                          x0) {
  { o.minimize(f, x0) } -> std::same_as<fit_expected<LbfgsOutput>>;
};

// A BoundedOptimizer additionally accepts per-parameter lower / upper bounds
// (`±std::numeric_limits<double>::infinity()` is the unbounded sentinel).
// Implementations that don't natively support bounds (e.g. plain L-BFGS) are
// just `Optimizer`s, not `BoundedOptimizer`s — they fail substitution at the
// `fit_bounded<...>` call site, which is the desired compile-time signal.
template <class O>
concept BoundedOptimizer = Optimizer<O> && requires(
    const O&                                                       o,
    std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)> f,
    const Eigen::VectorXd&                                          x0,
    const Eigen::VectorXd&                                          lb,
    const Eigen::VectorXd&                                          ub) {
  { o.minimize(f, x0, lb, ub) } -> std::same_as<fit_expected<LbfgsOutput>>;
};

// ============================================================================
// LsDiscrepancy / LsBoundedOptimizer — sum-of-squares refinements
// ============================================================================
//
// A `LsDiscrepancy<D>` is a `Discrepancy<D>` whose F has the structure
//
//     F(θ) = ½·||r(s, m(θ))||²
//
// with `r` a stacked vector of per-block, per-vech / per-mean residuals, and
// `J_r(θ) = ∂r/∂θ`. The contract requires:
//   • `r = d.residuals(s, m)`           — `fit_expected<VectorXd>`, length n_r.
//   • `Jr = d.residual_jacobian(s, m, J_sigma, J_mu)`  — `fit_expected<MatrixXd>`,
//                                          (n_r × n_free); J_sigma and J_mu have
//                                          the same shapes as in `Discrepancy`.
//
// Block weighting (n_b/N) is baked into `r` as `√(n_b/N)`, so the existing
// scalar `F = ½||r||²` and `∇_θ F = J_rᵀ r` identities hold without any extra
// weighting matrix on the optimizer side. Sign convention: `r = model − sample`
// so `Jr = +√(n_b/N) · [J_μ; J_σ]` (no flip relative to the model-implied
// Jacobian), matching the existing `gradient` formula.
template <class D>
concept LsDiscrepancy = Discrepancy<D> && requires(
    const D&                     d,
    const SampleStats&           s,
    const model::ImpliedMoments& m,
    const Eigen::MatrixXd&       J_sigma,
    const Eigen::MatrixXd&       J_mu) {
  { d.residuals(s, m) }
      -> std::same_as<fit_expected<Eigen::VectorXd>>;
  { d.residual_jacobian(s, m, J_sigma, J_mu) }
      -> std::same_as<fit_expected<Eigen::MatrixXd>>;
};

// `LsBoundedOptimizer<O>` is a `BoundedOptimizer<O>` that additionally accepts
// the LS-shape callback pair `(r_fn, J_fn)` plus the total residual length
// (so the cost function can be sized at construction). Optimizers that do
// not natively exploit LS structure (e.g. L-BFGS-B) satisfy this concept via
// a thin scalar adapter — same θ̂ as the scalar `minimize` path, with the
// LS form kept for caller-side consistency.
template <class O>
concept LsBoundedOptimizer = BoundedOptimizer<O> && requires(
    const O& o,
    std::function<fit_expected<Eigen::VectorXd>(const Eigen::VectorXd&)> r_fn,
    std::function<fit_expected<Eigen::MatrixXd>(const Eigen::VectorXd&)> J_fn,
    Eigen::Index                                                          n_resid,
    const Eigen::VectorXd&                                                x0,
    const Eigen::VectorXd&                                                lb,
    const Eigen::VectorXd&                                                ub) {
  { o.minimize_ls(r_fn, J_fn, n_resid, x0, lb, ub) }
      -> std::same_as<fit_expected<LbfgsOutput>>;
};

// Shared callback aliases — used by every `minimize_ls` overload so the lambda
// types in `fit_bounded` collapse to the same `std::function` instantiation.
using LsResidualFn = std::function<
    fit_expected<Eigen::VectorXd>(const Eigen::VectorXd& /*x*/)>;
using LsJacobianFn = std::function<
    fit_expected<Eigen::MatrixXd>(const Eigen::VectorXd& /*x*/)>;

}  // namespace magmaan::fit
