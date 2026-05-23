#include "magmaan/optim/lbfgs_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <LBFGS.h>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/optim/terminal_audit.hpp"

namespace magmaan::optim {

namespace {

FitError make_err(FitError::Kind k, std::string detail,
                  int iter = 0, double fval = 0.0) {
  return FitError{k, std::move(detail), iter, fval};
}

// ±infinity bound vectors for the unbounded path. The audit honours these as
// "never active" sentinels, so its projected-gradient construction reduces to
// the raw ‖g‖_∞ — exactly what the old hand-rolled success path computed.
Eigen::VectorXd unbounded(Eigen::Index n) {
  return Eigen::VectorXd::Constant(
      n, std::numeric_limits<double>::infinity());
}
Eigen::VectorXd unbounded_neg(Eigen::Index n) {
  return Eigen::VectorXd::Constant(
      n, -std::numeric_limits<double>::infinity());
}

// Adapter that wraps the user objective to the LBFGS++ functor signature.
// LBFGS's line search legitimately probes non-PD points (returning +inf
// from our objective is the contract for "this point is invalid; try a
// shorter step"). So we don't treat intermediate non-finite values as
// fatal — only the *final* converged objective is checked for finiteness
// after solver returns.
struct Functor {
  LbfgsOptimizer::Objective f;

  double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const {
    return f(x, grad);
  }
};

}  // namespace

fit_expected<LbfgsOutput>
LbfgsOptimizer::minimize(Objective f, const Eigen::VectorXd& x0) const {
  LBFGSpp::LBFGSParam<double> param;
  param.epsilon        = opts_.gtol;
  param.epsilon_rel    = 0;          // absolute gradient tolerance only
  param.max_iterations = opts_.max_iter;
  param.m              = opts_.history;
  // Bumped from default (20). Larger CFA models occasionally need more
  // line-search attempts near convergence; LBFGSpp throws on exhaustion
  // and that aborts under -fno-exceptions.
  param.max_linesearch = 100;
  // (LBFGSpp's `delta` / `past` mechanism is disabled by default —
  // gradient norm is the sole stopping criterion. Suits us: easy to
  // reason about, doesn't terminate early on flat sections.)

  Functor fn{std::move(f)};
  Eigen::VectorXd x = x0;
  double fval = 0.0;

  // LBFGSpp::LBFGSSolver throws std::runtime_error on line-search failure
  // (and std::invalid_argument on bad params). This TU is compiled with
  // -fexceptions precisely so we can catch it here and convert it to a
  // value; the throw site is deep in LBFGSpp's line-search loop, *outside*
  // any objective-callback frame, so unwinding stays within this
  // -fexceptions TU and never crosses a -fno-exceptions frame.
  //
  // The strong-Wolfe line search throws most often *at* the optimum: in the
  // flat neighbourhood of the minimum the objective no longer changes
  // measurably along any step before ‖g‖ falls under the absolute `gtol`.
  // When that happens `x` already holds a converged iterate, so we salvage it
  // (see the catch block) rather than discarding a good solution.
  LBFGSpp::LBFGSSolver<double> solver(param);
  int n_iter = 0;

  // Bind a thin ObjectiveFn for the audit (LBFGSpp's `fn` is a Functor; the
  // audit takes the std::function type). Cheap — captures by reference.
  const ObjectiveFn audit_fn =
      [&fn](const Eigen::VectorXd& xi, Eigen::VectorXd& g) { return fn(xi, g); };
  const Eigen::VectorXd lower = unbounded_neg(x.size());
  const Eigen::VectorXd upper = unbounded(x.size());

  try {
    n_iter = solver.minimize(fn, x, fval);
  } catch (const std::exception& e) {
    // Line search failed. `x` is the last accepted iterate; the audit decides
    // by GEOMETRY (relative projected-gradient stationarity) whether this is
    // a salvageable arrival at the optimum (the noise-floor case that the
    // ex5_4-style near-perfect-fit GLS objectives produce, where the
    // strong-Wolfe test cannot satisfy sufficient decrease because there is
    // no measurable decrease left) or a genuine failure mid-descent.
    TerminalAudit a = audit_terminal_iterate(audit_fn, x, fval, lower, upper);
    if (a.stationary) {
      LbfgsOutput out{std::move(x), a.f_recomputed, n_iter, 0, 0,
                      OptimStatus::LineSearchSalvaged, a.grad_inf_norm};
      out.audit = std::move(a);
      return out;
    }
    return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
        std::string("L-BFGS line search failed: ") + e.what() +
        " (model may need a box-constrained optimizer or better starts)",
        n_iter, fval));
  }

  if (!std::isfinite(fval)) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "final objective is non-finite (model likely under-identified or "
        "starting point too far from a valid PD region)",
        n_iter, fval));
  }
  if (n_iter >= opts_.max_iter) {
    // Budget exhausted. Apply the same geometric test the line-search throw
    // gets: a max-iter stop that nonetheless lands on a stationary point is
    // procedurally non-converged but geometrically converged, and the audit
    // distinguishes the two.
    TerminalAudit a = audit_terminal_iterate(audit_fn, x, fval, lower, upper);
    if (a.stationary) {
      LbfgsOutput out{std::move(x), a.f_recomputed, n_iter, 0, 0,
                      OptimStatus::LineSearchSalvaged, a.grad_inf_norm};
      out.audit = std::move(a);
      return out;
    }
    return std::unexpected(make_err(FitError::Kind::OptimizerNonConvergence,
        "max_iter reached without convergence",
        n_iter, fval));
  }

  // Success path: the audit replaces the hand-rolled gradient recompute (its
  // projected-gradient norm collapses to ‖g‖_∞ here because the bounds are
  // ±infinity). One extra f-evaluation, same as before; we additionally get
  // `f_consistent` and `f_recomputed` populated for downstream diagnostics.
  TerminalAudit a = audit_terminal_iterate(audit_fn, x, fval, lower, upper);
  LbfgsOutput out{std::move(x), fval, n_iter, 0, 0,
                  OptimStatus::Converged, a.grad_inf_norm};
  out.audit = std::move(a);
  return out;
}

}  // namespace magmaan::optim
