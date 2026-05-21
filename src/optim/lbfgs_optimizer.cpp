#include "magmaan/optim/lbfgs_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <LBFGS.h>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::optim {

namespace {

FitError make_err(FitError::Kind k, std::string detail,
                  int iter = 0, double fval = 0.0) {
  return FitError{k, std::move(detail), iter, fval};
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
  // (see the catch block) rather than discarding a good solution. See
  // docs/validation/convergence_diagnostics.md.
  LBFGSpp::LBFGSSolver<double> solver(param);
  int n_iter = 0;
  try {
    n_iter = solver.minimize(fn, x, fval);
  } catch (const std::exception& e) {
    // Line search failed. `x` is the last accepted iterate; if its gradient
    // is small the search has effectively converged -- accept it. A point
    // with a genuinely large gradient is still surfaced as a failure.
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
    const double f_at_x = fn(x, grad);
    const double gmax = x.size() > 0 ? grad.cwiseAbs().maxCoeff() : 0.0;
    const double salvage_gtol = std::max(1e-3, 1e3 * opts_.gtol);
    if (std::isfinite(f_at_x) && gmax <= salvage_gtol) {
      return LbfgsOutput{std::move(x), f_at_x, n_iter, 0, 0,
                         OptimStatus::LineSearchSalvaged, gmax};
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
    return std::unexpected(make_err(FitError::Kind::OptimizerNonConvergence,
        "max_iter reached without convergence",
        n_iter, fval));
  }

  // LBFGSpp does not return the final gradient; recompute it once at the
  // solution so callers get a stationarity measure (one extra evaluation).
  Eigen::VectorXd gfinal = Eigen::VectorXd::Zero(x.size());
  fn(x, gfinal);
  const double gnorm = x.size() > 0 ? gfinal.cwiseAbs().maxCoeff() : 0.0;
  return LbfgsOutput{std::move(x), fval, n_iter, 0, 0,
                     OptimStatus::Converged, gnorm};
}

}  // namespace magmaan::optim
