#include "magmaan/optim/lbfgs_optimizer.hpp"

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
  // -fexceptions TU and never crosses a -fno-exceptions frame. Known
  // trigger: single-group 3F CFA + meanstructure on the Holzinger sample
  // (lavaan's nlminb converges; LBFGS++ exhausts the line search). The
  // capable / box-constrained alternative now exists — `LbfgsBOptimizer`
  // via `fit_bounded(...)` (LBFGSpp's `LBFGSBSolver`) handles this case;
  // see `tests/unit/lbfgsb_integration_test.cpp` for the demonstration.
  LBFGSpp::LBFGSSolver<double> solver(param);
  int n_iter = 0;
  try {
    n_iter = solver.minimize(fn, x, fval);
  } catch (const std::exception& e) {
    return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
        std::string("L-BFGS line search failed: ") + e.what() +
        " (model may need a box-constrained optimizer or better starts)",
        0, fval));
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

  return LbfgsOutput{std::move(x), fval, n_iter};
}

}  // namespace magmaan::optim
