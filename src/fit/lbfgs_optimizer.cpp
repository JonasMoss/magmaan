#include "latva/fit/lbfgs_optimizer.hpp"

#include <cmath>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <LBFGS.h>

#include "latva/error.hpp"
#include "latva/expected.hpp"

namespace latva::fit {

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
  // or invalid arguments. Under -fno-exceptions, "throw" becomes
  // std::terminate. To stay value-returning we set max_linesearch high and
  // pre-validate inputs; line-search failures still abort. This is the
  // "real" risk surface from O3 in the plan; we'll wrap a retry/widened-
  // step fallback when we hit an actual case.
  LBFGSpp::LBFGSSolver<double> solver(param);
  const int n_iter = solver.minimize(fn, x, fval);

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

}  // namespace latva::fit
