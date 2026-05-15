#include "magmaan/optim/lbfgsb_optimizer.hpp"

#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <LBFGSB.h>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::optim {

namespace {

FitError make_err(FitError::Kind k, std::string detail,
                  int iter = 0, double fval = 0.0) {
  return FitError{k, std::move(detail), iter, fval};
}

// Adapter from our Objective to LBFGSpp's functor signature. Returning
// +inf from the user objective is the contract for "this point is
// invalid (e.g. non-PD Σ); try a shorter step" — same as in
// LbfgsOptimizer.
struct Functor {
  LbfgsBOptimizer::Objective f;

  double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const {
    return f(x, grad);
  }
};

}  // namespace

fit_expected<LbfgsOutput>
LbfgsBOptimizer::minimize(Objective f,
                          const Eigen::VectorXd& x0,
                          const Eigen::VectorXd& lower,
                          const Eigen::VectorXd& upper) const {
  // Size-check at the boundary — friendlier than letting LBFGSpp throw.
  if (lower.size() != x0.size() || upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "LbfgsBOptimizer: bounds size mismatch (lower=" +
            std::to_string(lower.size()) +
            ", upper=" + std::to_string(upper.size()) +
            ", x0=" + std::to_string(x0.size()) + ")",
        0, 0.0));
  }

  LBFGSpp::LBFGSBParam<double> param;
  param.epsilon        = opts_.gtol;
  param.epsilon_rel    = 0;          // absolute projected-gradient tolerance only
  param.max_iterations = opts_.max_iter;
  param.m              = opts_.history;
  // See header: with bounds, ‖Pg‖∞ plateaus on the active set; let the
  // delta-based stop on function-value stagnation also fire. `ftol`
  // (LbfgsOptions.ftol == 1e-10 by default) maps onto `delta`.
  param.past           = 1;
  param.delta          = opts_.ftol;
  // Mirror LbfgsOptimizer: 100 trials cushions CFA endgames where the
  // line search needs more probes near convergence.
  param.max_linesearch = 100;

  // LBFGSpp does not auto-project x0 into [lb, ub]; an out-of-box start
  // would feed a meaningless projected gradient into the convergence
  // test. Project defensively; callers from fit_bounded already supply
  // feasible starts via simple_start_values, so this is a no-op there.
  Eigen::VectorXd x = x0.cwiseMax(lower).cwiseMin(upper);
  double fval = 0.0;
  Functor fn{std::move(f)};

  // LBFGSpp::LBFGSBSolver throws (std::runtime_error on line-search
  // failure, std::invalid_argument on bad params). This TU is compiled
  // with -fexceptions; the throw site is inside LBFGSpp, outside any
  // objective-callback frame, so unwinding never crosses a
  // -fno-exceptions boundary. We catch and convert to a FitError value.
  LBFGSpp::LBFGSBSolver<double> solver(param);
  int n_iter = 0;
  try {
    n_iter = solver.minimize(fn, x, fval, lower, upper);
  } catch (const std::exception& e) {
    return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
        std::string("L-BFGS-B line search failed: ") + e.what() +
        " (consider tighter starts or a different optimizer backend)",
        0, fval));
  }

  if (!std::isfinite(fval)) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "final objective is non-finite (model likely under-identified "
        "or starting point too far from a valid PD region)",
        n_iter, fval));
  }
  if (n_iter >= opts_.max_iter) {
    return std::unexpected(make_err(FitError::Kind::OptimizerNonConvergence,
        "max_iter reached without convergence",
        n_iter, fval));
  }

  return LbfgsOutput{std::move(x), fval, n_iter};
}

fit_expected<LbfgsOutput>
LbfgsBOptimizer::minimize(Objective f, const Eigen::VectorXd& x0) const {
  const double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd lower = Eigen::VectorXd::Constant(x0.size(), -inf);
  Eigen::VectorXd upper = Eigen::VectorXd::Constant(x0.size(),  inf);
  return minimize(std::move(f), x0, lower, upper);
}

fit_expected<LbfgsOutput>
LbfgsBOptimizer::minimize_ls(LsResidualFn r_fn, LsJacobianFn J_fn,
                             Eigen::Index n_resid,
                             const Eigen::VectorXd& x0,
                             const Eigen::VectorXd& lower,
                             const Eigen::VectorXd& upper) const {
  // Scalar adapter: f(x) = ½||r(x)||², g(x) = J(x)ᵀ · r(x). The cached `r`
  // is reused inside `g` only when the same x is queried twice in a row —
  // simplest correctness contract is "recompute on every evaluation".
  // LBFGSpp's value+grad pattern computes both in the same call anyway.
  Objective scalar = [r_fn = std::move(r_fn), J_fn = std::move(J_fn), n_resid]
      (const Eigen::VectorXd& x, Eigen::VectorXd& grad) -> double {
    auto r = r_fn(x);
    if (!r.has_value() || r->size() != n_resid || !r->allFinite()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    auto J = J_fn(x);
    if (!J.has_value() || J->rows() != n_resid ||
        J->cols() != x.size() || !J->allFinite()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    grad.noalias() = J->transpose() * (*r);
    return 0.5 * r->squaredNorm();
  };
  return minimize(std::move(scalar), x0, lower, upper);
}

}  // namespace magmaan::optim
