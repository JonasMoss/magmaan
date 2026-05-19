#include "magmaan/optim/optimizers.hpp"

#include <limits>
#include <utility>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/optim/trust_region_optimizer.hpp"
#ifdef MAGMAAN_WITH_CERES
#include "magmaan/optim/ceres_optimizer.hpp"
#endif
#ifdef MAGMAAN_WITH_NLOPT
#include "magmaan/optim/nlopt_optimizer.hpp"
#endif

namespace magmaan::optim {

namespace {

OptimResult to_result(LbfgsOutput out) {
  return OptimResult{std::move(out.theta_hat), out.fmin, out.iterations};
}

}  // namespace

fit_expected<OptimResult>
lbfgs(const ScalarProblem& prob, const Eigen::VectorXd& x0,
      const Bounds& bounds, LbfgsOptions opts) {
  // Unbounded ⇒ plain LBFGSSolver; bounded ⇒ LBFGSBSolver. The two have
  // slightly different convergence criteria (the unbounded path relies on
  // the gradient norm only), so dispatching by `bounds.empty()` keeps each
  // path bit-for-bit what it was before the optimizer classes collapsed.
  fit_expected<LbfgsOutput> out =
      bounds.empty()
          ? LbfgsOptimizer{opts}.minimize(prob.f, x0)
          : LbfgsBOptimizer{opts}.minimize(prob.f, x0, bounds.lower,
                                           bounds.upper);
  if (!out.has_value()) return std::unexpected(out.error());
  return to_result(std::move(*out));
}

ScalarProblem scalarize(const GmmProblem& prob) {
  ScalarProblem s;
  s.n_param = prob.n_param;
  s.expand  = prob.expand;
  // f(x) = ½‖r(x)‖², ∇f(x) = J(x)ᵀ·r(x). A non-finite or wrong-shape
  // residual/Jacobian returns +inf — the "x is invalid, shorten the step"
  // contract every optimizer backend already honors.
  s.f = [r_fn = prob.r, J_fn = prob.J, eval_fn = prob.eval,
         n_resid = prob.n_resid](
            const Eigen::VectorXd& x, Eigen::VectorXd& grad) -> double {
    if (eval_fn) {
      auto e = eval_fn(x);
      if (!e.has_value() || e->residual.size() != n_resid ||
          e->jacobian.rows() != n_resid || e->jacobian.cols() != x.size() ||
          !e->residual.allFinite() || !e->jacobian.allFinite()) {
        grad.setZero();
        return std::numeric_limits<double>::infinity();
      }
      grad.noalias() = e->jacobian.transpose() * e->residual;
      return 0.5 * e->residual.squaredNorm();
    }
    auto r = r_fn(x);
    if (!r.has_value() || r->size() != n_resid || !r->allFinite()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    auto J = J_fn(x);
    if (!J.has_value() || J->rows() != n_resid || J->cols() != x.size() ||
        !J->allFinite()) {
      grad.setZero();
      return std::numeric_limits<double>::infinity();
    }
    grad.noalias() = J->transpose() * (*r);
    return 0.5 * r->squaredNorm();
  };
  return s;
}

fit_expected<OptimResult>
trust_region(const ScalarProblem& prob, const Eigen::VectorXd& x0,
             LbfgsOptions opts) {
  auto out = TrustRegionOptimizer{opts}.minimize(prob.f, x0);
  if (!out.has_value()) return std::unexpected(out.error());
  return to_result(std::move(*out));
}

#ifdef MAGMAAN_WITH_CERES
fit_expected<OptimResult>
ceres_lm(const GmmProblem& prob, const Eigen::VectorXd& x0,
         const Bounds& bounds, CeresOptions opts) {
  const CeresBoundedOptimizer solver{opts};
  const double inf = std::numeric_limits<double>::infinity();
  const Eigen::VectorXd lower =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(), -inf) : bounds.lower;
  const Eigen::VectorXd upper =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(), inf) : bounds.upper;
  auto out =
      solver.minimize_ls(prob.r, prob.J, prob.eval, prob.n_resid, x0, lower,
                         upper);
  if (!out.has_value()) return std::unexpected(out.error());
  return to_result(std::move(*out));
}

fit_expected<OptimResult>
ceres_bfgs(const GmmProblem& prob, const Eigen::VectorXd& x0,
           CeresOptions opts) {
  auto out = CeresBfgsOptimizer{opts}.minimize(scalarize(prob).f, x0);
  if (!out.has_value()) return std::unexpected(out.error());
  return to_result(std::move(*out));
}
#endif

#ifdef MAGMAAN_WITH_NLOPT
fit_expected<OptimResult>
nlopt_slsqp(const ScalarProblem& prob, const Eigen::VectorXd& x0,
            const Bounds& bounds, LbfgsOptions opts) {
  const double          inf = std::numeric_limits<double>::infinity();
  const Eigen::VectorXd lower =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(), -inf) : bounds.lower;
  const Eigen::VectorXd upper =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(), inf) : bounds.upper;
  auto out = NloptOptimizer{opts}.minimize(prob.f, x0, lower, upper);
  if (!out.has_value()) return std::unexpected(out.error());
  return to_result(std::move(*out));
}
#endif

}  // namespace magmaan::optim
