#include "magmaan/optim/optimizers.hpp"

#include <limits>
#include <utility>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#ifdef MAGMAAN_WITH_PORT
#include "magmaan/optim/port_optimizer.hpp"
#endif
#ifdef MAGMAAN_WITH_CERES
#include "magmaan/optim/ceres_optimizer.hpp"
#endif
#ifdef MAGMAAN_WITH_NLOPT
#include "magmaan/optim/nlopt_optimizer.hpp"
#endif

namespace magmaan::optim {

namespace {

OptimResult to_result(LbfgsOutput out) {
  return OptimResult{std::move(out.theta_hat), out.fmin, out.iterations,
                     out.f_evals, out.g_evals, out.status, out.grad_inf_norm};
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

#ifdef MAGMAAN_WITH_PORT
fit_expected<OptimResult>
port(const ScalarProblem& prob, const Eigen::VectorXd& x0,
     const Bounds& bounds, LbfgsOptions opts) {
  // PORT supports bounds natively (drmngb is the bounded variant of dmng).
  // Unbounded ⇒ pass ±infinity per coordinate; the adapter folds those into
  // PORT's ±1e308 sentinels internally.
  const double          inf = std::numeric_limits<double>::infinity();
  const Eigen::VectorXd lower =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(), -inf) : bounds.lower;
  const Eigen::VectorXd upper =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(),  inf) : bounds.upper;
  auto out = PortOptimizer{opts}.minimize(prob.f, x0, lower, upper);
  if (!out.has_value()) return std::unexpected(out.error());
  return to_result(std::move(*out));
}

fit_expected<OptimResult>
port_nls(const GmmProblem& prob, const Eigen::VectorXd& x0,
         const Bounds& bounds, LbfgsOptions opts) {
  // NL2SOL also supports bounds natively (drn2gb is the bounded variant of
  // drn2g). Same ±infinity sentinel handling as the scalar PORT entry.
  const double          inf = std::numeric_limits<double>::infinity();
  const Eigen::VectorXd lower =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(), -inf) : bounds.lower;
  const Eigen::VectorXd upper =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(),  inf) : bounds.upper;
  auto out = PortNlsOptimizer{opts}.minimize_ls(
      prob.r, prob.J, prob.eval, prob.n_resid, x0, lower, upper);
  if (!out.has_value()) return std::unexpected(out.error());
  return to_result(std::move(*out));
}
#endif

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
namespace {

// Shared body for the NLopt entry points: pack ±infinity bounds into NLopt's
// ±HUGE_VAL convention, construct a `NloptOptimizer` parameterised over
// the magmaan-local `NloptAlgorithm`, run it. Each named entry below picks a
// different algorithm; the rest of the call shape is identical.
fit_expected<OptimResult>
run_nlopt(const ScalarProblem& prob, const Eigen::VectorXd& x0,
          const Bounds& bounds, LbfgsOptions opts, NloptAlgorithm algo) {
  const double          inf = std::numeric_limits<double>::infinity();
  const Eigen::VectorXd lower =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(), -inf) : bounds.lower;
  const Eigen::VectorXd upper =
      bounds.empty() ? Eigen::VectorXd::Constant(x0.size(),  inf) : bounds.upper;
  auto out = NloptOptimizer{opts, algo}.minimize(prob.f, x0, lower, upper);
  if (!out.has_value()) return std::unexpected(out.error());
  return to_result(std::move(*out));
}

}  // namespace

fit_expected<OptimResult>
nlopt_slsqp(const ScalarProblem& prob, const Eigen::VectorXd& x0,
            const Bounds& bounds, LbfgsOptions opts) {
  return run_nlopt(prob, x0, bounds, opts, NloptAlgorithm::Slsqp);
}

fit_expected<OptimResult>
nlopt_bobyqa(const ScalarProblem& prob, const Eigen::VectorXd& x0,
             const Bounds& bounds, LbfgsOptions opts) {
  // BOBYQA needs *finite* bounds — there is no analogue of ±HUGE_VAL for a
  // derivative-free TR that builds a quadratic model from sampled points.
  // We reject `bounds.empty()` here with a clear error rather than letting
  // NLopt return NLOPT_INVALID_ARGS with no context.
  if (bounds.empty()) {
    return std::unexpected(magmaan::FitError{
        magmaan::FitError::Kind::NumericIssue,
        "nlopt BOBYQA requires finite bounds on every coordinate; supply a "
        "non-empty Bounds with concrete lower/upper, or use a different "
        "derivative-free backend (none currently wired) for unbounded fits",
        0, 0.0});
  }
  return run_nlopt(prob, x0, bounds, opts, NloptAlgorithm::Bobyqa);
}

fit_expected<OptimResult>
nlopt_tnewton(const ScalarProblem& prob, const Eigen::VectorXd& x0,
              const Bounds& bounds, LbfgsOptions opts) {
  return run_nlopt(prob, x0, bounds, opts, NloptAlgorithm::Tnewton);
}

fit_expected<OptimResult>
nlopt_var2(const ScalarProblem& prob, const Eigen::VectorXd& x0,
           const Bounds& bounds, LbfgsOptions opts) {
  return run_nlopt(prob, x0, bounds, opts, NloptAlgorithm::Var2);
}

fit_expected<OptimResult>
nlopt_lbfgs(const ScalarProblem& prob, const Eigen::VectorXd& x0,
            const Bounds& bounds, LbfgsOptions opts) {
  return run_nlopt(prob, x0, bounds, opts, NloptAlgorithm::Lbfgs);
}
#endif

}  // namespace magmaan::optim
