#include "magmaan/optim/nlopt_optimizer.hpp"

#ifdef MAGMAAN_WITH_NLOPT

#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <nlopt.h>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

// NLopt's objective callback type `nlopt_func` is declared inside the
// `extern "C"` block of <nlopt.h>, so the trampoline we hand to
// `nlopt_set_min_objective` must itself have C language linkage. Defining it
// `extern "C"` (and `static`, for internal linkage) makes its function-pointer
// type match `nlopt_func` exactly — no linkage-mismatch diagnostic under
// -Werror. It is defined at file scope, before the namespace, so it can be
// named without qualification troubles; it reaches the wrapped objective
// through NLopt's `void*` user-data slot.
extern "C" {
static double magmaan_nlopt_objective(unsigned n, const double* x,
                                      double* grad, void* data) {
  // `data` is the magmaan Objective handed in via nlopt_set_min_objective.
  const auto& f =
      *static_cast<const magmaan::optim::NloptOptimizer::Objective*>(data);
  const Eigen::Map<const Eigen::VectorXd> x_map(x, n);
  const Eigen::VectorXd xv = x_map;             // copy for the const& callback
  Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
  // The wrapped objective follows magmaan's contract: +inf and a zeroed
  // gradient for an invalid x (e.g. non-PD Σ). SLSQP treats the +inf as a
  // barrier and retreats, so no special handling is needed here.
  const double v = f(xv, g);
  if (grad) Eigen::Map<Eigen::VectorXd>(grad, n) = g;
  return v;
}
}  // extern "C"

namespace magmaan::optim {

namespace {

FitError make_err(FitError::Kind k, std::string detail,
                  int iter = 0, double fval = 0.0) {
  return FitError{k, std::move(detail), iter, fval};
}

}  // namespace

fit_expected<LbfgsOutput>
NloptOptimizer::minimize(Objective f,
                         const Eigen::VectorXd& x0,
                         const Eigen::VectorXd& lower,
                         const Eigen::VectorXd& upper) const {
  if (lower.size() != x0.size() || upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "NloptOptimizer: bounds size mismatch (lower=" +
            std::to_string(lower.size()) +
            ", upper=" + std::to_string(upper.size()) +
            ", x0=" + std::to_string(x0.size()) + ")"));
  }

  const unsigned n = static_cast<unsigned>(x0.size());
  nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, n);
  if (!opt) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "NloptOptimizer: nlopt_create failed for n=" + std::to_string(n)));
  }

  // The objective `f` outlives this whole call; NLopt invokes the trampoline
  // synchronously from nlopt_optimize, so passing &f through the void* slot is
  // safe.
  nlopt_set_min_objective(opt, &magmaan_nlopt_objective, &f);

  // NLopt reads ±HUGE_VAL as "no bound", which is exactly magmaan's ±infinity
  // sentinel — pass the bound vectors straight through (NLopt copies them).
  nlopt_set_lower_bounds(opt, lower.data());
  nlopt_set_upper_bounds(opt, upper.data());

  nlopt_set_ftol_rel(opt, opts_.ftol);
  nlopt_set_xtol_rel(opt, opts_.gtol);
  nlopt_set_maxeval(opt, opts_.max_iter);

  // SLSQP requires a feasible start; project x0 into the box defensively
  // (callers from fit_* already supply feasible starts).
  Eigen::VectorXd theta = x0.cwiseMax(lower).cwiseMin(upper);
  double fmin = std::numeric_limits<double>::quiet_NaN();
  const nlopt_result rc = nlopt_optimize(opt, theta.data(), &fmin);
  const int n_evals = nlopt_get_numevals(opt);
  nlopt_destroy(opt);

  switch (rc) {
    case NLOPT_SUCCESS:
    case NLOPT_STOPVAL_REACHED:
    case NLOPT_FTOL_REACHED:
    case NLOPT_XTOL_REACHED:
    case NLOPT_ROUNDOFF_LIMITED:   // result is still the best found — usable
      break;
    case NLOPT_MAXEVAL_REACHED:
    case NLOPT_MAXTIME_REACHED:
      return std::unexpected(make_err(FitError::Kind::OptimizerNonConvergence,
          "nlopt SLSQP: evaluation budget exhausted without convergence",
          n_evals, fmin));
    case NLOPT_FORCED_STOP:
      return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
          "nlopt SLSQP: forced stop", n_evals, fmin));
    case NLOPT_OUT_OF_MEMORY:
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "nlopt SLSQP: out of memory"));
    case NLOPT_INVALID_ARGS:
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "nlopt SLSQP: invalid arguments"));
    case NLOPT_FAILURE:
    default:
      return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
          "nlopt SLSQP: generic solver failure", n_evals, fmin));
  }

  if (!std::isfinite(fmin)) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "nlopt SLSQP: final objective is non-finite (model likely "
        "under-identified or the start too far from a valid region)",
        n_evals, fmin));
  }
  return LbfgsOutput{std::move(theta), fmin, n_evals};
}

fit_expected<LbfgsOutput>
NloptOptimizer::minimize(Objective f, const Eigen::VectorXd& x0) const {
  const double inf = std::numeric_limits<double>::infinity();
  return minimize(std::move(f), x0,
                  Eigen::VectorXd::Constant(x0.size(), -inf),
                  Eigen::VectorXd::Constant(x0.size(),  inf));
}

}  // namespace magmaan::optim

#endif  // MAGMAAN_WITH_NLOPT
