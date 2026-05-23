#include "magmaan/estimate/auglag.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "magmaan/error.hpp"
#include "magmaan/optim/optimizers.hpp"

namespace magmaan::estimate {

fit_expected<AugLagResult>
augmented_lagrangian(const optim::ScalarProblem& base, const ConstraintFn& h,
                     const ConstraintJacFn& jac, std::int32_t m,
                     const Eigen::VectorXd& x0, const Bounds& bounds,
                     OptimOptions opts, AugLagOptions al) {
  Eigen::VectorXd x      = x0;
  Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m);
  double rho             = al.rho0;
  double prev_viol       = std::numeric_limits<double>::infinity();
  int    outer           = 0;

  for (; outer < al.max_outer; ++outer) {
    // Inner objective L_ρ(x) = F(x) + λᵀh(x) + (ρ/2)‖h(x)‖². λ and ρ are
    // snapshot by value, so this outer iteration's penalty is fixed across
    // the inner solve; `base` / `h` / `jac` outlive every inner call.
    optim::ScalarProblem inner;
    inner.n_param = base.n_param;
    inner.expand  = base.expand;
    inner.f = [&base, &h, &jac, lambda, rho](const Eigen::VectorXd& xv,
                                             Eigen::VectorXd& grad) -> double {
      Eigen::VectorXd gF(xv.size());
      const double F = base.f(xv, gF);
      if (!std::isfinite(F)) { grad = gF; return F; }   // x invalid — propagate
      const Eigen::VectorXd hv = h(xv);
      const Eigen::MatrixXd H  = jac(xv);
      grad = gF + H.transpose() * (lambda + rho * hv);
      return F + lambda.dot(hv) + 0.5 * rho * hv.squaredNorm();
    };

    auto res = optim::nlopt_lbfgs(inner, x, bounds, opts);
    if (!res.has_value()) return std::unexpected(res.error());
    x = std::move(res->x);

    const Eigen::VectorXd hv = h(x);
    const double viol = (hv.size() > 0) ? hv.cwiseAbs().maxCoeff() : 0.0;
    if (viol <= al.feas_tol) { ++outer; break; }
    lambda += rho * hv;                          // Hestenes multiplier update
    if (viol > al.shrink * prev_viol) {          // violation stalled — penalize harder
      rho = std::min(rho * al.rho_grow, al.rho_max);
    }
    prev_viol = viol;
  }

  const Eigen::VectorXd hv = h(x);
  const double viol = (hv.size() > 0) ? hv.cwiseAbs().maxCoeff() : 0.0;
  if (viol > al.accept_tol) {
    return std::unexpected(FitError{FitError::Kind::NumericIssue,
        "nonlinear equality constraints did not converge — violation " +
            std::to_string(viol) + " (the constraint system may be infeasible)",
        0, 0.0});
  }
  Eigen::VectorXd scratch(x.size());
  const double F = base.f(x, scratch);
  return AugLagResult{std::move(x), F, outer, viol};
}

}  // namespace magmaan::estimate
