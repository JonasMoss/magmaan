// Unit changes of the same one-variance likelihood, through production adapters.
#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>
#include "magmaan/optim/optimizers.hpp"

int main() {
  using namespace magmaan;
  std::puts("variance,normalized,bounded,arm,returned,status,evaluations,invalid_evaluations,estimate_ratio,f,relative_error,stationary,gradient");
  for (double s : {1.0, 1e-2, 1e-6}) {
    for (bool normalized : {false, true}) for (bool bounded : {false, true}) {
      for (std::string_view arm : {"lbfgs", "lbfgs_budget", "lbfgs_tight", "port", "slsqp", "lbfgs_slsqp"}) {
        int evaluations = 0, invalid = 0;
        const double scale = normalized ? s : 1.0;
        optim::ScalarProblem problem;
        problem.n_param = 1;
        problem.f = [&](const Eigen::VectorXd& x, Eigen::VectorXd& g) {
          ++evaluations;
          const double variance = x(0) * scale;
          if (!(variance > 0.0)) {
            ++invalid;
            g.setZero();
            return std::numeric_limits<double>::infinity();
          }
          const double ratio = variance / s;
          g(0) = scale * (1.0 / variance - s / (variance * variance));
          return std::log(ratio) + 1.0 / ratio - 1.0;
        };
        Eigen::VectorXd x0(1); x0(0) = 10.0 * s / scale;
        estimate::Bounds bounds;
        if (bounded) {
          bounds.lower = Eigen::VectorXd::Zero(1);
          bounds.upper = Eigen::VectorXd::Constant(1, std::numeric_limits<double>::infinity());
        }
        optim::OptimOptions options;
        options.max_iter = 5000;
        if (arm == "lbfgs_budget") options.nlopt.max_eval = 20000;
        if (arm == "lbfgs_tight") {
          options.nlopt.ftol_rel = 1e-14;
          options.nlopt.xtol_rel = 1e-12;
          options.nlopt.tolg = 1e-10;
        }
        auto result = arm == "port" ? optim::port(problem, x0, bounds, options)
            : arm == "slsqp" ? optim::nlopt_slsqp(problem, x0, bounds, options)
            : arm == "lbfgs_slsqp" ? optim::nlopt_lbfgs_slsqp_fallback(problem, x0, bounds, options)
            : optim::nlopt_lbfgs(problem, x0, bounds, options);
        const double ratio = result ? result->x(0) * scale / s : std::numeric_limits<double>::quiet_NaN();
        std::printf("%.12g,%d,%d,%.*s,%d,%d,%d,%d,%.17g,%.17g,%.17g,%d,%.17g\n",
            s, normalized, bounded, static_cast<int>(arm.size()), arm.data(),
            result.has_value(), result ? static_cast<int>(result->status) : -1,
            evaluations, invalid, ratio, result ? result->fmin : std::numeric_limits<double>::quiet_NaN(), std::abs(ratio - 1.0), result && result->audit.stationary,
            result ? result->audit.grad_inf_norm : std::numeric_limits<double>::quiet_NaN());
      }
    }
  }
}
