#pragma once

#include <functional>
#include <string_view>

#include <Eigen/Core>

#include "latva/expected.hpp"

namespace latva::fit {

struct LbfgsOptions {
  int    max_iter = 1000;
  double ftol     = 1e-10;   // matches lavaan's optim.ftol default
  double gtol     = 1e-7;    // matches lavaan's optim.gradtol default
  int    history  = 10;      // L-BFGS history (m parameter)
};

struct LbfgsOutput {
  Eigen::VectorXd theta_hat;
  double          fmin = 0.0;
  int             iterations = 0;
};

// LBFGS++ adapter. The objective callable computes value + writes
// gradient into the second argument. The optimizer minimizes from x0.
//
// Returns FitError::OptimizerNonConvergence if the iteration limit is
// hit, NonFiniteObjective if the objective ever returns NaN/Inf,
// LineSearchFailed if LBFGSpp's line search fails (we catch that and
// surface it as a value).
class LbfgsOptimizer {
 public:
  static constexpr std::string_view name = "lbfgs";

  LbfgsOptimizer(LbfgsOptions opts = {}) noexcept : opts_(opts) {}

  using Objective = std::function<double(const Eigen::VectorXd& /*x*/,
                                         Eigen::VectorXd&       /*grad_out*/)>;

  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0) const;

 private:
  LbfgsOptions opts_;
};

}  // namespace latva::fit
