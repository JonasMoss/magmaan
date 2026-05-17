#include "magmaan/optim/trust_region_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>
#include <tuple>
#include <utility>

#include <Eigen/Core>

#include <cppoptlib/function.h>
#include <cppoptlib/solver/trust_region_newton.h>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::optim {

namespace {

FitError make_err(FitError::Kind k, std::string detail,
                  int iter = 0, double fval = 0.0) {
  return FitError{k, std::move(detail), iter, fval};
}

// CppNumericalSolvers function adapter. Declared Second-order so the
// TrustRegionNewton solver can request a Hessian; magmaan's objective supplies
// value+gradient analytically, and the Hessian is built by central finite
// differences of that gradient.
class TrFunction : public cppoptlib::function::FunctionCRTP<
                       TrFunction, double,
                       cppoptlib::function::DifferentiabilityMode::Second> {
 public:
  TrFunction(TrustRegionOptimizer::Objective f, int dim)
      : f_(std::move(f)), dim_(dim) {}

  // Required for a dynamically-sized FunctionCRTP.
  int GetDimension() const { return dim_; }

  ScalarType operator()(const VectorType& x, VectorType* grad,
                        MatrixType* hess) const {
    Eigen::VectorXd g = Eigen::VectorXd::Zero(x.size());
    const double v = f_(x, g);
    if (grad) *grad = g;
    if (hess) *hess = fd_hessian(x);
    return v;
  }

 private:
  // Central finite-difference Hessian of the analytic gradient:
  //   H[:,j] ≈ (∇f(x + h·e_j) − ∇f(x − h·e_j)) / (2h),
  // then symmetrized. The per-coordinate step is scaled by |x_j|; the base
  // ≈ eps^(1/3) is the accuracy sweet spot for differencing a gradient.
  Eigen::MatrixXd fd_hessian(const Eigen::VectorXd& x) const {
    const Eigen::Index n = x.size();
    constexpr double   base = 1e-5;
    Eigen::MatrixXd    H(n, n);
    Eigen::VectorXd    gp = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd    gm = Eigen::VectorXd::Zero(n);
    for (Eigen::Index j = 0; j < n; ++j) {
      const double    h  = base * std::max(1.0, std::abs(x[j]));
      Eigen::VectorXd xp = x;  xp[j] += h;
      Eigen::VectorXd xm = x;  xm[j] -= h;
      f_(xp, gp);
      f_(xm, gm);
      H.col(j) = (gp - gm) / (2.0 * h);
    }
    return 0.5 * (H + H.transpose());   // symmetrize away FD asymmetry
  }

  TrustRegionOptimizer::Objective f_;
  int                             dim_;
};

}  // namespace

fit_expected<LbfgsOutput>
TrustRegionOptimizer::minimize(Objective f, const Eigen::VectorXd& x0) const {
  if (x0.size() == 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "TrustRegionOptimizer: empty parameter vector"));
  }

  TrFunction fn(std::move(f), static_cast<int>(x0.size()));
  // The solver consumes a type-erased `FunctionExpr`: its unified `operator()`
  // carries the gradient/Hessian-pointer defaults the solver calls through,
  // whereas a raw `FunctionCRTP` subclass exposes only the fixed-arity
  // mode-specific overload. `FunctionExpr` deep-copies `fn` via `clone()`.
  cppoptlib::function::FunctionExpr objective(fn);
  cppoptlib::solver::TrustRegionNewton<decltype(objective)> solver;

  Eigen::VectorXd theta;
  double          fmin  = 0.0;
  int             iters = 0;
  // CppNumericalSolvers is header-only and compiled into this -fexceptions TU;
  // catch any throw here so nothing unwinds into the -fno-exceptions callers
  // (mirrors the Ceres adapter).
  try {
    auto [solution, progress] =
        solver.Minimize(objective, cppoptlib::function::FunctionState(x0));
    theta = solution.x;
    fmin  = solution.value;
    iters = static_cast<int>(progress.num_iterations);
  } catch (const std::exception& e) {
    return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
        std::string("trust-region: solver threw: ") + e.what()));
  }

  if (theta.size() != x0.size() || !theta.allFinite() ||
      !std::isfinite(fmin)) {
    return std::unexpected(make_err(FitError::Kind::NonFiniteObjective,
        "trust-region: solver returned a non-finite point or value "
        "(model likely under-identified or the start too far from a valid "
        "region)", iters, fmin));
  }
  return LbfgsOutput{std::move(theta), fmin, iters};
}

}  // namespace magmaan::optim
