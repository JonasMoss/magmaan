#include "magmaan/fit/ceres_optimizer.hpp"

#ifdef MAGMAAN_WITH_CERES

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>

#include <ceres/ceres.h>

#include <Eigen/Core>

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"

namespace magmaan::fit {

namespace {

FitError make_err(FitError::Kind k, std::string detail) {
  return FitError{k, std::move(detail), 0, 0.0};
}

// ε for the √(2F+ε) bounded-path trick: small enough to be numerically
// invisible relative to F on every realistic SEM problem, large enough that
// 1/√(2F+ε) doesn't overflow at F=0.
constexpr double kSqrtEps = 1e-30;

// Map Ceres' termination types onto our FitError taxonomy. Both
// GradientProblemSolver::Summary and Solver::Summary expose the same
// termination_type / message fields.
template <class Summary>
FitError summary_to_err(const Summary& s, const char* who) {
  std::string msg = std::string("ceres (") + who + "): " + s.message;
  if (s.termination_type == ceres::NO_CONVERGENCE) {
    return make_err(FitError::Kind::OptimizerNonConvergence, std::move(msg));
  }
  if (s.termination_type == ceres::FAILURE ||
      s.termination_type == ceres::USER_FAILURE) {
    // The most common Ceres FAILURE is a step-length collapse, which is the
    // moral equivalent of LBFGS++'s line-search failure.
    return make_err(FitError::Kind::LineSearchFailed, std::move(msg));
  }
  return make_err(FitError::Kind::NumericIssue, std::move(msg));
}

// FirstOrderFunction adapter for GradientProblemSolver. Wraps a value+gradient
// `Objective` (std::function<double(const VectorXd&, VectorXd&)>) so Ceres can
// call it with raw double*. We pay one VectorXd copy per Evaluate to satisfy
// the Objective signature (const VectorXd& not Ref) — acceptable; the
// per-iteration cost is dominated by the gradient evaluation.
class StdObjectiveFirstOrder : public ceres::FirstOrderFunction {
public:
  using Objective = CeresOptimizer::Objective;

  StdObjectiveFirstOrder(Objective f, int n)
      : f_(std::move(f)), n_(n) {}

  bool Evaluate(const double* parameters,
                double*       cost,
                double*       gradient) const override {
    Eigen::Map<const Eigen::VectorXd> x_map(parameters, n_);
    const Eigen::VectorXd             x = x_map;     // copy for const& bind
    Eigen::VectorXd                   grad_buf = Eigen::VectorXd::Zero(n_);
    const double v = f_(x, grad_buf);
    if (!std::isfinite(v)) return false;
    *cost = v;
    if (gradient) {
      Eigen::Map<Eigen::VectorXd> g_out(gradient, n_);
      g_out = grad_buf;
    }
    return true;
  }

  int NumParameters() const override { return n_; }

private:
  Objective f_;
  int       n_;
};

// CostFunction adapter for the per-discrepancy LS path. Holds the residual
// vector / Jacobian callbacks supplied by `fit_bounded` (which in turn pull
// them from `LsDiscrepancy::residuals` / `residual_jacobian`). `JᵀJ` is the
// natural Gauss–Newton normal matrix here — no rank deficiency — so LM
// converges cleanly on ULS/GLS/WLS where the scalar sqrt-trick stalls.
class LsCostFunction : public ceres::CostFunction {
public:
  LsCostFunction(LsResidualFn r_fn, LsJacobianFn J_fn,
                 int n_resid, int n_param)
      : r_fn_(std::move(r_fn)),
        J_fn_(std::move(J_fn)),
        n_resid_(n_resid),
        n_param_(n_param) {
    set_num_residuals(n_resid);
    mutable_parameter_block_sizes()->push_back(n_param);
  }

  bool Evaluate(const double* const* parameters,
                double*              residuals,
                double**             jacobians) const override {
    Eigen::Map<const Eigen::VectorXd> x_map(parameters[0], n_param_);
    const Eigen::VectorXd             x = x_map;       // copy for const& bind
    auto r = r_fn_(x);
    if (!r.has_value()) return false;
    if (r->size() != n_resid_) return false;
    if (!r->allFinite()) return false;
    Eigen::Map<Eigen::VectorXd>(residuals, n_resid_) = *r;

    if (jacobians && jacobians[0]) {
      auto J = J_fn_(x);
      if (!J.has_value()) return false;
      if (J->rows() != n_resid_ || J->cols() != n_param_) return false;
      if (!J->allFinite()) return false;
      // Ceres expects jacobians[0] in row-major order, n_resid × n_param.
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          J_out(jacobians[0], n_resid_, n_param_);
      J_out = *J;
    }
    return true;
  }

private:
  LsResidualFn r_fn_;
  LsJacobianFn J_fn_;
  int          n_resid_;
  int          n_param_;
};

// CostFunction adapter for the single-residual √(2F+ε) trick — flows a scalar
// objective through Ceres' bounds-aware Problem API. r₀ = √(2F+ε) means
// ½||r||² = F + ε/2 ≈ F, and ∂r₀/∂θ = (1/r₀)·∇F (chain rule). Kept as the
// scalar fallback for non-LS discrepancies (ML); see `LsCostFunction` above
// for the preferred path on ULS / GLS / WLS.
class StdObjectiveSqrtCost : public ceres::CostFunction {
public:
  using Objective = CeresBoundedOptimizer::Objective;

  StdObjectiveSqrtCost(Objective f, int n)
      : f_(std::move(f)), n_(n) {
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(n);
  }

  bool Evaluate(const double* const* parameters,
                double*              residuals,
                double**             jacobians) const override {
    Eigen::Map<const Eigen::VectorXd> x_map(parameters[0], n_);
    const Eigen::VectorXd             x = x_map;
    Eigen::VectorXd                   grad_buf = Eigen::VectorXd::Zero(n_);
    const double v = f_(x, grad_buf);
    if (!std::isfinite(v) || v < 0.0) return false;
    const double r = std::sqrt(2.0 * v + kSqrtEps);
    residuals[0] = r;
    if (jacobians && jacobians[0]) {
      const double inv_r = 1.0 / r;
      Eigen::Map<Eigen::VectorXd> J_out(jacobians[0], n_);
      J_out = inv_r * grad_buf;
    }
    return true;
  }

private:
  Objective f_;
  int       n_;
};

void apply_options(ceres::GradientProblemSolver::Options& o,
                   const CeresOptions& opts) {
  o.max_num_iterations           = opts.max_iter;
  o.function_tolerance           = opts.ftol;
  o.gradient_tolerance           = opts.gtol;
  o.parameter_tolerance          = opts.ptol;
  o.minimizer_progress_to_stdout = opts.verbose;
}

void apply_options(ceres::Solver::Options& o, const CeresOptions& opts) {
  o.max_num_iterations           = opts.max_iter;
  o.function_tolerance           = opts.ftol;
  o.gradient_tolerance           = opts.gtol;
  o.parameter_tolerance          = opts.ptol;
  o.minimizer_progress_to_stdout = opts.verbose;
}

}  // namespace

// ============================================================================
// CeresOptimizer — GradientProblemSolver
// ============================================================================

fit_expected<LbfgsOutput>
CeresOptimizer::minimize(Objective f, const Eigen::VectorXd& x0) const {
  const int n = static_cast<int>(x0.size());
  // ceres::GradientProblem takes ownership of the FirstOrderFunction*.
  ceres::GradientProblem problem(new StdObjectiveFirstOrder(std::move(f), n));

  Eigen::VectorXd theta = x0;
  ceres::GradientProblemSolver::Options options;
  apply_options(options, opts_);
  ceres::GradientProblemSolver::Summary summary;
  try {
    ceres::Solve(options, problem, theta.data(), &summary);
  } catch (const std::exception& e) {
    return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
        std::string("ceres (gradient) threw: ") + e.what()));
  }
  if (summary.termination_type != ceres::CONVERGENCE &&
      summary.termination_type != ceres::USER_SUCCESS) {
    return std::unexpected(summary_to_err(summary, "gradient"));
  }
  return LbfgsOutput{
      std::move(theta),
      summary.final_cost,
      static_cast<int>(summary.iterations.size())};
}

// ============================================================================
// CeresBoundedOptimizer — Problem API with native bounds
// ============================================================================

fit_expected<LbfgsOutput>
CeresBoundedOptimizer::minimize(Objective              f,
                                const Eigen::VectorXd& x0) const {
  // Unbounded path: delegate to the GradientProblemSolver, which avoids the
  // √(2F+ε) trick entirely when bounds aren't needed. This way the
  // discrepancy gradient flows through directly rather than via the
  // single-residual chain-rule rewrite.
  return CeresOptimizer{opts_}.minimize(std::move(f), x0);
}

fit_expected<LbfgsOutput>
CeresBoundedOptimizer::minimize(Objective              f,
                                const Eigen::VectorXd& x0,
                                const Eigen::VectorXd& lower,
                                const Eigen::VectorXd& upper) const {
  const int n = static_cast<int>(x0.size());
  if (lower.size() != x0.size() || upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "ceres-bounded: lower/upper bound size != x0 size"));
  }

  Eigen::VectorXd theta = x0;
  ceres::Problem problem;
  // Problem takes ownership of the CostFunction* via AddResidualBlock when
  // we pass `ceres::TAKE_OWNERSHIP` (the default in Problem::Options).
  problem.AddResidualBlock(new StdObjectiveSqrtCost(std::move(f), n),
                           nullptr,
                           theta.data());

  // Finite bounds only — ±inf means "no bound on this axis" and Ceres'
  // SetParameterLowerBound rejects infinity.
  for (int i = 0; i < n; ++i) {
    if (std::isfinite(lower(i))) {
      problem.SetParameterLowerBound(theta.data(), i, lower(i));
    }
    if (std::isfinite(upper(i))) {
      problem.SetParameterUpperBound(theta.data(), i, upper(i));
    }
  }

  ceres::Solver::Options options;
  apply_options(options, opts_);
  // The single-residual `r₀ = √(2F+ε)` trick has a 1×n Jacobian; `JᵀJ` is
  // rank 1 (singular for n > 1). Ceres' default SPARSE_NORMAL_CHOLESKY tries
  // an AMD reordering on the (degenerate) sparse structure and segfaults.
  // DENSE_QR handles the rank deficiency cleanly via LM damping. The problem
  // size is tiny by Ceres standards (n ≤ ~hundreds), so dense is fine here.
  options.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  try {
    ceres::Solve(options, &problem, &summary);
  } catch (const std::exception& e) {
    return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
        std::string("ceres (bounded) threw: ") + e.what()));
  }
  if (summary.termination_type != ceres::CONVERGENCE &&
      summary.termination_type != ceres::USER_SUCCESS) {
    return std::unexpected(summary_to_err(summary, "bounded"));
  }

  // summary.final_cost is ½·r₀² = ½·(2F + ε) = F + ε/2; back-transform.
  const double fmin = std::max(0.0, summary.final_cost - 0.5 * kSqrtEps);
  return LbfgsOutput{
      std::move(theta),
      fmin,
      static_cast<int>(summary.iterations.size())};
}

// ============================================================================
// CeresBoundedOptimizer::minimize_ls — true multi-residual LS path
// ============================================================================

fit_expected<LbfgsOutput>
CeresBoundedOptimizer::minimize_ls(LsResidualFn r_fn,
                                   LsJacobianFn J_fn,
                                   Eigen::Index n_resid,
                                   const Eigen::VectorXd& x0,
                                   const Eigen::VectorXd& lower,
                                   const Eigen::VectorXd& upper) const {
  const int n = static_cast<int>(x0.size());
  if (lower.size() != x0.size() || upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "ceres-bounded (ls): lower/upper bound size != x0 size"));
  }
  if (n_resid <= 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "ceres-bounded (ls): n_resid must be positive (got " +
            std::to_string(n_resid) + ")"));
  }

  Eigen::VectorXd theta = x0;
  ceres::Problem problem;
  problem.AddResidualBlock(
      new LsCostFunction(std::move(r_fn), std::move(J_fn),
                         static_cast<int>(n_resid), n),
      nullptr,
      theta.data());

  for (int i = 0; i < n; ++i) {
    if (std::isfinite(lower(i))) {
      problem.SetParameterLowerBound(theta.data(), i, lower(i));
    }
    if (std::isfinite(upper(i))) {
      problem.SetParameterUpperBound(theta.data(), i, upper(i));
    }
  }

  ceres::Solver::Options options;
  apply_options(options, opts_);
  // For typical SEM sizes n_resid is on the same order as n_param (and
  // certainly modest by Ceres standards), so a dense linear solver is the
  // right choice. DENSE_QR is more numerically stable than
  // DENSE_NORMAL_CHOLESKY when the residual count just barely exceeds the
  // parameter count, which is common for ULS on saturated models.
  options.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  try {
    ceres::Solve(options, &problem, &summary);
  } catch (const std::exception& e) {
    return std::unexpected(make_err(FitError::Kind::LineSearchFailed,
        std::string("ceres (bounded-ls) threw: ") + e.what()));
  }
  if (summary.termination_type != ceres::CONVERGENCE &&
      summary.termination_type != ceres::USER_SUCCESS) {
    return std::unexpected(summary_to_err(summary, "bounded-ls"));
  }

  // Ceres' final_cost is already ½‖r‖² — no sqrt-trick back-transform.
  return LbfgsOutput{
      std::move(theta),
      std::max(0.0, summary.final_cost),
      static_cast<int>(summary.iterations.size())};
}

}  // namespace magmaan::fit

#endif  // MAGMAAN_WITH_CERES
