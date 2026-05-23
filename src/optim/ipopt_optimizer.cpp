#include "magmaan/optim/ipopt_optimizer.hpp"

#ifdef MAGMAAN_WITH_IPOPT

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#if __has_include(<coin-or/IpStdCInterface.h>)
#include <coin-or/IpStdCInterface.h>
#else
#include <IpStdCInterface.h>
#endif

#include "magmaan/error.hpp"
#include "magmaan/expected.hpp"
#include "magmaan/optim/terminal_audit.hpp"

namespace magmaan::optim {

namespace {

constexpr double kIpoptInf = 2e19;
constexpr int kMaximumWallTimeExceededCode = -5;  // Named in IPOPT >= 3.14.

FitError make_err(FitError::Kind k, std::string detail,
                  int iter = 0, double fval = 0.0) {
  return FitError{k, std::move(detail), iter, fval};
}

ipnumber* ptr(std::vector<ipnumber>& v) noexcept {
  return v.empty() ? nullptr : v.data();
}

const char* status_name(ApplicationReturnStatus status) noexcept {
  switch (status) {
    case Solve_Succeeded: return "Solve_Succeeded";
    case Solved_To_Acceptable_Level: return "Solved_To_Acceptable_Level";
    case Infeasible_Problem_Detected: return "Infeasible_Problem_Detected";
    case Search_Direction_Becomes_Too_Small:
      return "Search_Direction_Becomes_Too_Small";
    case Diverging_Iterates: return "Diverging_Iterates";
    case User_Requested_Stop: return "User_Requested_Stop";
    case Feasible_Point_Found: return "Feasible_Point_Found";
    case Maximum_Iterations_Exceeded: return "Maximum_Iterations_Exceeded";
    case Restoration_Failed: return "Restoration_Failed";
    case Error_In_Step_Computation: return "Error_In_Step_Computation";
    case Maximum_CpuTime_Exceeded: return "Maximum_CpuTime_Exceeded";
    case Not_Enough_Degrees_Of_Freedom: return "Not_Enough_Degrees_Of_Freedom";
    case Invalid_Problem_Definition: return "Invalid_Problem_Definition";
    case Invalid_Option: return "Invalid_Option";
    case Invalid_Number_Detected: return "Invalid_Number_Detected";
    case Unrecoverable_Exception: return "Unrecoverable_Exception";
    case NonIpopt_Exception_Thrown: return "NonIpopt_Exception_Thrown";
    case Insufficient_Memory: return "Insufficient_Memory";
    case Internal_Error: return "Internal_Error";
    default:
      if (static_cast<int>(status) == kMaximumWallTimeExceededCode) {
        return "Maximum_WallTime_Exceeded";
      }
      return "Unknown";
  }
}

FitError::Kind status_error_kind(ApplicationReturnStatus status) noexcept {
  switch (status) {
    case Infeasible_Problem_Detected:
    case Not_Enough_Degrees_Of_Freedom:
    case Invalid_Problem_Definition:
    case Invalid_Option:
    case Invalid_Number_Detected:
      return FitError::Kind::NumericIssue;
    case Maximum_Iterations_Exceeded:
    case Maximum_CpuTime_Exceeded:
    case Diverging_Iterates:
    case Restoration_Failed:
      return FitError::Kind::OptimizerNonConvergence;
    default:
      if (static_cast<int>(status) == kMaximumWallTimeExceededCode) {
        return FitError::Kind::OptimizerNonConvergence;
      }
      return FitError::Kind::LineSearchFailed;
  }
}

std::vector<ipnumber> ipopt_bounds(const Eigen::VectorXd& b,
                                   bool lower) {
  std::vector<ipnumber> out(static_cast<std::size_t>(b.size()));
  for (Eigen::Index i = 0; i < b.size(); ++i) {
    const double x = b(i);
    out[static_cast<std::size_t>(i)] =
        std::isfinite(x) ? static_cast<ipnumber>(x)
                         : static_cast<ipnumber>(lower ? -kIpoptInf
                                                       :  kIpoptInf);
  }
  return out;
}

struct CallbackData {
  const ConstrainedScalarProblem* prob = nullptr;
  Eigen::Index n = 0;
  Eigen::Index m = 0;
  int f_evals = 0;
  int g_evals = 0;
};

bool eval_f(ipindex n, ipnumber* x, bool, ipnumber* obj_value,
            UserDataPtr user_data) {
  auto* data = static_cast<CallbackData*>(user_data);
  if (!data || !data->prob || n != data->n || !x || !obj_value) return false;
  const Eigen::Map<const Eigen::VectorXd> xv(x, n);
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(n);
  const double f = data->prob->objective.f(xv, grad);
  ++data->f_evals;
  if (!std::isfinite(f)) return false;
  *obj_value = static_cast<ipnumber>(f);
  return true;
}

bool eval_grad_f(ipindex n, ipnumber* x, bool, ipnumber* grad_f,
                 UserDataPtr user_data) {
  auto* data = static_cast<CallbackData*>(user_data);
  if (!data || !data->prob || n != data->n || !x || !grad_f) return false;
  const Eigen::Map<const Eigen::VectorXd> xv(x, n);
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(n);
  const double f = data->prob->objective.f(xv, grad);
  ++data->g_evals;
  if (!std::isfinite(f) || grad.size() != n || !grad.allFinite()) return false;
  Eigen::Map<Eigen::VectorXd>(grad_f, n) = grad;
  return true;
}

bool eval_g(ipindex n, ipnumber* x, bool, ipindex m, ipnumber* g,
            UserDataPtr user_data) {
  auto* data = static_cast<CallbackData*>(user_data);
  if (!data || !data->prob || n != data->n || m != data->m) return false;
  if (m == 0) return true;
  if (!x || !g) return false;
  const Eigen::Map<const Eigen::VectorXd> xv(x, n);
  const Eigen::VectorXd h = data->prob->h(xv);
  if (h.size() != m || !h.allFinite()) return false;
  Eigen::Map<Eigen::VectorXd>(g, m) = h;
  return true;
}

bool eval_jac_g(ipindex n, ipnumber* x, bool, ipindex m, ipindex nele_jac,
                ipindex* iRow, ipindex* jCol, ipnumber* values,
                UserDataPtr user_data) {
  auto* data = static_cast<CallbackData*>(user_data);
  if (!data || !data->prob || n != data->n || m != data->m) return false;
  if (nele_jac != n * m) return false;
  if (!values) {
    if (nele_jac == 0) return true;
    if (!iRow || !jCol) return false;
    ipindex k = 0;
    for (ipindex r = 0; r < m; ++r) {
      for (ipindex c = 0; c < n; ++c) {
        iRow[k] = r;
        jCol[k] = c;
        ++k;
      }
    }
    return true;
  }
  if (m == 0) return true;
  if (!x) return false;
  const Eigen::Map<const Eigen::VectorXd> xv(x, n);
  const Eigen::MatrixXd J = data->prob->J_h(xv);
  if (J.rows() != m || J.cols() != n || !J.allFinite()) return false;
  ipindex k = 0;
  for (ipindex r = 0; r < m; ++r) {
    for (ipindex c = 0; c < n; ++c) {
      values[k++] = static_cast<ipnumber>(J(r, c));
    }
  }
  return true;
}

bool add_str_option(IpoptProblem nlp, const char* key, const char* val) {
  return AddIpoptStrOption(nlp, const_cast<char*>(key),
                           const_cast<char*>(val));
}

bool add_num_option(IpoptProblem nlp, const char* key, double val) {
  return AddIpoptNumOption(nlp, const_cast<char*>(key),
                           static_cast<ipnumber>(val));
}

bool add_int_option(IpoptProblem nlp, const char* key, int val) {
  return AddIpoptIntOption(nlp, const_cast<char*>(key),
                           static_cast<ipindex>(val));
}

fit_expected<OptimOutput>
solve_ipopt(const ConstrainedScalarProblem& prob,
            const Eigen::VectorXd& x0,
            const Eigen::VectorXd& lower,
            const Eigen::VectorXd& upper,
            OptimOptions opts) {
  if (!prob.objective.f) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "IpoptOptimizer: empty objective callback"));
  }
  if (x0.size() == 0) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "IpoptOptimizer: empty parameter vector"));
  }
  if (prob.objective.n_param != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "IpoptOptimizer: objective n_param does not match x0"));
  }
  if (lower.size() != x0.size() || upper.size() != x0.size()) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "IpoptOptimizer: bounds size mismatch"));
  }
  const Eigen::Index m = prob.n_constraint;
  if (prob.constraint_lower.size() != m || prob.constraint_upper.size() != m) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "IpoptOptimizer: constraint bound size mismatch"));
  }
  if (m > 0 && (!prob.h || !prob.J_h)) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "IpoptOptimizer: constrained solve missing h/J_h callbacks"));
  }
  for (Eigen::Index i = 0; i < x0.size(); ++i) {
    if (!std::isfinite(x0(i))) {
      return std::unexpected(make_err(FitError::Kind::InvalidStartValues,
          "IpoptOptimizer: x0 contains non-finite values"));
    }
    if (lower(i) > upper(i)) {
      return std::unexpected(make_err(FitError::Kind::NumericIssue,
          "IpoptOptimizer: lower bound exceeds upper bound"));
    }
  }

  const ipindex n = static_cast<ipindex>(x0.size());
  const ipindex mc = static_cast<ipindex>(m);
  std::vector<ipnumber> x_L = ipopt_bounds(lower, true);
  std::vector<ipnumber> x_U = ipopt_bounds(upper, false);
  std::vector<ipnumber> g_L = ipopt_bounds(prob.constraint_lower, true);
  std::vector<ipnumber> g_U = ipopt_bounds(prob.constraint_upper, false);

  const ipindex nele_jac = n * mc;
  IpoptProblem nlp = CreateIpoptProblem(
      n, ptr(x_L), ptr(x_U), mc, ptr(g_L), ptr(g_U), nele_jac,
      /*nele_hess=*/0, /*index_style=*/0, eval_f, eval_g, eval_grad_f,
      eval_jac_g, nullptr);
  if (!nlp) {
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "IpoptOptimizer: CreateIpoptProblem failed"));
  }

  auto free_nlp = [&]() { FreeIpoptProblem(nlp); };
  if (!add_str_option(nlp, "hessian_approximation", "limited-memory") ||
      !add_str_option(nlp, "sb", "yes") ||
      !add_int_option(nlp, "print_level", 0) ||
      !add_int_option(nlp, "max_iter", opts.max_iter) ||
      !add_int_option(nlp, "limited_memory_max_history", opts.history) ||
      !add_num_option(nlp, "tol", opts.gtol) ||
      !add_num_option(nlp, "acceptable_tol", std::max(opts.ftol, opts.gtol))) {
    free_nlp();
    return std::unexpected(make_err(FitError::Kind::NumericIssue,
        "IpoptOptimizer: failed to set IPOPT options"));
  }

  std::vector<ipnumber> x(static_cast<std::size_t>(n));
  for (ipindex i = 0; i < n; ++i) {
    double xi = x0(i);
    if (std::isfinite(lower(i))) xi = std::max(xi, lower(i));
    if (std::isfinite(upper(i))) xi = std::min(xi, upper(i));
    x[static_cast<std::size_t>(i)] = static_cast<ipnumber>(xi);
  }
  std::vector<ipnumber> g(static_cast<std::size_t>(mc));
  std::vector<ipnumber> mult_g(static_cast<std::size_t>(mc));
  std::vector<ipnumber> mult_x_L(static_cast<std::size_t>(n));
  std::vector<ipnumber> mult_x_U(static_cast<std::size_t>(n));
  ipnumber obj = std::numeric_limits<ipnumber>::quiet_NaN();

  CallbackData data{&prob, x0.size(), m, 0, 0};
  const ApplicationReturnStatus status =
      IpoptSolve(nlp, x.data(), ptr(g), &obj, ptr(mult_g),
                 ptr(mult_x_L), ptr(mult_x_U), &data);
  free_nlp();

  Eigen::VectorXd x_final(n);
  for (ipindex i = 0; i < n; ++i) x_final(i) = x[static_cast<std::size_t>(i)];
  double f_final = static_cast<double>(obj);
  if (!std::isfinite(f_final)) {
    Eigen::VectorXd scratch(n);
    f_final = prob.objective.f(x_final, scratch);
  }

  const bool clean_success = status == Solve_Succeeded;
  const bool acceptable_success = status == Solved_To_Acceptable_Level;
  OptimStatus opt_status =
      acceptable_success ? OptimStatus::LineSearchSalvaged
                         : OptimStatus::Converged;

  TerminalAudit audit;
  double grad_inf = -1.0;
  if (m == 0 && std::isfinite(f_final)) {
    audit = audit_terminal_iterate(prob.objective.f, x_final, f_final,
                                   lower, upper);
    grad_inf = audit.grad_inf_norm;
  }

  if (!clean_success && !acceptable_success) {
    const bool salvage_unconstrained =
        m == 0 && std::isfinite(f_final) && audit.stationary &&
        (status == Search_Direction_Becomes_Too_Small ||
         status == Maximum_Iterations_Exceeded ||
         status == Maximum_CpuTime_Exceeded ||
         static_cast<int>(status) == kMaximumWallTimeExceededCode ||
         status == Feasible_Point_Found);
    if (salvage_unconstrained) {
      opt_status = OptimStatus::LineSearchSalvaged;
    } else {
      return std::unexpected(make_err(status_error_kind(status),
          std::string("IPOPT: ") + status_name(status),
          /*iter=*/0, f_final));
    }
  }

  OptimOutput out{std::move(x_final), f_final, /*iterations=*/0,
                  data.f_evals, data.g_evals, opt_status, grad_inf};
  out.audit = std::move(audit);
  return out;
}

}  // namespace

fit_expected<OptimOutput>
IpoptOptimizer::minimize(Objective f, const Eigen::VectorXd& x0) const {
  const double inf = std::numeric_limits<double>::infinity();
  return minimize(std::move(f), x0,
                  Eigen::VectorXd::Constant(x0.size(), -inf),
                  Eigen::VectorXd::Constant(x0.size(),  inf));
}

fit_expected<OptimOutput>
IpoptOptimizer::minimize(Objective f, const Eigen::VectorXd& x0,
                         const Eigen::VectorXd& lower,
                         const Eigen::VectorXd& upper) const {
  ConstrainedScalarProblem prob;
  prob.objective.f = std::move(f);
  prob.objective.n_param = x0.size();
  prob.objective.expand = [](const Eigen::VectorXd& x) { return x; };
  prob.n_constraint = 0;
  return solve_ipopt(prob, x0, lower, upper, opts_);
}

fit_expected<OptimOutput>
IpoptOptimizer::minimize_constrained(const ConstrainedScalarProblem& prob,
                                     const Eigen::VectorXd& x0,
                                     const Eigen::VectorXd& lower,
                                     const Eigen::VectorXd& upper) const {
  return solve_ipopt(prob, x0, lower, upper, opts_);
}

}  // namespace magmaan::optim

#endif  // MAGMAAN_WITH_IPOPT
