#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

namespace {

using magmaan::data::OrdinalEstimatorKind;
using magmaan::data::OrdinalGammaCache;
using magmaan::data::OrdinalMomentParameterization;
using magmaan::data::OrdinalStats;
using magmaan::data::OrdinalThresholdMode;
using magmaan::data::OrdinalWorkspacePurpose;
using magmaan::estimate::Backend;
using magmaan::estimate::Estimates;
using magmaan::estimate::OrdinalParameterization;

constexpr double quiet_nan() {
  return std::numeric_limits<double>::quiet_NaN();
}

struct Timing {
  double median_ms = quiet_nan();
  double min_ms = quiet_nan();
  double max_ms = quiet_nan();
};

struct TimedStatus {
  bool ok = true;
  Timing timing;
};

template <class Fn>
TimedStatus measure_reps(int reps, Fn&& fn) {
  std::vector<double> times;
  times.reserve(static_cast<std::size_t>(std::max(1, reps)));
  for (int rep = 0; rep < std::max(1, reps); ++rep) {
    const auto start = std::chrono::steady_clock::now();
    const bool ok = fn();
    const auto stop = std::chrono::steady_clock::now();
    if (!ok) return {.ok = false, .timing = {}};
    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    times.push_back(elapsed.count());
  }
  std::sort(times.begin(), times.end());
  const std::size_t mid = times.size() / 2U;
  const double median =
      (times.size() % 2U == 0U) ? 0.5 * (times[mid - 1U] + times[mid])
                                : times[mid];
  return {.ok = true,
          .timing = {.median_ms = median,
                     .min_ms = times.front(),
                     .max_ms = times.back()}};
}

int parse_int_arg(const char* arg, int fallback) {
  if (arg == nullptr) return fallback;
  char* end = nullptr;
  const long value = std::strtol(arg, &end, 10);
  if (end == arg || *end != '\0') return fallback;
  if (value < 0L ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(value);
}

std::string csv_escape(const std::string& value) {
  bool needs_quotes = false;
  for (const char ch : value) {
    if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) return value;
  std::string out = "\"";
  for (const char ch : value) {
    if (ch == '"') out += '"';
    out += ch;
  }
  out += '"';
  return out;
}

std::string bool_csv(bool value) {
  return value ? "1" : "0";
}

std::string optim_status_name(magmaan::optim::OptimStatus status) {
  switch (status) {
    case magmaan::optim::OptimStatus::Converged:
      return "converged";
    case magmaan::optim::OptimStatus::LineSearchSalvaged:
      return "line_search_salvaged";
    case magmaan::optim::OptimStatus::SingularConvergence:
      return "singular_convergence";
    case magmaan::optim::OptimStatus::NoisyObjective:
      return "noisy_objective";
    case magmaan::optim::OptimStatus::FalseConvergence:
      return "false_convergence";
    case magmaan::optim::OptimStatus::BudgetExhausted:
      return "budget_exhausted";
    case magmaan::optim::OptimStatus::Unknown:
      return "unknown";
  }
  return "unknown";
}

std::string estimator_name(OrdinalEstimatorKind estimator) {
  switch (estimator) {
    case OrdinalEstimatorKind::ULS:
      return "ULS";
    case OrdinalEstimatorKind::DWLS:
      return "DWLS";
    case OrdinalEstimatorKind::WLS:
      return "WLS";
  }
  return "unknown";
}

std::string parameterization_name(OrdinalMomentParameterization parameterization) {
  switch (parameterization) {
    case OrdinalMomentParameterization::Delta:
      return "delta";
    case OrdinalMomentParameterization::Theta:
      return "theta";
  }
  return "unknown";
}

OrdinalParameterization fit_parameterization(
    OrdinalMomentParameterization parameterization) {
  return parameterization == OrdinalMomentParameterization::Theta
             ? OrdinalParameterization::Theta
             : OrdinalParameterization::Delta;
}

std::string threshold_mode_name(OrdinalThresholdMode mode) {
  switch (mode) {
    case OrdinalThresholdMode::FreeIdentity:
      return "free";
    case OrdinalThresholdMode::LinearMap:
      return "shared";
    case OrdinalThresholdMode::FixedOrConstrained:
      return "fixed";
  }
  return "unknown";
}

OrdinalThresholdMode plan_threshold_mode(OrdinalThresholdMode mode) {
  return mode == OrdinalThresholdMode::FreeIdentity
      ? OrdinalThresholdMode::FreeIdentity
      : OrdinalThresholdMode::FixedOrConstrained;
}

std::vector<double> thresholds_for_categories(int categories,
                                              const std::string& balance) {
  if (balance == "skewed") {
    if (categories <= 2) return {0.80};
    if (categories == 3) return {0.20, 1.10};
    if (categories == 5) return {-0.20, 0.40, 1.00, 1.70};
    return {-0.40, 0.00, 0.45, 0.90, 1.35, 1.90};
  }
  if (categories <= 2) return {0.0};
  if (categories == 3) return {-0.55, 0.55};
  if (categories == 5) return {-1.20, -0.35, 0.35, 1.20};
  return {-1.50, -0.90, -0.35, 0.35, 0.90, 1.50};
}

std::string make_repeated_model_syntax(int q, int categories,
                                       OrdinalThresholdMode threshold_mode) {
  const int p = 4 * q;
  std::ostringstream syntax;
  for (int f = 0; f < 4; ++f) {
    syntax << "f" << (f + 1) << " =~ ";
    for (int j = 1; j <= q; ++j) {
      if (j > 1) syntax << " + ";
      syntax << "x" << (f * q + j);
    }
    syntax << "\n";
  }
  for (int j = 1; j <= q; ++j) {
    syntax << "x" << j << " ~~ x" << (q + j) << "\n";
    syntax << "x" << (2 * q + j) << " ~~ x" << (3 * q + j) << "\n";
  }

  const int n_threshold = categories - 1;
  for (int j = 1; j <= p; ++j) {
    syntax << "x" << j << " | ";
    for (int k = 1; k <= n_threshold; ++k) {
      if (k > 1) syntax << " + ";
      if (threshold_mode == OrdinalThresholdMode::FixedOrConstrained &&
          j == 1 && k == 1) {
        syntax << "0*";
      } else if (threshold_mode == OrdinalThresholdMode::LinearMap &&
                 j <= 2 && k == 1) {
        syntax << "a*";
      }
      syntax << "t" << k;
    }
    syntax << "\n";
  }
  for (int j = 1; j <= p; ++j) {
    syntax << "x" << j << " ~*~ 1*x" << j << "\n";
  }
  return syntax.str();
}

Eigen::MatrixXd equicor_chol(int size, double rho) {
  Eigen::MatrixXd cor = Eigen::MatrixXd::Constant(size, size, rho);
  cor.diagonal().setOnes();
  Eigen::LLT<Eigen::MatrixXd> llt(cor);
  return llt.matrixL();
}

Eigen::MatrixXd repeated_residual_chol(int q, double residual_var,
                                       double residual_cov) {
  const int p = 4 * q;
  Eigen::MatrixXd theta = Eigen::MatrixXd::Zero(p, p);
  theta.diagonal().setConstant(residual_var);
  for (int j = 0; j < q; ++j) {
    theta(j, q + j) = residual_cov;
    theta(q + j, j) = residual_cov;
    theta(2 * q + j, 3 * q + j) = residual_cov;
    theta(3 * q + j, 2 * q + j) = residual_cov;
  }
  Eigen::LLT<Eigen::MatrixXd> llt(theta);
  return llt.matrixL();
}

Eigen::MatrixXd simulate_repeated_ordinal(int n, int q, int categories,
                                          const std::string& balance, int seed,
                                          double loading = 0.7,
                                          double factor_cor = 0.3,
                                          double residual_var = 1.0,
                                          double residual_cov = 0.2) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<double> norm(0.0, 1.0);
  const int p = 4 * q;
  const auto thresholds = thresholds_for_categories(categories, balance);
  const Eigen::MatrixXd factor_L = equicor_chol(4, factor_cor);
  const Eigen::MatrixXd resid_L =
      repeated_residual_chol(q, residual_var, residual_cov);

  Eigen::MatrixXd x(n, p);
  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    Eigen::VectorXd z_eta(4);
    for (Eigen::Index k = 0; k < z_eta.size(); ++k) z_eta(k) = norm(rng);
    const Eigen::VectorXd eta = factor_L * z_eta;
    Eigen::VectorXd z_eps(p);
    for (Eigen::Index k = 0; k < z_eps.size(); ++k) z_eps(k) = norm(rng);
    const Eigen::VectorXd eps = resid_L * z_eps;
    for (int j = 0; j < p; ++j) {
      const int factor = j / q;
      const double y = loading * eta(factor) + eps(j);
      int category = 1;
      for (const double threshold : thresholds) {
        if (y > threshold) ++category;
      }
      x(i, j) = static_cast<double>(category);
    }
  }
  return x;
}

int moment_dimension(int p, int categories) {
  return p * (categories - 1) + (p * (p - 1)) / 2;
}

struct CacheFlags {
  int block_count = 0;
  bool has_diagonal = false;
  bool has_full = false;
  bool has_dwls_weight = false;
  bool has_wls_weight = false;
};

CacheFlags cache_flags(const OrdinalGammaCache* cache) {
  if (cache == nullptr || cache->blocks.empty()) return {};
  const auto& block = cache->blocks[0];
  return {.block_count = static_cast<int>(cache->blocks.size()),
          .has_diagonal = block.has_diagonal,
          .has_full = block.has_full,
          .has_dwls_weight = block.has_dwls_weight,
          .has_wls_weight = block.has_wls_weight};
}

struct CacheBundle {
  OrdinalGammaCache cache;
  bool use_cache = false;
  Timing cache_setup;
  Timing weight_setup;
  std::string error;

  OrdinalGammaCache* ptr() noexcept { return use_cache ? &cache : nullptr; }
};

CacheBundle make_fit_cache_unmeasured(const OrdinalStats& stats,
                                      OrdinalEstimatorKind estimator) {
  CacheBundle bundle;
  if (estimator == OrdinalEstimatorKind::ULS) return bundle;
  if (estimator == OrdinalEstimatorKind::DWLS) {
    bundle.cache =
        magmaan::data::ordinal_gamma_cache_from_diagonal(
            {stats.NACOV[0].diagonal()});
    bundle.use_cache = true;
    return bundle;
  }

  bundle.cache.blocks.resize(1);
  bundle.cache.blocks[0].gamma = stats.NACOV[0];
  bundle.cache.blocks[0].has_full = true;
  bundle.use_cache = true;
  auto ok = magmaan::data::ordinal_gamma_cache_ensure_wls_weights(
      bundle.cache);
  if (!ok.has_value()) bundle.error = ok.error().detail;
  return bundle;
}

CacheBundle make_fit_cache(const OrdinalStats& stats,
                           OrdinalEstimatorKind estimator) {
  CacheBundle bundle;
  if (estimator == OrdinalEstimatorKind::ULS) return bundle;
  if (estimator == OrdinalEstimatorKind::DWLS) {
    const auto timed = measure_reps(1, [&]() {
      bundle.cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
          {stats.NACOV[0].diagonal()});
      return true;
    });
    bundle.cache_setup = timed.timing;
    bundle.use_cache = true;
    return bundle;
  }

  const auto cache_timed = measure_reps(1, [&]() {
    bundle.cache.blocks.resize(1);
    bundle.cache.blocks[0].gamma = stats.NACOV[0];
    bundle.cache.blocks[0].has_full = true;
    bundle.use_cache = true;
    return true;
  });
  bundle.cache_setup = cache_timed.timing;
  const auto weight_timed = measure_reps(1, [&]() {
    auto ok = magmaan::data::ordinal_gamma_cache_ensure_wls_weights(
        bundle.cache);
    if (!ok.has_value()) {
      bundle.error = ok.error().detail;
      return false;
    }
    return true;
  });
  bundle.weight_setup = weight_timed.timing;
  if (!weight_timed.ok && bundle.error.empty()) {
    bundle.error = "failed to materialize WLS weight";
  }
  return bundle;
}

double theta_max_abs_diff(const Estimates& lhs, const Estimates& rhs) {
  if (lhs.theta.size() != rhs.theta.size()) return quiet_nan();
  if (lhs.theta.size() == 0) return 0.0;
  return (lhs.theta - rhs.theta).cwiseAbs().maxCoeff();
}

struct Row {
  std::string design;
  std::string case_id;
  int seed = 0;
  int n = 0;
  int q = 0;
  int p = 0;
  int categories = 0;
  std::string threshold_balance;
  std::string threshold_mode;
  int reps = 0;
  int max_iter = 0;
  int n_free = 0;
  int n_thresholds = 0;
  int moment_dim = 0;
  std::string parameterization = "delta";
  std::string estimator;
  std::string path;
  std::string construction = "precomputed_legacy";
  std::string status = "ok";
  std::string error;
  double model_setup_ms = quiet_nan();
  double stats_ms = quiet_nan();
  double moments_ms = quiet_nan();
  double starts_ms = quiet_nan();
  double cache_setup_ms = quiet_nan();
  double weight_setup_ms = quiet_nan();
  double fit_median_ms = quiet_nan();
  double fit_min_ms = quiet_nan();
  double fit_max_ms = quiet_nan();
  double total_median_ms = quiet_nan();
  double total_min_ms = quiet_nan();
  double total_max_ms = quiet_nan();
  double fmin = quiet_nan();
  int iterations = -1;
  int f_evals = -1;
  int g_evals = -1;
  std::string optimizer_status;
  double grad_inf_norm = quiet_nan();
  int n_nonlinear = -1;
  int n_linear = -1;
  int cache_blocks = 0;
  bool cache_has_diagonal = false;
  bool cache_has_full = false;
  bool cache_has_dwls_weight = false;
  bool cache_has_wls_weight = false;
  double theta_diff_profiled_bounded = quiet_nan();
  double fmin_diff_profiled_bounded = quiet_nan();
  double theta_diff_full_bounded = quiet_nan();
  double fmin_diff_full_bounded = quiet_nan();
};

void write_header(std::ostream& out) {
  out << "design,case_id,seed,n,q,p,categories,threshold_balance,"
         "threshold_mode,reps,max_iter,n_free,n_thresholds,moment_dim,"
         "parameterization,estimator,path,construction,status,error,"
         "model_setup_ms,stats_ms,moments_ms,starts_ms,"
         "cache_setup_ms,weight_setup_ms,fit_median_ms,fit_min_ms,"
         "fit_max_ms,total_median_ms,total_min_ms,total_max_ms,fmin,"
         "iterations,f_evals,"
         "g_evals,optimizer_status,grad_inf_norm,n_nonlinear,n_linear,"
         "cache_blocks,cache_has_diagonal,cache_has_full,"
         "cache_has_dwls_weight,cache_has_wls_weight,"
         "theta_diff_profiled_bounded,fmin_diff_profiled_bounded,"
         "theta_diff_full_bounded,fmin_diff_full_bounded\n";
}

void write_row(std::ostream& out, const Row& row) {
  out << csv_escape(row.design) << ',' << csv_escape(row.case_id) << ','
      << row.seed << ',' << row.n << ',' << row.q << ',' << row.p << ','
      << row.categories << ',' << csv_escape(row.threshold_balance) << ','
      << csv_escape(row.threshold_mode) << ',' << row.reps << ','
      << row.max_iter << ',' << row.n_free << ',' << row.n_thresholds << ','
      << row.moment_dim << ',' << csv_escape(row.parameterization) << ','
      << csv_escape(row.estimator) << ',' << csv_escape(row.path) << ','
      << csv_escape(row.construction) << ',' << csv_escape(row.status) << ','
      << csv_escape(row.error) << ',' << row.model_setup_ms << ','
      << row.stats_ms << ',' << row.moments_ms << ',' << row.starts_ms << ','
      << row.cache_setup_ms << ','
      << row.weight_setup_ms << ',' << row.fit_median_ms << ','
      << row.fit_min_ms << ',' << row.fit_max_ms << ','
      << row.total_median_ms << ',' << row.total_min_ms << ','
      << row.total_max_ms << ',' << row.fmin << ',' << row.iterations << ','
      << row.f_evals << ',' << row.g_evals << ','
      << csv_escape(row.optimizer_status) << ',' << row.grad_inf_norm << ','
      << row.n_nonlinear << ',' << row.n_linear << ',' << row.cache_blocks
      << ',' << bool_csv(row.cache_has_diagonal) << ','
      << bool_csv(row.cache_has_full) << ','
      << bool_csv(row.cache_has_dwls_weight) << ','
      << bool_csv(row.cache_has_wls_weight) << ','
      << row.theta_diff_profiled_bounded << ','
      << row.fmin_diff_profiled_bounded << ','
      << row.theta_diff_full_bounded << ','
      << row.fmin_diff_full_bounded << '\n';
}

void fill_estimate_fields(Row& row, const Estimates& est) {
  row.fmin = est.fmin;
  row.iterations = est.iterations;
  row.f_evals = est.f_evals;
  row.g_evals = est.g_evals;
  row.optimizer_status = optim_status_name(est.optimizer_status);
  row.grad_inf_norm = est.grad_inf_norm;
  row.n_nonlinear = est.n_nonlinear;
  row.n_linear = est.n_linear;
}

struct TimedFitResult {
  Row row;
  bool have_estimate = false;
  Estimates estimate;
};

template <class FitFn>
TimedFitResult run_timed_fit(Row base, int reps, FitFn&& fit_fn,
                             const Estimates* profiled_bounded_ref,
                             const Estimates* full_bounded_ref) {
  Estimates last;
  std::string error;
  const auto timed = measure_reps(reps, [&]() {
    auto fit = fit_fn();
    if (!fit.has_value()) {
      error = fit.error().detail;
      return false;
    }
    last = *fit;
    return true;
  });
  if (!timed.ok) {
    base.status = "error";
    base.error = error.empty() ? "fit failed" : error;
    return {.row = base};
  }
  base.fit_median_ms = timed.timing.median_ms;
  base.fit_min_ms = timed.timing.min_ms;
  base.fit_max_ms = timed.timing.max_ms;
  fill_estimate_fields(base, last);
  if (profiled_bounded_ref != nullptr) {
    base.theta_diff_profiled_bounded =
        theta_max_abs_diff(last, *profiled_bounded_ref);
    base.fmin_diff_profiled_bounded =
        std::abs(last.fmin - profiled_bounded_ref->fmin);
  }
  if (full_bounded_ref != nullptr) {
    base.theta_diff_full_bounded = theta_max_abs_diff(last, *full_bounded_ref);
    base.fmin_diff_full_bounded = std::abs(last.fmin - full_bounded_ref->fmin);
  }
  return {.row = base, .have_estimate = true, .estimate = std::move(last)};
}

template <class Fn>
TimedFitResult run_timed_estimate(Row base, int reps, Fn&& fn,
                                  const Estimates* profiled_bounded_ref,
                                  const Estimates* full_bounded_ref) {
  Estimates last;
  std::string error;
  const auto timed = measure_reps(reps, [&]() {
    return fn(last, error);
  });
  if (!timed.ok) {
    base.status = "error";
    base.error = error.empty() ? "fit failed" : error;
    return {.row = base};
  }
  base.fit_median_ms = timed.timing.median_ms;
  base.fit_min_ms = timed.timing.min_ms;
  base.fit_max_ms = timed.timing.max_ms;
  base.total_median_ms = timed.timing.median_ms;
  base.total_min_ms = timed.timing.min_ms;
  base.total_max_ms = timed.timing.max_ms;
  fill_estimate_fields(base, last);
  if (profiled_bounded_ref != nullptr) {
    base.theta_diff_profiled_bounded =
        theta_max_abs_diff(last, *profiled_bounded_ref);
    base.fmin_diff_profiled_bounded =
        std::abs(last.fmin - profiled_bounded_ref->fmin);
  }
  if (full_bounded_ref != nullptr) {
    base.theta_diff_full_bounded = theta_max_abs_diff(last, *full_bounded_ref);
    base.fmin_diff_full_bounded = std::abs(last.fmin - full_bounded_ref->fmin);
  }
  return {.row = base, .have_estimate = true, .estimate = std::move(last)};
}

struct DesignCell {
  std::string design;
  std::string case_id;
  int n = 0;
  int q = 0;
  int categories = 0;
  std::string threshold_balance;
  OrdinalThresholdMode threshold_mode = OrdinalThresholdMode::FreeIdentity;
  int seed = 0;
};

std::vector<DesignCell> make_design(bool smoke, int seed_base, int max_q) {
  std::vector<DesignCell> out;
  int offset = 0;
  const int capped_max_q = std::max(2, max_q);

  const int kreiberg_max_q = smoke ? std::min(4, capped_max_q) : capped_max_q;
  for (int q = 2; q <= kreiberg_max_q; ++q) {
    if (smoke && q == 3) continue;
    out.push_back({.design = "kreiberg_scaling",
                   .case_id = "kreiberg_q" + std::to_string(q),
                   .n = smoke ? 500 : 2000,
                   .q = q,
                   .categories = 5,
                   .threshold_balance = "symmetric",
                   .threshold_mode = OrdinalThresholdMode::FreeIdentity,
                   .seed = seed_base + offset});
    ++offset;
  }

  const int worked_q = smoke ? 4 : std::min(6, capped_max_q);
  out.push_back({.design = "worked_example",
                 .case_id = "ordinal_repeated_q" + std::to_string(worked_q),
                 .n = smoke ? 500 : 2000,
                 .q = worked_q,
                 .categories = 5,
                 .threshold_balance = "symmetric",
                 .threshold_mode = OrdinalThresholdMode::FreeIdentity,
                 .seed = seed_base + offset});
  ++offset;

  const std::vector<int> categories =
      smoke ? std::vector<int>{3, 5} : std::vector<int>{2, 3, 5, 7};
  const std::vector<std::string> balances =
      smoke ? std::vector<std::string>{"symmetric"}
            : std::vector<std::string>{"symmetric", "skewed"};
  const std::vector<OrdinalThresholdMode> modes =
      smoke ? std::vector<OrdinalThresholdMode>{
                  OrdinalThresholdMode::FreeIdentity,
                  OrdinalThresholdMode::LinearMap}
            : std::vector<OrdinalThresholdMode>{
                  OrdinalThresholdMode::FreeIdentity,
                  OrdinalThresholdMode::FixedOrConstrained,
                  OrdinalThresholdMode::LinearMap};
  for (const int cats : categories) {
    for (const auto& balance : balances) {
      for (const auto mode : modes) {
        out.push_back({.design = "threshold_stress",
                       .case_id = "stress_c" + std::to_string(cats) + "_" +
                                  balance + "_" + threshold_mode_name(mode),
                       .n = smoke ? 500 : 2000,
                       .q = smoke ? 3 : 4,
                       .categories = cats,
                       .threshold_balance = balance,
                       .threshold_mode = mode,
                       .seed = seed_base + offset});
        ++offset;
      }
    }
  }

  return out;
}

void print_help(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --out PATH       CSV output path (default: ordinal_snlls_speed.csv)\n"
      << "  --reps N         repeated timings per fit row (default: 5)\n"
      << "  --max-iter N     optimizer max_iter (default: 300)\n"
      << "  --seed-base N    deterministic seed base (default: 20260525)\n"
      << "  --max-q N        largest Kreiberg q in the scaling grid (default: 8)\n"
      << "  --smoke          run a small design\n"
      << "  --help           print this help\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "ordinal_snlls_speed.csv";
  int reps = 5;
  int max_iter = 300;
  int seed_base = 20260525;
  int max_q = 8;
  bool smoke = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0) {
      print_help(argv[0]);
      return 0;
    }
    if (std::strcmp(argv[i], "--smoke") == 0) {
      smoke = true;
      continue;
    }
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      ++i;
      out_path = argv[i];
      continue;
    }
    if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      ++i;
      reps = std::max(1, parse_int_arg(argv[i], reps));
      continue;
    }
    if (std::strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc) {
      ++i;
      max_iter = std::max(1, parse_int_arg(argv[i], max_iter));
      continue;
    }
    if (std::strcmp(argv[i], "--seed-base") == 0 && i + 1 < argc) {
      ++i;
      seed_base = parse_int_arg(argv[i], seed_base);
      continue;
    }
    if (std::strcmp(argv[i], "--max-q") == 0 && i + 1 < argc) {
      ++i;
      max_q = std::max(2, parse_int_arg(argv[i], max_q));
      continue;
    }
    std::cerr << "Unknown or incomplete option: " << argv[i] << "\n";
    print_help(argv[0]);
    return 2;
  }

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "Could not open output CSV: " << out_path << "\n";
    return 2;
  }
  write_header(out);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = max_iter;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  const auto design = make_design(smoke, seed_base, max_q);
  const std::vector<OrdinalEstimatorKind> estimators = {
      OrdinalEstimatorKind::ULS,
      OrdinalEstimatorKind::DWLS,
      OrdinalEstimatorKind::WLS};
  const std::vector<OrdinalMomentParameterization> parameterizations = {
      OrdinalMomentParameterization::Delta,
      OrdinalMomentParameterization::Theta};

  for (const auto& cell : design) {
    const int p = 4 * cell.q;
    const Eigen::MatrixXd data = simulate_repeated_ordinal(
        cell.n, cell.q, cell.categories, cell.threshold_balance, cell.seed);
    const std::string syntax =
        make_repeated_model_syntax(cell.q, cell.categories,
                                   cell.threshold_mode);

    magmaan::spec::LatentStructure pt;
    magmaan::model::MatrixRep rep;
    const auto model_timed = measure_reps(1, [&]() {
      auto flat = magmaan::parse::Parser::parse(syntax);
      if (!flat.has_value()) {
        std::cerr << "parse failed: " << flat.error().detail << "\n";
        return false;
      }
      auto built = magmaan::spec::build(*flat);
      if (!built.has_value()) {
        std::cerr << "lavaanify failed: " << built.error().detail << "\n";
        return false;
      }
      auto matrix_rep = magmaan::model::build_matrix_rep(*built);
      if (!matrix_rep.has_value()) {
        std::cerr << "matrix rep failed: " << matrix_rep.error().detail
                  << "\n";
        return false;
      }
      pt = *built;
      rep = *matrix_rep;
      return true;
    });
    if (!model_timed.ok) return 1;

    OrdinalStats stats;
    std::string stats_error;
    const auto stats_timed = measure_reps(1, [&]() {
      auto computed = magmaan::data::ordinal_stats_from_integer_data({data});
      if (!computed.has_value()) {
        stats_error = computed.error().detail;
        return false;
      }
      stats = *computed;
      return true;
    });
    if (!stats_timed.ok) {
      std::cerr << "ordinal stats failed: " << stats_error << "\n";
      return 1;
    }

    magmaan::data::OrdinalMoments moments;
    const auto moments_timed = measure_reps(1, [&]() {
      moments = magmaan::data::ordinal_moments_from_stats(stats);
      return true;
    });

    Eigen::VectorXd x0;
    std::string starts_error;
    const auto starts_timed = measure_reps(1, [&]() {
      auto starts =
          magmaan::estimate::ordinal_start_values(pt, rep, moments, {});
      if (!starts.has_value()) {
        starts_error = starts.error().detail;
        return false;
      }
      x0 = *starts;
      return true;
    });
    if (!starts_timed.ok) {
      std::cerr << "ordinal starts failed: " << starts_error << "\n";
      return 1;
    }

    Row base;
    base.design = cell.design;
    base.case_id = cell.case_id;
    base.seed = cell.seed;
    base.n = cell.n;
    base.q = cell.q;
    base.p = p;
    base.categories = cell.categories;
    base.threshold_balance = cell.threshold_balance;
    base.threshold_mode = threshold_mode_name(cell.threshold_mode);
    base.reps = reps;
    base.max_iter = max_iter;
    base.n_free = static_cast<int>(x0.size());
    base.n_thresholds =
        stats.thresholds.empty() ? 0
                                 : static_cast<int>(stats.thresholds[0].size());
    base.moment_dim = moment_dimension(p, cell.categories);
    base.model_setup_ms = model_timed.timing.median_ms;
    base.stats_ms = stats_timed.timing.median_ms;
    base.moments_ms = moments_timed.timing.median_ms;
    base.starts_ms = starts_timed.timing.median_ms;

    for (const auto estimator : estimators) {
      for (const auto parameterization : parameterizations) {
        const auto plan = magmaan::data::ordinal_weight_plan(
            OrdinalWorkspacePurpose::FitOnly, estimator, parameterization,
            plan_threshold_mode(cell.threshold_mode));
        const auto bounded_parameterization =
            fit_parameterization(parameterization);
        Row parameterized_base = base;
        parameterized_base.parameterization =
            parameterization_name(parameterization);

        Estimates full_bounded_ref;
        bool have_full_bounded = false;
        if (estimator != OrdinalEstimatorKind::ULS) {
          const auto weight_kind =
              estimator == OrdinalEstimatorKind::DWLS
                  ? magmaan::estimate::OrdinalWeightKind::DWLS
                  : magmaan::estimate::OrdinalWeightKind::WLS;
          Row row = parameterized_base;
          row.estimator = estimator_name(estimator);
          row.path = "full_bounded";
          auto result = run_timed_fit(
              row, reps,
              [&]() {
                return magmaan::estimate::fit_ordinal_bounded(
                    pt, rep, stats, {}, weight_kind, x0, Backend::NloptLbfgs,
                    opts, bounded_parameterization);
              },
              nullptr, nullptr);
          row = result.row;
          if (result.have_estimate) {
            have_full_bounded = true;
            full_bounded_ref = std::move(result.estimate);
            row.theta_diff_full_bounded = 0.0;
            row.fmin_diff_full_bounded = 0.0;
          }
          write_row(out, row);
        }

        Estimates profiled_bounded_ref;
        bool have_profiled_bounded = false;

        {
          auto cache = make_fit_cache(stats, estimator);
          Row row = parameterized_base;
          row.estimator = estimator_name(estimator);
          row.path = "threshold_profiled_bounded";
          row.cache_setup_ms = cache.cache_setup.median_ms;
          row.weight_setup_ms = cache.weight_setup.median_ms;
          if (!cache.error.empty()) {
            row.status = "error";
            row.error = cache.error;
          } else {
            auto result = run_timed_fit(
                row, reps,
                [&]() {
                  return magmaan::estimate::fit_ordinal_bounded(
                      pt, rep, moments, cache.ptr(), {}, plan, x0,
                      Backend::NloptLbfgs, opts);
                },
                nullptr, have_full_bounded ? &full_bounded_ref : nullptr);
            row = result.row;
            const auto flags = cache_flags(cache.ptr());
            row.cache_blocks = flags.block_count;
            row.cache_has_diagonal = flags.has_diagonal;
            row.cache_has_full = flags.has_full;
            row.cache_has_dwls_weight = flags.has_dwls_weight;
            row.cache_has_wls_weight = flags.has_wls_weight;
            if (result.have_estimate) {
              have_profiled_bounded = true;
              profiled_bounded_ref = std::move(result.estimate);
              row.theta_diff_profiled_bounded = 0.0;
              row.fmin_diff_profiled_bounded = 0.0;
            }
          }
          write_row(out, row);
        }

        {
          auto cache = make_fit_cache(stats, estimator);
          Row row = parameterized_base;
          row.estimator = estimator_name(estimator);
          row.path = "snlls_full_thresholds";
          row.cache_setup_ms = cache.cache_setup.median_ms;
          row.weight_setup_ms = cache.weight_setup.median_ms;
          if (!cache.error.empty()) {
            row.status = "error";
            row.error = cache.error;
          } else {
            auto result = run_timed_fit(
                row, reps,
                [&]() {
                  return magmaan::estimate::fit_ordinal_snlls_full_thresholds(
                      pt, rep, moments, cache.ptr(), plan, x0,
                      Backend::NloptLbfgs, opts);
                },
                have_profiled_bounded ? &profiled_bounded_ref : nullptr,
                have_full_bounded ? &full_bounded_ref : nullptr);
            row = result.row;
            const auto flags = cache_flags(cache.ptr());
            row.cache_blocks = flags.block_count;
            row.cache_has_diagonal = flags.has_diagonal;
            row.cache_has_full = flags.has_full;
            row.cache_has_dwls_weight = flags.has_dwls_weight;
            row.cache_has_wls_weight = flags.has_wls_weight;
          }
          write_row(out, row);
        }

        {
          auto cache = make_fit_cache(stats, estimator);
          Row row = parameterized_base;
          row.estimator = estimator_name(estimator);
          row.path = "snlls";
          row.cache_setup_ms = cache.cache_setup.median_ms;
          row.weight_setup_ms = cache.weight_setup.median_ms;
          if (!cache.error.empty()) {
            row.status = "error";
            row.error = cache.error;
          } else {
            auto result = run_timed_fit(
                row, reps,
                [&]() {
                  return magmaan::estimate::fit_ordinal_snlls(
                      pt, rep, moments, cache.ptr(), plan, x0,
                      Backend::NloptLbfgs, opts);
                },
                have_profiled_bounded ? &profiled_bounded_ref : nullptr,
                have_full_bounded ? &full_bounded_ref : nullptr);
            row = result.row;
            const auto flags = cache_flags(cache.ptr());
            row.cache_blocks = flags.block_count;
            row.cache_has_diagonal = flags.has_diagonal;
            row.cache_has_full = flags.has_full;
            row.cache_has_dwls_weight = flags.has_dwls_weight;
            row.cache_has_wls_weight = flags.has_wls_weight;
          }
          write_row(out, row);
        }

        {
          Row row = parameterized_base;
          row.estimator = estimator_name(estimator);
          row.path = "snlls_e2e_legacy";
          row.construction = "legacy_stats";
          CacheFlags last_flags;
          auto result = run_timed_estimate(
              row, reps,
              [&](Estimates &last, std::string &error) {
                auto computed =
                    magmaan::data::ordinal_stats_from_integer_data({data});
                if (!computed.has_value()) {
                  error = computed.error().detail;
                  return false;
                }
                auto local_moments =
                    magmaan::data::ordinal_moments_from_stats(*computed);
                auto starts = magmaan::estimate::ordinal_start_values(
                    pt, rep, local_moments, {});
                if (!starts.has_value()) {
                  error = starts.error().detail;
                  return false;
                }
                auto cache = make_fit_cache_unmeasured(*computed, estimator);
                if (!cache.error.empty()) {
                  error = cache.error;
                  return false;
                }
                last_flags = cache_flags(cache.ptr());
                auto fit = magmaan::estimate::fit_ordinal_snlls(
                    pt, rep, local_moments, cache.ptr(), plan, *starts,
                    Backend::NloptLbfgs, opts);
                if (!fit.has_value()) {
                  error = fit.error().detail;
                  return false;
                }
                last = *fit;
                return true;
              },
              have_profiled_bounded ? &profiled_bounded_ref : nullptr,
              have_full_bounded ? &full_bounded_ref : nullptr);
          row = result.row;
          row.cache_blocks = last_flags.block_count;
          row.cache_has_diagonal = last_flags.has_diagonal;
          row.cache_has_full = last_flags.has_full;
          row.cache_has_dwls_weight = last_flags.has_dwls_weight;
          row.cache_has_wls_weight = last_flags.has_wls_weight;
          write_row(out, row);
        }

        if (estimator == OrdinalEstimatorKind::ULS ||
            estimator == OrdinalEstimatorKind::DWLS) {
          Row row = parameterized_base;
          row.estimator = estimator_name(estimator);
          row.path = "snlls_e2e_lazy";
          row.construction = "lazy_workspace";
          row.stats_ms = quiet_nan();
          row.moments_ms = quiet_nan();
          row.starts_ms = quiet_nan();
          row.cache_setup_ms = quiet_nan();
          row.weight_setup_ms = quiet_nan();
          CacheFlags last_flags;
          auto result = run_timed_estimate(
              row, reps,
              [&](Estimates &last, std::string &error) {
                auto workspace =
                    magmaan::data::ordinal_workspace_from_integer_data({data},
                                                                       plan);
                if (!workspace.has_value()) {
                  error = workspace.error().detail;
                  return false;
                }
                auto starts = magmaan::estimate::ordinal_start_values(
                    pt, rep, workspace->moments, {});
                if (!starts.has_value()) {
                  error = starts.error().detail;
                  return false;
                }
                auto *cache_ptr = workspace->gamma_cache.block_count() == 0
                                      ? nullptr
                                      : &workspace->gamma_cache;
                last_flags = cache_flags(cache_ptr);
                auto fit = magmaan::estimate::fit_ordinal_snlls(
                    pt, rep, workspace->moments, cache_ptr, plan, *starts,
                    Backend::NloptLbfgs, opts);
                if (!fit.has_value()) {
                  error = fit.error().detail;
                  return false;
                }
                last = *fit;
                return true;
              },
              have_profiled_bounded ? &profiled_bounded_ref : nullptr,
              have_full_bounded ? &full_bounded_ref : nullptr);
          row = result.row;
          row.cache_blocks = last_flags.block_count;
          row.cache_has_diagonal = last_flags.has_diagonal;
          row.cache_has_full = last_flags.has_full;
          row.cache_has_dwls_weight = last_flags.has_dwls_weight;
          row.cache_has_wls_weight = last_flags.has_wls_weight;
          write_row(out, row);
        }
      }
    }
  }

  std::cout << "Wrote " << out_path << "\n";
  return 0;
}
