#include <algorithm>
#include <chrono>
#include <cmath>
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

template <class Fn> TimedStatus measure_reps(int reps, Fn &&fn) {
  std::vector<double> times;
  times.reserve(static_cast<std::size_t>(std::max(1, reps)));
  for (int rep = 0; rep < std::max(1, reps); ++rep) {
    const auto start = std::chrono::steady_clock::now();
    const bool ok = fn();
    const auto stop = std::chrono::steady_clock::now();
    if (!ok)
      return {.ok = false, .timing = {}};
    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    times.push_back(elapsed.count());
  }
  std::sort(times.begin(), times.end());
  const std::size_t mid = times.size() / 2U;
  const double median = (times.size() % 2U == 0U)
                            ? 0.5 * (times[mid - 1U] + times[mid])
                            : times[mid];
  return {.ok = true,
          .timing = {.median_ms = median,
                     .min_ms = times.front(),
                     .max_ms = times.back()}};
}

int parse_int_arg(const char *arg, int fallback) {
  if (arg == nullptr)
    return fallback;
  char *end = nullptr;
  const long value = std::strtol(arg, &end, 10);
  if (end == arg || *end != '\0')
    return fallback;
  if (value < 0L ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(value);
}

std::string csv_escape(const std::string &value) {
  bool needs_quotes = false;
  for (const char ch : value) {
    if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes)
    return value;
  std::string out = "\"";
  for (const char ch : value) {
    if (ch == '"')
      out += '"';
    out += ch;
  }
  out += '"';
  return out;
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
  case OrdinalEstimatorKind::DWLS:
    return "DWLS";
  case OrdinalEstimatorKind::WLS:
    return "WLS";
  case OrdinalEstimatorKind::ULS:
    return "ULS";
  }
  return "unknown";
}

Eigen::MatrixXd simulate_ordinal_cfa(int n, int p, int seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd x(n, p);
  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < x.cols(); ++j) {
      const double loading = 0.86 - 0.05 * static_cast<double>(j);
      const double sd = std::sqrt(std::max(0.05, 1.0 - loading * loading));
      const double shift = (j % 2 == 0) ? -0.10 : 0.12;
      const double y = shift + loading * eta + sd * norm(rng);
      x(i, j) = 1.0 + (y > -0.55) + (y > 0.45);
    }
  }
  return x;
}

std::string make_model_syntax(int p, const std::string &constraint_kind) {
  std::ostringstream syntax;
  syntax << "f =~ ";
  for (int j = 1; j <= p; ++j) {
    if (j > 1)
      syntax << " + ";
    syntax << "x" << j;
  }
  syntax << "\n";

  if (constraint_kind == "shared_label") {
    syntax << "x1 | a*t1 + t2\n";
    syntax << "x2 | a*t1 + t2\n";
    for (int j = 3; j <= p; ++j) {
      syntax << "x" << j << " | t1 + t2\n";
    }
  } else if (constraint_kind == "linear_symmetry_one") {
    syntax << "x1 | a*t1 + b*t2\n";
    for (int j = 2; j <= p; ++j) {
      syntax << "x" << j << " | t1 + t2\n";
    }
    syntax << "a + b == 0\n";
  } else if (constraint_kind == "linear_symmetry_all") {
    for (int j = 1; j <= p; ++j) {
      syntax << "x" << j << " | a" << j << "*t1 + b" << j << "*t2\n";
    }
    for (int j = 1; j <= p; ++j) {
      syntax << "a" << j << " + b" << j << " == 0\n";
    }
  } else {
    for (int j = 1; j <= p; ++j) {
      syntax << "x" << j << " | t1 + t2\n";
    }
  }

  for (int j = 1; j <= p; ++j) {
    syntax << "x" << j << " ~*~ 1*x" << j << "\n";
  }
  return syntax.str();
}

struct CacheBundle {
  OrdinalGammaCache cache;
  OrdinalGammaCache *cache_ptr = nullptr;
  Timing cache_setup;
  Timing weight_setup;
  std::string error;
};

CacheBundle make_fit_cache(const OrdinalStats &stats,
                           OrdinalEstimatorKind estimator) {
  CacheBundle bundle;
  if (estimator == OrdinalEstimatorKind::DWLS) {
    const auto timed = measure_reps(1, [&]() {
      bundle.cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
          {stats.NACOV[0].diagonal()});
      return true;
    });
    bundle.cache_setup = timed.timing;
    bundle.cache_ptr = &bundle.cache;
    return bundle;
  }

  const auto cache_timed = measure_reps(1, [&]() {
    bundle.cache.blocks.resize(1);
    bundle.cache.blocks[0].gamma = stats.NACOV[0];
    bundle.cache.blocks[0].has_full = true;
    return true;
  });
  bundle.cache_setup = cache_timed.timing;
  bundle.cache_ptr = &bundle.cache;
  const auto weight_timed = measure_reps(1, [&]() {
    auto ok =
        magmaan::data::ordinal_gamma_cache_ensure_wls_weights(bundle.cache);
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

double theta_max_abs_diff(const Estimates &lhs, const Estimates &rhs) {
  if (lhs.theta.size() != rhs.theta.size())
    return quiet_nan();
  if (lhs.theta.size() == 0)
    return 0.0;
  return (lhs.theta - rhs.theta).cwiseAbs().maxCoeff();
}

struct Row {
  std::string case_id;
  std::string constraint_kind;
  int seed = 0;
  int n = 0;
  int p = 0;
  int reps = 0;
  int max_iter = 0;
  int n_free = 0;
  std::string estimator;
  std::string path;
  std::string status = "ok";
  std::string error;
  double fit_median_ms = quiet_nan();
  double fit_min_ms = quiet_nan();
  double fit_max_ms = quiet_nan();
  double fmin = quiet_nan();
  int iterations = -1;
  int f_evals = -1;
  int g_evals = -1;
  std::string optimizer_status;
  double grad_inf_norm = quiet_nan();
  int n_nonlinear = -1;
  int n_linear = -1;
  double cache_setup_ms = quiet_nan();
  double weight_setup_ms = quiet_nan();
  double theta_diff_full_bounded = quiet_nan();
  double fmin_diff_full_bounded = quiet_nan();
  double theta_diff_profiled_bounded = quiet_nan();
  double fmin_diff_profiled_bounded = quiet_nan();
};

void write_header(std::ostream &out) {
  out << "case_id,constraint_kind,seed,n,p,reps,max_iter,n_free,estimator,path,"
         "status,error,fit_median_ms,fit_min_ms,fit_max_ms,fmin,iterations,"
         "f_evals,g_evals,optimizer_status,grad_inf_norm,n_nonlinear,n_linear,"
         "cache_setup_ms,weight_setup_ms,theta_diff_full_bounded,"
         "fmin_diff_full_bounded,theta_diff_profiled_bounded,"
         "fmin_diff_profiled_bounded\n";
}

void write_row(std::ostream &out, const Row &row) {
  out << csv_escape(row.case_id) << ',' << csv_escape(row.constraint_kind)
      << ',' << row.seed << ',' << row.n << ',' << row.p << ',' << row.reps
      << ',' << row.max_iter << ',' << row.n_free << ','
      << csv_escape(row.estimator) << ',' << csv_escape(row.path) << ','
      << csv_escape(row.status) << ',' << csv_escape(row.error) << ','
      << row.fit_median_ms << ',' << row.fit_min_ms << ',' << row.fit_max_ms
      << ',' << row.fmin << ',' << row.iterations << ',' << row.f_evals << ','
      << row.g_evals << ',' << csv_escape(row.optimizer_status) << ','
      << row.grad_inf_norm << ',' << row.n_nonlinear << ',' << row.n_linear
      << ',' << row.cache_setup_ms << ',' << row.weight_setup_ms << ','
      << row.theta_diff_full_bounded << ',' << row.fmin_diff_full_bounded << ','
      << row.theta_diff_profiled_bounded << ','
      << row.fmin_diff_profiled_bounded << '\n';
}

void fill_estimate_fields(Row &row, const Estimates &est) {
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
TimedFitResult run_timed_fit(Row base, int reps, FitFn &&fit_fn,
                             const Estimates *full_ref,
                             const Estimates *profiled_ref) {
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
  if (full_ref != nullptr) {
    base.theta_diff_full_bounded = theta_max_abs_diff(last, *full_ref);
    base.fmin_diff_full_bounded = std::abs(last.fmin - full_ref->fmin);
  }
  if (profiled_ref != nullptr) {
    base.theta_diff_profiled_bounded = theta_max_abs_diff(last, *profiled_ref);
    base.fmin_diff_profiled_bounded = std::abs(last.fmin - profiled_ref->fmin);
  }
  return {.row = base, .have_estimate = true, .estimate = std::move(last)};
}

struct DesignCell {
  std::string case_id;
  std::string constraint_kind;
  int n = 0;
  int p = 0;
  int seed = 0;
};

std::vector<DesignCell> make_design(bool smoke, int seed_base) {
  const int n = smoke ? 420 : 900;
  const int p = smoke ? 4 : 6;
  return {
      {.case_id = "free_thresholds",
       .constraint_kind = "free",
       .n = n,
       .p = p,
       .seed = seed_base},
      {.case_id = "shared_first_threshold",
       .constraint_kind = "shared_label",
       .n = n,
       .p = p,
       .seed = seed_base + 1},
      {.case_id = "one_item_symmetric_thresholds",
       .constraint_kind = "linear_symmetry_one",
       .n = n,
       .p = p,
       .seed = seed_base + 2},
      {.case_id = "all_items_symmetric_thresholds",
       .constraint_kind = "linear_symmetry_all",
       .n = n,
       .p = p,
       .seed = seed_base + 3},
  };
}

void print_help(const char *argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --out PATH       CSV output path (default: "
         "ordinal_threshold_constraints.csv)\n"
      << "  --reps N         repeated timings per fit row (default: 3)\n"
      << "  --max-iter N     optimizer max_iter (default: 300)\n"
      << "  --seed-base N    deterministic seed base (default: 20260601)\n"
      << "  --smoke          run a small design\n"
      << "  --help           print this help\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string out_path = "ordinal_threshold_constraints.csv";
  int reps = 3;
  int max_iter = 300;
  int seed_base = 20260601;
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

  const auto design = make_design(smoke, seed_base);
  const std::vector<OrdinalEstimatorKind> estimators = {
      OrdinalEstimatorKind::DWLS, OrdinalEstimatorKind::WLS};

  for (const auto &cell : design) {
    const Eigen::MatrixXd data =
        simulate_ordinal_cfa(cell.n, cell.p, cell.seed);
    const std::string syntax = make_model_syntax(cell.p, cell.constraint_kind);

    magmaan::spec::LatentStructure pt;
    magmaan::model::MatrixRep rep;
    auto flat = magmaan::parse::Parser::parse(syntax);
    if (!flat.has_value()) {
      std::cerr << "parse failed: " << flat.error().detail << "\n";
      return 1;
    }
    auto built = magmaan::spec::build(*flat);
    if (!built.has_value()) {
      std::cerr << "lavaanify failed: " << built.error().detail << "\n";
      return 1;
    }
    auto matrix_rep = magmaan::model::build_matrix_rep(*built);
    if (!matrix_rep.has_value()) {
      std::cerr << "matrix rep failed: " << matrix_rep.error().detail << "\n";
      return 1;
    }
    pt = *built;
    rep = *matrix_rep;

    auto stats_or = magmaan::data::ordinal_stats_from_integer_data({data});
    if (!stats_or.has_value()) {
      std::cerr << "ordinal stats failed: " << stats_or.error().detail << "\n";
      return 1;
    }
    OrdinalStats stats = *stats_or;
    auto moments = magmaan::data::ordinal_moments_from_stats(stats);

    auto starts_or =
        magmaan::estimate::ordinal_start_values(pt, rep, moments, {});
    if (!starts_or.has_value()) {
      std::cerr << "ordinal starts failed: " << starts_or.error().detail
                << "\n";
      return 1;
    }
    const Eigen::VectorXd x0 = *starts_or;

    Row base;
    base.case_id = cell.case_id;
    base.constraint_kind = cell.constraint_kind;
    base.seed = cell.seed;
    base.n = cell.n;
    base.p = cell.p;
    base.reps = reps;
    base.max_iter = max_iter;
    base.n_free = static_cast<int>(x0.size());

    for (const auto estimator : estimators) {
      const auto weight_kind = estimator == OrdinalEstimatorKind::DWLS
                                   ? magmaan::estimate::OrdinalWeightKind::DWLS
                                   : magmaan::estimate::OrdinalWeightKind::WLS;
      const auto plan = magmaan::data::ordinal_weight_plan(
          OrdinalWorkspacePurpose::FitOnly, estimator,
          OrdinalMomentParameterization::Delta,
          OrdinalThresholdMode::FixedOrConstrained);

      Estimates full_ref;
      bool have_full = false;
      {
        Row row = base;
        row.estimator = estimator_name(estimator);
        row.path = "full_bounded";
        auto result = run_timed_fit(
            row, reps,
            [&]() {
              return magmaan::estimate::fit_ordinal_bounded(
                  pt, rep, stats, {}, weight_kind, x0, Backend::NloptLbfgs,
                  opts);
            },
            nullptr, nullptr);
        row = result.row;
        if (result.have_estimate) {
          have_full = true;
          full_ref = std::move(result.estimate);
          row.theta_diff_full_bounded = 0.0;
          row.fmin_diff_full_bounded = 0.0;
        }
        write_row(out, row);
      }

      Estimates profiled_ref;
      bool have_profiled = false;
      {
        auto cache = make_fit_cache(stats, estimator);
        Row row = base;
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
                    pt, rep, moments, cache.cache_ptr, {}, plan, x0,
                    Backend::NloptLbfgs, opts);
              },
              have_full ? &full_ref : nullptr, nullptr);
          row = result.row;
          if (result.have_estimate) {
            have_profiled = true;
            profiled_ref = std::move(result.estimate);
            row.theta_diff_profiled_bounded = 0.0;
            row.fmin_diff_profiled_bounded = 0.0;
          }
        }
        write_row(out, row);
      }

      {
        auto cache = make_fit_cache(stats, estimator);
        Row row = base;
        row.estimator = estimator_name(estimator);
        row.path = "full_threshold_snlls";
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
                    pt, rep, moments, cache.cache_ptr, plan, x0,
                    Backend::NloptLbfgs, opts);
              },
              have_full ? &full_ref : nullptr,
              have_profiled ? &profiled_ref : nullptr);
          row = result.row;
        }
        write_row(out, row);
      }

      {
        auto cache = make_fit_cache(stats, estimator);
        Row row = base;
        row.estimator = estimator_name(estimator);
        row.path = "threshold_profiled_snlls";
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
                    pt, rep, moments, cache.cache_ptr, plan, x0,
                    Backend::NloptLbfgs, opts);
              },
              have_full ? &full_ref : nullptr,
              have_profiled ? &profiled_ref : nullptr);
          row = result.row;
        }
        write_row(out, row);
      }
    }
  }

  std::cout << "Wrote " << out_path << "\n";
  return 0;
}
