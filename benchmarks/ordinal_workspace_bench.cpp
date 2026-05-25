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

#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"
#include "magmaan/spec/start_hints.hpp"

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

template <class Fn>
TimedStatus measure_reps(int reps, Fn&& fn) {
  std::vector<double> times;
  times.reserve(static_cast<std::size_t>(std::max(1, reps)));
  for (int rep = 0; rep < std::max(1, reps); ++rep) {
    const auto start = std::chrono::steady_clock::now();
    const bool ok = fn();
    const auto stop = std::chrono::steady_clock::now();
    if (!ok) {
      return {.ok = false, .timing = {}};
    }
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
  if (value < 0L) return fallback;
  if (value > static_cast<long>(std::numeric_limits<int>::max())) {
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

std::string threshold_mode_name(OrdinalThresholdMode mode) {
  switch (mode) {
    case OrdinalThresholdMode::FreeIdentity:
      return "free";
    case OrdinalThresholdMode::LinearMap:
      return "linear_map";
    case OrdinalThresholdMode::FixedOrConstrained:
      return "fixed_or_shared";
  }
  return "unknown";
}

std::string make_model_syntax(int p, int categories,
                              OrdinalThresholdMode threshold_mode) {
  std::ostringstream syntax;
  syntax << "f =~ ";
  for (int j = 1; j <= p; ++j) {
    if (j > 1) syntax << " + ";
    syntax << "x" << j;
  }
  syntax << "\n";

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

std::vector<double> thresholds_for_categories(int categories) {
  if (categories <= 3) return {-0.55, 0.55};
  return {-1.20, -0.35, 0.35, 1.20};
}

Eigen::MatrixXd simulate_ordinal_data(int n, int p, int categories, int seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<double> norm(0.0, 1.0);
  const auto thresholds = thresholds_for_categories(categories);
  Eigen::MatrixXd x(n, p);
  const double denom = static_cast<double>(std::max(1, p - 1));
  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < x.cols(); ++j) {
      const double position = static_cast<double>(j) / denom;
      const double loading = 0.88 - 0.24 * position;
      const double shift = 0.16 * std::sin(1.7 * static_cast<double>(j));
      const double resid_sd =
          std::sqrt(std::max(1e-8, 1.0 - loading * loading));
      const double y = shift + loading * eta + resid_sd * norm(rng);
      int category = 1;
      for (const double threshold : thresholds) {
        if (y > threshold) ++category;
      }
      x(i, j) = static_cast<double>(category);
    }
  }
  return x;
}

double moment_dimension(int p, int categories) {
  const int thresholds = p * (categories - 1);
  const int correlations = (p * (p - 1)) / 2;
  return static_cast<double>(thresholds + correlations);
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
  OrdinalGammaCache* cache_ptr = nullptr;
  Timing cache_setup;
  Timing weight_setup;
  std::string error;
};

CacheBundle make_fit_only_cache(const OrdinalStats& stats,
                                OrdinalEstimatorKind estimator) {
  CacheBundle bundle;
  if (estimator == OrdinalEstimatorKind::ULS) {
    return bundle;
  }
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
  int seed = 0;
  int n = 0;
  int p = 0;
  int categories = 0;
  int reps = 0;
  int max_iter = 0;
  int n_free = 0;
  int n_thresholds = 0;
  int moment_dim = 0;
  std::string parameterization = "delta";
  std::string purpose = "fit_only";
  std::string threshold_mode;
  std::string estimator;
  std::string path;
  std::string fit_status = "ok";
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
  double theta_diff_bounded = quiet_nan();
  double fmin_diff_bounded = quiet_nan();
  double theta_diff_legacy = quiet_nan();
  double fmin_diff_legacy = quiet_nan();
};

void write_header(std::ostream& out) {
  out << "seed,n,p,categories,reps,max_iter,n_free,n_thresholds,moment_dim,"
         "parameterization,purpose,threshold_mode,estimator,path,fit_status,"
         "error,model_setup_ms,stats_ms,moments_ms,starts_ms,cache_setup_ms,"
         "weight_setup_ms,fit_median_ms,fit_min_ms,fit_max_ms,fmin,"
         "iterations,f_evals,g_evals,optimizer_status,grad_inf_norm,"
         "n_nonlinear,n_linear,cache_blocks,cache_has_diagonal,"
         "cache_has_full,cache_has_dwls_weight,cache_has_wls_weight,"
         "theta_diff_bounded,fmin_diff_bounded,theta_diff_legacy,"
         "fmin_diff_legacy\n";
}

void write_row(std::ostream& out, const Row& row) {
  out << row.seed << ',' << row.n << ',' << row.p << ',' << row.categories
      << ',' << row.reps << ',' << row.max_iter << ',' << row.n_free << ','
      << row.n_thresholds << ',' << row.moment_dim << ','
      << csv_escape(row.parameterization) << ',' << csv_escape(row.purpose)
      << ',' << csv_escape(row.threshold_mode) << ','
      << csv_escape(row.estimator) << ',' << csv_escape(row.path) << ','
      << csv_escape(row.fit_status) << ',' << csv_escape(row.error) << ','
      << row.model_setup_ms << ',' << row.stats_ms << ',' << row.moments_ms
      << ',' << row.starts_ms << ',' << row.cache_setup_ms << ','
      << row.weight_setup_ms << ',' << row.fit_median_ms << ','
      << row.fit_min_ms << ',' << row.fit_max_ms << ',' << row.fmin << ','
      << row.iterations << ',' << row.f_evals << ',' << row.g_evals << ','
      << csv_escape(row.optimizer_status) << ',' << row.grad_inf_norm << ','
      << row.n_nonlinear << ',' << row.n_linear << ',' << row.cache_blocks
      << ',' << bool_csv(row.cache_has_diagonal) << ','
      << bool_csv(row.cache_has_full) << ','
      << bool_csv(row.cache_has_dwls_weight) << ','
      << bool_csv(row.cache_has_wls_weight) << ','
      << row.theta_diff_bounded << ',' << row.fmin_diff_bounded << ','
      << row.theta_diff_legacy << ',' << row.fmin_diff_legacy << '\n';
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

std::string path_for_legacy(OrdinalEstimatorKind estimator) {
  if (estimator == OrdinalEstimatorKind::DWLS) return "materialized_bounded";
  if (estimator == OrdinalEstimatorKind::WLS) return "materialized_bounded";
  return "";
}

struct FitRefs {
  bool have_bounded = false;
  bool have_legacy = false;
  Estimates bounded;
  Estimates legacy;
};

struct TimedFitResult {
  Row row;
  bool have_estimate = false;
  Estimates estimate;
};

template <class FitFn>
TimedFitResult run_timed_fit(Row base, int reps, FitFn&& fit_fn,
                             const Estimates* bounded_ref,
                             const Estimates* legacy_ref) {
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
    base.fit_status = "error";
    base.error = error.empty() ? "fit failed" : error;
    return {.row = base};
  }
  base.fit_median_ms = timed.timing.median_ms;
  base.fit_min_ms = timed.timing.min_ms;
  base.fit_max_ms = timed.timing.max_ms;
  fill_estimate_fields(base, last);
  if (bounded_ref != nullptr) {
    base.theta_diff_bounded = theta_max_abs_diff(last, *bounded_ref);
    base.fmin_diff_bounded = std::abs(last.fmin - bounded_ref->fmin);
  }
  if (legacy_ref != nullptr) {
    base.theta_diff_legacy = theta_max_abs_diff(last, *legacy_ref);
    base.fmin_diff_legacy = std::abs(last.fmin - legacy_ref->fmin);
  }
  return {.row = base, .have_estimate = true, .estimate = std::move(last)};
}

struct DesignCell {
  int n = 0;
  int p = 0;
  int categories = 0;
  OrdinalThresholdMode threshold_mode = OrdinalThresholdMode::FreeIdentity;
  int seed = 0;
};

std::vector<DesignCell> make_design(bool smoke, int seed_base) {
  std::vector<DesignCell> out;
  const std::vector<int> ps = smoke ? std::vector<int>{4}
                                    : std::vector<int>{4, 8, 12};
  const std::vector<int> ns = smoke ? std::vector<int>{400}
                                    : std::vector<int>{500, 1200};
  const std::vector<int> categories = smoke ? std::vector<int>{3}
                                            : std::vector<int>{3, 5};
  const std::vector<OrdinalThresholdMode> modes =
      smoke ? std::vector<OrdinalThresholdMode>{
                  OrdinalThresholdMode::FreeIdentity,
                  OrdinalThresholdMode::FixedOrConstrained,
                  OrdinalThresholdMode::LinearMap}
            : std::vector<OrdinalThresholdMode>{
                  OrdinalThresholdMode::FreeIdentity,
                  OrdinalThresholdMode::FixedOrConstrained,
                  OrdinalThresholdMode::LinearMap};
  int offset = 0;
  for (const int n : ns) {
    for (const int p : ps) {
      for (const int category_count : categories) {
        for (const auto mode : modes) {
          out.push_back({.n = n,
                         .p = p,
                         .categories = category_count,
                         .threshold_mode = mode,
                         .seed = seed_base + offset});
          ++offset;
        }
      }
    }
  }
  return out;
}

void print_help(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --out PATH       CSV output path (default: ordinal_workspace_bench.csv)\n"
      << "  --reps N         repeated timings per fit row (default: 3)\n"
      << "  --max-iter N     optimizer max_iter (default: 300)\n"
      << "  --seed-base N    deterministic seed base (default: 20260525)\n"
      << "  --smoke          run a small three-cell design\n"
      << "  --help           print this help\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "ordinal_workspace_bench.csv";
  int reps = 3;
  int max_iter = 300;
  int seed_base = 20260525;
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
      OrdinalEstimatorKind::ULS,
      OrdinalEstimatorKind::DWLS,
      OrdinalEstimatorKind::WLS};

  for (const auto& cell : design) {
    const Eigen::MatrixXd data =
        simulate_ordinal_data(cell.n, cell.p, cell.categories, cell.seed);
    const std::string syntax =
        make_model_syntax(cell.p, cell.categories, cell.threshold_mode);

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

    const int n_thresholds =
        stats.thresholds.empty() ? 0
                                 : static_cast<int>(stats.thresholds[0].size());
    const int n_moments =
        static_cast<int>(moment_dimension(cell.p, cell.categories));
    Row base;
    base.seed = cell.seed;
    base.n = cell.n;
    base.p = cell.p;
    base.categories = cell.categories;
    base.reps = reps;
    base.max_iter = max_iter;
    base.n_free = static_cast<int>(x0.size());
    base.n_thresholds = n_thresholds;
    base.moment_dim = n_moments;
    base.threshold_mode = threshold_mode_name(cell.threshold_mode);
    base.model_setup_ms = model_timed.timing.median_ms;
    base.stats_ms = stats_timed.timing.median_ms;
    base.moments_ms = moments_timed.timing.median_ms;
    base.starts_ms = starts_timed.timing.median_ms;

    for (const auto estimator : estimators) {
      const std::string est_name = estimator_name(estimator);
      const auto plan = magmaan::data::ordinal_weight_plan(
          OrdinalWorkspacePurpose::FitOnly, estimator,
          OrdinalMomentParameterization::Delta,
          cell.threshold_mode == OrdinalThresholdMode::FreeIdentity
              ? OrdinalThresholdMode::FreeIdentity
              : OrdinalThresholdMode::FixedOrConstrained);

      FitRefs refs;
      const std::string legacy_path = path_for_legacy(estimator);
      if (!legacy_path.empty()) {
        Row row = base;
        row.estimator = est_name;
        row.path = legacy_path;
        TimedFitResult result;
        if (estimator == OrdinalEstimatorKind::DWLS) {
          result = run_timed_fit(
              row, reps,
              [&]() {
                return magmaan::estimate::fit_ordinal_bounded(
                    pt, rep, stats, {},
                    magmaan::estimate::OrdinalWeightKind::DWLS, x0,
                    Backend::NloptLbfgs, opts);
              },
              nullptr, nullptr);
        } else {
          result = run_timed_fit(
              row, reps,
              [&]() {
                return magmaan::estimate::fit_ordinal_bounded(
                    pt, rep, stats, {},
                    magmaan::estimate::OrdinalWeightKind::WLS, x0,
                    Backend::NloptLbfgs, opts);
              },
              nullptr, nullptr);
        }
        row = result.row;
        if (result.have_estimate) {
          refs.have_legacy = true;
          refs.legacy = std::move(result.estimate);
        }
        write_row(out, row);
      }

      {
        auto cache = make_fit_only_cache(stats, estimator);
        Row row = base;
        row.estimator = est_name;
        row.path = "cache_bounded";
        row.cache_setup_ms = cache.cache_setup.median_ms;
        row.weight_setup_ms = cache.weight_setup.median_ms;
        const auto flags = cache_flags(cache.cache_ptr);
        row.cache_blocks = flags.block_count;
        row.cache_has_diagonal = flags.has_diagonal;
        row.cache_has_full = flags.has_full;
        row.cache_has_dwls_weight = flags.has_dwls_weight;
        row.cache_has_wls_weight = flags.has_wls_weight;
        if (!cache.error.empty()) {
          row.fit_status = "error";
          row.error = cache.error;
        } else {
          auto result = run_timed_fit(
              row, reps,
              [&]() {
                return magmaan::estimate::fit_ordinal_bounded(
                    pt, rep, moments, cache.cache_ptr, {}, plan, x0,
                    Backend::NloptLbfgs, opts);
              },
              nullptr, refs.have_legacy ? &refs.legacy : nullptr);
          row = result.row;
          const auto post_flags = cache_flags(cache.cache_ptr);
          row.cache_blocks = post_flags.block_count;
          row.cache_has_diagonal = post_flags.has_diagonal;
          row.cache_has_full = post_flags.has_full;
          row.cache_has_dwls_weight = post_flags.has_dwls_weight;
          row.cache_has_wls_weight = post_flags.has_wls_weight;
          if (result.have_estimate) {
            refs.have_bounded = true;
            refs.bounded = std::move(result.estimate);
          }
        }
        write_row(out, row);
      }

      {
        auto cache = make_fit_only_cache(stats, estimator);
        Row row = base;
        row.estimator = est_name;
        row.path = "cache_snlls";
        row.cache_setup_ms = cache.cache_setup.median_ms;
        row.weight_setup_ms = cache.weight_setup.median_ms;
        const auto flags = cache_flags(cache.cache_ptr);
        row.cache_blocks = flags.block_count;
        row.cache_has_diagonal = flags.has_diagonal;
        row.cache_has_full = flags.has_full;
        row.cache_has_dwls_weight = flags.has_dwls_weight;
        row.cache_has_wls_weight = flags.has_wls_weight;
        if (!cache.error.empty()) {
          row.fit_status = "error";
          row.error = cache.error;
        } else {
          auto result = run_timed_fit(
              row, reps,
              [&]() {
                return magmaan::estimate::fit_ordinal_snlls(
                    pt, rep, moments, cache.cache_ptr, plan, x0,
                    Backend::NloptLbfgs, opts);
              },
              refs.have_bounded ? &refs.bounded : nullptr,
              refs.have_legacy ? &refs.legacy : nullptr);
          row = result.row;
          const auto post_flags = cache_flags(cache.cache_ptr);
          row.cache_blocks = post_flags.block_count;
          row.cache_has_diagonal = post_flags.has_diagonal;
          row.cache_has_full = post_flags.has_full;
          row.cache_has_dwls_weight = post_flags.has_dwls_weight;
          row.cache_has_wls_weight = post_flags.has_wls_weight;
        }
        write_row(out, row);
      }
    }
  }

  std::cout << "Wrote " << out_path << "\n";
  return 0;
}
