// Construction-time benchmark for the mixed continuous/ordinal sample-stats
// paths: the lazy MixedOrdinalWorkspace (ULS / DWLS plans) versus the legacy
// MixedOrdinalStats with and without the eager full-WLS inverse. Mirrors
// ordinal_construction_bench.cpp, which covers the all-ordinal paths; this
// harness exercises the polyserial/Pearson branches the all-ordinal bench
// never reaches.
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

namespace {

using magmaan::data::MixedOrdinalStats;

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
    if (!ok) {
      return {.ok = false, .timing = {}};
    }
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

std::vector<double> thresholds_for_categories(int categories) {
  if (categories <= 3)
    return {-0.55, 0.55};
  if (categories <= 5)
    return {-1.20, -0.35, 0.35, 1.20};
  return {-1.55, -0.95, -0.35, 0.35, 0.95, 1.55};
}

// Same two-factor latent design as the all-ordinal bench; even-indexed
// columns are discretized, odd-indexed columns stay continuous. `ordered`
// receives the per-column 0/1 ordinal mask.
Eigen::MatrixXd simulate_mixed_data(int n, int p, int categories, int seed,
                                    std::vector<std::int32_t> &ordered) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<double> norm(0.0, 1.0);
  const auto thresholds = thresholds_for_categories(categories);
  ordered.assign(static_cast<std::size_t>(p), 0);
  for (int j = 0; j < p; j += 2)
    ordered[static_cast<std::size_t>(j)] = 1;
  Eigen::MatrixXd x(n, p);
  const double denom = static_cast<double>(std::max(1, p - 1));
  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    const double eta = norm(rng);
    const double eta2 = 0.45 * eta + std::sqrt(1.0 - 0.45 * 0.45) * norm(rng);
    for (Eigen::Index j = 0; j < x.cols(); ++j) {
      const double position = static_cast<double>(j) / denom;
      const double loading = 0.86 - 0.22 * position;
      const double secondary = (j % 3 == 0) ? 0.18 : -0.10;
      const double signal = loading * eta + secondary * eta2;
      const double resid_sd = std::sqrt(
          std::max(1e-8, 1.0 - loading * loading - secondary * secondary));
      const double shift = 0.14 * std::sin(1.3 * static_cast<double>(j));
      const double y = shift + signal + resid_sd * norm(rng);
      if (j % 2 == 0) {
        int category = 1;
        for (const double threshold : thresholds) {
          if (y > threshold)
            ++category;
        }
        x(i, j) = static_cast<double>(category);
      } else {
        x(i, j) = 1.7 * y - 0.4;
      }
    }
  }
  return x;
}

struct Row {
  std::string design;
  int seed = 0;
  int n = 0;
  int p = 0;
  int p_ordinal = 0;
  int categories = 0;
  int reps = 0;
  int n_thresholds = 0;
  int moment_dim = 0;
  double gamma_mb = quiet_nan();
  std::string operation;
  std::string status = "ok";
  std::string error;
  double median_ms = quiet_nan();
  double min_ms = quiet_nan();
  double max_ms = quiet_nan();
  double checksum = quiet_nan();
};

void write_header(std::ostream &out) {
  out << "design,seed,n,p,p_ordinal,categories,reps,n_thresholds,moment_dim,"
         "gamma_mb,operation,status,error,median_ms,min_ms,max_ms,checksum\n";
}

void write_row(std::ostream &out, const Row &row) {
  out << csv_escape(row.design) << ',' << row.seed << ',' << row.n << ','
      << row.p << ',' << row.p_ordinal << ',' << row.categories << ','
      << row.reps << ',' << row.n_thresholds << ',' << row.moment_dim << ','
      << row.gamma_mb << ',' << csv_escape(row.operation) << ','
      << csv_escape(row.status) << ',' << csv_escape(row.error) << ','
      << row.median_ms << ',' << row.min_ms << ',' << row.max_ms << ','
      << row.checksum << '\n';
}

template <class Fn> Row run_operation(Row base, int reps, Fn &&fn) {
  double checksum = 0.0;
  std::string error;
  const auto timed = measure_reps(reps, [&]() { return fn(checksum, error); });
  if (!timed.ok) {
    base.status = "error";
    base.error = error.empty() ? "operation failed" : error;
    return base;
  }
  base.median_ms = timed.timing.median_ms;
  base.min_ms = timed.timing.min_ms;
  base.max_ms = timed.timing.max_ms;
  base.checksum = checksum;
  return base;
}

struct DesignCell {
  std::string design;
  int n = 0;
  int p = 0;
  int categories = 0;
  int seed = 0;
};

std::vector<DesignCell> make_design(bool smoke, int seed_base, int max_p) {
  const int n = smoke ? 420 : 900;
  const std::vector<int> p_values =
      smoke ? std::vector<int>{4, 8} : std::vector<int>{4, 8, 12, 16};
  const std::vector<int> categories =
      smoke ? std::vector<int>{3} : std::vector<int>{3, 5};
  std::vector<DesignCell> out;
  int offset = 0;
  for (const int cat : categories) {
    for (const int p : p_values) {
      if (p > max_p)
        continue;
      std::ostringstream id;
      id << "p" << p << "_c" << cat;
      out.push_back(DesignCell{.design = id.str(),
                               .n = n,
                               .p = p,
                               .categories = cat,
                               .seed = seed_base + offset});
      ++offset;
    }
  }
  return out;
}

void print_help(const char *argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --out PATH       CSV output path (default: "
         "mixed_ordinal_construction.csv)\n"
      << "  --reps N         repeated timings per operation (default: 3)\n"
      << "  --seed-base N    deterministic seed base (default: 20260612)\n"
      << "  --max-p N        largest indicator count in design (default: 16)\n"
      << "  --smoke          run a small design\n"
      << "  --help           print this help\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string out_path = "mixed_ordinal_construction.csv";
  int reps = 3;
  int seed_base = 20260612;
  int max_p = 16;
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
    if (std::strcmp(argv[i], "--seed-base") == 0 && i + 1 < argc) {
      ++i;
      seed_base = parse_int_arg(argv[i], seed_base);
      continue;
    }
    if (std::strcmp(argv[i], "--max-p") == 0 && i + 1 < argc) {
      ++i;
      max_p = std::max(1, parse_int_arg(argv[i], max_p));
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

  const auto design = make_design(smoke, seed_base, max_p);
  if (design.empty()) {
    std::cerr << "Design is empty; increase --max-p\n";
    return 2;
  }

  for (const auto &cell : design) {
    std::vector<std::int32_t> ordered_cols;
    const Eigen::MatrixXd data = simulate_mixed_data(
        cell.n, cell.p, cell.categories, cell.seed, ordered_cols);
    const std::vector<std::vector<std::int32_t>> ordered = {ordered_cols};

    int p_ordinal = 0;
    for (const std::int32_t flag : ordered_cols)
      p_ordinal += flag != 0 ? 1 : 0;
    const int p_continuous = cell.p - p_ordinal;
    const int n_thresholds = p_ordinal * (cell.categories - 1);
    // Mixed moment vector: thresholds, continuous means, continuous
    // variances, then the lower-triangle covariance/correlation entries.
    const int moment_dim = n_thresholds + 2 * p_continuous +
                           (cell.p * (cell.p - 1)) / 2;
    const double gamma_mb = static_cast<double>(moment_dim) *
                            static_cast<double>(moment_dim) * 8.0 / 1e6;

    Row base;
    base.design = cell.design;
    base.seed = cell.seed;
    base.n = cell.n;
    base.p = cell.p;
    base.p_ordinal = p_ordinal;
    base.categories = cell.categories;
    base.reps = reps;
    base.n_thresholds = n_thresholds;
    base.moment_dim = moment_dim;
    base.gamma_mb = gamma_mb;

    {
      Row row = base;
      row.operation = "lazy_uls_workspace_from_data";
      write_row(out, run_operation(row, reps, [&](double &checksum,
                                                  std::string &error) {
        auto plan = magmaan::data::ordinal_weight_plan(
            magmaan::data::OrdinalWorkspacePurpose::FitOnly,
            magmaan::data::OrdinalEstimatorKind::ULS);
        auto workspace =
            magmaan::data::mixed_ordinal_workspace_from_data({data}, ordered,
                                                             plan);
        if (!workspace.has_value()) {
          error = workspace.error().detail;
          return false;
        }
        checksum += workspace->moments.R[0](0, 0);
        checksum += workspace->moments.moments[0].sum();
        checksum += static_cast<double>(workspace->gamma_cache.block_count());
        return true;
      }));
    }

    {
      Row row = base;
      row.operation = "lazy_dwls_workspace_from_data";
      write_row(out, run_operation(row, reps, [&](double &checksum,
                                                  std::string &error) {
        auto plan = magmaan::data::ordinal_weight_plan(
            magmaan::data::OrdinalWorkspacePurpose::FitOnly,
            magmaan::data::OrdinalEstimatorKind::DWLS);
        auto workspace =
            magmaan::data::mixed_ordinal_workspace_from_data({data}, ordered,
                                                             plan);
        if (!workspace.has_value()) {
          error = workspace.error().detail;
          return false;
        }
        checksum += workspace->moments.R[0](0, 0);
        checksum += workspace->moments.moments[0].sum();
        checksum += workspace->gamma_cache.blocks[0].diagonal.sum();
        return true;
      }));
    }

    {
      Row row = base;
      row.operation = "legacy_stats_dwls_only";
      write_row(out, run_operation(row, reps, [&](double &checksum,
                                                  std::string &error) {
        auto stats = magmaan::data::mixed_ordinal_stats_from_data(
            {data}, ordered, /*full_wls_weight=*/false);
        if (!stats.has_value()) {
          error = stats.error().detail;
          return false;
        }
        checksum += stats->R[0](0, 0);
        checksum += stats->moments[0].sum();
        checksum += stats->NACOV[0](0, 0);
        checksum += stats->W_dwls[0](0, 0);
        return true;
      }));
    }

    {
      Row row = base;
      row.operation = "legacy_stats_full";
      write_row(out, run_operation(row, reps, [&](double &checksum,
                                                  std::string &error) {
        auto stats = magmaan::data::mixed_ordinal_stats_from_data(
            {data}, ordered, /*full_wls_weight=*/true);
        if (!stats.has_value()) {
          error = stats.error().detail;
          return false;
        }
        checksum += stats->R[0](0, 0);
        checksum += stats->moments[0].sum();
        checksum += stats->NACOV[0](0, 0);
        checksum += stats->W_wls[0](0, 0);
        return true;
      }));
    }
  }

  std::cout << "Wrote " << out_path << "\n";
  return 0;
}
