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

using magmaan::data::OrdinalGammaCache;
using magmaan::data::OrdinalMoments;
using magmaan::data::OrdinalStats;

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

std::string bool_csv(bool value) { return value ? "1" : "0"; }

std::vector<double> thresholds_for_categories(int categories) {
  if (categories <= 3)
    return {-0.55, 0.55};
  if (categories <= 5)
    return {-1.20, -0.35, 0.35, 1.20};
  return {-1.55, -0.95, -0.35, 0.35, 0.95, 1.55};
}

Eigen::MatrixXd simulate_ordinal_data(int n, int p, int categories, int seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<double> norm(0.0, 1.0);
  const auto thresholds = thresholds_for_categories(categories);
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
      int category = 1;
      for (const double threshold : thresholds) {
        if (y > threshold)
          ++category;
      }
      x(i, j) = static_cast<double>(category);
    }
  }
  return x;
}

struct Row {
  std::string design;
  int seed = 0;
  int n = 0;
  int p = 0;
  int categories = 0;
  int reps = 0;
  int n_thresholds = 0;
  int n_correlations = 0;
  int moment_dim = 0;
  double gamma_mb = quiet_nan();
  double diagonal_mb = quiet_nan();
  std::string operation;
  std::string status = "ok";
  std::string error;
  double median_ms = quiet_nan();
  double min_ms = quiet_nan();
  double max_ms = quiet_nan();
  bool has_moments = false;
  bool has_diagonal = false;
  bool has_full_gamma = false;
  bool has_dwls_weight = false;
  bool has_wls_weight = false;
  double checksum = quiet_nan();
};

void write_header(std::ostream &out) {
  out << "design,seed,n,p,categories,reps,n_thresholds,n_correlations,"
         "moment_dim,gamma_mb,diagonal_mb,operation,status,error,median_ms,"
         "min_ms,max_ms,has_moments,has_diagonal,has_full_gamma,"
         "has_dwls_weight,has_wls_weight,checksum\n";
}

void write_row(std::ostream &out, const Row &row) {
  out << csv_escape(row.design) << ',' << row.seed << ',' << row.n << ','
      << row.p << ',' << row.categories << ',' << row.reps << ','
      << row.n_thresholds << ',' << row.n_correlations << ',' << row.moment_dim
      << ',' << row.gamma_mb << ',' << row.diagonal_mb << ','
      << csv_escape(row.operation) << ',' << csv_escape(row.status) << ','
      << csv_escape(row.error) << ',' << row.median_ms << ',' << row.min_ms
      << ',' << row.max_ms << ',' << bool_csv(row.has_moments) << ','
      << bool_csv(row.has_diagonal) << ',' << bool_csv(row.has_full_gamma)
      << ',' << bool_csv(row.has_dwls_weight) << ','
      << bool_csv(row.has_wls_weight) << ',' << row.checksum << '\n';
}

struct TimedOpResult {
  Row row;
};

template <class Fn> TimedOpResult run_operation(Row base, int reps, Fn &&fn) {
  double checksum = 0.0;
  std::string error;
  const auto timed = measure_reps(reps, [&]() {
    const auto ok = fn(checksum, error);
    return ok;
  });
  if (!timed.ok) {
    base.status = "error";
    base.error = error.empty() ? "operation failed" : error;
    return {.row = base};
  }
  base.median_ms = timed.timing.median_ms;
  base.min_ms = timed.timing.min_ms;
  base.max_ms = timed.timing.max_ms;
  base.checksum = checksum;
  return {.row = base};
}

OrdinalGammaCache full_gamma_cache_from_stats(const OrdinalStats &stats) {
  OrdinalGammaCache cache;
  cache.blocks.resize(stats.NACOV.size());
  for (std::size_t b = 0; b < stats.NACOV.size(); ++b) {
    cache.blocks[b].gamma = stats.NACOV[b];
    cache.blocks[b].has_full = true;
  }
  return cache;
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
         "ordinal_construction.csv)\n"
      << "  --reps N         repeated timings per operation (default: 3)\n"
      << "  --seed-base N    deterministic seed base (default: 20260608)\n"
      << "  --max-p N        largest indicator count in design (default: 16)\n"
      << "  --smoke          run a small design\n"
      << "  --help           print this help\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string out_path = "ordinal_construction.csv";
  int reps = 3;
  int seed_base = 20260608;
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
    const Eigen::MatrixXd data =
        simulate_ordinal_data(cell.n, cell.p, cell.categories, cell.seed);

    OrdinalStats stats;
    {
      Row row;
      row.design = cell.design;
      row.seed = cell.seed;
      row.n = cell.n;
      row.p = cell.p;
      row.categories = cell.categories;
      row.reps = reps;
      row.n_correlations = (cell.p * (cell.p - 1)) / 2;
      row.operation = "legacy_stats_full";
      row.has_moments = true;
      row.has_diagonal = true;
      row.has_full_gamma = true;
      row.has_dwls_weight = true;
      row.has_wls_weight = true;
      auto result =
          run_operation(row, reps, [&](double &checksum, std::string &error) {
            auto computed =
                magmaan::data::ordinal_stats_from_integer_data({data});
            if (!computed.has_value()) {
              error = computed.error().detail;
              return false;
            }
            stats = std::move(*computed);
            checksum += stats.R[0](0, 0);
            checksum += stats.thresholds[0].sum();
            checksum += stats.NACOV[0](0, 0);
            checksum += stats.W_dwls[0](0, 0);
            checksum += stats.W_wls[0](0, 0);
            return true;
          });
      if (result.row.status != "ok") {
        write_row(out, result.row);
        std::cerr << "ordinal stats failed: " << result.row.error << "\n";
        return 1;
      }
      const int nth = static_cast<int>(stats.thresholds[0].size());
      const int mdim = static_cast<int>(stats.NACOV[0].rows());
      result.row.n_thresholds = nth;
      result.row.moment_dim = mdim;
      result.row.gamma_mb =
          static_cast<double>(mdim) * static_cast<double>(mdim) * 8.0 / 1e6;
      result.row.diagonal_mb = static_cast<double>(mdim) * 8.0 / 1e6;
      write_row(out, result.row);
    }

    const int n_thresholds = static_cast<int>(stats.thresholds[0].size());
    const int n_correlations = (cell.p * (cell.p - 1)) / 2;
    const int moment_dim = static_cast<int>(stats.NACOV[0].rows());
    const double gamma_mb = static_cast<double>(moment_dim) *
                            static_cast<double>(moment_dim) * 8.0 / 1e6;
    const double diagonal_mb = static_cast<double>(moment_dim) * 8.0 / 1e6;

    Row base;
    base.design = cell.design;
    base.seed = cell.seed;
    base.n = cell.n;
    base.p = cell.p;
    base.categories = cell.categories;
    base.reps = reps;
    base.n_thresholds = n_thresholds;
    base.n_correlations = n_correlations;
    base.moment_dim = moment_dim;
    base.gamma_mb = gamma_mb;
    base.diagonal_mb = diagonal_mb;

    {
      Row row = base;
      row.operation = "moments_projection_from_stats";
      row.has_moments = true;
      auto result =
          run_operation(row, reps, [&](double &checksum, std::string &error) {
            (void)error;
            OrdinalMoments moments =
                magmaan::data::ordinal_moments_from_stats(stats);
            checksum += moments.R[0](0, 0);
            checksum += moments.thresholds[0].sum();
            checksum += static_cast<double>(moments.n_obs[0]);
            return true;
          });
      write_row(out, result.row);
    }

    {
      Row row = base;
      row.operation = "cache_diagonal_from_legacy_gamma";
      row.has_diagonal = true;
      auto result =
          run_operation(row, reps, [&](double &checksum, std::string &error) {
            (void)error;
            auto cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
                {stats.NACOV[0].diagonal()});
            checksum += static_cast<double>(cache.blocks.size());
            checksum += cache.blocks[0].diagonal.sum();
            return true;
          });
      write_row(out, result.row);
    }

    {
      Row row = base;
      row.operation = "cache_full_gamma_copy";
      row.has_full_gamma = true;
      auto result =
          run_operation(row, reps, [&](double &checksum, std::string &error) {
            (void)error;
            auto cache = full_gamma_cache_from_stats(stats);
            checksum += static_cast<double>(cache.blocks.size());
            checksum += cache.blocks[0].gamma(0, 0);
            checksum += cache.blocks[0].gamma.diagonal().sum();
            return true;
          });
      write_row(out, result.row);
    }

    {
      Row row = base;
      row.operation = "cache_from_stats_copy_all";
      row.has_diagonal = true;
      row.has_full_gamma = true;
      row.has_dwls_weight = true;
      row.has_wls_weight = true;
      auto result =
          run_operation(row, reps, [&](double &checksum, std::string &error) {
            (void)error;
            auto cache = magmaan::data::ordinal_gamma_cache_from_stats(stats);
            checksum += static_cast<double>(cache.blocks.size());
            checksum += cache.blocks[0].diagonal.sum();
            checksum += cache.blocks[0].gamma(0, 0);
            checksum += cache.blocks[0].w_dwls(0, 0);
            checksum += cache.blocks[0].w_wls(0, 0);
            return true;
          });
      write_row(out, result.row);
    }

    {
      Row row = base;
      row.operation = "extract_diagonal_from_full_gamma";
      row.has_diagonal = true;
      row.has_full_gamma = true;
      auto result =
          run_operation(row, reps, [&](double &checksum, std::string &error) {
            auto cache = full_gamma_cache_from_stats(stats);
            auto ok = magmaan::data::ordinal_gamma_cache_ensure_diagonal(cache);
            if (!ok.has_value()) {
              error = ok.error().detail;
              return false;
            }
            checksum += cache.blocks[0].diagonal.sum();
            return true;
          });
      write_row(out, result.row);
    }

    {
      Row row = base;
      row.operation = "dwls_weight_from_diagonal";
      row.has_diagonal = true;
      row.has_dwls_weight = true;
      auto result =
          run_operation(row, reps, [&](double &checksum, std::string &error) {
            auto cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
                {stats.NACOV[0].diagonal()});
            auto ok =
                magmaan::data::ordinal_gamma_cache_ensure_dwls_weights(cache);
            if (!ok.has_value()) {
              error = ok.error().detail;
              return false;
            }
            checksum += cache.blocks[0].w_dwls.diagonal().sum();
            return true;
          });
      write_row(out, result.row);
    }

    {
      Row row = base;
      row.operation = "wls_weight_reinvert_full_gamma";
      row.has_full_gamma = true;
      row.has_wls_weight = true;
      auto result =
          run_operation(row, reps, [&](double &checksum, std::string &error) {
            auto cache = full_gamma_cache_from_stats(stats);
            auto ok =
                magmaan::data::ordinal_gamma_cache_ensure_wls_weights(cache);
            if (!ok.has_value()) {
              error = ok.error().detail;
              return false;
            }
            checksum += cache.blocks[0].w_wls(0, 0);
            checksum += cache.blocks[0].w_wls.diagonal().sum();
            return true;
          });
      write_row(out, result.row);
    }
  }

  std::cout << "Wrote " << out_path << "\n";
  return 0;
}
