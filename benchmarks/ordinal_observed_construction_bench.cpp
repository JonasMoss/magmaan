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
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/ordinal.hpp"

namespace {

constexpr double quiet_nan() {
  return std::numeric_limits<double>::quiet_NaN();
}

struct Timing {
  double median_ms = quiet_nan();
  double min_ms = quiet_nan();
  double max_ms = quiet_nan();
};

template <class Fn>
bool measure_reps(int reps, Fn&& fn, Timing& timing, std::string& error,
                  double& checksum) {
  std::vector<double> times;
  times.reserve(static_cast<std::size_t>(std::max(1, reps)));
  for (int rep = 0; rep < std::max(1, reps); ++rep) {
    const auto start = std::chrono::steady_clock::now();
    if (!fn(error, checksum)) return false;
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    times.push_back(elapsed.count());
  }
  std::sort(times.begin(), times.end());
  const std::size_t mid = times.size() / 2U;
  timing.median_ms = (times.size() % 2U == 0U)
                         ? 0.5 * (times[mid - 1U] + times[mid])
                         : times[mid];
  timing.min_ms = times.front();
  timing.max_ms = times.back();
  return true;
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

double parse_double_arg(const char* arg, double fallback) {
  if (arg == nullptr) return fallback;
  char* end = nullptr;
  const double value = std::strtod(arg, &end);
  if (end == arg || *end != '\0' || !std::isfinite(value)) return fallback;
  return value;
}

std::vector<double> thresholds_for_categories(int categories) {
  if (categories <= 3) return {-0.55, 0.55};
  if (categories <= 5) return {-1.20, -0.35, 0.35, 1.20};
  return {-1.55, -0.95, -0.35, 0.35, 0.95, 1.55};
}

Eigen::MatrixXd simulate_ordinal_data(int n, int p, int categories, int seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<double> norm(0.0, 1.0);
  const auto thresholds = thresholds_for_categories(categories);
  Eigen::MatrixXd x(n, p);
  const double denom = static_cast<double>(std::max(1, p - 1));
  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    const double eta1 = norm(rng);
    const double eta2 = 0.35 * eta1 + std::sqrt(1.0 - 0.35 * 0.35) * norm(rng);
    for (Eigen::Index j = 0; j < x.cols(); ++j) {
      const double position = static_cast<double>(j) / denom;
      const double primary = (j < x.cols() / 2) ? eta1 : eta2;
      const double secondary = (j < x.cols() / 2) ? eta2 : eta1;
      const double loading = 0.82 - 0.18 * position;
      const double cross_loading = (j % 4 == 0) ? 0.16 : 0.05;
      const double signal = loading * primary + cross_loading * secondary;
      const double resid_sd = std::sqrt(std::max(
          1e-8, 1.0 - loading * loading - cross_loading * cross_loading));
      const double y = signal + resid_sd * norm(rng);
      int category = 1;
      for (const double threshold : thresholds) {
        if (y > threshold) ++category;
      }
      x(i, j) = static_cast<double>(category);
    }
  }
  return x;
}

Eigen::MatrixXd apply_mcar_missing(Eigen::MatrixXd x, double missing_rate,
                                   int seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::bernoulli_distribution miss(std::clamp(missing_rate, 0.0, 0.95));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (Eigen::Index r = 0; r < x.rows(); ++r) {
    for (Eigen::Index c = 0; c < x.cols(); ++c) {
      if (miss(rng)) x(r, c) = nan;
    }
  }
  return x;
}

struct DesignCell {
  int n = 0;
  int p = 0;
  int categories = 0;
  int seed = 0;
};

std::vector<DesignCell> make_design(bool smoke, int seed_base, int n,
                                    int categories, int max_p) {
  const std::vector<int> p_values =
      smoke ? std::vector<int>{4, 8} : std::vector<int>{8, 12, 16};
  std::vector<DesignCell> out;
  int offset = 0;
  for (const int p : p_values) {
    if (p > max_p) continue;
    out.push_back(DesignCell{.n = n,
                             .p = p,
                             .categories = categories,
                             .seed = seed_base + offset});
    ++offset;
  }
  return out;
}

std::string gamma_label(magmaan::data::OrdinalPairwiseGammaKind kind) {
  return kind == magmaan::data::OrdinalPairwiseGammaKind::Overlap ? "overlap"
                                                                  : "nominal";
}

void print_help(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --out PATH        CSV output path (default: "
         "ordinal_observed_construction.csv)\n"
      << "  --reps N          repeated timings per operation (default: 5)\n"
      << "  --n N             rows per design cell (default: 1000)\n"
      << "  --categories N    ordinal categories per indicator (default: 5)\n"
      << "  --missing RATE    MCAR missing rate (default: 0.20)\n"
      << "  --seed-base N     deterministic seed base (default: 20260618)\n"
      << "  --max-p N         largest indicator count in design (default: 16)\n"
      << "  --smoke           run a small design\n"
      << "  --help            print this help\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "ordinal_observed_construction.csv";
  int reps = 5;
  int n = 1000;
  int categories = 5;
  int seed_base = 20260618;
  int max_p = 16;
  double missing_rate = 0.20;
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
      out_path = argv[++i];
      continue;
    }
    if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = std::max(1, parse_int_arg(argv[++i], reps));
      continue;
    }
    if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      n = std::max(2, parse_int_arg(argv[++i], n));
      continue;
    }
    if (std::strcmp(argv[i], "--categories") == 0 && i + 1 < argc) {
      categories = std::max(3, parse_int_arg(argv[++i], categories));
      continue;
    }
    if (std::strcmp(argv[i], "--missing") == 0 && i + 1 < argc) {
      missing_rate = std::clamp(parse_double_arg(argv[++i], missing_rate),
                                0.0, 0.95);
      continue;
    }
    if (std::strcmp(argv[i], "--seed-base") == 0 && i + 1 < argc) {
      seed_base = parse_int_arg(argv[++i], seed_base);
      continue;
    }
    if (std::strcmp(argv[i], "--max-p") == 0 && i + 1 < argc) {
      max_p = std::max(1, parse_int_arg(argv[++i], max_p));
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
  out << "n,p,categories,missing_rate,reps,seed,gamma_kind,moment_dim,"
         "status,error,median_ms,min_ms,max_ms,checksum\n";

  const auto design = make_design(smoke, seed_base, n, categories, max_p);
  if (design.empty()) {
    std::cerr << "Design is empty; increase --max-p\n";
    return 2;
  }

  for (const auto& cell : design) {
    const Eigen::MatrixXd complete =
        simulate_ordinal_data(cell.n, cell.p, cell.categories, cell.seed);
    const Eigen::MatrixXd observed =
        apply_mcar_missing(complete, missing_rate, cell.seed + 100000);
    const int n_thresholds = cell.p * (cell.categories - 1);
    const int n_correlations = (cell.p * (cell.p - 1)) / 2;
    const int moment_dim = n_thresholds + n_correlations;

    for (const auto kind : {magmaan::data::OrdinalPairwiseGammaKind::Overlap,
                            magmaan::data::OrdinalPairwiseGammaKind::Nominal}) {
      Timing timing;
      std::string error;
      double checksum = 0.0;
      const bool ok = measure_reps(
          reps,
          [&](std::string& err, double& sum) {
            auto stats = magmaan::data::ordinal_stats_from_observed_integer_data(
                {observed}, kind, false);
            if (!stats.has_value()) {
              err = stats.error().detail;
              return false;
            }
            sum += stats->R[0].trace();
            sum += stats->thresholds[0].sum();
            sum += stats->NACOV[0].diagonal().sum();
            sum += stats->W_dwls[0].diagonal().sum();
            sum += static_cast<double>(stats->moment_overlap_n_obs[0](0, 0));
            return true;
          },
          timing, error, checksum);

      out << cell.n << ',' << cell.p << ',' << cell.categories << ','
          << missing_rate << ',' << reps << ',' << cell.seed << ','
          << gamma_label(kind) << ',' << moment_dim << ','
          << (ok ? "ok" : "error") << ',' << error << ','
          << timing.median_ms << ',' << timing.min_ms << ','
          << timing.max_ms << ',' << checksum << '\n';
    }
  }

  return 0;
}
