#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/gmm/dls_weight.hpp"

namespace {

struct Config {
  int n = 800;
  int reps = 200;
  std::uint64_t seed = 20260518;
};

struct Summary {
  std::string name;
  int requested = 0;
  int used = 0;
  int failed = 0;
  double mean = std::numeric_limits<double>::quiet_NaN();
  double sd = std::numeric_limits<double>::quiet_NaN();
  double min = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();
};

Config parse_args(int argc, char** argv) {
  Config cfg;
  auto parse_int = [](std::string_view s, int fallback) {
    int out = fallback;
    const char* first = s.data();
    const char* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    if (ec != std::errc{} || ptr != last) out = fallback;
    return out;
  };
  auto parse_seed = [](std::string_view s, std::uint64_t fallback) {
    std::uint64_t out = fallback;
    const char* first = s.data();
    const char* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    if (ec != std::errc{} || ptr != last) out = fallback;
    return out;
  };

  for (int i = 1; i < argc; ++i) {
    const std::string_view a(argv[i]);
    auto value = [&](std::string_view key) -> std::string_view {
      return a.starts_with(key) ? a.substr(key.size()) : std::string_view{};
    };
    if (const auto v = value("--n="); !v.empty()) cfg.n = parse_int(v, cfg.n);
    if (const auto v = value("--reps="); !v.empty()) {
      cfg.reps = parse_int(v, cfg.reps);
    }
    if (const auto v = value("--seed="); !v.empty()) {
      cfg.seed = parse_seed(v, cfg.seed);
    }
  }
  return cfg;
}

double skewed_unit(std::mt19937_64& rng) {
  std::normal_distribution<double> z(0.0, 1.0);
  constexpr double sigma = 0.8;
  const double raw = std::exp(sigma * z(rng));
  const double mean = std::exp(0.5 * sigma * sigma);
  const double var = (std::exp(sigma * sigma) - 1.0) *
                     std::exp(sigma * sigma);
  return (raw - mean) / std::sqrt(var);
}

magmaan::data::RawData make_raw(int n, int condition, std::mt19937_64& rng) {
  std::normal_distribution<double> z(0.0, 1.0);
  std::chi_squared_distribution<double> chi2(5.0);
  const double lam[6] = {1.0, 0.9, 0.8, 1.1, 0.7, 1.2};
  Eigen::MatrixXd X(n, 6);

  for (int i = 0; i < n; ++i) {
    double scale = 1.0;
    if (condition == 1) scale = std::sqrt(5.0 / chi2(rng));
    const double f = (condition == 2) ? skewed_unit(rng) : z(rng);
    for (int j = 0; j < 6; ++j) {
      const double e = (condition == 2) ? skewed_unit(rng) : z(rng);
      X(i, j) = scale * (lam[j] * f + std::sqrt(0.5) * e);
    }
  }

  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  return raw;
}

Summary run_condition(std::string name, int condition, const Config& cfg) {
  std::mt19937_64 rng(cfg.seed + static_cast<std::uint64_t>(101 * condition));
  Summary out;
  out.name = std::move(name);
  out.requested = cfg.reps;

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(cfg.reps));
  for (int r = 0; r < cfg.reps; ++r) {
    auto raw = make_raw(cfg.n, condition, rng);
    auto samp = magmaan::data::sample_stats_from_raw(raw);
    if (!samp.has_value()) {
      ++out.failed;
      continue;
    }
    auto eb = magmaan::estimate::empirical_bayes_dls_mixing_scalar(*samp, raw);
    if (!eb.has_value()) {
      ++out.failed;
      continue;
    }
    values.push_back(eb->a);
  }

  out.used = static_cast<int>(values.size());
  if (values.empty()) return out;
  out.mean = std::accumulate(values.begin(), values.end(), 0.0) /
             static_cast<double>(values.size());
  double ss = 0.0;
  for (double v : values) ss += (v - out.mean) * (v - out.mean);
  out.sd = values.size() > 1
               ? std::sqrt(ss / static_cast<double>(values.size() - 1))
               : 0.0;
  auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
  out.min = *min_it;
  out.max = *max_it;
  return out;
}

void print_summary(const Summary& s) {
  std::cout << std::left << std::setw(14) << s.name
            << " requested=" << std::setw(4) << s.requested
            << " used=" << std::setw(4) << s.used
            << " failed=" << std::setw(4) << s.failed
            << " mean_a=" << std::fixed << std::setprecision(3) << s.mean
            << " sd=" << s.sd
            << " min=" << s.min
            << " max=" << s.max << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  const Config cfg = parse_args(argc, argv);
  std::cout << "DLS empirical-Bayes scalar sanity check"
            << " n=" << cfg.n << " reps=" << cfg.reps
            << " seed=" << cfg.seed << '\n';
  print_summary(run_condition("normal", 0, cfg));
  print_summary(run_condition("elliptical", 1, cfg));
  print_summary(run_condition("skewed", 2, cfg));
  return 0;
}
