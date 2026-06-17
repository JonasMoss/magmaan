#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

struct Timing {
  double median_ms = std::numeric_limits<double>::quiet_NaN();
  double min_ms = std::numeric_limits<double>::quiet_NaN();
  double max_ms = std::numeric_limits<double>::quiet_NaN();
};

template <class Fn>
Timing measure_reps(int reps, Fn&& fn) {
  std::vector<double> times;
  times.reserve(static_cast<std::size_t>(std::max(1, reps)));
  for (int rep = 0; rep < std::max(1, reps); ++rep) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    times.push_back(elapsed.count());
  }
  std::sort(times.begin(), times.end());
  const std::size_t mid = times.size() / 2U;
  const double median = (times.size() % 2U == 0U)
                            ? 0.5 * (times[mid - 1U] + times[mid])
                            : times[mid];
  return {.median_ms = median, .min_ms = times.front(), .max_ms = times.back()};
}

int parse_int_arg(const char* arg, int fallback) {
  if (!arg) return fallback;
  char* end = nullptr;
  const long value = std::strtol(arg, &end, 10);
  if (end == arg || *end != '\0' || value < 1L ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(value);
}

double parse_double_arg(const char* arg, double fallback) {
  if (!arg) return fallback;
  char* end = nullptr;
  const double value = std::strtod(arg, &end);
  if (end == arg || *end != '\0' || !std::isfinite(value)) return fallback;
  return value;
}

std::string cfa_model(int n_factors, int items_per_factor) {
  std::ostringstream out;
  for (int f = 0; f < n_factors; ++f) {
    out << "f" << (f + 1) << " =~ ";
    for (int j = 0; j < items_per_factor; ++j) {
      if (j > 0) out << " + ";
      out << "y" << (f * items_per_factor + j + 1);
    }
    out << '\n';
  }
  return out.str();
}

magmaan::data::RawData simulate_raw(int n, int n_factors, int items_per_factor,
                                    double missing_rate, int seed) {
  const int p = n_factors * items_per_factor;
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<double> norm(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  Eigen::MatrixXd X(n, p);
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask(n, p);
  mask.setOnes();

  for (Eigen::Index i = 0; i < n; ++i) {
    std::vector<double> eta(static_cast<std::size_t>(n_factors));
    for (int f = 0; f < n_factors; ++f) {
      eta[static_cast<std::size_t>(f)] = norm(rng);
    }
    int n_observed = 0;
    for (int f = 0; f < n_factors; ++f) {
      for (int j = 0; j < items_per_factor; ++j) {
        const int col = f * items_per_factor + j;
        const double loading = 0.82 - 0.05 * static_cast<double>(j);
        const double resid_sd =
            std::sqrt(std::max(1e-8, 1.0 - loading * loading));
        X(i, col) = 0.1 * static_cast<double>(j) +
                    loading * eta[static_cast<std::size_t>(f)] +
                    resid_sd * norm(rng);
        if (unif(rng) < missing_rate) {
          X(i, col) = std::numeric_limits<double>::quiet_NaN();
          mask(i, col) = 0U;
        } else {
          ++n_observed;
        }
      }
    }
    if (n_observed == 0) {
      X(i, 0) = norm(rng);
      mask(i, 0) = 1U;
    }
  }

  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  raw.mask.push_back(std::move(mask));
  return raw;
}

void print_row(const char* operation, const Timing& t, double checksum) {
  std::cout << operation << ",median_ms=" << t.median_ms
            << ",min_ms=" << t.min_ms
            << ",max_ms=" << t.max_ms
            << ",checksum=" << checksum << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = parse_int_arg(argc > 1 ? argv[1] : nullptr, 200);
  const int n = parse_int_arg(argc > 2 ? argv[2] : nullptr, 2800);
  const int n_factors = parse_int_arg(argc > 3 ? argv[3] : nullptr, 5);
  const int items_per_factor = parse_int_arg(argc > 4 ? argv[4] : nullptr, 5);
  const double missing_rate =
      parse_double_arg(argc > 5 ? argv[5] : nullptr, 0.15);

  const std::string model = cfa_model(n_factors, items_per_factor);
  auto parsed = magmaan::parse::Parser::parse(model);
  if (!parsed.has_value()) {
    std::cerr << "parse failed: " << parsed.error().detail << '\n';
    return 1;
  }
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto built = magmaan::spec::build(*parsed, opts);
  if (!built.has_value()) {
    std::cerr << "build failed: " << built.error().detail << '\n';
    return 1;
  }
  auto rep = magmaan::model::build_matrix_rep(*built);
  if (!rep.has_value()) {
    std::cerr << "matrix rep failed: " << rep.error().detail << '\n';
    return 1;
  }

  auto raw = simulate_raw(n, n_factors, items_per_factor, missing_rate,
                          20260617);
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  if (!pack.has_value()) {
    std::cerr << "fiml_pack failed: " << pack.error().detail << '\n';
    return 1;
  }
  auto x0 = magmaan::estimate::simple_start_values(
      *built, *rep, pack->start_stats, {});
  if (!x0.has_value()) {
    std::cerr << "start failed: " << x0.error().detail << '\n';
    return 1;
  }
  auto ev = magmaan::model::ModelEvaluator::build(*built, *rep);
  if (!ev.has_value()) {
    std::cerr << "evaluator build failed: " << ev.error().detail << '\n';
    return 1;
  }
  auto eval = ev->evaluate(*x0, true, true);
  if (!eval.has_value()) {
    std::cerr << "evaluate failed: " << eval.error().detail << '\n';
    return 1;
  }

  magmaan::estimate::fiml::FIML fiml;
  double kernel_checksum = 0.0;
  const Timing kernel = measure_reps(reps, [&] {
    auto vg = fiml.value_gradient(raw, pack->cache, eval->moments,
                                  eval->J_sigma, eval->J_mu);
    if (!vg.has_value()) {
      std::cerr << "value_gradient failed: " << vg.error().detail << '\n';
      std::exit(1);
    }
    kernel_checksum += vg->value + vg->gradient.squaredNorm() * 1e-9;
  });

  double full_checksum = 0.0;
  const Timing full = measure_reps(reps, [&] {
    auto e = ev->evaluate(*x0, true, true);
    if (!e.has_value()) {
      std::cerr << "evaluate failed: " << e.error().detail << '\n';
      std::exit(1);
    }
    auto vg = fiml.value_gradient(raw, pack->cache, e->moments,
                                  e->J_sigma, e->J_mu);
    if (!vg.has_value()) {
      std::cerr << "value_gradient failed: " << vg.error().detail << '\n';
      std::exit(1);
    }
    full_checksum += vg->value + vg->gradient.squaredNorm() * 1e-9;
  });

  std::cout << "reps=" << reps << ",n=" << n << ",p="
            << (n_factors * items_per_factor)
            << ",missing_rate=" << missing_rate
            << ",patterns=" << pack->cache.patterns.size() << '\n';
  print_row("value_gradient", kernel, kernel_checksum);
  print_row("evaluate_plus_value_gradient", full, full_checksum);
  return 0;
}
