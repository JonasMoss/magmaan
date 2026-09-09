#include "magmaan/data/ordinal.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto mid = values.size() / 2;
  return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

bool parse_positive(const char* arg, int& value) {
  const std::string_view text(arg);
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && value > 0;
}

Eigen::MatrixXd draw_data(int n, int p, int categories, int seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<double> normal;
  Eigen::MatrixXd X(n, p);
  for (int i = 0; i < n; ++i) {
    const double factor = normal(rng);
    for (int j = 0; j < p; ++j) {
      const double loading = 0.55 + 0.2 * static_cast<double>(j) / (p - 1);
      const double y = loading * factor + std::sqrt(1.0 - loading * loading) * normal(rng);
      const double probability = 0.5 * std::erfc(-y / std::sqrt(2.0));
      X(i, j) = 1 + std::min(categories - 1, static_cast<int>(categories * probability));
    }
  }
  return X;
}

template <class Fn, class Checksum>
bool measure(const std::string& prefix, const char* stage, int reps,
             Fn&& fn, Checksum&& checksum) {
  if (!fn()) return false;
  std::vector<double> wall, cpu;
  for (int rep = 0; rep < reps; ++rep) {
    const auto c0 = std::clock();
    const auto t0 = std::chrono::steady_clock::now();
    if (!fn()) return false;
    const auto t1 = std::chrono::steady_clock::now();
    const auto c1 = std::clock();
    wall.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    cpu.push_back(1000.0 * static_cast<double>(c1 - c0) / CLOCKS_PER_SEC);
  }
  std::cout << prefix << ',' << stage << ',' << median(wall) << ','
            << *std::min_element(wall.begin(), wall.end()) << ','
            << *std::max_element(wall.begin(), wall.end()) << ','
            << median(cpu) << ',' << checksum() << '\n' << std::flush;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int n = 300, p = 18, categories = 2, reps = 5, seed = 23260716;
  int* args[] = {&n, &p, &categories, &reps, &seed};
  auto usage = []() {
    std::cerr << "usage: magmaan_ordinal_gamma_influence_bench [n>=1] [p>=2] "
                 "[categories>=2] [reps>=1] [seed>=1]\n";
    return 2;
  };
  if (argc > 6) return usage();
  for (int i = 1; i < argc; ++i) {
    if (!parse_positive(argv[i], *args[i - 1])) return usage();
  }
  if (p < 2 || categories < 2) return usage();
  Eigen::setNbThreads(1);
  const Eigen::MatrixXd X = draw_data(n, p, categories, seed);
  const std::vector<Eigen::MatrixXd> blocks{X};
  magmaan::data::OrdinalStats stats;
  Eigen::MatrixXd direct, D, movement;
  const std::string prefix = std::to_string(n) + ',' + std::to_string(p) + ',' +
      std::to_string(categories) + ',' + std::to_string(seed) + ',' + std::to_string(reps);
  std::cout.precision(10);
  std::cout << "n,p,categories,seed,reps,stage,median_ms,min_ms,max_ms,median_cpu_ms,checksum\n";
  auto statistics = [&]() {
    auto result = magmaan::data::ordinal_stats_from_integer_data(blocks, false);
    if (!result) { std::cerr << result.error().detail << '\n'; return false; }
    stats = std::move(*result);
    return true;
  };
  if (!measure(prefix, "statistics", reps, statistics,
               [&]() { return stats.NACOV[0].trace(); })) return 1;
  auto gamma_direct = [&]() {
    auto result = magmaan::data::ordinal_gamma_diag_data_influence(
        stats.int_data[0], stats.n_levels[0], stats.thresholds[0], stats.R[0]);
    if (!result) { std::cerr << result.error().detail << '\n'; return false; }
    direct = std::move(*result);
    return direct.allFinite();
  };
  if (!measure(prefix, "gamma_direct", reps, gamma_direct,
               [&]() { return direct.squaredNorm(); })) return 1;
  auto gamma_jacobian = [&]() {
    auto result = magmaan::data::ordinal_gamma_diag_jacobian_fd(
        stats.int_data[0], stats.n_levels[0], stats.thresholds[0], stats.R[0]);
    if (!result) { std::cerr << result.error().detail << '\n'; return false; }
    D = std::move(*result);
    return D.allFinite();
  };
  if (!measure(prefix, "gamma_jacobian", reps, gamma_jacobian,
               [&]() { return D.squaredNorm(); })) return 1;
  if (!measure(prefix, "gamma_movement_product", reps, [&]() {
        movement.noalias() = stats.moment_influence[0] * D.transpose();
        return movement.allFinite();
      }, [&]() { return movement.squaredNorm(); })) return 1;

  // Probe the proposed item/pair dependency support against the current dense FD.
  const Eigen::Index nth = stats.thresholds[0].size();
  const Eigen::Index m = D.rows();
  Eigen::ArrayXXi support = Eigen::ArrayXXi::Zero(m, m);
  std::vector<Eigen::Index> starts(static_cast<std::size_t>(p));
  Eigen::Index off = 0;
  for (int j = 0; j < p; ++j) {
    const Eigen::Index len = stats.n_levels[0][static_cast<std::size_t>(j)] - 1;
    starts[static_cast<std::size_t>(j)] = off;
    support.block(off, off, len, len).setOnes();
    off += len;
  }
  Eigen::Index row = nth;
  for (int j = 0; j < p; ++j) {
    for (int i = j + 1; i < p; ++i, ++row) {
      support(row, row) = 1;
      for (const int v : {i, j}) {
        const auto vz = static_cast<std::size_t>(v);
        support.block(row, starts[vz], 1, stats.n_levels[0][vz] - 1).setOnes();
      }
    }
  }
  const double outside = (D.array().abs() * (support == 0).cast<double>()).maxCoeff();
  std::cerr << "moments=" << m << " support_entries=" << support.sum()
            << " dense_entries=" << m * m << " max_abs_outside_support=" << outside
            << " total_gamma_if_mean_max="
            << (direct + movement).colwise().mean().cwiseAbs().maxCoeff() << '\n';
  return 0;
}
