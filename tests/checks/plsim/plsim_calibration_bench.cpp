#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/sim/plsim.hpp"
#include "magmaan/sim/vale_maurelli.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
  int reps = 5;
  int sample_n = 30000;
  int hermite_order = 30;
  int quadrature_points = 31;
  std::uint64_t seed = 20260605;
};

struct Case {
  std::string name;
  Eigen::MatrixXd corr;
  Eigen::VectorXd skew;
  Eigen::VectorXd kurt;
};

struct MethodSummary {
  std::string name;
  std::string last_error;
  int ok = 0;
  int failed = 0;
  double mean_ms = 0.0;
  double max_strategy_resid = std::numeric_limits<double>::quiet_NaN();
  double max_quad_resid = std::numeric_limits<double>::quiet_NaN();
  double max_rect_resid = std::numeric_limits<double>::quiet_NaN();
  double max_rho_delta = std::numeric_limits<double>::quiet_NaN();
  double sample_corr_err = std::numeric_limits<double>::quiet_NaN();
  double sample_shape_err = std::numeric_limits<double>::quiet_NaN();
};

int parse_int(std::string_view s, int fallback) {
  int out = fallback;
  const char* first = s.data();
  const char* last = s.data() + s.size();
  auto [ptr, ec] = std::from_chars(first, last, out);
  if (ec != std::errc{} || ptr != last) out = fallback;
  return out;
}

std::uint64_t parse_seed(std::string_view s, std::uint64_t fallback) {
  std::uint64_t out = fallback;
  const char* first = s.data();
  const char* last = s.data() + s.size();
  auto [ptr, ec] = std::from_chars(first, last, out);
  if (ec != std::errc{} || ptr != last) out = fallback;
  return out;
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a(argv[i]);
    auto value = [&](std::string_view key) -> std::string_view {
      return a.starts_with(key) ? a.substr(key.size()) : std::string_view{};
    };
    if (const auto v = value("--reps="); !v.empty()) cfg.reps = parse_int(v, cfg.reps);
    if (const auto v = value("--sample-n="); !v.empty()) {
      cfg.sample_n = parse_int(v, cfg.sample_n);
    }
    if (const auto v = value("--hermite-order="); !v.empty()) {
      cfg.hermite_order = parse_int(v, cfg.hermite_order);
    }
    if (const auto v = value("--quadrature-points="); !v.empty()) {
      cfg.quadrature_points = parse_int(v, cfg.quadrature_points);
    }
    if (const auto v = value("--seed="); !v.empty()) cfg.seed = parse_seed(v, cfg.seed);
  }
  cfg.reps = std::max(1, cfg.reps);
  cfg.sample_n = std::max(1000, cfg.sample_n);
  cfg.hermite_order = std::max(1, cfg.hermite_order);
  cfg.quadrature_points = std::max(8, cfg.quadrature_points);
  return cfg;
}

Eigen::MatrixXd exchangeable_corr(int p, double r) {
  Eigen::MatrixXd out = Eigen::MatrixXd::Constant(p, p, r);
  out.diagonal().array() = 1.0;
  return out;
}

Eigen::MatrixXd ar1_corr(int p, double r) {
  Eigen::MatrixXd out(p, p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      out(i, j) = std::pow(r, std::abs(i - j));
    }
  }
  return out;
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;

  Case win;
  win.name = "p2 skew2-kurt5";
  win.corr.resize(2, 2);
  win.corr << 1.0, 0.30,
              0.30, 1.0;
  win.skew.resize(2);
  win.skew << 2.0, 0.75;
  win.kurt.resize(2);
  win.kurt << 5.0, 1.25;
  cases.push_back(win);

  Case mixed;
  mixed.name = "p5 mixed";
  mixed.corr = exchangeable_corr(5, 0.25);
  mixed.corr(0, 2) = mixed.corr(2, 0) = -0.15;
  mixed.corr(1, 4) = mixed.corr(4, 1) = 0.45;
  mixed.skew.resize(5);
  mixed.skew << 0.0, 0.75, -0.50, 1.25, 2.0;
  mixed.kurt.resize(5);
  mixed.kurt << 0.0, 1.0, 0.80, 2.50, 5.0;
  cases.push_back(mixed);

  Case high_corr;
  high_corr.name = "p2 high corr tails";
  high_corr.corr.resize(2, 2);
  high_corr.corr << 1.0, 0.85,
                    0.85, 1.0;
  high_corr.skew.resize(2);
  high_corr.skew << 2.0, -1.5;
  high_corr.kurt.resize(2);
  high_corr.kurt << 5.0, 4.0;
  cases.push_back(high_corr);

  Case neg_corr;
  neg_corr.name = "p2 negative corr tails";
  neg_corr.corr.resize(2, 2);
  neg_corr.corr << 1.0, -0.65,
                  -0.65, 1.0;
  neg_corr.skew.resize(2);
  neg_corr.skew << 2.0, 2.0;
  neg_corr.kurt.resize(2);
  neg_corr.kurt << 5.0, 5.0;
  cases.push_back(neg_corr);

  Case ar;
  ar.name = "p4 ar high shape";
  ar.corr = ar1_corr(4, 0.70);
  ar.skew.resize(4);
  ar.skew << 2.0, -1.5, 1.25, -0.75;
  ar.kurt.resize(4);
  ar.kurt << 5.0, 4.0, 3.0, 1.5;
  cases.push_back(ar);

  Case mixed_sign;
  mixed_sign.name = "p4 mixed sign";
  mixed_sign.corr.resize(4, 4);
  mixed_sign.corr << 1.0, 0.55, -0.30, 0.20,
                     0.55, 1.0, -0.25, 0.45,
                    -0.30, -0.25, 1.0, -0.50,
                     0.20, 0.45, -0.50, 1.0;
  mixed_sign.skew.resize(4);
  mixed_sign.skew << 2.0, 0.75, -1.5, 1.25;
  mixed_sign.kurt.resize(4);
  mixed_sign.kurt << 5.0, 1.25, 4.0, 3.0;
  cases.push_back(mixed_sign);

  return cases;
}

double sample_corr(const Eigen::MatrixXd& X, Eigen::Index a, Eigen::Index b) {
  const Eigen::VectorXd xa = X.col(a).array() - X.col(a).mean();
  const Eigen::VectorXd xb = X.col(b).array() - X.col(b).mean();
  return xa.dot(xb) / std::sqrt(xa.squaredNorm() * xb.squaredNorm());
}

double sample_skewness(const Eigen::MatrixXd& X, Eigen::Index j) {
  const Eigen::ArrayXd centered = X.col(j).array() - X.col(j).mean();
  const double var = centered.square().mean();
  return centered.pow(3).mean() / std::pow(var, 1.5);
}

double sample_excess_kurtosis(const Eigen::MatrixXd& X, Eigen::Index j) {
  const Eigen::ArrayXd centered = X.col(j).array() - X.col(j).mean();
  const double var = centered.square().mean();
  return centered.pow(4).mean() / (var * var) - 3.0;
}

double max_abs_offdiag(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  double out = 0.0;
  for (Eigen::Index i = 0; i < A.rows(); ++i) {
    for (Eigen::Index j = i + 1; j < A.cols(); ++j) {
      out = std::max(out, std::abs(A(i, j) - B(i, j)));
    }
  }
  return out;
}

double quadrature_residual(const magmaan::sim::PlsimCalibration& cal,
                           const Eigen::MatrixXd& target,
                           const magmaan::sim::PlsimOptions& options) {
  double out = 0.0;
  for (Eigen::Index i = 0; i < target.rows(); ++i) {
    for (Eigen::Index j = i + 1; j < target.cols(); ++j) {
      auto cov_or = magmaan::sim::plsim_covariance_quadrature(
          cal.marginals[static_cast<std::size_t>(i)],
          cal.marginals[static_cast<std::size_t>(j)],
          cal.intermediate_corr(i, j), options.quadrature_points);
      if (!cov_or.has_value()) return std::numeric_limits<double>::quiet_NaN();
      out = std::max(out, std::abs(*cov_or - target(i, j)));
    }
  }
  return out;
}

double rectangle_residual(const magmaan::sim::PlsimCalibration& cal,
                          const Eigen::MatrixXd& target,
                          const magmaan::sim::PlsimOptions& options) {
  double out = 0.0;
  for (Eigen::Index i = 0; i < target.rows(); ++i) {
    for (Eigen::Index j = i + 1; j < target.cols(); ++j) {
      auto cov_or = magmaan::sim::plsim_covariance_rectangle(
          cal.marginals[static_cast<std::size_t>(i)],
          cal.marginals[static_cast<std::size_t>(j)],
          cal.intermediate_corr(i, j), options);
      if (!cov_or.has_value()) return std::numeric_limits<double>::quiet_NaN();
      out = std::max(out, std::abs(*cov_or - target(i, j)));
    }
  }
  return out;
}

double sample_corr_error(const Eigen::MatrixXd& X, const Eigen::MatrixXd& target) {
  double out = 0.0;
  for (Eigen::Index i = 0; i < target.rows(); ++i) {
    for (Eigen::Index j = i + 1; j < target.cols(); ++j) {
      out = std::max(out, std::abs(sample_corr(X, i, j) - target(i, j)));
    }
  }
  return out;
}

double sample_shape_error(const Eigen::MatrixXd& X,
                          const Eigen::VectorXd& skew,
                          const Eigen::VectorXd& kurt) {
  double out = 0.0;
  for (Eigen::Index j = 0; j < X.cols(); ++j) {
    out = std::max(out, std::abs(sample_skewness(X, j) - skew(j)));
    out = std::max(out, std::abs(sample_excess_kurtosis(X, j) - kurt(j)));
  }
  return out;
}

MethodSummary run_method(const Case& c,
                         std::string name,
                         magmaan::sim::PlsimCovarianceMethod method,
                         const magmaan::sim::PlsimOptions& options,
                         const magmaan::sim::PlsimCalibration* reference,
                         const Config& cfg) {
  MethodSummary out;
  out.name = std::move(name);
  double total_ms = 0.0;
  magmaan::sim::PlsimCalibration last;

  for (int r = 0; r < cfg.reps; ++r) {
    const auto start = Clock::now();
    auto cal_or = magmaan::sim::calibrate_plsim(c.corr, c.skew, c.kurt, method, options);
    const auto stop = Clock::now();
    total_ms += std::chrono::duration<double, std::milli>(stop - start).count();
    if (!cal_or.has_value()) {
      ++out.failed;
      out.last_error = cal_or.error().detail;
      continue;
    }
    ++out.ok;
    last = std::move(*cal_or);
  }

  if (out.ok == 0) return out;
  out.mean_ms = total_ms / static_cast<double>(out.ok + out.failed);
  out.max_strategy_resid = max_abs_offdiag(last.achieved_corr, c.corr);
  out.max_quad_resid = quadrature_residual(last, c.corr, options);
  out.max_rect_resid = rectangle_residual(last, c.corr, options);
  if (reference != nullptr) {
    out.max_rho_delta =
        max_abs_offdiag(last.intermediate_corr, reference->intermediate_corr);
  }

  std::mt19937_64 rng(cfg.seed + static_cast<std::uint64_t>(17 * c.corr.rows()));
  auto X_or = magmaan::sim::simulate_plsim_matrix(cfg.sample_n, last, rng, options);
  if (X_or.has_value()) {
    out.sample_corr_err = sample_corr_error(*X_or, c.corr);
    out.sample_shape_err = sample_shape_error(*X_or, c.skew, c.kurt);
  }
  return out;
}

void print_summary(const Case& c, const std::vector<MethodSummary>& rows) {
  std::cout << "\ncase: " << c.name << " p=" << c.corr.rows() << "\n";
  std::cout << std::left << std::setw(23) << "method"
            << std::right << std::setw(8) << "ok"
            << std::setw(8) << "fail"
            << std::setw(12) << "ms"
            << std::setw(14) << "own_err"
            << std::setw(14) << "quad_err"
            << std::setw(14) << "rect_err"
            << std::setw(14) << "rho_delta"
            << std::setw(14) << "samp_corr"
            << std::setw(14) << "samp_shape" << "\n";
  for (const auto& r : rows) {
    std::cout << std::left << std::setw(23) << r.name
              << std::right << std::setw(8) << r.ok
              << std::setw(8) << r.failed
              << std::setw(12) << std::setprecision(4) << std::fixed << r.mean_ms
              << std::setw(14) << std::scientific << std::setprecision(3)
              << r.max_strategy_resid
              << std::setw(14) << r.max_quad_resid
              << std::setw(14) << r.max_rect_resid
              << std::setw(14) << r.max_rho_delta
              << std::setw(14) << r.sample_corr_err
              << std::setw(14) << r.sample_shape_err
              << std::defaultfloat << "\n";
    if (r.ok == 0 && !r.last_error.empty()) {
      std::cout << "  " << r.name << " failure: " << r.last_error << "\n";
    }
  }
}

void print_vm_status(const Case& c) {
  auto vm_or = magmaan::sim::calibrate_vale_maurelli(c.corr, c.skew, c.kurt);
  std::cout << "VM baseline: "
            << (vm_or.has_value() ? "calibrated" : vm_or.error().detail) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const Config cfg = parse_args(argc, argv);

  magmaan::sim::PlsimOptions options;
  options.hermite_order = cfg.hermite_order;
  options.quadrature_points = cfg.quadrature_points;
  options.correlation_tol = 1e-7;

  std::cout << "PLSIM calibration bench"
            << " reps=" << cfg.reps
            << " sample_n=" << cfg.sample_n
            << " hermite_order=" << options.hermite_order
            << " quadrature_points=" << options.quadrature_points << "\n";

  for (const auto& c : make_cases()) {
    print_vm_status(c);
    auto ref_or = magmaan::sim::calibrate_plsim(
        c.corr, c.skew, c.kurt, magmaan::sim::PlsimCovarianceMethod::Quadrature,
        options);
    const magmaan::sim::PlsimCalibration* ref =
        ref_or.has_value() ? &*ref_or : nullptr;

    std::vector<MethodSummary> rows;
    rows.push_back(run_method(
        c, "Hermite", magmaan::sim::PlsimCovarianceMethod::Hermite,
        options, ref, cfg));
    rows.push_back(run_method(
        c, "Quadrature", magmaan::sim::PlsimCovarianceMethod::Quadrature,
        options, ref, cfg));
    rows.push_back(run_method(
        c, "Rectangle", magmaan::sim::PlsimCovarianceMethod::Rectangle,
        options, ref, cfg));
    rows.push_back(run_method(
        c, "Hermite+Quadrature",
        magmaan::sim::PlsimCovarianceMethod::HermiteThenQuadrature,
        options, ref, cfg));
    rows.push_back(run_method(
        c, "Hermite+Rectangle",
        magmaan::sim::PlsimCovarianceMethod::HermiteThenRectangle,
        options, ref, cfg));
    print_summary(c, rows);
  }

  return 0;
}
