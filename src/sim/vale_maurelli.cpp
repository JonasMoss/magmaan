#include "magmaan/sim/vale_maurelli.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/QR>

namespace magmaan::sim {

namespace {

SimError make_err(SimError::Kind k, std::string detail) {
  return SimError{k, std::move(detail)};
}

bool approx_equal(double a, double b, double tol = 1e-12) noexcept {
  return std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)});
}

double normal_raw_moment(int k) noexcept {
  if (k < 0 || (k % 2) != 0) return 0.0;
  double out = 1.0;
  for (int j = 1; j <= k - 1; j += 2) out *= static_cast<double>(j);
  return out;
}

std::array<double, 13>
poly_power_moments(const FleishmanCoefficients& coef, int power) {
  std::array<double, 13> poly{};
  poly[0] = 1.0;
  int degree = 0;
  const std::array<double, 4> base{coef.a, coef.b, coef.c, coef.d};
  for (int p = 0; p < power; ++p) {
    std::array<double, 13> next{};
    for (int i = 0; i <= degree; ++i) {
      for (int j = 0; j <= 3; ++j) {
        next[static_cast<std::size_t>(i + j)] +=
            poly[static_cast<std::size_t>(i)] * base[static_cast<std::size_t>(j)];
      }
    }
    poly = next;
    degree += 3;
  }
  return poly;
}

double polynomial_expectation(const FleishmanCoefficients& coef, int power) {
  const auto poly = poly_power_moments(coef, power);
  double out = 0.0;
  for (int k = 0; k <= 12; ++k) {
    out += poly[static_cast<std::size_t>(k)] * normal_raw_moment(k);
  }
  return out;
}

sim_expected<Eigen::Vector3d>
fleishman_residual(double b,
                   double c,
                   double d,
                   double target_skewness,
                   double target_excess_kurtosis) {
  if (!std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d)) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "fit_fleishman_coefficients: non-finite coefficient iterate"));
  }

  const FleishmanCoefficients coef{-c, b, c, d};
  const double m1 = polynomial_expectation(coef, 1);
  const double m2 = polynomial_expectation(coef, 2);
  const double m3 = polynomial_expectation(coef, 3);
  const double m4 = polynomial_expectation(coef, 4);
  const double var = m2 - m1 * m1;
  if (!(var > 0.0) || !std::isfinite(var)) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "fit_fleishman_coefficients: variance became non-positive"));
  }
  const double sd = std::sqrt(var);
  const double mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1;
  const double mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 -
                    3.0 * m1 * m1 * m1 * m1;
  Eigen::Vector3d out;
  out(0) = var - 1.0;
  out(1) = mu3 / (sd * sd * sd) - target_skewness;
  out(2) = mu4 / (var * var) - 3.0 - target_excess_kurtosis;
  if (!out.allFinite()) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "fit_fleishman_coefficients: residual became non-finite"));
  }
  return out;
}

double residual_norm(const Eigen::Vector3d& r) noexcept {
  return std::max({std::abs(r(0)), std::abs(r(1)), std::abs(r(2))});
}

sim_expected<FleishmanCoefficients>
solve_fleishman_from_start(double skewness,
                           double excess_kurtosis,
                           const ValeMaurelliOptions& options,
                           const Eigen::Vector3d& start) {
  Eigen::Vector3d x = start;
  double best_norm = std::numeric_limits<double>::infinity();
  Eigen::Vector3d best_x = x;

  for (int iter = 0; iter < options.max_iter; ++iter) {
    auto r_or = fleishman_residual(x(0), x(1), x(2), skewness, excess_kurtosis);
    if (!r_or.has_value()) return std::unexpected(r_or.error());
    const Eigen::Vector3d r = *r_or;
    const double norm = residual_norm(r);
    if (norm < best_norm) {
      best_norm = norm;
      best_x = x;
    }
    if (norm <= options.coefficient_tol) {
      return FleishmanCoefficients{-x(1), x(0), x(1), x(2)};
    }

    Eigen::Matrix3d J;
    for (int j = 0; j < 3; ++j) {
      Eigen::Vector3d xp = x;
      const double h = 1e-5 * std::max(1.0, std::abs(x(j)));
      xp(j) += h;
      auto rp_or = fleishman_residual(
          xp(0), xp(1), xp(2), skewness, excess_kurtosis);
      if (!rp_or.has_value()) return std::unexpected(rp_or.error());
      J.col(j) = (*rp_or - r) / h;
    }

    const Eigen::Vector3d step = J.colPivHouseholderQr().solve(r);
    if (!step.allFinite()) {
      break;
    }

    bool accepted = false;
    double scale = 1.0;
    for (int ls = 0; ls < 24; ++ls) {
      const Eigen::Vector3d candidate = x - scale * step;
      auto rc_or = fleishman_residual(
          candidate(0), candidate(1), candidate(2), skewness, excess_kurtosis);
      if (rc_or.has_value() && residual_norm(*rc_or) < norm) {
        x = candidate;
        accepted = true;
        break;
      }
      scale *= 0.5;
    }
    if (!accepted) break;
  }

  if (best_norm <= std::max(options.coefficient_tol, 1e-8)) {
    return FleishmanCoefficients{-best_x(1), best_x(0), best_x(1), best_x(2)};
  }
  return std::unexpected(make_err(
      SimError::Kind::CalibrationFailed,
      "fit_fleishman_coefficients: Fleishman moment equations did not converge"));
}

sim_expected<void>
validate_options(const ValeMaurelliOptions& options) {
  if (options.max_iter <= 0 ||
      !std::isfinite(options.coefficient_tol) || options.coefficient_tol <= 0.0 ||
      !std::isfinite(options.correlation_tol) || options.correlation_tol <= 0.0 ||
      !std::isfinite(options.rho_bound) || options.rho_bound <= 0.0 ||
      options.rho_bound >= 1.0 ||
      !std::isfinite(options.cholesky_jitter) || options.cholesky_jitter < 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "Vale-Maurelli options are outside their valid ranges"));
  }
  return {};
}

void maybe_add_grid_start(std::vector<Eigen::Vector3d>& starts,
                          double skewness,
                          double excess_kurtosis,
                          double c,
                          double d) {
  const double disc = 1.0 - 2.0 * c * c - 6.0 * d * d;
  if (!(disc > 0.0)) return;
  for (double sign : {-1.0, 1.0}) {
    const double b = -3.0 * d + sign * std::sqrt(disc);
    if (!(b > 0.0) || !std::isfinite(b)) continue;
    auto r_or = fleishman_residual(b, c, d, skewness, excess_kurtosis);
    if (!r_or.has_value()) continue;
    const double norm = residual_norm(*r_or);
    auto pos = starts.end();
    for (auto it = starts.begin(); it != starts.end(); ++it) {
      auto old_or = fleishman_residual(
          (*it)(0), (*it)(1), (*it)(2), skewness, excess_kurtosis);
      if (old_or.has_value() && norm < residual_norm(*old_or)) {
        pos = it;
        break;
      }
    }
    starts.insert(pos, Eigen::Vector3d{b, c, d});
    if (starts.size() > 16u) starts.pop_back();
  }
}

sim_expected<void>
validate_vm_inputs(const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
                   const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
                   const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
                   const ValeMaurelliOptions& options) {
  if (auto ok = validate_options(options); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (target_corr.rows() != target_corr.cols()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_vale_maurelli: target_corr must be square"));
  }
  if (target_corr.rows() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_vale_maurelli: target_corr must not be empty"));
  }
  if (target_skewness.size() != target_corr.rows() ||
      target_excess_kurtosis.size() != target_corr.rows()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_vale_maurelli: moment vectors must match target_corr size"));
  }
  if (!target_corr.allFinite() || !target_skewness.allFinite() ||
      !target_excess_kurtosis.allFinite()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_vale_maurelli: inputs must be finite"));
  }
  for (Eigen::Index i = 0; i < target_corr.rows(); ++i) {
    if (!approx_equal(target_corr(i, i), 1.0, 1e-10)) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "calibrate_vale_maurelli: target_corr diagonal must equal one"));
    }
    if (target_excess_kurtosis(i) < target_skewness(i) * target_skewness(i) - 2.0) {
      return std::unexpected(make_err(
          SimError::Kind::CalibrationFailed,
          "calibrate_vale_maurelli: requested marginal moments are infeasible"));
    }
    for (Eigen::Index j = i + 1; j < target_corr.cols(); ++j) {
      if (!approx_equal(target_corr(i, j), target_corr(j, i), 1e-10)) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "calibrate_vale_maurelli: target_corr must be symmetric"));
      }
      if (std::abs(target_corr(i, j)) >= 1.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "calibrate_vale_maurelli: correlations must be strictly inside (-1, 1)"));
      }
    }
  }
  Eigen::LLT<Eigen::MatrixXd> llt(target_corr);
  if (llt.info() != Eigen::Success) {
    return std::unexpected(make_err(
        SimError::Kind::NonPositiveDefinite,
        "calibrate_vale_maurelli: target_corr is not positive definite"));
  }
  return {};
}

sim_expected<double>
solve_intermediate_corr(const FleishmanCoefficients& left,
                        const FleishmanCoefficients& right,
                        double target,
                        const ValeMaurelliOptions& options) {
  auto f_at = [&](double rho) -> double {
    auto cov_or = fleishman_covariance(left, right, rho);
    return cov_or.has_value() ? *cov_or - target
                              : std::numeric_limits<double>::quiet_NaN();
  };

  const double lo0 = -options.rho_bound;
  const double hi0 = options.rho_bound;
  double lo = lo0;
  double hi = hi0;
  double flo = f_at(lo);
  double fhi = f_at(hi);
  if (!std::isfinite(flo) || !std::isfinite(fhi) || flo * fhi > 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "calibrate_vale_maurelli: no intermediate correlation root in bounds"));
  }
  if (std::abs(flo) <= options.correlation_tol) return lo;
  if (std::abs(fhi) <= options.correlation_tol) return hi;

  for (int iter = 0; iter < options.max_iter; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double fmid = f_at(mid);
    if (!std::isfinite(fmid)) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "calibrate_vale_maurelli: non-finite correlation calibration residual"));
    }
    if (std::abs(fmid) <= options.correlation_tol ||
        std::abs(hi - lo) <= options.correlation_tol) {
      return mid;
    }
    if (flo * fmid <= 0.0) {
      hi = mid;
      fhi = fmid;
      (void)fhi;
    } else {
      lo = mid;
      flo = fmid;
    }
  }
  return 0.5 * (lo + hi);
}

sim_expected<Eigen::MatrixXd>
cholesky_factor_with_jitter(const Eigen::Ref<const Eigen::MatrixXd>& corr,
                            double jitter) {
  Eigen::LLT<Eigen::MatrixXd> llt(corr);
  if (llt.info() == Eigen::Success) return llt.matrixL();
  if (jitter > 0.0) {
    Eigen::MatrixXd adjusted = corr;
    adjusted.diagonal().array() += jitter;
    Eigen::LLT<Eigen::MatrixXd> llt_j(adjusted);
    if (llt_j.info() == Eigen::Success) return llt_j.matrixL();
  }
  return std::unexpected(make_err(
      SimError::Kind::NonPositiveDefinite,
      "simulate_vale_maurelli: intermediate correlation is not positive definite"));
}

double apply_fleishman(const FleishmanCoefficients& coef, double z) noexcept {
  return coef.a + z * (coef.b + z * (coef.c + z * coef.d));
}

}  // namespace

sim_expected<FleishmanCoefficients>
fit_fleishman_coefficients(double skewness,
                           double excess_kurtosis,
                           const ValeMaurelliOptions& options) {
  if (auto ok = validate_options(options); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (!std::isfinite(skewness) || !std::isfinite(excess_kurtosis)) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "fit_fleishman_coefficients: target moments must be finite"));
  }
  if (excess_kurtosis < skewness * skewness - 2.0) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_fleishman_coefficients: requested moments are infeasible"));
  }
  if (approx_equal(skewness, 0.0, options.coefficient_tol) &&
      approx_equal(excess_kurtosis, 0.0, options.coefficient_tol)) {
    return FleishmanCoefficients{};
  }

  std::vector<Eigen::Vector3d> starts{
      {1.0, skewness / 6.0, excess_kurtosis / 24.0},
      {1.0, skewness / 6.0, 0.0},
      {0.9, skewness / 8.0, excess_kurtosis / 36.0},
      {0.75, skewness / 10.0, excess_kurtosis / 48.0},
      {1.1, skewness / 6.0, excess_kurtosis / 60.0},
      {0.5, 0.0, std::max(0.0, excess_kurtosis) / 72.0}};

  for (int ic = -20; ic <= 20; ++ic) {
    const double c = static_cast<double>(ic) / 20.0;
    for (int id = -20; id <= 20; ++id) {
      const double d = static_cast<double>(id) / 120.0;
      maybe_add_grid_start(starts, skewness, excess_kurtosis, c, d);
    }
  }

  SimError last_error{SimError::Kind::CalibrationFailed,
                      "fit_fleishman_coefficients: Fleishman moment equations did not converge"};
  for (const auto& start : starts) {
    auto fit_or = solve_fleishman_from_start(
        skewness, excess_kurtosis, options, start);
    if (fit_or.has_value()) return fit_or;
    last_error = fit_or.error();
  }
  return std::unexpected(last_error);
}

sim_expected<double>
fleishman_covariance(const FleishmanCoefficients& left,
                     const FleishmanCoefficients& right,
                     double rho) {
  if (!std::isfinite(left.a) || !std::isfinite(left.b) ||
      !std::isfinite(left.c) || !std::isfinite(left.d) ||
      !std::isfinite(right.a) || !std::isfinite(right.b) ||
      !std::isfinite(right.c) || !std::isfinite(right.d) ||
      !std::isfinite(rho) || std::abs(rho) >= 1.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "fleishman_covariance: coefficients and rho must be finite with |rho| < 1"));
  }
  const double rho2 = rho * rho;
  const double rho3 = rho2 * rho;
  return rho * (left.b * right.b + 3.0 * left.b * right.d +
                3.0 * left.d * right.b + 9.0 * left.d * right.d) +
         2.0 * left.c * right.c * rho2 +
         6.0 * left.d * right.d * rho3;
}

sim_expected<ValeMaurelliCalibration>
calibrate_vale_maurelli(
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
    const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
    const ValeMaurelliOptions& options) {
  if (auto ok = validate_vm_inputs(
          target_corr, target_skewness, target_excess_kurtosis, options);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  const Eigen::Index p = target_corr.rows();
  ValeMaurelliCalibration out;
  out.coefficients.reserve(static_cast<std::size_t>(p));
  for (Eigen::Index j = 0; j < p; ++j) {
    auto coef_or = fit_fleishman_coefficients(
        target_skewness(j), target_excess_kurtosis(j), options);
    if (!coef_or.has_value()) return std::unexpected(coef_or.error());
    out.coefficients.push_back(*coef_or);
  }

  out.intermediate_corr = Eigen::MatrixXd::Identity(p, p);
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = i + 1; j < p; ++j) {
      auto rho_or = solve_intermediate_corr(
          out.coefficients[static_cast<std::size_t>(i)],
          out.coefficients[static_cast<std::size_t>(j)],
          target_corr(i, j),
          options);
      if (!rho_or.has_value()) return std::unexpected(rho_or.error());
      out.intermediate_corr(i, j) = *rho_or;
      out.intermediate_corr(j, i) = *rho_or;
    }
  }

  auto root_or = cholesky_factor_with_jitter(
      out.intermediate_corr, options.cholesky_jitter);
  if (!root_or.has_value()) return std::unexpected(root_or.error());
  return out;
}

sim_expected<Eigen::MatrixXd>
simulate_vale_maurelli_matrix(
    Eigen::Index n,
    const ValeMaurelliCalibration& calibration,
    std::mt19937_64& rng,
    const ValeMaurelliOptions& options) {
  if (auto ok = validate_options(options); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (n <= 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_vale_maurelli_matrix: n must be positive"));
  }
  const Eigen::Index p =
      static_cast<Eigen::Index>(calibration.coefficients.size());
  if (p == 0 || calibration.intermediate_corr.rows() != p ||
      calibration.intermediate_corr.cols() != p ||
      !calibration.intermediate_corr.allFinite()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_vale_maurelli_matrix: invalid calibration"));
  }

  auto root_or = cholesky_factor_with_jitter(
      calibration.intermediate_corr, options.cholesky_jitter);
  if (!root_or.has_value()) return std::unexpected(root_or.error());
  const Eigen::MatrixXd& root = *root_or;

  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd Z(n, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < p; ++j) Z(i, j) = normal(rng);
  }
  Z = Z * root.transpose();

  Eigen::MatrixXd X(n, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < p; ++j) {
      X(i, j) = apply_fleishman(
          calibration.coefficients[static_cast<std::size_t>(j)], Z(i, j));
    }
  }
  return X;
}

sim_expected<Eigen::MatrixXd>
simulate_vale_maurelli_matrix(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
    const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
    std::mt19937_64& rng,
    const ValeMaurelliOptions& options) {
  auto cal_or = calibrate_vale_maurelli(
      target_corr, target_skewness, target_excess_kurtosis, options);
  if (!cal_or.has_value()) return std::unexpected(cal_or.error());
  return simulate_vale_maurelli_matrix(n, *cal_or, rng, options);
}

sim_expected<data::RawData>
simulate_vale_maurelli_raw(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
    const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
    std::mt19937_64& rng,
    const ValeMaurelliOptions& options) {
  auto X_or = simulate_vale_maurelli_matrix(
      n, target_corr, target_skewness, target_excess_kurtosis, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  data::RawData raw;
  raw.X.push_back(std::move(*X_or));
  return raw;
}

}  // namespace magmaan::sim
