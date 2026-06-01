#include "magmaan/sim/norta.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "magmaan/sim/vale_maurelli.hpp"

#include "../detail_distribution_math.hpp"

namespace magmaan::sim {

namespace {

constexpr double kSqrt2 = 1.41421356237309504880;
constexpr double kSqrtPi = 1.77245385090551602730;

SimError make_err(SimError::Kind k, std::string detail) {
  return SimError{k, std::move(detail)};
}

struct GaussHermite {
  Eigen::VectorXd nodes;
  Eigen::VectorXd weights;
};

struct GaussLegendre {
  Eigen::VectorXd nodes;
  Eigen::VectorXd weights;
};

struct MarginalMoments {
  double mean = 0.0;
  double sd = 1.0;
};

struct MomentFitEval {
  MarginalSpec marginal;
  MarginalMomentSummary moments;
  Eigen::Vector2d residual = Eigen::Vector2d::Zero();
  double norm = 0.0;
};

struct PearsonIvKernel {
  double m = 0.0;
  double nu = 0.0;
  double log_peak = 0.0;

  double operator()(double theta) const noexcept {
    const double c = std::cos(theta);
    if (!(c > 0.0)) return 0.0;
    const double alpha = 2.0 * m - 2.0;
    return std::exp(-nu * theta + alpha * std::log(c) - log_peak);
  }
};

using detail::inverse_gamma_p;
using detail::inverse_regularized_beta;
using detail::nan;
using detail::student_t_cdf;
using detail::student_t_quantile;

bool approx_equal(double a, double b, double tol = 1e-12) noexcept {
  return std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)});
}

template <typename F>
double simpson_panel(const F& f, double a, double b) {
  const double m = 0.5 * (a + b);
  return (b - a) / 6.0 * (f(a) + 4.0 * f(m) + f(b));
}

template <typename F>
double adaptive_simpson_aux(const F& f,
                            double a,
                            double b,
                            double whole,
                            double abs_tol,
                            int depth) {
  const double m = 0.5 * (a + b);
  const double left = simpson_panel(f, a, m);
  const double right = simpson_panel(f, m, b);
  const double refined = left + right;
  const double delta = refined - whole;
  if (depth <= 0 || std::abs(delta) <= 15.0 * abs_tol) {
    return refined + delta / 15.0;
  }
  return adaptive_simpson_aux(f, a, m, left, 0.5 * abs_tol, depth - 1) +
         adaptive_simpson_aux(f, m, b, right, 0.5 * abs_tol, depth - 1);
}

template <typename F>
double adaptive_simpson(const F& f, double a, double b, double abs_tol) {
  if (!(b > a)) return 0.0;
  const double whole = simpson_panel(f, a, b);
  return adaptive_simpson_aux(f, a, b, whole, abs_tol, 28);
}

double pearson_type4_log_peak(double m, double nu) noexcept {
  const double alpha = 2.0 * m - 2.0;
  const double theta = std::atan(-nu / alpha);
  return -nu * theta + alpha * std::log(std::cos(theta));
}

double pearson_type4_integral(const PearsonIvKernel& kernel,
                              double a,
                              double b) {
  constexpr double tol = 2e-12;
  const double val = adaptive_simpson(kernel, a, b, tol);
  return std::max(0.0, val);
}

double pearson_type4_cdf_theta(double theta,
                               const PearsonIvKernel& kernel,
                               double total,
                               bool integrate_lower_tail) {
  constexpr double lo = -0.5 * std::numbers::pi;
  constexpr double hi = 0.5 * std::numbers::pi;
  if (theta <= lo) return 0.0;
  if (theta >= hi) return 1.0;
  if (!(total > 0.0) || !std::isfinite(total)) return nan();
  if (integrate_lower_tail) {
    return pearson_type4_integral(kernel, lo, theta) / total;
  }
  return 1.0 - pearson_type4_integral(kernel, theta, hi) / total;
}

double pearson_type4_quantile(double p,
                              double m,
                              double nu,
                              double location,
                              double scale) {
  if (!(p > 0.0) || !(p < 1.0) || !(m > 1.0) || !(scale > 0.0)) {
    return nan();
  }
  constexpr double lo0 = -0.5 * std::numbers::pi;
  constexpr double hi0 = 0.5 * std::numbers::pi;
  const PearsonIvKernel kernel{m, nu, pearson_type4_log_peak(m, nu)};
  const double total = pearson_type4_integral(kernel, lo0, hi0);
  if (!(total > 0.0) || !std::isfinite(total)) return nan();

  double lo = lo0;
  double hi = hi0;
  double mid = 0.0;
  const bool integrate_lower_tail = p <= 0.5;
  for (int iter = 0; iter < 90; ++iter) {
    mid = 0.5 * (lo + hi);
    const double cdf = pearson_type4_cdf_theta(
        mid, kernel, total, integrate_lower_tail);
    if (!std::isfinite(cdf)) return nan();
    if (cdf < p) lo = mid;
    else hi = mid;
  }
  const double theta = 0.5 * (lo + hi);
  return location + scale * std::tan(theta);
}

sim_expected<GaussHermite> gauss_hermite(int n) {
  if (n < 8) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "gauss_hermite: quadrature_points must be at least 8"));
  }

  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
  for (int i = 1; i < n; ++i) {
    const double off = std::sqrt(static_cast<double>(i) / 2.0);
    J(i - 1, i) = off;
    J(i, i - 1) = off;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
  if (es.info() != Eigen::Success) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "gauss_hermite: eigensolve failed"));
  }

  GaussHermite gh;
  gh.nodes = es.eigenvalues();
  gh.weights.resize(n);
  const auto V = es.eigenvectors();
  for (int i = 0; i < n; ++i) {
    gh.weights(i) = kSqrtPi * V(0, i) * V(0, i);
  }
  return gh;
}

sim_expected<GaussLegendre> gauss_legendre_unit(int n) {
  if (n < 8) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "gauss_legendre_unit: quadrature_points must be at least 8"));
  }

  GaussLegendre gl;
  gl.nodes.resize(n);
  gl.weights.resize(n);
  const int m = (n + 1) / 2;
  constexpr double eps = 1e-14;
  for (int i = 0; i < m; ++i) {
    double z = std::cos(std::numbers::pi *
                        (static_cast<double>(i) + 0.75) /
                        (static_cast<double>(n) + 0.5));
    double pp = 0.0;
    for (int iter = 0; iter < 80; ++iter) {
      double p1 = 1.0;
      double p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * static_cast<double>(j) - 1.0) * z * p2 -
              (static_cast<double>(j) - 1.0) * p3) /
             static_cast<double>(j);
      }
      pp = static_cast<double>(n) * (z * p1 - p2) / (z * z - 1.0);
      const double z_next = z - p1 / pp;
      if (std::abs(z_next - z) <= eps) {
        z = z_next;
        break;
      }
      z = z_next;
    }
    if (!std::isfinite(z) || !std::isfinite(pp) || pp == 0.0) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "gauss_legendre_unit: root solve failed"));
    }
    const double w = 2.0 / ((1.0 - z * z) * pp * pp);
    const int lo = i;
    const int hi = n - 1 - i;
    gl.nodes(lo) = 0.5 * (1.0 - z);
    gl.nodes(hi) = 0.5 * (1.0 + z);
    gl.weights(lo) = 0.5 * w;
    gl.weights(hi) = 0.5 * w;
  }
  return gl;
}

sim_expected<void> validate_marginal(const MarginalSpec& m) {
  if (!std::isfinite(m.mean) || !std::isfinite(m.sd) || m.sd <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidMarginal,
        "marginal: mean/sd must be finite and sd must be positive"));
  }
  switch (m.kind) {
    case MarginalKind::StandardNormal:
      return {};
    case MarginalKind::StandardizedLognormal:
      if (!std::isfinite(m.sigma_log) || m.sigma_log <= 0.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidMarginal,
            "standardized lognormal: sigma_log must be positive and finite"));
      }
      if (m.sigma_log > 5.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidMarginal,
            "standardized lognormal: sigma_log is too large for stable moments"));
      }
      return {};
    case MarginalKind::TukeyGH:
      if (!std::isfinite(m.g) || !std::isfinite(m.h) ||
          m.h < 0.0 || m.h >= 0.25) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidMarginal,
            "Tukey g-and-h: g must be finite and h must satisfy 0 <= h < 0.25"));
      }
      return {};
    case MarginalKind::Pearson:
      if (!std::isfinite(m.pearson_p1) || !std::isfinite(m.pearson_p2) ||
          !std::isfinite(m.pearson_p3) || !std::isfinite(m.pearson_p4)) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidMarginal,
            "Pearson: parameters must be finite"));
      }
      switch (m.pearson_type) {
        case 0:
          if (m.pearson_p2 > 0.0) return {};
          break;
        case 1:
          if (m.pearson_p1 > 0.0 && m.pearson_p2 > 0.0 &&
              m.pearson_p4 != 0.0) return {};
          break;
        case 2:
          if (m.pearson_p1 > 0.0 && m.pearson_p3 != 0.0) return {};
          break;
        case 3:
        case 5:
          if (m.pearson_p1 > 0.0 && m.pearson_p3 != 0.0) return {};
          break;
        case 4:
          if (m.pearson_p1 > 1.0 && m.pearson_p4 > 0.0) return {};
          break;
        case 6:
          if (m.pearson_p1 > 0.0 && m.pearson_p2 > 0.0 &&
              m.pearson_p4 != 0.0) return {};
          break;
        case 7:
          if (m.pearson_p1 > 0.0 && m.pearson_p3 != 0.0) return {};
          break;
        default:
          break;
      }
      return std::unexpected(make_err(
          SimError::Kind::InvalidMarginal,
          "Pearson: unsupported or invalid PearsonDS parameterization"));
    case MarginalKind::Johnson:
      if (!std::isfinite(m.johnson_gamma) ||
          !std::isfinite(m.johnson_delta) ||
          m.johnson_delta <= 0.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidMarginal,
            "Johnson: gamma must be finite and delta must be positive and finite"));
      }
      if (m.johnson_type == 1 || m.johnson_type == 2 ||
          m.johnson_type == 3) {
        return {};
      }
      return std::unexpected(make_err(
          SimError::Kind::InvalidMarginal,
          "Johnson: type must be 1 (SL), 2 (SU), or 3 (SB)"));
    case MarginalKind::Fleishman:
      if (!std::isfinite(m.fleishman_b) ||
          !std::isfinite(m.fleishman_c) ||
          !std::isfinite(m.fleishman_d)) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidMarginal,
            "Fleishman: polynomial coefficients must be finite"));
      }
      return {};
  }
  return std::unexpected(make_err(
      SimError::Kind::InvalidMarginal,
      "marginal: unknown marginal kind"));
}

double normal_from_z(const MarginalSpec&, double z) {
  return z;
}

double standardized_lognormal_from_z(const MarginalSpec& m, double z) {
  const double s2 = m.sigma_log * m.sigma_log;
  const double raw = std::exp(m.sigma_log * z);
  const double mean = std::exp(0.5 * s2);
  const double var = (std::exp(s2) - 1.0) * std::exp(s2);
  return (raw - mean) / std::sqrt(var);
}

double tukey_g_h_raw_from_z(const MarginalSpec& m, double z) {
  const double tg = (std::abs(m.g) < 1e-10)
      ? z
      : (std::exp(m.g * z) - 1.0) / m.g;
  return tg * std::exp(0.5 * m.h * z * z);
}

double qbeta_with_scale(double p,
                        double a,
                        double b,
                        double location,
                        double scale) {
  const double pp = scale < 0.0 ? 1.0 - p : p;
  const double q = inverse_regularized_beta(pp, a, b);
  return location + scale * q;
}

double pearson_raw_from_z(const MarginalSpec& m, double z) {
  const double p = normal_cdf(z);
  switch (m.pearson_type) {
    case 0:
      return m.pearson_p1 + m.pearson_p2 * z;
    case 1:
      return qbeta_with_scale(
          p, m.pearson_p1, m.pearson_p2, m.pearson_p3, m.pearson_p4);
    case 2:
      return qbeta_with_scale(
          p, m.pearson_p1, m.pearson_p1, m.pearson_p2, m.pearson_p3);
    case 3: {
      const double scale = m.pearson_p3;
      const double pp = scale < 0.0 ? 1.0 - p : p;
      const double q = inverse_gamma_p(pp, m.pearson_p1, std::abs(scale));
      return m.pearson_p2 + std::copysign(q, scale);
    }
    case 4:
      return pearson_type4_quantile(
          p, m.pearson_p1, m.pearson_p2, m.pearson_p3, m.pearson_p4);
    case 5: {
      const double scale = m.pearson_p3;
      const double pp = scale > 0.0 ? 1.0 - p : p;
      const double q = inverse_gamma_p(pp, m.pearson_p1, 1.0 / std::abs(scale));
      return m.pearson_p2 + std::copysign(1.0 / q, scale);
    }
    case 6: {
      const double scale = m.pearson_p4 * m.pearson_p1 / m.pearson_p2;
      const double pp = scale < 0.0 ? 1.0 - p : p;
      const double y = inverse_regularized_beta(
          pp, m.pearson_p1, m.pearson_p2);
      const double f = (2.0 * m.pearson_p2 * y) /
                       (2.0 * m.pearson_p1 * (1.0 - y));
      return m.pearson_p3 + scale * f;
    }
    case 7: {
      const double scale = m.pearson_p3;
      const double pp = scale < 0.0 ? 1.0 - p : p;
      return m.pearson_p2 + scale * student_t_quantile(pp, m.pearson_p1);
    }
    default:
      return nan();
  }
}

double logistic_stable(double x) {
  if (x >= 0.0) {
    const double e = std::exp(-x);
    return 1.0 / (1.0 + e);
  }
  const double e = std::exp(x);
  return e / (1.0 + e);
}

double johnson_raw_from_z(const MarginalSpec& m, double z) {
  const double y = (z - m.johnson_gamma) / m.johnson_delta;
  switch (m.johnson_type) {
    case 1:
      return std::exp(y);
    case 2:
      return std::sinh(y);
    case 3:
      return logistic_stable(y);
    default:
      return nan();
  }
}

double fleishman_raw_from_z(const MarginalSpec& m, double z) {
  const double z2 = z * z;
  return -m.fleishman_c + m.fleishman_b * z + m.fleishman_c * z2 +
         m.fleishman_d * z2 * z;
}

double raw_from_z(const MarginalSpec& m, double z) {
  switch (m.kind) {
    case MarginalKind::StandardNormal:
      return normal_from_z(m, z);
    case MarginalKind::StandardizedLognormal:
      return standardized_lognormal_from_z(m, z);
    case MarginalKind::TukeyGH:
      return tukey_g_h_raw_from_z(m, z);
    case MarginalKind::Pearson:
      return pearson_raw_from_z(m, z);
    case MarginalKind::Johnson:
      return johnson_raw_from_z(m, z);
    case MarginalKind::Fleishman:
      return fleishman_raw_from_z(m, z);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

sim_expected<MarginalMoments>
marginal_moments(const MarginalSpec& m, const GaussHermite& gh) {
  if (auto ok = validate_marginal(m); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (m.kind == MarginalKind::StandardNormal ||
      m.kind == MarginalKind::StandardizedLognormal) {
    return MarginalMoments{0.0, 1.0};
  }
  if (m.kind == MarginalKind::Pearson) {
    return MarginalMoments{m.mean, m.sd};
  }

  double e1 = 0.0;
  double e2 = 0.0;
  for (Eigen::Index a = 0; a < gh.nodes.size(); ++a) {
    const double z = kSqrt2 * gh.nodes(a);
    const double w = gh.weights(a) / kSqrtPi;
    const double y = raw_from_z(m, z);
    if (!std::isfinite(y)) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "marginal moments: non-finite marginal transform"));
    }
    e1 += w * y;
    e2 += w * y * y;
  }
  const double var = e2 - e1 * e1;
  if (!std::isfinite(var) || var <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "marginal moments: non-positive variance"));
  }
  return MarginalMoments{e1, std::sqrt(var)};
}

double standardized_from_z(const MarginalSpec& m,
                           const MarginalMoments& moments,
                           double z) {
  return (raw_from_z(m, z) - moments.mean) / moments.sd;
}

double open_unit(double u) noexcept {
  constexpr double lo = std::numeric_limits<double>::min();
  constexpr double hi = 1.0 - std::numeric_limits<double>::epsilon();
  if (u <= 0.0) return lo;
  if (u >= 1.0) return hi;
  return u;
}

sim_expected<double>
marginal_from_unit(const MarginalSpec& marginal,
                   const MarginalMoments& moments,
                   double u) {
  auto z_or = normal_quantile(open_unit(u));
  if (!z_or.has_value()) return std::unexpected(z_or.error());
  const double y = standardized_from_z(marginal, moments, *z_or);
  if (!std::isfinite(y)) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "copula simulation: non-finite marginal transform"));
  }
  return marginal.mean + marginal.sd * y;
}

sim_expected<void>
validate_target(const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
                const std::vector<MarginalSpec>& marginals,
                const NortaOptions& options) {
  if (target_corr.rows() != target_corr.cols()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_norta: target_corr must be square"));
  }
  if (target_corr.rows() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_norta: target_corr must not be empty"));
  }
  if (static_cast<Eigen::Index>(marginals.size()) != target_corr.rows()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_norta: marginal count must match target_corr dimension"));
  }
  if (options.rho_bound <= 0.0 || options.rho_bound >= 1.0 ||
      options.max_bisection_iter < 8 || options.calibration_tol <= 0.0 ||
      options.cholesky_jitter < 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_norta: invalid calibration options"));
  }

  constexpr double sym_tol = 1e-12;
  constexpr double diag_tol = 1e-12;
  for (Eigen::Index i = 0; i < target_corr.rows(); ++i) {
    if (!std::isfinite(target_corr(i, i)) ||
        std::abs(target_corr(i, i) - 1.0) > diag_tol) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "calibrate_norta: target_corr diagonal must equal 1"));
    }
    for (Eigen::Index j = 0; j < target_corr.cols(); ++j) {
      if (!std::isfinite(target_corr(i, j)) ||
          std::abs(target_corr(i, j) - target_corr(j, i)) > sym_tol) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "calibrate_norta: target_corr must be finite and symmetric"));
      }
      if (i != j && std::abs(target_corr(i, j)) >= 1.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "calibrate_norta: off-diagonal target correlations must satisfy |r| < 1"));
      }
    }
  }
  return {};
}

sim_expected<void>
validate_independent_inputs(Eigen::Index n,
                            const std::vector<MarginalSpec>& marginals,
                            const IndependentOptions& options) {
  if (n <= 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_independent_matrix: n must be positive"));
  }
  if (marginals.empty()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_independent_matrix: marginals must not be empty"));
  }
  if (options.quadrature_points < 8) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_independent_matrix: quadrature_points must be at least 8"));
  }
  for (const auto& marginal : marginals) {
    if (auto ok = validate_marginal(marginal); !ok.has_value()) {
      return std::unexpected(ok.error());
    }
  }
  return {};
}

sim_expected<void>
validate_t_copula_inputs(Eigen::Index n,
                         const TCopulaSpec& copula,
                         const std::vector<MarginalSpec>& marginals,
                         const TCopulaOptions& options) {
  if (n <= 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_t_copula_matrix: n must be positive"));
  }
  if (!std::isfinite(copula.df) || copula.df <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_t_copula_matrix: df must be positive and finite"));
  }
  if (copula.corr.rows() != copula.corr.cols()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_t_copula_matrix: copula correlation must be square"));
  }
  if (copula.corr.rows() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_t_copula_matrix: copula correlation must not be empty"));
  }
  if (static_cast<Eigen::Index>(marginals.size()) != copula.corr.rows()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_t_copula_matrix: marginal count must match copula dimension"));
  }
  if (options.quadrature_points < 8 || options.cholesky_jitter < 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_t_copula_matrix: invalid options"));
  }

  constexpr double sym_tol = 1e-12;
  constexpr double diag_tol = 1e-12;
  for (Eigen::Index i = 0; i < copula.corr.rows(); ++i) {
    if (!std::isfinite(copula.corr(i, i)) ||
        std::abs(copula.corr(i, i) - 1.0) > diag_tol) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "simulate_t_copula_matrix: copula correlation diagonal must equal 1"));
    }
    for (Eigen::Index j = 0; j < copula.corr.cols(); ++j) {
      if (!std::isfinite(copula.corr(i, j)) ||
          std::abs(copula.corr(i, j) - copula.corr(j, i)) > sym_tol) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "simulate_t_copula_matrix: copula correlation must be finite and symmetric"));
      }
      if (i != j && std::abs(copula.corr(i, j)) >= 1.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "simulate_t_copula_matrix: off-diagonal correlations must satisfy |r| < 1"));
      }
    }
  }

  for (const auto& marginal : marginals) {
    if (auto ok = validate_marginal(marginal); !ok.has_value()) {
      return std::unexpected(ok.error());
    }
    if (marginal.kind == MarginalKind::Fleishman) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidMarginal,
          "simulate_t_copula_matrix: Fleishman polynomial is not a copula quantile marginal"));
    }
  }
  return {};
}

sim_expected<void>
validate_bivariate_copula_spec(const BivariateCopulaSpec& copula);

sim_expected<void>
validate_bivariate_quantile_marginals(const std::vector<MarginalSpec>& marginals,
                                      const char* caller);

sim_expected<void>
validate_bivariate_copula_inputs(Eigen::Index n,
                                 const BivariateCopulaSpec& copula,
                                 const std::vector<MarginalSpec>& marginals,
                                 const BivariateCopulaOptions& options) {
  if (n <= 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_bivariate_copula_matrix: n must be positive"));
  }
  if (marginals.size() != 2u) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_bivariate_copula_matrix: exactly two marginals are required"));
  }
  if (options.quadrature_points < 8 || options.max_bisection_iter < 16 ||
      !std::isfinite(options.calibration_tol) || options.calibration_tol <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_bivariate_copula_matrix: invalid options"));
  }

  if (auto ok = validate_bivariate_copula_spec(copula); !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  return validate_bivariate_quantile_marginals(
      marginals, "simulate_bivariate_copula_matrix");
}

sim_expected<void>
validate_bivariate_copula_spec(const BivariateCopulaSpec& copula) {
  switch (copula.family) {
    case BivariateCopulaFamily::Independence:
      return {};
    case BivariateCopulaFamily::Clayton:
      if (std::isfinite(copula.theta) && copula.theta > 0.0) return {};
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "bivariate copula: Clayton theta must be positive"));
    case BivariateCopulaFamily::Gumbel:
      if (std::isfinite(copula.theta) && copula.theta >= 1.0) return {};
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "bivariate copula: Gumbel theta must be at least 1"));
    case BivariateCopulaFamily::Frank:
      if (std::isfinite(copula.theta) && std::abs(copula.theta) >= 1e-10) {
        return {};
      }
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "bivariate copula: Frank theta must be finite and nonzero"));
    case BivariateCopulaFamily::Joe:
      if (std::isfinite(copula.theta) && copula.theta >= 1.0) return {};
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "bivariate copula: Joe theta must be at least 1"));
  }
  return std::unexpected(make_err(
      SimError::Kind::InvalidInput,
      "bivariate copula: unknown family"));
}

sim_expected<void>
validate_bivariate_quantile_marginals(const std::vector<MarginalSpec>& marginals,
                                      const char* caller) {
  if (marginals.size() != 2u) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": exactly two marginals are required"));
  }
  for (const auto& marginal : marginals) {
    if (auto ok = validate_marginal(marginal); !ok.has_value()) {
      return std::unexpected(ok.error());
    }
    if (marginal.kind == MarginalKind::Fleishman) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidMarginal,
          std::string(caller) +
              ": Fleishman polynomial is not a copula quantile marginal"));
    }
  }
  return {};
}

sim_expected<void>
validate_bivariate_copula_matrix_target(
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const std::vector<MarginalSpec>& marginals,
    const BivariateCopulaOptions& options) {
  if (target_corr.rows() != target_corr.cols()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_bivariate_copula_correlation_matrix: target_corr must be square"));
  }
  if (target_corr.rows() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_bivariate_copula_correlation_matrix: target_corr must not be empty"));
  }
  if (static_cast<Eigen::Index>(marginals.size()) != target_corr.rows()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_bivariate_copula_correlation_matrix: marginal count must match target_corr dimension"));
  }
  if (options.quadrature_points < 8 || options.max_bisection_iter < 16 ||
      !std::isfinite(options.calibration_tol) || options.calibration_tol <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_bivariate_copula_correlation_matrix: invalid options"));
  }

  constexpr double sym_tol = 1e-12;
  constexpr double diag_tol = 1e-12;
  for (Eigen::Index i = 0; i < target_corr.rows(); ++i) {
    if (!std::isfinite(target_corr(i, i)) ||
        std::abs(target_corr(i, i) - 1.0) > diag_tol) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "calibrate_bivariate_copula_correlation_matrix: target_corr diagonal must equal 1"));
    }
    for (Eigen::Index j = 0; j < target_corr.cols(); ++j) {
      if (!std::isfinite(target_corr(i, j)) ||
          std::abs(target_corr(i, j) - target_corr(j, i)) > sym_tol) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "calibrate_bivariate_copula_correlation_matrix: target_corr must be finite and symmetric"));
      }
      if (i != j && std::abs(target_corr(i, j)) >= 1.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "calibrate_bivariate_copula_correlation_matrix: off-diagonal target correlations must satisfy |r| < 1"));
      }
    }
  }

  for (const auto& marginal : marginals) {
    if (auto ok = validate_marginal(marginal); !ok.has_value()) {
      return std::unexpected(ok.error());
    }
    if (marginal.kind == MarginalKind::Fleishman) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidMarginal,
          "calibrate_bivariate_copula_correlation_matrix: Fleishman polynomial is not a copula quantile marginal"));
    }
  }
  return {};
}

sim_expected<std::vector<MarginalMoments>>
build_marginal_moments(const std::vector<MarginalSpec>& marginals,
                       const GaussHermite& gh) {
  std::vector<MarginalMoments> moments;
  moments.reserve(marginals.size());
  for (const auto& marginal : marginals) {
    auto mom_or = marginal_moments(marginal, gh);
    if (!mom_or.has_value()) return std::unexpected(mom_or.error());
    moments.push_back(*mom_or);
  }
  return moments;
}

sim_expected<MarginalMomentSummary>
raw_moment_summary(const MarginalSpec& m, const GaussHermite& gh) {
  if (auto ok = validate_marginal(m); !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  double e1 = 0.0;
  double e2 = 0.0;
  double e3 = 0.0;
  double e4 = 0.0;
  for (Eigen::Index a = 0; a < gh.nodes.size(); ++a) {
    const double z = kSqrt2 * gh.nodes(a);
    const double w = gh.weights(a) / kSqrtPi;
    const double y = raw_from_z(m, z);
    if (!std::isfinite(y)) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "marginal moments: non-finite marginal transform"));
    }
    const double y2 = y * y;
    e1 += w * y;
    e2 += w * y2;
    e3 += w * y2 * y;
    e4 += w * y2 * y2;
  }

  const double var = e2 - e1 * e1;
  if (!std::isfinite(var) || var <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "marginal moments: non-positive variance"));
  }

  const double sd = std::sqrt(var);
  const double mu3 = e3 - 3.0 * e1 * e2 + 2.0 * e1 * e1 * e1;
  const double mu4 = e4 - 4.0 * e1 * e3 + 6.0 * e1 * e1 * e2 -
                     3.0 * e1 * e1 * e1 * e1;
  const double skew = mu3 / (sd * sd * sd);
  const double excess = mu4 / (var * var) - 3.0;
  if (!std::isfinite(skew) || !std::isfinite(excess)) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "marginal moments: non-finite shape moments"));
  }

  return MarginalMomentSummary{m.mean, m.sd, skew, excess};
}

double pair_observed_corr(const MarginalSpec& mi,
                          const MarginalMoments& mom_i,
                          const MarginalSpec& mj,
                          const MarginalMoments& mom_j,
                          double rho,
                          const GaussHermite& gh) {
  const double one_minus = std::max(0.0, 1.0 - rho * rho);
  const double s = std::sqrt(one_minus);
  double exy = 0.0;
  for (Eigen::Index a = 0; a < gh.nodes.size(); ++a) {
    const double z1 = kSqrt2 * gh.nodes(a);
    const double xi = standardized_from_z(mi, mom_i, z1);
    const double wa = gh.weights(a);
    for (Eigen::Index b = 0; b < gh.nodes.size(); ++b) {
      const double eps = kSqrt2 * gh.nodes(b);
      const double z2 = rho * z1 + s * eps;
      const double xj = standardized_from_z(mj, mom_j, z2);
      exy += wa * gh.weights(b) * xi * xj;
    }
  }
  return exy / std::numbers::pi;
}

double clayton_conditional_u(double u, double v, double theta) {
  const double log_u = std::log(u);
  const double a = std::exp(-theta * log_u) +
                   std::exp(-theta * std::log(v)) - 1.0;
  if (!(a > 0.0)) return nan();
  return std::exp((-theta - 1.0) * log_u +
                  (-1.0 / theta - 1.0) * std::log(a));
}

double gumbel_conditional_u(double u, double v, double theta) {
  const double lu = -std::log(u);
  const double lv = -std::log(v);
  const double a = std::pow(lu, theta);
  const double b = std::pow(lv, theta);
  const double s = a + b;
  if (!(s > 0.0)) return nan();
  const double c = std::exp(-std::pow(s, 1.0 / theta));
  if (lu == 0.0) return 1.0;
  return c * std::pow(s, 1.0 / theta - 1.0) *
         std::pow(lu, theta - 1.0) / u;
}

double frank_conditional_u(double u, double v, double theta) {
  const double eu = std::exp(-theta * u);
  const double ev = std::exp(-theta * v);
  const double d = std::expm1(-theta);
  const double denom = d + (eu - 1.0) * (ev - 1.0);
  if (denom == 0.0) return nan();
  return eu * (ev - 1.0) / denom;
}

double joe_conditional_u(double u, double v, double theta) {
  const double a = std::pow(1.0 - u, theta);
  const double b = std::pow(1.0 - v, theta);
  const double s = a + b - a * b;
  if (!(s > 0.0)) return nan();
  return (1.0 - b) * std::pow(s, 1.0 / theta - 1.0) *
         std::pow(1.0 - u, theta - 1.0);
}

double bivariate_conditional_u(const BivariateCopulaSpec& copula,
                               double u,
                               double v) {
  switch (copula.family) {
    case BivariateCopulaFamily::Independence:
      return v;
    case BivariateCopulaFamily::Clayton:
      return clayton_conditional_u(u, v, copula.theta);
    case BivariateCopulaFamily::Gumbel:
      return gumbel_conditional_u(u, v, copula.theta);
    case BivariateCopulaFamily::Frank:
      return frank_conditional_u(u, v, copula.theta);
    case BivariateCopulaFamily::Joe:
      return joe_conditional_u(u, v, copula.theta);
  }
  return nan();
}

double frank_debye1_integrand(double t) noexcept {
  if (std::abs(t) < 1e-6) {
    const double t2 = t * t;
    return 1.0 - 0.5 * t + t2 / 12.0 - t2 * t2 / 720.0;
  }
  return t / std::expm1(t);
}

double frank_debye1(double theta) {
  if (std::abs(theta) < 1e-5) {
    const double t2 = theta * theta;
    return 1.0 - 0.25 * theta + t2 / 36.0 - t2 * t2 / 3600.0;
  }
  const auto f = [](double t) { return frank_debye1_integrand(t); };
  const double integral = theta > 0.0
      ? adaptive_simpson(f, 0.0, theta, 1e-12)
      : -adaptive_simpson(f, theta, 0.0, 1e-12);
  return integral / theta;
}

double joe_tau_series(double theta) {
  double sum = 0.0;
  for (int k = 1; k < 200000; ++k) {
    const double dk = static_cast<double>(k);
    const double term = 1.0 /
        (dk * (theta * dk + 2.0) * (theta * (dk - 1.0) + 2.0));
    sum += term;
    if (term < 1e-15 * std::max(1.0, std::abs(sum))) break;
  }
  return 1.0 - 4.0 * sum;
}

double bivariate_tau_unchecked(const BivariateCopulaSpec& copula) {
  switch (copula.family) {
    case BivariateCopulaFamily::Independence:
      return 0.0;
    case BivariateCopulaFamily::Clayton:
      return copula.theta / (copula.theta + 2.0);
    case BivariateCopulaFamily::Gumbel:
      return 1.0 - 1.0 / copula.theta;
    case BivariateCopulaFamily::Frank: {
      const double a = std::abs(copula.theta);
      if (a > 50.0) {
        const double tau_abs = 1.0 - 4.0 / a +
            2.0 * std::numbers::pi * std::numbers::pi / (3.0 * a * a);
        return std::copysign(tau_abs, copula.theta);
      }
      return 1.0 + 4.0 * (frank_debye1(copula.theta) - 1.0) /
                       copula.theta;
    }
    case BivariateCopulaFamily::Joe:
      if (copula.theta == 1.0) return 0.0;
      return joe_tau_series(copula.theta);
  }
  return nan();
}

sim_expected<double>
invert_tau_bisection(BivariateCopulaFamily family,
                     double tau,
                     double lo,
                     double hi,
                     int max_iter) {
  BivariateCopulaSpec probe;
  probe.family = family;

  probe.theta = hi;
  double hi_tau = bivariate_tau_unchecked(probe);
  while (std::isfinite(hi_tau) && hi_tau < tau) {
    lo = hi;
    hi *= 2.0;
    if (!std::isfinite(hi) || hi > 1e8) {
      return std::unexpected(make_err(
          SimError::Kind::CalibrationFailed,
          "bivariate_copula_from_tau: could not bracket tau"));
    }
    probe.theta = hi;
    hi_tau = bivariate_tau_unchecked(probe);
  }
  if (!std::isfinite(hi_tau)) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "bivariate_copula_from_tau: non-finite tau during bracketing"));
  }

  double mid = hi;
  for (int iter = 0; iter < max_iter; ++iter) {
    mid = 0.5 * (lo + hi);
    probe.theta = mid;
    const double mid_tau = bivariate_tau_unchecked(probe);
    if (!std::isfinite(mid_tau)) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "bivariate_copula_from_tau: non-finite tau during inversion"));
    }
    if (mid_tau < tau) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

sim_expected<double>
invert_frank_tau(double tau, int max_iter) {
  if (tau > 0.0) {
    auto theta_or = invert_tau_bisection(
        BivariateCopulaFamily::Frank, tau, 1e-8, 1.0, max_iter);
    if (!theta_or.has_value()) return theta_or;
    return *theta_or;
  }

  const double target = -tau;
  auto theta_or = invert_tau_bisection(
      BivariateCopulaFamily::Frank, target, 1e-8, 1.0, max_iter);
  if (!theta_or.has_value()) return theta_or;
  return -*theta_or;
}

sim_expected<double>
invert_bivariate_conditional(const BivariateCopulaSpec& copula,
                             double u,
                             double w,
                             int max_iter) {
  if (copula.family == BivariateCopulaFamily::Independence) return w;
  u = open_unit(u);
  w = open_unit(w);
  double lo = std::numeric_limits<double>::min();
  double hi = 1.0 - std::numeric_limits<double>::epsilon();
  double mid = 0.5;
  for (int iter = 0; iter < max_iter; ++iter) {
    mid = 0.5 * (lo + hi);
    const double cdf = bivariate_conditional_u(copula, u, mid);
    if (!std::isfinite(cdf)) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "simulate_bivariate_copula_matrix: non-finite conditional copula CDF"));
    }
    if (cdf < w) lo = mid;
    else hi = mid;
  }
  return open_unit(0.5 * (lo + hi));
}

sim_expected<double>
calibrate_pair(double target,
               const MarginalSpec& mi,
               const MarginalMoments& mom_i,
               const MarginalSpec& mj,
               const MarginalMoments& mom_j,
               const GaussHermite& gh,
               const NortaOptions& options) {
  if (std::abs(target) < options.calibration_tol) {
    return 0.0;
  }

  double lo = -options.rho_bound;
  double hi = options.rho_bound;
  double flo = pair_observed_corr(mi, mom_i, mj, mom_j, lo, gh) - target;
  double fhi = pair_observed_corr(mi, mom_i, mj, mom_j, hi, gh) - target;
  if (!std::isfinite(flo) || !std::isfinite(fhi)) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "calibrate_norta: non-finite pair correlation during calibration"));
  }
  if (flo > 0.0 || fhi < 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "calibrate_norta: target correlation outside feasible NORTA range for a pair"));
  }

  double mid = 0.0;
  for (int iter = 0; iter < options.max_bisection_iter; ++iter) {
    mid = 0.5 * (lo + hi);
    const double fmid = pair_observed_corr(mi, mom_i, mj, mom_j, mid, gh) - target;
    if (!std::isfinite(fmid)) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "calibrate_norta: non-finite midpoint correlation"));
    }
    if (std::abs(fmid) <= options.calibration_tol ||
        0.5 * (hi - lo) <= options.calibration_tol) {
      return mid;
    }
    if (fmid < 0.0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return mid;
}

sim_expected<void>
validate_moment_match(const MomentMatchSpec& spec,
                      const MomentMatchOptions& options) {
  if (!std::isfinite(spec.mean) || !std::isfinite(spec.sd) || spec.sd <= 0.0 ||
      !std::isfinite(spec.shape.skewness) ||
      !std::isfinite(spec.shape.excess_kurtosis)) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "fit_marginal_to_moments: target moments must be finite and sd positive"));
  }
  if (options.quadrature_points < 8 || options.max_iter < 1 ||
      options.grid_points_g < 3 || options.grid_points_h < 2 ||
      options.objective_tol <= 0.0 || options.parameter_tol <= 0.0 ||
      options.finite_diff_step <= 0.0 ||
      !std::isfinite(options.objective_tol) ||
      !std::isfinite(options.parameter_tol) ||
      !std::isfinite(options.finite_diff_step) ||
      !std::isfinite(options.tukey_g_bound) ||
      !std::isfinite(options.tukey_h_upper) ||
      !std::isfinite(options.johnson_gamma_bound) ||
      !std::isfinite(options.johnson_log_delta_lower) ||
      !std::isfinite(options.johnson_log_delta_upper) ||
      options.tukey_g_bound <= 0.0 || options.tukey_h_upper <= 0.0 ||
      options.tukey_h_upper >= 0.25 ||
      options.johnson_gamma_bound <= 0.0 ||
      options.johnson_log_delta_lower >= options.johnson_log_delta_upper ||
      options.johnson_log_delta_lower < -2.0 ||
      options.johnson_log_delta_upper > 3.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "fit_marginal_to_moments: invalid moment-match options"));
  }
  switch (spec.family) {
    case MomentMatchFamily::TukeyGH:
      return {};
    case MomentMatchFamily::Pearson:
      return {};
    case MomentMatchFamily::Johnson:
      return {};
    case MomentMatchFamily::Fleishman:
      return {};
  }
  return std::unexpected(make_err(
      SimError::Kind::InvalidInput,
      "fit_marginal_to_moments: unknown moment-match family"));
}

Eigen::Vector2d project_tukey_params(Eigen::Vector2d x,
                                     const MomentMatchOptions& options) {
  x(0) = std::clamp(x(0), -options.tukey_g_bound, options.tukey_g_bound);
  x(1) = std::clamp(x(1), 0.0, options.tukey_h_upper);
  return x;
}

sim_expected<MomentFitEval>
eval_tukey_moment_fit(const Eigen::Vector2d& x,
                      const MomentMatchSpec& spec,
                      const GaussHermite& gh) {
  MomentFitEval out;
  out.marginal = MarginalSpec::tukey_g_h(x(0), x(1), spec.mean, spec.sd);
  auto moments_or = raw_moment_summary(out.marginal, gh);
  if (!moments_or.has_value()) return std::unexpected(moments_or.error());
  out.moments = *moments_or;
  out.residual(0) = out.moments.skewness - spec.shape.skewness;
  out.residual(1) = out.moments.excess_kurtosis - spec.shape.excess_kurtosis;
  out.norm = out.residual.norm();
  return out;
}

sim_expected<MomentFitEval>
initial_tukey_moment_fit(const MomentMatchSpec& spec,
                         const MomentMatchOptions& options,
                         const GaussHermite& gh) {
  MomentFitEval best;
  bool have_best = false;
  for (int ig = 0; ig < options.grid_points_g; ++ig) {
    const double wg = static_cast<double>(ig) /
                      static_cast<double>(options.grid_points_g - 1);
    const double g = -options.tukey_g_bound +
                     2.0 * options.tukey_g_bound * wg;
    for (int ih = 0; ih < options.grid_points_h; ++ih) {
      const double wh = static_cast<double>(ih) /
                        static_cast<double>(options.grid_points_h - 1);
      const double h = options.tukey_h_upper * wh;
      auto eval_or = eval_tukey_moment_fit(Eigen::Vector2d(g, h), spec, gh);
      if (!eval_or.has_value()) continue;
      if (!have_best || eval_or->norm < best.norm) {
        best = *eval_or;
        have_best = true;
      }
    }
  }
  if (!have_best) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: no finite Tukey g-and-h starting point"));
  }
  return best;
}

sim_expected<MomentMatchResult>
fit_tukey_moment_match(const MomentMatchSpec& spec,
                       const MomentMatchOptions& options,
                       const GaussHermite& gh) {
  if (spec.shape.excess_kurtosis < -options.objective_tol) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: Tukey g-and-h cannot match negative excess kurtosis"));
  }

  auto best_or = initial_tukey_moment_fit(spec, options, gh);
  if (!best_or.has_value()) return std::unexpected(best_or.error());
  MomentFitEval best = *best_or;
  MomentFitEval current = best;
  Eigen::Vector2d x(current.marginal.g, current.marginal.h);

  int iter = 0;
  for (; iter < options.max_iter; ++iter) {
    if (current.norm <= options.objective_tol) break;

    Eigen::Matrix2d J;
    bool ok_jac = true;
    for (int k = 0; k < 2; ++k) {
      const double step = options.finite_diff_step *
                          std::max(1.0, std::abs(x(k)));
      Eigen::Vector2d xp = project_tukey_params(x, options);
      Eigen::Vector2d xm = xp;
      xp(k) = std::clamp(x(k) + step,
                         k == 0 ? -options.tukey_g_bound : 0.0,
                         k == 0 ? options.tukey_g_bound : options.tukey_h_upper);
      xm(k) = std::clamp(x(k) - step,
                         k == 0 ? -options.tukey_g_bound : 0.0,
                         k == 0 ? options.tukey_g_bound : options.tukey_h_upper);
      if (std::abs(xp(k) - xm(k)) <= options.parameter_tol) {
        ok_jac = false;
        break;
      }
      auto fp_or = eval_tukey_moment_fit(xp, spec, gh);
      auto fm_or = eval_tukey_moment_fit(xm, spec, gh);
      if (!fp_or.has_value() || !fm_or.has_value()) {
        ok_jac = false;
        break;
      }
      J.col(k) = (fp_or->residual - fm_or->residual) / (xp(k) - xm(k));
    }
    if (!ok_jac || !J.allFinite()) break;

    Eigen::Matrix2d A = J.transpose() * J;
    const double ridge = 1e-10 * std::max(1.0, A.diagonal().cwiseAbs().maxCoeff());
    A.diagonal().array() += ridge;
    const Eigen::Vector2d step = A.ldlt().solve(-J.transpose() * current.residual);
    if (!step.allFinite()) break;
    if (step.norm() <= options.parameter_tol) break;

    bool accepted = false;
    double alpha = 1.0;
    for (int ls = 0; ls < 24; ++ls) {
      const Eigen::Vector2d trial_x = project_tukey_params(x + alpha * step, options);
      if ((trial_x - x).norm() <= options.parameter_tol) break;
      auto trial_or = eval_tukey_moment_fit(trial_x, spec, gh);
      if (trial_or.has_value() && trial_or->norm < current.norm) {
        current = *trial_or;
        x = trial_x;
        accepted = true;
        if (current.norm < best.norm) best = current;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) {
      double radius_g = std::max(0.01, 0.25 * std::max(1.0, std::abs(x(0))));
      double radius_h = 0.02;
      for (int search = 0; search < 32 && !accepted; ++search) {
        const Eigen::Vector2d trials[] = {
            Eigen::Vector2d( radius_g,  0.0),
            Eigen::Vector2d(-radius_g,  0.0),
            Eigen::Vector2d( 0.0,       radius_h),
            Eigen::Vector2d( 0.0,      -radius_h),
            Eigen::Vector2d( radius_g,  radius_h),
            Eigen::Vector2d( radius_g, -radius_h),
            Eigen::Vector2d(-radius_g,  radius_h),
            Eigen::Vector2d(-radius_g, -radius_h),
        };
        for (const auto& delta : trials) {
          const Eigen::Vector2d trial_x = project_tukey_params(x + delta, options);
          if ((trial_x - x).norm() <= options.parameter_tol) continue;
          auto trial_or = eval_tukey_moment_fit(trial_x, spec, gh);
          if (trial_or.has_value() && trial_or->norm < current.norm) {
            current = *trial_or;
            x = trial_x;
            accepted = true;
            if (current.norm < best.norm) best = current;
            break;
          }
        }
        if (!accepted) {
          radius_g *= 0.5;
          radius_h *= 0.5;
          if (std::max(radius_g, radius_h) <= options.parameter_tol) break;
        }
      }
    }
    if (!accepted) break;
  }

  if (best.norm > options.objective_tol) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: Tukey g-and-h moment solve did not converge"));
  }

  return MomentMatchResult{best.marginal, best.moments, best.norm, iter};
}

Eigen::Vector2d project_johnson_params(Eigen::Vector2d x,
                                       const MomentMatchOptions& options) {
  x(0) = std::clamp(x(0), -options.johnson_gamma_bound,
                    options.johnson_gamma_bound);
  x(1) = std::clamp(x(1), options.johnson_log_delta_lower,
                    options.johnson_log_delta_upper);
  return x;
}

sim_expected<MomentFitEval>
eval_johnson_moment_fit(int type,
                        const Eigen::Vector2d& x,
                        const MomentMatchSpec& spec,
                        const GaussHermite& gh) {
  MomentFitEval out;
  out.marginal = MarginalSpec::johnson(
      type, x(0), std::exp(x(1)), spec.mean, spec.sd);
  auto moments_or = raw_moment_summary(out.marginal, gh);
  if (!moments_or.has_value()) return std::unexpected(moments_or.error());
  out.moments = *moments_or;
  out.residual(0) = out.moments.skewness - spec.shape.skewness;
  out.residual(1) = out.moments.excess_kurtosis - spec.shape.excess_kurtosis;
  out.norm = out.residual.norm();
  return out;
}

sim_expected<MomentFitEval>
initial_johnson_moment_fit(int type,
                           const MomentMatchSpec& spec,
                           const MomentMatchOptions& options,
                           const GaussHermite& gh) {
  MomentFitEval best;
  bool have_best = false;
  for (int ig = 0; ig < options.grid_points_g; ++ig) {
    const double wg = static_cast<double>(ig) /
                      static_cast<double>(options.grid_points_g - 1);
    const double gamma = -options.johnson_gamma_bound +
                         2.0 * options.johnson_gamma_bound * wg;
    for (int id = 0; id < options.grid_points_h; ++id) {
      const double wd = static_cast<double>(id) /
                        static_cast<double>(options.grid_points_h - 1);
      const double log_delta = options.johnson_log_delta_lower +
          (options.johnson_log_delta_upper -
           options.johnson_log_delta_lower) * wd;
      auto eval_or = eval_johnson_moment_fit(
          type, Eigen::Vector2d(gamma, log_delta), spec, gh);
      if (!eval_or.has_value()) continue;
      if (!have_best || eval_or->norm < best.norm) {
        best = *eval_or;
        have_best = true;
      }
    }
  }
  if (!have_best) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: no finite Johnson starting point"));
  }
  return best;
}

sim_expected<MomentFitEval>
fit_one_johnson_moment_match(int type,
                             const MomentMatchSpec& spec,
                             const MomentMatchOptions& options,
                             const GaussHermite& gh,
                             int& iterations) {
  auto best_or = initial_johnson_moment_fit(type, spec, options, gh);
  if (!best_or.has_value()) return std::unexpected(best_or.error());
  MomentFitEval best = *best_or;
  MomentFitEval current = best;
  Eigen::Vector2d x(current.marginal.johnson_gamma,
                   std::log(current.marginal.johnson_delta));

  iterations = 0;
  for (; iterations < options.max_iter; ++iterations) {
    if (current.norm <= options.objective_tol) break;

    Eigen::Matrix2d J;
    bool ok_jac = true;
    for (int k = 0; k < 2; ++k) {
      const double step = options.finite_diff_step *
                          std::max(1.0, std::abs(x(k)));
      Eigen::Vector2d xp = project_johnson_params(x, options);
      Eigen::Vector2d xm = xp;
      xp(k) = std::clamp(x(k) + step,
                         k == 0 ? -options.johnson_gamma_bound
                                : options.johnson_log_delta_lower,
                         k == 0 ? options.johnson_gamma_bound
                                : options.johnson_log_delta_upper);
      xm(k) = std::clamp(x(k) - step,
                         k == 0 ? -options.johnson_gamma_bound
                                : options.johnson_log_delta_lower,
                         k == 0 ? options.johnson_gamma_bound
                                : options.johnson_log_delta_upper);
      if (std::abs(xp(k) - xm(k)) <= options.parameter_tol) {
        ok_jac = false;
        break;
      }
      auto fp_or = eval_johnson_moment_fit(type, xp, spec, gh);
      auto fm_or = eval_johnson_moment_fit(type, xm, spec, gh);
      if (!fp_or.has_value() || !fm_or.has_value()) {
        ok_jac = false;
        break;
      }
      J.col(k) = (fp_or->residual - fm_or->residual) / (xp(k) - xm(k));
    }
    if (!ok_jac || !J.allFinite()) break;

    Eigen::Matrix2d A = J.transpose() * J;
    const double ridge =
        1e-10 * std::max(1.0, A.diagonal().cwiseAbs().maxCoeff());
    A.diagonal().array() += ridge;
    const Eigen::Vector2d step =
        A.ldlt().solve(-J.transpose() * current.residual);
    if (!step.allFinite()) break;
    if (step.norm() <= options.parameter_tol) break;

    bool accepted = false;
    double alpha = 1.0;
    for (int ls = 0; ls < 24; ++ls) {
      const Eigen::Vector2d trial_x =
          project_johnson_params(x + alpha * step, options);
      if ((trial_x - x).norm() <= options.parameter_tol) break;
      auto trial_or = eval_johnson_moment_fit(type, trial_x, spec, gh);
      if (trial_or.has_value() && trial_or->norm < current.norm) {
        current = *trial_or;
        x = trial_x;
        accepted = true;
        if (current.norm < best.norm) best = current;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) {
      double radius_gamma = 0.25 * std::max(1.0, std::abs(x(0)));
      double radius_delta = 0.15;
      for (int search = 0; search < 32 && !accepted; ++search) {
        const Eigen::Vector2d trials[] = {
            Eigen::Vector2d( radius_gamma,  0.0),
            Eigen::Vector2d(-radius_gamma,  0.0),
            Eigen::Vector2d( 0.0,           radius_delta),
            Eigen::Vector2d( 0.0,          -radius_delta),
            Eigen::Vector2d( radius_gamma,  radius_delta),
            Eigen::Vector2d( radius_gamma, -radius_delta),
            Eigen::Vector2d(-radius_gamma,  radius_delta),
            Eigen::Vector2d(-radius_gamma, -radius_delta),
        };
        for (const auto& delta : trials) {
          const Eigen::Vector2d trial_x =
              project_johnson_params(x + delta, options);
          if ((trial_x - x).norm() <= options.parameter_tol) continue;
          auto trial_or = eval_johnson_moment_fit(type, trial_x, spec, gh);
          if (trial_or.has_value() && trial_or->norm < current.norm) {
            current = *trial_or;
            x = trial_x;
            accepted = true;
            if (current.norm < best.norm) best = current;
            break;
          }
        }
        if (!accepted) {
          radius_gamma *= 0.5;
          radius_delta *= 0.5;
          if (std::max(radius_gamma, radius_delta) <= options.parameter_tol) {
            break;
          }
        }
      }
    }
    if (!accepted) break;
  }
  return best;
}

sim_expected<MomentMatchResult>
fit_johnson_moment_match(const MomentMatchSpec& spec,
                         const MomentMatchOptions& options,
                         const GaussHermite& gh) {
  const double beta1 = spec.shape.skewness * spec.shape.skewness;
  const double beta2 = spec.shape.excess_kurtosis + 3.0;
  if (beta2 < beta1 + 1.0 - options.objective_tol) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: no distribution has the requested Johnson moments"));
  }

  MomentFitEval best;
  bool have_best = false;
  int best_iterations = 0;
  for (int type : {2, 3}) {
    int iterations = 0;
    auto fit_or = fit_one_johnson_moment_match(
        type, spec, options, gh, iterations);
    if (!fit_or.has_value()) continue;
    if (!have_best || fit_or->norm < best.norm) {
      best = *fit_or;
      best_iterations = iterations;
      have_best = true;
    }
  }
  if (!have_best) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: Johnson moment solve found no finite fit"));
  }
  if (best.norm > options.objective_tol) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: Johnson moment solve did not converge"));
  }
  return MomentMatchResult{
      best.marginal, best.moments, best.norm, best_iterations};
}

sim_expected<MomentMatchResult>
fit_fleishman_moment_match(const MomentMatchSpec& spec,
                           const MomentMatchOptions& options,
                           const GaussHermite& gh) {
  ValeMaurelliOptions vm_options;
  vm_options.max_iter = options.max_iter;
  vm_options.coefficient_tol = std::min(options.objective_tol, 1e-8);
  auto coef_or = fit_fleishman_coefficients(
      spec.shape.skewness, spec.shape.excess_kurtosis, vm_options);
  if (!coef_or.has_value()) return std::unexpected(coef_or.error());

  MomentMatchResult out;
  out.marginal = MarginalSpec::fleishman(
      coef_or->b, coef_or->c, coef_or->d, spec.mean, spec.sd);
  auto moments_or = raw_moment_summary(out.marginal, gh);
  if (!moments_or.has_value()) return std::unexpected(moments_or.error());
  out.moments = *moments_or;
  const Eigen::Vector2d residual(
      out.moments.skewness - spec.shape.skewness,
      out.moments.excess_kurtosis - spec.shape.excess_kurtosis);
  out.residual_norm = residual.norm();
  out.iterations = 0;
  if (out.residual_norm > options.objective_tol) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: Fleishman moment solve did not converge"));
  }
  return out;
}

sim_expected<MomentMatchResult>
fit_pearson_moment_match(const MomentMatchSpec& spec) {
  const double mean = spec.mean;
  const double variance = spec.sd * spec.sd;
  const double skew = spec.shape.skewness;
  const double kurt = spec.shape.excess_kurtosis + 3.0;
  const double skew2 = skew * skew;
  if (approx_equal(skew2, kurt - 1.0)) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: Pearson boundary is a two-point discrete distribution"));
  }
  if (skew2 > kurt - 1.0) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: no distribution has the requested Pearson moments"));
  }

  const double denom = 10.0 * kurt - 12.0 * skew2 - 18.0;
  if (approx_equal(denom, 0.0)) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "fit_marginal_to_moments: Pearson moment equations are singular"));
  }
  const double c0 = (4.0 * kurt - 3.0 * skew2) / denom * variance;
  const double c1 = skew * (kurt + 3.0) / denom * spec.sd;
  const double c2 = (2.0 * kurt - 3.0 * skew2 - 6.0) / denom;

  MarginalSpec marginal;
  if (approx_equal(skew, 0.0)) {
    if (approx_equal(kurt, 3.0)) {
      marginal = MarginalSpec::pearson(0, mean, spec.sd, 0.0, 0.0,
                                       mean, spec.sd);
    } else if (kurt < 3.0) {
      const double a1 = spec.sd / 2.0 *
          (-std::sqrt(-16.0 * kurt * (2.0 * kurt - 6.0)) /
           (2.0 * kurt - 6.0));
      const double m1 = -1.0 / (2.0 * c2);
      if (!(m1 > -1.0)) {
        return std::unexpected(make_err(
            SimError::Kind::CalibrationFailed,
            "fit_marginal_to_moments: invalid Pearson Type II shape"));
      }
      const double scale = 2.0 * a1;
      const double location = mean - scale / 2.0;
      marginal = MarginalSpec::pearson(2, 1.0 + m1, location, scale, 0.0,
                                       mean, spec.sd);
    } else {
      const double r = 6.0 * (kurt - 1.0) / (2.0 * kurt - 6.0);
      const double a = std::sqrt(variance * (r - 1.0));
      const double df = 1.0 + r;
      marginal = MarginalSpec::pearson(7, df, mean, a / std::sqrt(df), 0.0,
                                       mean, spec.sd);
    }
  } else if (!approx_equal(2.0 * kurt - 3.0 * skew2 - 6.0, 0.0)) {
    const double kap = 0.25 * skew2 * (kurt + 3.0) * (kurt + 3.0) /
        ((4.0 * kurt - 3.0 * skew2) *
         (2.0 * kurt - 3.0 * skew2 - 6.0));
    if (kap < 0.0) {
      double a1 = spec.sd / 2.0 *
          ((-skew * (kurt + 3.0) -
            std::sqrt(skew2 * (kurt + 3.0) * (kurt + 3.0) -
                      4.0 * (4.0 * kurt - 3.0 * skew2) *
                      (2.0 * kurt - 3.0 * skew2 - 6.0))) /
           (2.0 * kurt - 3.0 * skew2 - 6.0));
      double a2 = spec.sd / 2.0 *
          ((-skew * (kurt + 3.0) +
            std::sqrt(skew2 * (kurt + 3.0) * (kurt + 3.0) -
                      4.0 * (4.0 * kurt - 3.0 * skew2) *
                      (2.0 * kurt - 3.0 * skew2 - 6.0))) /
           (2.0 * kurt - 3.0 * skew2 - 6.0));
      if (a1 > 0.0) std::swap(a1, a2);
      const double disc = std::sqrt(
          skew2 * (kurt + 3.0) * (kurt + 3.0) -
          4.0 * (4.0 * kurt - 3.0 * skew2) *
          (2.0 * kurt - 3.0 * skew2 - 6.0));
      const double m1 = -(skew * (kurt + 3.0) +
          a1 * denom / spec.sd) / disc;
      const double m2 = -(-skew * (kurt + 3.0) -
          a2 * denom / spec.sd) / disc;
      if (!(m1 > -1.0) || !(m2 > -1.0)) {
        return std::unexpected(make_err(
            SimError::Kind::CalibrationFailed,
            "fit_marginal_to_moments: invalid Pearson Type I shape"));
      }
      const double scale = a2 - a1;
      const double location = mean - scale * (m1 + 1.0) / (m1 + m2 + 2.0);
      marginal = MarginalSpec::pearson(1, 1.0 + m1, 1.0 + m2,
                                       location, scale, mean, spec.sd);
    } else if (approx_equal(kap, 1.0)) {
      const double C1 = c1 / (2.0 * c2);
      const double scale = -(c1 - C1) / c2;
      marginal = MarginalSpec::pearson(5, 1.0 / c2 - 1.0, mean - C1,
                                       scale, 0.0, mean, spec.sd);
    } else if (kap > 1.0) {
      double a1 = spec.sd / 2.0 *
          ((-skew * (kurt + 3.0) -
            std::sqrt(skew2 * (kurt + 3.0) * (kurt + 3.0) -
                      4.0 * (4.0 * kurt - 3.0 * skew2) *
                      (2.0 * kurt - 3.0 * skew2 - 6.0))) /
           (2.0 * kurt - 3.0 * skew2 - 6.0));
      double a2 = spec.sd / 2.0 *
          ((-skew * (kurt + 3.0) +
            std::sqrt(skew2 * (kurt + 3.0) * (kurt + 3.0) -
                      4.0 * (4.0 * kurt - 3.0 * skew2) *
                      (2.0 * kurt - 3.0 * skew2 - 6.0))) /
           (2.0 * kurt - 3.0 * skew2 - 6.0));
      if (a1 > 0.0) std::swap(a1, a2);
      const double m1 = (c1 + a1) / (c2 * (a2 - a1));
      const double m2 = -(c1 + a2) / (c2 * (a2 - a1));
      const double scale = a2 - a1;
      const double location = mean + scale * (m2 + 1.0) / (m2 + m1 + 2.0);
      marginal = MarginalSpec::pearson(6, 1.0 + m2, -m2 - m1 - 1.0,
                                       location, scale, mean, spec.sd);
    } else if (kap > 0.0 && kap < 1.0) {
      const double r = 6.0 * (kurt - skew2 - 1.0) /
                       (2.0 * kurt - 3.0 * skew2 - 6.0);
      const double shape_term =
          16.0 * (r - 1.0) - skew2 * (r - 2.0) * (r - 2.0);
      if (!(shape_term > 0.0)) {
        return std::unexpected(make_err(
            SimError::Kind::CalibrationFailed,
            "fit_marginal_to_moments: invalid Pearson Type IV shape"));
      }
      const double nu = -r * (r - 2.0) * skew / std::sqrt(shape_term);
      const double scale = std::sqrt(variance * shape_term) / 4.0;
      const double location = mean - ((r - 2.0) * skew * spec.sd) / 4.0;
      const double m = 1.0 + 0.5 * r;
      marginal = MarginalSpec::pearson(4, m, nu, location, scale,
                                       mean, spec.sd);
    } else {
      return std::unexpected(make_err(
          SimError::Kind::CalibrationFailed,
          "fit_marginal_to_moments: invalid Pearson kappa"));
    }
  } else {
    const double m = c0 / (c1 * c1) - 1.0;
    marginal = MarginalSpec::pearson(3, m + 1.0, mean - c0 / c1, c1, 0.0,
                                     mean, spec.sd);
  }

  if (auto ok = validate_marginal(marginal); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  MomentMatchResult out;
  out.marginal = marginal;
  out.moments = MarginalMomentSummary{
      mean, spec.sd, spec.shape.skewness, spec.shape.excess_kurtosis};
  out.residual_norm = 0.0;
  out.iterations = 0;
  return out;
}

sim_expected<Eigen::MatrixXd>
cholesky_factor_with_jitter(const Eigen::Ref<const Eigen::MatrixXd>& corr,
                            double jitter) {
  Eigen::LLT<Eigen::MatrixXd> llt(corr);
  if (llt.info() == Eigen::Success) {
    return llt.matrixL();
  }
  if (jitter > 0.0) {
    Eigen::MatrixXd adjusted = corr;
    adjusted.diagonal().array() += jitter;
    Eigen::LLT<Eigen::MatrixXd> llt_j(adjusted);
    if (llt_j.info() == Eigen::Success) {
      return llt_j.matrixL();
    }
  }
  return std::unexpected(make_err(
      SimError::Kind::NonPositiveDefinite,
      "simulate_norta: calibrated latent correlation is not positive definite"));
}

sim_expected<void>
validate_ig_inputs(const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                   const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
                   const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
                   const IgOptions& options) {
  if (sigma.rows() != sigma.cols()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ig: sigma must be square"));
  }
  if (sigma.rows() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ig: sigma must not be empty"));
  }
  if (target_skewness.size() != sigma.rows() ||
      target_excess_kurtosis.size() != sigma.rows()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ig: target moment vector sizes must match sigma dimension"));
  }
  if (!std::isfinite(options.root_eigen_tol) || options.root_eigen_tol <= 0.0 ||
      !std::isfinite(options.moment_solve_tol) || options.moment_solve_tol <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ig: invalid root or moment-solve tolerances"));
  }

  constexpr double sym_tol = 1e-12;
  for (Eigen::Index i = 0; i < sigma.rows(); ++i) {
    if (!std::isfinite(target_skewness(i)) ||
        !std::isfinite(target_excess_kurtosis(i))) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "calibrate_ig: target skewness and kurtosis must be finite"));
    }
    if (!std::isfinite(sigma(i, i)) || sigma(i, i) <= 0.0) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "calibrate_ig: sigma diagonal must be finite and positive"));
    }
    for (Eigen::Index j = 0; j < sigma.cols(); ++j) {
      if (!std::isfinite(sigma(i, j)) ||
          std::abs(sigma(i, j) - sigma(j, i)) > sym_tol) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "calibrate_ig: sigma must be finite and symmetric"));
      }
    }
  }
  return {};
}

sim_expected<Eigen::MatrixXd>
ig_root_matrix(const Eigen::Ref<const Eigen::MatrixXd>& sigma,
               const IgOptions& options) {
  switch (options.root) {
    case IgRootKind::Cholesky: {
      Eigen::LLT<Eigen::MatrixXd> llt(sigma);
      if (llt.info() != Eigen::Success) {
        return std::unexpected(make_err(
            SimError::Kind::NonPositiveDefinite,
            "calibrate_ig: sigma is not positive definite"));
      }
      return llt.matrixL();
    }
    case IgRootKind::Symmetric: {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sigma);
      if (es.info() != Eigen::Success) {
        return std::unexpected(make_err(
            SimError::Kind::NumericIssue,
            "calibrate_ig: symmetric-root eigensolve failed"));
      }
      const Eigen::VectorXd evals = es.eigenvalues();
      if (evals.minCoeff() <= options.root_eigen_tol) {
        return std::unexpected(make_err(
            SimError::Kind::NonPositiveDefinite,
            "calibrate_ig: sigma has non-positive eigenvalues"));
      }
      return es.eigenvectors() * evals.cwiseSqrt().asDiagonal() *
             es.eigenvectors().transpose();
    }
  }
  return std::unexpected(make_err(
      SimError::Kind::InvalidInput,
      "calibrate_ig: unknown root kind"));
}

sim_expected<Eigen::VectorXd>
solve_ig_moment_system(const Eigen::MatrixXd& B,
                       const Eigen::VectorXd& target,
                       const char* label,
                       double tol) {
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(B);
  const Eigen::VectorXd x = qr.solve(target);
  if (!x.allFinite()) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        std::string("calibrate_ig: non-finite ") + label + " generator moments"));
  }
  const Eigen::VectorXd residual = B * x - target;
  const double scale = std::max(1.0, target.norm());
  if (!std::isfinite(residual.norm()) || residual.norm() > tol * scale) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        std::string("calibrate_ig: ") + label + " moment system is singular"));
  }
  return x;
}

sim_expected<void>
ig_generator_moments(const Eigen::MatrixXd& root,
                     const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
                     const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
                     double tol,
                     Eigen::VectorXd& generator_skewness,
                     Eigen::VectorXd& generator_excess_kurtosis) {
  const Eigen::Index p = root.rows();
  const Eigen::Index q = root.cols();
  Eigen::MatrixXd B3(p, q);
  Eigen::MatrixXd B4(p, q);
  for (Eigen::Index i = 0; i < p; ++i) {
    const double variance = root.row(i).squaredNorm();
    if (!std::isfinite(variance) || variance <= 0.0) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "calibrate_ig: root has a row with non-positive variance"));
    }
    const double sd = std::sqrt(variance);
    for (Eigen::Index j = 0; j < q; ++j) {
      const double a = root(i, j);
      const double a2 = a * a;
      B3(i, j) = a2 * a / (sd * sd * sd);
      B4(i, j) = a2 * a2 / (variance * variance);
    }
  }

  auto skew_or = solve_ig_moment_system(B3, target_skewness, "skewness", tol);
  if (!skew_or.has_value()) return std::unexpected(skew_or.error());
  auto kurt_or = solve_ig_moment_system(
      B4, target_excess_kurtosis, "kurtosis", tol);
  if (!kurt_or.has_value()) return std::unexpected(kurt_or.error());
  generator_skewness = std::move(*skew_or);
  generator_excess_kurtosis = std::move(*kurt_or);
  return {};
}

sim_expected<std::vector<MarginalSpec>>
fit_ig_generator_marginals(const Eigen::VectorXd& generator_skewness,
                           const Eigen::VectorXd& generator_excess_kurtosis,
                           const IgOptions& options) {
  std::vector<MarginalSpec> marginals;
  marginals.reserve(static_cast<std::size_t>(generator_skewness.size()));
  for (Eigen::Index j = 0; j < generator_skewness.size(); ++j) {
    MomentMatchSpec spec;
    spec.family = options.generator_family;
    spec.mean = 0.0;
    spec.sd = 1.0;
    spec.shape.skewness = generator_skewness(j);
    spec.shape.excess_kurtosis = generator_excess_kurtosis(j);
    auto fit_or = fit_marginal_to_moments(spec, options.moment_match);
    if (!fit_or.has_value()) return std::unexpected(fit_or.error());
    marginals.push_back(fit_or->marginal);
  }
  return marginals;
}

sim_expected<void>
validate_ig_root_sim_inputs(Eigen::Index n,
                            const Eigen::Ref<const Eigen::MatrixXd>& root,
                            const std::vector<MarginalSpec>& generator_marginals,
                            const IndependentOptions& options) {
  if (n <= 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_ig_matrix: n must be positive"));
  }
  if (root.rows() == 0 || root.cols() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_ig_matrix: root must not be empty"));
  }
  if (static_cast<Eigen::Index>(generator_marginals.size()) != root.cols()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_ig_matrix: generator marginal count must match root columns"));
  }
  if (!root.allFinite()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_ig_matrix: root must be finite"));
  }
  for (const auto& marginal : generator_marginals) {
    if (std::abs(marginal.mean) > 1e-12 ||
        std::abs(marginal.sd - 1.0) > 1e-12) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidMarginal,
          "simulate_ig_matrix: generator marginals must have mean 0 and sd 1"));
    }
  }
  return validate_independent_inputs(n, generator_marginals, options);
}

}  // namespace

MarginalSpec MarginalSpec::standard_normal(double mean, double sd) {
  MarginalSpec m;
  m.kind = MarginalKind::StandardNormal;
  m.mean = mean;
  m.sd = sd;
  return m;
}

MarginalSpec MarginalSpec::standardized_lognormal(double sigma_log,
                                                  double mean,
                                                  double sd) {
  MarginalSpec m;
  m.kind = MarginalKind::StandardizedLognormal;
  m.mean = mean;
  m.sd = sd;
  m.sigma_log = sigma_log;
  return m;
}

MarginalSpec MarginalSpec::tukey_g_h(double g,
                                     double h,
                                     double mean,
                                     double sd) {
  MarginalSpec m;
  m.kind = MarginalKind::TukeyGH;
  m.mean = mean;
  m.sd = sd;
  m.g = g;
  m.h = h;
  return m;
}

MarginalSpec MarginalSpec::pearson(int type,
                                   double p1,
                                   double p2,
                                   double p3,
                                   double p4,
                                   double mean,
                                   double sd) {
  MarginalSpec m;
  m.kind = MarginalKind::Pearson;
  m.mean = mean;
  m.sd = sd;
  m.pearson_type = type;
  m.pearson_p1 = p1;
  m.pearson_p2 = p2;
  m.pearson_p3 = p3;
  m.pearson_p4 = p4;
  return m;
}

MarginalSpec MarginalSpec::johnson(int type,
                                   double gamma,
                                   double delta,
                                   double mean,
                                   double sd) {
  MarginalSpec m;
  m.kind = MarginalKind::Johnson;
  m.mean = mean;
  m.sd = sd;
  m.johnson_type = type;
  m.johnson_gamma = gamma;
  m.johnson_delta = delta;
  return m;
}

MarginalSpec MarginalSpec::fleishman(double b,
                                     double c,
                                     double d,
                                     double mean,
                                     double sd) {
  MarginalSpec m;
  m.kind = MarginalKind::Fleishman;
  m.mean = mean;
  m.sd = sd;
  m.fleishman_b = b;
  m.fleishman_c = c;
  m.fleishman_d = d;
  return m;
}

double normal_cdf(double z) noexcept {
  return 0.5 * std::erfc(-z / kSqrt2);
}

sim_expected<double> normal_quantile(double u) {
  if (!std::isfinite(u) || u <= 0.0 || u >= 1.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "normal_quantile: u must satisfy 0 < u < 1"));
  }

  // Peter J. Acklam's rational approximation, refined once by Newton.
  constexpr double a1 = -3.969683028665376e+01;
  constexpr double a2 = 2.209460984245205e+02;
  constexpr double a3 = -2.759285104469687e+02;
  constexpr double a4 = 1.383577518672690e+02;
  constexpr double a5 = -3.066479806614716e+01;
  constexpr double a6 = 2.506628277459239e+00;
  constexpr double b1 = -5.447609879822406e+01;
  constexpr double b2 = 1.615858368580409e+02;
  constexpr double b3 = -1.556989798598866e+02;
  constexpr double b4 = 6.680131188771972e+01;
  constexpr double b5 = -1.328068155288572e+01;
  constexpr double c1 = -7.784894002430293e-03;
  constexpr double c2 = -3.223964580411365e-01;
  constexpr double c3 = -2.400758277161838e+00;
  constexpr double c4 = -2.549732539343734e+00;
  constexpr double c5 = 4.374664141464968e+00;
  constexpr double c6 = 2.938163982698783e+00;
  constexpr double d1 = 7.784695709041462e-03;
  constexpr double d2 = 3.224671290700398e-01;
  constexpr double d3 = 2.445134137142996e+00;
  constexpr double d4 = 3.754408661907416e+00;

  constexpr double plow = 0.02425;
  constexpr double phigh = 1.0 - plow;

  double x = 0.0;
  if (u < plow) {
    const double q = std::sqrt(-2.0 * std::log(u));
    x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  } else if (u > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - u));
    x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  } else {
    const double q = u - 0.5;
    const double r = q * q;
    x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  }

  const double err = normal_cdf(x) - u;
  const double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
  x -= err / pdf;
  return x;
}

sim_expected<double>
marginal_quantile(const MarginalSpec& marginal, double u) {
  if (marginal.kind == MarginalKind::Fleishman) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidMarginal,
        "marginal_quantile: Fleishman polynomial is a generator transform, not a guaranteed quantile"));
  }
  auto z_or = normal_quantile(u);
  if (!z_or.has_value()) return std::unexpected(z_or.error());
  auto gh_or = gauss_hermite(61);
  if (!gh_or.has_value()) return std::unexpected(gh_or.error());
  auto mom_or = marginal_moments(marginal, *gh_or);
  if (!mom_or.has_value()) return std::unexpected(mom_or.error());
  const double y = standardized_from_z(marginal, *mom_or, *z_or);
  return marginal.mean + marginal.sd * y;
}

sim_expected<MarginalMomentSummary>
marginal_moment_summary(const MarginalSpec& marginal, int quadrature_points) {
  auto gh_or = gauss_hermite(quadrature_points);
  if (!gh_or.has_value()) return std::unexpected(gh_or.error());
  return raw_moment_summary(marginal, *gh_or);
}

sim_expected<MomentMatchResult>
fit_marginal_to_moments(const MomentMatchSpec& spec,
                        const MomentMatchOptions& options) {
  if (auto ok = validate_moment_match(spec, options); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  switch (spec.family) {
    case MomentMatchFamily::TukeyGH: {
      auto gh_or = gauss_hermite(options.quadrature_points);
      if (!gh_or.has_value()) return std::unexpected(gh_or.error());
      return fit_tukey_moment_match(spec, options, *gh_or);
    }
    case MomentMatchFamily::Pearson:
      return fit_pearson_moment_match(spec);
    case MomentMatchFamily::Johnson: {
      auto gh_or = gauss_hermite(options.quadrature_points);
      if (!gh_or.has_value()) return std::unexpected(gh_or.error());
      return fit_johnson_moment_match(spec, options, *gh_or);
    }
    case MomentMatchFamily::Fleishman: {
      auto gh_or = gauss_hermite(options.quadrature_points);
      if (!gh_or.has_value()) return std::unexpected(gh_or.error());
      return fit_fleishman_moment_match(spec, options, *gh_or);
    }
  }
  return std::unexpected(make_err(
      SimError::Kind::InvalidInput,
      "fit_marginal_to_moments: unknown moment-match family"));
}

sim_expected<Eigen::MatrixXd>
simulate_independent_matrix(Eigen::Index n,
                            const std::vector<MarginalSpec>& marginals,
                            std::mt19937_64& rng,
                            const IndependentOptions& options) {
  if (auto ok = validate_independent_inputs(n, marginals, options); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  auto gh_or = gauss_hermite(options.quadrature_points);
  if (!gh_or.has_value()) return std::unexpected(gh_or.error());
  auto moments_or = build_marginal_moments(marginals, *gh_or);
  if (!moments_or.has_value()) return std::unexpected(moments_or.error());
  const auto& moments = *moments_or;

  const Eigen::Index p = static_cast<Eigen::Index>(marginals.size());
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index row = 0; row < n; ++row) {
    for (Eigen::Index j = 0; j < p; ++j) {
      const auto idx = static_cast<std::size_t>(j);
      const double z = normal(rng);
      const double y = standardized_from_z(marginals[idx], moments[idx], z);
      X(row, j) = marginals[idx].mean + marginals[idx].sd * y;
    }
  }
  return X;
}

sim_expected<data::RawData>
simulate_independent_raw(Eigen::Index n,
                         const std::vector<MarginalSpec>& marginals,
                         std::mt19937_64& rng,
                         const IndependentOptions& options) {
  auto X_or = simulate_independent_matrix(n, marginals, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  data::RawData raw;
  raw.X.push_back(std::move(*X_or));
  return raw;
}

sim_expected<IgCalibration>
calibrate_ig(const Eigen::Ref<const Eigen::MatrixXd>& sigma,
             const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
             const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
             const IgOptions& options) {
  if (auto ok = validate_ig_inputs(
          sigma, target_skewness, target_excess_kurtosis, options);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  auto root_or = ig_root_matrix(sigma, options);
  if (!root_or.has_value()) return std::unexpected(root_or.error());

  IgCalibration out;
  out.root = std::move(*root_or);
  if (auto ok = ig_generator_moments(out.root,
                                     target_skewness,
                                     target_excess_kurtosis,
                                     options.moment_solve_tol,
                                     out.generator_skewness,
                                     out.generator_excess_kurtosis);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  auto marginals_or = fit_ig_generator_marginals(
      out.generator_skewness, out.generator_excess_kurtosis, options);
  if (!marginals_or.has_value()) return std::unexpected(marginals_or.error());
  out.generator_marginals = std::move(*marginals_or);
  return out;
}

sim_expected<Eigen::MatrixXd>
simulate_ig_matrix(Eigen::Index n,
                   const Eigen::Ref<const Eigen::MatrixXd>& root,
                   const std::vector<MarginalSpec>& generator_marginals,
                   std::mt19937_64& rng,
                   const IndependentOptions& options) {
  if (auto ok = validate_ig_root_sim_inputs(
          n, root, generator_marginals, options);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  auto X_or = simulate_independent_matrix(n, generator_marginals, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  return (*X_or) * root.transpose();
}

sim_expected<Eigen::MatrixXd>
simulate_ig_matrix(Eigen::Index n,
                   const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                   const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
                   const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
                   std::mt19937_64& rng,
                   const IgOptions& options) {
  auto cal_or = calibrate_ig(
      sigma, target_skewness, target_excess_kurtosis, options);
  if (!cal_or.has_value()) return std::unexpected(cal_or.error());

  IndependentOptions independent_options;
  independent_options.quadrature_points = options.moment_match.quadrature_points;
  return simulate_ig_matrix(
      n, cal_or->root, cal_or->generator_marginals, rng, independent_options);
}

sim_expected<data::RawData>
simulate_ig_raw(Eigen::Index n,
                const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                const Eigen::Ref<const Eigen::VectorXd>& target_skewness,
                const Eigen::Ref<const Eigen::VectorXd>& target_excess_kurtosis,
                std::mt19937_64& rng,
                const IgOptions& options) {
  auto X_or = simulate_ig_matrix(
      n, sigma, target_skewness, target_excess_kurtosis, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  data::RawData raw;
  raw.X.push_back(std::move(*X_or));
  return raw;
}

sim_expected<NortaCalibration>
calibrate_norta(const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
                const std::vector<MarginalSpec>& marginals,
                const NortaOptions& options) {
  if (auto ok = validate_target(target_corr, marginals, options); !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  auto gh_or = gauss_hermite(options.quadrature_points);
  if (!gh_or.has_value()) return std::unexpected(gh_or.error());
  const auto& gh = *gh_or;

  auto moments_or = build_marginal_moments(marginals, gh);
  if (!moments_or.has_value()) return std::unexpected(moments_or.error());
  const auto& moments = *moments_or;

  const Eigen::Index p = target_corr.rows();
  NortaCalibration out;
  out.latent_corr = Eigen::MatrixXd::Identity(p, p);
  out.marginal_mean.resize(p);
  out.marginal_sd.resize(p);
  for (Eigen::Index i = 0; i < p; ++i) {
    out.marginal_mean(i) = marginals[static_cast<std::size_t>(i)].mean;
    out.marginal_sd(i) = marginals[static_cast<std::size_t>(i)].sd;
  }

  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = i + 1; j < p; ++j) {
      auto rho_or = calibrate_pair(
          target_corr(i, j),
          marginals[static_cast<std::size_t>(i)], moments[static_cast<std::size_t>(i)],
          marginals[static_cast<std::size_t>(j)], moments[static_cast<std::size_t>(j)],
          gh, options);
      if (!rho_or.has_value()) return std::unexpected(rho_or.error());
      out.latent_corr(i, j) = *rho_or;
      out.latent_corr(j, i) = *rho_or;
    }
  }

  auto L_or = cholesky_factor_with_jitter(out.latent_corr, options.cholesky_jitter);
  if (!L_or.has_value()) return std::unexpected(L_or.error());
  return out;
}

sim_expected<Eigen::MatrixXd>
simulate_norta_matrix(Eigen::Index n,
                      const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
                      const std::vector<MarginalSpec>& marginals,
                      std::mt19937_64& rng,
                      const NortaOptions& options) {
  if (n <= 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_norta_matrix: n must be positive"));
  }

  auto cal_or = calibrate_norta(target_corr, marginals, options);
  if (!cal_or.has_value()) return std::unexpected(cal_or.error());
  auto L_or = cholesky_factor_with_jitter(cal_or->latent_corr, options.cholesky_jitter);
  if (!L_or.has_value()) return std::unexpected(L_or.error());

  auto gh_or = gauss_hermite(options.quadrature_points);
  if (!gh_or.has_value()) return std::unexpected(gh_or.error());
  auto moments_or = build_marginal_moments(marginals, *gh_or);
  if (!moments_or.has_value()) return std::unexpected(moments_or.error());
  const auto& moments = *moments_or;

  const Eigen::Index p = target_corr.rows();
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index row = 0; row < n; ++row) {
    Eigen::VectorXd eps(p);
    for (Eigen::Index j = 0; j < p; ++j) eps(j) = normal(rng);
    const Eigen::VectorXd z = (*L_or) * eps;
    for (Eigen::Index j = 0; j < p; ++j) {
      const auto idx = static_cast<std::size_t>(j);
      const double y = standardized_from_z(marginals[idx], moments[idx], z(j));
      X(row, j) = marginals[idx].mean + marginals[idx].sd * y;
    }
  }
  return X;
}

sim_expected<data::RawData>
simulate_norta_raw(Eigen::Index n,
                   const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
                   const std::vector<MarginalSpec>& marginals,
                   std::mt19937_64& rng,
                   const NortaOptions& options) {
  auto X_or = simulate_norta_matrix(n, target_corr, marginals, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  data::RawData raw;
  raw.X.push_back(std::move(*X_or));
  return raw;
}

sim_expected<Eigen::MatrixXd>
simulate_t_copula_matrix(Eigen::Index n,
                         const TCopulaSpec& copula,
                         const std::vector<MarginalSpec>& marginals,
                         std::mt19937_64& rng,
                         const TCopulaOptions& options) {
  if (auto ok = validate_t_copula_inputs(n, copula, marginals, options);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  auto L_or = cholesky_factor_with_jitter(copula.corr, options.cholesky_jitter);
  if (!L_or.has_value()) return std::unexpected(L_or.error());

  auto gh_or = gauss_hermite(options.quadrature_points);
  if (!gh_or.has_value()) return std::unexpected(gh_or.error());
  auto moments_or = build_marginal_moments(marginals, *gh_or);
  if (!moments_or.has_value()) return std::unexpected(moments_or.error());
  const auto& moments = *moments_or;

  const Eigen::Index p = copula.corr.rows();
  std::normal_distribution<double> normal(0.0, 1.0);
  std::chi_squared_distribution<double> chi_square(copula.df);
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index row = 0; row < n; ++row) {
    Eigen::VectorXd eps(p);
    for (Eigen::Index j = 0; j < p; ++j) eps(j) = normal(rng);
    const Eigen::VectorXd z = (*L_or) * eps;
    const double w = chi_square(rng);
    if (!(w > 0.0) || !std::isfinite(w)) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "simulate_t_copula_matrix: non-positive chi-square draw"));
    }
    const double scale = std::sqrt(copula.df / w);
    for (Eigen::Index j = 0; j < p; ++j) {
      const auto idx = static_cast<std::size_t>(j);
      const double u = student_t_cdf(z(j) * scale, copula.df);
      if (!std::isfinite(u)) {
        return std::unexpected(make_err(
            SimError::Kind::NumericIssue,
            "simulate_t_copula_matrix: non-finite t copula probability"));
      }
      auto x_or = marginal_from_unit(marginals[idx], moments[idx], u);
      if (!x_or.has_value()) return std::unexpected(x_or.error());
      X(row, j) = *x_or;
    }
  }
  return X;
}

sim_expected<data::RawData>
simulate_t_copula_raw(Eigen::Index n,
                      const TCopulaSpec& copula,
                      const std::vector<MarginalSpec>& marginals,
                      std::mt19937_64& rng,
                      const TCopulaOptions& options) {
  auto X_or = simulate_t_copula_matrix(n, copula, marginals, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  data::RawData raw;
  raw.X.push_back(std::move(*X_or));
  return raw;
}

sim_expected<double>
bivariate_copula_conditional_cdf(const BivariateCopulaSpec& copula,
                                 double u,
                                 double v) {
  if (auto ok = validate_bivariate_copula_spec(copula); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (!std::isfinite(u) || !std::isfinite(v) ||
      u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "bivariate_copula_conditional_cdf: u and v must satisfy 0 < value < 1"));
  }
  const double h = bivariate_conditional_u(copula, u, v);
  if (!std::isfinite(h)) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "bivariate_copula_conditional_cdf: non-finite conditional CDF"));
  }
  return std::clamp(h, 0.0, 1.0);
}

sim_expected<double>
bivariate_copula_tau(const BivariateCopulaSpec& copula) {
  if (auto ok = validate_bivariate_copula_spec(copula); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  const double tau = bivariate_tau_unchecked(copula);
  if (!std::isfinite(tau)) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "bivariate_copula_tau: non-finite tau"));
  }
  return std::clamp(tau, -1.0, 1.0);
}

sim_expected<BivariateCopulaSpec>
bivariate_copula_from_tau(BivariateCopulaFamily family,
                          double tau,
                          const BivariateCopulaOptions& options) {
  if (!std::isfinite(tau) || tau <= -1.0 || tau >= 1.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "bivariate_copula_from_tau: tau must satisfy -1 < tau < 1"));
  }
  if (options.max_bisection_iter < 16) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "bivariate_copula_from_tau: max_bisection_iter must be at least 16"));
  }

  BivariateCopulaSpec out;
  out.family = family;
  switch (family) {
    case BivariateCopulaFamily::Independence:
      if (std::abs(tau) > 1e-14) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "bivariate_copula_from_tau: independence requires tau = 0"));
      }
      out.theta = 0.0;
      return out;
    case BivariateCopulaFamily::Clayton:
      if (tau < 0.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "bivariate_copula_from_tau: Clayton requires nonnegative tau"));
      }
      out.theta = tau == 0.0 ? 0.0 : 2.0 * tau / (1.0 - tau);
      if (out.theta == 0.0) {
        out.family = BivariateCopulaFamily::Independence;
      }
      return out;
    case BivariateCopulaFamily::Gumbel:
      if (tau < 0.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "bivariate_copula_from_tau: Gumbel requires nonnegative tau"));
      }
      out.theta = 1.0 / (1.0 - tau);
      return out;
    case BivariateCopulaFamily::Frank: {
      if (std::abs(tau) < 1e-14) {
        out.family = BivariateCopulaFamily::Independence;
        out.theta = 0.0;
        return out;
      }
      auto theta_or = invert_frank_tau(tau, options.max_bisection_iter);
      if (!theta_or.has_value()) return std::unexpected(theta_or.error());
      out.theta = *theta_or;
      return out;
    }
    case BivariateCopulaFamily::Joe: {
      if (tau < 0.0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "bivariate_copula_from_tau: Joe requires nonnegative tau"));
      }
      if (tau == 0.0) {
        out.theta = 1.0;
        return out;
      }
      auto theta_or = invert_tau_bisection(
          BivariateCopulaFamily::Joe, tau, 1.0, 2.0,
          options.max_bisection_iter);
      if (!theta_or.has_value()) return std::unexpected(theta_or.error());
      out.theta = *theta_or;
      return out;
    }
  }
  return std::unexpected(make_err(
      SimError::Kind::InvalidInput,
      "bivariate_copula_from_tau: unknown family"));
}

sim_expected<double>
bivariate_copula_observed_corr(
    const BivariateCopulaSpec& copula,
    const std::vector<MarginalSpec>& marginals,
    const BivariateCopulaOptions& options) {
  if (auto ok = validate_bivariate_copula_spec(copula); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (options.quadrature_points < 8 || options.max_bisection_iter < 16 ||
      !std::isfinite(options.calibration_tol) || options.calibration_tol <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "bivariate_copula_observed_corr: invalid options"));
  }
  if (auto ok = validate_bivariate_quantile_marginals(
          marginals, "bivariate_copula_observed_corr");
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  auto gl_or = gauss_legendre_unit(options.quadrature_points);
  if (!gl_or.has_value()) return std::unexpected(gl_or.error());
  auto gh_or = gauss_hermite(options.quadrature_points);
  if (!gh_or.has_value()) return std::unexpected(gh_or.error());
  auto moments_or = build_marginal_moments(marginals, *gh_or);
  if (!moments_or.has_value()) return std::unexpected(moments_or.error());
  const auto& gl = *gl_or;
  const auto& moments = *moments_or;

  double corr = 0.0;
  for (Eigen::Index a = 0; a < gl.nodes.size(); ++a) {
    const double u = open_unit(gl.nodes(a));
    auto x_or = marginal_from_unit(marginals[0], moments[0], u);
    if (!x_or.has_value()) return std::unexpected(x_or.error());
    const double x = (*x_or - marginals[0].mean) / marginals[0].sd;
    for (Eigen::Index b = 0; b < gl.nodes.size(); ++b) {
      const double w = open_unit(gl.nodes(b));
      auto v_or = invert_bivariate_conditional(
          copula, u, w, options.max_bisection_iter);
      if (!v_or.has_value()) return std::unexpected(v_or.error());
      auto y_or = marginal_from_unit(marginals[1], moments[1], *v_or);
      if (!y_or.has_value()) return std::unexpected(y_or.error());
      const double y = (*y_or - marginals[1].mean) / marginals[1].sd;
      corr += gl.weights(a) * gl.weights(b) * x * y;
    }
  }
  if (!std::isfinite(corr)) {
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "bivariate_copula_observed_corr: non-finite correlation"));
  }
  return std::clamp(corr, -1.0, 1.0);
}

sim_expected<BivariateCopulaCorrelationCalibration>
calibrate_bivariate_copula_correlation(
    BivariateCopulaFamily family,
    double target_corr,
    const std::vector<MarginalSpec>& marginals,
    const BivariateCopulaOptions& options) {
  if (!std::isfinite(target_corr) || target_corr <= -1.0 ||
      target_corr >= 1.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_bivariate_copula_correlation: target_corr must satisfy -1 < r < 1"));
  }
  if (options.quadrature_points < 8 || options.max_bisection_iter < 16 ||
      !std::isfinite(options.calibration_tol) || options.calibration_tol <= 0.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_bivariate_copula_correlation: invalid options"));
  }
  if (auto ok = validate_bivariate_quantile_marginals(
          marginals, "calibrate_bivariate_copula_correlation");
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  if (family == BivariateCopulaFamily::Independence) {
    if (std::abs(target_corr) > options.calibration_tol) {
      return std::unexpected(make_err(
          SimError::Kind::CalibrationFailed,
          "calibrate_bivariate_copula_correlation: independence can only target zero correlation"));
    }
    BivariateCopulaCorrelationCalibration out;
    out.copula = BivariateCopulaSpec{};
    out.target_corr = target_corr;
    out.achieved_corr = 0.0;
    out.lower_bound_corr = 0.0;
    out.upper_bound_corr = 0.0;
    out.iterations = 0;
    return out;
  }

  const auto eval_tau =
      [&](double tau) -> sim_expected<std::pair<BivariateCopulaSpec, double>> {
    auto spec_or = bivariate_copula_from_tau(family, tau, options);
    if (!spec_or.has_value()) return std::unexpected(spec_or.error());
    auto corr_or = bivariate_copula_observed_corr(*spec_or, marginals, options);
    if (!corr_or.has_value()) return std::unexpected(corr_or.error());
    return std::pair<BivariateCopulaSpec, double>{*spec_or, *corr_or};
  };

  const auto stable_endpoint =
      [&](double sign) -> sim_expected<std::pair<double, double>> {
    for (double tau_abs : {0.95, 0.90, 0.80, 0.70, 0.60, 0.50}) {
      const double tau = sign * tau_abs;
      auto eval_or = eval_tau(tau);
      if (eval_or.has_value()) {
        return std::pair<double, double>{tau, eval_or->second};
      }
      if (eval_or.error().kind != SimError::Kind::NumericIssue) {
        return std::unexpected(eval_or.error());
      }
    }
    return std::unexpected(make_err(
        SimError::Kind::NumericIssue,
        "calibrate_bivariate_copula_correlation: could not find stable endpoint"));
  };

  auto lo_endpoint_or = family == BivariateCopulaFamily::Frank
      ? stable_endpoint(-1.0)
      : sim_expected<std::pair<double, double>>{std::pair<double, double>{0.0, 0.0}};
  auto hi_endpoint_or = stable_endpoint(1.0);
  if (!lo_endpoint_or.has_value()) return std::unexpected(lo_endpoint_or.error());
  if (!hi_endpoint_or.has_value()) return std::unexpected(hi_endpoint_or.error());
  double lo_tau = lo_endpoint_or->first;
  double hi_tau = hi_endpoint_or->first;
  double lo_corr = lo_endpoint_or->second;
  double hi_corr = hi_endpoint_or->second;
  if (lo_corr > hi_corr) {
    std::swap(lo_tau, hi_tau);
    std::swap(lo_corr, hi_corr);
  }

  BivariateCopulaCorrelationCalibration out;
  out.target_corr = target_corr;
  out.lower_bound_corr = lo_corr;
  out.upper_bound_corr = hi_corr;
  if (target_corr < lo_corr - options.calibration_tol ||
      target_corr > hi_corr + options.calibration_tol) {
    return std::unexpected(make_err(
        SimError::Kind::CalibrationFailed,
        "calibrate_bivariate_copula_correlation: target correlation outside feasible family range"));
  }

  double best_tau = lo_tau;
  double best_corr = lo_corr;
  if (std::abs(target_corr - hi_corr) < std::abs(target_corr - best_corr)) {
    best_tau = hi_tau;
    best_corr = hi_corr;
  }

  for (int iter = 0; iter < options.max_bisection_iter; ++iter) {
    const double mid_tau = 0.5 * (lo_tau + hi_tau);
    auto mid_spec_or = bivariate_copula_from_tau(family, mid_tau, options);
    if (!mid_spec_or.has_value()) return std::unexpected(mid_spec_or.error());
    auto mid_corr_or = bivariate_copula_observed_corr(
        *mid_spec_or, marginals, options);
    if (!mid_corr_or.has_value()) return std::unexpected(mid_corr_or.error());
    const double mid_corr = *mid_corr_or;
    if (std::abs(target_corr - mid_corr) < std::abs(target_corr - best_corr)) {
      best_tau = mid_tau;
      best_corr = mid_corr;
    }
    out.iterations = iter + 1;
    if (std::abs(mid_corr - target_corr) <= options.calibration_tol ||
        std::abs(hi_tau - lo_tau) <= options.calibration_tol) {
      break;
    }
    if (mid_corr < target_corr) {
      lo_tau = mid_tau;
      lo_corr = mid_corr;
    } else {
      hi_tau = mid_tau;
      hi_corr = mid_corr;
    }
  }

  auto best_spec_or = bivariate_copula_from_tau(family, best_tau, options);
  if (!best_spec_or.has_value()) return std::unexpected(best_spec_or.error());
  out.copula = *best_spec_or;
  out.achieved_corr = best_corr;
  return out;
}

sim_expected<BivariateCopulaMatrixCalibration>
calibrate_bivariate_copula_correlation_matrix(
    BivariateCopulaFamily family,
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const std::vector<MarginalSpec>& marginals,
    const BivariateCopulaOptions& options) {
  if (auto ok = validate_bivariate_copula_matrix_target(
          target_corr, marginals, options);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  const Eigen::Index p = target_corr.rows();
  BivariateCopulaMatrixCalibration out;
  out.family = family;
  out.theta = Eigen::MatrixXd::Zero(p, p);
  out.achieved_corr = Eigen::MatrixXd::Identity(p, p);
  out.lower_bound_corr = Eigen::MatrixXd::Identity(p, p);
  out.upper_bound_corr = Eigen::MatrixXd::Identity(p, p);
  out.iterations = Eigen::MatrixXi::Zero(p, p);

  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = i + 1; j < p; ++j) {
      const std::vector<MarginalSpec> pair_marginals{
          marginals[static_cast<std::size_t>(i)],
          marginals[static_cast<std::size_t>(j)]};
      auto pair_or = calibrate_bivariate_copula_correlation(
          family, target_corr(i, j), pair_marginals, options);
      if (!pair_or.has_value()) return std::unexpected(pair_or.error());
      const auto& pair = *pair_or;
      out.theta(i, j) = pair.copula.theta;
      out.theta(j, i) = pair.copula.theta;
      out.achieved_corr(i, j) = pair.achieved_corr;
      out.achieved_corr(j, i) = pair.achieved_corr;
      out.lower_bound_corr(i, j) = pair.lower_bound_corr;
      out.lower_bound_corr(j, i) = pair.lower_bound_corr;
      out.upper_bound_corr(i, j) = pair.upper_bound_corr;
      out.upper_bound_corr(j, i) = pair.upper_bound_corr;
      out.iterations(i, j) = pair.iterations;
      out.iterations(j, i) = pair.iterations;
    }
  }

  return out;
}

sim_expected<double>
bivariate_copula_conditional_quantile(
    const BivariateCopulaSpec& copula,
    double u,
    double p,
    const BivariateCopulaOptions& options) {
  if (auto ok = validate_bivariate_copula_spec(copula); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (!std::isfinite(u) || !std::isfinite(p) ||
      u <= 0.0 || u >= 1.0 || p <= 0.0 || p >= 1.0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "bivariate_copula_conditional_quantile: u and p must satisfy 0 < value < 1"));
  }
  if (options.max_bisection_iter < 16) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "bivariate_copula_conditional_quantile: max_bisection_iter must be at least 16"));
  }
  return invert_bivariate_conditional(
      copula, u, p, options.max_bisection_iter);
}

sim_expected<Eigen::MatrixXd>
simulate_bivariate_copula_matrix(Eigen::Index n,
                                 const BivariateCopulaSpec& copula,
                                 const std::vector<MarginalSpec>& marginals,
                                 std::mt19937_64& rng,
                                 const BivariateCopulaOptions& options) {
  if (auto ok = validate_bivariate_copula_inputs(n, copula, marginals, options);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  auto gh_or = gauss_hermite(options.quadrature_points);
  if (!gh_or.has_value()) return std::unexpected(gh_or.error());
  auto moments_or = build_marginal_moments(marginals, *gh_or);
  if (!moments_or.has_value()) return std::unexpected(moments_or.error());
  const auto& moments = *moments_or;

  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  Eigen::MatrixXd X(n, 2);
  for (Eigen::Index row = 0; row < n; ++row) {
    const double u = open_unit(uniform(rng));
    const double w = open_unit(uniform(rng));
    auto v_or = invert_bivariate_conditional(
        copula, u, w, options.max_bisection_iter);
    if (!v_or.has_value()) return std::unexpected(v_or.error());

    auto x0_or = marginal_from_unit(marginals[0], moments[0], u);
    if (!x0_or.has_value()) return std::unexpected(x0_or.error());
    auto x1_or = marginal_from_unit(marginals[1], moments[1], *v_or);
    if (!x1_or.has_value()) return std::unexpected(x1_or.error());
    X(row, 0) = *x0_or;
    X(row, 1) = *x1_or;
  }
  return X;
}

sim_expected<data::RawData>
simulate_bivariate_copula_raw(Eigen::Index n,
                              const BivariateCopulaSpec& copula,
                              const std::vector<MarginalSpec>& marginals,
                              std::mt19937_64& rng,
                              const BivariateCopulaOptions& options) {
  auto X_or = simulate_bivariate_copula_matrix(
      n, copula, marginals, rng, options);
  if (!X_or.has_value()) return std::unexpected(X_or.error());
  data::RawData raw;
  raw.X.push_back(std::move(*X_or));
  return raw;
}

}  // namespace magmaan::sim
