#include "magmaan/sim/norta.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

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

struct MarginalMoments {
  double mean = 0.0;
  double sd = 1.0;
};

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

double raw_from_z(const MarginalSpec& m, double z) {
  switch (m.kind) {
    case MarginalKind::StandardNormal:
      return normal_from_z(m, z);
    case MarginalKind::StandardizedLognormal:
      return standardized_lognormal_from_z(m, z);
    case MarginalKind::TukeyGH:
      return tukey_g_h_raw_from_z(m, z);
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

  double e1 = 0.0;
  double e2 = 0.0;
  for (Eigen::Index a = 0; a < gh.nodes.size(); ++a) {
    const double z = kSqrt2 * gh.nodes(a);
    const double w = gh.weights(a) / kSqrtPi;
    const double y = raw_from_z(m, z);
    if (!std::isfinite(y)) {
      return std::unexpected(make_err(
          SimError::Kind::NumericIssue,
          "marginal moments: non-finite Tukey g-and-h transform"));
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
  auto z_or = normal_quantile(u);
  if (!z_or.has_value()) return std::unexpected(z_or.error());
  auto gh_or = gauss_hermite(61);
  if (!gh_or.has_value()) return std::unexpected(gh_or.error());
  auto mom_or = marginal_moments(marginal, *gh_or);
  if (!mom_or.has_value()) return std::unexpected(mom_or.error());
  const double y = standardized_from_z(marginal, *mom_or, *z_or);
  return marginal.mean + marginal.sd * y;
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

}  // namespace magmaan::sim
