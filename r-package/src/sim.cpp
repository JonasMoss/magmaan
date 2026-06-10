#include <RcppEigen.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "magmaan/error.hpp"
#include "magmaan/sim/norta.hpp"
#include "magmaan/sim/plsim.hpp"

namespace {

const char* sim_error_kind(magmaan::SimError::Kind k) {
  using K = magmaan::SimError::Kind;
  switch (k) {
    case K::InvalidInput: return "InvalidInput";
    case K::InvalidMarginal: return "InvalidMarginal";
    case K::CalibrationFailed: return "CalibrationFailed";
    case K::NonPositiveDefinite: return "NonPositiveDefinite";
    case K::NumericIssue: return "NumericIssue";
  }
  return "Unknown";
}

[[noreturn]] void stop_sim(const magmaan::SimError& e) {
  Rcpp::stop("magmaan simulation error [%s]: %s",
             sim_error_kind(e.kind), e.detail);
}

magmaan::sim::PlsimCovarianceMethod plsim_method_from_string(
    const std::string& method) {
  using M = magmaan::sim::PlsimCovarianceMethod;
  if (method == "hermite") return M::Hermite;
  if (method == "quadrature") return M::Quadrature;
  if (method == "rectangle") return M::Rectangle;
  if (method == "hermite_then_quadrature") return M::HermiteThenQuadrature;
  if (method == "hermite_then_rectangle") return M::HermiteThenRectangle;
  Rcpp::stop("unknown PLSIM covariance method: %s", method);
}

magmaan::sim::IgRootKind ig_root_from_string(const std::string& root) {
  using R = magmaan::sim::IgRootKind;
  if (root == "cholesky") return R::Cholesky;
  if (root == "symmetric") return R::Symmetric;
  Rcpp::stop("unknown IG root kind: %s", root);
}

magmaan::sim::MomentMatchFamily moment_match_family_from_string(
    const std::string& family) {
  using F = magmaan::sim::MomentMatchFamily;
  if (family == "tukey_gh") return F::TukeyGH;
  if (family == "pearson") return F::Pearson;
  if (family == "johnson") return F::Johnson;
  if (family == "fleishman") return F::Fleishman;
  Rcpp::stop("unknown moment-match family: %s", family);
}

magmaan::sim::IgOptions make_ig_options(
    const std::string& root,
    const std::string& generator_family,
    int quadrature_points,
    int max_iter,
    int grid_points_g,
    int grid_points_h,
    double objective_tol,
    double parameter_tol,
    double finite_diff_step,
    double tukey_g_bound,
    double tukey_h_upper,
    double johnson_gamma_bound,
    double johnson_log_delta_lower,
    double johnson_log_delta_upper,
    double root_eigen_tol,
    double moment_solve_tol) {
  magmaan::sim::IgOptions options;
  options.root = ig_root_from_string(root);
  options.generator_family = moment_match_family_from_string(generator_family);
  options.moment_match.quadrature_points = quadrature_points;
  options.moment_match.max_iter = max_iter;
  options.moment_match.grid_points_g = grid_points_g;
  options.moment_match.grid_points_h = grid_points_h;
  options.moment_match.objective_tol = objective_tol;
  options.moment_match.parameter_tol = parameter_tol;
  options.moment_match.finite_diff_step = finite_diff_step;
  options.moment_match.tukey_g_bound = tukey_g_bound;
  options.moment_match.tukey_h_upper = tukey_h_upper;
  options.moment_match.johnson_gamma_bound = johnson_gamma_bound;
  options.moment_match.johnson_log_delta_lower = johnson_log_delta_lower;
  options.moment_match.johnson_log_delta_upper = johnson_log_delta_upper;
  options.root_eigen_tol = root_eigen_tol;
  options.moment_solve_tol = moment_solve_tol;
  return options;
}

magmaan::sim::PlsimOptions make_plsim_options(
    int num_segments,
    bool monotone,
    int max_iter,
    int quadrature_points,
    int hermite_order,
    double marginal_tol,
    double correlation_tol,
    double rho_bound) {
  magmaan::sim::PlsimOptions options;
  options.num_segments = num_segments;
  options.monotone = monotone;
  options.max_iter = max_iter;
  options.quadrature_points = quadrature_points;
  options.hermite_order = hermite_order;
  options.marginal_tol = marginal_tol;
  options.correlation_tol = correlation_tol;
  options.rho_bound = rho_bound;
  return options;
}

magmaan::sim::NortaOptions make_norta_options(
    int quadrature_points,
    int max_bisection_iter,
    double rho_bound,
    double calibration_tol,
    double cholesky_jitter) {
  magmaan::sim::NortaOptions options;
  options.quadrature_points = quadrature_points;
  options.max_bisection_iter = max_bisection_iter;
  options.rho_bound = rho_bound;
  options.calibration_tol = calibration_tol;
  options.cholesky_jitter = cholesky_jitter;
  return options;
}

magmaan::sim::BivariateCopulaFamily bivariate_copula_family_from_string(
    const std::string& family) {
  using F = magmaan::sim::BivariateCopulaFamily;
  if (family == "independence" || family == "indep") return F::Independence;
  if (family == "clayton") return F::Clayton;
  if (family == "gumbel") return F::Gumbel;
  if (family == "frank") return F::Frank;
  if (family == "joe") return F::Joe;
  Rcpp::stop("unknown bivariate copula family: %s", family);
}

const char* bivariate_copula_family_to_string(
    magmaan::sim::BivariateCopulaFamily family) {
  using F = magmaan::sim::BivariateCopulaFamily;
  switch (family) {
    case F::Independence: return "independence";
    case F::Clayton: return "clayton";
    case F::Gumbel: return "gumbel";
    case F::Frank: return "frank";
    case F::Joe: return "joe";
  }
  return "unknown";
}

magmaan::sim::BivariateCopulaCorrelationRepairKind
bivariate_copula_repair_from_string(const std::string& repair) {
  using R = magmaan::sim::BivariateCopulaCorrelationRepairKind;
  if (repair == "none") return R::None;
  if (repair == "error") return R::Error;
  if (repair == "ridge") return R::Ridge;
  if (repair == "shrinkage") return R::Shrinkage;
  Rcpp::stop("unknown bivariate copula matrix repair kind: %s", repair);
}

const char* bivariate_copula_repair_to_string(
    magmaan::sim::BivariateCopulaCorrelationRepairKind repair) {
  using R = magmaan::sim::BivariateCopulaCorrelationRepairKind;
  switch (repair) {
    case R::None: return "none";
    case R::Error: return "error";
    case R::Ridge: return "ridge";
    case R::Shrinkage: return "shrinkage";
  }
  return "unknown";
}

magmaan::sim::BivariateCopulaOptions make_bivariate_copula_options(
    int quadrature_points,
    int max_bisection_iter,
    double calibration_tol,
    const std::string& matrix_repair = "none",
    double matrix_repair_min_eigenvalue = 1e-8) {
  magmaan::sim::BivariateCopulaOptions options;
  options.quadrature_points = quadrature_points;
  options.max_bisection_iter = max_bisection_iter;
  options.calibration_tol = calibration_tol;
  options.matrix_repair = bivariate_copula_repair_from_string(matrix_repair);
  options.matrix_repair_min_eigenvalue = matrix_repair_min_eigenvalue;
  return options;
}

const char* marginal_kind_to_string(magmaan::sim::MarginalKind kind) {
  using K = magmaan::sim::MarginalKind;
  switch (kind) {
    case K::StandardNormal: return "standard_normal";
    case K::StandardizedLognormal: return "standardized_lognormal";
    case K::TukeyGH: return "tukey_gh";
    case K::Pearson: return "pearson";
    case K::Johnson: return "johnson";
    case K::Fleishman: return "fleishman";
  }
  return "unknown";
}

magmaan::sim::MarginalKind marginal_kind_from_string(const std::string& kind) {
  using K = magmaan::sim::MarginalKind;
  if (kind == "standard_normal") return K::StandardNormal;
  if (kind == "standardized_lognormal") return K::StandardizedLognormal;
  if (kind == "tukey_gh") return K::TukeyGH;
  if (kind == "pearson") return K::Pearson;
  if (kind == "johnson") return K::Johnson;
  if (kind == "fleishman") return K::Fleishman;
  Rcpp::stop("unknown marginal kind: %s", kind);
}

bool has_named(Rcpp::List x, const char* name) {
  return x.containsElementNamed(name) && !Rcpp::RObject(x[name]).isNULL();
}

int list_int_or(Rcpp::List x, const char* name, int fallback) {
  return has_named(x, name) ? Rcpp::as<int>(x[name]) : fallback;
}

double list_double_or(Rcpp::List x, const char* name, double fallback) {
  return has_named(x, name) ? Rcpp::as<double>(x[name]) : fallback;
}

Rcpp::List marginal_spec_to_list(const magmaan::sim::MarginalSpec& m) {
  return Rcpp::List::create(
      Rcpp::_["kind"] = marginal_kind_to_string(m.kind),
      Rcpp::_["mean"] = m.mean,
      Rcpp::_["sd"] = m.sd,
      Rcpp::_["sigma_log"] = m.sigma_log,
      Rcpp::_["g"] = m.g,
      Rcpp::_["h"] = m.h,
      Rcpp::_["pearson_type"] = m.pearson_type,
      Rcpp::_["pearson_p1"] = m.pearson_p1,
      Rcpp::_["pearson_p2"] = m.pearson_p2,
      Rcpp::_["pearson_p3"] = m.pearson_p3,
      Rcpp::_["pearson_p4"] = m.pearson_p4,
      Rcpp::_["johnson_type"] = m.johnson_type,
      Rcpp::_["johnson_gamma"] = m.johnson_gamma,
      Rcpp::_["johnson_delta"] = m.johnson_delta,
      Rcpp::_["fleishman_b"] = m.fleishman_b,
      Rcpp::_["fleishman_c"] = m.fleishman_c,
      Rcpp::_["fleishman_d"] = m.fleishman_d);
}

magmaan::sim::MarginalSpec marginal_spec_from_list(Rcpp::List x) {
  magmaan::sim::MarginalSpec m;
  if (has_named(x, "kind")) {
    m.kind = marginal_kind_from_string(Rcpp::as<std::string>(x["kind"]));
  }
  m.mean = list_double_or(x, "mean", m.mean);
  m.sd = list_double_or(x, "sd", m.sd);
  m.sigma_log = list_double_or(x, "sigma_log", m.sigma_log);
  m.g = list_double_or(x, "g", m.g);
  m.h = list_double_or(x, "h", m.h);
  m.pearson_type = list_int_or(x, "pearson_type", m.pearson_type);
  m.pearson_p1 = list_double_or(x, "pearson_p1", m.pearson_p1);
  m.pearson_p2 = list_double_or(x, "pearson_p2", m.pearson_p2);
  m.pearson_p3 = list_double_or(x, "pearson_p3", m.pearson_p3);
  m.pearson_p4 = list_double_or(x, "pearson_p4", m.pearson_p4);
  m.johnson_type = list_int_or(x, "johnson_type", m.johnson_type);
  m.johnson_gamma = list_double_or(x, "johnson_gamma", m.johnson_gamma);
  m.johnson_delta = list_double_or(x, "johnson_delta", m.johnson_delta);
  m.fleishman_b = list_double_or(x, "fleishman_b", m.fleishman_b);
  m.fleishman_c = list_double_or(x, "fleishman_c", m.fleishman_c);
  m.fleishman_d = list_double_or(x, "fleishman_d", m.fleishman_d);
  return m;
}

Rcpp::List marginal_specs_to_list(
    const std::vector<magmaan::sim::MarginalSpec>& marginals) {
  Rcpp::List out(static_cast<R_xlen_t>(marginals.size()));
  for (std::size_t i = 0; i < marginals.size(); ++i) {
    out[static_cast<R_xlen_t>(i)] = marginal_spec_to_list(marginals[i]);
  }
  return out;
}

std::vector<magmaan::sim::MarginalSpec>
marginal_specs_from_list(Rcpp::List xs) {
  std::vector<magmaan::sim::MarginalSpec> out;
  out.reserve(static_cast<std::size_t>(xs.size()));
  for (R_xlen_t i = 0; i < xs.size(); ++i) {
    out.push_back(marginal_spec_from_list(Rcpp::as<Rcpp::List>(xs[i])));
  }
  return out;
}

Rcpp::List plsim_marginal_to_list(const magmaan::sim::PlsimMarginal& m) {
  return Rcpp::List::create(
      Rcpp::_["gamma"] = Rcpp::wrap(m.gamma),
      Rcpp::_["a"] = Rcpp::wrap(m.a),
      Rcpp::_["b"] = Rcpp::wrap(m.b),
      Rcpp::_["hermite_coefficients"] = Rcpp::wrap(m.hermite_coefficients),
      Rcpp::_["mean"] = m.mean,
      Rcpp::_["variance"] = m.variance,
      Rcpp::_["skewness"] = m.skewness,
      Rcpp::_["excess_kurtosis"] = m.excess_kurtosis);
}

magmaan::sim::PlsimMarginal plsim_marginal_from_list(Rcpp::List x) {
  magmaan::sim::PlsimMarginal m;
  m.gamma = Rcpp::as<Eigen::VectorXd>(x["gamma"]);
  m.a = Rcpp::as<Eigen::VectorXd>(x["a"]);
  m.b = Rcpp::as<Eigen::VectorXd>(x["b"]);
  m.hermite_coefficients =
      Rcpp::as<Eigen::VectorXd>(x["hermite_coefficients"]);
  m.mean = Rcpp::as<double>(x["mean"]);
  m.variance = Rcpp::as<double>(x["variance"]);
  m.skewness = Rcpp::as<double>(x["skewness"]);
  m.excess_kurtosis = Rcpp::as<double>(x["excess_kurtosis"]);
  return m;
}

Rcpp::List plsim_marginals_to_list(
    const std::vector<magmaan::sim::PlsimMarginal>& marginals) {
  Rcpp::List out(static_cast<R_xlen_t>(marginals.size()));
  for (std::size_t i = 0; i < marginals.size(); ++i) {
    out[static_cast<R_xlen_t>(i)] = plsim_marginal_to_list(marginals[i]);
  }
  return out;
}

std::vector<magmaan::sim::PlsimMarginal>
plsim_marginals_from_list(Rcpp::List xs) {
  std::vector<magmaan::sim::PlsimMarginal> out;
  out.reserve(static_cast<std::size_t>(xs.size()));
  for (R_xlen_t i = 0; i < xs.size(); ++i) {
    out.push_back(plsim_marginal_from_list(Rcpp::as<Rcpp::List>(xs[i])));
  }
  return out;
}

Rcpp::List plsim_options_to_list(const magmaan::sim::PlsimOptions& options,
                                 const std::string& method) {
  return Rcpp::List::create(
      Rcpp::_["method"] = method,
      Rcpp::_["num_segments"] = options.num_segments,
      Rcpp::_["monotone"] = options.monotone,
      Rcpp::_["max_iter"] = options.max_iter,
      Rcpp::_["quadrature_points"] = options.quadrature_points,
      Rcpp::_["hermite_order"] = options.hermite_order,
      Rcpp::_["marginal_tol"] = options.marginal_tol,
      Rcpp::_["correlation_tol"] = options.correlation_tol,
      Rcpp::_["rho_bound"] = options.rho_bound,
      Rcpp::_["cholesky_jitter"] = options.cholesky_jitter);
}

magmaan::sim::PlsimOptions plsim_options_from_list(Rcpp::List x) {
  magmaan::sim::PlsimOptions options = make_plsim_options(
      Rcpp::as<int>(x["num_segments"]),
      Rcpp::as<bool>(x["monotone"]),
      Rcpp::as<int>(x["max_iter"]),
      Rcpp::as<int>(x["quadrature_points"]),
      Rcpp::as<int>(x["hermite_order"]),
      Rcpp::as<double>(x["marginal_tol"]),
      Rcpp::as<double>(x["correlation_tol"]),
      Rcpp::as<double>(x["rho_bound"]));
  options.cholesky_jitter = Rcpp::as<double>(x["cholesky_jitter"]);
  return options;
}

Rcpp::List norta_options_to_list(const magmaan::sim::NortaOptions& options) {
  return Rcpp::List::create(
      Rcpp::_["quadrature_points"] = options.quadrature_points,
      Rcpp::_["max_bisection_iter"] = options.max_bisection_iter,
      Rcpp::_["rho_bound"] = options.rho_bound,
      Rcpp::_["calibration_tol"] = options.calibration_tol,
      Rcpp::_["cholesky_jitter"] = options.cholesky_jitter);
}

magmaan::sim::NortaOptions norta_options_from_list(Rcpp::List x) {
  return make_norta_options(
      list_int_or(x, "quadrature_points", 31),
      list_int_or(x, "max_bisection_iter", 80),
      list_double_or(x, "rho_bound", 0.999),
      list_double_or(x, "calibration_tol", 1e-8),
      list_double_or(x, "cholesky_jitter", 1e-10));
}

Rcpp::List bivariate_copula_options_to_list(
    const magmaan::sim::BivariateCopulaOptions& options) {
  return Rcpp::List::create(
      Rcpp::_["quadrature_points"] = options.quadrature_points,
      Rcpp::_["max_bisection_iter"] = options.max_bisection_iter,
      Rcpp::_["calibration_tol"] = options.calibration_tol,
      Rcpp::_["matrix_repair"] =
          bivariate_copula_repair_to_string(options.matrix_repair),
      Rcpp::_["matrix_repair_min_eigenvalue"] =
          options.matrix_repair_min_eigenvalue);
}

magmaan::sim::BivariateCopulaOptions bivariate_copula_options_from_list(
    Rcpp::List x) {
  const std::string matrix_repair = has_named(x, "matrix_repair")
      ? Rcpp::as<std::string>(x["matrix_repair"])
      : "none";
  return make_bivariate_copula_options(
      list_int_or(x, "quadrature_points", 31),
      list_int_or(x, "max_bisection_iter", 80),
      list_double_or(x, "calibration_tol", 1e-6),
      matrix_repair,
      list_double_or(x, "matrix_repair_min_eigenvalue", 1e-8));
}

Rcpp::List bivariate_copula_spec_to_list(
    const magmaan::sim::BivariateCopulaSpec& copula) {
  return Rcpp::List::create(
      Rcpp::_["family"] = bivariate_copula_family_to_string(copula.family),
      Rcpp::_["theta"] = copula.theta);
}

magmaan::sim::BivariateCopulaSpec bivariate_copula_spec_from_list(
    Rcpp::List x) {
  magmaan::sim::BivariateCopulaSpec copula;
  copula.family = bivariate_copula_family_from_string(
      Rcpp::as<std::string>(x["family"]));
  copula.theta = Rcpp::as<double>(x["theta"]);
  return copula;
}

Rcpp::List bivariate_copula_calibration_to_list(
    const magmaan::sim::BivariateCopulaCorrelationCalibration& cal) {
  return Rcpp::List::create(
      Rcpp::_["copula"] = bivariate_copula_spec_to_list(cal.copula),
      Rcpp::_["target_corr"] = cal.target_corr,
      Rcpp::_["achieved_corr"] = cal.achieved_corr,
      Rcpp::_["lower_bound_corr"] = cal.lower_bound_corr,
      Rcpp::_["upper_bound_corr"] = cal.upper_bound_corr,
      Rcpp::_["iterations"] = cal.iterations);
}

bool approx_equal(double a, double b, double tol = 1e-10) {
  return std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)});
}

bool pearson_type6_moments(const magmaan::sim::MarginalSpec& m,
                           double& mean,
                           double& sd) {
  if (m.kind != magmaan::sim::MarginalKind::Pearson ||
      m.pearson_type != 6 || !(m.pearson_p2 > 2.0)) {
    return false;
  }
  const double a = m.pearson_p1;
  const double b = m.pearson_p2;
  mean = m.pearson_p3 + m.pearson_p4 * a / (b - 1.0);
  const double var = m.pearson_p4 * m.pearson_p4 * a * (a + b - 1.0) /
      ((b - 2.0) * (b - 1.0) * (b - 1.0));
  if (!std::isfinite(mean) || !std::isfinite(var) || !(var > 0.0)) {
    return false;
  }
  sd = std::sqrt(var);
  return true;
}

bool can_use_r_pearson_type6_path(
    const std::vector<magmaan::sim::MarginalSpec>& marginals) {
  for (const auto& marginal : marginals) {
    double mean = 0.0;
    double sd = 1.0;
    if (!pearson_type6_moments(marginal, mean, sd)) return false;
    if (!approx_equal(mean, marginal.mean) || !approx_equal(sd, marginal.sd)) {
      return false;
    }
  }
  return true;
}

Eigen::MatrixXd simulate_ig_pearson_type6_r(
    Eigen::Index total_n,
    const Eigen::Ref<const Eigen::MatrixXd>& root,
    const std::vector<magmaan::sim::MarginalSpec>& marginals) {
  Rcpp::RNGScope scope;
  const Eigen::Index p = static_cast<Eigen::Index>(marginals.size());
  Eigen::MatrixXd G(total_n, p);
  for (Eigen::Index j = 0; j < p; ++j) {
    const auto& m = marginals[static_cast<std::size_t>(j)];
    const double a = m.pearson_p1;
    const double b = m.pearson_p2;
    const double fallback_beta = a / (a + b);
    const double location = m.pearson_p3;
    const double scale = m.pearson_p4;
    for (Eigen::Index row = 0; row < total_n; ++row) {
      const double x = R::rgamma(a, 1.0);
      const double y = R::rgamma(b, 1.0);
      const double beta = (x + y > 0.0) ? x / (x + y) : fallback_beta;
      const double one_minus_beta = std::max(
          1.0 - beta, std::numeric_limits<double>::min());
      G(row, j) = location + scale * beta / one_minus_beta;
    }
  }

  Eigen::MatrixXd out(total_n, root.rows());
  out.noalias() = G * root.transpose();
  return out;
}

Rcpp::List split_draw_matrix(const Eigen::MatrixXd& X, int n, int reps) {
  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    draws[i] = Rcpp::wrap(X.middleRows(
        static_cast<Eigen::Index>(i) * static_cast<Eigen::Index>(n),
        static_cast<Eigen::Index>(n)).eval());
  }
  return draws;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List sim_ig_batch_impl(
    Rcpp::NumericMatrix sigma,
    Rcpp::NumericVector target_skewness,
    Rcpp::NumericVector target_excess_kurtosis,
    int n,
    int reps,
    double seed_base,
    std::string root = "cholesky",
    std::string generator_family = "tukey_gh",
    int quadrature_points = 81,
    int max_iter = 80,
    int grid_points_g = 29,
    int grid_points_h = 25,
    double objective_tol = 1e-8,
    double parameter_tol = 1e-8,
    double finite_diff_step = 1e-4,
    double tukey_g_bound = 3.0,
    double tukey_h_upper = 0.249,
    double johnson_gamma_bound = 6.0,
    double johnson_log_delta_lower = -1.3862943611198906,
    double johnson_log_delta_upper = 2.0794415416798357,
    double root_eigen_tol = 1e-12,
    double moment_solve_tol = 1e-8) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const Eigen::Map<Eigen::MatrixXd> sigma_map(
      REAL(sigma), sigma.nrow(), sigma.ncol());
  const Eigen::Map<Eigen::VectorXd> skew(
      REAL(target_skewness), target_skewness.size());
  const Eigen::Map<Eigen::VectorXd> kurt(
      REAL(target_excess_kurtosis), target_excess_kurtosis.size());

  const magmaan::sim::IgOptions options = make_ig_options(
      root, generator_family, quadrature_points, max_iter, grid_points_g,
      grid_points_h, objective_tol, parameter_tol, finite_diff_step,
      tukey_g_bound, tukey_h_upper, johnson_gamma_bound,
      johnson_log_delta_lower, johnson_log_delta_upper, root_eigen_tol,
      moment_solve_tol);

  auto cal_or = magmaan::sim::calibrate_ig(sigma_map, skew, kurt, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  magmaan::sim::IndependentOptions independent_options;
  independent_options.quadrature_points = quadrature_points;

  const Eigen::Index total_n =
      static_cast<Eigen::Index>(n) * static_cast<Eigen::Index>(reps);
  Rcpp::List draws;
  if (can_use_r_pearson_type6_path(cal_or->generator_marginals)) {
    Rcpp::Function set_seed("set.seed");
    set_seed(static_cast<int>(seed_base) + 1);
    const Eigen::MatrixXd X = simulate_ig_pearson_type6_r(
        total_n, cal_or->root, cal_or->generator_marginals);
    draws = split_draw_matrix(X, n, reps);
  } else {
    draws = Rcpp::List(reps);
    for (int i = 0; i < reps; ++i) {
      std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                          static_cast<std::uint64_t>(i + 1));
      auto X_or = magmaan::sim::simulate_ig_matrix(
          static_cast<Eigen::Index>(n), cal_or->root,
          cal_or->generator_marginals, rng, independent_options);
      if (!X_or.has_value()) stop_sim(X_or.error());
      draws[i] = Rcpp::wrap(*X_or);
    }
  }

  return Rcpp::List::create(
      Rcpp::_["draws"] = draws,
      Rcpp::_["root"] = Rcpp::wrap(cal_or->root),
      Rcpp::_["generator_skewness"] = Rcpp::wrap(cal_or->generator_skewness),
          Rcpp::_["generator_excess_kurtosis"] =
          Rcpp::wrap(cal_or->generator_excess_kurtosis));
}

// [[Rcpp::export]]
Rcpp::List sim_ig_calibrate_impl(
    Rcpp::NumericMatrix sigma,
    Rcpp::NumericVector target_skewness,
    Rcpp::NumericVector target_excess_kurtosis,
    std::string root = "cholesky",
    std::string generator_family = "tukey_gh",
    int quadrature_points = 81,
    int max_iter = 80,
    int grid_points_g = 29,
    int grid_points_h = 25,
    double objective_tol = 1e-8,
    double parameter_tol = 1e-8,
    double finite_diff_step = 1e-4,
    double tukey_g_bound = 3.0,
    double tukey_h_upper = 0.249,
    double johnson_gamma_bound = 6.0,
    double johnson_log_delta_lower = -1.3862943611198906,
    double johnson_log_delta_upper = 2.0794415416798357,
    double root_eigen_tol = 1e-12,
    double moment_solve_tol = 1e-8) {
  const Eigen::Map<Eigen::MatrixXd> sigma_map(
      REAL(sigma), sigma.nrow(), sigma.ncol());
  const Eigen::Map<Eigen::VectorXd> skew(
      REAL(target_skewness), target_skewness.size());
  const Eigen::Map<Eigen::VectorXd> kurt(
      REAL(target_excess_kurtosis), target_excess_kurtosis.size());
  const magmaan::sim::IgOptions options = make_ig_options(
      root, generator_family, quadrature_points, max_iter, grid_points_g,
      grid_points_h, objective_tol, parameter_tol, finite_diff_step,
      tukey_g_bound, tukey_h_upper, johnson_gamma_bound,
      johnson_log_delta_lower, johnson_log_delta_upper, root_eigen_tol,
      moment_solve_tol);

  auto cal_or = magmaan::sim::calibrate_ig(sigma_map, skew, kurt, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["root"] = Rcpp::wrap(cal_or->root),
      Rcpp::_["generator_skewness"] = Rcpp::wrap(cal_or->generator_skewness),
      Rcpp::_["generator_excess_kurtosis"] =
          Rcpp::wrap(cal_or->generator_excess_kurtosis),
      Rcpp::_["generator_marginals"] =
          marginal_specs_to_list(cal_or->generator_marginals),
      Rcpp::_["quadrature_points"] = quadrature_points);
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_ig_calibration", "list");
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_ig_draw_impl(Rcpp::List calibration,
                            int n,
                            int reps,
                            double seed_base,
                            int quadrature_points = -1) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");
  Eigen::MatrixXd root = Rcpp::as<Eigen::MatrixXd>(calibration["root"]);
  std::vector<magmaan::sim::MarginalSpec> marginals =
      marginal_specs_from_list(Rcpp::as<Rcpp::List>(
          calibration["generator_marginals"]));
  magmaan::sim::IndependentOptions options;
  if (quadrature_points > 0) {
    options.quadrature_points = quadrature_points;
  } else if (calibration.containsElementNamed("quadrature_points")) {
    options.quadrature_points = Rcpp::as<int>(calibration["quadrature_points"]);
  }

  const Eigen::Index total_n =
      static_cast<Eigen::Index>(n) * static_cast<Eigen::Index>(reps);
  Rcpp::List draws;
  if (can_use_r_pearson_type6_path(marginals)) {
    Rcpp::Function set_seed("set.seed");
    set_seed(static_cast<int>(seed_base) + 1);
    const Eigen::MatrixXd X = simulate_ig_pearson_type6_r(
        total_n, root, marginals);
    draws = split_draw_matrix(X, n, reps);
  } else {
    draws = Rcpp::List(reps);
    for (int i = 0; i < reps; ++i) {
      std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                          static_cast<std::uint64_t>(i + 1));
      auto X_or = magmaan::sim::simulate_ig_matrix(
          static_cast<Eigen::Index>(n), root, marginals, rng, options);
      if (!X_or.has_value()) stop_sim(X_or.error());
      draws[i] = Rcpp::wrap(*X_or);
    }
  }
  return Rcpp::List::create(Rcpp::_["draws"] = draws);
}

// [[Rcpp::export]]
Rcpp::List sim_norta_batch_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::List marginals,
    int n,
    int reps,
    double seed_base,
    int quadrature_points = 31,
    int max_bisection_iter = 80,
    double rho_bound = 0.999,
    double calibration_tol = 1e-8,
    double cholesky_jitter = 1e-10) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const auto marginal_specs = marginal_specs_from_list(marginals);
  const magmaan::sim::NortaOptions options = make_norta_options(
      quadrature_points, max_bisection_iter, rho_bound, calibration_tol,
      cholesky_jitter);

  auto cal_or = magmaan::sim::calibrate_norta(corr, marginal_specs, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_norta_matrix(
        static_cast<Eigen::Index>(n), *cal_or, marginal_specs, rng, options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }

  return Rcpp::List::create(
      Rcpp::_["draws"] = draws,
      Rcpp::_["latent_corr"] = Rcpp::wrap(cal_or->latent_corr),
      Rcpp::_["marginal_mean"] = Rcpp::wrap(cal_or->marginal_mean),
      Rcpp::_["marginal_sd"] = Rcpp::wrap(cal_or->marginal_sd));
}

// [[Rcpp::export]]
Rcpp::List sim_norta_calibrate_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::List marginals,
    int quadrature_points = 31,
    int max_bisection_iter = 80,
    double rho_bound = 0.999,
    double calibration_tol = 1e-8,
    double cholesky_jitter = 1e-10) {
  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const auto marginal_specs = marginal_specs_from_list(marginals);
  const magmaan::sim::NortaOptions options = make_norta_options(
      quadrature_points, max_bisection_iter, rho_bound, calibration_tol,
      cholesky_jitter);

  auto cal_or = magmaan::sim::calibrate_norta(corr, marginal_specs, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["latent_corr"] = Rcpp::wrap(cal_or->latent_corr),
      Rcpp::_["marginal_mean"] = Rcpp::wrap(cal_or->marginal_mean),
      Rcpp::_["marginal_sd"] = Rcpp::wrap(cal_or->marginal_sd),
      Rcpp::_["marginals"] = marginal_specs_to_list(marginal_specs),
      Rcpp::_["options"] = norta_options_to_list(options));
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_norta_calibration", "list");
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_norta_draw_impl(Rcpp::List calibration,
                               int n,
                               int reps,
                               double seed_base,
                               int quadrature_points = -1,
                               double cholesky_jitter = -1.0) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  magmaan::sim::NortaCalibration cal;
  cal.latent_corr = Rcpp::as<Eigen::MatrixXd>(calibration["latent_corr"]);
  cal.marginal_mean = Rcpp::as<Eigen::VectorXd>(calibration["marginal_mean"]);
  cal.marginal_sd = Rcpp::as<Eigen::VectorXd>(calibration["marginal_sd"]);
  const auto marginal_specs = marginal_specs_from_list(
      Rcpp::as<Rcpp::List>(calibration["marginals"]));

  magmaan::sim::NortaOptions options;
  if (calibration.containsElementNamed("options")) {
    options = norta_options_from_list(Rcpp::as<Rcpp::List>(
        calibration["options"]));
  }
  if (quadrature_points > 0) options.quadrature_points = quadrature_points;
  if (cholesky_jitter >= 0.0) options.cholesky_jitter = cholesky_jitter;

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_norta_matrix(
        static_cast<Eigen::Index>(n), cal, marginal_specs, rng, options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }
  return Rcpp::List::create(Rcpp::_["draws"] = draws);
}

// [[Rcpp::export]]
Rcpp::List sim_bicop_batch_impl(double target_corr,
                                Rcpp::List marginals,
                                int n,
                                int reps,
                                double seed_base,
                                std::string family = "frank",
                                int quadrature_points = 31,
                                int max_bisection_iter = 80,
                                double calibration_tol = 1e-6) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const auto marginal_specs = marginal_specs_from_list(marginals);
  if (marginal_specs.size() != 2u) {
    Rcpp::stop("marginals must contain exactly two marginal specs");
  }
  const auto copula_family = bivariate_copula_family_from_string(family);
  const magmaan::sim::BivariateCopulaOptions options =
      make_bivariate_copula_options(
          quadrature_points, max_bisection_iter, calibration_tol);

  auto cal_or = magmaan::sim::calibrate_bivariate_copula_correlation(
      copula_family, target_corr, marginal_specs, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_bivariate_copula_matrix(
        static_cast<Eigen::Index>(n), cal_or->copula, marginal_specs, rng,
        options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }

  Rcpp::List out = bivariate_copula_calibration_to_list(*cal_or);
  out["draws"] = draws;
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_bicop_calibrate_impl(double target_corr,
                                    Rcpp::List marginals,
                                    std::string family = "frank",
                                    int quadrature_points = 31,
                                    int max_bisection_iter = 80,
                                    double calibration_tol = 1e-6) {
  const auto marginal_specs = marginal_specs_from_list(marginals);
  if (marginal_specs.size() != 2u) {
    Rcpp::stop("marginals must contain exactly two marginal specs");
  }
  const auto copula_family = bivariate_copula_family_from_string(family);
  const magmaan::sim::BivariateCopulaOptions options =
      make_bivariate_copula_options(
          quadrature_points, max_bisection_iter, calibration_tol);

  auto cal_or = magmaan::sim::calibrate_bivariate_copula_correlation(
      copula_family, target_corr, marginal_specs, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List out = bivariate_copula_calibration_to_list(*cal_or);
  out["marginals"] = marginal_specs_to_list(marginal_specs);
  out["options"] = bivariate_copula_options_to_list(options);
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_bicop_calibration", "list");
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_bicop_draw_impl(Rcpp::List calibration,
                               int n,
                               int reps,
                               double seed_base,
                               int quadrature_points = -1,
                               int max_bisection_iter = -1) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const auto copula = bivariate_copula_spec_from_list(
      Rcpp::as<Rcpp::List>(calibration["copula"]));
  const auto marginal_specs = marginal_specs_from_list(
      Rcpp::as<Rcpp::List>(calibration["marginals"]));
  if (marginal_specs.size() != 2u) {
    Rcpp::stop("calibration$marginals must contain exactly two marginal specs");
  }

  magmaan::sim::BivariateCopulaOptions options;
  if (calibration.containsElementNamed("options")) {
    options = bivariate_copula_options_from_list(
        Rcpp::as<Rcpp::List>(calibration["options"]));
  }
  if (quadrature_points > 0) options.quadrature_points = quadrature_points;
  if (max_bisection_iter > 0) {
    options.max_bisection_iter = max_bisection_iter;
  }

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_bivariate_copula_matrix(
        static_cast<Eigen::Index>(n), copula, marginal_specs, rng, options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }
  return Rcpp::List::create(Rcpp::_["draws"] = draws);
}

// [[Rcpp::export]]
Rcpp::List sim_plsim_batch_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::NumericVector target_skewness,
    Rcpp::NumericVector target_excess_kurtosis,
    int n,
    int reps,
    double seed_base,
    std::string method = "hermite",
    int num_segments = 12,
    bool monotone = false,
    int max_iter = 80,
    int quadrature_points = 31,
    int hermite_order = 24,
    double marginal_tol = 1e-8,
    double correlation_tol = 1e-8,
    double rho_bound = 0.999) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const Eigen::Map<Eigen::VectorXd> skew(
      REAL(target_skewness), target_skewness.size());
  const Eigen::Map<Eigen::VectorXd> kurt(
      REAL(target_excess_kurtosis), target_excess_kurtosis.size());

  magmaan::sim::PlsimOptions options = make_plsim_options(
      num_segments, monotone, max_iter, quadrature_points, hermite_order,
      marginal_tol, correlation_tol, rho_bound);

  const auto cov_method = plsim_method_from_string(method);
  auto cal_or = magmaan::sim::calibrate_plsim(
      corr, skew, kurt, cov_method, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_plsim_matrix(
        static_cast<Eigen::Index>(n), *cal_or, rng, options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }

  return Rcpp::List::create(
      Rcpp::_["draws"] = draws,
      Rcpp::_["intermediate_corr"] = Rcpp::wrap(cal_or->intermediate_corr),
      Rcpp::_["achieved_corr"] = Rcpp::wrap(cal_or->achieved_corr),
      Rcpp::_["iterations"] = Rcpp::wrap(cal_or->iterations));
}

// [[Rcpp::export]]
Rcpp::List sim_plsim_calibrate_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::NumericVector target_skewness,
    Rcpp::NumericVector target_excess_kurtosis,
    std::string method = "hermite",
    int num_segments = 12,
    bool monotone = false,
    int max_iter = 80,
    int quadrature_points = 31,
    int hermite_order = 24,
    double marginal_tol = 1e-8,
    double correlation_tol = 1e-8,
    double rho_bound = 0.999) {
  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const Eigen::Map<Eigen::VectorXd> skew(
      REAL(target_skewness), target_skewness.size());
  const Eigen::Map<Eigen::VectorXd> kurt(
      REAL(target_excess_kurtosis), target_excess_kurtosis.size());
  magmaan::sim::PlsimOptions options = make_plsim_options(
      num_segments, monotone, max_iter, quadrature_points, hermite_order,
      marginal_tol, correlation_tol, rho_bound);

  const auto cov_method = plsim_method_from_string(method);
  auto cal_or = magmaan::sim::calibrate_plsim(
      corr, skew, kurt, cov_method, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["marginals"] = plsim_marginals_to_list(cal_or->marginals),
      Rcpp::_["intermediate_corr"] = Rcpp::wrap(cal_or->intermediate_corr),
      Rcpp::_["achieved_corr"] = Rcpp::wrap(cal_or->achieved_corr),
      Rcpp::_["iterations"] = Rcpp::wrap(cal_or->iterations),
      Rcpp::_["options"] = plsim_options_to_list(options, method));
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_plsim_calibration", "list");
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_plsim_draw_impl(Rcpp::List calibration,
                               int n,
                               int reps,
                               double seed_base) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");
  magmaan::sim::PlsimCalibration cal;
  cal.marginals = plsim_marginals_from_list(
      Rcpp::as<Rcpp::List>(calibration["marginals"]));
  cal.intermediate_corr =
      Rcpp::as<Eigen::MatrixXd>(calibration["intermediate_corr"]);
  cal.achieved_corr = Rcpp::as<Eigen::MatrixXd>(calibration["achieved_corr"]);
  cal.iterations = Rcpp::as<Eigen::MatrixXi>(calibration["iterations"]);
  magmaan::sim::PlsimOptions options = plsim_options_from_list(
      Rcpp::as<Rcpp::List>(calibration["options"]));

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_plsim_matrix(
        static_cast<Eigen::Index>(n), cal, rng, options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }
  return Rcpp::List::create(Rcpp::_["draws"] = draws);
}
