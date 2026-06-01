#include <RcppEigen.h>

#include <cstdint>
#include <random>
#include <string>

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
  m.kind = marginal_kind_from_string(Rcpp::as<std::string>(x["kind"]));
  m.mean = Rcpp::as<double>(x["mean"]);
  m.sd = Rcpp::as<double>(x["sd"]);
  m.sigma_log = Rcpp::as<double>(x["sigma_log"]);
  m.g = Rcpp::as<double>(x["g"]);
  m.h = Rcpp::as<double>(x["h"]);
  m.pearson_type = Rcpp::as<int>(x["pearson_type"]);
  m.pearson_p1 = Rcpp::as<double>(x["pearson_p1"]);
  m.pearson_p2 = Rcpp::as<double>(x["pearson_p2"]);
  m.pearson_p3 = Rcpp::as<double>(x["pearson_p3"]);
  m.pearson_p4 = Rcpp::as<double>(x["pearson_p4"]);
  m.johnson_type = Rcpp::as<int>(x["johnson_type"]);
  m.johnson_gamma = Rcpp::as<double>(x["johnson_gamma"]);
  m.johnson_delta = Rcpp::as<double>(x["johnson_delta"]);
  m.fleishman_b = Rcpp::as<double>(x["fleishman_b"]);
  m.fleishman_c = Rcpp::as<double>(x["fleishman_c"]);
  m.fleishman_d = Rcpp::as<double>(x["fleishman_d"]);
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

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_ig_matrix(
        static_cast<Eigen::Index>(n), cal_or->root,
        cal_or->generator_marginals, rng, independent_options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
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

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_ig_matrix(
        static_cast<Eigen::Index>(n), root, marginals, rng, options);
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
