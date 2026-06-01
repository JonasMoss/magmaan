#include <RcppEigen.h>

#include <cstdint>
#include <random>
#include <string>

#include "magmaan/error.hpp"
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

}  // namespace

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

  magmaan::sim::PlsimOptions options;
  options.num_segments = num_segments;
  options.monotone = monotone;
  options.max_iter = max_iter;
  options.quadrature_points = quadrature_points;
  options.hermite_order = hermite_order;
  options.marginal_tol = marginal_tol;
  options.correlation_tol = correlation_tol;
  options.rho_bound = rho_bound;

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
