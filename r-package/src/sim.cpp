#include <RcppEigen.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "internal.hpp"
#include "magmaan/error.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/sim/model_implied.hpp"
#include "magmaan/sim/norta.hpp"
#include "magmaan/sim/ordinal_correlation.hpp"
#include "magmaan/sim/plsim.hpp"
#include "magmaan/sim/population.hpp"
#include "magmaan/sim/vale_maurelli.hpp"

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

magmaan::sim::GeneratorKind generator_kind_from_string(
    const std::string& generator) {
  using G = magmaan::sim::GeneratorKind;
  if (generator == "normal") return G::Normal;
  if (generator == "student_t" || generator == "t") return G::StudentT;
  if (generator == "scale_mixture") return G::ScaleMixture;
  if (generator == "contaminated_normal" || generator == "contaminated") {
    return G::ContaminatedNormal;
  }
  if (generator == "slash") return G::Slash;
  Rcpp::stop("unknown model-implied generator: %s", generator);
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

std::vector<double> nullable_numeric_to_vector(
    Rcpp::Nullable<Rcpp::NumericVector> x,
    const char* name) {
  if (x.isNull()) return {};
  Rcpp::NumericVector values(x);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(values.size()));
  for (R_xlen_t i = 0; i < values.size(); ++i) {
    const double value = values[i];
    if (!std::isfinite(value)) {
      Rcpp::stop("%s must contain finite values", name);
    }
    out.push_back(value);
  }
  return out;
}

Rcpp::IntegerVector observed_kinds_to_int(
    const std::vector<magmaan::sim::ObservedKind>& kinds) {
  Rcpp::IntegerVector out(static_cast<R_xlen_t>(kinds.size()));
  for (std::size_t i = 0; i < kinds.size(); ++i) {
    out[static_cast<R_xlen_t>(i)] =
        kinds[i] == magmaan::sim::ObservedKind::Ordinal ? 1 : 0;
  }
  return out;
}

std::vector<magmaan::sim::ObservedKind> observed_kinds_from_int(
    Rcpp::IntegerVector kinds) {
  std::vector<magmaan::sim::ObservedKind> out;
  out.reserve(static_cast<std::size_t>(kinds.size()));
  for (R_xlen_t i = 0; i < kinds.size(); ++i) {
    if (kinds[i] == 0) {
      out.push_back(magmaan::sim::ObservedKind::Continuous);
    } else if (kinds[i] == 1) {
      out.push_back(magmaan::sim::ObservedKind::Ordinal);
    } else {
      Rcpp::stop("model calibration kinds must be coded 0=continuous or 1=ordinal");
    }
  }
  return out;
}

Rcpp::IntegerVector int32_vector_to_r(const std::vector<std::int32_t>& x) {
  Rcpp::IntegerVector out(static_cast<R_xlen_t>(x.size()));
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[static_cast<R_xlen_t>(i)] = x[i];
  }
  return out;
}

Rcpp::List thresholds_to_list(
    const std::vector<Eigen::VectorXd>& thresholds) {
  Rcpp::List out(static_cast<R_xlen_t>(thresholds.size()));
  for (std::size_t j = 0; j < thresholds.size(); ++j) {
    out[static_cast<R_xlen_t>(j)] =
        thresholds[j].size() == 0
            ? R_NilValue
            : static_cast<SEXP>(Rcpp::wrap(thresholds[j]));
  }
  return out;
}

std::vector<Eigen::VectorXd> thresholds_from_list(Rcpp::List thresholds,
                                                  std::size_t p) {
  if (static_cast<std::size_t>(thresholds.size()) != p) {
    Rcpp::stop("model calibration threshold list length must match ov_names");
  }
  std::vector<Eigen::VectorXd> out(p);
  for (std::size_t j = 0; j < p; ++j) {
    Rcpp::RObject value(thresholds[static_cast<R_xlen_t>(j)]);
    if (value.isNULL()) {
      out[j] = Eigen::VectorXd{};
    } else {
      out[j] = Rcpp::as<Eigen::VectorXd>(value);
    }
  }
  return out;
}

magmaan::sim::ObservedCorrelationMetric observed_correlation_metric_from_string(
    const std::string& metric) {
  using M = magmaan::sim::ObservedCorrelationMetric;
  if (metric == "polychoric") return M::Polychoric;
  if (metric == "pearson_codes" || metric == "pearson") return M::PearsonCodes;
  if (metric == "polyserial") return M::Polyserial;
  Rcpp::stop("metric must be one of \"polychoric\", \"pearson_codes\", \"polyserial\"");
}

const char* observed_correlation_metric_to_string(
    magmaan::sim::ObservedCorrelationMetric metric) {
  using M = magmaan::sim::ObservedCorrelationMetric;
  switch (metric) {
    case M::Polychoric: return "polychoric";
    case M::PearsonCodes: return "pearson_codes";
    case M::Polyserial: return "polyserial";
  }
  return "polychoric";
}

magmaan::sim::OrdinalCorrelationOptions make_ordcorr_options(
    const std::string& metric,
    int max_bisection_iter,
    double calibration_tol,
    double rho_bound,
    const std::string& matrix_repair,
    double matrix_repair_min_eigenvalue) {
  magmaan::sim::OrdinalCorrelationOptions options;
  options.metric = observed_correlation_metric_from_string(metric);
  options.max_bisection_iter = max_bisection_iter;
  options.calibration_tol = calibration_tol;
  options.rho_bound = rho_bound;
  options.matrix_repair = bivariate_copula_repair_from_string(matrix_repair);
  options.matrix_repair_min_eigenvalue = matrix_repair_min_eigenvalue;
  return options;
}

// Per-variable marginals: an ordinal variable is a numeric vector of category
// proportions; a continuous variable is NULL.
std::vector<magmaan::sim::OrdinalMarginalSpec> ordinal_marginal_specs_from_list(
    Rcpp::List marginals) {
  std::vector<magmaan::sim::OrdinalMarginalSpec> out;
  out.reserve(static_cast<std::size_t>(marginals.size()));
  for (R_xlen_t i = 0; i < marginals.size(); ++i) {
    magmaan::sim::OrdinalMarginalSpec spec;
    Rcpp::RObject value(marginals[i]);
    if (value.isNULL()) {
      spec.kind = magmaan::sim::ObservedKind::Continuous;
    } else {
      spec.kind = magmaan::sim::ObservedKind::Ordinal;
      spec.proportions = Rcpp::as<Eigen::VectorXd>(value);
    }
    out.push_back(std::move(spec));
  }
  return out;
}

std::vector<Eigen::MatrixXd> ordinal_corr_matrices_from_list(
    Rcpp::List target_corrs) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(static_cast<std::size_t>(target_corrs.size()));
  for (R_xlen_t g = 0; g < target_corrs.size(); ++g) {
    Rcpp::NumericMatrix src = Rcpp::as<Rcpp::NumericMatrix>(target_corrs[g]);
    const Eigen::Map<Eigen::MatrixXd> corr(
        REAL(src), src.nrow(), src.ncol());
    out.emplace_back(corr);
  }
  return out;
}

std::vector<std::vector<magmaan::sim::OrdinalMarginalSpec>>
ordinal_marginal_groups_from_list(Rcpp::List marginals) {
  std::vector<std::vector<magmaan::sim::OrdinalMarginalSpec>> out;
  out.reserve(static_cast<std::size_t>(marginals.size()));
  for (R_xlen_t g = 0; g < marginals.size(); ++g) {
    out.push_back(ordinal_marginal_specs_from_list(
        Rcpp::as<Rcpp::List>(marginals[g])));
  }
  return out;
}

std::vector<std::vector<Eigen::VectorXd>>
ordinal_threshold_groups_from_list(Rcpp::List thresholds, std::size_t p) {
  std::vector<std::vector<Eigen::VectorXd>> out;
  out.reserve(static_cast<std::size_t>(thresholds.size()));
  for (R_xlen_t g = 0; g < thresholds.size(); ++g) {
    out.push_back(thresholds_from_list(
        Rcpp::as<Rcpp::List>(thresholds[g]), p));
  }
  return out;
}

Rcpp::DataFrame ordinal_pairs_to_df(
    const std::vector<magmaan::sim::OrdinalPairCalibration>& pairs) {
  const R_xlen_t m = static_cast<R_xlen_t>(pairs.size());
  Rcpp::IntegerVector i_idx(m);
  Rcpp::IntegerVector j_idx(m);
  Rcpp::IntegerVector iters(m);
  Rcpp::NumericVector target(m);
  Rcpp::NumericVector latent(m);
  Rcpp::NumericVector achieved(m);
  Rcpp::LogicalVector converged(m);
  Rcpp::LogicalVector infeasible(m);
  for (R_xlen_t k = 0; k < m; ++k) {
    const auto& p = pairs[static_cast<std::size_t>(k)];
    i_idx[k] = p.i + 1;  // 1-based for R
    j_idx[k] = p.j + 1;
    iters[k] = p.iterations;
    target[k] = p.target_corr;
    latent[k] = p.latent_rho;
    achieved[k] = p.achieved_corr;
    converged[k] = p.converged;
    infeasible[k] = p.infeasible;
  }
  return Rcpp::DataFrame::create(
      Rcpp::_["i"] = i_idx, Rcpp::_["j"] = j_idx,
      Rcpp::_["target"] = target, Rcpp::_["latent_rho"] = latent,
      Rcpp::_["achieved"] = achieved, Rcpp::_["iterations"] = iters,
      Rcpp::_["converged"] = converged, Rcpp::_["infeasible"] = infeasible,
      Rcpp::_["stringsAsFactors"] = false);
}

Rcpp::List ordcorr_calibration_to_list(
    const magmaan::sim::OrdinalCorrelationCalibration& cal) {
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["latent_corr"] = Rcpp::wrap(cal.latent_corr),
      Rcpp::_["target_corr"] = Rcpp::wrap(cal.target_corr),
      Rcpp::_["achieved_corr"] = Rcpp::wrap(cal.achieved_corr),
      Rcpp::_["kinds"] = observed_kinds_to_int(cal.kinds),
      Rcpp::_["thresholds"] = thresholds_to_list(cal.thresholds),
      Rcpp::_["pairs"] = ordinal_pairs_to_df(cal.pairs),
      Rcpp::_["metric"] =
          std::string(observed_correlation_metric_to_string(cal.metric)),
      Rcpp::_["max_abs_error"] = cal.max_abs_error,
      Rcpp::_["repair"] = Rcpp::List::create(
          Rcpp::_["applied"] = cal.repair_applied,
          Rcpp::_["raw_min_eigenvalue"] = cal.raw_min_eigenvalue,
          Rcpp::_["repaired_min_eigenvalue"] = cal.repaired_min_eigenvalue,
          Rcpp::_["ridge"] = cal.repair_ridge,
          Rcpp::_["shrinkage"] = cal.repair_shrinkage));
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_ordcorr_calibration", "list");
  return out;
}

magmaan::sim::OrdinalCorrelationCalibration ordcorr_calibration_from_list(
    Rcpp::List calibration) {
  magmaan::sim::OrdinalCorrelationCalibration cal;
  cal.latent_corr = Rcpp::as<Eigen::MatrixXd>(calibration["latent_corr"]);
  cal.kinds = observed_kinds_from_int(
      Rcpp::as<Rcpp::IntegerVector>(calibration["kinds"]));
  cal.thresholds = thresholds_from_list(
      Rcpp::as<Rcpp::List>(calibration["thresholds"]), cal.kinds.size());
  if (calibration.containsElementNamed("metric")) {
    cal.metric = observed_correlation_metric_from_string(
        Rcpp::as<std::string>(calibration["metric"]));
  }
  return cal;
}

Rcpp::List ordcorr_draw_to_list(
    const magmaan::sim::MixedPopulationDraw& draw) {
  return Rcpp::List::create(
      Rcpp::_["X"] = Rcpp::wrap(draw.observed.X),
      Rcpp::_["ordered"] = int32_vector_to_r(draw.observed.ordered),
      Rcpp::_["n_levels"] = int32_vector_to_r(draw.observed.n_levels),
      Rcpp::_["category_proportions"] =
          thresholds_to_list(draw.observed.category_proportions));
}

std::vector<std::string> group_labels_from_nullable(
    Rcpp::Nullable<Rcpp::CharacterVector> group_labels) {
  if (group_labels.isNull()) return {};
  return Rcpp::as<std::vector<std::string>>(
      Rcpp::CharacterVector(group_labels.get()));
}

std::vector<std::string> variable_names_from_marginals(Rcpp::List marginals) {
  Rcpp::CharacterVector names = marginals.names();
  if (names.size() != marginals.size()) return {};
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(names.size()));
  for (R_xlen_t i = 0; i < names.size(); ++i) {
    if (Rcpp::CharacterVector::is_na(names[i])) return {};
    const std::string name = Rcpp::as<std::string>(names[i]);
    if (name.empty()) return {};
    out.push_back(name);
  }
  return out;
}

Rcpp::List model_implied_population_to_list(
    const magmaan::sim::ModelImpliedPopulation& population) {
  Rcpp::List groups(static_cast<R_xlen_t>(population.groups.size()));
  for (std::size_t b = 0; b < population.groups.size(); ++b) {
    const auto& group = population.groups[b];
    groups[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
        Rcpp::_["mean"] = Rcpp::wrap(group.latent.mean),
        Rcpp::_["cov"] = Rcpp::wrap(group.latent.covariance),
        Rcpp::_["thresholds"] = thresholds_to_list(group.observed.thresholds));
  }
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["n_groups"] =
          static_cast<int>(population.groups.size()),
      Rcpp::_["ov_names"] = Rcpp::wrap(population.ov_names),
      Rcpp::_["kinds"] = observed_kinds_to_int(population.kinds),
      Rcpp::_["n_levels"] = int32_vector_to_r(population.n_levels),
      Rcpp::_["groups"] = groups);
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_model_calibration", "list");
  return out;
}

magmaan::sim::ModelImpliedPopulation model_implied_population_from_list(
    Rcpp::List calibration) {
  magmaan::sim::ModelImpliedPopulation population;
  population.ov_names =
      Rcpp::as<std::vector<std::string>>(calibration["ov_names"]);
  const std::size_t p = population.ov_names.size();
  population.kinds = observed_kinds_from_int(
      Rcpp::as<Rcpp::IntegerVector>(calibration["kinds"]));
  population.n_levels =
      Rcpp::as<std::vector<std::int32_t>>(calibration["n_levels"]);
  if (population.kinds.size() != p || population.n_levels.size() != p) {
    Rcpp::stop("model calibration kind/n_levels lengths must match ov_names");
  }

  Rcpp::List groups = Rcpp::as<Rcpp::List>(calibration["groups"]);
  population.groups.reserve(static_cast<std::size_t>(groups.size()));
  for (R_xlen_t b = 0; b < groups.size(); ++b) {
    Rcpp::List src = Rcpp::as<Rcpp::List>(groups[b]);
    magmaan::sim::MixedPopulation group;
    group.latent.mean = Rcpp::as<Eigen::VectorXd>(src["mean"]);
    group.latent.covariance = Rcpp::as<Eigen::MatrixXd>(src["cov"]);
    group.observed.kinds = population.kinds;
    group.observed.thresholds = thresholds_from_list(
        Rcpp::as<Rcpp::List>(src["thresholds"]), p);
    population.groups.push_back(std::move(group));
  }
  return population;
}

magmaan::sim::GeneratorSpec make_model_generator_spec(
    const std::string& generator,
    double df,
    Rcpp::Nullable<Rcpp::NumericVector> mixture_weights,
    Rcpp::Nullable<Rcpp::NumericVector> mixture_scale_multipliers,
    double contamination_probability,
    double contamination_scale_multiplier,
    double slash_q,
    double cholesky_jitter) {
  magmaan::sim::GeneratorSpec spec;
  spec.kind = generator_kind_from_string(generator);
  spec.df = df;
  spec.scale_mixture.weights =
      nullable_numeric_to_vector(mixture_weights, "mixture_weights");
  spec.scale_mixture.scale_multipliers =
      nullable_numeric_to_vector(mixture_scale_multipliers,
                                 "mixture_scale_multipliers");
  spec.contamination.contamination_probability =
      contamination_probability;
  spec.contamination.scale_multiplier = contamination_scale_multiplier;
  spec.slash.q = slash_q;
  spec.normal_options.cholesky_jitter = cholesky_jitter;
  spec.student_t_options.cholesky_jitter = cholesky_jitter;
  return spec;
}

magmaan::sim::ModelImpliedPopulation calibrate_model_from_arg(
    SEXP fit_or_partable,
    Rcpp::Nullable<Rcpp::NumericVector> theta) {
  const bool is_data_frame = Rf_inherits(fit_or_partable, "data.frame");
  if (!is_data_frame) {
    Rcpp::List fit(fit_or_partable);
    if (fit.containsElementNamed("partable") &&
        fit.containsElementNamed("theta")) {
      magmaanr::Ctx ctx = magmaanr::ctx_from_fit(fit);
      const magmaan::estimate::Estimates est = magmaanr::est_from_fit(fit);
      auto pop_or = magmaan::sim::lower_model_implied(
          ctx.pt, ctx.rep, est.theta);
      if (!pop_or.has_value()) stop_sim(pop_or.error());
      return std::move(*pop_or);
    }
  }

  auto parsed =
      magmaanr::parse_partable_df(Rcpp::DataFrame(fit_or_partable));
  auto rep_or = magmaan::model::build_matrix_rep(
      parsed.structure, &parsed.names);
  if (!rep_or.has_value()) magmaanr::stop_model(rep_or.error());
  Eigen::VectorXd theta_vec;
  if (theta.isNull()) {
    theta_vec = Eigen::VectorXd{};
  } else {
    theta_vec = Rcpp::as<Eigen::VectorXd>(Rcpp::NumericVector(theta));
  }
  auto pop_or = magmaan::sim::lower_model_implied(
      parsed.structure, *rep_or, theta_vec);
  if (!pop_or.has_value()) stop_sim(pop_or.error());
  return std::move(*pop_or);
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

Rcpp::List cvine_copula_spec_to_list(
    const magmaan::sim::CVineCopulaSpec& copula) {
  Rcpp::List pair_copulas(
      static_cast<R_xlen_t>(copula.pair_copulas.size()));
  for (std::size_t j = 0; j < copula.pair_copulas.size(); ++j) {
    Rcpp::List row(static_cast<R_xlen_t>(copula.pair_copulas[j].size()));
    for (std::size_t k = 0; k < copula.pair_copulas[j].size(); ++k) {
      row[static_cast<R_xlen_t>(k)] =
          bivariate_copula_spec_to_list(copula.pair_copulas[j][k]);
    }
    pair_copulas[static_cast<R_xlen_t>(j)] = row;
  }
  return Rcpp::List::create(Rcpp::_["pair_copulas"] = pair_copulas);
}

magmaan::sim::CVineCopulaSpec cvine_copula_spec_from_list(Rcpp::List x) {
  Rcpp::List pair_copulas = has_named(x, "pair_copulas")
      ? Rcpp::as<Rcpp::List>(x["pair_copulas"])
      : x;
  magmaan::sim::CVineCopulaSpec copula;
  copula.pair_copulas.resize(static_cast<std::size_t>(pair_copulas.size()));
  for (R_xlen_t j = 0; j < pair_copulas.size(); ++j) {
    Rcpp::List row = Rcpp::as<Rcpp::List>(pair_copulas[j]);
    if (row.size() != j) {
      Rcpp::stop(
          "copula$pair_copulas must be triangular with row j containing j entries");
    }
    auto& out_row = copula.pair_copulas[static_cast<std::size_t>(j)];
    out_row.reserve(static_cast<std::size_t>(row.size()));
    for (R_xlen_t k = 0; k < row.size(); ++k) {
      out_row.push_back(
          bivariate_copula_spec_from_list(Rcpp::as<Rcpp::List>(row[k])));
    }
  }
  return copula;
}

Rcpp::List cvine_calibration_to_list(
    const magmaan::sim::CVineCorrelationCalibration& cal) {
  return Rcpp::List::create(
      Rcpp::_["copula"] = cvine_copula_spec_to_list(cal.copula),
      Rcpp::_["family"] = bivariate_copula_family_to_string(cal.family),
      Rcpp::_["target_corr"] = Rcpp::wrap(cal.target_corr),
      Rcpp::_["achieved_corr"] = Rcpp::wrap(cal.achieved_corr),
      Rcpp::_["lower_bound_corr"] = Rcpp::wrap(cal.lower_bound_corr),
      Rcpp::_["upper_bound_corr"] = Rcpp::wrap(cal.upper_bound_corr),
      Rcpp::_["iterations"] = Rcpp::wrap(cal.iterations),
      Rcpp::_["max_abs_error"] = cal.max_abs_error);
}

Rcpp::List cvine3_copula_spec_to_list(
    const magmaan::sim::CVine3CopulaSpec& copula) {
  return Rcpp::List::create(
      Rcpp::_["copula_01"] =
          bivariate_copula_spec_to_list(copula.copula_01),
      Rcpp::_["copula_02"] =
          bivariate_copula_spec_to_list(copula.copula_02),
      Rcpp::_["copula_12_given_0"] =
          bivariate_copula_spec_to_list(copula.copula_12_given_0));
}

magmaan::sim::CVine3CopulaSpec cvine3_copula_spec_from_list(Rcpp::List x) {
  magmaan::sim::CVine3CopulaSpec copula;
  copula.copula_01 = bivariate_copula_spec_from_list(
      Rcpp::as<Rcpp::List>(x["copula_01"]));
  copula.copula_02 = bivariate_copula_spec_from_list(
      Rcpp::as<Rcpp::List>(x["copula_02"]));
  copula.copula_12_given_0 = bivariate_copula_spec_from_list(
      Rcpp::as<Rcpp::List>(x["copula_12_given_0"]));
  return copula;
}

magmaan::sim::CVine3FamilySpec cvine3_family_spec_from_nullable(
    Rcpp::Nullable<Rcpp::CharacterVector> families,
    magmaan::sim::BivariateCopulaFamily fallback) {
  magmaan::sim::CVine3FamilySpec out{fallback, fallback, fallback};
  if (families.isNull()) return out;

  Rcpp::CharacterVector values(families);
  if (values.size() == 1) {
    const auto family =
        bivariate_copula_family_from_string(Rcpp::as<std::string>(values[0]));
    return magmaan::sim::CVine3FamilySpec{family, family, family};
  }
  if (values.size() != 3) {
    Rcpp::stop("families must be NULL, length 1, or length 3");
  }
  out.family_01 =
      bivariate_copula_family_from_string(Rcpp::as<std::string>(values[0]));
  out.family_02 =
      bivariate_copula_family_from_string(Rcpp::as<std::string>(values[1]));
  out.family_12_given_0 =
      bivariate_copula_family_from_string(Rcpp::as<std::string>(values[2]));
  return out;
}

std::vector<magmaan::sim::BivariateCopulaFamily>
family_set_from_nullable(Rcpp::Nullable<Rcpp::CharacterVector> family_set,
                         magmaan::sim::BivariateCopulaFamily fallback) {
  std::vector<magmaan::sim::BivariateCopulaFamily> out;
  if (family_set.isNull()) {
    out.push_back(fallback);
    return out;
  }
  Rcpp::CharacterVector values(family_set);
  if (values.size() == 0) {
    Rcpp::stop("family_set must not be empty");
  }
  out.reserve(static_cast<std::size_t>(values.size()));
  for (R_xlen_t i = 0; i < values.size(); ++i) {
    out.push_back(
        bivariate_copula_family_from_string(Rcpp::as<std::string>(values[i])));
  }
  return out;
}

Eigen::Vector3i vector3i_from_list_field(Rcpp::List x, const char* name) {
  Rcpp::IntegerVector values = Rcpp::as<Rcpp::IntegerVector>(x[name]);
  if (values.size() != 3) {
    Rcpp::stop("%s must be an integer vector of length 3", name);
  }
  Eigen::Vector3i out;
  for (R_xlen_t i = 0; i < 3; ++i) out(i) = values[i];
  return out;
}

Rcpp::IntegerVector vector3i_to_integer_vector(const Eigen::Vector3i& x) {
  return Rcpp::IntegerVector::create(x(0), x(1), x(2));
}

Rcpp::List cvine3_calibration_to_list(
    const magmaan::sim::CVine3CorrelationCalibration& cal) {
  return Rcpp::List::create(
      Rcpp::_["copula"] = cvine3_copula_spec_to_list(cal.copula),
      Rcpp::_["target_corr"] = Rcpp::wrap(cal.target_corr),
      Rcpp::_["achieved_corr"] = Rcpp::wrap(cal.achieved_corr),
      Rcpp::_["variable_order"] = vector3i_to_integer_vector(cal.variable_order),
      Rcpp::_["root_index"] = cal.root_index,
      Rcpp::_["root_01"] = bivariate_copula_calibration_to_list(cal.root_01),
      Rcpp::_["root_02"] = bivariate_copula_calibration_to_list(cal.root_02),
      Rcpp::_["conditional_lower_bound_corr_12"] =
          cal.conditional_lower_bound_corr_12,
      Rcpp::_["conditional_upper_bound_corr_12"] =
          cal.conditional_upper_bound_corr_12,
      Rcpp::_["conditional_iterations"] = cal.conditional_iterations,
      Rcpp::_["max_abs_error"] = cal.max_abs_error);
}

magmaan::sim::CVine3CorrelationCalibration cvine3_calibration_from_list(
    Rcpp::List x) {
  magmaan::sim::CVine3CorrelationCalibration cal;
  cal.copula = cvine3_copula_spec_from_list(Rcpp::as<Rcpp::List>(x["copula"]));
  if (has_named(x, "target_corr")) {
    cal.target_corr = Rcpp::as<Eigen::MatrixXd>(x["target_corr"]);
  }
  if (has_named(x, "achieved_corr")) {
    cal.achieved_corr = Rcpp::as<Eigen::MatrixXd>(x["achieved_corr"]);
  }
  if (has_named(x, "variable_order")) {
    cal.variable_order = vector3i_from_list_field(x, "variable_order");
  }
  if (has_named(x, "root_index")) {
    cal.root_index = Rcpp::as<int>(x["root_index"]);
  }
  if (has_named(x, "root_01")) {
    Rcpp::List root = Rcpp::as<Rcpp::List>(x["root_01"]);
    cal.root_01.copula = bivariate_copula_spec_from_list(
        Rcpp::as<Rcpp::List>(root["copula"]));
    cal.root_01.target_corr = list_double_or(root, "target_corr", 0.0);
    cal.root_01.achieved_corr = list_double_or(root, "achieved_corr", 0.0);
    cal.root_01.lower_bound_corr =
        list_double_or(root, "lower_bound_corr", 0.0);
    cal.root_01.upper_bound_corr =
        list_double_or(root, "upper_bound_corr", 0.0);
    cal.root_01.iterations = list_int_or(root, "iterations", 0);
  }
  if (has_named(x, "root_02")) {
    Rcpp::List root = Rcpp::as<Rcpp::List>(x["root_02"]);
    cal.root_02.copula = bivariate_copula_spec_from_list(
        Rcpp::as<Rcpp::List>(root["copula"]));
    cal.root_02.target_corr = list_double_or(root, "target_corr", 0.0);
    cal.root_02.achieved_corr = list_double_or(root, "achieved_corr", 0.0);
    cal.root_02.lower_bound_corr =
        list_double_or(root, "lower_bound_corr", 0.0);
    cal.root_02.upper_bound_corr =
        list_double_or(root, "upper_bound_corr", 0.0);
    cal.root_02.iterations = list_int_or(root, "iterations", 0);
  }
  cal.conditional_lower_bound_corr_12 =
      list_double_or(x, "conditional_lower_bound_corr_12", 0.0);
  cal.conditional_upper_bound_corr_12 =
      list_double_or(x, "conditional_upper_bound_corr_12", 0.0);
  cal.conditional_iterations = list_int_or(x, "conditional_iterations", 0);
  cal.max_abs_error = list_double_or(x, "max_abs_error", 0.0);
  return cal;
}

magmaan::sim_expected<magmaan::sim::CVine3CorrelationCalibration>
calibrate_cvine3_from_args(
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const std::vector<magmaan::sim::MarginalSpec>& marginal_specs,
    const std::string& family,
    const std::string& selection,
    Rcpp::Nullable<Rcpp::CharacterVector> families,
    Rcpp::Nullable<Rcpp::CharacterVector> family_set,
    const magmaan::sim::BivariateCopulaOptions& options) {
  const auto fallback_family = bivariate_copula_family_from_string(family);
  if (selection == "fixed") {
    if (families.isNull()) {
      return magmaan::sim::calibrate_cvine3_copula_correlation(
          fallback_family, target_corr, marginal_specs, options);
    }
    const auto family_spec =
        cvine3_family_spec_from_nullable(families, fallback_family);
    return magmaan::sim::calibrate_cvine3_copula_correlation(
        family_spec, target_corr, marginal_specs, options);
  }
  if (selection == "select_root" || selection == "root") {
    return magmaan::sim::calibrate_cvine3_copula_correlation_select_root(
        fallback_family, target_corr, marginal_specs, options);
  }
  if (selection == "select_families" || selection == "families") {
    const auto families_to_try =
        family_set_from_nullable(family_set, fallback_family);
    return magmaan::sim::calibrate_cvine3_copula_correlation_select_families(
        families_to_try, target_corr, marginal_specs, options);
  }
  if (selection == "select_structure" || selection == "structure") {
    const auto families_to_try =
        family_set_from_nullable(family_set, fallback_family);
    return magmaan::sim::calibrate_cvine3_copula_correlation_select_structure(
        families_to_try, target_corr, marginal_specs, options);
  }
  Rcpp::stop(
      "selection must be 'fixed', 'select_root', 'select_families', or 'select_structure'");
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
Rcpp::List sim_ordcorr_calibrate_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::List marginals,
    std::string metric = "polychoric",
    int max_bisection_iter = 80,
    double calibration_tol = 1e-8,
    double rho_bound = 0.999,
    std::string matrix_repair = "none",
    double matrix_repair_min_eigenvalue = 1e-8) {
  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const auto marginal_specs = ordinal_marginal_specs_from_list(marginals);

  magmaan::sim::OrdinalCorrelationOptions options = make_ordcorr_options(
      metric, max_bisection_iter, calibration_tol, rho_bound, matrix_repair,
      matrix_repair_min_eigenvalue);

  auto cal_or = magmaan::sim::calibrate_ordinal_correlation(
      corr, marginal_specs, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());
  return ordcorr_calibration_to_list(*cal_or);
}

// [[Rcpp::export]]
Rcpp::List sim_ordcorr_summary_calibrate_impl(
    Rcpp::NumericMatrix latent_corr,
    Rcpp::IntegerVector kinds,
    Rcpp::List thresholds,
    std::string metric = "polychoric",
    std::string matrix_repair = "none",
    double matrix_repair_min_eigenvalue = 1e-8) {
  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(latent_corr), latent_corr.nrow(), latent_corr.ncol());
  const auto kind_vec = observed_kinds_from_int(kinds);
  const auto threshold_vec = thresholds_from_list(thresholds, kind_vec.size());

  magmaan::sim::OrdinalCorrelationOptions options;
  options.metric = observed_correlation_metric_from_string(metric);
  options.matrix_repair = bivariate_copula_repair_from_string(matrix_repair);
  options.matrix_repair_min_eigenvalue = matrix_repair_min_eigenvalue;

  auto cal_or = magmaan::sim::calibrate_ordinal_correlation_summary(
      corr, kind_vec, threshold_vec, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());
  return ordcorr_calibration_to_list(*cal_or);
}

// [[Rcpp::export]]
Rcpp::List sim_ordcorr_draw_impl(Rcpp::List calibration,
                                 int n,
                                 int reps,
                                 double seed_base,
                                 double cholesky_jitter = 0.0) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  magmaan::sim::OrdinalCorrelationCalibration cal =
      ordcorr_calibration_from_list(calibration);

  auto pop_or = magmaan::sim::ordinal_correlation_population(cal);
  if (!pop_or.has_value()) stop_sim(pop_or.error());

  magmaan::sim::NormalOptions normal_options;
  if (cholesky_jitter > 0.0) normal_options.cholesky_jitter = cholesky_jitter;

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto draw_or = magmaan::sim::simulate_mixed_population_normal(
        static_cast<Eigen::Index>(n), *pop_or, rng, normal_options);
    if (!draw_or.has_value()) stop_sim(draw_or.error());
    draws[i] = ordcorr_draw_to_list(*draw_or);
  }
  return Rcpp::List::create(Rcpp::_["draws"] = draws);
}

// [[Rcpp::export]]
Rcpp::List sim_ordcorr_batch_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::List marginals,
    int n,
    int reps,
    double seed_base,
    std::string metric = "polychoric",
    int max_bisection_iter = 80,
    double calibration_tol = 1e-8,
    double rho_bound = 0.999,
    std::string matrix_repair = "none",
    double matrix_repair_min_eigenvalue = 1e-8,
    double cholesky_jitter = 0.0) {
  Rcpp::List calibration = sim_ordcorr_calibrate_impl(
      target_corr, marginals, metric, max_bisection_iter, calibration_tol,
      rho_bound, matrix_repair, matrix_repair_min_eigenvalue);
  return sim_ordcorr_draw_impl(calibration, n, reps, seed_base, cholesky_jitter);
}

// [[Rcpp::export]]
Rcpp::List sim_ordcorr_mg_calibrate_impl(
    Rcpp::List target_corrs,
    Rcpp::List marginals,
    Rcpp::Nullable<Rcpp::CharacterVector> group_labels = R_NilValue,
    std::string metric = "polychoric",
    int max_bisection_iter = 80,
    double calibration_tol = 1e-8,
    double rho_bound = 0.999,
    std::string matrix_repair = "none",
    double matrix_repair_min_eigenvalue = 1e-8) {
  const auto corr_groups = ordinal_corr_matrices_from_list(target_corrs);
  const auto marginal_groups = ordinal_marginal_groups_from_list(marginals);
  const auto labels = group_labels_from_nullable(group_labels);
  magmaan::sim::OrdinalCorrelationOptions options = make_ordcorr_options(
      metric, max_bisection_iter, calibration_tol, rho_bound, matrix_repair,
      matrix_repair_min_eigenvalue);

  auto cal_or = magmaan::sim::calibrate_ordinal_correlation_multigroup(
      corr_groups, marginal_groups, labels, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());
  auto cal = std::move(*cal_or);
  if (marginals.size() > 0) {
    cal.variable_names =
        variable_names_from_marginals(Rcpp::as<Rcpp::List>(marginals[0]));
  }

  Rcpp::List groups(static_cast<R_xlen_t>(cal.groups.size()));
  for (std::size_t g = 0; g < cal.groups.size(); ++g) {
    groups[static_cast<R_xlen_t>(g)] =
        ordcorr_calibration_to_list(cal.groups[g]);
  }

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["groups"] = groups,
      Rcpp::_["group_labels"] = Rcpp::wrap(cal.group_labels),
      Rcpp::_["n_groups"] = static_cast<int>(cal.groups.size()),
      Rcpp::_["variable_names"] = Rcpp::wrap(cal.variable_names));
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_ordcorr_mg_calibration", "list");
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_ordcorr_mg_summary_calibrate_impl(
    Rcpp::List latent_corrs,
    Rcpp::IntegerVector kinds,
    Rcpp::List thresholds,
    Rcpp::Nullable<Rcpp::CharacterVector> group_labels = R_NilValue,
    std::string metric = "polychoric",
    std::string matrix_repair = "none",
    double matrix_repair_min_eigenvalue = 1e-8) {
  const auto corr_groups = ordinal_corr_matrices_from_list(latent_corrs);
  const auto kind_vec = observed_kinds_from_int(kinds);
  std::vector<std::vector<magmaan::sim::ObservedKind>> kind_groups(
      corr_groups.size(), kind_vec);
  const auto threshold_groups =
      ordinal_threshold_groups_from_list(thresholds, kind_vec.size());
  const auto labels = group_labels_from_nullable(group_labels);

  magmaan::sim::OrdinalCorrelationOptions options;
  options.metric = observed_correlation_metric_from_string(metric);
  options.matrix_repair = bivariate_copula_repair_from_string(matrix_repair);
  options.matrix_repair_min_eigenvalue = matrix_repair_min_eigenvalue;

  auto cal_or = magmaan::sim::calibrate_ordinal_correlation_summary_multigroup(
      corr_groups, kind_groups, threshold_groups, labels, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());
  auto cal = std::move(*cal_or);

  Rcpp::List groups(static_cast<R_xlen_t>(cal.groups.size()));
  for (std::size_t g = 0; g < cal.groups.size(); ++g) {
    groups[static_cast<R_xlen_t>(g)] =
        ordcorr_calibration_to_list(cal.groups[g]);
  }

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["groups"] = groups,
      Rcpp::_["group_labels"] = Rcpp::wrap(cal.group_labels),
      Rcpp::_["n_groups"] = static_cast<int>(cal.groups.size()),
      Rcpp::_["variable_names"] = Rcpp::CharacterVector());
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_ordcorr_mg_calibration", "list");
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_ordcorr_mg_draw_impl(
    Rcpp::List calibration,
    Rcpp::IntegerVector n,
    int reps,
    double seed_base,
    double cholesky_jitter = 0.0) {
  if (reps <= 0) Rcpp::stop("reps must be positive");

  Rcpp::List group_list = Rcpp::as<Rcpp::List>(calibration["groups"]);
  magmaan::sim::MultiGroupOrdinalCorrelationCalibration cal;
  cal.group_labels =
      Rcpp::as<std::vector<std::string>>(calibration["group_labels"]);
  if (calibration.containsElementNamed("variable_names")) {
    cal.variable_names =
        Rcpp::as<std::vector<std::string>>(calibration["variable_names"]);
  }
  cal.groups.reserve(static_cast<std::size_t>(group_list.size()));
  for (R_xlen_t g = 0; g < group_list.size(); ++g) {
    cal.groups.push_back(ordcorr_calibration_from_list(
        Rcpp::as<Rcpp::List>(group_list[g])));
  }

  std::vector<Eigen::Index> n_by_group(cal.groups.size());
  if (n.size() == 1) {
    std::fill(n_by_group.begin(), n_by_group.end(),
              static_cast<Eigen::Index>(n[0]));
  } else if (static_cast<std::size_t>(n.size()) == cal.groups.size()) {
    for (R_xlen_t g = 0; g < n.size(); ++g) {
      n_by_group[static_cast<std::size_t>(g)] =
          static_cast<Eigen::Index>(n[g]);
    }
  } else {
    Rcpp::stop("n must be a scalar or one value per group");
  }

  magmaan::sim::NormalOptions normal_options;
  if (cholesky_jitter > 0.0) normal_options.cholesky_jitter = cholesky_jitter;

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto draw_or = magmaan::sim::simulate_ordinal_correlation_multigroup_normal(
        n_by_group, cal, rng, normal_options);
    if (!draw_or.has_value()) stop_sim(draw_or.error());
    Rcpp::List groups(static_cast<R_xlen_t>(draw_or->size()));
    for (std::size_t g = 0; g < draw_or->size(); ++g) {
      groups[static_cast<R_xlen_t>(g)] = ordcorr_draw_to_list((*draw_or)[g]);
    }
    draws[i] = Rcpp::List::create(Rcpp::_["groups"] = groups);
  }

  return Rcpp::List::create(
      Rcpp::_["draws"] = draws,
      Rcpp::_["group_labels"] = Rcpp::wrap(cal.group_labels),
      Rcpp::_["variable_names"] = Rcpp::wrap(cal.variable_names));
}

// [[Rcpp::export]]
Rcpp::List sim_ordcorr_mg_batch_impl(
    Rcpp::List target_corrs,
    Rcpp::List marginals,
    Rcpp::IntegerVector n,
    int reps,
    double seed_base,
    Rcpp::Nullable<Rcpp::CharacterVector> group_labels = R_NilValue,
    std::string metric = "polychoric",
    int max_bisection_iter = 80,
    double calibration_tol = 1e-8,
    double rho_bound = 0.999,
    std::string matrix_repair = "none",
    double matrix_repair_min_eigenvalue = 1e-8,
    double cholesky_jitter = 0.0) {
  Rcpp::List calibration = sim_ordcorr_mg_calibrate_impl(
      target_corrs, marginals, group_labels, metric, max_bisection_iter,
      calibration_tol, rho_bound, matrix_repair, matrix_repair_min_eigenvalue);
  return sim_ordcorr_mg_draw_impl(
      calibration, n, reps, seed_base, cholesky_jitter);
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
Rcpp::List sim_cvine_batch_impl(Rcpp::NumericMatrix target_corr,
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

  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const auto marginal_specs = marginal_specs_from_list(marginals);
  const auto copula_family = bivariate_copula_family_from_string(family);
  const magmaan::sim::BivariateCopulaOptions options =
      make_bivariate_copula_options(
          quadrature_points, max_bisection_iter, calibration_tol);

  auto cal_or = magmaan::sim::calibrate_cvine_copula_correlation(
      copula_family, corr, marginal_specs, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_cvine_copula_matrix(
        static_cast<Eigen::Index>(n), cal_or->copula, marginal_specs, rng,
        options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }

  Rcpp::List out = cvine_calibration_to_list(*cal_or);
  out["draws"] = draws;
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_cvine_calibrate_impl(Rcpp::NumericMatrix target_corr,
                                    Rcpp::List marginals,
                                    std::string family = "frank",
                                    int quadrature_points = 31,
                                    int max_bisection_iter = 80,
                                    double calibration_tol = 1e-6) {
  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const auto marginal_specs = marginal_specs_from_list(marginals);
  const auto copula_family = bivariate_copula_family_from_string(family);
  const magmaan::sim::BivariateCopulaOptions options =
      make_bivariate_copula_options(
          quadrature_points, max_bisection_iter, calibration_tol);

  auto cal_or = magmaan::sim::calibrate_cvine_copula_correlation(
      copula_family, corr, marginal_specs, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List out = cvine_calibration_to_list(*cal_or);
  out["marginals"] = marginal_specs_to_list(marginal_specs);
  out["options"] = bivariate_copula_options_to_list(options);
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_cvine_calibration", "list");
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_cvine_draw_impl(Rcpp::List calibration,
                               int n,
                               int reps,
                               double seed_base,
                               int quadrature_points = -1,
                               int max_bisection_iter = -1) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const auto copula = cvine_copula_spec_from_list(
      Rcpp::as<Rcpp::List>(calibration["copula"]));
  const auto marginal_specs = marginal_specs_from_list(
      Rcpp::as<Rcpp::List>(calibration["marginals"]));

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
    auto X_or = magmaan::sim::simulate_cvine_copula_matrix(
        static_cast<Eigen::Index>(n), copula, marginal_specs, rng, options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }
  return Rcpp::List::create(Rcpp::_["draws"] = draws);
}

// [[Rcpp::export]]
Rcpp::List sim_cvine3_batch_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::List marginals,
    int n,
    int reps,
    double seed_base,
    std::string family = "frank",
    std::string selection = "fixed",
    Rcpp::Nullable<Rcpp::CharacterVector> families = R_NilValue,
    Rcpp::Nullable<Rcpp::CharacterVector> family_set = R_NilValue,
    int quadrature_points = 31,
    int max_bisection_iter = 80,
    double calibration_tol = 1e-6) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const auto marginal_specs = marginal_specs_from_list(marginals);
  const magmaan::sim::BivariateCopulaOptions options =
      make_bivariate_copula_options(
          quadrature_points, max_bisection_iter, calibration_tol);

  auto cal_or = calibrate_cvine3_from_args(
      corr, marginal_specs, family, selection, families, family_set, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_cvine3_copula_matrix(
        static_cast<Eigen::Index>(n), *cal_or, marginal_specs, rng, options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }

  Rcpp::List out = cvine3_calibration_to_list(*cal_or);
  out["selection"] = selection;
  out["marginals"] = marginal_specs_to_list(marginal_specs);
  out["options"] = bivariate_copula_options_to_list(options);
  out["draws"] = draws;
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_cvine3_calibrate_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::List marginals,
    std::string family = "frank",
    std::string selection = "fixed",
    Rcpp::Nullable<Rcpp::CharacterVector> families = R_NilValue,
    Rcpp::Nullable<Rcpp::CharacterVector> family_set = R_NilValue,
    int quadrature_points = 31,
    int max_bisection_iter = 80,
    double calibration_tol = 1e-6) {
  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const auto marginal_specs = marginal_specs_from_list(marginals);
  const magmaan::sim::BivariateCopulaOptions options =
      make_bivariate_copula_options(
          quadrature_points, max_bisection_iter, calibration_tol);

  auto cal_or = calibrate_cvine3_from_args(
      corr, marginal_specs, family, selection, families, family_set, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List out = cvine3_calibration_to_list(*cal_or);
  out["selection"] = selection;
  out["marginals"] = marginal_specs_to_list(marginal_specs);
  out["options"] = bivariate_copula_options_to_list(options);
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_cvine3_calibration", "list");
  return out;
}

// [[Rcpp::export]]
Rcpp::List sim_cvine3_draw_impl(Rcpp::List calibration,
                                int n,
                                int reps,
                                double seed_base,
                                int quadrature_points = -1,
                                int max_bisection_iter = -1) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  auto cal = cvine3_calibration_from_list(calibration);
  const auto marginal_specs = marginal_specs_from_list(
      Rcpp::as<Rcpp::List>(calibration["marginals"]));

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
    auto X_or = magmaan::sim::simulate_cvine3_copula_matrix(
        static_cast<Eigen::Index>(n), cal, marginal_specs, rng, options);
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

// [[Rcpp::export]]
Rcpp::List sim_model_calibrate_impl(
    SEXP fit_or_partable,
    Rcpp::Nullable<Rcpp::NumericVector> theta = R_NilValue) {
  auto population = calibrate_model_from_arg(fit_or_partable, theta);
  return model_implied_population_to_list(population);
}

// [[Rcpp::export]]
Rcpp::List sim_model_draw_impl(
    Rcpp::List calibration,
    int n,
    int reps,
    double seed_base,
    std::string generator = "normal",
    double df = 5.0,
    Rcpp::Nullable<Rcpp::NumericVector> mixture_weights = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> mixture_scale_multipliers = R_NilValue,
    double contamination_probability = 0.05,
    double contamination_scale_multiplier = 3.0,
    double slash_q = 5.0,
    double cholesky_jitter = 0.0) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const auto population = model_implied_population_from_list(calibration);
  const auto generator_spec = make_model_generator_spec(
      generator, df, mixture_weights, mixture_scale_multipliers,
      contamination_probability, contamination_scale_multiplier, slash_q,
      cholesky_jitter);

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto draw_or = magmaan::sim::simulate_model_implied(
        static_cast<Eigen::Index>(n), population, generator_spec, rng);
    if (!draw_or.has_value()) stop_sim(draw_or.error());

    Rcpp::List groups(static_cast<R_xlen_t>(draw_or->size()));
    for (std::size_t b = 0; b < draw_or->size(); ++b) {
      const auto& group = (*draw_or)[b];
      groups[static_cast<R_xlen_t>(b)] = Rcpp::List::create(
          Rcpp::_["X"] = Rcpp::wrap(group.observed.X),
          Rcpp::_["ordered"] = int32_vector_to_r(group.observed.ordered),
          Rcpp::_["n_levels"] = int32_vector_to_r(group.observed.n_levels));
    }
    draws[i] = Rcpp::List::create(Rcpp::_["groups"] = groups);
  }

  return Rcpp::List::create(
      Rcpp::_["draws"] = draws,
      Rcpp::_["ov_names"] = calibration["ov_names"]);
}

// [[Rcpp::export]]
Rcpp::List sim_model_batch_impl(
    SEXP fit_or_partable,
    int n,
    int reps,
    double seed_base,
    Rcpp::Nullable<Rcpp::NumericVector> theta = R_NilValue,
    std::string generator = "normal",
    double df = 5.0,
    Rcpp::Nullable<Rcpp::NumericVector> mixture_weights = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> mixture_scale_multipliers = R_NilValue,
    double contamination_probability = 0.05,
    double contamination_scale_multiplier = 3.0,
    double slash_q = 5.0,
    double cholesky_jitter = 0.0) {
  auto population = calibrate_model_from_arg(fit_or_partable, theta);
  Rcpp::List calibration = model_implied_population_to_list(population);
  return sim_model_draw_impl(
      calibration, n, reps, seed_base, generator, df, mixture_weights,
      mixture_scale_multipliers, contamination_probability,
      contamination_scale_multiplier, slash_q, cholesky_jitter);
}

namespace {

magmaan::sim::ValeMaurelliOptions make_vm_options(
    int max_iter, double coefficient_tol, double correlation_tol,
    double rho_bound, double cholesky_jitter) {
  magmaan::sim::ValeMaurelliOptions options;
  options.max_iter = max_iter;
  options.coefficient_tol = coefficient_tol;
  options.correlation_tol = correlation_tol;
  options.rho_bound = rho_bound;
  options.cholesky_jitter = cholesky_jitter;
  return options;
}

Rcpp::NumericMatrix vm_coefficients_to_matrix(
    const std::vector<magmaan::sim::FleishmanCoefficients>& coeffs) {
  Rcpp::NumericMatrix m(static_cast<int>(coeffs.size()), 4);
  for (std::size_t i = 0; i < coeffs.size(); ++i) {
    const int row = static_cast<int>(i);
    m(row, 0) = coeffs[i].a;
    m(row, 1) = coeffs[i].b;
    m(row, 2) = coeffs[i].c;
    m(row, 3) = coeffs[i].d;
  }
  Rcpp::colnames(m) = Rcpp::CharacterVector::create("a", "b", "c", "d");
  return m;
}

std::vector<magmaan::sim::FleishmanCoefficients> vm_coefficients_from_matrix(
    Rcpp::NumericMatrix m) {
  std::vector<magmaan::sim::FleishmanCoefficients> coeffs(
      static_cast<std::size_t>(m.nrow()));
  for (int i = 0; i < m.nrow(); ++i) {
    coeffs[static_cast<std::size_t>(i)].a = m(i, 0);
    coeffs[static_cast<std::size_t>(i)].b = m(i, 1);
    coeffs[static_cast<std::size_t>(i)].c = m(i, 2);
    coeffs[static_cast<std::size_t>(i)].d = m(i, 3);
  }
  return coeffs;
}

Rcpp::List vm_calibration_to_list(
    const magmaan::sim::ValeMaurelliCalibration& cal,
    const Eigen::MatrixXd& target_corr,
    const Eigen::VectorXd& skewness,
    const Eigen::VectorXd& excess_kurtosis) {
  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["coefficients"] = vm_coefficients_to_matrix(cal.coefficients),
      Rcpp::_["intermediate_corr"] = Rcpp::wrap(cal.intermediate_corr),
      Rcpp::_["target_corr"] = Rcpp::wrap(target_corr),
      Rcpp::_["skewness"] = Rcpp::wrap(skewness),
      Rcpp::_["excess_kurtosis"] = Rcpp::wrap(excess_kurtosis));
  out.attr("class") = Rcpp::CharacterVector::create(
      "magmaan_vm_calibration", "list");
  return out;
}

magmaan::sim::ValeMaurelliCalibration vm_calibration_from_list(
    Rcpp::List calibration) {
  magmaan::sim::ValeMaurelliCalibration cal;
  cal.coefficients = vm_coefficients_from_matrix(
      Rcpp::as<Rcpp::NumericMatrix>(calibration["coefficients"]));
  cal.intermediate_corr =
      Rcpp::as<Eigen::MatrixXd>(calibration["intermediate_corr"]);
  return cal;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List sim_vm_calibrate_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::NumericVector target_skewness,
    Rcpp::NumericVector target_excess_kurtosis,
    int max_iter = 80,
    double coefficient_tol = 1e-10,
    double correlation_tol = 1e-10,
    double rho_bound = 0.999,
    double cholesky_jitter = 1e-10) {
  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const Eigen::Map<Eigen::VectorXd> skew(
      REAL(target_skewness), target_skewness.size());
  const Eigen::Map<Eigen::VectorXd> kurt(
      REAL(target_excess_kurtosis), target_excess_kurtosis.size());
  const magmaan::sim::ValeMaurelliOptions options = make_vm_options(
      max_iter, coefficient_tol, correlation_tol, rho_bound, cholesky_jitter);

  auto cal_or = magmaan::sim::calibrate_vale_maurelli(corr, skew, kurt, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());
  return vm_calibration_to_list(*cal_or, corr, skew, kurt);
}

// [[Rcpp::export]]
Rcpp::List sim_vm_draw_impl(Rcpp::List calibration,
                            int n,
                            int reps,
                            double seed_base,
                            double cholesky_jitter = 1e-10) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const magmaan::sim::ValeMaurelliCalibration cal =
      vm_calibration_from_list(calibration);
  magmaan::sim::ValeMaurelliOptions options;
  options.cholesky_jitter = cholesky_jitter;

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_vale_maurelli_matrix(
        static_cast<Eigen::Index>(n), cal, rng, options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }
  return Rcpp::List::create(Rcpp::_["draws"] = draws);
}

// [[Rcpp::export]]
Rcpp::List sim_vm_batch_impl(
    Rcpp::NumericMatrix target_corr,
    Rcpp::NumericVector target_skewness,
    Rcpp::NumericVector target_excess_kurtosis,
    int n,
    int reps,
    double seed_base,
    int max_iter = 80,
    double coefficient_tol = 1e-10,
    double correlation_tol = 1e-10,
    double rho_bound = 0.999,
    double cholesky_jitter = 1e-10) {
  if (n <= 0) Rcpp::stop("n must be positive");
  if (reps <= 0) Rcpp::stop("reps must be positive");

  const Eigen::Map<Eigen::MatrixXd> corr(
      REAL(target_corr), target_corr.nrow(), target_corr.ncol());
  const Eigen::Map<Eigen::VectorXd> skew(
      REAL(target_skewness), target_skewness.size());
  const Eigen::Map<Eigen::VectorXd> kurt(
      REAL(target_excess_kurtosis), target_excess_kurtosis.size());
  const magmaan::sim::ValeMaurelliOptions options = make_vm_options(
      max_iter, coefficient_tol, correlation_tol, rho_bound, cholesky_jitter);

  auto cal_or = magmaan::sim::calibrate_vale_maurelli(corr, skew, kurt, options);
  if (!cal_or.has_value()) stop_sim(cal_or.error());

  Rcpp::List draws(reps);
  for (int i = 0; i < reps; ++i) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed_base) +
                        static_cast<std::uint64_t>(i + 1));
    auto X_or = magmaan::sim::simulate_vale_maurelli_matrix(
        static_cast<Eigen::Index>(n), *cal_or, rng, options);
    if (!X_or.has_value()) stop_sim(X_or.error());
    draws[i] = Rcpp::wrap(*X_or);
  }

  return Rcpp::List::create(
      Rcpp::_["draws"] = draws,
      Rcpp::_["coefficients"] = vm_coefficients_to_matrix(cal_or->coefficients),
      Rcpp::_["intermediate_corr"] = Rcpp::wrap(cal_or->intermediate_corr));
}
