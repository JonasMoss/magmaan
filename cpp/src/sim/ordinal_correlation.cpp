#include "magmaan/sim/ordinal_correlation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/error.hpp"
#include "magmaan/sim/detail_matrix_repair.hpp"

namespace magmaan::sim {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kInvSqrt2Pi = 0.39894228040143267794;

SimError make_err(SimError::Kind k, std::string detail) {
  return SimError{k, std::move(detail)};
}

double std_normal_pdf(double x) noexcept {
  return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

bool is_ordinal(ObservedKind k) noexcept { return k == ObservedKind::Ordinal; }

std::size_t ix(Eigen::Index i) noexcept { return static_cast<std::size_t>(i); }

std::vector<std::string> default_group_labels(std::size_t n_groups) {
  std::vector<std::string> labels;
  labels.reserve(n_groups);
  for (std::size_t g = 0; g < n_groups; ++g) {
    labels.push_back(std::to_string(g + 1));
  }
  return labels;
}

sim_expected<void>
validate_correlation_matrix(const Eigen::Ref<const Eigen::MatrixXd>& corr,
                            const char* caller) {
  const Eigen::Index p = corr.rows();
  if (p <= 0 || corr.cols() != p) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": correlation matrix must be non-empty and square"));
  }
  if (!corr.allFinite()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": correlation matrix must be finite"));
  }
  for (Eigen::Index i = 0; i < p; ++i) {
    if (std::abs(corr(i, i) - 1.0) > 1e-8) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          std::string(caller) + ": correlation diagonal must be one"));
    }
    for (Eigen::Index j = i + 1; j < p; ++j) {
      if (std::abs(corr(i, j) - corr(j, i)) > 1e-8) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            std::string(caller) + ": correlation matrix must be symmetric"));
      }
      if (std::abs(corr(i, j)) > 1.0 + 1e-8) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            std::string(caller) + ": off-diagonal correlations must be in [-1, 1]"));
      }
    }
  }
  return {};
}

sim_expected<void>
validate_summary_shape(const Eigen::Ref<const Eigen::MatrixXd>& corr,
                       const std::vector<ObservedKind>& kinds,
                       const std::vector<Eigen::VectorXd>& thresholds,
                       const char* caller) {
  if (auto ok = validate_correlation_matrix(corr, caller); !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  const std::size_t p = ix(corr.rows());
  if (kinds.size() != p || thresholds.size() != p) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        std::string(caller) + ": kinds and thresholds must match matrix dimension"));
  }
  for (std::size_t v = 0; v < p; ++v) {
    if (kinds[v] == ObservedKind::Continuous) {
      if (thresholds[v].size() != 0) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            std::string(caller) + ": continuous variables must have empty thresholds"));
      }
      continue;
    }
    if (thresholds[v].size() == 0) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          std::string(caller) + ": ordinal variables must have thresholds"));
    }
    if (!thresholds[v].allFinite()) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          std::string(caller) + ": thresholds must be finite"));
    }
    for (Eigen::Index k = 1; k < thresholds[v].size(); ++k) {
      if (!(thresholds[v](k - 1) < thresholds[v](k))) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            std::string(caller) + ": thresholds must be strictly increasing"));
      }
    }
  }
  return {};
}

// Category probabilities p_1..p_k from k-1 thresholds.
Eigen::VectorXd category_probs_from_thresholds(const Eigen::VectorXd& th) {
  const Eigen::Index k = th.size() + 1;
  Eigen::VectorXd p(k);
  double prev = 0.0;
  for (Eigen::Index c = 0; c < th.size(); ++c) {
    const double cdf = normal_cdf(th(c));
    p(c) = cdf - prev;
    prev = cdf;
  }
  p(k - 1) = 1.0 - prev;
  return p;
}

struct CodeMoments {
  double mean = 0.0;
  double sd = 0.0;
};

CodeMoments code_moments(const Eigen::VectorXd& th) {
  const Eigen::VectorXd p = category_probs_from_thresholds(th);
  double mean = 0.0;
  double m2 = 0.0;
  for (Eigen::Index c = 0; c < p.size(); ++c) {
    const double code = static_cast<double>(c + 1);
    mean += code * p(c);
    m2 += code * code * p(c);
  }
  return CodeMoments{mean, std::sqrt(std::max(0.0, m2 - mean * mean))};
}

double sum_phi_thresholds(const Eigen::VectorXd& th) {
  double s = 0.0;
  for (Eigen::Index c = 0; c < th.size(); ++c) s += std_normal_pdf(th(c));
  return s;
}

// [lo, hi) latent bounds for category a (1-based) given k-1 thresholds.
std::pair<double, double> category_bounds(const Eigen::VectorXd& th,
                                          Eigen::Index a) {
  const Eigen::Index k = th.size() + 1;
  const double lo = (a == 1) ? -kInf : th(a - 2);
  const double hi = (a == k) ? kInf : th(a - 1);
  return {lo, hi};
}

// Pearson r of the integer codes 1..k of two ordinal variables at latent rho.
double code_pearson_corr(const Eigen::VectorXd& th_i,
                         const Eigen::VectorXd& th_j, double rho) {
  const CodeMoments mi = code_moments(th_i);
  const CodeMoments mj = code_moments(th_j);
  if (!(mi.sd > 0.0) || !(mj.sd > 0.0)) return 0.0;
  const Eigen::Index ki = th_i.size() + 1;
  const Eigen::Index kj = th_j.size() + 1;
  double e_ij = 0.0;
  for (Eigen::Index a = 1; a <= ki; ++a) {
    const auto [lo_i, hi_i] = category_bounds(th_i, a);
    for (Eigen::Index b = 1; b <= kj; ++b) {
      const auto [lo_j, hi_j] = category_bounds(th_j, b);
      const double pab =
          data::ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho);
      e_ij += static_cast<double>(a) * static_cast<double>(b) * pab;
    }
  }
  return (e_ij - mi.mean * mj.mean) / (mi.sd * mj.sd);
}

// Polyserial correlation between ordinal codes (thresholds th) and a standard
// normal continuous variable with latent correlation rho:
//   corr = rho * sum_c phi(tau_c) / sd(codes).
double polyserial_corr(const Eigen::VectorXd& th, double rho) {
  const CodeMoments m = code_moments(th);
  if (!(m.sd > 0.0)) return 0.0;
  return rho * sum_phi_thresholds(th) / m.sd;
}

double clamp_rho(double rho, double bound, bool& hit_lower, bool& hit_upper) {
  if (rho < -bound) { hit_lower = true; return -bound; }
  if (rho > bound) { hit_upper = true; return bound; }
  return rho;
}

OrdinalPairCalibration
solve_pair_rho(ObservedCorrelationMetric metric,
               ObservedKind kind_i, const Eigen::VectorXd& th_i,
               ObservedKind kind_j, const Eigen::VectorXd& th_j,
               double target, const OrdinalCorrelationOptions& opt) {
  OrdinalPairCalibration rec;
  rec.target_corr = target;
  const double bound = opt.rho_bound;
  const bool ord_i = is_ordinal(kind_i);
  const bool ord_j = is_ordinal(kind_j);

  // continuous x continuous, and ordinal x ordinal under the closed-form metrics
  // (Polychoric/Polyserial): latent rho == target.
  const bool closed_form_identity =
      (!ord_i && !ord_j) ||
      (ord_i && ord_j && metric != ObservedCorrelationMetric::PearsonCodes);
  if (closed_form_identity) {
    rec.latent_rho = clamp_rho(target, bound, rec.hit_lower, rec.hit_upper);
    rec.infeasible = rec.hit_lower || rec.hit_upper;
    return rec;
  }

  // ordinal x continuous: polyserial closed form rho = target * sd / sum_phi.
  if (ord_i != ord_j) {
    const Eigen::VectorXd& th = ord_i ? th_i : th_j;
    const CodeMoments m = code_moments(th);
    const double sum_phi = sum_phi_thresholds(th);
    const double rho = (sum_phi > 0.0) ? target * m.sd / sum_phi : 0.0;
    rec.latent_rho = clamp_rho(rho, bound, rec.hit_lower, rec.hit_upper);
    rec.infeasible = rec.hit_lower || rec.hit_upper;
    return rec;
  }

  // ordinal x ordinal, PearsonCodes: monotone bisection on [-bound, bound].
  auto residual = [&](double rho) {
    return code_pearson_corr(th_i, th_j, rho) - target;
  };
  double lo = -bound;
  double hi = bound;
  if (residual(lo) > 0.0) {
    rec.latent_rho = lo;
    rec.hit_lower = true;
    rec.infeasible = true;
    return rec;
  }
  if (residual(hi) < 0.0) {
    rec.latent_rho = hi;
    rec.hit_upper = true;
    rec.infeasible = true;
    return rec;
  }
  double mid = 0.0;
  double fmid = 0.0;
  int it = 0;
  for (; it < opt.max_bisection_iter; ++it) {
    mid = 0.5 * (lo + hi);
    fmid = residual(mid);
    if (std::abs(fmid) <= opt.calibration_tol ||
        (hi - lo) <= 2.0 * opt.calibration_tol) {
      break;
    }
    if (fmid < 0.0) lo = mid; else hi = mid;
  }
  rec.iterations = it + 1;
  rec.latent_rho = mid;
  rec.converged = std::abs(fmid) <= std::max(opt.calibration_tol, 1e-7);
  return rec;
}

}  // namespace

sim_expected<double>
ordinal_pair_observed_corr(ObservedCorrelationMetric metric,
                           ObservedKind kind_i,
                           const Eigen::VectorXd& thresholds_i,
                           ObservedKind kind_j,
                           const Eigen::VectorXd& thresholds_j,
                           double latent_rho) {
  const bool ord_i = is_ordinal(kind_i);
  const bool ord_j = is_ordinal(kind_j);
  if (ord_i && thresholds_i.size() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "ordinal_pair_observed_corr: ordinal variable i has no thresholds"));
  }
  if (ord_j && thresholds_j.size() == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "ordinal_pair_observed_corr: ordinal variable j has no thresholds"));
  }
  if (!ord_i && !ord_j) return latent_rho;
  if (ord_i && !ord_j) return polyserial_corr(thresholds_i, latent_rho);
  if (!ord_i && ord_j) return polyserial_corr(thresholds_j, latent_rho);
  if (metric == ObservedCorrelationMetric::PearsonCodes) {
    return code_pearson_corr(thresholds_i, thresholds_j, latent_rho);
  }
  return latent_rho;
}

sim_expected<OrdinalCorrelationCalibration>
calibrate_ordinal_correlation(
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const std::vector<OrdinalMarginalSpec>& marginals,
    const OrdinalCorrelationOptions& options) {
  constexpr const char* caller = "calibrate_ordinal_correlation";
  const Eigen::Index p = target_corr.rows();
  if (auto ok = validate_correlation_matrix(target_corr, caller);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }
  if (marginals.size() != ix(p)) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ordinal_correlation: marginal count must match matrix dimension"));
  }

  OrdinalCorrelationCalibration cal;
  cal.metric = options.metric;
  cal.target_corr = target_corr;
  cal.kinds.resize(ix(p));
  cal.thresholds.resize(ix(p));
  for (Eigen::Index v = 0; v < p; ++v) {
    const OrdinalMarginalSpec& m = marginals[ix(v)];
    cal.kinds[ix(v)] = m.kind;
    if (m.kind == ObservedKind::Continuous) {
      cal.thresholds[ix(v)] = Eigen::VectorXd();
    } else {
      auto th_or = thresholds_from_probabilities(m.proportions);
      if (!th_or.has_value()) return std::unexpected(th_or.error());
      cal.thresholds[ix(v)] = *th_or;
    }
  }

  Eigen::MatrixXd latent = Eigen::MatrixXd::Identity(p, p);
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = i + 1; j < p; ++j) {
      OrdinalPairCalibration rec = solve_pair_rho(
          options.metric, cal.kinds[ix(i)], cal.thresholds[ix(i)],
          cal.kinds[ix(j)], cal.thresholds[ix(j)], target_corr(i, j), options);
      rec.i = static_cast<std::int32_t>(i);
      rec.j = static_cast<std::int32_t>(j);
      latent(i, j) = rec.latent_rho;
      latent(j, i) = rec.latent_rho;
      cal.pairs.push_back(rec);
    }
  }

  auto repair_or = repair_correlation_matrix_if_requested(
      latent, options.matrix_repair, options.matrix_repair_min_eigenvalue,
      "calibrate_ordinal_correlation");
  if (!repair_or.has_value()) return std::unexpected(repair_or.error());
  const MatrixRepairResult& rep = *repair_or;
  cal.latent_corr = rep.corr;
  cal.raw_min_eigenvalue = rep.raw_min_eigenvalue;
  cal.repaired_min_eigenvalue = rep.repaired_min_eigenvalue;
  cal.repair_ridge = rep.ridge;
  cal.repair_shrinkage = rep.shrinkage;
  cal.repair_applied = rep.repaired;

  cal.achieved_corr = Eigen::MatrixXd::Identity(p, p);
  double max_err = 0.0;
  std::size_t pair_idx = 0;
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = i + 1; j < p; ++j) {
      auto ach_or = ordinal_pair_observed_corr(
          options.metric, cal.kinds[ix(i)], cal.thresholds[ix(i)],
          cal.kinds[ix(j)], cal.thresholds[ix(j)], cal.latent_corr(i, j));
      if (!ach_or.has_value()) return std::unexpected(ach_or.error());
      const double ach = *ach_or;
      cal.achieved_corr(i, j) = ach;
      cal.achieved_corr(j, i) = ach;
      cal.pairs[pair_idx].achieved_corr = ach;
      max_err = std::max(max_err, std::abs(ach - target_corr(i, j)));
      ++pair_idx;
    }
  }
  cal.max_abs_error = max_err;
  return cal;
}

sim_expected<OrdinalCorrelationCalibration>
calibrate_ordinal_correlation_summary(
    const Eigen::Ref<const Eigen::MatrixXd>& latent_corr,
    const std::vector<ObservedKind>& kinds,
    const std::vector<Eigen::VectorXd>& thresholds,
    const OrdinalCorrelationOptions& options) {
  constexpr const char* caller = "calibrate_ordinal_correlation_summary";
  if (auto ok = validate_summary_shape(latent_corr, kinds, thresholds, caller);
      !ok.has_value()) {
    return std::unexpected(ok.error());
  }

  auto repair_or = repair_correlation_matrix_if_requested(
      latent_corr, options.matrix_repair,
      options.matrix_repair_min_eigenvalue, caller);
  if (!repair_or.has_value()) return std::unexpected(repair_or.error());
  const MatrixRepairResult& rep = *repair_or;

  const Eigen::Index p = latent_corr.rows();
  OrdinalCorrelationCalibration cal;
  cal.metric = options.metric;
  cal.kinds = kinds;
  cal.thresholds = thresholds;
  cal.target_corr = latent_corr;
  cal.latent_corr = rep.corr;
  cal.achieved_corr = rep.corr;
  cal.raw_min_eigenvalue = rep.raw_min_eigenvalue;
  cal.repaired_min_eigenvalue = rep.repaired_min_eigenvalue;
  cal.repair_ridge = rep.ridge;
  cal.repair_shrinkage = rep.shrinkage;
  cal.repair_applied = rep.repaired;

  double max_err = 0.0;
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = i + 1; j < p; ++j) {
      OrdinalPairCalibration rec;
      rec.i = static_cast<std::int32_t>(i);
      rec.j = static_cast<std::int32_t>(j);
      rec.target_corr = latent_corr(i, j);
      rec.latent_rho = latent_corr(i, j);
      rec.achieved_corr = rep.corr(i, j);
      rec.iterations = 0;
      rec.converged = true;
      cal.pairs.push_back(rec);
      max_err = std::max(max_err, std::abs(rep.corr(i, j) - latent_corr(i, j)));
    }
  }
  cal.max_abs_error = max_err;
  return cal;
}

sim_expected<MultiGroupOrdinalCorrelationCalibration>
calibrate_ordinal_correlation_multigroup(
    const std::vector<Eigen::MatrixXd>& target_corrs,
    const std::vector<std::vector<OrdinalMarginalSpec>>& marginals,
    const std::vector<std::string>& group_labels,
    const OrdinalCorrelationOptions& options) {
  const std::size_t n_groups = target_corrs.size();
  if (n_groups == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ordinal_correlation_multigroup: need at least one group"));
  }
  if (marginals.size() != n_groups) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ordinal_correlation_multigroup: target/marginal group counts must match"));
  }
  if (!group_labels.empty() && group_labels.size() != n_groups) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ordinal_correlation_multigroup: group_labels size must match group count"));
  }

  MultiGroupOrdinalCorrelationCalibration out;
  out.group_labels =
      group_labels.empty() ? default_group_labels(n_groups) : group_labels;
  out.groups.reserve(n_groups);
  for (std::size_t g = 0; g < n_groups; ++g) {
    auto cal_or = calibrate_ordinal_correlation(
        target_corrs[g], marginals[g], options);
    if (!cal_or.has_value()) return std::unexpected(cal_or.error());
    if (g > 0) {
      const auto& ref = out.groups.front();
      const auto& cur = *cal_or;
      if (cur.kinds.size() != ref.kinds.size()) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "calibrate_ordinal_correlation_multigroup: variable count must match across groups"));
      }
      for (std::size_t v = 0; v < ref.kinds.size(); ++v) {
        if (cur.kinds[v] != ref.kinds[v]) {
          return std::unexpected(make_err(
              SimError::Kind::InvalidInput,
              "calibrate_ordinal_correlation_multigroup: variable kinds must match across groups"));
        }
        if (cur.thresholds[v].size() != ref.thresholds[v].size()) {
          return std::unexpected(make_err(
              SimError::Kind::InvalidInput,
              "calibrate_ordinal_correlation_multigroup: ordinal category counts must match across groups"));
        }
      }
    }
    out.groups.push_back(std::move(*cal_or));
  }
  return out;
}

sim_expected<MultiGroupOrdinalCorrelationCalibration>
calibrate_ordinal_correlation_summary_multigroup(
    const std::vector<Eigen::MatrixXd>& latent_corrs,
    const std::vector<std::vector<ObservedKind>>& kinds,
    const std::vector<std::vector<Eigen::VectorXd>>& thresholds,
    const std::vector<std::string>& group_labels,
    const OrdinalCorrelationOptions& options) {
  const std::size_t n_groups = latent_corrs.size();
  if (n_groups == 0) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ordinal_correlation_summary_multigroup: need at least one group"));
  }
  if (kinds.size() != n_groups || thresholds.size() != n_groups) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ordinal_correlation_summary_multigroup: group counts must match"));
  }
  if (!group_labels.empty() && group_labels.size() != n_groups) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "calibrate_ordinal_correlation_summary_multigroup: group_labels size must match group count"));
  }

  MultiGroupOrdinalCorrelationCalibration out;
  out.group_labels =
      group_labels.empty() ? default_group_labels(n_groups) : group_labels;
  out.groups.reserve(n_groups);
  for (std::size_t g = 0; g < n_groups; ++g) {
    auto cal_or = calibrate_ordinal_correlation_summary(
        latent_corrs[g], kinds[g], thresholds[g], options);
    if (!cal_or.has_value()) return std::unexpected(cal_or.error());
    if (g > 0) {
      const auto& ref = out.groups.front();
      const auto& cur = *cal_or;
      if (cur.kinds.size() != ref.kinds.size()) {
        return std::unexpected(make_err(
            SimError::Kind::InvalidInput,
            "calibrate_ordinal_correlation_summary_multigroup: variable count must match across groups"));
      }
      for (std::size_t v = 0; v < ref.kinds.size(); ++v) {
        if (cur.kinds[v] != ref.kinds[v]) {
          return std::unexpected(make_err(
              SimError::Kind::InvalidInput,
              "calibrate_ordinal_correlation_summary_multigroup: variable kinds must match across groups"));
        }
        if (cur.thresholds[v].size() != ref.thresholds[v].size()) {
          return std::unexpected(make_err(
              SimError::Kind::InvalidInput,
              "calibrate_ordinal_correlation_summary_multigroup: ordinal category counts must match across groups"));
        }
      }
    }
    out.groups.push_back(std::move(*cal_or));
  }
  return out;
}

sim_expected<MixedPopulation>
ordinal_correlation_population(const OrdinalCorrelationCalibration& calibration) {
  const Eigen::Index p = calibration.latent_corr.rows();
  if (p <= 0 || calibration.latent_corr.cols() != p) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "ordinal_correlation_population: calibration has no latent correlation"));
  }
  if (calibration.kinds.size() != ix(p) ||
      calibration.thresholds.size() != ix(p)) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "ordinal_correlation_population: calibration kinds/thresholds size mismatch"));
  }
  MixedPopulation pop;
  pop.latent.mean = Eigen::VectorXd::Zero(p);
  pop.latent.covariance = calibration.latent_corr;
  pop.observed.kinds = calibration.kinds;
  pop.observed.thresholds = calibration.thresholds;
  return pop;
}

sim_expected<std::vector<MixedPopulation>>
ordinal_correlation_populations(
    const MultiGroupOrdinalCorrelationCalibration& calibration) {
  if (calibration.groups.empty()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "ordinal_correlation_populations: calibration has no groups"));
  }
  if (!calibration.group_labels.empty() &&
      calibration.group_labels.size() != calibration.groups.size()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "ordinal_correlation_populations: group_labels size must match group count"));
  }
  std::vector<MixedPopulation> out;
  out.reserve(calibration.groups.size());
  for (const auto& group : calibration.groups) {
    auto pop_or = ordinal_correlation_population(group);
    if (!pop_or.has_value()) return std::unexpected(pop_or.error());
    out.push_back(std::move(*pop_or));
  }
  return out;
}

sim_expected<MixedPopulationDraw>
simulate_ordinal_correlation_normal(
    Eigen::Index n,
    const Eigen::Ref<const Eigen::MatrixXd>& target_corr,
    const std::vector<OrdinalMarginalSpec>& marginals,
    std::mt19937_64& rng,
    const OrdinalCorrelationOptions& options,
    const NormalOptions& normal_options) {
  auto cal_or = calibrate_ordinal_correlation(target_corr, marginals, options);
  if (!cal_or.has_value()) return std::unexpected(cal_or.error());
  auto pop_or = ordinal_correlation_population(*cal_or);
  if (!pop_or.has_value()) return std::unexpected(pop_or.error());
  return simulate_mixed_population_normal(n, *pop_or, rng, normal_options);
}

sim_expected<std::vector<MixedPopulationDraw>>
simulate_ordinal_correlation_multigroup_normal(
    const std::vector<Eigen::Index>& n,
    const MultiGroupOrdinalCorrelationCalibration& calibration,
    std::mt19937_64& rng,
    const NormalOptions& normal_options) {
  if (n.size() != calibration.groups.size()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "simulate_ordinal_correlation_multigroup_normal: n size must match group count"));
  }
  auto pops_or = ordinal_correlation_populations(calibration);
  if (!pops_or.has_value()) return std::unexpected(pops_or.error());
  std::vector<MixedPopulationDraw> out;
  out.reserve(pops_or->size());
  for (std::size_t g = 0; g < pops_or->size(); ++g) {
    if (n[g] <= 0) {
      return std::unexpected(make_err(
          SimError::Kind::InvalidInput,
          "simulate_ordinal_correlation_multigroup_normal: group sample sizes must be positive"));
    }
    auto draw_or = simulate_mixed_population_normal(
        n[g], (*pops_or)[g], rng, normal_options);
    if (!draw_or.has_value()) return std::unexpected(draw_or.error());
    out.push_back(std::move(*draw_or));
  }
  return out;
}

sim_expected<std::vector<GroupOrdinalProportions>>
multigroup_category_proportions(
    const std::vector<std::string>& group_labels,
    const std::vector<MixedPopulationDraw>& draws) {
  if (draws.empty()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "multigroup_category_proportions: need at least one draw"));
  }
  if (!group_labels.empty() && group_labels.size() != draws.size()) {
    return std::unexpected(make_err(
        SimError::Kind::InvalidInput,
        "multigroup_category_proportions: group_labels size must match draw count"));
  }
  const std::vector<std::string> labels =
      group_labels.empty() ? default_group_labels(draws.size()) : group_labels;
  std::vector<GroupOrdinalProportions> out;
  out.reserve(draws.size());
  for (std::size_t g = 0; g < draws.size(); ++g) {
    out.push_back(GroupOrdinalProportions{
        labels[g], draws[g].observed.category_proportions});
  }
  return out;
}

}  // namespace magmaan::sim
