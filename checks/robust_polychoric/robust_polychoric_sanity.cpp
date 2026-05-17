#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/data/pairwise_mixed.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

namespace {

using magmaan::data::PairwiseOrdinalHWeightedStatsOptions;
using magmaan::data::PolychoricHScoreKind;
using magmaan::data::PolychoricHScoreOptions;

struct Config {
  int n = 1200;
  int pair_reps = 140;
  int matrix_reps = 70;
  int polyserial_reps = 50;
  int sem_reps = 30;
  std::uint64_t seed = 20260516;
};

struct CompareSummary {
  std::string name;
  int         requested = 0;
  int         used = 0;
  int         failed = 0;
  double      rel_fro = std::numeric_limits<double>::quiet_NaN();
  double      diag_min = std::numeric_limits<double>::quiet_NaN();
  double      diag_median = std::numeric_limits<double>::quiet_NaN();
  double      diag_max = std::numeric_limits<double>::quiet_NaN();
  double      max_abs_corr_diff = std::numeric_limits<double>::quiet_NaN();
};

struct PolyserialFullSummary {
  int requested = 0;
  int used = 0;
  int failed = 0;
  double ml_abs_error = 0.0;
  double full_dpd_abs_error = 0.0;
};

Config parse_args(int argc, char** argv) {
  Config cfg;
  auto parse_int = [](std::string_view s, int fallback) {
    int out = fallback;
    const char* first = s.data();
    const char* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    if (ec != std::errc{} || ptr != last) out = fallback;
    return out;
  };
  auto parse_seed = [](std::string_view s, std::uint64_t fallback) {
    std::uint64_t out = fallback;
    const char* first = s.data();
    const char* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    if (ec != std::errc{} || ptr != last) out = fallback;
    return out;
  };

  for (int i = 1; i < argc; ++i) {
    const std::string_view a(argv[i]);
    auto value = [&](std::string_view key) -> std::string_view {
      return a.starts_with(key) ? a.substr(key.size()) : std::string_view{};
    };
    if (const auto v = value("--n="); !v.empty()) cfg.n = parse_int(v, cfg.n);
    if (const auto v = value("--pair-reps="); !v.empty()) {
      cfg.pair_reps = parse_int(v, cfg.pair_reps);
    }
    if (const auto v = value("--matrix-reps="); !v.empty()) {
      cfg.matrix_reps = parse_int(v, cfg.matrix_reps);
    }
    if (const auto v = value("--polyserial-reps="); !v.empty()) {
      cfg.polyserial_reps = parse_int(v, cfg.polyserial_reps);
    }
    if (const auto v = value("--mixed-reps="); !v.empty()) {
      cfg.polyserial_reps = parse_int(v, cfg.polyserial_reps);
    }
    if (const auto v = value("--sem-reps="); !v.empty()) {
      cfg.sem_reps = parse_int(v, cfg.sem_reps);
    }
    if (const auto v = value("--seed="); !v.empty()) {
      cfg.seed = parse_seed(v, cfg.seed);
    }
  }
  return cfg;
}

int ordinal_level(double z, const Eigen::VectorXd& thresholds) {
  for (Eigen::Index k = 0; k < thresholds.size(); ++k) {
    if (z <= thresholds(k)) return static_cast<int>(k) + 1;
  }
  return static_cast<int>(thresholds.size()) + 1;
}

double normal_quantile(double p) noexcept {
  constexpr double a1 = -3.969683028665376e+01;
  constexpr double a2 =  2.209460984245205e+02;
  constexpr double a3 = -2.759285104469687e+02;
  constexpr double a4 =  1.383577518672690e+02;
  constexpr double a5 = -3.066479806614716e+01;
  constexpr double a6 =  2.506628277459239e+00;
  constexpr double b1 = -5.447609879822406e+01;
  constexpr double b2 =  1.615858368580409e+02;
  constexpr double b3 = -1.556989798598866e+02;
  constexpr double b4 =  6.680131188771972e+01;
  constexpr double b5 = -1.328068155288572e+01;
  constexpr double c1 = -7.784894002430293e-03;
  constexpr double c2 = -3.223964580411365e-01;
  constexpr double c3 = -2.400758277161838e+00;
  constexpr double c4 = -2.549732539343734e+00;
  constexpr double c5 =  4.374664141464968e+00;
  constexpr double c6 =  2.938163982698783e+00;
  constexpr double d1 =  7.784695709041462e-03;
  constexpr double d2 =  3.224671290700398e-01;
  constexpr double d3 =  2.445134137142996e+00;
  constexpr double d4 =  3.754408661907416e+00;
  constexpr double plow = 0.02425;
  constexpr double phigh = 1.0 - plow;
  p = std::clamp(p, 1e-12, 1.0 - 1e-12);
  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

Eigen::MatrixXd covariance_sqrt_n(const std::vector<Eigen::VectorXd>& values,
                                  double n) {
  const Eigen::Index d = values.front().size();
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
  for (const auto& v : values) mean += v;
  mean /= static_cast<double>(values.size());

  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(d, d);
  for (const auto& v : values) {
    const Eigen::VectorXd centered = std::sqrt(n) * (v - mean);
    cov.noalias() += centered * centered.transpose();
  }
  cov /= static_cast<double>(values.size() - 1);
  return 0.5 * (cov + cov.transpose());
}

Eigen::MatrixXd mean_matrix(const std::vector<Eigen::MatrixXd>& mats) {
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(mats.front().rows(),
                                              mats.front().cols());
  for (const auto& m : mats) out += m;
  out /= static_cast<double>(mats.size());
  return 0.5 * (out + out.transpose());
}

double max_abs_corr_diff(Eigen::MatrixXd emp, Eigen::MatrixXd theory) {
  const Eigen::Index d = emp.rows();
  double max_diff = 0.0;
  for (Eigen::Index i = 0; i < d; ++i) {
    for (Eigen::Index j = 0; j < i; ++j) {
      const double ei = emp(i, i);
      const double ej = emp(j, j);
      const double ti = theory(i, i);
      const double tj = theory(j, j);
      if (ei <= 0.0 || ej <= 0.0 || ti <= 0.0 || tj <= 0.0) continue;
      const double ce = emp(i, j) / std::sqrt(ei * ej);
      const double ct = theory(i, j) / std::sqrt(ti * tj);
      max_diff = std::max(max_diff, std::abs(ce - ct));
    }
  }
  return max_diff;
}

CompareSummary compare_covariances(std::string name,
                                    int requested,
                                    int failed,
                                    const std::vector<Eigen::VectorXd>& values,
                                    const std::vector<Eigen::MatrixXd>& theories,
                                    double n) {
  CompareSummary s;
  s.name = std::move(name);
  s.requested = requested;
  s.used = static_cast<int>(values.size());
  s.failed = failed;
  if (values.size() < 3 || theories.empty()) return s;

  const Eigen::MatrixXd emp = covariance_sqrt_n(values, n);
  const Eigen::MatrixXd theory = mean_matrix(theories);
  s.rel_fro = (emp - theory).norm() / std::max(1e-12, theory.norm());
  s.max_abs_corr_diff = max_abs_corr_diff(emp, theory);

  std::vector<double> ratios;
  ratios.reserve(static_cast<std::size_t>(emp.rows()));
  for (Eigen::Index k = 0; k < emp.rows(); ++k) {
    const double denom = theory(k, k);
    if (std::isfinite(denom) && std::abs(denom) > 1e-12) {
      ratios.push_back(emp(k, k) / denom);
    }
  }
  if (!ratios.empty()) {
    std::sort(ratios.begin(), ratios.end());
    s.diag_min = ratios.front();
    s.diag_median = ratios[ratios.size() / 2];
    s.diag_max = ratios.back();
  }
  return s;
}

void print_summary(const CompareSummary& s) {
  std::cout << std::left << std::setw(31) << s.name
            << " used=" << std::setw(4) << s.used
            << " fail=" << std::setw(3) << s.failed
            << " relF=" << std::setw(8) << std::setprecision(3) << s.rel_fro
            << " diag[min/med/max]=" << std::setprecision(3)
            << s.diag_min << "/" << s.diag_median << "/" << s.diag_max
            << " max|corr diff|=" << s.max_abs_corr_diff << "\n";
}

void print_summary(const PolyserialFullSummary& s) {
  std::cout << std::left << std::setw(31) << "polyserial full DPD"
            << " used=" << std::setw(4) << s.used
            << " fail=" << std::setw(3) << s.failed
            << " |rho-true| ml/full=" << std::setprecision(3)
            << s.ml_abs_error << "/" << s.full_dpd_abs_error << "\n";
}

PairwiseOrdinalHWeightedStatsOptions robust_options() {
  PairwiseOrdinalHWeightedStatsOptions opts;
  opts.rho.h_score = PolychoricHScoreOptions{
      .kind = PolychoricHScoreKind::WmaHardCap,
      .k = 1.30};
  opts.max_iter = 18;
  opts.gtol = 8e-5;
  return opts;
}

Eigen::MatrixXd simulate_pair_counts(int n,
                                     const Eigen::VectorXd& th_i,
                                     const Eigen::VectorXd& th_j,
                                     double rho,
                                     double contam,
                                     std::mt19937_64& rng) {
  const int li = static_cast<int>(th_i.size()) + 1;
  const int lj = static_cast<int>(th_j.size()) + 1;
  Eigen::MatrixXd counts = Eigen::MatrixXd::Zero(li, lj);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const double s = std::sqrt(std::max(0.0, 1.0 - rho * rho));
  for (int row = 0; row < n; ++row) {
    int xi = 0;
    int xj = 0;
    if (uniform(rng) < contam) {
      const bool flip = (row % 2) == 0;
      xi = flip ? 1 : li;
      xj = flip ? lj : 1;
    } else {
      const double z1 = normal(rng);
      const double z2 = rho * z1 + s * normal(rng);
      xi = ordinal_level(z1, th_i);
      xj = ordinal_level(z2, th_j);
    }
    counts(xi - 1, xj - 1) += 1.0;
  }
  return counts;
}

CompareSummary run_pair_check(const Config& cfg, std::mt19937_64& rng) {
  const Eigen::Vector3d th_i(-0.85, -0.05, 0.95);
  const Eigen::Vector3d th_j(-1.10, 0.15, 0.80);
  const double rho = 0.60;
  const double contam = 0.04;
  const auto opts = robust_options();

  std::vector<Eigen::VectorXd> estimates;
  std::vector<Eigen::MatrixXd> gammas;
  int failed = 0;
  for (int r = 0; r < cfg.pair_reps; ++r) {
    const Eigen::MatrixXd counts =
        simulate_pair_counts(cfg.n, th_i, th_j, rho, contam, rng);
    auto fit = magmaan::data::fit_ordinal_pair_joint_h_weighted(
        counts, magmaan::data::OrdinalPairJointHWeightedOptions{
                    .h_score = opts.rho.h_score});
    if (!fit.has_value() || !fit->converged) {
      ++failed;
      continue;
    }
    auto infl = magmaan::data::ordinal_pair_h_weighted_influence(
        counts, fit->thresholds_i, fit->thresholds_j, fit->rho,
        magmaan::data::OrdinalPairHWeightedInfluenceOptions{
            .h_score = opts.rho.h_score});
    if (!infl.has_value() || !infl->gamma.allFinite()) {
      ++failed;
      continue;
    }
    Eigen::VectorXd theta(fit->thresholds_i.size() + fit->thresholds_j.size() +
                          1);
    theta << fit->thresholds_i, fit->thresholds_j, fit->rho;
    estimates.push_back(std::move(theta));
    gammas.push_back(infl->gamma);
  }
  return compare_covariances("pair-local robust polychoric",
                             cfg.pair_reps, failed, estimates, gammas, cfg.n);
}

Eigen::MatrixXd hs_latent_correlation() {
  Eigen::MatrixXd phi(3, 3);
  phi << 1.00, 0.45, 0.35,
         0.45, 1.00, 0.30,
         0.35, 0.30, 1.00;
  Eigen::MatrixXd lambda = Eigen::MatrixXd::Zero(9, 3);
  lambda(0, 0) = 0.75;
  lambda(1, 0) = 0.70;
  lambda(2, 0) = 0.65;
  lambda(3, 1) = 0.82;
  lambda(4, 1) = 0.76;
  lambda(5, 1) = 0.70;
  lambda(6, 2) = 0.66;
  lambda(7, 2) = 0.61;
  lambda(8, 2) = 0.56;

  Eigen::MatrixXd sigma = lambda * phi * lambda.transpose();
  for (Eigen::Index i = 0; i < sigma.rows(); ++i) {
    const double communality = sigma(i, i);
    sigma(i, i) += std::max(0.15, 1.0 - communality);
  }
  for (Eigen::Index i = 0; i < sigma.rows(); ++i) {
    const double si = std::sqrt(sigma(i, i));
    for (Eigen::Index j = 0; j < sigma.cols(); ++j) {
      const double sj = std::sqrt(sigma(j, j));
      sigma(i, j) /= si * sj;
    }
  }
  return sigma;
}

std::vector<Eigen::Vector3d> hs_thresholds() {
  return {
      {-1.00, -0.15, 0.85}, {-0.80, 0.05, 1.05}, {-0.70, 0.20, 1.10},
      {-1.10, -0.25, 0.75}, {-0.90, 0.10, 0.95}, {-0.65, 0.25, 1.20},
      {-1.20, -0.35, 0.70}, {-0.95, -0.05, 1.00}, {-0.75, 0.15, 1.15}};
}

Eigen::MatrixXd simulate_hs_ordinal(int n, double contam, std::mt19937_64& rng) {
  const Eigen::MatrixXd sigma = hs_latent_correlation();
  const Eigen::LLT<Eigen::MatrixXd> llt(sigma);
  const Eigen::MatrixXd L = llt.matrixL();
  const auto thresholds = hs_thresholds();

  Eigen::MatrixXd X(n, 9);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (int row = 0; row < n; ++row) {
    Eigen::VectorXd z(9);
    for (Eigen::Index j = 0; j < z.size(); ++j) z(j) = normal(rng);
    z = L * z;
    for (Eigen::Index j = 0; j < 9; ++j) {
      X(row, j) = ordinal_level(z(j), thresholds[static_cast<std::size_t>(j)]);
    }
    if (uniform(rng) < contam) {
      if ((row % 2) == 0) {
        X(row, 0) = 4.0;
        X(row, 4) = 1.0;
        X(row, 8) = 4.0;
      } else {
        X(row, 0) = 1.0;
        X(row, 4) = 4.0;
        X(row, 8) = 1.0;
      }
    }
  }
  return X;
}

Eigen::VectorXd threshold_starts(const Eigen::VectorXi& cat) {
  const int levels = cat.maxCoeff() + 1;
  Eigen::VectorXd th(levels - 1);
  for (int k = 0; k < levels - 1; ++k) {
    int count = 0;
    for (Eigen::Index r = 0; r < cat.size(); ++r) {
      if (cat(r) <= k) ++count;
    }
    th(k) = normal_quantile(static_cast<double>(count) /
                            static_cast<double>(cat.size()));
  }
  return th;
}

PolyserialFullSummary run_polyserial_full_check(const Config& cfg,
                                                std::mt19937_64& rng) {
  const int reps = std::max(1, std::min(cfg.polyserial_reps, 18));
  constexpr double rho_true = 0.60;
  constexpr double alpha = 0.5;
  Eigen::VectorXd th_true(2);
  th_true << -0.45, 0.65;
  std::normal_distribution<double> norm(0.0, 1.0);
  PolyserialFullSummary s;
  s.requested = reps;
  for (int rep = 0; rep < reps; ++rep) {
    Eigen::VectorXi cat(cfg.n);
    Eigen::VectorXd x(cfg.n);
    const double sd = std::sqrt(1.0 - rho_true * rho_true);
    for (Eigen::Index r = 0; r < cfg.n; ++r) {
      const double u = norm(rng);
      const double y = rho_true * u + sd * norm(rng);
      x(r) = u;
      cat(r) = ordinal_level(y, th_true) - 1;
    }
    for (Eigen::Index r = cfg.n - cfg.n / 25; r < cfg.n; ++r) {
      x(r) = 5.5 + 0.01 * static_cast<double>(r - (cfg.n - cfg.n / 25));
      cat(r) = 0;
    }

    const double mu = x.mean();
    double var = 0.0;
    for (Eigen::Index r = 0; r < x.size(); ++r) {
      const double d = x(r) - mu;
      var += d * d;
    }
    var /= static_cast<double>(x.size());
    const Eigen::VectorXd u = (x.array() - mu) / std::sqrt(var);
    const Eigen::VectorXd th = threshold_starts(cat);
    auto ml = magmaan::data::fit_polyserial_pair_rho_ml(cat, u, th);
    auto full = magmaan::data::fit_polyserial_pair_joint_dpd(
        cat, x, magmaan::data::PolyserialPairJointDpdOptions{.alpha = alpha});
    if (!ml.has_value() || !full.has_value()) {
      ++s.failed;
      continue;
    }
    if (!std::isfinite(full->rho)) {
      ++s.failed;
      continue;
    }
    ++s.used;
    s.ml_abs_error += std::abs(ml->rho - rho_true);
    s.full_dpd_abs_error += std::abs(full->rho - rho_true);
  }
  if (s.used > 0) {
    const double inv = 1.0 / static_cast<double>(s.used);
    s.ml_abs_error *= inv;
    s.full_dpd_abs_error *= inv;
  }
  return s;
}

Eigen::VectorXd ordinal_moment_vector(const magmaan::data::OrdinalStats& stats) {
  const Eigen::MatrixXd& R = stats.R[0];
  const Eigen::VectorXd& th = stats.thresholds[0];
  const Eigen::Index p = R.rows();
  Eigen::VectorXd out(th.size() + p * (p - 1) / 2);
  out.head(th.size()) = th;
  Eigen::Index k = th.size();
  for (Eigen::Index j = 0; j < p; ++j) {
    for (Eigen::Index i = j + 1; i < p; ++i) {
      out(k++) = R(i, j);
    }
  }
  return out;
}

CompareSummary run_matrix_check(const Config& cfg, std::mt19937_64& rng) {
  const auto opts = robust_options();
  std::vector<Eigen::VectorXd> moments;
  std::vector<Eigen::MatrixXd> gammas;
  int failed = 0;
  for (int r = 0; r < cfg.matrix_reps; ++r) {
    const Eigen::MatrixXd X = simulate_hs_ordinal(cfg.n, 0.035, rng);
    auto stats = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
        {X}, opts);
    if (!stats.has_value() || !stats->stats.NACOV[0].allFinite()) {
      ++failed;
      continue;
    }
    moments.push_back(ordinal_moment_vector(stats->stats));
    gammas.push_back(stats->stats.NACOV[0]);
  }
  return compare_covariances("HS robust ordinal moments",
                             cfg.matrix_reps, failed, moments, gammas, cfg.n);
}

std::string hs_ordinal_model_syntax() {
  return
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9\n"
      "x1 | t11 + t12 + t13\n"
      "x2 | t21 + t22 + t23\n"
      "x3 | t31 + t32 + t33\n"
      "x4 | t41 + t42 + t43\n"
      "x5 | t51 + t52 + t53\n"
      "x6 | t61 + t62 + t63\n"
      "x7 | t71 + t72 + t73\n"
      "x8 | t81 + t82 + t83\n"
      "x9 | t91 + t92 + t93\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n"
      "x5 ~*~ 1*x5\n"
      "x6 ~*~ 1*x6\n"
      "x7 ~*~ 1*x7\n"
      "x8 ~*~ 1*x8\n"
      "x9 ~*~ 1*x9\n";
}

CompareSummary run_sem_check(const Config& cfg, std::mt19937_64& rng) {
  const auto parsed = magmaan::parse::Parser::parse(hs_ordinal_model_syntax());
  if (!parsed.has_value()) {
    std::cout << "SEM setup failed: parse: " << parsed.error().detail << "\n";
    CompareSummary s;
    s.name = "HS ordinal CFA parameters";
    s.requested = cfg.sem_reps;
    s.failed = cfg.sem_reps;
    return s;
  }
  auto pt = magmaan::spec::lavaanify(*parsed);
  if (!pt.has_value()) {
    std::cout << "SEM setup failed: lavaanify: " << pt.error().detail << "\n";
    CompareSummary s;
    s.name = "HS ordinal CFA parameters";
    s.requested = cfg.sem_reps;
    s.failed = cfg.sem_reps;
    return s;
  }
  auto rep = magmaan::model::build_matrix_rep(*pt);
  if (!rep.has_value()) {
    std::cout << "SEM setup failed: matrix_rep: " << rep.error().detail << "\n";
    CompareSummary s;
    s.name = "HS ordinal CFA parameters";
    s.requested = cfg.sem_reps;
    s.failed = cfg.sem_reps;
    return s;
  }

  const auto opts = robust_options();
  std::vector<Eigen::VectorXd> estimates;
  std::vector<Eigen::MatrixXd> vcovs_n;
  int failed = 0;
  for (int r = 0; r < cfg.sem_reps; ++r) {
    const Eigen::MatrixXd X = simulate_hs_ordinal(cfg.n, 0.035, rng);
    auto stats = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
        {X}, opts);
    if (!stats.has_value()) {
      ++failed;
      continue;
    }
    auto x0 = magmaan::estimate::ordinal_start_values(*pt, *rep, stats->stats);
    if (!x0.has_value()) {
      ++failed;
      continue;
    }
    auto est = magmaan::estimate::fit_ordinal_bounded(
        *pt, *rep, stats->stats, magmaan::estimate::Bounds{},
        magmaan::estimate::OrdinalWeightKind::DWLS, *x0);
    if (!est.has_value() || !est->theta.allFinite()) {
      ++failed;
      continue;
    }
    auto rob = magmaan::estimate::robust_ordinal(
        *pt, *rep, stats->stats, *est,
        magmaan::estimate::OrdinalWeightKind::DWLS);
    if (!rob.has_value() || !rob->vcov.allFinite()) {
      ++failed;
      continue;
    }
    estimates.push_back(est->theta);
    vcovs_n.push_back(rob->vcov * static_cast<double>(cfg.n));
  }
  return compare_covariances("HS ordinal CFA parameters",
                             cfg.sem_reps, failed, estimates, vcovs_n, cfg.n);
}

}  // namespace

int main(int argc, char** argv) {
  const Config cfg = parse_args(argc, argv);
  std::mt19937_64 rng(cfg.seed);

  std::cout << "robust polychoric sanity checks\n"
            << "seed=" << cfg.seed << " n=" << cfg.n
            << " pair_reps=" << cfg.pair_reps
            << " matrix_reps=" << cfg.matrix_reps
            << " polyserial_reps=" << cfg.polyserial_reps
            << " sem_reps=" << cfg.sem_reps
            << " h=wma_hard_cap(k=1.30)"
            << " polyserial_dpd_alpha=0.5\n\n";

  std::cout << "diagnostic columns: empirical cov of sqrt(N)*estimate vs "
               "average theoretical Gamma / N-scaled vcov\n";
  print_summary(run_pair_check(cfg, rng));
  print_summary(run_matrix_check(cfg, rng));
  print_summary(run_polyserial_full_check(cfg, rng));
  print_summary(run_sem_check(cfg, rng));

  std::cout << "\nInterpretation: diag ratios near 1 and modest relative "
               "Frobenius error are the desired Monte Carlo pattern. "
               "The HS checks are contaminated simulations. The polyserial "
               "line exercises the package pair-level full DPD primitive; "
               "SEM-level mixed DPD integration is intentionally absent until "
               "a shared-threshold mixed moment/Gamma contract is implemented.\n";
  return 0;
}
