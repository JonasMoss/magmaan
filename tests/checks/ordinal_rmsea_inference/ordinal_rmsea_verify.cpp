// Advisory verification for the estimated-weight RMSEA confidence interval
// (estimate::ordinal_rmsea_misspec_inference). RMSEA's metric IS the estimated
// weight, so its γ channel is large. Empirically the channel is
// variance-REDUCING (r and γ co-vary negatively): the estimated-weight variance
// matches Monte-Carlo and gives a calibrated CI, while the fixed-weight variance
// is much larger, so the fixed-weight CI is conservative (over-covers / too
// wide). Accounting for the estimated weight tightens the interval to nominal.
//
// Setup: the four-cycle (C4) binary population (p = 4 binary indicators,
// thresholds 0, latent correlations base = λ² on the two opposite pairs and
// base + ε on the four C4 edges). Fit the congeneric one-factor model; for ε > 0
// it is misspecified so the residual is nonzero and the γ channel is live;
// ε = 0 is correct specification. For each ε we draw `reps` fresh samples of
// size n, fit DWLS, and call ordinal_rmsea_misspec_inference with
// estimated_weight in {true,false}. Checks: bias (analytic E[N·F] vs MC mean),
// variance (analytic N·grad_var vs MC variance — the γ magnitude/sign gate),
// coverage (estimated covers near nominal; fixed under-covers under misspec),
// and ε=0 dormancy.

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

constexpr int p = 4;
constexpr double kConf = 0.90;

struct Config {
  std::int64_t reps = 2000;
  std::int64_t n = 1000;
  std::int64_t n_pop = 400000;
  std::uint64_t seed = 20260624;
  double lambda = 0.70;
};

Eigen::MatrixXd c4_correlation(double lambda, double eps) {
  const double base = lambda * lambda;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
  auto edge = [](int i, int j) {
    return (i == 0 && j == 1) || (i == 1 && j == 2) || (i == 2 && j == 3) ||
           (i == 0 && j == 3);
  };
  for (int i = 0; i < p; ++i)
    for (int j = i + 1; j < p; ++j) {
      const double v = base + (edge(i, j) ? eps : 0.0);
      R(i, j) = v;
      R(j, i) = v;
    }
  return R;
}

bool is_pd(const Eigen::MatrixXd& R) {
  return Eigen::LLT<Eigen::MatrixXd>(R).info() == Eigen::Success;
}

Eigen::MatrixXd c4_sample(const Eigen::MatrixXd& L, std::int64_t n,
                          std::mt19937_64& rng) {
  std::normal_distribution<double> z(0.0, 1.0);
  Eigen::MatrixXd X(n, p);
  for (std::int64_t i = 0; i < n; ++i) {
    Eigen::VectorXd e(p);
    for (int j = 0; j < p; ++j) e(j) = z(rng);
    const Eigen::VectorXd y = L * e;
    for (int j = 0; j < p; ++j) X(i, j) = 1.0 + static_cast<double>(y(j) > 0.0);
  }
  return X;
}

Eigen::MatrixXd c4_population_dataset(const Eigen::MatrixXd& L,
                                     std::int64_t n_pop, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::array<std::int64_t, 16> count{};
  const std::int64_t n_tally = 4'000'000;
  for (std::int64_t s = 0; s < n_tally; ++s) {
    Eigen::VectorXd e(p);
    for (int j = 0; j < p; ++j) e(j) = z(rng);
    const Eigen::VectorXd y = L * e;
    int pat = 0;
    for (int j = 0; j < p; ++j)
      if (y(j) > 0.0) pat |= (1 << j);
    ++count[static_cast<std::size_t>(pat)];
  }
  std::int64_t total = 0;
  std::array<std::int64_t, 16> rep{};
  for (int a = 0; a < 16; ++a) {
    rep[static_cast<std::size_t>(a)] = std::llround(
        static_cast<double>(count[static_cast<std::size_t>(a)]) /
        static_cast<double>(n_tally) * static_cast<double>(n_pop));
    total += rep[static_cast<std::size_t>(a)];
  }
  Eigen::MatrixXd X(total, p);
  Eigen::Index row = 0;
  for (int a = 0; a < 16; ++a)
    for (std::int64_t k = 0; k < rep[static_cast<std::size_t>(a)]; ++k) {
      for (int j = 0; j < p; ++j)
        X(row, j) = 1.0 + static_cast<double>((a >> j) & 1);
      ++row;
    }
  return X;
}

std::string syntax_congeneric() {
  return std::string("f =~ x1 + x2 + x3 + x4\n") +
         "x1 | t1\nx2 | t1\nx3 | t1\nx4 | t1\n"
         "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
}

struct Fitted {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
  magmaan::estimate::Estimates est;
  bool ok = false;
};

Fitted fit_dwls(const magmaan::data::OrdinalStats& stats) {
  Fitted out;
  auto fp = magmaan::parse::Parser::parse(syntax_congeneric());
  if (!fp.has_value()) return out;
  auto pt = magmaan::spec::build(*fp);
  if (!pt.has_value()) return out;
  auto mr = magmaan::model::build_matrix_rep(*pt);
  if (!mr.has_value()) return out;
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, stats, {});
  if (!x0.has_value()) return out;
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 2000;
  opts.ftol = 1e-12;
  opts.gtol = 1e-9;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  if (!fit.has_value()) return out;
  out.pt = std::move(*pt);
  out.rep = std::move(*mr);
  out.est = std::move(*fit);
  out.ok = true;
  return out;
}

std::int64_t parse_i64(std::string_view v, std::int64_t fb) {
  std::int64_t out = fb;
  std::from_chars(v.data(), v.data() + v.size(), out);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto eat = [&](std::string_view k, auto& d) {
      if (a.substr(0, k.size()) == k) {
        d = static_cast<std::decay_t<decltype(d)>>(
            parse_i64(a.substr(k.size()), static_cast<std::int64_t>(d)));
        return true;
      }
      return false;
    };
    if (eat("--reps=", cfg.reps)) continue;
    if (eat("--n=", cfg.n)) continue;
    if (eat("--n-pop=", cfg.n_pop)) continue;
    if (eat("--seed=", cfg.seed)) continue;
  }

  std::cout << "RMSEA estimated-weight CI verification (C4, congeneric fit)\n"
            << "  lambda=" << cfg.lambda << "  n=" << cfg.n
            << "  reps=" << cfg.reps << "  conf=" << kConf << "\n\n"
            << std::fixed << std::setprecision(4);

  const std::array<double, 3> epses = {0.0, 0.12, 0.24};
  int failures = 0;

  for (double eps : epses) {
    const Eigen::MatrixXd R = c4_correlation(cfg.lambda, eps);
    if (!is_pd(R)) { std::cout << "eps=" << eps << " not PD, skip\n"; continue; }
    const Eigen::MatrixXd L = Eigen::LLT<Eigen::MatrixXd>(R).matrixL();

    // Population reference: RMSEA and the discrepancy F0.
    double rmsea_pop = std::numeric_limits<double>::quiet_NaN();
    double F_pop = std::numeric_limits<double>::quiet_NaN();
    {
      const Eigen::MatrixXd Xp = c4_population_dataset(L, cfg.n_pop, cfg.seed);
      auto sp = magmaan::data::ordinal_stats_from_integer_data({Xp}, true);
      if (sp.has_value()) {
        Fitted fp = fit_dwls(*sp);
        if (fp.ok) {
          auto ep = magmaan::estimate::ordinal_rmsea_misspec_inference(
              fp.pt, fp.rep, *sp, fp.est);
          if (ep.has_value()) { rmsea_pop = ep->point; F_pop = ep->fmin; }
        }
      }
    }
    if (!std::isfinite(rmsea_pop) || !std::isfinite(F_pop)) {
      std::cout << "eps=" << eps << " population reference failed\n";
      ++failures;
      continue;
    }
    const double NF0 = static_cast<double>(cfg.n) * F_pop;

    std::mt19937_64 rng(cfg.seed + 202ULL +
                        7919ULL * static_cast<std::uint64_t>(eps * 1000));
    double sum_stat = 0, sum_stat2 = 0, sum_bias = 0, sum_gv_e = 0, sum_gv_f = 0;
    std::int64_t used = 0, cov_e = 0, cov_f = 0;
    for (std::int64_t r = 0; r < cfg.reps; ++r) {
      const Eigen::MatrixXd X = c4_sample(L, cfg.n, rng);
      auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
      if (!stats.has_value()) continue;
      Fitted f = fit_dwls(*stats);
      if (!f.ok) continue;
      auto e = magmaan::estimate::ordinal_rmsea_misspec_inference(
          f.pt, f.rep, *stats, f.est,
          magmaan::estimate::OrdinalParameterization::Delta, true);
      auto fx = magmaan::estimate::ordinal_rmsea_misspec_inference(
          f.pt, f.rep, *stats, f.est,
          magmaan::estimate::OrdinalParameterization::Delta, false);
      if (!e.has_value() || !fx.has_value()) continue;
      if (!std::isfinite(e->stat) || !std::isfinite(e->grad_var)) continue;
      sum_stat += e->stat;
      sum_stat2 += e->stat * e->stat;
      sum_bias += e->bias_trace;
      sum_gv_e += e->grad_var;
      sum_gv_f += fx->grad_var;
      cov_e += (e->ci_lower <= rmsea_pop && rmsea_pop <= e->ci_upper) ? 1 : 0;
      cov_f += (fx->ci_lower <= rmsea_pop && rmsea_pop <= fx->ci_upper) ? 1 : 0;
      ++used;
    }
    if (used < 10) { std::cout << "eps=" << eps << " too few reps\n"; ++failures;
                     continue; }
    const double d = static_cast<double>(used);
    const double mc_mean = sum_stat / d;
    const double mc_var = sum_stat2 / d - mc_mean * mc_mean;
    const double an_mean = NF0 + sum_bias / d;
    const double an_var_e = static_cast<double>(cfg.n) * (sum_gv_e / d);
    const double an_var_f = static_cast<double>(cfg.n) * (sum_gv_f / d);
    const double cove = static_cast<double>(cov_e) / d;
    const double covf = static_cast<double>(cov_f) / d;
    const double gamma_share =
        an_var_e > 1e-12 ? (an_var_e - an_var_f) / an_var_e : 0.0;

    std::cout << "eps=" << eps << "  RMSEA_pop=" << rmsea_pop << "  (used "
              << used << ")\n"
              << "  mean N*F: mc=" << mc_mean << " analytic=" << an_mean << "\n"
              << "  var  N*F: mc=" << mc_var << " analytic_est=" << an_var_e
              << " analytic_fixed=" << an_var_f << "  (gamma share " << gamma_share
              << ")\n"
              << "  coverage@" << kConf << ": est=" << cove << " fixed=" << covf
              << "\n";

    // Bias matches MC in every regime.
    if (mc_mean > 1e-9 && std::abs(an_mean - mc_mean) / mc_mean > 0.08)
      ++failures;

    if (eps == 0.0) {
      // Correct spec: gamma channel dormant. No coverage test -- the population
      // RMSEA is 0 (boundary), so any non-negative interval covers it trivially.
      if (std::abs(gamma_share) > 0.05) ++failures;
    } else {
      // Misspecification: the estimated-weight normal-theory variance matches MC
      // (gamma magnitude + sign gate), the fixed-weight variance is FURTHER from
      // MC (the large gamma channel matters), the estimated CI is calibrated,
      // and the fixed-weight CI is conservative (over-covers).
      if (mc_var > 1e-9 && std::abs(an_var_e - mc_var) / mc_var > 0.25)
        ++failures;
      if (std::abs(an_var_f - mc_var) <= std::abs(an_var_e - mc_var)) ++failures;
      if (cove < 0.85 || cove > 0.96) ++failures;
      if (covf < cove) ++failures;  // fixed-weight over-covers (conservative)
    }
    std::cout << "\n";
  }

  std::cout << (failures == 0 ? "PASS" : "FAIL") << ": " << failures
            << " failed criteria\n";
  return failures == 0 ? 0 : 1;
}
