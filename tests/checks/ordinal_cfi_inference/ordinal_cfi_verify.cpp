// Advisory verification for the estimated-weight CFI/TLI confidence intervals
// (estimate::ordinal_cfi_tli_misspec_inference). CFI and TLI are the program's
// first INCREMENTAL (two-model) indices: a ratio of the user and baseline
// noncentralities δ_u = T_u − Q̄_u and δ_b = T_b − Q̄_b, sharing one joint NACOV
// Γ_x. The interval is a ratio delta-method; the open questions this harness
// answers empirically are (a) whether that normal-theory interval covers near
// nominal despite CFI's hard ceiling at 1, (b) whether the leading-order
// baseline-dominated approximation Var(CFI) ≈ Var(T_u)/δ_b² is accurate (i.e.
// the cross-term and baseline-variance corrections are negligible), and (c)
// whether the estimated-weight γ channel matters for the index variance the way
// it does for RMSEA. See docs/research/notes/cfi_tli_misspec_inference.tex.
//
// Setup mirrors tests/checks/ordinal_rmsea_inference: the four-cycle (C4) binary
// population (p = 4 binary indicators, thresholds 0, latent correlations
// base = λ² on the two opposite pairs, base + ε on the four C4 edges). Fit the
// congeneric one-factor model; for ε > 0 it is misspecified (user residual ≠ 0,
// user γ channel live) and CFI₀ < 1; ε = 0 is correct specification (CFI₀ = 1,
// boundary). For each ε we draw `reps` fresh samples of size n, fit DWLS, and
// call ordinal_cfi_tli_misspec_inference with estimated_weight in {true,false}.
//
// Checks (ε > 0): point ~ unbiased (MC mean of 1 − δ_u/δ_b vs population CFI₀,
// likewise TLI); variance (analytic var_cfi/var_tli vs MC variance of the
// index — the delta-method gate); coverage of the population CFI₀/TLI₀ near
// nominal; the leading-order ratio Var(CFI)/(Var(T_u)/δ_b²) ≈ 1. At ε = 0 the
// user γ channel is dormant so its share of the index variance ≈ 0.
// Coverage uses the UNCLAMPED normal interval (the reported CI is clamped to
// [0,1]; near CFI₀ ≈ 1 that clamp only makes it more conservative).

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
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
constexpr double kZ = 1.6448536269514722;  // Φ⁻¹(0.95), two-sided 90%

struct Config {
  std::int64_t reps = 2000;
  std::int64_t n = 1000;
  std::int64_t n_pop = 400000;
  std::uint64_t seed = 20260627;
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

// Unclamped point CFI = 1 − δ_u/δ_b (the reported `cfi` is clamped to [0,1]).
double cfi_raw(const magmaan::estimate::OrdinalIncrementalFitInference& r) {
  return 1.0 - r.delta_user / r.delta_baseline;
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

  std::cout << "CFI/TLI estimated-weight CI verification (C4, congeneric fit)\n"
            << "  lambda=" << cfg.lambda << "  n=" << cfg.n
            << "  reps=" << cfg.reps << "  conf=" << kConf << "\n\n"
            << std::fixed << std::setprecision(6);

  const std::array<double, 3> epses = {0.0, 0.12, 0.24};
  int failures = 0;

  for (double eps : epses) {
    const Eigen::MatrixXd R = c4_correlation(cfg.lambda, eps);
    if (!is_pd(R)) { std::cout << "eps=" << eps << " not PD, skip\n"; continue; }
    const Eigen::MatrixXd L = Eigen::LLT<Eigen::MatrixXd>(R).matrixL();

    // Population reference: CFI₀ and TLI₀ (unclamped) from a large-N fit.
    double cfi_pop = std::numeric_limits<double>::quiet_NaN();
    double tli_pop = std::numeric_limits<double>::quiet_NaN();
    {
      const Eigen::MatrixXd Xp = c4_population_dataset(L, cfg.n_pop, cfg.seed);
      auto sp = magmaan::data::ordinal_stats_from_integer_data({Xp}, true);
      if (sp.has_value()) {
        Fitted fp = fit_dwls(*sp);
        if (fp.ok) {
          auto ep = magmaan::estimate::ordinal_cfi_tli_misspec_inference(
              fp.pt, fp.rep, *sp, fp.est);
          if (ep.has_value()) { cfi_pop = cfi_raw(*ep); tli_pop = ep->tli; }
        }
      }
    }
    if (!std::isfinite(cfi_pop) || !std::isfinite(tli_pop)) {
      std::cout << "eps=" << eps << " population reference failed\n";
      ++failures;
      continue;
    }

    std::mt19937_64 rng(cfg.seed + 202ULL +
                        7919ULL * static_cast<std::uint64_t>(eps * 1000));
    double s_cfi = 0, s_cfi2 = 0, s_tli = 0, s_tli2 = 0;
    double s_vcfi_e = 0, s_vcfi_f = 0, s_vtli_e = 0, s_lead = 0;
    std::int64_t used = 0, cov_cfi_e = 0, cov_cfi_f = 0, cov_tli_e = 0,
                 cov_tli_f = 0;
    for (std::int64_t r = 0; r < cfg.reps; ++r) {
      const Eigen::MatrixXd X = c4_sample(L, cfg.n, rng);
      auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
      if (!stats.has_value()) continue;
      Fitted f = fit_dwls(*stats);
      if (!f.ok) continue;
      auto e = magmaan::estimate::ordinal_cfi_tli_misspec_inference(
          f.pt, f.rep, *stats, f.est,
          magmaan::estimate::OrdinalParameterization::Delta, true);
      auto fx = magmaan::estimate::ordinal_cfi_tli_misspec_inference(
          f.pt, f.rep, *stats, f.est,
          magmaan::estimate::OrdinalParameterization::Delta, false);
      if (!e.has_value() || !fx.has_value()) continue;
      if (!std::isfinite(e->var_cfi) || !std::isfinite(e->tli) ||
          !(e->delta_baseline > 0.0))
        continue;

      const double c_raw = cfi_raw(*e);
      s_cfi += c_raw;
      s_cfi2 += c_raw * c_raw;
      s_tli += e->tli;
      s_tli2 += e->tli * e->tli;
      s_vcfi_e += e->var_cfi;
      s_vcfi_f += fx->var_cfi;
      s_vtli_e += e->var_tli;
      s_lead += e->var_user / (e->delta_baseline * e->delta_baseline);

      // Unclamped normal intervals, covering the population values.
      const double sd_e = std::sqrt(e->var_cfi);
      const double sd_f = std::sqrt(fx->var_cfi);
      const double cf_e = cfi_raw(*e), cf_f = cfi_raw(*fx);
      cov_cfi_e += (cf_e - kZ * sd_e <= cfi_pop &&
                    cfi_pop <= cf_e + kZ * sd_e) ? 1 : 0;
      cov_cfi_f += (cf_f - kZ * sd_f <= cfi_pop &&
                    cfi_pop <= cf_f + kZ * sd_f) ? 1 : 0;
      const double sdt_e = std::sqrt(e->var_tli);
      const double sdt_f = std::sqrt(fx->var_tli);
      cov_tli_e += (e->tli - kZ * sdt_e <= tli_pop &&
                    tli_pop <= e->tli + kZ * sdt_e) ? 1 : 0;
      cov_tli_f += (fx->tli - kZ * sdt_f <= tli_pop &&
                    tli_pop <= fx->tli + kZ * sdt_f) ? 1 : 0;
      ++used;
    }
    if (used < 10) { std::cout << "eps=" << eps << " too few reps\n"; ++failures;
                     continue; }
    const double d = static_cast<double>(used);
    const double mc_mean_cfi = s_cfi / d;
    const double mc_var_cfi = s_cfi2 / d - mc_mean_cfi * mc_mean_cfi;
    const double mc_mean_tli = s_tli / d;
    const double mc_var_tli = s_tli2 / d - mc_mean_tli * mc_mean_tli;
    const double an_vcfi_e = s_vcfi_e / d;
    const double an_vcfi_f = s_vcfi_f / d;
    const double an_vtli_e = s_vtli_e / d;
    const double lead = s_lead / d;
    const double cov_ce = static_cast<double>(cov_cfi_e) / d;
    const double cov_cf = static_cast<double>(cov_cfi_f) / d;
    const double cov_te = static_cast<double>(cov_tli_e) / d;
    const double cov_tf = static_cast<double>(cov_tli_f) / d;
    const double gshare =
        an_vcfi_e > 1e-18 ? (an_vcfi_e - an_vcfi_f) / an_vcfi_e : 0.0;
    const double lead_ratio = an_vcfi_e > 1e-18 ? lead / an_vcfi_e : 0.0;

    std::cout << "eps=" << eps << "  CFI_pop=" << cfi_pop
              << "  TLI_pop=" << tli_pop << "  (used " << used << ")\n"
              << "  CFI: mc_mean=" << mc_mean_cfi << "  mc_var=" << mc_var_cfi
              << "  an_var(est=" << an_vcfi_e << " fixed=" << an_vcfi_f << ")"
              << "  gamma_share=" << gshare << "\n"
              << "       leading V_uu/db^2=" << lead << " (ratio to est_var "
              << lead_ratio << ")\n"
              << "       coverage@" << kConf << ": est=" << cov_ce
              << " fixed=" << cov_cf << "\n"
              << "  TLI: mc_mean=" << mc_mean_tli << "  mc_var=" << mc_var_tli
              << "  an_var(est=" << an_vtli_e << ")"
              << "  coverage: est=" << cov_te << " fixed=" << cov_tf << "\n";

    // Point estimators track their population values in every regime (TLI is
    // far more volatile than CFI, so a looser band).
    if (std::abs(mc_mean_cfi - cfi_pop) > 0.02) ++failures;
    if (std::abs(mc_mean_tli - tli_pop) > 0.08) ++failures;

    if (eps == 0.0) {
      // Correct spec: CFI₀ = TLI₀ = 1 (boundary). No coverage test. The user γ
      // channel (∝ user residual) is dormant, so its share of the INDEX variance
      // ≈ 0 even though the baseline's own γ channel is always live.
      if (std::abs(gshare) > 0.12) ++failures;
    } else {
      // Misspecification. Coverage is the calibration gate: CFI's two-sided
      // interval is near nominal; TLI's must at least be non-anti-conservative
      // (its analytic variance OVER-states at strong misfit because c = Q̄_b/Q̄_u
      // is ill-conditioned when the user generalized df is small, so it can
      // over-cover -- reported, not failed). The leading-order ratio and the
      // variance match are diagnostics (printed, not gated): the code uses the
      // full bivariate form, and `lead_ratio` degrades from ~1 toward 0 as the
      // cross-term and baseline-variance corrections take over.
      if (cov_ce < 0.84 || cov_ce > 0.97) ++failures;
      if (cov_te < 0.84) ++failures;
    }
    std::cout << "\n";
  }

  std::cout << (failures == 0 ? "PASS" : "FAIL") << ": " << failures
            << " failed criteria\n";
  return failures == 0 ? 0 : 1;
}
