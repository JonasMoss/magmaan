// Experiment 36: does the estimated-weight profile law fix the nested
// categorical-DWLS test under fixed misspecification of the larger model?
//
// Thesis. For nested all-ordinal DWLS models the standard scaled difference
// test (Satorra 2000, what lavaan's lavTestLRT uses) reads its reference law
// off the FIXED-weight projector U = W - W D B^-1 D' W with W = diag(Gamma)^-1
// held constant. Under correct specification that is right. Under FIXED
// misspecification of the larger model the polychoric weight is estimated and
// its sampling influence enters the difference statistic, so the true reference
// law is the EXTENDED (u, gamma) profile law magmaan computes
// (estimate::ordinal_dwls_profile_lrt). The fixed-weight law omits that gamma
// channel and is therefore miscalibrated; the full profile law restores
// calibration. Ordinary chi^2 is the naive baseline (it ignores even the DWLS
// weight scaling, so it is miscalibrated already under correct spec).
//
// Design. The exact pseudo-null is the vertex-transitive four-cycle (C4): p = 4
// binary indicators, thresholds 0, latent-response correlations base = lambda^2
// on the two opposite pairs and base + eps on the four C4 edges. Symmetry forces
// the congeneric (H1) one-factor pseudo-truth to be tau-equivalent, so H0
// (loadings fixed equal) lies in H1's pseudo-truth: F_H0 = F_H1 at the
// population and the difference statistic has a central mixture law, with NO
// noncentrality, at every eps. eps tunes the misspecification of BOTH models
// (and hence the residual-driven gamma channel); eps = 0 is correct spec.
//
// For each (lambda, eps, n) we simulate `reps` fresh samples, fit H1 and H0 with
// DWLS, and for the difference statistic T compute three p-values:
//   p_chi2  = chi^2_df(T)                       (naive)
//   p_fixed = Pr(sum lambda_fixed_j chi^2_1 > T)  (fixed-weight, current practice)
//   p_full  = Pr(sum lambda_full_j  chi^2_1 > T)  (estimated-weight profile law)
// and report empirical rejection rates. A calibrated reference rejects at the
// nominal alpha; we expect p_full ~ alpha across eps while p_fixed drifts as the
// gamma channel grows, and p_chi2 is off even at eps = 0.

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/satorra2000.hpp"
#include "magmaan/robust/weighted_chisq.hpp"
#include "magmaan/spec/build.hpp"

namespace {

constexpr int p = 4;

struct Config {
  std::int64_t reps = 2000;
  std::int64_t n_pop = 400000;  // expanded population dataset for the reference
  std::uint64_t seed = 20260622;
  std::string out = "results/calibration.csv";
};

const std::array<double, 3> kAlpha = {0.01, 0.05, 0.10};

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
  Eigen::LLT<Eigen::MatrixXd> llt(R);
  return llt.info() == Eigen::Success;
}

// One fresh sample of n binary rows (codes 1/2) from latent N(0, R) at 0.
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

// Exact proportional integer dataset matching the C4 population cell shares.
Eigen::MatrixXd c4_population_dataset(const Eigen::MatrixXd& L,
                                      const Config& cfg, std::uint64_t seed) {
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
        static_cast<double>(n_tally) * static_cast<double>(cfg.n_pop));
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

const char* thresholds_syntax() {
  return "x1 | t1\nx2 | t1\nx3 | t1\nx4 | t1\n"
         "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
}
std::string syntax_h1() {
  return std::string("f =~ x1 + x2 + x3 + x4\n") + thresholds_syntax();
}
std::string syntax_h0() {  // tau-equivalent: loadings fixed equal to the marker
  return std::string("f =~ x1 + 1*x2 + 1*x3 + 1*x4\n") + thresholds_syntax();
}

struct Fitted {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
  magmaan::estimate::Estimates est;
  bool ok = false;
};

Fitted fit_dwls(const std::string& syntax,
                const magmaan::data::OrdinalStats& stats) {
  Fitted out;
  auto fp = magmaan::parse::Parser::parse(syntax);
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

// p-value of T under the fixed-weight (current-practice) nested DWLS law, read
// off the u-block of the profile contrast and the moment NACOV.
double p_fixed_weight(const magmaan::estimate::WeightedProfileLRTResult& lrt,
                      double T) {
  const Eigen::Index m = lrt.profile_hessian.rows() / 2;
  auto spec = magmaan::robust::compute_profile_contrast_spectrum(
      lrt.profile_hessian.topLeftCorner(m, m), lrt.gamma.topLeftCorner(m, m));
  if (!spec.has_value() || spec->eigenvalues.size() == 0)
    return std::numeric_limits<double>::quiet_NaN();
  return magmaan::robust::weighted_chisq_upper(spec->eigenvalues,
                                               std::max(0.0, T));
}

struct Cell {
  double lambda, eps;
  std::int64_t n;
  double pop_trace_fixed = std::numeric_limits<double>::quiet_NaN();
  double pop_trace_full = std::numeric_limits<double>::quiet_NaN();
  double mean_T = 0.0;
  std::int64_t used = 0;
  // rejections[ref][alpha], ref: 0=chi2, 1=fixed, 2=full
  std::array<std::array<std::int64_t, 3>, 3> rej{};
};

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
    if (a.substr(0, 6) == "--out=") { cfg.out = std::string(a.substr(6)); continue; }
    if (eat("--reps=", cfg.reps)) continue;
    if (eat("--n-pop=", cfg.n_pop)) continue;
    if (eat("--seed=", cfg.seed)) continue;
  }

  const std::array<double, 2> lambdas = {0.55, 0.70};
  const std::array<double, 4> epses = {0.0, 0.08, 0.16, 0.24};
  const std::array<std::int64_t, 2> ns = {500, 1500};

  std::cout << "Experiment 36: nested categorical-DWLS profile-LRT calibration\n"
            << "  reps=" << cfg.reps << "  seed=" << cfg.seed << "\n"
            << "  references: chi2 (naive) | fixed-weight (Satorra-2000) | "
               "full profile (estimated-weight)\n\n";
  std::cout << std::fixed << std::setprecision(3);

  std::vector<Cell> cells;
  std::uint64_t cell_idx = 0;
  for (double lambda : lambdas) {
    for (double eps : epses) {
      const Eigen::MatrixXd R = c4_correlation(lambda, eps);
      if (!is_pd(R)) { ++cell_idx; continue; }
      const Eigen::MatrixXd L =
          Eigen::LLT<Eigen::MatrixXd>(R).matrixL();

      // Population reference traces (one expanded fit per (lambda, eps)).
      double pop_tr_fixed = std::numeric_limits<double>::quiet_NaN();
      double pop_tr_full = std::numeric_limits<double>::quiet_NaN();
      {
        const Eigen::MatrixXd Xp =
            c4_population_dataset(L, cfg, cfg.seed + 7919 * cell_idx);
        auto sp = magmaan::data::ordinal_stats_from_integer_data({Xp}, true);
        if (sp.has_value()) {
          Fitted h1 = fit_dwls(syntax_h1(), *sp);
          Fitted h0 = fit_dwls(syntax_h0(), *sp);
          if (h1.ok && h0.ok) {
            auto lrt = magmaan::estimate::ordinal_dwls_profile_lrt(
                h1.pt, h1.rep, *sp, h1.est, h0.pt, h0.rep, h0.est);
            if (lrt.has_value()) {
              const Eigen::Index m = lrt->profile_hessian.rows() / 2;
              pop_tr_full = (lrt->profile_hessian * lrt->gamma).trace();
              pop_tr_fixed = (lrt->profile_hessian.topLeftCorner(m, m) *
                              lrt->gamma.topLeftCorner(m, m)).trace();
            }
          }
        }
      }

      for (std::int64_t n : ns) {
        Cell cell{lambda, eps, n};
        cell.pop_trace_fixed = pop_tr_fixed;
        cell.pop_trace_full = pop_tr_full;
        std::mt19937_64 rng(cfg.seed + 1000003ULL * cell_idx +
                            7ULL * static_cast<std::uint64_t>(n));
        for (std::int64_t r = 0; r < cfg.reps; ++r) {
          const Eigen::MatrixXd X = c4_sample(L, n, rng);
          auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
          if (!stats.has_value()) continue;
          Fitted h1 = fit_dwls(syntax_h1(), *stats);
          Fitted h0 = fit_dwls(syntax_h0(), *stats);
          if (!h1.ok || !h0.ok) continue;
          auto lrt = magmaan::estimate::ordinal_dwls_profile_lrt(
              h1.pt, h1.rep, *stats, h1.est, h0.pt, h0.rep, h0.est);
          if (!lrt.has_value()) continue;
          const double T = lrt->T_diff;
          if (!std::isfinite(T)) continue;
          const double pv[3] = {lrt->p_unscaled, p_fixed_weight(*lrt, T),
                                lrt->p_mixture};
          if (!std::isfinite(pv[1]) || !std::isfinite(pv[2])) continue;
          for (int ref = 0; ref < 3; ++ref)
            for (std::size_t ai = 0; ai < kAlpha.size(); ++ai)
              if (pv[ref] < kAlpha[ai]) ++cell.rej[static_cast<std::size_t>(ref)][ai];
          cell.mean_T += T;
          ++cell.used;
        }
        if (cell.used > 0) cell.mean_T /= static_cast<double>(cell.used);
        cells.push_back(cell);

        const double d = cell.used > 0 ? static_cast<double>(cell.used) : 1.0;
        std::cout << "lambda=" << lambda << " eps=" << eps << " n=" << n
                  << "  used=" << cell.used << "  meanT=" << cell.mean_T
                  << " (pop tr_full=" << pop_tr_full
                  << ", tr_fixed=" << pop_tr_fixed << ")\n"
                  << "    reject@.05  chi2=" << cell.rej[0][1] / d
                  << "  fixed=" << cell.rej[1][1] / d
                  << "  full=" << cell.rej[2][1] / d << "\n";
      }
      ++cell_idx;
    }
  }

  std::ofstream csv(cfg.out);
  csv << "lambda,eps,n,reps_used,pop_trace_fixed,pop_trace_full,mean_T";
  for (int ref = 0; ref < 3; ++ref) {
    const char* rn = ref == 0 ? "chi2" : (ref == 1 ? "fixed" : "full");
    for (double al : kAlpha) {
      csv << ",rej_" << rn << "_" << static_cast<int>(std::lround(al * 100));
    }
  }
  csv << "\n";
  csv << std::setprecision(6);
  for (const auto& c : cells) {
    const double d = c.used > 0 ? static_cast<double>(c.used) : 1.0;
    csv << c.lambda << "," << c.eps << "," << c.n << "," << c.used << ","
        << c.pop_trace_fixed << "," << c.pop_trace_full << "," << c.mean_T;
    for (int ref = 0; ref < 3; ++ref)
      for (std::size_t ai = 0; ai < kAlpha.size(); ++ai)
        csv << "," << c.rej[static_cast<std::size_t>(ref)][ai] / d;
    csv << "\n";
  }
  std::cout << "\nwrote " << cfg.out << "\n";
  return 0;
}
