// Advisory simulation for the robust (generalized / SB-scaled) score tests in
// inference::frontier. lavaan does not implement this statistic, so there is no
// package oracle; instead this validates the two things a number-match cannot:
//
//   1. Calibration. Under non-normal H0 data (a correctly-specified 1-factor
//      model, heavy-tailed errors), the robust modification index's rejection
//      rate at alpha should sit near alpha, while the ordinary normal-theory MI
//      over-rejects (leptokurtosis inflates the score variance the NT statistic
//      ignores). This is the "the correction is needed and works" demonstration.
//
//   2. Trinity equivalence. On the same data the robust score test of freeing a
//      parameter, the robust Wald z^2 of that parameter (refit + robust_se), and
//      the Satorra-Bentler scaled LRT difference all estimate the same thing and
//      converge as N grows. This anchors the new statistic to magmaan's
//      lavaan-validated robust_se and scaled-LRT machinery.
//
// Not part of ctest. Build + run via the local justfile.

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/spec/build.hpp"

namespace {

namespace inf = magmaan::inference;
namespace rob = magmaan::robust;
namespace est = magmaan::estimate;

struct Config {
  int           n = 2000;
  int           reps = 400;
  double        df_t = 5.0;     // multivariate-t dof (smaller ⇒ heavier tails)
  double        alpha = 0.05;
  std::uint64_t seed = 20260602;
};

struct Handles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep      rep;
};

std::optional<Handles> build_model(std::string_view src) {
  auto fp = magmaan::parse::Parser::parse(src);
  if (!fp.has_value()) return std::nullopt;
  auto pt = magmaan::spec::build(*fp);
  if (!pt.has_value()) return std::nullopt;
  auto rep = magmaan::model::build_matrix_rep(*pt);
  if (!rep.has_value()) return std::nullopt;
  return Handles{std::move(*pt), std::move(*rep)};
}

// Population covariance for an exact 1-factor model over 5 indicators: the
// residual cov(x1,x2) is truly 0, so freeing it is a genuine null.
Eigen::MatrixXd population_sigma() {
  Eigen::VectorXd lambda(5);
  lambda << 1.0, 0.8, 0.9, 0.7, 0.85;
  Eigen::VectorXd theta(5);
  theta << 0.6, 0.7, 0.5, 0.8, 0.65;
  const double phi = 1.0;
  return lambda * lambda.transpose() * phi + theta.asDiagonal().toDenseMatrix();
}

// Heavy-tailed multivariate-t with covariance ≈ Sigma.
Eigen::MatrixXd mvt_sample(std::mt19937& rng, int n, const Eigen::MatrixXd& Sigma,
                           double df) {
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  const Eigen::MatrixXd L = llt.matrixL();
  const Eigen::Index p = Sigma.rows();
  std::normal_distribution<double> z(0.0, 1.0);
  std::chi_squared_distribution<double> chi(df);
  const double scale = std::sqrt((df - 2.0) / df);
  Eigen::MatrixXd X(n, p);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd zi(p);
    for (Eigen::Index j = 0; j < p; ++j) zi(j) = z(rng);
    const double w = chi(rng) / df;
    X.row(i) = (scale * (L * zi) / std::sqrt(w)).transpose();
  }
  return X;
}

// Free-parameter index of the off-diagonal residual covariance (x1 ~~ x2).
int offdiag_cov_index(const magmaan::spec::LatentStructure& pt) {
  for (std::size_t r = 0; r < pt.size(); ++r) {
    if (pt.op[r] == magmaan::parse::Op::Covariance &&
        pt.lhs_var[r] != pt.rhs_var[r] && pt.free[r] > 0) {
      return pt.free[r] - 1;
    }
  }
  return -1;
}

struct Accum {
  long reps = 0;
  long reject_nt = 0;
  long reject_robust = 0;
  double sum_score = 0, sum_wald = 0, sum_lrt = 0;
  double sum_ratio_sw = 0, sum_ratio_sl = 0;
};

Config parse_args(int argc, char** argv) {
  Config cfg;
  auto num = [](std::string_view s, auto fallback) {
    auto out = fallback;
    std::from_chars(s.data(), s.data() + s.size(), out);
    return out;
  };
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto eq = a.find('=');
    if (eq == std::string_view::npos) continue;
    std::string_view k = a.substr(0, eq), v = a.substr(eq + 1);
    if (k == "--n") cfg.n = num(v, cfg.n);
    else if (k == "--reps") cfg.reps = num(v, cfg.reps);
    else if (k == "--df") cfg.df_t = num(v, cfg.df_t);
    else if (k == "--alpha") cfg.alpha = num(v, cfg.alpha);
    else if (k == "--seed") cfg.seed = num(v, cfg.seed);
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  const Config cfg = parse_args(argc, argv);
  const Eigen::MatrixXd Sigma = population_sigma();

  auto h0 = build_model("f =~ x1 + x2 + x3 + x4 + x5\nx1 ~~ 0*x2");  // restricted
  auto h1 = build_model("f =~ x1 + x2 + x3 + x4 + x5\nx1 ~~ x2");    // full
  if (!h0 || !h1) {
    std::cerr << "model build failed\n";
    return 1;
  }
  const int wald_idx = offdiag_cov_index(h1->pt);
  if (wald_idx < 0) {
    std::cerr << "could not locate x1~~x2 free index\n";
    return 1;
  }

  std::mt19937 rng(cfg.seed);
  const rob::InferenceSpec spec{rob::Information::Expected,
                                rob::WeightMoments::Structured,
                                rob::ScoreCovariance::Empirical};
  inf::frontier::RobustScoreOptions ropts;
  ropts.spec = spec;  // FixedRowsOnly, Expected by default

  Accum acc;
  long fit_fail = 0;
  for (int rep = 0; rep < cfg.reps; ++rep) {
    magmaan::data::RawData raw;
    raw.X.push_back(mvt_sample(rng, cfg.n, Sigma, cfg.df_t));
    auto samp = magmaan::data::sample_stats_from_raw(raw);
    if (!samp.has_value()) { ++fit_fail; continue; }

    auto x0_0 = est::simple_start_values(h0->pt, h0->rep, *samp);
    auto x0_1 = est::simple_start_values(h1->pt, h1->rep, *samp);
    if (!x0_0 || !x0_1) { ++fit_fail; continue; }
    auto e0 = est::fit_ml(h0->pt, h0->rep, *samp, *x0_0);
    auto e1 = est::fit_ml(h1->pt, h1->rep, *samp, *x0_1);
    if (!e0 || !e1) { ++fit_fail; continue; }

    // --- Calibration: NT vs robust MI of the null residual covariance ---
    auto nt_mi = inf::modification_indices(h0->pt, h0->rep, *samp, *e0,
                                           inf::ScoreInformation::Expected);
    auto rb_mi = inf::frontier::modification_indices_robust(
        h0->pt, h0->rep, *samp, raw, *e0, ropts);
    if (!nt_mi || !rb_mi || nt_mi->rows.empty() || rb_mi->rows.empty()) {
      ++fit_fail; continue;
    }
    const double p_nt = nt_mi->rows[0].p_value;
    const double p_rb = rb_mi->rows[0].p_value;
    const double t_score = rb_mi->rows[0].mi_scaled;

    // --- Trinity: robust Wald z^2 (H1) and SB scaled LRT diff ---
    auto wald = rob::robust_se(h1->pt, h1->rep, *samp, *e1, raw, spec);
    auto df0 = inf::df_stat(h0->pt, *samp, e0->theta);
    auto df1 = inf::df_stat(h1->pt, *samp, e1->theta);
    if (!wald || !df0 || !df1) { ++fit_fail; continue; }
    const double se = wald->se(wald_idx);
    const double t_wald = (e1->theta(wald_idx) / se) * (e1->theta(wald_idx) / se);

    const double T_H0 = inf::chi2_stat(*samp, *e0);
    const double T_H1 = inf::chi2_stat(*samp, *e1);
    auto lrt = rob::lr_test_satorra_bentler2001_from_data(
        h1->pt, h1->rep, e1->theta, h0->pt, h0->rep, e0->theta, raw, T_H0, T_H1,
        *df0, *df1, rob::GammaSource::Empirical);
    if (!lrt) { ++fit_fail; continue; }
    const double t_lrt = lrt->T_scaled;

    if (!(std::isfinite(t_score) && std::isfinite(t_wald) && std::isfinite(t_lrt)))
    { ++fit_fail; continue; }

    acc.reps++;
    if (p_nt < cfg.alpha) acc.reject_nt++;
    if (p_rb < cfg.alpha) acc.reject_robust++;
    acc.sum_score += t_score;
    acc.sum_wald += t_wald;
    acc.sum_lrt += t_lrt;
    acc.sum_ratio_sw += t_score / t_wald;
    acc.sum_ratio_sl += t_score / t_lrt;
  }

  const double R = static_cast<double>(acc.reps);
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "robust score sanity — n=" << cfg.n << " reps=" << cfg.reps
            << " df_t=" << cfg.df_t << " alpha=" << cfg.alpha << "\n";
  std::cout << "  used reps:          " << acc.reps << "  (skipped " << fit_fail
            << ")\n";
  std::cout << "  CALIBRATION (rejection rate of a true-null residual cov):\n";
  std::cout << "    NT  MI:           " << (acc.reject_nt / R)
            << "   (expected to OVER-reject vs alpha)\n";
  std::cout << "    robust MI:        " << (acc.reject_robust / R)
            << "   (expected ≈ alpha)\n";
  std::cout << "  TRINITY (mean statistic, df=1):\n";
  std::cout << "    robust score:     " << (acc.sum_score / R) << "\n";
  std::cout << "    robust Wald z^2:  " << (acc.sum_wald / R) << "\n";
  std::cout << "    SB scaled LRT:    " << (acc.sum_lrt / R) << "\n";
  std::cout << "    mean score/Wald:  " << (acc.sum_ratio_sw / R)
            << "   (→ 1)\n";
  std::cout << "    mean score/LRT:   " << (acc.sum_ratio_sl / R)
            << "   (→ 1)\n";
  return 0;
}
