#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/data/h_score.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/pairwise_mixed.hpp"
#include "magmaan/data/pairwise_ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/pairwise.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

namespace {

Eigen::MatrixXd ordinal_expected_counts(const Eigen::VectorXd& th_i,
                                        const Eigen::VectorXd& th_j,
                                        double rho,
                                        double total) {
  const double inf = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd out(th_i.size() + 1, th_j.size() + 1);
  for (Eigen::Index a = 0; a < out.rows(); ++a) {
    const double lo_i = (a == 0) ? -inf : th_i(a - 1);
    const double hi_i = (a + 1 == out.rows()) ? inf : th_i(a);
    for (Eigen::Index b = 0; b < out.cols(); ++b) {
      const double lo_j = (b == 0) ? -inf : th_j(b - 1);
      const double hi_j = (b + 1 == out.cols()) ? inf : th_j(b);
      out(a, b) = total * magmaan::data::ordinal_bvn_rect_prob(
          lo_i, hi_i, lo_j, hi_j, rho);
    }
  }
  return out;
}

double h_score_pair_objective(
    const Eigen::MatrixXd& counts,
    const Eigen::VectorXd& th_i,
    const Eigen::VectorXd& th_j,
    double rho,
    const magmaan::data::PolychoricHScoreOptions& options) {
  const double inf = std::numeric_limits<double>::infinity();
  const double total = counts.sum();
  double out = 0.0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -inf : th_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? inf : th_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -inf : th_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? inf : th_j(b);
      const double p = std::max(
          1.4901161193847656e-8,
          magmaan::data::ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
      const double t = counts(a, b) / (total * p);
      auto h = magmaan::data::eval_polychoric_h_score(t, options);
      if (!h.has_value()) return std::numeric_limits<double>::quiet_NaN();
      out += p * h->phi;
    }
  }
  return out;
}

double dpd_pair_objective(const Eigen::MatrixXd& counts,
                          const Eigen::VectorXd& th_i,
                          const Eigen::VectorXd& th_j,
                          double rho,
                          double alpha) {
  const double inf = std::numeric_limits<double>::infinity();
  const double total = counts.sum();
  double out = 0.0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    const double lo_i = (a == 0) ? -inf : th_i(a - 1);
    const double hi_i = (a + 1 == counts.rows()) ? inf : th_i(a);
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const double lo_j = (b == 0) ? -inf : th_j(b - 1);
      const double hi_j = (b + 1 == counts.cols()) ? inf : th_j(b);
      const double p = std::max(
          1.4901161193847656e-8,
          magmaan::data::ordinal_bvn_rect_prob(lo_i, hi_i, lo_j, hi_j, rho));
      const double fhat = counts(a, b) / total;
      const double pa = std::pow(p, alpha);
      out += p * pa - ((1.0 + alpha) / alpha) * fhat * pa;
    }
  }
  return out;
}

Eigen::MatrixXd ordinal_pair_score_rows_from_counts(
    const Eigen::MatrixXd& counts,
    const Eigen::VectorXd& th_i,
    const Eigen::VectorXd& th_j,
    double rho) {
  std::int64_t n = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      n += static_cast<std::int64_t>(std::llround(counts(a, b)));
    }
  }
  Eigen::VectorXi xi(n);
  Eigen::VectorXi xj(n);
  Eigen::Index row = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const auto reps = static_cast<Eigen::Index>(std::llround(counts(a, b)));
      for (Eigen::Index k = 0; k < reps; ++k) {
        xi(row) = static_cast<int>(a);
        xj(row) = static_cast<int>(b);
        ++row;
      }
    }
  }

  auto scores = magmaan::data::ordinal_pair_scores(xi, xj, rho, th_i, th_j);
  REQUIRE(scores.has_value());
  Eigen::MatrixXd out(n, th_i.size() + th_j.size() + 1);
  out.leftCols(th_i.size()) = scores->threshold_i;
  out.middleCols(th_i.size(), th_j.size()) = scores->threshold_j;
  out.col(out.cols() - 1) = scores->rho;
  return out;
}

Eigen::MatrixXd ordinal_data_from_pair_counts(const Eigen::MatrixXd& counts) {
  Eigen::Index n = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      n += static_cast<Eigen::Index>(std::llround(counts(a, b)));
    }
  }
  Eigen::MatrixXd out(n, 2);
  Eigen::Index row = 0;
  for (Eigen::Index a = 0; a < counts.rows(); ++a) {
    for (Eigen::Index b = 0; b < counts.cols(); ++b) {
      const auto reps = static_cast<Eigen::Index>(std::llround(counts(a, b)));
      for (Eigen::Index k = 0; k < reps; ++k) {
        out(row, 0) = static_cast<double>(a + 1);
        out(row, 1) = static_cast<double>(b + 1);
        ++row;
      }
    }
  }
  return out;
}

}  // namespace

TEST_CASE("Polychoric h-score API evaluates predefined caps") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  auto ml = magmaan::data::eval_polychoric_h_score(1.7);
  REQUIRE(ml.has_value());
  CHECK(ml->h == doctest::Approx(1.7));
  CHECK(ml->dh == doctest::Approx(1.0));
  CHECK(ml->phi == doctest::Approx(1.7 * std::log(1.7)));

  auto hard_inf = magmaan::data::eval_polychoric_h_score(
      2.4, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap,
               .k = std::numeric_limits<double>::infinity()});
  REQUIRE(hard_inf.has_value());
  CHECK(hard_inf->h == doctest::Approx(2.4));
  CHECK(hard_inf->dh == doctest::Approx(1.0));
  CHECK(hard_inf->phi == doctest::Approx(2.4 * std::log(2.4)));

  auto hard = magmaan::data::eval_polychoric_h_score(
      2.4, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap, .k = 1.6});
  REQUIRE(hard.has_value());
  CHECK(hard->h == doctest::Approx(1.6));
  CHECK(hard->dh == doctest::Approx(0.0));
  CHECK(hard->phi == doctest::Approx(2.4 * (std::log(1.6) + 1.0) - 1.6));

  auto hard_kink = magmaan::data::eval_polychoric_h_score(
      1.6, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap, .k = 1.6});
  REQUIRE(hard_kink.has_value());
  CHECK(hard_kink->h == doctest::Approx(1.6));
  CHECK(hard_kink->dh == doctest::Approx(0.0));

  auto smooth_low = magmaan::data::eval_polychoric_h_score(
      1.2, PolychoricHScoreOptions{.kind = PolychoricHScoreKind::SmoothCap});
  REQUIRE(smooth_low.has_value());
  CHECK(smooth_low->h == doctest::Approx(1.2));
  CHECK(smooth_low->dh == doctest::Approx(1.0));
  CHECK(smooth_low->phi == doctest::Approx(1.2 * std::log(1.2)));

  auto smooth_high = magmaan::data::eval_polychoric_h_score(
      2.8, PolychoricHScoreOptions{.kind = PolychoricHScoreKind::SmoothCap});
  REQUIRE(smooth_high.has_value());
  CHECK(smooth_high->h == doctest::Approx(1.9));
  CHECK(smooth_high->dh == doctest::Approx(0.0));
  CHECK(std::isfinite(smooth_high->phi));

  auto exp = magmaan::data::eval_polychoric_h_score(
      2.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::ExpCap, .k = 1.6, .lambda = 0.2});
  REQUIRE(exp.has_value());
  CHECK(exp->h == doctest::Approx(1.6 + 0.2 * (1.0 - std::exp(-2.0))));
  CHECK(exp->dh == doctest::Approx(std::exp(-2.0)));
  CHECK(std::isfinite(exp->phi));
}

TEST_CASE("Polychoric h-score objective contribution matches score identity") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  const PolychoricHScoreOptions options[] = {
      PolychoricHScoreOptions{.kind = PolychoricHScoreKind::ML},
      PolychoricHScoreOptions{.kind = PolychoricHScoreKind::WmaHardCap,
                              .k = 1.6},
      PolychoricHScoreOptions{.kind = PolychoricHScoreKind::SmoothCap},
      PolychoricHScoreOptions{.kind = PolychoricHScoreKind::ExpCap,
                              .k = 1.6,
                              .lambda = 0.2},
  };

  for (const auto& option : options) {
    const double t = 2.0;
    const double step = 1e-5;
    auto mid = magmaan::data::eval_polychoric_h_score(t, option);
    auto plus = magmaan::data::eval_polychoric_h_score(t + step, option);
    auto minus = magmaan::data::eval_polychoric_h_score(t - step, option);
    REQUIRE(mid.has_value());
    REQUIRE(plus.has_value());
    REQUIRE(minus.has_value());
    const double dphi = (plus->phi - minus->phi) / (2.0 * step);
    CHECK(mid->h == doctest::Approx(t * dphi - mid->phi).epsilon(1e-8));
  }
}

TEST_CASE("Polychoric h-score API rejects invalid inputs") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  auto negative = magmaan::data::eval_polychoric_h_score(-0.1);
  REQUIRE_FALSE(negative.has_value());
  CHECK(negative.error().detail.find("nonnegative") != std::string::npos);

  auto bad_hard = magmaan::data::eval_polychoric_h_score(
      1.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap, .k = 0.8});
  REQUIRE_FALSE(bad_hard.has_value());
  CHECK(bad_hard.error().detail.find("at least 1") != std::string::npos);

  auto bad_hard_inf = magmaan::data::eval_polychoric_h_score(
      1.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::WmaHardCap,
               .k = -std::numeric_limits<double>::infinity()});
  REQUIRE_FALSE(bad_hard_inf.has_value());
  CHECK(bad_hard_inf.error().detail.find("at least 1") != std::string::npos);

  auto bad_smooth = magmaan::data::eval_polychoric_h_score(
      1.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::SmoothCap, .a = 2.0, .b = 1.8});
  REQUIRE_FALSE(bad_smooth.has_value());
  CHECK(bad_smooth.error().detail.find("1 <= a < b") != std::string::npos);

  auto bad_exp = magmaan::data::eval_polychoric_h_score(
      1.0, PolychoricHScoreOptions{
               .kind = PolychoricHScoreKind::ExpCap, .k = 1.6, .lambda = 0.0});
  REQUIRE_FALSE(bad_exp.has_value());
  CHECK(bad_exp.error().detail.find("lambda > 0") != std::string::npos);
}

TEST_CASE("Ordinal pair ML kernel: probabilities, rho fit, and scores") {
  const double inf = std::numeric_limits<double>::infinity();
  Eigen::VectorXd thi(2);
  thi << -0.4, 0.7;
  Eigen::VectorXd thj(1);
  thj << 0.2;

  double total = 0.0;
  const double bi[4] = {-inf, thi(0), thi(1), inf};
  const double bj[3] = {-inf, thj(0), inf};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 2; ++b) {
      total += magmaan::data::ordinal_bvn_rect_prob(
          bi[a], bi[a + 1], bj[b], bj[b + 1], 0.35);
    }
  }
  CHECK(total == doctest::Approx(1.0).epsilon(1e-8));

  const double h = 1e-5;
  const double pp = magmaan::data::ordinal_bvn_rect_prob(
      thi(0), thi(1), -inf, thj(0), 0.3 + h);
  const double pm = magmaan::data::ordinal_bvn_rect_prob(
      thi(0), thi(1), -inf, thj(0), 0.3 - h);
  const double fd = (pp - pm) / (2.0 * h);
  const double analytic = magmaan::data::ordinal_bvn_rect_drho(
      thi(0), thi(1), -inf, thj(0), 0.3);
  CHECK(analytic == doctest::Approx(fd).epsilon(1e-4));

  Eigen::VectorXi xi(6);
  Eigen::VectorXi xj(6);
  xi << 0, 0, 1, 1, 2, 2;
  xj << 0, 1, 0, 1, 0, 1;
  auto table = magmaan::data::ordinal_pair_table(xi, xj, 3, 2);
  REQUIRE(table.has_value());
  CHECK((*table)(0, 0) == 1.0);
  CHECK((*table)(0, 1) == 1.0);
  CHECK((*table)(2, 0) == 1.0);
  CHECK((*table)(2, 1) == 1.0);

  auto fit = magmaan::data::fit_ordinal_pair_rho_ml(*table, thi, thj);
  REQUIRE(fit.has_value());
  CHECK(std::isfinite(fit->rho));
  CHECK(std::isfinite(fit->negloglik));

  auto scores = magmaan::data::ordinal_pair_scores(xi, xj, fit->rho, thi, thj);
  REQUIRE(scores.has_value());
  CHECK(scores->rho.size() == 6);
  CHECK(scores->threshold_i.rows() == 6);
  CHECK(scores->threshold_i.cols() == 2);
  CHECK(scores->threshold_j.rows() == 6);
  CHECK(scores->threshold_j.cols() == 1);
  CHECK(scores->rho.allFinite());
  CHECK(scores->threshold_i.allFinite());
  CHECK(scores->threshold_j.allFinite());
}

TEST_CASE("Ordinal pair ML kernel: independence and lavaan 2x2 adjustment") {
  Eigen::VectorXd th(1);
  th << 0.0;

  Eigen::MatrixXd balanced(2, 2);
  balanced << 10.0, 10.0,
              10.0, 10.0;
  auto independent = magmaan::data::fit_ordinal_pair_rho_ml(balanced, th, th);
  REQUIRE(independent.has_value());
  CHECK(independent->rho == doctest::Approx(0.0).epsilon(1e-12));
  CHECK(independent->iterations == 0);

  Eigen::MatrixXd sparse(2, 2);
  sparse << 0.0, 4.0,
            5.0, 6.0;
  auto adjusted = magmaan::data::fit_ordinal_pair_rho_ml(sparse, th, th);
  REQUIRE(adjusted.has_value());
  CHECK(adjusted->adjusted_counts(0, 0) == doctest::Approx(0.5));
  CHECK(adjusted->adjusted_counts(1, 1) == doctest::Approx(6.5));
  CHECK(adjusted->adjusted_counts(1, 0) == doctest::Approx(4.5));
  CHECK(adjusted->adjusted_counts(0, 1) == doctest::Approx(3.5));
  CHECK(std::isfinite(adjusted->rho));
  CHECK(adjusted->rho > -1.0);
  CHECK(adjusted->rho < 1.0);
}

TEST_CASE("Ordinal pair h-weighted rho preserves ML limits and diagnostics") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd thi(2);
  thi << -0.4, 0.7;
  Eigen::VectorXd thj(1);
  thj << 0.2;
  Eigen::MatrixXd counts(3, 2);
  counts << 8.0, 3.0,
            4.0, 7.0,
            2.0, 9.0;

  auto ml = magmaan::data::fit_ordinal_pair_rho_ml(counts, thi, thj);
  auto h_ml = magmaan::data::fit_ordinal_pair_rho_h_weighted(counts, thi, thj);
  auto hard_inf = magmaan::data::fit_ordinal_pair_rho_h_weighted(
      counts, thi, thj,
      magmaan::data::OrdinalPairHWeightedOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = std::numeric_limits<double>::infinity()}});
  REQUIRE(ml.has_value());
  REQUIRE(h_ml.has_value());
  REQUIRE(hard_inf.has_value());
  CHECK(h_ml->rho == doctest::Approx(ml->rho));
  CHECK(hard_inf->rho == doctest::Approx(ml->rho));
  CHECK(h_ml->converged);
  CHECK(hard_inf->converged);
  CHECK(std::isfinite(h_ml->objective));
  CHECK(std::isfinite(h_ml->score));
  CHECK(h_ml->probabilities.rows() == counts.rows());
  CHECK(h_ml->probabilities.cols() == counts.cols());
  CHECK(h_ml->expected_counts.rows() == counts.rows());
  CHECK(h_ml->residual_counts.isApprox(
      h_ml->adjusted_counts - h_ml->expected_counts, 1e-12));
  CHECK(h_ml->pearson_residuals.allFinite());
  CHECK(h_ml->weights.isApprox(Eigen::MatrixXd::Ones(3, 2), 1e-12));
  CHECK(hard_inf->weights.isApprox(Eigen::MatrixXd::Ones(3, 2), 1e-12));
}

TEST_CASE("Ordinal pair h-weighted rho downweights contaminated cells") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd th(2);
  th << -0.55, 0.75;
  const Eigen::MatrixXd clean = ordinal_expected_counts(th, th, 0.55, 5000.0);
  Eigen::MatrixXd contaminated = clean;
  contaminated(0, 2) += 900.0;

  auto clean_ml = magmaan::data::fit_ordinal_pair_rho_ml(clean, th, th);
  auto contaminated_ml =
      magmaan::data::fit_ordinal_pair_rho_ml(contaminated, th, th);
  auto robust = magmaan::data::fit_ordinal_pair_rho_h_weighted(
      contaminated, th, th,
      magmaan::data::OrdinalPairHWeightedOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = 1.15}});
  REQUIRE(clean_ml.has_value());
  REQUIRE(contaminated_ml.has_value());
  REQUIRE(robust.has_value());
  CHECK(robust->converged);
  CHECK(contaminated_ml->rho < clean_ml->rho);
  CHECK(robust->rho > contaminated_ml->rho);
  CHECK(std::abs(robust->rho - clean_ml->rho) <
        std::abs(contaminated_ml->rho - clean_ml->rho));
  CHECK(robust->weights(0, 2) < 1.0);
  CHECK(robust->pearson_residuals(0, 2) > 0.0);
}

TEST_CASE("Polyserial pair ML kernel: likelihood, rho fit, and scores") {
  Eigen::VectorXi cat(8);
  cat << 0, 0, 1, 1, 1, 2, 2, 2;
  Eigen::VectorXd u(8);
  u << -1.4, -0.8, -0.5, 0.0, 0.4, 0.7, 1.1, 1.5;
  Eigen::VectorXd th(2);
  th << -0.45, 0.65;

  auto nll0 = magmaan::data::polyserial_pair_negloglik(cat, u, th, 0.0);
  auto nll1 = magmaan::data::polyserial_pair_negloglik(cat, u, th, 0.45);
  REQUIRE(nll0.has_value());
  REQUIRE(nll1.has_value());
  CHECK(std::isfinite(*nll0));
  CHECK(std::isfinite(*nll1));
  CHECK(*nll1 < *nll0);

  auto fit = magmaan::data::fit_polyserial_pair_rho_ml(cat, u, th);
  REQUIRE(fit.has_value());
  CHECK(std::isfinite(fit->rho));
  CHECK(std::isfinite(fit->negloglik));
  CHECK(fit->rho > 0.0);
  CHECK(fit->rho > -1.0);
  CHECK(fit->rho < 1.0);

  auto scores = magmaan::data::polyserial_pair_scores(cat, u, fit->rho, th);
  REQUIRE(scores.has_value());
  CHECK(scores->rho.size() == cat.size());
  CHECK(scores->thresholds.rows() == cat.size());
  CHECK(scores->thresholds.cols() == th.size());
  CHECK(scores->rho.allFinite());
  CHECK(scores->thresholds.allFinite());
  CHECK(scores->score_contributions.rows() == cat.size());
  CHECK(scores->score_contributions.cols() == th.size() + 1);
  CHECK(scores->score_contributions.leftCols(th.size()).isApprox(
      scores->thresholds, 0.0));
  CHECK(scores->score_contributions.col(th.size()).isApprox(scores->rho, 0.0));
  CHECK(scores->score_gamma.rows() == th.size() + 1);
  CHECK(scores->score_gamma.cols() == th.size() + 1);
  CHECK(scores->score_gamma.isApprox(
      (scores->score_contributions.transpose() * scores->score_contributions) /
          static_cast<double>(cat.size()),
      1e-12));
}

TEST_CASE("Polyserial pair joint DPD estimates thresholds and downweights continuous-tail discordance") {
  std::mt19937 rng(20260516);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::VectorXd th(2);
  th << -0.45, 0.65;

  constexpr Eigen::Index n_clean = 420;
  constexpr Eigen::Index n_bad = 35;
  Eigen::VectorXi cat_clean(n_clean);
  Eigen::VectorXd u_clean(n_clean);
  const double rho_true = 0.62;
  const double sd = std::sqrt(1.0 - rho_true * rho_true);
  for (Eigen::Index r = 0; r < n_clean; ++r) {
    const double u = norm(rng);
    const double z = rho_true * u + sd * norm(rng);
    u_clean(r) = u;
    cat_clean(r) = (z > th(0)) + (z > th(1));
  }

  Eigen::VectorXi cat_cont(n_clean + n_bad);
  Eigen::VectorXd u_cont(n_clean + n_bad);
  Eigen::VectorXd x_cont(n_clean + n_bad);
  cat_cont.head(n_clean) = cat_clean;
  u_cont.head(n_clean) = u_clean;
  x_cont.head(n_clean) = u_clean;
  for (Eigen::Index r = 0; r < n_bad; ++r) {
    cat_cont(n_clean + r) = 0;
    u_cont(n_clean + r) = 5.5 + 0.01 * static_cast<double>(r);
    x_cont(n_clean + r) = u_cont(n_clean + r);
  }

  auto clean_ml = magmaan::data::fit_polyserial_pair_rho_ml(
      cat_clean, u_clean, th);
  auto cont_ml = magmaan::data::fit_polyserial_pair_rho_ml(
      cat_cont, u_cont, th);
  auto robust = magmaan::data::fit_polyserial_pair_joint_dpd(
      cat_cont, x_cont,
      magmaan::data::PolyserialPairJointDpdOptions{.alpha = 0.5});

  REQUIRE(clean_ml.has_value());
  REQUIRE(cont_ml.has_value());
  REQUIRE(robust.has_value());
  CHECK(cont_ml->rho < clean_ml->rho);
  CHECK(robust->rho > cont_ml->rho);
  CHECK(std::abs(robust->rho - clean_ml->rho) <
        std::abs(cont_ml->rho - clean_ml->rho));
  CHECK(robust->thresholds.size() == th.size());
  CHECK(robust->sd > 0.0);
  CHECK(robust->weights.tail(n_bad).maxCoeff() <
        robust->weights.head(n_clean).mean());
  CHECK(robust->probabilities.size() == cat_cont.size());
  CHECK(robust->joint_densities.size() == cat_cont.size());
  CHECK(robust->weights.size() == cat_cont.size());
  CHECK(robust->weights.allFinite());
}

TEST_CASE("Polyserial pair ML kernel rejects malformed inputs") {
  Eigen::VectorXi cat(3);
  cat << 0, 1, 3;
  Eigen::VectorXd u(3);
  u << -0.5, 0.0, 0.5;
  Eigen::VectorXd th(2);
  th << -0.4, 0.6;

  auto bad_cat = magmaan::data::fit_polyserial_pair_rho_ml(cat, u, th);
  REQUIRE_FALSE(bad_cat.has_value());
  CHECK(bad_cat.error().detail.find("category outside") != std::string::npos);

  cat << 0, 1, 2;
  th << 0.6, -0.4;
  auto bad_th = magmaan::data::fit_polyserial_pair_rho_ml(cat, u, th);
  REQUIRE_FALSE(bad_th.has_value());
  CHECK(bad_th.error().detail.find("strictly increasing") != std::string::npos);
}

TEST_CASE("Polyserial fixed-marginal DPD preserves ML limit and downweights discordance") {
  std::mt19937 rng(20260517);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::VectorXd th(2);
  th << -0.45, 0.65;

  constexpr Eigen::Index n_clean = 420;
  constexpr Eigen::Index n_bad = 35;
  Eigen::VectorXi cat_clean(n_clean);
  Eigen::VectorXd u_clean(n_clean);
  const double rho_true = 0.62;
  const double sd = std::sqrt(1.0 - rho_true * rho_true);
  for (Eigen::Index r = 0; r < n_clean; ++r) {
    const double u = norm(rng);
    const double z = rho_true * u + sd * norm(rng);
    u_clean(r) = u;
    cat_clean(r) = (z > th(0)) + (z > th(1));
  }

  Eigen::VectorXi cat_cont(n_clean + n_bad);
  Eigen::VectorXd u_cont(n_clean + n_bad);
  cat_cont.head(n_clean) = cat_clean;
  u_cont.head(n_clean) = u_clean;
  for (Eigen::Index r = 0; r < n_bad; ++r) {
    cat_cont(n_clean + r) = 0;
    u_cont(n_clean + r) = 5.5 + 0.01 * static_cast<double>(r);
  }

  auto clean_ml = magmaan::data::fit_polyserial_pair_rho_ml(
      cat_clean, u_clean, th);
  auto cont_ml = magmaan::data::fit_polyserial_pair_rho_ml(
      cat_cont, u_cont, th);
  auto dpd0 = magmaan::data::fit_polyserial_pair_rho_dpd(
      cat_cont, u_cont, th,
      magmaan::data::PolyserialPairDpdOptions{.alpha = 0.0});
  auto robust = magmaan::data::fit_polyserial_pair_rho_dpd(
      cat_cont, u_cont, th,
      magmaan::data::PolyserialPairDpdOptions{.alpha = 0.5});

  REQUIRE(clean_ml.has_value());
  REQUIRE(cont_ml.has_value());
  REQUIRE(dpd0.has_value());
  REQUIRE(robust.has_value());
  CHECK(dpd0->rho == doctest::Approx(cont_ml->rho));
  CHECK(cont_ml->rho < clean_ml->rho);
  CHECK(robust->rho > cont_ml->rho);
  CHECK(std::abs(robust->rho - clean_ml->rho) <
        std::abs(cont_ml->rho - clean_ml->rho));
  CHECK(robust->probabilities.size() == cat_cont.size());
  CHECK(robust->weights.tail(n_bad).maxCoeff() <
        robust->weights.head(n_clean).mean());

  auto ml_scores = magmaan::data::polyserial_pair_scores(
      cat_cont, u_cont, cont_ml->rho, th);
  auto dpd0_scores = magmaan::data::polyserial_pair_dpd_scores(
      cat_cont, u_cont, cont_ml->rho, th,
      magmaan::data::PolyserialPairDpdOptions{.alpha = 0.0});
  auto robust_scores = magmaan::data::polyserial_pair_dpd_scores(
      cat_cont, u_cont, robust->rho, th,
      magmaan::data::PolyserialPairDpdOptions{.alpha = 0.5});
  REQUIRE(ml_scores.has_value());
  REQUIRE(dpd0_scores.has_value());
  REQUIRE(robust_scores.has_value());
  CHECK(dpd0_scores->score_contributions.isApprox(
      ml_scores->score_contributions, 0.0));
  CHECK(dpd0_scores->bread.isApprox(ml_scores->score_gamma, 0.0));
  CHECK(robust_scores->score_contributions.rows() == cat_cont.size());
  CHECK(robust_scores->score_contributions.cols() == th.size() + 1);
  CHECK(robust_scores->score_gamma.isApprox(
      (robust_scores->score_contributions.transpose() *
       robust_scores->score_contributions) /
          static_cast<double>(cat_cont.size()),
      1e-12));
  CHECK(robust_scores->bread.rows() == th.size() + 1);
  CHECK(robust_scores->bread.cols() == th.size() + 1);
  CHECK(robust_scores->bread(th.size(), th.size()) > 0.0);
}

TEST_CASE("Continuous pair normal ML kernel returns complete-data pair diagnostics") {
  Eigen::VectorXd x(4);
  Eigen::VectorXd y(4);
  x << 1.0, 2.0, 3.0, 4.0;
  y << 2.0, 3.0, 5.0, 7.0;

  auto fit = magmaan::data::fit_continuous_pair_normal_ml(x, y);
  REQUIRE(fit.has_value());
  CHECK(fit->n_obs == 4);
  CHECK(fit->mean_i == doctest::Approx(2.5));
  CHECK(fit->mean_j == doctest::Approx(4.25));
  CHECK(fit->var_i == doctest::Approx(1.25));
  CHECK(fit->var_j == doctest::Approx(3.6875));
  CHECK(fit->cov == doctest::Approx(2.125));
  CHECK(fit->rho == doctest::Approx(2.125 / std::sqrt(1.25 * 3.6875)));
  CHECK(std::isfinite(fit->negloglik));
  CHECK(fit->score_contributions.rows() == x.size());
  CHECK(fit->score_contributions.cols() == 5);
  CHECK(fit->score_contributions.allFinite());
  CHECK(fit->score_contributions.colwise().sum().norm() < 1e-10);
  CHECK(fit->score_gamma.rows() == 5);
  CHECK(fit->score_gamma.cols() == 5);
  CHECK(fit->score_gamma.isApprox(
      (fit->score_contributions.transpose() * fit->score_contributions) /
          static_cast<double>(x.size()),
      1e-12));

  auto nll = magmaan::data::continuous_pair_normal_negloglik(
      x, y, fit->mean_i, fit->mean_j, fit->var_i, fit->var_j, fit->cov);
  REQUIRE(nll.has_value());
  CHECK(*nll == doctest::Approx(fit->negloglik));
  auto scores = magmaan::data::continuous_pair_normal_scores(
      x, y, fit->mean_i, fit->mean_j, fit->var_i, fit->var_j, fit->cov);
  REQUIRE(scores.has_value());
  CHECK(scores->score_contributions.isApprox(fit->score_contributions, 0.0));
  CHECK(scores->score_gamma.isApprox(fit->score_gamma, 0.0));

  y << 2.0, 4.0, 6.0, 8.0;
  auto singular = magmaan::data::fit_continuous_pair_normal_ml(x, y);
  REQUIRE_FALSE(singular.has_value());
  CHECK(singular.error().detail.find("positive definite") != std::string::npos);
}

TEST_CASE("Mixed pair labels match MixedOrdinalStats moment order") {
  std::vector<std::int32_t> ordered{0, 1, 0};
  std::vector<std::int32_t> threshold_ov{1, 1};
  std::vector<std::int32_t> threshold_level{1, 2};

  auto moments = magmaan::data::mixed_moment_labels(
      ordered, threshold_ov, threshold_level);
  REQUIRE(moments.has_value());
  REQUIRE(moments->size() == 9);

  CHECK((*moments)[0].kind == magmaan::data::MixedMomentKind::threshold);
  CHECK((*moments)[0].variable == 1);
  CHECK((*moments)[0].threshold_level == 1);
  CHECK((*moments)[1].kind == magmaan::data::MixedMomentKind::threshold);
  CHECK((*moments)[1].variable == 1);
  CHECK((*moments)[1].threshold_level == 2);
  CHECK((*moments)[2].kind == magmaan::data::MixedMomentKind::continuous_mean);
  CHECK((*moments)[2].variable == 0);
  CHECK((*moments)[3].kind == magmaan::data::MixedMomentKind::continuous_mean);
  CHECK((*moments)[3].variable == 2);
  CHECK((*moments)[4].kind == magmaan::data::MixedMomentKind::continuous_variance);
  CHECK((*moments)[4].variable == 0);
  CHECK((*moments)[5].kind == magmaan::data::MixedMomentKind::continuous_variance);
  CHECK((*moments)[5].variable == 2);

  auto pairs = magmaan::data::mixed_pair_labels(
      ordered, static_cast<std::int32_t>(threshold_ov.size()));
  REQUIRE(pairs.has_value());
  REQUIRE(pairs->size() == 3);
  CHECK((*pairs)[0].i == 1);
  CHECK((*pairs)[0].j == 0);
  CHECK((*pairs)[0].moment_index == 6);
  CHECK((*pairs)[0].kind == magmaan::data::MixedPairKind::continuous_ordinal);
  CHECK((*pairs)[1].i == 2);
  CHECK((*pairs)[1].j == 0);
  CHECK((*pairs)[1].moment_index == 7);
  CHECK((*pairs)[1].kind == magmaan::data::MixedPairKind::continuous_continuous);
  CHECK((*pairs)[2].i == 2);
  CHECK((*pairs)[2].j == 1);
  CHECK((*pairs)[2].moment_index == 8);
  CHECK((*pairs)[2].kind == magmaan::data::MixedPairKind::continuous_ordinal);

  CHECK((*moments)[6].kind == magmaan::data::MixedMomentKind::pair);
  CHECK((*moments)[6].variable_i == 1);
  CHECK((*moments)[6].variable_j == 0);
  CHECK((*moments)[6].pair_kind == magmaan::data::MixedPairKind::continuous_ordinal);
  CHECK((*moments)[7].kind == magmaan::data::MixedMomentKind::pair);
  CHECK((*moments)[7].variable_i == 2);
  CHECK((*moments)[7].variable_j == 0);
  CHECK((*moments)[7].pair_kind == magmaan::data::MixedPairKind::continuous_continuous);
  CHECK((*moments)[8].kind == magmaan::data::MixedMomentKind::pair);
  CHECK((*moments)[8].variable_i == 2);
  CHECK((*moments)[8].variable_j == 1);
  CHECK((*moments)[8].pair_kind == magmaan::data::MixedPairKind::continuous_ordinal);

  threshold_ov = {0};
  threshold_level = {1};
  auto bad = magmaan::data::mixed_moment_labels(
      ordered, threshold_ov, threshold_level);
  REQUIRE_FALSE(bad.has_value());
  CHECK(bad.error().detail.find("invalid ordered variable") != std::string::npos);
}

TEST_CASE("Ordinal pair observed table skips NaN by observed-pair semantics") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd xi(7);
  Eigen::VectorXd xj(7);
  xi << 1.0, 2.0, nan, 3.0, 1.0, 2.0, 3.0;
  xj << 1.0, 2.0, 1.0, nan, 2.0, 1.0, 2.0;

  auto table = magmaan::data::ordinal_pair_observed_table(xi, xj, 3, 2);
  REQUIRE(table.has_value());
  CHECK(table->n_obs == 5);
  CHECK(table->n_missing == 2);
  CHECK(table->counts.rows() == 3);
  CHECK(table->counts.cols() == 2);
  CHECK(table->counts(0, 0) == doctest::Approx(1.0));
  CHECK(table->counts(0, 1) == doctest::Approx(1.0));
  CHECK(table->counts(1, 0) == doctest::Approx(1.0));
  CHECK(table->counts(1, 1) == doctest::Approx(1.0));
  CHECK(table->counts(2, 0) == doctest::Approx(0.0));
  CHECK(table->counts(2, 1) == doctest::Approx(1.0));
}

TEST_CASE("Ordinal pair observed table rejects malformed observed categories") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd xi(3);
  Eigen::VectorXd xj(3);
  xi << 1.0, 2.5, nan;
  xj << 1.0, 2.0, 1.0;

  auto malformed = magmaan::data::ordinal_pair_observed_table(xi, xj, 2, 2);
  REQUIRE_FALSE(malformed.has_value());
  CHECK(malformed.error().detail.find("finite integers") != std::string::npos);

  xi << nan, nan, nan;
  xj << 1.0, 2.0, nan;
  auto empty = magmaan::data::ordinal_pair_observed_table(xi, xj, 2, 2);
  REQUIRE_FALSE(empty.has_value());
  CHECK(empty.error().detail.find("no observed pairs") != std::string::npos);
}

TEST_CASE("Ordinal pair observed ML wrappers reuse observed-pair table counts") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd xi(8);
  Eigen::VectorXd xj(8);
  xi << 1.0, 2.0, 3.0, nan, 1.0, 2.0, 3.0, 2.0;
  xj << 1.0, 2.0, 2.0, 1.0, nan, 1.0, 1.0, 2.0;
  Eigen::VectorXd thi(2);
  thi << -0.4, 0.7;
  Eigen::VectorXd thj(1);
  thj << 0.2;

  auto table = magmaan::data::ordinal_pair_observed_table(xi, xj, 3, 2);
  REQUIRE(table.has_value());
  auto direct_rho = magmaan::data::fit_ordinal_pair_rho_ml(
      table->counts, thi, thj);
  auto observed_rho = magmaan::data::fit_ordinal_pair_observed_rho_ml(
      xi, xj, 3, 2, thi, thj);
  REQUIRE(direct_rho.has_value());
  REQUIRE(observed_rho.has_value());
  CHECK(observed_rho->n_obs == table->n_obs);
  CHECK(observed_rho->n_missing == table->n_missing);
  CHECK(observed_rho->counts.isApprox(table->counts, 0.0));
  CHECK(observed_rho->fit.rho == doctest::Approx(direct_rho->rho));
  CHECK(observed_rho->fit.negloglik == doctest::Approx(direct_rho->negloglik));

  auto direct_joint = magmaan::data::fit_ordinal_pair_joint_ml(table->counts);
  auto observed_joint = magmaan::data::fit_ordinal_pair_observed_joint_ml(
      xi, xj, 3, 2);
  REQUIRE(direct_joint.has_value());
  REQUIRE(observed_joint.has_value());
  CHECK(observed_joint->n_obs == table->n_obs);
  CHECK(observed_joint->n_missing == table->n_missing);
  CHECK(observed_joint->counts.isApprox(table->counts, 0.0));
  CHECK(observed_joint->fit.thresholds_i.isApprox(direct_joint->thresholds_i, 1e-12));
  CHECK(observed_joint->fit.thresholds_j.isApprox(direct_joint->thresholds_j, 1e-12));
  CHECK(observed_joint->fit.rho == doctest::Approx(direct_joint->rho));
  CHECK(observed_joint->fit.negloglik == doctest::Approx(direct_joint->negloglik));
}

TEST_CASE("Ordinal pair joint ML estimates pair-local thresholds and rho") {
  Eigen::VectorXd thi(2);
  thi << -0.55, 0.85;
  Eigen::VectorXd thj(2);
  thj << -0.25, 0.65;
  const double rho = 0.42;
  const Eigen::MatrixXd counts = ordinal_expected_counts(thi, thj, rho, 50000.0);

  auto joint = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  REQUIRE(joint.has_value());
  CHECK(joint->thresholds_i.size() == 2);
  CHECK(joint->thresholds_j.size() == 2);
  CHECK(joint->thresholds_i(0) == doctest::Approx(thi(0)).epsilon(5e-4));
  CHECK(joint->thresholds_i(1) == doctest::Approx(thi(1)).epsilon(5e-4));
  CHECK(joint->thresholds_j(0) == doctest::Approx(thj(0)).epsilon(5e-4));
  CHECK(joint->thresholds_j(1) == doctest::Approx(thj(1)).epsilon(5e-4));
  CHECK(joint->rho == doctest::Approx(rho).epsilon(5e-4));
  CHECK(joint->thresholds_i(0) < joint->thresholds_i(1));
  CHECK(joint->thresholds_j(0) < joint->thresholds_j(1));
  CHECK(std::isfinite(joint->negloglik));
  CHECK(joint->adjusted_counts.isApprox(counts, 0.0));
}

TEST_CASE("Ordinal pair joint h-weighted estimator preserves ML limits") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd thi(2);
  thi << -0.55, 0.85;
  Eigen::VectorXd thj(2);
  thj << -0.25, 0.65;
  const Eigen::MatrixXd counts = ordinal_expected_counts(thi, thj, 0.42, 50000.0);

  auto ml = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  auto h_ml = magmaan::data::fit_ordinal_pair_joint_h_weighted(counts);
  auto hard_inf = magmaan::data::fit_ordinal_pair_joint_h_weighted(
      counts, magmaan::data::OrdinalPairJointHWeightedOptions{
                  .h_score = PolychoricHScoreOptions{
                      .kind = PolychoricHScoreKind::WmaHardCap,
                      .k = std::numeric_limits<double>::infinity()}});
  REQUIRE(ml.has_value());
  REQUIRE(h_ml.has_value());
  REQUIRE(hard_inf.has_value());
  CHECK(h_ml->thresholds_i.isApprox(ml->thresholds_i, 1e-12));
  CHECK(h_ml->thresholds_j.isApprox(ml->thresholds_j, 1e-12));
  CHECK(h_ml->rho == doctest::Approx(ml->rho));
  CHECK(hard_inf->thresholds_i.isApprox(ml->thresholds_i, 1e-12));
  CHECK(hard_inf->thresholds_j.isApprox(ml->thresholds_j, 1e-12));
  CHECK(hard_inf->rho == doctest::Approx(ml->rho));
  CHECK(h_ml->converged);
  CHECK(hard_inf->converged);
  CHECK(std::isfinite(h_ml->objective));
  CHECK(h_ml->expected_counts.rows() == counts.rows());
  CHECK(h_ml->expected_counts.cols() == counts.cols());
  CHECK(h_ml->residual_counts.isApprox(
      h_ml->adjusted_counts - h_ml->expected_counts, 1e-12));
  CHECK(h_ml->pearson_residuals.allFinite());
  CHECK(h_ml->weights.isApprox(Eigen::MatrixXd::Ones(3, 3), 1e-12));
}

TEST_CASE("Ordinal pair joint h-weighted estimator downweights contaminated cells") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd th(2);
  th << -0.55, 0.75;
  const Eigen::MatrixXd clean = ordinal_expected_counts(th, th, 0.55, 5000.0);
  Eigen::MatrixXd contaminated = clean;
  contaminated(0, 2) += 900.0;

  auto clean_ml = magmaan::data::fit_ordinal_pair_joint_ml(clean);
  auto contaminated_ml =
      magmaan::data::fit_ordinal_pair_joint_ml(contaminated);
  auto robust = magmaan::data::fit_ordinal_pair_joint_h_weighted(
      contaminated, magmaan::data::OrdinalPairJointHWeightedOptions{
                        .h_score = PolychoricHScoreOptions{
                            .kind = PolychoricHScoreKind::WmaHardCap,
                            .k = 1.15}});
  REQUIRE(clean_ml.has_value());
  REQUIRE(contaminated_ml.has_value());
  REQUIRE(robust.has_value());
  CHECK(robust->thresholds_i(0) < robust->thresholds_i(1));
  CHECK(robust->thresholds_j(0) < robust->thresholds_j(1));
  CHECK(robust->rho > contaminated_ml->rho);
  CHECK(std::abs(robust->rho - clean_ml->rho) <
        std::abs(contaminated_ml->rho - clean_ml->rho));
  CHECK(robust->weights(0, 2) < 1.0);
  CHECK(robust->pearson_residuals(0, 2) > 0.0);
  CHECK(robust->expected_counts.sum() ==
        doctest::Approx(robust->adjusted_counts.sum()).epsilon(0.03));

  const PolychoricHScoreOptions h_options{
      .kind = PolychoricHScoreKind::WmaHardCap, .k = 1.15};
  const double objective = h_score_pair_objective(
      contaminated, robust->thresholds_i, robust->thresholds_j, robust->rho,
      h_options);
  CHECK(robust->objective == doctest::Approx(objective).epsilon(1e-12));
  CHECK(objective < h_score_pair_objective(
      contaminated, contaminated_ml->thresholds_i, contaminated_ml->thresholds_j,
      contaminated_ml->rho, h_options));

  for (Eigen::Index k = 0; k < robust->thresholds_i.size(); ++k) {
    Eigen::VectorXd plus = robust->thresholds_i;
    Eigen::VectorXd minus = robust->thresholds_i;
    plus(k) += 0.02;
    minus(k) -= 0.02;
    CHECK(objective <= h_score_pair_objective(
        contaminated, plus, robust->thresholds_j, robust->rho, h_options));
    CHECK(objective <= h_score_pair_objective(
        contaminated, minus, robust->thresholds_j, robust->rho, h_options));
  }
  for (Eigen::Index k = 0; k < robust->thresholds_j.size(); ++k) {
    Eigen::VectorXd plus = robust->thresholds_j;
    Eigen::VectorXd minus = robust->thresholds_j;
    plus(k) += 0.02;
    minus(k) -= 0.02;
    CHECK(objective <= h_score_pair_objective(
        contaminated, robust->thresholds_i, plus, robust->rho, h_options));
    CHECK(objective <= h_score_pair_objective(
        contaminated, robust->thresholds_i, minus, robust->rho, h_options));
  }
  CHECK(objective <= h_score_pair_objective(
      contaminated, robust->thresholds_i, robust->thresholds_j,
      robust->rho + 0.02, h_options));
  CHECK(objective <= h_score_pair_objective(
      contaminated, robust->thresholds_i, robust->thresholds_j,
      robust->rho - 0.02, h_options));
}

TEST_CASE("Ordinal pair joint DPD estimator preserves ML limit") {
  Eigen::VectorXd thi(2);
  thi << -0.55, 0.85;
  Eigen::VectorXd thj(2);
  thj << -0.25, 0.65;
  const Eigen::MatrixXd counts = ordinal_expected_counts(thi, thj, 0.42, 50000.0);

  auto ml = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  auto dpd0 = magmaan::data::fit_ordinal_pair_joint_dpd(
      counts, magmaan::data::OrdinalPairJointDpdOptions{.alpha = 0.0});
  REQUIRE(ml.has_value());
  REQUIRE(dpd0.has_value());
  CHECK(dpd0->thresholds_i.isApprox(ml->thresholds_i, 1e-12));
  CHECK(dpd0->thresholds_j.isApprox(ml->thresholds_j, 1e-12));
  CHECK(dpd0->rho == doctest::Approx(ml->rho));
  CHECK(dpd0->converged);
  CHECK(dpd0->weights.isApprox(Eigen::MatrixXd::Ones(3, 3), 1e-12));
  CHECK(dpd0->objective ==
        doctest::Approx(ml->negloglik / counts.sum()).epsilon(1e-12));

  auto bad = magmaan::data::fit_ordinal_pair_joint_dpd(
      counts, magmaan::data::OrdinalPairJointDpdOptions{.alpha = -0.1});
  REQUIRE_FALSE(bad.has_value());
  CHECK(bad.error().detail.find("invalid options") != std::string::npos);
}

TEST_CASE("Ordinal pair joint DPD estimator tempers low-probability cells") {
  Eigen::VectorXd th(2);
  th << -0.55, 0.75;
  const Eigen::MatrixXd clean = ordinal_expected_counts(th, th, 0.55, 5000.0);
  Eigen::MatrixXd contaminated = clean;
  contaminated(0, 2) += 900.0;

  auto clean_ml = magmaan::data::fit_ordinal_pair_joint_ml(clean);
  auto contaminated_ml =
      magmaan::data::fit_ordinal_pair_joint_ml(contaminated);
  auto dpd = magmaan::data::fit_ordinal_pair_joint_dpd(
      contaminated, magmaan::data::OrdinalPairJointDpdOptions{.alpha = 0.35});
  REQUIRE(clean_ml.has_value());
  REQUIRE(contaminated_ml.has_value());
  REQUIRE(dpd.has_value());
  CHECK(dpd->converged);
  CHECK(dpd->thresholds_i(0) < dpd->thresholds_i(1));
  CHECK(dpd->thresholds_j(0) < dpd->thresholds_j(1));
  CHECK(contaminated_ml->rho != doctest::Approx(clean_ml->rho));
  CHECK(dpd->rho != doctest::Approx(contaminated_ml->rho));
  CHECK(dpd->weights(0, 2) < dpd->weights(1, 1));
  CHECK(dpd->pearson_residuals(0, 2) > 0.0);

  const double objective = dpd_pair_objective(
      contaminated, dpd->thresholds_i, dpd->thresholds_j, dpd->rho, 0.35);
  CHECK(dpd->objective == doctest::Approx(objective).epsilon(1e-12));
  CHECK(objective < dpd_pair_objective(
      contaminated, contaminated_ml->thresholds_i, contaminated_ml->thresholds_j,
      contaminated_ml->rho, 0.35));
}

TEST_CASE("Ordinal pair h-weighted influence gives casewise sandwich rows") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::MatrixXd counts(3, 3);
  counts << 20.0, 10.0, 5.0,
             8.0, 25.0, 12.0,
             4.0, 14.0, 22.0;

  auto fit = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  REQUIRE(fit.has_value());
  auto influence = magmaan::data::ordinal_pair_h_weighted_influence(
      counts, fit->thresholds_i, fit->thresholds_j, fit->rho);
  REQUIRE(influence.has_value());
  const Eigen::MatrixXd ordinary_scores = ordinal_pair_score_rows_from_counts(
      counts, fit->thresholds_i, fit->thresholds_j, fit->rho);
  Eigen::MatrixXd centered_scores = ordinary_scores;
  centered_scores.rowwise() -= centered_scores.colwise().mean();
  CHECK(influence->n_obs == counts.sum());
  CHECK(influence->estimating_functions.isApprox(centered_scores, 1e-12));
  CHECK(influence->score_gamma.isApprox(
      (centered_scores.transpose() * centered_scores) /
          static_cast<double>(influence->n_obs),
      1e-12));
  CHECK(influence->bread.rows() == ordinary_scores.cols());
  CHECK(influence->bread.cols() == ordinary_scores.cols());
  CHECK(influence->influence.rows() == ordinary_scores.rows());
  CHECK(influence->influence.cols() == ordinary_scores.cols());
  CHECK(influence->gamma.isApprox(
      (influence->influence.transpose() * influence->influence) /
          static_cast<double>(influence->n_obs),
      1e-12));
  CHECK(influence->gamma.isApprox(influence->gamma.transpose(), 1e-12));
  CHECK(influence->gamma.allFinite());

  auto hard_inf = magmaan::data::ordinal_pair_h_weighted_influence(
      counts, fit->thresholds_i, fit->thresholds_j, fit->rho,
      magmaan::data::OrdinalPairHWeightedInfluenceOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = std::numeric_limits<double>::infinity()}});
  REQUIRE(hard_inf.has_value());
  CHECK(hard_inf->estimating_functions.isApprox(
      influence->estimating_functions, 1e-12));
  CHECK(hard_inf->gamma.isApprox(influence->gamma, 1e-12));
}

TEST_CASE("Ordinal pair h-weighted influence reports hard-cap kink convention") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::VectorXd th(1);
  th << 0.0;
  Eigen::MatrixXd counts(2, 2);
  counts << 16.0, 8.0,
             8.0, 8.0;
  auto influence = magmaan::data::ordinal_pair_h_weighted_influence(
      counts, th, th, 0.0,
      magmaan::data::OrdinalPairHWeightedInfluenceOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = 1.6}});
  REQUIRE(influence.has_value());
  CHECK(influence->ratios(0, 0) == doctest::Approx(1.6));
  CHECK(influence->h_values(0, 0) == doctest::Approx(1.6));
  CHECK(influence->dh_values(0, 0) == doctest::Approx(0.0));
  CHECK(influence->weights(0, 0) == doctest::Approx(1.0));
  CHECK(influence->ratios(0, 1) == doctest::Approx(0.8));
  CHECK(influence->dh_values(0, 1) == doctest::Approx(1.0));
}

TEST_CASE("Ordinal pair h-weighted influence downweights inflated cells and scales Gamma") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::MatrixXd counts(3, 3);
  counts << 594.0, 365.0, 858.0,
            453.0, 715.0, 372.0,
            311.0, 614.0, 718.0;
  auto robust = magmaan::data::fit_ordinal_pair_joint_h_weighted(
      counts, magmaan::data::OrdinalPairJointHWeightedOptions{
                  .h_score = PolychoricHScoreOptions{
                      .kind = PolychoricHScoreKind::WmaHardCap,
                      .k = 1.15}});
  REQUIRE(robust.has_value());
  auto influence = magmaan::data::ordinal_pair_h_weighted_influence(
      counts, robust->thresholds_i, robust->thresholds_j, robust->rho,
      magmaan::data::OrdinalPairHWeightedInfluenceOptions{
          .h_score = PolychoricHScoreOptions{
              .kind = PolychoricHScoreKind::WmaHardCap,
              .k = 1.15}});
  REQUIRE(influence.has_value());
  CHECK(influence->weights(0, 2) < 1.0);
  CHECK(influence->estimating_functions.rows() == counts.sum());
  CHECK(influence->estimating_functions.cols() == 5);
  CHECK(influence->score_gamma.isApprox(
      (influence->estimating_functions.transpose() *
       influence->estimating_functions) /
          static_cast<double>(influence->n_obs),
      1e-12));
  CHECK(influence->gamma.isApprox(
      (influence->influence.transpose() * influence->influence) /
          static_cast<double>(influence->n_obs),
      1e-12));
  CHECK(influence->gamma.allFinite());
  CHECK(influence->gamma.diagonal().minCoeff() > 0.0);
}

TEST_CASE("Ordinal pair joint ML rejects empty marginal categories") {
  Eigen::MatrixXd counts(3, 2);
  counts << 3.0, 2.0,
            0.0, 0.0,
            4.0, 5.0;

  auto joint = magmaan::data::fit_ordinal_pair_joint_ml(counts);
  REQUIRE_FALSE(joint.has_value());
  CHECK(joint.error().detail.find("marginal categories") != std::string::npos);
}

TEST_CASE("Pairwise ordinal stats wrap OrdinalStats and expose diagnostics") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto base = magmaan::data::ordinal_stats_from_integer_data({X});
  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(base.has_value());
  REQUIRE(pairwise.has_value());
  REQUIRE(pairwise->stats.R.size() == 1);
  CHECK(pairwise->stats.R[0].isApprox(base->R[0], 0.0));
  CHECK(pairwise->stats.thresholds[0].isApprox(base->thresholds[0], 0.0));
  CHECK(pairwise->stats.NACOV[0].isApprox(base->NACOV[0], 0.0));
  CHECK(pairwise->stats.W_dwls[0].isApprox(base->W_dwls[0], 0.0));
  CHECK(pairwise->stats.W_wls[0].isApprox(base->W_wls[0], 0.0));
  CHECK(pairwise->stats.n_obs == base->n_obs);
  CHECK(pairwise->stats.n_levels == base->n_levels);
  CHECK(pairwise->stats.threshold_ov == base->threshold_ov);
  CHECK(pairwise->stats.threshold_level == base->threshold_level);

  REQUIRE(pairwise->block_diagnostics.size() == 1);
  const auto& bd = pairwise->block_diagnostics[0];
  REQUIRE(bd.pair_diagnostics.size() == 3);
  CHECK(bd.moment_influence.rows() == 24);
  CHECK(bd.moment_influence.cols() == pairwise->stats.NACOV[0].rows());
  CHECK(bd.moment_influence.allFinite());
  CHECK(bd.gamma.isApprox(pairwise->stats.NACOV[0], 0.0));
  const Eigen::MatrixXd gamma_from_if =
      (bd.moment_influence.transpose() * bd.moment_influence) / 24.0;
  CHECK(gamma_from_if.isApprox(pairwise->stats.NACOV[0], 1e-10));
  CHECK(bd.pair_diagnostics[0].label.i == 1);
  CHECK(bd.pair_diagnostics[0].label.j == 0);
  CHECK(bd.pair_diagnostics[1].label.i == 2);
  CHECK(bd.pair_diagnostics[1].label.j == 0);
  CHECK(bd.pair_diagnostics[2].label.i == 2);
  CHECK(bd.pair_diagnostics[2].label.j == 1);
  for (const auto& pd : bd.pair_diagnostics) {
    CHECK(pd.label.block == 0);
    CHECK(pd.label.n_levels_i == 3);
    CHECK(pd.label.n_levels_j == 3);
    CHECK(std::isfinite(pd.rho));
    CHECK(std::isfinite(pd.negloglik));
    CHECK(pd.n_obs == 24);
    CHECK(pd.n_missing == 0);
    CHECK_FALSE(pd.ridge_applied);
    CHECK(pd.ridge == doctest::Approx(0.0));
    CHECK_FALSE(pd.shrinkage_applied);
    CHECK(pd.shrinkage_intensity == doctest::Approx(0.0));
    CHECK(pd.counts.rows() == 3);
    CHECK(pd.counts.cols() == 3);
    CHECK(pd.adjusted_counts.rows() == 3);
    CHECK(pd.adjusted_counts.cols() == 3);
    CHECK(pd.expected_counts.rows() == 3);
    CHECK(pd.expected_counts.cols() == 3);
    CHECK(pd.residual_counts.rows() == 3);
    CHECK(pd.residual_counts.cols() == 3);
    CHECK(pd.expected_counts.allFinite());
    CHECK(pd.residual_counts.allFinite());
    CHECK(pd.expected_counts.sum() == doctest::Approx(pd.adjusted_counts.sum()));
    CHECK(pd.residual_counts.sum() == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(pd.residual_counts.isApprox(pd.adjusted_counts - pd.expected_counts, 1e-12));
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(pairwise->stats.R[0]);
  REQUIRE(es.info() == Eigen::Success);
  CHECK(bd.min_eigen_r == doctest::Approx(es.eigenvalues().minCoeff()));
}

TEST_CASE("Pairwise ordinal stats diagnostics report lavaan 2x2 adjustment") {
  Eigen::MatrixXd X(15, 2);
  Eigen::Index r = 0;
  for (int k = 0; k < 4; ++k) X.row(r++) << 1, 2;
  for (int k = 0; k < 5; ++k) X.row(r++) << 2, 1;
  for (int k = 0; k < 6; ++k) X.row(r++) << 2, 2;

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());
  REQUIRE(pairwise->block_diagnostics.size() == 1);
  REQUIRE(pairwise->block_diagnostics[0].pair_diagnostics.size() == 1);
  CHECK(pairwise->block_diagnostics[0].gamma.isApprox(pairwise->stats.NACOV[0], 0.0));
  const auto& pd = pairwise->block_diagnostics[0].pair_diagnostics[0];
  CHECK(pd.counts(0, 0) == doctest::Approx(0.0));
  CHECK(pd.counts(0, 1) == doctest::Approx(5.0));
  CHECK(pd.counts(1, 0) == doctest::Approx(4.0));
  CHECK(pd.counts(1, 1) == doctest::Approx(6.0));
  CHECK(pd.adjusted_counts(0, 0) == doctest::Approx(0.5));
  CHECK(pd.adjusted_counts(0, 1) == doctest::Approx(4.5));
  CHECK(pd.adjusted_counts(1, 0) == doctest::Approx(3.5));
  CHECK(pd.adjusted_counts(1, 1) == doctest::Approx(6.5));
  CHECK(pd.expected_counts.rows() == 2);
  CHECK(pd.expected_counts.cols() == 2);
  CHECK(pd.expected_counts.sum() == doctest::Approx(pd.adjusted_counts.sum()));
  CHECK(pd.residual_counts.isApprox(pd.adjusted_counts - pd.expected_counts, 1e-12));
  CHECK(pd.n_obs == 15);
  CHECK(pd.n_missing == 0);
  CHECK_FALSE(pd.ridge_applied);
  CHECK_FALSE(pd.shrinkage_applied);
  CHECK(std::isfinite(pd.rho));
  CHECK(std::isfinite(pairwise->block_diagnostics[0].min_eigen_r));
}

TEST_CASE("Pairwise ordinal h-weighted stats preserve ML limit") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto base = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  auto robust =
      magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data({X});
  REQUIRE(base.has_value());
  REQUIRE(robust.has_value());
  CHECK(robust->stats.thresholds[0].isApprox(base->stats.thresholds[0], 0.0));
  CHECK(robust->stats.R[0].isApprox(base->stats.R[0], 1e-7));
  CHECK(robust->stats.NACOV[0].rows() == base->stats.NACOV[0].rows());
  CHECK(robust->stats.NACOV[0].cols() == base->stats.NACOV[0].cols());
  CHECK(robust->stats.NACOV[0].allFinite());
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(robust->block_diagnostics[0].gamma.isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  for (const auto& pd : robust->block_diagnostics[0].pair_diagnostics) {
    CHECK_FALSE(pd.h_weighted);
    CHECK(pd.converged);
    CHECK(pd.weights.rows() == pd.counts.rows());
    CHECK(pd.weights.cols() == pd.counts.cols());
  }

  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto fit = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, robust->stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  CHECK(std::isfinite(fit->fmin));
}

TEST_CASE("Pairwise ordinal h-weighted stats jointly estimate shared thresholds and Gamma") {
  using magmaan::data::PolychoricHScoreKind;
  using magmaan::data::PolychoricHScoreOptions;

  Eigen::MatrixXd counts(3, 3);
  counts << 594.0, 365.0, 858.0,
            453.0, 715.0, 372.0,
            311.0, 614.0, 718.0;
  const Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  auto base = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(base.has_value());

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions options;
  options.rho.h_score = PolychoricHScoreOptions{
      .kind = PolychoricHScoreKind::WmaHardCap,
      .k = 1.15};
  auto robust = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, options);
  REQUIRE(robust.has_value());

  CHECK_FALSE(robust->stats.thresholds[0].isApprox(
      base->stats.thresholds[0], 1e-6));
  CHECK(std::abs(robust->stats.R[0](1, 0) - base->stats.R[0](1, 0)) > 1e-4);
  CHECK(robust->stats.NACOV[0].isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  CHECK_FALSE(robust->stats.NACOV[0].isApprox(base->stats.NACOV[0], 1e-6));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  REQUIRE(robust->block_diagnostics[0].pair_diagnostics.size() == 1);
  const auto& pd = robust->block_diagnostics[0].pair_diagnostics[0];
  CHECK(pd.h_weighted);
  CHECK(pd.weights.rows() == counts.cols());
  CHECK(pd.weights.cols() == counts.rows());
  CHECK(pd.weights(2, 0) < 1.0);
  CHECK(pd.expected_counts.sum() == doctest::Approx(pd.adjusted_counts.sum()));
  CHECK(pd.expected_counts.allFinite());
}

TEST_CASE("Pairwise ordinal DPD stats jointly estimate shared thresholds") {
  Eigen::MatrixXd counts(3, 3);
  counts << 594.0, 365.0, 858.0,
            453.0, 715.0, 372.0,
            311.0, 614.0, 718.0;
  const Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  auto base = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(base.has_value());

  magmaan::data::PairwiseOrdinalDpdStatsOptions options;
  options.alpha = 0.35;
  auto robust = magmaan::data::pairwise_ordinal_stats_dpd_from_integer_data(
      {X}, options);
  REQUIRE(robust.has_value());
  options.alpha = 0.0;
  auto ml_limit = magmaan::data::pairwise_ordinal_stats_dpd_from_integer_data(
      {X}, options);
  REQUIRE(ml_limit.has_value());
  CHECK(ml_limit->stats.thresholds[0].isApprox(base->stats.thresholds[0], 0.0));
  CHECK(ml_limit->stats.R[0].isApprox(base->stats.R[0], 0.0));
  CHECK_FALSE(robust->stats.thresholds[0].isApprox(
      base->stats.thresholds[0], 1e-6));
  CHECK(std::abs(robust->stats.R[0](1, 0) - base->stats.R[0](1, 0)) > 1e-4);
  CHECK(robust->stats.NACOV[0].isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  REQUIRE(robust->block_diagnostics[0].pair_diagnostics.size() == 1);
  const auto& pd = robust->block_diagnostics[0].pair_diagnostics[0];
  CHECK_FALSE(pd.h_weighted);
  CHECK(pd.weights.minCoeff() > 0.0);
  CHECK(pd.expected_counts.sum() == doctest::Approx(pd.adjusted_counts.sum()));
}

TEST_CASE("Pairwise ordinal h-weighted stats repair low-eigen robust R on request") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions raw_options;
  raw_options.rho.h_score = magmaan::data::PolychoricHScoreOptions{
      .kind = magmaan::data::PolychoricHScoreKind::WmaHardCap,
      .k = 1.30};
  auto raw = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, raw_options);
  REQUIRE(raw.has_value());
  const auto& raw_bd = raw->block_diagnostics[0];
  CHECK(raw_bd.raw_min_eigen_r == doctest::Approx(raw_bd.min_eigen_r));
  CHECK_FALSE(raw_bd.r_repair_applied);
  CHECK(raw_bd.r_ridge == doctest::Approx(0.0));
  CHECK(raw_bd.r_shrinkage_intensity == doctest::Approx(0.0));
  REQUIRE(raw_bd.min_eigen_r < 0.95);

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions err_options;
  err_options.rho.h_score = raw_options.rho.h_score;
  err_options.correlation_repair.kind =
      magmaan::data::PairwiseOrdinalCorrelationRepairKind::Error;
  err_options.correlation_repair.min_eigenvalue = 0.95;
  auto err = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, err_options);
  REQUIRE_FALSE(err.has_value());
  CHECK(err.error().detail.find("minimum eigenvalue") != std::string::npos);

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions shrink_options;
  shrink_options.rho.h_score = raw_options.rho.h_score;
  shrink_options.correlation_repair.kind =
      magmaan::data::PairwiseOrdinalCorrelationRepairKind::Shrinkage;
  shrink_options.correlation_repair.min_eigenvalue = 0.95;
  auto shrunk = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, shrink_options);
  REQUIRE(shrunk.has_value());
  const auto& shrunk_bd = shrunk->block_diagnostics[0];
  CHECK(shrunk_bd.raw_min_eigen_r == doctest::Approx(raw_bd.raw_min_eigen_r));
  CHECK(shrunk_bd.min_eigen_r == doctest::Approx(0.95).epsilon(1e-10));
  CHECK(shrunk_bd.r_repair_applied);
  CHECK(shrunk_bd.r_ridge == doctest::Approx(0.0));
  REQUIRE(shrunk_bd.r_shrinkage_intensity > 0.0);
  CHECK(shrunk_bd.r_shrinkage_intensity < 1.0);
  CHECK(shrunk->stats.R[0].diagonal().isOnes(1e-12));
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> shrunk_es(shrunk->stats.R[0]);
  REQUIRE(shrunk_es.info() == Eigen::Success);
  CHECK(shrunk_es.eigenvalues().minCoeff() == doctest::Approx(0.95).epsilon(1e-10));
  CHECK(shrunk->stats.NACOV[0].isApprox(
      (shrunk_bd.moment_influence.transpose() *
       shrunk_bd.moment_influence) /
          static_cast<double>(shrunk->stats.n_obs[0]),
      1e-12));
  for (const auto& pd : shrunk_bd.pair_diagnostics) {
    CHECK_FALSE(pd.ridge_applied);
    CHECK(pd.ridge == doctest::Approx(0.0));
    CHECK(pd.shrinkage_applied);
    CHECK(pd.shrinkage_intensity ==
          doctest::Approx(shrunk_bd.r_shrinkage_intensity));
  }

  magmaan::data::PairwiseOrdinalHWeightedStatsOptions ridge_options;
  ridge_options.rho.h_score = raw_options.rho.h_score;
  ridge_options.correlation_repair.kind =
      magmaan::data::PairwiseOrdinalCorrelationRepairKind::Ridge;
  ridge_options.correlation_repair.min_eigenvalue = 0.95;
  auto ridged = magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
      {X}, ridge_options);
  REQUIRE(ridged.has_value());
  const auto& ridged_bd = ridged->block_diagnostics[0];
  CHECK(ridged_bd.raw_min_eigen_r == doctest::Approx(raw_bd.raw_min_eigen_r));
  CHECK(ridged_bd.min_eigen_r == doctest::Approx(0.95).epsilon(1e-10));
  CHECK(ridged_bd.r_repair_applied);
  CHECK(ridged_bd.r_ridge > 0.0);
  CHECK(ridged_bd.r_shrinkage_intensity == doctest::Approx(0.0));
  CHECK(ridged->stats.R[0].isApprox(shrunk->stats.R[0], 1e-12));
  for (const auto& pd : ridged_bd.pair_diagnostics) {
    CHECK(pd.ridge_applied);
    CHECK(pd.ridge == doctest::Approx(ridged_bd.r_ridge));
    CHECK_FALSE(pd.shrinkage_applied);
    CHECK(pd.shrinkage_intensity == doctest::Approx(0.0));
  }
}

TEST_CASE("Pairwise ordinal composite objective uses pair diagnostics and explicit scaling") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());

  auto obj = magmaan::estimate::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, pairwise->stats.R);
  REQUIRE(obj.has_value());
  REQUIRE(obj->blocks.size() == 1);
  REQUIRE(obj->blocks[0].pairs.size() == 3);

  double expected_nll = 0.0;
  for (const auto& pd : pairwise->block_diagnostics[0].pair_diagnostics) {
    expected_nll += pd.negloglik;
  }
  CHECK(obj->negloglik == doctest::Approx(expected_nll));
  CHECK(obj->weighted_negloglik == doctest::Approx(expected_nll));
  CHECK(obj->scaling_denominator == doctest::Approx(72.0));
  CHECK(obj->objective == doctest::Approx(expected_nll / 72.0));
  CHECK_FALSE(obj->reports_chisq);
  CHECK(obj->df == -1);
  CHECK_FALSE(obj->blocks[0].reports_chisq);
  CHECK(obj->blocks[0].df == -1);

  for (const auto& pair : obj->blocks[0].pairs) {
    CHECK(pair.n_obs == 24);
    CHECK(pair.n_missing == 0);
    CHECK(pair.scaling_weight == doctest::Approx(24.0));
    CHECK(pair.expected_counts.sum() == doctest::Approx(pair.counts.sum()));
    CHECK(pair.residual_counts.isApprox(pair.counts - pair.expected_counts, 1e-12));
  }

  auto sum_obj = magmaan::estimate::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, pairwise->stats.R,
      magmaan::estimate::PairwiseOrdinalCompositeOptions{
          .weighting = magmaan::estimate::PairwiseCompositeWeighting::ObservedPairCount,
          .scaling = magmaan::estimate::PairwiseCompositeScaling::SumNegLogLik});
  REQUIRE(sum_obj.has_value());
  CHECK(sum_obj->objective == doctest::Approx(expected_nll));
}

TEST_CASE("Pairwise ordinal composite objective validates implied pair mapping") {
  Eigen::MatrixXd X(15, 2);
  Eigen::Index r = 0;
  for (int k = 0; k < 4; ++k) X.row(r++) << 1, 1;
  for (int k = 0; k < 3; ++k) X.row(r++) << 1, 2;
  for (int k = 0; k < 2; ++k) X.row(r++) << 2, 1;
  for (int k = 0; k < 6; ++k) X.row(r++) << 2, 2;

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());

  auto implied_r = pairwise->stats.R;
  implied_r[0](1, 0) = implied_r[0](0, 1) = 0.25;
  auto obj = magmaan::estimate::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, implied_r);
  REQUIRE(obj.has_value());
  REQUIRE(obj->blocks[0].pairs.size() == 1);
  CHECK(obj->blocks[0].pairs[0].rho == doctest::Approx(0.25));

  implied_r[0](1, 0) = implied_r[0](0, 1) = 1.0;
  auto bad_rho = magmaan::estimate::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, implied_r);
  REQUIRE_FALSE(bad_rho.has_value());
  CHECK(bad_rho.error().detail.find("inside (-1, 1)") != std::string::npos);

  auto bad_thresholds = pairwise->stats.thresholds;
  bad_thresholds[0](0) = std::numeric_limits<double>::quiet_NaN();
  auto bad_th = magmaan::estimate::pairwise_ordinal_composite_objective(
      *pairwise, bad_thresholds, pairwise->stats.R);
  REQUIRE_FALSE(bad_th.has_value());
  CHECK(bad_th.error().detail.find("non-finite") != std::string::npos);
}

TEST_CASE("Pairwise ordinal joint composite objective fits complete pair-local margins") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());
  auto joint = magmaan::estimate::pairwise_ordinal_joint_composite_objective(
      *pairwise);
  REQUIRE(joint.has_value());
  REQUIRE(joint->blocks.size() == 1);
  REQUIRE(joint->blocks[0].pairs.size() == 3);
  CHECK_FALSE(joint->reports_chisq);
  CHECK(joint->df == -1);

  double expected_nll = 0.0;
  for (std::size_t k = 0; k < joint->blocks[0].pairs.size(); ++k) {
    const auto& pd = pairwise->block_diagnostics[0].pair_diagnostics[k];
    const auto direct = magmaan::data::fit_ordinal_pair_joint_ml(pd.counts);
    REQUIRE(direct.has_value());
    const auto& pair = joint->blocks[0].pairs[k];
    expected_nll += direct->negloglik;
    CHECK(pair.label.i == pd.label.i);
    CHECK(pair.label.j == pd.label.j);
    CHECK(pair.thresholds_i.isApprox(direct->thresholds_i, 1e-12));
    CHECK(pair.thresholds_j.isApprox(direct->thresholds_j, 1e-12));
    CHECK(pair.rho == doctest::Approx(direct->rho));
    CHECK(pair.negloglik == doctest::Approx(direct->negloglik));
    CHECK(pair.counts.isApprox(pd.counts, 0.0));
    CHECK(pair.adjusted_counts.isApprox(direct->adjusted_counts, 0.0));
    CHECK(pair.expected_counts.sum() == doctest::Approx(pair.adjusted_counts.sum()));
    CHECK(pair.residual_counts.isApprox(pair.adjusted_counts - pair.expected_counts, 1e-12));
    CHECK(pair.scaling_weight == doctest::Approx(24.0));
    const Eigen::Index n_score_cols =
        pair.thresholds_i.size() + pair.thresholds_j.size() + 1;
    CHECK(pair.score_contributions.rows() == pair.n_obs);
    CHECK(pair.score_contributions.cols() == n_score_cols);
    CHECK(pair.score_contributions.allFinite());
    CHECK(pair.score_gamma.rows() == n_score_cols);
    CHECK(pair.score_gamma.cols() == n_score_cols);
    CHECK(pair.score_gamma.isApprox(
        (pair.score_contributions.transpose() * pair.score_contributions) /
            static_cast<double>(pair.n_obs),
        1e-12));
  }
  CHECK(joint->negloglik == doctest::Approx(expected_nll));
  CHECK(joint->weighted_negloglik == doctest::Approx(expected_nll));
  CHECK(joint->scaling_denominator == doctest::Approx(72.0));
  CHECK(joint->objective == doctest::Approx(expected_nll / 72.0));
}

TEST_CASE("Pairwise ordinal joint composite objective preserves lavaan 2x2 adjustment choice") {
  Eigen::MatrixXd X(15, 2);
  Eigen::Index r = 0;
  for (int k = 0; k < 4; ++k) X.row(r++) << 1, 2;
  for (int k = 0; k < 5; ++k) X.row(r++) << 2, 1;
  for (int k = 0; k < 6; ++k) X.row(r++) << 2, 2;

  auto pairwise = magmaan::data::pairwise_ordinal_stats_from_integer_data({X});
  REQUIRE(pairwise.has_value());
  auto adjusted = magmaan::estimate::pairwise_ordinal_joint_composite_objective(
      *pairwise);
  auto raw = magmaan::estimate::pairwise_ordinal_joint_composite_objective(
      *pairwise,
      magmaan::estimate::PairwiseOrdinalCompositeOptions{
          .lavaan_adjust_2x2 = false});
  REQUIRE(adjusted.has_value());
  REQUIRE(raw.has_value());
  REQUIRE(adjusted->blocks[0].pairs.size() == 1);
  REQUIRE(raw->blocks[0].pairs.size() == 1);

  const auto& adj_pair = adjusted->blocks[0].pairs[0];
  const auto& raw_pair = raw->blocks[0].pairs[0];
  CHECK(adj_pair.counts.isApprox(raw_pair.counts, 0.0));
  CHECK(adj_pair.adjusted_counts(0, 0) == doctest::Approx(0.5));
  CHECK(raw_pair.adjusted_counts(0, 0) == doctest::Approx(0.0));
  CHECK(adj_pair.negloglik != doctest::Approx(raw_pair.negloglik));
}

TEST_CASE("Pairwise ordinal observed joint composite objective preserves pairwise missingness") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(8, 3);
  X << 1.0, 1.0, 1.0,
       2.0, 2.0, 1.0,
       3.0, 1.0, 2.0,
       nan, 2.0, 2.0,
       1.0, nan, 1.0,
       2.0, 1.0, nan,
       3.0, 2.0, 1.0,
       nan, nan, 2.0;
  std::vector<std::vector<std::int32_t>> levels{{3, 2, 2}};

  auto observed =
      magmaan::estimate::pairwise_ordinal_observed_joint_composite_objective(
          {X}, levels);
  REQUIRE(observed.has_value());
  REQUIRE(observed->blocks.size() == 1);
  REQUIRE(observed->blocks[0].pairs.size() == 3);
  CHECK(observed->blocks[0].n_obs == 8);
  CHECK_FALSE(observed->reports_chisq);
  CHECK(observed->df == -1);

  double expected_nll = 0.0;
  std::int64_t expected_pair_n = 0;
  for (const auto& pair : observed->blocks[0].pairs) {
    auto direct = magmaan::data::fit_ordinal_pair_observed_joint_ml(
        X.col(pair.label.i), X.col(pair.label.j),
        pair.label.n_levels_i, pair.label.n_levels_j);
    REQUIRE(direct.has_value());
    expected_nll += direct->fit.negloglik;
    expected_pair_n += direct->n_obs;
    CHECK(pair.thresholds_i.isApprox(direct->fit.thresholds_i, 1e-12));
    CHECK(pair.thresholds_j.isApprox(direct->fit.thresholds_j, 1e-12));
    CHECK(pair.rho == doctest::Approx(direct->fit.rho));
    CHECK(pair.negloglik == doctest::Approx(direct->fit.negloglik));
    CHECK(pair.n_obs == direct->n_obs);
    CHECK(pair.n_missing == direct->n_missing);
    CHECK(pair.n_obs == 5);
    CHECK(pair.n_missing == 3);
    CHECK(pair.counts.isApprox(direct->counts, 0.0));
    CHECK(pair.adjusted_counts.isApprox(direct->fit.adjusted_counts, 0.0));
    CHECK(pair.expected_counts.sum() ==
          doctest::Approx(pair.adjusted_counts.sum()).epsilon(0.03));
    CHECK(pair.residual_counts.isApprox(pair.adjusted_counts - pair.expected_counts, 1e-12));
    const Eigen::Index n_score_cols =
        pair.thresholds_i.size() + pair.thresholds_j.size() + 1;
    CHECK(pair.score_contributions.rows() == pair.n_obs);
    CHECK(pair.score_contributions.cols() == n_score_cols);
    CHECK(pair.score_contributions.allFinite());
    CHECK(pair.score_gamma.rows() == n_score_cols);
    CHECK(pair.score_gamma.cols() == n_score_cols);
    CHECK(pair.score_gamma.isApprox(
        (pair.score_contributions.transpose() * pair.score_contributions) /
            static_cast<double>(pair.n_obs),
        1e-12));
  }
  CHECK(observed->negloglik == doctest::Approx(expected_nll));
  CHECK(observed->weighted_negloglik == doctest::Approx(expected_nll));
  CHECK(observed->scaling_denominator == doctest::Approx(static_cast<double>(expected_pair_n)));
  CHECK(observed->objective == doctest::Approx(expected_nll / static_cast<double>(expected_pair_n)));
}

TEST_CASE("Pairwise ordinal observed joint composite objective rejects invalid observed pairs") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::vector<std::int32_t>> levels{{2, 2}};

  Eigen::MatrixXd all_missing(3, 2);
  all_missing << nan, 1.0,
                 nan, 2.0,
                 nan, nan;
  auto missing =
      magmaan::estimate::pairwise_ordinal_observed_joint_composite_objective(
          {all_missing}, levels);
  REQUIRE_FALSE(missing.has_value());
  CHECK(missing.error().detail.find("no observed pairs") != std::string::npos);

  Eigen::MatrixXd empty_margin(3, 2);
  empty_margin << 1.0, 1.0,
                  1.0, 2.0,
                  nan, 1.0;
  auto empty =
      magmaan::estimate::pairwise_ordinal_observed_joint_composite_objective(
          {empty_margin}, levels);
  REQUIRE_FALSE(empty.has_value());
  CHECK(empty.error().detail.find("marginal categories") != std::string::npos);
}

TEST_CASE("Ordinal stats: thresholds, polychoric R, and weights have expected shapes") {
  Eigen::MatrixXd X(320, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 5; ++rep) {
    for (int x1 = 1; x1 <= 4; ++x1) {
      for (int x2 = 1; x2 <= 4; ++x2) {
        for (int x3 = 1; x3 <= 4; ++x3) {
          X(r, 0) = x1;
          X(r, 1) = x2;
          X(r, 2) = x3;
          ++r;
        }
      }
    }
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  REQUIRE(stats->R.size() == 1);
  CHECK(stats->R[0].rows() == 3);
  CHECK(stats->R[0].cols() == 3);
  CHECK(stats->R[0].diagonal().isOnes(1e-12));
  CHECK(std::abs(stats->R[0](1, 0)) < 1e-6);
  CHECK(std::abs(stats->R[0](2, 0)) < 1e-6);
  CHECK(std::abs(stats->R[0](2, 1)) < 1e-6);
  CHECK(stats->thresholds[0].size() == 9);
  CHECK(stats->thresholds[0](0) == doctest::Approx(-0.67448975).epsilon(1e-7));
  CHECK(stats->thresholds[0](1) == doctest::Approx(0.0).epsilon(1e-12));
  CHECK(stats->thresholds[0](2) == doctest::Approx(0.67448975).epsilon(1e-7));
  CHECK(stats->threshold_ov[0].size() == 9);
  CHECK(stats->threshold_ov[0] == std::vector<std::int32_t>({0, 0, 0, 1, 1, 1, 2, 2, 2}));
  CHECK(stats->threshold_level[0] == std::vector<std::int32_t>({1, 2, 3, 1, 2, 3, 1, 2, 3}));
  CHECK(stats->NACOV[0].rows() == 12);
  CHECK(stats->NACOV[0].cols() == 12);
  CHECK(stats->W_dwls[0].rows() == 12);
  CHECK(stats->W_dwls[0].cols() == 12);
  CHECK(stats->W_wls[0].rows() == 12);
  CHECK(stats->W_wls[0].cols() == 12);
  CHECK((stats->W_wls[0] * stats->NACOV[0])
            .isApprox(Eigen::MatrixXd::Identity(12, 12), 1e-8));
  for (Eigen::Index k = 0; k < 12; ++k) {
    CHECK(stats->W_dwls[0](k, k) == doctest::Approx(1.0 / stats->NACOV[0](k, k)));
  }
  CHECK(stats->n_obs[0] == 320);
}

TEST_CASE("Ordinal stats: metadata and moments use documented order") {
  Eigen::MatrixXd X(24, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int c = 1; c <= 3; ++c) {
      X(r, 0) = c;
      X(r, 1) = 1 + ((c + rep) % 3);
      X(r, 2) = 1 + ((2 * c + rep) % 3);
      ++r;
      X(r, 0) = c;
      X(r, 1) = 1 + ((2 * c + rep) % 3);
      X(r, 2) = 1 + ((c + rep) % 3);
      ++r;
    }
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  CHECK(stats->n_obs[0] == 24);
  CHECK(stats->n_levels[0] == std::vector<std::int32_t>({3, 3, 3}));
  CHECK(stats->threshold_ov[0] == std::vector<std::int32_t>({0, 0, 1, 1, 2, 2}));
  CHECK(stats->threshold_level[0] == std::vector<std::int32_t>({1, 2, 1, 2, 1, 2}));

  Eigen::VectorXd moments(9);
  moments.head(6) = stats->thresholds[0];
  moments(6) = stats->R[0](1, 0);
  moments(7) = stats->R[0](2, 0);
  moments(8) = stats->R[0](2, 1);
  CHECK(moments.allFinite());
  CHECK(stats->NACOV[0].rows() == moments.size());
  CHECK(stats->W_dwls[0].rows() == moments.size());
  CHECK(stats->W_wls[0].rows() == moments.size());
}

TEST_CASE("Ordinal stats: empty marginal categories are explicit errors") {
  Eigen::MatrixXd X(4, 2);
  X << 1, 1,
       1, 2,
       3, 2,
       3, 3;
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  CHECK_FALSE(stats.has_value());
}

TEST_CASE("Ordinal stats: near-empty categories stay finite") {
  Eigen::MatrixXd X(200, 3);
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    X(r, 0) = r == 0 ? 1 : (r + 1 == X.rows() ? 5 : 2 + (r % 3));
    X(r, 1) = 1 + ((2 * r + r / 7) % 5);
    X(r, 2) = 1 + ((3 * r + r / 11) % 5);
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  CHECK(stats->thresholds[0].allFinite());
  CHECK(stats->R[0].allFinite());
  CHECK(stats->NACOV[0].allFinite());
  CHECK(stats->W_dwls[0].allFinite());
  CHECK(stats->W_wls[0].allFinite());
  CHECK(stats->n_levels[0] == std::vector<std::int32_t>({5, 5, 5}));
}

TEST_CASE("Ordinal rows round-trip through lavaan-shaped partables and matrix_rep ignores them") {
  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  int threshold_rows = 0;
  int scale_rows = 0;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->op[i] == magmaan::parse::Op::Threshold) {
      ++threshold_rows;
      CHECK_FALSE(mr->cell_for_row[i].used);
    }
    if (pt->op[i] == magmaan::parse::Op::ResponseScale) {
      ++scale_rows;
      CHECK_FALSE(mr->cell_for_row[i].used);
    }
  }
  CHECK(threshold_rows == 6);
  CHECK(scale_rows == 3);
}

TEST_CASE("Ordinal delta preparation fixes response variances and compacts free indices") {
  Eigen::MatrixXd X(320, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 5; ++rep) {
    for (int x1 = 1; x1 <= 4; ++x1) {
      for (int x2 = 1; x2 <= 4; ++x2) {
        for (int x3 = 1; x3 <= 4; ++x3) {
          X(r, 0) = x1;
          X(r, 1) = x2;
          X(r, 2) = x3;
          ++r;
        }
      }
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2 + t3\n"
      "x2 | t1 + t2 + t3\n"
      "x3 | t1 + t2 + t3\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  magmaan::spec::Starts starts;
  auto pt = magmaan::spec::lavaanify(*fp, {}, &starts);
  REQUIRE(pt.has_value());
  const std::int32_t old_n = pt->n_free();
  starts.hint.resize(static_cast<std::size_t>(old_n),
                     std::numeric_limits<double>::quiet_NaN());
  for (std::int32_t k = 1; k <= old_n; ++k) {
    starts.hint[static_cast<std::size_t>(k - 1)] = static_cast<double>(k);
  }

  std::int32_t old_latent_var_free = 0;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->op[i] == magmaan::parse::Op::Covariance &&
        pt->lhs_var[i] == pt->rhs_var[i] && pt->lhs_var[i] >= 0 &&
        pt->is_user_latent[static_cast<std::size_t>(pt->lhs_var[i])] != 0) {
      old_latent_var_free = pt->free[i];
    }
  }
  REQUIRE(old_latent_var_free > 0);

  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(*pt, *stats, &starts);
  REQUIRE(prep.has_value());
  CHECK(pt->n_free() == old_n - 3);
  CHECK(starts.hint.size() == static_cast<std::size_t>(pt->n_free()));
  CHECK(starts.hint.back() == doctest::Approx(static_cast<double>(old_latent_var_free)));

  int fixed_response_variances = 0;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->op[i] != magmaan::parse::Op::Covariance ||
        pt->lhs_var[i] != pt->rhs_var[i] || pt->lhs_var[i] < 0) {
      continue;
    }
    const bool is_observed =
        pt->ov_pos[static_cast<std::size_t>(pt->lhs_var[i])] >= 0;
    const bool is_latent =
        pt->is_user_latent[static_cast<std::size_t>(pt->lhs_var[i])] != 0;
    if (is_observed) {
      ++fixed_response_variances;
      CHECK(pt->free[i] == 0);
      CHECK(pt->fixed_value[i] == doctest::Approx(1.0));
    }
    if (is_latent) {
      CHECK(pt->free[i] == pt->n_free());
    }
  }
  CHECK(fixed_response_variances == 3);
}

TEST_CASE("Ordinal theta parameterization is a valid reparameterization of delta") {
  // Delta and Theta fit the same model to the same data — they are
  // reparameterizations, so the discrepancy at the optimum (fmin / χ²) must
  // agree. Theta differs only in the fit objective: the implied latent-
  // response moments are standardized before comparison with the polychorics.
  std::mt19937 rng(20260518);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(600, 4);
  const double loading[4] = {0.9, 0.8, 0.7, 0.6};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.6) + (y > 0.5);   // 3 categories
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  using magmaan::estimate::OrdinalParameterization;
  using magmaan::estimate::OrdinalWeightKind;

  // The prepared partable is parameterization-independent.
  auto prep = magmaan::estimate::prepare_ordinal_partable(
      *pt, *stats, OrdinalParameterization::Theta);
  REQUIRE(prep.has_value());

  auto delta = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::Lbfgs, {}, OrdinalParameterization::Delta);
  auto theta = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::Lbfgs, {}, OrdinalParameterization::Theta);
  REQUIRE_MESSAGE(delta.has_value(),
      "delta fit failed: " << (delta.has_value() ? "" : delta.error().detail));
  REQUIRE_MESSAGE(theta.has_value(),
      "theta fit failed: " << (theta.has_value() ? "" : theta.error().detail));

  CHECK(std::isfinite(theta->fmin));
  CHECK(theta->iterations > 0);
  // Reparameterization invariance: same minimized discrepancy.
  CHECK(theta->fmin == doctest::Approx(delta->fmin).epsilon(1e-4));
  // The parameter vectors themselves differ — theta loadings are unstandardized.
  CHECK((theta->theta - delta->theta).cwiseAbs().maxCoeff() > 1e-3);
}

TEST_CASE("Ordinal robust reporting returns sandwich SEs and scaled-test eigenvalues") {
  std::mt19937 rng(20240514);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(800, 4);
  const double loading[4] = {0.95, 0.85, 0.75, 0.65};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.7) + (y > 0.0) + (y > 0.7);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2 + t3\n"
      "x2 | t1 + t2 + t3\n"
      "x3 | t1 + t2 + t3\n"
      "x4 | t1 + t2 + t3\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto fit = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  auto rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(rob.has_value());

  CHECK(rob->vcov.rows() == fit->theta.size());
  CHECK(rob->vcov.cols() == fit->theta.size());
  CHECK(rob->se.size() == fit->theta.size());
  CHECK(rob->se.allFinite());
  CHECK(rob->df == 2);
  CHECK(rob->eigvals.size() == rob->df);
  CHECK(rob->eigvals.minCoeff() > 0.0);
  CHECK(rob->chisq_standard == doctest::Approx(800.0 * fit->fmin));
  CHECK(rob->satorra_bentler.df == rob->df);
  CHECK(std::isfinite(rob->satorra_bentler.scale_c));
  CHECK(std::isfinite(rob->mean_var_adjusted.df_adj));
  CHECK(std::isfinite(rob->scaled_shifted.scale_a));
}

TEST_CASE("Mixed ordinal stats and DWLS fit use continuous and threshold moments") {
  std::mt19937 rng(20240515);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(600, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    const double e1 = norm(rng);
    const double e2 = norm(rng);
    const double y1 = 0.8 * eta + 0.6 * e1;
    const double y2 = 0.7 * eta + 0.7 * e2;
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.4);
    X(i, 1) = 1.0 + (0.65 * eta + 0.76 * norm(rng) > 0.1);
    X(i, 2) = y1 + 0.2;
    X(i, 3) = y2 - 0.1;
  }
  std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());
  REQUIRE(stats->R.size() == 1);
  CHECK(stats->thresholds[0].size() == 3);
  CHECK(stats->mean[0].size() == 4);
  CHECK(stats->moments[0].size() == 13);
  CHECK(stats->NACOV[0].rows() == 13);
  CHECK(stats->W_dwls[0].rows() == 13);
  CHECK(stats->W_wls[0].rows() == 13);
  CHECK(stats->R[0](0, 0) == doctest::Approx(1.0));
  CHECK(stats->R[0](1, 1) == doctest::Approx(1.0));
  CHECK(stats->R[0](2, 2) > 0.0);
  CHECK(stats->R[0](3, 3) > 0.0);

  magmaan::spec::LavaanifyOptions opts;
  opts.meanstructure = true;

  const char* th_syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n";
  auto fpt = magmaan::parse::Parser::parse(th_syntax);
  REQUIRE(fpt.has_value());
  auto pt2 = magmaan::spec::lavaanify(*fpt, opts);
  REQUIRE(pt2.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt2);
  REQUIRE(mr.has_value());
  auto fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt2, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  CHECK(fit->theta.allFinite());
  CHECK(std::isfinite(fit->fmin));

  auto rob = magmaan::estimate::robust_mixed_ordinal(
      *pt2, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(rob.has_value());
  CHECK(rob->vcov.rows() == fit->theta.size());
  CHECK(rob->se.size() == fit->theta.size());
  CHECK(rob->se.allFinite());
}

TEST_CASE("Mixed ordinal polyserial DPD keeps shared marginals and fits DWLS") {
  std::mt19937 rng(20260517);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.4);
    X(i, 1) = 1.0 + (0.65 * eta + 0.76 * norm(rng) > 0.1);
    X(i, 2) = 0.8 * eta + 0.6 * norm(rng) + 0.2;
    X(i, 3) = 0.7 * eta + 0.7 * norm(rng) - 0.1;
  }
  for (Eigen::Index i = 0; i < 28; ++i) {
    X(i, 0) = 1.0;
    X(i, 2) = 7.0 + 0.02 * static_cast<double>(i);
  }
  std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};

  auto base = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  auto ml_limit = magmaan::data::mixed_ordinal_stats_polyserial_dpd_from_data(
      {X}, ordered, magmaan::data::PolyserialPairDpdOptions{.alpha = 0.0});
  auto robust = magmaan::data::mixed_ordinal_stats_polyserial_dpd_from_data(
      {X}, ordered, magmaan::data::PolyserialPairDpdOptions{.alpha = 0.45});
  REQUIRE(base.has_value());
  REQUIRE(ml_limit.has_value());
  REQUIRE(robust.has_value());

  CHECK(ml_limit->stats.thresholds[0].isApprox(base->thresholds[0], 0.0));
  CHECK(ml_limit->stats.R[0].isApprox(base->R[0], 1e-10));
  CHECK(ml_limit->stats.NACOV[0].isApprox(base->NACOV[0], 1e-8));
  CHECK(robust->stats.thresholds[0].isApprox(base->thresholds[0], 0.0));
  CHECK(robust->stats.mean[0].isApprox(base->mean[0], 0.0));
  CHECK(robust->stats.R[0](0, 0) == doctest::Approx(base->R[0](0, 0)));
  CHECK(robust->stats.R[0](1, 1) == doctest::Approx(base->R[0](1, 1)));
  CHECK(robust->stats.R[0](2, 2) == doctest::Approx(base->R[0](2, 2)));
  CHECK(robust->stats.R[0](3, 3) == doctest::Approx(base->R[0](3, 3)));
  CHECK(std::abs(robust->stats.R[0](2, 0) - base->R[0](2, 0)) > 1e-4);
  CHECK(robust->stats.NACOV[0].isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  REQUIRE(robust->block_diagnostics.size() == 1);
  CHECK(robust->block_diagnostics[0].dpd_pairs.size() == 4);
  CHECK(robust->block_diagnostics[0].dpd_fits.size() == 4);
  CHECK(robust->block_diagnostics[0].dpd_fits[0].weights.allFinite());

  magmaan::spec::LavaanifyOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, robust->stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  CHECK(fit->theta.allFinite());
  CHECK(std::isfinite(fit->fmin));
}

TEST_CASE("Huber residual clipping API covers hard, smooth, Tukey, and no-clip") {
  using magmaan::data::HuberResidualClipKind;
  using magmaan::data::HuberResidualClipOptions;

  auto none = magmaan::data::eval_huber_residual_clip(
      2.0, HuberResidualClipOptions{.kind = HuberResidualClipKind::None});
  REQUIRE(none.has_value());
  CHECK(none->psi == doctest::Approx(2.0));
  CHECK(none->dpsi == doctest::Approx(1.0));

  auto hard = magmaan::data::eval_huber_residual_clip(
      2.0, HuberResidualClipOptions{
               .kind = HuberResidualClipKind::HardHuber, .k = 1.25});
  REQUIRE(hard.has_value());
  CHECK(hard->psi == doctest::Approx(1.25));
  CHECK(hard->dpsi == doctest::Approx(0.0));
  CHECK(hard->weight == doctest::Approx(0.625));

  auto pseudo = magmaan::data::eval_huber_residual_clip(
      2.0, HuberResidualClipOptions{
               .kind = HuberResidualClipKind::PseudoHuber, .k = 1.0});
  REQUIRE(pseudo.has_value());
  CHECK(pseudo->psi == doctest::Approx(2.0 / std::sqrt(5.0)));
  CHECK(pseudo->dpsi == doctest::Approx(1.0 / (std::sqrt(5.0) * 5.0)));

  auto tukey = magmaan::data::eval_huber_residual_clip(
      2.0, HuberResidualClipOptions{
               .kind = HuberResidualClipKind::TukeyBiweight, .k = 1.5});
  REQUIRE(tukey.has_value());
  CHECK(tukey->psi == doctest::Approx(0.0));
  CHECK(tukey->dpsi == doctest::Approx(0.0));
}

TEST_CASE("Mixed ordinal Huber residual no-clip preserves single-ordinal ML Gamma") {
  std::mt19937 rng(20260519);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(360, 3);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.65) + (eta > 0.45);
    X(i, 1) = 0.8 * eta + 0.6 * norm(rng);
    X(i, 2) = 0.55 * eta + 0.84 * norm(rng);
  }
  for (Eigen::Index i = 0; i < 18; ++i) {
    X(i, 0) = 1.0;
    X(i, 1) = 5.5 + 0.02 * static_cast<double>(i);
  }
  std::vector<std::vector<std::int32_t>> ordered = {{1, 0, 0}};

  auto base = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  auto none = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      {X}, ordered,
      magmaan::data::MixedOrdinalHuberResidualOptions{
          .clip = magmaan::data::HuberResidualClipOptions{
              .kind = magmaan::data::HuberResidualClipKind::None},
          .correlation_repair = {}});
  auto robust = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      {X}, ordered,
      magmaan::data::MixedOrdinalHuberResidualOptions{
          .clip = magmaan::data::HuberResidualClipOptions{
              .kind = magmaan::data::HuberResidualClipKind::HardHuber,
              .k = 1.345},
          .correlation_repair = {}});
  REQUIRE(base.has_value());
  REQUIRE(none.has_value());
  REQUIRE(robust.has_value());

  CHECK(none->stats.thresholds[0].isApprox(base->thresholds[0], 0.0));
  CHECK(none->stats.R[0].isApprox(base->R[0], 1e-12));
  CHECK(none->stats.NACOV[0].isApprox(base->NACOV[0], 1e-10));
  CHECK(none->block_diagnostics[0].gamma.isApprox(
      (none->block_diagnostics[0].moment_influence.transpose() *
       none->block_diagnostics[0].moment_influence) /
          static_cast<double>(none->stats.n_obs[0]),
      1e-12));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(robust->block_diagnostics[0].robust_pairs.size() == 2);
  CHECK_FALSE(robust->stats.NACOV[0].isApprox(base->NACOV[0], 1e-6));
}

TEST_CASE("Mixed ordinal Huber residual stats rebuild Gamma and preserve continuous moments") {
  std::mt19937 rng(20260518);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(420, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.55) + (eta > 0.55);
    X(i, 1) = 1.0 + (0.7 * eta + 0.72 * norm(rng) > 0.0);
    X(i, 2) = 0.8 * eta + 0.6 * norm(rng);
    X(i, 3) = 0.6 * eta + 0.8 * norm(rng);
  }
  for (Eigen::Index i = 0; i < 20; ++i) {
    X(i, 0) = 1.0;
    X(i, 1) = 2.0;
    X(i, 2) = 6.0 + 0.01 * static_cast<double>(i);
  }
  std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};

  auto base = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  auto none = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      {X}, ordered,
      magmaan::data::MixedOrdinalHuberResidualOptions{
          .clip = magmaan::data::HuberResidualClipOptions{
              .kind = magmaan::data::HuberResidualClipKind::None},
          .correlation_repair = {}});
  auto robust = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
      {X}, ordered,
      magmaan::data::MixedOrdinalHuberResidualOptions{
          .clip = magmaan::data::HuberResidualClipOptions{
              .kind = magmaan::data::HuberResidualClipKind::HardHuber,
              .k = 1.345},
          .correlation_repair = {}});
  REQUIRE(base.has_value());
  REQUIRE(none.has_value());
  REQUIRE(robust.has_value());

  CHECK(none->stats.mean[0].isApprox(base->mean[0], 0.0));
  CHECK(robust->stats.mean[0].isApprox(base->mean[0], 0.0));
  CHECK(robust->stats.R[0](2, 2) == doctest::Approx(base->R[0](2, 2)));
  CHECK(robust->stats.R[0](3, 3) == doctest::Approx(base->R[0](3, 3)));
  CHECK(std::abs(robust->stats.R[0](2, 0) - base->R[0](2, 0)) > 1e-5);
  CHECK(robust->stats.NACOV[0].isApprox(
      (robust->block_diagnostics[0].moment_influence.transpose() *
       robust->block_diagnostics[0].moment_influence) /
          static_cast<double>(robust->stats.n_obs[0]),
      1e-12));
  CHECK(robust->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(robust->block_diagnostics[0].robust_pairs.size() == 5);
  CHECK(robust->block_diagnostics[0].rho.size() == 5);

  magmaan::spec::LavaanifyOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, robust->stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  CHECK(fit->theta.allFinite());
}

TEST_CASE("Mixed ordinal validation rejects malformed stats before fitting") {
  std::mt19937 rng(20240516);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(240, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.4);
    X(i, 1) = 1.0 + (0.65 * eta + 0.76 * norm(rng) > 0.1);
    X(i, 2) = 0.8 * eta + 0.6 * norm(rng);
    X(i, 3) = 0.7 * eta + 0.7 * norm(rng);
  }
  auto stats = magmaan::data::mixed_ordinal_stats_from_data(
      {X}, std::vector<std::vector<std::int32_t>>{{1, 1, 0, 0}});
  REQUIRE(stats.has_value());

  magmaan::spec::LavaanifyOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto expect_error = [&](magmaan::data::MixedOrdinalStats bad,
                          const char* needle) {
    // Call the core directly: malformed stats must be rejected by
    // validate_stats, which runs before the x0 size check — so an empty x0
    // is fine here.
    auto fit = magmaan::estimate::fit_mixed_ordinal_bounded(
        *pt, *mr, bad, {}, magmaan::estimate::OrdinalWeightKind::DWLS,
        Eigen::VectorXd{});
    REQUIRE_FALSE(fit.has_value());
    CHECK(fit.error().detail.find(needle) != std::string::npos);
  };

  {
    auto bad = *stats;
    bad.ordered[0][0] = 2;
    expect_error(std::move(bad), "ordered mask");
  }
  {
    auto bad = *stats;
    bad.ordered[0][3] = 1;
    expect_error(std::move(bad), "missing thresholds");
  }
  {
    auto bad = *stats;
    bad.threshold_ov[0][0] = 2;
    expect_error(std::move(bad), "continuous variable");
  }
  {
    auto bad = *stats;
    bad.NACOV[0](0, 0) = -1.0;
    expect_error(std::move(bad), "NACOV diagonal");
  }
  {
    auto bad = *stats;
    bad.mean[0](0) = std::numeric_limits<double>::quiet_NaN();
    expect_error(std::move(bad), "non-finite");
  }
}

TEST_CASE("Mixed ordinal theta parameterization fails explicitly") {
  Eigen::MatrixXd X(80, 3);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    X(i, 0) = 1.0 + (i % 3);
    X(i, 1) = static_cast<double>(i) / 10.0;
    X(i, 2) = std::sin(static_cast<double>(i));
  }
  auto stats = magmaan::data::mixed_ordinal_stats_from_data(
      {X}, std::vector<std::vector<std::int32_t>>{{1, 0, 0}});
  REQUIRE(stats.has_value());

  magmaan::spec::LavaanifyOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x1 ~*~ 1*x1\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp, opts);
  REQUIRE(pt.has_value());
  auto prep = magmaan::estimate::prepare_mixed_ordinal_partable(
      *pt, *stats, magmaan::estimate::OrdinalParameterization::Theta);
  REQUIRE_FALSE(prep.has_value());
  CHECK(prep.error().detail.find("mixed ordinal theta parameterization") !=
        std::string::npos);
}
