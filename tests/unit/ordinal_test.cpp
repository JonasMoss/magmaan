#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

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

}  // namespace

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

  auto nll = magmaan::data::continuous_pair_normal_negloglik(
      x, y, fit->mean_i, fit->mean_j, fit->var_i, fit->var_j, fit->cov);
  REQUIRE(nll.has_value());
  CHECK(*nll == doctest::Approx(fit->negloglik));

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

TEST_CASE("Ordinal theta parameterization fails explicitly") {
  Eigen::MatrixXd X(16, 2);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 4; ++rep) {
    for (int x1 = 1; x1 <= 2; ++x1) {
      for (int x2 = 1; x2 <= 2; ++x2) {
        X(r, 0) = x1;
        X(r, 1) = x2;
        ++r;
      }
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2\n"
      "x1 | t1\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto prep = magmaan::estimate::prepare_ordinal_partable(
      *pt, *stats, magmaan::estimate::OrdinalParameterization::Theta);
  REQUIRE_FALSE(prep.has_value());
  CHECK(prep.error().detail.find("theta parameterization") != std::string::npos);
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

  auto fit = magmaan::estimate::fit_ordinal_bounded(
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
  auto fit = magmaan::estimate::fit_mixed_ordinal_bounded(
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
    auto fit = magmaan::estimate::fit_mixed_ordinal_bounded(
        *pt, *mr, bad, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
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
