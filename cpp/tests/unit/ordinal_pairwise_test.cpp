#include "ordinal_test_helpers.hpp"

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
  auto pt = magmaan::spec::build(*fp);
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

TEST_CASE("Pairwise ordinal Huber residual stats preserve ML limit and downweight contamination") {
  Eigen::MatrixXd clean_counts(3, 3);
  clean_counts << 250.0, 120.0, 35.0,
                  110.0, 300.0, 120.0,
                  25.0, 130.0, 260.0;
  Eigen::MatrixXd contaminated_counts = clean_counts;
  contaminated_counts(0, 2) += 900.0;
  const Eigen::MatrixXd clean = ordinal_data_from_pair_counts(clean_counts);
  const Eigen::MatrixXd contaminated =
      ordinal_data_from_pair_counts(contaminated_counts);

  auto clean_ml = magmaan::data::pairwise_ordinal_stats_from_integer_data({clean});
  auto contaminated_ml =
      magmaan::data::pairwise_ordinal_stats_from_integer_data({contaminated});
  magmaan::data::PairwiseOrdinalHuberResidualStatsOptions huber_ml_options;
  huber_ml_options.clip.kind = magmaan::data::HuberResidualClipKind::None;
  auto huber_ml = magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
      {contaminated}, huber_ml_options);
  magmaan::data::PairwiseOrdinalHuberResidualStatsOptions huber_options;
  huber_options.clip.kind = magmaan::data::HuberResidualClipKind::PseudoHuber;
  huber_options.clip.k = 1.25;
  auto huber = magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
      {contaminated}, huber_options);
  magmaan::data::PairwiseOrdinalHWeightedStatsOptions h_options;
  h_options.rho.h_score = magmaan::data::PolychoricHScoreOptions{
      .kind = magmaan::data::PolychoricHScoreKind::WmaHardCap,
      .k = 1.15};
  auto h_weighted =
      magmaan::data::pairwise_ordinal_stats_h_weighted_from_integer_data(
          {contaminated}, h_options);
  magmaan::data::PairwiseOrdinalDpdStatsOptions dpd_options;
  dpd_options.alpha = 0.35;
  auto dpd = magmaan::data::pairwise_ordinal_stats_dpd_from_integer_data(
      {contaminated}, dpd_options);
  REQUIRE(clean_ml.has_value());
  REQUIRE(contaminated_ml.has_value());
  REQUIRE(huber_ml.has_value());
  REQUIRE(huber.has_value());
  REQUIRE(h_weighted.has_value());
  REQUIRE(dpd.has_value());

  CHECK(huber_ml->stats.thresholds[0].isApprox(
      contaminated_ml->stats.thresholds[0], 0.0));
  CHECK(huber_ml->stats.R[0].isApprox(contaminated_ml->stats.R[0], 0.0));
  CHECK(huber_ml->stats.NACOV[0].isApprox(contaminated_ml->stats.NACOV[0], 0.0));

  const double contaminated_r = contaminated_ml->stats.R[0](1, 0);
  const double huber_r = huber->stats.R[0](1, 0);
  CHECK(std::isfinite(huber_r));
  CHECK(std::abs(huber_r - contaminated_r) > 1e-4);
  CHECK(std::abs(h_weighted->stats.R[0](1, 0) - contaminated_r) > 1e-4);
  CHECK(std::abs(dpd->stats.R[0](1, 0) - contaminated_r) > 1e-4);
  CHECK(huber->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(h_weighted->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  CHECK(dpd->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  REQUIRE(huber->block_diagnostics[0].pair_diagnostics.size() == 1);
  const auto& pd = huber->block_diagnostics[0].pair_diagnostics[0];
  CHECK(pd.weights.minCoeff() < 1.0);
  CHECK(pd.pearson_residuals.cwiseAbs().maxCoeff() > 1.25);
  CHECK(huber->stats.NACOV[0].isApprox(
      (huber->block_diagnostics[0].moment_influence.transpose() *
       huber->block_diagnostics[0].moment_influence) /
          static_cast<double>(huber->stats.n_obs[0]),
      1e-12));
}

TEST_CASE("Pairwise ordinal Huber residual repair keeps correlation influence stable") {
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

  magmaan::data::PairwiseOrdinalHuberResidualStatsOptions raw_options;
  raw_options.clip = magmaan::data::HuberResidualClipOptions{
      .kind = magmaan::data::HuberResidualClipKind::HardHuber,
      .k = 1.20};
  auto raw = magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
      {X}, raw_options);
  REQUIRE(raw.has_value());
  const auto& raw_bd = raw->block_diagnostics[0];
  REQUIRE(raw_bd.min_eigen_r < 0.95);

  auto ridge_options = raw_options;
  ridge_options.correlation_repair.kind =
      magmaan::data::PairwiseOrdinalCorrelationRepairKind::Ridge;
  ridge_options.correlation_repair.min_eigenvalue = 0.95;
  auto ridged =
      magmaan::data::pairwise_ordinal_stats_huber_residual_from_integer_data(
          {X}, ridge_options);
  REQUIRE(ridged.has_value());
  const auto& ridged_bd = ridged->block_diagnostics[0];
  CHECK(ridged_bd.raw_min_eigen_r == doctest::Approx(raw_bd.raw_min_eigen_r));
  CHECK(ridged_bd.min_eigen_r == doctest::Approx(0.95).epsilon(1e-10));
  CHECK(ridged_bd.r_repair_applied);
  REQUIRE(ridged_bd.r_ridge > 0.0);

  const Eigen::Index nth = ridged->stats.thresholds[0].size();
  const Eigen::Index ncorr = ridged->stats.R[0].cols() *
      (ridged->stats.R[0].cols() - 1) / 2;
  const double corr_scale = 1.0 / (1.0 + ridged_bd.r_ridge);
  CHECK(ridged_bd.moment_influence.leftCols(nth).isApprox(
      raw_bd.moment_influence.leftCols(nth), 1e-12));
  CHECK(ridged_bd.moment_influence.rightCols(ncorr).isApprox(
      corr_scale * raw_bd.moment_influence.rightCols(ncorr), 1e-12));
  CHECK(ridged->stats.NACOV[0].isApprox(
      (ridged_bd.moment_influence.transpose() *
       ridged_bd.moment_influence) /
          static_cast<double>(ridged->stats.n_obs[0]),
      1e-12));
  CHECK(ridged->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
  for (const auto& pd : ridged_bd.pair_diagnostics) {
    CHECK(pd.ridge_applied);
    CHECK(pd.ridge == doctest::Approx(ridged_bd.r_ridge));
  }
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

  auto obj = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
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

  auto sum_obj = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, pairwise->stats.R,
      magmaan::estimate::frontier::PairwiseOrdinalCompositeOptions{
          .weighting = magmaan::estimate::frontier::PairwiseCompositeWeighting::ObservedPairCount,
          .scaling = magmaan::estimate::frontier::PairwiseCompositeScaling::SumNegLogLik});
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
  auto obj = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, implied_r);
  REQUIRE(obj.has_value());
  REQUIRE(obj->blocks[0].pairs.size() == 1);
  CHECK(obj->blocks[0].pairs[0].rho == doctest::Approx(0.25));

  implied_r[0](1, 0) = implied_r[0](0, 1) = 1.0;
  auto bad_rho = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
      *pairwise, pairwise->stats.thresholds, implied_r);
  REQUIRE_FALSE(bad_rho.has_value());
  CHECK(bad_rho.error().detail.find("inside (-1, 1)") != std::string::npos);

  auto bad_thresholds = pairwise->stats.thresholds;
  bad_thresholds[0](0) = std::numeric_limits<double>::quiet_NaN();
  auto bad_th = magmaan::estimate::frontier::pairwise_ordinal_composite_objective(
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
  auto joint = magmaan::estimate::frontier::pairwise_ordinal_joint_composite_objective(
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
  auto adjusted = magmaan::estimate::frontier::pairwise_ordinal_joint_composite_objective(
      *pairwise);
  auto raw = magmaan::estimate::frontier::pairwise_ordinal_joint_composite_objective(
      *pairwise,
      magmaan::estimate::frontier::PairwiseOrdinalCompositeOptions{
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
      magmaan::estimate::frontier::pairwise_ordinal_observed_joint_composite_objective(
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
      magmaan::estimate::frontier::pairwise_ordinal_observed_joint_composite_objective(
          {all_missing}, levels);
  REQUIRE_FALSE(missing.has_value());
  CHECK(missing.error().detail.find("no observed pairs") != std::string::npos);

  Eigen::MatrixXd empty_margin(3, 2);
  empty_margin << 1.0, 1.0,
                  1.0, 2.0,
                  nan, 1.0;
  auto empty =
      magmaan::estimate::frontier::pairwise_ordinal_observed_joint_composite_objective(
          {empty_margin}, levels);
  REQUIRE_FALSE(empty.has_value());
  CHECK(empty.error().detail.find("marginal categories") != std::string::npos);
}

TEST_CASE("Pairwise ordinal composite fit and Godambe handle observed-pair missingness") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::mt19937 rng(20260618);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(90, 3);
  const double loading[3] = {0.85, 0.75, 0.65};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.35) + (y > 0.55);
    }
  }
  X(3, 0) = nan;
  X(14, 1) = nan;
  X(29, 2) = nan;
  X(51, 0) = nan;

  std::vector<std::vector<std::int32_t>> levels{{3, 3, 3}};
  auto data = magmaan::estimate::frontier::pairwise_ordinal_observed_data(
      {X}, levels);
  REQUIRE(data.has_value());
  REQUIRE(data->saturated.blocks.size() == 1);
  REQUIRE(data->saturated.blocks[0].pairs.size() == 3);
  bool saw_missing = false;
  for (const auto& pair : data->saturated.blocks[0].pairs) {
    saw_missing = saw_missing || pair.n_missing > 0;
  }
  CHECK(saw_missing);

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
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 80;
  auto fit = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      *pt, *mr, *data, {}, {}, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "pairwise composite fit failed: "
          << (fit.has_value() ? "" : fit.error().detail));
  CHECK(std::isfinite(fit->objective.negloglik));
  CHECK(fit->objective.negloglik <= data->saturated.negloglik + 100.0);

  auto god = magmaan::estimate::frontier::pairwise_ordinal_composite_godambe(
      *pt, *mr, *data, fit->estimates, 2e-5);
  REQUIRE_MESSAGE(god.has_value(),
      "pairwise composite Godambe failed: "
          << (god.has_value() ? "" : god.error().detail));
  CHECK(god->vcov.rows() == fit->estimates.theta.size());
  CHECK(god->vcov.cols() == fit->estimates.theta.size());
  CHECK(god->se.size() == fit->estimates.theta.size());
  CHECK(god->casewise_scores.rows() == X.rows());
  CHECK(god->casewise_scores.cols() == fit->estimates.theta.size());
  CHECK(god->vcov.allFinite());
  CHECK(god->se.allFinite());
}

TEST_CASE("Pairwise ordinal composite nested LR reports Satorra spectrum") {
  std::mt19937 rng(20260619);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(110, 3);
  const double loading[3] = {0.82, 0.70, 0.70};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.40) + (y > 0.50);
    }
  }

  auto data = magmaan::estimate::frontier::pairwise_ordinal_observed_data(
      {X}, {{3, 3, 3}});
  REQUIRE(data.has_value());
  const char* h1_syntax =
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  const char* h0_syntax =
      "f =~ x1 + a*x2 + a*x3\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n";
  auto fp1 = magmaan::parse::Parser::parse(h1_syntax);
  auto fp0 = magmaan::parse::Parser::parse(h0_syntax);
  REQUIRE(fp1.has_value());
  REQUIRE(fp0.has_value());
  auto pt1 = magmaan::spec::build(*fp1);
  auto pt0 = magmaan::spec::build(*fp0);
  REQUIRE(pt1.has_value());
  REQUIRE(pt0.has_value());
  auto mr1 = magmaan::model::build_matrix_rep(*pt1);
  auto mr0 = magmaan::model::build_matrix_rep(*pt0);
  REQUIRE(mr1.has_value());
  REQUIRE(mr0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 80;
  auto fit1 = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      *pt1, *mr1, *data, {}, {}, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto fit0 = magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      *pt0, *mr0, *data, {}, {}, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit1.has_value(),
      "H1 pairwise composite fit failed: "
          << (fit1.has_value() ? "" : fit1.error().detail));
  REQUIRE_MESSAGE(fit0.has_value(),
      "H0 pairwise composite fit failed: "
          << (fit0.has_value() ? "" : fit0.error().detail));

  auto lr = magmaan::estimate::frontier::lr_test_pairwise_ordinal_composite(
      *pt1, *mr1, *data, *fit1, *pt0, *mr0, *fit0,
      magmaan::robust::SatorraAMethod::Exact, 2e-5);
  REQUIRE_MESSAGE(lr.has_value(),
      "pairwise composite LR failed: "
          << (lr.has_value() ? "" : lr.error().detail));
  CHECK(lr->df_diff == 1);
  CHECK(lr->T_diff == doctest::Approx(
      2.0 * (fit0->objective.negloglik - fit1->objective.negloglik)));
  CHECK(lr->eigenvalues.size() == 1);
  CHECK(std::isfinite(lr->p_scaled));
  CHECK(std::isfinite(lr->p_adjusted));
  CHECK(std::isfinite(lr->p_mixture));

  auto god0 = magmaan::estimate::frontier::pairwise_ordinal_composite_godambe(
      *pt0, *mr0, *data, fit0->estimates, 2e-5);
  REQUIRE_MESSAGE(god0.has_value(),
      "constrained pairwise composite Godambe failed: "
          << (god0.has_value() ? "" : god0.error().detail));
  CHECK(god0->vcov.rows() == fit0->estimates.theta.size());
  CHECK(god0->vcov.cols() == fit0->estimates.theta.size());
  CHECK(god0->se.allFinite());
}
