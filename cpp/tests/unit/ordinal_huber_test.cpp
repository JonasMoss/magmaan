#include "ordinal_test_helpers.hpp"

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

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, robust->stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(fit.has_value());
  CHECK(fit->theta.allFinite());
}

TEST_CASE("Mixed ordinal Huber residual Monte Carlo keeps sparse contaminated Gamma stable") {
  std::normal_distribution<double> norm(0.0, 1.0);
  std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  double clean_to_ml = 0.0;
  double clean_to_huber = 0.0;
  double huber_log_condition = 0.0;
  int stable_reps = 0;

  for (int rep = 0; rep < 6; ++rep) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(20260600 + rep));
    Eigen::MatrixXd clean(300, 4);
    for (Eigen::Index i = 0; i < clean.rows(); ++i) {
      const double eta = norm(rng);
      const double z1 = 0.78 * eta + 0.63 * norm(rng);
      const double z2 = 0.62 * eta + 0.78 * norm(rng);
      clean(i, 0) = 1.0 + (z1 > -1.65) + (z1 > -0.25) + (z1 > 1.35);
      clean(i, 1) = 1.0 + (z2 > -1.15) + (z2 > 0.35);
      clean(i, 2) = 0.75 * eta + 0.66 * norm(rng);
      clean(i, 3) = 0.55 * eta + 0.84 * norm(rng);
    }

    Eigen::MatrixXd contaminated(clean.rows() + 24, clean.cols());
    contaminated.topRows(clean.rows()) = clean;
    for (Eigen::Index i = 0; i < 24; ++i) {
      contaminated(clean.rows() + i, 0) = 1.0;
      contaminated(clean.rows() + i, 1) = 3.0;
      contaminated(clean.rows() + i, 2) = 6.0 + 0.03 * static_cast<double>(i);
      contaminated(clean.rows() + i, 3) = -5.5 - 0.02 * static_cast<double>(i);
    }

    auto clean_stats = magmaan::data::mixed_ordinal_stats_from_data(
        {clean}, ordered);
    auto base = magmaan::data::mixed_ordinal_stats_from_data(
        {contaminated}, ordered);
    auto huber = magmaan::data::mixed_ordinal_stats_huber_residual_from_data(
        {contaminated}, ordered,
        magmaan::data::MixedOrdinalHuberResidualOptions{
            .clip = magmaan::data::HuberResidualClipOptions{
                .kind = magmaan::data::HuberResidualClipKind::HardHuber,
                .k = 1.20},
            .correlation_repair =
                magmaan::data::MixedOrdinalCorrelationRepairOptions{
                    .kind = magmaan::data::MixedOrdinalCorrelationRepairKind::Ridge,
                    .min_eigenvalue = 1e-6}});
    auto dpd = magmaan::data::mixed_ordinal_stats_polyserial_dpd_from_data(
        {contaminated}, ordered,
        magmaan::data::PolyserialPairDpdOptions{.alpha = 0.35});
    REQUIRE(clean_stats.has_value());
    REQUIRE(base.has_value());
    REQUIRE(huber.has_value());
    REQUIRE(dpd.has_value());

    const Eigen::MatrixXd& gamma = huber->stats.NACOV[0];
    CHECK(gamma.allFinite());
    CHECK(huber->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
    CHECK(huber->stats.W_dwls[0].diagonal().allFinite());
    CHECK(huber->block_diagnostics[0].gamma.isApprox(
        (huber->block_diagnostics[0].moment_influence.transpose() *
         huber->block_diagnostics[0].moment_influence) /
            static_cast<double>(huber->stats.n_obs[0]),
        1e-12));
    CHECK(dpd->stats.W_dwls[0].diagonal().minCoeff() > 0.0);
    CHECK(dpd->stats.W_dwls[0].diagonal().allFinite());

    const double base_cond = symmetric_condition_number(base->NACOV[0]);
    const double huber_cond = symmetric_condition_number(gamma);
    CHECK(std::isfinite(base_cond));
    CHECK(std::isfinite(huber_cond));
    CHECK(huber_cond < 1e14);
    huber_log_condition += std::log(huber_cond);

    const double clean_r = clean_stats->R[0](2, 0);
    const double base_r = base->R[0](2, 0);
    const double huber_r = huber->stats.R[0](2, 0);
    clean_to_ml += std::abs(base_r - clean_r);
    clean_to_huber += std::abs(huber_r - clean_r);
    CHECK(std::abs(huber_r - base_r) > 1e-4);
    CHECK(huber->block_diagnostics[0].objective.allFinite());
    ++stable_reps;
  }

  REQUIRE(stable_reps == 6);
  CHECK(clean_to_huber < clean_to_ml);
  CHECK(huber_log_condition / static_cast<double>(stable_reps) < std::log(1e8));
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

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
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

TEST_CASE("Mixed ordinal theta parameterization fits and supports post-fit reporting") {
  std::mt19937 rng(20260520);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.4);
    X(i, 1) = 1.0 + (0.65 * eta + 0.76 * norm(rng) > 0.1);
    X(i, 2) = 0.8 * eta + 0.6 * norm(rng) + 0.2;
    X(i, 3) = 0.7 * eta + 0.7 * norm(rng) - 0.1;
  }
  auto stats = magmaan::data::mixed_ordinal_stats_from_data(
      {X}, std::vector<std::vector<std::int32_t>>{{1, 1, 0, 0}});
  REQUIRE(stats.has_value());

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  using magmaan::estimate::OrdinalParameterization;
  using magmaan::estimate::OrdinalWeightKind;
  auto delta = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::NloptLbfgs, {}, OrdinalParameterization::Delta);
  auto theta = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::NloptLbfgs, {}, OrdinalParameterization::Theta);
  REQUIRE_MESSAGE(delta.has_value(),
      "delta fit failed: " << (delta.has_value() ? "" : delta.error().detail));
  REQUIRE_MESSAGE(theta.has_value(),
      "theta fit failed: " << (theta.has_value() ? "" : theta.error().detail));
  CHECK(theta->theta.allFinite());
  CHECK(std::isfinite(theta->fmin));
  CHECK((theta->theta - delta->theta).cwiseAbs().maxCoeff() > 1e-3);

  auto rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, *stats, *theta, OrdinalWeightKind::DWLS,
      OrdinalParameterization::Theta);
  REQUIRE(rob.has_value());
  CHECK(rob->vcov.rows() == theta->theta.size());
  CHECK(rob->se.allFinite());
  CHECK(rob->eigvals.size() == rob->df);

  magmaan::inference::ModificationIndexOptions mi_opts;
  mi_opts.candidates = magmaan::inference::ScoreCandidateSet::WithAbsentRows;
  auto mi = magmaan::estimate::modification_indices_mixed_ordinal(
      *pt, *mr, *stats, *theta, OrdinalWeightKind::DWLS, mi_opts,
      OrdinalParameterization::Theta);
  REQUIRE(mi.has_value());
  CHECK_FALSE(mi->rows.empty());

  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_mixed_ordinal_partable(
      pt_prepared, *stats, OrdinalParameterization::Theta);
  REQUIRE(prep.has_value());
  auto ev_dbg = magmaan::model::ModelEvaluator::build(pt_prepared, *mr);
  REQUIRE(ev_dbg.has_value());
  CHECK(ev_dbg->param_locations().size() ==
        static_cast<std::size_t>(theta->theta.size()));
  auto std_all = magmaan::measures::standardize::standardize_all(
      pt_prepared, *mr, *theta, rob->vcov);
  REQUIRE_MESSAGE(std_all.has_value(),
      "standardize_all failed: " <<
      (std_all.has_value() ? "" : std_all.error().detail));
  if (std_all.has_value()) {
    CHECK(std_all->theta.size() == theta->theta.size());
    CHECK(std_all->se.allFinite());
  }
}
