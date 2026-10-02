#include "ordinal_test_helpers.hpp"

TEST_CASE("robust_ordinal_ij DWLS and WLS support observed MCAR ordinal stats") {
  Eigen::MatrixXd X =
      ordinal_test_block(20260622, 520, {0.82, 0.76, 0.70, 0.64}, -0.45, 0.55);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    for (Eigen::Index c = 0; c < X.cols(); ++c) {
      if (((r + 5 * c) % 19) == 0) X(r, c) = nan;
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap,
      /*full_wls_weight=*/true);
  REQUIRE(stats.has_value());
  REQUIRE(stats->moment_influence.size() == 1);
  REQUIRE(stats->int_data.size() == 1);
  REQUIRE(stats->W_wls.size() == 1);
  REQUIRE(stats->W_wls[0].size() > 0);

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1500;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  for (const auto weight : {magmaan::estimate::OrdinalWeightKind::DWLS,
                            magmaan::estimate::OrdinalWeightKind::WLS}) {
    auto fit = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, *stats, {}, weight, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    REQUIRE_MESSAGE(fit.has_value(),
        "observed MCAR ordinal fit failed: "
            << (fit.has_value() ? "" : fit.error().detail));

    auto fixed = magmaan::estimate::robust_ordinal(
        *pt, *mr, *stats, *fit, weight,
        magmaan::estimate::OrdinalParameterization::Delta,
        magmaan::robust::Information::Observed);
    auto ij = magmaan::estimate::robust_ordinal_ij(
        *pt, *mr, *stats, *fit, weight);
    REQUIRE_MESSAGE(fixed.has_value(),
        "fixed observed robust failed: "
            << (fixed.has_value() ? "" : fixed.error().detail));
    REQUIRE_MESSAGE(ij.has_value(),
        "observed MCAR IJ robust failed: "
            << (ij.has_value() ? "" : ij.error().detail));
    CHECK(ij->df == fixed->df);
    CHECK(ij->chisq_standard == doctest::Approx(fixed->chisq_standard));
    CHECK(ij->vcov.rows() == fixed->vcov.rows());
    CHECK(ij->vcov.cols() == fixed->vcov.cols());
    CHECK(ij->vcov.allFinite());
    CHECK(ij->se.allFinite());
  }
}

TEST_CASE("robust_ordinal_ij WLS carries dense estimated-weight channel") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260620, 420, {0.82, 0.76, 0.70, 0.64}, -0.45, 0.55);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());
  REQUIRE(stats->W_wls.size() == 1);
  REQUIRE(stats->W_wls[0].size() > 0);

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1500;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "WLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto fixed = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::WLS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Observed);
  auto ij = magmaan::estimate::robust_ordinal_ij(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE_MESSAGE(fixed.has_value(),
      "fixed WLS robust failed: "
          << (fixed.has_value() ? "" : fixed.error().detail));
  REQUIRE_MESSAGE(ij.has_value(),
      "WLS IJ robust failed: " << (ij.has_value() ? "" : ij.error().detail));
  CHECK(ij->df == fixed->df);
  CHECK(ij->chisq_standard == doctest::Approx(fixed->chisq_standard));
  CHECK(ij->vcov.rows() == fixed->vcov.rows());
  CHECK(ij->vcov.cols() == fixed->vcov.cols());
  CHECK(ij->vcov.allFinite());
  CHECK(ij->se.allFinite());
}

TEST_CASE("ordinal_casewise_influence_ij Gram reproduces the DWLS IJ vcov") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260623, 500, {0.82, 0.76, 0.70, 0.64}, -0.45, 0.55);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, true);
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1500;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  // Reference: the estimated-weight ("complete-sandwich") ordinal IJ vcov.
  auto ij = magmaan::estimate::robust_ordinal_ij(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE(ij.has_value());

  auto infl = magmaan::estimate::ordinal_casewise_influence_ij(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(infl.has_value(),
      "ordinal casewise influence failed: "
          << (infl.has_value() ? "" : infl.error().detail));

  CHECK(infl->n_total == static_cast<std::int64_t>(X.rows()));
  CHECK(infl->influence.rows() == X.rows());
  CHECK(infl->influence.cols() == ij->vcov.rows());

  // Σ_i c_i c_iᵀ == the ordinal estimated-weight vcov, to machine precision.
  const Eigen::MatrixXd gram = infl->influence.transpose() * infl->influence;
  CHECK(gram.isApprox(ij->vcov, 1e-8));

  // The estimated diagonal polychoric weight ⇒ complete differs from naive.
  const double diag =
      (infl->influence - infl->influence_naive).cwiseAbs().maxCoeff();
  CHECK(diag > 1e-8);

  // Recover the unchanged moment-to-parameter map from the naive influence,
  // then independently supply the retained dense WLS Gamma channels. This
  // checks the complete covariance without reproducing the SEM bread code.
  auto prepared = *pt;
  REQUIRE(magmaan::estimate::prepare_ordinal_delta_partable(prepared, *stats).has_value());
  auto evaluator = magmaan::model::ModelEvaluator::build(prepared, *mr);
  REQUIRE(evaluator.has_value());
  auto evaluated = evaluator->evaluate(fit->theta, false, false);
  REQUIRE(evaluated.has_value());
  const auto& G = stats->moment_influence[0];
  const Eigen::Index m = G.cols();
  Eigen::VectorXd residual(m);
  Eigen::Index k = 0;
  for (std::size_t row = 0; row < prepared.size(); ++row) {
    if (prepared.op[row] != magmaan::parse::Op::Threshold) continue;
    REQUIRE(prepared.free[row] > 0);
    residual(k) = fit->theta(prepared.free[row] - 1) - stats->thresholds[0](k);
    ++k;
  }
  REQUIRE(k == stats->thresholds[0].size());
  for (Eigen::Index j = 0; j < 4; ++j)
    for (Eigen::Index i = j + 1; i < 4; ++i)
      residual(k++) = evaluated->moments.sigma[0](i, j) - stats->R[0](i, j);
  REQUIRE(residual.tail(6).norm() > 1e-4);
  auto full_IF = magmaan::data::ordinal_gamma_data_influence(
      stats->int_data[0], stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  auto full_D = magmaan::data::ordinal_gamma_jacobian_fd(
      stats->int_data[0], stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(full_IF.has_value());
  REQUIRE(full_D.has_value());
  Eigen::MatrixXd correction(G.rows(), m);
  for (k = 0; k < m; ++k) {
    const double gamma = stats->NACOV[0](k, k);
    correction.col(k) = residual(k) / (gamma * gamma) *
        (full_IF->col(k * (m + 1)) + G * full_D->row(k * (m + 1)).transpose());
  }
  const Eigen::MatrixXd weighted_G = G * stats->W_dwls[0];
  const Eigen::MatrixXd transfer = weighted_G.colPivHouseholderQr().solve(infl->influence_naive);
  REQUIRE((weighted_G * transfer).isApprox(infl->influence_naive, 1e-9));
  const Eigen::MatrixXd expected = (weighted_G + correction) * transfer;
  CHECK(expected.isApprox(infl->influence, 1e-8));
  CHECK((expected.transpose() * expected).isApprox(ij->vcov, 1e-8));
}

TEST_CASE("Observed ordinal stats expose overlap counts and nominal gamma variant") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(10, 3);
  X << 1,   1,   1,
       2,   1,   nan,
       1,   2,   1,
       2,   2,   2,
       1,   nan, 2,
       2,   1,   1,
       nan, 2,   2,
       1,   1,   2,
       2,   2,   1,
       nan, nan, 2;

  auto overlap = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap);
  auto nominal = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Nominal);
  REQUIRE(overlap.has_value());
  REQUIRE(nominal.has_value());
  CHECK(overlap->R[0].isApprox(nominal->R[0], 1e-12));
  CHECK(overlap->thresholds[0].isApprox(nominal->thresholds[0], 1e-12));
  REQUIRE(overlap->moment_influence.size() == 1);
  CHECK(nominal->moment_influence.empty());
  const Eigen::MatrixXd rebuilt =
      overlap->moment_influence[0].transpose() *
      overlap->moment_influence[0] / static_cast<double>(X.rows());
  CHECK(rebuilt.isApprox(overlap->NACOV[0], 1e-10));
  REQUIRE(overlap->moment_n_obs.size() == 1);
  REQUIRE(overlap->moment_overlap_n_obs.size() == 1);
  const auto& nobs = overlap->moment_n_obs[0];
  const auto& ovlp = overlap->moment_overlap_n_obs[0];
  REQUIRE(nobs.size() == 6);
  CHECK(nobs[0] == 8);   // y1 threshold
  CHECK(nobs[1] == 8);   // y2 threshold
  CHECK(nobs[2] == 9);   // y3 threshold
  CHECK(nobs[3] == 7);   // y2,y1 polychoric
  CHECK(nobs[4] == 7);   // y3,y1 polychoric
  CHECK(nobs[5] == 7);   // y3,y2 polychoric
  CHECK(ovlp(0, 0) == 8);
  CHECK(ovlp(0, 1) == 7);
  CHECK(ovlp(3, 3) == 7);
  CHECK(ovlp(3, 5) == 6);

  const Eigen::MatrixXd& Gp = overlap->NACOV[0];
  const Eigen::MatrixXd& Gn = nominal->NACOV[0];
  CHECK(Gp(0, 0) / Gn(0, 0) == doctest::Approx(10.0 / 8.0).epsilon(1e-10));
  CHECK(Gp(3, 3) / Gn(3, 3) == doctest::Approx(10.0 / 7.0).epsilon(1e-10));
  CHECK((Gp - Gn).cwiseAbs().maxCoeff() > 1e-3);
}

TEST_CASE("Ordinal stage-2 weights reuse observed Gamma and expose DLS endpoints") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(24, 2);
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    X(r, 0) = static_cast<double>((r % 2) + 1);
    X(r, 1) = static_cast<double>(((r / 2) % 2) + 1);
  }
  X(3, 0) = nan;
  X(7, 1) = nan;
  X(11, 0) = nan;
  X(19, 1) = nan;

  auto stats_or = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap,
      /*full_wls_weight=*/true);
  REQUIRE(stats_or.has_value());
  const auto& stats = *stats_or;
  REQUIRE(stats.W_wls.size() == 1);
  REQUIRE(stats.W_wls[0].size() > 0);

  namespace mf = magmaan::estimate::frontier;
  auto uls = mf::ordinal_stage2_weight_blocks(stats, mf::OrdinalStage2Weight::Uls);
  auto dwls = mf::ordinal_stage2_weight_blocks(stats, mf::OrdinalStage2Weight::Dwls);
  auto wls = mf::ordinal_stage2_weight_blocks(stats, mf::OrdinalStage2Weight::Wls);
  auto nt = mf::ordinal_stage2_weight_blocks(stats, mf::OrdinalStage2Weight::Nt);
  auto dls0 = mf::ordinal_stage2_weight_blocks(
      stats, mf::OrdinalStage2Weight::Dls, {0.0});
  auto dls1 = mf::ordinal_stage2_weight_blocks(
      stats, mf::OrdinalStage2Weight::Dls, {1.0});

  REQUIRE(uls.has_value());
  REQUIRE(dwls.has_value());
  REQUIRE(wls.has_value());
  REQUIRE(nt.has_value());
  REQUIRE(dls0.has_value());
  REQUIRE(dls1.has_value());

  const Eigen::Index mdim = stats.NACOV[0].rows();
  CHECK((*uls)[0].isApprox(Eigen::MatrixXd::Identity(mdim, mdim), 1e-12));
  CHECK((*dwls)[0].isApprox(stats.W_dwls[0], 1e-12));
  CHECK((*wls)[0].isApprox(stats.W_wls[0], 1e-9));
  CHECK((*dls0)[0].isApprox((*nt)[0], 1e-9));
  CHECK((*dls1)[0].isApprox((*wls)[0], 1e-9));

  auto adapted = mf::ordinal_stats_with_stage2_weight(
      stats, mf::OrdinalStage2Weight::Dls, {0.35});
  REQUIRE(adapted.has_value());
  REQUIRE(adapted->W_wls.size() == 1);
  CHECK(adapted->NACOV[0].isApprox(stats.NACOV[0], 1e-12));
  CHECK(adapted->W_dwls[0].isApprox(stats.W_dwls[0], 1e-12));
  CHECK(adapted->W_wls[0].rows() == mdim);
  Eigen::LLT<Eigen::MatrixXd> llt(adapted->W_wls[0]);
  CHECK(llt.info() == Eigen::Success);
}

TEST_CASE("Observed ordinal stats count four-variable moment support overlaps") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(12, 4);
  X << 1,   1,   1,   1,
       1,   1,   2,   2,
       1,   2,   1,   2,
       1,   2,   2,   1,
       2,   1,   1,   2,
       2,   1,   2,   1,
       2,   2,   1,   1,
       2,   2,   2,   2,
       nan, 1,   1,   1,
       1,   nan, 1,   1,
       1,   1,   nan, 1,
       1,   1,   1,   nan;

  auto overlap = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap);
  auto nominal = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Nominal);
  REQUIRE(overlap.has_value());
  REQUIRE(nominal.has_value());
  REQUIRE(overlap->moment_n_obs.size() == 1);
  REQUIRE(overlap->moment_overlap_n_obs.size() == 1);
  const auto& nobs = overlap->moment_n_obs[0];
  const auto& ovlp = overlap->moment_overlap_n_obs[0];
  REQUIRE(nobs.size() == 10);
  CHECK(nobs[4] == 10);   // y2,y1 polychoric
  CHECK(nobs[9] == 10);   // y4,y3 polychoric
  CHECK(ovlp(4, 4) == 10);
  CHECK(ovlp(9, 9) == 10);
  CHECK(ovlp(4, 9) == 8); // union support y1,y2,y3,y4

  const Eigen::MatrixXd& Gp = overlap->NACOV[0];
  const Eigen::MatrixXd& Gn = nominal->NACOV[0];
  CHECK(Gp(4, 4) / Gn(4, 4) == doctest::Approx(12.0 / 10.0).epsilon(1e-10));
}
