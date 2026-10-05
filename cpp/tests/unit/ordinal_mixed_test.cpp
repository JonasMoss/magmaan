#include "ordinal_test_helpers.hpp"

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
  REQUIRE(stats->moment_influence.size() == 1);
  CHECK(stats->moment_influence[0].rows() == X.rows());
  CHECK(stats->moment_influence[0].cols() == stats->moments[0].size());
  REQUIRE(stats->raw_data.size() == 1);
  CHECK(stats->raw_data[0].isApprox(X, 0.0));
  CHECK(((stats->moment_influence[0].transpose() *
          stats->moment_influence[0]) /
         static_cast<double>(X.rows()))
            .isApprox(stats->NACOV[0], 1e-10));
  CHECK(stats->R[0](0, 0) == doctest::Approx(1.0));
  CHECK(stats->R[0](1, 1) == doctest::Approx(1.0));
  CHECK(stats->R[0](2, 2) > 0.0);
  CHECK(stats->R[0](3, 3) > 0.0);

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;

  const char* th_syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x2 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n";
  auto fpt = magmaan::parse::Parser::parse(th_syntax);
  REQUIRE(fpt.has_value());
  auto pt2 = magmaan::spec::build(*fpt, opts);
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

TEST_CASE("Observed mixed ordinal stats reduce to complete-data mixed stats") {
  std::mt19937 rng(20260624);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(420, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta + 0.35 * norm(rng) > -0.45) +
              (eta + 0.35 * norm(rng) > 0.55);
    X(i, 1) = 0.62 * eta + 0.78 * norm(rng) + 0.20;
    X(i, 2) = 1.0 + (0.68 * eta + 0.74 * norm(rng) > 0.05);
    X(i, 3) = 0.54 * eta + 0.84 * norm(rng) - 0.15;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 0, 1, 0}};
  auto complete =
      magmaan::data::mixed_ordinal_stats_from_data({X}, ordered, false);
  auto observed =
      magmaan::data::mixed_ordinal_stats_from_observed_data({X}, ordered, false);
  REQUIRE(complete.has_value());
  REQUIRE(observed.has_value());

  CHECK(observed->R[0].isApprox(complete->R[0], 1e-10));
  CHECK(observed->mean[0].isApprox(complete->mean[0], 1e-12));
  CHECK(observed->thresholds[0].isApprox(complete->thresholds[0], 1e-12));
  CHECK(observed->moments[0].isApprox(complete->moments[0], 1e-10));
  CHECK(observed->moment_influence[0].isApprox(
      complete->moment_influence[0], 1e-8));
  CHECK(observed->NACOV[0].isApprox(complete->NACOV[0], 1e-8));
  CHECK(observed->W_dwls[0].isApprox(complete->W_dwls[0], 1e-8));
  REQUIRE(observed->raw_data.size() == 1);
  CHECK(observed->raw_data[0].isApprox(X, 0.0));
}

TEST_CASE("robust_mixed_ordinal_ij ULS supports observed MCAR") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::mt19937 rng(20260625);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(760, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta + 0.30 * norm(rng) > -0.55) +
              (eta + 0.30 * norm(rng) > 0.45);
    X(i, 1) = 0.70 * eta + 0.72 * norm(rng) + 0.15;
    X(i, 2) = 1.0 + (0.72 * eta + 0.69 * norm(rng) > 0.10);
    X(i, 3) = 0.58 * eta + 0.82 * norm(rng) - 0.10;
  }
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    if (i % 7 == 0) X(i, 0) = nan;
    if (i % 11 == 0) X(i, 1) = nan;
    if (i % 13 == 0) X(i, 2) = nan;
    if (i % 17 == 0) X(i, 3) = nan;
  }

  const std::vector<std::vector<std::int32_t>> ordered = {{1, 0, 1, 0}};
  auto stats =
      magmaan::data::mixed_ordinal_stats_from_observed_data({X}, ordered, false);
  REQUIRE_MESSAGE(stats.has_value(),
      "observed mixed stats failed: "
          << (stats.has_value() ? "" : stats.error().detail));
  REQUIRE(stats->moment_influence.size() == 1);
  CHECK(stats->moment_influence[0].rows() == X.rows());
  CHECK(stats->moment_influence[0].cols() == stats->moments[0].size());
  CHECK(((stats->moment_influence[0].transpose() *
          stats->moment_influence[0]) /
         static_cast<double>(X.rows()))
            .isApprox(stats->NACOV[0], 1e-10));
  REQUIRE(stats->raw_data.size() == 1);
  CHECK(matrix_matches_with_nan(stats->raw_data[0], X, 0.0));

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x3 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x3 ~*~ 1*x3\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::ULS);
  REQUIRE_MESSAGE(fit.has_value(),
      "observed mixed ULS fit failed: "
          << (fit.has_value() ? "" : fit.error().detail));
  auto fixed = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Observed);
  auto ij = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS);
  REQUIRE_MESSAGE(fixed.has_value(),
      "observed fixed mixed robust failed: "
          << (fixed.has_value() ? "" : fixed.error().detail));
  REQUIRE_MESSAGE(ij.has_value(),
      "observed mixed ULS IJ failed: "
          << (ij.has_value() ? "" : ij.error().detail));
  CHECK(ij->df == fixed->df);
  CHECK(ij->chisq_standard == doctest::Approx(fixed->chisq_standard));
  CHECK(ij->vcov.isApprox(fixed->vcov, 1e-8));
  CHECK(ij->se.isApprox(fixed->se, 1e-8));
}

TEST_CASE("observed mixed Gamma helpers reduce to complete-data helpers") {
  std::mt19937 rng(20260627);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(360, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta + 0.35 * norm(rng) > -0.35) +
              (eta + 0.35 * norm(rng) > 0.60);
    X(i, 1) = 0.58 * eta + 0.82 * norm(rng) + 0.18;
    X(i, 2) = 1.0 + (0.70 * eta + 0.70 * norm(rng) > 0.10);
    X(i, 3) = 0.48 * eta + 0.88 * norm(rng) - 0.12;
  }

  const std::vector<std::vector<std::int32_t>> ordered = {{1, 0, 1, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());
  REQUIRE(stats->raw_data.size() == 1);

  auto IFG = magmaan::data::mixed_gamma_diag_data_influence(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto IFG_obs = magmaan::data::mixed_observed_gamma_diag_data_influence(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto IFG_full = magmaan::data::mixed_gamma_data_influence(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto IFG_full_obs = magmaan::data::mixed_observed_gamma_data_influence(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto D = magmaan::data::mixed_gamma_diag_jacobian_fd(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto D_obs = magmaan::data::mixed_observed_gamma_diag_jacobian_fd(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto D_full = magmaan::data::mixed_gamma_jacobian_fd(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto D_full_obs = magmaan::data::mixed_observed_gamma_jacobian_fd(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  REQUIRE(IFG.has_value());
  REQUIRE(IFG_obs.has_value());
  REQUIRE(IFG_full.has_value());
  REQUIRE(IFG_full_obs.has_value());
  REQUIRE(D.has_value());
  REQUIRE(D_obs.has_value());
  REQUIRE(D_full.has_value());
  REQUIRE(D_full_obs.has_value());
  CHECK(IFG_obs->isApprox(*IFG, 1e-8));
  CHECK(IFG_full_obs->isApprox(*IFG_full, 1e-8));
  CHECK(D_obs->isApprox(*D, 1e-7));
  CHECK(D_full_obs->isApprox(*D_full, 1e-7));
}

TEST_CASE("robust_mixed_ordinal_ij DWLS and WLS support observed MCAR") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::mt19937 rng(20260628);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(980, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta + 0.30 * norm(rng) > -0.50) +
              (eta + 0.30 * norm(rng) > 0.48);
    X(i, 1) = 0.70 * eta + 0.72 * norm(rng) + 0.12;
    X(i, 2) = 1.0 + (0.72 * eta + 0.69 * norm(rng) > 0.08);
    X(i, 3) = 0.58 * eta + 0.82 * norm(rng) - 0.08;
  }
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    if (i % 7 == 0) X(i, 0) = nan;
    if (i % 11 == 0) X(i, 1) = nan;
    if (i % 13 == 0) X(i, 2) = nan;
    if (i % 17 == 0) X(i, 3) = nan;
  }

  const std::vector<std::vector<std::int32_t>> ordered = {{1, 0, 1, 0}};
  auto stats =
      magmaan::data::mixed_ordinal_stats_from_observed_data({X}, ordered, true);
  REQUIRE_MESSAGE(stats.has_value(),
      "observed mixed stats failed: "
          << (stats.has_value() ? "" : stats.error().detail));
  REQUIRE(stats->raw_data.size() == 1);
  CHECK(matrix_matches_with_nan(stats->raw_data[0], X, 0.0));
  REQUIRE(stats->W_wls.size() == 1);
  REQUIRE(stats->W_wls[0].rows() == stats->moments[0].size());

  auto IFG = magmaan::data::mixed_observed_gamma_diag_data_influence(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto IFG_full = magmaan::data::mixed_observed_gamma_data_influence(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto D = magmaan::data::mixed_observed_gamma_diag_jacobian_fd(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto D_full = magmaan::data::mixed_observed_gamma_jacobian_fd(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  REQUIRE(IFG.has_value());
  REQUIRE(IFG_full.has_value());
  REQUIRE(D.has_value());
  REQUIRE(D_full.has_value());
  const Eigen::Index m = stats->moments[0].size();
  CHECK(IFG->rows() == X.rows());
  CHECK(IFG->cols() == m);
  CHECK(IFG_full->rows() == X.rows());
  CHECK(IFG_full->cols() == m * m);
  CHECK(D->rows() == m);
  CHECK(D->cols() == m);
  CHECK(D_full->rows() == m * m);
  CHECK(D_full->cols() == m);
  CHECK(IFG->allFinite());
  CHECK(IFG_full->allFinite());
  CHECK(D->allFinite());
  CHECK(D_full->allFinite());
  for (Eigen::Index k = 0; k < m; ++k) {
    CHECK((IFG->col(k) - IFG_full->col(k + k * m)).norm() < 1e-10);
    CHECK((D->row(k) - D_full->row(k + k * m)).norm() < 1e-10);
  }
  const Eigen::MatrixXd full_if =
      *IFG + stats->moment_influence[0] * D->transpose();
  const Eigen::MatrixXd full_matrix_if =
      *IFG_full + stats->moment_influence[0] * D_full->transpose();
  const double scale = 1.0 + full_if.cwiseAbs().maxCoeff();
  const double full_scale = 1.0 + full_matrix_if.cwiseAbs().maxCoeff();
  CHECK(IFG->colwise().mean().norm() < 1e-8 * scale);
  CHECK(full_if.colwise().mean().norm() < 1e-3 * scale);
  CHECK(IFG_full->colwise().mean().norm() < 1e-8 * full_scale);
  CHECK(full_matrix_if.colwise().mean().norm() < 1e-3 * full_scale);

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x3 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x3 ~*~ 1*x3\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto dwls_fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(dwls_fit.has_value(),
      "observed mixed DWLS fit failed: "
          << (dwls_fit.has_value() ? "" : dwls_fit.error().detail));
  auto dwls = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, *stats, *dwls_fit,
      magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(dwls.has_value(),
      "observed mixed DWLS IJ failed: "
          << (dwls.has_value() ? "" : dwls.error().detail));
  CHECK(dwls->vcov.allFinite());
  CHECK(dwls->se.allFinite());

  auto fallback_stats = *stats;
  fallback_stats.gamma_diag_influence.clear();
  fallback_stats.gamma_full_influence.clear();
  auto dwls_fallback = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, fallback_stats, *dwls_fit,
      magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(dwls_fallback.has_value(),
      "observed mixed DWLS IJ raw fallback failed: "
          << (dwls_fallback.has_value() ? "" : dwls_fallback.error().detail));
  CHECK(dwls_fallback->vcov.isApprox(dwls->vcov, 1e-10));
  CHECK(dwls_fallback->se.isApprox(dwls->se, 1e-10));

  auto prof = magmaan::estimate::mixed_ordinal_dwls_profile_rmsea(
      *pt, *mr, *stats, *dwls_fit,
      magmaan::estimate::OrdinalParameterization::Delta);
  auto prof_fallback = magmaan::estimate::mixed_ordinal_dwls_profile_rmsea(
      *pt, *mr, fallback_stats, *dwls_fit,
      magmaan::estimate::OrdinalParameterization::Delta);
  REQUIRE_MESSAGE(prof.has_value(),
      "observed mixed DWLS profile failed: "
          << (prof.has_value() ? "" : prof.error().detail));
  REQUIRE_MESSAGE(prof_fallback.has_value(),
      "observed mixed DWLS profile raw fallback failed: "
          << (prof_fallback.has_value() ? "" : prof_fallback.error().detail));
  CHECK(prof_fallback->gamma.isApprox(prof->gamma, 1e-10));
  CHECK(prof_fallback->eigvals.isApprox(prof->eigvals, 1e-9));
  CHECK(prof_fallback->rmsea == doctest::Approx(prof->rmsea));

  auto wls_fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE_MESSAGE(wls_fit.has_value(),
      "observed mixed WLS fit failed: "
          << (wls_fit.has_value() ? "" : wls_fit.error().detail));
  auto wls = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, *stats, *wls_fit,
      magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE_MESSAGE(wls.has_value(),
      "observed mixed WLS IJ failed: "
          << (wls.has_value() ? "" : wls.error().detail));
  auto wls_fallback = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, fallback_stats, *wls_fit,
      magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE_MESSAGE(wls_fallback.has_value(),
      "observed mixed WLS IJ raw fallback failed: "
          << (wls_fallback.has_value() ? "" : wls_fallback.error().detail));
  CHECK(wls->df == dwls->df);
  CHECK(wls->vcov.allFinite());
  CHECK(wls->se.allFinite());
  CHECK(wls_fallback->vcov.isApprox(wls->vcov, 1e-10));
  CHECK(wls_fallback->se.isApprox(wls->se, 1e-10));
}

TEST_CASE("hybrid FIML mixed ordinal stats support MCAR continuous information") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::mt19937 rng(20260629);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(820, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    const double eps1 = norm(rng);
    const double eps2 = norm(rng);
    X(i, 0) = 1.0 + (eta + 0.34 * eps1 > -0.55) +
              (eta + 0.34 * eps1 > 0.48);
    X(i, 1) = 0.70 * eta + 0.72 * norm(rng) + 0.12;
    X(i, 2) = 1.0 + (0.72 * eta + 0.69 * eps2 > 0.08);
    X(i, 3) = 0.58 * eta + 0.82 * norm(rng) - 0.08;
  }
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    if (i % 7 == 0) X(i, 0) = nan;
    if (i % 11 == 0) X(i, 1) = nan;
    if (i % 13 == 0) X(i, 2) = nan;
    if (i % 17 == 0) X(i, 3) = nan;
  }

  const std::vector<std::vector<std::int32_t>> ordered = {{1, 0, 1, 0}};
  auto stats = magmaan::estimate::fiml::
      mixed_ordinal_stats_hybrid_fiml_from_observed_data({X}, ordered, true);
  REQUIRE_MESSAGE(stats.has_value(),
      "hybrid mixed stats failed: "
          << (stats.has_value() ? "" : stats.error().detail));
  REQUIRE(stats->moment_influence.size() == 1);
  REQUIRE(stats->gamma_diag_influence.size() == 1);
  REQUIRE(stats->gamma_full_influence.size() == 1);
  const Eigen::Index m = stats->moments[0].size();
  CHECK(stats->moment_influence[0].rows() == X.rows());
  CHECK(stats->moment_influence[0].cols() == m);
  CHECK(stats->gamma_diag_influence[0].rows() == X.rows());
  CHECK(stats->gamma_diag_influence[0].cols() == m);
  CHECK(stats->gamma_full_influence[0].rows() == X.rows());
  CHECK(stats->gamma_full_influence[0].cols() == m * m);
  CHECK(((stats->moment_influence[0].transpose() *
          stats->moment_influence[0]) /
         static_cast<double>(X.rows()))
            .isApprox(stats->NACOV[0], 1e-10));
  CHECK(stats->moment_influence[0].allFinite());
  CHECK(stats->gamma_diag_influence[0].allFinite());
  CHECK(stats->gamma_full_influence[0].allFinite());
  CHECK(stats->gamma_diag_influence[0].colwise().mean().norm() < 1e-9);
  CHECK(stats->gamma_full_influence[0].colwise().mean().norm() < 1e-9);

  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\n"
      "x3 | t1\n"
      "x1 ~*~ 1*x1\n"
      "x3 ~*~ 1*x3\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  auto fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(fit.has_value(),
      "hybrid mixed DWLS fit failed: "
          << (fit.has_value() ? "" : fit.error().detail));
  auto ij = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(ij.has_value(),
      "hybrid mixed DWLS IJ failed: "
          << (ij.has_value() ? "" : ij.error().detail));
  auto prof = magmaan::estimate::mixed_ordinal_dwls_profile_rmsea(
      *pt, *mr, *stats, *fit,
      magmaan::estimate::OrdinalParameterization::Delta);
  REQUIRE_MESSAGE(prof.has_value(),
      "hybrid mixed DWLS profile failed: "
          << (prof.has_value() ? "" : prof.error().detail));
  CHECK(ij->vcov.allFinite());
  CHECK(ij->se.allFinite());
  CHECK(prof->gamma.allFinite());
  CHECK(std::isfinite(prof->rmsea));
}

TEST_CASE("mixed Gamma influence has mixed moment order and zero mean") {
  std::mt19937 rng(20260622);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(360, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta + 0.35 * norm(rng) > -0.35) +
              (eta + 0.35 * norm(rng) > 0.65);
    X(i, 1) = 0.55 * eta + 0.84 * norm(rng) + 0.25;
    X(i, 2) = 1.0 + (0.70 * eta + 0.70 * norm(rng) > 0.10);
    X(i, 3) = 0.45 * eta + 0.90 * norm(rng) - 0.15;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 0, 1, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());
  REQUIRE(stats->raw_data.size() == 1);
  REQUIRE(stats->moment_influence.size() == 1);

  auto IFG = magmaan::data::mixed_gamma_diag_data_influence(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto IFG_full = magmaan::data::mixed_gamma_data_influence(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto D = magmaan::data::mixed_gamma_diag_jacobian_fd(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  auto D_full = magmaan::data::mixed_gamma_jacobian_fd(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  REQUIRE(IFG.has_value());
  REQUIRE(IFG_full.has_value());
  REQUIRE(D.has_value());
  REQUIRE(D_full.has_value());
  const Eigen::Index m = stats->moments[0].size();
  CHECK(IFG->rows() == X.rows());
  CHECK(IFG->cols() == m);
  CHECK(IFG_full->rows() == X.rows());
  CHECK(IFG_full->cols() == m * m);
  CHECK(D->rows() == m);
  CHECK(D->cols() == m);
  CHECK(D_full->rows() == m * m);
  CHECK(D_full->cols() == m);
  CHECK(IFG->allFinite());
  CHECK(IFG_full->allFinite());
  CHECK(D->allFinite());
  CHECK(D_full->allFinite());
  auto probe = mixed_gamma_diag_influence_probe(
      stats->raw_data[0], stats->ordered[0], stats->n_levels[0],
      stats->thresholds[0], stats->mean[0], stats->R[0]);
  CHECK((probe.Gamma - stats->NACOV[0]).cwiseAbs().maxCoeff() < 1e-8);

  double max_abs = 0.0;
  double max_full_abs = 0.0;
  for (Eigen::Index i = 0; i < X.rows(); i += 41) {
    const Eigen::VectorXd fd =
        finite_diff_mixed_gamma_diag_case_influence(probe, i);
    max_abs = std::max(
        max_abs, (IFG->row(i).transpose() - fd).cwiseAbs().maxCoeff());
    const Eigen::MatrixXd fd_full =
        finite_diff_mixed_gamma_case_influence(probe, i);
    const Eigen::VectorXd full_vec = IFG_full->row(i).transpose();
    const Eigen::Map<const Eigen::MatrixXd> full_i(full_vec.data(), m, m);
    max_full_abs = std::max(
        max_full_abs, (full_i - fd_full).cwiseAbs().maxCoeff());
  }
  CHECK(max_abs < 8e-5 * (1.0 + IFG->cwiseAbs().maxCoeff()));
  CHECK(max_full_abs < 2e-4 * (1.0 + IFG_full->cwiseAbs().maxCoeff()));

  for (Eigen::Index k = 0; k < m; ++k) {
    CHECK((IFG->col(k) - IFG_full->col(k + k * m)).norm() < 1e-10);
    CHECK((D->row(k) - D_full->row(k + k * m)).norm() < 1e-10);
  }

  const Eigen::MatrixXd full_if =
      *IFG + stats->moment_influence[0] * D->transpose();
  const Eigen::MatrixXd full_matrix_if =
      *IFG_full + stats->moment_influence[0] * D_full->transpose();
  const double scale = 1.0 + full_if.cwiseAbs().maxCoeff();
  const double full_scale = 1.0 + full_matrix_if.cwiseAbs().maxCoeff();
  CHECK(IFG->colwise().mean().norm() < 1e-8 * scale);
  CHECK(full_if.colwise().mean().norm() < 1e-6 * scale);
  CHECK(IFG_full->colwise().mean().norm() < 1e-8 * full_scale);
  CHECK(full_matrix_if.colwise().mean().norm() < 1e-6 * full_scale);
}

TEST_CASE("robust_mixed_ordinal_ij supports mixed ULS DWLS and WLS") {
  std::mt19937 rng(20260621);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.35);
    X(i, 1) = 1.0 + (0.70 * eta + 0.72 * norm(rng) > 0.05);
    X(i, 2) = 0.76 * eta + 0.64 * norm(rng) + 0.20;
    X(i, 3) = 0.66 * eta + 0.75 * norm(rng) - 0.10;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());
  REQUIRE(stats->moment_influence.size() == 1);

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
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::ULS);
  REQUIRE_MESSAGE(fit.has_value(),
      "mixed ULS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto fixed = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Observed);
  auto ij = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS);
  REQUIRE_MESSAGE(fixed.has_value(),
      "fixed mixed ULS robust failed: "
          << (fixed.has_value() ? "" : fixed.error().detail));
  REQUIRE_MESSAGE(ij.has_value(),
      "mixed ULS IJ failed: " << (ij.has_value() ? "" : ij.error().detail));
  CHECK(ij->df == fixed->df);
  CHECK(ij->chisq_standard == doctest::Approx(fixed->chisq_standard));
  CHECK(ij->vcov.isApprox(fixed->vcov, 1e-8));
  CHECK(ij->se.isApprox(fixed->se, 1e-8));

  auto ev = magmaan::model::ModelEvaluator::build(*pt, *mr);
  REQUIRE(ev.has_value());
  Eigen::Index k_loading = -1;
  const auto locs = ev->param_locations();
  for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(locs.size()); ++k) {
    const auto& loc = locs[static_cast<std::size_t>(k)];
    if (loc.mat == magmaan::model::MatId::Lambda &&
        loc.row == 1 && loc.col == 0) {
      k_loading = k;
      break;
    }
  }
  REQUIRE(k_loading >= 0);
  magmaan::optim::OptimOptions prof_opts;
  prof_opts.max_iter = 350;
  prof_opts.ftol = 1e-9;
  prof_opts.gtol = 1e-7;
  auto uls_profile =
      magmaan::estimate::frontier::profile_lrt_parameter_mixed_ordinal(
          *pt, *mr, *stats, *fit, k_loading, 0.97 * fit->theta(k_loading),
          {}, magmaan::estimate::OrdinalWeightKind::ULS,
          magmaan::estimate::Backend::NloptSlsqp, prof_opts,
          magmaan::estimate::OrdinalParameterization::Delta, 1e-6, true,
          magmaan::estimate::frontier::ScalarProfileReference::RobustScaled);
  REQUIRE_MESSAGE(uls_profile.has_value(),
      "mixed ULS robust profile LRT failed: "
          << (uls_profile.has_value() ? "" : uls_profile.error().detail));
  CHECK(uls_profile->scaling_factor > 0.0);
  CHECK(std::isfinite(uls_profile->T_scaled));

  auto dwls_fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(dwls_fit.has_value(),
      "mixed DWLS fit failed: "
          << (dwls_fit.has_value() ? "" : dwls_fit.error().detail));
  auto dwls = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, *stats, *dwls_fit,
      magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(dwls.has_value(),
      "mixed DWLS IJ failed: " << (dwls.has_value() ? "" : dwls.error().detail));
  CHECK(dwls->df == fixed->df);
  CHECK(dwls->vcov.allFinite());
  CHECK(dwls->se.allFinite());

  auto no_raw = *stats;
  no_raw.raw_data.clear();
  auto missing = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, no_raw, *dwls_fit,
      magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_FALSE(missing.has_value());
  CHECK(missing.error().detail.find("estimated-weight influence unavailable") !=
        std::string::npos);

  auto wls_fit = magmaan::test::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE_MESSAGE(wls_fit.has_value(),
      "mixed WLS fit failed: "
          << (wls_fit.has_value() ? "" : wls_fit.error().detail));
  auto wls = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, *stats, *wls_fit,
      magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE_MESSAGE(wls.has_value(),
      "mixed WLS IJ failed: " << (wls.has_value() ? "" : wls.error().detail));
  CHECK(wls->df == fixed->df);
  CHECK(wls->vcov.allFinite());
  CHECK(wls->se.allFinite());

  auto missing_wls = magmaan::estimate::robust_mixed_ordinal_ij(
      *pt, *mr, no_raw, *wls_fit,
      magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE_FALSE(missing_wls.has_value());
  CHECK(missing_wls.error().detail.find("estimated-weight influence unavailable") !=
        std::string::npos);
}

TEST_CASE("Sparse mixed Gamma diagonal agrees with retained dense case matrices") {
  // Each group is evaluated independently, as in grouped post-fit inference.
  for (bool all_ordinal : {false, true})
  for (int categories : {2, 5}) for (int groups : {1, 2}) {
    std::mt19937 rng(static_cast<unsigned>(88000 + categories + groups));
    std::normal_distribution<double> normal;
    for (int group = 0; group < groups; ++group) {
      Eigen::MatrixXd x(160, 4);
      for (Eigen::Index i = 0; i < x.rows(); ++i) {
        const double factor = normal(rng);
        for (Eigen::Index j = 0; j < x.cols(); ++j) {
          const double z = 0.6 * factor + normal(rng);
          x(i, j) = all_ordinal || j % 2 == 0
              ? (categories == 2 ? 1.0 + (z > 0.0)
                                : 1.0 + (z > -1.0) + (z > -0.3) +
                                  (z > 0.3) + (z > 1.0))
              : z;
        }
      }
      const std::vector<std::int32_t> ordered = all_ordinal
          ? std::vector<std::int32_t>{1, 1, 1, 1}
          : std::vector<std::int32_t>{1, 0, 1, 0};
      // The mixed influence primitive also accepts the all-ordinal endpoint.
      magmaan::data::MixedOrdinalStats mixed;
      if (!all_ordinal) {
        auto built = magmaan::data::mixed_ordinal_stats_from_data({x}, {ordered}, false);
        REQUIRE(built.has_value());
        mixed = std::move(*built);
      } else {
        auto stats = magmaan::data::ordinal_stats_from_integer_data({x}, false);
        REQUIRE(stats.has_value());
        mixed.n_levels = stats->n_levels;
        mixed.thresholds = stats->thresholds;
        mixed.R = stats->R;
        mixed.mean = {Eigen::VectorXd::Zero(x.cols())};
      }
      for (bool observed : {false, true}) {
        auto diagonal = observed
            ? magmaan::data::mixed_observed_gamma_diag_data_influence(
                x, ordered, mixed.n_levels[0], mixed.thresholds[0], mixed.mean[0], mixed.R[0])
            : magmaan::data::mixed_gamma_diag_data_influence(
                x, ordered, mixed.n_levels[0], mixed.thresholds[0], mixed.mean[0], mixed.R[0]);
        auto full = observed
            ? magmaan::data::mixed_observed_gamma_data_influence(
                x, ordered, mixed.n_levels[0], mixed.thresholds[0], mixed.mean[0], mixed.R[0])
            : magmaan::data::mixed_gamma_data_influence(
                x, ordered, mixed.n_levels[0], mixed.thresholds[0], mixed.mean[0], mixed.R[0]);
        REQUIRE(diagonal.has_value());
        REQUIRE(full.has_value());
        const Eigen::Index m = diagonal->cols();
        Eigen::MatrixXd reference(x.rows(), m);
        for (Eigen::Index j = 0; j < m; ++j) reference.col(j) = full->col(j * (m + 1));
        CHECK(diagonal->isApprox(reference, 1e-10));
      }
    }
  }
}
