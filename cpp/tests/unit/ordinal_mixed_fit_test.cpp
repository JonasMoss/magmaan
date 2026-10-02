#include "ordinal_test_helpers.hpp"

TEST_CASE("Mixed ordinal full-threshold SNLLS matches bounded DWLS/WLS") {
  std::mt19937 rng(20260612);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(560, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.7) + (eta > 0.35);
    X(i, 1) = 1.0 + (0.68 * eta + 0.74 * norm(rng) > 0.05);
    X(i, 2) = 0.82 * eta + 0.57 * norm(rng) + 0.15;
    X(i, 3) = 0.63 * eta + 0.78 * norm(rng) - 0.12;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());

  magmaan::spec::BuildOptions build_opts;
  build_opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                                          "x1 | t1 + t2\n"
                                          "x2 | t1\n"
                                          "x1 ~*~ 1*x1\n"
                                          "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::mixed_ordinal_start_values(*pt, *mr, *stats, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 500;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto check_weight = [&](magmaan::estimate::OrdinalWeightKind weight) {
    auto bounded = magmaan::estimate::fit_mixed_ordinal_bounded(
        *pt, *mr, *stats, {}, weight, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto snlls = magmaan::estimate::fit_mixed_ordinal_snlls_full_thresholds(
        *pt, *mr, *stats, weight, *x0, magmaan::estimate::Backend::NloptLbfgs,
        opts);
    REQUIRE_MESSAGE(bounded.has_value(),
                    "mixed bounded failed: "
                        << (bounded.has_value() ? "" : bounded.error().detail));
    REQUIRE_MESSAGE(snlls.has_value(),
                    "mixed SNLLS failed: "
                        << (snlls.has_value() ? "" : snlls.error().detail));
    CHECK(snlls->fmin == doctest::Approx(bounded->fmin).epsilon(2e-6));
    CHECK((snlls->theta - bounded->theta).cwiseAbs().maxCoeff() < 8e-4);
    CHECK(snlls->n_nonlinear > 0);
    CHECK(snlls->n_linear > 0);
    CHECK(snlls->diagnostics.geometric_stationarity.checked);
    CHECK(magmaan::estimate::fit_verdict(*snlls).objective ==
          magmaan::estimate::FitCheck::Passed);
  };

  check_weight(magmaan::estimate::OrdinalWeightKind::DWLS);
  check_weight(magmaan::estimate::OrdinalWeightKind::WLS);

  // Theta flows through the same full-threshold stack: the standardized
  // covariance moments make the non-threshold block nonlinear, so only the
  // thresholds stay Golub-Pereyra linear, and the fit must agree with the
  // bounded theta fit. The delta model above keeps a binary indicator, but
  // under theta that model has a near-flat lambda/psi ridge (both optimizers
  // stall at arbitrary ridge points), so the theta parity check uses a
  // well-identified design with three-category ordinal indicators.
  std::mt19937 rng_theta(20260614);
  Eigen::MatrixXd Xt(560, 4);
  for (Eigen::Index i = 0; i < Xt.rows(); ++i) {
    const double eta = norm(rng_theta);
    const double y1 = 0.85 * eta + 0.53 * norm(rng_theta);
    const double y2 = 0.78 * eta + 0.63 * norm(rng_theta);
    Xt(i, 0) = 1.0 + (y1 > -0.7) + (y1 > 0.35);
    Xt(i, 1) = 1.0 + (y2 > -0.5) + (y2 > 0.55);
    Xt(i, 2) = 0.82 * eta + 0.57 * norm(rng_theta) + 0.15;
    Xt(i, 3) = 0.63 * eta + 0.78 * norm(rng_theta) - 0.12;
  }
  auto stats_theta =
      magmaan::data::mixed_ordinal_stats_from_data({Xt}, ordered);
  REQUIRE(stats_theta.has_value());
  auto fp_theta = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                                                "x1 | t1 + t2\n"
                                                "x2 | t1 + t2\n"
                                                "x1 ~*~ 1*x1\n"
                                                "x2 ~*~ 1*x2\n");
  REQUIRE(fp_theta.has_value());
  auto pt_theta = magmaan::spec::build(*fp_theta, build_opts);
  REQUIRE(pt_theta.has_value());
  auto mr_theta = magmaan::model::build_matrix_rep(*pt_theta);
  REQUIRE(mr_theta.has_value());
  auto x0_theta = magmaan::estimate::mixed_ordinal_start_values(
      *pt_theta, *mr_theta, *stats_theta, {});
  REQUIRE(x0_theta.has_value());

  auto check_theta = [&](magmaan::estimate::OrdinalWeightKind weight) {
    auto bounded = magmaan::estimate::fit_mixed_ordinal_bounded(
        *pt_theta, *mr_theta, *stats_theta, {}, weight, *x0_theta,
        magmaan::estimate::Backend::NloptLbfgs, opts,
        magmaan::estimate::OrdinalParameterization::Theta);
    auto snlls = magmaan::estimate::fit_mixed_ordinal_snlls_full_thresholds(
        *pt_theta, *mr_theta, *stats_theta, weight, *x0_theta,
        magmaan::estimate::Backend::NloptLbfgs, opts,
        magmaan::estimate::OrdinalParameterization::Theta);
    REQUIRE_MESSAGE(bounded.has_value(),
                    "mixed bounded theta failed: "
                        << (bounded.has_value() ? "" : bounded.error().detail));
    REQUIRE_MESSAGE(snlls.has_value(),
                    "mixed SNLLS theta failed: "
                        << (snlls.has_value() ? "" : snlls.error().detail));
    CHECK(snlls->fmin == doctest::Approx(bounded->fmin).epsilon(2e-6));
    CHECK((snlls->theta - bounded->theta).cwiseAbs().maxCoeff() < 8e-4);
    CHECK(snlls->n_nonlinear > 0);
    CHECK(snlls->n_linear > 0);
    CHECK(snlls->diagnostics.geometric_stationarity.checked);
    CHECK(magmaan::estimate::fit_verdict(*snlls).objective ==
          magmaan::estimate::FitCheck::Passed);
  };
  check_theta(magmaan::estimate::OrdinalWeightKind::DWLS);
  check_theta(magmaan::estimate::OrdinalWeightKind::WLS);
}

TEST_CASE("Mixed ordinal fit-only workspace supplies DWLS diagonal fits") {
  std::mt19937 rng(20260613);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.65) + (eta > 0.3);
    X(i, 1) = 1.0 + (0.66 * eta + 0.75 * norm(rng) > 0.08);
    X(i, 2) = 0.84 * eta + 0.55 * norm(rng) + 0.12;
    X(i, 3) = 0.61 * eta + 0.79 * norm(rng) - 0.08;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());

  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Delta);
  auto workspace =
      magmaan::data::mixed_ordinal_workspace_from_data({X}, ordered, plan);
  REQUIRE(workspace.has_value());
  REQUIRE(workspace->gamma_cache.blocks.size() == 1);
  CHECK(workspace->gamma_cache.blocks[0].has_diagonal);
  CHECK_FALSE(workspace->gamma_cache.blocks[0].has_full);
  CHECK(workspace->moments.moments[0].size() == stats->moments[0].size());
  const double moment_diff =
      (workspace->moments.moments[0] - stats->moments[0])
          .cwiseAbs()
          .maxCoeff();
  CHECK(moment_diff < 1e-10);
  REQUIRE(stats->NACOV.size() == 1);
  REQUIRE(workspace->gamma_cache.blocks[0].diagonal.size() ==
          stats->NACOV[0].rows());
  CHECK((workspace->gamma_cache.blocks[0].diagonal -
         stats->NACOV[0].diagonal())
            .cwiseAbs()
            .maxCoeff() < 1e-8);

  magmaan::spec::BuildOptions build_opts;
  build_opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                                          "x1 | t1 + t2\n"
                                          "x2 | t1\n"
                                          "x1 ~*~ 1*x1\n"
                                          "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0_full =
      magmaan::estimate::mixed_ordinal_start_values(*pt, *mr, *stats, {});
  auto x0_lazy = magmaan::estimate::mixed_ordinal_start_values(
      *pt, *mr, workspace->moments, {});
  REQUIRE(x0_full.has_value());
  REQUIRE(x0_lazy.has_value());
  REQUIRE(x0_full->size() == x0_lazy->size());
  CHECK((*x0_full - *x0_lazy).cwiseAbs().maxCoeff() < 1e-10);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 500;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto full_bounded = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS,
      *x0_full, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto lazy_bounded = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, workspace->moments, &workspace->gamma_cache, {}, plan,
      *x0_lazy, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto lazy_snlls =
      magmaan::estimate::fit_mixed_ordinal_snlls_full_thresholds(
          *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0_lazy,
          magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(full_bounded.has_value(),
                  "full mixed bounded failed: "
                      << (full_bounded.has_value() ? ""
                                                   : full_bounded.error().detail));
  REQUIRE_MESSAGE(lazy_bounded.has_value(),
                  "lazy mixed bounded failed: "
                      << (lazy_bounded.has_value() ? ""
                                                   : lazy_bounded.error().detail));
  REQUIRE_MESSAGE(lazy_snlls.has_value(),
                  "lazy mixed SNLLS failed: "
                      << (lazy_snlls.has_value() ? ""
                                                 : lazy_snlls.error().detail));
  CHECK(lazy_bounded->fmin ==
        doctest::Approx(full_bounded->fmin).epsilon(2e-6));
  CHECK(lazy_snlls->fmin == doctest::Approx(full_bounded->fmin).epsilon(2e-6));
  for (const auto* fit : {&*full_bounded, &*lazy_bounded, &*lazy_snlls}) {
    CHECK(fit->diagnostics.geometric_stationarity.checked);
    CHECK(magmaan::estimate::fit_verdict(*fit).objective ==
          magmaan::estimate::FitCheck::Passed);
  }
  CHECK((lazy_bounded->theta - full_bounded->theta).cwiseAbs().maxCoeff() <
        8e-4);
  CHECK((lazy_snlls->theta - full_bounded->theta).cwiseAbs().maxCoeff() <
        8e-4);

  // A WLS plan carries the full Gamma but defers the O(m^3) inverse: the
  // workspace must hand back has_full without has_wls_weight, the cache-aware
  // fit builds the weight on demand and matches the eagerly materialized fit,
  // and the cache retains the weight afterwards.
  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Delta);
  auto wls_workspace =
      magmaan::data::mixed_ordinal_workspace_from_data({X}, ordered, wls_plan);
  REQUIRE(wls_workspace.has_value());
  REQUIRE(wls_workspace->gamma_cache.blocks.size() == 1);
  CHECK(wls_workspace->gamma_cache.blocks[0].has_full);
  CHECK_FALSE(wls_workspace->gamma_cache.blocks[0].has_wls_weight);
  auto full_wls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS,
      *x0_full, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto lazy_wls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, wls_workspace->moments, &wls_workspace->gamma_cache, {},
      wls_plan, *x0_lazy, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(full_wls.has_value(),
                  "full mixed WLS failed: "
                      << (full_wls.has_value() ? "" : full_wls.error().detail));
  REQUIRE_MESSAGE(lazy_wls.has_value(),
                  "lazy mixed WLS failed: "
                      << (lazy_wls.has_value() ? "" : lazy_wls.error().detail));
  CHECK(lazy_wls->fmin == doctest::Approx(full_wls->fmin).epsilon(1e-10));
  CHECK((lazy_wls->theta - full_wls->theta).cwiseAbs().maxCoeff() < 1e-8);
  CHECK(wls_workspace->gamma_cache.blocks[0].has_wls_weight);
}

TEST_CASE("Mixed ordinal full-Gamma cache supplies robust reporting") {
  std::mt19937 rng(20260614);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(540, 4);
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.7) + (eta > 0.35);
    X(i, 1) = 1.0 + (0.62 * eta + 0.78 * norm(rng) > 0.02);
    X(i, 2) = 0.80 * eta + 0.62 * norm(rng) + 0.16;
    X(i, 3) = 0.58 * eta + 0.82 * norm(rng) - 0.05;
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());

  magmaan::spec::BuildOptions build_opts;
  build_opts.meanstructure = true;
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
                                          "x1 | t1 + t2\n"
                                          "x2 | t1\n"
                                          "x1 ~*~ 1*x1\n"
                                          "x2 ~*~ 1*x2\n");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  const auto moments = magmaan::data::mixed_ordinal_moments_from_stats(*stats);
  auto x0 = magmaan::estimate::mixed_ordinal_start_values(
      *pt, *mr, moments, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 500;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto full_dwls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS,
      *x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(full_dwls.has_value(),
      "full mixed DWLS failed: " <<
      (full_dwls.has_value() ? "" : full_dwls.error().detail));

  auto dwls_cache = magmaan::data::ordinal_gamma_cache_from_stats(*stats);
  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitPlusInference,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Delta);
  auto cached_dwls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, moments, &dwls_cache, {}, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(cached_dwls.has_value(),
      "cached mixed DWLS failed: " <<
      (cached_dwls.has_value() ? "" : cached_dwls.error().detail));
  CHECK(cached_dwls->fmin == doctest::Approx(full_dwls->fmin).epsilon(2e-6));
  CHECK((cached_dwls->theta - full_dwls->theta).cwiseAbs().maxCoeff() < 8e-4);
  CHECK(dwls_cache.blocks[0].has_full);
  CHECK(dwls_cache.blocks[0].has_diagonal);
  CHECK(dwls_cache.blocks[0].has_dwls_weight);

  auto materialized_dwls_rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, *stats, *cached_dwls,
      magmaan::estimate::OrdinalWeightKind::DWLS);
  auto cached_dwls_rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, moments, dwls_cache, *cached_dwls, dwls_plan);
  REQUIRE_MESSAGE(materialized_dwls_rob.has_value(),
      "materialized mixed DWLS robust failed: " <<
      (materialized_dwls_rob.has_value()
           ? ""
           : materialized_dwls_rob.error().detail));
  REQUIRE_MESSAGE(cached_dwls_rob.has_value(),
      "cached mixed DWLS robust failed: " <<
      (cached_dwls_rob.has_value() ? "" : cached_dwls_rob.error().detail));
  CHECK(cached_dwls_rob->chisq_standard ==
        doctest::Approx(materialized_dwls_rob->chisq_standard).epsilon(1e-12));
  CHECK(cached_dwls_rob->df == materialized_dwls_rob->df);
  CHECK((cached_dwls_rob->se - materialized_dwls_rob->se)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK((cached_dwls_rob->eigvals - materialized_dwls_rob->eigvals)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK(cached_dwls_rob->satorra_bentler.chi2_scaled ==
        doctest::Approx(materialized_dwls_rob->satorra_bentler.chi2_scaled));
  CHECK(cached_dwls_rob->mean_var_adjusted.chi2_adj ==
        doctest::Approx(materialized_dwls_rob->mean_var_adjusted.chi2_adj));
  CHECK(cached_dwls_rob->scaled_shifted.chi2_adj ==
        doctest::Approx(materialized_dwls_rob->scaled_shifted.chi2_adj));

  auto full_wls = magmaan::estimate::fit_mixed_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS,
      *x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(full_wls.has_value(),
      "full mixed WLS failed: " <<
      (full_wls.has_value() ? "" : full_wls.error().detail));
  auto materialized_wls_rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, *stats, *full_wls,
      magmaan::estimate::OrdinalWeightKind::WLS);

  auto wls_cache = magmaan::data::ordinal_gamma_cache_from_stats(*stats);
  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::InferenceOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Delta);
  auto cached_wls_rob = magmaan::estimate::robust_mixed_ordinal(
      *pt, *mr, moments, wls_cache, *full_wls, wls_plan);
  REQUIRE_MESSAGE(materialized_wls_rob.has_value(),
      "materialized mixed WLS robust failed: " <<
      (materialized_wls_rob.has_value()
           ? ""
           : materialized_wls_rob.error().detail));
  REQUIRE_MESSAGE(cached_wls_rob.has_value(),
      "cached mixed WLS robust failed: " <<
      (cached_wls_rob.has_value() ? "" : cached_wls_rob.error().detail));
  CHECK(wls_cache.blocks[0].has_full);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK((cached_wls_rob->vcov - materialized_wls_rob->vcov)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK((cached_wls_rob->eigvals - materialized_wls_rob->eigvals)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
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
  CHECK(std::isfinite(fit->fmin));
}
