#include "ordinal_test_helpers.hpp"

TEST_CASE("categorical preparation rejects fixed covariates and retains joint models") {
  using namespace magmaan;
  auto parsed = parse::Parser::parse("y ~ x\ny | t1\nx | t1\n");
  REQUIRE(parsed.has_value());
  data::OrdinalStats ordinal;
  ordinal.R = {Eigen::MatrixXd::Identity(2, 2)};
  ordinal.threshold_ov = {{0, 1}};
  data::MixedOrdinalStats mixed;
  mixed.R = ordinal.R;
  mixed.mean = {Eigen::VectorXd::Zero(2)};
  mixed.ordered = {{1, 0}};
  mixed.thresholds = {Eigen::VectorXd::Zero(1)};
  mixed.threshold_ov = {{0}};
  mixed.threshold_level = {{1}};
  mixed.moments = {(Eigen::Vector4d() << 0.0, 0.0, 1.0, 0.0).finished()};
  mixed.n_obs = {100};
  mixed.n_levels = {{2, 0}};

  for (bool fixed_x : {true, false}) {
    CAPTURE(fixed_x);
    spec::BuildOptions opts;
    opts.fixed_x = fixed_x;
    opts.meanstructure = true;
    auto built = spec::build(*parsed, opts);
    REQUIRE(built.has_value());
    auto check = [&](const auto& result) {
      CHECK(result.has_value() == !fixed_x);
      if (fixed_x) {
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().kind == FitError::Kind::NumericIssue);
        CHECK(result.error().detail.find("conditional moments") != std::string::npos);
      }
    };
    auto pt = *built;
    check(estimate::prepare_ordinal_delta_partable(pt, ordinal));
    pt = *built;
    check(estimate::prepare_ordinal_delta_partable(
        pt, data::ordinal_moments_from_stats(ordinal)));
    pt = *built;
    check(estimate::prepare_mixed_ordinal_delta_partable(pt, mixed));
    if (fixed_x) {
      // The cached mixed route prepares its model through the moments overload.
      auto rep = model::build_matrix_rep(*built);
      REQUIRE(rep.has_value());
      check(estimate::mixed_ordinal_start_values(
          *built, *rep, data::mixed_ordinal_moments_from_stats(mixed), {}));
    }
  }
}

TEST_CASE("Ordinal workspace adapters split moments from Gamma cache") {
  magmaan::data::OrdinalStats stats;
  Eigen::MatrixXd R(2, 2);
  R << 1.0, 0.35,
       0.35, 1.0;
  Eigen::VectorXd thresholds(2);
  thresholds << -0.25, 0.40;
  Eigen::MatrixXd gamma(3, 3);
  gamma << 4.0, 0.2, 0.1,
           0.2, 3.0, 0.3,
           0.1, 0.3, 2.0;
  stats.R.push_back(R);
  stats.thresholds.push_back(thresholds);
  stats.threshold_ov.push_back({0, 1});
  stats.threshold_level.push_back({1, 1});
  stats.NACOV.push_back(gamma);
  stats.W_dwls.push_back(gamma.diagonal().cwiseInverse().asDiagonal());
  stats.W_wls.push_back(gamma.inverse());
  stats.n_obs.push_back(80);
  stats.n_levels.push_back({2, 2});
  stats.ov_names.push_back({"y1", "y2"});

  auto moments = magmaan::data::ordinal_moments_from_stats(stats);
  CHECK(moments.R[0].isApprox(stats.R[0], 0.0));
  CHECK(moments.thresholds[0].isApprox(stats.thresholds[0], 0.0));
  CHECK(moments.threshold_ov[0] == stats.threshold_ov[0]);
  CHECK(moments.n_obs[0] == 80);
  CHECK(moments.ov_names[0][1] == "y2");

  auto cache = magmaan::data::ordinal_gamma_cache_from_stats(stats);
  REQUIRE(cache.block_count() == 1);
  CHECK(cache.blocks[0].has_full);
  CHECK(cache.blocks[0].has_diagonal);
  CHECK(cache.blocks[0].has_dwls_weight);
  CHECK(cache.blocks[0].has_wls_weight);
  CHECK(cache.blocks[0].gamma.isApprox(gamma, 0.0));
  CHECK(cache.blocks[0].diagonal.isApprox(gamma.diagonal(), 0.0));

  for (bool placeholder : {false, true}) {
    CAPTURE(placeholder);
    stats.W_wls.clear();
    if (placeholder) stats.W_wls.emplace_back();
    auto deferred = magmaan::data::ordinal_gamma_cache_from_stats(stats);
    CHECK_FALSE(deferred.blocks[0].has_wls_weight);
    CHECK(deferred.blocks[0].w_wls.size() == 0);
    REQUIRE(magmaan::data::ordinal_gamma_cache_ensure_wls_weights(deferred)
                .has_value());
    CHECK(deferred.blocks[0].has_wls_weight);
    CHECK(deferred.blocks[0].w_wls.isApprox(gamma.inverse(), 1e-12));
  }
}

TEST_CASE("OrdinalGammaCache can materialize DWLS without full Gamma") {
  Eigen::VectorXd diagonal(3);
  diagonal << 2.0, 4.0, 5.0;
  auto cache = magmaan::data::ordinal_gamma_cache_from_diagonal({diagonal});

  REQUIRE(cache.block_count() == 1);
  CHECK(cache.blocks[0].has_diagonal);
  CHECK_FALSE(cache.blocks[0].has_full);
  CHECK_FALSE(cache.blocks[0].has_dwls_weight);

  auto dwls = magmaan::data::ordinal_gamma_cache_ensure_dwls_weights(cache);
  REQUIRE(dwls.has_value());
  CHECK(cache.blocks[0].has_dwls_weight);
  CHECK(cache.blocks[0].w_dwls.diagonal().isApprox(
      diagonal.cwiseInverse(), 0.0));

  auto wls = magmaan::data::ordinal_gamma_cache_ensure_wls_weights(cache);
  CHECK_FALSE(wls.has_value());
}

TEST_CASE("OrdinalWeightPlan encodes fit-only Gamma cost rules") {
  using magmaan::data::OrdinalEstimatorKind;
  using magmaan::data::OrdinalGammaMaterialization;
  using magmaan::data::OrdinalWorkspacePurpose;

  auto uls = magmaan::data::ordinal_weight_plan(
      OrdinalWorkspacePurpose::FitOnly, OrdinalEstimatorKind::ULS);
  CHECK(uls.materialization == OrdinalGammaMaterialization::None);

  auto dwls = magmaan::data::ordinal_weight_plan(
      OrdinalWorkspacePurpose::FitOnly, OrdinalEstimatorKind::DWLS);
  CHECK(dwls.materialization == OrdinalGammaMaterialization::Diagonal);

  auto wls = magmaan::data::ordinal_weight_plan(
      OrdinalWorkspacePurpose::FitOnly, OrdinalEstimatorKind::WLS);
  CHECK(wls.materialization == OrdinalGammaMaterialization::Full);

  auto inference = magmaan::data::ordinal_weight_plan(
      OrdinalWorkspacePurpose::FitPlusInference, OrdinalEstimatorKind::DWLS);
  CHECK(inference.materialization == OrdinalGammaMaterialization::Full);
}

TEST_CASE("Ordinal raw workspace builder honors fit-only materialization") {
  std::mt19937 rng(20260609);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(420, 4);
  const double loading[4] = {0.88, 0.80, 0.72, 0.64};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.45);
    }
  }

  auto legacy = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(legacy.has_value());

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS);
  auto uls = magmaan::data::ordinal_workspace_from_integer_data({X}, uls_plan);
  REQUIRE(uls.has_value());
  REQUIRE(uls->moments.R.size() == 1);
  CHECK(uls->moments.R[0].isApprox(legacy->R[0], 0.0));
  CHECK(uls->moments.thresholds[0].isApprox(legacy->thresholds[0], 0.0));
  CHECK(uls->moments.threshold_ov[0] == legacy->threshold_ov[0]);
  CHECK(uls->moments.threshold_level[0] == legacy->threshold_level[0]);
  CHECK(uls->moments.n_levels[0] == legacy->n_levels[0]);
  CHECK(uls->gamma_cache.block_count() == 0);

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto dwls =
      magmaan::data::ordinal_workspace_from_integer_data({X}, dwls_plan);
  REQUIRE(dwls.has_value());
  REQUIRE(dwls->gamma_cache.block_count() == 1);
  CHECK(dwls->moments.R[0].isApprox(legacy->R[0], 0.0));
  CHECK(dwls->moments.thresholds[0].isApprox(legacy->thresholds[0], 0.0));
  CHECK(dwls->gamma_cache.blocks[0].has_diagonal);
  CHECK_FALSE(dwls->gamma_cache.blocks[0].has_full);
  CHECK_FALSE(dwls->gamma_cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(dwls->gamma_cache.blocks[0].has_wls_weight);
  CHECK(dwls->gamma_cache.blocks[0].diagonal.isApprox(
      legacy->NACOV[0].diagonal(), 1e-8));

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  auto wls = magmaan::data::ordinal_workspace_from_integer_data({X}, wls_plan);
  REQUIRE(wls.has_value());
  REQUIRE(wls->gamma_cache.block_count() == 1);
  CHECK(wls->gamma_cache.blocks[0].has_full);
  CHECK(wls->gamma_cache.blocks[0].has_wls_weight);

  auto inference_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitPlusInference,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto inference =
      magmaan::data::ordinal_workspace_from_integer_data({X}, inference_plan);
  REQUIRE(inference.has_value());
  const auto& block = inference->gamma_cache.blocks[0];
  CHECK(block.has_full);
  CHECK(block.has_diagonal);
  CHECK(block.has_dwls_weight);
  CHECK_FALSE(block.has_wls_weight);
  CHECK(block.w_wls.size() == 0);
  CHECK(block.gamma.isApprox(legacy->NACOV[0], 0.0));
  CHECK(block.diagonal.isApprox(legacy->NACOV[0].diagonal(), 0.0));
  CHECK(block.w_dwls.isApprox(legacy->W_dwls[0], 0.0));
  CHECK(inference->moments.R[0].isApprox(legacy->R[0], 0.0));
  CHECK(inference->moments.thresholds[0].isApprox(legacy->thresholds[0], 0.0));
}

TEST_CASE("Ordinal inference workspace keeps singular Gamma without a WLS weight") {
  using namespace magmaan::data;
  // Four cases and six moments give rank-deficient Gamma with valid diagonals.
  Eigen::MatrixXd X(4, 3);
  X << 1, 1, 1,
       1, 2, 2,
       2, 1, 2,
       2, 2, 1;
  auto stats = ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  REQUIRE(stats->W_wls[0].size() == 0);
  CHECK((stats->NACOV[0].diagonal().array() > 0.0).all());
  auto cache = ordinal_gamma_cache_from_stats(*stats);
  CHECK_FALSE(cache.blocks[0].has_wls_weight);
  CHECK_FALSE(ordinal_gamma_cache_ensure_wls_weights(cache).has_value());
  CHECK_FALSE(cache.blocks[0].has_wls_weight);
  CHECK(cache.blocks[0].w_wls.size() == 0);

  for (auto estimator : {OrdinalEstimatorKind::DWLS, OrdinalEstimatorKind::WLS}) {
    auto plan = ordinal_weight_plan(OrdinalWorkspacePurpose::FitPlusInference,
                                   estimator);
    auto workspace = ordinal_workspace_from_integer_data({X}, plan);
    REQUIRE(workspace.has_value());
    const auto& block = workspace->gamma_cache.blocks[0];
    CHECK(block.has_full);
    CHECK(block.has_diagonal);
    CHECK(block.has_dwls_weight);
    CHECK_FALSE(block.has_wls_weight);
    CHECK(block.w_wls.size() == 0);
    CHECK(block.gamma.isApprox(stats->NACOV[0], 0.0));
    CHECK(block.w_dwls.isApprox(stats->W_dwls[0], 0.0));
  }
}

TEST_CASE("Cached ordinal DWLS fit consumes lazy raw workspace diagonal") {
  std::mt19937 rng(20260610);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(500, 3);
  const double loading[3] = {0.90, 0.78, 0.68};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.45) + (y > 0.55);
    }
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto workspace =
      magmaan::data::ordinal_workspace_from_integer_data({X}, plan);
  REQUIRE(workspace.has_value());

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

  auto x0 = magmaan::estimate::ordinal_start_values(
      *pt, *mr, workspace->moments, {});
  REQUIRE(x0.has_value());
  auto legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0);
  auto cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, workspace->moments, &workspace->gamma_cache, {}, plan, *x0);
  REQUIRE_MESSAGE(legacy.has_value(),
      "legacy DWLS failed: " << (legacy.has_value() ? "" : legacy.error().detail));
  REQUIRE_MESSAGE(cached.has_value(),
      "cached DWLS failed: " << (cached.has_value() ? "" : cached.error().detail));

  CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
  CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 1e-6);
  REQUIRE(workspace->gamma_cache.block_count() == 1);
  CHECK(workspace->gamma_cache.blocks[0].has_diagonal);
  CHECK_FALSE(workspace->gamma_cache.blocks[0].has_full);
}

TEST_CASE("Cached ordinal DWLS fit consumes diagonal Gamma only") {
  std::mt19937 rng(20260525);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(500, 3);
  const double loading[3] = {0.90, 0.78, 0.68};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.45) + (y > 0.55);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);
  auto cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});

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

  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());
  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
      pt_prepared, moments);
  REQUIRE(prep.has_value());
  Eigen::VectorXd x0_profile = *x0;
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0) {
      x0_profile(pt_prepared.free[row] - 1) += 1.25;
    }
  }
  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0);
  auto cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &cache, {}, plan, x0_profile);
  REQUIRE_MESSAGE(legacy.has_value(),
      "legacy DWLS failed: " << (legacy.has_value() ? "" : legacy.error().detail));
  REQUIRE_MESSAGE(cached.has_value(),
      "cached DWLS failed: " << (cached.has_value() ? "" : cached.error().detail));

  CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
  CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 1e-6);
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0) {
      CHECK(cached->theta(pt_prepared.free[row] - 1) ==
            doctest::Approx((*x0)(pt_prepared.free[row] - 1)));
    }
  }
  REQUIRE(cache.block_count() == 1);
  CHECK(cache.blocks[0].has_diagonal);
  CHECK_FALSE(cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(cache.blocks[0].has_full);
  CHECK_FALSE(cache.blocks[0].has_wls_weight);
}

TEST_CASE("Cached ordinal DWLS fit-plus-inference reuses Gamma for robust reporting") {
  std::mt19937 rng(20260527);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(560, 4);
  const double loading[4] = {0.88, 0.80, 0.72, 0.64};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.50) + (y > 0.45);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

  auto workspace = magmaan::data::ordinal_workspace_from_integer_data(
      {X}, magmaan::data::ordinal_weight_plan(
               magmaan::data::OrdinalWorkspacePurpose::FitPlusInference,
               magmaan::data::OrdinalEstimatorKind::DWLS));
  REQUIRE(workspace.has_value());
  auto fit_cache = std::move(workspace->gamma_cache);
  CHECK_FALSE(fit_cache.blocks[0].has_wls_weight);
  CHECK(fit_cache.blocks[0].w_wls.size() == 0);

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

  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());
  auto fit_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitPlusInference,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto cached_fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &fit_cache, {}, fit_plan, *x0);
  REQUIRE_MESSAGE(cached_fit.has_value(),
      "cached DWLS fit-plus-inference failed: "
          << (cached_fit.has_value() ? "" : cached_fit.error().detail));
  CHECK(fit_cache.blocks[0].has_full);
  CHECK(fit_cache.blocks[0].has_diagonal);
  CHECK(fit_cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(fit_cache.blocks[0].has_wls_weight);

  auto materialized_rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *cached_fit,
      magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(materialized_rob.has_value(),
      "materialized robust ordinal failed: "
          << (materialized_rob.has_value() ? "" : materialized_rob.error().detail));

  magmaan::data::OrdinalGammaCache inference_cache;
  inference_cache.blocks.resize(1);
  inference_cache.blocks[0].gamma = stats->NACOV[0];
  inference_cache.blocks[0].has_full = true;
  auto infer_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::InferenceOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto cached_rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, moments, inference_cache, *cached_fit, infer_plan);
  REQUIRE_MESSAGE(cached_rob.has_value(),
      "cached robust ordinal failed: "
          << (cached_rob.has_value() ? "" : cached_rob.error().detail));

  CHECK(inference_cache.blocks[0].has_full);
  CHECK(inference_cache.blocks[0].has_diagonal);
  CHECK(inference_cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(inference_cache.blocks[0].has_wls_weight);
  CHECK(cached_rob->chisq_standard ==
        doctest::Approx(materialized_rob->chisq_standard).epsilon(1e-12));
  CHECK(cached_rob->df == materialized_rob->df);
  CHECK((cached_rob->se - materialized_rob->se).cwiseAbs().maxCoeff() < 1e-10);
  CHECK((cached_rob->vcov - materialized_rob->vcov).cwiseAbs().maxCoeff() <
        1e-10);
  CHECK((cached_rob->eigvals - materialized_rob->eigvals)
            .cwiseAbs()
            .maxCoeff() < 1e-10);

  auto materialized_wls_rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *cached_fit,
      magmaan::estimate::OrdinalWeightKind::WLS);
  REQUIRE_MESSAGE(materialized_wls_rob.has_value(),
      "materialized WLS robust ordinal failed: "
          << (materialized_wls_rob.has_value()
                  ? ""
                  : materialized_wls_rob.error().detail));

  magmaan::data::OrdinalGammaCache wls_cache;
  wls_cache.blocks.resize(1);
  wls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_cache.blocks[0].has_full = true;
  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::InferenceOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  auto cached_wls_rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, moments, wls_cache, *cached_fit, wls_plan);
  REQUIRE_MESSAGE(cached_wls_rob.has_value(),
      "cached WLS robust ordinal failed: "
          << (cached_wls_rob.has_value() ? "" : cached_wls_rob.error().detail));
  CHECK(wls_cache.blocks[0].has_full);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(wls_cache.blocks[0].has_dwls_weight);
  CHECK((cached_wls_rob->se - materialized_wls_rob->se)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK((cached_wls_rob->vcov - materialized_wls_rob->vcov)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
  CHECK((cached_wls_rob->eigvals - materialized_wls_rob->eigvals)
            .cwiseAbs()
            .maxCoeff() < 1e-10);
}

// Anchors the ordinal FMG path (Paper 2 gate): the estimator-agnostic FMG
// eigenvalue-tail transform fed the `robust_ordinal` (chisq_standard, df,
// eigvals) triple reproduces the same Satorra-Bentler scaling `robust_ordinal`
// already reports, and the tail transforms (pEBA/pOLS) yield proper p-values on
// the polychoric UGamma spectrum. This is exactly the composition the R
// `fmg_tests_ordinal()` wrapper performs (infer_ordinal_robust -> infer_fmg_test).
TEST_CASE("Ordinal FMG transforms consume the robust_ordinal UGamma spectrum") {
  std::mt19937 rng(20260613);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(600, 4);
  const double loading[4] = {0.86, 0.78, 0.70, 0.62};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.50);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
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

  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0);
  REQUIRE_MESSAGE(fit.has_value(),
      "DWLS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto rob = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::DWLS);
  REQUIRE_MESSAGE(rob.has_value(),
      "robust_ordinal failed: " << (rob.has_value() ? "" : rob.error().detail));
  REQUIRE(rob->df > 0);
  REQUIRE(rob->eigvals.size() == rob->df);
  // The projected UGamma spectrum is positive-definite, so FMG's default
  // negative-eigenvalue truncation is a no-op here (keeps the SB comparison exact).
  CHECK(rob->eigvals.minCoeff() > 0.0);

  // FMG SatorraBentler must reproduce robust_ordinal's stored SB scaling, since
  // both apply the same robust::satorra_bentler() to the same (T, df, eigvals).
  const auto fmg_sb = magmaan::robust::frontier::fmg_test(
      rob->chisq_standard, rob->df, rob->eigvals,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::SatorraBentler,
          .truncate_negative = false});
  const double sb_p_direct = magmaan::inference::chi2_pvalue(
      rob->satorra_bentler.chi2_scaled, rob->satorra_bentler.df);
  CHECK(fmg_sb.p_value == doctest::Approx(sb_p_direct).epsilon(1e-12));

  // The eigenvalue-tail transforms (the actual FMG winners) yield proper
  // p-values on the ordinal spectrum.
  for (const auto method : {magmaan::robust::frontier::FmgMethod::Peba,
                            magmaan::robust::frontier::FmgMethod::Pols,
                            magmaan::robust::frontier::FmgMethod::PenalizedAll}) {
    const auto r = magmaan::robust::frontier::fmg_test(
        rob->chisq_standard, rob->df, rob->eigvals,
        magmaan::robust::frontier::FmgOptions{.method = method, .param = 4.0});
    CHECK(std::isfinite(r.p_value));
    CHECK(r.p_value >= 0.0);
    CHECK(r.p_value <= 1.0);
  }
}
