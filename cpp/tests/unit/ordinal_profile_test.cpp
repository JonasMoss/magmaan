#include "ordinal_test_helpers.hpp"

TEST_CASE("Cached ordinal WLS fit uses Schur threshold profiling") {
  std::mt19937 rng(20260526);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  const double loading[4] = {0.88, 0.76, 0.70, 0.62};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.35);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);
  magmaan::data::OrdinalGammaCache cache;
  cache.blocks.resize(1);
  cache.blocks[0].gamma = stats->NACOV[0];
  cache.blocks[0].has_full = true;

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
  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
      pt_prepared, moments);
  REQUIRE(prep.has_value());
  Eigen::VectorXd x0_profile = *x0;
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0) {
      x0_profile(pt_prepared.free[row] - 1) -= 0.90;
    }
  }

  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  auto legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0);
  auto cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &cache, {}, plan, x0_profile);
  REQUIRE_MESSAGE(legacy.has_value(),
      "legacy WLS failed: " << (legacy.has_value() ? "" : legacy.error().detail));
  REQUIRE_MESSAGE(cached.has_value(),
      "cached WLS failed: " << (cached.has_value() ? "" : cached.error().detail));

  CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-7));
  CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 2e-5);
  CHECK(cache.blocks[0].has_full);
  CHECK(cache.blocks[0].has_wls_weight);
  CHECK_FALSE(cache.blocks[0].has_dwls_weight);
}

TEST_CASE("Cached ordinal profiling keeps fixed threshold rows in the objective") {
  std::mt19937 rng(20260529);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(540, 4);
  const double loading[4] = {0.86, 0.78, 0.70, 0.62};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.60) + (y > 0.40);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  REQUIRE(std::abs(stats->thresholds[0](0)) > 0.05);
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | 0*t1 + t2\n"
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

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  auto uls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, nullptr, {}, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(uls_cached.has_value(),
      "cached ULS failed: "
          << (uls_cached.has_value() ? "" : uls_cached.error().detail));
  CHECK(std::isfinite(uls_cached->fmin));

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  auto dwls_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto dwls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto dwls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &dwls_cache, {}, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(dwls_legacy.has_value(),
      "legacy DWLS failed: "
          << (dwls_legacy.has_value() ? "" : dwls_legacy.error().detail));
  REQUIRE_MESSAGE(dwls_cached.has_value(),
      "cached DWLS failed: "
          << (dwls_cached.has_value() ? "" : dwls_cached.error().detail));
  CHECK(dwls_cached->fmin ==
        doctest::Approx(dwls_legacy->fmin).epsilon(1e-8));
  CHECK((dwls_cached->theta - dwls_legacy->theta).cwiseAbs().maxCoeff() <
        2e-5);
  CHECK(dwls_cache.blocks[0].has_diagonal);
  CHECK_FALSE(dwls_cache.blocks[0].has_full);
  CHECK_FALSE(dwls_cache.blocks[0].has_dwls_weight);

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  magmaan::data::OrdinalGammaCache wls_cache;
  wls_cache.blocks.resize(1);
  wls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_cache.blocks[0].has_full = true;
  auto wls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto wls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &wls_cache, {}, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_legacy.has_value(),
      "legacy WLS failed: "
          << (wls_legacy.has_value() ? "" : wls_legacy.error().detail));
  REQUIRE_MESSAGE(wls_cached.has_value(),
      "cached WLS failed: "
          << (wls_cached.has_value() ? "" : wls_cached.error().detail));
  CHECK(wls_cached->fmin ==
        doctest::Approx(wls_legacy->fmin).epsilon(1e-7));
  CHECK((wls_cached->theta - wls_legacy->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(wls_cache.blocks[0].has_full);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(wls_cache.blocks[0].has_dwls_weight);

  magmaan::data::OrdinalGammaCache snlls_wls_cache;
  snlls_wls_cache.blocks.resize(1);
  snlls_wls_cache.blocks[0].gamma = stats->NACOV[0];
  snlls_wls_cache.blocks[0].has_full = true;
  auto wls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &snlls_wls_cache, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_snlls.has_value(),
      "SNLLS WLS failed: "
          << (wls_snlls.has_value() ? "" : wls_snlls.error().detail));
  CHECK(wls_snlls->fmin ==
        doctest::Approx(wls_cached->fmin).epsilon(1e-8));
  CHECK((wls_snlls->theta - wls_cached->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(snlls_wls_cache.blocks[0].has_full);
  CHECK(snlls_wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(snlls_wls_cache.blocks[0].has_dwls_weight);
}

TEST_CASE("Cached ordinal profiling handles shared threshold labels") {
  std::mt19937 rng(20260530);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(560, 4);
  const double loading[4] = {0.84, 0.76, 0.68, 0.60};
  const double shift[4] = {-0.15, 0.10, 0.00, 0.05};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = shift[j] + loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.45);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  REQUIRE(std::abs(stats->thresholds[0](0) - stats->thresholds[0](2)) >
          0.01);
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | a*t1 + t2\n"
      "x2 | a*t1 + t2\n"
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

  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
      pt_prepared, moments);
  REQUIRE(prep.has_value());
  std::vector<Eigen::Index> shared_threshold_free;
  std::int32_t shared_group = -1;
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] != magmaan::parse::Op::Threshold ||
        pt_prepared.free[row] <= 0) {
      continue;
    }
    const std::int32_t group =
        pt_prepared.eq_groups[static_cast<std::size_t>(pt_prepared.free[row] - 1)];
    int count = 0;
    for (std::size_t other = 0; other < pt_prepared.size(); ++other) {
      if (pt_prepared.op[other] == magmaan::parse::Op::Threshold &&
          pt_prepared.free[other] > 0 &&
          pt_prepared.eq_groups[static_cast<std::size_t>(
              pt_prepared.free[other] - 1)] == group) {
        ++count;
      }
    }
    if (count == 2) {
      shared_group = group;
      break;
    }
  }
  REQUIRE(shared_group >= 0);
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0 &&
        pt_prepared.eq_groups[static_cast<std::size_t>(
            pt_prepared.free[row] - 1)] == shared_group) {
      shared_threshold_free.push_back(pt_prepared.free[row] - 1);
    }
  }
  REQUIRE(shared_threshold_free.size() == 2);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  auto uls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, nullptr, {}, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(uls_cached.has_value(),
      "cached ULS failed: "
          << (uls_cached.has_value() ? "" : uls_cached.error().detail));
  CHECK(std::isfinite(uls_cached->fmin));
  CHECK(uls_cached->theta(shared_threshold_free[0]) ==
        doctest::Approx(uls_cached->theta(shared_threshold_free[1])).epsilon(1e-12));

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  auto dwls_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto dwls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto dwls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &dwls_cache, {}, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(dwls_legacy.has_value(),
      "legacy DWLS failed: "
          << (dwls_legacy.has_value() ? "" : dwls_legacy.error().detail));
  REQUIRE_MESSAGE(dwls_cached.has_value(),
      "cached DWLS failed: "
          << (dwls_cached.has_value() ? "" : dwls_cached.error().detail));
  CHECK(dwls_cached->fmin ==
        doctest::Approx(dwls_legacy->fmin).epsilon(1e-8));
  CHECK((dwls_cached->theta - dwls_legacy->theta).cwiseAbs().maxCoeff() <
        2e-5);
  CHECK(dwls_cache.blocks[0].has_diagonal);
  CHECK_FALSE(dwls_cache.blocks[0].has_full);
  CHECK_FALSE(dwls_cache.blocks[0].has_dwls_weight);

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Delta,
      magmaan::data::OrdinalThresholdMode::FixedOrConstrained);
  magmaan::data::OrdinalGammaCache wls_cache;
  wls_cache.blocks.resize(1);
  wls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_cache.blocks[0].has_full = true;
  auto wls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto wls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &wls_cache, {}, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_legacy.has_value(),
      "legacy WLS failed: "
          << (wls_legacy.has_value() ? "" : wls_legacy.error().detail));
  REQUIRE_MESSAGE(wls_cached.has_value(),
      "cached WLS failed: "
          << (wls_cached.has_value() ? "" : wls_cached.error().detail));
  CHECK(wls_cached->fmin ==
        doctest::Approx(wls_legacy->fmin).epsilon(1e-7));
  CHECK((wls_cached->theta - wls_legacy->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(wls_cache.blocks[0].has_full);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(wls_cache.blocks[0].has_dwls_weight);

  magmaan::data::OrdinalGammaCache snlls_wls_cache;
  snlls_wls_cache.blocks.resize(1);
  snlls_wls_cache.blocks[0].gamma = stats->NACOV[0];
  snlls_wls_cache.blocks[0].has_full = true;
  auto wls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &snlls_wls_cache, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_snlls.has_value(),
      "SNLLS WLS failed: "
          << (wls_snlls.has_value() ? "" : wls_snlls.error().detail));
  CHECK(wls_snlls->fmin ==
        doctest::Approx(wls_cached->fmin).epsilon(1e-8));
  CHECK((wls_snlls->theta - wls_cached->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(snlls_wls_cache.blocks[0].has_full);
  CHECK(snlls_wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(snlls_wls_cache.blocks[0].has_dwls_weight);
}

TEST_CASE("Cached ordinal ULS fit does not require Gamma") {
  Eigen::MatrixXd X(240, 3);
  Eigen::Index r = 0;
  for (int rep = 0; rep < 10; ++rep) {
    for (int x1 = 1; x1 <= 3; ++x1) {
      for (int x2 = 1; x2 <= 3; ++x2) {
        for (int x3 = 1; x3 <= 3; ++x3) {
          if (r >= X.rows()) break;
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
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

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

  magmaan::data::OrdinalGammaCache cache;
  cache.blocks.resize(1);
  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS);
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &cache, {}, plan, *x0);
  REQUIRE_MESSAGE(fit.has_value(),
      "cached ULS failed: " << (fit.has_value() ? "" : fit.error().detail));
  CHECK(std::isfinite(fit->fmin));
  CHECK_FALSE(cache.blocks[0].has_diagonal);
  CHECK_FALSE(cache.blocks[0].has_full);
  CHECK_FALSE(cache.blocks[0].has_dwls_weight);
  CHECK_FALSE(cache.blocks[0].has_wls_weight);
}

TEST_CASE("Ordinal SNLLS profiles thresholds and linear covariance block") {
  std::mt19937 rng(20260528);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(620, 4);
  const double loading[4] = {0.90, 0.82, 0.74, 0.66};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.45) + (y > 0.50);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

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

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS);
  auto uls_bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, nullptr, {}, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto uls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, nullptr, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto uls_snlls_full_thresholds =
      magmaan::estimate::fit_ordinal_snlls_full_thresholds(
          *pt, *mr, moments, nullptr, uls_plan, *x0,
          magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(uls_bounded.has_value(),
      "bounded ULS failed: "
          << (uls_bounded.has_value() ? "" : uls_bounded.error().detail));
  REQUIRE_MESSAGE(uls_snlls.has_value(),
      "SNLLS ULS failed: "
          << (uls_snlls.has_value() ? "" : uls_snlls.error().detail));
  REQUIRE_MESSAGE(uls_snlls_full_thresholds.has_value(),
      "full-threshold SNLLS ULS failed: "
          << (uls_snlls_full_thresholds.has_value()
                  ? ""
                  : uls_snlls_full_thresholds.error().detail));
  CHECK(uls_snlls->fmin ==
        doctest::Approx(uls_bounded->fmin).epsilon(1e-8));
  CHECK((uls_snlls->theta - uls_bounded->theta).cwiseAbs().maxCoeff() < 2e-5);
  CHECK(uls_snlls_full_thresholds->fmin ==
        doctest::Approx(uls_bounded->fmin).epsilon(1e-8));
  CHECK((uls_snlls_full_thresholds->theta - uls_bounded->theta)
            .cwiseAbs()
            .maxCoeff() < 2e-5);

  auto pt_prepared = *pt;
  auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
      pt_prepared, moments);
  REQUIRE(prep.has_value());
  for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
    if (pt_prepared.op[row] == magmaan::parse::Op::Threshold &&
        pt_prepared.free[row] > 0) {
      CHECK(uls_snlls->theta(pt_prepared.free[row] - 1) ==
            doctest::Approx((*x0)(pt_prepared.free[row] - 1)));
    }
  }

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto bounded_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto snlls_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto dwls_bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &bounded_cache, {}, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto dwls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &snlls_cache, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto full_threshold_dwls_cache = magmaan::data::ordinal_gamma_cache_from_diagonal(
      {stats->NACOV[0].diagonal()});
  auto dwls_snlls_full_thresholds =
      magmaan::estimate::fit_ordinal_snlls_full_thresholds(
          *pt, *mr, moments, &full_threshold_dwls_cache, dwls_plan, *x0,
          magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(dwls_bounded.has_value(),
      "bounded DWLS failed: "
          << (dwls_bounded.has_value() ? "" : dwls_bounded.error().detail));
  REQUIRE_MESSAGE(dwls_snlls.has_value(),
      "SNLLS DWLS failed: "
          << (dwls_snlls.has_value() ? "" : dwls_snlls.error().detail));
  REQUIRE_MESSAGE(dwls_snlls_full_thresholds.has_value(),
      "full-threshold SNLLS DWLS failed: "
          << (dwls_snlls_full_thresholds.has_value()
                  ? ""
                  : dwls_snlls_full_thresholds.error().detail));
  CHECK(dwls_snlls->fmin ==
        doctest::Approx(dwls_bounded->fmin).epsilon(1e-8));
  CHECK((dwls_snlls->theta - dwls_bounded->theta).cwiseAbs().maxCoeff() <
        2e-5);
  CHECK(dwls_snlls_full_thresholds->fmin ==
        doctest::Approx(dwls_bounded->fmin).epsilon(1e-8));
  CHECK((dwls_snlls_full_thresholds->theta - dwls_bounded->theta)
            .cwiseAbs()
            .maxCoeff() < 2e-5);
  CHECK(snlls_cache.blocks[0].has_diagonal);
  CHECK_FALSE(snlls_cache.blocks[0].has_full);
  CHECK_FALSE(snlls_cache.blocks[0].has_dwls_weight);

  // A fit-only cache must consume its fitting W, not invert sampling Gamma.
  auto supplied = *stats;
  const auto q = supplied.NACOV[0].rows();
  supplied.W_dwls[0] = Eigen::VectorXd::LinSpaced(q, 0.7, 1.3).asDiagonal();
  auto supplied_cache = magmaan::data::ordinal_gamma_cache_from_stats(supplied);
  auto supplied_reference = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, supplied, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE(supplied_reference);
  auto supplied_bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &supplied_cache, {}, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto supplied_profiled = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &supplied_cache, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto supplied_full_thresholds = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
      *pt, *mr, moments, &supplied_cache, dwls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE(supplied_bounded); REQUIRE(supplied_profiled); REQUIRE(supplied_full_thresholds);
  for (const auto* fit : {&*supplied_bounded, &*supplied_profiled, &*supplied_full_thresholds}) {
    CHECK(fit->fmin == doctest::Approx(supplied_reference->fmin).epsilon(1e-8));
    CHECK((fit->theta - supplied_reference->theta).cwiseAbs().maxCoeff() < 2e-5);
  }
  CHECK(supplied_cache.blocks[0].gamma.isApprox(stats->NACOV[0], 0.0));

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  magmaan::data::OrdinalGammaCache bounded_wls_cache;
  bounded_wls_cache.blocks.resize(1);
  bounded_wls_cache.blocks[0].gamma = stats->NACOV[0];
  bounded_wls_cache.blocks[0].has_full = true;
  magmaan::data::OrdinalGammaCache snlls_wls_cache;
  snlls_wls_cache.blocks.resize(1);
  snlls_wls_cache.blocks[0].gamma = stats->NACOV[0];
  snlls_wls_cache.blocks[0].has_full = true;
  magmaan::data::OrdinalGammaCache full_threshold_wls_cache;
  full_threshold_wls_cache.blocks.resize(1);
  full_threshold_wls_cache.blocks[0].gamma = stats->NACOV[0];
  full_threshold_wls_cache.blocks[0].has_full = true;
  auto wls_bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &bounded_wls_cache, {}, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto wls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &snlls_wls_cache, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto wls_snlls_full_thresholds =
      magmaan::estimate::fit_ordinal_snlls_full_thresholds(
          *pt, *mr, moments, &full_threshold_wls_cache, wls_plan, *x0,
          magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_bounded.has_value(),
      "bounded WLS failed: "
          << (wls_bounded.has_value() ? "" : wls_bounded.error().detail));
  REQUIRE_MESSAGE(wls_snlls.has_value(),
      "SNLLS WLS failed: "
          << (wls_snlls.has_value() ? "" : wls_snlls.error().detail));
  REQUIRE_MESSAGE(wls_snlls_full_thresholds.has_value(),
      "full-threshold SNLLS WLS failed: "
          << (wls_snlls_full_thresholds.has_value()
                  ? ""
                  : wls_snlls_full_thresholds.error().detail));
  CHECK(wls_snlls->fmin ==
        doctest::Approx(wls_bounded->fmin).epsilon(1e-8));
  CHECK((wls_snlls->theta - wls_bounded->theta).cwiseAbs().maxCoeff() <
        3e-5);
  CHECK(wls_snlls_full_thresholds->fmin ==
        doctest::Approx(wls_bounded->fmin).epsilon(1e-8));
  CHECK((wls_snlls_full_thresholds->theta - wls_bounded->theta)
            .cwiseAbs()
            .maxCoeff() < 3e-5);
  CHECK(snlls_wls_cache.blocks[0].has_full);
  CHECK(snlls_wls_cache.blocks[0].has_wls_weight);
  CHECK_FALSE(snlls_wls_cache.blocks[0].has_dwls_weight);
}

TEST_CASE("Ordinal full-threshold SNLLS accepts linear threshold constraints") {
  std::mt19937 rng(20260602);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(520, 4);
  const double loading[4] = {0.86, 0.79, 0.72, 0.68};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.55) + (y > 0.45);
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | a*t1 + b*t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n"
      "a + b == 0\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(*pt, *mr, moments, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS);
  auto cache =
      magmaan::data::ordinal_gamma_cache_from_diagonal({stats->NACOV[0].diagonal()});
  auto snlls = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
      *pt, *mr, moments, &cache, plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);

  REQUIRE_MESSAGE(bounded.has_value(),
      "bounded DWLS failed: "
          << (bounded.has_value() ? "" : bounded.error().detail));
  REQUIRE_MESSAGE(snlls.has_value(),
      "full-threshold SNLLS DWLS failed: "
          << (snlls.has_value() ? "" : snlls.error().detail));
  CHECK(snlls->fmin == doctest::Approx(bounded->fmin).epsilon(1e-8));
  CHECK((snlls->theta - bounded->theta).cwiseAbs().maxCoeff() < 3e-5);

  // The threshold-profiled path absorbs threshold-only linear constraints
  // into the threshold design (general H) and must agree with the
  // full-threshold SNLLS and bounded fits.
  auto profiled = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &cache, plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(profiled.has_value(),
      "threshold-profiled SNLLS DWLS failed: "
          << (profiled.has_value() ? "" : profiled.error().detail));
  CHECK(profiled->fmin == doctest::Approx(bounded->fmin).epsilon(1e-8));
  CHECK((profiled->theta - bounded->theta).cwiseAbs().maxCoeff() < 3e-5);
  // The constraint itself holds at the profiled solution.
  double a_hat = 0.0;
  double b_hat = 0.0;
  bool found_a = false;
  bool found_b = false;
  for (std::size_t row = 0; row < pt->size(); ++row) {
    if (pt->op[row] != magmaan::parse::Op::Threshold) continue;
    if (pt->free[row] <= 0) continue;
    // x1's two thresholds carry labels a and b; x1 rows come first.
    if (!found_a) {
      a_hat = profiled->theta(pt->free[row] - 1);
      found_a = true;
    } else if (!found_b) {
      b_hat = profiled->theta(pt->free[row] - 1);
      found_b = true;
    }
  }
  REQUIRE(found_a);
  REQUIRE(found_b);
  CHECK(std::abs(a_hat + b_hat) < 1e-10);
}

namespace {

constexpr const char* k2GroupOrdinalCfa =
    "f =~ x1 + x2 + x3 + x4\n"
    "x1 | t1 + t2\n"
    "x2 | t1 + t2\n"
    "x3 | t1 + t2\n"
    "x4 | t1 + t2\n"
    "x1 ~*~ 1*x1\n"
    "x2 ~*~ 1*x2\n"
    "x3 ~*~ 1*x3\n"
    "x4 ~*~ 1*x4\n";

}  // namespace

// Regression for the joint cross-block profiling refactor: the per-block
// profiled solve used to build each block's normal matrix over the global
// gamma coordinate (singular for any second group), so the profiled bounded
// and SNLLS paths could not fit multi-group ordinal models at all. The two
// groups deliberately have different sample sizes so a missing n_b/N weight
// in the joint normal equations would break parity with the legacy
// unprofiled fit.
TEST_CASE("Ordinal SNLLS rejects released-scale delta before profiling") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260610, 400, {0.85, 0.78, 0.71, 0.66}, -0.45, 0.55);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X, X});
  REQUIRE(stats.has_value());
  magmaan::spec::BuildOptions options;
  options.n_groups = 2;
  auto parsed = magmaan::parse::Parser::parse(k2GroupOrdinalCfa);
  REQUIRE(parsed.has_value());
  auto pt = magmaan::spec::build(*parsed, options);
  REQUIRE(pt.has_value());
  pt->group_equal = {magmaan::spec::GroupEqual::Thresholds};
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  auto moments = magmaan::data::ordinal_moments_from_stats(*stats);
  auto start = magmaan::estimate::ordinal_start_values(*pt, *rep, moments, {});
  REQUIRE(start.has_value());
  const auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS,
      magmaan::data::OrdinalMomentParameterization::Delta);
  auto profiled = magmaan::estimate::fit_ordinal_snlls(
      *pt, *rep, moments, nullptr, plan, *start);
  auto full = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
      *pt, *rep, moments, nullptr, plan, *start);
  REQUIRE_FALSE(profiled.has_value());
  REQUIRE_FALSE(full.has_value());
  CHECK(profiled.error().detail.find("released-scale delta") != std::string::npos);
  CHECK(full.error().detail.find("released-scale delta") != std::string::npos);
}

TEST_CASE("Joint threshold profiling handles two-group ordinal fits") {
  const Eigen::MatrixXd X1 =
      ordinal_test_block(20260610, 700, {0.85, 0.78, 0.71, 0.66}, -0.45, 0.55);
  const Eigen::MatrixXd X2 =
      ordinal_test_block(20260611, 450, {0.80, 0.74, 0.69, 0.62}, -0.30, 0.70);

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X1, X2});
  REQUIRE(stats.has_value());

  magmaan::spec::BuildOptions build_opts;
  build_opts.n_groups = 2;
  auto fp = magmaan::parse::Parser::parse(k2GroupOrdinalCfa);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  const std::array<std::pair<magmaan::data::OrdinalEstimatorKind,
                             magmaan::estimate::OrdinalWeightKind>,
                   3>
      kinds = {{{magmaan::data::OrdinalEstimatorKind::ULS,
                 magmaan::estimate::OrdinalWeightKind::ULS},
                {magmaan::data::OrdinalEstimatorKind::DWLS,
                 magmaan::estimate::OrdinalWeightKind::DWLS},
                {magmaan::data::OrdinalEstimatorKind::WLS,
                 magmaan::estimate::OrdinalWeightKind::WLS}}};
  for (const auto& [estimator, weight_kind] : kinds) {
    CAPTURE(static_cast<int>(estimator));
    auto plan = magmaan::data::ordinal_weight_plan(
        magmaan::data::OrdinalWorkspacePurpose::FitOnly, estimator);
    auto workspace =
        magmaan::data::ordinal_workspace_from_integer_data({X1, X2}, plan);
    REQUIRE(workspace.has_value());
    auto x0 = magmaan::estimate::ordinal_start_values(
        *pt, *mr, workspace->moments, {});
    REQUIRE(x0.has_value());

    auto legacy = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, *stats, {}, weight_kind, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto cached = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, {}, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto snlls = magmaan::estimate::fit_ordinal_snlls(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);

    REQUIRE_MESSAGE(legacy.has_value(),
        "legacy failed: " << (legacy.has_value() ? "" : legacy.error().detail));
    REQUIRE_MESSAGE(cached.has_value(),
        "profiled bounded failed: "
            << (cached.has_value() ? "" : cached.error().detail));
    REQUIRE_MESSAGE(snlls.has_value(),
        "profiled SNLLS failed: "
            << (snlls.has_value() ? "" : snlls.error().detail));

    CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK(snlls->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
    CHECK((snlls->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
  }
}

// Cross-group threshold invariance is a structured threshold design: shared
// labels replicate across groups and merge into joint gamma coordinates. The
// legacy unprofiled fit enforces the same equalities through the constrained
// solver and is the parity oracle.
TEST_CASE("Joint threshold profiling enforces cross-group threshold invariance") {
  const Eigen::MatrixXd X1 =
      ordinal_test_block(20260612, 650, {0.84, 0.77, 0.70, 0.64}, -0.40, 0.60);
  const Eigen::MatrixXd X2 =
      ordinal_test_block(20260613, 480, {0.81, 0.73, 0.68, 0.61}, -0.35, 0.65);

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X1, X2});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t11*t1 + t12*t2\n"
      "x2 | t21*t1 + t22*t2\n"
      "x3 | t31*t1 + t32*t2\n"
      "x4 | t41*t1 + t42*t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n";
  magmaan::spec::BuildOptions build_opts;
  build_opts.n_groups = 2;
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp, build_opts);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 400;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  for (const auto estimator : {magmaan::data::OrdinalEstimatorKind::DWLS,
                               magmaan::data::OrdinalEstimatorKind::WLS}) {
    CAPTURE(static_cast<int>(estimator));
    const auto weight_kind =
        estimator == magmaan::data::OrdinalEstimatorKind::DWLS
            ? magmaan::estimate::OrdinalWeightKind::DWLS
            : magmaan::estimate::OrdinalWeightKind::WLS;
    auto plan = magmaan::data::ordinal_weight_plan(
        magmaan::data::OrdinalWorkspacePurpose::FitOnly, estimator);
    auto workspace =
        magmaan::data::ordinal_workspace_from_integer_data({X1, X2}, plan);
    REQUIRE(workspace.has_value());
    auto x0 = magmaan::estimate::ordinal_start_values(
        *pt, *mr, workspace->moments, {});
    REQUIRE(x0.has_value());

    auto legacy = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, *stats, {}, weight_kind, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto cached = magmaan::estimate::fit_ordinal_bounded(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, {}, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto snlls = magmaan::estimate::fit_ordinal_snlls(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto full = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
        *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);

    REQUIRE_MESSAGE(legacy.has_value(),
        "legacy failed: " << (legacy.has_value() ? "" : legacy.error().detail));
    REQUIRE_MESSAGE(cached.has_value(),
        "profiled bounded failed: "
            << (cached.has_value() ? "" : cached.error().detail));
    REQUIRE_MESSAGE(snlls.has_value(),
        "profiled SNLLS failed: "
            << (snlls.has_value() ? "" : snlls.error().detail));
    REQUIRE_MESSAGE(full.has_value(),
        "full-threshold SNLLS failed: "
            << (full.has_value() ? "" : full.error().detail));

    CHECK(cached->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK(snlls->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK(full->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
    CHECK((cached->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
    CHECK((snlls->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
    CHECK((full->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);

    // The invariance must be materialized (some equality group spans two
    // distinct threshold free parameters) and hold exactly in the profiled
    // solution.
    auto pt_prepared = *pt;
    auto prep = magmaan::estimate::prepare_ordinal_delta_partable(
        pt_prepared, workspace->moments);
    REQUIRE(prep.has_value());
    REQUIRE(pt_prepared.eq_groups.size() ==
            static_cast<std::size_t>(pt_prepared.n_free()));
    std::map<std::int32_t, std::vector<std::int32_t>> threshold_groups;
    for (std::size_t row = 0; row < pt_prepared.size(); ++row) {
      if (pt_prepared.op[row] != magmaan::parse::Op::Threshold) continue;
      const std::int32_t fr = pt_prepared.free[row];
      if (fr <= 0) continue;
      threshold_groups[pt_prepared.eq_groups[static_cast<std::size_t>(fr - 1)]]
          .push_back(fr);
    }
    bool found_cross_group_merge = false;
    for (auto& [group, members] : threshold_groups) {
      std::sort(members.begin(), members.end());
      members.erase(std::unique(members.begin(), members.end()),
                    members.end());
      if (members.size() < 2) continue;
      found_cross_group_merge = true;
      for (std::size_t k = 1; k < members.size(); ++k) {
        CHECK(snlls->theta(members[k] - 1) ==
              doctest::Approx(snlls->theta(members[0] - 1)).epsilon(1e-10));
        CHECK(cached->theta(members[k] - 1) ==
              doctest::Approx(cached->theta(members[0] - 1)).epsilon(1e-10));
      }
    }
    REQUIRE(found_cross_group_merge);
  }
}

// WLS couples the threshold and correlation residual blocks (W_tr != 0), so
// the profiled threshold rows stay active and depend on the correlation
// residual. General-H profiling must reproduce the bounded and full-threshold
// fits in exactly this regime, and contradictory threshold constraints must
// fail loudly.
TEST_CASE("Profiled WLS threshold maps couple thresholds and correlations") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260614, 560, {0.86, 0.79, 0.72, 0.68}, -0.55, 0.45);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  const char* syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | a*t1 + b*t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n"
      "a + b == 0\n";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 400;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS);
  auto workspace =
      magmaan::data::ordinal_workspace_from_integer_data({X}, plan);
  REQUIRE(workspace.has_value());
  auto x0 = magmaan::estimate::ordinal_start_values(
      *pt, *mr, workspace->moments, {});
  REQUIRE(x0.has_value());

  auto legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto full = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
      *pt, *mr, workspace->moments, &workspace->gamma_cache, plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);

  REQUIRE_MESSAGE(legacy.has_value(),
      "legacy WLS failed: "
          << (legacy.has_value() ? "" : legacy.error().detail));
  REQUIRE_MESSAGE(snlls.has_value(),
      "profiled SNLLS WLS failed: "
          << (snlls.has_value() ? "" : snlls.error().detail));
  REQUIRE_MESSAGE(full.has_value(),
      "full-threshold SNLLS WLS failed: "
          << (full.has_value() ? "" : full.error().detail));
  CHECK(snlls->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
  CHECK(full->fmin == doctest::Approx(legacy->fmin).epsilon(1e-8));
  CHECK((snlls->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);
  CHECK((full->theta - legacy->theta).cwiseAbs().maxCoeff() < 3e-5);

  // Contradictory threshold constraints must produce a clear error, not a
  // silent fit.
  const char* bad_syntax =
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | a*t1 + b*t2\n"
      "x2 | t1 + t2\n"
      "x3 | t1 + t2\n"
      "x4 | t1 + t2\n"
      "x1 ~*~ 1*x1\n"
      "x2 ~*~ 1*x2\n"
      "x3 ~*~ 1*x3\n"
      "x4 ~*~ 1*x4\n"
      "a + b == 0\n"
      "a + b == 0.5\n";
  auto bad_fp = magmaan::parse::Parser::parse(bad_syntax);
  REQUIRE(bad_fp.has_value());
  auto bad_pt = magmaan::spec::build(*bad_fp);
  REQUIRE(bad_pt.has_value());
  auto bad_mr = magmaan::model::build_matrix_rep(*bad_pt);
  REQUIRE(bad_mr.has_value());
  auto bad_x0 = magmaan::estimate::ordinal_start_values(
      *bad_pt, *bad_mr, workspace->moments, {});
  REQUIRE(bad_x0.has_value());
  auto bad = magmaan::estimate::fit_ordinal_snlls(
      *bad_pt, *bad_mr, workspace->moments, &workspace->gamma_cache, plan,
      *bad_x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_FALSE(bad.has_value());
  CHECK(bad.error().detail.find("infeasible") != std::string::npos);
}
