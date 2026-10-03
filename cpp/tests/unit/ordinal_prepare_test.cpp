#include "ordinal_test_helpers.hpp"
#include "magmaan/compat/lavaan/partable_view.hpp"

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
    X(r, 0) = static_cast<double>(r == 0 ? 1 : (r + 1 == X.rows() ? 5 : 2 + (r % 3)));
    X(r, 1) = static_cast<double>(1 + ((2 * r + r / 7) % 5));
    X(r, 2) = static_cast<double>(1 + ((3 * r + r / 11) % 5));
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
  auto pt = magmaan::spec::build(*fp);
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
  auto pt = magmaan::spec::build(*fp, {}, &starts);
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
  auto pt = magmaan::spec::build(*fp);
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
      magmaan::estimate::Backend::NloptLbfgs, {}, OrdinalParameterization::Delta);
  auto theta = magmaan::test::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, OrdinalWeightKind::DWLS,
      magmaan::estimate::Backend::NloptLbfgs, {}, OrdinalParameterization::Theta);
  REQUIRE_MESSAGE(delta.has_value(),
      "delta fit failed: " << (delta.has_value() ? "" : delta.error().detail));
  REQUIRE_MESSAGE(theta.has_value(),
      "theta fit failed: " << (theta.has_value() ? "" : theta.error().detail));

  CHECK(std::isfinite(theta->fmin));
  // NB: no iteration / f_eval assertion — `simple_start_values` on this
  // 4-indicator synthetic lands close enough to the optimum that NLopt's
  // gradient at the start point is already under tolerance, returning at
  // n_evals = 0. The test's invariant is reparameterization equivalence
  // (same fmin), not the optimizer's evaluation count.
  // Reparameterization invariance: same minimized discrepancy.
  CHECK(theta->fmin == doctest::Approx(delta->fmin).epsilon(1e-4));
  // The parameter vectors themselves differ — theta loadings are unstandardized.
  CHECK((theta->theta - delta->theta).cwiseAbs().maxCoeff() > 1e-3);
}

TEST_CASE("Theta threshold profile preserves objective and off-optimum gradient") {
  for (bool diagonal : {true, false}) {
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(5, 5);
    F.diagonal() << 0.8, 1.4, 0.9, 1.3, 1.1;
    if (!diagonal) {
      F(2, 0) = 0.4; F(3, 1) = -0.3; F(4, 0) = 0.2;
      F(4, 2) = 0.25;
    }
    // A diagonal weight keeps a diagonal Schur complement, so the profile must
    // hand back a Diagonal operator rather than a materialized dense matrix.
    const auto FW = diagonal
        ? magmaan::detail::WhitenFactor::diagonal(F.diagonal())
        : magmaan::detail::WhitenFactor::dense(F);
    auto w = magmaan::detail::theta_threshold_profile(FW, 2, diagonal);
    REQUIRE(w.has_value());
    CHECK(w->factor.kind() == (diagonal
        ? magmaan::detail::WhitenFactor::Kind::Diagonal
        : magmaan::detail::WhitenFactor::Kind::Dense));
    for (double beta : {-0.7, 0.2, 1.1}) {
      const auto residual = [&](double b) {
        return Eigen::Vector3d(std::sin(b) - 0.2, b*b - 0.4, b + 0.3);
      };
      const Eigen::Vector3d d = residual(beta);
      Eigen::VectorXd full(5);
      full.head(2) = -w->threshold_from_corr * d;
      full.tail(3) = d;
      const Eigen::VectorXd wr = F.transpose() * full;
      const Eigen::VectorXd reduced = w->factor.apply(d);
      CHECK(wr.squaredNorm() == doctest::Approx(reduced.squaredNorm()).epsilon(1e-12));
      CHECK((F.topRows(2) * wr).norm() < 1e-12);
      const Eigen::Vector3d derivative(std::cos(beta), 2*beta, 1);
      const double gradient = w->factor.apply(derivative).dot(reduced);
      const double eps = 1e-6;
      const double fd = ((w->factor.apply(residual(beta + eps))).squaredNorm() -
                        (w->factor.apply(residual(beta - eps))).squaredNorm()) / (4*eps);
      CHECK(gradient == doctest::Approx(fd).epsilon(1e-8));
    }
  }
}

TEST_CASE("Cache-aware ordinal theta fits and SNLLS use fit-only workspaces") {
  std::mt19937 rng(20260611);
  std::normal_distribution<double> norm(0.0, 1.0);
  Eigen::MatrixXd X(640, 4);
  const double loading[4] = {0.88, 0.80, 0.70, 0.62};
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      const double eps = std::sqrt(1.0 - loading[j] * loading[j]) * norm(rng);
      const double y = loading[j] * eta + eps;
      X(i, j) = 1.0 + (y > -0.65) + (y > 0.45);
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
  opts.max_iter = 500;
  opts.ftol = 1e-10;
  opts.gtol = 1e-7;

  auto uls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::ULS,
      magmaan::data::OrdinalMomentParameterization::Theta);
  auto uls_workspace =
      magmaan::data::ordinal_workspace_from_integer_data({X}, uls_plan);
  REQUIRE(uls_workspace.has_value());
  CHECK(uls_workspace->gamma_cache.block_count() == 0);
  auto uls_bounded = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, uls_workspace->moments, nullptr, {}, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  auto uls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, uls_workspace->moments, nullptr, uls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(uls_bounded.has_value(),
      "theta ULS bounded failed: "
          << (uls_bounded.has_value() ? "" : uls_bounded.error().detail));
  REQUIRE_MESSAGE(uls_snlls.has_value(),
      "theta ULS SNLLS failed: "
          << (uls_snlls.has_value() ? "" : uls_snlls.error().detail));
  CHECK(uls_snlls->fmin == doctest::Approx(uls_bounded->fmin).epsilon(1e-7));
  CHECK((uls_snlls->theta - uls_bounded->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(uls_snlls->n_linear > 0);

  auto dwls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::DWLS,
      magmaan::data::OrdinalMomentParameterization::Theta);
  auto dwls_workspace =
      magmaan::data::ordinal_workspace_from_integer_data({X}, dwls_plan);
  REQUIRE(dwls_workspace.has_value());
  REQUIRE(dwls_workspace->gamma_cache.block_count() == 1);
  CHECK(dwls_workspace->gamma_cache.blocks[0].has_diagonal);
  CHECK_FALSE(dwls_workspace->gamma_cache.blocks[0].has_full);
  auto dwls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::DWLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts,
      magmaan::estimate::OrdinalParameterization::Theta);
  auto dwls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, dwls_workspace->moments, &dwls_workspace->gamma_cache, {},
      dwls_plan, *x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  auto dwls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, dwls_workspace->moments, &dwls_workspace->gamma_cache,
      dwls_plan, *x0, magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(dwls_legacy.has_value(),
      "theta legacy DWLS failed: "
          << (dwls_legacy.has_value() ? "" : dwls_legacy.error().detail));
  REQUIRE_MESSAGE(dwls_cached.has_value(),
      "theta cached DWLS failed: "
          << (dwls_cached.has_value() ? "" : dwls_cached.error().detail));
  REQUIRE_MESSAGE(dwls_snlls.has_value(),
      "theta DWLS SNLLS failed: "
          << (dwls_snlls.has_value() ? "" : dwls_snlls.error().detail));
  CHECK(dwls_cached->fmin ==
        doctest::Approx(dwls_legacy->fmin).epsilon(1e-8));
  CHECK((dwls_cached->theta - dwls_legacy->theta).cwiseAbs().maxCoeff() <
        5e-5);
  CHECK(dwls_snlls->fmin ==
        doctest::Approx(dwls_legacy->fmin).epsilon(1e-7));
  CHECK((dwls_snlls->theta - dwls_legacy->theta).cwiseAbs().maxCoeff() <
        8e-5);
  CHECK(dwls_snlls->n_linear > 0);

  auto wls_plan = magmaan::data::ordinal_weight_plan(
      magmaan::data::OrdinalWorkspacePurpose::FitOnly,
      magmaan::data::OrdinalEstimatorKind::WLS,
      magmaan::data::OrdinalMomentParameterization::Theta);
  magmaan::data::OrdinalGammaCache wls_cache;
  wls_cache.blocks.resize(1);
  wls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_cache.blocks[0].has_full = true;
  auto wls_legacy = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::WLS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts,
      magmaan::estimate::OrdinalParameterization::Theta);
  auto wls_cached = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, moments, &wls_cache, {}, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  magmaan::data::OrdinalGammaCache wls_snlls_cache;
  wls_snlls_cache.blocks.resize(1);
  wls_snlls_cache.blocks[0].gamma = stats->NACOV[0];
  wls_snlls_cache.blocks[0].has_full = true;
  auto wls_snlls = magmaan::estimate::fit_ordinal_snlls(
      *pt, *mr, moments, &wls_snlls_cache, wls_plan, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(wls_legacy.has_value(),
      "theta legacy WLS failed: "
          << (wls_legacy.has_value() ? "" : wls_legacy.error().detail));
  REQUIRE_MESSAGE(wls_cached.has_value(),
      "theta cached WLS failed: "
          << (wls_cached.has_value() ? "" : wls_cached.error().detail));
  REQUIRE_MESSAGE(wls_snlls.has_value(),
      "theta WLS SNLLS failed: "
          << (wls_snlls.has_value() ? "" : wls_snlls.error().detail));
  CHECK(wls_cached->fmin ==
        doctest::Approx(wls_legacy->fmin).epsilon(1e-7));
  CHECK((wls_cached->theta - wls_legacy->theta).cwiseAbs().maxCoeff() <
        1e-4);
  CHECK(wls_snlls->fmin ==
        doctest::Approx(wls_legacy->fmin).epsilon(1e-6));
  CHECK((wls_snlls->theta - wls_legacy->theta).cwiseAbs().maxCoeff() <
        2e-4);
  CHECK(wls_cache.blocks[0].has_wls_weight);
  CHECK(wls_snlls_cache.blocks[0].has_wls_weight);

  for (const auto estimator : {magmaan::data::OrdinalEstimatorKind::ULS,
                              magmaan::data::OrdinalEstimatorKind::DWLS,
                              magmaan::data::OrdinalEstimatorKind::WLS}) {
    const auto plan = magmaan::data::ordinal_weight_plan(
        magmaan::data::OrdinalWorkspacePurpose::FitOnly, estimator,
        magmaan::data::OrdinalMomentParameterization::Theta);
    auto cache = magmaan::data::OrdinalGammaCache{};
    cache.blocks.resize(1);
    cache.blocks[0].gamma = stats->NACOV[0];
    cache.blocks[0].has_full = true;
    auto fast = magmaan::estimate::fit_ordinal_snlls(
        *pt, *mr, moments, &cache, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto generic = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
        *pt, *mr, moments, &cache, plan, *x0,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    REQUIRE(fast.has_value());
    REQUIRE(generic.has_value());
    CHECK(fast->fmin == doctest::Approx(generic->fmin).epsilon(1e-8));
    CHECK((fast->theta - generic->theta).norm() < 3e-4);
    CHECK(magmaan::estimate::fit_verdict(*fast).status == magmaan::estimate::FitCheck::Passed);
    CHECK(magmaan::estimate::fit_verdict(*generic).status == magmaan::estimate::FitCheck::Passed);
  }
  for (bool shared : {false, true}) {
    std::string constrained_syntax(syntax);
    const auto first = constrained_syntax.find("x1 | t1");
    constrained_syntax.replace(first, 7, shared ? "x1 | a*t1" : "x1 | 0*t1");
    if (shared) {
      const auto second = constrained_syntax.find("x2 | t1");
      constrained_syntax.replace(second, 7, "x2 | a*t1");
    }
    auto parsed = magmaan::parse::Parser::parse(constrained_syntax);
    REQUIRE(parsed.has_value());
    auto constrained = magmaan::spec::build(*parsed);
    REQUIRE(constrained.has_value());
    auto rep = magmaan::model::build_matrix_rep(*constrained);
    REQUIRE(rep.has_value());
    auto start = magmaan::estimate::ordinal_start_values(*constrained, *rep, moments, {});
    REQUIRE(start.has_value());
    auto fallback = magmaan::estimate::fit_ordinal_snlls(
        *constrained, *rep, moments, nullptr, uls_plan, *start,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    auto generic = magmaan::estimate::fit_ordinal_snlls_full_thresholds(
        *constrained, *rep, moments, nullptr, uls_plan, *start,
        magmaan::estimate::Backend::NloptLbfgs, opts);
    REQUIRE(fallback.has_value());
    REQUIRE(generic.has_value());
    CHECK(fallback->f_evals == generic->f_evals);
    CHECK((fallback->theta - generic->theta).norm() == 0);
  }

}

TEST_CASE("Ordinal preparation provenance preserves released scales and rejects changed layouts") {
  using namespace magmaan;
  data::OrdinalStats stats;
  stats.R = {Eigen::MatrixXd::Identity(4, 4)};
  stats.threshold_ov = {{0, 0, 1, 1, 2, 2, 3, 3}};
  stats.threshold_level = {{1, 2, 1, 2, 1, 2, 1, 2}};
  stats.thresholds = {Eigen::VectorXd::Zero(8)};
  for (const char* scale : {"NA", "1"}) {
    const std::string syntax = std::string("f =~ y1+a*y2+b*y3+y4\na == 2*b\n") +
        "y1 | t1+t2\ny2 | t1+t2\ny3 | t1+t2\ny4 | t1+t2\n" +
        "y1 ~*~ " + scale + "*y1\n";
    auto parsed = parse::Parser::parse(syntax);
    REQUIRE(parsed);
    spec::Starts starts;
    spec::LatentNames names;
    auto pt = spec::build(*parsed, {}, &starts, &names);
    REQUIRE(pt);
    for (std::size_t k = 0; k < starts.hint.size(); ++k)
      starts.hint[k] = static_cast<double>(k + 1);
    REQUIRE(estimate::prepare_ordinal_delta_partable(*pt, stats, &starts, &names.row_user));
    CHECK(pt->n_free() == (std::string(scale) == "NA" ? 13 : 12));
    REQUIRE_FALSE(pt->lin_constraint_R.empty());
    const auto prepared = *pt;
    const auto hints = starts.hint;
    REQUIRE(estimate::prepare_ordinal_delta_partable(*pt, stats, &starts));
    CHECK(pt->free == prepared.free);
    CHECK(pt->eq_groups == prepared.eq_groups);
    CHECK(pt->lin_constraint_R == prepared.lin_constraint_R);
    CHECK(pt->lin_constraint_d == prepared.lin_constraint_d);
    REQUIRE(starts.hint.size() == hints.size());
    for (std::size_t k = 0; k < hints.size(); ++k) {
      if (std::isnan(hints[k])) CHECK(std::isnan(starts.hint[k]));
      else CHECK(starts.hint[k] == hints[k]);
    }
    for (std::size_t i = 0; i < pt->size(); ++i) {
      if (std::isnan(prepared.fixed_value[i])) CHECK(std::isnan(pt->fixed_value[i]));
      else CHECK(pt->fixed_value[i] == prepared.fixed_value[i]);
    }
    auto projection = compat::lavaan::to_lavaan_partable(*pt, names, starts);
    auto restored = compat::lavaan::from_lavaan_partable(projection);
    CHECK(restored.structure.ordinal_preparation == pt->ordinal_preparation);
    REQUIRE(estimate::prepare_ordinal_delta_partable(restored.structure, stats, &restored.starts));
    CHECK(restored.structure.free == pt->free);
    data::OrdinalMoments moments;
    moments.R = stats.R;
    moments.threshold_ov = stats.threshold_ov;
    REQUIRE(estimate::prepare_ordinal_delta_partable(*pt, moments, &starts));
    data::MixedOrdinalStats mixed;
    mixed.R = stats.R;
    mixed.ordered = {{1, 1, 1, 1}};
    mixed.threshold_ov = stats.threshold_ov;
    REQUIRE(estimate::prepare_mixed_ordinal_delta_partable(*pt, mixed, &starts));
    mixed.ordered[0][0] = 0;
    CHECK_FALSE(estimate::prepare_mixed_ordinal_delta_partable(*pt, mixed, &starts));
    CHECK(pt->free == prepared.free);
    REQUIRE(starts.hint.size() == hints.size());
    for (std::size_t k = 0; k < hints.size(); ++k) {
      if (std::isnan(hints[k])) CHECK(std::isnan(starts.hint[k]));
      else CHECK(starts.hint[k] == hints[k]);
    }
    auto changed = stats;
    changed.threshold_ov[0][1] = 1; // y1 now binary: its scale release is vetoed.
    CHECK_FALSE(estimate::prepare_ordinal_delta_partable(*pt, changed));
    changed.threshold_ov[0] = {1, 1, 2, 2, 3, 3}; // y1 is no longer ordered.
    CHECK_FALSE(estimate::prepare_ordinal_delta_partable(*pt, changed));
    CHECK(pt->free == prepared.free);
  }
}

TEST_CASE("Explicit response scales retain residual coordinates with auto_var disabled") {
  auto flat = magmaan::parse::Parser::parse(
      "f =~ x1+x2+x3\nf ~~ f\n"
      "x1 | t1+t2\nx2 | t1+t2\nx3 | t1+t2\n"
      "x1 ~*~ c(1,NA)*x1\nx2 ~*~ c(1,1)*x2\nx3 ~*~ c(1,1)*x3");
  REQUIRE(flat.has_value());
  magmaan::spec::BuildOptions options;
  options.auto_var = false;
  options.n_groups = 2;
  magmaan::spec::LatentNames names;
  magmaan::spec::Starts starts;
  auto pt = magmaan::spec::build(*flat, options, &starts, &names);
  REQUIRE(pt.has_value());
  int residuals = 0;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->op[i] != magmaan::parse::Op::Covariance ||
        names.row_lhs[i] != names.row_rhs[i] || names.row_lhs[i] == "f") continue;
    ++residuals;
    CHECK(names.row_user[i] == 0);
    CHECK(pt->free[i] == 0);
    CHECK(pt->fixed_value[i] == 1.0);
  }
  CHECK(residuals == 6);
  magmaan::data::OrdinalStats stats;
  stats.R = {Eigen::MatrixXd::Identity(3,3), Eigen::MatrixXd::Identity(3,3)};
  stats.threshold_ov = {{0,0,1,1,2,2},{0,0,1,1,2,2}};
  stats.threshold_level = {{1,2,1,2,1,2},{1,2,1,2,1,2}};
  stats.thresholds = {Eigen::VectorXd::Zero(6),Eigen::VectorXd::Zero(6)};
  REQUIRE(magmaan::estimate::prepare_ordinal_delta_partable(
      *pt, stats, &starts, &names.row_user).has_value());
  int released = 0;
  for (std::size_t i = 0; i < pt->size(); ++i) {
    if (pt->op[i] == magmaan::parse::Op::Covariance &&
        names.row_lhs[i] == "x1" && names.row_rhs[i] == "x1" && pt->group[i] == 2) {
      CHECK(pt->free[i] > 0);
      ++released;
    }
  }
  CHECK(released == 1);
}

TEST_CASE("DELTA scale restrictions fail before losing their coordinates") {
  using namespace magmaan;
  data::OrdinalStats stats;
  stats.R = {Eigen::MatrixXd::Identity(3, 3)};
  stats.threshold_ov = {{0, 0, 1, 1, 2, 2}};
  stats.threshold_level = {{1, 2, 1, 2, 1, 2}};
  stats.thresholds = {Eigen::VectorXd::Zero(6)};
  for (const char* restriction : {"x1 ~*~ shared*x1; x2 ~*~ shared*x2",
      "x1 ~*~ shared*x1; f =~ shared*x2",
      "x1 ~*~ a*x1; x2 ~*~ b*x2; a == 2*b",
      "x1 ~*~ 0.8*x1"}) {
    auto flat = parse::Parser::parse(std::string(
        "f =~ x1+x2+x3\nx1 | t1+t2\nx2 | t1+t2\nx3 | t1+t2\n") + restriction);
    REQUIRE(flat);
    auto pt = spec::build(*flat);
    REQUIRE(pt);
    const auto original = *pt;
    auto prepared = estimate::prepare_ordinal_delta_partable(*pt, stats);
    REQUIRE_FALSE(prepared);
    CHECK(prepared.error().kind == FitError::Kind::NumericIssue);
    CHECK(prepared.error().detail.find("unsupported DELTA response scale") != std::string::npos);
    CHECK(prepared.error().detail.find("theta") != std::string::npos);
    CHECK(pt->free == original.free);
    CHECK(pt->eq_groups == original.eq_groups);
    CHECK(pt->ordinal_preparation.empty());
  }
}
