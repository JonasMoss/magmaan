#include "magmaan/estimate/nt.hpp"
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

TEST_CASE("Ordinal exact first stage matches replicated case weights and keeps OPG default") {
  using namespace magmaan;
  for (int groups : {1, 2}) {
    std::vector<Eigen::MatrixXd> blocks;
    for (int g = 0; g < groups; ++g)
      blocks.push_back(ordinal_test_block(7400u + static_cast<unsigned>(g), 150, {0.82, 0.76, 0.70, 0.64}, -0.45, 0.55));
    auto stats = data::ordinal_stats_from_integer_data(blocks, false);
    REQUIRE(stats.has_value());
    for (int g = 0; g < groups; ++g) {
      const auto b = static_cast<std::size_t>(g);
      auto exact = data::ordinal_moment_sampling_influence(stats->int_data[b],
          stats->n_levels[b], stats->thresholds[b], stats->R[b]);
      REQUIRE_MESSAGE(exact.has_value(), (exact.has_value() ? "" : exact.error().detail));
      CHECK((exact->gamma - exact->rows.transpose()*exact->rows/150.0).norm() == 0.0);
      for (int row : {0, 17, 91}) {
        constexpr int copies = 1000;
        Eigen::VectorXd kappa[2];
        for (int side = 0; side < 2; ++side) {
          const int step = side == 0 ? -1 : 1;
          Eigen::MatrixXd repeated(150*copies+step, 4);
          int off = 0;
          for (int i = 0; i < 150; ++i)
            for (int j = 0; j < copies+(i == row ? step : 0); ++j)
              repeated.row(off++) = blocks[b].row(i);
          auto perturbed = data::ordinal_stats_from_integer_data({repeated}, false);
          REQUIRE(perturbed.has_value());
          kappa[side].resize(stats->thresholds[b].size()+6);
          kappa[side].head(stats->thresholds[b].size()) = perturbed->thresholds[0];
          Eigen::Index k = stats->thresholds[b].size();
          for (int j = 0; j < 4; ++j)
            for (int i = j+1; i < 4; ++i) kappa[side](k++) = perturbed->R[0](i,j);
        }
        const Eigen::VectorXd fd = (kappa[1]-kappa[0])*(150*copies/2.0);
        const double error = (fd-exact->rows.row(row).transpose()).norm()/fd.norm();
        MESSAGE("exact ordinal case-weight relative error " << error);
        CHECK(error <= 1e-5);
        const Eigen::Index nth = stats->thresholds[b].size();
        CHECK((fd.head(nth)-exact->rows.row(row).head(nth).transpose()).norm()/fd.head(nth).norm() <= 1e-5);
        CHECK((fd.tail(6)-exact->rows.row(row).tail(6).transpose()).norm()/fd.tail(6).norm() <= 1e-5);
      }
    }
    auto parsed = parse::Parser::parse(
        "f =~ x1 + 0.9*x2 + x3 + x4\n"
        "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
        "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n");
    REQUIRE(parsed.has_value());
    spec::BuildOptions build; build.n_groups = groups;
    auto pt = spec::build(*parsed, build);
    REQUIRE(pt.has_value());
    auto rep = model::build_matrix_rep(*pt);
    REQUIRE(rep.has_value());
    for (auto par : {estimate::OrdinalParameterization::Delta, estimate::OrdinalParameterization::Theta}) {
      optim::OptimOptions options; options.max_iter = 1500;
      auto fit = test::fit_ordinal_bounded(*pt, *rep, *stats, {},
          estimate::OrdinalWeightKind::DWLS, estimate::Backend::NloptLbfgs, options, par);
      REQUIRE_MESSAGE(fit.has_value(), (fit.has_value() ? "" : fit.error().detail));
      auto implicit = estimate::robust_ordinal_ij(*pt,*rep,*stats,*fit,estimate::OrdinalWeightKind::DWLS,par);
      auto opg = estimate::robust_ordinal_ij(*pt,*rep,*stats,*fit,estimate::OrdinalWeightKind::DWLS,par,nullptr,estimate::OrdinalFirstStage::OPG);
      auto exact = estimate::robust_ordinal_ij(*pt,*rep,*stats,*fit,estimate::OrdinalWeightKind::DWLS,par,nullptr,estimate::OrdinalFirstStage::Exact);
      REQUIRE(implicit.has_value()); REQUIRE(opg.has_value());
      REQUIRE_MESSAGE(exact.has_value(), (exact.has_value() ? "" : exact.error().detail));
      CHECK((implicit->vcov.array() == opg->vcov.array()).all());
      CHECK((implicit->se.array() == opg->se.array()).all());
      CHECK(exact->vcov.allFinite());
      CHECK((exact->vcov-opg->vcov).norm() > 0.0);
    }
  }
}

TEST_CASE("Ordinal exact and OPG IJ converge under a Gaussian copula") {
  using namespace magmaan;
  auto parsed = parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n");
  REQUIRE(parsed.has_value());
  auto pt = spec::build(*parsed); REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep.has_value());
  double errors[3]{};
  const int sizes[3]{250, 1000, 4000};
  for (int k = 0; k < 3; ++k) {
    for (unsigned seed = 0; seed < 8; ++seed) {
      auto X = ordinal_test_block(7450u+seed, sizes[k], {0.82,0.76,0.70,0.64}, -0.45,0.55);
      auto stats = data::ordinal_stats_from_integer_data({X},false); REQUIRE(stats.has_value());
      auto fit = test::fit_ordinal_bounded(*pt,*rep,*stats,{},estimate::OrdinalWeightKind::DWLS);
      REQUIRE(fit.has_value());
      auto opg = estimate::robust_ordinal_ij(*pt,*rep,*stats,*fit,estimate::OrdinalWeightKind::DWLS);
      auto exact = estimate::robust_ordinal_ij(*pt,*rep,*stats,*fit,estimate::OrdinalWeightKind::DWLS,
          estimate::OrdinalParameterization::Delta,nullptr,estimate::OrdinalFirstStage::Exact);
      REQUIRE(opg.has_value()); REQUIRE(exact.has_value());
      const double error = (opg->vcov-exact->vcov).norm()/exact->vcov.norm();
      errors[k] += error*error/8.0;
    }
    errors[k] = std::sqrt(errors[k]);
    MESSAGE("Gaussian ordinal IJ RMS relative gap N=" << sizes[k] << ": " << errors[k]);
  }
  CHECK(errors[1] < errors[0]);
  CHECK(errors[2] < errors[1]);
  // Sixteenfold N should give about a fourfold reduction; allow Monte Carlo
  // variation across the eight deterministic Gaussian samples.
  CHECK(errors[0]/errors[2] > 2.0);
  CHECK(errors[0]/errors[2] < 8.0);
}

TEST_CASE("association ML IJ joint transport and independent objective derivatives") {
  using namespace magmaan;
  auto stats = data::ordinal_stats_from_integer_data({ordinal_test_block(
      322, 500, {0.82,0.76,0.7,0.64}, -0.45,0.55)}, false);
  REQUIRE(stats.has_value());
  auto parsed = parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2");
  REQUIRE(parsed.has_value());
  auto pt = spec::build(*parsed); REQUIRE(pt.has_value());
  REQUIRE(estimate::prepare_ordinal_delta_partable(*pt,*stats).has_value());
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep.has_value());
  auto start = estimate::ordinal_start_values(*pt,*rep,*stats,{}); REQUIRE(start.has_value());
  auto fit = estimate::frontier::fit_ml(*pt,*rep,*stats,*start);
  REQUIRE_MESSAGE(fit.has_value(), (fit ? "" : fit.error().detail));
  // Evaluate off the optimum to ensure observed curvature is being tested.
  fit->theta(0) *= 1.04;
  auto ij = estimate::frontier::association_ml_ij(*pt,*rep,*stats,*fit);
  REQUIRE_MESSAGE(ij.has_value(), (ij ? "" : ij.error().detail));
  auto ev = model::ModelEvaluator::build(*pt,*rep); REQUIRE(ev.has_value());
  auto q = [&](const Eigen::VectorXd& theta) {
    auto evaluation = ev->evaluate(theta,false,false); REQUIRE(evaluation.has_value());
    auto corr = model::correlation_evaluation(*evaluation); REQUIRE(corr.has_value());
    const auto& C = corr->moments.sigma[0];
    return 0.5*(std::log(C.determinant()) + (stats->R[0]*C.inverse()).trace()
                - std::log(stats->R[0].determinant())-static_cast<double>(C.rows()));
  };
  CHECK(ij->value == doctest::Approx(q(fit->theta)).epsilon(1e-10));
  const auto& K = ij->coordinates;
  constexpr double h = 1e-4;
  for (Eigen::Index j = 0; j < K.cols(); ++j) {
    CHECK(ij->score(j) == doctest::Approx((q(fit->theta+h*K.col(j))-q(fit->theta-h*K.col(j)))/(2*h)).scale(1).epsilon(1e-7));
    for (Eigen::Index k = 0; k < K.cols(); ++k) {
      double H = (q(fit->theta+h*K.col(j)+h*K.col(k))
                - q(fit->theta+h*K.col(j)-h*K.col(k))
                - q(fit->theta-h*K.col(j)+h*K.col(k))
                + q(fit->theta-h*K.col(j)-h*K.col(k)))/(4*h*h);
      CHECK(ij->sensitivity(j,k) == doctest::Approx(H).scale(1).epsilon(2e-6));
    }
  }
  CHECK((ij->vcov-ij->influence[0].transpose()*ij->influence[0]/250000.0).norm() < 1e-14);
  CHECK((ij->vcov_active-ij->influence_active[0].transpose()*ij->influence_active[0]/250000.0).norm() < 1e-12);
  CHECK(ij->influence[0].colwise().mean().norm() < 1e-10);
  // Perturb the association target while holding the evaluation point fixed.
  for (Eigen::Index c = 0, col = 0; c < 4; ++c) for (Eigen::Index r = c+1; r < 4; ++r, ++col) {
    auto plus = *stats, minus = *stats;
    plus.R[0](r,c) += 1e-5; plus.R[0](c,r) += 1e-5;
    minus.R[0](r,c) -= 1e-5; minus.R[0](c,r) -= 1e-5;
    auto hi = estimate::frontier::association_ml_ij(*pt,*rep,plus,*fit);
    auto lo = estimate::frontier::association_ml_ij(*pt,*rep,minus,*fit);
    REQUIRE(hi.has_value()); REQUIRE(lo.has_value());
    CHECK(((hi->score-lo->score)/2e-5-ij->target_derivative[0].col(col)).norm() < 1e-9);
  }
  // A population target exactly on the model eliminates the curvature term.
  auto evaluated = ev->evaluate(fit->theta,true,false); REQUIRE(evaluated.has_value());
  auto correlation = model::correlation_evaluation(*evaluated); REQUIRE(correlation.has_value());
  const auto& population = correlation->moments.sigma[0];
  std::array<double,4> loading;
  loading[0] = std::sqrt(population(0,1)*population(0,2)/population(1,2));
  for (std::size_t j = 1; j < loading.size(); ++j)
    loading[j] = population(0,static_cast<Eigen::Index>(j))/loading[0];
  auto large = data::ordinal_stats_from_integer_data({ordinal_test_block(
      3224,4000,loading,-0.45,0.55)},false);
  REQUIRE(large.has_value());
  auto exact_stats = *large;
  exact_stats.R = correlation->moments.sigma;
  auto exact_fit = *fit;
  auto exact_layout = estimate::ordinal_association_layout(*pt,*rep,exact_stats,fit->theta);
  REQUIRE(exact_layout.has_value()); exact_fit.theta = exact_layout->theta;
  auto exact = estimate::frontier::association_ml_ij(*pt,*rep,exact_stats,exact_fit);
  REQUIRE(exact.has_value());
  const Eigen::MatrixXd inverse = exact_stats.R[0].inverse();
  const Eigen::MatrixXd J = correlation->J_sigma*K;
  Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(K.cols(),K.cols());
  for (Eigen::Index j = 0; j < K.cols(); ++j) {
    Eigen::MatrixXd dC(4,4);
    for (Eigen::Index c = 0, row = 0; c < 4; ++c) for (Eigen::Index r = c; r < 4; ++r, ++row)
      dC(r,c) = dC(c,r) = J(row,j);
    for (Eigen::Index k = 0; k < K.cols(); ++k) {
      Eigen::MatrixXd eC(4,4);
      for (Eigen::Index c = 0, row = 0; c < 4; ++c) for (Eigen::Index r = c; r < 4; ++r, ++row)
        eC(r,c) = eC(c,r) = J(row,k);
      expected(j,k) = .5*(inverse*dC*inverse*eC).trace();
    }
  }
  CHECK((exact->sensitivity-expected).norm()/expected.norm() < 1e-7);
  auto sampling = data::ordinal_moment_sampling_influence(exact_stats.int_data[0],exact_stats.n_levels[0],
      exact_stats.thresholds[0],exact_stats.R[0]); REQUIRE(sampling.has_value());
  const Eigen::MatrixXd fixed = expected.inverse()*exact->target_derivative[0]*
      sampling->gamma.bottomRightCorner(6,6)*exact->target_derivative[0].transpose()*expected.inverse()/4000.0;
  CHECK((fixed-exact->vcov_active).norm()/fixed.norm() < 1e-7);
  auto missing = *stats; missing.int_data[0](0,0) = -1;
  CHECK_FALSE(estimate::frontier::association_ml_ij(*pt,*rep,missing,*fit).has_value());
  CHECK_FALSE(estimate::frontier::association_ml_ij(*pt,*rep,*stats,*fit,nullptr,true).has_value());
}

namespace {
Eigen::MatrixXd association_skewed_cases(unsigned seed, int n) {
  std::mt19937 rng(seed);
  std::gamma_distribution<double> factor(1.5,2.0), error(2.5,2.0);
  Eigen::MatrixXd X(n,4);
  for (int i = 0; i < n; ++i) {
    const double f = (factor(rng)-3)/std::sqrt(6.0);
    for (int j = 0; j < 4; ++j) {
      const double l = .8-.07*j;
      const double z = l*f+std::sqrt(1-l*l)*(error(rng)-5)/std::sqrt(10.0);
      X(i,j) = z < -.45 ? 1 : (z < .55 ? 2 : 3);
    }
  }
  return X;
}
}

TEST_CASE("association ML IJ nonnormal case weights and stratified delete one") {
  using namespace magmaan;
  auto parsed = parse::Parser::parse("f =~ x1 + l2*x2 + l3*x3 + x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2");
  REQUIRE(parsed.has_value());
  for (int n : {250,1000,4000}) {
    CAPTURE(n);
    std::vector<Eigen::MatrixXd> X{association_skewed_cases(3221,n/2),
                                 association_skewed_cases(3222,n-n/2)};
    auto stats = data::ordinal_stats_from_integer_data(X,false); REQUIRE(stats.has_value());
    spec::BuildOptions build; build.n_groups = 2;
    auto pt = spec::build(*parsed,build); REQUIRE(pt.has_value());
    REQUIRE(estimate::prepare_ordinal_delta_partable(*pt,*stats).has_value());
    auto rep = model::build_matrix_rep(*pt); REQUIRE(rep.has_value());
    auto start = estimate::ordinal_start_values(*pt,*rep,*stats,{}); REQUIRE(start.has_value());
    optim::OptimOptions options; options.ftol = 1e-14; options.gtol = 1e-11; options.max_iter = 1000;
    auto refit = [&](const data::OrdinalStats& target, const Eigen::VectorXd& initial) {
      auto fitted = estimate::frontier::fit_ml(*pt,*rep,target,initial,estimate::Backend::NloptLbfgs,options);
      REQUIRE_MESSAGE(fitted.has_value(), (fitted ? "" : fitted.error().detail));
      // Newton polish analytic objective gradients independently of the IJ.
      auto layout = estimate::ordinal_association_layout(*pt,*rep,target,fitted->theta); REQUIRE(layout.has_value());
      const auto K = layout->constraints.K();
      auto ev = model::ModelEvaluator::build(*pt,*rep); REQUIRE(ev.has_value());
      data::SampleStats sample{target.R,{},target.n_obs};
      auto cache = estimate::ml_prepare(sample,model::MomentTarget::Correlation); REQUIRE(cache.has_value());
      auto gradient = [&](const Eigen::VectorXd& theta) {
        auto e = ev->evaluate(theta,true,false); REQUIRE(e.has_value());
        auto c = model::correlation_evaluation(*e); REQUIRE(c.has_value());
        auto vg = estimate::ml_value_gradient(sample,*cache,c->moments,c->J_sigma); REQUIRE(vg.has_value());
        return Eigen::VectorXd(.5*K.transpose()*vg->gradient);
      };
      for (int iter = 0; iter < 8; ++iter) {
        auto s = gradient(fitted->theta);
        if (s.norm() < 1e-13) break;
        Eigen::MatrixXd H(K.cols(),K.cols());
        for (Eigen::Index j = 0; j < K.cols(); ++j)
          H.col(j) = (gradient(fitted->theta+1e-4*K.col(j))-gradient(fitted->theta-1e-4*K.col(j)))/2e-4;
        fitted->theta -= K*H.fullPivLu().solve(s);
      }
      return *fitted;
    };
    auto fitted = refit(*stats,*start);
    auto ij = estimate::frontier::association_ml_ij(*pt,*rep,*stats,fitted);
    REQUIRE_MESSAGE(ij.has_value(), (ij ? "" : ij.error().detail));
    if (n == 250) for (std::size_t b = 0; b < X.size(); ++b) {
      auto plus = *stats, minus = *stats;
      plus.R[b](1,0) += 1e-4; plus.R[b](0,1) += 1e-4;
      minus.R[b](1,0) -= 1e-4; minus.R[b](0,1) -= 1e-4;
      const Eigen::VectorXd fd = (refit(plus,fitted.theta).theta-refit(minus,fitted.theta).theta)/2e-4;
      const double w = static_cast<double>(stats->n_obs[b])/n;
      const Eigen::VectorXd prediction = -w*ij->coordinates*ij->sensitivity.fullPivLu().solve(ij->target_derivative[b].col(0));
      CHECK((fd-prediction).norm()/prediction.norm() < 1e-5);
    }
    Eigen::MatrixXd jk = Eigen::MatrixXd::Zero(fitted.theta.size(),fitted.theta.size());
    for (std::size_t b = 0; b < X.size(); ++b) {
      const auto nb = X[b].rows();
      Eigen::MatrixXd deleted(nb,fitted.theta.size());
      for (Eigen::Index i = 0; i < nb; ++i) {
        auto blocks = X;
        blocks[b].resize(nb-1,4);
        if (i > 0) blocks[b].topRows(i) = X[b].topRows(i);
        if (i+1 < nb) blocks[b].bottomRows(nb-i-1) = X[b].bottomRows(nb-i-1);
        auto target = data::ordinal_stats_from_integer_data(blocks,false); REQUIRE(target.has_value());
        // Fixed allocation: deleting a row changes its empirical distribution,
        // not the Stage-2 stratum proportion.
        target->n_obs = stats->n_obs;
        deleted.row(i) = refit(*target,fitted.theta).theta;
      }
      const Eigen::RowVectorXd mean = deleted.colwise().mean();
      deleted.rowwise() -= mean;
      jk.noalias() += (static_cast<double>(nb-1)/static_cast<double>(nb))*deleted.transpose()*deleted;
      if (n == 250) for (int row : {0,17,91}) for (int copies : {50,100}) {
        // Keep perturbations above the pairwise solver's 1e-9 rho resolution;
        // two central-difference steps gate both truncation and solver error.
        Eigen::VectorXd weighted[2], stage1[2];
        for (int side = 0; side < 2; ++side) {
          const int step = side == 0 ? -1 : 1;
          auto blocks = X;
          blocks[b].resize(nb*copies+step,4);
          Eigen::Index off = 0;
          for (Eigen::Index i = 0; i < nb; ++i)
            for (int j = 0; j < copies+(i == row ? step : 0); ++j)
              blocks[b].row(off++) = X[b].row(i);
          auto target = data::ordinal_stats_from_integer_data(blocks,false); REQUIRE(target.has_value());
          stage1[side].resize(target->thresholds[b].size()+6);
          stage1[side].head(target->thresholds[b].size()) = target->thresholds[b];
          Eigen::Index kappa_col = target->thresholds[b].size();
          for (int c = 0; c < 4; ++c) for (int r = c+1; r < 4; ++r)
            stage1[side](kappa_col++) = target->R[b](r,c);
          target->n_obs = stats->n_obs;
          weighted[side] = refit(*target,fitted.theta).theta;
        }
        Eigen::VectorXd fd = (weighted[1]-weighted[0])*(n*copies/2.0);
        const double relative = (fd-ij->influence[b].row(row).transpose()).norm()/fd.norm();
        auto stage = data::ordinal_moment_sampling_influence(stats->int_data[b],stats->n_levels[b],stats->thresholds[b],stats->R[b]);
        REQUIRE(stage.has_value());
        const Eigen::VectorXd stage_fd = (stage1[1]-stage1[0])*(static_cast<double>(nb)*copies/2.0);
        MESSAGE("association nonnormal case-weight error " << relative << " group " << b
                << " Stage1 error " << (stage_fd-stage->rows.row(row).transpose()).norm()/stage_fd.norm());
        CHECK(relative <= 1e-5);
        const Eigen::VectorXd alpha_fd = ij->coordinates.colPivHouseholderQr().solve(fd);
        CHECK((alpha_fd-ij->influence_active[b].row(row).transpose()).norm()/alpha_fd.norm() <= 1e-5);
        CHECK((stage_fd-stage->rows.row(row).transpose()).norm()/stage_fd.norm() <= 1e-5);
      }
    }
    const double relative = (jk-ij->vcov).norm()/ij->vcov.norm();
    MESSAGE("association stratified delete-one N=" << n << " relative covariance error " << relative);
    CHECK(relative < (n == 250 ? .2 : (n == 1000 ? .08 : .03)));
  }
}

TEST_CASE("association ML IJ identification charts and typed refusals") {
  using namespace magmaan;
  auto X = association_skewed_cases(3223,700);
  auto stats = data::ordinal_stats_from_integer_data({X},false); REQUIRE(stats.has_value());
  auto parsed = parse::Parser::parse("f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2");
  REQUIRE(parsed.has_value());
  auto unidentified_parsed = parse::Parser::parse("f =~ NA*x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2");
  REQUIRE(unidentified_parsed.has_value());
  auto unidentified = spec::build(*unidentified_parsed); REQUIRE(unidentified.has_value());
  REQUIRE(estimate::prepare_ordinal_delta_partable(*unidentified,*stats).has_value());
  auto unidentified_rep = model::build_matrix_rep(*unidentified); REQUIRE(unidentified_rep.has_value());
  auto unidentified_start = estimate::ordinal_start_values(*unidentified,*unidentified_rep,*stats,{});
  REQUIRE(unidentified_start.has_value());
  estimate::Estimates unidentified_point; unidentified_point.theta = *unidentified_start;
  auto unidentified_result = estimate::frontier::association_ml_ij(*unidentified,*unidentified_rep,*stats,unidentified_point);
  REQUIRE_FALSE(unidentified_result.has_value());
  CHECK(unidentified_result.error().detail.find("unidentified") != std::string::npos);
  CHECK(unidentified_result.error().kind == PostError::Kind::InfoMatrixSingular);
  Eigen::MatrixXd correlation_covariance;
  for (bool std_lv : {false,true}) {
    spec::BuildOptions options; options.std_lv = std_lv;
    auto pt = spec::build(*parsed,options); REQUIRE(pt.has_value());
    REQUIRE(estimate::prepare_ordinal_delta_partable(*pt,*stats).has_value());
    auto rep = model::build_matrix_rep(*pt); REQUIRE(rep.has_value());
    auto start = estimate::ordinal_start_values(*pt,*rep,*stats,{}); REQUIRE(start.has_value());
    optim::OptimOptions opts; opts.ftol = 1e-14; opts.gtol = 1e-10;
    auto fit = estimate::frontier::fit_ml(*pt,*rep,*stats,*start,estimate::Backend::NloptLbfgs,opts);
    REQUIRE(fit.has_value());
    auto ij = estimate::frontier::association_ml_ij(*pt,*rep,*stats,*fit); REQUIRE(ij.has_value());
    auto ev = model::ModelEvaluator::build(*pt,*rep); REQUIRE(ev.has_value());
    auto evaluation = ev->evaluate(fit->theta,true,false); REQUIRE(evaluation.has_value());
    auto corr = model::correlation_evaluation(*evaluation); REQUIRE(corr.has_value());
    Eigen::MatrixXd covariance = corr->J_sigma*ij->vcov*corr->J_sigma.transpose();
    if (!std_lv) correlation_covariance = covariance;
    else CHECK((correlation_covariance-covariance).norm()/covariance.norm() < 1e-5);
    auto unsupported = *stats; unsupported.n_levels[0][0] = 0;
    CHECK_FALSE(estimate::frontier::association_ml_ij(*pt,*rep,unsupported,*fit).has_value());
    auto nonlinear = *pt; nonlinear.nonlinear_eq_rows.push_back(0);
    auto refusal = estimate::frontier::association_ml_ij(nonlinear,*rep,*stats,*fit);
    REQUIRE_FALSE(refusal.has_value());
    CHECK(refusal.error().kind == PostError::Kind::UnsupportedInference);
    // A vanishing latent variance destroys the association Jacobian rank.
    if (!std_lv) {
      auto deficient = *fit;
      for (std::size_t row = 0; row < pt->size(); ++row)
        if (pt->op[row] == parse::Op::Covariance && pt->lhs_var[row] == pt->rhs_var[row] &&
            pt->lhs_var[row] >= 0 && pt->ov_pos[static_cast<std::size_t>(pt->lhs_var[row])] < 0 && pt->free[row] > 0)
          deficient.theta(pt->free[row]-1) = 0;
      auto boundary = estimate::frontier::association_ml_ij(*pt,*rep,*stats,deficient);
      REQUIRE_FALSE(boundary.has_value());
      CHECK(boundary.error().kind == PostError::Kind::UnsupportedInference);
      CHECK(boundary.error().detail.find("PSD boundary") != std::string::npos);
    }
  }
}

TEST_CASE("association ML global and nested spectral reconstruction") {
  using namespace magmaan;
  const std::string thresholds = "\nx1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2";
  for (bool grouped : {false,true}) {
    CAPTURE(grouped);
    auto block = ordinal_test_block(3233,500,{.8,.73,.66,.59},-.45,.55);
    auto stats = data::ordinal_stats_from_integer_data(
        grouped ? std::vector<Eigen::MatrixXd>{block,block} : std::vector<Eigen::MatrixXd>{block},false);
    REQUIRE(stats.has_value());
    // Identical strata have identical pseudo-true parameters even when the
    // larger one-factor model is misspecified by a residual association.
    if (grouped) for (auto& R : stats->R) { R(3,0) += .07; R(0,3) += .07; }
    auto build = [&](const std::string& syntax, spec::LatentNames* names = nullptr) {
      auto parsed = parse::Parser::parse(syntax+thresholds); REQUIRE(parsed.has_value());
      spec::BuildOptions opts; opts.n_groups = grouped ? 2 : 1;
      auto pt = spec::build(*parsed,opts,nullptr,names); REQUIRE(pt.has_value());
      REQUIRE(estimate::prepare_ordinal_delta_partable(*pt,*stats).has_value());
      return *pt;
    };
    spec::LatentNames alt_names, null_names;
    auto alt = build("f =~ x1 + x2 + x3 + x4",&alt_names);
    auto null = build(grouped ? "f =~ x1 + l2*x2 + x3 + x4" : "f =~ x1 + l*x2 + l*x3 + x4",&null_names);
    auto ra = model::build_matrix_rep(alt,&alt_names), rn = model::build_matrix_rep(null,&null_names);
    REQUIRE(ra.has_value()); REQUIRE(rn.has_value());
    auto fit = [&](const spec::LatentStructure& pt,const model::MatrixRep& rep) {
      auto start = estimate::ordinal_start_values(pt,rep,*stats,{}); REQUIRE(start.has_value());
      optim::OptimOptions opts; opts.ftol = 1e-14; opts.gtol = 1e-10; opts.max_iter = 1000;
      auto result = estimate::frontier::fit_ml(pt,rep,*stats,*start,estimate::Backend::NloptLbfgs,opts);
      REQUIRE_MESSAGE(result.has_value(),(result ? "" : result.error().detail));
      return *result;
    };
    auto ea = fit(alt,*ra), en = fit(null,*rn);
    auto ij = estimate::frontier::association_ml_ij(alt,*ra,*stats,ea); REQUIRE(ij.has_value());
    auto global = estimate::frontier::association_ml_global_test(alt,*ra,*stats,ea);
    REQUIRE_MESSAGE(global.has_value(),(global ? "" : global.error().detail));
    const int groups = grouped ? 2 : 1;
    const double N = 500*groups;
    CHECK(global->df == 2*groups);
    CHECK(global->statistic == doctest::Approx(2*N*ij->value).epsilon(1e-12));
    auto ev = model::ModelEvaluator::build(alt,*ra); REQUIRE(ev.has_value());
    auto evaluated = ev->evaluate(ea.theta,true,false); REQUIRE(evaluated.has_value());
    auto corr = model::correlation_evaluation(*evaluated); REQUIRE(corr.has_value());
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(6*groups,6*groups);
    Eigen::MatrixXd G = V;
    Eigen::MatrixXd Delta(6*groups,ij->coordinates.cols());
    // Independent explicit basis reconstruction, including stratified N scale.
    for (int b = 0; b < groups; ++b) {
      const auto block_index = static_cast<std::size_t>(b);
      Eigen::MatrixXd inverse = corr->moments.sigma[block_index].inverse();
      std::vector<Eigen::MatrixXd> E;
      for (int c = 0; c < 4; ++c) for (int r = c+1; r < 4; ++r) {
        Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(4,4); basis(r,c)=basis(c,r)=1;
        E.push_back(basis);
        const int vech = c*4-c*(c-1)/2+r-c;
        Delta.row(6*b+static_cast<int>(E.size())-1) = corr->J_sigma.row(10*b+vech)*ij->coordinates;
      }
      for (int j=0;j<6;++j) for (int k=0;k<6;++k)
        V(6*b+j,6*b+k) = (inverse*E[static_cast<std::size_t>(j)]*inverse*E[static_cast<std::size_t>(k)]).trace()/(2*groups);
      auto sampling = data::ordinal_moment_sampling_influence(stats->int_data[block_index],stats->n_levels[block_index],stats->thresholds[block_index],stats->R[block_index]);
      REQUIRE(sampling.has_value());
      Eigen::MatrixXd rows = sampling->rows.rightCols(6);
      G.block(6*b,6*b,6,6) = groups*rows.transpose()*rows/500;
    }
    Eigen::MatrixXd U = V-V*Delta*(Delta.transpose()*V*Delta).inverse()*Delta.transpose()*V;
    Eigen::EigenSolver<Eigen::MatrixXd> eigen(U*G,false);
    REQUIRE(eigen.info()==Eigen::Success);
    Eigen::VectorXd lambda = eigen.eigenvalues().real();
    std::sort(lambda.data(),lambda.data()+lambda.size());
    CHECK((global->spectrum-lambda.tail(global->df)).norm() < 1e-10);
    CHECK((global->metric-V).norm() < 1e-12);
    CHECK((global->gamma-G).norm() < 1e-12);
    CHECK((global->tangent-Delta).norm() < 1e-12);
    auto nested = estimate::frontier::association_ml_nested_test(alt,*ra,*stats,ea,null,*rn,en);
    REQUIRE_MESSAGE(nested.has_value(),(nested ? "" : nested.error().detail));
    CHECK(nested->df == 1);
    auto i0 = estimate::frontier::association_ml_ij(null,*rn,*stats,en); REQUIRE(i0.has_value());
    CHECK(nested->statistic == doctest::Approx(std::max(0.0,2*N*(i0->value-ij->value))).scale(1).epsilon(1e-12));
    const Eigen::MatrixXd Hinv = ij->sensitivity.inverse();
    // Construct the known loading restriction without the nesting worker.
    Eigen::RowVectorXd full_A = Eigen::RowVectorXd::Zero(ea.theta.size());
    int found = 0;
    for (std::size_t row = 0; row < alt.size(); ++row) {
      if (alt.op[row] != parse::Op::Measurement || alt.free[row] <= 0) continue;
      const auto ov = alt.ov_pos[static_cast<std::size_t>(alt.rhs_var[row])];
      if ((grouped && ov == 1) || (!grouped && (ov == 1 || ov == 2)))
        full_A(alt.free[row]-1) = found++ == 0 ? 1 : -1;
    }
    REQUIRE(found==2);
    Eigen::MatrixXd independent_A = full_A*ij->coordinates;
    Eigen::MatrixXd independent_C = independent_A*Hinv*independent_A.transpose();
    Eigen::MatrixXd independent_S = independent_A*Hinv*ij->meat*Hinv.transpose()*independent_A.transpose();
    CHECK(nested->spectrum(0) == doctest::Approx(independent_S(0,0)/independent_C(0,0)).epsilon(1e-10));
    const auto& A = nested->restriction;
    Eigen::MatrixXd C = A*Hinv*A.transpose();
    Eigen::MatrixXd S = A*Hinv*ij->meat*Hinv.transpose()*A.transpose();
    CHECK((nested->C-C).norm() < 1e-10);
    CHECK((nested->S-S).norm() < 1e-10);
    CHECK(nested->spectrum(0) == doctest::Approx(S(0,0)/C(0,0)).epsilon(1e-10));
    for (const auto* reference : {&nested->all,&nested->sb,&nested->peba4}) {
      CHECK(reference->p_value >= 0); CHECK(reference->p_value <= 1);
      CHECK(reference->lambdas_reference.sum() == doctest::Approx(nested->spectrum.sum()).epsilon(1e-12));
    }
    if (grouped) {
      CHECK(ij->value > 1e-4);
      CHECK(nested->statistic < 1e-7);
    }
    auto identical = estimate::frontier::association_ml_nested_test(alt,*ra,*stats,ea,alt,*ra,ea);
    REQUIRE(identical.has_value()); CHECK(identical->df==0); CHECK(identical->statistic==0);
    CHECK(std::isnan(identical->all.p_value));
    auto refused = estimate::frontier::association_ml_global_test(alt,*ra,*stats,ea,nullptr,true);
    REQUIRE_FALSE(refused.has_value()); CHECK(refused.error().kind==PostError::Kind::UnsupportedInference);
    if (!grouped) {
      spec::LatentNames other_names;
      auto other_model = build("g =~ x1 + x2 + x3 + x4",&other_names);
      auto other_rep = model::build_matrix_rep(other_model,&other_names); REQUIRE(other_rep.has_value());
      auto other_fit = fit(other_model,*other_rep);
      auto moment_pair = estimate::frontier::association_ml_nested_test(
          alt,*ra,*stats,ea,other_model,*other_rep,other_fit);
      REQUIRE_FALSE(moment_pair.has_value());
      CHECK((moment_pair.error().kind == PostError::Kind::NotNested ||
             moment_pair.error().kind == PostError::Kind::UnsupportedNesting));
      auto reversed = estimate::frontier::association_ml_nested_test(null,*rn,*stats,en,alt,*ra,ea);
      CHECK_FALSE(reversed.has_value());
      auto exact = *stats; exact.R = corr->moments.sigma;
      exact.int_data[0] = exact.int_data[0].replicate(8,1).eval();
      exact.n_obs[0] *= 8;  // N=4000 with the same empirical target distribution.
      auto test = estimate::frontier::association_ml_global_test(alt,*ra,exact,ea); REQUIRE(test.has_value());
      auto exact_ij = estimate::frontier::association_ml_ij(alt,*ra,exact,ea); REQUIRE(exact_ij.has_value());
      CHECK(test->statistic < 1e-9);
      CHECK((exact_ij->sensitivity-Delta.transpose()*V*Delta).norm() < 1e-8);
      Eigen::EigenSolver<Eigen::MatrixXd> reconstructed(test->residual*test->gamma,false);
      Eigen::VectorXd l = reconstructed.eigenvalues().real(); std::sort(l.data(),l.data()+l.size());
      CHECK((test->spectrum-l.tail(test->df)).norm() < 1e-10);
    }
  }
  // Three-indicator one-factor model is saturated in association coordinates.
  Eigen::MatrixXd block = ordinal_test_block(3234,500,{.8,.73,.66,.59},-.45,.55).leftCols(3);
  auto stats = data::ordinal_stats_from_integer_data({block},false); REQUIRE(stats.has_value());
  auto parsed = parse::Parser::parse("f =~ x1 + x2 + x3\nx1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2"); REQUIRE(parsed.has_value());
  auto pt = spec::build(*parsed); REQUIRE(pt.has_value());
  REQUIRE(estimate::prepare_ordinal_delta_partable(*pt,*stats).has_value());
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep.has_value());
  auto start = estimate::ordinal_start_values(*pt,*rep,*stats,{});
  REQUIRE_MESSAGE(start.has_value(),(start ? "" : start.error().detail));
  if (!start) return;
  auto fit = estimate::frontier::fit_ml(*pt,*rep,*stats,*start);
  REQUIRE_MESSAGE(fit.has_value(),(fit ? "" : fit.error().detail));
  if (!fit) return;
  auto test = estimate::frontier::association_ml_global_test(*pt,*rep,*stats,*fit);
  REQUIRE(test.has_value()); CHECK(test->df==0); CHECK(test->spectrum.size()==0);
  CHECK(std::isnan(test->all.p_value));
}
