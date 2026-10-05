#include "ordinal_test_helpers.hpp"
#include "../../src/data/detail_gamma_reference.hpp"

namespace {

void check_local_ordinal_gamma_against_dense(
    const Eigen::MatrixXi& Xcat, const std::vector<std::int32_t>& levels,
    const Eigen::VectorXd& thresholds, const Eigen::MatrixXd& R, double step) {
  // The full-Gamma WLS helpers retain the global row-score/sandwich algebra.
  auto direct = magmaan::data::ordinal_gamma_diag_data_influence(Xcat, levels, thresholds, R);
  auto full = magmaan::data::ordinal_gamma_data_influence(Xcat, levels, thresholds, R);
  auto D = magmaan::data::ordinal_gamma_diag_jacobian_fd(Xcat, levels, thresholds, R, step);
  auto Dfull = magmaan::data::ordinal_gamma_jacobian_fd(Xcat, levels, thresholds, R, step);
  REQUIRE_MESSAGE(direct.has_value(), (direct.has_value() ? "" : direct.error().detail));
  REQUIRE(full.has_value());
  REQUIRE_MESSAGE(D.has_value(), (D.has_value() ? "" : D.error().detail));
  REQUIRE(Dfull.has_value());
  const Eigen::Index m = D->rows();
  REQUIRE(full->cols() == m * m);
  REQUIRE(Dfull->rows() == m * m);
  REQUIRE(direct->allFinite());
  REQUIRE(D->allFinite());
  for (Eigen::Index k = 0; k < m; ++k) {
    CAPTURE(k);
    const auto reference_if = full->col(k * (m + 1));
    const auto reference_D = Dfull->row(k * (m + 1));
    CHECK((direct->col(k) - reference_if).cwiseAbs().maxCoeff() <
          1e-9 * (1.0 + reference_if.cwiseAbs().maxCoeff()));
    CHECK((D->row(k) - reference_D).cwiseAbs().maxCoeff() <
          2e-7 * (1.0 + reference_D.cwiseAbs().maxCoeff()));
  }
  CHECK(direct->colwise().mean().cwiseAbs().maxCoeff() <
        1e-10 * (1.0 + direct->cwiseAbs().maxCoeff()));
}

}  // namespace

TEST_CASE("Ordinal local Gamma matches dense algebra across items and off-root moments") {
  for (const std::vector<std::int32_t>& levels :
       {std::vector<std::int32_t>{2, 2, 2, 2}, {2, 3, 4, 2}}) {
    std::mt19937 rng(20260909);
    std::normal_distribution<double> normal;
    Eigen::MatrixXd X(120, 4);
    for (Eigen::Index r = 0; r < X.rows(); ++r) {
      const double factor = normal(rng);
      for (Eigen::Index j = 0; j < X.cols(); ++j) {
        const double y = 0.65 * factor + std::sqrt(1.0 - 0.65 * 0.65) * normal(rng);
        const int count = levels[static_cast<std::size_t>(j)];
        X(r, j) = 1 + std::min(count - 1, static_cast<int>(count * std_normal_cdf(y)));
      }
    }
    auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, false);
    REQUIRE(stats.has_value());
    REQUIRE(stats->n_levels[0] == levels);
    for (const bool perturb : {false, true}) {
      CAPTURE(perturb);
      Eigen::VectorXd th = stats->thresholds[0];
      Eigen::MatrixXd R = stats->R[0];
      if (perturb) {
        for (Eigen::Index k = 0; k < th.size(); ++k) th(k) += k % 2 ? 0.031 : -0.047;
        R(2, 0) = R(0, 2) = R(2, 0) + 0.035;
      }
      for (const double step : {1e-5, 1e-4, 3e-4}) {
        CAPTURE(step);
        check_local_ordinal_gamma_against_dense(stats->int_data[0], levels, th, R, step);
      }
    }
  }
}

TEST_CASE("Ordinal local Gamma preserves sparse-cell and high-correlation calculations") {
  Eigen::MatrixXd counts(2, 2);
  counts << 40, 0, 3, 41;
  const Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X}, false);
  REQUIRE(stats.has_value());
  for (const double rho : {0.96, 0.995}) {
    CAPTURE(rho);
    Eigen::MatrixXd R = stats->R[0];
    R(1, 0) = R(0, 1) = rho;
    for (const double step : {1e-5, 1e-4})
      check_local_ordinal_gamma_against_dense(stats->int_data[0], stats->n_levels[0],
                                             stats->thresholds[0], R, step);
  }
}

TEST_CASE("Ordinal local Gamma retains the global threshold conditioning gate") {
  Eigen::MatrixXi X(120, 2);
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    X(r, 0) = static_cast<int>(r % 2);
    X(r, 1) = static_cast<int>((r / 2) % 2);
  }
  Eigen::VectorXd th(2);
  th << 7.8, 0;
  Eigen::MatrixXd R(2, 2);
  R << 1, 0.3, 0.3, 1;
  const std::vector<std::int32_t> levels{2, 2};
  // Both scalar blocks pass individually; the off-root tail threshold fails
  // the relative cutoff set by the much larger other item's information.
  for (Eigen::Index j = 0; j < 2; ++j) {
    const Eigen::MatrixXi item = X.col(j);
    REQUIRE(magmaan::data::ordinal_gamma_diag_data_influence(
        item, {2}, th.segment(j, 1), Eigen::MatrixXd::Identity(1, 1)).has_value());
  }
  CHECK_FALSE(magmaan::data::ordinal_gamma_data_influence(X, levels, th, R).has_value());
  CHECK_FALSE(magmaan::data::ordinal_gamma_diag_data_influence(X, levels, th, R).has_value());
  CHECK_FALSE(magmaan::data::ordinal_gamma_diag_jacobian_fd(X, levels, th, R).has_value());
}

TEST_CASE("Ordinal local Gamma rejects malformed complete-data inputs and FD steps") {
  Eigen::MatrixXi X(4, 2);
  X << 0, 0, 0, 1, 1, 0, 1, 1;
  const Eigen::VectorXd th = Eigen::VectorXd::Zero(2);
  const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2);
  const std::vector<std::int32_t> levels{2, 2};
  for (const double step : {0.0, -1e-4, std::numeric_limits<double>::infinity()})
    CHECK_FALSE(magmaan::data::ordinal_gamma_diag_jacobian_fd(X, levels, th, R, step).has_value());
  CHECK_FALSE(magmaan::data::ordinal_gamma_diag_data_influence(X, {2}, th, R).has_value());
  X(0, 0) = -1;
  CHECK_FALSE(magmaan::data::ordinal_gamma_diag_data_influence(X, levels, th, R).has_value());
  X(0, 0) = 2;
  CHECK_FALSE(magmaan::data::ordinal_gamma_diag_jacobian_fd(X, levels, th, R).has_value());
}

TEST_CASE("ordinal_gamma_diag_data_influence matches case-weight finite differences") {
  Eigen::MatrixXd counts(3, 3);
  counts << 11, 7, 5,
             6, 14, 9,
             4, 10, 12;
  const Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  Eigen::MatrixXi Xcat = (X.cast<int>().array() - 1).matrix();

  auto IFG = magmaan::data::ordinal_gamma_diag_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(IFG.has_value());
  auto probe = gamma_diag_influence_probe_2var(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  CHECK((probe.Gamma - stats->NACOV[0]).cwiseAbs().maxCoeff() < 1e-8);

  double max_abs = 0.0;
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const Eigen::VectorXd fd = finite_diff_gamma_diag_case_influence(probe, i);
    max_abs = std::max(
        max_abs, (IFG->row(i).transpose() - fd).cwiseAbs().maxCoeff());
  }
  const double scale = 1.0 + IFG->cwiseAbs().maxCoeff();
  CHECK(max_abs < 5e-5 * scale);
}

TEST_CASE("ordinal_gamma_data_influence matches case-weight finite differences") {
  Eigen::MatrixXd counts(3, 3);
  counts << 11, 7, 5,
             6, 14, 9,
             4, 10, 12;
  const Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  Eigen::MatrixXi Xcat = (X.cast<int>().array() - 1).matrix();

  auto IFG = magmaan::data::ordinal_gamma_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(IFG.has_value());
  auto IFG_diag = magmaan::data::ordinal_gamma_diag_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(IFG_diag.has_value());
  auto D = magmaan::data::ordinal_gamma_jacobian_fd(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(D.has_value());
  auto D_diag = magmaan::data::ordinal_gamma_diag_jacobian_fd(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(D_diag.has_value());

  const Eigen::Index m = stats->NACOV[0].rows();
  REQUIRE(IFG->cols() == m * m);
  REQUIRE(D->rows() == m * m);
  auto probe = gamma_diag_influence_probe_2var(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  CHECK((probe.Gamma - stats->NACOV[0]).cwiseAbs().maxCoeff() < 1e-8);

  double max_abs = 0.0;
  double max_diag = 0.0;
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const Eigen::MatrixXd fd = finite_diff_gamma_case_influence(probe, i);
    const Eigen::VectorXd got_vec = IFG->row(i).transpose();
    const Eigen::Map<const Eigen::MatrixXd> got(got_vec.data(), m, m);
    max_abs = std::max(max_abs, (got - fd).cwiseAbs().maxCoeff());
    max_diag = std::max(
        max_diag, (got.diagonal() - IFG_diag->row(i).transpose())
                      .cwiseAbs().maxCoeff());
  }
  for (Eigen::Index l = 0; l < m; ++l) {
    const Eigen::Map<const Eigen::MatrixXd> dG(D->col(l).data(), m, m);
    max_diag = std::max(
        max_diag, (dG.diagonal() - D_diag->col(l)).cwiseAbs().maxCoeff());
  }

  const double scale = 1.0 + IFG->cwiseAbs().maxCoeff();
  CHECK(max_abs < 5e-5 * scale);
  CHECK(max_diag < 1e-8 * scale);
}

TEST_CASE("ordinal observed gamma influence handles pairwise missing data") {
  Eigen::MatrixXd counts(3, 3);
  counts << 11, 7, 5,
             6, 14, 9,
             4, 10, 12;
  Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    if (r % 11 == 0) X(r, 0) = nan;
    if (r % 13 == 0) X(r, 1) = nan;
  }

  auto stats = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap);
  REQUIRE(stats.has_value());
  REQUIRE(stats->int_data.size() == 1);
  const Eigen::MatrixXi& Xcat = stats->int_data[0];

  auto IFG = magmaan::data::ordinal_observed_gamma_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(IFG.has_value());
  auto IFG_diag = magmaan::data::ordinal_observed_gamma_diag_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(IFG_diag.has_value());
  auto D = magmaan::data::ordinal_observed_gamma_jacobian_fd(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(D.has_value());
  auto D_diag = magmaan::data::ordinal_observed_gamma_diag_jacobian_fd(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  REQUIRE(D_diag.has_value());

  const Eigen::Index m = stats->NACOV[0].rows();
  REQUIRE(IFG->rows() == X.rows());
  REQUIRE(IFG->cols() == m * m);
  REQUIRE(D->rows() == m * m);
  auto probe = gamma_diag_influence_probe_2var(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  CHECK((probe.Gamma - stats->NACOV[0]).cwiseAbs().maxCoeff() < 1e-8);

  double max_abs = 0.0;
  double max_diag = 0.0;
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    const Eigen::MatrixXd fd = finite_diff_gamma_case_influence(probe, i);
    const Eigen::VectorXd got_vec = IFG->row(i).transpose();
    const Eigen::Map<const Eigen::MatrixXd> got(got_vec.data(), m, m);
    max_abs = std::max(max_abs, (got - fd).cwiseAbs().maxCoeff());
    max_diag = std::max(max_diag,
        (got.diagonal() - IFG_diag->row(i).transpose()).cwiseAbs().maxCoeff());
  }
  for (Eigen::Index l = 0; l < m; ++l) {
    const Eigen::Map<const Eigen::MatrixXd> dG(D->col(l).data(), m, m);
    max_diag = std::max(
        max_diag, (dG.diagonal() - D_diag->col(l)).cwiseAbs().maxCoeff());
  }

  const double scale = 1.0 + IFG->cwiseAbs().maxCoeff();
  CHECK(max_abs < 5e-5 * scale);
  CHECK(max_diag < 1e-8 * scale);
}

TEST_CASE("ordinal observed gamma influence reduces to complete-data influence") {
  Eigen::MatrixXd counts(3, 3);
  counts << 11, 7, 5,
             6, 14, 9,
             4, 10, 12;
  const Eigen::MatrixXd X = ordinal_data_from_pair_counts(counts);
  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());
  Eigen::MatrixXi Xcat = (X.cast<int>().array() - 1).matrix();

  auto complete_diag = magmaan::data::ordinal_gamma_diag_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  auto observed_diag = magmaan::data::ordinal_observed_gamma_diag_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  auto complete_full = magmaan::data::ordinal_gamma_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  auto observed_full = magmaan::data::ordinal_observed_gamma_data_influence(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  auto complete_D = magmaan::data::ordinal_gamma_jacobian_fd(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  auto observed_D = magmaan::data::ordinal_observed_gamma_jacobian_fd(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  auto complete_D_diag = magmaan::data::ordinal_gamma_diag_jacobian_fd(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);
  auto observed_D_diag = magmaan::data::ordinal_observed_gamma_diag_jacobian_fd(
      Xcat, stats->n_levels[0], stats->thresholds[0], stats->R[0]);

  REQUIRE(complete_diag.has_value());
  REQUIRE(observed_diag.has_value());
  REQUIRE(complete_full.has_value());
  REQUIRE(observed_full.has_value());
  REQUIRE(complete_D.has_value());
  REQUIRE(observed_D.has_value());
  REQUIRE(complete_D_diag.has_value());
  REQUIRE(observed_D_diag.has_value());
  CHECK(observed_diag->isApprox(*complete_diag, 1e-10));
  CHECK(observed_full->isApprox(*complete_full, 1e-10));
  CHECK(observed_D->isApprox(*complete_D, 1e-8));
  CHECK(observed_D_diag->isApprox(*complete_D_diag, 1e-8));
}

TEST_CASE("robust_ordinal_ij rejects estimated-weight stats without integer data") {
  Eigen::MatrixXd counts(3, 3);
  counts << 11, 7, 5,
             6, 14, 9,
             4, 10, 12;
  auto stats = magmaan::data::ordinal_stats_from_integer_data(
      {ordinal_data_from_pair_counts(counts)});
  REQUIRE(stats.has_value());
  stats->int_data.clear();

  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(mr.has_value());

  magmaan::estimate::Estimates est;
  for (const auto weight : {magmaan::estimate::OrdinalWeightKind::DWLS,
                            magmaan::estimate::OrdinalWeightKind::WLS}) {
    auto r = magmaan::estimate::robust_ordinal_ij(*pt, *mr, *stats, est, weight);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().detail.find("int_data") != std::string::npos);
  }
}

TEST_CASE("robust_ordinal_ij ULS matches observed-bread fixed-weight sandwich") {
  const Eigen::MatrixXd X =
      ordinal_test_block(20260619, 320, {0.82, 0.76, 0.70, 0.64}, -0.45, 0.55);
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

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 1000;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::ULS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "ULS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto fixed = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Observed);
  auto ij = magmaan::estimate::robust_ordinal_ij(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS);
  REQUIRE_MESSAGE(fixed.has_value(),
      "fixed robust failed: "
          << (fixed.has_value() ? "" : fixed.error().detail));
  REQUIRE_MESSAGE(ij.has_value(),
      "IJ robust failed: " << (ij.has_value() ? "" : ij.error().detail));
  CHECK(ij->df == fixed->df);
  CHECK(ij->chisq_standard == doctest::Approx(fixed->chisq_standard));
  CHECK(ij->vcov.isApprox(fixed->vcov, 1e-8));
  CHECK(ij->se.isApprox(fixed->se, 1e-8));
}

TEST_CASE("robust_ordinal_ij ULS supports observed MCAR ordinal stats") {
  Eigen::MatrixXd X =
      ordinal_test_block(20260621, 360, {0.82, 0.76, 0.70, 0.64}, -0.45, 0.55);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    for (Eigen::Index c = 0; c < X.cols(); ++c) {
      if (((r + 3 * c) % 17) == 0) X(r, c) = nan;
    }
  }
  auto stats = magmaan::data::ordinal_stats_from_observed_integer_data(
      {X}, magmaan::data::OrdinalPairwiseGammaKind::Overlap,
      /*full_wls_weight=*/false);
  REQUIRE(stats.has_value());
  REQUIRE(stats->moment_influence.size() == 1);
  const Eigen::MatrixXd rebuilt =
      stats->moment_influence[0].transpose() * stats->moment_influence[0] /
      static_cast<double>(stats->n_obs[0]);
  CHECK(rebuilt.isApprox(stats->NACOV[0], 1e-9));

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
  opts.max_iter = 1000;
  opts.ftol = 1e-12;
  opts.gtol = 1e-8;
  auto fit = magmaan::estimate::fit_ordinal_bounded(
      *pt, *mr, *stats, {}, magmaan::estimate::OrdinalWeightKind::ULS, *x0,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(fit.has_value(),
      "ULS fit failed: " << (fit.has_value() ? "" : fit.error().detail));

  auto fixed = magmaan::estimate::robust_ordinal(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS,
      magmaan::estimate::OrdinalParameterization::Delta,
      magmaan::robust::Information::Observed);
  auto ij = magmaan::estimate::robust_ordinal_ij(
      *pt, *mr, *stats, *fit, magmaan::estimate::OrdinalWeightKind::ULS);
  REQUIRE_MESSAGE(fixed.has_value(),
      "fixed robust failed: "
          << (fixed.has_value() ? "" : fixed.error().detail));
  REQUIRE_MESSAGE(ij.has_value(),
      "IJ robust failed: " << (ij.has_value() ? "" : ij.error().detail));
  CHECK(ij->df == fixed->df);
  CHECK(ij->chisq_standard == doctest::Approx(fixed->chisq_standard));
  CHECK(ij->vcov.isApprox(fixed->vcov, 1e-8));
  CHECK(ij->se.isApprox(fixed->se, 1e-8));
}

TEST_CASE("Ordinal exact sampling refuses missing data and invalid steps") {
  Eigen::MatrixXi X(4,2); X << 0,0,0,1,1,0,1,1;
  const auto th = Eigen::VectorXd::Zero(2).eval();
  const auto R = Eigen::MatrixXd::Identity(2,2).eval();
  for (double step : {0.0,-1e-5,std::numeric_limits<double>::infinity()})
    CHECK_FALSE(magmaan::data::ordinal_moment_sampling_influence(X,{2,2},th,R,step).has_value());
  X(0,0) = -1;
  CHECK_FALSE(magmaan::data::ordinal_moment_sampling_influence(X,{2,2},th,R).has_value());
}

TEST_CASE("Sparse Gamma movement matches dense FD for ordinal and mixed missing groups") {
  for (bool mixed : {false, true}) for (int categories : {2, 5, 7})
    for (int groups : {1, 2}) for (int group = 0; group < groups; ++group) {
      CAPTURE(mixed); CAPTURE(categories); CAPTURE(groups); CAPTURE(group);
      std::mt19937 rng(static_cast<unsigned>(89000 + categories * 100 + groups * 10 + group));
      std::normal_distribution<double> normal;
      const std::vector<std::int32_t> ordered = mixed
          ? std::vector<std::int32_t>{1, 0, 1, 0}
          : std::vector<std::int32_t>{1, 1, 1, 1};
      std::vector<std::int32_t> levels(4, categories);
      Eigen::VectorXd thresholds((mixed ? 2 : 4) * (categories - 1));
      Eigen::Index start = 0;
      for (int j = 0; j < 4; ++j) {
        if (!ordered[static_cast<std::size_t>(j)]) { levels[static_cast<std::size_t>(j)] = 0; continue; }
        for (int k = 0; k < categories - 1; ++k)
          thresholds(start++) = categories == 2 ? 0.35 : -1.15 + 2.6 * k / (categories - 2.0) + 0.08 * j;
      }
      Eigen::MatrixXd x(500, 4);
      for (Eigen::Index r = 0; r < x.rows(); ++r) {
        const double factor = normal(rng);
        Eigen::Index offset = 0;
        for (Eigen::Index j = 0; j < x.cols(); ++j) {
          const double z = 0.5 * factor + normal(rng);
          if (ordered[static_cast<std::size_t>(j)]) {
            int category = 1;
            for (int k = 0; k < categories - 1; ++k) category += z > thresholds(offset + k);
            x(r, j) = category;
            offset += categories - 1;
          } else x(r, j) = 0.2 + 1.1 * z;
        }
      }
      Eigen::MatrixXd R = Eigen::MatrixXd::Constant(4, 4, 0.2);
      R.diagonal().setOnes();
      Eigen::VectorXd mean = Eigen::VectorXd::Zero(4);
      for (int j = 0; j < 4; ++j) if (!ordered[static_cast<std::size_t>(j)]) {
        R(j, j) = 1.3; mean(j) = 0.2;
      }
      for (bool missing : {false, true}) {
        CAPTURE(missing);
        Eigen::MatrixXd data = x;
        if (missing) for (Eigen::Index r = 0; r < data.rows(); ++r)
          for (Eigen::Index j = 0; j < data.cols(); ++j)
            if ((r * 7 + j * 11) % 17 == 0)
              data(r, j) = std::numeric_limits<double>::quiet_NaN();
        auto sparse = missing
            ? magmaan::data::mixed_observed_gamma_diag_jacobian_fd(data, ordered, levels, thresholds, mean, R, 1e-4)
            : magmaan::data::mixed_gamma_diag_jacobian_fd(data, ordered, levels, thresholds, mean, R, 1e-4);
        auto dense = missing
            ? magmaan::data::mixed_observed_gamma_diag_jacobian_fd_dense(data, ordered, levels, thresholds, mean, R, 1e-4)
            : magmaan::data::mixed_gamma_diag_jacobian_fd_dense(data, ordered, levels, thresholds, mean, R, 1e-4);
        REQUIRE_MESSAGE(sparse.has_value(), (sparse.has_value() ? "" : sparse.error().detail));
        REQUIRE(dense.has_value());
        CHECK(sparse->isApprox(*dense, 1e-8));
        CHECK(((sparse.value() - dense.value()).array().abs() /
              (1.0 + dense->array().abs())).maxCoeff() <= 1e-8);
        if (!mixed) {
          Eigen::MatrixXi cat(data.rows(), data.cols());
          for (Eigen::Index r = 0; r < data.rows(); ++r)
            for (Eigen::Index j = 0; j < data.cols(); ++j)
              cat(r, j) = std::isfinite(data(r, j)) ? static_cast<int>(data(r, j)) - 1 : -1;
          auto ordinal = missing
              ? magmaan::data::ordinal_observed_gamma_diag_jacobian_fd(cat, levels, thresholds, R, 1e-4)
              : magmaan::data::ordinal_gamma_diag_jacobian_fd(cat, levels, thresholds, R, 1e-4);
          REQUIRE_MESSAGE(ordinal.has_value(), (ordinal.has_value() ? "" : ordinal.error().detail));
          CHECK(ordinal->isApprox(*dense, 1e-8));
          CHECK(((ordinal.value() - dense.value()).array().abs() /
                (1.0 + dense->array().abs())).maxCoeff() <= 1e-8);
          if (missing) {
            auto reference = magmaan::data::ordinal_observed_gamma_diag_jacobian_fd_dense(cat, levels, thresholds, R, 1e-4);
            REQUIRE(reference.has_value());
            CHECK(((ordinal.value() - reference.value()).array().abs() /
                  (1.0 + reference->array().abs())).maxCoeff() <= 1e-8);
          }
        }
      }
    }
}
