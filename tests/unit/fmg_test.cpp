#include <doctest/doctest.h>

#include <cmath>

#include <Eigen/Core>

#include "magmaan/inference/inference.hpp"
#include "magmaan/robust/frontier/fmg.hpp"

TEST_CASE("FMG standard chi-square ignores the UGamma spectrum") {
  Eigen::VectorXd eig(3);
  eig << 10.0, 0.5, 2.0;

  const auto r = magmaan::robust::frontier::fmg_test(
      7.0, 4, eig,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::StandardChiSquare});

  CHECK(r.p_value == doctest::Approx(
                         magmaan::inference::chi2_pvalue(7.0, 4)));
  CHECK(r.chi2_equiv == doctest::Approx(7.0).epsilon(1e-10));
}

TEST_CASE("FMG selects top df eigenvalues and truncates negative noise") {
  Eigen::VectorXd eig(4);
  eig << -1.0, 2.0, 0.25, -0.5;

  const auto r = magmaan::robust::frontier::fmg_test(
      3.0, 3, eig,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::All});

  REQUIRE(r.lambdas_raw.size() == 3);
  CHECK(r.lambdas_raw(0) == doctest::Approx(2.0));
  CHECK(r.lambdas_raw(1) == doctest::Approx(0.25));
  CHECK(r.lambdas_raw(2) == doctest::Approx(-0.5));
  CHECK(r.lambdas(2) == doctest::Approx(0.0));
  CHECK(r.n_truncated == 1);
}

TEST_CASE("FMG exact and classical approximations collapse for equal lambdas") {
  const Eigen::VectorXd eig = Eigen::VectorXd::Ones(4);
  const double chi2 = 7.0;
  const double p = magmaan::inference::chi2_pvalue(chi2, 4);

  for (const auto method : {
           magmaan::robust::frontier::FmgMethod::All,
           magmaan::robust::frontier::FmgMethod::SatorraBentler,
           magmaan::robust::frontier::FmgMethod::ScaledShifted,
           magmaan::robust::frontier::FmgMethod::ScaledF}) {
    const auto r = magmaan::robust::frontier::fmg_test(
        chi2, 4, eig, magmaan::robust::frontier::FmgOptions{.method = method});
    CHECK(r.p_value == doctest::Approx(p).epsilon(2e-5));
    CHECK(r.chi2_equiv == doctest::Approx(chi2).epsilon(2e-5));
  }
}

TEST_CASE("FMG PALL shrinks every eigenvalue halfway to the mean") {
  Eigen::VectorXd eig(3);
  eig << 3.0, 2.0, 1.0;

  const auto r = magmaan::robust::frontier::fmg_test(
      4.0, 3, eig,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::PenalizedAll});

  REQUIRE(r.lambdas_reference.size() == 3);
  CHECK(r.lambdas_reference(0) == doctest::Approx(2.5));
  CHECK(r.lambdas_reference(1) == doctest::Approx(2.0));
  CHECK(r.lambdas_reference(2) == doctest::Approx(1.5));
}

TEST_CASE("FMG pEBA uses lavaan/semTests column-block averaging") {
  Eigen::VectorXd eig(5);
  eig << 8.0, 6.0, 4.0, 2.0, 1.0;

  const auto r = magmaan::robust::frontier::fmg_test(
      6.0, 5, eig,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::Peba,
          .param = 2.0});

  REQUIRE(r.lambdas_reference.size() == 5);
  CHECK(r.lambdas_reference(0) == doctest::Approx(5.1));
  CHECK(r.lambdas_reference(1) == doctest::Approx(5.1));
  CHECK(r.lambdas_reference(2) == doctest::Approx(5.1));
  CHECK(r.lambdas_reference(3) == doctest::Approx(2.85));
  CHECK(r.lambdas_reference(4) == doctest::Approx(2.85));
}

TEST_CASE("FMG EBA averages eigenvalue blocks without penalization") {
  Eigen::VectorXd eig(5);
  eig << 8.0, 6.0, 4.0, 2.0, 1.0;

  const auto r = magmaan::robust::frontier::fmg_test(
      6.0, 5, eig,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::Eba,
          .param = 2.0});

  // Same blocks as pEBA (means 6 and 1.5), but used directly -- no shrink to
  // the global mean (4.2). Contrast with the pEBA case above (5.1 / 2.85).
  REQUIRE(r.lambdas_reference.size() == 5);
  CHECK(r.lambdas_reference(0) == doctest::Approx(6.0));
  CHECK(r.lambdas_reference(1) == doctest::Approx(6.0));
  CHECK(r.lambdas_reference(2) == doctest::Approx(6.0));
  CHECK(r.lambdas_reference(3) == doctest::Approx(1.5));
  CHECK(r.lambdas_reference(4) == doctest::Approx(1.5));
}

TEST_CASE("FMG pOLS shrinks the spectrum trend by gamma") {
  Eigen::VectorXd eig(4);
  eig << 8.0, 6.0, 4.0, 2.0;

  const auto r = magmaan::robust::frontier::fmg_test(
      6.0, 4, eig,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::Pols,
          .param = 2.0});

  REQUIRE(r.lambdas_reference.size() == 4);
  CHECK(r.lambdas_reference(0) == doctest::Approx(6.5));
  CHECK(r.lambdas_reference(1) == doctest::Approx(5.5));
  CHECK(r.lambdas_reference(2) == doctest::Approx(4.5));
  CHECK(r.lambdas_reference(3) == doctest::Approx(3.5));
}

TEST_CASE("FMG reduced-matrix entry point eigensolves before p-value") {
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(3, 3);
  M(0, 0) = 3.0;
  M(1, 1) = 2.0;

  auto r_or = magmaan::robust::frontier::fmg_test_from_reduced_matrix(
      4.0, 2, M,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::PenalizedAll});
  REQUIRE(r_or.has_value());
  CHECK(r_or->lambdas(0) == doctest::Approx(3.0));
  CHECK(r_or->lambdas(1) == doctest::Approx(2.0));
}

TEST_CASE("FMG nested wrapper rejects single-model approximations") {
  magmaan::robust::SatorraDiffResult sd;
  sd.eigenvalues = Eigen::VectorXd::Ones(2);

  auto r_or = magmaan::robust::frontier::lr_test_fmg(
      4.0, sd,
      magmaan::robust::frontier::FmgOptions{
          .method = magmaan::robust::frontier::FmgMethod::SatorraBentler});
  CHECK_FALSE(r_or.has_value());
}
