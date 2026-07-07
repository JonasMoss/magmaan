#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "magmaan/estimate/frontier/communality.hpp"

using magmaan::estimate::frontier::CommunalityMethod;
using magmaan::estimate::frontier::estimate_h_communalities;
using magmaan::estimate::frontier::estimate_h2_communalities;

namespace {

Eigen::MatrixXd one_factor_corr(const std::vector<double>& lambda) {
  const Eigen::Index p = static_cast<Eigen::Index>(lambda.size());
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(p, p);
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = i + 1; j < p; ++j) {
      R(i, j) = lambda[static_cast<std::size_t>(i)] *
                lambda[static_cast<std::size_t>(j)];
      R(j, i) = R(i, j);
    }
  }
  return R;
}

}  // namespace

TEST_CASE("communality rules recover exact one-factor h2") {
  const std::vector<double> lambda = {0.5, 0.7, 0.9, 0.6, 0.8};
  const Eigen::MatrixXd R = one_factor_corr(lambda);
  const std::vector<std::int32_t> blocks(lambda.size(), 0);

  Eigen::VectorXd expected(static_cast<Eigen::Index>(lambda.size()));
  for (Eigen::Index i = 0; i < expected.size(); ++i) {
    expected(i) = lambda[static_cast<std::size_t>(i)] *
                  lambda[static_cast<std::size_t>(i)];
  }

  for (CommunalityMethod method :
       {CommunalityMethod::AverageRatio,
        CommunalityMethod::RatioOfSums,
        CommunalityMethod::TriadLeastSquares,
        CommunalityMethod::AnchorTriadLeastSquares,
        CommunalityMethod::GmmBlock,
        CommunalityMethod::GmmFull}) {
    auto h2 = estimate_h2_communalities(R, blocks, method);
    REQUIRE(h2.has_value());
    CHECK((*h2 - expected).cwiseAbs().maxCoeff() < 1e-10);
  }
}

TEST_CASE("anchor communality recovers exact two-factor h2") {
  const std::vector<double> lambda = {0.5, 0.7, 0.9, 0.6, 0.8, 0.55};
  const std::vector<std::int32_t> blocks = {0, 0, 0, 1, 1, 1};

  for (const double rho : {0.0, 0.4}) {
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(
        static_cast<Eigen::Index>(lambda.size()),
        static_cast<Eigen::Index>(lambda.size()));
    for (Eigen::Index i = 0; i < R.rows(); ++i) {
      for (Eigen::Index j = i + 1; j < R.cols(); ++j) {
        const double phi = blocks[static_cast<std::size_t>(i)] ==
                                   blocks[static_cast<std::size_t>(j)]
                               ? 1.0
                               : rho;
        R(i, j) = lambda[static_cast<std::size_t>(i)] *
                  lambda[static_cast<std::size_t>(j)] * phi;
        R(j, i) = R(i, j);
      }
    }

    Eigen::VectorXd expected(static_cast<Eigen::Index>(lambda.size()));
    for (Eigen::Index i = 0; i < expected.size(); ++i) {
      expected(i) = lambda[static_cast<std::size_t>(i)] *
                    lambda[static_cast<std::size_t>(i)];
    }

    auto h2 = estimate_h2_communalities(
        R, blocks, CommunalityMethod::AnchorTriadLeastSquares);
    REQUIRE(h2.has_value());
    CHECK((*h2 - expected).cwiseAbs().maxCoeff() < 1e-10);
  }
}

TEST_CASE("communality three-indicator GMM collapses to the just-identified "
          "triad solution") {
  Eigen::MatrixXd R(3, 3);
  R << 1.0, 0.30, 0.45,
       0.30, 1.0, 0.50,
       0.45, 0.50, 1.0;
  const std::vector<std::int32_t> blocks = {0, 0, 0};

  auto rs = estimate_h2_communalities(R, blocks, CommunalityMethod::RatioOfSums);
  auto gb = estimate_h2_communalities(R, blocks, CommunalityMethod::GmmBlock);
  auto gf = estimate_h2_communalities(R, blocks, CommunalityMethod::GmmFull);
  REQUIRE(rs.has_value());
  REQUIRE(gb.has_value());
  REQUIRE(gf.has_value());

  CHECK((*rs)(0) == doctest::Approx(0.30 * 0.45 / 0.50));
  CHECK((*rs - *gb).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((*rs - *gf).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("communality H result replaces only the covariance diagonal") {
  Eigen::MatrixXd S(3, 3);
  S << 4.0, 0.60, 1.20,
       0.60, 9.0, 1.50,
       1.20, 1.50, 16.0;
  const std::vector<std::int32_t> blocks = {0, 0, 0};

  auto out = estimate_h_communalities(S, blocks, CommunalityMethod::RatioOfSums);
  REQUIRE(out.has_value());

  CHECK(out->H(0, 1) == doctest::Approx(S(0, 1)));
  CHECK(out->H(0, 2) == doctest::Approx(S(0, 2)));
  CHECK(out->H(1, 2) == doctest::Approx(S(1, 2)));
  for (Eigen::Index i = 0; i < 3; ++i) {
    CHECK(out->h_diag(i) == doctest::Approx(S(i, i) * out->h2(i)));
    CHECK(out->H(i, i) == doctest::Approx(out->h_diag(i)));
  }
}

TEST_CASE("communality estimator rejects under-sized factor blocks") {
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(4, 4);
  const std::vector<std::int32_t> blocks = {0, 0, 1, 1};

  auto h2 = estimate_h2_communalities(
      R, blocks, CommunalityMethod::TriadLeastSquares);
  CHECK_FALSE(h2.has_value());
}
