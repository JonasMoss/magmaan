#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "magmaan/estimate/frontier/communality.hpp"

using magmaan::estimate::frontier::CommunalityMethod;
using magmaan::estimate::frontier::estimate_h_communalities;
using magmaan::estimate::frontier::estimate_h2_communalities_constrained;
using magmaan::estimate::frontier::estimate_h2_communalities_constrained_jacobian;
using magmaan::estimate::frontier::estimate_h2_communalities;
using magmaan::estimate::frontier::estimate_h2_communalities_directional;
using magmaan::estimate::frontier::estimate_h2_communalities_jacobian;

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
       {CommunalityMethod::TriadMean,
        CommunalityMethod::TriadPooled,
        CommunalityMethod::TriadLeastSquares,
        CommunalityMethod::ExtendedTriadLeastSquares,
        CommunalityMethod::TriadWls,
        CommunalityMethod::TriadWlsJoint}) {
    auto h2 = estimate_h2_communalities(R, blocks, method);
    REQUIRE(h2.has_value());
    CHECK((*h2 - expected).cwiseAbs().maxCoeff() < 1e-10);
  }
}

TEST_CASE("communality directional derivative matches central finite differences") {
  Eigen::MatrixXd S = one_factor_corr({0.55, 0.70, 0.85, 0.62, 0.78});
  S(0, 0) = 1.20;
  S(1, 1) = 0.90;
  S(3, 3) = 1.35;
  Eigen::MatrixXd dS = Eigen::MatrixXd::Zero(5, 5);
  dS(0, 0) = 0.30;
  dS(1, 1) = -0.20;
  dS(0, 2) = 0.11;
  dS(2, 0) = 0.11;
  dS(1, 4) = -0.07;
  dS(4, 1) = -0.07;
  const std::vector<std::int32_t> blocks(5, 0);
  constexpr double eps = 2e-6;

  for (CommunalityMethod method :
       {CommunalityMethod::TriadMean,
        CommunalityMethod::TriadPooled,
        CommunalityMethod::TriadLeastSquares,
        CommunalityMethod::ExtendedTriadLeastSquares,
        CommunalityMethod::TriadWls,
        CommunalityMethod::TriadWlsJoint}) {
    INFO("method ordinal: " << static_cast<int>(method));
    auto analytic = estimate_h2_communalities_directional(S, dS, blocks, method);
    auto plus = estimate_h2_communalities(S + eps * dS, blocks, method);
    auto minus = estimate_h2_communalities(S - eps * dS, blocks, method);
    REQUIRE(analytic.has_value());
    REQUIRE(plus.has_value());
    REQUIRE(minus.has_value());
    const Eigen::VectorXd fd = (*plus - *minus) / (2.0 * eps);
    CHECK((*analytic - fd).cwiseAbs().maxCoeff() < 5e-5);
  }
}

TEST_CASE("communality batched Jacobian matches directional derivatives") {
  const std::vector<std::int32_t> blocks = {0, 0, 0, 1, 1, 1, 1, 1};
  const std::vector<double> lambda = {0.58, 0.74, 0.91, 0.52,
                                      0.67, 0.81, 0.73, 0.88};
  const std::vector<double> sd = {1.20, 0.90, 1.45, 0.85,
                                  1.65, 1.10, 1.35, 0.95};
  const double rho = 0.36;

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
  Eigen::MatrixXd S = R;
  for (Eigen::Index i = 0; i < S.rows(); ++i) {
    for (Eigen::Index j = 0; j < S.cols(); ++j)
      S(i, j) *= sd[static_cast<std::size_t>(i)] *
                 sd[static_cast<std::size_t>(j)];
  }

  const Eigen::Index p = S.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  for (CommunalityMethod method :
       {CommunalityMethod::TriadLeastSquares,
        CommunalityMethod::ExtendedTriadLeastSquares,
        CommunalityMethod::TriadWls}) {
    INFO("method ordinal: " << static_cast<int>(method));
    auto J = estimate_h2_communalities_jacobian(S, blocks, method);
    REQUIRE(J.has_value());
    REQUIRE(J->rows() == p);
    REQUIRE(J->cols() == pstar);

    Eigen::Index col = 0;
    for (Eigen::Index c = 0; c < p; ++c) {
      for (Eigen::Index r = c; r < p; ++r) {
        Eigen::MatrixXd dS = Eigen::MatrixXd::Zero(p, p);
        dS(r, c) = 1.0;
        if (r != c) dS(c, r) = 1.0;
        auto directional =
            estimate_h2_communalities_directional(S, dS, blocks, method);
        REQUIRE(directional.has_value());
        CHECK((J->col(col) - *directional).cwiseAbs().maxCoeff() < 1e-8);
        ++col;
      }
    }
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
        R, blocks, CommunalityMethod::ExtendedTriadLeastSquares);
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

  auto rs = estimate_h2_communalities(R, blocks, CommunalityMethod::TriadPooled);
  auto gb = estimate_h2_communalities(R, blocks, CommunalityMethod::TriadWls);
  auto gf = estimate_h2_communalities(R, blocks, CommunalityMethod::TriadWlsJoint);
  REQUIRE(rs.has_value());
  REQUIRE(gb.has_value());
  REQUIRE(gf.has_value());

  CHECK((*rs)(0) == doctest::Approx(0.30 * 0.45 / 0.50));
  CHECK((*rs - *gb).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((*rs - *gf).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("communality constrained system preserves the current GMM solution "
          "with no rows") {
  const std::vector<double> lambda = {0.5, 0.7, 0.9, 0.6, 0.8, 0.55};
  const std::vector<std::int32_t> blocks = {0, 0, 0, 1, 1, 1};
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(
      static_cast<Eigen::Index>(lambda.size()),
      static_cast<Eigen::Index>(lambda.size()));
  for (Eigen::Index i = 0; i < R.rows(); ++i) {
    for (Eigen::Index j = i + 1; j < R.cols(); ++j) {
      const double phi = blocks[static_cast<std::size_t>(i)] ==
                                 blocks[static_cast<std::size_t>(j)]
                             ? 1.0
                             : 0.35;
      R(i, j) = lambda[static_cast<std::size_t>(i)] *
                lambda[static_cast<std::size_t>(j)] * phi;
      R(j, i) = R(i, j);
    }
  }

  const Eigen::MatrixXd R0(0, R.rows());
  const Eigen::VectorXd r0(0);
  auto base = estimate_h2_communalities(R, blocks, CommunalityMethod::TriadWls);
  auto con = estimate_h2_communalities_constrained(
      R, blocks, CommunalityMethod::TriadWls, R0, r0);
  REQUIRE(base.has_value());
  REQUIRE(con.has_value());
  CHECK((*base - *con).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("communality constrained system imposes linear h2 rows") {
  Eigen::MatrixXd R(4, 4);
  R << 1.0, 0.30, 0.45, 0.28,
       0.30, 1.0, 0.50, 0.35,
       0.45, 0.50, 1.0, 0.42,
       0.28, 0.35, 0.42, 1.0;
  const std::vector<std::int32_t> blocks = {0, 0, 0, 0};
  Eigen::MatrixXd C(1, 4);
  C << 1.0, -1.0, 0.0, 0.0;
  Eigen::VectorXd d(1);
  d << 0.0;

  auto h2 = estimate_h2_communalities_constrained(
      R, blocks, CommunalityMethod::TriadWls, C, d);
  REQUIRE(h2.has_value());
  CHECK((*h2)(0) == doctest::Approx((*h2)(1)).epsilon(1e-10));
}

TEST_CASE("communality constrained Jacobian matches finite differences") {
  const std::vector<std::int32_t> blocks = {0, 0, 0, 0, 0, 1, 1, 1};
  const std::vector<double> lambda = {0.55, 0.70, 0.85, 0.62,
                                      0.78, 0.58, 0.76, 0.68};
  const std::vector<double> sd = {1.20, 0.90, 1.35, 1.10,
                                  1.55, 0.85, 1.45, 1.05};
  const double rho = 0.32;

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
  Eigen::MatrixXd S = R;
  for (Eigen::Index i = 0; i < S.rows(); ++i) {
    for (Eigen::Index j = 0; j < S.cols(); ++j)
      S(i, j) *= sd[static_cast<std::size_t>(i)] *
                 sd[static_cast<std::size_t>(j)];
  }
  const auto constraints = [](const Eigen::MatrixXd& X) {
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(1, X.rows());
    C(0, 0) = -X(0, 0);
    C(0, 1) = X(1, 1);
    Eigen::VectorXd d(1);
    d(0) = -X(0, 0) + X(1, 1);
    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{C, d};
  };
  auto [C, d] = constraints(S);

  const Eigen::Index p = S.rows();
  constexpr double eps = 2e-6;
  for (CommunalityMethod method :
       {CommunalityMethod::TriadLeastSquares,
        CommunalityMethod::ExtendedTriadLeastSquares,
        CommunalityMethod::TriadWls}) {
    INFO("method ordinal: " << static_cast<int>(method));
    auto J = estimate_h2_communalities_constrained_jacobian(
        S, blocks, method, C, d);
    REQUIRE(J.has_value());

    Eigen::Index col = 0;
    for (Eigen::Index c = 0; c < p; ++c) {
      for (Eigen::Index r = c; r < p; ++r) {
        Eigen::MatrixXd dS = Eigen::MatrixXd::Zero(p, p);
        dS(r, c) = 1.0;
        if (r != c) dS(c, r) = 1.0;
        auto [Cp, dp] = constraints(S + eps * dS);
        auto [Cm, dm] = constraints(S - eps * dS);
        auto hp = estimate_h2_communalities_constrained(
            S + eps * dS, blocks, method, Cp, dp);
        auto hm = estimate_h2_communalities_constrained(
            S - eps * dS, blocks, method, Cm, dm);
        REQUIRE(hp.has_value());
        REQUIRE(hm.has_value());
        const Eigen::VectorXd fd = (*hp - *hm) / (2.0 * eps);
        CHECK((J->col(col) - fd).cwiseAbs().maxCoeff() < 5e-5);
        ++col;
      }
    }
  }
}

TEST_CASE("anchor communality is the anchor joint LS system with no rows") {
  const std::vector<double> lambda = {0.5, 0.7, 0.9, 0.6, 0.8, 0.55};
  const std::vector<std::int32_t> blocks = {0, 0, 0, 1, 1, 1};
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(
      static_cast<Eigen::Index>(lambda.size()),
      static_cast<Eigen::Index>(lambda.size()));
  for (Eigen::Index i = 0; i < R.rows(); ++i) {
    for (Eigen::Index j = i + 1; j < R.cols(); ++j) {
      const double phi = blocks[static_cast<std::size_t>(i)] ==
                                 blocks[static_cast<std::size_t>(j)]
                             ? 1.0
                             : 0.4;
      R(i, j) = lambda[static_cast<std::size_t>(i)] *
                lambda[static_cast<std::size_t>(j)] * phi;
      R(j, i) = R(i, j);
    }
  }
  const Eigen::MatrixXd R0(0, R.rows());
  const Eigen::VectorXd r0(0);
  auto item = estimate_h2_communalities(
      R, blocks, CommunalityMethod::ExtendedTriadLeastSquares);
  auto joint = estimate_h2_communalities_constrained(
      R, blocks, CommunalityMethod::ExtendedTriadLeastSquares, R0, r0);
  REQUIRE(item.has_value());
  REQUIRE(joint.has_value());
  CHECK((*item - *joint).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("communality H result replaces only the covariance diagonal") {
  Eigen::MatrixXd S(3, 3);
  S << 4.0, 0.60, 1.20,
       0.60, 9.0, 1.50,
       1.20, 1.50, 16.0;
  const std::vector<std::int32_t> blocks = {0, 0, 0};

  auto out = estimate_h_communalities(S, blocks, CommunalityMethod::TriadPooled);
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
