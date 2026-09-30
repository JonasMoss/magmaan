#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/data/pairwise_cov.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

struct BuiltModel {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep      rep;
};

BuiltModel build_cfa() {
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return BuiltModel{std::move(*pt), std::move(*rep)};
}

Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  return A * A.transpose() + Eigen::MatrixXd::Identity(p, p) * static_cast<double>(p);
}

Eigen::MatrixXd sample_mvn(std::mt19937& rng, const Eigen::VectorXd& mu,
                           const Eigen::MatrixXd& Sigma, int n) {
  const Eigen::Index p = mu.size();
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd X(n, p);
  Eigen::VectorXd z(p);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index k = 0; k < p; ++k) z(k) = nd(rng);
    X.row(i) = (mu + L * z).transpose();
  }
  return X;
}

}  // namespace

TEST_CASE("pairwise_sample_stats: numeric overflow returns a block error") {
  Eigen::MatrixXd X(3, 1);
  SUBCASE("marginal means overflow") { X.setConstant(1e308); }
  SUBCASE("covariance products overflow") { X << -1e200, 0.0, 1e200; }
  SUBCASE("covariance accumulation overflows") { X << -1e154, 0.0, 1e154; }
  magmaan::data::RawData raw;
  raw.X = {Eigen::MatrixXd::Ones(3, 1), X};
  auto result = magmaan::data::pairwise_sample_stats(raw);
  CHECK_FALSE(result.has_value());
  if (!result) {
    CHECK(result.error().kind == magmaan::PostError::Kind::NumericIssue);
    CHECK(result.error().detail.find("block 1") != std::string::npos);
  }
}

TEST_CASE("pairwise_sample_stats: numeric validation preserves missingness and singular moments") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(4, 2);
  X << 1.0, 2.0, 1.0, 2.0, nan, 2.0, 1.0, nan;
  magmaan::data::RawData raw;
  raw.X = {X};
  auto inferred = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(inferred.has_value());
  if (!inferred) return;
  CHECK(inferred->S[0].isZero());
  CHECK(inferred->mean[0](0) == 1.0);
  CHECK(inferred->mean[0](1) == 2.0);

  raw.mask = {X.array().isFinite().cast<std::uint8_t>()};
  raw.X[0](2, 0) = std::numeric_limits<double>::infinity();
  raw.X[0](3, 1) = 1e308;
  auto masked = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(masked.has_value());
  if (!masked) return;
  CHECK((masked->S[0] - inferred->S[0]).norm() == 0.0);
  CHECK((masked->mean[0] - inferred->mean[0]).norm() == 0.0);
  CHECK((masked->n_pair[0] - inferred->n_pair[0]).norm() == 0);

  raw.mask.clear();
  raw.X = {Eigen::MatrixXd(3, 1)};
  raw.X[0] << -1e100, 0.0, 1e100;
  auto large = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(large.has_value());
  if (large) CHECK(large->S[0](0, 0) == doctest::Approx(2e200 / 3.0));
}

TEST_CASE("gamma_nt_pairwise: complete-data degeneracy matches gamma_nt") {
  std::mt19937 rng(20260601);
  const Eigen::Index p = 4;
  const Eigen::MatrixXd Sigma = random_pd(rng, p);
  const Eigen::VectorXd mu = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd X = sample_mvn(rng, mu, Sigma, 200);

  magmaan::data::RawData raw;
  raw.X.push_back(X);
  // no mask: complete data

  auto pw_or = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw_or.has_value());
  auto gnt_pw_or = magmaan::data::gamma_nt_pairwise(raw, *pw_or);
  REQUIRE(gnt_pw_or.has_value());
  auto gnt_or = magmaan::data::gamma_nt(pw_or->S[0]);
  REQUIRE(gnt_or.has_value());

  CHECK((gnt_pw_or->at(0) - *gnt_or).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("gamma_nt_pairwise: inferred missingness matches explicit masks and overlaps") {
  magmaan::data::RawData raw;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X(6, 2);
  X << -1.0, 2.0, 1.0, 4.0, nan, 6.0, nan, 8.0, 3.0, nan, 5.0, nan;
  raw.X = {Eigen::MatrixXd::Ones(6, 1), X};
  auto pw = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw.has_value());
  if (!pw) return;
  auto inferred = magmaan::data::gamma_nt_pairwise(raw, *pw);
  REQUIRE(inferred.has_value());
  if (!inferred) return;

  using Mask = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
  raw.mask = {Mask::Ones(6, 1), X.array().isFinite().cast<std::uint8_t>()};
  auto masked = magmaan::data::gamma_nt_pairwise(raw, *pw);
  REQUIRE(masked.has_value());
  if (!masked) return;
  CHECK(((*inferred)[0] - (*masked)[0]).norm() == 0.0);
  CHECK(((*inferred)[1] - (*masked)[1]).norm() < 1e-12);

  auto nt = magmaan::data::gamma_nt(pw->S[1]);
  REQUIRE(nt.has_value());
  if (!nt) return;
  const int first[] = {0, 1, 1};
  const int second[] = {0, 0, 1};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      int overlap = 0;
      for (Eigen::Index r = 0; r < X.rows(); ++r) {
        overlap += std::isfinite(X(r, first[a])) && std::isfinite(X(r, second[a])) &&
                   std::isfinite(X(r, first[b])) && std::isfinite(X(r, second[b]));
      }
      const double pi_a = pw->pi_hat[1](first[a], second[a]);
      const double pi_b = pw->pi_hat[1](first[b], second[b]);
      const double expected = (*nt)(a, b) * (static_cast<double>(overlap) / 6.0) /
                              (pi_a * pi_b);
      CHECK((*inferred)[1](a, b) == doctest::Approx(expected).epsilon(1e-12));
    }
  }
}

TEST_CASE("gamma_nt_pairwise: malformed summaries return NumericIssue") {
  magmaan::data::RawData raw;
  raw.X = {Eigen::MatrixXd::Ones(4, 2)};
  auto stats = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(stats.has_value());
  if (!stats) return;
  auto pw = *stats;
  using Mask = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

  SUBCASE("empty inputs") { raw.X.clear(); pw.S.clear(); pw.pi_hat.clear(); }
  SUBCASE("missing covariance blocks") { pw.S.clear(); }
  SUBCASE("missing availability blocks") { pw.pi_hat.clear(); }
  SUBCASE("extra availability blocks") { pw.pi_hat.push_back(pw.pi_hat[0]); }
  SUBCASE("raw column mismatch") { raw.X[0].resize(4, 1); }
  SUBCASE("too few observations") { raw.X[0].resize(1, 2); }
  SUBCASE("zero columns") {
    raw.X[0].resize(4, 0); pw.S[0].resize(0, 0); pw.pi_hat[0].resize(0, 0);
  }
  SUBCASE("nonsquare covariance") { pw.S[0].resize(2, 1); }
  SUBCASE("short availability rows") { pw.pi_hat[0].resize(1, 2); }
  SUBCASE("short availability columns") { pw.pi_hat[0].resize(2, 1); }
  SUBCASE("mask block mismatch") { raw.mask = {Mask::Ones(4, 2), Mask::Ones(4, 2)}; }
  SUBCASE("mask row mismatch") { raw.mask = {Mask::Ones(3, 2)}; }
  SUBCASE("mask column mismatch") { raw.mask = {Mask::Ones(4, 1)}; }
  SUBCASE("non-finite covariance") {
    pw.S[0](0, 0) = std::numeric_limits<double>::infinity();
  }
  SUBCASE("invalid availability") {
    for (double bad : {0.0, -0.1, 1.1, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
      CAPTURE(bad);
      pw.pi_hat[0](1, 0) = bad;
      auto result = magmaan::data::gamma_nt_pairwise(raw, pw);
      CHECK_FALSE(result.has_value());
      if (!result) CHECK(result.error().kind == magmaan::PostError::Kind::NumericIssue);
    }
    return;
  }
  auto result = magmaan::data::gamma_nt_pairwise(raw, pw);
  CHECK_FALSE(result.has_value());
  if (!result) CHECK(result.error().kind == magmaan::PostError::Kind::NumericIssue);
}

TEST_CASE("pairwise MCAR moments compose with ML, GLS, ULS and fixed-weight WLS") {
  std::mt19937 rng(20260603);
  auto model = build_cfa();
  const Eigen::Index p = 3;

  Eigen::Vector3d lam(1.0, 0.8, 0.9);
  Eigen::Matrix3d Sigma = 1.5 * (lam * lam.transpose());
  Sigma.diagonal() += Eigen::Vector3d(0.7, 0.6, 0.5);
  Eigen::Vector3d mu = Eigen::Vector3d::Zero();

  Eigen::MatrixXd X = sample_mvn(rng, mu, Sigma, 400);

  // Drop ~12% per column completely-at-random; never leave a row fully blank.
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(X.rows(), p);
  M.setOnes();
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    int kept = 0;
    for (Eigen::Index c = 0; c < p; ++c) {
      if (u(rng) < 0.12) M(r, c) = 0;
      else ++kept;
    }
    if (kept == 0) M(r, 0) = 1;
    for (Eigen::Index c = 0; c < p; ++c) {
      if (M(r, c) == 0) X(r, c) = std::numeric_limits<double>::quiet_NaN();
    }
  }
  magmaan::data::RawData raw;
  raw.X.push_back(X);
  raw.mask.push_back(M);

  auto pw = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw.has_value());

  // Γ_NT^pw must be symmetric and PD.
  auto gnt_pw = magmaan::data::gamma_nt_pairwise(raw, *pw);
  REQUIRE(gnt_pw.has_value());
  const Eigen::MatrixXd& G = gnt_pw->at(0);
  CHECK((G - G.transpose()).cwiseAbs().maxCoeff() < 1e-10);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(G);
  REQUIRE(eig.info() == Eigen::Success);
  CHECK(eig.eigenvalues().minCoeff() > 0.0);

  magmaan::data::SampleStats samp;
  samp.S = pw->S;
  samp.mean = pw->mean;
  samp.n_obs = pw->n_obs;
  auto x0 = magmaan::estimate::simple_start_values(model.pt, model.rep, samp, {});
  REQUIRE(x0.has_value());

  Eigen::MatrixXd W = G.llt().solve(
      Eigen::MatrixXd::Identity(G.rows(), G.cols()));
  auto weight = magmaan::estimate::gmm::dense_weight(
      {W}, magmaan::FitError::Kind::NumericIssue, "pairwise MCAR metric");
  REQUIRE(weight.has_value());
  for (const auto& fit : {
      magmaan::estimate::fit_ml(model.pt, model.rep, samp, *x0),
      magmaan::estimate::fit_gls(model.pt, model.rep, samp, *x0),
      magmaan::estimate::fit_gmm(model.pt, model.rep, samp, *x0),
      magmaan::estimate::fit_gmm(model.pt, model.rep, samp, *x0, *weight)}) {
    REQUIRE(fit.has_value());
    CHECK(fit->theta.allFinite());
    CHECK(fit->fmin >= 0.0);
  }
}
