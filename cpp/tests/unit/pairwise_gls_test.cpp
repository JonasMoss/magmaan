#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

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

TEST_CASE("fit_gls_pairwise: complete-data fit matches fit_gls") {
  std::mt19937 rng(20260602);
  auto model = build_cfa();
  const Eigen::Index p = 3;

  // Pick a "true" Σ via the model — start at a θ that makes a clean SEM.
  const Eigen::VectorXd mu = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd Sigma(p, p);
  // f loadings 1.0, 0.8, 0.9; var(f) = 1.5; residual variances 0.7, 0.6, 0.5.
  const double l1 = 1.0, l2 = 0.8, l3 = 0.9;
  const double psi = 1.5;
  Eigen::Vector3d theta_res(0.7, 0.6, 0.5);
  Eigen::Vector3d lam(l1, l2, l3);
  Sigma = psi * (lam * lam.transpose());
  Sigma.diagonal() += theta_res;

  Eigen::MatrixXd X = sample_mvn(rng, mu, Sigma, 400);

  // Complete-data path.
  magmaan::data::RawData raw;
  raw.X.push_back(X);
  auto pw = magmaan::data::pairwise_sample_stats(raw);
  REQUIRE(pw.has_value());

  // fit_gls with samp.S = pw.S (Σ-only weight, the literature convention).
  magmaan::data::SampleStats samp;
  samp.S = pw->S;
  samp.mean = pw->mean;
  samp.n_obs = pw->n_obs;
  auto x0 = magmaan::estimate::simple_start_values(model.pt, model.rep, samp, {});
  REQUIRE(x0.has_value());
  auto fit_a = magmaan::estimate::fit_gls(model.pt, model.rep, samp, *x0);
  REQUIRE(fit_a.has_value());

  // fit_gls_pairwise with the Γ_NT^pw weight.
  auto fit_b = magmaan::estimate::fit_gls_pairwise(model.pt, model.rep, raw,
                                                    *pw, *x0);
  REQUIRE(fit_b.has_value());

  // On complete data, the two W matrices coincide, so the fits agree.
  REQUIRE(fit_a->theta.size() == fit_b->theta.size());
  CHECK((fit_a->theta - fit_b->theta).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("fit_gls_pairwise: missing-data sanity") {
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

  auto fit = magmaan::estimate::fit_gls_pairwise(model.pt, model.rep, raw,
                                                 *pw, *x0);
  REQUIRE(fit.has_value());
  CHECK(fit->theta.allFinite());
  CHECK(fit->fmin >= 0.0);
}
