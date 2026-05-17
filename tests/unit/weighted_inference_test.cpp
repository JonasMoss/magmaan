#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/optim/lbfgsb_optimizer.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

struct OneFactorFixture {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
  magmaan::data::RawData raw;
  magmaan::data::SampleStats samp;
};

Eigen::MatrixXd mvn_sample(Eigen::Index n,
                           const Eigen::VectorXd& mu,
                           const Eigen::MatrixXd& Sigma) {
  std::mt19937 rng(42);
  std::normal_distribution<double> z(0.0, 1.0);
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  Eigen::MatrixXd X(n, mu.size());
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::VectorXd zi(mu.size());
    for (Eigen::Index j = 0; j < zi.size(); ++j) zi(j) = z(rng);
    X.row(i) = (mu + L * zi).transpose();
  }
  return X;
}

OneFactorFixture one_factor_fixture() {
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());

  Eigen::Vector4d lambda;
  lambda << 0.8, 0.7, 0.9, 0.75;
  Eigen::Matrix4d Sigma = lambda * lambda.transpose();
  Sigma.diagonal().array() += 0.45;
  Eigen::Vector4d mu;
  mu << 0.0, 0.0, 0.0, 0.0;

  magmaan::data::RawData raw;
  raw.X.push_back(mvn_sample(420, mu, Sigma));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  return OneFactorFixture{std::move(*pt), std::move(*rep), std::move(raw),
                          std::move(*samp)};
}

std::vector<Eigen::MatrixXd> wls_weights_from_sample(
    const magmaan::data::SampleStats& samp) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(samp.S.size());
  for (const auto& S : samp.S) {
    auto G = magmaan::data::gamma_nt(S);
    REQUIRE(G.has_value());
    Eigen::LDLT<Eigen::MatrixXd> ldlt(*G);
    REQUIRE(ldlt.info() == Eigen::Success);
    REQUIRE(ldlt.isPositive());
    out.push_back(ldlt.solve(Eigen::MatrixXd::Identity(G->rows(), G->cols())));
  }
  return out;
}

double total_n(const magmaan::data::SampleStats& samp) {
  double out = 0.0;
  for (auto n : samp.n_obs) out += static_cast<double>(n);
  return out;
}

// The GLS moment weight (normal-theory weight built from S).
magmaan::estimate::gmm::Weight gls_weight(const magmaan::spec::LatentStructure& pt,
                                const magmaan::model::MatrixRep& rep,
                                const magmaan::data::SampleStats& samp,
                                const Eigen::VectorXd& theta) {
  auto ev = magmaan::model::ModelEvaluator::build(pt, rep);
  REQUIRE(ev.has_value());
  auto w = magmaan::estimate::gmm::normal_theory_weight(*ev, samp, theta);
  REQUIRE(w.has_value());
  return *w;
}

}  // namespace

TEST_CASE("robust_weighted_moments computes sandwich and U-Gamma for one block") {
  magmaan::estimate::WeightedMomentBlock block;
  block.jacobian.resize(2, 1);
  block.jacobian << 1.0, 0.0;
  block.weight = Eigen::MatrixXd::Identity(2, 2);
  block.gamma = Eigen::MatrixXd::Identity(2, 2);
  block.n_obs = 100;

  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(1, 1);
  auto out = magmaan::estimate::robust_weighted_moments({block}, K, 0.5);
  REQUIRE(out.has_value());
  CHECK(out->df == 1);
  CHECK(out->vcov.rows() == 1);
  CHECK(out->vcov.cols() == 1);
  CHECK(out->vcov(0, 0) == doctest::Approx(0.01));
  CHECK(out->se(0) == doctest::Approx(0.1));
  CHECK(out->chisq_standard == doctest::Approx(50.0));
  REQUIRE(out->eigvals.size() == 1);
  CHECK(out->eigvals(0) == doctest::Approx(1.0));
  CHECK(out->satorra_bentler.scale_c == doctest::Approx(1.0));
  CHECK(out->satorra_bentler.chi2_scaled == doctest::Approx(50.0));
}

TEST_CASE("robust_weighted_moments respects per-block sample weighting and K") {
  magmaan::estimate::WeightedMomentBlock b1;
  b1.jacobian.resize(1, 2);
  b1.jacobian << 1.0, 0.0;
  b1.weight.resize(1, 1);
  b1.weight << 4.0;
  b1.gamma.resize(1, 1);
  b1.gamma << 2.0;
  b1.n_obs = 50;

  magmaan::estimate::WeightedMomentBlock b2;
  b2.jacobian.resize(1, 2);
  b2.jacobian << 0.0, 1.0;
  b2.weight.resize(1, 1);
  b2.weight << 3.0;
  b2.gamma.resize(1, 1);
  b2.gamma << 5.0;
  b2.n_obs = 150;

  Eigen::MatrixXd K(2, 1);
  K << 1.0, 1.0;
  auto out = magmaan::estimate::robust_weighted_moments({b1, b2}, K, 0.25);
  REQUIRE(out.has_value());
  CHECK(out->df == 1);
  REQUIRE(out->vcov.rows() == 2);
  CHECK(out->vcov(0, 0) == doctest::Approx(41.75 / (3.25 * 3.25) / 200.0));
  CHECK(out->vcov(1, 1) == doctest::Approx(out->vcov(0, 0)));
  CHECK(out->vcov(0, 1) == doctest::Approx(out->vcov(0, 0)));
  CHECK(out->chisq_standard == doctest::Approx(50.0));
  REQUIRE(out->eigvals.size() == 1);
  CHECK(out->eigvals(0) >= 0.0);
}

TEST_CASE("evaluate_ls_objective matches fitted fmin for continuous LS") {
  auto fx = one_factor_fixture();
  const magmaan::optim::LbfgsOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};

  auto est_uls = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Lbfgs, opt);
  REQUIRE(est_uls.has_value());
  auto f_uls = magmaan::estimate::evaluate_ls_objective(
      fx.pt, fx.rep, fx.samp, est_uls->theta, magmaan::estimate::gmm::Weight{});
  REQUIRE(f_uls.has_value());
  CHECK(*f_uls == doctest::Approx(est_uls->fmin).epsilon(1e-12));

  auto est_gls = magmaan::test::fit_gls(
      fx.pt, fx.rep, fx.samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Lbfgs, opt);
  REQUIRE(est_gls.has_value());
  auto f_gls = magmaan::estimate::evaluate_ls_objective(
      fx.pt, fx.rep, fx.samp, est_gls->theta,
      gls_weight(fx.pt, fx.rep, fx.samp, est_gls->theta));
  REQUIRE(f_gls.has_value());
  CHECK(*f_gls == doctest::Approx(est_gls->fmin).epsilon(1e-12));

  magmaan::estimate::gmm::Weight wls = wls_weights_from_sample(fx.samp);
  auto est_wls = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, wls, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Lbfgs, opt);
  REQUIRE(est_wls.has_value());
  auto f_wls = magmaan::estimate::evaluate_ls_objective(
      fx.pt, fx.rep, fx.samp, est_wls->theta, wls);
  REQUIRE(f_wls.has_value());
  CHECK(*f_wls == doctest::Approx(est_wls->fmin).epsilon(1e-12));
}

TEST_CASE("evaluate_ls_objective reports data objective under LS constraints") {
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + b2*x2 + b3*x3\n"
      "b2 + b3 == 1.5");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());

  const Eigen::Vector3d lam(1.0, 0.7, 0.8);
  const Eigen::Vector3d th(0.5, 0.6, 0.4);
  magmaan::data::SampleStats samp;
  samp.S = {lam * lam.transpose() * 1.8 + th.asDiagonal().toDenseMatrix()};
  samp.n_obs = {250};

  const magmaan::optim::LbfgsOptions opt{
      .max_iter = 5000, .ftol = 1e-14, .gtol = 1e-9};
  auto est = magmaan::test::fit_gmm(
      *pt, *rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Lbfgs, opt);
  REQUIRE(est.has_value());

  auto f = magmaan::estimate::evaluate_ls_objective(
      *pt, *rep, samp, est->theta, magmaan::estimate::gmm::Weight{});
  REQUIRE(f.has_value());
  CHECK(*f == doctest::Approx(est->fmin).epsilon(1e-12));
}

TEST_CASE("robust_continuous_ls: raw and supplied Gamma agree for ULS") {
  auto fx = one_factor_fixture();
  const magmaan::optim::LbfgsOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
  auto est = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Lbfgs, opt);
  REQUIRE(est.has_value());

  auto G = magmaan::data::empirical_gamma(fx.raw.X[0]);
  REQUIRE(G.has_value());
  auto by_gamma = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est, magmaan::estimate::gmm::Weight{}, {*G});
  auto by_raw = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est, magmaan::estimate::gmm::Weight{}, fx.raw);
  REQUIRE(by_gamma.has_value());
  REQUIRE(by_raw.has_value());

  CHECK((by_gamma->vcov - by_raw->vcov).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((by_gamma->se - by_raw->se).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((by_gamma->eigvals - by_raw->eigvals).cwiseAbs().maxCoeff() < 1e-10);
  CHECK(by_raw->chisq_standard ==
        doctest::Approx(2.0 * total_n(fx.samp) * est->fmin));
}

TEST_CASE("robust_continuous_ls: GLS and WLS preserve continuous LS statistic scale") {
  auto fx = one_factor_fixture();
  const magmaan::optim::LbfgsOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};

  auto est_gls = magmaan::test::fit_gls(
      fx.pt, fx.rep, fx.samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Lbfgs, opt);
  REQUIRE(est_gls.has_value());
  auto rob_gls = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est_gls,
      gls_weight(fx.pt, fx.rep, fx.samp, est_gls->theta), fx.raw);
  REQUIRE(rob_gls.has_value());
  CHECK(rob_gls->chisq_standard ==
        doctest::Approx(2.0 * total_n(fx.samp) * est_gls->fmin));
  CHECK(rob_gls->se.allFinite());

  magmaan::estimate::gmm::Weight wls = wls_weights_from_sample(fx.samp);
  auto est_wls = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, wls, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Lbfgs, opt);
  REQUIRE(est_wls.has_value());
  auto rob_wls = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est_wls, wls, fx.raw);
  REQUIRE(rob_wls.has_value());
  CHECK(rob_wls->chisq_standard ==
        doctest::Approx(2.0 * total_n(fx.samp) * est_wls->fmin));
  CHECK(rob_wls->se.allFinite());
}

TEST_CASE("robust_continuous_ls: validates Gamma block dimensions") {
  auto fx = one_factor_fixture();
  const magmaan::optim::LbfgsOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
  auto est = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Lbfgs, opt);
  REQUIRE(est.has_value());
  std::vector<Eigen::MatrixXd> bad{
      Eigen::MatrixXd::Identity(2, 2)};
  CHECK_FALSE(magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est, magmaan::estimate::gmm::Weight{}, bad).has_value());
}
