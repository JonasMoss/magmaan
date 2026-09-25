#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <random>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include "magmaan/api/policy.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/frontier/fmg.hpp"
#include "magmaan/robust/prepared_ntml.hpp"
#include "magmaan/spec/build.hpp"

// The ordinary-user inference policy for complete-data ML. References:
//   * covariance: the observed-bread sandwich of casewise likelihood scores,
//     with the scores differentiated numerically from each case's normal
//     log-density, so the weighted Jacobian, the mean-shift correction and the
//     observed information are all checked independently;
//   * global tests: the shared NTML geometry and fmg_test, called directly.

namespace {

namespace api = magmaan::api;
namespace inf = magmaan::inference;
namespace ntml = magmaan::robust::frontier;
using magmaan::spec::GroupEqual;

struct Model {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

Model build(std::string_view src, bool means, int groups = 1,
            std::vector<GroupEqual> equal = {}) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = means;
  opts.n_groups = groups;
  opts.group_equal = std::move(equal);
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Model{std::move(*pt), std::move(*rep)};
}

// Heavy-tailed rows (multivariate t, 7 df) with the given mean, so the
// sandwich meat differs from the normal-theory information.
Eigen::MatrixXd t_rows(std::mt19937& rng, Eigen::Index n, const Eigen::VectorXd& mean) {
  Eigen::Matrix4d Sigma;
  Sigma << 2.0, 1.2, 1.0, 1.3,
           1.2, 1.9, 0.8, 1.0,
           1.0, 0.8, 1.7, 0.9,
           1.3, 1.0, 0.9, 1.8;
  const Eigen::Matrix4d L = Eigen::LLT<Eigen::Matrix4d>(Sigma).matrixL();
  std::normal_distribution<double> z(0.0, 1.0);
  std::chi_squared_distribution<double> chi(7.0);
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::Vector4d zi;
    for (int j = 0; j < 4; ++j) zi(j) = z(rng);
    X.row(i) = (mean + std::sqrt(5.0 / 7.0) * L * zi / std::sqrt(chi(rng) / 7.0)).transpose();
  }
  return X;
}

struct Prepared {
  Model model;
  magmaan::data::RawData raw;
  std::shared_ptr<ntml::NTMLData> data;
  magmaan::estimate::Estimates est;
  std::shared_ptr<ntml::NTMLFit> fit;
};

Prepared prepare(Model model, magmaan::data::RawData raw, bool means) {
  auto data = ntml::prepare_ntml_data(raw, means, ntml::ContributionStorage::Casewise);
  REQUIRE(data.has_value());
  auto est = magmaan::test::fit(model.pt, model.rep, (*data)->sample);
  REQUIRE(est.has_value());
  auto fit = ntml::prepare_ntml_fit(*data, model.pt, model.rep, *est);
  REQUIRE(fit.has_value());
  return Prepared{std::move(model), std::move(raw), *data, *est, *fit};
}

// Casewise scores d/dtheta of log N(x_i; mu(theta), Sigma(theta)) by central
// differences, stacked over groups in row order.
Eigen::MatrixXd numeric_scores(const Prepared& p) {
  auto ev = magmaan::model::ModelEvaluator::build(p.model.pt, p.model.rep);
  REQUIRE(ev.has_value());
  const Eigen::VectorXd theta = p.est.theta;
  Eigen::Index n = 0;
  for (const auto& x : p.raw.X) n += x.rows();
  auto loglik = [&](const Eigen::VectorXd& th) {
    auto m = ev->sigma(th);
    REQUIRE(m.has_value());
    Eigen::VectorXd out(n);
    Eigen::Index row = 0;
    for (std::size_t b = 0; b < p.raw.X.size(); ++b) {
      Eigen::LLT<Eigen::MatrixXd> llt(m->sigma[b]);
      REQUIRE(llt.info() == Eigen::Success);
      const double logdet = 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
      const Eigen::VectorXd mu = m->mu.empty() || m->mu[b].size() == 0
          ? Eigen::VectorXd(p.raw.X[b].colwise().mean().transpose()) : m->mu[b];
      for (Eigen::Index i = 0; i < p.raw.X[b].rows(); ++i, ++row) {
        const Eigen::VectorXd r = p.raw.X[b].row(i).transpose() - mu;
        out(row) = -0.5 * (logdet + r.dot(llt.solve(r)));
      }
    }
    return out;
  };
  Eigen::MatrixXd S(n, theta.size());
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    const double h = 1e-5 * std::max(1.0, std::abs(theta(k)));
    Eigen::VectorXd up = theta, down = theta;
    up(k) += h;
    down(k) -= h;
    S.col(k) = (loglik(up) - loglik(down)) / (2.0 * h);
  }
  return S;
}

double relative(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).norm() / b.norm();
}

Eigen::MatrixXd observed_bread(const Prepared& p) {
  auto H = inf::information_observed_analytic(p.model.pt, p.model.rep, p.data->sample, p.est);
  REQUIRE(H.has_value());
  auto V = inf::vcov(*H, p.model.pt, p.est.theta);
  REQUIRE(V.has_value());
  return *V;
}

}  // namespace

TEST_CASE("policy covariance: observed-bread sandwich of casewise scores") {
  std::mt19937 rng(20260925u);
  magmaan::data::RawData raw;
  raw.X.push_back(t_rows(rng, 300, Eigen::Vector4d::Zero()));
  auto p = prepare(build("f =~ x1 + x2 + x3 + x4", false), raw, false);
  auto out = api::policy_inference_ml(*p.fit, {});
  REQUIRE(out.covariance_reason == api::InferenceReason::Available);
  const Eigen::MatrixXd V = observed_bread(p);
  const Eigen::MatrixXd S = numeric_scores(p);
  const Eigen::MatrixXd reference = V * S.transpose() * S * V;
  CHECK(relative(out.covariance, reference) < 1e-6);
  // Without a mean structure the centered-moment scores are the likelihood
  // scores up to the first-order condition.
  auto centered = inf::casewise_scores(p.model.pt, p.model.rep, p.data->sample, p.raw, p.est);
  REQUIRE(centered.has_value());
  CHECK(relative(out.covariance, V * centered->transpose() * *centered * V) < 1e-6);
}

TEST_CASE("policy covariance: exact scores under a misspecified structured mean") {
  // Scalar invariance across two groups whose indicator means differ in
  // pattern, so the fitted means are not the sample means.
  std::mt19937 rng(9251u);
  magmaan::data::RawData raw;
  Eigen::Vector4d m1, m2;
  m1 << 0.0, 0.4, -0.3, 0.2;
  m2 << 0.5, 0.1, 0.6, -0.2;
  raw.X.push_back(t_rows(rng, 220, m1));
  raw.X.push_back(t_rows(rng, 260, m2));
  auto p = prepare(build("f =~ x1 + x2 + x3 + x4", true, 2,
                         {GroupEqual::Loadings, GroupEqual::Intercepts}),
                   raw, true);
  auto out = api::policy_inference_ml(*p.fit, {});
  REQUIRE(out.covariance_reason == api::InferenceReason::Available);
  const Eigen::MatrixXd V = observed_bread(p);
  const Eigen::MatrixXd S = numeric_scores(p);
  CHECK(relative(out.covariance, V * S.transpose() * S * V) < 1e-6);
  // Sample-mean-centered scores would give a different, inconsistent meat.
  auto centered = inf::casewise_scores(p.model.pt, p.model.rep, p.data->sample, p.raw, p.est);
  REQUIRE(centered.has_value());
  CHECK(relative(V * centered->transpose() * *centered * V, out.covariance) > 1e-3);
}

TEST_CASE("policy global tests: shared geometry with SB and PEBA4") {
  std::mt19937 rng(77u);
  magmaan::data::RawData raw;
  raw.X.push_back(t_rows(rng, 400, Eigen::Vector4d::Zero()));
  auto p = prepare(build("f =~ x1 + x2 + x3 + x4", false), raw, false);
  auto out = api::policy_inference_ml(*p.fit, {});
  auto q = prepare(build("f =~ x1 + x2 + x3 + x4", false), raw, false);
  for (bool score : {true, false}) {
    const auto& test = score ? out.score : out.lr;
    REQUIRE(test.reason == api::InferenceReason::Available);
    auto quadratic = ntml::ntml_quadratic(*q.fit, score);
    REQUIRE(quadratic.has_value());
    auto spectrum = ntml::ntml_spectrum(**quadratic);
    REQUIRE(spectrum.has_value());
    CHECK(test.df == 2);
    CHECK(test.statistic == doctest::Approx((*quadratic)->statistic).epsilon(1e-12));
    CHECK((test.eigenvalues - **spectrum).norm() < 1e-12);
    using ntml::FmgMethod;
    CHECK(test.p_peba4 == doctest::Approx(ntml::fmg_test(test.statistic, 2, **spectrum,
                                          {FmgMethod::Peba, 4.0, true}).p_value));
    CHECK(test.p_sb == doctest::Approx(inf::chi2_pvalue(test.statistic / test.sb_scale, 2)));
    CHECK(test.sb_scale > 1.0);  // heavy tails inflate the UGamma spectrum
  }
  CHECK(out.lr.statistic == doctest::Approx(inf::chi2_stat(p.data->sample, p.est)));
}

TEST_CASE("policy: saturated models get the covariance but no global test") {
  std::mt19937 rng(5u);
  magmaan::data::RawData raw;
  raw.X.push_back(t_rows(rng, 200, Eigen::Vector4d::Zero()).leftCols(3));
  auto p = prepare(build("f =~ x1 + x2 + x3", false), raw, false);
  auto out = api::policy_inference_ml(*p.fit, {});
  CHECK(out.covariance_reason == api::InferenceReason::Available);
  CHECK(out.covariance.rows() == 6);
  CHECK(out.score.reason == api::InferenceReason::Saturated);
  CHECK(out.lr.reason == api::InferenceReason::Saturated);
  CHECK(api::reason_name(out.lr.reason) == "saturated");
}

TEST_CASE("policy: the fit state gates every component") {
  std::mt19937 rng(11u);
  magmaan::data::RawData raw;
  raw.X.push_back(t_rows(rng, 150, Eigen::Vector4d::Zero()));
  auto p = prepare(build("f =~ x1 + x2 + x3 + x4", false), raw, false);
  const auto state = api::policy_fit_state(p.est);
  CHECK(state.converged);
  CHECK_FALSE(state.psd_boundary);
  auto failed = api::policy_inference_ml(*p.fit, {false, false});
  CHECK(failed.covariance_reason == api::InferenceReason::NotConverged);
  CHECK(failed.score.reason == api::InferenceReason::NotConverged);
  CHECK(failed.covariance.size() == 0);
  auto boundary = api::policy_inference_ml(*p.fit, {true, true});
  CHECK(boundary.covariance_reason == api::InferenceReason::PsdBoundary);
  CHECK(boundary.lr.reason == api::InferenceReason::PsdBoundary);
  CHECK(api::reason_name(boundary.lr.reason) == "psd_boundary");
}
