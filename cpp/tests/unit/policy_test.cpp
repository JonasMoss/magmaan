#include <doctest/doctest.h>
#include "../test_fit.hpp"
#include "../oracle.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "magmaan/api/policy.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/frontier/fmg.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"
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

TEST_CASE("policy: fully specified global tests use all covariance and mean directions") {
  for (bool means : {false, true}) {
    for (int groups : {1, 2}) {
      for (auto storage : {ntml::ContributionStorage::Casewise, ntml::ContributionStorage::Tiled}) {
        CAPTURE(means);
        CAPTURE(groups);
        CAPTURE(static_cast<int>(storage));
        std::mt19937 rng(8u);
        magmaan::data::RawData raw;
        raw.X.push_back(t_rows(rng, 120, Eigen::Vector4d::Constant(0.3)));
        if (groups == 2)
          raw.X.push_back(t_rows(rng, 170, Eigen::Vector4d::Constant(-0.4)));
        std::string syntax = "f =~ 1*x1 + 0.9*x2 + 0.8*x3 + 0.7*x4\nf ~~ 1*f\n"
                             "x1 ~~ 1*x1\nx2 ~~ 1*x2\nx3 ~~ 1*x3\nx4 ~~ 1*x4";
        if (means) syntax += "\nx1 ~ 0.1*1\nx2 ~ -0.2*1\nx3 ~ 0.2*1\nx4 ~ -0.1*1";
        auto p = prepare(build(syntax, means, groups), raw, means);
        p.data->storage = storage;
        REQUIRE(p.est.theta.size() == 0);
        auto out = api::policy_inference_ml(*p.fit, {});
        CHECK(out.covariance_reason == api::InferenceReason::Available);
        CHECK(out.covariance.rows() == 0);
        CHECK(out.covariance.cols() == 0);
        REQUIRE(out.score.reason == api::InferenceReason::Available);
        REQUIRE(out.lr.reason == api::InferenceReason::Available);

        // Independent saturated-normal score and LR formulas. With no model
        // directions to remove, the full moment space is tested. Construct
        // Gamma_NT and each empirical score/moment row directly from raw X,
        // rather than using the library's Jacobian, U or contribution helpers.
        Eigen::Vector4d loading, mu;
        loading << 1.0, 0.9, 0.8, 0.7;
        mu << 0.1, -0.2, 0.2, -0.1;
        const Eigen::Matrix4d sigma = loading * loading.transpose() + Eigen::Matrix4d::Identity();
        const Eigen::Matrix4d inverse = sigma.inverse();
        const int dimension = means ? 14 : 10;
        Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(dimension, dimension);
        if (means) gamma.topLeftCorner(4, 4) = sigma;
        int a = means ? 4 : 0;
        for (int j = 0; j < 4; ++j) for (int i = j; i < 4; ++i, ++a) {
          int b = means ? 4 : 0;
          for (int l = 0; l < 4; ++l) for (int k = l; k < 4; ++k, ++b)
            gamma(a, b) = sigma(i, k) * sigma(j, l) + sigma(i, l) * sigma(j, k);
        }
        const Eigen::LLT<Eigen::MatrixXd> metric(gamma);
        double score_reference = 0.0, lr_reference = 0.0;
        Eigen::VectorXd score_eigen(groups * dimension), lr_eigen(groups * dimension);
        for (int group = 0; group < groups; ++group) {
          const auto& x = raw.X[static_cast<std::size_t>(group)];
          const double n = static_cast<double>(x.rows());
          const Eigen::Vector4d mean = x.colwise().mean().transpose();
          const Eigen::MatrixXd centered = x.rowwise() - mean.transpose();
          const Eigen::Matrix4d sample = centered.transpose() * centered / n;
          const Eigen::Vector4d shift = means ? Eigen::Vector4d(mean - mu) : Eigen::Vector4d::Zero();
          const Eigen::Matrix4d error = sample + shift * shift.transpose() - sigma;
          score_reference += n * (shift.dot(inverse * shift) +
                                  0.5 * (inverse * error * inverse * error).trace());
          lr_reference += n * ((inverse * sample).trace() -
                               std::log(sample.determinant() / sigma.determinant()) - 4.0 +
                               shift.dot(inverse * shift));
          Eigen::MatrixXd score_rows(x.rows(), dimension), lr_rows(x.rows(), dimension);
          for (Eigen::Index row = 0; row < x.rows(); ++row) {
            const Eigen::Vector4d z = centered.row(row).transpose();
            const Eigen::Vector4d residual = z + shift;
            if (means) {
              score_rows.row(row).head(4) = residual.transpose();
              lr_rows.row(row).head(4) = z.transpose();
            }
            int col = means ? 4 : 0;
            for (int j = 0; j < 4; ++j) for (int i = j; i < 4; ++i, ++col) {
              score_rows(row, col) = residual(i) * residual(j) - sigma(i, j);
              lr_rows(row, col) = z(i) * z(j) - sample(i, j);
            }
          }
          auto eigenvalues = [&](const Eigen::MatrixXd& rows) -> Eigen::VectorXd {
            const Eigen::MatrixXd whitened = metric.matrixL().solve(rows.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(whitened * whitened.transpose() / n);
            REQUIRE(eig.info() == Eigen::Success);
            return eig.eigenvalues();
          };
          score_eigen.segment(group * dimension, dimension) = eigenvalues(score_rows);
          lr_eigen.segment(group * dimension, dimension) = eigenvalues(lr_rows);
        }
        std::sort(score_eigen.data(), score_eigen.data() + score_eigen.size());
        std::sort(lr_eigen.data(), lr_eigen.data() + lr_eigen.size());
        CHECK(out.score.statistic == doctest::Approx(score_reference).epsilon(1e-10));
        CHECK(out.lr.statistic == doctest::Approx(lr_reference).epsilon(1e-10));
        for (bool score : {true, false}) {
          const auto& test = score ? out.score : out.lr;
          const auto& eigen = score ? score_eigen : lr_eigen;
          CHECK(test.df == groups * dimension);
          CHECK((test.eigenvalues - eigen).norm() < 1e-10);
          CHECK(test.sb_scale == doctest::Approx(eigen.mean()).epsilon(1e-10));
          CHECK(test.p_sb == doctest::Approx(inf::chi2_pvalue(test.statistic / eigen.mean(), test.df)));
          CHECK(test.p_peba4 == doctest::Approx(ntml::fmg_test(test.statistic, test.df, eigen,
                                              {ntml::FmgMethod::Peba, 4.0, true}).p_value));
        }
        // With fixed means, the zero-column mean Jacobian still has rows.
        REQUIRE(p.fit->geometry.has_value());
        CHECK(p.fit->geometry->q == 0);
        CHECK(p.fit->geometry->base.has_means == means);
        CHECK(p.fit->geometry->base.total_rows == groups * dimension);
        auto failed = api::policy_inference_ml(*p.fit, {false, false});
        CHECK(failed.score.reason == api::InferenceReason::NotConverged);
        CHECK(failed.lr.reason == api::InferenceReason::NotConverged);
      }
    }
  }
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
  // A PSD boundary estimate still gets every component, flagged: the regular
  // limits hold when the population is interior.
  auto interior = api::policy_inference_ml(*p.fit, {true, false});
  auto boundary = api::policy_inference_ml(*p.fit, {true, true});
  CHECK_FALSE(interior.psd_boundary);
  CHECK(boundary.psd_boundary);
  CHECK(boundary.covariance_reason == api::InferenceReason::Available);
  CHECK(boundary.lr.reason == api::InferenceReason::Available);
  CHECK((boundary.covariance - interior.covariance).norm() == 0.0);
  CHECK(boundary.score.statistic == interior.score.statistic);
}

namespace {

std::shared_ptr<ntml::NTMLFit> prepare_on(const std::shared_ptr<ntml::NTMLData>& data,
                                          const Model& model) {
  auto est = magmaan::test::fit(model.pt, model.rep, data->sample);
  REQUIRE(est.has_value());
  auto fit = ntml::prepare_ntml_fit(data, model.pt, model.rep, *est);
  REQUIRE(fit.has_value());
  return *fit;
}

}  // namespace

TEST_CASE("policy nested tests: the hypothesis quadratics with SB and PEBA4") {
  std::mt19937 rng(314u);
  magmaan::data::RawData raw;
  raw.X.push_back(t_rows(rng, 400, Eigen::Vector4d::Zero()));
  auto data = ntml::prepare_ntml_data(raw, false, ntml::ContributionStorage::Casewise);
  REQUIRE(data.has_value());
  const Model alt_model = build("f =~ x1 + x2 + x3 + x4", false);
  const Model null_model = build("f =~ x1 + a*x2 + a*x3 + a*x4", false);
  auto alt = prepare_on(*data, alt_model);
  auto null = prepare_on(*data, null_model);
  auto out = api::policy_nested_ml(null, {}, alt, {});
  CHECK_FALSE(out.psd_boundary);
  // References on fresh snapshots of the same fits.
  auto h = ntml::prepare_ntml_hypothesis(prepare_on(*data, null_model), prepare_on(*data, alt_model));
  REQUIRE(h.has_value());
  for (bool score : {true, false}) {
    const auto& test = score ? out.score : out.lr;
    REQUIRE(test.reason == api::InferenceReason::Available);
    auto quadratic = ntml::ntml_quadratic(**h, score);
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
  }
  CHECK(out.lr.statistic == doctest::Approx(inf::chi2_stat((*data)->sample, null->estimates) -
                                            inf::chi2_stat((*data)->sample, alt->estimates)));
  // Under a true null the score and LR statistics agree to first order.
  CHECK(std::abs(out.score.statistic - out.lr.statistic) < 0.1 * (1.0 + out.lr.statistic));
}

TEST_CASE("policy nested tests: gating and nesting") {
  std::mt19937 rng(2718u);
  magmaan::data::RawData raw;
  raw.X.push_back(t_rows(rng, 250, Eigen::Vector4d::Zero()));
  auto data = ntml::prepare_ntml_data(raw, false, ntml::ContributionStorage::Casewise);
  REQUIRE(data.has_value());
  auto alt = prepare_on(*data, build("f =~ x1 + x2 + x3 + x4", false));
  auto null = prepare_on(*data, build("f =~ x1 + a*x2 + a*x3 + x4", false));
  auto swapped = api::policy_nested_ml(alt, {}, null, {});
  CHECK(swapped.lr.reason == api::InferenceReason::NotNested);
  CHECK(api::reason_name(swapped.score.reason) == "not_nested");
  auto failed = api::policy_nested_ml(null, {false, false}, alt, {});
  CHECK(failed.lr.reason == api::InferenceReason::NotConverged);
  auto boundary = api::policy_nested_ml(null, {true, true}, alt, {});
  CHECK(boundary.psd_boundary);
  CHECK(boundary.lr.reason == api::InferenceReason::Available);
  // Different observations cannot be compared.
  auto other = ntml::prepare_ntml_data(raw, false, ntml::ContributionStorage::Casewise);
  REQUIRE(other.has_value());
  auto elsewhere = api::policy_nested_ml(prepare_on(*other, build("f =~ x1 + a*x2 + a*x3 + x4", false)),
                                         {}, alt, {});
  CHECK(elsewhere.score.reason == api::InferenceReason::NotNested);
}

TEST_CASE("policy nested embedding: omitted, fixed and equality paths share null geometry") {
  std::mt19937 rng(817u);
  magmaan::data::RawData raw;
  raw.X.push_back(t_rows(rng, 350, Eigen::Vector4d::Zero()));
  auto data = ntml::prepare_ntml_data(raw, false, ntml::ContributionStorage::Casewise);
  REQUIRE(data.has_value());
  const auto named_model = [](const std::string& syntax) {
    auto parsed = magmaan::parse::Parser::parse(syntax);
    REQUIRE(parsed.has_value());
    magmaan::spec::LatentNames names;
    auto pt = magmaan::spec::build(*parsed, {}, nullptr, &names);
    REQUIRE(pt.has_value());
    auto rep = magmaan::model::build_matrix_rep(*pt, &names);
    REQUIRE(rep.has_value());
    return Model{*pt, *rep};
  };
  const std::string base = "f =~ x1 + x2 + x3 + x4";
  auto alt = prepare_on(*data, named_model(base + "\nx1 ~~ x2"));
  auto null = prepare_on(*data, named_model(base));
  auto expected = api::policy_nested_ml(null, {}, alt, {});
  REQUIRE(expected.score.reason == api::InferenceReason::Available);
  REQUIRE(expected.lr.reason == api::InferenceReason::Available);
  // Evaluate exactly the same fitted point in all spellings. This isolates
  // inference invariance from optimizer tolerances in independent fits.
  for (const std::string tail : {"\nx1 ~~ 0*x2", "\nx1 ~~ a*x2\na == 0"}) {
    auto model = named_model(base + tail);
    auto c1 = magmaan::estimate::build_eq_constraints(model.pt);
    auto c0 = magmaan::estimate::build_eq_constraints(null->pt);
    REQUIRE(c1.has_value()); REQUIRE(c0.has_value());
    auto embedding = magmaan::robust::embed_nested_null(model.pt, model.rep,
        null->pt, null->rep, null->estimates.theta, *c1, *c0);
    REQUIRE(embedding.has_value());
    auto estimates = null->estimates;
    estimates.theta = embedding->theta;
    auto spelling = ntml::prepare_ntml_fit(*data, model.pt, model.rep, estimates);
    REQUIRE(spelling.has_value());
    auto result = api::policy_nested_ml(*spelling, {}, alt, {});
    if (tail.find("a == 0")!=std::string::npos) {
      auto hypothesis=ntml::prepare_ntml_hypothesis(*spelling,alt);
      REQUIRE(hypothesis.has_value());
      CHECK_FALSE((*hypothesis)->embedded_null);
      auto alternative_constraints=magmaan::estimate::build_eq_constraints(alt->pt);
      REQUIRE(alternative_constraints.has_value());
      auto before=magmaan::robust::restriction_alpha_from_K(*alternative_constraints,*c1);
      REQUIRE(before.has_value());
      CHECK((before->A-(*hypothesis)->restriction.A).norm()==0.0);
      CHECK((before->b-(*hypothesis)->restriction.b).norm()==0.0);
    }
    for (bool score : {true, false}) {
      const auto& x = score ? result.score : result.lr;
      const auto& y = score ? expected.score : expected.lr;
      REQUIRE(x.reason == api::InferenceReason::Available);
      CHECK(std::abs(x.statistic-y.statistic) < 1e-10);
      CHECK((x.eigenvalues-y.eigenvalues).norm() < 1e-10);
      CHECK(std::abs(x.p_sb-y.p_sb) < 1e-10);
      CHECK(std::abs(x.p_peba4-y.p_peba4) < 1e-10);
    }
  }
}

TEST_CASE("policy nested embedding: reparameterized null and boundary have typed outcomes") {
  const auto named_model = [](const std::string& syntax, bool std_lv) {
    auto parsed = magmaan::parse::Parser::parse(syntax);
    REQUIRE(parsed.has_value());
    magmaan::spec::LatentNames names;
    magmaan::spec::BuildOptions opts;
    opts.std_lv = std_lv;
    auto pt = magmaan::spec::build(*parsed, opts, nullptr, &names);
    REQUIRE(pt.has_value());
    auto rep = magmaan::model::build_matrix_rep(*pt, &names);
    REQUIRE(rep.has_value());
    return Model{*pt, *rep};
  };
  auto alt = named_model("f =~ x1 + x2 + x3 + x4", false);
  auto null = named_model("g =~ x1 + a*x2 + a*x3 + x4", true);
  std::mt19937 rng(163u);
  magmaan::data::RawData raw;
  raw.X.push_back(t_rows(rng, 350, Eigen::Vector4d::Zero()));
  auto data = ntml::prepare_ntml_data(raw, false);
  REQUIRE(data.has_value());
  auto h1 = prepare_on(*data, alt), h0 = prepare_on(*data, null);
  auto c1 = magmaan::estimate::build_eq_constraints(alt.pt);
  auto c0 = magmaan::estimate::build_eq_constraints(null.pt);
  REQUIRE(c1.has_value()); REQUIRE(c0.has_value());
  auto unsupported = magmaan::robust::embed_nested_null(alt.pt,alt.rep,
      null.pt,null.rep,h0->estimates.theta,*c1,*c0);
  REQUIRE_FALSE(unsupported.has_value());
  CHECK(unsupported.error().kind == magmaan::PostError::Kind::UnsupportedNesting);
  auto same_point = magmaan::robust::embed_nested_null(alt.pt,alt.rep,
      null.pt,null.rep,h0->estimates.theta,*c1,*c0,true,&h1->estimates.theta);
  REQUIRE(same_point.has_value());
  CHECK(same_point->through_moments);
  auto policy = api::policy_nested_ml(h0,{},h1,{});
  CHECK(policy.lr.reason == api::InferenceReason::Available);
  CHECK(policy.score.reason == api::InferenceReason::Available);

  auto two = named_model("f =~ x1 + x2\ng =~ x3 + x4", true);
  auto one = named_model("h =~ x1 + x2 + x3 + x4", true);
  auto single = prepare_on(*data,one);
  auto ct = magmaan::estimate::build_eq_constraints(two.pt);
  auto cs = magmaan::estimate::build_eq_constraints(one.pt);
  REQUIRE(ct.has_value()); REQUIRE(cs.has_value());
  auto boundary = magmaan::robust::embed_nested_null(two.pt,two.rep,
      one.pt,one.rep,single->estimates.theta,*ct,*cs,true);
  REQUIRE_FALSE(boundary.has_value());
  CHECK(boundary.error().kind == magmaan::PostError::Kind::BoundaryNesting);
}

TEST_CASE("policy nested embedding: frozen lavaan dropped-loading score and exact LR") {
  auto fixture = magmaan::test::read_fixture(magmaan::test::fixtures_dir()+"/nested_embedding.json");
  REQUIRE(fixture.has_value());
  const auto j=nlohmann::json::parse(*fixture,nullptr,false);
  REQUIRE_FALSE(j.is_discarded());
  magmaan::data::RawData raw;
  raw.X.push_back(magmaan::test::matrix_from_json(j["X"]));
  auto data=ntml::prepare_ntml_data(raw,false);
  REQUIRE(data.has_value());
  const auto model_fit = [&](const std::string& syntax,const nlohmann::json& rows,double fmin) {
    auto parsed=magmaan::parse::Parser::parse(syntax);
    REQUIRE(parsed.has_value());
    magmaan::spec::LatentNames names;
    auto pt=magmaan::spec::build(*parsed,{},nullptr,&names);
    REQUIRE(pt.has_value());
    auto rep=magmaan::model::build_matrix_rep(*pt,&names);
    REQUIRE(rep.has_value());
    magmaan::estimate::Estimates est;
    est.theta=Eigen::VectorXd::Zero(pt->n_free()); est.fmin=fmin;
    for (std::size_t i=0;i<pt->size();++i) {
      if (pt->free[i]==0) continue;
      for (const auto& row:rows) {
        if (row["lhs"]==names.row_lhs[i] && row["rhs"]==names.row_rhs[i] &&
            row["op"]==magmaan::parse::to_string(pt->op[i]) && row["group"]==pt->group[i])
          est.theta(pt->free[i]-1)=row["est"].get<double>();
      }
    }
    auto fit=ntml::prepare_ntml_fit(*data,*pt,*rep,est);
    REQUIRE(fit.has_value());
    return *fit;
  };
  auto alt=model_fit(j["model_H1"].get<std::string>(),j["rows_H1"],j["fmin_H1"].get<double>());
  const std::string base=j["model_H0"].get<std::string>();
  for (const std::string tail:{"", "\nvisual =~ 0*x9", "\nvisual =~ a*x9\na == 0"}) {
    auto null=model_fit(base+tail,j["rows_H0"],j["fmin_H0"].get<double>());
    auto result=api::policy_nested_ml(null,{},alt,{});
    REQUIRE(result.score.reason==api::InferenceReason::Available);
    REQUIRE(result.lr.reason==api::InferenceReason::Available);
    CHECK(std::abs(result.score.statistic-j["score"].get<double>())<1e-5);
    CHECK(std::abs(result.lr.statistic-j["lr"].get<double>())<1e-10);
    auto c1=magmaan::estimate::build_eq_constraints(alt->pt);
    auto c0=magmaan::estimate::build_eq_constraints(null->pt);
    REQUIRE(c1.has_value()); REQUIRE(c0.has_value());
    auto lr=magmaan::robust::lr_test_satorra2000_from_data(
        alt->pt,alt->rep,alt->estimates.theta,*c1,
        null->pt,null->rep,null->estimates.theta,*c0,raw.X,
        {raw.X[0].colwise().mean().transpose()},
        {static_cast<std::int32_t>(raw.X[0].rows())},{1.0},
        2.0*static_cast<double>(raw.X[0].rows())*null->estimates.fmin,
        2.0*static_cast<double>(raw.X[0].rows())*alt->estimates.fmin,
        1,0,{});
    REQUIRE(lr.has_value());
    CHECK(std::abs(lr->T_scaled-j["lr_scaled_expected"].get<double>())<1e-5);
  }
}

TEST_CASE("policy nested rank checks preserve variable units") {
  std::mt19937 rng(6105u);
  magmaan::data::RawData raw;
  raw.X.push_back(t_rows(rng, 400, Eigen::Vector4d::Zero()));
  auto data = ntml::prepare_ntml_data(raw, false, ntml::ContributionStorage::Casewise);
  REQUIRE(data.has_value());
  auto alt = prepare_on(*data, build("f =~ x1 + x2 + x3 + x4", false));
  auto null = prepare_on(*data, build("f =~ x1 + a*x2 + a*x3 + x4", false));
  const auto reference = api::policy_nested_ml(null, {}, alt, {});
  for (double units : {0.01, 100.0, 1000.0}) {
    // Change only the units, carrying fitted points through the exact parameter
    // transformation to keep optimizer accuracy out of the rank regression.
    auto scaled = raw;
    scaled.X[0].col(3) *= units;
    auto scaled_data = ntml::prepare_ntml_data(scaled, false, ntml::ContributionStorage::Casewise);
    REQUIRE(scaled_data.has_value());
    auto transform = [&](const std::shared_ptr<ntml::NTMLFit>& fit) {
      auto est = fit->estimates;
      for (std::size_t r = 0; r < fit->pt.free.size(); ++r) {
        const auto& cell = fit->rep.cell_for_row[r];
        if (!cell.used || fit->pt.free[r] <= 0) continue;
        double multiplier = 1.0;
        // LISREL: x4's loading and residual variance carry its units.
        if (cell.mat == magmaan::model::MatId::Lambda && cell.row == 3) multiplier = units;
        if (cell.mat == magmaan::model::MatId::Theta && cell.row == 3 && cell.col == 3) multiplier = units*units;
        est.theta(fit->pt.free[r]-1) *= multiplier;
      }
      auto out = ntml::prepare_ntml_fit(*scaled_data, fit->pt, fit->rep, est);
      REQUIRE(out.has_value());
      return *out;
    };
    const auto result = api::policy_nested_ml(transform(null), {}, transform(alt), {});
    for (bool score : {false, true}) {
      const auto& expected = score ? reference.score : reference.lr;
      const auto& test = score ? result.score : result.lr;
      INFO("units=", units, ", score=", score, ", reason=", api::reason_name(test.reason));
      REQUIRE(expected.reason == api::InferenceReason::Available);
      REQUIRE(test.reason == api::InferenceReason::Available);
      CHECK(test.statistic == doctest::Approx(expected.statistic).epsilon(1e-8));
      CHECK(test.eigenvalues.isApprox(expected.eigenvalues, 1e-8));
      CHECK(test.p_sb == doctest::Approx(expected.p_sb).epsilon(1e-8));
      CHECK(test.p_peba4 == doctest::Approx(expected.p_peba4).epsilon(1e-8));
    }
  }
}
