#include <doctest/doctest.h>
#include "magmaan/api/sem.hpp"
#include "magmaan/api/policy.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include <Eigen/Cholesky>
#include <cmath>
#include <random>

namespace {
using namespace magmaan;
const api::PolicyFitIndex& index(const api::PolicyFitMeasures& out, std::string_view name) {
  for (const auto& row : out.indices) if (row.index == name) return row;
  REQUIRE(false);
  return out.indices.front();
}
data::RawData rows(int n = 256) {
  std::mt19937 rng(101);
  std::normal_distribution<double> normal;
  data::RawData raw;
  raw.X.emplace_back(n, 4);
  for (int i = 0; i < n; ++i) {
    const double f = normal(rng), nuisance = normal(rng);
    for (int j = 0; j < 4; ++j) raw.X[0](i,j) = 0.7 * f + (j < 2 ? 0.4 * nuisance : 0) + normal(rng);
  }
  return raw;
}
api::Fit fitted(const data::RawData& raw, bool fiml = false, bool uls = false) {
  api::ModelOptions opts;
  opts.build.meanstructure = fiml;
  opts.build.fixed_x = false;
  opts.build.n_groups = static_cast<int>(raw.X.size());
  auto model = api::model_from_lavaan("f =~ x1 + x2 + x3 + x4", opts);
  REQUIRE(model);
  auto sample = data::sample_stats_from_raw(raw);
  REQUIRE(sample);
  if (!fiml) sample->mean.clear();
  auto d = fiml ? api::data_from_raw(*model, raw) : api::data_from_sample_stats(*model, *sample);
  REQUIRE(d);
  auto fit = api::fit(*model, *d, uls ? api::uls() : fiml ? api::fiml() : api::ml());
  REQUIRE_MESSAGE(fit, (fit ? "" : fit.error().detail));
  return *fit;
}
}

TEST_CASE("policy fit measures: Takeuchi equals independent profile Hessian plus moment bias") {
  using namespace magmaan;
  data::RawData raw;
  raw.X.emplace_back(128, 1);
  for (int i = 0; i < 128; ++i) raw.X[0](i,0) = std::sin(i * 0.17) + 0.4 * std::cos(i * 0.31);
  raw.X[0].array() -= raw.X[0].mean();
  raw.X[0] /= std::sqrt(raw.X[0].squaredNorm() / 128.0);
  raw.X[0].array() += 1.0;
  api::ModelOptions opts;
  opts.build.meanstructure = true;
  auto model = api::model_from_lavaan("x ~~ x\nx ~ 0*1", opts);
  REQUIRE(model);
  estimate::Estimates est;
  est.theta = Eigen::VectorXd::Constant(1, 2.0);
  est.fmin = std::log(2.0) / 2.0;
  const auto out = api::policy_fit_measures(model->structure(), model->matrix_rep(), raw, est, {});
  Eigen::MatrixXd influence(128, 2);
  influence.col(0) = raw.X[0].col(0).array() - 1.0;
  influence.col(1) = influence.col(0).array().square() - 1.0;
  const Eigen::Matrix2d gamma = influence.transpose() * influence / 128.0;
  // Independent analytic profile of F(m,S) = log(S+m^2)-log(S).
  Eigen::Matrix2d h;
  h << 0.0, -0.5, -0.5, 0.75;
  const double profile_trace = 0.5 * (h * gamma).trace();
  const auto f = [](double mean, double variance) {
    return std::log(variance + mean * mean) - std::log(variance);
  };
  constexpr double step = 1e-4;
  Eigen::Matrix2d fd;
  fd(0,0) = (f(1+step,1)-2*f(1,1)+f(1-step,1))/(step*step);
  fd(1,1) = (f(1,1+step)-2*f(1,1)+f(1,1-step))/(step*step);
  fd(0,1) = fd(1,0) = (f(1+step,1+step)-f(1+step,1-step)-f(1-step,1+step)+f(1-step,1-step))/(4*step*step);
  const double moment_bias = 0.5; // grad_S(F)*(-S), N-divisor covariance
  CHECK(out.user.discrepancy == doctest::Approx(std::log(2.0)).epsilon(1e-12));
  CHECK(out.user.trace == doctest::Approx(profile_trace + moment_bias).epsilon(1e-6));
  CHECK(out.user.trace == doctest::Approx(0.5*(fd*gamma).trace()+moment_bias).epsilon(1e-6));
  CHECK(out.user.df == 1);
  CHECK(index(out, "rmsea").reason == api::InferenceReason::Available);
}

TEST_CASE("policy fit measures: ML and ULS duplicated groups preserve indices") {
  using namespace magmaan;
  auto raw = rows();
  for (bool uls : {false, true}) {
    auto single = fitted(raw, false, uls);
    auto duplicate = raw;
    duplicate.X.push_back(raw.X[0]);
    auto grouped = fitted(duplicate, false, uls);
    auto one = api::policy_fit_measures(single.model().structure(),single.model().matrix_rep(),raw,single.estimates(),{},false,uls);
    auto two = api::policy_fit_measures(grouped.model().structure(),grouped.model().matrix_rep(),duplicate,grouped.estimates(),{},false,uls);
    for (auto name : {"rmsea", "cfi", "tli", "srmr"}) {
      REQUIRE_MESSAGE(index(one, name).reason == api::InferenceReason::Available, index(one,name).detail);
      REQUIRE_MESSAGE(index(two, name).reason == api::InferenceReason::Available, index(two,name).detail);
      CHECK(index(one,name).estimate == doctest::Approx(index(two,name).estimate).epsilon(1e-6));
    }
    CHECK(two.user.trace == doctest::Approx(2 * one.user.trace).epsilon(1e-6));
    CHECK(two.user.df == 2 * one.user.df);
  }
}

TEST_CASE("policy fit measures: likelihood helpers and state reasons") {
  using namespace magmaan;
  auto raw = rows();
  auto fit = fitted(raw);
  auto out = api::policy_fit_measures(fit.model().structure(),fit.model().matrix_rep(),raw,fit.estimates(),{});
  auto sample = data::sample_stats_from_raw(raw);
  REQUIRE(sample);
  sample->mean.clear();
  auto extras = measures::fit_extras(fit.model().structure(), fit.model().matrix_rep(), *sample, fit.estimates());
  REQUIRE(extras);
  CHECK(index(out,"logl").estimate == extras->logl);
  CHECK(index(out,"unrestricted.logl").estimate == extras->unrestricted_logl);
  CHECK(index(out,"aic").estimate == extras->aic);
  CHECK(index(out,"bic").estimate == extras->bic);
  CHECK(out.residual_uncorrected == doctest::Approx(extras->srmr).epsilon(1e-12));
  api::PolicyFitState state;
  state.converged = false;
  auto refused = api::policy_fit_measures(fit.model().structure(), fit.model().matrix_rep(), raw, fit.estimates(), state);
  for (const auto& i : refused.indices) CHECK(i.reason == api::InferenceReason::NotConverged);
  state.penalized = true;
  refused = api::policy_fit_measures(fit.model().structure(), fit.model().matrix_rep(), raw, fit.estimates(), state);
  for (const auto& i : refused.indices) CHECK(i.reason == api::InferenceReason::Penalized);
}

TEST_CASE("policy fit measures: saturated ML has per-index reasons") {
  using namespace magmaan;
  auto raw = rows();
  auto m = api::model_from_lavaan("x1 ~~ x1 + x2 + x3 + x4\nx2 ~~ x2 + x3 + x4\nx3 ~~ x3 + x4\nx4 ~~ x4");
  REQUIRE(m);
  auto sample = data::sample_stats_from_raw(raw); REQUIRE(sample);
  sample->mean.clear();
  auto d = api::data_from_sample_stats(*m,*sample); REQUIRE(d);
  auto f = api::fit(*m,*d,api::ml()); REQUIRE(f);
  auto out = api::policy_fit_measures(m->structure(),m->matrix_rep(),raw,f->estimates(),{});
  CHECK(index(out,"rmsea").reason == api::InferenceReason::Saturated);
  CHECK(index(out,"tli").reason == api::InferenceReason::Saturated);
  CHECK(index(out,"cfi").reason == api::InferenceReason::Available);
  CHECK(index(out,"cfi").estimate == doctest::Approx(1.0));
}
