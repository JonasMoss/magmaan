#include <doctest/doctest.h>
#include "magmaan/api/sem.hpp"
#include "magmaan/api/policy.hpp"
#include "magmaan/api/frontier/fit_index_intervals.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include <Eigen/Cholesky>
#include <cmath>
#include <random>
#include <iostream>

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
api::Fit fitted(const data::RawData& raw, bool fiml = false, bool uls = false, bool means = false) {
  api::ModelOptions opts;
  opts.build.meanstructure = fiml || means;
  opts.build.fixed_x = false;
  opts.build.n_groups = static_cast<int>(raw.X.size());
  auto model = api::model_from_lavaan("f =~ x1 + x2 + x3 + x4", opts);
  REQUIRE(model);
  auto complete = raw;
  complete.mask.clear();
  auto sample = data::sample_stats_from_raw(complete);
  REQUIRE(sample);
  if (!fiml && !means) sample->mean.clear();
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

TEST_CASE("policy fit measures: covariance-only ML profiles shifted group means") {
  using namespace magmaan;
  for (int groups : {1, 2}) {
    auto raw = rows();
    if (groups == 2) raw.X.push_back(raw.X[0]);
    auto fit = fitted(raw);
    auto shifted = raw;
    for (auto& x : shifted.X) x.array() += 10.0;
    const auto& m = fit.model();
    auto a = api::policy_fit_measures(m.structure(), m.matrix_rep(), raw, fit.estimates(), {});
    auto b = api::policy_fit_measures(m.structure(), m.matrix_rep(), shifted, fit.estimates(), {});
    for (auto name : {"rmsea", "cfi", "tli"}) {
      REQUIRE(index(a,name).reason == api::InferenceReason::Available);
      REQUIRE(index(b,name).reason == api::InferenceReason::Available);
      CHECK(index(a,name).estimate == doctest::Approx(index(b,name).estimate).epsilon(1e-8));
    }
    CHECK(a.user.trace == doctest::Approx(b.user.trace).epsilon(1e-8));
    CHECK(a.baseline.trace == doctest::Approx(b.baseline.trace).epsilon(1e-8));
    CHECK(a.user.corrected == doctest::Approx(b.user.corrected).epsilon(1e-8));
  }
}

TEST_CASE("policy fit measures: continuous affine invariance at transformed fitted points") {
  using namespace magmaan;
  for (int groups : {1, 2}) for (int route : {0, 1, 2}) {
    auto raw = rows(500);
    const bool fiml = route == 2, means = route != 0;
    if (fiml) {
      raw.mask.emplace_back(Eigen::Matrix<std::uint8_t,Eigen::Dynamic,Eigen::Dynamic>::Ones(500,4));
      for (int r = 2; r < 500; r += 7) raw.mask[0](r,1) = 0;
    }
    if (groups == 2) {
      raw.X.push_back(raw.X[0]);
      if (fiml) raw.mask.push_back(raw.mask[0]);
    }
    auto f = fitted(raw, fiml, false, means);
    const auto& m = f.model();
    auto a = api::policy_fit_measures(m.structure(),m.matrix_rep(),raw,f.estimates(),{},fiml);
    for (const auto& scales : {Eigen::Vector4d::Ones().eval(),
         Eigen::Vector4d::Constant(2.0).eval(), Eigen::Vector4d(0.5,2.0,3.0,1.5)}) {
      auto transformed = raw;
      for (auto& x : transformed.X) {
        x *= scales.asDiagonal();
        x.array() += 10.0;
      }
      auto est = f.estimates();
      for (std::size_t r = 0; r < m.structure().free.size(); ++r) {
        if (m.structure().free[r] <= 0) continue;
        const auto& cell = m.matrix_rep().cell_for_row[r];
        auto& value = est.theta(m.structure().free[r]-1);
        if (cell.mat == model::MatId::Theta) value *= scales(cell.row)*scales(cell.col);
        else if (cell.mat == model::MatId::Psi) value *= scales(0)*scales(0);
        else if (cell.mat == model::MatId::Lambda) value *= scales(cell.row)/scales(0);
        else if (cell.mat == model::MatId::Nu) value = scales(cell.row)*value+10.0;
      }
      auto b = api::policy_fit_measures(m.structure(),m.matrix_rep(),transformed,est,{},fiml);
      for (auto name : {"rmsea", "cfi", "tli"}) {
        REQUIRE_MESSAGE(index(b,name).reason == api::InferenceReason::Available,index(b,name).detail);
        CHECK(index(a,name).estimate == doctest::Approx(index(b,name).estimate).epsilon(1e-8));
      }
      CHECK(a.user.trace == doctest::Approx(b.user.trace).epsilon(1e-8));
      CHECK(a.baseline.trace == doctest::Approx(b.baseline.trace).epsilon(1e-8));
      CHECK(a.user.corrected == doctest::Approx(b.user.corrected).epsilon(1e-8));
      CHECK(a.baseline.corrected == doctest::Approx(b.baseline.corrected).epsilon(1e-8));
    }
  }
}

TEST_CASE("fit index Wald intervals: continuous likelihood and point gates") {
  using namespace magmaan;
  for (int groups : {1,2}) for (int route : {0,1,2}) {
    auto raw = rows(500);
    for (int r = 0; r < 500; ++r) {
      const double residual = 1.5*std::sin(r*0.83);
      raw.X[0](r,0) += residual; raw.X[0](r,1) += residual;
    }
    const bool fiml = route == 2, means = route != 0;
    if (fiml) {
      raw.mask.emplace_back(Eigen::Matrix<std::uint8_t,Eigen::Dynamic,Eigen::Dynamic>::Ones(500,4));
      for (int r = 2; r < 500; r += 7) raw.mask[0](r,1) = 0;
    }
    if (groups == 2) {
      raw.X.push_back(raw.X[0]);
      if (fiml) raw.mask.push_back(raw.mask[0]);
    }
    auto f = fitted(raw,fiml,false,means);
    auto out = api::frontier::fit_index_intervals_misspec(f.model().structure(),f.model().matrix_rep(),raw,f.estimates(),{},fiml);
    REQUIRE(out.indices.size() == 6);
    for (const auto& row : out.indices) {
      REQUIRE_MESSAGE(row.reason == api::InferenceReason::Available,row.detail);
      CHECK(row.lower <= row.estimate);
      CHECK(row.upper >= row.estimate);
      CHECK(row.v1 >= 0); CHECK(row.v2 >= 0);
    }
    auto point = api::policy_fit_measures(f.model().structure(),f.model().matrix_rep(),raw,f.estimates(),{},fiml);
    CHECK(out.casewise_discrepancy == doctest::Approx(point.user.discrepancy).epsilon(1e-10));
    CHECK(out.points.user.corrected == doctest::Approx(point.user.corrected).epsilon(1e-12));
    CHECK(out.points.user.corrected == doctest::Approx(std::max(out.casewise_discrepancy-point.user.trace/(500*groups),0.0)).epsilon(1e-10));
    const double width = 1.6448536269514722*std::sqrt(out.indices[0].v1+out.indices[0].v2);
    CHECK(out.indices[0].lower == doctest::Approx(std::sqrt(groups*std::max(point.user.corrected-width,0.0)/point.user.df)).epsilon(1e-12));
    CHECK(out.indices[0].upper == doctest::Approx(std::sqrt(groups*(point.user.corrected+width)/point.user.df)).epsilon(1e-12));
    CHECK(out.indices[0].v2 == doctest::Approx(2*out.quadratic_squared_trace/std::pow(500.0*groups,2)).epsilon(1e-12));
  }
}

TEST_CASE("fit index Wald intervals: exact covariance population and typed reasons") {
  using namespace magmaan;
  auto raw = rows(500);
  auto stats = data::sample_stats_from_raw(raw); REQUIRE(stats);
  Eigen::LLT<Eigen::MatrixXd> sample_llt(stats->S[0]);
  Eigen::Matrix4d sigma = Eigen::Matrix4d::Constant(0.49);
  sigma.diagonal().setConstant(1.0);
  Eigen::LLT<Eigen::Matrix4d> target_llt(sigma);
  raw.X[0].rowwise() -= stats->mean[0].transpose();
  raw.X[0] = raw.X[0]*sample_llt.matrixU().solve(Eigen::Matrix4d::Identity())*target_llt.matrixU();
  for (int route : {0,1,2}) {
    auto f = fitted(raw,route==2,false,route!=0);
    auto out = api::frontier::fit_index_intervals_misspec(f.model().structure(),f.model().matrix_rep(),raw,f.estimates(),{},route==2);
    REQUIRE_MESSAGE(out.indices[0].reason == api::InferenceReason::Available,out.indices[0].detail);
    CHECK(out.indices[0].v1 < 1e-15);
    CHECK(out.indices[0].v2 > 0);
    CHECK(out.indices[1].reason == api::InferenceReason::Inapplicable);
    std::cout << "Wald exact population route " << route << ": tr(M)=" << out.quadratic_trace
      << " Takeuchi=" << out.points.user.trace << " df=" << out.points.user.df << '\n';
  }
  auto uls_fit = fitted(raw,false,true);
  auto unsupported = api::frontier::fit_index_intervals_misspec(uls_fit);
  for (const auto& row : unsupported.indices) CHECK(row.reason == api::InferenceReason::UnsupportedModel);
  auto f = fitted(raw);
  api::PolicyFitState state; state.penalized = true;
  auto penalized = api::frontier::fit_index_intervals_misspec(f.model().structure(),f.model().matrix_rep(),raw,f.estimates(),state);
  for (const auto& row : penalized.indices) CHECK(row.reason == api::InferenceReason::Penalized);
}
