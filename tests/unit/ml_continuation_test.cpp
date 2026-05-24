#include <doctest/doctest.h>

#include <cmath>
#include <string_view>
#include <utility>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/ml_continuation.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

struct BuiltModel {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

BuiltModel build_model(std::string_view syntax) {
  auto flat = magmaan::parse::Parser::parse(syntax);
  REQUIRE(flat.has_value());
  auto pt = magmaan::spec::build(*flat);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return BuiltModel{std::move(*pt), std::move(*rep)};
}

magmaan::data::SampleStats near_singular_cfa_stats() {
  Eigen::Vector4d lambda;
  lambda << 1.0, 0.99, 1.01, 0.98;
  const Eigen::Vector4d theta =
      Eigen::Vector4d::Constant(0.035);

  magmaan::data::SampleStats samp;
  Eigen::Matrix4d S = lambda * lambda.transpose();
  S.diagonal() += theta;
  samp.S.push_back(std::move(S));
  samp.n_obs.push_back(80);
  return samp;
}

}  // namespace

TEST_CASE("ml ridge continuation reaches the ordinary ML endpoint") {
  auto model = build_model("f =~ x1 + x2 + x3 + x4");
  auto samp = near_singular_cfa_stats();
  auto x0 = magmaan::estimate::simple_start_values(model.pt, model.rep, samp,
                                                   {});
  REQUIRE(x0.has_value());

  magmaan::estimate::frontier::MlRidgeContinuationOptions cont;
  cont.alphas = {0.4, 0.1};

  magmaan::optim::OptimOptions opt;
  opt.max_iter = 400;
  auto fit = magmaan::estimate::frontier::fit_ml_ridge_continuation(
      model.pt, model.rep, samp, *x0, {}, magmaan::estimate::Backend::NloptLbfgs,
      opt, cont);
  REQUIRE_MESSAGE(fit.has_value(),
                  "continuation failed: "
                      << (fit.has_value() ? "" : fit.error().detail));

  REQUIRE(fit->steps.size() == 3);
  CHECK(fit->steps[0].alpha == doctest::Approx(0.4));
  CHECK(fit->steps[1].alpha == doctest::Approx(0.1));
  CHECK(fit->steps[2].alpha == doctest::Approx(0.0));
  CHECK(fit->steps[0].sample_condition < fit->steps[2].sample_condition);
  CHECK(std::isfinite(fit->final.fmin));
  CHECK(fit->final.fmin < 1e-6);
  CHECK(fit->total_iterations >= fit->final.iterations);
  CHECK(fit->final_sample_stats.S[0].isApprox(samp.S[0], 1e-14));
}

TEST_CASE("ml ridge continuation rejects invalid path values") {
  auto model = build_model("f =~ x1 + x2 + x3 + x4");
  auto samp = near_singular_cfa_stats();
  auto x0 = magmaan::estimate::simple_start_values(model.pt, model.rep, samp,
                                                   {});
  REQUIRE(x0.has_value());

  magmaan::estimate::frontier::MlRidgeContinuationOptions cont;
  cont.alphas = {0.5, -0.1};

  auto fit = magmaan::estimate::frontier::fit_ml_ridge_continuation(
      model.pt, model.rep, samp, *x0, {}, magmaan::estimate::Backend::NloptLbfgs,
      {}, cont);
  CHECK_FALSE(fit.has_value());
}
