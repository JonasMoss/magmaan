#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include <cmath>

#include "magmaan/estimate/fit.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/measures/residuals.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

// Load the 3-factor Holzinger-Swineford sample covariance fixture.
magmaan::data::SampleStats load_hs_samp() {
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0002_three_factor_hs.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss;
  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());
  const auto& M = j["sample_cov"][0]["matrix"];
  const Eigen::Index p = static_cast<Eigen::Index>(M.size());
  Eigen::MatrixXd S(p, p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      S(r, c) = M[static_cast<std::size_t>(r)]
                 [static_cast<std::size_t>(c)].get<double>();
  magmaan::data::SampleStats samp;
  samp.S = {S};
  samp.n_obs = {j["n_obs"].get<std::int64_t>()};
  return samp;
}

}  // namespace

TEST_CASE("residuals: cov[0] equals S − Σ̂ recomputed independently") {
  auto fp = magmaan::parse::Parser::parse(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);          REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  auto samp = load_hs_samp();
  auto est = magmaan::test::fit(*pt, *mr, samp).value();

  auto res = magmaan::measures::residuals(*pt, *mr, samp, est);
  REQUIRE(res.has_value());
  REQUIRE(res->cov.size() == 1);
  REQUIRE(res->mean.size() == 1);

  // Independently recompute Σ̂(θ̂) and S − Σ̂.
  auto ev = magmaan::model::ModelEvaluator::build(*pt, *mr);
  REQUIRE(ev.has_value());
  auto sm = ev->sigma(est.theta);
  REQUIRE(sm.has_value());
  const Eigen::MatrixXd& S = samp.S[0];
  const Eigen::MatrixXd expected = S - sm->sigma[0];

  const Eigen::Index p = S.rows();
  REQUIRE(res->cov[0].rows() == p);
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      CHECK(res->cov[0](r, c) == doctest::Approx(expected(r, c)).epsilon(1e-10));

  // The raw covariance residual is symmetric.
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = r + 1; c < p; ++c)
      CHECK(res->cov[0](r, c) == doctest::Approx(res->cov[0](c, r)).epsilon(1e-12));

  // A covariance-only CFA has no mean structure ⇒ the mean residual is empty.
  CHECK(res->mean[0].size() == 0);
}

TEST_CASE("standardized_residuals: correlation metric and SRMR are consistent") {
  auto fp = magmaan::parse::Parser::parse(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);          REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  auto samp = load_hs_samp();
  auto est = magmaan::test::fit(*pt, *mr, samp).value();

  auto sr = magmaan::measures::standardized_residuals(*pt, *mr, samp, est);
  REQUIRE(sr.has_value());
  REQUIRE(sr->cov_raw.size() == 1);
  REQUIRE(sr->cov_cor.size() == 1);
  REQUIRE(sr->cov_se.size() == 1);
  REQUIRE(sr->cov_z.size() == 1);

  // cov_raw matches the standalone residuals() accessor.
  auto raw = magmaan::measures::residuals(*pt, *mr, samp, est);
  REQUIRE(raw.has_value());
  const Eigen::Index p = sr->cov_raw[0].rows();
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      CHECK(sr->cov_raw[0](r, c) ==
            doctest::Approx(raw->cov[0](r, c)).epsilon(1e-12));

  // cov_cor is the Bentler-standardized residual and is symmetric.
  const Eigen::MatrixXd& S = samp.S[0];
  for (Eigen::Index c = 0; c < p; ++c) {
    CHECK(sr->cov_cor[0](c, c) ==
          doctest::Approx(sr->cov_raw[0](c, c) / S(c, c)).epsilon(1e-12));
    for (Eigen::Index r = c + 1; r < p; ++r) {
      const double expect = sr->cov_raw[0](r, c) / std::sqrt(S(r, r) * S(c, c));
      CHECK(sr->cov_cor[0](r, c) == doctest::Approx(expect).epsilon(1e-12));
      CHECK(sr->cov_cor[0](c, r) == doctest::Approx(expect).epsilon(1e-12));
    }
  }

  // srmr is the single fit_extras definition — not re-derived.
  auto fx = magmaan::measures::fit_extras(*pt, *mr, samp, est);
  REQUIRE(fx.has_value());
  CHECK(sr->srmr == doctest::Approx(fx->srmr).epsilon(1e-12));

  // cov_z matches lavaan's default lavResiduals(type = "cor.bentler")
  // standardized residual z-statistics for the Holzinger-Swineford CFA.
  CHECK(sr->cov_z[0](1, 0) == doctest::Approx(-1.99605930).epsilon(1e-4));
  CHECK(sr->cov_z[0](2, 1) == doctest::Approx(2.68898258).epsilon(1e-4));
  CHECK(sr->cov_z[0](6, 0) == doctest::Approx(-3.77308971).epsilon(1e-4));
  CHECK(sr->cov_z[0](7, 6) == doctest::Approx(4.82307041).epsilon(1e-4));
  CHECK(sr->cov_z[0](8, 7) == doctest::Approx(-4.13205676).epsilon(1e-4));
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = r + 1; c < p; ++c)
      CHECK(sr->cov_z[0](r, c) ==
            doctest::Approx(sr->cov_z[0](c, r)).epsilon(1e-12));

  // SRMR is the RMS of the lower-triangle correlation residuals (no mean
  // structure ⇒ pstar = p(p+1)/2).
  double sum_sq = 0.0;
  for (Eigen::Index c = 0; c < p; ++c)
    for (Eigen::Index r = c; r < p; ++r)
      sum_sq += sr->cov_cor[0](r, c) * sr->cov_cor[0](r, c);
  const double pstar = static_cast<double>(p) * static_cast<double>(p + 1) / 2.0;
  CHECK(sr->srmr == doctest::Approx(std::sqrt(sum_sq / pstar)).epsilon(1e-10));

  // No mean structure ⇒ mean residuals are empty.
  CHECK(sr->mean_raw[0].size() == 0);
  CHECK(sr->mean_cor[0].size() == 0);
  CHECK(sr->mean_se[0].size() == 0);
  CHECK(sr->mean_z[0].size() == 0);
}

TEST_CASE("residuals: rejects a θ of the wrong size") {
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);          REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  auto samp = load_hs_samp();
  // Trim the 9×9 fixture down to the first 3 indicators.
  Eigen::MatrixXd S3 = samp.S[0].topLeftCorner(3, 3);
  samp.S = {S3};

  magmaan::estimate::Estimates bad;
  bad.theta = Eigen::VectorXd::Zero(2);  // model has > 2 free params
  auto res = magmaan::measures::residuals(*pt, *mr, samp, bad);
  CHECK(!res.has_value());
}
