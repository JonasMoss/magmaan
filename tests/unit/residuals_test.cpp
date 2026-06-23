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

  // $summary cov column matches lavResiduals(fit)$summary (type = cor.bentler)
  // for the Holzinger-Swineford CFA, full precision from lavaan 0.7.1.2691.
  REQUIRE(sr->summary.size() == 1);
  const auto& sm = sr->summary[0];
  CHECK(sm.has_mean == false);
  CHECK(sm.cov.srmr == doctest::Approx(0.0652050571844).epsilon(1e-7));
  CHECK(sm.cov.srmr_se == doctest::Approx(0.00573544473881).epsilon(1e-6));
  CHECK(sm.cov.srmr_exactfit_z == doctest::Approx(6.06302113974).epsilon(1e-6));
  CHECK(sm.cov.srmr_exactfit_pvalue ==
        doctest::Approx(6.67940480703e-10).epsilon(1e-4));
  CHECK(sm.cov.usrmr == doctest::Approx(0.058003188781).epsilon(1e-6));
  CHECK(sm.cov.usrmr_se == doctest::Approx(0.00962198180933).epsilon(1e-6));
  CHECK(sm.cov.usrmr_ci_lower == doctest::Approx(0.0421764371034).epsilon(1e-6));
  CHECK(sm.cov.usrmr_ci_upper == doctest::Approx(0.0738299404585).epsilon(1e-6));
  CHECK(sm.cov.usrmr_closefit_h0 == doctest::Approx(0.05));
  CHECK(sm.cov.usrmr_closefit_z == doctest::Approx(0.831760955236).epsilon(1e-6));
  CHECK(sm.cov.usrmr_closefit_pvalue ==
        doctest::Approx(0.202771943379).epsilon(1e-5));

  // The summary srmr is the same scalar the struct already reports.
  CHECK(sm.cov.srmr == doctest::Approx(sr->srmr).epsilon(1e-10));
}

TEST_CASE("standardized_residuals: $summary mean/total match lavaan with means") {
  // Over-identified 3-factor HS fit *with* a (saturated) mean structure. The
  // covariance residuals are unchanged from the cov-only fit, so the `cov`
  // column matches the no-mean case; the `mean` column is degenerate (means are
  // reproduced exactly) and the `total` column combines the two. Targets are
  // lavResiduals(cfa(HS.model, meanstructure = TRUE))$summary, lavaan 0.7.1.2691.
  auto fp = magmaan::parse::Parser::parse(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto pt = magmaan::spec::build(*fp, opts);          REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt);    REQUIRE(mr.has_value());

  auto samp = load_hs_samp();
  Eigen::VectorXd mean(9);
  mean << 4.93576965637874, 6.08803986710963, 2.25041528239203,
          3.06090808684385, 4.34053156146179, 2.18557190176080,
          4.18590206677741, 5.52707641196013, 5.37412329169435;
  samp.mean = {mean};

  auto est = magmaan::test::fit(*pt, *mr, samp).value();
  auto sr = magmaan::measures::standardized_residuals(*pt, *mr, samp, est);
  REQUIRE(sr.has_value());
  REQUIRE(sr->summary.size() == 1);
  const auto& sm = sr->summary[0];
  CHECK(sm.has_mean == true);

  // cov column — identical to the cov-only fit.
  CHECK(sm.cov.srmr == doctest::Approx(0.0652050571844).epsilon(1e-7));
  CHECK(sm.cov.usrmr == doctest::Approx(0.058003188781).epsilon(1e-6));
  CHECK(sm.cov.usrmr_closefit_z ==
        doctest::Approx(0.831760955236).epsilon(1e-6));

  // total column — mean ++ vech(cov), pstar = 9 + 45 = 54.
  CHECK(sm.total.srmr == doctest::Approx(0.0595238011388).epsilon(1e-7));
  CHECK(sm.total.srmr_se == doctest::Approx(0.00523572076795).epsilon(1e-6));
  CHECK(sm.total.srmr_exactfit_z == doctest::Approx(6.06302113974).epsilon(1e-6));
  CHECK(sm.total.usrmr == doctest::Approx(0.0529494248376).epsilon(1e-6));
  CHECK(sm.total.usrmr_se == doctest::Approx(0.00878362747479).epsilon(1e-6));
  CHECK(sm.total.usrmr_ci_lower ==
        doctest::Approx(0.0385016433279).epsilon(1e-6));
  CHECK(sm.total.usrmr_ci_upper ==
        doctest::Approx(0.0673972063473).epsilon(1e-6));
  CHECK(sm.total.usrmr_closefit_z ==
        doctest::Approx(0.335786649206).epsilon(1e-5));
  CHECK(sm.total.usrmr_closefit_pvalue ==
        doctest::Approx(0.368515879307).epsilon(1e-5));

  // mean column — saturated means ⇒ RMS == 0 with non-finite inferential
  // fields (lavaan reports NA), except the close-fit H0 constant.
  CHECK(sm.mean.srmr == doctest::Approx(0.0).epsilon(1e-8));
  CHECK(!std::isfinite(sm.mean.srmr_se));
  CHECK(sm.mean.usrmr_closefit_h0 == doctest::Approx(0.05));
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
