#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "magmaan/estimate/fit.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

Eigen::MatrixXd random_pd(std::mt19937& rng, Eigen::Index p) {
  std::uniform_real_distribution<double> d(-0.5, 0.5);
  Eigen::MatrixXd A(p, p);
  for (Eigen::Index i = 0; i < p; ++i)
    for (Eigen::Index j = 0; j < p; ++j) A(i, j) = d(rng);
  return A * A.transpose() + Eigen::MatrixXd::Identity(p, p) * static_cast<double>(p);
}

}  // namespace

TEST_CASE("baseline_chi2: closed form matches log|diag(S)| − log|S|") {
  std::mt19937 rng(2026);
  Eigen::MatrixXd S = random_pd(rng, 4);
  magmaan::data::SampleStats samp;
  samp.S     = {S};
  samp.n_obs = {200};

  auto bl = magmaan::measures::baseline_chi2(samp);

  // Hand calculation.
  Eigen::LLT<Eigen::MatrixXd> llt(S);
  REQUIRE(llt.info() == Eigen::Success);
  double log_det_S = 0.0;
  for (Eigen::Index i = 0; i < 4; ++i) log_det_S += std::log(llt.matrixL()(i, i));
  log_det_S *= 2.0;
  double log_det_diag = 0.0;
  for (Eigen::Index i = 0; i < 4; ++i) log_det_diag += std::log(S(i, i));
  const double F_expected = log_det_diag - log_det_S;

  CHECK(bl.chi2 == doctest::Approx(200.0 * F_expected).epsilon(1e-12));
  CHECK(bl.df   == 4 * 3 / 2);   // p(p-1)/2 = 6
}

TEST_CASE("fit_measures: CFI, TLI, RMSEA on a known nontrivial fit") {
  // Use the 3F Holzinger fit fixture so we have meaningful T_user and df_u.
  auto fp = magmaan::parse::Parser::parse(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0002_three_factor_hs.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
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
  samp.S = {S}; samp.n_obs = {j["n_obs"].get<std::int64_t>()};

  auto est   = magmaan::test::fit(*pt, *mr, samp).value();
  auto chi2  = magmaan::inference::chi2_stat(samp, est);
  auto df    = magmaan::inference::df_stat(*pt, samp).value();
  auto bl    = magmaan::measures::baseline_chi2(samp);

  auto fm = magmaan::measures::fit_measures(chi2, df, bl, samp);

  // Sanity checks: all measures in expected ranges for a non-trivial CFA fit.
  CHECK(df == 24);
  CHECK(bl.df  == 9 * 8 / 2);    // p(p-1)/2 = 36
  // T_user < T_baseline for a meaningful model.
  CHECK(chi2 < bl.chi2);
  // Lavaan reports for this model:
  //   CFI ≈ 0.9305, TLI ≈ 0.8959, RMSEA ≈ 0.0921
  // Verify our values land in tight neighborhoods.
  CHECK(fm.cfi   == doctest::Approx(0.9305).epsilon(1e-3));
  CHECK(fm.tli   == doctest::Approx(0.8959).epsilon(1e-3));
  CHECK(fm.rmsea == doctest::Approx(0.0921).epsilon(1e-3));
  // RMSEA 90% CI — lavaan reports rmsea.ci.lower ≈ 0.0714185,
  // rmsea.ci.upper ≈ 0.1136780 for this fit (fitMeasures(fit)).
  CHECK(fm.rmsea_ci_lower == doctest::Approx(0.071418490439939).epsilon(1e-4));
  CHECK(fm.rmsea_ci_upper == doctest::Approx(0.11367801681196).epsilon(1e-4));
  CHECK(fm.rmsea_close_h0 == doctest::Approx(0.05));
  CHECK(fm.rmsea_notclose_h0 == doctest::Approx(0.08));
  CHECK(fm.rmsea_pvalue == doctest::Approx(
      1.0 - magmaan::inference::noncentral_chisq_cdf(
          chi2, static_cast<double>(df),
          301.0 * static_cast<double>(df) * 0.05 * 0.05)).epsilon(1e-12));
  CHECK(fm.rmsea_notclose_pvalue == doctest::Approx(
      magmaan::inference::noncentral_chisq_cdf(
          chi2, static_cast<double>(df),
          301.0 * static_cast<double>(df) * 0.08 * 0.08)).epsilon(1e-12));
  // The CI brackets the point estimate.
  CHECK(fm.rmsea_ci_lower <= fm.rmsea);
  CHECK(fm.rmsea          <= fm.rmsea_ci_upper);

  // fit_extras: SRMR + log-likelihood + AIC/BIC. Lavaan reports
  //   srmr ≈ 0.06520, logl ≈ -3737.745, AIC ≈ 7517.490, BIC ≈ 7595.339,
  //   BIC2 ≈ 7528.739, npar = 21.
  auto fx = magmaan::measures::fit_extras(*pt, *mr, samp, est).value();
  CHECK(fx.npar   == 21);
  CHECK(fx.ntotal == 301);
  CHECK(fx.srmr   == doctest::Approx(0.0652050571843865).epsilon(1e-4));
  CHECK(fx.logl   == doctest::Approx(-3737.7449266262).epsilon(1e-5));
  CHECK(fx.unrestricted_logl == doctest::Approx(-3695.09216574121).epsilon(1e-6));
  // Algebraic identities — independent of lavaan.
  CHECK(fx.aic  == doctest::Approx(-2.0 * fx.logl + 2.0 * fx.npar).epsilon(1e-12));
  CHECK(fx.bic  == doctest::Approx(-2.0 * fx.logl + fx.npar * std::log(301.0)).epsilon(1e-12));
  CHECK(fx.bic2 == doctest::Approx(-2.0 * fx.logl + fx.npar * std::log((301.0 + 2.0) / 24.0)).epsilon(1e-12));
  // logl ≡ unrestricted_logl − χ²/2 (cov-only model).
  CHECK(fx.logl == doctest::Approx(fx.unrestricted_logl - chi2 / 2.0).epsilon(1e-9));
}

TEST_CASE("fit_measures: RMSEA CI edge cases (df<1, small χ², G>1 scaling)") {
  using magmaan::measures::BaselineFit;
  using magmaan::measures::FitMeasures;
  using magmaan::data::SampleStats;
  using magmaan::measures::fit_measures;

  auto mk_samp = [](std::initializer_list<std::int64_t> ns) {
    SampleStats s;
    for (auto n : ns) {
      s.S.push_back(Eigen::MatrixXd::Identity(3, 3));   // shape only — size = G
      s.n_obs.push_back(n);
    }
    return s;
  };
  BaselineFit bl{500.0, 36};

  // df_u < 1 (saturated): both bounds 0.
  {
    auto fm = fit_measures(0.0, 0, bl, mk_samp({300}));
    CHECK(fm.tli == doctest::Approx(1.0));
    CHECK(fm.rmsea_ci_lower == 0.0);
    CHECK(fm.rmsea_ci_upper == 0.0);
  }
  // Tiny χ² < df: lower bound 0 (central CDF below 0.95). Upper may also be 0
  // if χ² is below the 5% point.
  {
    auto fm = fit_measures(2.0, 24, bl, mk_samp({300}));
    CHECK(fm.rmsea_ci_lower == 0.0);
    CHECK(fm.rmsea_ci_upper == 0.0);   // χ²=2 is far below qchisq(0.05, 24)
  }
  // χ² well above df: a genuine interval, lower < upper, both > 0.
  {
    auto fm = fit_measures(85.305521769973, 24, bl, mk_samp({301}));
    CHECK(fm.rmsea_ci_lower > 0.0);
    CHECK(fm.rmsea_ci_lower < fm.rmsea_ci_upper);
  }
  // Multi-group: same χ²/df/N split across G blocks scales the bounds by √G
  // (lavaan's convention; the point RMSEA already carries the same factor).
  {
    auto fm1 = fit_measures(85.305521769973, 24, bl, mk_samp({300}));
    auto fm4 = fit_measures(85.305521769973, 24, bl, mk_samp({75, 75, 75, 75}));
    CHECK(fm4.rmsea_ci_lower == doctest::Approx(2.0 * fm1.rmsea_ci_lower).epsilon(1e-12));
    CHECK(fm4.rmsea_ci_upper == doctest::Approx(2.0 * fm1.rmsea_ci_upper).epsilon(1e-12));
    CHECK(fm4.rmsea          == doctest::Approx(2.0 * fm1.rmsea).epsilon(1e-12));
    CHECK(fm4.rmsea_pvalue != doctest::Approx(fm1.rmsea_pvalue));
  }
}

TEST_CASE("robust_fit_measures: lavaan MLM scaled and robust conventions") {
  magmaan::measures::RobustFitMeasureInputs in;
  in.chi2 = 85.30552177;
  in.df = 24;
  in.chi2_scaled = 80.87178349;
  in.scaling_factor = 1.054824;
  in.baseline_chi2 = 918.85158929;
  in.baseline_df = 36;
  in.baseline_chi2_scaled = 789.29750421;
  in.baseline_scaling_factor = 1.164138;
  in.n_total = 301;
  in.n_groups = 1;

  const auto fm = magmaan::measures::robust_fit_measures(in);

  CHECK(fm.chisq_scaled == doctest::Approx(80.87178349));
  CHECK(fm.df_scaled == 24);
  CHECK(fm.baseline_chisq_scaled == doctest::Approx(789.29750421));
  CHECK(fm.baseline_df_scaled == 36);
  CHECK(fm.chisq_scaling_factor == doctest::Approx(1.054824));
  CHECK(fm.baseline_chisq_scaling_factor == doctest::Approx(1.164138));

  CHECK(fm.cfi_scaled == doctest::Approx(0.92450289).epsilon(1e-7));
  CHECK(fm.tli_scaled == doctest::Approx(0.88675434).epsilon(1e-7));
  CHECK(fm.cfi_robust == doctest::Approx(0.93159217).epsilon(1e-7));
  CHECK(fm.tli_robust == doctest::Approx(0.89738826).epsilon(1e-7));

  CHECK(fm.rmsea_scaled == doctest::Approx(0.08872777).epsilon(1e-7));
  CHECK(fm.rmsea_ci_lower_scaled == doctest::Approx(0.06841236).epsilon(1e-7));
  CHECK(fm.rmsea_ci_upper_scaled == doctest::Approx(0.10982939).epsilon(1e-7));
  CHECK(fm.rmsea_pvalue_scaled == doctest::Approx(0.001277219).epsilon(1e-6));
  CHECK(fm.rmsea_notclose_pvalue_scaled ==
        doctest::Approx(0.771981712).epsilon(1e-6));

  CHECK(fm.rmsea_robust == doctest::Approx(0.09112753).epsilon(1e-7));
  CHECK(fm.rmsea_ci_lower_robust == doctest::Approx(0.06971130).epsilon(1e-7));
  CHECK(fm.rmsea_ci_upper_robust == doctest::Approx(0.11339582).epsilon(1e-7));
  CHECK(fm.rmsea_pvalue_robust == doctest::Approx(0.001211059).epsilon(1e-6));
  CHECK(fm.rmsea_notclose_pvalue_robust ==
        doctest::Approx(0.813162430).epsilon(1e-6));
}

TEST_CASE("fit_extras: saturated 1F model — SRMR ≈ 0, logl == unrestricted_logl") {
  // A just-identified one-factor model (3 indicators) — df = 0, Σ̂ = S, so
  // every residual vanishes: SRMR ≈ 0, F_ML ≈ 0, logl ≈ saturated logl.
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) +
                   "/fit/0001_one_factor_cfa.fit.json");
  REQUIRE(in.is_open());
  std::stringstream ss; ss << in.rdbuf();
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
  samp.S = {S}; samp.n_obs = {j["n_obs"].get<std::int64_t>()};

  auto est = magmaan::test::fit(*pt, *mr, samp).value();
  CHECK(std::abs(est.fmin) < 1e-8);                   // F_ML ≈ 0

  auto fx = magmaan::measures::fit_extras(*pt, *mr, samp, est).value();
  CHECK(fx.npar == 6);
  CHECK(std::abs(fx.srmr) < 1e-6);
  CHECK(fx.logl == doctest::Approx(fx.unrestricted_logl).epsilon(1e-9));
  CHECK(fx.aic  == doctest::Approx(-2.0 * fx.logl + 12.0).epsilon(1e-12));
}
