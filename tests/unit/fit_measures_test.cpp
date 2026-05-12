#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "latva/fit/fit.hpp"
#include "latva/fit/fit_measures.hpp"
#include "latva/fit/inference.hpp"
#include "latva/fit/sample_stats.hpp"
#include "latva/model/matrix_rep.hpp"
#include "latva/parse/parser.hpp"
#include "latva/partable/lavaanify.hpp"

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
  latva::fit::SampleStats samp;
  samp.S     = {S};
  samp.n_obs = {200};

  auto bl = latva::fit::baseline_chi2(samp);

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
  auto fp = latva::parse::Parser::parse(
      "visual =~ x1 + x2 + x3\n"
      "textual =~ x4 + x5 + x6\n"
      "speed =~ x7 + x8 + x9");
  REQUIRE(fp.has_value());
  auto pt = latva::partable::lavaanify(*fp);  REQUIRE(pt.has_value());
  auto mr = latva::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  std::ifstream in(std::string(LATVA_FIXTURES_DIR) +
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
  latva::fit::SampleStats samp;
  samp.S = {S}; samp.n_obs = {j["n_obs"].get<std::int64_t>()};

  auto est = latva::fit::fit(*pt, *mr, samp).value();
  auto inf = latva::fit::ExpectedInfoSE{}.compute(*pt, *mr, samp, est).value();
  auto bl  = latva::fit::baseline_chi2(samp);

  auto fm = latva::fit::fit_measures(inf, bl, samp);

  // Sanity checks: all measures in expected ranges for a non-trivial CFA fit.
  CHECK(inf.df == 24);
  CHECK(bl.df  == 9 * 8 / 2);    // p(p-1)/2 = 36
  // T_user < T_baseline for a meaningful model.
  CHECK(inf.chi2 < bl.chi2);
  // Lavaan reports for this model:
  //   CFI ≈ 0.9305, TLI ≈ 0.8959, RMSEA ≈ 0.0921
  // Verify our values land in tight neighborhoods.
  CHECK(fm.cfi   == doctest::Approx(0.9305).epsilon(1e-3));
  CHECK(fm.tli   == doctest::Approx(0.8959).epsilon(1e-3));
  CHECK(fm.rmsea == doctest::Approx(0.0921).epsilon(1e-3));

  // fit_extras: SRMR + log-likelihood + AIC/BIC. Lavaan reports
  //   srmr ≈ 0.06520, logl ≈ -3737.745, AIC ≈ 7517.490, BIC ≈ 7595.339,
  //   BIC2 ≈ 7528.739, npar = 21.
  auto fx = latva::fit::fit_extras(*pt, *mr, samp, est).value();
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
  CHECK(fx.logl == doctest::Approx(fx.unrestricted_logl - inf.chi2 / 2.0).epsilon(1e-9));
}

TEST_CASE("fit_extras: saturated 1F model — SRMR ≈ 0, logl == unrestricted_logl") {
  // A just-identified one-factor model (3 indicators) — df = 0, Σ̂ = S, so
  // every residual vanishes: SRMR ≈ 0, F_ML ≈ 0, logl ≈ saturated logl.
  auto fp = latva::parse::Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = latva::partable::lavaanify(*fp);  REQUIRE(pt.has_value());
  auto mr = latva::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  std::ifstream in(std::string(LATVA_FIXTURES_DIR) +
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
  latva::fit::SampleStats samp;
  samp.S = {S}; samp.n_obs = {j["n_obs"].get<std::int64_t>()};

  auto est = latva::fit::fit(*pt, *mr, samp).value();
  CHECK(std::abs(est.fmin) < 1e-8);                   // F_ML ≈ 0

  auto fx = latva::fit::fit_extras(*pt, *mr, samp, est).value();
  CHECK(fx.npar == 6);
  CHECK(std::abs(fx.srmr) < 1e-6);
  CHECK(fx.logl == doctest::Approx(fx.unrestricted_logl).epsilon(1e-9));
  CHECK(fx.aic  == doctest::Approx(-2.0 * fx.logl + 12.0).epsilon(1e-12));
}
