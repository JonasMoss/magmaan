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
}
