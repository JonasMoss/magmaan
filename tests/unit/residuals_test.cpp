#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include <cmath>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/measures/residuals.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/weighted_inference.hpp"
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

// A fixed 4-indicator population covariance with one extra residual covariance,
// for the estimated-weight frontier residual tests.
Eigen::Matrix4d four_indicator_cov() {
  Eigen::Vector4d lambda;  lambda << 1.0, 0.8, 0.7, 0.9;
  Eigen::Vector4d theta;   theta  << 0.6, 0.7, 0.8, 0.5;
  Eigen::Matrix4d S =
      lambda * lambda.transpose() * 1.4 + theta.asDiagonal().toDenseMatrix();
  S(1, 0) += 0.18;
  S(0, 1) = S(1, 0);
  return S;
}

// Heavy-tailed multivariate-t with covariance ≈ Sigma ⇒ Γ̂ ≠ Γ_NT.
Eigen::MatrixXd mvt_sample(std::mt19937& rng, Eigen::Index n,
                           const Eigen::MatrixXd& Sigma, double df) {
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  const Eigen::MatrixXd L = llt.matrixL();
  const Eigen::Index p = Sigma.rows();
  std::normal_distribution<double> z(0.0, 1.0);
  std::chi_squared_distribution<double> chi(df);
  const double scale = std::sqrt((df - 2.0) / df);
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::VectorXd zi(p);
    for (Eigen::Index j = 0; j < p; ++j) zi(j) = z(rng);
    const double w = chi(rng) / df;
    X.row(i) = (scale * (L * zi) / std::sqrt(w)).transpose();
  }
  return X;
}

struct CfaHandles {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
};

CfaHandles build_4indicator_cfa() {
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);              REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt); REQUIRE(rep.has_value());
  return CfaHandles{std::move(*pt), std::move(*rep)};
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

TEST_CASE("frontier estimated-weight residuals: Fixed mode == hand-built "
          "Q·(Γ̂/n)·Qᵀ projection") {
  // With a fixed weight (no IF(Ŵ)) the estimated-weight residual ACOV must equal
  // the residual-maker Q = I − Δ(Δ'WΔ)⁻¹Δ'W applied to the empirical moment
  // ACOV Γ̂/n: acov_res = Q·(Γ̂/n)·Qᵀ. Re-derive that here from primitives and
  // check the standardized residual SE matches. This anchors the projector and
  // the IJ residual assembly exactly (no mean structure, single group).
  auto h = build_4indicator_cfa();
  std::mt19937 rng(20260623u);
  const Eigen::Matrix4d Sigma = four_indicator_cov();
  magmaan::data::RawData raw;
  raw.X.push_back(mvt_sample(rng, 2000, Sigma, 8.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  auto G = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(G.has_value());
  Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(G->rows(), G->cols());
  for (Eigen::Index k = 0; k < G->rows(); ++k) W_dwls(k, k) = 1.0 / (*G)(k, k);
  magmaan::estimate::gmm::Weight weight{W_dwls};

  auto est = magmaan::test::fit_gmm(h.pt, h.rep, *samp, weight);
  REQUIRE(est.has_value());

  auto ew = magmaan::measures::frontier::standardized_residuals_estimated_weight(
      h.pt, h.rep, *samp, *est, weight, raw,
      magmaan::estimate::ContinuousLsIJWeightMode::Fixed);
  REQUIRE(ew.has_value());
  REQUIRE(ew->cov_se.size() == 1);

  // Independent Q·(Γ̂/n)·Qᵀ in the vech(cov) moment metric.
  auto ev = magmaan::model::ModelEvaluator::build(h.pt, h.rep);
  REQUIRE(ev.has_value());
  auto Delta = ev->dsigma_dtheta(est->theta);
  REQUIRE(Delta.has_value());
  const Eigen::MatrixXd& D = *Delta;            // pstar × q
  const Eigen::MatrixXd A = D.transpose() * W_dwls * D;
  const Eigen::MatrixXd Ainv = A.ldlt().solve(
      Eigen::MatrixXd::Identity(A.rows(), A.cols()));
  const Eigen::Index m = D.rows();
  const Eigen::MatrixXd Q =
      Eigen::MatrixXd::Identity(m, m) - D * Ainv * D.transpose() * W_dwls;
  const double n = static_cast<double>(samp->n_obs[0]);
  const Eigen::MatrixXd acov_res = Q * (*G / n) * Q.transpose();

  const Eigen::Index p = ew->cov_se[0].rows();
  auto vech_idx = [](Eigen::Index pp, Eigen::Index r, Eigen::Index c) {
    return c * pp - (c * (c - 1)) / 2 + (r - c);
  };
  const Eigen::MatrixXd& S = samp->S[0];
  for (Eigen::Index c = 0; c < p; ++c)
    for (Eigen::Index r = c; r < p; ++r) {
      const Eigen::Index row = vech_idx(p, r, c);
      const double se_ref =
          std::sqrt(acov_res(row, row) / (S(r, r) * S(c, c)));
      CHECK(ew->cov_se[0](r, c) ==
            doctest::Approx(se_ref).epsilon(1e-8));
    }

  // The $summary cov SRMR matches the deterministic SRMR; inference is finite.
  REQUIRE(ew->summary.size() == 1);
  CHECK(ew->summary[0].cov.srmr == doctest::Approx(ew->srmr).epsilon(1e-10));
  CHECK(std::isfinite(ew->summary[0].cov.srmr_se));
  CHECK(std::isfinite(ew->summary[0].cov.usrmr));
}

TEST_CASE("frontier estimated-weight residuals: DWLS weight correction shifts "
          "the residual SE under non-normality") {
  // Same DWLS weight, same projection, same empirical Γ̂ meat: the ONLY
  // difference between Fixed and SampleEmpiricalDwls is the data-dependent-weight
  // term C = IF(Ŵ). On heavy-tailed data it must move the residual SE (and so
  // the $summary), the residual-space analogue of the Stage-A MI shift.
  auto h = build_4indicator_cfa();
  std::mt19937 rng(20260623u);
  const Eigen::Matrix4d Sigma = four_indicator_cov();
  magmaan::data::RawData raw;
  raw.X.push_back(mvt_sample(rng, 4000, Sigma, 6.0));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  auto G = magmaan::data::empirical_gamma(raw.X[0]);
  REQUIRE(G.has_value());
  Eigen::MatrixXd W_dwls = Eigen::MatrixXd::Zero(G->rows(), G->cols());
  for (Eigen::Index k = 0; k < G->rows(); ++k) W_dwls(k, k) = 1.0 / (*G)(k, k);
  magmaan::estimate::gmm::Weight weight{W_dwls};

  auto est = magmaan::test::fit_gmm(h.pt, h.rep, *samp, weight);
  REQUIRE(est.has_value());

  auto fixed =
      magmaan::measures::frontier::standardized_residuals_estimated_weight(
          h.pt, h.rep, *samp, *est, weight, raw,
          magmaan::estimate::ContinuousLsIJWeightMode::Fixed);
  REQUIRE(fixed.has_value());
  auto ew = magmaan::measures::frontier::standardized_residuals_estimated_weight(
      h.pt, h.rep, *samp, *est, weight, raw,
      magmaan::estimate::ContinuousLsIJWeightMode::SampleEmpiricalDwls);
  REQUIRE(ew.has_value());

  const Eigen::Index p = ew->cov_se[0].rows();
  // The deterministic residual matrices are identical (same θ̂); only the ACOV
  // (SE/z/summary) carries the weight correction.
  for (Eigen::Index r = 0; r < p; ++r)
    for (Eigen::Index c = 0; c < p; ++c)
      CHECK(ew->cov_cor[0](r, c) ==
            doctest::Approx(fixed->cov_cor[0](r, c)).epsilon(1e-10));

  double max_abs_dse = 0.0;
  for (Eigen::Index c = 0; c < p; ++c)
    for (Eigen::Index r = c; r < p; ++r) {
      CHECK(std::isfinite(ew->cov_se[0](r, c)));
      max_abs_dse = std::max(max_abs_dse,
                             std::abs(ew->cov_se[0](r, c) - fixed->cov_se[0](r, c)));
    }
  CHECK(max_abs_dse > 1e-4);
  // The summary close-fit inference moves too.
  CHECK(std::abs(ew->summary[0].cov.usrmr_se -
                 fixed->summary[0].cov.usrmr_se) > 1e-6);
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
