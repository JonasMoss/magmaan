#include <doctest/doctest.h>
#include "../test_fit.hpp"
#include "../oracle.hpp"
#include <random>
#include <Eigen/Cholesky>
#include "magmaan/api/policy.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"
#include "magmaan/robust/restriction.hpp"

namespace {
using namespace magmaan;
struct Model { spec::LatentStructure pt; model::MatrixRep rep; };
Model build(const char* syntax, int groups = 1) {
  auto parsed = parse::Parser::parse(syntax); REQUIRE(parsed);
  spec::BuildOptions options; options.meanstructure = true; options.n_groups = groups;
  auto pt = spec::build(*parsed, options); REQUIRE(pt);
  auto rep = model::build_matrix_rep(*pt); REQUIRE(rep);
  return {*pt, *rep};
}
data::RawData rows(int n, int missing = 0, int groups = 1, bool wrong = false) {
  std::mt19937 rng(137);
  std::normal_distribution<double> z;
  std::uniform_real_distribution<double> u;
  data::RawData raw;
  for (int g = 0; g < groups; ++g) {
    Eigen::MatrixXd X(n, 4);
    for (int i = 0; i < n; ++i) {
      double f = z(rng), r = z(rng);
      for (int j = 0; j < 4; ++j) X(i,j) = (0.7 + 0.1*j)*f + z(rng) + 0.2*j + g;
      if (wrong) { X(i,0) += 0.8*r; X(i,1) += 0.8*r; }
    }
    raw.X.push_back(X);
    if (missing) {
      auto mask = Eigen::Matrix<std::uint8_t,Eigen::Dynamic,Eigen::Dynamic>::Ones(n,4).eval();
      for (int i = 0; i < n; ++i) for (int j = 1; j < 4; ++j) {
        const double probability = missing == 1 ? 0.2 : 1.0/(1.0+std::exp(1.6-0.7*X(i,0)));
        if (u(rng) < probability) mask(i,j) = 0;
      }
      raw.mask.push_back(mask);
    }
  }
  return raw;
}
estimate::Estimates fit(const Model& m, const data::RawData& raw) {
  optim::OptimOptions options; options.max_iter = 4000;
  auto est = test::fit_fiml(m.pt, m.rep, raw, options); REQUIRE(est);
  return *est;
}
void available(const api::PolicyTest& t, int df) {
  INFO(t.detail); REQUIRE(t.reason == api::InferenceReason::Available);
  CHECK(t.df == df); CHECK(t.eigenvalues.size() == df);
  CHECK(t.eigenvalues.allFinite()); CHECK(t.eigenvalues.minCoeff() >= 0);
  CHECK(std::isfinite(t.statistic)); CHECK(std::isfinite(t.p_sb)); CHECK(std::isfinite(t.p_peba4));
}
// Independent normal observed-pattern log density, differentiated casewise.
Eigen::MatrixXd numeric_scores(const Model& m, const data::RawData& raw,
                               const estimate::Estimates& est) {
  auto evaluator = model::ModelEvaluator::build(m.pt, m.rep); REQUIRE(evaluator);
  int total = 0; for (const auto& X : raw.X) total += X.rows();
  auto loglik = [&](const Eigen::VectorXd& theta) {
    auto moments = evaluator->sigma(theta); REQUIRE(moments);
    Eigen::VectorXd result(total); int at = 0;
    for (std::size_t b = 0; b < raw.X.size(); ++b) for (int i = 0; i < raw.X[b].rows(); ++i) {
      std::vector<int> observed;
      for (int j = 0; j < raw.X[b].cols(); ++j)
        if (raw.mask.empty() || raw.mask[b](i,j)) observed.push_back(j);
      Eigen::MatrixXd S(observed.size(), observed.size()); Eigen::VectorXd d(observed.size());
      for (std::size_t j = 0; j < observed.size(); ++j) {
        d(static_cast<Eigen::Index>(j)) = raw.X[b](i,observed[j]) - moments->mu[b](observed[j]);
        for (std::size_t k = 0; k < observed.size(); ++k) S(static_cast<Eigen::Index>(j),static_cast<Eigen::Index>(k)) = moments->sigma[b](observed[j],observed[k]);
      }
      Eigen::LLT<Eigen::MatrixXd> llt(S); REQUIRE(llt.info() == Eigen::Success);
      result(at++) = -0.5*(2*llt.matrixL().toDenseMatrix().diagonal().array().log().sum()+d.dot(llt.solve(d)));
    }
    return result;
  };
  Eigen::MatrixXd scores(total, est.theta.size());
  for (int j = 0; j < est.theta.size(); ++j) {
    double h = 1e-5*std::max(1.0,std::abs(est.theta(j)));
    auto plus = est.theta, minus = est.theta; plus(j)+=h; minus(j)-=h;
    scores.col(j) = (loglik(plus)-loglik(minus))/(2*h);
  }
  return scores;
}
}

TEST_CASE("FIML policy: complete-data reductions and typed reasons") {
  auto raw = rows(1200);
  auto m1 = build("f =~ x1 + x2 + x3 + x4");
  auto m0 = build("f =~ x1 + a*x2 + a*x3 + x4");
  auto e1 = fit(m1,raw), e0 = fit(m0,raw);
  auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack);
  auto d = robust::frontier::prepare_ntml_data(raw,true); REQUIRE(d);
  // The same point isolates inference reduction from optimizer differences.
  auto sample = (*d)->sample;
  auto ml1 = e1, ml0 = e0;
  auto ev1 = model::ModelEvaluator::build(m1.pt,m1.rep); REQUIRE(ev1);
  auto ev0 = model::ModelEvaluator::build(m0.pt,m0.rep); REQUIRE(ev0);
  auto im1 = ev1->sigma(e1.theta); REQUIRE(im1);
  auto im0 = ev0->sigma(e0.theta); REQUIRE(im0);
  auto v1 = estimate::ml_value(sample,*im1); REQUIRE(v1);
  auto v0 = estimate::ml_value(sample,*im0); REQUIRE(v0);
  ml1.fmin = 0.5 * *v1; ml0.fmin = 0.5 * *v0;
  auto f1 = robust::frontier::prepare_ntml_fit(*d,m1.pt,m1.rep,ml1); REQUIRE(f1);
  auto f0 = robust::frontier::prepare_ntml_fit(*d,m0.pt,m0.rep,ml0); REQUIRE(f0);
  auto fp = api::policy_inference_fiml(m1.pt,m1.rep,raw,*pack,e1,{});
  auto mp = api::policy_inference_ml(**f1,{});
  available(fp.score,2); available(fp.lr,2);
  CHECK((fp.covariance-mp.covariance).norm() < 1e-9);
  CHECK(fp.lr.statistic == doctest::Approx(mp.lr.statistic).epsilon(1e-9));
  // Global spectra and score statistics use different finite-sample geometry;
  // their reduction is asymptotic, not an equality at the fitted sample.
  auto fn = api::policy_nested_fiml(m0.pt,m0.rep,e0,{},m1.pt,m1.rep,e1,{},raw,*pack);
  auto mn = api::policy_nested_ml(*f0,{},*f1,{});
  available(fn.score,1); available(fn.lr,1);
  CHECK(fn.lr.statistic == doctest::Approx(mn.lr.statistic).epsilon(1e-8));
  CHECK((fn.lr.eigenvalues-mn.lr.eigenvalues).norm() < 1e-9);
  CHECK((fn.score.eigenvalues-mn.score.eigenvalues).norm() < 1e-8);
  api::PolicyFitState failed; failed.converged = false;
  CHECK(api::policy_inference_fiml(m1.pt,m1.rep,raw,*pack,e1,failed).score.reason == api::InferenceReason::NotConverged);
  failed.penalized = true;
  CHECK(api::policy_inference_fiml(m1.pt,m1.rep,raw,*pack,e1,failed).covariance_reason == api::InferenceReason::Penalized);
  auto reversed = api::policy_nested_fiml(m1.pt,m1.rep,e1,{},m0.pt,m0.rep,e0,{},raw,*pack);
  CHECK(reversed.lr.reason == api::InferenceReason::NotNested);
  auto saturated = build("x1 ~~ x2 + x3 + x4\nx2 ~~ x3 + x4\nx3 ~~ x4");
  auto es = fit(saturated,raw);
  CHECK(api::policy_inference_fiml(saturated.pt,saturated.rep,raw,*pack,es,{}).lr.reason == api::InferenceReason::Saturated);
}

TEST_CASE("ML nested LR: restricted misspecified means use exact casewise scores") {
  for (bool restricted_means : {false, true}) {
    auto raw = rows(1200,0,2);
    // Groups have the same covariance and differ by a common mean shift.
    // Swapping groups about their shared fitted mean preserves the population
    // objective, so equal group loadings hold at its symmetric pseudo-true
    // point while the shared intercept structure is misspecified.
    const std::string means = restricted_means
        ? "\nx1 ~ m1*1\nx2 ~ m2*1\nx3 ~ m3*1\nx4 ~ m4*1" : "";
    auto m1 = build(("f =~ x1 + x2 + x3 + x4" + means).c_str(),2);
    auto m0 = build(("f =~ x1 + a*x2 + x3 + x4" + means).c_str(),2);
    auto e1 = fit(m1,raw), e0 = fit(m0,raw);
    auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack);
    auto direct = api::policy_nested_fiml(m0.pt,m0.rep,e0,{},m1.pt,m1.rep,e1,{},raw,*pack);
    available(direct.lr,1);
    auto mb = estimate::fiml::fiml_score_meat_bread(m1.pt,m1.rep,raw,*pack,e1); REQUIRE(mb);
    auto k1 = estimate::build_eq_constraints(m1.pt); REQUIRE(k1);
    auto k0 = estimate::build_eq_constraints(m0.pt); REQUIRE(k0);
    auto embedding = robust::embed_nested_null(m1.pt,m1.rep,m0.pt,m0.rep,e0.theta,*k1,*k0); REQUIRE(embedding);
    const auto numeric = numeric_scores(m1,raw,e1);
    auto reference = robust::compute_satorra2000_from_sandwich(
        (static_cast<double>(pack->cache.n_total)/2.0)*k1->K().transpose()*mb->hessian*k1->K(),
        k1->K().transpose()*numeric.transpose()*numeric*k1->K(),embedding->restriction.A); REQUIRE(reference);
    CHECK((direct.lr.eigenvalues-reference->eigenvalues).norm() < 1e-7);
    for (auto storage : {robust::frontier::ContributionStorage::Casewise,
                         robust::frontier::ContributionStorage::Tiled}) {
      auto d = robust::frontier::prepare_ntml_data(raw,true,storage); REQUIRE(d);
      auto ml1 = e1, ml0 = e0;
      for (auto pair : {std::pair{&m1,&ml1},std::pair{&m0,&ml0}}) {
        auto ev = model::ModelEvaluator::build(pair.first->pt,pair.first->rep); REQUIRE(ev);
        auto im = ev->sigma(pair.second->theta); REQUIRE(im);
        auto value = estimate::ml_value((*d)->sample,*im); REQUIRE(value);
        pair.second->fmin = 0.5 * *value;
      }
      auto f1 = robust::frontier::prepare_ntml_fit(*d,m1.pt,m1.rep,ml1); REQUIRE(f1);
      auto f0 = robust::frontier::prepare_ntml_fit(*d,m0.pt,m0.rep,ml0); REQUIRE(f0);
      auto result = api::policy_nested_ml(*f0,{},*f1,{});
      available(result.lr,1);
      CAPTURE(restricted_means);
      CHECK((result.lr.eigenvalues-direct.lr.eigenvalues).norm() < 1e-9);
      CHECK((result.lr.eigenvalues-reference->eigenvalues).norm() < 1e-7);
      auto info = robust::frontier::ntml_information(**f1); REQUIRE(info);
      auto expected_reference = robust::compute_satorra2000_from_sandwich(
          k1->K().transpose()* **info * k1->K(),
          k1->K().transpose()*numeric.transpose()*numeric*k1->K(),embedding->restriction.A); REQUIRE(expected_reference);
      auto hypothesis = robust::frontier::prepare_ntml_hypothesis(*f0,*f1); REQUIRE(hypothesis);
      auto expected = robust::frontier::ntml_quadratic(**hypothesis,false); REQUIRE(expected);
      auto spectrum = robust::frontier::ntml_spectrum(**expected); REQUIRE(spectrum);
      CHECK((**spectrum-expected_reference->eigenvalues).norm() < 1e-7);
      auto cached = robust::frontier::ntml_quadratic(**hypothesis,false); REQUIRE(cached);
      CHECK(*cached == *expected);
      if (!restricted_means) {
        // Reconstruct the pre-fix centered-moment meat independently, to
        // pin saturated-mean comparisons to their previous spectrum.
        auto geometry = robust::frontier::ntml_geometry(**f1); REQUIRE(geometry);
        Eigen::MatrixXd wd = (*geometry)->Delta;
        for (const auto& block : (*geometry)->base.blocks) {
          wd.middleRows(block.row_offset,block.pstar) = block.llt_gamma_nt.solve(
              (*geometry)->Delta.middleRows(block.row_offset,block.pstar));
          wd.middleRows(block.mu_off,block.p) = block.llt_M.solve(
              (*geometry)->Delta.middleRows(block.mu_off,block.p));
        }
        auto moments = robust::casewise_contributions(raw,(*d)->sample,true); REQUIRE(moments);
        const Eigen::MatrixXd old_scores = *moments * wd * k1->K();
        auto observed = robust::frontier::ntml_observed_information(**f1); REQUIRE(observed);
        auto old = robust::compute_satorra2000_from_sandwich(
            k1->K().transpose()* **observed * k1->K(),
            old_scores.transpose()*old_scores,embedding->restriction.A); REQUIRE(old);
        CHECK((result.lr.eigenvalues-old->eigenvalues).norm()/old->eigenvalues.norm() <= 1e-12);
      }
    }
  }
}

TEST_CASE("FIML policy: MCAR MAR grouped sandwich and evaluation-point score meat") {
  for (int missing : {1,2}) for (int groups : {1,2}) {
    auto raw = rows(900,missing,groups,true);
    auto m1 = build("f =~ x1 + x2 + x3 + x4",groups);
    auto m0 = build("f =~ x1 + a*x2 + a*x3 + x4",groups);
    auto e1 = fit(m1,raw), e0 = fit(m0,raw);
    auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack);
    auto h1 = estimate::fiml::fiml_h1_moments(raw,*pack); REQUIRE(h1);
    auto p = api::policy_inference_fiml(m1.pt,m1.rep,raw,*pack,e1,{});
    available(p.score,2*groups); available(p.lr,2*groups);
    auto mlr = estimate::fiml::fiml_robust_mlr(m1.pt,m1.rep,raw,e1,p.lr.df,p.lr.statistic,*pack,*h1); REQUIRE(mlr);
    // Transitive lavaan gate: fiml_robust_mlr is checked on frozen MLR fixtures.
    CHECK((p.covariance-mlr->vcov).norm() < 1e-9);
    auto nested = api::policy_nested_fiml(m0.pt,m0.rep,e0,{},m1.pt,m1.rep,e1,{},raw,*pack);
    api::FimlPolicyFit c0(m0.pt,m0.rep,raw,*pack,e0), c1(m1.pt,m1.rep,raw,*pack,e1);
    const auto cached = api::policy_inference_fiml(c1,{});
    const api::PolicyFitState boundary{true,true,false,false};
    const auto flagged = api::policy_inference_fiml(c1,boundary);
    const auto fresh_flagged = api::policy_inference_fiml(m1.pt,m1.rep,raw,*pack,e1,boundary);
    CHECK(flagged.psd_boundary == fresh_flagged.psd_boundary);
    CHECK(flagged.verdict_disagreement == fresh_flagged.verdict_disagreement);
    auto unsupported_pt = m1.pt; unsupported_pt.has_inequality_constraints = true;
    api::FimlPolicyFit unsupported(unsupported_pt,m1.rep,raw,*pack,e1);
    const auto refused = api::policy_inference_fiml(unsupported,boundary);
    const auto fresh_refused = api::policy_inference_fiml(unsupported_pt,m1.rep,raw,*pack,e1,boundary);
    CHECK(refused.covariance_reason == fresh_refused.covariance_reason);
    CHECK(refused.psd_boundary == fresh_refused.psd_boundary);
    CHECK(refused.verdict_disagreement == fresh_refused.verdict_disagreement);
    CHECK(api::policy_ingredient_builds(unsupported) == 0);
    CHECK((cached.covariance.array() == p.covariance.array()).all());
    for (const auto& tests : {std::pair{cached.score,p.score}, std::pair{cached.lr,p.lr}}) {
      CHECK(tests.first.statistic == tests.second.statistic);
      CHECK(tests.first.p_sb == tests.second.p_sb);
      CHECK(tests.first.p_peba4 == tests.second.p_peba4);
      CHECK((tests.first.eigenvalues.array() == tests.second.eigenvalues.array()).all());
    }
    for (int i = 0; i < 2; ++i) {
      const auto got = api::policy_nested_fiml(c0,{},c1,{});
      for (const auto& tests : {std::pair{got.score,nested.score}, std::pair{got.lr,nested.lr}}) {
        CHECK(tests.first.statistic == tests.second.statistic);
        CHECK(tests.first.p_sb == tests.second.p_sb);
        CHECK(tests.first.p_peba4 == tests.second.p_peba4);
        CHECK((tests.first.eigenvalues.array() == tests.second.eigenvalues.array()).all());
      }
      CHECK(api::policy_ingredient_builds(c1) == 1);
    }
    auto elsewhere = raw; elsewhere.X[0](0,0) += 1;
    api::FimlPolicyFit wrong(m0.pt,m0.rep,elsewhere,*pack,e0);
    CHECK(api::policy_nested_fiml(wrong,{},c1,{}).lr.reason == api::InferenceReason::NotNested);

    // The repeated label also ties loadings across groups: four slots
    // collapse to one in the two-group null.
    const int restrictions = 2*groups-1;
    available(nested.score,restrictions); available(nested.lr,restrictions);
    auto mb = estimate::fiml::fiml_score_meat_bread(m1.pt,m1.rep,raw,*pack,e1); REQUIRE(mb);
    const Eigen::MatrixXd analytic = -0.5*mb->scores;
    const auto numeric = numeric_scores(m1,raw,e1);
    CHECK((analytic-numeric).norm()/numeric.norm() < 1e-7);
    CHECK((analytic.transpose()*analytic-numeric.transpose()*numeric).norm()/(numeric.transpose()*numeric).norm() < 1e-7);
    auto k1 = estimate::build_eq_constraints(m1.pt); REQUIRE(k1);
    auto k0 = estimate::build_eq_constraints(m0.pt); REQUIRE(k0);
    auto embedding = robust::embed_nested_null(m1.pt,m1.rep,m0.pt,m0.rep,e0.theta,*k1,*k0); REQUIRE(embedding);
    auto score_components = inference::frontier::nested_score_components(m1.pt,m1.rep,
        m0.pt,m0.rep,nullptr,raw,&*pack,e0,
        inference::frontier::ScoreSensitivity::ObservedInformation); REQUIRE(score_components);
    auto embedded_est = e0; embedded_est.theta = embedding->theta;
    const auto numeric_null = numeric_scores(m1,raw,embedded_est);
    CHECK((score_components->rows-numeric_null).norm()/numeric_null.norm() < 1e-7);
    auto reference = robust::compute_satorra2000_from_sandwich(
        (static_cast<double>(pack->cache.n_total)/2.0)*k1->K().transpose()*mb->hessian*k1->K(),
        k1->K().transpose()*numeric.transpose()*numeric*k1->K(),embedding->restriction.A); REQUIRE(reference);
    CHECK((nested.lr.eigenvalues-reference->eigenvalues).norm() < 1e-7);
  }
}

TEST_CASE("FIML policy: direct and transported nested spectra agree under large-N correct specification") {
  auto raw = rows(20000,1);
  auto m1 = build("f =~ x1 + x2 + x3 + x4");
  auto m0 = build("f =~ x1 + a*x2 + a*x3 + x4");
  auto e1 = fit(m1,raw), e0 = fit(m0,raw);
  auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack);
  auto direct = api::policy_nested_fiml(m0.pt,m0.rep,e0,{},m1.pt,m1.rep,e1,{},raw,*pack);
  available(direct.lr,1);
  auto k1 = estimate::build_eq_constraints(m1.pt); REQUIRE(k1);
  auto k0 = estimate::build_eq_constraints(m0.pt); REQUIRE(k0);
  auto transported = robust::lr_test_satorra2000_fiml_from_data(m1.pt,m1.rep,e1.theta,*k1,
      m0.pt,m0.rep,e0.theta,*k0,raw,2.0*static_cast<double>(pack->cache.n_total)*e0.fmin,2.0*static_cast<double>(pack->cache.n_total)*e1.fmin,3,2);
  REQUIRE(transported);
  // O(N^-1/2) agreement, not a finite-sample identity. At N=20000, 5%
  // accommodates fitted-point versus H1 transport estimation variation.
  CHECK((direct.lr.eigenvalues-transported->eigenvalues).norm() < 0.05);
}


TEST_CASE("FIML policy: frozen MCAR MAR and lavaan MLR covariance gates") {
  const auto fixtures = test::load_json_fixture("fitting/lavaan_fiml_0_7_2.json");
  REQUIRE_FALSE(fixtures.is_discarded());
  // First four cases are MCAR/MAR, each with one and two groups.
  for (std::size_t at = 0; at < 4; ++at) {
    const auto& c = fixtures["cases"][at];
    CAPTURE(c["mechanism"]); CAPTURE(c["groups"]);
    const auto syntax = c["model"].get<std::string>();
    auto m = build(syntax.c_str(),c["groups"].get<int>());
    data::RawData raw;
    for (const auto& block : c["raw"]) {
      Eigen::MatrixXd X(static_cast<Eigen::Index>(block.size()),4);
      Eigen::Matrix<std::uint8_t,Eigen::Dynamic,Eigen::Dynamic> mask(X.rows(),4);
      for (Eigen::Index i = 0; i < X.rows(); ++i) for (int j = 0; j < 4; ++j) {
        const auto& value = block[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        mask(i,j) = value.is_null() ? 0 : 1;
        X(i,j) = value.is_null() ? 0.0 : value.get<double>();
      }
      raw.X.push_back(std::move(X)); raw.mask.push_back(std::move(mask));
    }
    auto e = fit(m,raw);
    auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack);
    auto h1 = estimate::fiml::fiml_h1_moments(raw,*pack); REQUIRE(h1);
    auto p = api::policy_inference_fiml(m.pt,m.rep,raw,*pack,e,{});
    auto mlr = estimate::fiml::fiml_robust_mlr(m.pt,m.rep,raw,e,p.lr.df,p.lr.statistic,*pack,*h1); REQUIRE(mlr);
    REQUIRE(p.covariance_reason == api::InferenceReason::Available);
    CHECK((p.covariance-mlr->vcov).norm() < 1e-9);
  }
  const auto oracle = test::load_json_fixture("fiml/0001_one_factor_hs_fiml.fit.json");
  REQUIRE_FALSE(oracle.is_discarded());
  const auto syntax = oracle["input"].get<std::string>();
  auto m = build(syntax.c_str());
  auto raw = test::raw_from_fixture(oracle);
  auto e = fit(m,raw);
  auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack);
  auto p = api::policy_inference_fiml(m.pt,m.rep,raw,*pack,e,{});
  REQUIRE(p.covariance_reason == api::InferenceReason::Available);
  const auto oracle_se = test::vector_from_json(oracle["se_robust_huberwhite"]);
  // Existing FIML golden tolerance, unchanged: optimizer and Hessian variation.
  CHECK((p.covariance.diagonal().array().sqrt().matrix()-oracle_se).cwiseAbs().maxCoeff() < 3e-4);
}
