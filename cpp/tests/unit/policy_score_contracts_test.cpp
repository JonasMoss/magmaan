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
Model build(const char* syntax, int groups = 1, std::vector<spec::GroupEqual> equal = {}) {
  auto parsed = parse::Parser::parse(syntax); REQUIRE(parsed);
  spec::BuildOptions options; options.meanstructure = true; options.n_groups = groups; options.group_equal = std::move(equal);
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

// Raw score second moments and observed bread are checked at the same point;
// centering must never change the observed score used by the policy.
TEST_CASE("policy score contracts: stationarity raw meat and projection") {
  for (int missing : {0,1,2}) for (int groups : {1,2}) {
    CAPTURE(missing); CAPTURE(groups);
    auto raw = rows(900,missing,groups,true);
    // Include the most sparse nonempty MAR pattern explicitly.
    if (missing == 2) for (int j=1;j<4;++j) raw.mask[0](0,j)=0;
    auto alt = build("f =~ x1 + x2 + x3 + x4",groups);
    auto null = build("f =~ x1 + a*x2 + a*x3 + x4",groups);
    auto ea = fit(alt,raw), en = fit(null,raw);
    auto pack = estimate::fiml::fiml_pack(raw); REQUIRE(pack);
    for (const auto& pair : {std::pair{&alt,&ea},std::pair{&null,&en}}) {
      const auto& m=*pair.first; const auto& e=*pair.second;
      auto mb=estimate::fiml::fiml_score_meat_bread(m.pt,m.rep,raw,*pack,e); REQUIRE(mb);
      auto eq=estimate::build_eq_constraints(m.pt); REQUIRE(eq);
      const Eigen::MatrixXd scores=-0.5*mb->scores;
      const auto numeric=numeric_scores(m,raw,e);
      CHECK((scores-numeric).norm()/numeric.norm()<1e-7);
      const double n=static_cast<double>(pack->cache.n_total);
      // Terminal absolute deviance-gradient tolerance is 1e-3; log scores
      // have half that scale. K maps free coordinates into full theta.
      CHECK((scores.colwise().sum()*eq->K()/n).cwiseAbs().maxCoeff() < 0.5e-3);
      const Eigen::MatrixXd H=(n/2.0)*eq->K().transpose()*mb->hessian*eq->K();
      // Differentiate summed casewise scores to check observed bread,
      // including the structural second-derivative contribution.
      Eigen::MatrixXd numeric_H(e.theta.size(),e.theta.size());
      for (int j=0;j<e.theta.size();++j) {
        const double h=1e-5*std::max(1.0,std::abs(e.theta(j)));
        auto plus=e,minus=e; plus.theta(j)+=h; minus.theta(j)-=h;
        auto sp=estimate::fiml::fiml_casewise_deviance_scores(m.pt,m.rep,raw,*pack,plus);
        auto sm=estimate::fiml::fiml_casewise_deviance_scores(m.pt,m.rep,raw,*pack,minus);
        REQUIRE(sp); REQUIRE(sm);
        numeric_H.col(j)=0.5*(sp->colwise().sum()-sm->colwise().sum()).transpose()/(2*h);
      }
      CHECK((H-eq->K().transpose()*numeric_H*eq->K()).norm()/H.norm()<1e-7);
      const Eigen::MatrixXd B=eq->K().transpose()*scores.transpose()*scores*eq->K();
      Eigen::LDLT<Eigen::MatrixXd> ldlt(H); REQUIRE(ldlt.info()==Eigen::Success);
      const Eigen::MatrixXd inverse=ldlt.solve(Eigen::MatrixXd::Identity(H.rows(),H.cols()));
      const Eigen::MatrixXd expected=eq->K()*inverse*B*inverse*eq->K().transpose();
      auto policy=api::policy_inference_fiml(m.pt,m.rep,raw,*pack,e,{});
      REQUIRE(policy.covariance_reason==api::InferenceReason::Available);
      CHECK((policy.covariance-expected).norm()/expected.norm()<1e-9);
    }
    auto components=inference::frontier::nested_score_components(alt.pt,alt.rep,null.pt,null.rep,
      nullptr,raw,&*pack,en,inference::frontier::ScoreSensitivity::ObservedInformation);
    REQUIRE(components);
    auto k1=estimate::build_eq_constraints(alt.pt); REQUIRE(k1);
    auto k0=estimate::build_eq_constraints(null.pt); REQUIRE(k0);
    auto embedding=robust::embed_nested_null(alt.pt,alt.rep,null.pt,null.rep,en.theta,*k1,*k0); REQUIRE(embedding);
    auto embedded_est=en; embedded_est.theta=embedding->theta;
    const auto direct=numeric_scores(alt,raw,embedded_est);
    CHECK((components->rows-direct).norm()/direct.norm()<1e-7);
    auto projected=inference::frontier::project_scores(*components,true,false); REQUIRE(projected);
    CHECK((projected->projection.transpose()*components->sensitivity*components->nuisance).norm()<1e-8);
    CHECK((projected->rows-direct*projected->projection).norm()/projected->rows.norm()<1e-7);
    CHECK(projected->rows.allFinite());
    CHECK((projected->score-projected->projection.transpose()*components->score).norm()<1e-10);
    CHECK((projected->rows-components->rows*projected->projection).norm()<1e-10);
    CHECK((projected->meat-projected->rows.transpose()*projected->rows).norm()<1e-9);
    auto centered=inference::frontier::project_scores(*components,true,true); REQUIRE(centered);
    CHECK((centered->score-projected->score).norm()==0.0);
    // At the embedded null the nonzero score makes centering observable.
    CHECK((centered->meat-projected->meat).norm()>1e-6);
    inference::frontier::ScoreGeometryOptions geometry;
    geometry.sensitivity=inference::frontier::ScoreSensitivity::ObservedInformation;
    auto global=inference::frontier::global_score_components(alt.pt,alt.rep,raw,*pack,ea,geometry); REQUIRE(global);
    auto gp=inference::frontier::project_scores(*global,true,false); REQUIRE(gp);
    auto spectrum=inference::frontier::score_spectrum(*gp); REQUIRE(spectrum);
    auto policy=api::policy_inference_fiml(alt.pt,alt.rep,raw,*pack,ea,{});
    CHECK(gp->statistic==doctest::Approx(policy.score.statistic).epsilon(1e-10));
    CHECK((*spectrum-policy.score.eigenvalues).norm()<1e-9);
    CHECK((gp->meat-gp->rows.transpose()*gp->rows).norm()<1e-9);
    auto saturated=build("x1 ~~ x2 + x3 + x4\nx2 ~~ x3 + x4\nx3 ~~ x4",groups);
    auto es=fit(saturated,raw);
    CHECK(api::policy_inference_fiml(saturated.pt,saturated.rep,raw,*pack,es,{}).score.reason
      ==api::InferenceReason::Saturated);
  }
}

TEST_CASE("policy score contracts: two-group complete-data ML reduction") {
  auto raw=rows(900,0,2,true);
  auto alt=build("f =~ x1 + x2 + x3 + x4",2);
  auto null=build("f =~ x1 + a*x2 + a*x3 + x4",2);
  auto ea=fit(alt,raw), en=fit(null,raw);
  auto pack=estimate::fiml::fiml_pack(raw); REQUIRE(pack);
  auto data=robust::frontier::prepare_ntml_data(raw,true); REQUIRE(data);
  auto prepare=[&](const Model& m, estimate::Estimates e) {
    auto ev=model::ModelEvaluator::build(m.pt,m.rep); REQUIRE(ev);
    auto im=ev->sigma(e.theta); REQUIRE(im);
    auto value=estimate::ml_value((*data)->sample,*im); REQUIRE(value);
    e.fmin=0.5* *value;
    auto f=robust::frontier::prepare_ntml_fit(*data,m.pt,m.rep,e); REQUIRE(f);
    return *f;
  };
  auto fa=prepare(alt,ea), fn=prepare(null,en);
  auto f=api::policy_inference_fiml(alt.pt,alt.rep,raw,*pack,ea,{});
  auto m=api::policy_inference_ml(*fa,{});
  CHECK((f.covariance-m.covariance).norm()<1e-9);
  auto nf=api::policy_nested_fiml(null.pt,null.rep,en,{},alt.pt,alt.rep,ea,{},raw,*pack);
  auto nm=api::policy_nested_ml(fn,{},fa,{});
  available(nf.score,3); available(nm.score,3);
  CHECK(nf.score.statistic==doctest::Approx(nm.score.statistic).epsilon(1e-8));
  CHECK((nf.score.eigenvalues-nm.score.eigenvalues).norm()<1e-8);
  auto direct=inference::frontier::nested_score_components(alt.pt,alt.rep,null.pt,null.rep,
    &(*data)->sample,raw,nullptr,en,inference::frontier::ScoreSensitivity::ObservedInformation);
  REQUIRE(direct);
  auto projected=inference::frontier::project_scores(*direct,true,false); REQUIRE(projected);
  CHECK(projected->statistic==doctest::Approx(nm.score.statistic).epsilon(1e-8));
}

TEST_CASE("policy score contracts: equivalent constraints at a common null point") {
  // group.equal is a cross-group operation; in one group compare labels
  // and == rows. In two groups all three spellings tie the same loadings.
  for (int groups : {1,2}) for (int missing : {0,2}) {
    CAPTURE(groups); CAPTURE(missing);
    auto raw=rows(900,missing,groups,true);
    auto pack=estimate::fiml::fiml_pack(raw); REQUIRE(pack);
    auto alt=build("f =~ x1 + x2 + x3 + x4",groups);
    std::vector<Model> nulls;
    if (groups==1) {
      nulls.push_back(build("f =~ x1 + a*x2 + a*x3 + x4"));
      nulls.push_back(build("f =~ x1 + a*x2 + b*x3 + x4\na == b"));
    } else {
      nulls.push_back(build("f =~ x1 + a*x2 + b*x3 + c*x4",2));
      nulls.push_back(build("f =~ x1 + c(a,d)*x2 + c(b,e)*x3 + c(c,h)*x4\na == d\nb == e\nc == h",2));
      nulls.push_back(build("f =~ x1 + x2 + x3 + x4",2,{spec::GroupEqual::Loadings}));
    }
    auto en=fit(nulls[0],raw), ea=fit(alt,raw);
    auto k0=estimate::build_eq_constraints(nulls[0].pt); REQUIRE(k0);
    auto ka=estimate::build_eq_constraints(alt.pt); REQUIRE(ka);
    auto base_embedding=robust::embed_nested_null(alt.pt,alt.rep,nulls[0].pt,nulls[0].rep,en.theta,*ka,*k0);
    REQUIRE(base_embedding);
    auto data=robust::frontier::prepare_ntml_data(raw,true);
    if (missing==0) REQUIRE(data);
    auto prepare_ml=[&](const Model& m, estimate::Estimates e) {
      auto ev=model::ModelEvaluator::build(m.pt,m.rep); REQUIRE(ev);
      auto im=ev->sigma(e.theta); REQUIRE(im);
      auto v=estimate::ml_value((*data)->sample,*im); REQUIRE(v);
      e.fmin=0.5* *v;
      auto f=robust::frontier::prepare_ntml_fit(*data,m.pt,m.rep,e); REQUIRE(f); return *f;
    };
    auto reference=api::policy_inference_fiml(nulls[0].pt,nulls[0].rep,raw,*pack,en,{});
    auto reference_nested=api::policy_nested_fiml(nulls[0].pt,nulls[0].rep,en,{},alt.pt,alt.rep,ea,{},raw,*pack);
    auto covariance_at_alt=[&](const Model& m,const estimate::Estimates& e,const Eigen::MatrixXd& covariance) {
      // Different spellings can change theta dimension. Differentiate the
      // embedding into the common larger-model coordinates to compare V.
      auto k=estimate::build_eq_constraints(m.pt); REQUIRE(k);
      Eigen::MatrixXd J(ea.theta.size(),e.theta.size());
      for (int j=0;j<e.theta.size();++j) {
        auto plus=e.theta, minus=e.theta; plus(j)+=1e-5; minus(j)-=1e-5;
        auto ep=robust::embed_nested_null(alt.pt,alt.rep,m.pt,m.rep,plus,*ka,*k); REQUIRE(ep);
        auto em=robust::embed_nested_null(alt.pt,alt.rep,m.pt,m.rep,minus,*ka,*k); REQUIRE(em);
        J.col(j)=(ep->theta-em->theta)/2e-5;
      }
      return Eigen::MatrixXd(J*covariance*J.transpose());
    };
    const auto reference_covariance=covariance_at_alt(nulls[0],en,reference.covariance);
    for (const auto& m : nulls) {
      auto k=estimate::build_eq_constraints(m.pt); REQUIRE(k);
      // Embed the reference restriction in the equivalent representation.
      auto embedded=robust::embed_nested_null(m.pt,m.rep,nulls[0].pt,nulls[0].rep,en.theta,*k,*k0);
      REQUIRE(embedded);
      auto e=en; e.theta=embedded->theta;
      auto p=api::policy_inference_fiml(m.pt,m.rep,raw,*pack,e,{});
      REQUIRE(p.covariance_reason==api::InferenceReason::Available);
      CHECK((covariance_at_alt(m,e,p.covariance)-reference_covariance).norm()/reference_covariance.norm()<1e-8);
      auto nested=api::policy_nested_fiml(m.pt,m.rep,e,{},alt.pt,alt.rep,ea,{},raw,*pack);
      for (const auto& pair : {std::pair{p.score,reference.score},std::pair{p.lr,reference.lr},
          std::pair{nested.score,reference_nested.score},std::pair{nested.lr,reference_nested.lr}}) {
        REQUIRE(pair.first.reason==api::InferenceReason::Available);
        CHECK(pair.first.statistic==doctest::Approx(pair.second.statistic).epsilon(1e-8));
        CHECK((pair.first.eigenvalues-pair.second.eigenvalues).norm()/pair.second.eigenvalues.norm()<1e-8);
      }
      if (missing==0) {
        auto ml=prepare_ml(m,e), ml_alt=prepare_ml(alt,ea);
        auto mp=api::policy_inference_ml(*ml,{});
        auto baseline=prepare_ml(nulls[0],en);
        auto bp=api::policy_inference_ml(*baseline,{});
        auto bn=api::policy_nested_ml(baseline,{},ml_alt,{});
        CHECK((mp.covariance-p.covariance).norm()/p.covariance.norm()<1e-8);
        auto mn=api::policy_nested_ml(ml,{},ml_alt,{});
        CHECK(mn.score.statistic==doctest::Approx(nested.score.statistic).epsilon(1e-8));
        CHECK((mn.score.eigenvalues-nested.score.eigenvalues).norm()<1e-8);
        for (const auto& pair : {std::pair{mp.score,bp.score},std::pair{mp.lr,bp.lr},
            std::pair{mn.score,bn.score},std::pair{mn.lr,bn.lr}}) {
          REQUIRE(pair.first.reason==api::InferenceReason::Available);
          CHECK(pair.first.statistic==doctest::Approx(pair.second.statistic).epsilon(1e-8));
          CHECK((pair.first.eigenvalues-pair.second.eigenvalues).norm()/pair.second.eigenvalues.norm()<1e-8);
        }
      }
    }
  }
}

TEST_CASE("policy score contracts: actual Heywood PSD boundary has typed inference") {
  // Reuse the PSD-fit regression covariance, with deterministic raw rows
  // having exactly that covariance so both policy routes share the point.
  Eigen::Matrix3d S; S << 1.0,0.7,0.7, 0.7,1.0,0.3, 0.7,0.3,1.0;
  Eigen::MatrixXd X=Eigen::MatrixXd::Zero(200,3);
  Eigen::Matrix3d L=Eigen::LLT<Eigen::Matrix3d>(S).matrixL();
  for (int j=0;j<3;++j) {
    X.row(2*j)=10.0*L.col(j).transpose();
    X.row(2*j+1)=-10.0*L.col(j).transpose();
  }
  data::RawData raw; raw.X.push_back(X);
  auto m=build("f =~ x1 + x2 + x3");
  auto data=robust::frontier::prepare_ntml_data(raw,true); REQUIRE(data);
  auto ordinary=test::fit(m.pt,m.rep,(*data)->sample); REQUIRE(ordinary);
  auto bounded=estimate::frontier::fit_ml_psd(m.pt,m.rep,(*data)->sample,ordinary->theta); REQUIRE(bounded);
  auto state=api::policy_fit_state(*bounded);
  REQUIRE(state.psd_boundary);
  auto f=robust::frontier::prepare_ntml_fit(*data,m.pt,m.rep,*bounded); REQUIRE(f);
  auto ml=api::policy_inference_ml(**f,state);
  auto pack=estimate::fiml::fiml_pack(raw); REQUIRE(pack);
  auto fiml=api::policy_inference_fiml(m.pt,m.rep,raw,*pack,*bounded,state);
  for (const auto& p : {ml,fiml}) {
    CHECK(p.psd_boundary);
    if (p.covariance_reason==api::InferenceReason::Available) CHECK(p.covariance.allFinite());
    else CHECK_FALSE(api::reason_name(p.covariance_reason).empty());
    for (const auto& t : {p.score,p.lr}) {
      if (t.reason==api::InferenceReason::Available) {
        CHECK(std::isfinite(t.statistic)); CHECK(t.eigenvalues.allFinite());
      } else CHECK_FALSE(api::reason_name(t.reason).empty());
    }
  }
}
