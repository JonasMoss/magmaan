#include <doctest/doctest.h>
#include "../test_fit.hpp"
#include "magmaan/data/ordinal.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include <random>

namespace {
using namespace magmaan;
using estimate::OrdinalParameterization;
using estimate::OrdinalWeightKind;

// One factor with an omitted residual association: estimated weights matter
// at the population projection, rather than only through sampling residuals.
Eigen::MatrixXd mixed_block(unsigned seed, Eigen::Index n) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> z;
  Eigen::MatrixXd x(n, 6);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double f = z(rng), nuisance = z(rng);
    for (int j = 0; j < 6; ++j) {
      const double loading = 0.8 - 0.04*j;
      const double shared = j < 2 ? 0.35 : 0.0;
      const double y = loading*f + shared*nuisance +
          std::sqrt(1-loading*loading-shared*shared)*z(rng);
      x(i,j) = j < 3 ? 1.0+(y > -0.5)+(y > 0.5) : y+0.1*j;
    }
  }
  return x;
}
struct MixedModel { spec::LatentStructure pt; model::MatrixRep rep; };
MixedModel mixed_model(int groups) {
  auto parsed = parse::Parser::parse(
      "f =~ x1 + x2 + x3 + x4 + x5 + x6\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\n");
  REQUIRE(parsed.has_value());
  spec::BuildOptions opts;
  opts.meanstructure = true;
  opts.n_groups = groups;
  auto pt = spec::build(*parsed, opts);
  REQUIRE(pt.has_value());
  auto rep = model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return {*pt,*rep};
}
optim::OptimOptions tight() {
  optim::OptimOptions opts;
  opts.max_iter=3000; opts.ftol=1e-14; opts.gtol=1e-10;
  return opts;
}
}

TEST_CASE("Mixed DWLS IJ covariance agrees with the delete-one jackknife") {
  for (auto parameterization : {OrdinalParameterization::Delta, OrdinalParameterization::Theta}) {
    for (int groups : {1,2}) {
      double previous_error=1.0;
      for (int n : {600,1200}) {
        CAPTURE(n); CAPTURE(groups); CAPTURE(static_cast<int>(parameterization));
        std::vector<Eigen::MatrixXd> blocks;
        std::vector<std::vector<std::int32_t>> ordered(static_cast<std::size_t>(groups), {1,1,1,0,0,0});
        for (int g=0;g<groups;++g) blocks.push_back(mixed_block(6700u+static_cast<unsigned>(g),n));
        auto stats=data::mixed_ordinal_stats_from_data(blocks,ordered,false);
        REQUIRE(stats.has_value());
        const auto m=mixed_model(groups);
        auto fit=test::fit_mixed_ordinal_bounded(m.pt,m.rep,*stats,{},OrdinalWeightKind::DWLS,
            estimate::Backend::NloptLbfgs,tight(),parameterization);
        REQUIRE_MESSAGE(fit.has_value(), (fit.has_value()? "" : fit.error().detail));
        auto ij=estimate::robust_mixed_ordinal_ij(m.pt,m.rep,*stats,*fit,OrdinalWeightKind::DWLS,parameterization);
        REQUIRE_MESSAGE(ij.has_value(), (ij.has_value()? "" : ij.error().detail));
        auto fixed=estimate::robust_mixed_ordinal(m.pt,m.rep,*stats,*fit,OrdinalWeightKind::DWLS,
            parameterization,robust::Information::Observed);
        REQUIRE(fixed.has_value());
        const Eigen::Index k=fit->theta.size();
        Eigen::MatrixXd jackknife=Eigen::MatrixXd::Zero(k,k);
        for (int g=0;g<groups;++g) {
          Eigen::MatrixXd thetas(n,k);
          for (int i=0;i<n;++i) {
            auto reduced=blocks;
            auto& x=reduced[static_cast<std::size_t>(g)];
            Eigen::MatrixXd y(n-1,x.cols());
            y.topRows(i)=x.topRows(i); y.bottomRows(n-1-i)=x.bottomRows(n-1-i);
            x=std::move(y);
            auto s=data::mixed_ordinal_stats_from_data(reduced,ordered,false);
            REQUIRE(s.has_value());
            auto refit=estimate::fit_mixed_ordinal_bounded(m.pt,m.rep,*s,{},OrdinalWeightKind::DWLS,
                fit->theta,estimate::Backend::NloptLbfgs,tight(),parameterization);
            REQUIRE_MESSAGE(refit.has_value(), (refit.has_value()? "" : refit.error().detail));
            thetas.row(i)=refit->theta.transpose();
          }
          const Eigen::MatrixXd centered=thetas.rowwise()-thetas.colwise().mean();
          jackknife += (double(n-1)/n)*centered.transpose()*centered;
        }
        const double error=(ij->vcov-jackknife).norm()/jackknife.norm();
        const Eigen::VectorXd diagonal=jackknife.diagonal();
        const double diagonal_error=(ij->vcov.diagonal()-diagonal).norm()/diagonal.norm();
        const double fixed_error=(fixed->vcov.diagonal()-diagonal).norm()/diagonal.norm();
        MESSAGE("mixed IJ relative Frobenius error: " << error << ", diagonal error " << diagonal_error
            << ", fixed-weight diagonal error " << fixed_error);
        // Provisional O(1/N) target: this gate currently fails and must not
        // be treated as a validated tolerance or evidence of a core defect.
        CHECK(error < 12.0/n);
        if (n==1200) CHECK(error < previous_error);
        previous_error=error;
      }
    }
  }
}

TEST_CASE("Mixed IJ reduction routes require both variable types") {
  const Eigen::MatrixXd x=mixed_block(6700u,300);
  // The public mixed builder deliberately rejects the endpoints. A direct
  // mixed-fit reduction to either pure route is consequently unavailable.
  for (int flag : {0,1}) {
    const Eigen::MatrixXd pure=flag ? x.leftCols(3).eval() : x.rightCols(3).eval();
    auto stats=data::mixed_ordinal_stats_from_data({pure},{{flag,flag,flag}},false);
    REQUIRE_FALSE(stats.has_value());
    CHECK(stats.error().kind == PostError::Kind::NumericIssue);
    CHECK(stats.error().detail.find("must contain both ordered and continuous") != std::string::npos);
  }
}


TEST_CASE("Mixed DWLS IJ agrees with replicated case-weight finite differences") {
  // Integer replication implements empirical case weights independently of
  // the influence formulas. +/- one copy is a central weight step 1/copies.
  const int n=150, copies=100;
  const auto x=mixed_block(6700u,n);
  const std::vector<std::vector<std::int32_t>> ordered{{1,1,1,0,0,0}};
  auto stats=data::mixed_ordinal_stats_from_data({x},ordered,false);
  REQUIRE(stats.has_value());
  auto direct=data::mixed_gamma_diag_data_influence(x,stats->ordered[0],stats->n_levels[0],
      stats->thresholds[0],stats->mean[0],stats->R[0]);
  auto movement=data::mixed_gamma_diag_jacobian_fd(x,stats->ordered[0],stats->n_levels[0],
      stats->thresholds[0],stats->mean[0],stats->R[0]);
  REQUIRE(direct.has_value()); REQUIRE(movement.has_value());
  const Eigen::MatrixXd gamma_if=*direct+stats->moment_influence[0]*movement->transpose();
  const auto m=mixed_model(1);
  for (auto parameterization : {OrdinalParameterization::Delta,OrdinalParameterization::Theta}) {
    auto fit=test::fit_mixed_ordinal_bounded(m.pt,m.rep,*stats,{},OrdinalWeightKind::DWLS,
        estimate::Backend::NloptLbfgs,tight(),parameterization);
    REQUIRE(fit.has_value());
    for (int row : {0,17,91}) {
      CAPTURE(row); CAPTURE(static_cast<int>(parameterization));
      Eigen::VectorXd theta[2];
      data::MixedOrdinalStats perturbed[2];
      for (int side=0;side<2;++side) {
        const int step=side==0 ? -1 : 1;
        Eigen::MatrixXd repeated(n*copies+step,x.cols());
        int off=0;
        for (int i=0;i<n;++i)
          for (int j=0;j<copies+(i==row ? step : 0);++j) repeated.row(off++)=x.row(i);
        auto st=data::mixed_ordinal_stats_from_data({repeated},ordered,false);
        REQUIRE(st.has_value());
        perturbed[side]=std::move(*st);
        auto refit=estimate::fit_mixed_ordinal_bounded(m.pt,m.rep,perturbed[side],{},
            OrdinalWeightKind::DWLS,fit->theta,estimate::Backend::NloptLbfgs,tight(),parameterization);
        REQUIRE(refit.has_value());
        theta[side]=refit->theta;
      }
      const Eigen::VectorXd derivative=(theta[1]-theta[0])*(copies/2.0);
      const Eigen::VectorXd moment_fd=(perturbed[1].moments[0]-perturbed[0].moments[0])*(n*copies/2.0);
      // Marginal sample mean and ML variance have exact empirical
      // derivatives, independent of any normal-information approximation.
      Eigen::VectorXd marginal(6);
      for (int j=0;j<3;++j) {
        const double residual=x(row,j+3)-stats->mean[0](j+3);
        marginal(j)=-residual;
        marginal(j+3)=residual*residual-stats->R[0](j+3,j+3);
      }
      CHECK((moment_fd.segment(6,6)-marginal).norm()/marginal.norm() < 1e-6);
      const double moment_error=(moment_fd-stats->moment_influence[0].row(row).transpose()).norm()/moment_fd.norm();
      const Eigen::VectorXd gamma_fd=(perturbed[1].NACOV[0].diagonal()-
          perturbed[0].NACOV[0].diagonal())*(n*copies/2.0);
      const double gamma_error=(gamma_fd-gamma_if.row(row).transpose()).norm()/gamma_fd.norm();
      // Isolate the selected row so its covariance is the outer product of
      // its predicted parameter derivative (the API does not expose rows).
      auto isolated=*stats;
      isolated.moment_influence[0].setZero();
      isolated.moment_influence[0].row(row)=stats->moment_influence[0].row(row);
      isolated.gamma_diag_influence={Eigen::MatrixXd::Zero(n,gamma_if.cols())};
      isolated.gamma_diag_influence[0].row(row)=gamma_if.row(row);
      auto ij=estimate::robust_mixed_ordinal_ij(m.pt,m.rep,isolated,*fit,
          OrdinalWeightKind::DWLS,parameterization);
      REQUIRE(ij.has_value());
      const Eigen::MatrixXd fd_outer=derivative*derivative.transpose();
      const double error=(ij->vcov-fd_outer).norm()/fd_outer.norm();
      MESSAGE("case-weight outer-product relative error " << error << ", Gamma row error " << gamma_error << ", moment row error " << moment_error);
      CHECK(error < 2e-5);
      CHECK(gamma_error < 1e-5);
      CHECK(moment_error < 1e-5);
    }
  }
}
