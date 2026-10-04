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
      for (int n : {150,300}) {
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
        MESSAGE("mixed IJ relative Frobenius error: " << error);
        // Provisional O(1/N) target: this gate currently fails and must not
        // be treated as a validated tolerance or evidence of a core defect.
        CHECK(error < 12.0/n);
        if (n==300) CHECK(error < previous_error);
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
