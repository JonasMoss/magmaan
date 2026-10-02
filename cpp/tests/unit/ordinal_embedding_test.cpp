#include "ordinal_test_helpers.hpp"

TEST_CASE("Pairwise ordinal exact embedding accepts a fixed residual covariance") {
  std::mt19937 rng(71231);
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd X(140,4);
  for (Eigen::Index i=0;i<X.rows();++i) {
    const double eta=normal(rng);
    for (Eigen::Index j=0;j<X.cols();++j) {
      const double y=0.65*eta+0.76*normal(rng);
      X(i,j)=1.0+(y> -0.4)+(y>0.6);
    }
  }
  auto data=magmaan::estimate::frontier::pairwise_ordinal_observed_data({X},{{3,3,3,3}});
  REQUIRE(data.has_value());
  const std::string base="f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\nx4 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\nx4 ~*~ 1*x4\n";
  auto make_model=[&](const std::string& syntax) {
    auto fp=magmaan::parse::Parser::parse(syntax);
    REQUIRE(fp.has_value());
    magmaan::spec::LatentNames names;
    auto pt=magmaan::spec::build(*fp,{},nullptr,&names);
    REQUIRE(pt.has_value());
    auto mr=magmaan::model::build_matrix_rep(*pt,&names);
    REQUIRE(mr.has_value());
    return std::pair{*pt,*mr};
  };
  auto [pt1,mr1]=make_model(base+"x1 ~~ x2");
  magmaan::optim::OptimOptions options;
  options.max_iter=120;
  auto fit1=magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
      pt1,mr1,*data,{},{},magmaan::estimate::Backend::NloptLbfgs,options);
  REQUIRE(fit1.has_value());
  std::optional<magmaan::robust::LRSatorra2000Result> reference;
  for (const std::string tail:{"", "x1 ~~ 0*x2", "x1 ~~ a*x2\na == 0"}) {
    auto [pt0,mr0]=make_model(base+tail);
    auto fit0=magmaan::estimate::frontier::fit_pairwise_ordinal_composite(
        pt0,mr0,*data,{},{},magmaan::estimate::Backend::NloptLbfgs,options);
    REQUIRE(fit0.has_value());
    auto lr=magmaan::estimate::frontier::lr_test_pairwise_ordinal_composite(
        pt1,mr1,*data,*fit1,pt0,mr0,*fit0,magmaan::robust::SatorraAMethod::Exact,2e-5);
    REQUIRE_MESSAGE(lr.has_value(),(lr.has_value() ? "" : lr.error().detail));
    if (!reference) reference=*lr;
    CHECK(lr->df_diff==1);
    CHECK((lr->eigenvalues-reference->eigenvalues).norm()<1e-10);
    CHECK(std::abs(lr->T_diff-reference->T_diff)<1e-5);
  }
}
