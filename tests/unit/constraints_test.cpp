#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "magmaan/error.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/weighted_chisq.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/spec/build.hpp"

#include "../inference_bundle.hpp"

using magmaan::estimate::build_eq_constraints;
using magmaan::estimate::build_nl_constraints;
using magmaan::estimate::NonlinearEqConstraints;
using magmaan::data::SampleStats;
using magmaan::test::expected_inference;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatId;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::build;
using magmaan::spec::BuildOptions;
using magmaan::spec::LatentStructure;

namespace {

LatentStructure must_lavaanify(std::string_view src, BuildOptions opts = {}) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = build(*fp, opts);
  REQUIRE_MESSAGE(pt.has_value(),
      "lavaanify failed: " << (pt.has_value() ? "" : pt.error().detail));
  return std::move(*pt);
}

// 3×3 sample covariance (x1,x2,x3) + n from the saturated 1F-CFA fixture.
SampleStats fixture_samp_3() {
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
      S(r, c) = M[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)].get<double>();
  SampleStats samp;
  samp.S.push_back(std::move(S));
  samp.n_obs.push_back(j["n_obs"].get<std::int64_t>());
  return samp;
}

magmaan::data::RawData multivariate_t_raw(std::mt19937& rng, Eigen::Index n,
                                          const Eigen::MatrixXd& Sigma,
                                          double df) {
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  const Eigen::Index p = Sigma.rows();
  std::normal_distribution<double> z(0.0, 1.0);
  std::chi_squared_distribution<double> chi(df);
  const double scale = std::sqrt((df - 2.0) / df);
  Eigen::MatrixXd X(n, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::VectorXd zi(p);
    for (Eigen::Index j = 0; j < p; ++j) zi(j) = z(rng);
    X.row(i) = (scale * (L * zi) / std::sqrt(chi(rng) / df)).transpose();
  }
  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  return raw;
}

// Free index of the cell Λ[row, 0] (the loading of the `row`-th observed
// variable on the sole latent), as a 0-based θ ordinal; -1 if not found.
Eigen::Index lambda_free_idx(const ModelEvaluator& ev, std::int16_t row) {
  const auto locs = ev.param_locations();
  for (std::size_t k = 0; k < locs.size(); ++k) {
    if (locs[k].mat == MatId::Lambda && locs[k].row == row) {
      return static_cast<Eigen::Index>(k);
    }
  }
  return -1;
}

double parameter_profile_scale(const magmaan::robust::ParamSpaceSandwich& sw,
                               Eigen::Index parameter) {
  REQUIRE(sw.K_con.size() == 0);
  REQUIRE(sw.A1.rows() == sw.A1.cols());
  REQUIRE(sw.B1.rows() == sw.B1.cols());
  REQUIRE(sw.A1.rows() == sw.B1.rows());
  REQUIRE(parameter >= 0);
  REQUIRE(parameter < sw.A1.rows());
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(sw.A1.rows());
  grad(parameter) = 1.0;
  const Eigen::MatrixXd A = 0.5 * (sw.A1 + sw.A1.transpose()).eval();
  const Eigen::MatrixXd B = 0.5 * (sw.B1 + sw.B1.transpose()).eval();
  Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
  REQUIRE(ldlt.info() == Eigen::Success);
  const Eigen::VectorXd x = ldlt.solve(grad);
  REQUIRE(ldlt.info() == Eigen::Success);
  const double bread_q = x.dot(A * x);
  const double meat_q = x.dot(B * x);
  REQUIRE(bread_q > 0.0);
  REQUIRE(meat_q > 0.0);
  return meat_q / bread_q;
}

}  // namespace

TEST_CASE("build_eq_constraints: identity reparameterization on a model with no `==` rows") {
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  auto con_or = build_eq_constraints(pt);
  REQUIRE(con_or.has_value());
  const auto& con = *con_or;
  CHECK_FALSE(con.active());
  CHECK(con.rank == 0);
  CHECK(con.npar == pt.n_free());
  CHECK(con.n_alpha == pt.n_free());
  for (std::int32_t k = 0; k < con.npar; ++k) {
    CHECK(con.group[static_cast<std::size_t>(k)] == k);
  }
}

TEST_CASE("build_eq_constraints: a shared label merges the two tied loadings") {
  auto pt = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  auto con_or = build_eq_constraints(pt);
  REQUIRE(con_or.has_value());
  const auto& con = *con_or;
  CHECK(con.active());
  CHECK(con.rank == 1);
  CHECK(con.npar == pt.n_free());
  CHECK(con.n_alpha == pt.n_free() - 1);

  auto rep = build_matrix_rep(pt).value();
  auto ev  = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  const Eigen::Index k_x3 = lambda_free_idx(ev, 2);
  REQUIRE(k_x2 >= 0);
  REQUIRE(k_x3 >= 0);
  CHECK(con.group[static_cast<std::size_t>(k_x2)] ==
        con.group[static_cast<std::size_t>(k_x3)]);
}

TEST_CASE("fit: an equality constraint is actually enforced (the tied loadings come out equal)") {
  auto samp = fixture_samp_3();
  auto pt   = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  auto rep  = build_matrix_rep(pt).value();

  auto est_or = magmaan::test::fit(pt, rep, samp);
  REQUIRE_MESSAGE(est_or.has_value(),
      "constrained fit failed: " << (est_or.has_value() ? "" : est_or.error().detail));
  const auto& est = *est_or;
  // The returned θ̂ is the FULL parameter vector (length n_free), not the
  // reduced α — downstream inference / R bindings expect this.
  CHECK(static_cast<std::int32_t>(est.theta.size()) == pt.n_free());

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  const Eigen::Index k_x3 = lambda_free_idx(ev, 2);
  REQUIRE(k_x2 >= 0);
  REQUIRE(k_x3 >= 0);
  CHECK(est.theta(k_x2) == doctest::Approx(est.theta(k_x3)).epsilon(1e-12));
}

TEST_CASE("Inference under an equality constraint: df += 1, tied SEs, χ² reflects the cost") {
  auto samp = fixture_samp_3();

  // Unconstrained baseline: saturated 1F CFA, df = 0, χ² ≈ 0.
  auto pt_u  = must_lavaanify("f =~ x1 + x2 + x3");
  auto rep_u = build_matrix_rep(pt_u).value();
  auto est_u = magmaan::test::fit(pt_u, rep_u, samp).value();
  auto inf_u = expected_inference(pt_u, rep_u, samp, est_u).value();
  CHECK(inf_u.df == 0);
  CHECK(inf_u.chi2 < 1e-6);

  // Constrained: x2 & x3 loadings tied → one fewer effective free param.
  auto pt_c  = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  auto rep_c = build_matrix_rep(pt_c).value();
  auto est_c = magmaan::test::fit(pt_c, rep_c, samp).value();
  auto inf_or = expected_inference(pt_c, rep_c, samp, est_c);
  REQUIRE_MESSAGE(inf_or.has_value(),
      "constrained expected_inference failed: " <<
          (inf_or.has_value() ? "" : inf_or.error().detail));
  const auto& inf = *inf_or;

  CHECK(inf.df == inf_u.df + 1);
  CHECK(inf.info.rows() == pt_c.n_free());   // full n_free × n_free info
  CHECK(inf.info.cols() == pt_c.n_free());
  CHECK(inf.vcov.rows() == pt_c.n_free());
  CHECK(inf.se.size()   == pt_c.n_free());
  CHECK(inf.chi2 >= inf_u.chi2);             // constraining can't improve fit
  CHECK(inf.chi2 > 1e-8);                    // on real data it costs something

  auto ev = ModelEvaluator::build(pt_c, rep_c).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  const Eigen::Index k_x3 = lambda_free_idx(ev, 2);
  REQUIRE(k_x2 >= 0);
  REQUIRE(k_x3 >= 0);
  CHECK(inf.se(k_x2) == doctest::Approx(inf.se(k_x3)).epsilon(1e-10));
}

TEST_CASE("build_eq_constraints: inequality (`<` / `>`) rows error out — not yet enforced") {
  // A `b > 0` line on a labeled factor variance produces a GtConstraint row.
  auto pt = must_lavaanify("f =~ x1 + x2 + x3\nf ~~ b*f\nb > 0");
  auto con_or = build_eq_constraints(pt);
  REQUIRE_FALSE(con_or.has_value());
  CHECK(con_or.error().kind == magmaan::PostError::Kind::NumericIssue);
}

// === P9 phase 2 — general LINEAR equality constraints ======================

TEST_CASE("build_eq_constraints: pure-merge path unchanged — no lin_constraint rows, 0/1 K") {
  auto pt = must_lavaanify("f =~ x1 + a*x2 + a*x3");
  CHECK(pt.lin_constraint_R.empty());
  CHECK(pt.lin_constraint_d.empty());
  auto con = build_eq_constraints(pt).value();
  CHECK(con.active());
  CHECK_FALSE(con.group.empty());                 // pure-merge ⇒ diagnostic group filled
  CHECK(con.Kmat.rows() == pt.n_free());
  CHECK(con.Kmat.cols() == con.n_alpha);
  CHECK(con.theta0.isZero());                      // θ₀ = 0 in the merge case
  // K is the 0/1 group-membership matrix.
  for (std::int32_t k = 0; k < con.npar; ++k) {
    for (std::int32_t g = 0; g < con.n_alpha; ++g) {
      const double want = (con.group[static_cast<std::size_t>(k)] == g) ? 1.0 : 0.0;
      CHECK(con.Kmat(k, g) == want);
    }
  }
}

TEST_CASE("build_eq_constraints: general linear `a == 2*b + c`") {
  // x0 is the marker (fixed), so a/b/c (loadings of x1/x2/x3) are all free.
  auto pt = must_lavaanify("f =~ x0 + a*x1 + b*x2 + c*x3\na == 2*b + c");
  CHECK_FALSE(pt.has_unenforced_constraints);     // linear ⇒ resolved, not flagged
  CHECK(pt.lin_constraint_d.size() == 1);
  auto con = build_eq_constraints(pt).value();
  CHECK(con.active());
  CHECK(con.rank == 1);
  CHECK(con.npar == pt.n_free());
  CHECK(con.n_alpha == pt.n_free() - 1);
  CHECK(con.Kmat.cols() == con.n_alpha);
  CHECK(con.group.empty());                        // general path ⇒ no merge partition
  // K's columns are orthonormal (an SVD kernel basis).
  Eigen::MatrixXd KtK = con.Kmat.transpose() * con.Kmat;
  CHECK(KtK.isApprox(Eigen::MatrixXd::Identity(con.n_alpha, con.n_alpha), 1e-9));
}

TEST_CASE("fit: a linear equality constraint `b2 + b3 == 1.5` is enforced") {
  auto samp = fixture_samp_3();
  auto pt   = must_lavaanify("f =~ x1 + b2*x2 + b3*x3\nb2 + b3 == 1.5");
  CHECK_FALSE(pt.has_unenforced_constraints);
  auto rep  = build_matrix_rep(pt).value();

  auto est_or = magmaan::test::fit(pt, rep, samp);
  REQUIRE_MESSAGE(est_or.has_value(),
      "constrained fit failed: " << (est_or.has_value() ? "" : est_or.error().detail));
  const auto& est = *est_or;
  CHECK(static_cast<std::int32_t>(est.theta.size()) == pt.n_free());

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  const Eigen::Index k_x3 = lambda_free_idx(ev, 2);
  REQUIRE(k_x2 >= 0);
  REQUIRE(k_x3 >= 0);
  CHECK(est.theta(k_x2) + est.theta(k_x3) == doctest::Approx(1.5).epsilon(1e-9));

  // df increases by 1 vs the unconstrained model.
  auto pt_u  = must_lavaanify("f =~ x1 + b2*x2 + b3*x3");
  auto rep_u = build_matrix_rep(pt_u).value();
  auto est_u = magmaan::test::fit(pt_u, rep_u, samp).value();
  auto inf_u = expected_inference(pt_u, rep_u, samp, est_u).value();
  auto inf_c = expected_inference(pt, rep, samp, est).value();
  CHECK(inf_c.df == inf_u.df + 1);
  CHECK(inf_c.chi2 >= inf_u.chi2 - 1e-9);
  CHECK(inf_c.se.size() == pt.n_free());           // full n_free-length SEs
}

TEST_CASE("fit: `d == 0` pins a loading to zero") {
  auto samp = fixture_samp_3();
  auto pt   = must_lavaanify("f =~ x1 + d*x2 + x3\nd == 0");
  CHECK_FALSE(pt.has_unenforced_constraints);
  auto rep  = build_matrix_rep(pt).value();
  auto est  = magmaan::test::fit(pt, rep, samp).value();
  auto ev   = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  REQUIRE(k_x2 >= 0);
  CHECK(std::abs(est.theta(k_x2)) < 1e-8);
}

TEST_CASE("build_eq_constraints: an infeasible linear system errors out") {
  auto pt = must_lavaanify("f =~ x1 + d*x2 + x3\nd == 1\nd == 2");
  CHECK_FALSE(pt.has_unenforced_constraints);       // both rows ARE linear...
  auto con_or = build_eq_constraints(pt);           // ...but jointly infeasible
  REQUIRE_FALSE(con_or.has_value());
  CHECK(con_or.error().kind == magmaan::PostError::Kind::NumericIssue);
  CHECK(con_or.error().detail.find("infeasible") != std::string::npos);
}

TEST_CASE("build_eq_constraints: a nonlinear `==` is classified, not yet enforced") {
  auto pt = must_lavaanify("f =~ x1 + a*x2 + b*x3\na == b*b");
  CHECK_FALSE(pt.has_unenforced_constraints);        // well-formed: `a`, `b` known
  CHECK(pt.nonlinear_eq_rows.size() == 1);           // `b*b` ⇒ nonlinear equality
  auto con_or = build_eq_constraints(pt);
  REQUIRE_FALSE(con_or.has_value());                 // skipped only with allow_nonlinear
  CHECK(con_or.error().kind == magmaan::PostError::Kind::NumericIssue);
  CHECK(con_or.error().detail.find("nonlinear") != std::string::npos);
}

TEST_CASE("build_eq_constraints: an inequality constraint gets a specific error") {
  auto pt = must_lavaanify("f =~ x1 + a*x2 + b*x3\na > b");
  CHECK(pt.has_inequality_constraints);
  CHECK_FALSE(pt.has_unenforced_constraints);
  CHECK(pt.nonlinear_eq_rows.empty());
  auto con_or = build_eq_constraints(pt);
  REQUIRE_FALSE(con_or.has_value());
  CHECK(con_or.error().detail.find("inequality") != std::string::npos);
}

TEST_CASE("build_eq_constraints: an `==` to an unknown parameter is malformed") {
  auto pt = must_lavaanify("f =~ x1 + a*x2 + x3\na == nope");
  CHECK(pt.has_unenforced_constraints);
  CHECK(pt.nonlinear_eq_rows.empty());
  auto con_or = build_eq_constraints(pt);
  REQUIRE_FALSE(con_or.has_value());
  CHECK(con_or.error().detail.find("malformed") != std::string::npos);
}

// === nonlinear equality constraints ========================================

TEST_CASE("NonlinearEqConstraints: h and jacobian of a hand-built tree") {
  using magmaan::spec::NlConstraint;
  using magmaan::spec::NlExprNode;
  // h = θ0 − θ1^2  (the constraint θ0 == θ1^2).
  NlExprNode n0; n0.kind = NlExprNode::Kind::Param;  n0.free_idx = 0;   // θ0
  NlExprNode n1; n1.kind = NlExprNode::Kind::Param;  n1.free_idx = 1;   // θ1
  NlExprNode n2; n2.kind = NlExprNode::Kind::Const;  n2.constant = 2.0; // 2
  NlExprNode n3; n3.kind = NlExprNode::Kind::Binary;
  n3.bin_op = magmaan::parse::BinOp::Pow; n3.lhs = 1; n3.rhs = 2;       // θ1^2
  NlExprNode n4; n4.kind = NlExprNode::Kind::Binary;
  n4.bin_op = magmaan::parse::BinOp::Sub; n4.lhs = 0; n4.rhs = 3;       // θ0 − θ1^2
  NlConstraint c;
  c.nodes = {n0, n1, n2, n3, n4};
  c.root  = 4;

  NonlinearEqConstraints nl;
  nl.rows = {c};
  nl.npar = 2;
  REQUIRE(nl.active());
  CHECK(nl.m() == 1);

  Eigen::VectorXd theta(2);
  theta << 2.0, 3.0;
  const Eigen::VectorXd hv = nl.h(theta);
  REQUIRE(hv.size() == 1);
  CHECK(hv(0) == doctest::Approx(2.0 - 9.0));   // 2 − 3²

  const Eigen::MatrixXd J = nl.jacobian(theta);
  REQUIRE(J.rows() == 1);
  REQUIRE(J.cols() == 2);
  CHECK(J(0, 0) == doctest::Approx(1.0));        // ∂h/∂θ0
  CHECK(J(0, 1) == doctest::Approx(-6.0));       // ∂h/∂θ1 = −2·3
}

TEST_CASE("resolve_lin_constraints: compiles a nonlinear `==` into nl_constraints") {
  auto pt = must_lavaanify("f =~ x1 + a*x2 + b*x3\na == b*b");
  REQUIRE(pt.nonlinear_eq_rows.size() == 1);
  REQUIRE(pt.nl_constraints.size() == 1);
  CHECK(pt.nl_constraints[0].root >= 0);
  CHECK_FALSE(pt.nl_constraints[0].nodes.empty());

  auto nl = build_nl_constraints(pt);
  CHECK(nl.m() == 1);
  CHECK(nl.npar == pt.n_free());

  // At θ ≡ 2, h = a − b·b = 2 − 4 = −2, whatever free indices `a` and `b`
  // landed on. The Jacobian row touches exactly those two parameters:
  // ∂h/∂a = 1, ∂h/∂b = −2b = −4, every other column 0.
  Eigen::VectorXd theta = Eigen::VectorXd::Constant(pt.n_free(), 2.0);
  CHECK(nl.h(theta)(0) == doctest::Approx(-2.0));
  const Eigen::MatrixXd J = nl.jacobian(theta);
  int nonzero = 0;
  for (Eigen::Index k = 0; k < J.cols(); ++k)
    if (J(0, k) != 0.0) ++nonzero;
  CHECK(nonzero == 2);
  CHECK(J.row(0).sum() == doctest::Approx(1.0 - 4.0));
}

TEST_CASE("nonlinear `==` with exp/log: compiled and AD-evaluated") {
  // exp() / log() in a nonlinear constraint compile into the node pool and
  // evaluate (value + θ-gradient) through the forward-mode AD sweep.
  for (const char* model : {"f =~ x1 + a*x2 + b*x3\na == exp(b)",
                            "f =~ x1 + a*x2 + b*x3\na == log(b)"}) {
    auto pt = must_lavaanify(model);
    REQUIRE(pt.nonlinear_eq_rows.size() == 1);
    REQUIRE(pt.nl_constraints.size() == 1);
    auto nl = build_nl_constraints(pt);
    REQUIRE(nl.m() == 1);

    const bool is_exp =
        std::string_view(model).find("exp") != std::string_view::npos;
    const double fb  = is_exp ? std::exp(0.7) : std::log(0.7);
    const double dfb = is_exp ? std::exp(0.7) : 1.0 / 0.7;

    // h = a − f(b) at θ ≡ 0.7.
    Eigen::VectorXd theta = Eigen::VectorXd::Constant(pt.n_free(), 0.7);
    CHECK(nl.h(theta)(0) == doctest::Approx(0.7 - fb));
    // Jacobian row: ∂h/∂a = 1, ∂h/∂b = −f'(b), every other column 0.
    const Eigen::MatrixXd J = nl.jacobian(theta);
    int nonzero = 0;
    for (Eigen::Index k = 0; k < J.cols(); ++k)
      if (J(0, k) != 0.0) ++nonzero;
    CHECK(nonzero == 2);
    CHECK(J.row(0).sum() == doctest::Approx(1.0 - dfb));
  }
}

TEST_CASE("fit: nonlinear equality constraints require a constrained backend") {
  auto samp = fixture_samp_3();
  auto pt = must_lavaanify("f =~ x1 + a*x2 + b*x3\na == b^2");
  auto rep = build_matrix_rep(pt).value();
  auto est_or = magmaan::test::fit(pt, rep, samp);
  REQUIRE_FALSE(est_or.has_value());
  CHECK(est_or.error().detail.find("nlopt-slsqp") != std::string::npos);
}

TEST_CASE("fit: nonlinear equality constraints accept the NLopt SLSQP backend") {
  Eigen::MatrixXd S(3, 3);
  S << 1.3583698455127435, 0.40737133015270244, 0.57989932234369646,
       0.40737133015270244, 1.3817838655202481,  0.45106393693226349,
       0.57989932234369646, 0.45106393693226349, 1.2748648607631261;
  SampleStats samp;
  samp.S.push_back(S);
  samp.n_obs.push_back(301);

  auto pt = must_lavaanify("visual =~ x1 + a*x2 + b*x3\na == b^2");
  auto rep = build_matrix_rep(pt).value();
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 2000;
  auto est_or = magmaan::test::fit(
      pt, rep, samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(est_or.has_value(), "SLSQP constrained fit failed: "
      << (est_or.has_value() ? std::string{} : est_or.error().detail));
  const auto& est = *est_or;

  Eigen::VectorXd ref(6);
  ref << 0.76844012067838263, 0.87660715038973203, 0.75267986994138880,
         1.04132952146332847, 0.74953748710182533, 0.62738942422421695;
  for (Eigen::Index k = 0; k < 6; ++k)
    CHECK(est.theta(k) == doctest::Approx(ref(k)).epsilon(1e-4));

  auto nl = build_nl_constraints(pt);
  CHECK(std::abs(nl.h(est.theta)(0)) < 1e-5);
  CHECK(est.audit.constrained);
  CHECK(est.audit.constraint_violation_inf <= 1e-6);
  CHECK(est.audit.stationary);
}

TEST_CASE("frontier fit_ml_constrained appends a programmatic scalar equality") {
  auto samp = fixture_samp_3();
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  auto rep = build_matrix_rep(pt).value();
  auto est = magmaan::test::fit(pt, rep, samp).value();

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  REQUIRE(k_x2 >= 0);
  const double target = 0.95 * est.theta(k_x2);

  magmaan::estimate::frontier::ExtraNonlinearEqConstraints extra;
  extra.n_constraint = 1;
  extra.h = [k_x2, target](const Eigen::VectorXd& theta) {
    Eigen::VectorXd h(1);
    h(0) = theta(k_x2) - target;
    return h;
  };
  extra.jacobian = [k_x2](const Eigen::VectorXd& theta) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, theta.size());
    J(0, k_x2) = 1.0;
    return J;
  };

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto con = magmaan::estimate::frontier::fit_ml_constrained(
      pt, rep, samp, est.theta, std::move(extra), {},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(con.has_value(), "programmatic constrained fit failed: "
      << (con.has_value() ? std::string{} : con.error().detail));
  CHECK(con->theta(k_x2) == doctest::Approx(target).epsilon(1e-7));
  CHECK(con->fmin >= est.fmin - 1e-10);
}

TEST_CASE("frontier fit_ml_psd agrees with ordinary ML on an interior CFA") {
  const auto samp = fixture_samp_3();
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  const auto rep = build_matrix_rep(pt).value();
  auto ordinary = magmaan::test::fit(pt, rep, samp);
  REQUIRE(ordinary.has_value());
  REQUIRE(ordinary->diagnostics.admissibility.admissible);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto constrained = magmaan::estimate::frontier::fit_ml_psd(
      pt, rep, samp, ordinary->theta,
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(constrained.has_value(), "PSD ML fit failed: "
      << (constrained.has_value() ? std::string{}
                                  : constrained.error().detail));
  CHECK(constrained->fmin ==
        doctest::Approx(ordinary->fmin).epsilon(1e-7));
  CHECK((constrained->theta - ordinary->theta).cwiseAbs().maxCoeff() < 1e-5);
  CHECK(constrained->diagnostics.admissibility.admissible);
  CHECK(constrained->audit.constrained);
  CHECK(constrained->audit.constraint_violation_inf <= 1e-6);
  CHECK(constrained->audit.stationary);

#ifdef MAGMAAN_WITH_IPOPT
  auto ipopt = magmaan::estimate::frontier::fit_ml_psd(
      pt, rep, samp, ordinary->theta,
      magmaan::estimate::Backend::Ipopt, opts);
  REQUIRE_MESSAGE(ipopt.has_value(), "IPOPT PSD ML fit failed: "
      << (ipopt.has_value() ? std::string{} : ipopt.error().detail));
  CHECK(ipopt->fmin == doctest::Approx(constrained->fmin).epsilon(1e-7));
  CHECK((ipopt->theta - constrained->theta).cwiseAbs().maxCoeff() < 1e-5);
  CHECK(ipopt->audit.constrained);
  CHECK(ipopt->audit.constraint_violation_inf <= 1e-6);
  CHECK(ipopt->audit.stationary);
#endif
}

TEST_CASE("frontier fit_ml_psd preserves shared covariance parameters") {
  const auto samp = fixture_samp_3();
  auto pt = must_lavaanify(
      "f =~ x1 + x2 + x3\n"
      "x1 ~~ a*x1\n"
      "x2 ~~ a*x2");
  const auto rep = build_matrix_rep(pt).value();
  auto x0 = magmaan::estimate::simple_start_values(pt, rep, samp, {});
  REQUIRE(x0.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 5000;
  auto constrained = magmaan::estimate::frontier::fit_ml_psd(
      pt, rep, samp, *x0, magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(constrained.has_value(), "shared-covariance PSD fit failed: "
      << (constrained.has_value() ? std::string{}
                                  : constrained.error().detail));
  std::vector<Eigen::Index> residual_variances;
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const auto& cell = rep.cell_for_row[i];
    if (cell.used && cell.mat == MatId::Theta &&
        cell.row == cell.col && pt.free[i] > 0) {
      residual_variances.push_back(pt.free[i] - 1);
    }
  }
  REQUIRE(residual_variances.size() == 3);
  const Eigen::Index k1 = residual_variances[0];
  const Eigen::Index k2 = residual_variances[1];
  REQUIRE(pt.eq_groups[static_cast<std::size_t>(k1)] ==
          pt.eq_groups[static_cast<std::size_t>(k2)]);
  CHECK(constrained->theta(k1) ==
        doctest::Approx(constrained->theta(k2)).epsilon(1e-10));
  CHECK(constrained->diagnostics.admissibility.admissible);
}

TEST_CASE("frontier fit_ml_psd separates covariance-connected components") {
  SampleStats samp;
  Eigen::Matrix3d S;
  S << 1.0, 0.66, 0.48,
       0.66, 1.0, 0.42,
       0.48, 0.42, 1.0;
  samp.S.push_back(S);
  samp.n_obs.push_back(300);

  auto pt = must_lavaanify(
      "f =~ x1 + x2 + x3\n"
      "x1 ~~ x2");
  const auto rep = build_matrix_rep(pt).value();
  auto x0 = magmaan::estimate::simple_start_values(pt, rep, samp, {});
  REQUIRE(x0.has_value());
  auto ordinary = magmaan::estimate::fit_ml(pt, rep, samp, *x0);
  REQUIRE(ordinary.has_value());
  REQUIRE(ordinary->diagnostics.admissibility.admissible);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto constrained = magmaan::estimate::frontier::fit_ml_psd(
      pt, rep, samp, ordinary->theta,
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(constrained.has_value(), "component PSD fit failed: "
      << (constrained.has_value() ? std::string{}
                                  : constrained.error().detail));
  CHECK(constrained->fmin ==
        doctest::Approx(ordinary->fmin).epsilon(1e-7));
  CHECK((constrained->theta - ordinary->theta).cwiseAbs().maxCoeff() < 1e-5);
  CHECK(constrained->diagnostics.admissibility.admissible);
}

TEST_CASE("frontier fit_ml_psd handles reduced-LISREL structural zeros") {
  SampleStats samp;
  Eigen::Matrix2d S;
  S << 2.0, 0.8,
       0.8, 1.5;
  samp.S.push_back(S);
  samp.n_obs.push_back(300);

  auto pt = must_lavaanify("y ~ x");
  const auto rep = build_matrix_rep(pt).value();
  REQUIRE(rep.form == magmaan::model::RepForm::Reduced);
  auto x0 = magmaan::estimate::simple_start_values(pt, rep, samp, {});
  REQUIRE(x0.has_value());
  auto ordinary = magmaan::estimate::fit_ml(pt, rep, samp, *x0);
  REQUIRE(ordinary.has_value());
  REQUIRE(ordinary->diagnostics.admissibility.admissible);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto constrained = magmaan::estimate::frontier::fit_ml_psd(
      pt, rep, samp, ordinary->theta,
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(constrained.has_value(), "structural PSD fit failed: "
      << (constrained.has_value() ? std::string{}
                                  : constrained.error().detail));
  CHECK(constrained->fmin ==
        doctest::Approx(ordinary->fmin).epsilon(1e-7));
  CHECK((constrained->theta - ordinary->theta).cwiseAbs().maxCoeff() < 1e-5);
  CHECK(constrained->diagnostics.admissibility.admissible);
}

TEST_CASE("frontier fit_ml_psd replaces a negative residual Heywood optimum") {
  SampleStats samp;
  Eigen::Matrix3d S;
  S << 1.0, 0.7, 0.7,
       0.7, 1.0, 0.3,
       0.7, 0.3, 1.0;
  samp.S.push_back(S);
  samp.n_obs.push_back(200);

  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  const auto rep = build_matrix_rep(pt).value();
  auto x0 = magmaan::estimate::simple_start_values(pt, rep, samp, {});
  REQUIRE(x0.has_value());
  auto ordinary = magmaan::estimate::fit_ml(pt, rep, samp, *x0);
  REQUIRE(ordinary.has_value());
  REQUIRE_FALSE(ordinary->diagnostics.admissibility.admissible);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 5000;
  opts.gtol = 1e-8;
  auto constrained = magmaan::estimate::frontier::fit_ml_psd(
      pt, rep, samp, ordinary->theta,
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(constrained.has_value(), "PSD Heywood fit failed: "
      << (constrained.has_value() ? std::string{}
                                  : constrained.error().detail));
  CHECK(constrained->diagnostics.admissibility.admissible);
  CHECK(constrained->fmin > ordinary->fmin + 1e-6);
  CHECK(constrained->audit.constrained);
  CHECK(constrained->audit.constraint_violation_inf <= 1e-6);
  CHECK(constrained->audit.stationary);
}

TEST_CASE("frontier fit_ml_psd enforces joint 3x3 Psi PSD") {
  // Every 2x2 principal submatrix of Psi has correlation magnitude .9, but
  // the full sign pattern has one negative eigenvalue. Adding fixed
  // Theta=.9I keeps the observed sample covariance positive definite.
  SampleStats samp;
  Eigen::Matrix3d S;
  S << 1.9,  0.9,  0.9,
       0.9,  1.9, -0.9,
       0.9, -0.9,  1.9;
  samp.S.push_back(S);
  samp.n_obs.push_back(500);

  auto pt = must_lavaanify(
      "f1 =~ 1*x1\n"
      "f2 =~ 1*x2\n"
      "f3 =~ 1*x3\n"
      "x1 ~~ 0.9*x1\n"
      "x2 ~~ 0.9*x2\n"
      "x3 ~~ 0.9*x3");
  const auto rep = build_matrix_rep(pt).value();
  auto x0 = magmaan::estimate::simple_start_values(pt, rep, samp, {});
  REQUIRE(x0.has_value());
  auto ordinary = magmaan::estimate::fit_ml(pt, rep, samp, *x0);
  REQUIRE(ordinary.has_value());
  REQUIRE_FALSE(ordinary->diagnostics.admissibility.admissible);
  REQUIRE(ordinary->diagnostics.admissibility.psi_blocks.size() == 1);
  CHECK(ordinary->diagnostics.admissibility.psi_blocks[0]
            .negative_variance_rows.empty());
  CHECK(ordinary->diagnostics.admissibility.psi_blocks[0]
            .invalid_correlation_rows.empty());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 5000;
  opts.gtol = 1e-8;
  auto constrained = magmaan::estimate::frontier::fit_ml_psd(
      pt, rep, samp, ordinary->theta,
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(constrained.has_value(), "joint-Psi PSD fit failed: "
      << (constrained.has_value() ? std::string{}
                                  : constrained.error().detail));
  CHECK(constrained->diagnostics.admissibility.admissible);
  CHECK(constrained->fmin > ordinary->fmin + 1e-6);
}

TEST_CASE("frontier profile_lrt_parameter_ml reports the ordinary df-1 LR statistic") {
  auto samp = fixture_samp_3();
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  auto rep = build_matrix_rep(pt).value();
  auto est = magmaan::test::fit(pt, rep, samp).value();

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  REQUIRE(k_x2 >= 0);
  const double target = 0.95 * est.theta(k_x2);

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto lrt = magmaan::estimate::frontier::profile_lrt_parameter_ml(
      pt, rep, samp, est, k_x2, target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(lrt.has_value(), "parameter profile LRT failed: "
      << (lrt.has_value() ? std::string{} : lrt.error().detail));

  CHECK(lrt->unrestricted_value == doctest::Approx(est.theta(k_x2)));
  CHECK(lrt->constrained_value == doctest::Approx(target).epsilon(1e-7));
  CHECK(std::abs(lrt->constraint_residual) < 1e-7);
  CHECK(lrt->fmin_unrestricted == doctest::Approx(est.fmin));
  CHECK(lrt->T == doctest::Approx(
      2.0 * static_cast<double>(samp.n_obs[0]) *
      (lrt->fmin_constrained - est.fmin)));
  CHECK(lrt->p_value == doctest::Approx(
      magmaan::inference::chi2_pvalue(lrt->T, 1)));
  CHECK(lrt->df == 1);
}

TEST_CASE("frontier continuous profile LRT supports robust scaling and scaled CI") {
  Eigen::Vector4d lam;
  lam << 1.0, 0.82, 0.74, 0.68;
  Eigen::MatrixXd Sigma = lam * lam.transpose();
  Sigma.diagonal().array() += 0.45;
  std::mt19937 rng(20260703);
  auto raw = multivariate_t_raw(rng, 360, Sigma, 5.0);
  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE_MESSAGE(samp_or.has_value(),
      "sample_stats_from_raw failed: "
          << (samp_or.has_value() ? "" : samp_or.error().detail));
  SampleStats samp = std::move(*samp_or);
  samp.mean.clear();

  auto pt = must_lavaanify("f =~ x1 + x2 + x3 + x4");
  auto rep = build_matrix_rep(pt).value();
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;

  auto ml = magmaan::test::fit(
      pt, rep, samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptSlsqp, opts).value();
  auto gmm = magmaan::test::fit_gmm(
      pt, rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptSlsqp, opts).value();
  auto gls = magmaan::test::fit_gls(
      pt, rep, samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptSlsqp, opts).value();

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  REQUIRE(k_x2 >= 0);
  const double ml_target = 0.95 * ml.theta(k_x2);
  const double gmm_target = 0.95 * gmm.theta(k_x2);
  const double gls_target = 0.95 * gls.theta(k_x2);

  auto ml_plain = magmaan::estimate::frontier::profile_lrt_parameter_ml(
      pt, rep, samp, ml, k_x2, ml_target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE(ml_plain.has_value());
  auto ml_robust = magmaan::estimate::frontier::profile_lrt_parameter_ml(
      pt, rep, samp, ml, k_x2, ml_target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw);
  REQUIRE_MESSAGE(ml_robust.has_value(),
      "ML robust profile LRT failed: "
          << (ml_robust.has_value() ? "" : ml_robust.error().detail));
  CHECK(ml_robust->T == doctest::Approx(ml_plain->T).epsilon(1e-8));
  CHECK(std::isfinite(ml_robust->scaling_factor));
  CHECK(ml_robust->scaling_factor > 0.0);
  CHECK(ml_robust->T_scaled ==
        doctest::Approx(ml_robust->T / ml_robust->scaling_factor));
  CHECK(ml_robust->p_value_scaled == doctest::Approx(
      magmaan::inference::chi2_pvalue(ml_robust->T_scaled, 1)));
  auto ml_misspec = magmaan::estimate::frontier::profile_lrt_parameter_ml(
      pt, rep, samp, ml, k_x2, ml_target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw,
      magmaan::estimate::frontier::ScalarProfileReference::MisspecMixture);
  REQUIRE_MESSAGE(ml_misspec.has_value(),
      "ML misspec profile LRT failed: "
          << (ml_misspec.has_value() ? "" : ml_misspec.error().detail));
  CHECK(ml_misspec->T == doctest::Approx(ml_plain->T).epsilon(1e-8));
  CHECK(ml_misspec->misspec_scaling_factor > 0.0);
  REQUIRE(ml_misspec->misspec_eigvals.size() == 1);
  CHECK(ml_misspec->misspec_eigvals(0) ==
        doctest::Approx(ml_misspec->misspec_scaling_factor));
  CHECK(ml_misspec->T_misspec_scaled ==
        doctest::Approx(ml_misspec->T / ml_misspec->misspec_scaling_factor));
  CHECK(ml_misspec->p_value_misspec_mixture == doctest::Approx(
      magmaan::robust::weighted_chisq_upper(
          ml_misspec->misspec_eigvals, ml_misspec->T)));

  auto gmm_plain = magmaan::estimate::frontier::profile_lrt_parameter_gmm(
      pt, rep, samp, gmm, {}, k_x2, gmm_target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE(gmm_plain.has_value());
  auto gmm_robust = magmaan::estimate::frontier::profile_lrt_parameter_gmm(
      pt, rep, samp, gmm, {}, k_x2, gmm_target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw);
  REQUIRE_MESSAGE(gmm_robust.has_value(),
      "GMM robust profile LRT failed: "
          << (gmm_robust.has_value() ? "" : gmm_robust.error().detail));
  CHECK(gmm_robust->T == doctest::Approx(gmm_plain->T).epsilon(1e-8));
  CHECK(std::isfinite(gmm_robust->scaling_factor));
  CHECK(gmm_robust->scaling_factor > 0.0);
  CHECK(gmm_robust->T_scaled ==
        doctest::Approx(gmm_robust->T / gmm_robust->scaling_factor));
  CHECK(gmm_robust->p_value_scaled == doctest::Approx(
      magmaan::inference::chi2_pvalue(gmm_robust->T_scaled, 1)));
  auto gmm_misspec = magmaan::estimate::frontier::profile_lrt_parameter_gmm(
      pt, rep, samp, gmm, {}, k_x2, gmm_target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw, {},
      magmaan::estimate::frontier::ScalarProfileReference::MisspecMixture);
  REQUIRE_MESSAGE(gmm_misspec.has_value(),
      "GMM misspec profile LRT failed: "
          << (gmm_misspec.has_value() ? "" : gmm_misspec.error().detail));
  CHECK(gmm_misspec->T == doctest::Approx(gmm_plain->T).epsilon(1e-8));
  CHECK(gmm_misspec->misspec_scaling_factor > 0.0);
  REQUIRE(gmm_misspec->misspec_eigvals.size() == 1);
  CHECK(gmm_misspec->misspec_eigvals(0) ==
        doctest::Approx(gmm_misspec->misspec_scaling_factor));

  auto fitw_plain =
      magmaan::estimate::frontier::profile_lrt_parameter_gmm_fitted_weight(
          pt, rep, samp, gmm, k_x2, gmm_target, {},
          magmaan::estimate::Bounds{},
          magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE(fitw_plain.has_value());
  auto fitw_robust =
      magmaan::estimate::frontier::profile_lrt_parameter_gmm_fitted_weight(
          pt, rep, samp, gmm, k_x2, gmm_target, {},
          magmaan::estimate::Bounds{},
          magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw);
  REQUIRE_MESSAGE(fitw_robust.has_value(),
      "fitted-weight GMM robust profile LRT failed: "
          << (fitw_robust.has_value() ? "" : fitw_robust.error().detail));
  CHECK(fitw_robust->T == doctest::Approx(fitw_plain->T).epsilon(1e-8));
  CHECK(std::isfinite(fitw_robust->scaling_factor));
  CHECK(fitw_robust->scaling_factor > 0.0);
  CHECK(fitw_robust->T_scaled ==
        doctest::Approx(fitw_robust->T / fitw_robust->scaling_factor));
  CHECK(fitw_robust->p_value_scaled == doctest::Approx(
      magmaan::inference::chi2_pvalue(fitw_robust->T_scaled, 1)));
  auto fitw_misspec =
      magmaan::estimate::frontier::profile_lrt_parameter_gmm_fitted_weight(
          pt, rep, samp, gmm, k_x2, gmm_target, {},
          magmaan::estimate::Bounds{},
          magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw,
          magmaan::estimate::frontier::ScalarProfileReference::MisspecMixture);
  REQUIRE_MESSAGE(fitw_misspec.has_value(),
      "fitted-weight GMM misspec profile LRT failed: "
          << (fitw_misspec.has_value() ? "" : fitw_misspec.error().detail));
  CHECK(fitw_misspec->T == doctest::Approx(fitw_plain->T).epsilon(1e-8));
  CHECK(fitw_misspec->misspec_scaling_factor > 0.0);
  REQUIRE(fitw_misspec->misspec_eigvals.size() == 1);
  CHECK(fitw_misspec->misspec_eigvals(0) ==
        doctest::Approx(fitw_misspec->misspec_scaling_factor));

  magmaan::estimate::frontier::ScalarProfileCiOptions ci_opts;
  ci_opts.reference =
      magmaan::estimate::frontier::ScalarProfileReference::RobustScaled;
  ci_opts.target_tol = 1e-4;
  ci_opts.statistic_tol = 1e-3;
  ci_opts.initial_step = 0.05 * std::abs(gmm.theta(k_x2));
  ci_opts.max_iter = 40;

  auto ci = magmaan::estimate::frontier::profile_lrt_ci_parameter_gmm(
      pt, rep, samp, gmm, {}, k_x2, ci_opts, {},
      magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw);
  REQUIRE_MESSAGE(ci.has_value(),
      "GMM robust profile CI failed: "
          << (ci.has_value() ? "" : ci.error().detail));
  CHECK(ci->lower < gmm.theta(k_x2));
  CHECK(ci->upper > gmm.theta(k_x2));
  CHECK(ci->lower_profile.scaling_factor > 0.0);
  CHECK(ci->upper_profile.scaling_factor > 0.0);
  CHECK(ci->lower_profile.T_scaled ==
        doctest::Approx(ci->cutoff).epsilon(2e-3));
  CHECK(ci->upper_profile.T_scaled ==
        doctest::Approx(ci->cutoff).epsilon(2e-3));

  auto ci_misspec_scaled_opts = ci_opts;
  ci_misspec_scaled_opts.reference =
      magmaan::estimate::frontier::ScalarProfileReference::MisspecScaled;
  auto ci_misspec_scaled =
      magmaan::estimate::frontier::profile_lrt_ci_parameter_gmm(
          pt, rep, samp, gmm, {}, k_x2, ci_misspec_scaled_opts, {},
          magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw);
  REQUIRE_MESSAGE(ci_misspec_scaled.has_value(),
      "GMM misspec-scaled profile CI failed: "
          << (ci_misspec_scaled.has_value() ? ""
              : ci_misspec_scaled.error().detail));
  CHECK(ci_misspec_scaled->lower_profile.T_misspec_scaled ==
        doctest::Approx(ci_misspec_scaled->cutoff).epsilon(2e-3));
  CHECK(ci_misspec_scaled->upper_profile.T_misspec_scaled ==
        doctest::Approx(ci_misspec_scaled->cutoff).epsilon(2e-3));

  auto ci_mixture_opts = ci_opts;
  ci_mixture_opts.reference =
      magmaan::estimate::frontier::ScalarProfileReference::MisspecMixture;
  auto ci_mixture = magmaan::estimate::frontier::profile_lrt_ci_parameter_gmm(
      pt, rep, samp, gmm, {}, k_x2, ci_mixture_opts, {},
      magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw);
  REQUIRE_MESSAGE(ci_mixture.has_value(),
      "GMM misspec-mixture profile CI failed: "
          << (ci_mixture.has_value() ? "" : ci_mixture.error().detail));
  CHECK(ci_mixture->lower_cutoff == doctest::Approx(
      magmaan::robust::weighted_chisq_quantile(
          ci_mixture->lower_profile.misspec_eigvals,
          ci_mixture->confidence_level)));
  CHECK(ci_mixture->upper_cutoff == doctest::Approx(
      magmaan::robust::weighted_chisq_quantile(
          ci_mixture->upper_profile.misspec_eigvals,
          ci_mixture->confidence_level)));
  CHECK(ci_mixture->lower_profile.T ==
        doctest::Approx(ci_mixture->lower_cutoff).epsilon(2e-3));
  CHECK(ci_mixture->upper_profile.T ==
        doctest::Approx(ci_mixture->upper_cutoff).epsilon(2e-3));

  auto W_gls = magmaan::estimate::gmm::normal_theory_weight(ev, samp, gls.theta);
  REQUIRE_MESSAGE(W_gls.has_value(),
      "normal_theory_weight failed: "
          << (W_gls.has_value() ? "" : W_gls.error().detail));
  auto gls_fixed = magmaan::estimate::frontier::profile_lrt_parameter_gmm(
      pt, rep, samp, gls, *W_gls, k_x2, gls_target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw);
  REQUIRE_MESSAGE(gls_fixed.has_value(),
      "GLS fixed-weight robust profile LRT failed: "
          << (gls_fixed.has_value() ? "" : gls_fixed.error().detail));

  magmaan::estimate::frontier::GmmProfileRobustOptions ew_opts;
  ew_opts.estimated_weight = true;
  ew_opts.ij_weight_mode =
      magmaan::estimate::ContinuousLsIJWeightMode::SampleNormalTheory;
  auto gls_ew = magmaan::estimate::frontier::profile_lrt_parameter_gmm(
      pt, rep, samp, gls, *W_gls, k_x2, gls_target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw, ew_opts);
  REQUIRE_MESSAGE(gls_ew.has_value(),
      "GLS estimated-weight robust profile LRT failed: "
          << (gls_ew.has_value() ? "" : gls_ew.error().detail));
  CHECK(gls_ew->T == doctest::Approx(gls_fixed->T).epsilon(1e-8));
  CHECK(std::isfinite(gls_ew->scaling_factor));
  CHECK(gls_ew->scaling_factor > 0.0);
  CHECK(gls_ew->T_scaled ==
        doctest::Approx(gls_ew->T / gls_ew->scaling_factor));
  CHECK(std::abs(gls_ew->scaling_factor - gls_fixed->scaling_factor) > 1e-5);

  auto sw_ij = magmaan::estimate::continuous_ls_param_space_sandwich_ij(
      pt, rep, samp, gls_ew->constrained, *W_gls, raw,
      magmaan::estimate::ContinuousLsIJWeightMode::SampleNormalTheory);
  REQUIRE_MESSAGE(sw_ij.has_value(),
      "IJ sandwich failed: "
          << (sw_ij.has_value() ? "" : sw_ij.error().detail));
  CHECK(gls_ew->scaling_factor ==
        doctest::Approx(parameter_profile_scale(*sw_ij, k_x2)).epsilon(1e-8));

  auto ci_gls_ew = magmaan::estimate::frontier::profile_lrt_ci_parameter_gmm(
      pt, rep, samp, gls, *W_gls, k_x2, ci_opts, {},
      magmaan::estimate::Backend::NloptSlsqp, opts, 1e-6, &raw, ew_opts);
  REQUIRE_MESSAGE(ci_gls_ew.has_value(),
      "GLS estimated-weight robust profile CI failed: "
          << (ci_gls_ew.has_value() ? "" : ci_gls_ew.error().detail));
  CHECK(ci_gls_ew->lower < gls.theta(k_x2));
  CHECK(ci_gls_ew->upper > gls.theta(k_x2));
  CHECK(ci_gls_ew->lower_profile.T_scaled ==
        doctest::Approx(ci_gls_ew->cutoff).epsilon(2e-3));
  CHECK(ci_gls_ew->upper_profile.T_scaled ==
        doctest::Approx(ci_gls_ew->cutoff).epsilon(2e-3));
}

TEST_CASE("frontier fit_gmm_constrained appends a programmatic scalar equality") {
  auto samp = fixture_samp_3();
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  auto rep = build_matrix_rep(pt).value();
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto est = magmaan::test::fit_gmm(
      pt, rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptSlsqp, opts).value();

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  REQUIRE(k_x2 >= 0);
  const double target = 0.95 * est.theta(k_x2);

  magmaan::estimate::frontier::ExtraNonlinearEqConstraints extra;
  extra.n_constraint = 1;
  extra.h = [k_x2, target](const Eigen::VectorXd& theta) {
    Eigen::VectorXd h(1);
    h(0) = theta(k_x2) - target;
    return h;
  };
  extra.jacobian = [k_x2](const Eigen::VectorXd& theta) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, theta.size());
    J(0, k_x2) = 1.0;
    return J;
  };

  auto con = magmaan::estimate::frontier::fit_gmm_constrained(
      pt, rep, samp, est.theta, {}, std::move(extra), {},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(con.has_value(), "programmatic constrained GMM fit failed: "
      << (con.has_value() ? std::string{} : con.error().detail));
  CHECK(con->theta(k_x2) == doctest::Approx(target).epsilon(1e-7));
  CHECK(con->fmin >= est.fmin - 1e-10);
}

TEST_CASE("frontier profile_lrt_parameter_gmm reports the ordinary df-1 statistic") {
  auto samp = fixture_samp_3();
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  auto rep = build_matrix_rep(pt).value();
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto est = magmaan::test::fit_gmm(
      pt, rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptSlsqp, opts).value();

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  REQUIRE(k_x2 >= 0);
  const double target = 0.95 * est.theta(k_x2);

  auto lrt = magmaan::estimate::frontier::profile_lrt_parameter_gmm(
      pt, rep, samp, est, {}, k_x2, target, {},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(lrt.has_value(), "parameter GMM profile LRT failed: "
      << (lrt.has_value() ? std::string{} : lrt.error().detail));

  CHECK(lrt->unrestricted_value == doctest::Approx(est.theta(k_x2)));
  CHECK(lrt->constrained_value == doctest::Approx(target).epsilon(1e-7));
  CHECK(std::abs(lrt->constraint_residual) < 1e-7);
  CHECK(lrt->fmin_unrestricted == doctest::Approx(est.fmin));
  CHECK(lrt->T == doctest::Approx(
      2.0 * static_cast<double>(samp.n_obs[0]) *
      (lrt->fmin_constrained - est.fmin)));
  CHECK(lrt->p_value == doctest::Approx(
      magmaan::inference::chi2_pvalue(lrt->T, 1)));
  CHECK(lrt->df == 1);
}

TEST_CASE("frontier profile_lrt_parameter_gmm_fitted_weight reports the ordinary df-1 statistic") {
  auto samp = fixture_samp_3();
  auto pt = must_lavaanify("f =~ x1 + x2 + x3");
  auto rep = build_matrix_rep(pt).value();
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto start = magmaan::test::fit_gmm(
      pt, rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptSlsqp, opts).value();

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  REQUIRE(k_x2 >= 0);
  const double target = 0.95 * start.theta(k_x2);

  auto lrt = magmaan::estimate::frontier::profile_lrt_parameter_gmm_fitted_weight(
      pt, rep, samp, start, k_x2, target, {},
      magmaan::estimate::Bounds{}, magmaan::estimate::Backend::NloptSlsqp,
      opts);
  REQUIRE_MESSAGE(lrt.has_value(), "fitted-weight parameter GMM profile LRT failed: "
      << (lrt.has_value() ? std::string{} : lrt.error().detail));

  CHECK(lrt->constrained_value == doctest::Approx(target).epsilon(1e-7));
  CHECK(std::abs(lrt->constraint_residual) < 1e-7);
  CHECK(lrt->T == doctest::Approx(
      2.0 * static_cast<double>(samp.n_obs[0]) *
      (lrt->fmin_constrained - lrt->fmin_unrestricted)));
  CHECK(lrt->p_value == doctest::Approx(
      magmaan::inference::chi2_pvalue(lrt->T, 1)));
  CHECK(lrt->df == 1);
}

TEST_CASE("frontier profile_lrt_ci_parameter_gmm inverts the df-1 cutoff") {
  Eigen::Vector4d lam(1.0, 0.82, 0.74, 0.68);
  Eigen::MatrixXd S = lam * lam.transpose();
  S.diagonal().array() += 0.45;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {1500};

  auto pt = must_lavaanify("f =~ x1 + x2 + x3 + x4");
  auto rep = build_matrix_rep(pt).value();
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 3000;
  auto est = magmaan::test::fit_gmm(
      pt, rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptSlsqp, opts).value();

  auto ev = ModelEvaluator::build(pt, rep).value();
  const Eigen::Index k_x2 = lambda_free_idx(ev, 1);
  REQUIRE(k_x2 >= 0);

  magmaan::estimate::frontier::ScalarProfileCiOptions ci_opts;
  ci_opts.target_tol = 1e-4;
  ci_opts.statistic_tol = 1e-4;
  ci_opts.initial_step = 0.05 * std::abs(est.theta(k_x2));
  ci_opts.max_iter = 40;

  auto ci = magmaan::estimate::frontier::profile_lrt_ci_parameter_gmm(
      pt, rep, samp, est, {}, k_x2, ci_opts, {},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(ci.has_value(), "parameter GMM profile CI failed: "
      << (ci.has_value() ? std::string{} : ci.error().detail));

  CHECK(ci->lower < est.theta(k_x2));
  CHECK(ci->upper > est.theta(k_x2));
  CHECK_FALSE(ci->lower_at_bound);
  CHECK_FALSE(ci->upper_at_bound);
  CHECK(ci->lower_profile.T == doctest::Approx(ci->cutoff).epsilon(1e-3));
  CHECK(ci->upper_profile.T == doctest::Approx(ci->cutoff).epsilon(1e-3));
  CHECK(magmaan::inference::chi2_pvalue(ci->cutoff, 1) ==
        doctest::Approx(1.0 - ci->confidence_level).epsilon(1e-6));
}

TEST_CASE("fit_gmm: linear and nonlinear equality constraints accept NLopt SLSQP") {
  Eigen::Vector4d lam(1.0, 0.49, 0.7, 1.19);
  Eigen::MatrixXd S = lam * lam.transpose();
  S.diagonal().array() += 0.5;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {400};

  auto pt = must_lavaanify(
      "f =~ x1 + a*x2 + b*x3 + c*x4\nc == a + b\na == b^2");
  auto rep = build_matrix_rep(pt).value();
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 2000;
  auto est_or = magmaan::test::fit_gmm(
      pt, rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(est_or.has_value(), "SLSQP constrained GMM fit failed: "
      << (est_or.has_value() ? std::string{} : est_or.error().detail));

  auto nl = build_nl_constraints(pt);
  CHECK(std::abs(nl.h(est_or->theta)(0)) < 1e-5);

  auto con = build_eq_constraints(pt, /*allow_nonlinear=*/true).value();
  REQUIRE(con.active());
  CHECK((con.A_eq * est_or->theta - con.b_eq).cwiseAbs().maxCoeff() < 1e-8);
}

#ifdef MAGMAAN_WITH_IPOPT
TEST_CASE("fit: a nonlinear equality constraint (a == b^2) is enforced and matches lavaan") {
  // visual =~ x1 + a*x2 + b*x3 with the nonlinear constraint a == b^2, fit by
  // ML to the HolzingerSwineford1939 x1-x3 N-divisor covariance (n = 301).
  // Reference values from lavaan 0.6-22 `cfa()` with `a == b^2` in syntax.
  Eigen::MatrixXd S(3, 3);
  S << 1.3583698455127435, 0.40737133015270244, 0.57989932234369646,
       0.40737133015270244, 1.3817838655202481,  0.45106393693226349,
       0.57989932234369646, 0.45106393693226349, 1.2748648607631261;
  SampleStats samp;
  samp.S.push_back(S);
  samp.n_obs.push_back(301);

  auto pt = must_lavaanify("visual =~ x1 + a*x2 + b*x3\na == b^2");
  REQUIRE(pt.nl_constraints.size() == 1);
  auto rep = build_matrix_rep(pt).value();
  auto est_or = magmaan::test::fit(
      pt, rep, samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Ipopt);
  if (!est_or.has_value()) {
    FAIL("constrained fit failed: " << est_or.error().detail);
    return;
  }
  const auto& est = *est_or;
  REQUIRE(est.theta.size() == 6);

  // lavaan reference, magmaan free order [a, b, x1~~x1, x2~~x2, x3~~x3, psi].
  Eigen::VectorXd ref(6);
  ref << 0.76844012067838263, 0.87660715038973203, 0.75267986994138880,
         1.04132952146332847, 0.74953748710182533, 0.62738942422421695;
  for (Eigen::Index k = 0; k < 6; ++k)
    CHECK(est.theta(k) == doctest::Approx(ref(k)).epsilon(1e-4));

  // IPOPT drove the constraint h(θ̂) = a − b² to zero.
  auto nl = build_nl_constraints(pt);
  CHECK(std::abs(nl.h(est.theta)(0)) < 1e-5);
}

TEST_CASE("inference: vcov / df for a nonlinear-equality model match lavaan") {
  // Same a == b^2 model and HS x1-x3 covariance as the fit test above.
  Eigen::MatrixXd S(3, 3);
  S << 1.3583698455127435, 0.40737133015270244, 0.57989932234369646,
       0.40737133015270244, 1.3817838655202481,  0.45106393693226349,
       0.57989932234369646, 0.45106393693226349, 1.2748648607631261;
  SampleStats samp;
  samp.S.push_back(S);
  samp.n_obs.push_back(301);

  auto pt  = must_lavaanify("visual =~ x1 + a*x2 + b*x3\na == b^2");
  auto rep = build_matrix_rep(pt).value();
  auto est_or = magmaan::test::fit(
      pt, rep, samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Ipopt);
  if (!est_or.has_value()) {
    FAIL("constrained fit failed: " << est_or.error().detail);
    return;
  }
  const auto& est = *est_or;

  auto info_or = magmaan::inference::information_expected(pt, rep, samp, est);
  REQUIRE(info_or.has_value());

  // vcov via the null-space projection of the constraint Jacobian H(θ̂).
  auto vcov_or = magmaan::inference::vcov(*info_or, pt, est.theta);
  REQUIRE_MESSAGE(vcov_or.has_value(), "constrained vcov failed: "
      << (vcov_or.has_value() ? std::string{} : vcov_or.error().detail));
  const Eigen::VectorXd se = magmaan::inference::se(*vcov_or);
  REQUIRE(se.size() == 6);

  // lavaan reference SEs, free order [a, b, x1~~x1, x2~~x2, x3~~x3, psi].
  Eigen::VectorXd ref_se(6);
  ref_se << 0.141589294898072537, 0.080759833430825323, 0.112218326796740542,
            0.106358953281192381, 0.082513979989703973, 0.124620815146450709;
  for (Eigen::Index k = 0; k < 6; ++k)
    CHECK(se(k) == doctest::Approx(ref_se(k)).epsilon(1e-3));

  // The nonlinear `==` adds one degree of freedom (rank of H(θ̂) = 1).
  auto df_or = magmaan::inference::df_stat(pt, samp, est.theta);
  REQUIRE(df_or.has_value());
  CHECK(*df_or == 1);

  // The two-argument forms (no θ̂) must reject a nonlinear-constraint model.
  CHECK_FALSE(magmaan::inference::vcov(*info_or, pt).has_value());
  CHECK_FALSE(magmaan::inference::df_stat(pt, samp).has_value());
}

TEST_CASE("fit: general-linear + nonlinear equality constraints in one model") {
  // f =~ x1 + a*x2 + b*x3 + c*x4 with a general-linear equality `c == a + b`
  // AND a nonlinear equality `a == b^2`. IPOPT runs in the linear-reduced
  // α-space. S is synthesized from Λ = (1, 0.49, 0.7, 1.19) — so b = 0.7,
  // a = b² = 0.49, c = a + b = 1.19 — with ψ = 1, Θ = 0.5·I.
  Eigen::Vector4d lam(1.0, 0.49, 0.7, 1.19);
  Eigen::MatrixXd S = lam * lam.transpose();
  S.diagonal().array() += 0.5;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {400};

  auto pt = must_lavaanify(
      "f =~ x1 + a*x2 + b*x3 + c*x4\nc == a + b\na == b^2");
  REQUIRE(pt.nl_constraints.size() == 1);          // the nonlinear `==`
  REQUIRE(pt.lin_constraint_d.size() == 1);        // the general-linear `==`

  auto rep = build_matrix_rep(pt).value();
  auto est_or = magmaan::test::fit(
      pt, rep, samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Ipopt);
  if (!est_or.has_value()) {
    FAIL("combined-constraint fit failed: " << est_or.error().detail);
    return;
  }
  const auto& est = *est_or;

  // The nonlinear constraint h(θ̂) = a − b² is driven to zero.
  auto nl = build_nl_constraints(pt);
  CHECK(std::abs(nl.h(est.theta)(0)) < 1e-5);

  // The linear constraint A_eq·θ̂ = b_eq holds — it is built into the
  // α-reparameterization, so it is satisfied exactly.
  auto con = build_eq_constraints(pt, /*allow_nonlinear=*/true).value();
  REQUIRE(con.active());
  CHECK((con.A_eq * est.theta - con.b_eq).cwiseAbs().maxCoeff() < 1e-8);

  // df adds the linear-constraint rank (1) and the nonlinear rank (1) on top
  // of the unconstrained 10 − 8 = 2.
  auto df = magmaan::inference::df_stat(pt, samp, est.theta);
  REQUIRE(df.has_value());
  CHECK(*df == 4);
}

TEST_CASE("fit: shared-label (merge) + nonlinear equality constraints together") {
  // f =~ x1 + a*x2 + a*x3 + b*x4 — the shared label `a` is a pure-merge linear
  // equality — plus the nonlinear `b == a^2`. Exercises the pure-merge branch
  // of the combined-constraint α-space path. S from Λ = (1, 0.7, 0.7, 0.49).
  Eigen::Vector4d lam(1.0, 0.7, 0.7, 0.49);
  Eigen::MatrixXd S = lam * lam.transpose();
  S.diagonal().array() += 0.5;
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {400};

  auto pt = must_lavaanify("f =~ x1 + a*x2 + a*x3 + b*x4\nb == a^2");
  REQUIRE(pt.nl_constraints.size() == 1);

  auto rep = build_matrix_rep(pt).value();
  auto est_or = magmaan::test::fit(
      pt, rep, samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::Ipopt);
  if (!est_or.has_value()) {
    FAIL("merge + nonlinear fit failed: " << est_or.error().detail);
    return;
  }
  const auto& est = *est_or;

  auto nl = build_nl_constraints(pt);
  CHECK(std::abs(nl.h(est.theta)(0)) < 1e-5);

  // The shared label is a pure-merge reparameterization (con.group populated).
  auto con = build_eq_constraints(pt, /*allow_nonlinear=*/true).value();
  REQUIRE(con.active());
  CHECK_FALSE(con.group.empty());
}
#endif  // MAGMAAN_WITH_IPOPT

// === effect coding =========================================================

TEST_CASE("lavaanify: effect_coding synthesizes `Σλ == #indicators` and frees everything") {
  BuildOptions opts; opts.effect_coding = true;
  auto pt = must_lavaanify("f =~ x1 + x2 + x3", opts);
  CHECK_FALSE(pt.has_unenforced_constraints);
  CHECK(pt.lin_constraint_d.size() == 1);           // one latent, one group
  CHECK(pt.lin_constraint_d[0] == doctest::Approx(3.0));
  // The R row has a 1 on each of the three loading columns and 0 elsewhere.
  REQUIRE(pt.lin_constraint_R.size() == static_cast<std::size_t>(pt.n_free()));
  double rowsum = 0.0; int ones = 0;
  for (double v : pt.lin_constraint_R) { rowsum += v; if (v == 1.0) ++ones; }
  CHECK(rowsum == doctest::Approx(3.0));
  CHECK(ones == 3);
  CHECK(pt.free[0] > 0);                             // first loading is free, not the marker

  // f ~~ f stays free under effect coding.
  auto rep = build_matrix_rep(pt).value();
  auto ev  = ModelEvaluator::build(pt, rep).value();
  bool psi_free = false;
  for (const auto& loc : ev.param_locations())
    if (loc.mat == MatId::Psi && loc.row == loc.col) psi_free = true;
  CHECK(psi_free);
}

TEST_CASE("lavaanify: effect_coding and std_lv are mutually exclusive") {
  BuildOptions opts; opts.effect_coding = true; opts.std_lv = true;
  auto fp = Parser::parse("f =~ x1 + x2 + x3");
  REQUIRE(fp.has_value());
  auto pt = build(*fp, opts);
  REQUIRE_FALSE(pt.has_value());
  CHECK(pt.error().kind == magmaan::PartableError::Kind::BadGroupSpec);
}

TEST_CASE("fit: effect_coding — loadings sum to #indicators; χ²/df match the marker fit") {
  auto samp = fixture_samp_3();
  BuildOptions opts; opts.effect_coding = true;
  auto pt  = must_lavaanify("f =~ x1 + x2 + x3", opts);
  auto rep = build_matrix_rep(pt).value();
  auto est_or = magmaan::test::fit(pt, rep, samp);
  REQUIRE_MESSAGE(est_or.has_value(),
      "effect-coding fit failed: " << (est_or.has_value() ? "" : est_or.error().detail));
  const auto& est = *est_or;

  auto ev = ModelEvaluator::build(pt, rep).value();
  const double lsum = est.theta(lambda_free_idx(ev, 0)) +
                      est.theta(lambda_free_idx(ev, 1)) +
                      est.theta(lambda_free_idx(ev, 2));
  CHECK(lsum == doctest::Approx(3.0).epsilon(1e-7));

  auto inf_ec = expected_inference(pt, rep, samp, est).value();
  // Marker fit of the same model — bijective reparam ⇒ identical χ² and df.
  auto pt_m  = must_lavaanify("f =~ x1 + x2 + x3");
  auto rep_m = build_matrix_rep(pt_m).value();
  auto est_m = magmaan::test::fit(pt_m, rep_m, samp).value();
  auto inf_m = expected_inference(pt_m, rep_m, samp, est_m).value();
  CHECK(inf_ec.df == inf_m.df);
  CHECK(inf_ec.chi2 == doctest::Approx(inf_m.chi2).epsilon(1e-6));
}

TEST_CASE("constraints: multi-group shared-label LS fits via K-reparameterization") {
  // Two groups; the two non-marker loadings carry bare shared labels, so they
  // are equated across groups (cross-group metric invariance). fit_bounded's
  // LS path reparameterizes θ = θ₀ + K·α and optimizes the reduced bounded
  // problem — exact constraints, no penalty. The earlier 1e10-penalty path
  // made the bounded quasi-Newton search loop forever on this model.
  BuildOptions opts;
  opts.n_groups = 2;
  auto pt  = must_lavaanify("f =~ x1 + l2*x2 + l3*x3", opts);
  auto rep = build_matrix_rep(pt).value();

  // Two distinct 1-factor-structured covariances, one per group.
  auto cov1f = [](double psi) -> Eigen::Matrix3d {
    const Eigen::Vector3d lam(1.0, 0.8, 0.65);
    const Eigen::Vector3d th(0.6, 0.5, 0.7);
    return lam * lam.transpose() * psi + th.asDiagonal().toDenseMatrix();
  };
  SampleStats samp;
  samp.S = {cov1f(2.0), cov1f(2.6)};
  samp.n_obs = {180, 150};

  const magmaan::optim::OptimOptions opt{
      .max_iter = 5000, .ftol = 1e-14, .gtol = 1e-9};
  auto est_or = magmaan::test::fit_gmm(
      pt, rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE_MESSAGE(est_or.has_value(),
      "constrained multi-group LS fit failed: "
          << (est_or.has_value() ? "" : est_or.error().detail));
  CHECK(std::isfinite(est_or->fmin));

  // The shared labels must hold exactly in θ̂: A_eq·θ̂ = b_eq.
  auto con = build_eq_constraints(pt).value();
  REQUIRE(con.active());
  const double eq_resid =
      (con.A_eq * est_or->theta - con.b_eq).cwiseAbs().maxCoeff();
  CHECK(eq_resid < 1e-9);
}

TEST_CASE("constraints: general-linear equality LS fits via K-reparameterization") {
  // `b2 + b3 == 1.5` is arithmetic — a general-linear lin_constraint row, not
  // a pure-merge group. fit_bounded's LS path reparameterizes θ = θ₀ + K·α
  // with the SVD-kernel K (which rotates the parameter axes) and optimizes the
  // reduced problem — exact constraint, no penalty.
  auto pt  = must_lavaanify("f =~ x1 + b2*x2 + b3*x3\nb2 + b3 == 1.5");
  auto rep = build_matrix_rep(pt).value();

  // A 1-factor-structured 3×3 covariance.
  const Eigen::Vector3d lam(1.0, 0.7, 0.8);
  const Eigen::Vector3d th(0.5, 0.6, 0.4);
  Eigen::Matrix3d S =
      lam * lam.transpose() * 1.8 + th.asDiagonal().toDenseMatrix();
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {250};

  const magmaan::optim::OptimOptions opt{
      .max_iter = 5000, .ftol = 1e-14, .gtol = 1e-9};
  auto est_or = magmaan::test::fit_gmm(
      pt, rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE_MESSAGE(est_or.has_value(),
      "general-linear LS fit failed: "
          << (est_or.has_value() ? "" : est_or.error().detail));
  CHECK(std::isfinite(est_or->fmin));

  auto con = build_eq_constraints(pt).value();
  REQUIRE(con.active());
  REQUIRE(con.group.empty());   // general-linear path, not pure-merge
  const double eq_resid =
      (con.A_eq * est_or->theta - con.b_eq).cwiseAbs().maxCoeff();
  CHECK(eq_resid < 1e-9);
}
