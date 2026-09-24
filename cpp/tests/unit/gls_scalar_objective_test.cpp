#include <doctest/doctest.h>

#include <random>
#include <string>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/gmm/weight.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

// ============================================================================
// `gmm::normal_theory_objective` — the scalar trace-identity form of GLS — must
// agree with the dense moment-quadratic path (`residuals` with
// `normal_theory_weight`, adapted by `optim::scalarize`) in both F and ∇F, at
// arbitrary θ, with and without mean structure, single- and multi-block.
//
// The trace form is what makes GLS affordable: O(p³ + p²·n_free) per
// evaluation against the dense path's O(q²·n_free) with q = p + p(p+1)/2, and
// it never materializes the q×q weight or its Cholesky. It is only legitimate
// if it is numerically the same objective, which is what this pins.
//
// Also covers `gmm::BlockWeight::NormalTheory`, the structured weight carrying
// chol(A): its quadratic form and its dense materialization must match the
// weight `normal_theory_weight` builds today.
// ============================================================================

#define REQUIRE_OK(value)                                                     \
  do {                                                                        \
    INFO("error: " << ((value).has_value() ? "" : (value).error().detail));   \
    REQUIRE((value).has_value());                                             \
    if (!(value).has_value()) return;                                         \
  } while (false)

namespace {

using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::build;
namespace gmm = magmaan::estimate::gmm;

// Independent dense reference for the normal-theory weight, written the way
// `gmm::normal_theory_weight` used to build it: an explicit
// W_kl = tr(A⁻¹ E_k A⁻¹ E_l) double loop over the symmetric lower-vech basis,
// halved on the covariance block, with A⁻¹ on the mean block.
//
// This must stay a *separate* implementation from `BlockWeight::nt_to_dense`,
// which now derives the same entries in closed form. Comparing the structured
// weight against `normal_theory_weight` alone would be circular, since that
// function returns a NormalTheory block itself.
Eigen::MatrixXd dense_nt_reference(const Eigen::MatrixXd& A, bool has_means) {
  const Eigen::Index p = A.rows();
  const Eigen::Index pstar = p * (p + 1) / 2;
  const Eigen::Index off = has_means ? p : 0;
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  const Eigen::MatrixXd Ainv = llt.solve(Eigen::MatrixXd::Identity(p, p));

  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(off + pstar, off + pstar);
  if (has_means) W.block(0, 0, p, p) = Ainv;

  auto vech_index = [p](Eigen::Index r, Eigen::Index c) {
    return c * p - (c * (c - 1)) / 2 + (r - c);
  };
  for (Eigen::Index c1 = 0; c1 < p; ++c1) {
    for (Eigen::Index r1 = c1; r1 < p; ++r1) {
      Eigen::MatrixXd E1 = Eigen::MatrixXd::Zero(p, p);
      E1(r1, c1) = 1.0;
      E1(c1, r1) = 1.0;
      const Eigen::MatrixXd A1 = Ainv * E1 * Ainv;
      for (Eigen::Index c2 = 0; c2 < p; ++c2) {
        for (Eigen::Index r2 = c2; r2 < p; ++r2) {
          Eigen::MatrixXd E2 = Eigen::MatrixXd::Zero(p, p);
          E2(r2, c2) = 1.0;
          E2(c2, r2) = 1.0;
          W(off + vech_index(r1, c1), off + vech_index(r2, c2)) =
              0.5 * (A1 * E2).trace();
        }
      }
    }
  }
  return W;
}

// A perturbation of each implied block as the sample statistic, so the
// residual is nonzero (a saturated-fit S would make every gradient vanish and
// pin nothing). Blocks come from the model's own implied moments, so the
// count always agrees with the evaluator.
SampleStats make_stats(const ModelEvaluator& ev, const Eigen::VectorXd& x0,
                       unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd(0.0, 0.02);
  SampleStats samp;
  auto m0 = ev.sigma(x0);
  if (!m0.has_value()) return samp;
  for (std::size_t b = 0; b < m0->sigma.size(); ++b) {
    Eigen::MatrixXd S = m0->sigma[b];
    for (Eigen::Index i = 0; i < S.rows(); ++i) {
      for (Eigen::Index j = 0; j <= i; ++j) {
        S(i, j) += nd(rng);
        S(j, i) = S(i, j);
      }
    }
    samp.S.push_back(S);
    samp.n_obs.push_back(200 + 37 * static_cast<std::int64_t>(b));
    // A mean block only exists when the *model* carries one; supplying sample
    // means against a covariance-only model leaves `ls_has_means` false.
    if (b < m0->mu.size() && m0->mu[b].size() > 0) {
      Eigen::VectorXd mn = m0->mu[b];
      for (Eigen::Index i = 0; i < mn.size(); ++i) mn(i) += nd(rng);
      samp.mean.push_back(mn);
    }
  }
  return samp;
}

void check_equivalence(const std::string& syntax, int n_groups,
                       bool meanstructure, unsigned seed) {
  INFO("syntax=" << syntax << " n_groups=" << n_groups
                 << " meanstructure=" << meanstructure);
  auto fp = Parser::parse(syntax);
  REQUIRE_OK(fp);
  magmaan::spec::BuildOptions bo;
  bo.meanstructure = meanstructure;
  bo.n_groups = n_groups;
  auto pt = build(*fp, bo);
  REQUIRE_OK(pt);
  auto mr = build_matrix_rep(*pt);
  REQUIRE_OK(mr);
  auto evr = ModelEvaluator::build(*pt, *mr);
  REQUIRE_OK(evr);
  const auto& ev = *evr;

  const Eigen::Index nf = static_cast<Eigen::Index>(ev.n_free());
  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(nf, 0.8);
  SampleStats samp = make_stats(ev, x0, seed);
  REQUIRE(samp.S.size() == static_cast<std::size_t>(n_groups));
  CHECK(samp.mean.size() == (meanstructure ? samp.S.size() : 0u));

  auto W = gmm::normal_theory_weight(ev, samp, x0);
  REQUIRE_OK(W);
  auto dense_prob = gmm::residuals(ev, samp, x0, *W);
  REQUIRE_OK(dense_prob);
  auto scalar_dense = magmaan::optim::scalarize(*dense_prob);

  auto trace_prob = gmm::normal_theory_objective(ev, samp, x0);
  REQUIRE_OK(trace_prob);

  std::mt19937 rng(seed + 991);
  std::normal_distribution<double> jitter(0.0, 0.12);
  int compared = 0;
  for (int t = 0; t < 25; ++t) {
    Eigen::VectorXd x(nf);
    for (Eigen::Index i = 0; i < nf; ++i) x(i) = 0.8 + jitter(rng);

    Eigen::VectorXd g_dense = Eigen::VectorXd::Zero(nf);
    Eigen::VectorXd g_trace = Eigen::VectorXd::Zero(nf);
    const double f_dense = scalar_dense.f(x, g_dense);
    const double f_trace = trace_prob->f(x, g_trace);

    if (!std::isfinite(f_dense) || !std::isfinite(f_trace)) {
      // Both adapters report an invalid θ the same way.
      CHECK(std::isfinite(f_dense) == std::isfinite(f_trace));
      continue;
    }
    ++compared;
    CHECK(f_trace == doctest::Approx(f_dense).epsilon(1e-11));
    const double gscale = std::max(1.0, g_dense.cwiseAbs().maxCoeff());
    CHECK((g_trace - g_dense).cwiseAbs().maxCoeff() / gscale < 1e-9);
  }
  CHECK(compared > 0);
}

}  // namespace

TEST_CASE("GLS scalar trace objective matches the dense moment-quadratic path") {
  const std::string one_factor = "f =~ x1 + x2 + x3 + x4";
  const std::string two_factor =
      "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf1 ~~ f2";

  SUBCASE("1 factor, covariance only")   { check_equivalence(one_factor, 1, false, 11); }
  SUBCASE("1 factor, mean structure")    { check_equivalence(one_factor, 1, true,  12); }
  SUBCASE("1 factor, 2 groups")          { check_equivalence(one_factor, 2, false, 13); }
  SUBCASE("1 factor, 3 groups + means")  { check_equivalence(one_factor, 3, true,  14); }
  SUBCASE("2 factors, covariance only")  { check_equivalence(two_factor, 1, false, 15); }
  SUBCASE("2 factors, mean structure")   { check_equivalence(two_factor, 1, true,  16); }
  SUBCASE("2 factors, 2 groups + means") { check_equivalence(two_factor, 2, true,  17); }
}

// The closed-form materialization is new math (the old path ran an explicit
// tr(A E_k A E_l) double loop), and the basis-expansion multiplicity differs
// between diagonal and off-diagonal vech entries. Pin it directly over a range
// of p, without needing a model to produce the covariance.
TEST_CASE("BlockWeight::NormalTheory closed-form to_dense matches the trace loop") {
  std::mt19937 rng(4242);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (Eigen::Index p : {2, 3, 4, 5, 6, 7, 8, 9, 11, 14}) {
    for (bool has_means : {false, true}) {
      CAPTURE(p);
      CAPTURE(has_means);
      Eigen::MatrixXd B(p, p);
      for (Eigen::Index i = 0; i < p; ++i) {
        for (Eigen::Index j = 0; j < p; ++j) B(i, j) = nd(rng);
      }
      const Eigen::MatrixXd A =
          B * B.transpose() +
          static_cast<double>(p) * Eigen::MatrixXd::Identity(p, p);

      auto bw = gmm::BlockWeight::normal_theory(
          A, has_means, magmaan::FitError::Kind::NumericIssue, "closed-form");
      REQUIRE_OK(bw);

      const Eigen::MatrixXd Wref = dense_nt_reference(A, has_means);
      REQUIRE(bw->rows() == Wref.rows());
      CHECK((bw->to_dense() - Wref).cwiseAbs().maxCoeff() < 1e-12);

      // ...and the whitening reproduces the same quadratic form.
      for (int t = 0; t < 10; ++t) {
        Eigen::VectorXd d(Wref.rows());
        for (Eigen::Index i = 0; i < d.size(); ++i) d(i) = nd(rng);
        const Eigen::VectorXd r = bw->t_apply(1.0, d);
        CHECK(r.squaredNorm() ==
              doctest::Approx(d.dot(Wref * d)).epsilon(1e-10));
      }
    }
  }
}

// Is the normal-theory weight literally the inverse of the normal-theory
// moment ACOV? `dls_weight.cpp` asserts it in a comment ("at a = 0 this equals
// gmm::normal_theory_weight's covariance block; its 0.5 scaling cancels the
// symmetric-basis factor 2") and `fiml::two_stage_stage2_weight_blocks` builds
// its `Nt` weight as `blockdiag(Sigma^-1, gamma_nt(Sigma)^-1)`.
//
// If that holds, the `Nt` Stage-2 weight is exactly BlockWeight::NormalTheory
// and can stop being materialized. This pins the claim rather than assuming
// it, because the duplication-matrix scaling differs between diagonal and
// off-diagonal vech entries and that is easy to get wrong in either direction.
TEST_CASE("gamma_nt(Sigma) inverse is the NormalTheory covariance weight") {
  std::mt19937 rng(20260918);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (Eigen::Index p : {2, 3, 4, 5, 6, 8, 10}) {
    CAPTURE(p);
    Eigen::MatrixXd B(p, p);
    for (Eigen::Index i = 0; i < p; ++i) {
      for (Eigen::Index j = 0; j < p; ++j) B(i, j) = nd(rng);
    }
    const Eigen::MatrixXd S =
        B * B.transpose() +
        static_cast<double>(p) * Eigen::MatrixXd::Identity(p, p);

    auto G = magmaan::data::gamma_nt(S);
    REQUIRE_OK(G);
    const Eigen::MatrixXd Ginv = G->inverse();

    auto bw = gmm::BlockWeight::normal_theory(
        S, /*has_means=*/false, magmaan::FitError::Kind::NumericIssue, "chk");
    REQUIRE_OK(bw);
    const Eigen::MatrixXd W = bw->to_dense();

    REQUIRE(W.rows() == Ginv.rows());
    const double rel =
        (Ginv - W).cwiseAbs().maxCoeff() /
        std::max(1e-300, W.cwiseAbs().maxCoeff());
    INFO("relative discrepancy = " << rel);
    CHECK(rel < 1e-9);
  }
}

TEST_CASE("BlockWeight::NormalTheory reproduces the dense normal-theory weight") {
  for (bool meanstructure : {false, true}) {
    CAPTURE(meanstructure);
    auto fp = Parser::parse("f =~ x1 + x2 + x3 + x4");
    REQUIRE_OK(fp);
    magmaan::spec::BuildOptions bo;
    bo.meanstructure = meanstructure;
    auto pt = build(*fp, bo);
    REQUIRE_OK(pt);
    auto mr = build_matrix_rep(*pt);
    REQUIRE_OK(mr);
    auto evr = ModelEvaluator::build(*pt, *mr);
    REQUIRE_OK(evr);
    const auto& ev = *evr;

    const Eigen::Index nf = static_cast<Eigen::Index>(ev.n_free());
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(nf, 0.8);
    SampleStats samp = make_stats(ev, x0, 23);
    const Eigen::MatrixXd Wref = dense_nt_reference(samp.S[0], meanstructure);

    // The weight the fit path actually builds is now structured; it must still
    // materialize to the reference.
    auto W = gmm::normal_theory_weight(ev, samp, x0);
    REQUIRE_OK(W);
    REQUIRE(W->size() == 1u);
    CHECK((*W)[0].kind() == gmm::BlockWeight::Kind::NormalTheory);
    CHECK(((*W)[0].to_dense() - Wref).cwiseAbs().maxCoeff() < 1e-10);

    auto bw = gmm::BlockWeight::normal_theory(
        samp.S[0], meanstructure, magmaan::FitError::Kind::NumericIssue,
        "test");
    REQUIRE_OK(bw);
    REQUIRE(bw->rows() == Wref.rows());

    // Closed-form materialization agrees with the explicit trace double loop.
    CHECK((bw->to_dense() - Wref).cwiseAbs().maxCoeff() < 1e-10);

    // And the whitening reproduces the quadratic form without forming q×q.
    std::mt19937 rng(77);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int t = 0; t < 20; ++t) {
      Eigen::VectorXd d(Wref.rows());
      for (Eigen::Index i = 0; i < d.size(); ++i) d(i) = nd(rng);
      const Eigen::VectorXd r = bw->t_apply(1.0, d);
      CHECK(r.squaredNorm() == doctest::Approx(d.dot(Wref * d)).epsilon(1e-10));
    }
  }
}
