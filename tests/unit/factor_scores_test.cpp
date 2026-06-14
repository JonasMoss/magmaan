#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/measures/factor_scores.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

// Generate raw observations from a 2-factor CFA:
//   x1..x3 load on f1, x4..x6 on f2; Cov(f1,f2) = 0.3, both unit variance;
//   residual variances 0.5. Returns the data and the true factor scores.
struct Generated {
  Eigen::MatrixXd X;        // n × 6
  Eigen::MatrixXd factors;  // n × 2 — the η that produced X
};

Generated generate(std::mt19937& rng, Eigen::Index n) {
  std::normal_distribution<double> z(0.0, 1.0);

  Eigen::MatrixXd Lambda(6, 2);
  Lambda << 1.0, 0.0,
            0.8, 0.0,
            1.2, 0.0,
            0.0, 1.0,
            0.0, 0.9,
            0.0, 1.1;
  Eigen::MatrixXd Phi(2, 2);
  Phi << 1.0, 0.3,
         0.3, 1.0;
  const Eigen::MatrixXd Lphi = Eigen::LLT<Eigen::MatrixXd>(Phi).matrixL();
  const double resid_sd = std::sqrt(0.5);

  Generated g;
  g.X.resize(n, 6);
  g.factors.resize(n, 2);
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::Vector2d zf(z(rng), z(rng));
    Eigen::Vector2d eta = Lphi * zf;
    g.factors.row(i) = eta.transpose();
    Eigen::VectorXd eps(6);
    for (Eigen::Index j = 0; j < 6; ++j) eps(j) = resid_sd * z(rng);
    g.X.row(i) = (Lambda * eta + eps).transpose();
  }
  return g;
}

// Pearson correlation of two column vectors.
double corr(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
  const Eigen::VectorXd ca = a.array() - a.mean();
  const Eigen::VectorXd cb = b.array() - b.mean();
  return ca.dot(cb) / std::sqrt(ca.squaredNorm() * cb.squaredNorm());
}

}  // namespace

TEST_CASE("factor_scores: regression and Bartlett scores on a 2-factor CFA") {
  std::mt19937 rng(20260518);
  const Eigen::Index n = 800;
  auto gen = generate(rng, n);

  magmaan::data::RawData raw;
  raw.X = {gen.X};

  auto fp = magmaan::parse::Parser::parse(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);             REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  auto est = magmaan::test::fit(*pt, *mr, *samp).value();

  using magmaan::measures::FactorScoreMethod;
  auto reg = magmaan::measures::factor_scores(*pt, *mr, raw, est,
                                              FactorScoreMethod::Regression);
  auto bar = magmaan::measures::factor_scores(*pt, *mr, raw, est,
                                              FactorScoreMethod::Bartlett);
  auto bad = magmaan::measures::factor_scores(*pt, *mr, raw, est,
                                              FactorScoreMethod::Ebm);
  REQUIRE(reg.has_value());
  REQUIRE(bar.has_value());
  CHECK_FALSE(bad.has_value());
  REQUIRE(reg->scores.size() == 1);
  REQUIRE(bar->scores.size() == 1);
  REQUIRE(reg->scores[0].rows() == n);
  REQUIRE(reg->scores[0].cols() == 2);
  REQUIRE(bar->scores[0].rows() == n);
  REQUIRE(bar->scores[0].cols() == 2);

  // Regression scores are conditional means E[η|y]; with no mean structure
  // they are centered on the sample mean ⇒ exact column means of 0.
  for (Eigen::Index c = 0; c < 2; ++c)
    CHECK(reg->scores[0].col(c).mean() == doctest::Approx(0.0).epsilon(1e-9));

  // Both score sets must be finite, and recover the latent factors that
  // produced the data (the marker indicator fixes the sign).
  for (Eigen::Index c = 0; c < 2; ++c) {
    CHECK(std::isfinite(reg->scores[0](0, c)));
    CHECK(std::isfinite(bar->scores[0](0, c)));
    CHECK(corr(reg->scores[0].col(c), gen.factors.col(c)) > 0.7);
    CHECK(corr(bar->scores[0].col(c), gen.factors.col(c)) > 0.7);
  }

  // Regression scores are shrunk toward the mean relative to Bartlett's
  // unbiased scores ⇒ smaller spread.
  for (Eigen::Index c = 0; c < 2; ++c) {
    const double v_reg = (reg->scores[0].col(c).array() -
                          reg->scores[0].col(c).mean()).square().sum();
    const double v_bar = (bar->scores[0].col(c).array() -
                          bar->scores[0].col(c).mean()).square().sum();
    CHECK(v_reg < v_bar);
  }

  // Bartlett scores are conditionally unbiased: the (Λᵀ Θ⁻¹ Λ)⁻¹ Λᵀ Θ⁻¹
  // coefficient times Λ is the identity. Re-derive the coefficient from the
  // assembled matrices and verify.
  auto ev = magmaan::model::ModelEvaluator::build(*pt, *mr);
  REQUIRE(ev.has_value());
  auto asm_ = ev->assembled(est.theta);
  REQUIRE(asm_.has_value());
  const auto& B = asm_->blocks[0];
  const Eigen::MatrixXd ThiL =
      Eigen::LLT<Eigen::MatrixXd>(B.Theta).solve(B.Lambda);
  const Eigen::MatrixXd M = B.Lambda.transpose() * ThiL;
  const Eigen::MatrixXd C = Eigen::LLT<Eigen::MatrixXd>(M).solve(ThiL.transpose());
  const Eigen::MatrixXd CL = C * B.Lambda;
  for (Eigen::Index r = 0; r < CL.rows(); ++r)
    for (Eigen::Index c = 0; c < CL.cols(); ++c)
      CHECK(CL(r, c) ==
            doctest::Approx(r == c ? 1.0 : 0.0).epsilon(1e-9));

  // The Bartlett scores returned must equal (X − x̄)·Cᵀ.
  const Eigen::MatrixXd expect =
      (gen.X.rowwise() - gen.X.colwise().mean()) * C.transpose();
  for (Eigen::Index i = 0; i < n; i += 97)
    for (Eigen::Index c = 0; c < 2; ++c)
      CHECK(bar->scores[0](i, c) ==
            doctest::Approx(expect(i, c)).epsilon(1e-9));
}

TEST_CASE("factor_scores: missing data is rejected") {
  std::mt19937 rng(7);
  auto gen = generate(rng, 50);
  magmaan::data::RawData raw;
  raw.X = {gen.X};
  raw.mask = {Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(
      50, 6)};

  auto fp = magmaan::parse::Parser::parse("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);             REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt); REQUIRE(mr.has_value());

  magmaan::estimate::Estimates est;
  est.theta = Eigen::VectorXd::Zero(pt->n_free());
  auto fs = magmaan::measures::factor_scores(
      *pt, *mr, raw, est, magmaan::measures::FactorScoreMethod::Regression);
  CHECK(!fs.has_value());
}
