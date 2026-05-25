#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/data/pairwise_cov.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/spec/build.hpp"

namespace {

struct BuiltModel {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep      rep;
};

BuiltModel build_cfa_5() {
  // 5-indicator CFA — overidentified (df > 0), enough to make U-factor
  // non-trivial without being slow.
  auto fp = magmaan::parse::Parser::parse("f =~ x1 + x2 + x3 + x4 + x5");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return BuiltModel{std::move(*pt), std::move(*rep)};
}

Eigen::MatrixXd sample_mvn(std::mt19937& rng, const Eigen::MatrixXd& Sigma,
                           int n) {
  const Eigen::Index p = Sigma.rows();
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd X(n, p);
  Eigen::VectorXd z(p);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index k = 0; k < p; ++k) z(k) = nd(rng);
    X.row(i) = (L * z).transpose();
  }
  return X;
}

// Common: simulate complete-data CFA, build pw, fit by GLS, return all
// pieces needed for inference tests.
struct Fixture {
  BuiltModel model;
  magmaan::data::RawData raw;
  magmaan::data::PairwiseSampleStats pw;
  magmaan::data::SampleStats samp;
  magmaan::estimate::Estimates est;
};

Fixture build_complete_fixture(std::uint32_t seed) {
  Fixture fx;
  fx.model = build_cfa_5();
  std::mt19937 rng(seed);
  // Mild correlation structure.
  Eigen::VectorXd lam(5);
  lam << 0.7, 0.75, 0.65, 0.8, 0.7;
  Eigen::MatrixXd Sigma = lam * lam.transpose();
  Sigma.diagonal() += (Eigen::VectorXd::Constant(5, 1.0).array() - lam.array().square()).matrix();
  Eigen::MatrixXd X = sample_mvn(rng, Sigma, 400);
  fx.raw.X.push_back(X);
  // No mask: complete data.

  auto pw_or = magmaan::data::pairwise_sample_stats(fx.raw);
  REQUIRE(pw_or.has_value());
  fx.pw = std::move(*pw_or);

  fx.samp.S = fx.pw.S;
  fx.samp.mean = fx.pw.mean;
  fx.samp.n_obs = fx.pw.n_obs;

  auto x0 = magmaan::estimate::simple_start_values(fx.model.pt, fx.model.rep,
                                                   fx.samp, {});
  REQUIRE(x0.has_value());
  auto fit = magmaan::estimate::fit_gls(fx.model.pt, fx.model.rep, fx.samp,
                                        *x0);
  REQUIRE(fit.has_value());
  fx.est = std::move(*fit);
  return fx;
}

// Same fixture but with MCAR missingness in x2..x5 (keep x1 intact).
Fixture build_missing_fixture(std::uint32_t seed) {
  Fixture fx = build_complete_fixture(seed);
  std::mt19937 rng(seed + 11u);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const Eigen::Index n = fx.raw.X[0].rows();
  const Eigen::Index p = fx.raw.X[0].cols();
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(n, p);
  M.setOnes();
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index c = 1; c < p; ++c) {
      if (u(rng) < 0.15) {
        M(r, c) = 0;
        fx.raw.X[0](r, c) = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }
  fx.raw.mask.push_back(M);
  // Rebuild pw + samp + fit on the masked data.
  auto pw_or = magmaan::data::pairwise_sample_stats(fx.raw);
  REQUIRE(pw_or.has_value());
  fx.pw = std::move(*pw_or);
  fx.samp.S = fx.pw.S;
  fx.samp.mean = fx.pw.mean;
  fx.samp.n_obs = fx.pw.n_obs;
  auto x0 = magmaan::estimate::simple_start_values(fx.model.pt, fx.model.rep,
                                                   fx.samp, {});
  REQUIRE(x0.has_value());
  auto fit = magmaan::estimate::fit_gls_pairwise(fx.model.pt, fx.model.rep,
                                                  fx.raw, fx.pw, *x0);
  REQUIRE(fit.has_value());
  fx.est = std::move(*fit);
  return fx;
}

}  // namespace

TEST_CASE("build_u_factor pairwise: complete-data degeneracy vs Unstructured") {
  // data::gamma_nt_pairwise uses pw.S (the pairwise sample covariance) as
  // its inner Σ. On complete data pw.S = S, so Pairwise bread collapses to
  // Unstructured bread (Γ_NT(S)), NOT Structured (which uses model-implied
  // Σ̂_b). This is the right reference comparison.
  auto fx = build_complete_fixture(20260620);

  magmaan::robust::InferenceSpec spec_unstr{
      magmaan::robust::Information::Expected,
      magmaan::robust::WeightMoments::Unstructured,
      magmaan::robust::ScoreCovariance::ModelImplied};
  magmaan::robust::InferenceSpec spec_pw{
      magmaan::robust::Information::Expected,
      magmaan::robust::WeightMoments::Pairwise,
      magmaan::robust::ScoreCovariance::ModelImplied};

  auto uf_u_or = magmaan::robust::build_u_factor(fx.model.pt, fx.model.rep,
                                                 fx.samp, fx.est, spec_unstr);
  REQUIRE(uf_u_or.has_value());
  auto uf_p_or = magmaan::robust::build_u_factor(fx.model.pt, fx.model.rep,
                                                 fx.samp, fx.est, fx.raw,
                                                 fx.pw, spec_pw);
  REQUIRE(uf_p_or.has_value());

  CHECK(uf_u_or->df == uf_p_or->df);
  CHECK(uf_u_or->B.rows() == uf_p_or->B.rows());
  CHECK(uf_u_or->B.cols() == uf_p_or->B.cols());
  // B is determined up to a sign per column; compare via U = B·Bᵀ.
  const Eigen::MatrixXd U_u = uf_u_or->B * uf_u_or->B.transpose();
  const Eigen::MatrixXd U_p = uf_p_or->B * uf_p_or->B.transpose();
  CHECK((U_u - U_p).cwiseAbs().maxCoeff() < 1e-10);
}

TEST_CASE("reduced_gamma_nt_pairwise: complete-data degeneracy vs reduced_gamma_nt") {
  auto fx = build_complete_fixture(20260621);

  magmaan::robust::InferenceSpec spec_struct{
      magmaan::robust::Information::Expected,
      magmaan::robust::WeightMoments::Structured,
      magmaan::robust::ScoreCovariance::ModelImplied};
  auto uf_or = magmaan::robust::build_u_factor(fx.model.pt, fx.model.rep,
                                               fx.samp, fx.est, spec_struct);
  REQUIRE(uf_or.has_value());

  auto M_full_or = magmaan::robust::reduced_gamma_nt(*uf_or);
  REQUIRE(M_full_or.has_value());
  auto M_pw_or =
      magmaan::robust::reduced_gamma_nt_pairwise(*uf_or, fx.raw, fx.pw);
  REQUIRE(M_pw_or.has_value());

  CHECK((*M_full_or - *M_pw_or).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("reduced_gamma_nt_pairwise: matches materialised B'·Γ_NT^pw·B under missingness") {
  auto fx = build_missing_fixture(20260622);

  // data::gamma_nt_pairwise uses pw.S as its inner Σ. The reducer applies
  // Γ_NT(M_b) where M_b = Σ̂_b (Structured) or S_b (Unstructured). For the
  // cross-check we want the reducer's inner Σ to match pw.S, so use
  // Unstructured (which reads samp.S = pw.S in this fixture).
  magmaan::robust::InferenceSpec spec{
      magmaan::robust::Information::Expected,
      magmaan::robust::WeightMoments::Unstructured,
      magmaan::robust::ScoreCovariance::ModelImplied};
  auto uf_or = magmaan::robust::build_u_factor(fx.model.pt, fx.model.rep,
                                               fx.samp, fx.est, spec);
  REQUIRE(uf_or.has_value());

  auto gnt_pw_or = magmaan::data::gamma_nt_pairwise(fx.raw, fx.pw);
  REQUIRE(gnt_pw_or.has_value());
  // Single-block fixture — assemble materialised B'·Γ_NT^pw·B against the σ
  // slice. The reducer's μ-block is independent of the pattern (uses Σ̂_b
  // for the μ-rows), so we compare just the σ contribution by extracting
  // the σ-rows of B.
  const auto& blk = uf_or->blocks[0];
  Eigen::MatrixXd B_sigma = uf_or->B.middleRows(blk.row_offset, blk.pstar);
  Eigen::MatrixXd M_mat   = B_sigma.transpose() * gnt_pw_or->at(0) * B_sigma;

  auto M_op_or =
      magmaan::robust::reduced_gamma_nt_pairwise(*uf_or, fx.raw, fx.pw);
  REQUIRE(M_op_or.has_value());

  // The operator path produces (B_μ' Σ̂_b B_μ + B_σ' Γ_NT^pw B_σ); the
  // complete-data μ contribution is identical between full reducer and
  // any restricted comparison. For a cov-only model (has_means = false)
  // these match exactly; here we have means on by default — subtract the
  // μ-block contribution from the operator path before comparing.
  Eigen::MatrixXd M_op_minus_mu = *M_op_or;
  if (uf_or->has_means) {
    Eigen::MatrixXd B_mu = uf_or->B.middleRows(blk.mu_off, blk.p);
    M_op_minus_mu.noalias() -= B_mu.transpose() * blk.Sigma_hat * B_mu;
  }
  CHECK((M_op_minus_mu - M_mat).cwiseAbs().maxCoeff() < 1e-10);
}

TEST_CASE("Pairwise bread + Pairwise meat: U·Γ_NT^pw eigenvalues = 1") {
  auto fx = build_missing_fixture(20260623);
  magmaan::robust::InferenceSpec spec{
      magmaan::robust::Information::Expected,
      magmaan::robust::WeightMoments::Pairwise,
      magmaan::robust::ScoreCovariance::ModelImplied};
  auto uf_or = magmaan::robust::build_u_factor(fx.model.pt, fx.model.rep,
                                               fx.samp, fx.est, fx.raw,
                                               fx.pw, spec);
  REQUIRE(uf_or.has_value());

  auto M_or =
      magmaan::robust::reduced_gamma_nt_pairwise(*uf_or, fx.raw, fx.pw);
  REQUIRE(M_or.has_value());
  auto eig_or = magmaan::robust::ugamma_eigenvalues(*M_or);
  REQUIRE(eig_or.has_value());

  // df eigenvalues should be ≈ 1 (rank-df projector identity for the
  // bread-meat collapse). Tolerance allowed for the μ-block compromise
  // (μ uses Σ̂_b, not the pairwise μ ACOV) — the collapse is exact on the
  // σ-block.
  REQUIRE(eig_or->size() == uf_or->df);
  for (Eigen::Index k = 0; k < eig_or->size(); ++k) {
    CHECK(std::abs((*eig_or)(k) - 1.0) < 1e-8);
  }
}
