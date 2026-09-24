#include <doctest/doctest.h>

#include <random>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

// ============================================================================
// `inference::information_expected_per_case_blocks` computes
//
//   I_b[a,c] = ½·tr(Σ_b⁻¹ Δ_a Σ_b⁻¹ Δ_c) + ν_a' Σ_b⁻¹ ν_c   (per case)
//
// via the normal-theory whitened Jacobian: with Y_b = Fᵀ[dμ/dθ ; dvech(Σ)/dθ]_b
// and F Fᵀ = W_b the NormalTheory `gmm::BlockWeight` for Σ_b, I_b = Y_bᵀ Y_b.
//
// That identity rests on the whitening convention inside `BlockWeight` — the
// 1/√2 on vech diagonal entries, and the mean block being L⁻¹ rather than
// Σ⁻¹ — so it is worth pinning directly against the explicit trace form rather
// than trusting that downstream SEs would notice. `reference_per_case_blocks`
// below is the independent implementation: form Σ⁻¹ densely, build
// T_k = Σ⁻¹·unvech(J[:,k]) per parameter, and reduce each pair with an explicit
// elementwise trace. It deliberately shares no code with the implementation.
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

constexpr Eigen::Index vech_len_(Eigen::Index p) { return p * (p + 1) / 2; }
constexpr Eigen::Index vech_index_(Eigen::Index p, Eigen::Index r,
                                   Eigen::Index c) {
  return c * p - (c * (c - 1)) / 2 + (r - c);
}

// Explicit-trace reference: Σ⁻¹ dense, T_k = Σ⁻¹·unvech(J[:,k]), pairwise
// elementwise trace, plus the ν' Σ⁻¹ ν mean term at its factor of 2.
std::vector<Eigen::MatrixXd> reference_per_case_blocks(
    const ModelEvaluator& ev, const Eigen::VectorXd& theta) {
  const std::size_t n_free = ev.n_free();
  auto sm  = ev.sigma(theta);
  auto J_  = ev.dsigma_dtheta(theta);
  auto Jm_ = ev.dmu_dtheta(theta);
  std::vector<Eigen::MatrixXd> out;
  if (!sm.has_value() || !J_.has_value() || !Jm_.has_value()) return out;
  const Eigen::MatrixXd& J   = *J_;
  const Eigen::MatrixXd& Jmu = *Jm_;
  const bool         has_means = (Jmu.size() > 0);
  const std::size_t  n_blocks  = sm->sigma.size();
  const Eigen::Index nf        = static_cast<Eigen::Index>(n_free);

  std::vector<Eigen::MatrixXd> SigmaInv(n_blocks);
  std::vector<Eigen::Index>    p_dim(n_blocks, 0), vech_off(n_blocks, 0),
      mu_off(n_blocks, 0);
  Eigen::Index running = 0, running_p = 0;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const Eigen::MatrixXd Sb =
        0.5 * (sm->sigma[b] + sm->sigma[b].transpose());
    const Eigen::Index p = Sb.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(Sb);
    if (llt.info() != Eigen::Success) return {};
    SigmaInv[b] = llt.solve(Eigen::MatrixXd::Identity(p, p));
    p_dim[b]    = p;
    vech_off[b] = running;
    mu_off[b]   = running_p;
    running   += vech_len_(p);
    running_p += p;
  }

  std::vector<std::vector<Eigen::MatrixXd>> T(
      n_free, std::vector<Eigen::MatrixXd>(n_blocks));
  for (std::size_t k = 0; k < n_free; ++k) {
    for (std::size_t b = 0; b < n_blocks; ++b) {
      const Eigen::Index p = p_dim[b];
      Eigen::MatrixXd M = Eigen::MatrixXd::Zero(p, p);
      for (Eigen::Index c = 0; c < p; ++c) {
        for (Eigen::Index r = c; r < p; ++r) {
          const double v = J(vech_off[b] + vech_index_(p, r, c),
                             static_cast<Eigen::Index>(k));
          M(r, c) = v;
          if (r != c) M(c, r) = v;
        }
      }
      T[k][b].noalias() = SigmaInv[b] * M;
    }
  }

  out.assign(n_blocks, Eigen::MatrixXd::Zero(nf, nf));
  for (std::size_t a = 0; a < n_free; ++a) {
    for (std::size_t c = a; c < n_free; ++c) {
      for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        double pb = (T[a][blk].transpose().array() * T[c][blk].array()).sum();
        if (has_means) {
          const Eigen::VectorXd nu_a =
              Jmu.col(static_cast<Eigen::Index>(a))
                  .segment(mu_off[blk], p_dim[blk]);
          const Eigen::VectorXd nu_c =
              Jmu.col(static_cast<Eigen::Index>(c))
                  .segment(mu_off[blk], p_dim[blk]);
          pb += 2.0 * nu_a.dot(SigmaInv[blk] * nu_c);
        }
        const double v = 0.5 * pb;
        out[blk](static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(c)) = v;
        if (a != c)
          out[blk](static_cast<Eigen::Index>(c), static_cast<Eigen::Index>(a)) = v;
      }
    }
  }
  return out;
}

void check_against_reference(const std::string& syntax, int n_groups,
                             bool meanstructure, unsigned seed) {
  INFO("syntax=" << syntax << " n_groups=" << n_groups
                 << " meanstructure=" << meanstructure);
  auto fp = Parser::parse(syntax);
  REQUIRE_OK(fp);
  magmaan::spec::BuildOptions bo;
  bo.meanstructure = meanstructure;
  bo.n_groups      = n_groups;
  auto pt = build(*fp, bo);
  REQUIRE_OK(pt);
  auto mr = build_matrix_rep(*pt);
  REQUIRE_OK(mr);
  auto evr = ModelEvaluator::build(*pt, *mr);
  REQUIRE_OK(evr);
  const auto& ev = *evr;

  const Eigen::Index nf = static_cast<Eigen::Index>(ev.n_free());
  const Eigen::VectorXd theta = Eigen::VectorXd::Constant(nf, 0.8);

  auto m0 = ev.sigma(theta);
  REQUIRE_OK(m0);
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd(0.0, 0.02);
  SampleStats samp;
  for (std::size_t b = 0; b < m0->sigma.size(); ++b) {
    Eigen::MatrixXd S = m0->sigma[b];
    for (Eigen::Index i = 0; i < S.rows(); ++i)
      for (Eigen::Index j = 0; j <= i; ++j) {
        S(i, j) += nd(rng);
        S(j, i) = S(i, j);
      }
    samp.S.push_back(S);
    samp.n_obs.push_back(300 + 41 * static_cast<std::int64_t>(b));
    if (b < m0->mu.size() && m0->mu[b].size() > 0) {
      Eigen::VectorXd mn = m0->mu[b];
      for (Eigen::Index i = 0; i < mn.size(); ++i) mn(i) += nd(rng);
      samp.mean.push_back(mn);
    }
  }

  magmaan::estimate::Estimates est;
  est.theta = theta;

  auto got = magmaan::inference::information_expected_per_case_blocks(
      *pt, *mr, samp, est);
  REQUIRE_OK(got);
  const auto ref = reference_per_case_blocks(ev, theta);
  REQUIRE(ref.size() == got->size());
  REQUIRE(!ref.empty());

  for (std::size_t b = 0; b < ref.size(); ++b) {
    CAPTURE(b);
    const double scale = std::max(1.0, ref[b].cwiseAbs().maxCoeff());
    CHECK((ref[b] - (*got)[b]).cwiseAbs().maxCoeff() / scale < 1e-10);
    // Gram-form construction: symmetric, and PSD by construction.
    CHECK(((*got)[b] - (*got)[b].transpose()).cwiseAbs().maxCoeff() /
              scale < 1e-14);
  }
}

}  // namespace

TEST_CASE("expected information via the whitened Jacobian matches the trace form") {
  const std::string one_factor = "f =~ x1 + x2 + x3 + x4";
  const std::string two_factor =
      "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nf1 ~~ f2";

  SUBCASE("1 factor, covariance only")   { check_against_reference(one_factor, 1, false, 31); }
  SUBCASE("1 factor, mean structure")    { check_against_reference(one_factor, 1, true,  32); }
  SUBCASE("1 factor, 2 groups")          { check_against_reference(one_factor, 2, false, 33); }
  SUBCASE("1 factor, 3 groups + means")  { check_against_reference(one_factor, 3, true,  34); }
  SUBCASE("2 factors, covariance only")  { check_against_reference(two_factor, 1, false, 35); }
  SUBCASE("2 factors, mean structure")   { check_against_reference(two_factor, 1, true,  36); }
  SUBCASE("2 factors, 4 groups + means") { check_against_reference(two_factor, 4, true,  37); }
}
