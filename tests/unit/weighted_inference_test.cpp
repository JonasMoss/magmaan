#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <random>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/constraints.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/robust/weighted_chisq.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

struct OneFactorFixture {
  magmaan::spec::LatentStructure pt;
  magmaan::model::MatrixRep rep;
  magmaan::data::RawData raw;
  magmaan::data::SampleStats samp;
};

Eigen::MatrixXd mvn_sample(Eigen::Index n,
                           const Eigen::VectorXd& mu,
                           const Eigen::MatrixXd& Sigma) {
  std::mt19937 rng(42);
  std::normal_distribution<double> z(0.0, 1.0);
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  Eigen::MatrixXd X(n, mu.size());
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::VectorXd zi(mu.size());
    for (Eigen::Index j = 0; j < zi.size(); ++j) zi(j) = z(rng);
    X.row(i) = (mu + L * zi).transpose();
  }
  return X;
}

OneFactorFixture one_factor_fixture(bool meanstructure = false) {
  const std::string syntax =
      meanstructure
          ? "f =~ x1 + x2 + x3 + x4\n"
            "x1 ~ 1\nx2 ~ 1\nx3 ~ 1\nx4 ~ 1"
          : "f =~ x1 + x2 + x3 + x4";
  auto fp = magmaan::parse::Parser::parse(syntax);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());

  Eigen::Vector4d lambda;
  lambda << 0.8, 0.7, 0.9, 0.75;
  Eigen::Matrix4d Sigma = lambda * lambda.transpose();
  Sigma.diagonal().array() += 0.45;
  Eigen::Vector4d mu;
  if (meanstructure) {
    mu << 0.4, -0.2, 0.7, 0.1;
  } else {
    mu << 0.0, 0.0, 0.0, 0.0;
  }

  magmaan::data::RawData raw;
  raw.X.push_back(mvn_sample(420, mu, Sigma));
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());
  return OneFactorFixture{std::move(*pt), std::move(*rep), std::move(raw),
                          std::move(*samp)};
}

std::vector<Eigen::MatrixXd> wls_weights_from_sample(
    const magmaan::data::SampleStats& samp) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(samp.S.size());
  for (const auto& S : samp.S) {
    auto G = magmaan::data::gamma_nt(S);
    REQUIRE(G.has_value());
    Eigen::LDLT<Eigen::MatrixXd> ldlt(*G);
    REQUIRE(ldlt.info() == Eigen::Success);
    REQUIRE(ldlt.isPositive());
    out.push_back(ldlt.solve(Eigen::MatrixXd::Identity(G->rows(), G->cols())));
  }
  return out;
}

double total_n(const magmaan::data::SampleStats& samp) {
  double out = 0.0;
  for (auto n : samp.n_obs) out += static_cast<double>(n);
  return out;
}

// The GLS moment weight (normal-theory weight built from S).
magmaan::estimate::gmm::Weight gls_weight(const magmaan::spec::LatentStructure& pt,
                                const magmaan::model::MatrixRep& rep,
                                const magmaan::data::SampleStats& samp,
                                const Eigen::VectorXd& theta) {
  auto ev = magmaan::model::ModelEvaluator::build(pt, rep);
  REQUIRE(ev.has_value());
  auto w = magmaan::estimate::gmm::normal_theory_weight(*ev, samp, theta);
  REQUIRE(w.has_value());
  return *w;
}

Eigen::Index vech_len(Eigen::Index p) {
  return p * (p + 1) / 2;
}

Eigen::VectorXd vech_lower(const Eigen::Ref<const Eigen::MatrixXd>& M) {
  Eigen::VectorXd out(vech_len(M.rows()));
  Eigen::Index k = 0;
  for (Eigen::Index c = 0; c < M.cols(); ++c) {
    for (Eigen::Index r = c; r < M.rows(); ++r) out(k++) = M(r, c);
  }
  return out;
}

void unpack_vech(const Eigen::Ref<const Eigen::VectorXd>& x,
                 Eigen::Index p,
                 Eigen::Ref<Eigen::MatrixXd> M) {
  Eigen::Index k = 0;
  for (Eigen::Index c = 0; c < p; ++c) {
    for (Eigen::Index r = c; r < p; ++r) {
      M(r, c) = x(k);
      if (r != c) M(c, r) = x(k);
      ++k;
    }
  }
}

struct TestLsLayout {
  bool has_means = false;
  std::vector<Eigen::Index> block_rows;
  std::vector<Eigen::Index> mu_offsets;
  std::vector<Eigen::Index> sigma_offsets;
  Eigen::Index total_mu_rows = 0;
  Eigen::Index total_sigma_rows = 0;
};

TestLsLayout make_test_layout(const magmaan::data::SampleStats& samp,
                              const magmaan::model::ImpliedMoments& moments) {
  TestLsLayout layout;
  for (std::size_t b = 0; b < moments.mu.size(); ++b) {
    if (moments.mu[b].size() > 0 && b < samp.mean.size() &&
        samp.mean[b].size() > 0) {
      layout.has_means = true;
    }
  }
  layout.block_rows.resize(moments.sigma.size());
  layout.mu_offsets.resize(moments.sigma.size());
  layout.sigma_offsets.resize(moments.sigma.size());
  for (std::size_t b = 0; b < moments.sigma.size(); ++b) {
    const Eigen::Index p = moments.sigma[b].rows();
    layout.mu_offsets[b] = layout.total_mu_rows;
    if (layout.has_means) layout.total_mu_rows += p;
    layout.sigma_offsets[b] = layout.total_sigma_rows;
    layout.total_sigma_rows += vech_len(p);
    layout.block_rows[b] = (layout.has_means ? p : 0) + vech_len(p);
  }
  return layout;
}

Eigen::MatrixXd test_jacobian_block(
    const TestLsLayout& layout,
    const magmaan::model::ImpliedMoments& moments,
    const Eigen::MatrixXd& J_sigma,
    const Eigen::MatrixXd& J_mu,
    std::size_t b) {
  const Eigen::Index p = moments.sigma[b].rows();
  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index n_free = J_sigma.cols();
  Eigen::MatrixXd Jb(layout.block_rows[b], n_free);
  Eigen::Index off = 0;
  if (layout.has_means) {
    Jb.topRows(p).setZero();
    if (b < moments.mu.size() && moments.mu[b].size() == p) {
      Jb.topRows(p) = J_mu.block(layout.mu_offsets[b], 0, p, n_free);
    }
    off = p;
  }
  Jb.block(off, 0, pstar, n_free) =
      J_sigma.block(layout.sigma_offsets[b], 0, pstar, n_free);
  return Jb;
}

Eigen::VectorXd test_residual_block(const magmaan::data::SampleStats& samp,
                                    const magmaan::model::ImpliedMoments& moments,
                                    const TestLsLayout& layout,
                                    std::size_t b) {
  const Eigen::Index p = moments.sigma[b].rows();
  const Eigen::Index pstar = vech_len(p);
  Eigen::VectorXd d(layout.block_rows[b]);
  Eigen::Index off = 0;
  if (layout.has_means) {
    const Eigen::VectorXd mu_model =
        (b < moments.mu.size() && moments.mu[b].size() == p)
            ? moments.mu[b]
            : Eigen::VectorXd::Zero(p);
    const Eigen::VectorXd mean_s =
        (b < samp.mean.size() && samp.mean[b].size() == p)
            ? samp.mean[b]
            : Eigen::VectorXd::Zero(p);
    d.head(p) = mu_model - mean_s;
    off = p;
  }
  d.segment(off, pstar) = vech_lower(moments.sigma[b] - samp.S[b]);
  return d;
}

std::vector<Eigen::MatrixXd> test_moment_influence_rows(
    const magmaan::data::RawData& raw,
    const magmaan::data::SampleStats& samp,
    const TestLsLayout& layout) {
  std::vector<Eigen::MatrixXd> out;
  out.reserve(raw.X.size());
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const Eigen::MatrixXd& X = raw.X[b];
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    const Eigen::Index pstar = vech_len(p);
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(n, layout.block_rows[b]);
    const Eigen::VectorXd mean_b =
        (b < samp.mean.size() && samp.mean[b].size() == p)
            ? samp.mean[b]
            : X.colwise().mean().transpose().eval();
    const Eigen::MatrixXd Xc = X.rowwise() - mean_b.transpose();
    const Eigen::VectorXd s_vech = vech_lower(samp.S[b]);
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::VectorXd xi = Xc.row(i).transpose();
      Eigen::Index off = 0;
      if (layout.has_means) {
        Z.block(i, 0, 1, p) = xi.transpose();
        off = p;
      }
      Z.block(i, off, 1, pstar) =
          (vech_lower(Eigen::MatrixXd(xi * xi.transpose())) - s_vech)
              .transpose();
    }
    out.push_back(std::move(Z));
  }
  return out;
}

magmaan::estimate::gmm::Weight adf_weight_from_rows(
    const std::vector<Eigen::MatrixXd>& rows) {
  magmaan::estimate::gmm::Weight out;
  out.reserve(rows.size());
  for (const auto& Z : rows) {
    Eigen::MatrixXd Gamma =
        (Z.transpose() * Z) / static_cast<double>(Z.rows());
    Gamma = 0.5 * (Gamma + Gamma.transpose()).eval();
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Gamma);
    REQUIRE(ldlt.info() == Eigen::Success);
    REQUIRE(ldlt.isPositive());
    out.push_back(ldlt.solve(
        Eigen::MatrixXd::Identity(Gamma.rows(), Gamma.cols())));
  }
  return out;
}

magmaan::estimate::gmm::Weight dwls_weight_from_rows(
    const std::vector<Eigen::MatrixXd>& rows) {
  magmaan::estimate::gmm::Weight out;
  out.reserve(rows.size());
  for (const auto& Z : rows) {
    const Eigen::RowVectorXd gamma =
        Z.array().square().colwise().mean().matrix();
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(Z.cols(), Z.cols());
    for (Eigen::Index k = 0; k < gamma.size(); ++k) {
      REQUIRE(gamma(k) > 0.0);
      W(k, k) = 1.0 / gamma(k);
    }
    out.push_back(std::move(W));
  }
  return out;
}

Eigen::MatrixXd dls_gamma_nt_from_cov(const Eigen::MatrixXd& S,
                                      const TestLsLayout& layout,
                                      std::size_t b) {
  auto g_nt = magmaan::data::gamma_nt(S);
  REQUIRE(g_nt.has_value());
  if (!layout.has_means) return *g_nt;

  Eigen::MatrixXd out =
      Eigen::MatrixXd::Zero(layout.block_rows[b], layout.block_rows[b]);
  const Eigen::Index p = S.rows();
  const Eigen::Index pstar = vech_len(p);
  out.topLeftCorner(p, p) = S;
  out.bottomRightCorner(pstar, pstar) = *g_nt;
  return out;
}

Eigen::MatrixXd weighted_covariance(const Eigen::MatrixXd& X,
                                    Eigen::Index focus,
                                    double eps) {
  const Eigen::Index n = X.rows();
  Eigen::VectorXd w =
      Eigen::VectorXd::Constant(n, (1.0 - eps) / static_cast<double>(n));
  w(focus) += eps;
  const Eigen::VectorXd mean = X.transpose() * w;
  const Eigen::MatrixXd Xc = X.rowwise() - mean.transpose();
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(X.cols(), X.cols());
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::VectorXd xi = Xc.row(i).transpose();
    S.noalias() += w(i) * (xi * xi.transpose());
  }
  return S;
}

Eigen::MatrixXd weighted_empirical_gamma(
    const Eigen::MatrixXd& X,
    const TestLsLayout& layout,
    std::size_t b,
    Eigen::Index focus,
    double eps) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const Eigen::Index pstar = vech_len(p);
  Eigen::VectorXd w =
      Eigen::VectorXd::Constant(n, (1.0 - eps) / static_cast<double>(n));
  w(focus) += eps;
  const Eigen::VectorXd mean = X.transpose() * w;
  const Eigen::MatrixXd Xc = X.rowwise() - mean.transpose();
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(p, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::VectorXd xi = Xc.row(i).transpose();
    S.noalias() += w(i) * (xi * xi.transpose());
  }
  const Eigen::VectorXd s_vech = vech_lower(S);

  Eigen::MatrixXd Gamma =
      Eigen::MatrixXd::Zero(layout.block_rows[b], layout.block_rows[b]);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::VectorXd xi = Xc.row(i).transpose();
    Eigen::RowVectorXd z = Eigen::RowVectorXd::Zero(layout.block_rows[b]);
    Eigen::Index off = 0;
    if (layout.has_means) {
      z.head(p) = xi.transpose();
      off = p;
    }
    z.segment(off, pstar) =
        (vech_lower(Eigen::MatrixXd(xi * xi.transpose())) - s_vech)
            .transpose();
    Gamma.noalias() += w(i) * (z.transpose() * z);
  }
  return 0.5 * (Gamma + Gamma.transpose()).eval();
}

Eigen::MatrixXd weighted_dls_gamma(
    const Eigen::MatrixXd& X,
    const TestLsLayout& layout,
    std::size_t b,
    Eigen::Index focus,
    double eps,
    double a) {
  const Eigen::MatrixXd S = weighted_covariance(X, focus, eps);
  const Eigen::MatrixXd Gnt = dls_gamma_nt_from_cov(S, layout, b);
  const Eigen::MatrixXd Gadf =
      weighted_empirical_gamma(X, layout, b, focus, eps);
  return (1.0 - a) * Gnt + a * Gadf;
}

Eigen::RowVectorXd weighted_empirical_gamma_diag(
    const Eigen::MatrixXd& X,
    const TestLsLayout& layout,
    std::size_t b,
    Eigen::Index focus,
    double eps) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const Eigen::Index pstar = vech_len(p);
  Eigen::VectorXd w =
      Eigen::VectorXd::Constant(n, (1.0 - eps) / static_cast<double>(n));
  w(focus) += eps;
  const Eigen::VectorXd mean = X.transpose() * w;
  const Eigen::MatrixXd Xc = X.rowwise() - mean.transpose();
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(p, p);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::VectorXd xi = Xc.row(i).transpose();
    S.noalias() += w(i) * (xi * xi.transpose());
  }
  const Eigen::VectorXd s_vech = vech_lower(S);

  Eigen::RowVectorXd gamma =
      Eigen::RowVectorXd::Zero(layout.block_rows[b]);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::VectorXd xi = Xc.row(i).transpose();
    Eigen::RowVectorXd z = Eigen::RowVectorXd::Zero(layout.block_rows[b]);
    Eigen::Index off = 0;
    if (layout.has_means) {
      z.head(p) = xi.transpose();
      off = p;
    }
    z.segment(off, pstar) =
        (vech_lower(Eigen::MatrixXd(xi * xi.transpose())) - s_vech)
            .transpose();
    gamma.array() += w(i) * z.array().square();
  }
  return gamma;
}

Eigen::MatrixXd finite_wls_weight_correction(
    const magmaan::data::RawData& raw,
    const TestLsLayout& layout,
    const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& rows,
    std::size_t b) {
  Eigen::MatrixXd correction = Eigen::MatrixXd::Zero(rows.rows(), rows.cols());
  const double eps = 1e-6;
  for (Eigen::Index i = 0; i < rows.rows(); ++i) {
    const Eigen::MatrixXd Gp =
        weighted_empirical_gamma(raw.X[b], layout, b, i, eps);
    const Eigen::MatrixXd Gm =
        weighted_empirical_gamma(raw.X[b], layout, b, i, -eps);
    Eigen::LDLT<Eigen::MatrixXd> lp(Gp);
    Eigen::LDLT<Eigen::MatrixXd> lm(Gm);
    REQUIRE(lp.info() == Eigen::Success);
    REQUIRE(lm.info() == Eigen::Success);
    REQUIRE(lp.isPositive());
    REQUIRE(lm.isPositive());
    const Eigen::MatrixXd Wp =
        lp.solve(Eigen::MatrixXd::Identity(Gp.rows(), Gp.cols()));
    const Eigen::MatrixXd Wm =
        lm.solve(Eigen::MatrixXd::Identity(Gm.rows(), Gm.cols()));
    correction.row(i) = -residual.transpose() *
                        ((Wp - Wm) / (2.0 * eps));
  }
  return correction;
}

Eigen::MatrixXd finite_dwls_weight_correction(
    const magmaan::data::RawData& raw,
    const TestLsLayout& layout,
    const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& rows,
    std::size_t b) {
  Eigen::MatrixXd correction = Eigen::MatrixXd::Zero(rows.rows(), rows.cols());
  const double eps = 1e-6;
  for (Eigen::Index i = 0; i < rows.rows(); ++i) {
    const Eigen::RowVectorXd gp =
        weighted_empirical_gamma_diag(raw.X[b], layout, b, i, eps);
    const Eigen::RowVectorXd gm =
        weighted_empirical_gamma_diag(raw.X[b], layout, b, i, -eps);
    Eigen::MatrixXd Wp = Eigen::MatrixXd::Zero(gp.size(), gp.size());
    Eigen::MatrixXd Wm = Eigen::MatrixXd::Zero(gm.size(), gm.size());
    for (Eigen::Index k = 0; k < gp.size(); ++k) {
      REQUIRE(gp(k) > 0.0);
      REQUIRE(gm(k) > 0.0);
      Wp(k, k) = 1.0 / gp(k);
      Wm(k, k) = 1.0 / gm(k);
    }
    correction.row(i) = -residual.transpose() *
                        ((Wp - Wm) / (2.0 * eps));
  }
  return correction;
}

Eigen::MatrixXd finite_dls_weight_correction(
    const magmaan::data::RawData& raw,
    const TestLsLayout& layout,
    const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& rows,
    std::size_t b,
    double a) {
  Eigen::MatrixXd correction = Eigen::MatrixXd::Zero(rows.rows(), rows.cols());
  const double eps = 1e-6;
  for (Eigen::Index i = 0; i < rows.rows(); ++i) {
    const Eigen::MatrixXd Gp =
        weighted_dls_gamma(raw.X[b], layout, b, i, eps, a);
    const Eigen::MatrixXd Gm =
        weighted_dls_gamma(raw.X[b], layout, b, i, -eps, a);
    Eigen::LDLT<Eigen::MatrixXd> lp(Gp);
    Eigen::LDLT<Eigen::MatrixXd> lm(Gm);
    REQUIRE(lp.info() == Eigen::Success);
    REQUIRE(lm.info() == Eigen::Success);
    REQUIRE(lp.isPositive());
    REQUIRE(lm.isPositive());
    const Eigen::MatrixXd Wp =
        lp.solve(Eigen::MatrixXd::Identity(Gp.rows(), Gp.cols()));
    const Eigen::MatrixXd Wm =
        lm.solve(Eigen::MatrixXd::Identity(Gm.rows(), Gm.cols()));
    correction.row(i) =
        -residual.transpose() * ((Wp - Wm) / (2.0 * eps));
  }
  return correction;
}

Eigen::VectorXd test_ls_gradient(const magmaan::model::ModelEvaluator& ev,
                                 const magmaan::data::SampleStats& samp,
                                 const magmaan::estimate::gmm::Weight& weight,
                                 double N_total,
                                 const Eigen::VectorXd& theta) {
  auto eval = ev.evaluate(theta, true, true);
  REQUIRE(eval.has_value());
  const TestLsLayout layout = make_test_layout(samp, eval->moments);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(theta.size());
  for (std::size_t b = 0; b < samp.S.size(); ++b) {
    const Eigen::MatrixXd Jb = test_jacobian_block(
        layout, eval->moments, eval->J_sigma, eval->J_mu, b);
    const Eigen::VectorXd d_b =
        test_residual_block(samp, eval->moments, layout, b);
    const double w_b = static_cast<double>(samp.n_obs[b]) / N_total;
    g.noalias() += w_b * (Jb.transpose() * weight[b] * d_b);
  }
  return g;
}

Eigen::MatrixXd finite_gls_weight_correction(
    const magmaan::spec::LatentStructure& pt,
    const magmaan::model::MatrixRep& rep,
    const magmaan::data::SampleStats& samp,
    const Eigen::VectorXd& theta,
    const TestLsLayout& layout,
    const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& rows,
    std::size_t b) {
  const Eigen::Index p = samp.S[b].rows();
  const Eigen::Index pstar = vech_len(p);
  const Eigen::Index cov_off = layout.has_means ? p : 0;
  Eigen::MatrixXd correction = Eigen::MatrixXd::Zero(rows.rows(), rows.cols());
  Eigen::MatrixXd dS(p, p);
  const double eps = 1e-6;
  for (Eigen::Index i = 0; i < rows.rows(); ++i) {
    unpack_vech(rows.row(i).segment(cov_off, pstar).transpose(), p, dS);
    magmaan::data::SampleStats plus = samp;
    magmaan::data::SampleStats minus = samp;
    plus.S[b] += eps * dS;
    minus.S[b] -= eps * dS;
    const auto Wp = gls_weight(pt, rep, plus, theta);
    const auto Wm = gls_weight(pt, rep, minus, theta);
    correction.row(i) = -residual.transpose() * ((Wp[b] - Wm[b]) / (2.0 * eps));
  }
  return correction;
}

}  // namespace

TEST_CASE("robust_weighted_moments computes sandwich and U-Gamma for one block") {
  magmaan::estimate::WeightedMomentBlock block;
  block.jacobian.resize(2, 1);
  block.jacobian << 1.0, 0.0;
  block.weight = Eigen::MatrixXd::Identity(2, 2);
  block.gamma = Eigen::MatrixXd::Identity(2, 2);
  block.n_obs = 100;

  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(1, 1);
  auto out = magmaan::estimate::robust_weighted_moments({block}, K, 0.5);
  REQUIRE(out.has_value());
  CHECK(out->df == 1);
  CHECK(out->vcov.rows() == 1);
  CHECK(out->vcov.cols() == 1);
  CHECK(out->vcov(0, 0) == doctest::Approx(0.01));
  CHECK(out->se(0) == doctest::Approx(0.1));
  CHECK(out->chisq_standard == doctest::Approx(50.0));
  REQUIRE(out->eigvals.size() == 1);
  CHECK(out->eigvals(0) == doctest::Approx(1.0));
  CHECK(out->satorra_bentler.scale_c == doctest::Approx(1.0));
  CHECK(out->satorra_bentler.chi2_scaled == doctest::Approx(50.0));
}

TEST_CASE("robust_weighted_moments respects per-block sample weighting and K") {
  magmaan::estimate::WeightedMomentBlock b1;
  b1.jacobian.resize(1, 2);
  b1.jacobian << 1.0, 0.0;
  b1.weight.resize(1, 1);
  b1.weight << 4.0;
  b1.gamma.resize(1, 1);
  b1.gamma << 2.0;
  b1.n_obs = 50;

  magmaan::estimate::WeightedMomentBlock b2;
  b2.jacobian.resize(1, 2);
  b2.jacobian << 0.0, 1.0;
  b2.weight.resize(1, 1);
  b2.weight << 3.0;
  b2.gamma.resize(1, 1);
  b2.gamma << 5.0;
  b2.n_obs = 150;

  Eigen::MatrixXd K(2, 1);
  K << 1.0, 1.0;
  auto out = magmaan::estimate::robust_weighted_moments({b1, b2}, K, 0.25);
  REQUIRE(out.has_value());
  CHECK(out->df == 1);
  REQUIRE(out->vcov.rows() == 2);
  CHECK(out->vcov(0, 0) == doctest::Approx(41.75 / (3.25 * 3.25) / 200.0));
  CHECK(out->vcov(1, 1) == doctest::Approx(out->vcov(0, 0)));
  CHECK(out->vcov(0, 1) == doctest::Approx(out->vcov(0, 0)));
  CHECK(out->chisq_standard == doctest::Approx(50.0));
  REQUIRE(out->eigvals.size() == 1);
  CHECK(out->eigvals(0) >= 0.0);
}

TEST_CASE("weighted_moment_profile_rmsea reports positive spectrum size") {
  magmaan::estimate::WeightedMomentBlock block;
  block.jacobian.resize(3, 1);
  block.jacobian << 1.0, 0.0, 0.0;
  block.weight = Eigen::MatrixXd::Identity(3, 3);
  block.gamma = Eigen::Vector3d(2.0, 3.0, 5.0).asDiagonal();
  block.n_obs = 100;

  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(1, 1);
  Eigen::MatrixXd observed_bread(1, 1);
  observed_bread << 2.0;

  auto out = magmaan::estimate::weighted_moment_profile_rmsea(
      {block}, K, 0.5, observed_bread);
  REQUIRE(out.has_value());

  CHECK(out->df == 2);
  CHECK(out->spectrum_size == 3);
  REQUIRE(out->eigvals.size() == 3);
  CHECK(out->eigvals(0) == doctest::Approx(1.0));
  CHECK(out->eigvals(1) == doctest::Approx(3.0));
  CHECK(out->eigvals(2) == doctest::Approx(5.0));
  CHECK(out->bias_trace == doctest::Approx(9.0));
  CHECK(out->bias_trace_sq == doctest::Approx(35.0));
  CHECK(out->trace_signed == doctest::Approx(9.0));
  CHECK(out->negative_trace_abs == doctest::Approx(0.0));
  CHECK(out->negative_spectrum_size == 0);
  CHECK(out->spectrum_rank == 3);
  CHECK(out->chisq_standard == doctest::Approx(50.0));
  CHECK(out->rmsea ==
        doctest::Approx(std::sqrt((0.5 - 9.0 / 100.0) / 2.0)));
  CHECK(out->rmsea_positive_trace == doctest::Approx(out->rmsea));
  CHECK(out->rmsea_df ==
        doctest::Approx(std::sqrt((0.5 - 2.0 / 100.0) / 2.0)));
  CHECK(out->warnings.empty());
}

TEST_CASE("weighted_moment_profile_rmsea exposes signed trace for indefinite QGamma") {
  magmaan::estimate::WeightedMomentBlock block;
  block.jacobian.resize(2, 1);
  block.jacobian << 1.0, 0.0;
  block.weight = Eigen::MatrixXd::Identity(2, 2);
  block.gamma = Eigen::MatrixXd::Identity(2, 2);
  block.n_obs = 100;

  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(1, 1);
  Eigen::MatrixXd observed_bread(1, 1);
  observed_bread << 0.5;

  auto out = magmaan::estimate::weighted_moment_profile_rmsea(
      {block}, K, 0.20, observed_bread);
  REQUIRE(out.has_value());

  CHECK(out->df == 1);
  CHECK(out->spectrum_size == 1);
  CHECK(out->negative_spectrum_size == 1);
  CHECK(out->spectrum_rank == 2);
  REQUIRE(out->eigvals.size() == 1);
  CHECK(out->eigvals(0) == doctest::Approx(1.0));
  CHECK(out->bias_trace == doctest::Approx(1.0));
  CHECK(out->trace_signed == doctest::Approx(0.0));
  CHECK(out->negative_trace_abs == doctest::Approx(1.0));
  CHECK(out->rmsea == doctest::Approx(std::sqrt(0.20)));
  CHECK(out->rmsea_positive_trace ==
        doctest::Approx(std::sqrt(0.20 - 1.0 / 100.0)));
  CHECK_FALSE(out->warnings.empty());
}

TEST_CASE("weighted_moment_profile_lrt separates nominal df from spectrum size") {
  magmaan::estimate::WeightedProfileRMSEAResult h1;
  h1.profile_hessian = Eigen::MatrixXd::Zero(3, 3);
  h1.gamma = Eigen::Vector3d(2.0, 3.0, 5.0).asDiagonal();
  h1.fmin = 0.10;
  h1.df = 2;
  h1.ntotal = 100;
  h1.n_groups = 1;

  magmaan::estimate::WeightedProfileRMSEAResult h0 = h1;
  h0.profile_hessian = Eigen::MatrixXd::Identity(3, 3);
  h0.fmin = 0.50;
  h0.df = 3;

  auto out = magmaan::estimate::weighted_moment_profile_lrt(h1, h0);
  REQUIRE(out.has_value());

  CHECK(out->df_diff == 1);
  CHECK(out->spectrum_size == 3);
  REQUIRE(out->eigvals.size() == 3);
  CHECK(out->eigvals(0) == doctest::Approx(2.0));
  CHECK(out->eigvals(1) == doctest::Approx(3.0));
  CHECK(out->eigvals(2) == doctest::Approx(5.0));
  CHECK(out->bias_trace == doctest::Approx(10.0));
  CHECK(out->bias_trace_sq == doctest::Approx(38.0));
  CHECK(out->trace_signed == doctest::Approx(10.0));
  CHECK(out->negative_trace_abs == doctest::Approx(0.0));
  CHECK(out->negative_spectrum_size == 0);
  CHECK(out->spectrum_rank == 3);
  CHECK(out->fmin_diff == doctest::Approx(0.40));
  CHECK(out->T_diff == doctest::Approx(40.0));
  CHECK(out->scale_c == doctest::Approx(10.0 / 3.0));
  CHECK(out->T_scaled == doctest::Approx(12.0));
  CHECK(out->p_unscaled ==
        doctest::Approx(magmaan::inference::chi2_pvalue(40.0, 1)));
  CHECK(out->p_scaled ==
        doctest::Approx(magmaan::inference::chi2_pvalue(12.0, 3)));
  CHECK(out->p_mixture ==
        doctest::Approx(magmaan::robust::weighted_chisq_upper(out->eigvals,
                                                              40.0)));
  CHECK(out->scaled_shifted.df == 3);
  CHECK(out->warnings.empty());
}

TEST_CASE("weighted_moment_profile_lrt exposes signed trace for indefinite contrasts") {
  magmaan::estimate::WeightedProfileRMSEAResult h1;
  h1.profile_hessian = Eigen::MatrixXd::Zero(2, 2);
  h1.gamma = Eigen::MatrixXd::Identity(2, 2);
  h1.fmin = 0.10;
  h1.df = 1;
  h1.ntotal = 100;
  h1.n_groups = 1;

  magmaan::estimate::WeightedProfileRMSEAResult h0 = h1;
  h0.profile_hessian.resize(2, 2);
  h0.profile_hessian << 1.0, 0.0,
                        0.0, -0.25;
  h0.fmin = 0.50;
  h0.df = 2;

  auto out = magmaan::estimate::weighted_moment_profile_lrt(h1, h0);
  REQUIRE(out.has_value());

  CHECK(out->df_diff == 1);
  CHECK(out->spectrum_size == 1);
  CHECK(out->negative_spectrum_size == 1);
  CHECK(out->spectrum_rank == 2);
  REQUIRE(out->eigvals.size() == 1);
  CHECK(out->eigvals(0) == doctest::Approx(1.0));
  CHECK(out->bias_trace == doctest::Approx(1.0));
  CHECK(out->trace_signed == doctest::Approx(0.75));
  CHECK(out->negative_trace_abs == doctest::Approx(0.25));
  CHECK(out->p_mixture ==
        doctest::Approx(magmaan::robust::weighted_chisq_upper(out->eigvals,
                                                              40.0)));
  CHECK_FALSE(out->warnings.empty());
}

TEST_CASE("weighted_moment_profile_rmsea_two_metric uses data and projection metrics") {
  magmaan::estimate::WeightedProfileMomentBlock block;
  block.jacobian.resize(3, 1);
  block.jacobian << 1.0, 0.0, 0.0;
  block.data_metric = Eigen::Vector3d(2.0, 3.0, 5.0).asDiagonal();
  block.projection_metric = Eigen::MatrixXd::Identity(3, 3);
  block.gamma = Eigen::Vector3d(7.0, 11.0, 13.0).asDiagonal();
  block.n_obs = 100;

  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(1, 1);
  Eigen::MatrixXd observed_bread(1, 1);
  observed_bread << 1.0;

  auto out = magmaan::estimate::weighted_moment_profile_rmsea_two_metric(
      {block}, K, 0.5, observed_bread);
  REQUIRE(out.has_value());

  CHECK(out->df == 2);
  CHECK(out->spectrum_size == 3);
  REQUIRE(out->eigvals.size() == 3);
  CHECK(out->eigvals(0) == doctest::Approx(7.0));
  CHECK(out->eigvals(1) == doctest::Approx(33.0));
  CHECK(out->eigvals(2) == doctest::Approx(65.0));
  CHECK(out->bias_trace == doctest::Approx(105.0));
  CHECK(out->bias_trace_sq == doctest::Approx(7.0 * 7.0 + 33.0 * 33.0 +
                                              65.0 * 65.0));
}

TEST_CASE("weighted_moment_profile_rmsea_estimated_weight matches finite-difference Q") {
  using namespace magmaan::estimate;
  const Eigen::Index m = 5;
  const Eigen::Index q = 2;
  Eigen::MatrixXd A(m, q);
  A << 1.0, 0.0,
       0.8, 0.2,
       0.6, 0.4,
       0.3, 0.7,
       0.1, 0.9;
  Eigen::VectorXd gamma0(m);
  gamma0 << 0.5, 0.8, 1.2, 0.9, 0.6;
  Eigen::VectorXd u0(m);
  u0 << 0.9, 0.7, 0.55, 0.45, 0.2;  // not in col(A): residual is nonzero

  // Profile score s(x) of the DWLS value function over x = (u, gamma), with the
  // linear-model closed-form inner solve theta_hat = (A'WA)^{-1} A'W u.
  auto profile_score = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
    Eigen::VectorXd u = x.head(m);
    Eigen::VectorXd g = x.tail(m);
    Eigen::MatrixXd W = g.cwiseInverse().asDiagonal();
    Eigen::MatrixXd AtW = A.transpose() * W;
    Eigen::VectorXd theta = (AtW * A).ldlt().solve(AtW * u);
    Eigen::VectorXd r = A * theta - u;
    Eigen::VectorXd s(2 * m);
    s.head(m) = -(W * r);
    s.tail(m) = -0.5 * (r.array().square() / g.array().square()).matrix();
    return s;
  };

  Eigen::VectorXd x0(2 * m);
  x0.head(m) = u0;
  x0.tail(m) = gamma0;

  // Central-difference Jacobian of the profile score == value-function Hessian.
  const double h = 1e-6;
  Eigen::MatrixXd Q_fd(2 * m, 2 * m);
  for (Eigen::Index k = 0; k < 2 * m; ++k) {
    Eigen::VectorXd xp = x0;
    Eigen::VectorXd xm = x0;
    xp(k) += h;
    xm(k) -= h;
    Q_fd.col(k) = (profile_score(xp) - profile_score(xm)) / (2.0 * h);
  }
  Q_fd = 0.5 * (Q_fd + Q_fd.transpose()).eval();

  // Analytic assembly at the fitted point.
  Eigen::MatrixXd W0 = gamma0.cwiseInverse().asDiagonal();
  Eigen::MatrixXd AtW0 = A.transpose() * W0;
  Eigen::VectorXd theta0 = (AtW0 * A).ldlt().solve(AtW0 * u0);
  Eigen::VectorXd r0 = A * theta0 - u0;

  WeightedEstimatedWeightProfileBlock blk;
  blk.jacobian = A;
  blk.weight_diag = gamma0;
  blk.residual = r0;
  blk.gamma = Eigen::MatrixXd::Identity(2 * m, 2 * m);
  blk.n_obs = 200;

  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(q, q);
  Eigen::MatrixXd B = AtW0 * A;  // linear model: observed bread is exact

  auto out = weighted_moment_profile_rmsea_estimated_weight({blk}, K, 0.3, B);
  REQUIRE(out.has_value());
  REQUIRE(out->profile_hessian.rows() == 2 * m);
  CHECK((out->profile_hessian - Q_fd).cwiseAbs().maxCoeff() < 1e-6);

  // Classical df is over the u-moments only, not the doubled extended space.
  CHECK(out->df == static_cast<int>(m - q));
}

TEST_CASE("weighted_moment_profile_rmsea_estimated_weight collapses to fixed weight at r=0") {
  using namespace magmaan::estimate;
  const Eigen::Index m = 5;
  const Eigen::Index q = 2;
  Eigen::MatrixXd A(m, q);
  A << 1.0, 0.0,
       0.8, 0.2,
       0.6, 0.4,
       0.3, 0.7,
       0.1, 0.9;
  Eigen::VectorXd gamma0(m);
  gamma0 << 0.5, 0.8, 1.2, 0.9, 0.6;
  Eigen::Vector2d theta_star(0.5, 0.3);
  Eigen::VectorXd u0 = A * theta_star;  // in col(A): residual vanishes at fit

  Eigen::MatrixXd W0 = gamma0.cwiseInverse().asDiagonal();
  Eigen::MatrixXd B = A.transpose() * W0 * A;
  Eigen::VectorXd r0 = Eigen::VectorXd::Zero(m);  // exact fit
  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(q, q);

  WeightedEstimatedWeightProfileBlock blk;
  blk.jacobian = A;
  blk.weight_diag = gamma0;
  blk.residual = r0;
  blk.gamma = Eigen::MatrixXd::Identity(2 * m, 2 * m);
  blk.n_obs = 200;
  auto ew = weighted_moment_profile_rmsea_estimated_weight({blk}, K, 0.0, B);
  REQUIRE(ew.has_value());

  // Off-diagonal and gamma-gamma blocks are exactly zero when r = 0.
  CHECK(ew->profile_hessian.block(0, m, m, m).cwiseAbs().maxCoeff() == 0.0);
  CHECK(ew->profile_hessian.block(m, m, m, m).cwiseAbs().maxCoeff() == 0.0);

  // The uu block equals the plain fixed-weight profile Hessian.
  WeightedMomentBlock fixed_blk;
  fixed_blk.jacobian = A;
  fixed_blk.weight = W0;
  fixed_blk.gamma = Eigen::MatrixXd::Identity(m, m);
  fixed_blk.n_obs = 200;
  auto fixed = weighted_moment_profile_rmsea({fixed_blk}, K, 0.0, B);
  REQUIRE(fixed.has_value());
  CHECK((ew->profile_hessian.topLeftCorner(m, m) - fixed->profile_hessian)
            .cwiseAbs()
            .maxCoeff() < 1e-12);
}

TEST_CASE("weighted_moment_profile_lrt accepts nested estimated-weight fits") {
  using namespace magmaan::estimate;
  const Eigen::Index m = 5;
  Eigen::MatrixXd A(m, 2);
  A << 1.0, 0.0,
       0.8, 0.2,
       0.6, 0.4,
       0.3, 0.7,
       0.1, 0.9;
  Eigen::VectorXd gamma0(m);
  gamma0 << 0.5, 0.8, 1.2, 0.9, 0.6;
  Eigen::VectorXd u0(m);
  u0 << 0.9, 0.7, 0.55, 0.45, 0.2;
  Eigen::MatrixXd W0 = gamma0.cwiseInverse().asDiagonal();
  Eigen::MatrixXd Gamma_x = Eigen::MatrixXd::Identity(2 * m, 2 * m);

  auto build = [&](const Eigen::MatrixXd& D, double fmin) {
    Eigen::MatrixXd AtW = D.transpose() * W0;
    Eigen::VectorXd theta = (AtW * D).ldlt().solve(AtW * u0);
    Eigen::VectorXd r = D * theta - u0;
    WeightedEstimatedWeightProfileBlock blk;
    blk.jacobian = D;
    blk.weight_diag = gamma0;
    blk.residual = r;
    blk.gamma = Gamma_x;
    blk.n_obs = 200;
    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(D.cols(), D.cols());
    return weighted_moment_profile_rmsea_estimated_weight({blk}, K, fmin,
                                                          AtW * D);
  };

  auto h1 = build(A, 0.20);                 // full model, q = 2, df = 3
  auto h0 = build(A.leftCols(1), 0.35);     // submodel, q = 1, df = 4
  REQUIRE(h1.has_value());
  REQUIRE(h0.has_value());
  CHECK(h1->df == 3);
  CHECK(h0->df == 4);

  auto lrt = weighted_moment_profile_lrt(*h1, *h0);
  REQUIRE(lrt.has_value());
  CHECK(lrt->df_diff == 1);
  CHECK(lrt->spectrum_size > 0);
  CHECK(lrt->T_diff == doctest::Approx(200.0 * (0.35 - 0.20)));
}

TEST_CASE("robust_weighted_moment_ij fixed-weight path matches weighted sandwich") {
  Eigen::MatrixXd G(4, 2);
  G << 1.0, 0.0,
      -1.0, 0.0,
       0.0, 1.0,
       0.0, -1.0;

  magmaan::estimate::WeightedMomentBlock block;
  block.jacobian.resize(2, 1);
  block.jacobian << 1.0, 0.0;
  block.weight = Eigen::MatrixXd::Identity(2, 2);
  block.gamma = (G.transpose() * G) / 4.0;
  block.n_obs = 4;

  magmaan::estimate::WeightedMomentIJBlock ij_block;
  ij_block.jacobian = block.jacobian;
  ij_block.weight = block.weight;
  ij_block.moment_influence = G;
  ij_block.n_obs = block.n_obs;

  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(1, 1);
  Eigen::MatrixXd bread(1, 1);
  bread << 1.0;

  auto fixed = magmaan::estimate::robust_weighted_moments({block}, K, 0.5, bread);
  auto ij = magmaan::estimate::robust_weighted_moment_ij({ij_block}, K, 0.5, bread);
  REQUIRE(fixed.has_value());
  REQUIRE(ij.has_value());
  CHECK(ij->vcov.isApprox(fixed->vcov, 1e-14));
  CHECK(ij->se.isApprox(fixed->se, 1e-14));
  CHECK(ij->chisq_standard == doctest::Approx(fixed->chisq_standard));
  CHECK(ij->df == fixed->df);
  CHECK(ij->eigvals.size() == 0);
}

TEST_CASE("robust_weighted_moment_ij includes casewise weight corrections") {
  Eigen::MatrixXd G(4, 2);
  G << 1.0, 0.0,
      -1.0, 0.0,
       0.0, 1.0,
       0.0, -1.0;
  Eigen::MatrixXd correction = Eigen::MatrixXd::Zero(4, 2);
  correction(0, 0) = 1.0;
  correction(1, 0) = -0.5;

  magmaan::estimate::WeightedMomentIJBlock ij_block;
  ij_block.jacobian.resize(2, 1);
  ij_block.jacobian << 1.0, 0.0;
  ij_block.weight = Eigen::MatrixXd::Identity(2, 2);
  ij_block.moment_influence = G;
  ij_block.weight_correction = correction;
  ij_block.n_obs = 4;

  Eigen::MatrixXd K = Eigen::MatrixXd::Identity(1, 1);
  Eigen::MatrixXd bread(1, 1);
  bread << 1.0;
  auto ij = magmaan::estimate::robust_weighted_moment_ij({ij_block}, K, 0.5, bread);
  REQUIRE(ij.has_value());

  const Eigen::MatrixXd V = G + correction;
  const Eigen::MatrixXd DbK = ij_block.jacobian * K;
  const double expected =
      (DbK.transpose() * V.transpose() * V * DbK)(0, 0) / (4.0 * 4.0);
  CHECK(ij->vcov(0, 0) == doctest::Approx(expected));
}

TEST_CASE("evaluate_ls_objective matches fitted fmin for continuous LS") {
  auto fx = one_factor_fixture();
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};

  auto est_uls = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est_uls.has_value());
  auto f_uls = magmaan::estimate::evaluate_ls_objective(
      fx.pt, fx.rep, fx.samp, est_uls->theta, magmaan::estimate::gmm::Weight{});
  REQUIRE(f_uls.has_value());
  CHECK(*f_uls == doctest::Approx(est_uls->fmin).epsilon(1e-12));

  auto est_gls = magmaan::test::fit_gls(
      fx.pt, fx.rep, fx.samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est_gls.has_value());
  auto f_gls = magmaan::estimate::evaluate_ls_objective(
      fx.pt, fx.rep, fx.samp, est_gls->theta,
      gls_weight(fx.pt, fx.rep, fx.samp, est_gls->theta));
  REQUIRE(f_gls.has_value());
  CHECK(*f_gls == doctest::Approx(est_gls->fmin).epsilon(1e-12));

  magmaan::estimate::gmm::Weight wls = wls_weights_from_sample(fx.samp);
  auto est_wls = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, wls, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est_wls.has_value());
  auto f_wls = magmaan::estimate::evaluate_ls_objective(
      fx.pt, fx.rep, fx.samp, est_wls->theta, wls);
  REQUIRE(f_wls.has_value());
  CHECK(*f_wls == doctest::Approx(est_wls->fmin).epsilon(1e-12));
}

TEST_CASE("evaluate_ls_objective reports data objective under LS constraints") {
  auto fp = magmaan::parse::Parser::parse(
      "f =~ x1 + b2*x2 + b3*x3\n"
      "b2 + b3 == 1.5");
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());

  const Eigen::Vector3d lam(1.0, 0.7, 0.8);
  const Eigen::Vector3d th(0.5, 0.6, 0.4);
  magmaan::data::SampleStats samp;
  samp.S = {lam * lam.transpose() * 1.8 + th.asDiagonal().toDenseMatrix()};
  samp.n_obs = {250};

  const magmaan::optim::OptimOptions opt{
      .max_iter = 5000, .ftol = 1e-14, .gtol = 1e-9};
  auto est = magmaan::test::fit_gmm(
      *pt, *rep, samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est.has_value());

  auto f = magmaan::estimate::evaluate_ls_objective(
      *pt, *rep, samp, est->theta, magmaan::estimate::gmm::Weight{});
  REQUIRE(f.has_value());
  CHECK(*f == doctest::Approx(est->fmin).epsilon(1e-12));
}

TEST_CASE("robust_continuous_ls: raw and supplied Gamma agree for ULS") {
  auto fx = one_factor_fixture();
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
  auto est = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est.has_value());

  auto G = magmaan::data::empirical_gamma(fx.raw.X[0]);
  REQUIRE(G.has_value());
  auto by_gamma = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est, magmaan::estimate::gmm::Weight{}, {*G});
  auto by_raw = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est, magmaan::estimate::gmm::Weight{}, fx.raw);
  REQUIRE(by_gamma.has_value());
  REQUIRE(by_raw.has_value());

  CHECK((by_gamma->vcov - by_raw->vcov).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((by_gamma->se - by_raw->se).cwiseAbs().maxCoeff() < 1e-12);
  CHECK((by_gamma->eigvals - by_raw->eigvals).cwiseAbs().maxCoeff() < 1e-10);
  CHECK(by_raw->chisq_standard ==
        doctest::Approx(2.0 * total_n(fx.samp) * est->fmin));
}

TEST_CASE("continuous_ls_profile_rmsea raw and supplied Gamma agree") {
  auto fx = one_factor_fixture();
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
  auto est = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est.has_value());

  auto G = magmaan::data::empirical_gamma(fx.raw.X[0]);
  REQUIRE(G.has_value());
  auto by_gamma = magmaan::estimate::continuous_ls_profile_rmsea(
      fx.pt, fx.rep, fx.samp, *est, magmaan::estimate::gmm::Weight{}, {*G});
  auto by_raw = magmaan::estimate::continuous_ls_profile_rmsea(
      fx.pt, fx.rep, fx.samp, *est, magmaan::estimate::gmm::Weight{}, fx.raw);
  REQUIRE(by_gamma.has_value());
  REQUIRE(by_raw.has_value());

  CHECK(by_raw->df == by_gamma->df);
  CHECK(by_raw->spectrum_size == by_gamma->spectrum_size);
  CHECK(by_raw->profile_hessian.isApprox(by_gamma->profile_hessian, 1e-12));
  CHECK(by_raw->gamma.isApprox(by_gamma->gamma, 1e-12));
  CHECK(by_raw->eigvals.isApprox(by_gamma->eigvals, 1e-10));
  CHECK(by_raw->bias_trace == doctest::Approx(by_gamma->bias_trace));
  CHECK(by_raw->rmsea == doctest::Approx(by_gamma->rmsea));
  CHECK(by_raw->chisq_standard ==
        doctest::Approx(2.0 * total_n(fx.samp) * est->fmin));
  CHECK(by_raw->spectrum_size == by_raw->eigvals.size());
}

TEST_CASE("continuous_ls_profile_lrt raw and supplied Gamma agree") {
  auto fx = one_factor_fixture();
  auto fp0 = magmaan::parse::Parser::parse(
      "f =~ x1 + a*x2 + a*x3 + x4");
  REQUIRE(fp0.has_value());
  auto pt0 = magmaan::spec::build(*fp0);
  REQUIRE(pt0.has_value());
  auto rep0 = magmaan::model::build_matrix_rep(*pt0);
  REQUIRE(rep0.has_value());

  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
  auto est1 = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  auto est0 = magmaan::test::fit_gmm(
      *pt0, *rep0, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est1.has_value());
  REQUIRE(est0.has_value());

  auto G = magmaan::data::empirical_gamma(fx.raw.X[0]);
  REQUIRE(G.has_value());
  auto by_gamma = magmaan::estimate::continuous_ls_profile_lrt(
      fx.pt, fx.rep, fx.samp, *est1, *pt0, *rep0, *est0,
      magmaan::estimate::gmm::Weight{}, {*G});
  auto by_raw = magmaan::estimate::continuous_ls_profile_lrt(
      fx.pt, fx.rep, fx.samp, *est1, *pt0, *rep0, *est0,
      magmaan::estimate::gmm::Weight{}, fx.raw);
  REQUIRE(by_gamma.has_value());
  REQUIRE(by_raw.has_value());

  CHECK(by_raw->df_diff == 1);
  CHECK(by_raw->spectrum_size == by_gamma->spectrum_size);
  CHECK(by_raw->profile_hessian.isApprox(by_gamma->profile_hessian, 1e-12));
  CHECK(by_raw->gamma.isApprox(by_gamma->gamma, 1e-12));
  CHECK(by_raw->eigvals.isApprox(by_gamma->eigvals, 1e-10));
  CHECK(by_raw->bias_trace == doctest::Approx(by_gamma->bias_trace));
  CHECK(by_raw->T_diff == doctest::Approx(by_gamma->T_diff));
  CHECK(by_raw->p_mixture == doctest::Approx(by_gamma->p_mixture));
  CHECK(by_raw->spectrum_size == by_raw->eigvals.size());
}

TEST_CASE("ml_profile_rmsea covariance-only raw and supplied Gamma agree") {
  auto fx = one_factor_fixture();
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
  auto est = magmaan::test::fit_bounded(
      fx.pt, fx.rep, fx.samp, magmaan::estimate::Bounds{}, opt);
  REQUIRE(est.has_value());

  auto G = magmaan::data::empirical_gamma(fx.raw.X[0]);
  REQUIRE(G.has_value());
  auto by_gamma = magmaan::estimate::ml_profile_rmsea(
      fx.pt, fx.rep, fx.samp, *est, {*G});
  auto by_raw = magmaan::estimate::ml_profile_rmsea(
      fx.pt, fx.rep, fx.samp, *est, fx.raw);
  REQUIRE(by_gamma.has_value());
  REQUIRE(by_raw.has_value());

  CHECK(by_raw->df == 2);
  CHECK(by_raw->spectrum_size == by_gamma->spectrum_size);
  CHECK(by_raw->profile_hessian.isApprox(by_gamma->profile_hessian, 1e-12));
  CHECK(by_raw->gamma.isApprox(by_gamma->gamma, 1e-12));
  CHECK(by_raw->eigvals.isApprox(by_gamma->eigvals, 1e-10));
  CHECK(by_raw->bias_trace == doctest::Approx(by_gamma->bias_trace));
  CHECK(by_raw->rmsea == doctest::Approx(by_gamma->rmsea));
  CHECK(by_raw->chisq_standard ==
        doctest::Approx(2.0 * total_n(fx.samp) * est->fmin));
  CHECK(by_raw->spectrum_size == by_raw->eigvals.size());
}

TEST_CASE("ml_profile_lrt covariance-only raw and supplied Gamma agree") {
  auto fx = one_factor_fixture();
  auto fp0 = magmaan::parse::Parser::parse(
      "f =~ x1 + a*x2 + a*x3 + x4");
  REQUIRE(fp0.has_value());
  auto pt0 = magmaan::spec::build(*fp0);
  REQUIRE(pt0.has_value());
  auto rep0 = magmaan::model::build_matrix_rep(*pt0);
  REQUIRE(rep0.has_value());

  const magmaan::optim::OptimOptions opt{
      .max_iter = 5000, .ftol = 1e-13, .gtol = 1e-8};
  auto est1 = magmaan::test::fit_bounded(
      fx.pt, fx.rep, fx.samp, magmaan::estimate::Bounds{}, opt);
  auto est0 = magmaan::test::fit_bounded(
      *pt0, *rep0, fx.samp, magmaan::estimate::Bounds{}, opt);
  REQUIRE(est1.has_value());
  REQUIRE(est0.has_value());

  auto G = magmaan::data::empirical_gamma(fx.raw.X[0]);
  REQUIRE(G.has_value());
  auto by_gamma = magmaan::estimate::ml_profile_lrt(
      fx.pt, fx.rep, fx.samp, *est1, *pt0, *rep0, *est0, {*G});
  auto by_raw = magmaan::estimate::ml_profile_lrt(
      fx.pt, fx.rep, fx.samp, *est1, *pt0, *rep0, *est0, fx.raw);
  REQUIRE(by_gamma.has_value());
  REQUIRE(by_raw.has_value());

  CHECK(by_raw->df_diff == 1);
  CHECK(by_raw->spectrum_size == by_gamma->spectrum_size);
  CHECK(by_raw->profile_hessian.isApprox(by_gamma->profile_hessian, 1e-12));
  CHECK(by_raw->gamma.isApprox(by_gamma->gamma, 1e-12));
  CHECK(by_raw->eigvals.isApprox(by_gamma->eigvals, 1e-10));
  CHECK(by_raw->bias_trace == doctest::Approx(by_gamma->bias_trace));
  CHECK(by_raw->T_diff == doctest::Approx(by_gamma->T_diff));
  CHECK(by_raw->p_mixture == doctest::Approx(by_gamma->p_mixture));
  CHECK(by_raw->spectrum_size == by_raw->eigvals.size());
}

TEST_CASE("robust_continuous_ls_fixed_weight_ij reduces to observed-bread sandwich") {
  auto fx = one_factor_fixture();
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};

  auto est_uls = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est_uls.has_value());
  auto fixed_uls = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est_uls, magmaan::estimate::gmm::Weight{},
      fx.raw, magmaan::robust::Information::Observed);
  auto ij_uls = magmaan::estimate::robust_continuous_ls_fixed_weight_ij(
      fx.pt, fx.rep, fx.samp, *est_uls, magmaan::estimate::gmm::Weight{},
      fx.raw);
  REQUIRE(fixed_uls.has_value());
  REQUIRE(ij_uls.has_value());
  CHECK(ij_uls->df == fixed_uls->df);
  CHECK(ij_uls->chisq_standard == doctest::Approx(fixed_uls->chisq_standard));
  CHECK(ij_uls->vcov.isApprox(fixed_uls->vcov, 1e-8));
  CHECK(ij_uls->se.isApprox(fixed_uls->se, 1e-8));
  CHECK(ij_uls->eigvals.size() == 0);

  auto est_gls = magmaan::test::fit_gls(
      fx.pt, fx.rep, fx.samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est_gls.has_value());
  const magmaan::estimate::gmm::Weight w_gls =
      gls_weight(fx.pt, fx.rep, fx.samp, est_gls->theta);
  auto fixed_gls = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est_gls, w_gls, fx.raw,
      magmaan::robust::Information::Observed);
  auto ij_gls = magmaan::estimate::robust_continuous_ls_fixed_weight_ij(
      fx.pt, fx.rep, fx.samp, *est_gls, w_gls, fx.raw);
  REQUIRE(fixed_gls.has_value());
  REQUIRE(ij_gls.has_value());
  CHECK(ij_gls->df == fixed_gls->df);
  CHECK(ij_gls->chisq_standard == doctest::Approx(fixed_gls->chisq_standard));
  CHECK(ij_gls->vcov.isApprox(fixed_gls->vcov, 1e-8));
  CHECK(ij_gls->se.isApprox(fixed_gls->se, 1e-8));
  CHECK(ij_gls->eigvals.size() == 0);
}

TEST_CASE("robust_continuous_ls_gls_ij matches finite-difference weight influence") {
  auto fx = one_factor_fixture();
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};

  auto est = magmaan::test::fit_gls(
      fx.pt, fx.rep, fx.samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est.has_value());
  const auto W = gls_weight(fx.pt, fx.rep, fx.samp, est->theta);
  auto got = magmaan::estimate::robust_continuous_ls_gls_ij(
      fx.pt, fx.rep, fx.samp, *est, fx.raw);
  REQUIRE(got.has_value());

  auto ev_or = magmaan::model::ModelEvaluator::build(fx.pt, fx.rep);
  REQUIRE(ev_or.has_value());
  auto eval = ev_or->evaluate(est->theta, true, true);
  REQUIRE(eval.has_value());
  const TestLsLayout layout = make_test_layout(fx.samp, eval->moments);
  const auto rows = test_moment_influence_rows(fx.raw, fx.samp, layout);
  auto con = magmaan::estimate::build_eq_constraints(fx.pt);
  REQUIRE(con.has_value());
  const Eigen::MatrixXd& K = con->K();
  const double N_total = total_n(fx.samp);

  auto grad_at = [&](const Eigen::VectorXd& theta)
      -> magmaan::post_expected<Eigen::VectorXd> {
    return test_ls_gradient(*ev_or, fx.samp, W, N_total, theta);
  };
  auto bread = magmaan::estimate::observed_moment_bread_fd(
      grad_at, est->theta, K);
  REQUIRE(bread.has_value());

  std::vector<magmaan::estimate::WeightedMomentIJBlock> blocks;
  blocks.reserve(fx.samp.S.size());
  for (std::size_t b = 0; b < fx.samp.S.size(); ++b) {
    const Eigen::MatrixXd Jb = test_jacobian_block(
        layout, eval->moments, eval->J_sigma, eval->J_mu, b);
    const Eigen::VectorXd d_b =
        test_residual_block(fx.samp, eval->moments, layout, b);
    blocks.push_back(magmaan::estimate::WeightedMomentIJBlock{
        .jacobian = Jb,
        .weight = W[b],
        .moment_influence = rows[b],
        .weight_correction = finite_gls_weight_correction(
            fx.pt, fx.rep, fx.samp, est->theta, layout, d_b, rows[b], b),
        .n_obs = fx.samp.n_obs[b]});
  }
  auto expected = magmaan::estimate::robust_weighted_moment_ij(
      blocks, K, 2.0 * est->fmin, *bread);
  REQUIRE(expected.has_value());

  CHECK(got->df == expected->df);
  CHECK(got->chisq_standard == doctest::Approx(expected->chisq_standard));
  CHECK(got->vcov.isApprox(expected->vcov, 5e-7));
  CHECK(got->se.isApprox(expected->se, 5e-7));
  CHECK(got->eigvals.size() == 0);
}

TEST_CASE("robust_continuous_ls_wls_ij matches finite-difference empirical "
          "weight influence") {
  const magmaan::optim::OptimOptions opt{
      .max_iter = 5000, .ftol = 1e-13, .gtol = 1e-8};

  for (const bool meanstructure : {false, true}) {
    CAPTURE(meanstructure);
    auto fx = one_factor_fixture(meanstructure);

    auto ev_or = magmaan::model::ModelEvaluator::build(fx.pt, fx.rep);
    REQUIRE(ev_or.has_value());
    auto x0 = magmaan::estimate::simple_start_values(
        fx.pt, fx.rep, fx.samp, {});
    REQUIRE(x0.has_value());
    auto eval0 = ev_or->evaluate(*x0, true, true);
    REQUIRE(eval0.has_value());
    const TestLsLayout layout0 = make_test_layout(fx.samp, eval0->moments);
    const auto rows0 = test_moment_influence_rows(fx.raw, fx.samp, layout0);
    const magmaan::estimate::gmm::Weight W = adf_weight_from_rows(rows0);

    auto est = magmaan::test::fit_gmm(
        fx.pt, fx.rep, fx.samp, W, magmaan::estimate::Bounds{},
        magmaan::estimate::Backend::NloptLbfgs, opt);
    REQUIRE(est.has_value());
    auto got = magmaan::estimate::robust_continuous_ls_wls_ij(
        fx.pt, fx.rep, fx.samp, *est, fx.raw);
    REQUIRE(got.has_value());

    auto eval = ev_or->evaluate(est->theta, true, true);
    REQUIRE(eval.has_value());
    const TestLsLayout layout = make_test_layout(fx.samp, eval->moments);
    const auto rows = test_moment_influence_rows(fx.raw, fx.samp, layout);
    const magmaan::estimate::gmm::Weight W_fit = adf_weight_from_rows(rows);
    REQUIRE(W_fit.size() == W.size());
    for (std::size_t b = 0; b < W.size(); ++b) {
      CHECK(W_fit[b].isApprox(W[b], 1e-12));
    }

    auto con = magmaan::estimate::build_eq_constraints(fx.pt);
    REQUIRE(con.has_value());
    const Eigen::MatrixXd& K = con->K();
    const double N_total = total_n(fx.samp);

    auto grad_at = [&](const Eigen::VectorXd& theta)
        -> magmaan::post_expected<Eigen::VectorXd> {
      return test_ls_gradient(*ev_or, fx.samp, W_fit, N_total, theta);
    };
    auto bread = magmaan::estimate::observed_moment_bread_fd(
        grad_at, est->theta, K);
    REQUIRE(bread.has_value());

    std::vector<magmaan::estimate::WeightedMomentIJBlock> blocks;
    blocks.reserve(fx.samp.S.size());
    for (std::size_t b = 0; b < fx.samp.S.size(); ++b) {
      const Eigen::MatrixXd Jb = test_jacobian_block(
          layout, eval->moments, eval->J_sigma, eval->J_mu, b);
      const Eigen::VectorXd d_b =
          test_residual_block(fx.samp, eval->moments, layout, b);
      blocks.push_back(magmaan::estimate::WeightedMomentIJBlock{
          .jacobian = Jb,
          .weight = W_fit[b],
          .moment_influence = rows[b],
          .weight_correction = finite_wls_weight_correction(
              fx.raw, layout, d_b, rows[b], b),
          .n_obs = fx.samp.n_obs[b]});
    }
    auto expected = magmaan::estimate::robust_weighted_moment_ij(
        blocks, K, 2.0 * est->fmin, *bread);
    REQUIRE(expected.has_value());

    CHECK(got->df == expected->df);
    CHECK(got->chisq_standard == doctest::Approx(expected->chisq_standard));
    CHECK(got->vcov.isApprox(expected->vcov, 1e-5));
    CHECK(got->se.isApprox(expected->se, 1e-5));
    CHECK(got->eigvals.size() == 0);
  }
}

TEST_CASE("robust_continuous_ls_dwls_ij matches finite-difference diagonal "
          "empirical weight influence") {
  const magmaan::optim::OptimOptions opt{
      .max_iter = 5000, .ftol = 1e-13, .gtol = 1e-8};

  for (const bool meanstructure : {false, true}) {
    CAPTURE(meanstructure);
    auto fx = one_factor_fixture(meanstructure);

    auto ev_or = magmaan::model::ModelEvaluator::build(fx.pt, fx.rep);
    REQUIRE(ev_or.has_value());
    auto x0 = magmaan::estimate::simple_start_values(
        fx.pt, fx.rep, fx.samp, {});
    REQUIRE(x0.has_value());
    auto eval0 = ev_or->evaluate(*x0, true, true);
    REQUIRE(eval0.has_value());
    const TestLsLayout layout0 = make_test_layout(fx.samp, eval0->moments);
    const auto rows0 = test_moment_influence_rows(fx.raw, fx.samp, layout0);
    const magmaan::estimate::gmm::Weight W = dwls_weight_from_rows(rows0);

    auto est = magmaan::test::fit_gmm(
        fx.pt, fx.rep, fx.samp, W, magmaan::estimate::Bounds{},
        magmaan::estimate::Backend::NloptLbfgs, opt);
    REQUIRE(est.has_value());
    auto got = magmaan::estimate::robust_continuous_ls_dwls_ij(
        fx.pt, fx.rep, fx.samp, *est, fx.raw);
    REQUIRE(got.has_value());

    auto eval = ev_or->evaluate(est->theta, true, true);
    REQUIRE(eval.has_value());
    const TestLsLayout layout = make_test_layout(fx.samp, eval->moments);
    const auto rows = test_moment_influence_rows(fx.raw, fx.samp, layout);
    const magmaan::estimate::gmm::Weight W_fit = dwls_weight_from_rows(rows);
    REQUIRE(W_fit.size() == W.size());
    for (std::size_t b = 0; b < W.size(); ++b) {
      CHECK(W_fit[b].isApprox(W[b], 1e-12));
    }

    auto con = magmaan::estimate::build_eq_constraints(fx.pt);
    REQUIRE(con.has_value());
    const Eigen::MatrixXd& K = con->K();
    const double N_total = total_n(fx.samp);

    auto grad_at = [&](const Eigen::VectorXd& theta)
        -> magmaan::post_expected<Eigen::VectorXd> {
      return test_ls_gradient(*ev_or, fx.samp, W_fit, N_total, theta);
    };
    auto bread = magmaan::estimate::observed_moment_bread_fd(
        grad_at, est->theta, K);
    REQUIRE(bread.has_value());

    std::vector<magmaan::estimate::WeightedMomentIJBlock> blocks;
    blocks.reserve(fx.samp.S.size());
    for (std::size_t b = 0; b < fx.samp.S.size(); ++b) {
      const Eigen::MatrixXd Jb = test_jacobian_block(
          layout, eval->moments, eval->J_sigma, eval->J_mu, b);
      const Eigen::VectorXd d_b =
          test_residual_block(fx.samp, eval->moments, layout, b);
      blocks.push_back(magmaan::estimate::WeightedMomentIJBlock{
          .jacobian = Jb,
          .weight = W_fit[b],
          .moment_influence = rows[b],
          .weight_correction = finite_dwls_weight_correction(
              fx.raw, layout, d_b, rows[b], b),
          .n_obs = fx.samp.n_obs[b]});
    }
    auto expected = magmaan::estimate::robust_weighted_moment_ij(
        blocks, K, 2.0 * est->fmin, *bread);
    REQUIRE(expected.has_value());

    CHECK(got->df == expected->df);
    CHECK(got->chisq_standard == doctest::Approx(expected->chisq_standard));
    CHECK(got->vcov.isApprox(expected->vcov, 5e-6));
    CHECK(got->se.isApprox(expected->se, 5e-6));
    CHECK(got->eigvals.size() == 0);
  }
}

TEST_CASE("robust_continuous_ls_dls_ij matches finite-difference mixed "
          "Gamma weight influence") {
  const magmaan::optim::OptimOptions opt{
      .max_iter = 5000, .ftol = 1e-13, .gtol = 1e-8};
  const magmaan::estimate::frontier::DlsWeightOptions dls_opts{.a = 0.35};

  for (const bool meanstructure : {false, true}) {
    CAPTURE(meanstructure);
    auto fx = one_factor_fixture(meanstructure);

    auto ev_or = magmaan::model::ModelEvaluator::build(fx.pt, fx.rep);
    REQUIRE(ev_or.has_value());
    auto x0 = magmaan::estimate::simple_start_values(
        fx.pt, fx.rep, fx.samp, {});
    REQUIRE(x0.has_value());
    auto W_or = magmaan::estimate::frontier::dls_weight(
        *ev_or, fx.samp, fx.raw, *x0, dls_opts);
    REQUIRE(W_or.has_value());
    const magmaan::estimate::gmm::Weight W = *W_or;

    auto est = magmaan::test::fit_gmm(
        fx.pt, fx.rep, fx.samp, W, magmaan::estimate::Bounds{},
        magmaan::estimate::Backend::NloptLbfgs, opt);
    REQUIRE(est.has_value());
    auto got = magmaan::estimate::robust_continuous_ls_dls_ij(
        fx.pt, fx.rep, fx.samp, *est, fx.raw, dls_opts);
    REQUIRE(got.has_value());

    auto eval = ev_or->evaluate(est->theta, true, true);
    REQUIRE(eval.has_value());
    const TestLsLayout layout = make_test_layout(fx.samp, eval->moments);
    const auto rows = test_moment_influence_rows(fx.raw, fx.samp, layout);
    auto W_fit_or = magmaan::estimate::frontier::dls_weight(
        *ev_or, fx.samp, fx.raw, est->theta, dls_opts);
    REQUIRE(W_fit_or.has_value());
    const magmaan::estimate::gmm::Weight W_fit = *W_fit_or;
    REQUIRE(W_fit.size() == W.size());
    for (std::size_t b = 0; b < W.size(); ++b) {
      CHECK(W_fit[b].isApprox(W[b], 1e-12));
    }

    auto con = magmaan::estimate::build_eq_constraints(fx.pt);
    REQUIRE(con.has_value());
    const Eigen::MatrixXd& K = con->K();
    const double N_total = total_n(fx.samp);

    auto grad_at = [&](const Eigen::VectorXd& theta)
        -> magmaan::post_expected<Eigen::VectorXd> {
      return test_ls_gradient(*ev_or, fx.samp, W_fit, N_total, theta);
    };
    auto bread = magmaan::estimate::observed_moment_bread_fd(
        grad_at, est->theta, K);
    REQUIRE(bread.has_value());

    std::vector<magmaan::estimate::WeightedMomentIJBlock> blocks;
    blocks.reserve(fx.samp.S.size());
    for (std::size_t b = 0; b < fx.samp.S.size(); ++b) {
      const Eigen::MatrixXd Jb = test_jacobian_block(
          layout, eval->moments, eval->J_sigma, eval->J_mu, b);
      const Eigen::VectorXd d_b =
          test_residual_block(fx.samp, eval->moments, layout, b);
      blocks.push_back(magmaan::estimate::WeightedMomentIJBlock{
          .jacobian = Jb,
          .weight = W_fit[b],
          .moment_influence = rows[b],
          .weight_correction = finite_dls_weight_correction(
              fx.raw, layout, d_b, rows[b], b, dls_opts.a),
          .n_obs = fx.samp.n_obs[b]});
    }
    auto expected = magmaan::estimate::robust_weighted_moment_ij(
        blocks, K, 2.0 * est->fmin, *bread);
    REQUIRE(expected.has_value());

    CHECK(got->df == expected->df);
    CHECK(got->chisq_standard == doctest::Approx(expected->chisq_standard));
    CHECK(got->vcov.isApprox(expected->vcov, 1e-5));
    CHECK(got->se.isApprox(expected->se, 1e-5));
    CHECK(got->eigvals.size() == 0);
  }
}

TEST_CASE("robust_continuous_ls: GLS and WLS preserve continuous LS statistic scale") {
  auto fx = one_factor_fixture();
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};

  auto est_gls = magmaan::test::fit_gls(
      fx.pt, fx.rep, fx.samp, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est_gls.has_value());
  auto rob_gls = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est_gls,
      gls_weight(fx.pt, fx.rep, fx.samp, est_gls->theta), fx.raw);
  REQUIRE(rob_gls.has_value());
  CHECK(rob_gls->chisq_standard ==
        doctest::Approx(2.0 * total_n(fx.samp) * est_gls->fmin));
  CHECK(rob_gls->se.allFinite());

  magmaan::estimate::gmm::Weight wls = wls_weights_from_sample(fx.samp);
  auto est_wls = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, wls, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est_wls.has_value());
  auto rob_wls = magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est_wls, wls, fx.raw);
  REQUIRE(rob_wls.has_value());
  CHECK(rob_wls->chisq_standard ==
        doctest::Approx(2.0 * total_n(fx.samp) * est_wls->fmin));
  CHECK(rob_wls->se.allFinite());
}

TEST_CASE("robust_continuous_ls: validates Gamma block dimensions") {
  auto fx = one_factor_fixture();
  const magmaan::optim::OptimOptions opt{
      .max_iter = 4000, .ftol = 1e-13, .gtol = 1e-8};
  auto est = magmaan::test::fit_gmm(
      fx.pt, fx.rep, fx.samp, {}, magmaan::estimate::Bounds{},
      magmaan::estimate::Backend::NloptLbfgs, opt);
  REQUIRE(est.has_value());
  std::vector<Eigen::MatrixXd> bad{
      Eigen::MatrixXd::Identity(2, 2)};
  CHECK_FALSE(magmaan::estimate::robust_continuous_ls(
      fx.pt, fx.rep, fx.samp, *est, magmaan::estimate::gmm::Weight{}, bad).has_value());
}
