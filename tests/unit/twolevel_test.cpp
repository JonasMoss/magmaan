#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/estimate/twolevel.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"

// Private src/ vech helper, included by relative path (src/ is private to the
// magmaan target; the inline functions are header-only).
#include "../../src/detail_vech.hpp"

namespace tl = magmaan::estimate::twolevel;
using magmaan::data::ClusterGroupStats;
using magmaan::data::ClusterSampleStats;
using magmaan::data::ClusterSizePattern;
using magmaan::model::BlockInfo;
using magmaan::model::BlockLevel;
using magmaan::model::ImpliedMoments;
using magmaan::model::MatrixRep;

namespace {

using Cluster = std::vector<Eigen::VectorXd>;          // rows of one cluster
using Clusters = std::vector<Cluster>;

// A minimal MatrixRep carrying only the block-role map (all level_block_pairs
// reads). Block 0 = within, block 1 = between, same group.
MatrixRep two_block_rep() {
  MatrixRep rep;
  rep.block_info = {BlockInfo{0, 0, BlockLevel::Within},
                    BlockInfo{0, 1, BlockLevel::Between}};
  return rep;
}

// Build the per-group two-level sufficient statistics from raw cluster rows
// (the job Stream B will do from a data matrix; done here directly so the test
// is independent of Stream B).
ClusterGroupStats stats_from_clusters(const Clusters& clusters) {
  const Eigen::Index p = clusters.at(0).at(0).size();
  ClusterGroupStats g;
  g.p_within = p;
  g.p_between = p;
  g.within_scatter = Eigen::MatrixXd::Zero(p, p);
  Eigen::VectorXd gsum = Eigen::VectorXd::Zero(p);
  std::map<std::int64_t, ClusterSizePattern> by_size;
  std::int64_t N = 0, J = 0;
  for (const auto& cl : clusters) {
    const std::int64_t d = static_cast<std::int64_t>(cl.size());
    Eigen::VectorXd ybar = Eigen::VectorXd::Zero(p);
    for (const auto& y : cl) { ybar += y; gsum += y; }
    ybar /= static_cast<double>(d);
    for (const auto& y : cl) {
      const Eigen::VectorXd e = y - ybar;
      g.within_scatter += e * e.transpose();
    }
    auto& sp = by_size[d];
    if (sp.n_clusters == 0) {
      sp.cluster_size = d;
      sp.sum_cluster_mean = Eigen::VectorXd::Zero(p);
      sp.sum_cluster_mean_cp = Eigen::MatrixXd::Zero(p, p);
    }
    sp.n_clusters += 1;
    sp.sum_cluster_mean += ybar;
    sp.sum_cluster_mean_cp += ybar * ybar.transpose();
    N += d;
    J += 1;
  }
  g.n_within = N;
  g.n_clusters = J;
  g.grand_mean = gsum / static_cast<double>(N);
  for (const auto& [d, sp] : by_size) g.size_patterns.push_back(sp);
  return g;
}

// Brute-force two-level -2logL (minus the n*p*log(2pi) constant), building the
// full V_j = I_n (x) Sw + J_n (x) Sb for each cluster. Independent of the
// grouped-by-size form the kernel uses.
double brute_force_neg2ll(const Clusters& clusters, const Eigen::MatrixXd& Sw,
                          const Eigen::MatrixXd& Sb, const Eigen::VectorXd& mu) {
  const Eigen::Index p = Sw.rows();
  double val = 0.0;
  for (const auto& cl : clusters) {
    const Eigen::Index n = static_cast<Eigen::Index>(cl.size());
    Eigen::MatrixXd V(n * p, n * p);
    Eigen::VectorXd Y(n * p);
    for (Eigen::Index i = 0; i < n; ++i) {
      Y.segment(i * p, p) = cl[static_cast<std::size_t>(i)] - mu;
      for (Eigen::Index j = 0; j < n; ++j) {
        Eigen::MatrixXd blk = Sb;
        if (i == j) blk += Sw;
        V.block(i * p, j * p, p, p) = blk;
      }
    }
    Eigen::LLT<Eigen::MatrixXd> llt(V);
    const auto L = llt.matrixL();
    double logdet = 0.0;
    for (Eigen::Index i = 0; i < n * p; ++i) logdet += std::log(L(i, i));
    val += 2.0 * logdet + Y.dot(llt.solve(Y));
  }
  return val;
}

Eigen::VectorXd vec3(double a, double b, double c) {
  Eigen::VectorXd v(3);
  v << a, b, c;
  return v;
}
Eigen::VectorXd vec2(double a, double b) {
  Eigen::VectorXd v(2);
  v << a, b;
  return v;
}

// Identity selection Jacobians for the parametrization
// theta = [vech(Sw); vech(Sb); mu], so a gradient w.r.t. theta is exactly the
// stacked (vech-doubled G_W, G_B, g_mu). Lets a test read the matrix score off
// twolevel_value_gradient.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> identity_jacobians(Eigen::Index p) {
  const Eigen::Index pv = magmaan::detail::vech_len(p);
  const Eigen::Index nfree = 2 * pv + p;
  Eigen::MatrixXd Js = Eigen::MatrixXd::Zero(2 * pv, nfree);
  Js.block(0, 0, pv, pv).setIdentity();
  Js.block(pv, pv, pv, pv).setIdentity();
  Eigen::MatrixXd Jm = Eigen::MatrixXd::Zero(2 * p, nfree);
  Jm.block(p, 2 * pv, p, p).setIdentity();
  return {Js, Jm};
}

ImpliedMoments moments_of(const Eigen::MatrixXd& Sw, const Eigen::MatrixXd& Sb,
                          const Eigen::VectorXd& mu) {
  ImpliedMoments m;
  m.sigma = {Sw, Sb};
  m.mu = {Eigen::VectorXd(), mu};  // within has no mean structure
  return m;
}

// d x (d-1) Helmert basis: orthonormal columns, each orthogonal to 1_d.
// Column k (0-based) is the contrast (1,...,1,-(k+1),0,...,0) with k+1 leading
// ones, normalized. Used to build zero-sum within-cluster deviations.
Eigen::MatrixXd helmert(int d) {
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(d, d - 1);
  for (int k = 0; k < d - 1; ++k) {
    for (int i = 0; i <= k; ++i) Q(i, k) = 1.0;
    Q(k + 1, k) = -static_cast<double>(k + 1);
    Q.col(k) /= Q.col(k).norm();
  }
  return Q;
}

// Construct clustered data that *exactly* reproduces a chosen two-level model
// at (Sw, Sb, mu): every pseudo-block residual vanishes, so the observed and
// expected information coincide. For each distinct size d we emit J_d = 2p
// clusters whose cluster-mean cross-product about mu equals (J_d/d)·C_d (the
// size-d pseudo-block scatter d·A_d then equals J_d·C_d) and whose pooled
// within scatter sums to (N-J)·Sw. Sizes are deliberately unequal so the
// size-pattern sum is exercised.
Clusters perfect_following(const Eigen::MatrixXd& Sw, const Eigen::MatrixXd& Sb,
                           const Eigen::VectorXd& mu, const std::vector<int>& sizes) {
  const Eigen::Index p = Sw.rows();
  const Eigen::LLT<Eigen::MatrixXd> llt_w(Sw);
  const Eigen::MatrixXd Lw = llt_w.matrixL();  // Sw = Lw Lw'
  Clusters clusters;
  for (int d : sizes) {
    const Eigen::MatrixXd Cd = Sw + static_cast<double>(d) * Sb;
    const Eigen::MatrixXd Lc = Eigen::LLT<Eigen::MatrixXd>(Cd).matrixL();
    // Within-deviation block E (d x p): E'E = (d-1)·Sw, column sums zero.
    const Eigen::MatrixXd Q = helmert(d);                // d x (d-1)
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(d - 1, p);  // R'R = Sw
    R.topRows(p) = Lw.transpose();
    const Eigen::MatrixXd E =
        std::sqrt(static_cast<double>(d - 1)) * (Q * R);  // d x p
    // Cluster means: 2p clusters at mu ± sqrt(p/d)·Lc[:,i], i = 0..p-1.
    const double s = std::sqrt(static_cast<double>(p) / static_cast<double>(d));
    for (Eigen::Index i = 0; i < p; ++i) {
      for (double sign : {+1.0, -1.0}) {
        const Eigen::VectorXd ybar = mu + sign * s * Lc.col(i);
        Cluster cl;
        for (int row = 0; row < d; ++row)
          cl.push_back(Eigen::VectorXd(ybar + E.row(row).transpose()));
        clusters.push_back(std::move(cl));
      }
    }
  }
  return clusters;
}

}  // namespace

TEST_CASE("twolevel_value matches the brute-force stacked-cluster -2logL") {
  // Unbalanced design: cluster sizes {2, 2, 3}, p = 2.
  Clusters clusters = {
      {vec2(1.0, 0.5), vec2(0.2, -0.4)},
      {vec2(-0.3, 1.1), vec2(0.7, 0.0)},
      {vec2(0.4, -0.2), vec2(-0.5, 0.9), vec2(1.3, 0.3)}};

  ClusterSampleStats cs;
  cs.groups = {stats_from_clusters(clusters)};

  Eigen::MatrixXd Sw(2, 2);
  Sw << 1.7, 0.4, 0.4, 1.2;
  Eigen::MatrixXd Sb(2, 2);
  Sb << 0.9, 0.2, 0.2, 0.6;
  const Eigen::VectorXd mu = vec2(0.3, -0.1);

  ImpliedMoments m;
  m.sigma = {Sw, Sb};
  m.mu = {Eigen::VectorXd(), mu};  // within has no mean structure

  const MatrixRep rep = two_block_rep();
  auto cache = tl::twolevel_prepare(cs);
  REQUIRE(cache.has_value());

  auto F = tl::twolevel_value(cs, *cache, m, rep);
  if (!F) {
    CHECK_MESSAGE(false, "twolevel_value failed: " << F.error().detail);
    return;
  }
  const double brute = brute_force_neg2ll(clusters, Sw, Sb, mu);
  CHECK(*F == doctest::Approx(brute).epsilon(1e-9));
}

TEST_CASE("twolevel_value_gradient matches central finite differences") {
  // p = 3 to exercise off-diagonal vech doubling; unbalanced sizes {2, 2, 3}.
  Clusters clusters = {
      {vec3(1.0, 0.5, -0.2), vec3(0.2, -0.4, 0.8)},
      {vec3(-0.3, 1.1, 0.1), vec3(0.7, 0.0, -0.6)},
      {vec3(0.4, -0.2, 0.5), vec3(-0.5, 0.9, 0.2), vec3(1.3, 0.3, -0.1)}};

  ClusterSampleStats cs;
  cs.groups = {stats_from_clusters(clusters)};
  const MatrixRep rep = two_block_rep();
  auto cache = tl::twolevel_prepare(cs);
  REQUIRE(cache.has_value());

  // Identity parametrization: theta = [vech(Sw); vech(Sb); mu], so J_sigma /
  // J_mu are pure selection matrices and the FD check isolates the kernel's
  // G_W / G_B / g_mu assembly and the vech-doubled contraction.
  const Eigen::Index p = 3;
  const Eigen::Index pv = magmaan::detail::vech_len(p);  // 6
  const Eigen::Index nfree = 2 * pv + p;                 // 15

  Eigen::MatrixXd Sw0(3, 3), Sb0(3, 3);
  Sw0 << 2.0, 0.3, 0.1, 0.3, 1.5, 0.2, 0.1, 0.2, 1.0;
  Sb0 << 1.0, 0.2, 0.0, 0.2, 0.8, 0.1, 0.0, 0.1, 0.6;
  const Eigen::VectorXd mu0 = vec3(0.5, -0.3, 0.2);

  Eigen::VectorXd theta(nfree);
  theta << magmaan::detail::vech_lower(Sw0), magmaan::detail::vech_lower(Sb0),
      mu0;

  auto moments_at = [&](const Eigen::VectorXd& th) {
    Eigen::MatrixXd Sw(p, p), Sb(p, p);
    magmaan::detail::vech_unpack(th.segment(0, pv), p, Sw);
    magmaan::detail::vech_unpack(th.segment(pv, pv), p, Sb);
    ImpliedMoments mm;
    mm.sigma = {Sw, Sb};
    mm.mu = {Eigen::VectorXd(), Eigen::VectorXd(th.segment(2 * pv, p))};
    return mm;
  };
  auto value_at = [&](const Eigen::VectorXd& th) -> double {
    auto F = tl::twolevel_value(cs, *cache, moments_at(th), rep);
    REQUIRE(F.has_value());
    return *F;
  };

  // J_sigma (2*pv x nfree): within vech <- cols [0,pv), between vech <- [pv,2pv).
  Eigen::MatrixXd J_sigma = Eigen::MatrixXd::Zero(2 * pv, nfree);
  J_sigma.block(0, 0, pv, pv).setIdentity();
  J_sigma.block(pv, pv, pv, pv).setIdentity();
  // J_mu (2*p x nfree): within mean rows zero; between mean rows <- last p cols.
  Eigen::MatrixXd J_mu = Eigen::MatrixXd::Zero(2 * p, nfree);
  J_mu.block(p, 2 * pv, p, p).setIdentity();

  auto vg = tl::twolevel_value_gradient(cs, *cache, moments_at(theta), J_sigma,
                                        J_mu, rep);
  if (!vg) {
    CHECK_MESSAGE(false, "twolevel_value_gradient failed: " << vg.error().detail);
    return;
  }
  CHECK(vg->value == doctest::Approx(value_at(theta)).epsilon(1e-10));

  const double h = 1e-6;
  for (Eigen::Index k = 0; k < nfree; ++k) {
    Eigen::VectorXd tp = theta, tm = theta;
    tp(k) += h;
    tm(k) -= h;
    const double fd = (value_at(tp) - value_at(tm)) / (2.0 * h);
    CHECK(vg->gradient(k) ==
          doctest::Approx(fd).epsilon(1e-5).scale(1.0 + std::abs(fd)));
  }
}

TEST_CASE("twolevel_h1_moments recovers the balanced closed form") {
  // 5 clusters, each size 4, p = 2; strong between separation -> Sigma_B PD.
  // Within deviations sum to zero per cluster, so ybar_j == base_j exactly.
  const std::vector<Eigen::VectorXd> bases = {vec2(0, 0), vec2(3, 1),
                                              vec2(-2, 2), vec2(1, -3),
                                              vec2(4, 0)};
  const std::vector<Eigen::VectorXd> devs = {vec2(0.2, 0.1), vec2(-0.1, 0.2),
                                             vec2(0.1, -0.2), vec2(-0.2, -0.1)};
  Clusters clusters;
  for (const auto& b : bases) {
    Cluster cl;
    for (const auto& dv : devs) cl.push_back(Eigen::VectorXd(b + dv));
    clusters.push_back(cl);
  }
  ClusterSampleStats cs;
  cs.groups = {stats_from_clusters(clusters)};
  const MatrixRep rep = two_block_rep();

  auto h1 = tl::twolevel_h1_moments(cs);
  REQUIRE(h1.has_value());
  REQUIRE(h1->sigma_w.size() == 1);

  // Closed form from the same data: Sw = S_PW, mu = mean(base_j),
  // Sb = cov(base_j) - S_PW/d.
  const double d = 4.0, J = 5.0, Ntot = 20.0;
  const Eigen::MatrixXd S_PW = cs.groups[0].within_scatter / (Ntot - J);
  Eigen::VectorXd muhat = Eigen::VectorXd::Zero(2);
  for (const auto& b : bases) muhat += b;
  muhat /= J;
  Eigen::MatrixXd Adm = Eigen::MatrixXd::Zero(2, 2);
  for (const auto& b : bases) Adm += (b - muhat) * (b - muhat).transpose();
  const Eigen::MatrixXd Sb_ref = Adm / J - S_PW / d;

  CHECK((h1->mu_b[0] - muhat).norm() < 1e-6);
  CHECK((h1->sigma_w[0] - S_PW).norm() < 1e-6);
  CHECK((h1->sigma_b[0] - Sb_ref).norm() < 1e-6);

  // Score at the H1 solution is ~ 0.
  auto cache = tl::twolevel_prepare(cs);
  REQUIRE(cache.has_value());
  auto [Js, Jm] = identity_jacobians(2);
  auto vg = tl::twolevel_value_gradient(
      cs, *cache, moments_of(h1->sigma_w[0], h1->sigma_b[0], h1->mu_b[0]), Js,
      Jm, rep);
  REQUIRE(vg.has_value());
  CHECK(vg->gradient.norm() < 1e-5);
}

TEST_CASE("twolevel_h1_moments drives the score to zero (unbalanced)") {
  // Varying cluster sizes {2,3,2,4,3} with strong, clearly-PD between
  // separation so the saturated MLE is interior (Sigma_B off the PSD boundary).
  const std::vector<Eigen::VectorXd> bases = {vec2(0, 0), vec2(3, 1),
                                              vec2(-2, 2), vec2(1, -3),
                                              vec2(4, 0)};
  const std::vector<int> sizes = {2, 3, 2, 4, 3};
  Clusters clusters;
  for (std::size_t j = 0; j < bases.size(); ++j) {
    Cluster cl;
    for (int i = 0; i < sizes[j]; ++i) {
      const double s = 0.15 * ((i % 2 == 0) ? 1.0 : -1.0);
      cl.push_back(vec2(bases[j](0) + s, bases[j](1) - 0.5 * s + 0.1 * i));
    }
    clusters.push_back(cl);
  }
  ClusterSampleStats cs;
  cs.groups = {stats_from_clusters(clusters)};
  const MatrixRep rep = two_block_rep();

  auto h1 = tl::twolevel_h1_moments(cs, {500, 1e-12});
  REQUIRE(h1.has_value());

  auto cache = tl::twolevel_prepare(cs);
  REQUIRE(cache.has_value());
  const ImpliedMoments mm =
      moments_of(h1->sigma_w[0], h1->sigma_b[0], h1->mu_b[0]);
  auto [Js, Jm] = identity_jacobians(2);
  auto vg = tl::twolevel_value_gradient(cs, *cache, mm, Js, Jm, rep);
  REQUIRE(vg.has_value());
  CHECK(vg->gradient.norm() < 1e-5);

  // The reported H1 deviance equals F at the returned moments.
  auto F = tl::twolevel_value(cs, *cache, mm, rep);
  REQUIRE(F.has_value());
  CHECK(*F == doctest::Approx(h1->value).epsilon(1e-10));
}

TEST_CASE(
    "twolevel_expected_information equals observed FD info at residual-zero "
    "(unbalanced)") {
  // The closed-form expected (Fisher) information ½·E[∂²F/∂θ²] and the observed
  // information ½·∂²F/∂θ² coincide exactly when every pseudo-block residual
  // vanishes. perfect_following() builds unbalanced clustered data (sizes
  // {3, 4}, so the per-size sum is exercised) that reproduces the chosen model
  // moments exactly, so the analytic expected info must match a central finite
  // difference of the analytic full-deviance gradient to FD tolerance.
  const Eigen::Index p = 2;
  Eigen::MatrixXd Sw(2, 2), Sb(2, 2);
  Sw << 1.6, 0.45, 0.45, 1.1;
  Sb << 0.8, 0.25, 0.25, 0.5;
  const Eigen::VectorXd mu = vec2(0.4, -0.2);
  const std::vector<int> sizes = {3, 4};

  ClusterSampleStats cs;
  cs.groups = {stats_from_clusters(perfect_following(Sw, Sb, mu, sizes))};
  const MatrixRep rep = two_block_rep();
  auto cache = tl::twolevel_prepare(cs);
  REQUIRE(cache.has_value());

  const Eigen::Index pv = magmaan::detail::vech_len(p);  // 3
  const Eigen::Index nfree = 2 * pv + p;                 // 8
  auto [Js, Jm] = identity_jacobians(p);

  // Sanity: residuals really do vanish here -> this point is the optimum.
  auto vg0 = tl::twolevel_value_gradient(cs, *cache, moments_of(Sw, Sb, mu), Js,
                                         Jm, rep);
  REQUIRE(vg0.has_value());
  CHECK(vg0->gradient.norm() < 1e-9);

  // Analytic expected information at the model moments.
  auto info_exp = tl::twolevel_expected_information(
      cs, *cache, moments_of(Sw, Sb, mu), Js, Jm, rep);
  if (!info_exp) {
    CHECK_MESSAGE(false, "twolevel_expected_information failed: "
                             << info_exp.error().detail);
    return;
  }
  REQUIRE(info_exp->rows() == nfree);

  // Observed info: ½ × central-FD Hessian of the analytic gradient ∂F/∂θ.
  Eigen::VectorXd theta(nfree);
  theta << magmaan::detail::vech_lower(Sw), magmaan::detail::vech_lower(Sb), mu;
  auto grad_at = [&](const Eigen::VectorXd& th) -> Eigen::VectorXd {
    Eigen::MatrixXd Sw_(p, p), Sb_(p, p);
    magmaan::detail::vech_unpack(th.segment(0, pv), p, Sw_);
    magmaan::detail::vech_unpack(th.segment(pv, pv), p, Sb_);
    ImpliedMoments mm;
    mm.sigma = {Sw_, Sb_};
    mm.mu = {Eigen::VectorXd(), Eigen::VectorXd(th.segment(2 * pv, p))};
    auto vg = tl::twolevel_value_gradient(cs, *cache, mm, Js, Jm, rep);
    REQUIRE(vg.has_value());
    return vg->gradient;
  };
  const double h = 1e-6;
  Eigen::MatrixXd Hess(nfree, nfree);
  for (Eigen::Index j = 0; j < nfree; ++j) {
    Eigen::VectorXd tp = theta, tm = theta;
    tp(j) += h;
    tm(j) -= h;
    Hess.col(j) = (grad_at(tp) - grad_at(tm)) / (2.0 * h);
  }
  Hess = (0.5 * (Hess + Hess.transpose())).eval();
  const Eigen::MatrixXd info_obs = 0.5 * Hess;

  const double max_abs = (*info_exp - info_obs).cwiseAbs().maxCoeff();
  const double scale = info_obs.cwiseAbs().maxCoeff();
  CHECK(max_abs / scale < 1e-5);

  // The expected info is symmetric positive definite, so it inverts to a vcov.
  CHECK((*info_exp - info_exp->transpose()).cwiseAbs().maxCoeff() < 1e-10);
  Eigen::LLT<Eigen::MatrixXd> llt(*info_exp);
  CHECK(llt.info() == Eigen::Success);
}
