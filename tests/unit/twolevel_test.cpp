#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <map>
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
