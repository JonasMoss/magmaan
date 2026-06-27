// Unit tests for data::cluster_sample_stats — the two-level sufficient
// statistics (Stream B, Contract 2). Two independent validations:
//
//   (1) A hand-built balanced dataset whose SSW, size buckets, grand mean, and
//       per-size sum_cluster_mean / sum_cluster_mean_cp are computable by hand.
//
//   (2) Lavaan parity. Small balanced + unbalanced fixtures whose data and
//       reference statistics were emitted by
//         lavInspect(sem(model, data, cluster="id"), "sampstat")$within$cov
//       (== within_scatter/(N-J)), the per-size cluster-mean sums, and the
//       cluster-size table. lavaan 0.7.1.2691; the generator script lives in
//       this stream's scratchpad. CI never invokes R — the references are
//       embedded constants.

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <doctest/doctest.h>

#include "magmaan/data/cluster_stats.hpp"

using magmaan::data::ClusterSampleStats;
using magmaan::data::cluster_sample_stats;
using magmaan::data::cluster_sample_stats_multigroup;

namespace {

// Build (X, cluster_id) from a flat {id, v0, v1, ...} row table where column 0
// is the (0-based) cluster id and the rest are the p observed variables.
template <std::size_t NRows, std::size_t NCol>
std::pair<Eigen::MatrixXd, std::vector<std::int32_t>>
unpack(const std::array<std::array<double, NCol>, NRows>& rows) {
  static_assert(NCol >= 2, "need an id column plus at least one variable");
  constexpr Eigen::Index p = static_cast<Eigen::Index>(NCol) - 1;
  Eigen::MatrixXd X(static_cast<Eigen::Index>(NRows), p);
  std::vector<std::int32_t> id(NRows);
  for (std::size_t r = 0; r < NRows; ++r) {
    id[r] = static_cast<std::int32_t>(rows[r][0]);
    for (Eigen::Index k = 0; k < p; ++k)
      X(static_cast<Eigen::Index>(r), k) = rows[r][static_cast<std::size_t>(k) + 1];
  }
  return {std::move(X), std::move(id)};
}

constexpr double kTol = 1e-9;

}  // namespace

TEST_CASE("cluster_stats: hand-built balanced data matches by-hand values") {
  // Two variables, three clusters of size two. Tiny enough that SSW, the
  // grand mean, and the size-2 bucket are computable by hand.
  //
  //   cluster 0: (0,0), (2,0)   -> ȳ = (1, 0)
  //   cluster 1: (4,1), (6,3)   -> ȳ = (5, 2)
  //   cluster 2: (1,2), (3,4)   -> ȳ = (2, 3)
  const std::array<std::array<double, 3>, 6> rows = {{
      {{0, 0, 0}}, {{0, 2, 0}},
      {{1, 4, 1}}, {{1, 6, 3}},
      {{2, 1, 2}}, {{2, 3, 4}},
  }};
  auto [X, id] = unpack(rows);
  std::vector<std::int32_t> cols = {0, 1};

  auto res = cluster_sample_stats(X, id, cols, cols);
  REQUIRE(res.has_value());
  const auto& cs = *res;
  REQUIRE(cs.groups.size() == 1);
  const auto& g = cs.groups[0];

  CHECK(g.n_within == 6);
  CHECK(g.n_clusters == 3);
  CHECK(g.p_within == 2);
  CHECK(g.p_between == 2);

  // Grand mean over all 6 rows: column sums (16, 10) / 6.
  CHECK(g.grand_mean(0) == doctest::Approx(16.0 / 6.0));
  CHECK(g.grand_mean(1) == doctest::Approx(10.0 / 6.0));

  // SSW = Σ_j Σ_i (y − ȳ_j)(y − ȳ_j)ᵀ, raw (no N−J divisor).
  //   cluster 0 deviations: (−1,0),(+1,0) -> [[2,0],[0,0]]
  //   cluster 1 deviations: (−1,−1),(+1,+1) -> [[2,2],[2,2]]
  //   cluster 2 deviations: (−1,−1),(+1,+1) -> [[2,2],[2,2]]
  // Sum: [[6,4],[4,4]].
  CHECK(g.within_scatter(0, 0) == doctest::Approx(6.0));
  CHECK(g.within_scatter(0, 1) == doctest::Approx(4.0));
  CHECK(g.within_scatter(1, 0) == doctest::Approx(4.0));
  CHECK(g.within_scatter(1, 1) == doctest::Approx(4.0));

  // One distinct size (d = 2), three clusters.
  REQUIRE(g.size_patterns.size() == 1);
  const auto& pat = g.size_patterns[0];
  CHECK(pat.cluster_size == 2);
  CHECK(pat.n_clusters == 3);

  // sum_cluster_mean = (1,0)+(5,2)+(2,3) = (8,5).
  CHECK(pat.sum_cluster_mean(0) == doctest::Approx(8.0));
  CHECK(pat.sum_cluster_mean(1) == doctest::Approx(5.0));

  // sum_cluster_mean_cp = Σ ȳ ȳᵀ
  //   (1,0): [[1,0],[0,0]]
  //   (5,2): [[25,10],[10,4]]
  //   (2,3): [[4,6],[6,9]]
  // Sum: [[30,16],[16,13]].
  CHECK(pat.sum_cluster_mean_cp(0, 0) == doctest::Approx(30.0));
  CHECK(pat.sum_cluster_mean_cp(0, 1) == doctest::Approx(16.0));
  CHECK(pat.sum_cluster_mean_cp(1, 0) == doctest::Approx(16.0));
  CHECK(pat.sum_cluster_mean_cp(1, 1) == doctest::Approx(13.0));

  // Index vectors round-trip the selectors.
  CHECK(cs.within_ov_index == cols);
  CHECK(cs.between_ov_index == cols);
}

TEST_CASE("cluster_stats: per-size cross-products reconstruct the total") {
  // Self-consistency, no external oracle: the per-size sum_cluster_mean_cp
  // summed over sizes equals Σ_all_clusters ȳ ȳᵀ, computed independently.
  const std::array<std::array<double, 3>, 7> rows = {{
      {{0, 1.0, 2.0}}, {{0, 3.0, 4.0}},                 // d=2
      {{1, 0.5, 9.0}},                                  // d=1
      {{2, 2.0, 2.0}}, {{2, 4.0, 6.0}}, {{2, 6.0, 1.0}},// d=3
      {{3, 5.0, 5.0}},                                  // d=1
  }};
  auto [X, id] = unpack(rows);
  std::vector<std::int32_t> cols = {0, 1};
  auto res = cluster_sample_stats(X, id, cols, cols);
  REQUIRE(res.has_value());
  const auto& g = res->groups[0];

  // sizes present: 1 (two clusters) and 2 and 3.
  std::int64_t total_clusters = 0;
  Eigen::MatrixXd cp_sum = Eigen::MatrixXd::Zero(2, 2);
  Eigen::VectorXd mean_sum = Eigen::VectorXd::Zero(2);
  for (const auto& pat : g.size_patterns) {
    total_clusters += pat.n_clusters;
    cp_sum += pat.sum_cluster_mean_cp;
    mean_sum += pat.sum_cluster_mean;
  }
  CHECK(total_clusters == 4);
  CHECK(g.n_clusters == 4);

  // Independent: cluster means are (2,3),(0.5,9),(4,3),(5,5).
  Eigen::MatrixXd cp_ref = Eigen::MatrixXd::Zero(2, 2);
  Eigen::VectorXd mean_ref = Eigen::VectorXd::Zero(2);
  for (Eigen::Vector2d ybar : {Eigen::Vector2d(2.0, 3.0), Eigen::Vector2d(0.5, 9.0),
                               Eigen::Vector2d(4.0, 3.0), Eigen::Vector2d(5.0, 5.0)}) {
    cp_ref += ybar * ybar.transpose();
    mean_ref += ybar;
  }
  CHECK((cp_sum - cp_ref).cwiseAbs().maxCoeff() < kTol);
  CHECK((mean_sum - mean_ref).cwiseAbs().maxCoeff() < kTol);
}

// ---------------------------------------------------------------------------
// Lavaan parity fixtures (emitted from lavaan 0.7.1.2691). p = 2.
// ---------------------------------------------------------------------------

namespace {

// === balanced: 48 rows, 12 clusters of size 4 ===
const std::array<std::array<double, 3>, 48> balanced_data = {{
    {{0, 2.4958, 1.7483}},  {{0, 3.759, 2.8512}},   {{0, 2.5813, 3.1148}},  {{0, 2.9687, 2.9519}},
    {{1, 3.6041, -0.1625}}, {{1, 2.8804, -1.035}},  {{1, 3.9333, -1.0195}}, {{1, 3.3044, -4.5964}},
    {{2, 2.372, -0.8445}},  {{2, 2.4224, -0.3793}}, {{2, 2.8788, -0.1292}}, {{2, 2.8194, 0.3763}},
    {{3, 1.6467, -1.1167}}, {{3, 4.6406, 0.1165}},  {{3, 3.3763, -0.6208}}, {{3, 2.809, -1.0246}},
    {{4, 1.1465, -1.6269}}, {{4, 4.9871, -2.2979}}, {{4, 5.0537, -3.6019}}, {{4, 5.0712, -0.6644}},
    {{5, 4.2787, -2.2078}}, {{5, 4.0727, -1.8442}}, {{5, 2.1581, -3.8103}}, {{5, 2.3248, -1.8853}},
    {{6, 5.1175, -1.5179}}, {{6, 2.519, 0.2861}},   {{6, 4.6943, 0.2169}},  {{6, 4.5295, -0.2537}},
    {{7, 3.1443, -1.2849}}, {{7, 1.4515, -2.4189}}, {{7, 2.3401, -1.0142}}, {{7, 2.0255, 0.1962}},
    {{8, 5.83, 0.5858}},    {{8, 2.2947, 0.8546}},  {{8, 3.9805, -1.433}},  {{8, 5.3766, -0.6111}},
    {{9, 5.7316, 0.342}},   {{9, 5.2115, 0.9443}},  {{9, 4.401, 1.1443}},   {{9, 4.7006, 0.1938}},
    {{10, 3.473, -0.7932}}, {{10, 2.4322, -1.3934}},{{10, 2.0108, 0.2715}}, {{10, 2.91, -2.2943}},
    {{11, 4.6695, 0.5148}}, {{11, 6.7389, -1.2149}},{{11, 7.0161, -0.8888}},{{11, 8.3786, 0.8237}},
}};
// lavaan S.PW.start (== SSW/(N-J)), extracted at full precision from the
// fitted object's internal YLp (lavInspect "sampstat" prints only 3 decimals).
const std::array<std::array<double, 2>, 2> balanced_within_cov = {{
    {{1.2211655846527776, 0.00091398590277779065}},
    {{0.00091398590277779065, 1.0511348264583333}},
}};
const std::array<double, 2> balanced_grand_mean = {{3.7200479166666667, -0.55109374999999994}};
// d=4 J_d=12 scm={44.640574999999998,-6.6131250000000001}
//   cp_diag={183.12242132062499,23.616608679374998} cp01=-22.190407395625005

// === unbalanced: 56 rows; sizes 2 (x6), 4 (x5), 6 (x4) ===
const std::array<std::array<double, 3>, 56> unbalanced_data = {{
    {{0, 3.3046, -1.8515}},  {{0, 1.9983, -3.9375}},
    {{1, 5.3715, 2.5523}},   {{1, 5.9776, -0.4493}},
    {{2, 3.1446, -0.4598}},  {{2, 4.1144, 2.239}},
    {{3, 5.3726, 0.9056}},   {{3, 1.8417, -1.3929}},
    {{4, 1.8941, 0.0446}},   {{4, 4.8617, -1.192}},
    {{5, 5.5754, -0.777}},   {{5, 6.0411, -0.8042}},
    {{6, 4.3232, -0.1535}},  {{6, 3.7109, -1.2895}},  {{6, 3.4663, -1.7597}},  {{6, 4.6104, 0.3244}},
    {{7, 4.0591, -2.0183}},  {{7, 3.3147, -1.8613}},  {{7, 3.1946, -0.6125}},  {{7, 7.117, -0.4986}},
    {{8, 5.7101, -0.1771}},  {{8, 1.6158, -1.7113}},  {{8, 0.9404, -2.6293}},  {{8, 3.2771, 0.7329}},
    {{9, 4.298, -1.2872}},   {{9, 3.9105, 0.3796}},   {{9, 3.6039, -0.4879}},  {{9, 2.675, -0.7085}},
    {{10, 4.2473, -3.0516}}, {{10, 3.4924, 0.3252}},  {{10, 0.7205, -3.2569}}, {{10, 0.5701, -3.4026}},
    {{11, 3.7132, -1.7764}}, {{11, 5.1787, -0.3156}}, {{11, 4.3344, -0.8738}}, {{11, 5.975, -0.5793}},
    {{11, 3.2585, -0.8115}}, {{11, 5.5146, -2.6509}},
    {{12, 6.551, 1.7833}},   {{12, 6.4307, -1.2823}}, {{12, 6.5579, 2.0954}},  {{12, 2.8028, -0.9087}},
    {{12, 3.7222, 1.9065}},  {{12, 6.2896, 0.113}},
    {{13, 3.3798, 0.9517}},  {{13, 1.9281, 1.9733}},  {{13, 4.4673, 1.447}},   {{13, 4.0162, -0.2096}},
    {{13, 4.6262, 0.319}},   {{13, 2.917, -0.4564}},
    {{14, 2.5753, -1.7462}}, {{14, 5.4581, -3.0553}}, {{14, 5.5857, -0.6971}}, {{14, 4.8772, -1.3586}},
    {{14, 3.5555, -1.0001}}, {{14, 6.1961, 1.6439}},
}};
// lavaan S.PW.start (== SSW/(N-J)), full precision (see balanced note).
const std::array<std::array<double, 2>, 2> unbalanced_within_cov = {{
    {{2.0337854890853659, 0.60562591380081299}},
    {{0.60562591380081299, 1.6611855013617887}},
}};
const std::array<double, 2> unbalanced_grand_mean = {{4.0761785714285717, -0.60276964285714285}};
// per-size buckets (d, J_d, scm0, scm1, cp00, cp11, cp01):
struct SizeRef {
  std::int64_t d, J_d;
  double scm0, scm1, cp00, cp11, cp01;
};
const std::array<SizeRef, 3> unbalanced_size_ref = {{
    {2, 6, 24.748799999999999, -2.56135, 110.56148265, 11.2887160325, -5.8878152774999997},
    {4, 5, 17.214325000000002, -5.7859249999999998, 62.313275638124999, 8.7523964518750006,
     -18.347667756875001},
    {6, 4, 18.318516666666667, -0.91478333333333328, 85.624175682499995, 3.2682042402777784,
     -4.6036348505555544},
}};

}  // namespace

TEST_CASE("cluster_stats: lavaan parity (balanced)") {
  auto [X, id] = unpack(balanced_data);
  std::vector<std::int32_t> cols = {0, 1};
  auto res = cluster_sample_stats(X, id, cols, cols);
  REQUIRE(res.has_value());
  const auto& g = res->groups[0];

  CHECK(g.n_within == 48);
  CHECK(g.n_clusters == 12);
  REQUIRE(g.size_patterns.size() == 1);
  CHECK(g.size_patterns[0].cluster_size == 4);
  CHECK(g.size_patterns[0].n_clusters == 12);

  // grand mean.
  CHECK(g.grand_mean(0) == doctest::Approx(balanced_grand_mean[0]).epsilon(1e-12));
  CHECK(g.grand_mean(1) == doctest::Approx(balanced_grand_mean[1]).epsilon(1e-12));

  // within_scatter / (N - J) == lavaan $within$cov.
  const double denom = static_cast<double>(g.n_within - g.n_clusters);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      CHECK(g.within_scatter(i, j) / denom ==
            doctest::Approx(balanced_within_cov[static_cast<std::size_t>(i)]
                                              [static_cast<std::size_t>(j)])
                .epsilon(1e-9));

  // per-size sum_cluster_mean / sum_cluster_mean_cp.
  const auto& p = g.size_patterns[0];
  CHECK(p.sum_cluster_mean(0) == doctest::Approx(44.640574999999998).epsilon(1e-10));
  CHECK(p.sum_cluster_mean(1) == doctest::Approx(-6.6131250000000001).epsilon(1e-10));
  CHECK(p.sum_cluster_mean_cp(0, 0) == doctest::Approx(183.12242132062499).epsilon(1e-10));
  CHECK(p.sum_cluster_mean_cp(1, 1) == doctest::Approx(23.616608679374998).epsilon(1e-10));
  CHECK(p.sum_cluster_mean_cp(0, 1) == doctest::Approx(-22.190407395625005).epsilon(1e-10));
}

TEST_CASE("cluster_stats: lavaan parity (unbalanced)") {
  auto [X, id] = unpack(unbalanced_data);
  std::vector<std::int32_t> cols = {0, 1};
  auto res = cluster_sample_stats(X, id, cols, cols);
  REQUIRE(res.has_value());
  const auto& g = res->groups[0];

  CHECK(g.n_within == 56);
  CHECK(g.n_clusters == 15);

  // Distinct cluster sizes 2, 4, 6 (ascending), with counts 6, 5, 4.
  REQUIRE(g.size_patterns.size() == 3);
  for (std::size_t k = 0; k < 3; ++k) {
    CHECK(g.size_patterns[k].cluster_size == unbalanced_size_ref[k].d);
    CHECK(g.size_patterns[k].n_clusters == unbalanced_size_ref[k].J_d);
  }

  // grand mean over all 56 rows.
  CHECK(g.grand_mean(0) == doctest::Approx(unbalanced_grand_mean[0]).epsilon(1e-12));
  CHECK(g.grand_mean(1) == doctest::Approx(unbalanced_grand_mean[1]).epsilon(1e-12));

  // within_scatter / (N - J) == lavaan $within$cov.
  const double denom = static_cast<double>(g.n_within - g.n_clusters);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      CHECK(g.within_scatter(i, j) / denom ==
            doctest::Approx(unbalanced_within_cov[static_cast<std::size_t>(i)]
                                                 [static_cast<std::size_t>(j)])
                .epsilon(1e-9));

  // per-size between cross-products vs lavaan.
  for (std::size_t k = 0; k < 3; ++k) {
    const auto& p = g.size_patterns[k];
    const auto& r = unbalanced_size_ref[k];
    CHECK(p.sum_cluster_mean(0) == doctest::Approx(r.scm0).epsilon(1e-10));
    CHECK(p.sum_cluster_mean(1) == doctest::Approx(r.scm1).epsilon(1e-10));
    CHECK(p.sum_cluster_mean_cp(0, 0) == doctest::Approx(r.cp00).epsilon(1e-10));
    CHECK(p.sum_cluster_mean_cp(1, 1) == doctest::Approx(r.cp11).epsilon(1e-10));
    CHECK(p.sum_cluster_mean_cp(0, 1) == doctest::Approx(r.cp01).epsilon(1e-10));
    CHECK(p.sum_cluster_mean_cp(1, 0) == doctest::Approx(r.cp01).epsilon(1e-10));
  }
}

TEST_CASE("cluster_stats: malformed input is rejected") {
  Eigen::MatrixXd X(3, 2);
  X << 1, 2, 3, 4, 5, 6;
  std::vector<std::int32_t> cols = {0, 1};

  // cluster_id length mismatch.
  {
    std::vector<std::int32_t> id = {0, 0};  // too short
    auto res = cluster_sample_stats(X, id, cols, cols);
    CHECK_FALSE(res.has_value());
  }
  // empty matrix.
  {
    Eigen::MatrixXd empty(0, 2);
    std::vector<std::int32_t> id;
    auto res = cluster_sample_stats(empty, id, cols, cols);
    CHECK_FALSE(res.has_value());
  }
  // within != between (v1 shared-set violation).
  {
    std::vector<std::int32_t> id = {0, 0, 1};
    std::vector<std::int32_t> other = {1, 0};
    auto res = cluster_sample_stats(X, id, cols, other);
    CHECK_FALSE(res.has_value());
  }
  // column out of range.
  {
    std::vector<std::int32_t> id = {0, 0, 1};
    std::vector<std::int32_t> bad = {0, 5};
    auto res = cluster_sample_stats(X, id, bad, bad);
    CHECK_FALSE(res.has_value());
  }
  // empty selector.
  {
    std::vector<std::int32_t> id = {0, 0, 1};
    std::vector<std::int32_t> none;
    auto res = cluster_sample_stats(X, id, none, none);
    CHECK_FALSE(res.has_value());
  }
}

TEST_CASE("cluster_stats: multigroup merge preserves per-group statistics") {
  // Build a two-group ClusterSampleStats from the balanced (group 0) and
  // unbalanced (group 1) lavaan-parity datasets, and assert each merged group
  // equals the single-group reduction. The multigroup builder is just a
  // per-group fan-out of the single-group reducer, so this pins that contract.
  auto [Xb, idb] = unpack(balanced_data);
  auto [Xu, idu] = unpack(unbalanced_data);
  std::vector<std::int32_t> cols = {0, 1};

  auto single_b = cluster_sample_stats(Xb, idb, cols, cols);
  auto single_u = cluster_sample_stats(Xu, idu, cols, cols);
  REQUIRE(single_b.has_value());
  REQUIRE(single_u.has_value());

  std::vector<Eigen::MatrixXd> X_by_group = {Xb, Xu};
  std::vector<std::vector<std::int32_t>> id_by_group = {idb, idu};
  auto mg = cluster_sample_stats_multigroup(X_by_group, id_by_group, cols, cols);
  REQUIRE(mg.has_value());
  REQUIRE(mg->groups.size() == 2);
  CHECK(mg->within_ov_index == cols);
  CHECK(mg->between_ov_index == cols);

  // group 0 == single balanced; group 1 == single unbalanced.
  const auto check_group = [](const auto& got, const auto& want) {
    CHECK(got.n_within == want.n_within);
    CHECK(got.n_clusters == want.n_clusters);
    CHECK(got.p_within == want.p_within);
    CHECK(got.p_between == want.p_between);
    CHECK((got.grand_mean - want.grand_mean).cwiseAbs().maxCoeff() < kTol);
    CHECK((got.within_scatter - want.within_scatter).cwiseAbs().maxCoeff() < kTol);
    REQUIRE(got.size_patterns.size() == want.size_patterns.size());
    for (std::size_t s = 0; s < got.size_patterns.size(); ++s) {
      CHECK(got.size_patterns[s].cluster_size == want.size_patterns[s].cluster_size);
      CHECK(got.size_patterns[s].n_clusters == want.size_patterns[s].n_clusters);
      CHECK((got.size_patterns[s].sum_cluster_mean -
             want.size_patterns[s].sum_cluster_mean).cwiseAbs().maxCoeff() < kTol);
      CHECK((got.size_patterns[s].sum_cluster_mean_cp -
             want.size_patterns[s].sum_cluster_mean_cp).cwiseAbs().maxCoeff() < kTol);
    }
  };
  check_group(mg->groups[0], single_b->groups[0]);
  check_group(mg->groups[1], single_u->groups[0]);
}

TEST_CASE("cluster_stats: multigroup malformed input is rejected") {
  auto [Xb, idb] = unpack(balanced_data);
  std::vector<std::int32_t> cols = {0, 1};

  // empty group list.
  {
    std::vector<Eigen::MatrixXd> X;
    std::vector<std::vector<std::int32_t>> id;
    auto res = cluster_sample_stats_multigroup(X, id, cols, cols);
    CHECK_FALSE(res.has_value());
  }
  // group / cluster_id count mismatch.
  {
    std::vector<Eigen::MatrixXd> X = {Xb, Xb};
    std::vector<std::vector<std::int32_t>> id = {idb};  // one short
    auto res = cluster_sample_stats_multigroup(X, id, cols, cols);
    CHECK_FALSE(res.has_value());
  }
  // a per-group reduction failure (cluster_id length mismatch in group 1)
  // propagates as an overall failure.
  {
    std::vector<Eigen::MatrixXd> X = {Xb, Xb};
    std::vector<std::vector<std::int32_t>> id = {idb, {0, 0, 1}};
    auto res = cluster_sample_stats_multigroup(X, id, cols, cols);
    CHECK_FALSE(res.has_value());
  }
}

TEST_CASE("cluster_stats: non-contiguous / out-of-order cluster ids") {
  // Cluster labels need not be contiguous or sorted; grouping is by value.
  const std::array<std::array<double, 3>, 5> rows = {{
      {{7, 1.0, 1.0}},
      {{3, 2.0, 2.0}},
      {{7, 3.0, 3.0}},
      {{3, 4.0, 4.0}},
      {{99, 5.0, 5.0}},
  }};
  auto [X, id] = unpack(rows);
  std::vector<std::int32_t> cols = {0, 1};
  auto res = cluster_sample_stats(X, id, cols, cols);
  REQUIRE(res.has_value());
  const auto& g = res->groups[0];
  CHECK(g.n_within == 5);
  CHECK(g.n_clusters == 3);
  // sizes: cluster 7 -> 2, cluster 3 -> 2, cluster 99 -> 1.
  REQUIRE(g.size_patterns.size() == 2);
  CHECK(g.size_patterns[0].cluster_size == 1);
  CHECK(g.size_patterns[0].n_clusters == 1);
  CHECK(g.size_patterns[1].cluster_size == 2);
  CHECK(g.size_patterns[1].n_clusters == 2);
}
