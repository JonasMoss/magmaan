#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::Estimates;
using magmaan::model::MatrixRep;
using magmaan::model::build_matrix_rep;
using magmaan::parse::Parser;
using magmaan::spec::LatentStructure;
using magmaan::spec::lavaanify;

namespace {

struct Handles {
  LatentStructure pt;
  MatrixRep rep;
};

Handles build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto rep = build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  return Handles{std::move(*pt), std::move(*rep)};
}

nlohmann::json fixture_json(const std::string& rel) {
  std::ifstream in(std::string(MAGMAAN_FIXTURES_DIR) + rel);
  REQUIRE(in.is_open());
  std::stringstream ss;
  ss << in.rdbuf();
  auto j = nlohmann::json::parse(ss.str(), nullptr, false);
  REQUIRE(!j.is_discarded());
  return j;
}

Eigen::MatrixXd matrix_from_json(const nlohmann::json& arr) {
  const Eigen::Index nr = static_cast<Eigen::Index>(arr.size());
  const Eigen::Index nc = static_cast<Eigen::Index>(arr[0].size());
  Eigen::MatrixXd out(nr, nc);
  for (Eigen::Index i = 0; i < nr; ++i) {
    for (Eigen::Index j = 0; j < nc; ++j) {
      out(i, j) = arr[static_cast<std::size_t>(i)]
                     [static_cast<std::size_t>(j)].get<double>();
    }
  }
  return out;
}

SampleStats sample_from_fixture(const nlohmann::json& j) {
  SampleStats samp;
  samp.S.push_back(matrix_from_json(j["sample_cov"][0]["matrix"]));
  samp.n_obs.push_back(j["n_obs"].get<std::int64_t>());
  return samp;
}

bool has_covariance_candidate(const magmaan::nt::infer::ScoreTestTable& table) {
  for (const auto& row : table.rows) {
    if (row.candidate.op == magmaan::parse::Op::Covariance &&
        row.mi >= 0.0 && row.information > 0.0 &&
        std::isfinite(row.epc) && std::isfinite(row.p_value)) {
      return true;
    }
  }
  return false;
}

Eigen::Matrix4d four_indicator_sample_cov() {
  Eigen::Vector4d lambda;
  lambda << 1.0, 0.8, 0.7, 0.9;
  Eigen::Vector4d theta;
  theta << 0.6, 0.7, 0.8, 0.5;
  Eigen::Matrix4d S =
      lambda * lambda.transpose() * 1.4 + theta.asDiagonal().toDenseMatrix();
  S(1, 0) += 0.18;
  S(0, 1) = S(1, 0);
  return S;
}

}  // namespace

TEST_CASE("modification_indices: complete ML reports finite fixed-row tests") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  SampleStats samp;
  samp.S = {four_indicator_sample_cov()};
  samp.n_obs = {400};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  auto mi = magmaan::nt::infer::modification_indices(h.pt, h.rep, samp, *est);
  REQUIRE(mi.has_value());
  CHECK(has_covariance_candidate(*mi));
}

TEST_CASE("modification_indices: ULS uses LS residual information") {
  auto h = build("f =~ x1 + x2 + x3 + x4\nx1 ~~ 0*x2");
  SampleStats samp;
  samp.S = {four_indicator_sample_cov()};
  samp.n_obs = {400};
  auto est = magmaan::test::fit_gmm(h.pt, h.rep, samp, {},
                                    magmaan::estimate::Bounds{});
  REQUIRE(est.has_value());

  auto mi = magmaan::nt::infer::modification_indices(h.pt, h.rep, samp, *est,
                                                     magmaan::gmm::Weight{});
  REQUIRE(mi.has_value());
  CHECK(has_covariance_candidate(*mi));
}

TEST_CASE("score_tests: equality releases are reported in constrained ML models") {
  auto h = build("f =~ a*x1 + b*x2 + x3\na == b");
  const auto j = fixture_json("/fit/0001_one_factor_cfa.fit.json");
  const SampleStats samp = sample_from_fixture(j);
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  auto st = magmaan::nt::infer::score_tests(h.pt, h.rep, samp, *est);
  REQUIRE(st.has_value());
  REQUIRE(st->rows.size() >= 1);
  CHECK(st->rows[0].candidate.kind == magmaan::nt::infer::ScoreCandidateKind::EqualityRelease);
  CHECK(st->rows[0].mi >= 0.0);
  CHECK(std::isfinite(st->rows[0].epc));
}

// ============================================================================
// Phase 5 — absent-row generation and standardized EPC.
// ============================================================================

namespace {

namespace inf = magmaan::nt::infer;

// HS 2-factor sample covariance (lavInspect(cfa(...), "sampstat")$cov on
// HolzingerSwineford1939, x1..x6 — the N-divisor covariance magmaan consumes).
Eigen::MatrixXd hs_6var_cov() {
  Eigen::MatrixXd S(6, 6);
  S << 1.3583698455, 0.4073713302, 0.5798993223, 0.5048349962, 0.4406155374, 0.4548081306,
       0.4073713302, 1.3817838655, 0.4510639369, 0.2089233703, 0.2110910752, 0.2475445713,
       0.5798993223, 0.4510639369, 1.2748648608, 0.2081696129, 0.1122962909, 0.2441089815,
       0.5048349962, 0.2089233703, 0.2081696129, 1.3506645079, 1.0977527816, 0.8955157094,
       0.4406155374, 0.2110910752, 0.1122962909, 1.0977527816, 1.6597857640, 1.0145240421,
       0.4548081306, 0.2475445713, 0.2441089815, 0.8955157094, 1.0145240421, 1.1963583780;
  return S;
}

int ov_index(const LatentStructure& pt, std::int32_t v) {
  return (v >= 0 && static_cast<std::size_t>(v) < pt.ov_pos.size())
             ? pt.ov_pos[static_cast<std::size_t>(v)]
             : -1;
}

// 0-based observed index of a factor's unit-fixed marker indicator.
int factor_marker_ov(const LatentStructure& pt, std::int32_t f) {
  for (std::size_t i = 0; i < pt.size(); ++i) {
    if (pt.op[i] == magmaan::parse::Op::Measurement && pt.lhs_var[i] == f &&
        pt.free[i] == 0 && pt.fixed_value[i] == 1.0) {
      return ov_index(pt, pt.rhs_var[i]);
    }
  }
  return -1;
}

const inf::ScoreTestResult*
find_residual_cov(const inf::ScoreTestTable& t, const LatentStructure& pt,
                  int ov_a, int ov_b) {
  for (const auto& r : t.rows) {
    const auto& c = r.candidate;
    if (c.op != magmaan::parse::Op::Covariance || c.lhs_var == c.rhs_var) continue;
    const int a = ov_index(pt, c.lhs_var);
    const int b = ov_index(pt, c.rhs_var);
    if ((a == ov_a && b == ov_b) || (a == ov_b && b == ov_a)) return &r;
  }
  return nullptr;
}

const inf::ScoreTestResult*
find_cross_loading(const inf::ScoreTestTable& t, const LatentStructure& pt,
                   int marker_ov, int indicator_ov) {
  for (const auto& r : t.rows) {
    const auto& c = r.candidate;
    if (c.op != magmaan::parse::Op::Measurement) continue;
    if (ov_index(pt, c.rhs_var) == indicator_ov &&
        factor_marker_ov(pt, c.lhs_var) == marker_ov) {
      return &r;
    }
  }
  return nullptr;
}

}  // namespace

TEST_CASE("modification_indices: WithAbsentRows is a superset of FixedRowsOnly") {
  auto h = build("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6");
  SampleStats samp;
  samp.S = {hs_6var_cov()};
  samp.n_obs = {301};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  auto fixed = inf::modification_indices(h.pt, h.rep, samp, *est);
  REQUIRE(fixed.has_value());

  inf::ModificationIndexOptions opts;
  opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto absent = inf::modification_indices(h.pt, h.rep, samp, *est, opts);
  REQUIRE(absent.has_value());

  // The model has all parameters present, so FixedRowsOnly finds nothing;
  // WithAbsentRows enumerates the 6 cross-loadings + 15 residual covariances.
  CHECK(absent->rows.size() > fixed->rows.size());
  int loadings = 0, covs = 0;
  for (const auto& r : absent->rows) {
    if (r.candidate.op == magmaan::parse::Op::Measurement) ++loadings;
    if (r.candidate.op == magmaan::parse::Op::Covariance) ++covs;
    CHECK(r.candidate.op != magmaan::parse::Op::Regression);  // off by default
  }
  CHECK(loadings == 6);   // f1=~x4..x6, f2=~x1..x3
  CHECK(covs == 15);      // C(6,2) residual covariances
}

TEST_CASE("modification_indices: an absent candidate scores like an explicit "
          "fixed-at-0 row") {
  SampleStats samp;
  samp.S = {hs_6var_cov()};
  samp.n_obs = {301};

  // Same model, once with the cross-loading f1=~x4 explicitly fixed at 0.
  auto base = build("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6");
  auto expl = build("f1 =~ x1 + x2 + x3 + 0*x4\nf2 =~ x4 + x5 + x6");

  auto est_base = magmaan::test::fit(base.pt, base.rep, samp);
  auto est_expl = magmaan::test::fit(expl.pt, expl.rep, samp);
  REQUIRE(est_base.has_value());
  REQUIRE(est_expl.has_value());

  inf::ModificationIndexOptions opts;
  opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto gen = inf::modification_indices(base.pt, base.rep, samp, *est_base, opts);
  REQUIRE(gen.has_value());
  auto fixed = inf::modification_indices(expl.pt, expl.rep, samp, *est_expl);
  REQUIRE(fixed.has_value());

  // f1's marker is x1 (ov 0); the cross-loading is onto x4 (ov 3).
  const auto* g = find_cross_loading(*gen, base.pt, 0, 3);
  const auto* f = find_cross_loading(*fixed, expl.pt, 0, 3);
  REQUIRE(g != nullptr);
  REQUIRE(f != nullptr);
  CHECK(g->mi == doctest::Approx(f->mi).epsilon(1e-6));
  CHECK(g->epc == doctest::Approx(f->epc).epsilon(1e-6));
  CHECK(g->epc_lv == doctest::Approx(f->epc_lv).epsilon(1e-6));
  CHECK(g->epc_all == doctest::Approx(f->epc_all).epsilon(1e-6));
}

TEST_CASE("modification_indices: absent-row MI / EPC match lavaan modindices") {
  auto h = build("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6");
  SampleStats samp;
  samp.S = {hs_6var_cov()};
  samp.n_obs = {301};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions opts;
  opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  auto mi = inf::modification_indices(h.pt, h.rep, samp, *est, opts);
  REQUIRE(mi.has_value());

  // Oracle: modindices(cfa(model, HolzingerSwineford1939)), lavaan 0.6.22.
  struct Loading { int marker_ov, ind_ov; double mi, epc, sepc_lv, sepc_all; };
  const Loading loadings[] = {
      {3, 0, 10.433306,  0.440040,  0.435894,  0.374000},  // f2 =~ x1
      {0, 4,  7.966681, -0.217557, -0.197241, -0.153099},  // f1 =~ x5
  };
  for (const auto& L : loadings) {
    const auto* r = find_cross_loading(*mi, h.pt, L.marker_ov, L.ind_ov);
    REQUIRE(r != nullptr);
    CHECK(r->mi == doctest::Approx(L.mi).epsilon(0.02));
    CHECK(r->epc == doctest::Approx(L.epc).epsilon(0.02));
    CHECK(r->epc_lv == doctest::Approx(L.sepc_lv).epsilon(0.02));
    CHECK(r->epc_all == doctest::Approx(L.sepc_all).epsilon(0.02));
  }

  struct Cov { int ov_a, ov_b; double mi, epc, sepc_lv, sepc_all; };
  const Cov covs[] = {
      {1, 2, 10.433328,  0.267458,  0.267458,  0.271456},  // x2 ~~ x3
      {0, 1,  9.203159, -0.354376, -0.354376, -0.456186},  // x1 ~~ x2
      {3, 5,  7.966665, -0.278343, -0.278343, -0.767433},  // x4 ~~ x6
  };
  for (const auto& C : covs) {
    const auto* r = find_residual_cov(*mi, h.pt, C.ov_a, C.ov_b);
    REQUIRE(r != nullptr);
    CHECK(r->mi == doctest::Approx(C.mi).epsilon(0.02));
    CHECK(r->epc == doctest::Approx(C.epc).epsilon(0.02));
    CHECK(r->epc_lv == doctest::Approx(C.sepc_lv).epsilon(0.02));
    CHECK(r->epc_all == doctest::Approx(C.sepc_all).epsilon(0.02));
    // Residual covariances are unchanged under std.lv (observed not rescaled).
    CHECK(r->epc_lv == doctest::Approx(r->epc));
  }
}

TEST_CASE("modification_indices: include flags scope the candidate set") {
  auto h = build("f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6");
  SampleStats samp;
  samp.S = {hs_6var_cov()};
  samp.n_obs = {301};
  auto est = magmaan::test::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  inf::ModificationIndexOptions opts;
  opts.candidates = inf::ScoreCandidateSet::WithAbsentRows;
  opts.include_covariances = false;   // cross-loadings only
  auto mi = inf::modification_indices(h.pt, h.rep, samp, *est, opts);
  REQUIRE(mi.has_value());
  for (const auto& r : mi->rows) {
    CHECK(r.candidate.op == magmaan::parse::Op::Measurement);
  }
  CHECK(mi->rows.size() == 6);
}
