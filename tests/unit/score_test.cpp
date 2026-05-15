#include <doctest/doctest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/bounds.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/nt/score.hpp"
#include "magmaan/gls/uls.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::Estimates;
using magmaan::gls::ULS;
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
  auto est = magmaan::estimate::fit(h.pt, h.rep, samp);
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
  auto est = magmaan::estimate::fit_bounded(h.pt, h.rep, samp,
                                            magmaan::estimate::Bounds{}, ULS{});
  REQUIRE(est.has_value());

  auto mi = magmaan::nt::infer::modification_indices(h.pt, h.rep, samp, *est, ULS{});
  REQUIRE(mi.has_value());
  CHECK(has_covariance_candidate(*mi));
}

TEST_CASE("score_tests: equality releases are reported in constrained ML models") {
  auto h = build("f =~ a*x1 + b*x2 + x3\na == b");
  const auto j = fixture_json("/fit/0001_one_factor_cfa.fit.json");
  const SampleStats samp = sample_from_fixture(j);
  auto est = magmaan::estimate::fit(h.pt, h.rep, samp);
  REQUIRE(est.has_value());

  auto st = magmaan::nt::infer::score_tests(h.pt, h.rep, samp, *est);
  REQUIRE(st.has_value());
  REQUIRE(st->rows.size() >= 1);
  CHECK(st->rows[0].candidate.kind == magmaan::nt::infer::ScoreCandidateKind::EqualityRelease);
  CHECK(st->rows[0].mi >= 0.0);
  CHECK(std::isfinite(st->rows[0].epc));
}
