#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::FabinVariant;
using magmaan::estimate::fabin_start_values;
using magmaan::estimate::simple_start_values;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatrixRep;
using magmaan::parse::Parser;
using magmaan::spec::LatentStructure;
using magmaan::spec::lavaanify;

namespace {

struct Built {
  LatentStructure pt;
  MatrixRep rep;
};

Built build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = lavaanify(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return Built{std::move(*pt), std::move(*mr)};
}

// Exact 1-factor population covariance: Σ = ψ·λλᵀ + diag(θ).
Eigen::MatrixXd one_factor_cov(const Eigen::VectorXd& lambda, double psi,
                               double theta) {
  const Eigen::Index p = lambda.size();
  Eigen::MatrixXd S = psi * (lambda * lambda.transpose());
  for (Eigen::Index i = 0; i < p; ++i) S(i, i) += theta;
  return S;
}

}  // namespace

TEST_CASE("FABIN3 recovers loadings on an exact 1-factor covariance") {
  // True 1-factor model with marker x1 (loading 1).
  Eigen::VectorXd lambda(5);
  lambda << 1.0, 0.8, 1.2, 0.6, 0.9;
  const Eigen::MatrixXd S = one_factor_cov(lambda, 1.0, 0.5);

  Built b = build("f =~ x1 + x2 + x3 + x4 + x5\n");
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {300};

  auto x0 = fabin_start_values(b.pt, b.rep, samp, {}, FabinVariant::Fabin3);
  REQUIRE(x0.has_value());

  // Every free loading must equal the true loading of its indicator row.
  int checked = 0;
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (!c.used || c.mat != magmaan::model::MatId::Lambda) continue;
    if (b.pt.free[i] == 0) continue;  // marker
    const double got = (*x0)(b.pt.free[i] - 1);
    CHECK(got == doctest::Approx(lambda(c.row)).epsilon(1e-8));
    ++checked;
  }
  CHECK(checked == 4);  // x2..x5
}

TEST_CASE("FABIN2 also recovers the 1-factor truth") {
  Eigen::VectorXd lambda(6);
  lambda << 1.0, 0.5, 1.4, 0.9, 0.7, 1.1;
  const Eigen::MatrixXd S = one_factor_cov(lambda, 0.8, 0.6);

  Built b = build("f =~ x1 + x2 + x3 + x4 + x5 + x6\n");
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {300};

  auto x0 = fabin_start_values(b.pt, b.rep, samp, {}, FabinVariant::Fabin2);
  REQUIRE(x0.has_value());
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (!c.used || c.mat != magmaan::model::MatId::Lambda) continue;
    if (b.pt.free[i] == 0) continue;
    CHECK((*x0)(b.pt.free[i] - 1) ==
          doctest::Approx(lambda(c.row)).epsilon(1e-8));
  }
}

TEST_CASE("FABIN closed form recovers loadings for a 3-indicator factor") {
  Eigen::VectorXd lambda(3);
  lambda << 1.0, 0.7, 1.3;
  const Eigen::MatrixXd S = one_factor_cov(lambda, 1.0, 0.4);

  Built b = build("f =~ x1 + x2 + x3\n");
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {300};

  auto x0 = fabin_start_values(b.pt, b.rep, samp);
  REQUIRE(x0.has_value());
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (!c.used || c.mat != magmaan::model::MatId::Lambda) continue;
    if (b.pt.free[i] == 0) continue;
    CHECK((*x0)(b.pt.free[i] - 1) ==
          doctest::Approx(lambda(c.row)).epsilon(1e-8));
  }
}

TEST_CASE("FABIN differs from simple only in free loadings") {
  Eigen::VectorXd lambda(4);
  lambda << 1.0, 0.8, 1.2, 0.9;
  const Eigen::MatrixXd S = one_factor_cov(lambda, 1.0, 0.5);

  Built b = build("f =~ x1 + x2 + x3 + x4\n");
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {300};

  auto simple = simple_start_values(b.pt, b.rep, samp);
  auto fabin = fabin_start_values(b.pt, b.rep, samp);
  REQUIRE(simple.has_value());
  REQUIRE(fabin.has_value());
  REQUIRE(simple->size() == fabin->size());

  bool any_loading_changed = false;
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (!c.used || b.pt.free[i] == 0) continue;
    const Eigen::Index k = b.pt.free[i] - 1;
    if (c.mat == magmaan::model::MatId::Lambda) {
      if ((*simple)(k) != (*fabin)(k)) any_loading_changed = true;
    } else {
      // Non-loading parameters must be byte-identical to the simple scheme.
      CHECK((*fabin)(k) == (*simple)(k));
    }
  }
  CHECK(any_loading_changed);
}

TEST_CASE("FABIN keeps a user hint on a free loading") {
  Eigen::VectorXd lambda(4);
  lambda << 1.0, 0.8, 1.2, 0.9;
  const Eigen::MatrixXd S = one_factor_cov(lambda, 1.0, 0.5);

  Built b = build("f =~ x1 + x2 + x3 + x4\n");
  SampleStats samp;
  samp.S = {S};
  samp.n_obs = {300};

  // Hint the loading of x3 (find its free index).
  Eigen::Index x3_loading = -1;
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (c.used && c.mat == magmaan::model::MatId::Lambda &&
        b.pt.free[i] > 0 && c.row == 2) {
      x3_loading = b.pt.free[i] - 1;
    }
  }
  REQUIRE(x3_loading >= 0);

  magmaan::spec::Starts hints;
  hints.hint.assign(static_cast<std::size_t>(b.pt.n_free()),
                    std::numeric_limits<double>::quiet_NaN());
  hints.hint[static_cast<std::size_t>(x3_loading)] = 0.123;

  auto x0 = fabin_start_values(b.pt, b.rep, samp, hints);
  REQUIRE(x0.has_value());
  CHECK((*x0)(x3_loading) == doctest::Approx(0.123));
}
