#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/start_values.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::data::SampleStats;
using magmaan::estimate::bentler1982_start_values;
using magmaan::estimate::guttman_start_values;
using magmaan::estimate::jamesstein_start_values;
using magmaan::estimate::simple_start_values;
using magmaan::model::build_matrix_rep;
using magmaan::model::MatrixRep;
using magmaan::parse::Parser;
using magmaan::spec::LatentStructure;

namespace {

struct Built {
  LatentStructure pt;
  MatrixRep rep;
};

Built build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE(fp.has_value());
  auto pt = magmaan::spec::build(*fp);
  REQUIRE(pt.has_value());
  auto mr = build_matrix_rep(*pt);
  REQUIRE(mr.has_value());
  return Built{std::move(*pt), std::move(*mr)};
}

// Exact 2-factor population covariance: Σ = Λ Ψ Λᵀ + diag(θ).
// f1 ← x1,x2,x3 (markers x1); f2 ← x4,x5,x6 (marker x4).
Eigen::MatrixXd two_factor_cov() {
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(6, 2);
  L(0, 0) = 1.0; L(1, 0) = 0.8; L(2, 0) = 1.2;
  L(3, 1) = 1.0; L(4, 1) = 0.7; L(5, 1) = 1.3;
  Eigen::MatrixXd Psi(2, 2);
  Psi << 1.0, 0.3, 0.3, 1.0;
  Eigen::MatrixXd S = L * Psi * L.transpose();
  for (Eigen::Index i = 0; i < 6; ++i) S(i, i) += 0.5;
  return S;
}

// True free loadings keyed by observed row (markers are rows 0 and 3).
double true_loading(std::int16_t row) {
  switch (row) {
    case 1: return 0.8;
    case 2: return 1.2;
    case 4: return 0.7;
    case 5: return 1.3;
    default: return 1.0;
  }
}

constexpr const char* kTwoFactor =
    "f1 =~ x1 + x2 + x3\n"
    "f2 =~ x4 + x5 + x6\n"
    "f1 ~~ f2\n";

}  // namespace

TEST_CASE("Guttman/MGM recovers loadings on an exact 2-factor covariance") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};

  auto x0 = guttman_start_values(b.pt, b.rep, samp);
  REQUIRE(x0.has_value());

  int checked = 0;
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (!c.used || c.mat != magmaan::model::MatId::Lambda) continue;
    if (b.pt.free[i] == 0) continue;
    CHECK((*x0)(b.pt.free[i] - 1) ==
          doctest::Approx(true_loading(c.row)).epsilon(1e-6));
    ++checked;
  }
  CHECK(checked == 4);
}

TEST_CASE("Bentler-1982 recovers loadings on an exact 2-factor covariance") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  // A large n keeps the non-iterative PD correction negligible, so Bentler
  // recovers the population loadings to high precision.
  samp.n_obs = {100000000};

  auto x0 = bentler1982_start_values(b.pt, b.rep, samp);
  REQUIRE(x0.has_value());

  int checked = 0;
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (!c.used || c.mat != magmaan::model::MatId::Lambda) continue;
    if (b.pt.free[i] == 0) continue;
    CHECK((*x0)(b.pt.free[i] - 1) ==
          doctest::Approx(true_loading(c.row)).epsilon(1e-5));
    ++checked;
  }
  CHECK(checked == 4);
}

TEST_CASE("James-Stein recovers loadings on an exact 2-factor covariance") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  // Large n keeps the James-Stein shrinkage factor R ≈ ψ/(ψ+θ), so plain JS
  // recovers the population loadings to high precision.
  samp.n_obs = {100000000};

  auto x0 = jamesstein_start_values(b.pt, b.rep, samp);
  REQUIRE(x0.has_value());

  int checked = 0;
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (!c.used || c.mat != magmaan::model::MatId::Lambda) continue;
    if (b.pt.free[i] == 0) continue;
    CHECK((*x0)(b.pt.free[i] - 1) ==
          doctest::Approx(true_loading(c.row)).epsilon(1e-5));
    ++checked;
  }
  CHECK(checked == 4);
}

TEST_CASE("James-Stein falls back to simple for a structural model") {
  Built b = build(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6\n"
      "f2 ~ f1\n");
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};

  auto simple = simple_start_values(b.pt, b.rep, samp);
  auto js = jamesstein_start_values(b.pt, b.rep, samp);
  REQUIRE(simple.has_value());
  REQUIRE(js.has_value());
  CHECK((*js - *simple).cwiseAbs().maxCoeff() == doctest::Approx(0.0));
}

TEST_CASE("Guttman/Bentler differ from simple only in free loadings") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};

  auto simple = simple_start_values(b.pt, b.rep, samp);
  REQUIRE(simple.has_value());
  for (const auto& x0 : {guttman_start_values(b.pt, b.rep, samp),
                         bentler1982_start_values(b.pt, b.rep, samp)}) {
    REQUIRE(x0.has_value());
    REQUIRE(x0->size() == simple->size());
    bool loading_changed = false;
    for (std::size_t i = 0; i < b.pt.size(); ++i) {
      const auto& c = b.rep.cell_for_row[i];
      if (!c.used || b.pt.free[i] == 0) continue;
      const Eigen::Index k = b.pt.free[i] - 1;
      if (c.mat == magmaan::model::MatId::Lambda) {
        if ((*simple)(k) != (*x0)(k)) loading_changed = true;
      } else {
        CHECK((*x0)(k) == (*simple)(k));
      }
    }
    CHECK(loading_changed);
  }
}

TEST_CASE("Guttman/Bentler fall back to simple for a structural model") {
  // A `~` regression makes this a Reduced-form model; both producers are
  // CFA-only and must return the simple baseline unchanged.
  Built b = build(
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ x4 + x5 + x6\n"
      "f2 ~ f1\n");
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};

  auto simple = simple_start_values(b.pt, b.rep, samp);
  auto guttman = guttman_start_values(b.pt, b.rep, samp);
  auto bentler = bentler1982_start_values(b.pt, b.rep, samp);
  REQUIRE(simple.has_value());
  REQUIRE(guttman.has_value());
  REQUIRE(bentler.has_value());
  CHECK((*guttman - *simple).cwiseAbs().maxCoeff() == doctest::Approx(0.0));
  CHECK((*bentler - *simple).cwiseAbs().maxCoeff() == doctest::Approx(0.0));
}

TEST_CASE("Guttman/Bentler keep a user hint on a free loading") {
  Built b = build(kTwoFactor);
  SampleStats samp;
  samp.S = {two_factor_cov()};
  samp.n_obs = {400};

  Eigen::Index x5_loading = -1;
  for (std::size_t i = 0; i < b.pt.size(); ++i) {
    const auto& c = b.rep.cell_for_row[i];
    if (c.used && c.mat == magmaan::model::MatId::Lambda &&
        b.pt.free[i] > 0 && c.row == 4) {
      x5_loading = b.pt.free[i] - 1;
    }
  }
  REQUIRE(x5_loading >= 0);

  magmaan::spec::Starts hints;
  hints.hint.assign(static_cast<std::size_t>(b.pt.n_free()),
                    std::numeric_limits<double>::quiet_NaN());
  hints.hint[static_cast<std::size_t>(x5_loading)] = 0.321;

  auto g = guttman_start_values(b.pt, b.rep, samp, hints);
  auto bn = bentler1982_start_values(b.pt, b.rep, samp, hints);
  REQUIRE(g.has_value());
  REQUIRE(bn.has_value());
  CHECK((*g)(x5_loading) == doctest::Approx(0.321));
  CHECK((*bn)(x5_loading) == doctest::Approx(0.321));
}
