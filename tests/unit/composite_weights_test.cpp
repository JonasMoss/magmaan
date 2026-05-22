#include <doctest/doctest.h>

#include <string>
#include <string_view>
#include <utility>

#include <Eigen/Core>

#include "magmaan/estimate/fit.hpp"
#include "magmaan/measures/composite_weights.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::estimate::Estimates;
using magmaan::measures::composite::composite_weights;
using magmaan::measures::composite::CompositeWeights;
using magmaan::parse::Parser;
using magmaan::spec::build;
using magmaan::spec::BuildOptions;
using magmaan::spec::CompositeMode;
using magmaan::spec::LatentNames;
using magmaan::spec::LatentStructure;

namespace {

struct Model {
  LatentStructure pt;
  LatentNames     names;
};

Model must_build(std::string_view src) {
  auto fp = Parser::parse(src);
  REQUIRE_MESSAGE(fp.has_value(), "parser failed: " << fp.error().detail);
  LatentNames names;
  BuildOptions opts;
  opts.composite_mode = CompositeMode::HenselerOgasawara;
  auto pt = build(*fp, opts, nullptr, &names);
  REQUIRE_MESSAGE(pt.has_value(), "build failed: " << pt.error().detail);
  return Model{std::move(*pt), std::move(names)};
}

}  // namespace

TEST_CASE("composite_weights: K=3 zero loadings recover the H-O permutation") {
  // With every free loading at 0 the K×K block is a permutation matrix, so
  // the emergent composite is carried entirely by its marker indicator (x3).
  Model m = must_build("C <~ x1 + x2 + x3\n y ~ C");
  Estimates est;
  est.theta = Eigen::VectorXd::Zero(m.pt.n_free());
  const Eigen::MatrixXd vcov =
      Eigen::MatrixXd::Identity(m.pt.n_free(), m.pt.n_free());

  auto w = composite_weights(m.pt, m.names, est, vcov);
  REQUIRE_MESSAGE(w.has_value(),
                  "composite_weights failed: " << w.error().detail);
  REQUIRE(w->size() == 1);
  const CompositeWeights& cw = (*w)[0];
  CHECK(cw.composite == "C");
  CHECK(cw.group == 1);
  REQUIRE(cw.weight.size() == 3);
  CHECK(cw.weight(0) == doctest::Approx(0.0));
  CHECK(cw.weight(1) == doctest::Approx(0.0));
  CHECK(cw.weight(2) == doctest::Approx(1.0));
  REQUIRE(cw.se.size() == 3);
  for (Eigen::Index p = 0; p < 3; ++p) CHECK(cw.se(p) >= 0.0);
}

TEST_CASE("composite_weights: K=2 recovers the inverse loading block") {
  // Every free parameter at 0.5 ⇒ loading block Λ = [[0.5, 1], [1, 0.5]];
  // the weight vector is Λ⁻¹ row 0 = [-2/3, 4/3].
  Model m = must_build("C <~ x1 + x2\n y ~ C");
  Estimates est;
  est.theta = Eigen::VectorXd::Constant(m.pt.n_free(), 0.5);
  const Eigen::MatrixXd vcov =
      Eigen::MatrixXd::Identity(m.pt.n_free(), m.pt.n_free());

  auto w = composite_weights(m.pt, m.names, est, vcov);
  REQUIRE_MESSAGE(w.has_value(),
                  "composite_weights failed: " << w.error().detail);
  REQUIRE(w->size() == 1);
  const CompositeWeights& cw = (*w)[0];
  REQUIRE(cw.weight.size() == 2);
  CHECK(cw.weight(0) == doctest::Approx(-2.0 / 3.0));
  CHECK(cw.weight(1) == doctest::Approx(4.0 / 3.0));
  CHECK(cw.se(0) > 0.0);
  CHECK(cw.se(1) > 0.0);
}

TEST_CASE("composite_weights: a model with no composites yields an empty result") {
  Model m = must_build("f =~ x1 + x2 + x3");
  Estimates est;
  est.theta = Eigen::VectorXd::Zero(m.pt.n_free());
  const Eigen::MatrixXd vcov =
      Eigen::MatrixXd::Identity(m.pt.n_free(), m.pt.n_free());
  auto w = composite_weights(m.pt, m.names, est, vcov);
  REQUIRE(w.has_value());
  CHECK(w->empty());
}
