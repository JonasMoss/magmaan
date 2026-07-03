#include <doctest/doctest.h>

#include <cmath>

#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/frontier/sam.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"
#include "magmaan/spec/partable.hpp"

using magmaan::estimate::frontier::fit_sam;
using magmaan::estimate::frontier::mapping_matrix;
using magmaan::estimate::frontier::SamMapping;
using magmaan::estimate::frontier::SamMethod;
using magmaan::estimate::frontier::SamOptions;
using magmaan::estimate::frontier::SamSe;

namespace {

// A 6-indicator, 2-factor loading matrix with a clean simple structure, plus a
// diagonal residual Θ and the model-implied S = Λ Ψ Λᵀ + Θ (Ψ = I here).
struct TwoFactor {
  Eigen::MatrixXd Lambda;
  Eigen::MatrixXd Theta;
  Eigen::MatrixXd S;
};

TwoFactor make_two_factor() {
  Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(6, 2);
  Lambda(0, 0) = 1.0;
  Lambda(1, 0) = 0.8;
  Lambda(2, 0) = 1.2;
  Lambda(3, 1) = 1.0;
  Lambda(4, 1) = 0.9;
  Lambda(5, 1) = 1.1;
  Eigen::VectorXd theta_diag(6);
  theta_diag << 0.5, 0.6, 0.7, 0.55, 0.65, 0.45;
  Eigen::MatrixXd Theta = theta_diag.asDiagonal();
  Eigen::MatrixXd Psi = Eigen::MatrixXd::Identity(2, 2);
  Psi(0, 1) = Psi(1, 0) = 0.3;
  Eigen::MatrixXd S = Lambda * Psi * Lambda.transpose() + Theta;
  return {Lambda, Theta, S};
}

}  // namespace

TEST_CASE("mapping_matrix satisfies M*Lambda = I for all three methods") {
  const TwoFactor tf = make_two_factor();
  const Eigen::MatrixXd I2 = Eigen::MatrixXd::Identity(2, 2);

  for (SamMapping method : {SamMapping::ML, SamMapping::GLS, SamMapping::ULS}) {
    auto M_or = mapping_matrix(tf.Lambda, tf.Theta, tf.S, method);
    REQUIRE(M_or.has_value());
    if (!M_or.has_value()) return;
    const Eigen::MatrixXd& M = *M_or;
    CHECK(M.rows() == 2);
    CHECK(M.cols() == 6);
    const Eigen::MatrixXd ML = M * tf.Lambda;
    CHECK((ML - I2).cwiseAbs().maxCoeff() == doctest::Approx(0.0).epsilon(1e-10));
  }
}

TEST_CASE("mapping_matrix ML and GLS differ off the model manifold") {
  const TwoFactor tf = make_two_factor();
  // Perturb S away from the Λ Ψ Λᵀ + Θ manifold; ML (uses Θ) and GLS (uses S)
  // must then produce different mappings even though both satisfy M·Λ = I.
  Eigen::MatrixXd Sperturbed = tf.S;
  Sperturbed(0, 0) += 0.4;
  auto ml = mapping_matrix(tf.Lambda, tf.Theta, Sperturbed, SamMapping::ML);
  auto gls = mapping_matrix(tf.Lambda, tf.Theta, Sperturbed, SamMapping::GLS);
  REQUIRE(ml.has_value());
  REQUIRE(gls.has_value());
  if (!ml.has_value() || !gls.has_value()) return;
  CHECK((*ml - *gls).cwiseAbs().maxCoeff() > 1e-6);
}

TEST_CASE("mapping_matrix rejects degenerate inputs") {
  const TwoFactor tf = make_two_factor();
  // Non-PD Θ under ML is rejected (Wall-Amemiya zero-residual case deferred).
  Eigen::MatrixXd bad_theta = tf.Theta;
  bad_theta(0, 0) = 0.0;
  auto ml = mapping_matrix(tf.Lambda, bad_theta, tf.S, SamMapping::ML);
  CHECK_FALSE(ml.has_value());

  // More latents than indicators is rejected.
  Eigen::MatrixXd fat = Eigen::MatrixXd::Identity(2, 3);
  auto uls = mapping_matrix(fat, Eigen::MatrixXd(), Eigen::MatrixXd(),
                            SamMapping::ULS);
  CHECK_FALSE(uls.has_value());
}

namespace {

// lavaan's model sample covariance for the 2-factor + structural-path fixture
// (external/... sam_ref.R, N = 300), ordered x1,x2,x3,y1,y2,y3.
Eigen::MatrixXd lavaan_S() {
  Eigen::MatrixXd S(6, 6);
  S << 1.5385323080, 0.7646196841, 1.1749898429, 0.5595638960, 0.5195712638, 0.5905333215,
       0.7646196841, 1.2859399879, 0.9463390813, 0.4506578337, 0.4341929790, 0.5396205706,
       1.1749898429, 0.9463390813, 1.9836289068, 0.7661793862, 0.7363430095, 0.8168780232,
       0.5595638960, 0.4506578337, 0.7661793862, 1.8177216554, 1.1013066447, 1.3007371312,
       0.5195712638, 0.4341929790, 0.7363430095, 1.1013066447, 1.5875257054, 1.0802328369,
       0.5905333215, 0.5396205706, 0.8168780232, 1.3007371312, 1.0802328369, 1.7845931312;
  return S;
}

struct BuiltModel {
  magmaan::spec::LatentStructure pt;
  magmaan::spec::LatentNames     names;
  magmaan::model::MatrixRep      rep;
};

BuiltModel build_two_factor() {
  const char* src =
      "f1 =~ x1 + x2 + x3\n"
      "f2 =~ y1 + y2 + y3\n"
      "f2 ~ f1\n";
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  magmaan::spec::Starts starts;
  magmaan::spec::LatentNames names;
  auto pt = magmaan::spec::build(*fp, opts, &starts, &names);
  REQUIRE(pt.has_value());
  auto mr = magmaan::model::build_matrix_rep(*pt, &names);
  REQUIRE(mr.has_value());
  return BuiltModel{std::move(*pt), std::move(names), std::move(*mr)};
}

magmaan::data::SampleStats lavaan_samp() {
  magmaan::data::SampleStats s;
  s.S = {lavaan_S()};
  s.n_obs = {300};
  return s;
}

Eigen::MatrixXd robust_raw_X() {
  const Eigen::Index n = 240;
  Eigen::MatrixXd X(n, 6);
  for (Eigen::Index r = 0; r < n; ++r) {
    const int i = static_cast<int>(r + 1);
    const double t = static_cast<double>(i);
    const double f1 = std::sin(0.071 * t) + 0.35 * std::cos(0.017 * t) +
                      0.002 * static_cast<double>((i % 11) - 5);
    const double f2 = 0.62 * f1 + 0.75 * std::cos(0.053 * t) +
                      0.20 * std::sin(0.113 * t);
    X(r, 0) = f1 + 0.45 * std::sin(0.19 * t);
    X(r, 1) = 0.82 * f1 + 0.55 * std::cos(0.13 * t) +
              0.10 * static_cast<double>((i % 5) - 2);
    X(r, 2) = 1.18 * f1 + 0.50 * std::sin(0.07 * t + 0.4);
    X(r, 3) = f2 + 0.50 * std::cos(0.11 * t);
    X(r, 4) = 0.92 * f2 + 0.45 * std::sin(0.09 * t) +
              0.08 * static_cast<double>((i % 7) - 3);
    X(r, 5) = 1.08 * f2 + 0.40 * std::cos(0.15 * t + 0.2);
  }
  return X;
}

magmaan::data::RawData robust_raw() {
  magmaan::data::RawData raw;
  raw.X = {robust_raw_X()};
  return raw;
}

}  // namespace

TEST_CASE("fit_sam local point estimates match lavaan::sam(sam.method=local)") {
  BuiltModel bm = build_two_factor();
  SamOptions opts;
  opts.method = SamMethod::Local;
  opts.local.mapping = SamMapping::ML;
  opts.se = SamSe::None;

  auto res = fit_sam(bm.pt, bm.rep, bm.names, lavaan_samp(), opts);
  REQUIRE(res.has_value());
  if (!res.has_value()) return;

  // latent_order is [f1, f2] (first-seen). lavaan's implied VETA:
  REQUIRE(res->VETA.rows() == 2);
  CHECK(res->VETA(0, 0) == doctest::Approx(0.949364).epsilon(1e-4));
  CHECK(res->VETA(1, 1) == doctest::Approx(1.326106).epsilon(1e-4));
  CHECK(res->VETA(0, 1) == doctest::Approx(0.632308).epsilon(1e-4));

  // Structural coefficient beta = f2~f1 = VETA[f1,f2]/VETA[f1,f1] (just-identified).
  const double beta = res->VETA(0, 1) / res->VETA(0, 0);
  CHECK(beta == doctest::Approx(0.66603).epsilon(1e-4));
  REQUIRE(res->theta.size() == 13);
  CHECK(res->theta(4) == doctest::Approx(beta).epsilon(1e-6));

  // Measurement block loadings and residuals (block f1: x1,x2,x3).
  REQUIRE(res->measurement.size() == 2);
  const auto& b0 = res->measurement[0];
  CHECK(b0.Lambda(0, 0) == doctest::Approx(1.0));
  CHECK(b0.Lambda(1, 0) == doctest::Approx(0.805402).epsilon(1e-4));
  CHECK(b0.Lambda(2, 0) == doctest::Approx(1.23766).epsilon(1e-4));
  CHECK(b0.Theta(0, 0) == doctest::Approx(0.589168).epsilon(1e-4));

  for (double r : res->reliability) CHECK((r > 0.0));
}

TEST_CASE("fit_sam global equals local for a clean two-block model") {
  BuiltModel bm = build_two_factor();
  SamOptions lopts;
  lopts.method = SamMethod::Local;
  lopts.se = SamSe::None;
  SamOptions gopts;
  gopts.method = SamMethod::Global;
  gopts.se = SamSe::None;

  auto lres = fit_sam(bm.pt, bm.rep, bm.names, lavaan_samp(), lopts);
  auto gres = fit_sam(bm.pt, bm.rep, bm.names, lavaan_samp(), gopts);
  REQUIRE(lres.has_value());
  REQUIRE(gres.has_value());
  if (!lres.has_value() || !gres.has_value()) return;
  CHECK((lres->VETA - gres->VETA).cwiseAbs().maxCoeff() ==
        doctest::Approx(0.0).epsilon(1e-8));
}

TEST_CASE("fit_sam GLS and ULS mappings match lavaan local.options M.method") {
  BuiltModel bm = build_two_factor();

  struct Target { SamMapping mapping; double v00, v11, v01; };
  const Target targets[] = {
      {SamMapping::GLS, 0.936503, 1.315407, 0.632666},
      {SamMapping::ULS, 0.949364, 1.326089, 0.633893},
  };
  for (const Target& t : targets) {
    SamOptions opts;
    opts.method = SamMethod::Local;
    opts.local.mapping = t.mapping;
    opts.se = SamSe::None;
    auto res = fit_sam(bm.pt, bm.rep, bm.names, lavaan_samp(), opts);
    REQUIRE(res.has_value());
    if (!res.has_value()) return;
    REQUIRE(res->theta.size() == 13);
    CHECK(res->VETA(0, 0) == doctest::Approx(t.v00).epsilon(1e-4));
    CHECK(res->VETA(1, 1) == doctest::Approx(t.v11).epsilon(1e-4));
    CHECK(res->VETA(0, 1) == doctest::Approx(t.v01).epsilon(1e-4));
  }
}

TEST_CASE("fit_sam twostep standard errors match lavaan::sam(se=twostep)") {
  BuiltModel bm = build_two_factor();
  SamOptions opts;
  opts.method = SamMethod::Local;
  opts.se = SamSe::Twostep;
  auto res = fit_sam(bm.pt, bm.rep, bm.names, lavaan_samp(), opts);
  REQUIRE(res.has_value());
  if (!res.has_value()) return;
  REQUIRE(res->theta.size() == 13);
  REQUIRE(res->se.size() == 13);
  // Joint free order mirrors lavaan: f1=~x2(0) f1=~x3(1) f2=~y2(2) f2=~y3(3)
  // f2~f1(4) x1~~x1(5)..y3~~y3(10) f1~~f1(11) f2~~f2(12).
  CHECK(res->theta(4) == doctest::Approx(0.666030).epsilon(1e-4));
  CHECK(res->se(0) == doctest::Approx(0.070991).epsilon(1e-4));   // f1=~x2
  CHECK(res->se(4) == doctest::Approx(0.081207).epsilon(1e-4));   // f2~f1
  CHECK(res->se(5) == doctest::Approx(0.078815).epsilon(1e-4));   // x1~~x1
  CHECK(res->se(11) == doctest::Approx(0.131773).epsilon(1e-4));  // f1~~f1
  CHECK(res->se(12) == doctest::Approx(0.117461).epsilon(1e-4));  // f2~~f2
}

TEST_CASE("fit_sam twostep.robust structural SEs match lavaan raw-data SAM") {
  BuiltModel bm = build_two_factor();
  SamOptions opts;
  opts.method = SamMethod::Local;
  opts.se = SamSe::TwostepRobust;

  auto res = fit_sam(bm.pt, bm.rep, bm.names, robust_raw(), opts);
  REQUIRE(res.has_value());
  if (!res.has_value()) return;
  REQUIRE(res->theta.size() == 13);
  REQUIRE(res->se.size() == 13);

  CHECK(res->structural.theta(0) == doctest::Approx(0.798053).epsilon(1e-4));
  CHECK(res->structural.theta(1) == doctest::Approx(0.611640).epsilon(1e-4));
  CHECK(res->structural.theta(2) == doctest::Approx(0.305760).epsilon(1e-4));
  CHECK(res->se(4) == doctest::Approx(0.050554).epsilon(1e-4));   // f2~f1
  CHECK(res->se(11) == doctest::Approx(0.053457).epsilon(1e-4));  // f1~~f1
  CHECK(res->se(12) == doctest::Approx(0.023753).epsilon(1e-4));  // f2~~f2
}

TEST_CASE("fit_sam twostep.robust rejects unsupported entry points") {
  BuiltModel bm = build_two_factor();
  const magmaan::data::RawData raw = robust_raw();

  SamOptions robust;
  robust.method = SamMethod::Local;
  robust.se = SamSe::TwostepRobust;
  CHECK_FALSE(fit_sam(bm.pt, bm.rep, bm.names, lavaan_samp(), robust).has_value());

  SamOptions global = robust;
  global.method = SamMethod::Global;
  CHECK_FALSE(fit_sam(bm.pt, bm.rep, bm.names, raw, global).has_value());

  SamOptions means = robust;
  means.meanstructure = true;
  CHECK_FALSE(fit_sam(bm.pt, bm.rep, bm.names, raw, means).has_value());

  SamOptions alpha = robust;
  alpha.local.alpha_correction = 1;
  CHECK_FALSE(fit_sam(bm.pt, bm.rep, bm.names, raw, alpha).has_value());

  SamOptions naive;
  naive.se = SamSe::Naive;
  CHECK_FALSE(fit_sam(bm.pt, bm.rep, bm.names, lavaan_samp(), naive).has_value());
}
