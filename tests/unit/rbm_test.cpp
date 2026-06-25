#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <memory>
#include <string_view>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/frontier/rbm.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

struct BuiltModel {
  std::unique_ptr<magmaan::spec::LatentStructure> pt;
  std::unique_ptr<magmaan::model::MatrixRep> rep;
  magmaan::model::ModelEvaluator ev;
};

BuiltModel build_mean_model(std::string_view src) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  auto pt = magmaan::spec::build(*fp, opts);
  REQUIRE(pt.has_value());
  auto rep = magmaan::model::build_matrix_rep(*pt);
  REQUIRE(rep.has_value());
  auto pt_keep = std::make_unique<magmaan::spec::LatentStructure>(std::move(*pt));
  auto rep_keep = std::make_unique<magmaan::model::MatrixRep>(std::move(*rep));
  auto ev = magmaan::model::ModelEvaluator::build(*pt_keep, *rep_keep);
  REQUIRE(ev.has_value());
  return BuiltModel{std::move(pt_keep), std::move(rep_keep), std::move(*ev)};
}

Eigen::MatrixXd deterministic_z(Eigen::Index n, Eigen::Index p) {
  Eigen::MatrixXd Z(n, p);
  for (Eigen::Index r = 0; r < n; ++r) {
    for (Eigen::Index c = 0; c < p; ++c) {
      const double rr = static_cast<double>(r + 1);
      const double cc = static_cast<double>(c + 1);
      Z(r, c) = std::sin(0.37 * rr * cc) +
                std::cos(0.19 * (rr + 1.0) * (cc + 2.0));
    }
  }
  for (Eigen::Index c = 0; c < p; ++c) {
    Z.col(c).array() -= Z.col(c).mean();
  }
  return Z;
}

magmaan::data::RawData model_raw(const BuiltModel& built,
                                 const Eigen::VectorXd& theta,
                                 Eigen::Index n) {
  auto truth = built.ev.sigma(theta);
  REQUIRE(truth.has_value());
  Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[0]);
  REQUIRE(llt.info() == Eigen::Success);
  magmaan::data::RawData raw;
  raw.X.push_back((deterministic_z(n, truth->sigma[0].rows()) *
                   llt.matrixL().transpose()).rowwise() +
                  truth->mu[0].transpose());
  return raw;
}

BuiltModel rbm_fixture(magmaan::data::RawData& raw,
                       magmaan::data::SampleStats& samp,
                       magmaan::estimate::Estimates& est) {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.65);
  theta0.tail(4) << 0.5, 1.0, 1.5, 2.0;
  raw = model_raw(built, theta0, 64);
  auto samp_or = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp_or.has_value());
  samp = std::move(*samp_or);

  magmaan::optim::OptimOptions fit_opts;
  fit_opts.max_iter = 200;
  auto fit = magmaan::estimate::fit_ml(*built.pt, *built.rep, samp, theta0,
                                       {}, magmaan::estimate::Backend::NloptLbfgs,
                                       fit_opts);
  if (!fit.has_value()) {
    FAIL(fit.error().detail);
    return built;
  }
  est = std::move(*fit);
  return built;
}

template <class Derived>
double max_abs(const Eigen::MatrixBase<Derived>& x) {
  return x.size() == 0 ? 0.0 : x.cwiseAbs().maxCoeff();
}

}  // namespace

TEST_CASE("RBM explicit ML returns finite correction and diagnostics") {
  magmaan::data::RawData raw;
  magmaan::data::SampleStats samp;
  magmaan::estimate::Estimates est;
  auto built = rbm_fixture(raw, samp, est);

  magmaan::estimate::frontier::RBMOptions opts;
  auto rbm = magmaan::estimate::frontier::rbm_explicit_ml(
      *built.pt, *built.rep, samp, raw, est, {}, opts);
  if (!rbm.has_value()) {
    FAIL(rbm.error().detail);
    return;
  }

  const Eigen::Index q = est.theta.size();
  CHECK(rbm->estimates.theta.size() == q);
  CHECK(rbm->correction.size() == q);
  CHECK(rbm->adjustment.size() == q);
  CHECK(rbm->information.rows() == q);
  CHECK(rbm->information.cols() == q);
  CHECK(rbm->meat.rows() == q);
  CHECK(rbm->meat.cols() == q);
  CHECK(rbm->information_reduced.rows() == q);
  CHECK(rbm->information_reduced.cols() == q);
  CHECK(rbm->meat_reduced.rows() == q);
  CHECK(rbm->meat_reduced.cols() == q);
  CHECK(max_abs(rbm->information_reduced - rbm->information) < 1e-8);
  CHECK(max_abs(rbm->meat_reduced - rbm->meat) < 1e-8);
  CHECK(rbm->correction.allFinite());
  CHECK(rbm->adjustment.allFinite());
  CHECK(std::isfinite(rbm->trace_term));
  CHECK(rbm->penalty == doctest::Approx(-0.5 * rbm->trace_term));
  CHECK(rbm->penalized_fmin ==
        doctest::Approx(rbm->estimates.fmin + rbm->penalty_per_observation));
}

TEST_CASE("RBM explicit FIML complete-data path matches ML scaling") {
  magmaan::data::RawData raw;
  magmaan::data::SampleStats samp;
  magmaan::estimate::Estimates est;
  auto built = rbm_fixture(raw, samp, est);

  magmaan::estimate::frontier::RBMOptions opts;
  auto ml = magmaan::estimate::frontier::rbm_explicit_ml(
      *built.pt, *built.rep, samp, raw, est, {}, opts);
  if (!ml.has_value()) {
    FAIL(ml.error().detail);
    return;
  }

  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto fiml = magmaan::estimate::frontier::rbm_explicit_fiml(
      *built.pt, *built.rep, raw, *pack, est, {}, opts);
  if (!fiml.has_value()) {
    FAIL(fiml.error().detail);
    return;
  }

  CHECK(max_abs(fiml->correction - ml->correction) < 1e-6);
  CHECK(max_abs(fiml->adjustment - ml->adjustment) < 1e-5);
  CHECK(max_abs(fiml->information - ml->information) < 1e-6);
  CHECK(max_abs(fiml->meat - ml->meat) < 1e-6);
  CHECK(max_abs(fiml->information_reduced - ml->information_reduced) < 1e-6);
  CHECK(max_abs(fiml->meat_reduced - ml->meat_reduced) < 1e-6);
  CHECK(fiml->trace_term == doctest::Approx(ml->trace_term).epsilon(1e-7));
}

TEST_CASE("RBM explicit continuous ULS returns finite moment diagnostics") {
  magmaan::data::RawData raw;
  magmaan::data::SampleStats samp;
  magmaan::estimate::Estimates ml_est;
  auto built = rbm_fixture(raw, samp, ml_est);

  magmaan::optim::OptimOptions fit_opts;
  fit_opts.max_iter = 250;
  auto uls = magmaan::test::fit_gmm(
      *built.pt, *built.rep, samp, {}, {}, magmaan::estimate::Backend::NloptLbfgs,
      fit_opts);
  if (!uls.has_value()) {
    FAIL(uls.error().detail);
    return;
  }

  magmaan::estimate::frontier::RBMOptions opts;
  auto rbm = magmaan::estimate::frontier::rbm_explicit_continuous_ls(
      *built.pt, *built.rep, samp, *uls, {}, raw,
      magmaan::estimate::ContinuousLsIJWeightMode::Fixed, {}, {}, opts);
  if (!rbm.has_value()) {
    FAIL(rbm.error().detail);
    return;
  }

  const Eigen::Index q = uls->theta.size();
  CHECK(rbm->estimates.theta.size() == q);
  CHECK(rbm->correction.size() == q);
  CHECK(rbm->information.rows() == q);
  CHECK(rbm->information.cols() == q);
  CHECK(rbm->information_reduced.rows() == q);
  CHECK(rbm->information_reduced.cols() == q);
  CHECK(rbm->meat_reduced.rows() == q);
  CHECK(rbm->meat_reduced.cols() == q);
  CHECK(rbm->correction.allFinite());
  CHECK(rbm->adjustment.allFinite());
  CHECK(rbm->information_reduced.allFinite());
  CHECK(rbm->meat_reduced.allFinite());
  CHECK(std::isfinite(rbm->trace_term));
  CHECK(rbm->penalty == doctest::Approx(-0.5 * rbm->trace_term));
}
