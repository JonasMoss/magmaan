#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

namespace {

struct BuiltModel {
  std::unique_ptr<magmaan::spec::LatentStructure> pt;
  std::unique_ptr<magmaan::model::MatrixRep> rep;
  magmaan::model::ModelEvaluator ev;
};

BuiltModel build_mean_model(std::string_view src, int n_groups = 1) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = true;
  opts.n_groups = n_groups;
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

magmaan::data::RawData small_missing_raw() {
  const double na = std::numeric_limits<double>::quiet_NaN();
  magmaan::data::RawData raw;
  Eigen::MatrixXd X(5, 3);
  X << 1.0, 2.0, 3.0,
       2.0, 3.0, 4.0,
       1.5, na,  2.5,
       na,  2.2, 3.5,
       0.8, 1.9, na;
  raw.X.push_back(X);

  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(5, 3);
  M << 1, 1, 1,
       1, 1, 1,
       1, 0, 1,
       0, 1, 1,
       1, 1, 0;
  raw.mask.push_back(M);
  return raw;
}

magmaan::data::RawData well_conditioned_missing_raw() {
  const double na = std::numeric_limits<double>::quiet_NaN();
  magmaan::data::RawData raw;
  Eigen::MatrixXd X(8, 3);
  X << 1.0, 2.0, 3.0,
       2.0, 1.0, 4.0,
       3.0, 4.0, 2.0,
       4.0, 3.0, 5.0,
       5.0, 5.0, 1.0,
       6.0, 4.0, 6.0,
       7.0, 7.0, 4.0,
       8.0, 6.0, 7.0;
  X(2, 1) = na;
  X(5, 2) = na;
  X(6, 0) = na;
  raw.X.push_back(X);

  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(8, 3);
  M << 1, 1, 1,
       1, 1, 1,
       1, 0, 1,
       1, 1, 1,
       1, 1, 1,
       1, 1, 0,
       0, 1, 1,
       1, 1, 1;
  raw.mask.push_back(M);
  return raw;
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

double log_det_pd(const Eigen::MatrixXd& A) {
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  REQUIRE(llt.info() == Eigen::Success);
  const auto L = llt.matrixL();
  double out = 0.0;
  for (Eigen::Index i = 0; i < A.rows(); ++i) out += std::log(L(i, i));
  return 2.0 * out;
}

}  // namespace

TEST_CASE("FIML: prepare compresses rows into observed-value patterns") {
  const auto raw = small_missing_raw();
  magmaan::estimate::fiml::FIML fiml;
  auto cache_or = fiml.prepare(raw);
  REQUIRE(cache_or.has_value());
  const auto& cache = *cache_or;

  CHECK(cache.n_total == 5);
  CHECK(cache.patterns.size() == 4u);
  bool saw_full = false;
  bool saw_x1_x3 = false;
  bool saw_x2_x3 = false;
  bool saw_x1_x2 = false;
  for (const auto& pat : cache.patterns) {
    saw_full  = saw_full  || (pat.observed == std::vector<Eigen::Index>{0, 1, 2} && pat.n_obs == 2);
    saw_x1_x3 = saw_x1_x3 || (pat.observed == std::vector<Eigen::Index>{0, 2} && pat.n_obs == 1);
    saw_x2_x3 = saw_x2_x3 || (pat.observed == std::vector<Eigen::Index>{1, 2} && pat.n_obs == 1);
    saw_x1_x2 = saw_x1_x2 || (pat.observed == std::vector<Eigen::Index>{0, 1} && pat.n_obs == 1);
  }
  CHECK(saw_full);
  CHECK(saw_x1_x3);
  CHECK(saw_x2_x3);
  CHECK(saw_x1_x2);
}

TEST_CASE("FIML: analytic gradient matches finite differences") {
  auto built = build_mean_model("f =~ x1 + x2 + x3");
  const auto raw = small_missing_raw();
  magmaan::estimate::fiml::FIML fiml;
  auto cache = fiml.prepare(raw);
  REQUIRE(cache.has_value());

  Eigen::VectorXd theta(static_cast<Eigen::Index>(built.ev.n_free()));
  for (Eigen::Index k = 0; k < theta.size(); ++k) theta(k) = 0.7;
  theta.tail(3) << 1.0, 2.0, 3.0;  // indicator intercept starts

  auto eval = built.ev.evaluate(theta, true, true);
  REQUIRE(eval.has_value());
  auto vg = fiml.value_gradient(raw, *cache, eval->moments,
                                eval->J_sigma, eval->J_mu);
  REQUIRE(vg.has_value());

  Eigen::VectorXd g_fd(theta.size());
  const double h = 1e-6;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    Eigen::VectorXd tp = theta; tp(k) += h;
    Eigen::VectorXd tm = theta; tm(k) -= h;
    auto ep = built.ev.sigma(tp);
    auto em = built.ev.sigma(tm);
    REQUIRE(ep.has_value());
    REQUIRE(em.has_value());
    auto fp = fiml.value(raw, *cache, *ep);
    auto fm = fiml.value(raw, *cache, *em);
    REQUIRE(fp.has_value());
    REQUIRE(fm.has_value());
    g_fd(k) = (*fp - *fm) / (2.0 * h);
  }

  const double diff = (vg->gradient - g_fd).cwiseAbs().maxCoeff();
  CHECK(diff < 1e-5);
}

TEST_CASE("fit_fiml: complete-data path fits a saturated mean CFA near zero gradient") {
  auto built = build_mean_model("f =~ x1 + x2 + x3");

  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  for (Eigen::Index k = 0; k < theta0.size(); ++k) theta0(k) = 0.7;
  theta0.tail(3) << 1.0, 2.0, 3.0;
  auto truth = built.ev.sigma(theta0);
  REQUIRE(truth.has_value());
  Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[0]);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  const double a = std::sqrt(3.0);
  Eigen::MatrixXd Z(6, 3);
  Z <<  a, 0.0, 0.0,
       -a, 0.0, 0.0,
       0.0,  a, 0.0,
       0.0, -a, 0.0,
       0.0, 0.0,  a,
       0.0, 0.0, -a;

  magmaan::data::RawData raw;
  raw.X.push_back((Z * L.transpose()).rowwise() + truth->mu[0].transpose());

  magmaan::optim::LbfgsOptions opts;
  opts.max_iter = 100;
  auto est = magmaan::estimate::fit_fiml(*built.pt, *built.rep, raw, theta0,
                                         magmaan::estimate::fiml::FIML{}, opts);
  if (!est.has_value()) {
    FAIL(est.error().detail);
    return;
  }

  magmaan::estimate::fiml::FIML fiml;
  auto cache = fiml.prepare(raw);
  REQUIRE(cache.has_value());
  auto ev = magmaan::model::ModelEvaluator::build(*built.pt, *built.rep);
  REQUIRE(ev.has_value());
  auto eval = ev->evaluate(est->theta, true, true);
  REQUIRE(eval.has_value());
  auto vg = fiml.value_gradient(raw, *cache, eval->moments,
                                eval->J_sigma, eval->J_mu);
  if (!vg.has_value()) {
    FAIL(vg.error().detail);
    return;
  }
  CHECK(vg->gradient.cwiseAbs().maxCoeff() < 1e-4);
}

TEST_CASE("FIML complete-data objective and gradient match ML up to constants") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4", /*n_groups=*/2);

  Eigen::VectorXd theta(static_cast<Eigen::Index>(built.ev.n_free()));
  theta.setConstant(0.55);
  auto truth = built.ev.sigma(theta);
  REQUIRE(truth.has_value());

  magmaan::data::RawData raw;
  for (std::size_t b = 0; b < truth->sigma.size(); ++b) {
    Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[b]);
    REQUIRE(llt.info() == Eigen::Success);
    const Eigen::MatrixXd L = llt.matrixL();
    const Eigen::Index n = b == 0 ? 31 : 27;
    Eigen::MatrixXd Z = deterministic_z(n, truth->sigma[b].rows());
    if (b == 1) Z.array() *= 1.12;
    raw.X.push_back((Z * L.transpose()).rowwise() +
                    truth->mu[b].transpose());
  }
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  auto eval = built.ev.evaluate(theta, true, true);
  REQUIRE(eval.has_value());

  magmaan::estimate::fiml::FIML fiml;
  auto fiml_cache = fiml.prepare(raw);
  REQUIRE(fiml_cache.has_value());
  auto fiml_vg = fiml.value_gradient(raw, *fiml_cache, eval->moments,
                                     eval->J_sigma, eval->J_mu);
  REQUIRE(fiml_vg.has_value());

  auto ml_cache = magmaan::estimate::ml_prepare(*samp);
  REQUIRE(ml_cache.has_value());
  auto ml_vg = magmaan::estimate::ml_value_gradient(*samp, *ml_cache, eval->moments,
                                              eval->J_sigma, eval->J_mu);
  REQUIRE(ml_vg.has_value());

  double constant = 0.0;
  for (std::size_t b = 0; b < samp->S.size(); ++b) {
    const double weight = static_cast<double>(samp->n_obs[b]) /
                          static_cast<double>(ml_cache->n_total);
    constant += weight *
        (log_det_pd(samp->S[b]) + static_cast<double>(samp->S[b].rows()));
  }

  CHECK(fiml_vg->value ==
        doctest::Approx(ml_vg->value + constant).epsilon(1e-10));
  CHECK((fiml_vg->gradient - ml_vg->gradient).cwiseAbs().maxCoeff() < 1e-10);
}

TEST_CASE("fiml_extras: complete data matches SampleStats fit_extras") {
  auto built = build_mean_model("f =~ x1 + x2 + x3");

  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  for (Eigen::Index k = 0; k < theta0.size(); ++k) theta0(k) = 0.7;
  theta0.tail(3) << 1.0, 2.0, 3.0;
  auto truth = built.ev.sigma(theta0);
  REQUIRE(truth.has_value());
  Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[0]);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  const double a = std::sqrt(3.0);
  Eigen::MatrixXd Z(6, 3);
  Z <<  a, 0.0, 0.0,
       -a, 0.0, 0.0,
       0.0,  a, 0.0,
       0.0, -a, 0.0,
       0.0, 0.0,  a,
       0.0, 0.0, -a;

  magmaan::data::RawData raw;
  raw.X.push_back((Z * L.transpose()).rowwise() + truth->mu[0].transpose());
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  magmaan::optim::LbfgsOptions opts;
  opts.max_iter = 100;
  auto est = magmaan::estimate::fit_fiml(*built.pt, *built.rep, raw, theta0,
                                         magmaan::estimate::fiml::FIML{}, opts);
  REQUIRE(est.has_value());

  auto fiml_fx = magmaan::estimate::fiml::fiml_extras(*built.pt, *built.rep, raw, *est);
  REQUIRE(fiml_fx.has_value());
  auto ml_fx = magmaan::measures::fit_extras(*built.pt, *built.rep, *samp, *est);
  REQUIRE(ml_fx.has_value());

  CHECK(fiml_fx->logl == doctest::Approx(ml_fx->logl).epsilon(1e-9));
  CHECK(fiml_fx->unrestricted_logl ==
        doctest::Approx(ml_fx->unrestricted_logl).epsilon(1e-9));
  CHECK(fiml_fx->npar == ml_fx->npar);
  CHECK(fiml_fx->ntotal == ml_fx->ntotal);
}

TEST_CASE("fiml_baseline_chi2: complete data matches SampleStats baseline") {
  auto built = build_mean_model("f =~ x1 + x2 + x3");

  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  for (Eigen::Index k = 0; k < theta0.size(); ++k) theta0(k) = 0.7;
  theta0.tail(3) << 1.0, 2.0, 3.0;
  auto truth = built.ev.sigma(theta0);
  REQUIRE(truth.has_value());
  Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[0]);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  const double a = std::sqrt(3.0);
  Eigen::MatrixXd Z(6, 3);
  Z <<  a, 0.0, 0.0,
       -a, 0.0, 0.0,
       0.0,  a, 0.0,
       0.0, -a, 0.0,
       0.0, 0.0,  a,
       0.0, 0.0, -a;

  magmaan::data::RawData raw;
  raw.X.push_back((Z * L.transpose()).rowwise() + truth->mu[0].transpose());
  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  auto fiml_bl = magmaan::estimate::fiml::fiml_baseline_chi2(raw);
  REQUIRE(fiml_bl.has_value());
  const auto ml_bl = magmaan::measures::baseline_chi2(*samp);

  CHECK(fiml_bl->chi2 == doctest::Approx(ml_bl.chi2).epsilon(1e-10));
  CHECK(fiml_bl->df == ml_bl.df);
}

TEST_CASE("fiml_robust_mlr: multi-block H1 trace handles unequal row counts") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4", /*n_groups=*/2);

  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.6);
  auto truth = built.ev.sigma(theta0);
  REQUIRE(truth.has_value());
  REQUIRE(truth->sigma.size() == 2);

  magmaan::data::RawData raw;
  for (std::size_t b = 0; b < truth->sigma.size(); ++b) {
    Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[b]);
    REQUIRE(llt.info() == Eigen::Success);
    const Eigen::MatrixXd L = llt.matrixL();
    const Eigen::Index n = (b == 0) ? 24 : 19;
    Eigen::MatrixXd Z = deterministic_z(n, truth->sigma[b].rows());
    if (b == 1) Z.array() *= 1.15;
    raw.X.push_back((Z * L.transpose()).rowwise() +
                    truth->mu[b].transpose());
  }

  magmaan::estimate::Estimates est;
  est.theta = theta0;

  constexpr int df = 4;
  auto rob = magmaan::estimate::fiml::fiml_robust_mlr(*built.pt, *built.rep, raw,
                                                est, df, /*chisq=*/8.0);
  REQUIRE_MESSAGE(rob.has_value(),
      "fiml_robust_mlr failed: " <<
      (rob.has_value() ? "" : rob.error().detail));

  CHECK(rob->ntotal == 43);
  CHECK(rob->df == df);
  CHECK(rob->se.size() == theta0.size());
  CHECK(rob->vcov.rows() == theta0.size());
  CHECK(rob->vcov.cols() == theta0.size());
  CHECK(std::isfinite(rob->trace_ugamma_h1));
  CHECK(std::isfinite(rob->trace_ugamma_h0));
  CHECK(std::isfinite(rob->trace_ugamma));
  CHECK(std::isfinite(rob->scaling_factor));
  CHECK(std::isfinite(rob->chisq_scaled));
}

TEST_CASE("fiml_baseline_chi2: missing data produces finite baseline") {
  const auto raw = well_conditioned_missing_raw();
  auto bl = magmaan::estimate::fiml::fiml_baseline_chi2(raw);
  if (!bl.has_value()) {
    FAIL(bl.error().detail);
    return;
  }
  CHECK(std::isfinite(bl->chi2));
  CHECK(bl->chi2 >= 0.0);
  CHECK(bl->df == 3);
}

TEST_CASE("fiml_baseline_chi2: rejects a column with no observed values") {
  auto raw = small_missing_raw();
  raw.mask[0].col(1).setZero();
  for (Eigen::Index r = 0; r < raw.X[0].rows(); ++r) {
    raw.X[0](r, 1) = std::numeric_limits<double>::quiet_NaN();
  }

  auto bl = magmaan::estimate::fiml::fiml_baseline_chi2(raw);
  REQUIRE(!bl.has_value());
  CHECK(bl.error().detail.find("column 1 has no observed values") !=
        std::string::npos);
}
