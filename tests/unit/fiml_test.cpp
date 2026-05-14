#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/nt/fiml.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/lavaanify.hpp"

namespace {

struct BuiltModel {
  std::unique_ptr<magmaan::spec::LatentStructure> pt;
  std::unique_ptr<magmaan::model::MatrixRep> rep;
  magmaan::model::ModelEvaluator ev;
};

BuiltModel build_mean_model(std::string_view src) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::LavaanifyOptions opts;
  opts.meanstructure = true;
  auto pt = magmaan::spec::lavaanify(*fp, opts);
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

}  // namespace

TEST_CASE("FIML: prepare compresses rows into observed-value patterns") {
  const auto raw = small_missing_raw();
  magmaan::nt::fiml::FIML fiml;
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
  magmaan::nt::fiml::FIML fiml;
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

  magmaan::partable::Starts starts;
  starts.hint.resize(static_cast<std::size_t>(theta0.size()));
  for (Eigen::Index k = 0; k < theta0.size(); ++k) {
    starts.hint[static_cast<std::size_t>(k)] = theta0(k);
  }

  magmaan::optim::LbfgsOptions opts;
  opts.max_iter = 100;
  magmaan::optim::LbfgsOptimizer opt(opts);
  auto est = magmaan::estimate::fit_fiml(*built.pt, *built.rep, raw,
                                         magmaan::nt::fiml::FIML{}, opt,
                                         std::move(starts));
  if (!est.has_value()) {
    FAIL(est.error().detail);
    return;
  }

  magmaan::nt::fiml::FIML fiml;
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
