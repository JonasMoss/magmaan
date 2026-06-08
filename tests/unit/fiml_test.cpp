#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/robust/lr_test_satorra.hpp"
#include "magmaan/robust/restriction.hpp"
#include "magmaan/robust/robust.hpp"
#include "magmaan/estimate/nl_constraints.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/optim/problem.hpp"
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

magmaan::data::RawData
model_missing_raw(const BuiltModel& built,
                  const Eigen::Ref<const Eigen::VectorXd>& theta,
                  const std::vector<Eigen::Index>& n_per_group) {
  auto truth = built.ev.sigma(theta);
  REQUIRE(truth.has_value());
  REQUIRE(truth->sigma.size() == n_per_group.size());

  magmaan::data::RawData raw;
  for (std::size_t b = 0; b < truth->sigma.size(); ++b) {
    const Eigen::Index p = truth->sigma[b].rows();
    Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[b]);
    REQUIRE(llt.info() == Eigen::Success);
    Eigen::MatrixXd Z = deterministic_z(n_per_group[b], p);
    if (b > 0) Z.array() *= 1.0 + 0.07 * static_cast<double>(b);
    Eigen::MatrixXd X =
        (Z * llt.matrixL().transpose()).rowwise() + truth->mu[b].transpose();
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M =
        Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Ones(
            X.rows(), X.cols());
    for (Eigen::Index r = 0; r < X.rows(); ++r) {
      if (r % 5 == 0) {
        const Eigen::Index c = (r + static_cast<Eigen::Index>(b)) % p;
        M(r, c) = 0;
        X(r, c) = std::numeric_limits<double>::quiet_NaN();
      } else if (r % 11 == 0 && p > 2) {
        const Eigen::Index c = (r + 2 + static_cast<Eigen::Index>(b)) % p;
        M(r, c) = 0;
        X(r, c) = std::numeric_limits<double>::quiet_NaN();
      }
    }
    raw.X.push_back(std::move(X));
    raw.mask.push_back(std::move(M));
  }
  return raw;
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

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 100;
  auto est = magmaan::estimate::fit_fiml(*built.pt, *built.rep, raw, theta0,
                                         magmaan::estimate::fiml::FIML{},
                                         magmaan::estimate::Backend::NloptLbfgs,
                                         opts);
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

TEST_CASE("fit_fiml: nonlinear equality constraints accept NLopt SLSQP") {
  auto built = build_mean_model("f =~ x1 + a*x2 + b*x3\na == b^2");

  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.7);
  theta0(0) = 0.49;
  theta0(1) = 0.7;
  theta0.tail(3) << 1.0, 2.0, 3.0;
  auto truth = built.ev.sigma(theta0);
  REQUIRE(truth.has_value());
  Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[0]);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  const Eigen::MatrixXd Z = deterministic_z(24, 3);

  magmaan::data::RawData raw;
  raw.X.push_back((Z * L.transpose()).rowwise() + truth->mu[0].transpose());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 300;
  auto est = magmaan::estimate::fit_fiml(
      *built.pt, *built.rep, raw, theta0, magmaan::estimate::fiml::FIML{},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(est.has_value(), "SLSQP constrained FIML failed: "
      << (est.has_value() ? std::string{} : est.error().detail));

  auto nl = magmaan::estimate::build_nl_constraints(*built.pt);
  CHECK(std::abs(nl.h(est->theta)(0)) < 1e-5);
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

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 100;
  auto est = magmaan::estimate::fit_fiml(*built.pt, *built.rep, raw, theta0,
                                         magmaan::estimate::fiml::FIML{},
                                         magmaan::estimate::Backend::NloptLbfgs,
                                         opts);
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

TEST_CASE("fiml_ugamma_spectrum: complete data matches the unstructured robust spectrum") {
  // The first-principles FIML UΓ spectrum (V = saturated info, Γ = H⁻¹JH⁻¹) must,
  // on complete data, reproduce the complete-data UNSTRUCTURED robust spectrum
  // eig(U·Γ̂) with U built from the sample-moment (unstructured) weight — the
  // h1.information = "unstructured" convention that is natural for FIML's EM
  // saturated model. (Validated against lavaan's unstructured UGamma in the R
  // example; this is the lavaan-free C++ guard.)
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4 + x5");
  const Eigen::Index nfree = static_cast<Eigen::Index>(built.ev.n_free());
  Eigen::VectorXd theta0(nfree);
  theta0.setConstant(0.6);
  auto truth = built.ev.sigma(theta0);
  REQUIRE(truth.has_value());
  Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[0]);
  REQUIRE(llt.info() == Eigen::Success);
  const Eigen::MatrixXd L = llt.matrixL();
  const Eigen::Index p = truth->sigma[0].rows();
  Eigen::MatrixXd Z = deterministic_z(220, p);
  magmaan::data::RawData raw;
  raw.X.push_back((Z * L.transpose()).rowwise() + truth->mu[0].transpose());

  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 500;
  auto est = magmaan::estimate::fit_fiml(*built.pt, *built.rep, raw, theta0,
                                         magmaan::estimate::fiml::FIML{},
                                         magmaan::estimate::Backend::NloptLbfgs,
                                         opts);
  REQUIRE(est.has_value());
  auto fx = magmaan::estimate::fiml::fiml_extras(*built.pt, *built.rep, raw, *est);
  REQUIRE(fx.has_value());

  // df = saturated moments (means + vech) − free params, single group.
  const int df = static_cast<int>(p + p * (p + 1) / 2 - nfree);
  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, *est, df, fx->chi2);
  REQUIRE_MESSAGE(sp.has_value(),
      "fiml_ugamma_spectrum failed: " << (sp.has_value() ? "" : sp.error().detail));

  // Contract.
  CHECK(sp->df == df);
  CHECK(sp->eigvals.size() == df);
  CHECK(sp->eigvals.allFinite());
  CHECK(sp->eigvals.minCoeff() > 0.0);
  CHECK(sp->chi2_lrt == doctest::Approx(fx->chi2));
  CHECK(sp->trace_xcheck == doctest::Approx(sp->eigvals.sum()));

  // Reference: complete-data unstructured robust spectrum eig(U·Γ̂).
  auto uf = magmaan::robust::build_u_factor(
      *built.pt, *built.rep, *samp, *est,
      magmaan::robust::InferenceSpec{
          magmaan::robust::Information::Expected,
          magmaan::robust::WeightMoments::Unstructured,
          magmaan::robust::ScoreCovariance::Empirical});
  REQUIRE(uf.has_value());
  auto Zc = magmaan::robust::casewise_contributions(raw, *samp, /*include_means=*/true);
  REQUIRE(Zc.has_value());
  auto M = magmaan::robust::reduced_gamma_sample(
      *uf, *Zc, static_cast<double>(raw.X[0].rows()));
  REQUIRE(M.has_value());
  auto ev_ref = magmaan::robust::ugamma_eigenvalues(*M);
  REQUIRE(ev_ref.has_value());
  REQUIRE(static_cast<int>(uf->df) == df);

  Eigen::VectorXd a = sp->eigvals;          // ascending
  Eigen::VectorXd b = ev_ref->tail(df);     // top-df, ascending
  CHECK((a - b).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("fiml_ugamma_spectrum: missing-data trace matches fiml_robust_mlr") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  auto raw = model_missing_raw(built, theta0, {160});

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 600;
  auto est = magmaan::estimate::fit_fiml(
      *built.pt, *built.rep, raw, theta0, magmaan::estimate::fiml::FIML{},
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(est.has_value(),
      "fit_fiml failed: " << (est.has_value() ? "" : est.error().detail));

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df_or = magmaan::inference::df_stat(*built.pt, *samp, est->theta);
  REQUIRE(df_or.has_value());
  auto fx = magmaan::estimate::fiml::fiml_extras(
      *built.pt, *built.rep, raw, *est);
  REQUIRE(fx.has_value());

  constexpr double h_step = 1e-4;
  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, *est, *df_or, fx->chi2,
      magmaan::estimate::fiml::FIML{}, h_step);
  REQUIRE_MESSAGE(sp.has_value(),
      "fiml_ugamma_spectrum failed: " <<
      (sp.has_value() ? "" : sp.error().detail));
  auto rob = magmaan::estimate::fiml::fiml_robust_mlr(
      *built.pt, *built.rep, raw, *est, *df_or, fx->chi2,
      magmaan::estimate::fiml::FIML{}, h_step);
  REQUIRE_MESSAGE(rob.has_value(),
      "fiml_robust_mlr failed: " <<
      (rob.has_value() ? "" : rob.error().detail));

  CHECK(sp->df == *df_or);
  CHECK(sp->eigvals.size() == *df_or);
  CHECK(sp->eigvals.allFinite());
  CHECK(sp->eigvals.minCoeff() > 0.0);
  INFO("trace_xcheck = ", sp->trace_xcheck,
       " robust trace = ", rob->trace_ugamma);
  const double trace_scale =
      std::max(1.0, std::abs(rob->trace_ugamma));
  CHECK(std::abs(sp->trace_xcheck - rob->trace_ugamma) / trace_scale < 1e-2);
}

TEST_CASE("fiml_ugamma_spectrum: multi-group missing-data spectrum is finite") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4", /*n_groups=*/2);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.58);
  auto raw = model_missing_raw(built, theta0, {90, 84});

  magmaan::estimate::Estimates est;
  est.theta = theta0;
  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df_or = magmaan::inference::df_stat(*built.pt, *samp, est.theta);
  REQUIRE(df_or.has_value());
  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, est, *df_or, /*chi2_lrt=*/7.0);
  REQUIRE_MESSAGE(sp.has_value(),
      "fiml_ugamma_spectrum failed: " <<
      (sp.has_value() ? "" : sp.error().detail));

  CHECK(sp->df == *df_or);
  CHECK(sp->eigvals.size() == *df_or);
  CHECK(sp->eigvals.allFinite());
  CHECK(sp->trace_xcheck == doctest::Approx(sp->eigvals.sum()));
  CHECK(sp->eigvals.minCoeff() > 0.0);
}

TEST_CASE("fiml_ugamma_spectrum: rejects nonlinear equality constraints explicitly") {
  auto built = build_mean_model("f =~ x1 + a*x2 + b*x3\na == b^2");
  magmaan::data::RawData raw;
  raw.X.push_back(deterministic_z(40, 3));
  magmaan::estimate::Estimates est;
  est.theta = Eigen::VectorXd::Constant(
      static_cast<Eigen::Index>(built.ev.n_free()), 0.6);

  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, est, /*df=*/1, /*chi2_lrt=*/1.0);
  REQUIRE_FALSE(sp.has_value());
  CHECK(sp.error().detail.find("nonlinear equality constraints") !=
        std::string::npos);
  CHECK(sp.error().detail.find("tangent-space") != std::string::npos);
}

TEST_CASE("fiml_ugamma_spectrum: rejects bad arguments") {
  auto built = build_mean_model("f =~ x1 + x2 + x3");
  magmaan::data::RawData raw;
  raw.X.push_back(deterministic_z(40, 3));
  magmaan::estimate::Estimates est;
  est.theta = Eigen::VectorXd::Constant(
      static_cast<Eigen::Index>(built.ev.n_free()), 0.5);
  auto bad_df = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, est, /*df=*/0, /*chi2_lrt=*/1.0);
  CHECK_FALSE(bad_df.has_value());
}

TEST_CASE("nested FIML restriction map: NT gamma gives all-one eigenvalues") {
  auto h1 = build_mean_model("f =~ x1 + x2 + x3 + x4");
  auto h0 = build_mean_model("f =~ x1 + a*x2 + a*x3 + x4");
  Eigen::VectorXd theta1(static_cast<Eigen::Index>(h1.ev.n_free()));
  theta1.setConstant(0.6);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(h0.ev.n_free()));
  theta0.setConstant(0.6);
  auto raw = model_missing_raw(h1, theta1, {140});

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df1 = magmaan::inference::df_stat(*h1.pt, *samp, theta1);
  auto df0 = magmaan::inference::df_stat(*h0.pt, *samp, theta0);
  REQUIRE(df1.has_value());
  REQUIRE(df0.has_value());
  REQUIRE(*df0 - *df1 == 1);
  auto K1 = magmaan::estimate::build_eq_constraints(*h1.pt);
  auto K0 = magmaan::estimate::build_eq_constraints(*h0.pt);
  REQUIRE(K1.has_value());
  REQUIRE(K0.has_value());

  auto r = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/5.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::NT,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(r.has_value(),
      "nested FIML NT failed: " << (r.has_value() ? "" : r.error().detail));
  REQUIRE(r->eigenvalues.size() == 1);
  CHECK(r->eigenvalues(0) == doctest::Approx(1.0).epsilon(1e-10));
  CHECK(r->scale_c == doctest::Approx(1.0).epsilon(1e-10));
  CHECK(r->T_scaled == doctest::Approx(3.0).epsilon(1e-10));
}

TEST_CASE("nested FIML restriction map: reduced and dense eta-space eigensolves agree") {
  auto h1 = build_mean_model("f =~ x1 + x2 + x3 + x4");
  auto h0 = build_mean_model("f =~ x1 + a*x2 + a*x3 + x4");
  Eigen::VectorXd theta1(static_cast<Eigen::Index>(h1.ev.n_free()));
  theta1.setConstant(0.6);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(h0.ev.n_free()));
  theta0.setConstant(0.6);
  auto raw = model_missing_raw(h1, theta1, {150});

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df1 = magmaan::inference::df_stat(*h1.pt, *samp, theta1);
  auto df0 = magmaan::inference::df_stat(*h0.pt, *samp, theta0);
  REQUIRE(df1.has_value());
  REQUIRE(df0.has_value());
  auto K1 = magmaan::estimate::build_eq_constraints(*h1.pt);
  auto K0 = magmaan::estimate::build_eq_constraints(*h0.pt);
  REQUIRE(K1.has_value());
  REQUIRE(K0.has_value());

  auto exact = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/5.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  auto delta = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/5.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Delta);
  REQUIRE_MESSAGE(exact.has_value(),
      "nested FIML exact failed: " << (exact.has_value() ? "" : exact.error().detail));
  REQUIRE_MESSAGE(delta.has_value(),
      "nested FIML delta failed: " << (delta.has_value() ? "" : delta.error().detail));
  REQUIRE(exact->eigenvalues.size() == 1);
  REQUIRE(delta->eigenvalues.size() == 1);
  CHECK(exact->eigenvalues(0) == doctest::Approx(delta->eigenvalues(0)).epsilon(1e-7));

  auto sm = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE(sm.has_value());
  magmaan::estimate::Estimates est1;
  est1.theta = theta1;
  auto D1 = magmaan::estimate::fiml::fiml_eta_jacobian(
      *h1.pt, *h1.rep, raw, est1);
  REQUIRE(D1.has_value());
  const Eigen::MatrixXd Delta1 = D1->Delta_theta * K1->Kmat;
  auto restr = magmaan::robust::restriction_alpha_from_K(*K1, *K0);
  REQUIRE(restr.has_value());
  auto reduced = magmaan::robust::compute_fiml_satorra2000(
      Delta1, sm->H, sm->acov, restr->A);
  REQUIRE(reduced.has_value());

  const Eigen::MatrixXd V = 0.5 * (sm->H + sm->H.transpose());
  const Eigen::MatrixXd G = 0.5 * (sm->acov + sm->acov.transpose());
  const Eigen::MatrixXd VD = V * Delta1;
  Eigen::MatrixXd P = Delta1.transpose() * VD;
  P = 0.5 * (P + P.transpose()).eval();
  Eigen::LDLT<Eigen::MatrixXd> ldlt_P(P);
  REQUIRE(ldlt_P.info() == Eigen::Success);
  const Eigen::MatrixXd Pinv =
      ldlt_P.solve(Eigen::MatrixXd::Identity(P.rows(), P.cols()));
  Eigen::MatrixXd C = restr->A * Pinv * restr->A.transpose();
  C = 0.5 * (C + C.transpose()).eval();
  Eigen::LDLT<Eigen::MatrixXd> ldlt_C(C);
  REQUIRE(ldlt_C.info() == Eigen::Success);
  const Eigen::MatrixXd Cinv =
      ldlt_C.solve(Eigen::MatrixXd::Identity(C.rows(), C.cols()));
  Eigen::MatrixXd U = VD * Pinv * restr->A.transpose() * Cinv *
                      restr->A * Pinv * VD.transpose();
  U = 0.5 * (U + U.transpose()).eval();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_g(G);
  REQUIRE(es_g.info() == Eigen::Success);
  const Eigen::VectorXd d =
      es_g.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  const Eigen::MatrixXd Gsqrt =
      es_g.eigenvectors() * d.asDiagonal() * es_g.eigenvectors().transpose();
  Eigen::MatrixXd dense_op = Gsqrt * U * Gsqrt;
  dense_op = 0.5 * (dense_op + dense_op.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_dense(dense_op);
  REQUIRE(es_dense.info() == Eigen::Success);
  const Eigen::VectorXd ev_dense =
      es_dense.eigenvalues().tail(reduced->eigenvalues.size());
  CHECK(ev_dense(0) == doctest::Approx(reduced->eigenvalues(0)).epsilon(1e-7));
}

TEST_CASE("nested FIML restriction map: rejects df mismatch and reversed nesting") {
  auto h1 = build_mean_model("f =~ x1 + x2 + x3 + x4");
  auto h0 = build_mean_model("f =~ x1 + a*x2 + a*x3 + x4");
  Eigen::VectorXd theta1(static_cast<Eigen::Index>(h1.ev.n_free()));
  theta1.setConstant(0.6);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(h0.ev.n_free()));
  theta0.setConstant(0.6);
  auto raw = model_missing_raw(h1, theta1, {120});
  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df1 = magmaan::inference::df_stat(*h1.pt, *samp, theta1);
  auto df0 = magmaan::inference::df_stat(*h0.pt, *samp, theta0);
  REQUIRE(df1.has_value());
  REQUIRE(df0.has_value());
  auto K1 = magmaan::estimate::build_eq_constraints(*h1.pt);
  auto K0 = magmaan::estimate::build_eq_constraints(*h0.pt);
  REQUIRE(K1.has_value());
  REQUIRE(K0.has_value());

  auto mismatch = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/5.0, /*T_H1=*/2.0, *df0 + 1, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_FALSE(mismatch.has_value());
  CHECK(mismatch.error().detail.find("df_diff mismatch") != std::string::npos);

  auto reversed = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h0.pt, *h0.rep, theta0, *K0,
      *h1.pt, *h1.rep, theta1, *K1,
      raw, /*T_H0=*/2.0, /*T_H1=*/5.0, *df1, *df0,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_FALSE(reversed.has_value());
  CHECK(reversed.error().detail.find("df_H0 - df_H1 is negative") !=
        std::string::npos);
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
