#include <doctest/doctest.h>
#include "../test_fit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "magmaan/data/raw_data.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/inference/inference.hpp"
#include "magmaan/inference/score.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/measures/fit_measures.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/robust/weighted_inference.hpp"
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

bool same_or_both_nan(double a, double b) {
  return (std::isnan(a) && std::isnan(b)) || a == b;
}

void check_same_score_table(const magmaan::inference::ScoreTestTable& a,
                            const magmaan::inference::ScoreTestTable& b) {
  REQUIRE(a.rows.size() == b.rows.size());
  for (std::size_t i = 0; i < a.rows.size(); ++i) {
    const auto& x = a.rows[i];
    const auto& y = b.rows[i];
    CHECK(x.candidate.kind == y.candidate.kind);
    CHECK(x.candidate.row == y.candidate.row);
    CHECK(x.candidate.op == y.candidate.op);
    CHECK(x.candidate.lhs_var == y.candidate.lhs_var);
    CHECK(x.candidate.rhs_var == y.candidate.rhs_var);
    CHECK(x.candidate.group == y.candidate.group);
    CHECK(same_or_both_nan(x.score, y.score));
    CHECK(same_or_both_nan(x.information, y.information));
    CHECK(same_or_both_nan(x.mi, y.mi));
    CHECK(x.df == y.df);
    CHECK(same_or_both_nan(x.p_value, y.p_value));
    CHECK(same_or_both_nan(x.epc, y.epc));
    CHECK(same_or_both_nan(x.epc_lv, y.epc_lv));
    CHECK(same_or_both_nan(x.epc_all, y.epc_all));
    CHECK(same_or_both_nan(x.v_eff, y.v_eff));
    CHECK(same_or_both_nan(x.mi_scaled, y.mi_scaled));
    CHECK(same_or_both_nan(x.scaling_factor, y.scaling_factor));
  }
}

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

magmaan::data::RawData duplicate_raw_rows(const magmaan::data::RawData& raw,
                                          int times) {
  REQUIRE(times > 0);
  magmaan::data::RawData out;
  out.X.reserve(raw.X.size());
  out.mask.reserve(raw.mask.size());
  for (std::size_t b = 0; b < raw.X.size(); ++b) {
    const Eigen::MatrixXd& X = raw.X[b];
    Eigen::MatrixXd Xdup(X.rows() * times, X.cols());
    for (int t = 0; t < times; ++t) {
      Xdup.block(t * X.rows(), 0, X.rows(), X.cols()) = X;
    }
    out.X.push_back(std::move(Xdup));

    if (!raw.mask.empty()) {
      const auto& M = raw.mask[b];
      Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> Mdup(
          M.rows() * times, M.cols());
      for (int t = 0; t < times; ++t) {
        Mdup.block(t * M.rows(), 0, M.rows(), M.cols()) = M;
      }
      out.mask.push_back(std::move(Mdup));
    }
  }
  return out;
}

Eigen::MatrixXd symmetrized(const Eigen::MatrixXd& A) {
  return 0.5 * (A + A.transpose());
}

Eigen::MatrixXd inverse_ldlt(const Eigen::MatrixXd& A) {
  Eigen::LDLT<Eigen::MatrixXd> ldlt(symmetrized(A));
  REQUIRE(ldlt.info() == Eigen::Success);
  return ldlt.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
}

struct FimlProjectionPieces {
  Eigen::MatrixXd H;
  Eigen::MatrixXd J;
  Eigen::MatrixXd Gamma;
  Eigen::MatrixXd Delta;
  Eigen::MatrixXd P_inv;
  Eigen::MatrixXd U;
};

FimlProjectionPieces fiml_projection_pieces(
    const BuiltModel& built,
    const magmaan::data::RawData& raw,
    const magmaan::estimate::Estimates& est) {
  auto sm = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE_MESSAGE(sm.has_value(),
      "saturated_em_moments failed: " << (sm.has_value() ? "" : sm.error().detail));
  auto D = magmaan::estimate::fiml::fiml_eta_jacobian(
      *built.pt, *built.rep, raw, est);
  REQUIRE_MESSAGE(D.has_value(),
      "fiml_eta_jacobian failed: " << (D.has_value() ? "" : D.error().detail));
  auto con = magmaan::estimate::build_eq_constraints(*built.pt);
  REQUIRE_MESSAGE(con.has_value(),
      "build_eq_constraints failed: " << (con.has_value() ? "" : con.error().detail));

  Eigen::MatrixXd Delta = D->Delta_theta;
  if (con->active()) Delta = (Delta * con->K()).eval();

  FimlProjectionPieces out;
  out.H = symmetrized(sm->H);
  out.J = symmetrized(sm->J);
  out.Gamma = symmetrized(sm->acov);
  out.Delta = std::move(Delta);

  const Eigen::MatrixXd HD = out.H * out.Delta;
  const Eigen::MatrixXd P = symmetrized(out.Delta.transpose() * HD);
  out.P_inv = inverse_ldlt(P);
  out.U = symmetrized(out.H - HD * out.P_inv * HD.transpose());
  return out;
}

Eigen::VectorXd projected_ugamma_eigenvalues(const Eigen::MatrixXd& U,
                                             const Eigen::MatrixXd& Gamma,
                                             int df) {
  const Eigen::MatrixXd Gs = symmetrized(Gamma);
  Eigen::MatrixXd reduced;
  Eigen::LLT<Eigen::MatrixXd> llt(Gs);
  if (llt.info() == Eigen::Success) {
    const Eigen::MatrixXd R = llt.matrixL();
    reduced = R.transpose() * U * R;
  } else {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_g(Gs);
    REQUIRE(es_g.info() == Eigen::Success);
    const Eigen::VectorXd d =
        es_g.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    const Eigen::MatrixXd sq =
        es_g.eigenvectors() * d.asDiagonal() * es_g.eigenvectors().transpose();
    reduced = sq * U * sq;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetrized(reduced),
                                                    Eigen::EigenvaluesOnly);
  REQUIRE(es.info() == Eigen::Success);
  REQUIRE(static_cast<Eigen::Index>(df) <= es.eigenvalues().size());
  return es.eigenvalues().tail(df);
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

namespace {

// Analytic-vs-FD comparator for the observed FIML information. Evaluating at
// an arbitrary (non-optimal) θ exercises the second-order chain-rule term
// hardest: away from the optimum the pattern-aggregated moment gradient that
// contracts with ∂²Σ/∂θ_a∂θ_b and ∂²μ/∂θ_a∂θ_b is far from zero.
void check_fiml_observed_information_analytic_vs_fd(
    const BuiltModel& built,
    const magmaan::data::RawData& raw,
    const Eigen::VectorXd& theta) {
  magmaan::estimate::Estimates est;
  est.theta = theta;

  auto an = magmaan::estimate::fiml::fiml_observed_information(
      *built.pt, *built.rep, raw, est);
  REQUIRE_MESSAGE(an.has_value(),
      "analytic observed information failed: "
          << (an.has_value() ? "" : an.error().detail));
  auto fd = magmaan::estimate::fiml::diagnostic::fiml_observed_information_fd(
      *built.pt, *built.rep, raw, est);
  REQUIRE_MESSAGE(fd.has_value(),
      "FD observed information failed: "
          << (fd.has_value() ? "" : fd.error().detail));

  CHECK((*an - an->transpose()).cwiseAbs().maxCoeff() < 1e-9);
  const double scale = std::max(1.0, fd->cwiseAbs().maxCoeff());
  INFO("max rel diff = ", (*an - *fd).cwiseAbs().maxCoeff() / scale);
  CHECK((*an - *fd).cwiseAbs().maxCoeff() / scale < 1e-6);
}

Eigen::VectorXd perturbed_theta(const Eigen::VectorXd& theta0) {
  Eigen::VectorXd theta = theta0;
  for (Eigen::Index k = 0; k < theta.size(); ++k) {
    theta(k) += 0.04 * std::sin(1.0 + static_cast<double>(k));
  }
  return theta;
}

}  // namespace

TEST_CASE("fiml_observed_information: analytic matches FD for a mean CFA") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  const auto raw = model_missing_raw(built, theta0, {120});
  check_fiml_observed_information_analytic_vs_fd(built, raw,
                                                 perturbed_theta(theta0));
}

TEST_CASE("fiml_observed_information: analytic matches FD with structural "
          "latent means") {
  // y ~ f exercises the Β second-derivative paths (Λ-Β, Ψ-Β, Β-Β), and the
  // free latent mean f ~ 1 turns on the mean-map curvature (Λ-α, α-Β, Β-Β).
  auto built = build_mean_model("f =~ x1 + x2 + x3\ny ~ f\nf ~ 1");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  const auto raw = model_missing_raw(built, theta0, {130});
  check_fiml_observed_information_analytic_vs_fd(built, raw,
                                                 perturbed_theta(theta0));
}

TEST_CASE("fiml_observed_information: analytic matches FD multi-group") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4", 2);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  const auto raw = model_missing_raw(built, theta0, {70, 90});
  check_fiml_observed_information_analytic_vs_fd(built, raw,
                                                 perturbed_theta(theta0));
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
  CHECK(sp->h1_information ==
        magmaan::estimate::fiml::FIMLH1Information::Saturated);
  INFO("trace_xcheck = ", sp->trace_xcheck,
       " robust trace = ", rob->trace_ugamma);
  const double trace_scale =
      std::max(1.0, std::abs(rob->trace_ugamma));
  CHECK(std::abs(sp->trace_xcheck - rob->trace_ugamma) / trace_scale < 1e-2);

  auto sp_struct = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, *est, *df_or, fx->chi2,
      magmaan::estimate::fiml::FIML{}, h_step,
      magmaan::estimate::fiml::FIMLH1Information::Structured);
  REQUIRE_MESSAGE(sp_struct.has_value(),
      "structured fiml_ugamma_spectrum failed: " <<
      (sp_struct.has_value() ? "" : sp_struct.error().detail));
  CHECK(sp_struct->h1_information ==
        magmaan::estimate::fiml::FIMLH1Information::Structured);
  CHECK(sp_struct->df == *df_or);
  CHECK(sp_struct->chi2_lrt == doctest::Approx(fx->chi2));
  CHECK(sp_struct->eigvals.size() == *df_or);
  CHECK(sp_struct->eigvals.allFinite());
  CHECK(sp_struct->eigvals.minCoeff() > -1e-8);
  CHECK(sp_struct->trace_xcheck ==
        doctest::Approx(sp_struct->eigvals.sum()).epsilon(1e-12));
}

TEST_CASE("fiml_ugamma_spectrum: saturated-space trace identity is algebraic") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  auto raw = model_missing_raw(built, theta0, {180});

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 700;
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
  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, *est, *df_or, fx->chi2);
  REQUIRE_MESSAGE(sp.has_value(),
      "fiml_ugamma_spectrum failed: " <<
      (sp.has_value() ? "" : sp.error().detail));

  const auto pieces = fiml_projection_pieces(built, raw, *est);
  const double trace_h1 = (pieces.H * pieces.Gamma).trace();
  const double trace_h0 =
      (pieces.P_inv *
       pieces.Delta.transpose() * pieces.J * pieces.Delta).trace();
  const double trace_from_split = trace_h1 - trace_h0;
  const double trace_direct = (pieces.U * pieces.Gamma).trace();

  INFO("trace_h1 = ", trace_h1,
       " trace_h0 = ", trace_h0,
       " trace_from_split = ", trace_from_split,
       " trace_direct = ", trace_direct,
       " spectrum trace = ", sp->trace_xcheck);
  CHECK(trace_direct == doctest::Approx(trace_from_split).epsilon(1e-9));
  CHECK(sp->trace_xcheck == doctest::Approx(trace_from_split).epsilon(1e-9));
}

TEST_CASE("fiml_ugamma_spectrum: NT saturated gamma gives all-one eigenvalues") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  auto raw = model_missing_raw(built, theta0, {170});

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 700;
  auto est = magmaan::estimate::fit_fiml(
      *built.pt, *built.rep, raw, theta0, magmaan::estimate::fiml::FIML{},
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(est.has_value(),
      "fit_fiml failed: " << (est.has_value() ? "" : est.error().detail));
  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df_or = magmaan::inference::df_stat(*built.pt, *samp, est->theta);
  REQUIRE(df_or.has_value());

  const auto pieces = fiml_projection_pieces(built, raw, *est);
  const Eigen::MatrixXd gamma_nt = inverse_ldlt(pieces.H);
  const Eigen::VectorXd eig =
      projected_ugamma_eigenvalues(pieces.U, gamma_nt, *df_or);
  REQUIRE(eig.size() == *df_or);
  INFO("NT eigenvalues = ", eig.transpose());
  CHECK((eig.array() - 1.0).abs().maxCoeff() < 1e-8);
}

TEST_CASE("fiml_ugamma_spectrum: row duplication leaves eigenvalues invariant") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  auto raw = model_missing_raw(built, theta0, {150});

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 700;
  auto est = magmaan::estimate::fit_fiml(
      *built.pt, *built.rep, raw, theta0, magmaan::estimate::fiml::FIML{},
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE_MESSAGE(est.has_value(),
      "fit_fiml failed: " << (est.has_value() ? "" : est.error().detail));

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df_or = magmaan::inference::df_stat(*built.pt, *samp, est->theta);
  REQUIRE(df_or.has_value());

  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, *est, *df_or, /*chi2_lrt=*/3.0);
  REQUIRE_MESSAGE(sp.has_value(),
      "fiml_ugamma_spectrum failed: " <<
      (sp.has_value() ? "" : sp.error().detail));

  const auto raw_dup = duplicate_raw_rows(raw, 3);
  auto sp_dup = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw_dup, *est, *df_or, /*chi2_lrt=*/3.0);
  REQUIRE_MESSAGE(sp_dup.has_value(),
      "duplicated fiml_ugamma_spectrum failed: " <<
      (sp_dup.has_value() ? "" : sp_dup.error().detail));

  REQUIRE(sp->eigvals.size() == sp_dup->eigvals.size());
  INFO("original eigenvalues = ", sp->eigvals.transpose(),
       " duplicated eigenvalues = ", sp_dup->eigvals.transpose());
  CHECK((sp->eigvals - sp_dup->eigvals).cwiseAbs().maxCoeff() < 1e-9);
  CHECK(sp_dup->trace_xcheck == doctest::Approx(sp->trace_xcheck).epsilon(1e-9));
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

TEST_CASE("fiml_ugamma_spectrum: nonlinear equality constraints use tangent space") {
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

  magmaan::data::RawData raw;
  raw.X.push_back((deterministic_z(34, 3) * L.transpose()).rowwise() +
                  truth->mu[0].transpose());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 350;
  auto est = magmaan::estimate::fit_fiml(
      *built.pt, *built.rep, raw, theta0, magmaan::estimate::fiml::FIML{},
      magmaan::estimate::Backend::NloptSlsqp, opts);
  REQUIRE_MESSAGE(est.has_value(), "SLSQP constrained FIML failed: "
      << (est.has_value() ? std::string{} : est.error().detail));

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df_or = magmaan::inference::df_stat(*built.pt, *samp, est->theta);
  REQUIRE(df_or.has_value());
  REQUIRE(*df_or == 1);

  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, *est, *df_or, /*chi2_lrt=*/1.0);
  REQUIRE_MESSAGE(sp.has_value(),
      "fiml_ugamma_spectrum failed: " <<
      (sp.has_value() ? "" : sp.error().detail));
  CHECK(sp->df == *df_or);
  CHECK(sp->eigvals.size() == *df_or);
  CHECK(sp->eigvals.allFinite());
  CHECK(sp->eigvals.minCoeff() > 0.0);
  CHECK(sp->trace_xcheck == doctest::Approx(sp->eigvals.sum()));
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
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto sat_h1 = magmaan::estimate::fiml::fiml_h1_moments(raw, *pack);
  REQUIRE(sat_h1.has_value());
  auto r_pack = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/5.0, /*T_H1=*/2.0, *df0, *df1,
      *pack, *sat_h1,
      magmaan::robust::GammaSource::NT,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(r_pack.has_value(),
      "nested FIML NT pack failed: " <<
          (r_pack.has_value() ? "" : r_pack.error().detail));
  CHECK(r->T_scaled == r_pack->T_scaled);
  CHECK(r->scale_c == r_pack->scale_c);
  CHECK(r->eigenvalues == r_pack->eigenvalues);
  REQUIRE(r->eigenvalues.size() == 1);
  CHECK(r->eigenvalues(0) == doctest::Approx(1.0).epsilon(1e-10));
  CHECK(r->scale_c == doctest::Approx(1.0).epsilon(1e-10));
  CHECK(r->T_scaled == doctest::Approx(3.0).epsilon(1e-10));
}

TEST_CASE("nested FIML restriction map: precomputed SaturatedMoments is "
          "bit-identical to the rebuild (2000 and 2001)") {
  auto h1 = build_mean_model("f =~ x1 + x2 + x3 + x4");
  auto h0 = build_mean_model("f =~ x1 + a*x2 + a*x3 + x4");
  Eigen::VectorXd theta1(static_cast<Eigen::Index>(h1.ev.n_free()));
  theta1.setConstant(0.6);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(h0.ev.n_free()));
  theta0.setConstant(0.6);
  auto raw = model_missing_raw(h1, theta1, {160});

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

  // The saturated moments depend only on the data, so the caller can build
  // them once and hand them in; the result must match the from-scratch rebuild
  // to the bit (the EM is deterministic).
  auto sm = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE(sm.has_value());

  // method 2000 (restriction map), empirical Gamma + exact A.
  auto r_rebuild = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1, *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/5.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact, /*h_step=*/1e-4, nullptr);
  auto r_reuse = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1, *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/5.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact, /*h_step=*/1e-4, &*sm);
  REQUIRE_MESSAGE(r_rebuild.has_value(),
      "2000 rebuild failed: " << (r_rebuild.has_value() ? "" : r_rebuild.error().detail));
  REQUIRE_MESSAGE(r_reuse.has_value(),
      "2000 reuse failed: " << (r_reuse.has_value() ? "" : r_reuse.error().detail));
  CHECK(r_reuse->T_scaled == r_rebuild->T_scaled);
  CHECK(r_reuse->scale_c == r_rebuild->scale_c);
  CHECK(r_reuse->eigenvalues == r_rebuild->eigenvalues);

  // method 2001 (difference spectrum).
  auto d_rebuild = magmaan::robust::lr_test_satorra2001_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *h0.pt, *h0.rep, theta0,
      raw, /*T_H0=*/5.0, /*T_H1=*/2.0, *df0, *df1, /*h_step=*/1e-4, nullptr);
  auto d_reuse = magmaan::robust::lr_test_satorra2001_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *h0.pt, *h0.rep, theta0,
      raw, /*T_H0=*/5.0, /*T_H1=*/2.0, *df0, *df1, /*h_step=*/1e-4, &*sm);
  REQUIRE_MESSAGE(d_rebuild.has_value(),
      "2001 rebuild failed: " << (d_rebuild.has_value() ? "" : d_rebuild.error().detail));
  REQUIRE_MESSAGE(d_reuse.has_value(),
      "2001 reuse failed: " << (d_reuse.has_value() ? "" : d_reuse.error().detail));
  CHECK(d_reuse->T_scaled == d_rebuild->T_scaled);
  CHECK(d_reuse->scale_c == d_rebuild->scale_c);
  CHECK(d_reuse->eigenvalues == d_rebuild->eigenvalues);
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

TEST_CASE("nested FIML restriction map: trace_CinvS matches C/S algebra") {
  auto h1 = build_mean_model("f =~ x1 + x2 + x3 + x4");
  auto h0 = build_mean_model("f =~ x1 + a*x2 + a*x3 + a*x4");
  Eigen::VectorXd theta1(static_cast<Eigen::Index>(h1.ev.n_free()));
  theta1.setConstant(0.6);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(h0.ev.n_free()));
  theta0.setConstant(0.6);
  auto raw = model_missing_raw(h1, theta1, {160});

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df1 = magmaan::inference::df_stat(*h1.pt, *samp, theta1);
  auto df0 = magmaan::inference::df_stat(*h0.pt, *samp, theta0);
  REQUIRE(df1.has_value());
  REQUIRE(df0.has_value());
  REQUIRE(*df0 - *df1 == 2);
  auto K1 = magmaan::estimate::build_eq_constraints(*h1.pt);
  auto K0 = magmaan::estimate::build_eq_constraints(*h0.pt);
  REQUIRE(K1.has_value());
  REQUIRE(K0.has_value());

  auto exact = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/7.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(exact.has_value(),
      "nested FIML exact failed: " << (exact.has_value() ? "" : exact.error().detail));
  REQUIRE(exact->eigenvalues.size() == 2);

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
  REQUIRE_MESSAGE(reduced.has_value(),
      "compute_fiml_satorra2000 failed: " <<
      (reduced.has_value() ? "" : reduced.error().detail));
  REQUIRE(reduced->eigenvalues.size() == 2);

  const Eigen::MatrixXd Cinv = inverse_ldlt(reduced->C);
  const double trace_direct = (Cinv * reduced->S).trace();
  INFO("trace_direct = ", trace_direct,
       " trace_CinvS = ", reduced->trace_CinvS,
       " eigenvalues = ", reduced->eigenvalues.transpose());
  CHECK(trace_direct == doctest::Approx(reduced->trace_CinvS).epsilon(1e-9));
  CHECK(reduced->trace_CinvS ==
        doctest::Approx(reduced->eigenvalues.sum()).epsilon(1e-12));
  CHECK(reduced->trace_CinvS_sq ==
        doctest::Approx(reduced->eigenvalues.squaredNorm()).epsilon(1e-12));
  CHECK((exact->eigenvalues - reduced->eigenvalues).cwiseAbs().maxCoeff() < 1e-9);
  CHECK(exact->scale_c ==
        doctest::Approx(reduced->trace_CinvS / 2.0).epsilon(1e-12));
}

TEST_CASE("nested FIML restriction map: row duplication leaves eigenvalues invariant") {
  auto h1 = build_mean_model("f =~ x1 + x2 + x3 + x4");
  auto h0 = build_mean_model("f =~ x1 + a*x2 + a*x3 + a*x4");
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
  REQUIRE(*df0 - *df1 == 2);
  auto K1 = magmaan::estimate::build_eq_constraints(*h1.pt);
  auto K0 = magmaan::estimate::build_eq_constraints(*h0.pt);
  REQUIRE(K1.has_value());
  REQUIRE(K0.has_value());

  auto exact = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/7.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(exact.has_value(),
      "nested FIML exact failed: " << (exact.has_value() ? "" : exact.error().detail));

  const auto raw_dup = duplicate_raw_rows(raw, 3);
  auto exact_dup = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw_dup, /*T_H0=*/7.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(exact_dup.has_value(),
      "duplicated nested FIML exact failed: " <<
      (exact_dup.has_value() ? "" : exact_dup.error().detail));

  REQUIRE(exact->eigenvalues.size() == exact_dup->eigenvalues.size());
  INFO("original eigenvalues = ", exact->eigenvalues.transpose(),
       " duplicated eigenvalues = ", exact_dup->eigenvalues.transpose());
  CHECK((exact->eigenvalues - exact_dup->eigenvalues).cwiseAbs().maxCoeff() < 1e-9);
  CHECK(exact_dup->scale_c == doctest::Approx(exact->scale_c).epsilon(1e-9));
  CHECK(exact_dup->adjust_d0 == doctest::Approx(exact->adjust_d0).epsilon(1e-9));
}

TEST_CASE("nested FIML restriction map: nonlinear equality uses local tangent") {
  auto h1 = build_mean_model("f =~ x1 + a*x2 + b*x3");
  auto h0 = build_mean_model("f =~ x1 + a*x2 + b*x3\na == b^2");
  Eigen::VectorXd theta(static_cast<Eigen::Index>(h1.ev.n_free()));
  theta.setConstant(0.7);
  theta(0) = 0.49;
  theta(1) = 0.7;
  theta.tail(3) << 1.0, 2.0, 3.0;
  auto raw = model_missing_raw(h0, theta, {150});

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df1 = magmaan::inference::df_stat(*h1.pt, *samp, theta);
  auto df0 = magmaan::inference::df_stat(*h0.pt, *samp, theta);
  REQUIRE(df1.has_value());
  REQUIRE(df0.has_value());
  REQUIRE(*df0 - *df1 == 1);
  auto K1 = magmaan::estimate::build_eq_constraints(
      *h1.pt, /*allow_nonlinear=*/true);
  auto K0 = magmaan::estimate::build_eq_constraints(
      *h0.pt, /*allow_nonlinear=*/true);
  REQUIRE(K1.has_value());
  REQUIRE(K0.has_value());

  auto exact = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta, *K1,
      *h0.pt, *h0.rep, theta, *K0,
      raw, /*T_H0=*/6.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(exact.has_value(),
      "nested FIML nonlinear exact failed: " <<
      (exact.has_value() ? "" : exact.error().detail));
  auto delta = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta, *K1,
      *h0.pt, *h0.rep, theta, *K0,
      raw, /*T_H0=*/6.0, /*T_H1=*/2.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Delta);
  REQUIRE_MESSAGE(delta.has_value(),
      "nested FIML nonlinear delta failed: " <<
      (delta.has_value() ? "" : delta.error().detail));

  REQUIRE(exact->eigenvalues.size() == 1);
  CHECK(exact->eigenvalues.allFinite());
  CHECK(exact->eigenvalues.minCoeff() >= 0.0);
  CHECK((exact->eigenvalues - delta->eigenvalues).cwiseAbs().maxCoeff() < 1e-12);
  bool warned = false;
  for (const auto& w : exact->warnings) {
    warned = warned || w.find("local tangent") != std::string::npos;
  }
  CHECK(warned);
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

// ===========================================================================
// Multi-group + cross-group equality (measurement-invariance) coverage.
//
// The nested-FIML cases above are all single-group. lr_test_satorra2000_fiml_
// from_data and fiml_ugamma_spectrum claim multi-group, cross-group-equality
// (metric/scalar), and mean-structure support, but nothing exercises that
// seam. These cases drive it: H1 = configural (loadings free per group),
// H0 = metric (a single bare label per non-marker loading, which the build
// step replicates across groups => equal-across-groups). A scalar variant
// ties the intercepts too, exercising the (μ_b, vech(Σ_b)) η layout under a
// mean-block constraint.
// ===========================================================================

TEST_CASE("nested FIML restriction map: multi-group metric NT gamma gives all-one eigenvalues") {
  auto h1 = build_mean_model("f =~ x1 + x2 + x3 + x4", /*n_groups=*/2);
  auto h0 = build_mean_model("f =~ x1 + a2*x2 + a3*x3 + a4*x4", /*n_groups=*/2);
  Eigen::VectorXd theta1(static_cast<Eigen::Index>(h1.ev.n_free()));
  theta1.setConstant(0.6);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(h0.ev.n_free()));
  theta0.setConstant(0.6);
  auto raw = model_missing_raw(h1, theta1, {120, 100});

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df1 = magmaan::inference::df_stat(*h1.pt, *samp, theta1);
  auto df0 = magmaan::inference::df_stat(*h0.pt, *samp, theta0);
  REQUIRE(df1.has_value());
  REQUIRE(df0.has_value());
  // 3 non-marker loadings, each tied across 2 groups => 3 restrictions.
  REQUIRE(*df0 - *df1 == 3);
  auto K1 = magmaan::estimate::build_eq_constraints(*h1.pt);
  auto K0 = magmaan::estimate::build_eq_constraints(*h0.pt);
  REQUIRE(K1.has_value());
  REQUIRE(K0.has_value());

  auto r = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/9.0, /*T_H1=*/3.0, *df0, *df1,
      magmaan::robust::GammaSource::NT,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(r.has_value(),
      "multi-group nested FIML NT failed: " <<
      (r.has_value() ? "" : r.error().detail));
  REQUIRE(r->eigenvalues.size() == 3);
  INFO("NT eigenvalues = ", r->eigenvalues.transpose());
  CHECK((r->eigenvalues.array() - 1.0).abs().maxCoeff() < 1e-8);
  CHECK(r->scale_c == doctest::Approx(1.0).epsilon(1e-9));
  CHECK(r->T_scaled == doctest::Approx(6.0).epsilon(1e-9));  // T_diff = 6, c = 1
}

TEST_CASE("nested FIML restriction map: multi-group metric trace_CinvS algebra and row-dup invariance") {
  auto h1 = build_mean_model("f =~ x1 + x2 + x3 + x4", /*n_groups=*/2);
  auto h0 = build_mean_model("f =~ x1 + a2*x2 + a3*x3 + a4*x4", /*n_groups=*/2);
  Eigen::VectorXd theta1(static_cast<Eigen::Index>(h1.ev.n_free()));
  theta1.setConstant(0.6);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(h0.ev.n_free()));
  theta0.setConstant(0.6);
  auto raw = model_missing_raw(h1, theta1, {160, 130});

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df1 = magmaan::inference::df_stat(*h1.pt, *samp, theta1);
  auto df0 = magmaan::inference::df_stat(*h0.pt, *samp, theta0);
  REQUIRE(df1.has_value());
  REQUIRE(df0.has_value());
  REQUIRE(*df0 - *df1 == 3);
  auto K1 = magmaan::estimate::build_eq_constraints(*h1.pt);
  auto K0 = magmaan::estimate::build_eq_constraints(*h0.pt);
  REQUIRE(K1.has_value());
  REQUIRE(K0.has_value());

  auto exact = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw, /*T_H0=*/9.0, /*T_H1=*/3.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(exact.has_value(),
      "multi-group nested FIML exact failed: " <<
      (exact.has_value() ? "" : exact.error().detail));
  REQUIRE(exact->eigenvalues.size() == 3);

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
  REQUIRE_MESSAGE(reduced.has_value(),
      "compute_fiml_satorra2000 failed: " <<
      (reduced.has_value() ? "" : reduced.error().detail));
  REQUIRE(reduced->eigenvalues.size() == 3);

  const Eigen::MatrixXd Cinv = inverse_ldlt(reduced->C);
  const double trace_direct = (Cinv * reduced->S).trace();
  INFO("trace_direct = ", trace_direct,
       " trace_CinvS = ", reduced->trace_CinvS,
       " eigenvalues = ", reduced->eigenvalues.transpose());
  CHECK(trace_direct == doctest::Approx(reduced->trace_CinvS).epsilon(1e-9));
  CHECK(reduced->trace_CinvS ==
        doctest::Approx(reduced->eigenvalues.sum()).epsilon(1e-12));
  CHECK(reduced->trace_CinvS_sq ==
        doctest::Approx(reduced->eigenvalues.squaredNorm()).epsilon(1e-12));
  CHECK((exact->eigenvalues - reduced->eigenvalues).cwiseAbs().maxCoeff() < 1e-9);

  const auto raw_dup = duplicate_raw_rows(raw, 3);
  auto exact_dup = magmaan::robust::lr_test_satorra2000_fiml_from_data(
      *h1.pt, *h1.rep, theta1, *K1,
      *h0.pt, *h0.rep, theta0, *K0,
      raw_dup, /*T_H0=*/9.0, /*T_H1=*/3.0, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(exact_dup.has_value(),
      "duplicated multi-group nested FIML exact failed: " <<
      (exact_dup.has_value() ? "" : exact_dup.error().detail));
  REQUIRE(exact->eigenvalues.size() == exact_dup->eigenvalues.size());
  INFO("original eigenvalues = ", exact->eigenvalues.transpose(),
       " duplicated eigenvalues = ", exact_dup->eigenvalues.transpose());
  CHECK((exact->eigenvalues - exact_dup->eigenvalues).cwiseAbs().maxCoeff() < 1e-9);
  CHECK(exact_dup->scale_c == doctest::Approx(exact->scale_c).epsilon(1e-9));
  CHECK(exact_dup->adjust_d0 == doctest::Approx(exact->adjust_d0).epsilon(1e-9));
}

TEST_CASE("fiml_ugamma_spectrum: cross-group metric equality spectrum is finite with trace identity") {
  auto built = build_mean_model("f =~ x1 + a2*x2 + a3*x3 + a4*x4", /*n_groups=*/2);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.58);
  auto raw = model_missing_raw(built, theta0, {130, 110});

  magmaan::estimate::Estimates est;
  est.theta = theta0;
  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df_or = magmaan::inference::df_stat(*built.pt, *samp, est.theta);
  REQUIRE(df_or.has_value());
  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, est, *df_or, /*chi2_lrt=*/6.0);
  REQUIRE_MESSAGE(sp.has_value(),
      "cross-group metric fiml_ugamma_spectrum failed: " <<
      (sp.has_value() ? "" : sp.error().detail));

  CHECK(sp->df == *df_or);
  CHECK(sp->eigvals.size() == *df_or);
  CHECK(sp->eigvals.allFinite());
  CHECK(sp->eigvals.minCoeff() > 0.0);
  CHECK(sp->trace_xcheck == doctest::Approx(sp->eigvals.sum()));

  // The split trace identity routes Δ through K (cross-group equality ties
  // parameters between the two stacked group blocks); a block-ordering bug in
  // the Δ←Δ·K projection breaks it.
  const auto pieces = fiml_projection_pieces(built, raw, est);
  const double trace_h1 = (pieces.H * pieces.Gamma).trace();
  const double trace_h0 =
      (pieces.P_inv *
       pieces.Delta.transpose() * pieces.J * pieces.Delta).trace();
  const double trace_from_split = trace_h1 - trace_h0;
  const double trace_direct = (pieces.U * pieces.Gamma).trace();
  INFO("trace_from_split = ", trace_from_split,
       " trace_direct = ", trace_direct,
       " spectrum trace = ", sp->trace_xcheck);
  CHECK(trace_direct == doctest::Approx(trace_from_split).epsilon(1e-9));
  CHECK(sp->trace_xcheck == doctest::Approx(trace_from_split).epsilon(1e-9));
}

TEST_CASE("fiml_ugamma_spectrum: cross-group scalar (equal intercepts) spectrum exercises mean-block layout") {
  auto built = build_mean_model(
      "f =~ x1 + a2*x2 + a3*x3 + a4*x4\n"
      "x1 ~ i1*1\nx2 ~ i2*1\nx3 ~ i3*1\nx4 ~ i4*1",
      /*n_groups=*/2);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.5);
  auto raw = model_missing_raw(built, theta0, {140, 120});

  magmaan::estimate::Estimates est;
  est.theta = theta0;
  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df_or = magmaan::inference::df_stat(*built.pt, *samp, est.theta);
  REQUIRE(df_or.has_value());
  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, est, *df_or, /*chi2_lrt=*/6.0);
  REQUIRE_MESSAGE(sp.has_value(),
      "cross-group scalar fiml_ugamma_spectrum failed: " <<
      (sp.has_value() ? "" : sp.error().detail));

  CHECK(sp->df == *df_or);
  CHECK(sp->eigvals.size() == *df_or);
  CHECK(sp->eigvals.allFinite());
  CHECK(sp->eigvals.minCoeff() > 0.0);
  CHECK(sp->trace_xcheck == doctest::Approx(sp->eigvals.sum()));

  const auto pieces = fiml_projection_pieces(built, raw, est);
  const double trace_from_split =
      (pieces.H * pieces.Gamma).trace() -
      (pieces.P_inv * pieces.Delta.transpose() * pieces.J * pieces.Delta).trace();
  const double trace_direct = (pieces.U * pieces.Gamma).trace();
  INFO("trace_from_split = ", trace_from_split,
       " trace_direct = ", trace_direct,
       " spectrum trace = ", sp->trace_xcheck);
  CHECK(trace_direct == doctest::Approx(trace_from_split).epsilon(1e-9));
  CHECK(sp->trace_xcheck == doctest::Approx(trace_from_split).epsilon(1e-9));
}

TEST_CASE("fiml_ugamma_spectrum: complete-data multi-group metric matches the unstructured robust spectrum") {
  // Degeneracy guard, multi-group + cross-group-equality (metric invariance)
  // generalization of the single-group case above: on COMPLETE data the
  // first-principles FIML UΓ spectrum must reproduce the complete-data
  // UNSTRUCTURED robust spectrum eig(U·Γ̂) with the loadings tied across two
  // groups of UNEQUAL size (200 vs 170). This is the case that regressed when
  // build_u_factor's kernel projector omitted the per-group weight √(n_b/N):
  // the omission tilts ker(Aᵀ) for cross-group-coupled columns of unequal-size
  // blocks. With the weight restored, the FIML saturated spectrum and the
  // complete-data unstructured spectrum agree to ~1e-6 here, and the production
  // Structured FMG path matches lavaan across configural/metric/scalar in
  // experiments/21-fiml-measurement-invariance-fmg.
  auto built = build_mean_model("f =~ x1 + a2*x2 + a3*x3 + a4*x4", /*n_groups=*/2);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.6);
  auto truth = built.ev.sigma(theta0);
  REQUIRE(truth.has_value());
  REQUIRE(truth->sigma.size() == 2);

  magmaan::data::RawData raw;
  const std::array<Eigen::Index, 2> ns = {200, 170};
  for (std::size_t b = 0; b < 2; ++b) {
    Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[b]);
    REQUIRE(llt.info() == Eigen::Success);
    const Eigen::MatrixXd L = llt.matrixL();
    Eigen::MatrixXd Z = deterministic_z(ns[b], truth->sigma[b].rows());
    if (b == 1) Z.array() *= 1.08;
    raw.X.push_back((Z * L.transpose()).rowwise() + truth->mu[b].transpose());
  }

  auto samp = magmaan::data::sample_stats_from_raw(raw);
  REQUIRE(samp.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 800;
  auto est = magmaan::estimate::fit_fiml(*built.pt, *built.rep, raw, theta0,
                                         magmaan::estimate::fiml::FIML{},
                                         magmaan::estimate::Backend::NloptLbfgs,
                                         opts);
  REQUIRE_MESSAGE(est.has_value(),
      "multi-group metric fit_fiml failed: " <<
      (est.has_value() ? "" : est.error().detail));
  auto fx = magmaan::estimate::fiml::fiml_extras(*built.pt, *built.rep, raw, *est);
  REQUIRE(fx.has_value());

  auto df_or = magmaan::inference::df_stat(*built.pt, *samp, est->theta);
  REQUIRE(df_or.has_value());
  auto sp = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, *est, *df_or, fx->chi2);
  REQUIRE_MESSAGE(sp.has_value(),
      "multi-group metric fiml_ugamma_spectrum failed: " <<
      (sp.has_value() ? "" : sp.error().detail));

  // Reference: complete-data unstructured robust spectrum eig(U·Γ̂), multi-block.
  auto uf = magmaan::robust::build_u_factor(
      *built.pt, *built.rep, *samp, *est,
      magmaan::robust::InferenceSpec{
          magmaan::robust::Information::Expected,
          magmaan::robust::WeightMoments::Unstructured,
          magmaan::robust::ScoreCovariance::Empirical});
  REQUIRE(uf.has_value());
  auto Zc = magmaan::robust::casewise_contributions(raw, *samp, /*include_means=*/true);
  REQUIRE(Zc.has_value());
  Eigen::VectorXd denom(2);
  denom << static_cast<double>(raw.X[0].rows()),
           static_cast<double>(raw.X[1].rows());
  auto M = magmaan::robust::reduced_gamma_sample(*uf, *Zc, denom);
  REQUIRE(M.has_value());
  auto ev_ref = magmaan::robust::ugamma_eigenvalues(*M);
  REQUIRE(ev_ref.has_value());
  REQUIRE(static_cast<int>(uf->df) == *df_or);

  Eigen::VectorXd a = sp->eigvals;             // ascending, length df
  Eigen::VectorXd b = ev_ref->tail(*df_or);    // top-df, ascending
  INFO("fiml eig = ", a.transpose(), " unstructured eig = ", b.transpose());
  CHECK((a - b).cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("two_stage_em_ml_inference: complete-data multi-group matches complete-data robust path") {
  auto built = build_mean_model("f =~ x1 + a2*x2 + a3*x3 + a4*x4", /*n_groups=*/2);
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.6);
  auto truth = built.ev.sigma(theta0);
  REQUIRE(truth.has_value());
  REQUIRE(truth->sigma.size() == 2);

  magmaan::data::RawData raw;
  const std::array<Eigen::Index, 2> ns = {180, 145};
  for (std::size_t b = 0; b < 2; ++b) {
    Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[b]);
    REQUIRE(llt.info() == Eigen::Success);
    Eigen::MatrixXd Z = deterministic_z(ns[b], truth->sigma[b].rows());
    if (b == 1) Z.array() *= 1.05;
    raw.X.push_back((Z * llt.matrixL().transpose()).rowwise() +
                    truth->mu[b].transpose());
  }

  auto sm = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE_MESSAGE(sm.has_value(),
      "saturated_em_moments failed: " << (sm.has_value() ? "" : sm.error().detail));
  magmaan::data::SampleStats samp;
  samp.S = sm->cov;
  samp.mean = sm->mean;
  samp.n_obs = sm->n_obs;

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 800;
  auto est = magmaan::estimate::fit_ml(
      *built.pt, *built.rep, samp, theta0, {}, magmaan::estimate::Backend::NloptLbfgs,
      opts);
  REQUIRE_MESSAGE(est.has_value(),
      "two-stage ML fit failed: " << (est.has_value() ? "" : est.error().detail));

  auto ml2s = magmaan::estimate::fiml::two_stage_em_ml_inference(
      *built.pt, *built.rep, raw, *est);
  REQUIRE_MESSAGE(ml2s.has_value(),
      "two_stage_em_ml_inference failed: " <<
      (ml2s.has_value() ? "" : ml2s.error().detail));

  // ML2S follows the robust.two.stage convention: Unstructured (sample/saturated
  // h1) weight moments, not Structured (model-implied). On complete data this is
  // robust.two.stage, NOT robust.sem/MLM (which is structured); the two genuinely
  // differ when Σ̂ ≠ Σ(θ̂). The explicit reference path must use the same spec.
  auto rob = magmaan::robust::robust_se(
      *built.pt, *built.rep, samp, *est, raw,
      magmaan::robust::InferenceSpec{
          magmaan::robust::Information::Expected,
          magmaan::robust::WeightMoments::Unstructured,
          magmaan::robust::ScoreCovariance::Empirical});
  REQUIRE(rob.has_value());
  CHECK((ml2s->vcov - rob->vcov).cwiseAbs().maxCoeff() < 1e-7);
  CHECK((ml2s->se - rob->se).cwiseAbs().maxCoeff() < 1e-7);

  auto uf = magmaan::robust::build_u_factor(
      *built.pt, *built.rep, samp, *est,
      magmaan::robust::InferenceSpec{
          magmaan::robust::Information::Expected,
          magmaan::robust::WeightMoments::Unstructured,
          magmaan::robust::ScoreCovariance::Empirical});
  REQUIRE(uf.has_value());
  auto Zc = magmaan::robust::casewise_contributions(raw, samp, /*include_means=*/true);
  REQUIRE(Zc.has_value());
  Eigen::VectorXd denom(2);
  denom << static_cast<double>(raw.X[0].rows()),
           static_cast<double>(raw.X[1].rows());
  auto M = magmaan::robust::reduced_gamma_sample(*uf, *Zc, denom);
  REQUIRE(M.has_value());
  auto ev_ref = magmaan::robust::ugamma_eigenvalues(*M);
  REQUIRE(ev_ref.has_value());

  REQUIRE(ml2s->df == static_cast<int>(uf->df));
  CHECK(ml2s->eigvals.size() == ev_ref->size());
  CHECK((ml2s->eigvals - *ev_ref).cwiseAbs().maxCoeff() < 1e-7);
  CHECK(ml2s->trace_ugamma == doctest::Approx(ml2s->eigvals.sum()));
  CHECK(ml2s->scaling_factor ==
        doctest::Approx(ml2s->trace_ugamma / static_cast<double>(ml2s->df)));
  CHECK(ml2s->chisq == doctest::Approx(magmaan::inference::chi2_stat(samp, *est)));
}

TEST_CASE("nested ML2S restriction map: empirical spectrum and NT collapse") {
  auto h1 = build_mean_model("f =~ x1 + x2 + x3 + x4");
  auto h0 = build_mean_model("f =~ x1 + a*x2 + a*x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(h1.ev.n_free()));
  theta0.setConstant(0.6);
  auto truth = h1.ev.sigma(theta0);
  REQUIRE(truth.has_value());
  REQUIRE(truth->sigma.size() == 1);

  magmaan::data::RawData raw;
  Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[0]);
  REQUIRE(llt.info() == Eigen::Success);
  raw.X.push_back((deterministic_z(220, truth->sigma[0].rows()) *
                   llt.matrixL().transpose()).rowwise() +
                  truth->mu[0].transpose());

  auto sm = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE_MESSAGE(sm.has_value(),
      "saturated_em_moments failed: " <<
      (sm.has_value() ? "" : sm.error().detail));
  magmaan::data::SampleStats samp;
  samp.S = sm->cov;
  samp.mean = sm->mean;
  samp.n_obs = sm->n_obs;

  Eigen::VectorXd start1(static_cast<Eigen::Index>(h1.ev.n_free()));
  start1.setConstant(0.55);
  Eigen::VectorXd start0(static_cast<Eigen::Index>(h0.ev.n_free()));
  start0.setConstant(0.55);
  magmaan::optim::OptimOptions opts;
  opts.max_iter = 900;
  auto est1 = magmaan::estimate::fit_ml(
      *h1.pt, *h1.rep, samp, start1, {}, magmaan::estimate::Backend::NloptLbfgs,
      opts);
  REQUIRE_MESSAGE(est1.has_value(),
      "H1 stage-2 ML fit failed: " << (est1.has_value() ? "" : est1.error().detail));
  auto est0 = magmaan::estimate::fit_ml(
      *h0.pt, *h0.rep, samp, start0, {}, magmaan::estimate::Backend::NloptLbfgs,
      opts);
  REQUIRE_MESSAGE(est0.has_value(),
      "H0 stage-2 ML fit failed: " << (est0.has_value() ? "" : est0.error().detail));

  auto df1 = magmaan::inference::df_stat(*h1.pt, samp, est1->theta);
  auto df0 = magmaan::inference::df_stat(*h0.pt, samp, est0->theta);
  REQUIRE(df1.has_value());
  REQUIRE(df0.has_value());
  REQUIRE(*df0 - *df1 == 1);
  const double T1 = magmaan::inference::chi2_stat(samp, *est1);
  const double T0 = magmaan::inference::chi2_stat(samp, *est0);

  auto K1 = magmaan::estimate::build_eq_constraints(*h1.pt);
  auto K0 = magmaan::estimate::build_eq_constraints(*h0.pt);
  REQUIRE(K1.has_value());
  REQUIRE(K0.has_value());

  auto ml2s = magmaan::robust::lr_test_satorra2000_ml2s_from_data(
      *h1.pt, *h1.rep, est1->theta, *K1,
      *h0.pt, *h0.rep, est0->theta, *K0,
      raw, T0, T1, *df0, *df1,
      magmaan::robust::GammaSource::Empirical,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(ml2s.has_value(),
      "nested ML2S complete-data failed: " <<
      (ml2s.has_value() ? "" : ml2s.error().detail));
  if (!ml2s.has_value()) return;

  REQUIRE(ml2s->eigenvalues.size() == 1);
  CHECK(ml2s->eigenvalues.allFinite());
  CHECK(ml2s->eigenvalues.minCoeff() > 0.0);
  CHECK(ml2s->T_diff == doctest::Approx(T0 - T1).epsilon(1e-10));
  CHECK(ml2s->df_diff == *df0 - *df1);
  CHECK(std::isfinite(ml2s->scale_c));
  CHECK(ml2s->scale_c > 0.0);

  auto nt = magmaan::robust::lr_test_satorra2000_ml2s_from_data(
      *h1.pt, *h1.rep, est1->theta, *K1,
      *h0.pt, *h0.rep, est0->theta, *K0,
      raw, T0, T1, *df0, *df1,
      magmaan::robust::GammaSource::NT,
      magmaan::robust::SatorraAMethod::Exact);
  REQUIRE_MESSAGE(nt.has_value(),
      "nested ML2S NT failed: " << (nt.has_value() ? "" : nt.error().detail));
  if (!nt.has_value()) return;
  REQUIRE(nt->eigenvalues.size() == 1);
  INFO("ML2S NT eigenvalues = ", nt->eigenvalues.transpose());
  CHECK((nt->eigenvalues.array() - 1.0).abs().maxCoeff() < 1e-8);
  CHECK(nt->scale_c == doctest::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("two_stage_em_ml_inference: missing data returns finite corrected output") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  auto raw = model_missing_raw(built, theta0, {170});

  auto sm = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE_MESSAGE(sm.has_value(),
      "saturated_em_moments failed: " << (sm.has_value() ? "" : sm.error().detail));
  magmaan::data::SampleStats samp;
  samp.S = sm->cov;
  samp.mean = sm->mean;
  samp.n_obs = sm->n_obs;

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 700;
  auto est = magmaan::estimate::fit_ml(
      *built.pt, *built.rep, samp, theta0, {}, magmaan::estimate::Backend::NloptLbfgs,
      opts);
  REQUIRE_MESSAGE(est.has_value(),
      "two-stage ML fit failed: " << (est.has_value() ? "" : est.error().detail));

  auto ml2s = magmaan::estimate::fiml::two_stage_em_ml_inference(
      *built.pt, *built.rep, raw, *est);
  REQUIRE_MESSAGE(ml2s.has_value(),
      "two_stage_em_ml_inference failed: " <<
      (ml2s.has_value() ? "" : ml2s.error().detail));

  CHECK(ml2s->df > 0);
  CHECK(ml2s->ntotal == 170);
  CHECK(ml2s->vcov.rows() == est->theta.size());
  CHECK(ml2s->vcov.cols() == est->theta.size());
  CHECK(ml2s->se.size() == est->theta.size());
  CHECK(ml2s->vcov.allFinite());
  CHECK(ml2s->se.allFinite());
  CHECK(ml2s->eigvals.size() == ml2s->df);
  CHECK(ml2s->eigvals.allFinite());
  CHECK(ml2s->trace_ugamma > 0.0);
  CHECK(ml2s->scaling_factor > 0.0);
  CHECK(std::isfinite(ml2s->chisq));
  CHECK(std::isfinite(ml2s->chisq_scaled));
}

TEST_CASE("two-stage Stage-2 weights: NT robust_continuous_ls reproduces the NT "
          "spectrum; ADF collapses to c=1; DLS endpoints match Nt/Adf") {
  namespace mf = magmaan::estimate::fiml;

  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  auto raw = model_missing_raw(built, theta0, {220});

  auto sm = mf::saturated_em_moments(raw);
  REQUIRE_MESSAGE(sm.has_value(),
      "saturated_em_moments failed: " << (sm.has_value() ? "" : sm.error().detail));

  magmaan::data::SampleStats samp;
  samp.S = sm->cov;
  samp.mean = sm->mean;
  samp.n_obs = sm->n_obs;

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 700;
  auto est = magmaan::estimate::fit_ml(
      *built.pt, *built.rep, samp, theta0, {},
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE(est.has_value());

  // Trusted normal-theory two-stage path (build_u_factor + reduced_gamma).
  auto nt = mf::two_stage_em_ml_inference(*built.pt, *built.rep, raw, *est);
  REQUIRE(nt.has_value());

  // Per-block Γ_FIML (n-scaled test convention), sliced from the assembled meat.
  auto gfull = mf::two_stage_gamma_from_acov(*sm, /*se_weighted=*/false);
  REQUIRE(gfull.has_value());
  std::vector<Eigen::MatrixXd> gamma;
  Eigen::Index off = 0;
  for (std::size_t b = 0; b < sm->cov.size(); ++b) {
    const Eigen::Index p = sm->cov[b].rows();
    const Eigen::Index q = p + p * (p + 1) / 2;
    gamma.push_back(gfull->block(off, off, q, q));
    off += q;
  }

  // (1) The NT weight through the explicit-weight robust sandwich, at the SAME
  //     estimate, reproduces the NT-path UΓ spectrum (identical U, Γ, Δ).
  auto w_nt = mf::two_stage_stage2_weight_blocks(*sm, mf::TwoStageWeight::Nt);
  REQUIRE(w_nt.has_value());
  auto rr_nt = magmaan::estimate::robust_continuous_ls(
      *built.pt, *built.rep, samp, *est, *w_nt, gamma);
  REQUIRE_MESSAGE(rr_nt.has_value(),
      "robust_continuous_ls (NT) failed: " <<
      (rr_nt.has_value() ? "" : rr_nt.error().detail));
  REQUIRE(rr_nt->eigvals.size() == nt->eigvals.size());
  Eigen::VectorXd a = rr_nt->eigvals;
  Eigen::VectorXd b = nt->eigvals;
  std::sort(a.data(), a.data() + a.size());
  std::sort(b.data(), b.data() + b.size());
  CHECK((a - b).cwiseAbs().maxCoeff() < 1e-7);
  CHECK(std::abs(rr_nt->satorra_bentler.scale_c - nt->scaling_factor) < 1e-7);

  // (2) ADF (W = Γ_FIML⁻¹): UΓ is a projector, so every eigenvalue is 1 and the
  //     robust scaling collapses to c = 1 — independent of the estimate.
  auto w_adf = mf::two_stage_stage2_weight_blocks(*sm, mf::TwoStageWeight::Adf);
  REQUIRE(w_adf.has_value());
  auto rr_adf = magmaan::estimate::robust_continuous_ls(
      *built.pt, *built.rep, samp, *est, *w_adf, gamma);
  REQUIRE(rr_adf.has_value());
  CHECK((rr_adf->eigvals.array() - 1.0).abs().maxCoeff() < 1e-6);
  CHECK(std::abs(rr_adf->satorra_bentler.scale_c - 1.0) < 1e-6);

  // (3) DLS endpoints: a = 0 recovers the NT weight, a = 1 recovers ADF.
  auto w_dls0 = mf::two_stage_stage2_weight_blocks(
      *sm, mf::TwoStageWeight::Dls, {0.0});
  auto w_dls1 = mf::two_stage_stage2_weight_blocks(
      *sm, mf::TwoStageWeight::Dls, {1.0});
  REQUIRE(w_dls0.has_value());
  REQUIRE(w_dls1.has_value());
  REQUIRE(w_dls0->size() == w_nt->size());
  REQUIRE(w_dls1->size() == w_adf->size());
  for (std::size_t bb = 0; bb < w_nt->size(); ++bb) {
    CHECK(((*w_dls0)[bb] - (*w_nt)[bb]).cwiseAbs().maxCoeff() < 1e-7);
    CHECK(((*w_dls1)[bb] - (*w_adf)[bb]).cwiseAbs().maxCoeff() < 1e-9);
  }
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

TEST_CASE("FIML pack overloads reproduce the raw-data paths exactly") {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Eigen::VectorXd theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.55);
  const auto raw = model_missing_raw(built, theta0, {160});

  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto h1 = magmaan::estimate::fiml::fiml_h1_moments(raw, *pack);
  REQUIRE(h1.has_value());

  magmaan::optim::OptimOptions opts;
  opts.max_iter = 600;
  auto est_raw = magmaan::estimate::fit_fiml(
      *built.pt, *built.rep, raw, theta0, magmaan::estimate::fiml::FIML{},
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE(est_raw.has_value());
  auto est_pack = magmaan::estimate::fit_fiml(
      *built.pt, *built.rep, raw, theta0, *pack,
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE(est_pack.has_value());
  CHECK(est_raw->theta == est_pack->theta);
  const auto& est = *est_raw;

  auto fx_raw = magmaan::estimate::fiml::fiml_extras(
      *built.pt, *built.rep, raw, est);
  REQUIRE(fx_raw.has_value());
  auto fx_pack = magmaan::estimate::fiml::fiml_extras(
      *built.pt, *built.rep, raw, est, *pack, *h1);
  REQUIRE(fx_pack.has_value());
  CHECK(fx_raw->logl == fx_pack->logl);
  CHECK(fx_raw->unrestricted_logl == fx_pack->unrestricted_logl);
  CHECK(fx_raw->chi2 == fx_pack->chi2);
  CHECK(fx_raw->srmr == fx_pack->srmr);

  auto info_raw = magmaan::estimate::fiml::fiml_observed_information(
      *built.pt, *built.rep, raw, est);
  REQUIRE(info_raw.has_value());
  auto info_pack = magmaan::estimate::fiml::fiml_observed_information(
      *built.pt, *built.rep, raw, est, *pack);
  REQUIRE(info_pack.has_value());
  CHECK(*info_raw == *info_pack);

  auto samp = magmaan::estimate::fiml::fiml_start_sample_stats(raw);
  REQUIRE(samp.has_value());
  auto df_or = magmaan::inference::df_stat(*built.pt, *samp, est.theta);
  REQUIRE(df_or.has_value());

  auto rob_raw = magmaan::estimate::fiml::fiml_robust_mlr(
      *built.pt, *built.rep, raw, est, *df_or, fx_raw->chi2);
  REQUIRE(rob_raw.has_value());
  auto rob_pack = magmaan::estimate::fiml::fiml_robust_mlr(
      *built.pt, *built.rep, raw, est, *df_or, fx_raw->chi2, *pack, *h1);
  REQUIRE(rob_pack.has_value());
  CHECK(rob_raw->chisq_scaled == rob_pack->chisq_scaled);
  CHECK(rob_raw->trace_ugamma == rob_pack->trace_ugamma);
  CHECK(rob_raw->se == rob_pack->se);

  auto sm_raw = magmaan::estimate::fiml::saturated_em_moments(raw);
  REQUIRE(sm_raw.has_value());
  auto sm_pack = magmaan::estimate::fiml::saturated_em_moments(raw, *pack, *h1);
  REQUIRE(sm_pack.has_value());
  CHECK(sm_raw->H == sm_pack->H);
  CHECK(sm_raw->J == sm_pack->J);
  CHECK(sm_raw->acov == sm_pack->acov);
  REQUIRE(sm_raw->mean.size() == sm_pack->mean.size());
  for (std::size_t b = 0; b < sm_raw->mean.size(); ++b) {
    CHECK(sm_raw->mean[b] == sm_pack->mean[b]);
    CHECK(sm_raw->cov[b] == sm_pack->cov[b]);
  }

  auto bl_raw = magmaan::estimate::fiml::fiml_baseline_chi2(*built.pt, raw);
  REQUIRE(bl_raw.has_value());
  auto bl_pack = magmaan::estimate::fiml::fiml_baseline_chi2(
      *built.pt, raw, *pack, *h1);
  REQUIRE(bl_pack.has_value());
  CHECK(bl_raw->chi2 == bl_pack->chi2);
  CHECK(bl_raw->df == bl_pack->df);

  auto sp_raw = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, est, *df_or, fx_raw->chi2);
  REQUIRE(sp_raw.has_value());
  auto sp_pack = magmaan::estimate::fiml::fiml_ugamma_spectrum(
      *built.pt, *built.rep, raw, est, *df_or, fx_raw->chi2, *pack, *h1);
  REQUIRE(sp_pack.has_value());
  CHECK(sp_raw->eigvals == sp_pack->eigvals);

  auto mi_raw = magmaan::inference::modification_indices_fiml(
      *built.pt, *built.rep, raw, est);
  REQUIRE(mi_raw.has_value());
  auto mi_pack = magmaan::inference::modification_indices_fiml(
      *built.pt, *built.rep, raw, est, *pack);
  REQUIRE(mi_pack.has_value());
  check_same_score_table(*mi_raw, *mi_pack);

  magmaan::inference::ModificationIndexOptions mi_opts;
  mi_opts.candidates =
      magmaan::inference::ScoreCandidateSet::WithAbsentRows;
  auto rmi_raw = magmaan::inference::frontier::modification_indices_fiml_robust(
      *built.pt, *built.rep, raw, est, mi_opts);
  REQUIRE(rmi_raw.has_value());
  auto rmi_pack = magmaan::inference::frontier::modification_indices_fiml_robust(
      *built.pt, *built.rep, raw, est, *pack, mi_opts);
  REQUIRE(rmi_pack.has_value());
  check_same_score_table(*rmi_raw, *rmi_pack);

  auto eq_built = build_mean_model("f =~ x1 + a*x2 + a*x3 + x4");
  Eigen::VectorXd eq_theta0(static_cast<Eigen::Index>(eq_built.ev.n_free()));
  eq_theta0.setConstant(0.55);
  const auto eq_raw = model_missing_raw(eq_built, eq_theta0, {160});
  auto eq_pack = magmaan::estimate::fiml::fiml_pack(eq_raw);
  REQUIRE(eq_pack.has_value());
  auto eq_est = magmaan::estimate::fit_fiml(
      *eq_built.pt, *eq_built.rep, eq_raw, eq_theta0,
      magmaan::estimate::fiml::FIML{},
      magmaan::estimate::Backend::NloptLbfgs, opts);
  REQUIRE(eq_est.has_value());

  auto st_raw = magmaan::inference::score_tests_fiml(
      *eq_built.pt, *eq_built.rep, eq_raw, *eq_est);
  REQUIRE(st_raw.has_value());
  auto st_pack = magmaan::inference::score_tests_fiml(
      *eq_built.pt, *eq_built.rep, eq_raw, *eq_est, *eq_pack);
  REQUIRE(st_pack.has_value());
  check_same_score_table(*st_raw, *st_pack);

  auto rst_raw = magmaan::inference::frontier::score_tests_fiml_robust(
      *eq_built.pt, *eq_built.rep, eq_raw, *eq_est);
  REQUIRE(rst_raw.has_value());
  auto rst_pack = magmaan::inference::frontier::score_tests_fiml_robust(
      *eq_built.pt, *eq_built.rep, eq_raw, *eq_est, *eq_pack);
  REQUIRE(rst_pack.has_value());
  check_same_score_table(*rst_raw, *rst_pack);
}
