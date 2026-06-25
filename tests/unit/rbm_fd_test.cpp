// Independent finite-difference correctness checks for the reduced-bias
// M-estimation (RBM) "adjusted estimating equations" driver. The RBM
// ingredients j(θ) (observed bread) and e(θ) (estimating-function meat) are
// already FD-gated indirectly: `weighted_moment_rbm_parts` builds the identical
// meat from the identical IJ blocks that `robust_weighted_moment_ij` uses, and
// those blocks are gated in weighted_inference_test / ordinal_test / fiml_test
// (observed_moment_bread_fd, the *_ij case-weight finite differences, the
// Γ-influence FD gates). What is untested is the RBM *driver's assembly on top*:
// the penalty P(θ)=−½·tr(j⁻¹e), the explicit Newton correction, and whether the
// implicit θ_RBM actually solves the adjusted estimating equation
// ∇[F + tr(j⁻¹e)/(2N)] = 0. This file gates that assembly, three ways:
//
//   Check A — explicit one-step reconstruction. Independently central-difference
//     the penalty P(θ)=−½·tr(j(θ)⁻¹e(θ)), solve j_α·δ = ∇P, and assert
//     correction ≈ δ and adjustment ≈ ∇P. Guards the sign, the −½, the
//     K-reduction, and the solve. Runs on EVERY family (it needs only the
//     explicit result and the independent trace(θ); no optimization).
//   Check B — explicit ↔ implicit cross-validation. The analytic one-step and
//     the numerically-minimized penalized objective must agree to the O(‖corr‖²)
//     second Newton step. Needs N large enough to be in the second-order regime.
//   Check C — implicit stationarity (the adjusted estimating equation, verbatim).
//     Reconstruct M(θ)=base.f(θ)+trace(θ)/(2N) entirely from public calls,
//     central-difference ∇M at the returned implicit θ_RBM, assert ‖∇M‖≈0.
//
// Scope. B and C need the implicit RBM solve, which minimizes M by a
// finite-difference gradient (rbm.cpp `objective_gradient_fd`) that rebuilds the
// parts ~2q times per iterate. For the estimated-weight families that carry the
// IF(Ŵ) channel (ordinal/mixed/ML2S — especially missing-data ML2S with its
// per-case Stage-1 Γ-influence) the parts are too expensive to iterate in a unit
// test, so those families run Check A only. The full A/B/C runs on the
// cheap-implicit families (continuous LS, ML, FIML); the implicit-optimizer
// wiring it exercises is shared across all families.
//
// trace(θ): the weighted families read it directly from the public *_rbm_parts
// (`weighted_trace`), independent of the driver's penalty assembly; ML and FIML
// have no public parts builder, so they use the "base-swap" identity instead —
// rbm_explicit_<fam> stores trace_term at base.theta and its parts_at matches
// the implicit path's, so calling explicit with base={.theta=θ} yields trace(θ).
// base.f(θ) (Check C) is reconstructed per family from its public objective
// constructor. Fixtures are unconstrained, so K = I (verified per case by
// information_reduced == information).

#include <doctest/doctest.h>

#include "../test_fit.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include "magmaan/data/ordinal.hpp"
#include "magmaan/data/raw_data.hpp"
#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/ordinal.hpp"
#include "magmaan/estimate/fiml.hpp"
#include "magmaan/estimate/frontier/dls_weight.hpp"
#include "magmaan/estimate/frontier/rbm.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/nt.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/optim/optimizers.hpp"
#include "magmaan/optim/problem.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/robust/weighted_inference.hpp"
#include "magmaan/spec/build.hpp"

namespace {

namespace mf = magmaan::estimate::frontier;
using Vec = Eigen::VectorXd;
using ScalarFn = std::function<double(const Vec&)>;
constexpr double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------
// Fixtures (mirrors tests/unit/rbm_test.cpp; the helpers there live in that
// TU's anonymous namespace and are not linkable across TUs).
// ---------------------------------------------------------------------------

struct BuiltModel {
  std::unique_ptr<magmaan::spec::LatentStructure> pt;
  std::unique_ptr<magmaan::model::MatrixRep> rep;
  magmaan::model::ModelEvaluator ev;
};

BuiltModel build_model(std::string_view src, bool meanstructure) {
  auto fp = magmaan::parse::Parser::parse(src);
  REQUIRE(fp.has_value());
  magmaan::spec::BuildOptions opts;
  opts.meanstructure = meanstructure;
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

BuiltModel build_mean_model(std::string_view src) {
  return build_model(src, /*meanstructure=*/true);
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

Eigen::MatrixXd model_data_matrix(const BuiltModel& built,
                                  const Eigen::VectorXd& theta, Eigen::Index n) {
  auto truth = built.ev.sigma(theta);
  REQUIRE(truth.has_value());
  Eigen::LLT<Eigen::MatrixXd> llt(truth->sigma[0]);
  REQUIRE(llt.info() == Eigen::Success);
  return (deterministic_z(n, truth->sigma[0].rows()) *
          llt.matrixL().transpose())
             .rowwise() +
         truth->mu[0].transpose();
}

magmaan::data::RawData model_raw(const BuiltModel& built,
                                 const Eigen::VectorXd& theta, Eigen::Index n) {
  magmaan::data::RawData raw;
  raw.X.push_back(model_data_matrix(built, theta, n));
  return raw;
}

// Complete model data with a deterministic MCAR mask poked into two columns;
// no row is left all-missing.
magmaan::data::RawData model_raw_missing(const BuiltModel& built,
                                         const Eigen::VectorXd& theta,
                                         Eigen::Index n) {
  const double na = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd X = model_data_matrix(built, theta, n);
  Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> M(X.rows(),
                                                                X.cols());
  M.setOnes();
  for (Eigen::Index r = 0; r < X.rows(); ++r) {
    if (r % 7 == 0) {
      X(r, 1) = na;
      M(r, 1) = 0;
    }
    if (r % 9 == 4) {
      X(r, 3) = na;
      M(r, 3) = 0;
    }
  }
  magmaan::data::RawData raw;
  raw.X.push_back(std::move(X));
  raw.mask.push_back(std::move(M));
  return raw;
}

template <class Derived>
double max_abs(const Eigen::MatrixBase<Derived>& x) {
  return x.size() == 0 ? 0.0 : x.cwiseAbs().maxCoeff();
}

double total_n(const std::vector<std::int64_t>& n_obs) {
  double n = 0.0;
  for (const auto nb : n_obs) n += static_cast<double>(nb);
  return n;
}

// trace(θ) = tr(j⁻¹ e) from the public moment-quadratic parts, matching the
// driver's `parts_from_weighted` (FullPivLU on the K-reduced, total-N bread).
// This is the independent trace ingredient for the weighted families (cheaper
// than re-running a full explicit RBM, and not routed through the driver's own
// penalty assembly).
double weighted_trace(const magmaan::estimate::WeightedMomentRBMParts& p) {
  return p.information.fullPivLu().solve(p.meat).trace();
}

// ---------------------------------------------------------------------------
// FD helpers — central difference with the production composite step.
// ---------------------------------------------------------------------------

double fd_h(double x, double abs_s, double rel_s) {
  return abs_s + rel_s * std::max(1.0, std::abs(x));
}

Vec central_grad(const ScalarFn& f, const Vec& x, double abs_s, double rel_s) {
  Vec g(x.size());
  for (Eigen::Index k = 0; k < x.size(); ++k) {
    const double h = fd_h(x(k), abs_s, rel_s);
    Vec xp = x, xm = x;
    xp(k) += h;
    xm(k) -= h;
    const double fp = f(xp);
    const double fm = f(xm);
    g(k) = (fp - fm) / (2.0 * h);
  }
  return g;
}

// ---------------------------------------------------------------------------
// The shared A/B/C harness. `expl`/`impl` are the two RBM forms on the same
// fixture; `trace_at`/`base_f` are independent public reconstructions.
// ---------------------------------------------------------------------------

// Self-checks + Check A (explicit one-step reconstruction). This is the cheap
// part: it needs only the explicit result and the independent trace(θ), no
// implicit optimization. The estimated-weight families (ordinal/mixed/ML2S)
// stop here — their implicit RBM solve is an FD-gradient-over-expensive-parts
// optimization that rebuilds the (IF(Ŵ)-carrying) parts hundreds of times and
// is too slow for a unit test. The implicit-form gates (B, C) run on the
// cheap-implicit families (continuous/ML/FIML) below.
void check_explicit(std::string_view label, const mf::RBMResult& expl,
                    const ScalarFn& trace_at, const Vec& theta0, double tolA) {
  INFO("case = ", label);
  const double abs_s = 1e-6;
  const double rel_s = 1e-4;

  // Cheap exact self-checks: penalty identity, penalized fmin, and K = I.
  CHECK(expl.penalty == doctest::Approx(-0.5 * expl.trace_term));
  CHECK(expl.penalized_fmin ==
        doctest::Approx(expl.estimates.fmin + expl.penalty_per_observation));
  CHECK(max_abs((expl.information_reduced - expl.information).eval()) < 1e-8);
  CHECK(max_abs((expl.meat_reduced - expl.meat).eval()) < 1e-8);

  // Check A — explicit one-step reconstruction.
  ScalarFn penalty = [&](const Vec& th) {
    const double t = trace_at(th);
    return std::isfinite(t) ? -0.5 * t : kInf;
  };
  const Vec gradP = central_grad(penalty, theta0, abs_s, rel_s);
  REQUIRE(gradP.allFinite());
  const Vec delta = expl.information_reduced.fullPivLu().solve(gradP);
  CHECK(max_abs((expl.adjustment - gradP).eval()) < tolA);
  CHECK(max_abs((expl.correction - delta).eval()) < tolA);
}

// Full A/B/C harness for the cheap-implicit families. `expl`/`impl` are the two
// RBM forms on the same fixture; `trace_at`/`base_f` are independent public
// reconstructions.
void run_abc(std::string_view label, const mf::RBMResult& expl,
             const mf::RBMResult& impl, const ScalarFn& trace_at,
             const ScalarFn& base_f, const Vec& theta0, double N, double tolA,
             double b_c, double b_floor, double tolC) {
  check_explicit(label, expl, trace_at, theta0, tolA);
  const double abs_s = 1e-6;
  const double rel_s = 1e-4;

  // Check B — explicit ↔ implicit cross-validation.
  const double nrm = expl.correction.norm();
  const double b_gap = max_abs((expl.correction - impl.correction).eval());
  INFO("Check B: gap = ", b_gap, ", corr = ", nrm, ", corr2 = ", nrm * nrm);
  CHECK(b_gap < b_c * nrm * nrm + b_floor);

  // Check C — implicit stationarity of the adjusted estimating equation.
  ScalarFn objective = [&](const Vec& th) {
    const double b = base_f(th);
    const double t = trace_at(th);
    return (std::isfinite(b) && std::isfinite(t)) ? b + t / (2.0 * N) : kInf;
  };
  const Vec gM = central_grad(objective, impl.estimates.theta, abs_s, rel_s);
  REQUIRE(gM.allFinite());
  const double scale = std::max(1.0, std::abs(objective(impl.estimates.theta)));
  const double c_resid = gM.cwiseAbs().maxCoeff() / scale;
  INFO("Check C: scaled gradM = ", c_resid);
  CHECK(c_resid < tolC);
}

// The implicit RBM solve minimizes the penalized objective with a finite-
// difference gradient (rbm.cpp `objective_gradient_fd`), whose noise floor is
// ~1e-9; a `gtol` below that never converges and spins to `max_iter` on the
// (expensive) per-iterate parts rebuild. 1e-7 is comfortably reachable and
// still leaves the returned θ stationary to well under the Check C tolerance.
mf::RBMOptions implicit_opts() {
  mf::RBMOptions opts;
  opts.optim.max_iter = 500;
  opts.optim.gtol = 1e-7;
  return opts;
}

}  // namespace

// ===========================================================================
// Continuous moment-quadratic LS
// ===========================================================================

TEST_CASE("RBM FD: continuous ULS A/B/C") {
  magmaan::data::RawData raw;
  magmaan::data::SampleStats samp;
  magmaan::estimate::Estimates ml_est;
  auto built = [&] {
    auto b = build_mean_model("f =~ x1 + x2 + x3 + x4");
    Vec theta0(static_cast<Eigen::Index>(b.ev.n_free()));
    theta0.setConstant(0.65);
    theta0.tail(4) << 0.5, 1.0, 1.5, 2.0;
    raw = model_raw(b, theta0, 400);
    auto s = magmaan::data::sample_stats_from_raw(raw);
    REQUIRE(s.has_value());
    samp = std::move(*s);
    return b;
  }();

  magmaan::optim::OptimOptions fopts;
  fopts.max_iter = 250;
  auto est = magmaan::test::fit_gmm(*built.pt, *built.rep, samp, {}, {},
                                    magmaan::estimate::Backend::NloptLbfgs, fopts);
  if (!est.has_value()) {
    FAIL(est.error().detail);
    return;
  }

  const magmaan::estimate::gmm::Weight W{};
  const auto mode = magmaan::estimate::ContinuousLsIJWeightMode::Fixed;
  auto opts = implicit_opts();

  auto expl = mf::rbm_explicit_continuous_ls(*built.pt, *built.rep, samp, *est,
                                             W, raw, mode, {}, {}, opts);
  auto impl = mf::rbm_implicit_continuous_ls(*built.pt, *built.rep, samp, *est,
                                             W, raw, mode, {}, {}, opts);
  if (!expl.has_value()) { FAIL(expl.error().detail); return; }
  if (!impl.has_value()) { FAIL(impl.error().detail); return; }

  ScalarFn trace_at = [&](const Vec& th) {
    magmaan::estimate::Estimates e;
    e.theta = th;
    auto p = magmaan::estimate::continuous_ls_rbm_parts(
        *built.pt, *built.rep, samp, e, W, raw, mode, {});
    return p.has_value() ? weighted_trace(*p) : kInf;
  };
  auto gp = magmaan::estimate::gmm::residuals(built.ev, samp, est->theta, W);
  REQUIRE(gp.has_value());
  auto sp = magmaan::optim::scalarize(*gp);
  ScalarFn base_f = [&sp](const Vec& th) {
    Vec g(th.size());
    return sp.f(th, g);
  };

  run_abc("continuous ULS", *expl, *impl, trace_at, base_f, est->theta,
          total_n(samp.n_obs), 1e-7, 30.0, 1e-4, 5e-6);
}

TEST_CASE("RBM FD: continuous GLS estimated-weight A/B/C") {
  magmaan::data::RawData raw;
  magmaan::data::SampleStats samp;
  auto built = [&] {
    auto b = build_mean_model("f =~ x1 + x2 + x3 + x4");
    Vec theta0(static_cast<Eigen::Index>(b.ev.n_free()));
    theta0.setConstant(0.65);
    theta0.tail(4) << 0.5, 1.0, 1.5, 2.0;
    raw = model_raw(b, theta0, 400);
    auto s = magmaan::data::sample_stats_from_raw(raw);
    REQUIRE(s.has_value());
    samp = std::move(*s);
    return b;
  }();

  auto est = magmaan::test::fit_gls(*built.pt, *built.rep, samp);
  if (!est.has_value()) { FAIL(est.error().detail); return; }
  auto Wor = magmaan::estimate::gmm::normal_theory_weight(built.ev, samp,
                                                          est->theta);
  REQUIRE(Wor.has_value());
  const magmaan::estimate::gmm::Weight W = *Wor;
  const auto mode = magmaan::estimate::ContinuousLsIJWeightMode::SampleNormalTheory;
  auto opts = implicit_opts();

  auto expl = mf::rbm_explicit_continuous_ls(*built.pt, *built.rep, samp, *est,
                                             W, raw, mode, {}, {}, opts);
  auto impl = mf::rbm_implicit_continuous_ls(*built.pt, *built.rep, samp, *est,
                                             W, raw, mode, {}, {}, opts);
  if (!expl.has_value()) { FAIL(expl.error().detail); return; }
  if (!impl.has_value()) { FAIL(impl.error().detail); return; }

  ScalarFn trace_at = [&](const Vec& th) {
    magmaan::estimate::Estimates e;
    e.theta = th;
    auto p = magmaan::estimate::continuous_ls_rbm_parts(
        *built.pt, *built.rep, samp, e, W, raw, mode, {});
    return p.has_value() ? weighted_trace(*p) : kInf;
  };
  auto gp = magmaan::estimate::gmm::residuals(built.ev, samp, est->theta, W);
  REQUIRE(gp.has_value());
  auto sp = magmaan::optim::scalarize(*gp);
  ScalarFn base_f = [&sp](const Vec& th) {
    Vec g(th.size());
    return sp.f(th, g);
  };

  run_abc("continuous GLS", *expl, *impl, trace_at, base_f, est->theta,
          total_n(samp.n_obs), 1e-7, 30.0, 1e-4, 5e-6);
}

// Check A only: the estimated-weight implicit B/C is already covered by the GLS
// case above (same implicit-optimizer wiring; only the IF(Ŵ) weight channel
// differs). This keeps the WLS empirical-Γ implicit solve off the suite.
TEST_CASE("RBM FD: continuous WLS estimated-weight A (explicit)") {
  magmaan::data::RawData raw;
  magmaan::data::SampleStats samp;
  Vec theta0;
  auto built = [&] {
    auto b = build_mean_model("f =~ x1 + x2 + x3 + x4");
    theta0.resize(static_cast<Eigen::Index>(b.ev.n_free()));
    theta0.setConstant(0.65);
    theta0.tail(4) << 0.5, 1.0, 1.5, 2.0;
    raw = model_raw(b, theta0, 400);
    auto s = magmaan::data::sample_stats_from_raw(raw);
    REQUIRE(s.has_value());
    samp = std::move(*s);
    return b;
  }();

  auto Wor = mf::dls_weight(built.ev, samp, raw, theta0,
                            mf::DlsWeightOptions{1.0});
  REQUIRE(Wor.has_value());
  const magmaan::estimate::gmm::Weight W = *Wor;
  auto est = magmaan::test::fit_gmm(*built.pt, *built.rep, samp, W);
  if (!est.has_value()) { FAIL(est.error().detail); return; }
  const auto mode = magmaan::estimate::ContinuousLsIJWeightMode::SampleEmpiricalWls;

  auto expl = mf::rbm_explicit_continuous_ls(*built.pt, *built.rep, samp, *est,
                                             W, raw, mode, {}, {}, {});
  if (!expl.has_value()) { FAIL(expl.error().detail); return; }

  ScalarFn trace_at = [&](const Vec& th) {
    magmaan::estimate::Estimates e;
    e.theta = th;
    auto p = magmaan::estimate::continuous_ls_rbm_parts(
        *built.pt, *built.rep, samp, e, W, raw, mode, {});
    return p.has_value() ? weighted_trace(*p) : kInf;
  };

  check_explicit("continuous WLS", *expl, trace_at, est->theta, 1e-7);
}

// ===========================================================================
// Complete-data ML (fixed-weight control)
// ===========================================================================

TEST_CASE("RBM FD: ML A/B/C") {
  magmaan::data::RawData raw;
  magmaan::data::SampleStats samp;
  magmaan::estimate::Estimates est;
  auto built = [&] {
    auto b = build_mean_model("f =~ x1 + x2 + x3 + x4");
    Vec theta0(static_cast<Eigen::Index>(b.ev.n_free()));
    theta0.setConstant(0.65);
    theta0.tail(4) << 0.5, 1.0, 1.5, 2.0;
    raw = model_raw(b, theta0, 400);
    auto s = magmaan::data::sample_stats_from_raw(raw);
    REQUIRE(s.has_value());
    samp = std::move(*s);
    magmaan::optim::OptimOptions fopts;
    fopts.max_iter = 200;
    auto f = magmaan::estimate::fit_ml(*b.pt, *b.rep, samp, theta0, {},
                                       magmaan::estimate::Backend::NloptLbfgs,
                                       fopts);
    REQUIRE(f.has_value());
    est = std::move(*f);
    return b;
  }();

  auto opts = implicit_opts();
  auto expl = mf::rbm_explicit_ml(*built.pt, *built.rep, samp, raw, est, {},
                                  opts);
  auto impl = mf::rbm_implicit_ml(*built.pt, *built.rep, samp, raw, est, {},
                                  opts);
  if (!expl.has_value()) { FAIL(expl.error().detail); return; }
  if (!impl.has_value()) { FAIL(impl.error().detail); return; }

  ScalarFn trace_at = [&](const Vec& th) {
    magmaan::estimate::Estimates e;
    e.theta = th;
    auto r = mf::rbm_explicit_ml(*built.pt, *built.rep, samp, raw, e, {}, opts);
    return r.has_value() ? r->trace_term : kInf;
  };
  auto sp = magmaan::estimate::ml_objective(built.ev, samp);
  REQUIRE(sp.has_value());
  ScalarFn base_f = [&sp](const Vec& th) {
    Vec g(th.size());
    return sp->f(th, g);
  };

  run_abc("ML", *expl, *impl, trace_at, base_f, est.theta,
          total_n(samp.n_obs), 1e-7, 30.0, 1e-4, 5e-6);
}

// ===========================================================================
// FIML (complete and missing data)
// ===========================================================================

void run_fiml_fd_case(std::string_view label, magmaan::data::RawData raw) {
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  auto pack = magmaan::estimate::fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());

  auto est = magmaan::test::fit_fiml(*built.pt, *built.rep, raw);
  if (!est.has_value()) { FAIL(est.error().detail); return; }

  auto opts = implicit_opts();
  auto expl = mf::rbm_explicit_fiml(*built.pt, *built.rep, raw, *pack, *est, {},
                                    opts);
  auto impl = mf::rbm_implicit_fiml(*built.pt, *built.rep, raw, *pack, *est, {},
                                    opts);
  if (!expl.has_value()) { FAIL(expl.error().detail); return; }
  if (!impl.has_value()) { FAIL(impl.error().detail); return; }

  ScalarFn trace_at = [&](const Vec& th) {
    magmaan::estimate::Estimates e;
    e.theta = th;
    auto r = mf::rbm_explicit_fiml(*built.pt, *built.rep, raw, *pack, e, {},
                                   opts);
    return r.has_value() ? r->trace_term : kInf;
  };
  ScalarFn base_f = [&](const Vec& th) {
    auto m = built.ev.sigma(th);
    if (!m.has_value()) return kInf;
    auto v = magmaan::estimate::fiml::FIML{}.value(raw, pack->cache, *m);
    return v.has_value() ? 0.5 * *v : kInf;
  };

  run_abc(label, *expl, *impl, trace_at, base_f, est->theta,
          static_cast<double>(pack->cache.n_total), 1e-7, 30.0, 1e-4, 5e-6);
}

TEST_CASE("RBM FD: FIML complete-data A/B/C") {
  auto raw = [] {
    auto b = build_mean_model("f =~ x1 + x2 + x3 + x4");
    Vec theta0(static_cast<Eigen::Index>(b.ev.n_free()));
    theta0.setConstant(0.65);
    theta0.tail(4) << 0.5, 1.0, 1.5, 2.0;
    return model_raw(b, theta0, 400);
  }();
  run_fiml_fd_case("FIML complete", std::move(raw));
}

TEST_CASE("RBM FD: FIML missing-data A/B/C") {
  auto raw = [] {
    auto b = build_mean_model("f =~ x1 + x2 + x3 + x4");
    Vec theta0(static_cast<Eigen::Index>(b.ev.n_free()));
    theta0.setConstant(0.65);
    theta0.tail(4) << 0.5, 1.0, 1.5, 2.0;
    return model_raw_missing(b, theta0, 400);
  }();
  run_fiml_fd_case("FIML missing", std::move(raw));
}

// ===========================================================================
// All-ordinal DWLS / WLS (estimated-weight) and ULS (fixed-weight control)
// ===========================================================================

void run_ordinal_fd_case(std::string_view label,
                         magmaan::estimate::OrdinalWeightKind weights) {
  std::mt19937 rng(20260625);
  std::normal_distribution<double> norm(0.0, 1.0);
  const Eigen::Index n = 500;
  Eigen::MatrixXd X(n, 3);
  const double load[3] = {0.80, 0.72, 0.68};
  for (Eigen::Index i = 0; i < n; ++i) {
    const double eta = norm(rng);
    for (Eigen::Index j = 0; j < 3; ++j) {
      const double y =
          load[j] * eta + std::sqrt(1.0 - load[j] * load[j]) * norm(rng);
      X(i, j) = 1.0 + (y > -0.5) + (y > 0.6);  // 3 categories
    }
  }

  auto stats = magmaan::data::ordinal_stats_from_integer_data({X});
  REQUIRE(stats.has_value());

  auto built = build_model(
      "f =~ x1 + x2 + x3\n"
      "x1 | t1 + t2\nx2 | t1 + t2\nx3 | t1 + t2\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\nx3 ~*~ 1*x3\n",
      /*meanstructure=*/false);

  const auto par = magmaan::estimate::OrdinalParameterization::Delta;
  auto est = magmaan::test::fit_ordinal_bounded(*built.pt, *built.rep, *stats,
                                                {}, weights);
  if (!est.has_value()) { FAIL(est.error().detail); return; }

  auto expl = mf::rbm_explicit_ordinal(*built.pt, *built.rep, *stats, *est,
                                       weights, par, {}, {});
  if (!expl.has_value()) { FAIL(expl.error().detail); return; }

  ScalarFn trace_at = [&](const Vec& th) {
    magmaan::estimate::Estimates e;
    e.theta = th;
    auto p = magmaan::estimate::ordinal_rbm_parts(*built.pt, *built.rep, *stats,
                                                  e, weights, par);
    return p.has_value() ? weighted_trace(*p) : kInf;
  };

  check_explicit(label, *expl, trace_at, est->theta, 1e-7);
}

TEST_CASE("RBM FD: ordinal DWLS A (explicit)") {
  run_ordinal_fd_case("ordinal DWLS", magmaan::estimate::OrdinalWeightKind::DWLS);
}
TEST_CASE("RBM FD: ordinal WLS A (explicit)") {
  run_ordinal_fd_case("ordinal WLS", magmaan::estimate::OrdinalWeightKind::WLS);
}
TEST_CASE("RBM FD: ordinal ULS A (explicit)") {
  run_ordinal_fd_case("ordinal ULS", magmaan::estimate::OrdinalWeightKind::ULS);
}

// ===========================================================================
// Mixed continuous/ordinal DWLS (estimated diagonal weight)
// ===========================================================================

TEST_CASE("RBM FD: mixed-ordinal DWLS A (explicit)") {
  std::mt19937 rng(20260626);
  std::normal_distribution<double> norm(0.0, 1.0);
  const Eigen::Index n = 600;
  Eigen::MatrixXd X(n, 4);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double eta = norm(rng);
    X(i, 0) = 1.0 + (eta > -0.6) + (eta > 0.35);                 // 3-cat ord
    X(i, 1) = 1.0 + (0.70 * eta + 0.72 * norm(rng) > 0.05);      // binary ord
    X(i, 2) = 0.76 * eta + 0.64 * norm(rng) + 0.20;             // continuous
    X(i, 3) = 0.66 * eta + 0.75 * norm(rng) - 0.10;             // continuous
  }
  const std::vector<std::vector<std::int32_t>> ordered = {{1, 1, 0, 0}};
  auto stats = magmaan::data::mixed_ordinal_stats_from_data({X}, ordered);
  REQUIRE(stats.has_value());

  auto built = build_model(
      "f =~ x1 + x2 + x3 + x4\n"
      "x1 | t1 + t2\nx2 | t1\n"
      "x1 ~*~ 1*x1\nx2 ~*~ 1*x2\n",
      /*meanstructure=*/true);

  const auto weights = magmaan::estimate::OrdinalWeightKind::DWLS;
  const auto par = magmaan::estimate::OrdinalParameterization::Delta;
  auto est = magmaan::test::fit_mixed_ordinal_bounded(*built.pt, *built.rep,
                                                      *stats, {}, weights);
  if (!est.has_value()) { FAIL(est.error().detail); return; }

  auto expl = mf::rbm_explicit_mixed_ordinal(*built.pt, *built.rep, *stats, *est,
                                             weights, par, {}, {});
  if (!expl.has_value()) { FAIL(expl.error().detail); return; }

  ScalarFn trace_at = [&](const Vec& th) {
    magmaan::estimate::Estimates e;
    e.theta = th;
    auto p = magmaan::estimate::mixed_ordinal_rbm_parts(*built.pt, *built.rep,
                                                        *stats, e, weights, par);
    return p.has_value() ? weighted_trace(*p) : kInf;
  };

  check_explicit("mixed-ordinal DWLS", *expl, trace_at, est->theta, 1e-7);
}

// ===========================================================================
// Two-stage ML (ML2S): Nt (fixed-weight control), Dwls and Adf (estimated)
// ===========================================================================

void run_ml2s_fd_case(std::string_view label,
                      magmaan::estimate::fiml::TwoStageWeight kind) {
  namespace fiml = magmaan::estimate::fiml;
  auto built = build_mean_model("f =~ x1 + x2 + x3 + x4");
  Vec theta0(static_cast<Eigen::Index>(built.ev.n_free()));
  theta0.setConstant(0.65);
  theta0.tail(4) << 0.5, 1.0, 1.5, 2.0;
  auto raw = model_raw_missing(built, theta0, 120);

  auto pack = fiml::fiml_pack(raw);
  REQUIRE(pack.has_value());
  auto h1 = fiml::fiml_h1_moments(raw, *pack);
  REQUIRE(h1.has_value());
  auto sm = fiml::saturated_em_moments(raw, *pack, *h1);
  REQUIRE(sm.has_value());

  magmaan::data::SampleStats samp;
  samp.S = sm->cov;
  samp.mean = sm->mean;
  samp.n_obs = sm->n_obs;
  const fiml::TwoStageDlsOptions dls{};
  auto w = fiml::two_stage_stage2_weight_blocks(*sm, kind, dls);
  REQUIRE(w.has_value());
  auto est = magmaan::test::fit_gmm(*built.pt, *built.rep, samp, *w);
  if (!est.has_value()) { FAIL(est.error().detail); return; }

  auto expl = mf::rbm_explicit_two_stage(*built.pt, *built.rep, raw, *pack, *h1,
                                         *est, kind, dls, {}, {});
  if (!expl.has_value()) { FAIL(expl.error().detail); return; }

  ScalarFn trace_at = [&](const Vec& th) {
    magmaan::estimate::Estimates e;
    e.theta = th;
    auto p = fiml::two_stage_rbm_parts(*built.pt, *built.rep, raw, e, *pack, *h1,
                                       *sm, kind, dls);
    return p.has_value() ? weighted_trace(*p) : kInf;
  };

  check_explicit(label, *expl, trace_at, est->theta, 1e-7);
}

TEST_CASE("RBM FD: ML2S Nt A (explicit)") {
  run_ml2s_fd_case("ML2S Nt", magmaan::estimate::fiml::TwoStageWeight::Nt);
}
TEST_CASE("RBM FD: ML2S Dwls A (explicit)") {
  run_ml2s_fd_case("ML2S Dwls", magmaan::estimate::fiml::TwoStageWeight::Dwls);
}
TEST_CASE("RBM FD: ML2S Adf A (explicit)") {
  run_ml2s_fd_case("ML2S Adf", magmaan::estimate::fiml::TwoStageWeight::Adf);
}
