// Does sharing the p×p normal-theory factor across the four NT sites buy
// anything measurable?
//
// `project/backlog/todo.md` carries a "share the p×p factor" item: ½D'(A⁻¹⊗A⁻¹)D
// is built independently by the GLS weight (A = S), the IRLS reweight
// (A = Σ(θ_k)), `inference::expected_information` (A = Σ(θ̂)), and the NT Γ
// behind the SB corrections, and ML already factors Σ(θ) every iteration for
// its own trace identity and throws it away.
//
// That item was written when a `gmm::Weight` block was a dense q×q with
// q = p + p(p+1)/2, so "share the factor" meant sharing an O(q³) Cholesky.
// After the structured-weight work a NormalTheory block stores chol(A) only,
// so the shareable quantity is now a p×p Cholesky — O(p³/3). This benchmark
// measures what fraction of an IRLS fit that actually is, i.e. the ceiling on
// any sharing scheme, before anyone builds the plumbing to share it.
//
// Reported per p:
//   evaluate_ms   ev.evaluate(θ)                         — inside the reweight
//   llt_ms        BlockWeight::normal_theory(Σ) per block — the shareable part
//   weight_ms     gmm::expected_information_weight(...)   — evaluate + llt
//   fit_ms        full fit_ml_irls
//   outer         outer iterations actually taken
//
// The sharing ceiling is (outer × llt_ms) / fit_ms.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "magmaan/data/sample_stats.hpp"
#include "magmaan/estimate/fit.hpp"
#include "magmaan/estimate/gmm/moment_quadratic.hpp"
#include "magmaan/estimate/gmm/weight.hpp"
#include "magmaan/model/matrix_rep.hpp"
#include "magmaan/model/model_evaluator.hpp"
#include "magmaan/parse/parser.hpp"
#include "magmaan/spec/build.hpp"

#include "timing/timing.hpp"

namespace {

using magmaan::bench::Arm;
using magmaan::bench::TimingOptions;

using magmaan::data::SampleStats;
using magmaan::model::build_matrix_rep;
using magmaan::model::ModelEvaluator;
using magmaan::parse::Parser;
using magmaan::spec::build;
namespace gmm = magmaan::estimate::gmm;
namespace est = magmaan::estimate;

double ms_of(const Arm& arm, const TimingOptions& opt) {
  return magmaan::bench::time_arm(arm, opt).median_ns / 1e6;
}

// A p-indicator congeneric one-factor model: p loadings, p residual variances.
std::string one_factor_syntax(int p) {
  std::string s = "f =~ x1";
  for (int i = 2; i <= p; ++i) s += " + x" + std::to_string(i);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> ps = {6, 12, 24, 48};
  if (argc > 1) {
    ps.clear();
    for (int i = 1; i < argc; ++i) ps.push_back(std::atoi(argv[i]));
  }

  std::printf("%4s %8s %11s %9s %10s %7s %8s %9s\n", "p", "n_free",
              "evaluate_ms", "llt_ms", "weight_ms", "outer", "fit_ms",
              "ceiling");
  for (int p : ps) {
    auto fp = Parser::parse(one_factor_syntax(p));
    if (!fp.has_value()) { std::printf("p=%d parse failed\n", p); continue; }
    auto pt = build(*fp, {});
    if (!pt.has_value()) { std::printf("p=%d build failed\n", p); continue; }
    auto mr = build_matrix_rep(*pt);
    if (!mr.has_value()) { std::printf("p=%d rep failed\n", p); continue; }
    auto evr = ModelEvaluator::build(*pt, *mr);
    if (!evr.has_value()) { std::printf("p=%d evaluator failed\n", p); continue; }
    const auto& ev = *evr;

    const Eigen::Index nf = static_cast<Eigen::Index>(ev.n_free());
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(nf, 0.7);

    // Population-shaped S from the model at a nearby θ, lightly perturbed so
    // the fit has something to do.
    auto m0 = ev.sigma(x0);
    if (!m0.has_value()) { std::printf("p=%d sigma failed\n", p); continue; }
    std::mt19937 rng(20260918u + static_cast<unsigned>(p));
    std::normal_distribution<double> nd(0.0, 0.01);
    SampleStats samp;
    for (std::size_t b = 0; b < m0->sigma.size(); ++b) {
      Eigen::MatrixXd S = m0->sigma[b];
      for (Eigen::Index i = 0; i < S.rows(); ++i)
        for (Eigen::Index j = 0; j <= i; ++j) {
          S(i, j) += nd(rng);
          S(j, i) = S(i, j);
        }
      samp.S.push_back(S);
      samp.n_obs.push_back(1000);
    }

    TimingOptions opt;
    opt.reps    = 11;
    opt.warmups = 2;

    const double t_eval = ms_of(Arm{"evaluate", [&] {
      auto e = ev.evaluate(x0, false, false);
      return e.has_value() ? e->moments.sigma[0].sum() : 0.0;
    }}, opt);

    // The shareable quantity: chol(Σ_b) for every block, exactly what
    // `BlockWeight::normal_theory` does and what ML's objective already has.
    const double t_llt = ms_of(Arm{"chol", [&] {
      double acc = 0.0;
      for (const auto& Sigma : m0->sigma) {
        auto bw = gmm::BlockWeight::normal_theory(
            Sigma, false, magmaan::FitError::Kind::NonPositiveDefiniteSigma,
            "bench");
        if (bw.has_value()) acc += static_cast<double>(bw->rows());
      }
      return acc;
    }}, opt);

    const double t_weight = ms_of(Arm{"weight", [&] {
      auto w = gmm::expected_information_weight(ev, samp, x0);
      return w.has_value() ? static_cast<double>(w->size()) : 0.0;
    }}, opt);

    est::IrlsOptions iopts;
    std::int32_t outer = -1;
    TimingOptions fit_opt;
    fit_opt.reps    = 5;
    fit_opt.warmups = 1;
    const double t_fit = ms_of(Arm{"fit", [&] {
      auto f = est::fit_ml_irls(*pt, *mr, samp, x0, {},
                                est::Backend::PortNls, {}, iopts);
      if (!f.has_value()) return 0.0;
      outer = f->iterations;
      return f->fmin;
    }}, fit_opt);

    const double ceiling =
        (outer > 0 && t_fit > 0.0) ? 100.0 * (outer * t_llt) / t_fit : -1.0;
    std::printf("%4d %8lld %11.4f %9.4f %10.4f %7d %8.2f %8.3f%%\n", p,
                static_cast<long long>(nf), t_eval, t_llt, t_weight, outer,
                t_fit, ceiling);
  }
  return 0;
}
