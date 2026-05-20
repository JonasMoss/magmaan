#pragma once

#include <functional>
#include <string_view>

#include <Eigen/Core>

#include "magmaan/expected.hpp"
#include "magmaan/optim/lbfgs_optimizer.hpp"   // shared LbfgsOptions / LbfgsOutput

namespace magmaan::optim {

// PORT tuning reuses `LbfgsOptions` for interface parity with the other
// adapters (the public option-vocabulary across magmaan's optimizers is one
// shared struct). PORT carries its own well-tuned step-control defaults
// internally; only `max_iter` (mapped to PORT's IV(MXITER)) is forwarded,
// matching the cross-backend convention that the option struct's tolerance
// fields tune scalar L-BFGS by default.
using PortOptions = LbfgsOptions;

// PortOptimizer — wraps PORT's `drmngb_` (Dennis-Gay-Welsch model-Hessian
// trust region with simple bounds; TOMS 611). PORT is the algorithm behind
// R's `nlminb`, so `Backend::Port` gives magmaan a trust-region cross-check
// that is *the same* algorithm lavaan itself uses through nlminb — modulo
// our analytic-gradient vs nlminb's finite-difference fallback path.
//
// The vendored source lives at `third_party/port/`; see its README.md for
// the AMPL/ASL + Fermi-LAT provenance and the BSD-3 license chain. PORT is
// built into a static `port` library by `cmake/PortVendor.cmake` and is
// gated behind `MAGMAAN_WITH_PORT` (default ON).
//
// Bounds: PORT supports per-variable simple bounds natively, so the bounded
// and unbounded overloads share an internal entry — unbounded calls pass
// ±1e308 sentinels per coordinate. Conceptually identical to NLopt's
// ±HUGE_VAL convention.
//
// The objective callback follows magmaan's contract:
//   double f(x, grad_out)
//      → returns F(x); writes ∇F(x) into grad_out; returns +inf if x is
//        invalid (e.g. non-PD Σ for ML), which sets PORT's IV(TOOBIG) so the
//        next step is shortened.
//
// Error mapping (PORT IV(1) return codes):
//   3..7   → success (X-, RFCTOL-, both-, AFCTOL-, singular-convergence).
//   8      → NumericIssue       — noisy objective detected.
//   9      → LineSearchFailed   — false convergence (X step below precision).
//   10     → OptimizerNonConvergence — max iterations or evaluations.
//   11..   → LineSearchFailed   — internal/parameter errors (rare; bad
//                                  bounds, allocation failure, etc.).
class PortOptimizer {
 public:
  static constexpr std::string_view name = "port";

  PortOptimizer(PortOptions opts = {}) noexcept : opts_(opts) {}

  PortOptions options() const noexcept { return opts_; }

  using Objective = std::function<double(const Eigen::VectorXd& /*x*/,
                                         Eigen::VectorXd&       /*grad_out*/)>;

  // Unbounded entry point. Implemented in terms of the bounded overload
  // with ±1e308 sentinels per coordinate.
  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0) const;

  // Bounded entry point. `lower`/`upper` must each match `x0`'s size; use
  // ±std::numeric_limits<double>::infinity() on coordinates that are
  // genuinely unbounded (the adapter translates them to PORT's ±1e308
  // sentinels at the call boundary).
  fit_expected<LbfgsOutput>
  minimize(Objective f, const Eigen::VectorXd& x0,
           const Eigen::VectorXd& lower,
           const Eigen::VectorXd& upper) const;

 private:
  PortOptions opts_;
};

}  // namespace magmaan::optim
