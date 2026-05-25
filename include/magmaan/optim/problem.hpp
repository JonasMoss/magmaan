#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "magmaan/expected.hpp"

// The contract between objective builders and optimizers.
//
// Objective builders (`gmm::residuals`, `gmm::gp`, `estimate::ml_objective`)
// package a model + sample into one of two plain structs of closures;
// optimizers (`optim::nlopt_lbfgs`, `optim::ceres_lm`) consume them. No
// templates, no concepts — the closures are `std::function`, so the optimizer
// never sees the builder's type. Composition happens at the call site.

namespace magmaan::optim {

// x ↦ whitened residual r̃(x); F = ½‖r̃‖². May fail (non-PD Σ at x, etc.).
using ResidualFn  = std::function<fit_expected<Eigen::VectorXd>(const Eigen::VectorXd&)>;
// x ↦ whitened Jacobian J̃(x) = ∂r̃/∂x.
using JacobianFn  = std::function<fit_expected<Eigen::MatrixXd>(const Eigen::VectorXd&)>;
// x ↦ whitened residual and Jacobian at the same point. Optional fast path for
// builders that can compute both from one model evaluation.
struct LsEvaluation {
  Eigen::VectorXd residual;
  Eigen::MatrixXd jacobian;
};
using LsEvaluationFn = std::function<fit_expected<LsEvaluation>(const Eigen::VectorXd&)>;
// (x, grad_out) ↦ F(x), writing ∇F into grad_out. +inf signals "x invalid".
using ObjectiveFn = std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)>;
// x ↦ full θ. The optimizer drives `x` (which may be a reduced parameter —
// profiled β, or constraint-reduced α); `expand` recovers the full vector.
using ExpandFn    = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
// Nonlinear optimizer constraints in the driven coordinate system.
// `h(x)` is the constraint residual vector and `J_h(x)` its Jacobian.
using ConstraintFn    = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
using ConstraintJacFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

// LS-shape callback aliases — the single-argument residual / Jacobian closures
// the bounded optimizers' `minimize_ls` overloads take. Identical to
// `ResidualFn` / `JacobianFn`; kept under their own names for those signatures.
using LsResidualFn = ResidualFn;
using LsJacobianFn = JacobianFn;

// Least-squares-shaped problem: F(x) = ½‖r(x)‖², with the full Jacobian J(x)
// available so Levenberg–Marquardt sees the true Gauss–Newton structure.
// `n_param` is the dimension the optimizer drives (≤ dim θ when profiled or
// constraint-reduced); `expand` maps the solution back to full θ.
struct GmmProblem {
  ResidualFn   r;
  JacobianFn   J;
  LsEvaluationFn eval;
  Eigen::Index n_resid = 0;
  Eigen::Index n_param = 0;
  ExpandFn     expand;
};

// Scalar-shaped problem: F(x) and ∇F(x) only — no sum-of-squares structure.
// `estimate::ml_objective` produces this; `optim::scalarize` adapts a GmmProblem.
struct ScalarProblem {
  ObjectiveFn  f;
  Eigen::Index n_param = 0;
  ExpandFn     expand;
};

// Scalar nonlinear program: minimize F(x) subject to
// constraint_lower <= h(x) <= constraint_upper and simple bounds on x. Equality
// constraints use identical lower/upper entries, e.g. 0 == h_j(x).
struct ConstrainedScalarProblem {
  ScalarProblem   objective;
  ConstraintFn    h;
  ConstraintJacFn J_h;
  Eigen::Index    n_constraint = 0;
  Eigen::VectorXd constraint_lower;
  Eigen::VectorXd constraint_upper;
};

// Optimizer termination status. The error path (`fit_expected`'s unexpected
// branch) already carries hard failures — max-iter, non-finite objective,
// unrecoverable line-search abort. This enum refines the *success* path so a
// caller can tell a clean stationary stop from a salvaged or singular one.
//
// The `NoisyObjective` / `FalseConvergence` / `BudgetExhausted` triplet
// returns iterates PORT would otherwise refuse (IV(1) = 8 / 9 / 10). Near
// SEM optima, PORT's internal consistency check between the trust-region
// model's predicted reduction and the actual reduction fires on
// floating-point cancellation noise rather than on a real optimization
// failure; the wrapper-level terminal audit gives a sharper verdict than
// PORT's heuristic does. Returning the iterate (instead of erroring) lets
// callers read `theta` / `fmin` / `audit` and decide policy in R — see
// `papers/snlls-continuous/dev/audits/convergence-audit-notes.md` §3.2 for
// the seven corpus rows this promotes out of the rejection bucket.
enum class OptimStatus {
  Converged,            // clean stop: (projected) gradient norm under tolerance
  LineSearchSalvaged,   // line search aborted, but the iterate was stationary
  SingularConvergence,  // PORT singular convergence (IV(1)=7): Hessian singular
  NoisyObjective,       // PORT IV(1)=8: predicted vs actual reduction disagree;
                        //   typically floating-point noise near tight optima
  FalseConvergence,     // PORT IV(1)=9: step length below PORT's resolvable
                        //   precision; the iterate may still be near a minimum
  BudgetExhausted,      // PORT IV(1)=10 or analogous: max_iter hit; with the
                        //   uniform-stop discipline (paper dev/todo.md §D),
                        //   common signature of an iterate the audit will
                        //   still confirm as stationary
  Unknown,              // backend did not report a refined status
};

struct OptimOptions {
  int    max_iter = 1000;
  double ftol     = 1e-10;   // matches lavaan's optim.ftol default
  double gtol     = 1e-7;    // matches lavaan's optim.gradtol default
  int    history  = 10;      // optimizer history where supported
};

// --- Terminal audit -------------------------------------------------------
//
// The types belong here (alongside OptimStatus / OptimResult) because they
// are part of the optimizer's contract: every scalar backend may carry a
// TerminalAudit on its OptimResult / OptimOutput. The free-function declaration
// and the narrative rationale live in `terminal_audit.hpp`, which includes
// this header.

// Knobs for `audit_terminal_iterate`. v1 defaults are tuned to match lavaan's
// own `check.gradient = TRUE` / `optim.dx.tol = 0.001` convergence post-check
// for cross-package comparability — see docs/design/terminal-audit.md and
// the backlog "Tolerance calibration" entry. The shape of the test (relative
// vs absolute) is genuinely under-calibrated for SEM workloads; the Relative
// mode below stays available as a research-grade alternative.
struct TerminalAuditOptions {
  // Whether the stationarity test compares the projected-gradient norm
  // against an absolute or a relative right-hand side.
  //
  //   Absolute (v1 default): ‖Pg‖_∞ ≤ absolute_tol.
  //     Matches lavaan: a fit is non-stationary if any non-bound coordinate
  //     has |∂f/∂θ| greater than `absolute_tol`. Cross-package honest;
  //     defensible without a magmaan-specific calibration study.
  //
  //   Relative: ‖Pg‖_∞ ≤ stationarity_tol · (1 + |f|).
  //     The strict-but-uncalibrated choice the earlier audit used. Kept
  //     because |f| spans many orders of magnitude across the SEM corpus,
  //     so an absolute threshold may under- or over-reject at the extremes.
  //     Flip to this mode (and tune `stationarity_tol`) once an empirical
  //     calibration study justifies it.
  //
  // This is the first hard design call in magmaan and the calibration is
  // genuinely unstable — the choice is encoded as a mode rather than baked
  // in so the experiment is one option flip away.
  enum class StationarityMode { Absolute, Relative };
  StationarityMode stationarity_mode = StationarityMode::Absolute;

  // Used by Absolute mode. Default 1e-3 matches lavaan's `optim.dx.tol`.
  double absolute_tol      = 1e-3;
  // Used by Relative mode. Default 1e-6 is the strict choice from the
  // earlier relative-only audit; only consulted when `stationarity_mode`
  // is `Relative`.
  double stationarity_tol  = 1e-6;

  // Coordinate distance at which we declare `x[i]` to be on a finite bound,
  // for the projected-gradient construction. Small absolute — we are masking
  // a gradient component, not measuring active-set membership for inference.
  double active_bound_tol  = 1e-12;
  // RELATIVE consistency check: |f(x) − reported_f| ≤ f_consistency_rel ·
  // (1 + |reported_f|). This catches "backend left last-tried point, not
  // best" — a gross-divergence detector, not a precision measurement.
  double f_consistency_rel = 1e-6;
};

// What the audit found. Plain struct, carried up via `OptimOutput` /
// `OptimResult` and surfaced to R as the nested `fit$audit` sub-record. Never
// produces a `FitError` — observation only.
struct TerminalAudit {
  bool        stationary       = false;
  double      grad_inf_norm    = -1.0;   // -1 sentinel = could not compute
  double      stationarity_rhs = -1.0;   // the tol · (1 + |f|) it compared against
  double      f_recomputed     = std::numeric_limits<double>::quiet_NaN();
  bool        f_consistent     = false;
  bool        f_finite         = false;
  // Active set in DRIVEN coordinates (size matches optimizer's x):
  // {-1 = at lower, 0 = interior, +1 = at upper}. Distinct from L2's
  // `active_bounds_full`, which indexes the expanded θ.
  std::vector<std::int8_t> active_set;
  // Advisory only — the wrapper owns the actual returned status.
  OptimStatus advisory_status  = OptimStatus::Unknown;
};

struct OptimOutput {
  Eigen::VectorXd theta_hat;
  double          fmin = 0.0;
  int             iterations = 0;
  int             f_evals = 0;
  int             g_evals = 0;
  // Refined success status and the (projected) gradient infinity-norm at the
  // solution. `grad_inf_norm < 0` means the backend did not compute it.
  OptimStatus     status        = OptimStatus::Converged;
  double          grad_inf_norm = -1.0;
  // Terminal audit at the returned point. Default-constructed (sentinels,
  // `stationary = false`) when the wrapper did not run an audit. The
  // `= {}` keeps existing `OptimOutput{...}` aggregate-inits passing under
  // `-Wmissing-field-initializers`.
  TerminalAudit   audit         = {};
};

// Optimizer output. `x` is in the driven parameter space; the caller applies
// the problem's `expand` to recover full θ.
struct OptimResult {
  Eigen::VectorXd x;
  double          fmin       = 0.0;
  int             iterations = 0;
  int             f_evals    = 0;
  int             g_evals    = 0;
  // Refined success status and the (projected) gradient infinity-norm at the
  // returned point. `grad_inf_norm < 0` means the backend did not compute it.
  OptimStatus     status        = OptimStatus::Converged;
  double          grad_inf_norm = -1.0;
  // Terminal audit at the returned point. Default-constructed (`stationary
  // = false`, sentinels) when the wrapper did not run an audit — e.g. on a
  // hard failure that bypasses it, or on a backend not wired in v1. The
  // `= {}` is load-bearing: without it, `-Wmissing-field-initializers`
  // would force every existing `OptimResult{...}` aggregate-init site to add
  // a trailing `, {}`.
  TerminalAudit   audit         = {};
};

}  // namespace magmaan::optim
