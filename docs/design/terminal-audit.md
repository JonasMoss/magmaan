# Terminal Audit (L1) and Fit Diagnostics (L2)

## Context

magmaan's Newsom corpus speed survey turned up cases where the L-BFGS Full
GLS fit reaches the *objective value* of the optimum (`ex5_4`:
`f = 0.00301331`, matching SNLLS to 6 digits and lavaan to 5) but is then
discarded as a `LineSearchFailed`. NLopt's `nlopt_optimize` returns
`NLOPT_FAILURE` because its line search cannot find a measurable decrease
step at the floating-point noise floor of a near-perfect-fit GLS objective;
`src/optim/nlopt_optimizer.cpp` mapped that straight to
`FitError::LineSearchFailed` and discarded the iterate **without checking
whether it was a stationary point**. The gradient-norm computation at the
bottom of the function sat on the success branch only, so the wrapper never
examined the geometry of the point it just rejected.

This is structurally the wrong shape. The principle is well-established in
optimization engineering (Nocedal & Wright; Ceres' explicit projected-gradient
KKT termination; NLopt's own acknowledgement that `ROUNDOFF_LIMITED` may
still leave a useful minimum; lme4's convergence guidance treating optimizer
status as diagnostic, not verdict): **the optimizer's return code is a hint;
first-order stationarity at the returned point is the fact**.

This change introduces a terminal audit as two cooperating layers, both as
plain structs carried on the fit record, with R-side surfacing as nested
`fit$audit` and `fit$diagnostics` sub-lists.

## Principle

**Optimizers propose; the audit disposes.** Classification of the returned
iterate is by *geometry* (first-order stationarity via projected gradient),
not by the backend's return code. The two layers separate concerns that the
old code conflated:

- **L1 — Optimizer Terminal Audit** lives in `src/optim/`. Operates in
  **driven coordinates** — the reduced/profiled space the optimizer actually
  minimized over. Answers "is this point first-order stationary?"
- **L2 — Fit Finalization Audit** lives in `src/estimate/`. Operates on
  **expanded full θ**. Answers "is this fit usable for downstream inference?"

Two coordinate systems, two questions, one fit record. L1's stationarity
verdict is what `fit$audit$stationary` reports; L2 records what
SE/χ²/robust-correction code needs (implied-Σ PD per group, equality residual,
active-bound set on full θ).

## L1: optimizer terminal audit

**Files:**
- `include/magmaan/optim/terminal_audit.hpp` — declaration of the free
  function `audit_terminal_iterate`.
- `include/magmaan/optim/problem.hpp` — `TerminalAudit` and
  `TerminalAuditOptions` structs (placed alongside `OptimResult` because they
  are part of the optimizer-output contract).
- `src/optim/terminal_audit.cpp` — implementation.

### Signature

```cpp
TerminalAudit audit_terminal_iterate(
    const ObjectiveFn&     f,
    const Eigen::VectorXd& x,
    double                 reported_f,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,
    TerminalAuditOptions   opts = {});
```

The audit recomputes `f(x, grad)` at the returned `x`, builds a projected
gradient infinity-norm against the (possibly ±∞) bounds, applies a RELATIVE
stationarity test `‖Pg‖∞ ≤ tol · (1 + |f|)`, records an active-set readout
in driven coordinates, and produces an *advisory* `OptimStatus` (it never
overrides the wrapper's actual returned status).

### Options

```cpp
struct TerminalAuditOptions {
  double stationarity_tol  = 2e-6;
  double active_bound_tol  = 1e-12;
  double f_consistency_rel = 1e-6;
};
```

- `stationarity_tol` is **relative**: an absolute threshold would mis-classify
  across a 290-model corpus where `|f|` spans many orders of magnitude. The
  test reduces to `tol` when `|f| ≪ 1` and scales with `|f|` when `|f| ≫ 1`.
  The default is two parts per million, which keeps ordinary ML line-search
  noise-floor stops salvageable without admitting the 1e-3-scale
  non-stationary textbook-corpus failures.
- `active_bound_tol` is a coordinate-distance threshold for the
  projected-gradient construction — masking a gradient component, not a
  diagnostic readout.
- `f_consistency_rel` catches the rare case where a backend leaves the
  *last-tried* iterate in `x` rather than the *best-found* one. Relative
  because the same floating-point noise that motivates the audit also
  applies here.

### Promotion policy (v1)

The audit is **observation only**. The wrapper still owns the final
`OptimStatus` it returns. The v1 policy:

| Backend reports | Audit says stationary | Wrapper returns |
|---|---|---|
| clean success | yes | `OptimStatus::Converged` (unchanged) |
| clean success | no | `OptimStatus::Converged` (unchanged — v1 does NOT downgrade) |
| soft failure | yes | **PROMOTE** to `OptimStatus::LineSearchSalvaged` |
| soft failure | no | original `FitError` (unchanged) |
| hard failure | (audit skipped) | original `FitError` |

Hard failures (no usable `x` or `f`) bypass the audit. A future PR may add
symmetric downgrade — if a backend declares success at a non-stationary
point, surface that — but this is a semantics-change separate from v1.

### Wired backends

| Backend | File | Soft-failure sites that now audit |
|---|---|---|
| `LbfgsOptimizer` (LBFGS++) | `src/optim/lbfgs_optimizer.cpp` | line-search throw; max-iter (refactor of pre-existing ad-hoc `max(1e-3, 1e3·gtol)` salvage) |
| `LbfgsBOptimizer` (LBFGS++) | `src/optim/lbfgsb_optimizer.cpp` | line-search throw; max-iter |
| `NloptOptimizer` (NLopt C API) | `src/optim/nlopt_optimizer.cpp` | `MAXEVAL/MAXTIME`; `FORCED_STOP`; generic `FAILURE`; `ROUNDOFF_LIMITED` (kept as salvaged regardless, audit fills `grad_inf_norm`) |
| `PortOptimizer` (PORT `drmngb`) | `src/optim/port_optimizer.cpp` | IV(1)=8 noisy; IV(1)=9 false convergence; IV(1)=10 budget; IV(1)≥11 other; IV(1)=7 singular keeps `SingularConvergence` |

Every success path also calls the audit so `OptimResult::grad_inf_norm` has
a single source of truth (the four bespoke projected-gradient loops are
gone). The audit costs one extra `f(x, grad)` evaluation per fit — exactly
what the success paths already did inline.

**Not wired in v1:** `PortNlsOptimizer` and `ceres_lm` (residual-driven,
no scalar `ObjectiveFn` to audit); `CeresBfgsOptimizer` (no surveyed
failures motivate it).

### Semantic shift: PORT IV(1)=8

PORT's "noisy gradient detected" status (`IV(1)=8`) was previously a hard
`FitError::NumericIssue`. After v1, IV(1)=8 plus a stationary audit becomes
`OptimStatus::LineSearchSalvaged` (success); IV(1)=8 plus non-stationary
keeps `NumericIssue`. The "noisy" message remains informative about *why*
the audit failed when it does.

## L2: fit finalization audit

**Files:**
- `include/magmaan/estimate/diagnostics.hpp` — `FitDiagnostics`,
  `DiagnosticsOptions`, `finalize_fit_diagnostics` free function.
- `src/estimate/diagnostics.cpp` — implementation; reuses the PD-check
  pattern from `src/measures/fit_measures.cpp:130-135`,
  `estimate::active_bounds`, and `model::ModelEvaluator::sigma`.

```cpp
struct FitDiagnostics {
  std::vector<bool>      sigma_pd_per_block;
  bool                   sigma_pd_all = false;
  double                 lin_eq_residual_inf = 0.0;
  bool                   lin_eq_satisfied    = true;
  Eigen::VectorXd        nl_eq_residual;          // empty when no NL constraints
  double                 nl_eq_residual_inf  = 0.0;
  bool                   nl_eq_satisfied     = true;
  ActiveBoundDiagnostics active_bounds_full;      // indexes full θ (distinct from L1 active_set)
  bool                   snlls_profile_fallback = false;  // see v1 non-goals
};
```

`finalize_fit_diagnostics(theta_full, ev, con, nl, bounds, ...)` is called
from every public `fit_*` entry point that holds a `ModelEvaluator` (i.e.
the MatrixRep paths: `fit_ml`, `fit_gls`, `fit_gmm`, `fit_snlls`,
`fit_snlls_gls`). The FCSEM path uses a different evaluator type and is
excluded — its `fit$diagnostics` is the default-constructed schema slot.

### What L2 *records* vs *gates*

L2 never blocks a fit. It records what downstream consumers need:

- `sigma_pd_per_block` / `sigma_pd_all` — ML/GLS need `Σ⁻¹` so a non-PD
  block invalidates SEs; ULS only needs finite, so a consumer reading this
  decides per-formula. (A blanket gate would wrongly reject ULS fits whose
  Σ has a singular block that ULS does not actually require to invert.)
- `lin_eq_residual_inf` — the K-reparameterization enforces `A_eq·θ = b_eq`
  by construction; a residual exceeding `lin_eq_residual_tol` signals the
  expansion itself is misbehaving (a correctness signal, not an optimizer
  signal).
- `nl_eq_residual` / `_inf` — the AL outer loop drives `h(θ̂) → 0`; recording
  the achieved infinity-norm tells the user whether AL actually converged
  the constraints (separate from whether the augmented objective stopped).
- `active_bounds_full` — the Heywood-case detector: a variance at its 0
  bound has a one-sided derivative, and the standard info-matrix SE for it
  is not valid. Downstream `magmaan_se()` etc. can flag or fall back.

## R schema

Surfacing happens in `r-package/src/fit.cpp` via two helpers
(`audit_to_r`, `diagnostics_to_r`) called from `fit_result()` — the single
shared assembler for ML/GLS/ULS/WLS/SNLLS — and mirrored in
`fcsem_fit_result()`. Active-bound indices convert 0-based → 1-based at the
R boundary so they index `theta` / `partable` rows directly.

```
fit$audit$
  stationary       (logical)   geometric verdict
  grad_inf_norm    (numeric)   projected ‖g‖∞ at the iterate
  stationarity_rhs (numeric)   the tol · (1 + |f|) it was compared against
  f_recomputed     (numeric)   f at x, recomputed by the audit
  f_consistent     (logical)   |f_recomputed - reported| ≤ rel·(1+|reported|)
  f_finite         (logical)
  active_set       (integer)   in DRIVEN coords: {-1, 0, +1}
  advisory_status  (character) "converged" / "line_search_salvaged" / ...

fit$diagnostics$
  sigma_pd_per_block      (logical)   one per group
  sigma_pd_all            (logical)
  lin_eq_residual_inf     (numeric)
  lin_eq_satisfied        (logical)
  nl_eq_residual          (numeric)   empty when no NL constraints
  nl_eq_residual_inf      (numeric)
  nl_eq_satisfied         (logical)
  active_bounds_lower     (integer)   1-based θ indices (Heywood detector)
  active_bounds_upper     (integer)
  snlls_profile_fallback  (logical)
```

**Back-compat:** `fit$converged` stays strict (true only for
`OptimStatus::Converged`). Existing R consumers
(`r-package/R/model_data.R:1300` print method; paper harness
`harness-benchmark.R`, `harness-sim-benchmark.R`) keep working bit-for-bit.
New code reads `fit$audit$stationary` for the geometric verdict and
`fit$optimizer_status` for the refined status string.

**Two coordinate systems, two active-set readouts.** `fit$audit$active_set`
is in the driven (reduced/profiled) coordinates the optimizer minimized
over; `fit$diagnostics$active_bounds_lower/upper` indexes the expanded full
θ. They will not match in general — this is the L1/L2 split made visible.

## What the corpus survey revealed

The intended PR success criterion was "Newsom `Full-fail` drops 3 → 1." The
empirical result is *no change in Full-fail count* across all 290 corpus
models — and that turns out to be the more informative outcome.

For `ex5_4` / `ex5_4c` the audit confirms: the iterate sits at the
*objective value* of the optimum (`f = 0.003`, matching lavaan to 5
digits) but the projected gradient there is **0.0015 / 0.0073** — not
machine zero, not noise-floor tiny. The objective surface is near-flat in
some directions of the constraint-reduced α space; the optimizer's line
search stopped because no further `f` decrease was measurable, and the
non-zero gradient remains.

This is exactly the distinction the audit was built to make. The
pre-existing `LbfgsOptimizer` salvage at `max(1e-3, 1e3·gtol)` wouldn't have
caught these either (`0.0015 > 1e-3` borderline, `0.0073 ≫ 1e-3`), so v1
introduces no behavioral regression on previously-salvaged iterates while
adding principled stationarity verification everywhere. The honest
classification of `ex5_4` / `ex5_4c` as **non-stationary** points to the
right next investigation (evaluator accuracy, re-parameterization), rather
than masking the problem with a looser tolerance.

`ex12_3` is a separate mechanism — NLopt L-BFGS gets stuck early at
`f = 982` (true optimum `3.06`); PORT and SNLLS both converge it. The audit
correctly reports non-stationary at `f = 982` (large gradient there). Not
fixed by tolerance tuning.

## v1 non-goals (explicit out-of-scope)

1. **Symmetric downgrade.** If a backend reports `Converged` but the audit
   says non-stationary, v1 keeps the reported status. The
   `fit$audit$stationary` field records the geometric verdict for
   consumers. A separate semantics-change PR can flip this once a survey
   pass confirms no surprise.
2. **`allFit`-style cross-backend agreement** (Layer 3).
3. **LS-backend audit wiring** (`PortNlsOptimizer`, `ceres_lm`):
   residual-driven, no scalar `ObjectiveFn`.
4. **Per-estimator tolerance tuning.** Every backend uses the same
   `TerminalAuditOptions` defaults. The `stationarity_tol = 2e-6` v1 pick
   clears ordinary one-ppm line-search remainders while still rejecting the
   non-stationary Newsom cases; tuning per estimator-class is a follow-up if
   a specific class genuinely needs it.
5. **`fit$converged` boolean semantics:** unchanged.
6. **`snlls_profile_fallback` plumbing:** the flag exists on
   `FitDiagnostics` and surfaces to R, but the v1 SNLLS expand site leaves
   it `false`. Wiring requires a small flag on `GpProblem` — follow-up.
   Documented as a known v1 gap.
7. **No retry / fallback / warm-restart inside the audit.** The audit
   observes; it never re-runs the optimizer. That's Layer 3.

## References

- Implementation roadmap: [`docs/architecture/roadmap.md`](../architecture/roadmap.md) — see the "Optimizer backends" section
  for the existing `OptimStatus` / `grad_norm` surfacing that the audit
  now sits behind.
- Open backlog: [`docs/backlog/newsom-corpus-failures.md`](../backlog/newsom-corpus-failures.md) — section 1 amended
  after the audit's empirical findings on `ex5_4` / `ex5_4c`.
- Primary bug site (now fixed): `src/optim/nlopt_optimizer.cpp` — the
  `NLOPT_FAILURE` path that previously discarded any returned iterate
  without examining its geometry.
- Prior art:
  - Nocedal & Wright, *Numerical Optimization* — line-search convergence
    analyzed via gradient-norm / first-order stationarity, not "the line
    search returned success."
  - Ceres Solver: projected-gradient KKT termination — `‖x − Π(x − g)‖∞ ≤
    gradient_tolerance` (equivalent to the per-component active-bound
    masking used here, for box bounds).
  - NLopt manual: `ROUNDOFF_LIMITED` may still leave a useful minimum;
    the API returns `xopt`, `fmin`, and status separately to support
    auditing the returned point.
  - lme4 convergence guidance: optimizer warnings as diagnostics
    cross-checked with gradients, Hessians, and alternate optimizers;
    `allFit` as the practical cross-check pattern.
