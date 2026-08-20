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
iterate is by *geometry* (first-order stationarity via a projected gradient
or equality-constrained KKT residual), not by the backend's return code. The
two layers separate concerns that the old code conflated:

- **L1 — Optimizer Terminal Audit** lives in `src/optim/`. Operates in
  **driven coordinates** — the reduced/profiled/lifted space the optimizer
  actually minimized over. Answers "is this point primal-feasible and
  first-order stationary for the problem the backend solved?"
- **L2 — Fit Finalization Audit** lives in `src/estimate/`. Operates on
  **expanded full θ**. It records admissibility and feasibility, and—when the
  fit path supplies the ordinary full-`theta` objective gradient—audits the
  same terminal differential against the model's covariance-cone geometry.

Two coordinate systems, two questions, one fit record. L1's stationarity
verdict is what `fit$audit$stationary` reports; L2 records what
SE/χ²/robust-correction code needs and exposes the cross-method stationarity
comparison at `fit$diagnostics$geometric_stationarity`. The latter is
additive: it never overwrites the lavaan-compatible L1 result.

## L1: optimizer terminal audit

**Files:**
- `include/magmaan/optim/terminal_audit.hpp` — declarations of the
  `audit_terminal_iterate` and
  `audit_equality_constrained_terminal_iterate` free functions.
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

TerminalAudit audit_equality_constrained_terminal_iterate(
    const ConstrainedScalarProblem& prob,
    const Eigen::VectorXd&          x,
    double                          reported_f,
    const Eigen::VectorXd&          lower,
    const Eigen::VectorXd&          upper,
    TerminalAuditOptions            opts = {});
```

The audit recomputes `f(x, grad)` at the returned `x`, builds a projected
gradient infinity-norm against the (possibly ±∞) bounds, applies the configured
absolute or relative stationarity test, records an active-set readout in
driven coordinates, and produces an *advisory* `OptimStatus`.

For equality-constrained scalar problems, the companion audit additionally
recomputes \(h(x)\) and \(J_h(x)\), solves the rank-revealing least-squares
multiplier problem, and tests

```text
|| project_box(grad f(x) + J_h(x)' lambda) ||_inf
```

together with primal equality feasibility. Rank-deficient constraint
Jacobians are allowed and their numerical rank is recorded. With active box
bounds, multipliers are fitted from interior coordinates before the one-sided
bound KKT signs are applied; when multipliers are non-unique this is
conservative. This constrained audit is essential for nonlinear `==` models
and for PSD-ML's Cholesky lift, whose covariance links are nonlinear
equalities even though the resulting primitive covariance matrices are PSD by
construction.

### Options

```cpp
struct TerminalAuditOptions {
  enum class StationarityMode { Absolute, Relative };
  StationarityMode stationarity_mode = StationarityMode::Absolute;
  double absolute_tol      = 1e-3;
  double stationarity_tol  = 1e-6;
  double active_bound_tol  = 1e-12;
  double f_consistency_rel = 1e-6;
  double constraint_tol    = 1e-6;
};
```

- `stationarity_mode` selects the shape of the stationarity test.

  - **Absolute (v1 default)**: `‖Pg‖_∞ ≤ absolute_tol`. The default
    `absolute_tol = 1e-3` matches lavaan's `optim.dx.tol` default (lavaan's
    `check.gradient = TRUE` path applies the same KKT-aware projected-
    gradient check at that tolerance). v1 ships Absolute so that magmaan's
    convergence verdict is on the same yardstick as lavaan's, which makes
    cross-package comparisons honest without an internal calibration study
    we don't yet have. NOTE: since the `fmin = ½·F` unification (see
    [numerical-conventions.md](numerical-conventions.md)) the audit sees a
    gradient on the `½F` (discrepancy-half) scale for EVERY estimator,
    matching lavaan (which minimises `½F` for ML too). Before that change ML/NT
    presented the audit a full-`F` gradient — 2× lavaan's — so `1e-3` was
    effectively 2× too tight for ML; it is now genuinely apples-to-apples. The
    `absolute_tol` value itself is unchanged and still a defensive cross-package
    match, not a calibrated choice.
  - **Relative**: `‖Pg‖_∞ ≤ stationarity_tol · (1 + |f|)`. The shape an
    earlier revision shipped as the default. The argument for Relative is
    that `|f|` spans many orders of magnitude across the SEM corpus, so an
    absolute threshold may under- or over-reject at the extremes. The
    argument against is that there is no calibration study justifying a
    specific relative tolerance against downstream inference (SE, χ²,
    LRT), so the choice is opinionated. Relative remains available as a
    research-grade alternative, and the test file
    `tests/unit/terminal_audit_test.cpp` keeps explicit Relative-mode
    coverage so the path doesn't bit-rot.

  This is the first genuinely hard design call in magmaan and the choice
  is genuinely unstable. The mode flag encodes the decision in one place
  so future experiments are one option flip away; the "Tolerance
  calibration" section below sketches the empirical work that would let
  us revisit the default with data instead of a defensive choice.

- `absolute_tol` is consulted only in Absolute mode. `stationarity_tol` is
  consulted only in Relative mode. They are not interchangeable.
- `active_bound_tol` is a coordinate-distance threshold for the
  projected-gradient construction — masking a gradient component, not a
  diagnostic readout.
- `f_consistency_rel` catches the rare case where a backend leaves the
  *last-tried* iterate in `x` rather than the *best-found* one. Relative
  because the same floating-point noise that motivates the audit also
  applies here.
- `constraint_tol` is the maximum equality residual accepted by the
  constrained audit. It certifies the returned point; it does not control the
  optimizer's search tolerance.

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
| `NloptOptimizer` (NLopt C API) | `src/optim/nlopt_optimizer.cpp` | `MAXEVAL/MAXTIME`; `FORCED_STOP`; generic `FAILURE`; `ROUNDOFF_LIMITED` (kept as salvaged regardless, audit fills `grad_inf_norm`); SLSQP equality problems use the KKT audit |
| `PortOptimizer` (PORT `drmngb`) | `src/optim/port_optimizer.cpp` | IV(1)=8 noisy; IV(1)=9 false convergence; IV(1)=10 budget; IV(1)≥11 other; IV(1)=7 singular keeps `SingularConvergence` |
| `IpoptOptimizer` | `src/optim/ipopt_optimizer.cpp` | unconstrained problems use projected gradients; equality-constrained problems use the same KKT audit and may salvage stationary budget/small-step stops |

Every success path also calls the audit so `OptimResult::grad_inf_norm` has
a single source of truth (the bespoke projected-gradient loops are gone).
The audit costs one extra `f(x, grad)` evaluation per fit — exactly
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
- `nl_eq_residual` / `_inf` — IPOPT drives `h(θ̂) → 0`; recording the achieved
  infinity-norm tells the user whether the nonlinear constraints were actually
  satisfied (separate from the optimizer's terminal status).
- `active_bounds_full` — the Heywood-case detector: a variance at its 0
  bound has a one-sided derivative, and the standard info-matrix SE for it
  is not valid. Downstream `magmaan_se()` etc. can flag or fall back.

### Common-coordinate covariance-cone stationarity

The lifted Cholesky audit and an ordinary partable-coordinate audit are not
the same first-order question at a singular covariance boundary. For the
scalar analogue (v=ell^2), an inward objective gradient at (v=0) becomes
zero after differentiating through (ell=0). Consequently, equality KKT in
the lifted chart can declare the terminal point stationary even though a
feasible first-order direction exists in the original covariance variable.

`audit_geometric_stationarity()` avoids that chart singularity. It receives
the analytic objective gradient in ordinary full-`theta` coordinates and
computes two normal-cone projection residuals:

1. **Ambient residual:** linear-equality normals, nonlinear-equality tangent
   normals, and active box-bound normals.
2. **Cone residual:** the same normals plus
   $-D_b^*(U_{0b} H_b U_{0b}^{\mathsf T})$, $H_b \succeq 0$, for every
   singular primitive covariance block. Here $U_{0b}$ spans the numerical
   null space of $\Theta_b$ or $\Psi_b$, and $D_b^*$ maps a covariance
   differential back to full model coordinates.

The cone residual is therefore the distance of the objective differential
from the KKT normal cone of the original PSD-constrained model. At a
positive-definite point every covariance null space is empty, so it reduces
exactly to the ambient equality/bound residual.

A scalar residual requires a metric. The declared default is
`model_frobenius`: the product Frobenius metric induced by the assembled
LISREL matrices. Symmetric off-diagonal covariance entries have weight two,
and shared coordinates accumulate every matrix occurrence. This is invariant
to orthogonal changes of basis inside covariance blocks and avoids privileging
the Cholesky chart. It is not invariant to arbitrary changes of measurement
units; such invariance is neither claimed nor numerically possible without a
separate scale convention.

The stationarity verdict thresholds the metric-dual L2 distance, which is
unchanged by isometric changes of model coordinates. The coordinatewise
infinity residual is also retained as a familiar diagnostic, but it does not
drive the verdict because it depends on the selected basis. Both residuals use
the same half-discrepancy objective scale and numerical `1e-3` default as the
L1 audit; this common number does not make the L2 and infinity-norm criteria
identical. Feasibility uses the original
equalities, bounds, and PSD blocks. The multiplier projection is observation
only and never changes fit return semantics. Complete-data ML/GMM/LS, FIML,
PSD FIML, ML2S through its ML/GMM Stage 2, ordinal/mixed-ordinal LS, and CatML
fit paths supply full-model gradients. Profiled SNLLS and extra callback
constraints remain unchecked until their eliminated/extra constraint normals
are available in the common representation.

## R schema

Surfacing happens in `r-package/src/fit.cpp` via two helpers
(`audit_to_r`, `diagnostics_to_r`) called from `fit_result()` — the single
shared assembler for ML/GLS/ULS/WLS/SNLLS — and mirrored in
`fcsem_fit_result()`. Active-bound indices convert 0-based → 1-based at the
R boundary so they index `theta` / `partable` rows directly.

```
fit$audit$
  stationary       (logical)   geometric + primal-feasibility verdict
  grad_inf_norm    (numeric)   projected objective/Lagrangian gradient norm
  raw_grad_inf_norm (numeric)  unprojected objective-gradient norm
  grad_scaled_inf  (numeric)   scale-aware version of grad_inf_norm
  stationarity_rhs (numeric)   configured absolute/relative comparison RHS
  f_recomputed     (numeric)   f at x, recomputed by the audit
  f_consistent     (logical)   |f_recomputed - reported| ≤ rel·(1+|reported|)
  f_finite         (logical)
  constrained      (logical)   equality-KKT audit ran
  constraint_violation_inf (numeric) max primal equality residual
  constraint_jacobian_rank (integer) numerical rank of J_h(x)
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
  geometric_stationarity  (list)
    checked               (logical)
    metric                (character) "model_frobenius"
    ambient_stationary    (logical)   equality/bound geometry only
    ambient_residual_inf  (numeric)
    ambient_residual_l2   (numeric)   metric-dual norm; drives verdict
    cone_stationary       (logical)   adds primitive PSD normal cones
    cone_residual_inf     (numeric)
    cone_residual_l2      (numeric)   metric-dual norm; drives verdict
    covariance_nullity    (integer)   summed active null-space dimension
  snlls_profile_fallback  (logical)
```

**Back-compat:** `fit$converged` stays strict (true only for
`OptimStatus::Converged`). Existing R consumers
(`r-package/R/model_data.R:1300` print method; paper harness
`harness-benchmark.R`, `harness-sim-benchmark.R`) keep working bit-for-bit.
Compatibility code reads `fit$audit$stationary` for the driven-coordinate
lavaan-style verdict and `fit$optimizer_status` for the refined status
string. Cross-method studies read
`fit$diagnostics$geometric_stationarity$cone_stationary`.

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

This is exactly the distinction the audit was built to make. The previous
ad-hoc salvage at `max(1e-3, 1e3·gtol)` wouldn't have caught these either
(`0.0015 > 1e-3` borderline, `0.0073 ≫ 1e-3`), so v1
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
   `fit$audit$stationary` field records the driven-coordinate verdict for
   consumers. A separate semantics-change PR can flip this once a survey
   pass confirms no surprise.
2. **`allFit`-style cross-backend agreement** (Layer 3).
3. **LS-backend audit wiring** (`PortNlsOptimizer`, `ceres_lm`):
   residual-driven, no scalar `ObjectiveFn`.
4. **Per-estimator tolerance tuning.** Every backend uses the same
   `TerminalAuditOptions` defaults. The v1 Absolute / 1e-3 default is a
   defensive cross-package match (lavaan), not a calibrated choice; per-
   estimator tuning, or a switch to Relative with a calibrated tolerance,
   may emerge from the "Tolerance calibration" study below if ML / GLS /
   ULS / DWLS / FIML show systematically different gradient noise floors.

## Tolerance calibration (open follow-up)

v1 ships Absolute mode at `absolute_tol = 1e-3` because that matches
lavaan's `check.gradient` default and makes cross-package convergence
comparisons honest without a magmaan-specific calibration study. The
default is therefore defensive, not calibrated — the real question is:
*what stationarity threshold reliably predicts that downstream inference
gives the same answer on this iterate that it would at the geometric
optimum?* That question doesn't have a model-independent answer, and it's
the same question whether the threshold is absolute or relative. Until a
study answers it, matching lavaan is the conservative cross-package choice.

Sketch of a study that would pin it down:

1. **Salvage-candidate gallery.** Across our corpora (geiser, kline, brown,
   mplus, little, newsom, paper, textbook) under each estimator, identify
   fits where (a) the optimizer's primary return is a soft failure AND
   (b) at least one other backend (or lavaan) converges cleanly. That's the
   population where salvage actually has a chance to matter.
2. **Record both `θ_failed` and `θ_true`.** Per candidate, capture the
   recomputed `‖Pg‖∞`, the function gap, `‖θ_f − θ_t‖`, SE deltas, χ² /
   RMSEA / CFI deltas.
3. **Threshold = the largest `T`** such that every candidate with
   either `‖Pg‖∞ ≤ T` (Absolute mode) or `‖Pg‖∞ ≤ T·(1+|f|)` (Relative
   mode) has inferential deltas under (say) 1% relative. The mode itself
   is part of what the study should decide.
4. **Stratify by estimator.** Distinct evaluators carry distinct
   cancellation floors (the Newsom GLS investigation already showed GLS's
   floor is materially worse than ML's). The right answer may be a vector
   indexed by estimator.
5. **Distribution-aware reporting.** A density plot of "true-optimum
   `‖Pg‖∞`" per estimator across the corpus gives the natural calibration
   band; the threshold should sit a comfortable margin above the upper
   whisker.

The infrastructure for step 1-2 is what the corpus-survey and
convergence-sim scripts already do; adding the audit fields to their
output yields the per-candidate data essentially for free. The study
becomes a small Quarto report that re-runs as the corpus grows — a living
calibration rather than a one-shot.
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
