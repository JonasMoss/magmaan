### Optimizer backends

User-facing the optimizer is selected by a single kebab-case `optimizer = "..."`
string threaded through `fit_model()` and the underlying `fit_ml/fit_uls/fit_gls/
fit_wls/fit_*_snlls/fit_*_ordinal` entries; the table lives in
`cpp/include/magmaan/estimate/backend_strings.hpp` and parses into the C++
`Backend` enum. Accepted strings: `"ceres"`, `"ceres-bfgs"`,
`"nlopt-slsqp"`, `"nlopt-bobyqa"`, `"nlopt-tnewton"`, `"nlopt-var2"`,
`"nlopt-lbfgs"`, `"nlopt-lbfgs-slsqp-fallback"`, `"ipopt"`, `"port"`,
`"port-nls"`. The R side passes the same
string through to a single Rcpp shim per fit family — no per-Backend wrapper
explosion. Solver tuning rides on a generic `control = list(max_iter, ftol,
gtol, history)` argument.

The R complete-data ML and LS helpers also expose box constraints through
`bounds = list(lower, upper)` or the named bound builders
`bounds_variance()` / `bounds_pos_var()`, `bounds_standard()`,
`bounds_wide()`, and `bounds_loading()`. The high-level `fit_model()` ML, ULS,
GLS, WLS, and ordinal LS paths thread the same bounds object into the C++
fit layer; FIML remains unbounded at the R surface.

Optimizer outputs carry function/gradient evaluation counts plus a refined
success status and final stationarity diagnostic. The C++ `OptimResult` and
`estimate::Estimates` report `OptimStatus` (`Converged`,
`LineSearchSalvaged`, `SingularConvergence`, or `Unknown`) and a final
(projected, when bounded) gradient infinity norm when the backend can compute
one. R fit lists expose these as `optimizer_status` and `grad_norm`; the
`converged` field projects the common verdict (TRUE/FALSE/NA), independently
of the optimizer stop. A returned estimate need not pass that verdict.


- `Backend::NloptLbfgs` is the complete-data scalar default. FIML defaults to
  `Backend::NloptLbfgsSlsqpFallback`, which retries NLopt SLSQP from the same
  start if L-BFGS fails or returns a non-clean optimizer status. NLopt is a
  required dependency in the ordinary build, so both pieces of the fallback are
  always available.
  Known limitation: the NLopt Luksan L-BFGS line search allows ten step
  reductions, which can be insufficient to reach a finite objective from a
  valid start when parameter scales produce a large gradient. This can abort
  after 12 evaluations regardless of the overall evaluation budget. The
  standalone variance probe in `cpp/tests/checks/nlopt_lbfgs_domain.c` isolates the
  issue; the domain-recovery task is tracked in `project/backlog/todo.md`.
- `Backend::Port` is the trust-region cross-check: vendored PORT (Bell Labs)
  `drmngb_` (TOMS 611 Dennis-Gay-Welsch model-Hessian trust region; the
  algorithm behind R's `nlminb`), supports bounds natively. Vendored at
  `cpp/third_party/port/` from AMPL/ASL + Fermi-LAT (both BSD-3, manifest in
  `cpp/third_party/port/README.md`). Replaces the previous CppNumericalSolvers
  `Backend::TrustRegion`.
- `Backend::PortNls` is the least-squares-shape counterpart: PORT `drn2gb_`
  (TOMS 573 NL2SOL adaptive trust region — the algorithm behind R's `nls`).
  Drives the multi-residual `GmmProblem` directly through reverse
  communication (alternating R/J requests), so NL2SOL sees the true
  Gauss-Newton-plus-secant model Hessian instead of the scalarised
  ½‖r‖² collapse. On non-convex SNLLS problems with multiple local
  optima (e.g., Bollen's democracy SEM under GLS) it may converge to a
  different basin than the gradient backends; the cross-check tests
  document this rather than enforce agreement.
- `Backend::Ceres` / `Backend::CeresBfgs` cover Ceres Levenberg-Marquardt and
  dense line-search BFGS on the least-squares path (build with
  `MAGMAAN_WITH_CERES=ON`).
- `Backend::Ipopt` is the optional system-IPOPT interior-point backend (build
  with `MAGMAAN_WITH_IPOPT=ON`). It is available as a general scalar optimizer
  and accepts nonlinear equality constraints. v1 uses IPOPT's limited-memory
  Hessian approximation, so magmaan supplies objective gradients and
  constraint Jacobians but not exact Lagrangian Hessians.
- `Backend::NloptSlsqp` exposes NLopt's SLSQP for ML, FIML, and LS scalar
  paths (gradient SQP, Kraft 1988). It accepts simple bounds and nonlinear
  equality constraints via analytic constraint Jacobians.
- `Backend::NloptBobyqa`, `Backend::NloptTnewton`, `Backend::NloptVar2`,
  `Backend::NloptLbfgs` round out the NLopt roster with distinct algorithm
  ideas: Powell 2009 derivative-free quadratic-model trust region (BOBYQA,
  finite bounds required); Nash 1985 preconditioned truncated Newton with
  CG inner solve (TNEWTON); Shanno-Phua 1980 full (dense) variable-metric
  BFGS (VAR2); and NLopt's own L-BFGS. All five share the one
  `NloptOptimizer` adapter parameterised over an opaque
  `NloptAlgorithm` enum.
- The local `ceres` and `ipopt` presets are optional optimizer comparison
  builds. Ceres is FetchContent-managed; IPOPT is deliberately a system
  dependency because its BLAS/LAPACK and sparse-linear-solver stack is a
  toolchain choice. `just r-install-ceres` and `just r-install-ipopt` mirror
  the relevant compile definition into the R shared object so the accepted
  optimizer strings are executable from the R dev surface in one install.

### Equality-constrained automatic start fallback

When equality constraints prevent std.lv start transport, the automatic start
pipeline constructs the fallback in normalized sample units and maps it back
to caller units. This scales the simple baseline's latent variance constants
with the data, including for FABIN2/3, and preserves supplied start hints.
The branch remains native and reports `equality-constraints-unsupported`:
normalizing data units does not change the identification chart. Explicit
native starts retain their constructor conventions; required transport still
refuses equality constraints. Nonlinear constraints retain the native fallback
because ML normalization does not support them. This repair does not change
barrier weights, marker rules, defaults or near-pole verdicts.

### Explicit start reporting

Ordinary fits supplied with a start table or previous fit report
`fitting$requested$starts = "table"`; `fitting$effective$starts` appends
`+table` to the resolved convention (for example `lavaan-0.7.2+table`).
This describes attempt 1: matched parameters use the table, unmatched parameters
use that convention, and retry attempts retain their own recorded starts.

### Versioned fitting post-check

Fits whose resolved convergence component is `lavaan-0.7.2` retain
`FittingReport::post_check`, exposed as `fit$fitting$post_check` in magmaanlab
and `as_lab_fit(fit)$fitting$post_check` in magmaan. The list contains `ok`,
`var_na`, `ov_variance_negative`, `lv_variance_negative`, `cov_lv_not_pd`, and
`theta_not_pd`; other convergence components report `NULL`.
The shared C++ check follows lavaan 0.7.2's variance-row NA/negative branches,
uses the propagated latent covariance with dummy latents removed, and checks
only continuous response entries of theta at the absolute eigenvalue cutoff
`-epsilon^(3/4)`. Ordinal checks use reconstructed partable estimates.
This report emits no warning and leaves convergence, inference gating and
magmaan admissibility diagnostics unchanged.

### Versioned FIML H1

The `lavaan-0.7.2` FIML preset uses marginal diagonal H1 starts,
SQUAREM acceleration and lavaan's covariance repair: when the minimum
eigenvalue is below `1e-6`, add `max(diag(Sigma))*1e-8` to the diagonal.
Native FIML H1 options and tolerances are unchanged. Converged lavaan H1
means and covariances are gated at `1e-10` relative (with unit absolute floor).
When lavaan's H1 EM stalls, the preset must also report non-convergence and
use the same repair rule; endpoint moments and their derived first starts
are outside this compatibility contract. The retained N=20 seed 12953003
has a covariance gap of `1.1e-7`; two equivalent lavaan call forms themselves
differ by `1.8e-4`. Both stall. The internal prepared H1 accessor and
`fit$fitting$h1` expose per-group `converged`, EM update `iterations`,
`covariance_repairs` and the `lavaan_covariance_ridge` rule flag.
Frozen C++ witnesses and live lavaan 0.7.2 tests cover the stalled case and
two converged controls (TASK-129.7).

The opt-in installed-lavaan 0.7.2 simulation gate is
[`test_preset_simulation_parity.R`](../../../r-package/tests/testthat/test_preset_simulation_parity.R)
(`MAGMAAN_PARITY=1`). Its eight named cases cover complete ML, FIML and
all-ordinal delta/theta DWLS, including grouped equalities. The comparison is
per parameter, `abs(a-b) <= 1e-5 * (1 + max(abs(a), abs(b)))`, matching the
pinned fixture gate. The approved alternative for non-retry endpoints requires
identical convergence/verdicts, both maximum gradients <=1e-3 in lavaan units,
lavaan statistics (2N times the objective) agreeing within 1e-6 and maximum estimate difference divided
by the same-fit lavaan standard SE <=1e-3. The test evaluates the preset endpoint
with lavaan's objective/gradient in oracle coordinates and records every metric;
contract-route counts print per model. Rescaled retries keep the task-47 contract.
The installed-lavaan 0.7.2 gate passes all 160 replicates (about 35 seconds).
The split between routes varies slightly between builds because the drift
is rounding noise: in the merge run (2026-10-03) 157 pass estimates and three
PoliticalDemocracy seeds (590202, 590203, 590210) pass the endpoint
alternative, with chi-square differences at most 3.3e-8, gradients at most
1.7e-6 and SE-scaled differences at most 1.7e-4; all other models use zero
endpoint alternatives. All verdicts agree and no rescaled retry occurs. An
earlier 1e-9 relative objective condition proved build-sensitive: near an
optimum the objective gap is second order in the estimate difference, so it
is judged on the statistic's scale.
The default suite skips this opt-in gate.

The hard-case companion
[`test_preset_hard_parity.R`](../../../r-package/tests/testthat/test_preset_hard_parity.R)
uses literal populations across 18 small-sample cells: weak markers, regression,
linear and constrained latent-basis growth, MIMIC, MCAR FIML, ordinal/mixed DWLS,
and loading-invariant groups. Run with `MAGMAAN_PARITY=1`; lavaan must be exactly
0.7.2. Both fits use structured means and random covariates. Lavaan uses
`se = "none", test = "none"`; post.check remains available. Oracle SEs are
computed only when the alternative endpoint contract needs them. Temporary
optimizer-exit instrumentation records unscaled starts before constraint
projection, PORT statuses and retries, excluding the independence model.
Starts are aligned by `(lhs, op, rhs, group)` on every route, with maximum
relative gap `|a-b|/max(1,|a|,|b|)` and the planner-amended 1e-6 threshold.
When both searches fail with equal attempt counts and matching starts and
reported decisions, `both_failed` counts as agreement and is reported separately;
failed endpoints are not evidence of a fitting-rule difference.

Per-replicate data, class counts, design timing and maximum start/first-stage
gaps go to `~/.cache/magmaan-logs/task-129.4/`. DWLS first-stage gaps compare
thresholds and ordinal/mixed sample matrices; FIML compares saturated h1 means
and covariances. The baseline's opaque FIML h1 pointer requires a read-only
accessor supplied through `options(magmaan.hard.h1_accessor = function(pointer))`,
returning `mean` and `cov`; unavailable diagnostics are explicit in the CSV and
fail final acceptance. The measurement helper and runner are retained beside
the untracked evidence, without changing fitting code.

The gate requests 100 replicates per cell in complete rounds;
`MAGMAAN_HARD_PARITY_REPS` reduces replication and
`MAGMAAN_HARD_PARITY_WORKERS` selects one or two workers. It stops starting
rounds after 28 minutes, reserving time to finish the current round.
The baseline permits documented missing marker/post-check features and the
proved standardized affine-retry defect as separate classes. The approved
TASK-129.7 amendment also classifies FIML start gaps above 1e-6 accompanied by
h1 gaps within a factor of ten as `pending_feature` in the baseline only.
Rule differences fail the gate. The merger sets `MAGMAAN_HARD_PARITY_FINAL=1`
after TASK-129.1–.3 and .7:
missing features also fail, and path-divergence rates above 5% require a decision.
Endpoints use the existing estimate/gradient/statistic/SE-unit contract above.

The completed main baseline (9ad5cb3d) uses 20 replicates per cell, two workers,
and 909.690 seconds: 279 agree, 35 both_failed, 44 pending_feature, one
path_divergence and one rule_difference. Evidence is in the `completed/`
subdirectory. D7 N=30/group seed 12958009 has matching starts (5.079077e-16),
but lavaan rejects its first attempt (gradient .001185135) and accepts a
standardized retry, while magmaan accepts one attempt (gradient .000148144).
Both final fits converge; the endpoint contract fails, so the registered
classification requires a decision. D7 seed 12958008 is path_divergence
(5% of that cell). Maximum FIML start/h1 gaps are 9.438837e-4/1.080801e-3;
these remain TASK-129.7 pending features. No final acceptance is implied.
The existing 160-fit gate still passes.

Task-59's focused seed 590214 investigation temporarily recorded each preset
objective callback's coordinates/value/gradient and traced installed lavaan's
`lav_model_grad` coordinates, with `control = list(trace = 1)` for PORT output.
The temporary instrumentation was removed before the final build. Parameter
keys, native/optimizer starts (maximum difference 1.8e-15), unit parameter
scales and PORT controls agree; this model has no equality reduction. Aligning
oracle gradient evaluations with the preset's callback points (which also
include objective-only rejected trials), the first shared coordinates differ
by 1.8e-15, then 2.7e-15, 1.8e-14, 1.1e-13, 2.4e-12, 2.4e-11,
8.1e-10 and 2.4e-7. Evaluating lavaan at every finite preset callback point
gives maximum objective error 3.65e-15 and gradient error 6.28e-14. Invalid
non-PD trial points are excluded from this same-point derivative comparison,
not from the simulation gate. This localizes the initial divergence to
rounding, followed by amplification along the optimization path.

In the instrumented run, seed 590214 stopped after 80 versus 64 iterations;
its endpoint objectives differed by 4.36e-11 and maximum gradients were
1.30e-6 versus 7.41e-7. In the restored build, the five previously failing
seeds have starts within 6.3e-15, endpoint objective differences below
5.5e-11 and maximum gradients below 1.8e-6, comfortably within the preset's
0.001 acceptance threshold. Expected-information directional curvature along
the endpoint difference is 0.0134–0.0249, consistent with a weak direction.
The evidence supports floating-point search-path amplification rather than a
systematic objective/gradient or coordinate discrepancy. This evidence supports
the approved endpoint alternative above; the estimate tolerance remains unchanged
and no fitting code was changed.
Diagnostic scripts and logs are retained under `~/.cache/magmaan-logs/task-59-*`;
the trace scripts require the temporary callback instrumentation described above.

### lavaan marker component (ML, FIML and ordinal/mixed DWLS)

`FittingOptions::marker` resolves to `default` or `lavaan-0.7.2`; the preset
selects the latter and explicit component overrides mark it modified. The
core item-rest correlation rule consumes named h1 covariance blocks, averages
signed correlations ignoring nonfinite entries and switches a weak first
marker below 0.1 to the first absolute maximum at least 0.1. Builder marker
maps select loading rows before numbering/constraints without reordering.
R replays the same fitting route on a rebuilt specification, then the original
if the switched fit fails its selected convergence rule. Both interfaces retain
the actual model and signed rounded switch information with a revert flag.
Inference/reporting consume the fitted partable; nested inference and
`refit_from_null()` refuse adapted marker coordinates until their mapping is
implemented. Casewise contrasts also refuse automatic adaptation because a
changed marker changes the parameter metric. Ordinary fitting calls rerun the
rule on each dataset.
FIML reads the preset EM H1 covariance; ordinal DWLS reads polychoric
correlations, and mixed DWLS reads the polychoric/polyserial/Pearson H1
covariance. Prepared and ordinary fits replay the same route after switching.
Switch parity requires converged H1 EM on both sides; stalled endpoints retain
the TASK-129.7 boundary. Conditional-x residual H1 and composite adaptation
are unsupported.

### Unwired objective-coordinate scale primitive

`estimate::frontier::objective_coordinate_scale()` accepts sample units and
an actual scalar-objective GN diagonal in matching equality-reduced coordinates.
It clamps two-sided multipliers to [1e-6, 1e6], or downward-only multipliers to
[1e-3, 1], and retains sample units for invalid or degenerate diagonal entries.
Malformed units/sizes and nonfinite or nonpositive outputs are typed failures.
The helper does not compute NT information. TASK-33.10.5.4 owns subsequent fit
integration and opt-in selection; this additive primitive changes no defaults.

### Layered starts with overlapping measurement blocks

TASK-136 retains a second layered candidate when an indicator loads on multiple
latents in a block. It seeds measurement loading ratios with FABIN3, completes
the same scale, structural, equality, hint and PD stages, and selects the lower
caller-sample ML discrepancy with a provenance note. Non-overlapping models
retain bit-identical starts. TASK-132/133 establish the direct variable inventory;
TASK-134 identifies the relative-sign failure. The retained Geiser latent-AR
fixture now reaches the lavaan optimum with L-BFGS and PORT under ML and GLS.

The layered PORT/L-BFGS corpus comparison of main `d1dab5dc` and candidate
`10db66a7` attempted 398 cases under both ML and GLS. The same 118 case/estimator
jobs failed preparation; 678 returned both arms, giving 1,356 paired rows.
There were 82 endpoint/convergence changes, four convergence gains and no
convergence losses. Among 1,340 doubly converged rows, the maximum relative
objective increase was 2.96e-10 and maximum absolute relative change 1.83e-9.
The amended gate in board TASK-136 comment 10 blocks only lost convergence or
an increase exceeding 1e-8 relative when both fits converge; it was specified
after the initial partial comparison. Three baseline and four candidate GLS
jobs hit the 180-second cap; separate missing-arm retries at 600 seconds all
completed without changing optimizer controls. Raw comparisons and hashes stay
in `~/.cache/magmaan-logs/task-136-corpus/`, as requested.

Eight changed endpoints were unconverged in both runs and do not gate adoption.
These values describe where the optimizers stopped, not comparable optima:

| Newsom 2024 case | Estimator | Arm | Baseline objective | Candidate objective |
|---|---|---|---:|---:|
| ex6_1c | GLS | L-BFGS | 0.0847015271560526 | 0.0847082183201977 |
| ex6_1c | GLS | PORT | 0.0851384089574275 | 0.0852369337412223 |
| ex6_1c | ML | L-BFGS | 0.0909224900248713 | 0.0909232712360382 |
| ex6_1c | ML | PORT | 0.0909196701776249 | 0.0909196715271197 |
| ex8_5c | GLS | L-BFGS | 0.0104402876684403 | 0.0104402876684411 |
| ex8_5c | GLS | PORT | 0.0104402876685199 | 0.0104402876684467 |
| ex8_5c | ML | L-BFGS | 0.0230056841197252 | 0.0116990628930935 |
| ex8_5c | ML | PORT | 0.0116990628931142 | 0.0116990628931121 |
