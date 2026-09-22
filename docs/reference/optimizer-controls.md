# Optimizer search controls

Search termination and the independent terminal audit are separate. Neither a
small step nor a backend success flag guarantees stationarity. The generic optimizer defaults remain available; complete-data ML uses the
validated profile below. The proposed interior Newton budget is not deployed
as a production acceptance rule.

`OptimOptions` retains its four legacy fields for source compatibility.
Explicit backend fields override the corresponding legacy setting. In R, the
same fields are nested lists inside `control`; the common parser passes these
through estimator wrappers. Each backend reads its own block, allowing a
fallback policy to carry settings for both solvers. Other backend blocks are
inactive. Unknown or duplicate names inside a backend block are errors in R.
An option unsupported by the selected NLopt algorithm is an error.

```cpp
magmaan::optim::OptimOptions options;
options.nlopt.ftol_rel = 1e-12;
options.nlopt.xtol_rel = 1e-10;
options.nlopt.vector_storage = 0; // automatic, not memory 10
```

```r
control <- list(nlopt = list(ftol_rel = 1e-12, xtol_rel = 1e-10))
# Pass control to the selected fitting entry point.
```

## Complete-data ML defaults (2026-09-22)

Bare C++ `fit_ml`, `frontier::fit_ml_psd`, the staged `api::ml()` factory,
and their R wrappers now use the validated ML numerical profile:

- NLopt relative objective tolerance 1e-12, relative step tolerance 1e-10,
  and maximum 5000 objective evaluations. Luksan gradient tolerance and
  memory remain backend defaults. PSD ML uses constraint tolerance 1e-8.
- High-level ML starts use `ml_start_values`: std.lv FABIN starts transported
  to the original marker chart when safe, native FABIN otherwise. Fixed values,
  linear/nonlinear equalities, unsupported marker layouts and failed transport
  cause native fallback. Finite user hints override transported values.
  Low-level fitters continue to use the caller's explicit start vector.
  R `ml_start_policy` reports transported, fallback, or explicitly selected starts.
- Ordinary ML with L-BFGS, SLSQP or their existing fallback adapter uses
  sample-derived scaling in equality-reduced coordinates. Pure-merge bounds
  are transformed exactly; nonlinear constraints and general affine
  constraints combined with bounds retain the existing unscaled adapter.
  `Estimates::ml_sample_scaling_applied` (R: `ml_sample_scaling`) reports
  the branch actually used. Diagnostics are evaluated in original coordinates.
- PSD ML enables its existing lifted expected-information diagonal scaling.
  It does not add a retry or change the PSD feasible set.

Explicit C++ `OptimOptions` replace the default argument: `{}` retains generic
legacy controls and disables ordinary ML scaling. To customize the new policy,
start with `ml_optim_options()` or `frontier::ml_psd_optim_options()` and edit
its backend fields. An explicit `api::OptimizerSpec` likewise replaces that
factory's optimizer options; use the ML profile when constructing it. Selecting
a backend without controls, e.g. `ml().optimizer(nlopt_slsqp())`, inherits the
ML profile. Passing `nlopt_slsqp(OptimOptions{})` explicitly opts out.
For PSD scaling, the default argument is `ml_psd_options()`; an explicit
`PsdFitOptions{}` retains its former unscaled behavior.

In R, `control=list(start="fabin3", ml_sample_scaling=FALSE)` selects native
starts and unscaled ordinary ML. `control$start` also accepts `scaled-fabin`,
`simple` and the existing named start methods. Use `preconditioning="none"`
for unscaled PSD ML. Legacy `max_iter`, `ftol` and `gtol` explicitly override
profile defaults; a supplied nested NLopt field overrides the corresponding
legacy field. Other estimators retain their existing start/control defaults.

These search changes do not implement the research Newton audit, change the
production fit verdict, or make PSD fitting the default estimator. The
validation and remaining boundary/fallback limitations are documented in
[the numerical study](../research/interior-newton-audit.md).

## NLopt: L-BFGS, SLSQP, VAR2, TNEWTON and BOBYQA

| Explicit field | Meaning | When absent |
|---|---|---|
| `ftol_rel` | Relative objective-change stopping | legacy `ftol` = 1e-10 |
| `ftol_abs` | Absolute objective-change stopping | NLopt default (disabled) |
| `xtol_rel` | Relative parameter-step stopping | legacy `gtol` = 1e-7 |
| `xtol_abs` | Same absolute step threshold for each coordinate | NLopt default (disabled) |
| `max_eval` | Function-evaluation budget, positive integer | legacy `max_iter` = 1000 |
| `tolg` | Luksan internal gradient stopping: L-BFGS, VAR2, TNEWTON | backend default (1e-8 in inspected NLopt 2.10.1) |
| `vector_storage` | Luksan storage control: L-BFGS, VAR2, TNEWTON | automatic backend choice |
| `constraint_tol` | Equality feasibility tolerance in constrained SLSQP | max(legacy `gtol`, 1e-12) |

Zero disables the f/x tolerances; `tolg=0` selects the internal default,
not a disabled gradient criterion. `vector_storage=0` selects automatic
storage. The stopping tests are alternatives, so tightening only one need
not improve accuracy. Memory semantics depend on the Luksan algorithm; it
is not a universal BFGS history length. Legacy `history` remains ignored
by NLopt to preserve old calls; use explicit `vector_storage` instead.
In the L-BFGS-to-SLSQP fallback policy, `tolg` and `vector_storage` apply
only to L-BFGS; common f/x/evaluation controls apply to both stages.
The ten-reduction L-BFGS backtracking limit is not exposed by NLopt and is
still a separate recovery task. Constraint tolerance is independent of
explicit step tolerance and does not change the terminal audit threshold.

## PORT scalar and PORT-NLS

| Explicit field | PORT setting | When absent |
|---|---|---|
| `rel_f_tol` | V(RFCTOL), relative function convergence | positive legacy `ftol`; otherwise PORT default |
| `abs_f_tol` | V(AFCTOL), absolute objective **size**, not change | PORT default |
| `x_tol` | V(XCTOL), scaled relative step convergence | PORT default |
| `false_conv_tol` | V(XFTOL), false-convergence step threshold | PORT default |
| `max_eval` | IV(MXFCAL), function-evaluation budget | 10 × legacy `max_iter` |

`max_iter` remains the iteration budget. Legacy `gtol` and `history` are
unused. Explicit nonnegative tolerances are passed verbatim, including zero;
PORT's native validity checks and stopping rules apply. Do not interpret
zero as a universal disable switch. PORT distinguishes x, relative-function,
absolute-function, singular and false-convergence exits; these are not a
shared gradient test. See vendored `dv7dfl.c`, `dparck.c` and `da7sst.c`
for defaults, permitted ranges and the coupled stopping conditions.

## IPOPT and Ceres

IPOPT's block exposes native `tol` (overall scaled optimality tolerance),
`acceptable_tol`, `acceptable_iter` (zero disables acceptable termination),
and `limited_memory_max_history`. Absent overrides, the adapter retains
legacy `gtol`, max(`ftol`, `gtol`), IPOPT's acceptable-iteration default,
and legacy `history`, respectively. These are not objective/step tolerances.
IPOPT validates native ranges.

Ceres's block exposes `function_tolerance`, `gradient_tolerance` and
`parameter_tolerance` in the ordinary and ordinal LS estimator bridges.
Their absence preserves legacy `ftol`, legacy `gtol`, and CeresOptions'
`ptol=1e-8`. Direct C++ Ceres entry points retain their existing separate
`CeresOptions` (`ftol`, `gtol`, `ptol`, `max_iter`, `verbose`). These are
Ceres stopping criteria, not NLopt equivalents.

## Scope and remaining work

Shared optimizer-backed estimator paths receive the backend blocks without
changing objectives or estimator defaults. Specialized scoring/EM/IRLS outer
loops retain their own controls; a backend block tunes the inner optimizer
where one is used, not an outer loop. Effective-control and raw stopping-code
reporting through every fitted result remains a separate backlog item.
The outer R `control` list also carries estimator-specific options and is
not globally restricted by the shared optimizer parser.

Validation for this change: 39 NLopt/PORT tests (147 assertions), four IPOPT
checks, and the R `optimizer_controls.R` and `common_verdict.R` examples passed.
The optimized core and R package built successfully. Ceres was unavailable
in the validation environment, so its optional bridges were not run.
The repository-wide layering check still flags pre-existing pinned-paper
references; no finding named a file changed by this cleanup.
