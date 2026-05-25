# `evaluate_at`: the no-optimizer audit

## Context

magmaan's `fit_*` composers package the canonical pipeline `data →
objective → optimize → audit → diagnostics`. The audit and diagnostics
layers (`audit_terminal_iterate`, `finalize_fit_diagnostics`) are
*post-fit* operations — they ask "is the iterate the optimizer just
returned a defensible answer?" by recomputing F and ∇F at the iterate
and applying lavaan's `optim.dx.tol = 1e-3` projected-gradient test.

Sometimes a caller wants to ask the same question about a parameter
vector that *didn't* come from a magmaan optimization run. Concrete
needs:

1. **Cross-package parity.** The snlls-constrained paper's lavaan-audit
   parity supplement asks: "does the audit's verdict agree with
   `lavInspect(fit, "converged")` on lavaan's own returned θ̂?" To
   answer it, the supplement needs to compute F, ∇F, and the audit at
   θ̂ without re-fitting in magmaan.
2. **Ground-truth comparison.** Replications of the Geiser / Ernst
   designs may want to ask whether a published reference solution
   passes the audit on the simulated data.
3. **Seed sensitivity.** Two starting seeds may converge to two
   different points; auditing each, against the other's objective,
   reveals which is in the deeper basin.

Before this helper, the snlls-constrained supplement implemented (1) by
writing θ̂ into `spec$partable$ustart`, calling `fit_uls`/`fit_gls`/etc.
with `max_iter = 50`, and reading the audit at the optimizer's exit
point. Empirically every stationary-start cell exits with `iter = 0` so
the audit fires at θ̂, but the path threw `OptimizerNonConvergence` on
24 / 756 cells (NLopt budget-exhausted exits get rejected before the
audit fires) and read as a workaround rather than an API call. See the
paper's `TODO.md` loose-end-B section for the full path-A description.

## API

```cpp
namespace magmaan::estimate {

enum class Estimator { ULS, GLS, WLS, ML };

fit_expected<Estimates>
evaluate_at(spec::LatentStructure pt, const model::MatrixRep& rep,
            const SampleStats& samp, const Eigen::VectorXd& theta_full,
            Estimator estimator,
            const gmm::Weight& weight = {},
            Bounds bounds = {},
            optim::TerminalAuditOptions audit_opts = {});

}
```

`Estimates` is the same struct the `fit_*` composers return. The
documented "exact/closed-form solve where no outer optimizer ran"
defaults apply: `iterations = 0`, `f_evals = 1`, `g_evals = 1`,
`optimizer_status = Converged` (a sentinel — `audit.advisory_status` is
the authoritative verdict), and `audit` / `diagnostics` filled from
the standard post-fit calls.

R surface (in `r-package/R/model_data.R`):

```r
evaluate_at(model, data, theta,
            estimator = c("ULS", "GLS", "WLS", "ML"),
            W = NULL, bounds = NULL, audit_options = NULL)
```

Returns the same fit-result list as `fit_uls` / `fit_gls` /
`fit_wls` / `fit_ml`.

## Composition

```
                  ┌──────── partable + sample stats
                  │
   resolve_fixed_x_from_sample
                  │
        ModelEvaluator::build       (linear/nonlinear constraint scan)
                  │
       ┌──────────┴─────────┐
       │                    │
  gmm::residuals      ml_objective       ← branched on Estimator
  (ULS/GLS/WLS)            (ML)
       │                    │
   optim::scalarize         │            ← LS path only
       │                    │
       └──────────┬─────────┘
                  │
              ScalarProblem.f(theta_full, &grad)    ← compute F, ∇F
                  │
                  ├──→ audit_terminal_iterate(...)  ← L1 verdict
                  │
                  └──→ finalize_fit_diagnostics(...) ← L2 diagnostics
```

The objective builders are exactly the ones `fit_uls` / `fit_gls` /
`fit_wls` / `fit_ml` use internally, so the F the audit recomputes is
byte-identical to what the optimizer would have minimised.

## SNLLS coordination

The audit operates in **driven coordinates** by design (see
`docs/design/terminal-audit.md` and `terminal_audit.hpp`'s narrative).
For SNLLS, driven coordinates are β-space — the optimizer never sees
the profiled α block.

`evaluate_at` is for the **unprofiled** question: "is this full θ
stationary for the moment objective?" That matches the cross-package
parity use case: lavaan's optimizer minimises the full-θ objective, so
the natural verdict on θ_lavaan is the full-θ audit, not the β-space
one. The SNLLS β-space audit (which has its own correct answer for the
profiled question) is a separate entry point — `evaluate_at_snlls(β)`
would be the analogous no-optimizer helper, and the `GpProblem` layer
in `gmm/gp.hpp` is already factored to support it. Not implemented in
v1 because no consumer needs it yet.

## Bounds default

Default is `variance_bounds(pt)` (lavaan's `pos.var` preset: lower=0 on
variance diagonals, ±∞ everywhere else). The rationale:

- Lavaan's own audit applies `optim.dx.tol` against the projected
  gradient under whichever box the user / preset chose. Defaulting to
  unbounded would make `evaluate_at` more permissive than lavaan on
  boundary-variance cases — at a Heywood point, the unbounded
  gradient is non-zero (it wants to push variance further negative)
  even though the constrained problem is locally optimal.
- `variance_bounds` is data-free (it inspects only the partable), so
  the default has no hidden cost.
- Standard data-derived bounds (`standard_bounds(pt, samp)`) are
  available but require sample stats and are stricter than lavaan's
  own default for plain ML/GLS calls. Callers who want lavaan-strict
  bounds pass them explicitly.

To override: pass an explicit `Bounds` argument. To request unbounded
audits: pass a `Bounds` whose `lower`/`upper` are full-size ±∞
vectors.

## Audit options

`TerminalAuditOptions` defaults to `StationarityMode::Absolute` with
`absolute_tol = 1e-3` — the lavaan-compatible setting. Callers can
flip to relative mode or tighten the tolerance via the optional
argument. The R wrapper accepts the same fields as a named list:

```r
evaluate_at(spec, dat, theta, "ULS",
            audit_options = list(stationarity_mode = "absolute",
                                 absolute_tol = 1e-4))
```

## Tests

`tests/unit/evaluate_at_test.cpp` covers six properties, all on the
same 1F CFA fixture:

1. ULS audit at the converged θ is stationary, grad_inf < 1e-6, iter = 0.
2. ULS audit at a 1.5×-displaced θ is non-stationary, grad_inf > 1e-3.
3. GLS audit at fit_gls's converged θ is stationary.
4. ML audit at fit_ml's converged θ is stationary.
5. WLS with an empty weight returns an error.
6. theta_full with wrong size returns an error.

Run via `ctest --test-dir build/dev -R evaluate_at`.

## Future work

- **SNLLS-coordinate variant** (`evaluate_at_snlls(beta)`): same shape,
  but takes a β in driven coordinates and runs the audit there. The
  GpProblem builder already exposes the right interface; the helper
  would be a thin wrapper.
- **FCSEM / FIML**: a non-trivial extension since the FCSEM
  `ml_objective` overload returns a `ScalarProblem` but the constraint
  shape differs. Add when a consumer needs it.
- **Direct evaluator**: a true `evaluate_f_and_grad(spec, dat, theta)`
  that doesn't allocate the audit / diagnostics outputs — for users
  who only want F and ∇F at θ, not the audit verdict. The
  `ScalarProblem.f` closure is already that function; the missing
  surface is just an R-facing wrapper.
