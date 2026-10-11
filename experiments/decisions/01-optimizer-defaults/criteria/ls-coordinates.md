# Criteria: least-squares search coordinates (lane ls-coordinates)

**Status: proposed 2026-10-11 by the planner (TASK-33.10.5.5) for the user's
review. It becomes binding when the user approves it, before any decision run;
the adoption decision itself stays with the user.**

## Decision

For continuous fixed-weight least-squares fits, separately for ULS, DWLS
(continuous, diagonal weight) and WLS (ADF), and for ML2S Stage-2 least squares:
keep the current default search coordinates (information-refined sample units,
`coordinate_scaling = "information"`) or adopt one of the opt-in modes built by
TASK-33.10.5.4:

| Arm | Coordinates | Role |
|---|---|---|
| `default` | library default, called without options | baseline |
| `objective_downward` | Gauss-Newton diagonal of the fitting objective, multipliers clamped to [1e-3, 1] | candidate (smaller change: may only shrink units) |
| `objective` | the same diagonal, multipliers clamped to [1e-6, 1e6] | candidate (answers whether enlarging units is safe) |

Ordinal and mixed routes refuse these modes (TASK-33.10.5.6) and are not
covered; ML, GLS and FIML keep their own settled coordinates.

## Evidence that motivates the lane

TASK-33.10.5.4: `objective` cut the ULS failures in the retained witnesses
sharply but those fits still failed the terminal audit; DWLS and WLS were
unchanged (the failures ended at 12 evaluations in every mode); higher-order
mixed ULS cells converged to primitive-inadmissible endpoints; none of the six
witnesses passed. The lane therefore asks whether the candidates help on fresh,
held-out problems, not on the witnesses that motivated them.

## Problems

- The decisions/01 test and control families (`R/families.R`): regular,
  weak-loading, mixed-unit and collapsed/Heywood-prone populations, under the
  native, x100, x0.01 and separate-unit transforms, at the families' existing
  sample sizes. Controls are reported and never gate.
- ML2S Stage-2 cells reuse the same populations with MAR missingness on half of
  the variables (single-cause, 15%), fitted by the library's ML2S route.
- Replications: 100 per population and N, to be confirmed by a pilot that
  prices the run (non-confirmatory, its own seed base). **Seed base:**
  2026101101, distinct from every earlier decisions/01 run and the pilot.

## Judge and rules

The judge is `fit$converged` (the library verdict), with the objective compared
against the best known objective for that problem across arms and a long
reference fit. Rates are reported per family and estimator, never pooled.

For each estimator separately, a candidate is **eligible** when, in every test
family:

1. **Primary.** It certifies at least as many attainable problems as `default`.
2. **Best known.** It has no net loss against `default` in reaching the best
   known objective (within the library's accuracy budget).
3. **Invariance.** Its certified count does not fall under any unit transform
   relative to the native units by more than `default`'s does.

**Outcome.** If both candidates are eligible, `objective_downward` is chosen
(the smaller change) unless `objective` certifies more pooled test problems and
has no net loss against `objective_downward` in any test family. If one is
eligible, it is chosen; if none is, the default stays and the report names the
failing families. Every loss against `default` is listed, and median
evaluations and seconds among successes are reported (secondary, not gating).
The user decides adoption from the report; an adopted candidate is promoted
with a reproduction run on the same seed base, as in the ml-gls lane.

## Compute

Expected to be similar to an ml-gls run (about 10,000 tasks per estimator; the
2026-10-09 ml-gls rerun took about 25 minutes on four local workers). Run
locally within the work slice or on simbox under nice 10, after the pilot.
