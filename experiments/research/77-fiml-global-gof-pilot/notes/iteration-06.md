# Iteration 06: latent-SEM null calibration gate

Date: 2026-08-20

## Decision

Proceed to matched sparse and diffuse misspecification panels. The curved-SEM
global effective multiplier is sufficiently well calibrated in the
representative null grid to justify measuring power. Retain explicit numerical
tangent-rank diagnostics and fail closed on non-nominal geometry in the power
analysis.

## Design

- Four correctly specified models: six-indicator one-factor CFA,
  ten-indicator correlated two-factor CFA, twelve-indicator orthogonal
  bifactor model, and five-wave linear growth model.
- Normal, moderate/severe Vale--Maurelli, and matched moderate/severe
  independent-generator data.
- Complete data and 30% MCAR, `n=120`.
- 40 cells, 500 replications per cell, 20,000 attempted fits.
- Effective multiplier with 199 Rademacher draws, compared on the same datasets
  with MLR and FMG SB, SS, pEBA(4), and `all`.

## Main result

Across the 40 null cells, the effective multiplier had mean rejection .0584,
mean absolute size error .0161, range .018--.122, and 33/40 cells in
[.025,.075]. The corresponding summaries were:

| method | mean rejection | mean absolute error | range | cells in band |
|---|---:|---:|---:|---:|
| effective multiplier | .058 | .016 | .018--.122 | 33/40 |
| FMG all | .065 | .024 | .018--.155 | 27/40 |
| FMG SS | .072 | .027 | .024--.173 | 24/40 |
| FMG SB | .163 | .113 | .066--.440 | 2/40 |
| FMG pEBA(4) | .135 | .085 | .064--.373 | 4/40 |
| MLR | .364 | .314 | .068--.930 | 2/40 |

Pooled over models, multiplier rejection was .043/.037 under normal
complete/MCAR, .053/.046 under moderate VM, .058/.062 under moderate IG,
.085/.070 under severe VM, and .061/.072 under severe IG. The VM/IG distinction
therefore remains scientifically active even when the multiplier is broadly
calibrated.

## Numerical health

The run took 446 seconds on four workers with observed 3.68x speedup. Nine
FIML fits failed to converge. All 19,991 converged fits produced finite
effective-multiplier p-values. FMG returned for 19,581 fits and MLR was finite
for 19,949.

Twenty-five finite multiplier calls had a numerical tangent rank below the
model's prespecified rank, hence a flip df above the nominal model df. Sixteen
of those 25 rejected. They were concentrated in the two-factor CFA under VM
plus MCAR. Restricting to the 19,966 nominal-rank calls changed the mean cell
rejection from .0584 to .0577 and left 33/40 cells in [.025,.075]. The anomaly
does not drive the positive calibration result, but it is too rejection-prone
to classify as ordinary success.

## Next gate

Add prespecified sparse and diffuse population misspecifications to the same
40-cell generator/observation panel. Report raw and size-adjusted power, with
fit, method-finite, and nominal-tangent denominators separated. A later
1,000--2,000-replication null confirmation is useful for publication precision,
not for deciding whether to proceed.
