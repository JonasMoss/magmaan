# Compute budget (planning estimate)

Bottom-up CPU-core-hour estimate for one 6-month period, anchored on the **one
grid we have actually timed**: experiment 61's complete-data invariance
permutation grid. The four flagship studies carry the number; the rest of the
program sits in aggregate lines. Order-of-magnitude planning figures, to be
refined by the paper leads, not promises.

## Anchor (measured)

| Quantity | Value | Source |
| --- | --- | --- |
| exp 61 complete full grid | 963 cells x 1000 reps x B=199 | measured |
| Per-rep cost | ~1 s (p=8) to ~5.5 s (p=16), B=199 | measured |
| Complete grid total | **~700 core-hours** | measured extrapolation |
| FIML multiplier | **10-100x** (iterative refit per permutation) | estimate, measure before submitting |

A "unit" complete-data grid is ~0.7 kCPU-h. A FIML or resampling-heavy study is
~7-70 kCPU-h. Ordinal/polychoric studies sit between (slower fits, similar grids).
Multiply raw figures by ~1.2 for the calibration pre-passes, sensitivity slices,
and revision reruns every real study needs.

## Budget

| Line | kCPU-h | Notes |
| --- | --- | --- |
| **Flagship 1** - robust SE under misspecification (`estimated-weight-se`) | 20 | coverage grids x estimators x misspec x n; MI extension |
| **Flagship 2** - FIML invariance + permutation (`fiml-fmg`) | 45 | heaviest; FIML x resampling x missingness x non-normality |
| **Flagship 3** - ordinal robust reference laws (`ordinal-fmg`) | 20 | slow polychoric/WLSMV fits, large categorical grid |
| **Flagship 4** - non-iterative CFA inference (`guttman-inference`) | 10 | large coverage/size grids, convergence-free |
| Additional active studies (6, aggregate) | 25 | closed-form-omega, composite-ml, snlls-continuous, ordinal-snlls, second-order-reliability-gaps, h-polychorics-sem |
| Methods-dev / engine validation (55 experiments, ongoing) | 15 | shared parity and calibration runs |
| **Subtotal (raw)** | **135** | |
| **x1.2 iteration/revision factor** | **~160** | |

## Headline ask

Request in the **120-150 kCPU-h** band for the 1 Oct 2026 - 31 Mar 2027 period.
It sits above the measured anchor by a wide margin (so it is clearly usable),
is backed by a concrete per-flagship breakdown, and is modest enough to clear the
committee on a first application. Grow it in the April 2027 round once there is a
Saga usage history, or top up mid-period (extra allocations need only a short
justification and can be submitted any time).

## Storage

Results are CSV, fixtures are JSON. A few hundred GB of NIRD project storage. No
large primary datasets.

## Lock before submitting

- [ ] PI + Norwegian-institution affiliation (Sigma2 eligibility).
- [ ] Funding source (mandatory MAS field).
- [ ] Final headline kCPU-h (pick within 120-150).
- [ ] Confirm the FIML multiplier with a small measured Saga (or local) trial.
      That single number dominates the budget; measuring it turns 10-100x into a
      figure and hardens the whole justification.
