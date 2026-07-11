# Compute budget (planning estimate)

Bottom-up CPU-core-hour estimate for one 6-month period, anchored on the **one
grid we have actually timed**: experiment 61's complete-data invariance
permutation grid. Everything else is scaled from that anchor and should be
refined by the paper leads. These are order-of-magnitude planning figures, not
promises.

## Anchor (measured)

| Quantity | Value | Source |
| --- | --- | --- |
| exp 61 complete full grid | 963 cells x 1000 reps x B=199 | measured |
| Per-rep cost | ~1 s (p=8) to ~5.5 s (p=16), B=199 | measured on this machine |
| Complete grid total | **~700 core-hours** | measured extrapolation |
| FIML multiplier | **10-100x** (iterative refit per permutation) | estimate |
| Modal reference cost | ~$33 CPU for the same 700 core-hr | measured rate |

So a "unit study" (one complete-data grid) is ~0.7 kCPU-h. A FIML/permutation or
bootstrap study is ~7-70 kCPU-h. Ordinal/polychoric studies sit in between
(slower fits, similar grids). Multiply raw estimates by ~1.5 for the calibration
pre-passes, sensitivity slices, and revision reruns every real paper needs.

## Per-paper estimate (refine with leads)

| Paper (papers/<slug>) | Study character | Raw kCPU-h | Notes |
| --- | --- | --- | --- |
| fiml-fmg | FIML robust MI, permutation extension | 40 | heaviest; FIML x resampling |
| exp 61 permutation-MI (this) | complete now + FIML later | 50 | complete ~3, FIML ~45 |
| h-polychorics-sem | polychoric SEM grids | 20 | slow categorical fits |
| ordinal-fmg | ordinal robust tests | 20 | slow categorical fits |
| ordinal-snlls | ordinal SNLLS | 12 | |
| estimated-weight-se | misspec-robust SE sandwich | 8 | |
| guttman-inference | closed-form CFA inference | 8 | |
| snlls-continuous | SNLLS robustness | 8 | |
| closed-form-omega | reliability inference | 5 | |
| composite-ml | composite / Henseler-Ogasawara | 5 | |
| second-order-reliability-gaps | reliability gaps | 5 | |
| methods-dev / exploratory experiments | ongoing (55 experiments) | 30 | shared engine validation |
| **Subtotal (raw)** | | **~211** | |
| **x1.5 iteration/revision factor** | | **~315** | |

## Headline ask options

| Strategy | kCPU-h | Rationale |
| --- | --- | --- |
| Conservative first ask | **50-100** | modest, easy to justify, scale next period |
| Portfolio-matched | **150-250** | matches the bottom-up total with headroom |
| Full pipeline + growth | **300+** | the raw x1.5 figure; needs strong justification |

Recommendation for a **first** application: request in the **100-150 kCPU-h**
band. It is well above the measured anchor (so it is clearly usable), backed by a
concrete per-paper breakdown, and modest enough to clear the committee. Grow the
ask in the April 2027 round once there is a Saga usage history to point at.

## Storage

Results are CSV; fixtures are JSON. Estimate a few hundred GB of NIRD project
storage. No large primary datasets.

## What to lock before submitting

- [ ] PI + Norwegian-institution affiliation (Sigma2 eligibility).
- [ ] Final headline kCPU-h (pick a band above).
- [ ] Trim/confirm the per-paper rows with each lead; drop papers that will be
      done before 1 Oct and add any not yet listed.
- [ ] Confirm the FIML multiplier with a small measured Saga trial if time
      allows (turns the 10-100x guess into a number).
