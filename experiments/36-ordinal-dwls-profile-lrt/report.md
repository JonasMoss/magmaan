# Experiment 36 — Nested categorical-DWLS difference test under misspecification

## Question

When two nested ordinal CFA models are compared with the usual scaled difference
test (DWLS/WLSMV, the test `lavaan::lavTestLRT` reports), does that test stay
calibrated if the *larger* model is itself misspecified — and if not, does the
estimated-weight reference law magmaan now computes restore calibration? This is
the difference-test analogue of experiment 35's estimated-weight standard error.

## Short answer

**No — the standard test becomes anti-conservative as the larger model's
misspecification grows, and the estimated-weight law fixes it.** On an exact
pseudo-null (the true restriction holds even though both models are wrong), the
standard fixed-weight scaled test rejects a true 3-df restriction at **10.5%**
for a nominal 5% in the strong design, climbing monotonically from 4.9% as
misspecification grows; the estimated-weight reference stays flat at **~4.2%**
across the whole range. The reason is direct: the average difference statistic
(≈ **2.63** at the strongest setting) equals the estimated-weight reference's
mean (population trace **2.63**), not the fixed-weight reference's (**2.13**),
and that gap widens with misspecification. Plain χ² is miscalibrated even under
correct specification, as expected for DWLS.

## Evidence

Exact pseudo-null: a four-variable binary CFA whose population correlations have
a symmetric four-cycle pattern, so the congeneric model's best fit is forced to
be tau-equivalent. The restriction (tau-equivalence, 3 df) is therefore *true*
at every misspecification level `eps`, while both models are wrong for `eps > 0`;
`eps = 0` is correct specification. Rejection rate of the true restriction at the
5% level, strong design (high loadings, n = 1500); a calibrated test reads 0.05.

| Misspecification `eps` | χ² (naive) | Fixed-weight (current) | Estimated-weight (this work) |
|--:|--:|--:|--:|
| 0.00 | 3.0% | 4.9% | 4.3% |
| 0.08 | 3.5% | 5.8% | 4.2% |
| 0.16 | 4.8% | 8.5% | 4.3% |
| 0.24 | 4.6% | 10.5% | 4.1% |

Full grid (both loading levels, n ∈ {500, 1500}, levels 1/5/10%, and the
population reference traces) is in `results/calibration.csv`. The decisive column
there is `mean_T` versus `pop_trace_full` and `pop_trace_fixed`: the empirical
mean of the statistic tracks the estimated-weight trace and diverges from the
fixed-weight trace as `eps` grows.

## Caveats

- One deliberately clean design: a symmetric four-cycle binary pseudo-null, so
  the restriction is exact by construction at every `eps`. It is a controlled
  demonstration, not a survey of ordinal models.
- Type-I (calibration) only; no power arm. Under this balanced design the
  estimated-weight correction is real but moderate; less balanced designs make
  the fixed-weight inflation larger.
- The estimated-weight reference is mildly conservative at the smaller sample
  size; it approaches nominal as n grows. Its value is removing the
  fixed-weight test's anti-conservative inflation, not perfect small-sample size.
- The comparison isolates the reference *law*: all three p-values are applied to
  the same difference statistic, so differences are purely the reference each
  test assumes.

## Reproduce

Build the optimized core once (`cmake --build --preset opt`), then:

```sh
just quick   # reps=300, fast smoke
just all     # reps=4000, tight
```

Override the simulator directly:
`build/ordinal_dwls_profile_lrt_sim --reps=… --seed=… --out=results/calibration.csv`.
Columns: `lambda, eps, n, reps_used, pop_trace_fixed, pop_trace_full, mean_T`,
then rejection rates `rej_{chi2,fixed,full}_{1,5,10}` (percent of the nominal
level). Results in this folder are from `reps = 2000`.
