# robust_score advisory simulation

Advisory Monte-Carlo sanity for the robust (generalized / Satorra-Bentler-scaled)
score & modification-index tests in `inference::frontier`. **Not** part of the
default test suite (`ctest`) — it is stochastic and slow. lavaan does not
implement this statistic (it falls back to the ordinary one), so there is no
package oracle; this validates the two things an exact number-match cannot.

## What it checks

Population: an exact 1-factor model over 5 indicators, so the residual
covariance `x1 ~~ x2` is **truly zero**. Data are heavy-tailed multivariate-t
(`--df`, smaller ⇒ more excess kurtosis) with covariance ≈ the model-implied one.

1. **Calibration.** Over reps, the rejection rate (at `--alpha`) of the
   true-null residual covariance:
   - ordinary **NT** modification index → expected to **over-reject** (leptokurtosis
     inflates the score variance the NT χ² ignores);
   - **robust** modification index → expected **≈ alpha**.
   This is the substantive "the correction is needed and works" demonstration.

2. **Trinity equivalence.** On the same data the df-1 statistics
   - robust **score** test of freeing `x1 ~~ x2` (`modification_indices_robust`),
   - robust **Wald** `z²` of `x1 ~~ x2` (refit full model + `robust_se`),
   - Satorra-Bentler **scaled LRT** difference (`lr_test_satorra_bentler2001`)
   all estimate the same thing; the mean `score/Wald` and `score/LRT` ratios → 1
   as N grows. This anchors the new statistic to magmaan's lavaan-validated
   `robust_se` and scaled-LRT machinery.

## Run

```sh
just quick      # ~120 reps, n=1500
just all        # ~800 reps, n=4000
```

`just lib` (re)builds the optimized `build/opt/libmagmaan.a` the harness links
against. Flags: `--n`, `--reps`, `--df`, `--alpha`, `--seed`.
