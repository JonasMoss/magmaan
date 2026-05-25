# Maydeu-Olivares (2017) Replication

Reproduces the simulation in Maydeu-Olivares (2017),
*Maximum Likelihood Estimation of Structural Equation Models for Continuous
Data: Standard Errors and Goodness of Fit*, against magmaan's own ML
inference machinery.

The paper crosses seven SE methods and five χ² statistics over a two-factor
CFA fit as one-factor, varying nonnormality (induced by 5-category
thresholding of MVN data), sample size, and degree of misspecification. The
headline conclusion is that **MLMV** (Asparouhov–Muthén mean-and-variance-
adjusted χ²) gives the best Type I error and **MLR** (sandwich with observed
information) gives the most accurate SEs under nonnormality.

## Method coverage

| Mplus name | What it is | magmaan equivalent |
|---|---|---|
| ML observed | Hessian-based SEs | `inference_information_observed_analytic` |
| ML expected | Fisher-based SEs | `inference_information_expected` |
| MLF | Cross-products SEs | `inference_information_cross_products` (added for this experiment) |
| MLM | Satorra–Bentler mean-adjusted χ² | `robust_satorra_bentler` |
| MLMV | Mean+variance-adjusted χ² | `robust_mean_var_adjusted` |
| MLR | Yuan–Bentler sandwich (observed outer) | `robust_se_raw_fit(bread="observed")` |
| MLR expected | Sandwich with expected outer | `robust_se_raw_fit(bread="expected")` |

The data is multivariate normal sampled from a two-factor CFA covariance,
then discretized at thresholds chosen to make the discrete-variable
moments hit one of three (skewness, excess-kurtosis) targets:

- `(0, 0)` — symmetric, light tails (≈ normal after discretization)
- `(0, 2)` — symmetric, heavy tails
- `(-2, 3.18)` — left-skewed, heavy tails

The analysis model is **unidimensional** (one factor). Under population
factor correlation ρ = 1.0 this is correctly specified; ρ = 0.8 / 0.4 give
"minor" and "large" misspecification.

## Run

Install the R package first if needed:

```sh
just r-install
```

Quick smoke (default `--reps 10`, runs the full 54-cell grid):

```sh
Rscript experiments/07-maydeu-olivares-2017/run_experiment.R
```

Tiny smoke (2 reps, all cells):

```sh
Rscript experiments/07-maydeu-olivares-2017/run_experiment.R --reps 2
```

Publication-grade run (slow):

```sh
Rscript experiments/07-maydeu-olivares-2017/run_experiment.R --reps 1000
```

Filter cells:

```sh
Rscript experiments/07-maydeu-olivares-2017/run_experiment.R \
  --reps 200 --cells 'n_items=16'
```

Lavaan parity check (one extra fit per cell, magmaan vs lavaan side-by-side):

```sh
Rscript experiments/07-maydeu-olivares-2017/run_experiment.R \
  --reps 5 --lavaan-parity
```

Render the report:

```sh
(cd experiments/07-maydeu-olivares-2017 && quarto render report.qmd)
```

## Outputs

Generated files live under `results/` and are gitignored:

- `metadata.csv` — session info, command-line arguments, design summary.
- `cells.csv` — the 54-cell design grid: `(n_items, rho, skew, kurt, N)`.
- `thresholds.csv` — the threshold sets chosen per `(skew, kurt)` target,
  plus the achieved marginal moments.
- `fits.csv` — one row per `(cell, rep)`: ML loading point estimates, the
  seven SE vectors (in long form), the five χ² statistics, sample RMSEA,
  SRMR, population RMSEA, convergence flags.
- `summary.csv` — aggregated tables: SE relative bias by `(cell, method)`,
  empirical rejection rate at 5% by `(cell, statistic)`.
- `lavaan_parity.csv` — only when `--lavaan-parity` is passed; one row per
  cell with magmaan vs lavaan side-by-side outputs.

## Design grid

54 cells = 2 (n_items) × 3 (ρ) × 3 (distribution) × 3 (N). With the default
`--reps 10` that is 540 fits in total — under a minute on a laptop.

## Caveats

- Single-threshold strategy: thresholds are chosen by numerical
  moment-matching on the standard-normal quantile; if the search fails for
  a given (skew, kurt) target the row is logged in `thresholds.csv`. The
  paper uses the closed-form expressions of Maydeu-Olivares, Coffman &
  Hartmann (2007); ours is asymptotically equivalent.
- No Mplus oracle: the comparison is against the paper's published tables,
  not a re-run with Mplus.
- No Monte-Carlo error bars on the rejection-rate / bias estimates
  (consistent with the `feedback-mc-error` rule for magmaan papers).
