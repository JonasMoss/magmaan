# Robust ordinal SEM paper plan

Working title:

> Joint robust polychoric moments for ordinal SEM

Short-paper constraint: target 8--10 pages. Keep theory to definitions,
implementation contracts, and simulation evidence. Move derivations and
large grids to repo notes/supplement if needed.

## Main claim

Welz, Mair, and Alfons give a robust bivariate polychoric estimator. For SEM,
the missing step is a coherent ordinal moment builder: one shared threshold
vector per observed variable, one robust polychoric per pair, and a matching
casewise influence/Gamma object for DWLS/WLS weights and robust reporting.

The paper-level contribution is therefore:

1. shared-threshold joint robust ordinal moment estimation;
2. a small h-score family containing ML, WMA hard cap, and smooth capped
   h-polychorics;
3. Huberized Pearson-residual moments as an interpretable comparator;
4. simulation and benchmark evidence for robustness, stability, and SEM-facing
   downstream behavior.

Do not frame the paper as a new bivariate robust polychoric estimator. The
bivariate pieces are validation and diagnostics; the SEM-grade object is the
joint threshold-plus-correlation moment vector.

Reference anchors:

- Welz, Mair, and Alfons (2026), local copy:
  `resources/papers/robust_ordinal/welz_etal_2025_robust_polychoric_correlation.pdf`.
- VITA/covsim for copula stress tests: Grønneberg, Foldnes, and Marcoulides
  (2022), Journal of Statistical Software, doi: `10.18637/jss.v102.i03`.

## Estimators to report

- ML/two-step ordinal moments: lavaan-compatible baseline.
- WMA hard cap: `h(t) = min(t, k)`, using the Welz et al. tuning target for
  comparability.
- Smooth h cap: same overcount-cell story, but differentiable through the
  transition. This is the numerical-stability candidate.
- Huberized Pearson residual: minimize or solve clipped table residual
  equations; use as the most readable comparator for SEM readers.

Optional only if plots demand it:

- DPD: implemented and useful as a different robustness mechanism, but may
  make the short paper sprawl.
- Pair-local joint WMA: report only as a validation check against robcat/Welz,
  not as the SEM method.

Scope note: WMA is a *polychoric* (cell-overcount) recipe and does not extend to
polyserial (continuous-ordinal) pairs; the mixed/polyserial robust extension
forces a recipe-choice question (uniform DPD vs mixing WMA-cells with
clipped-conditional polyserials vs keeping this paper all-ordinal). That design
discussion and its resume checklist are in
[robust_mixed_recipe_taxonomy.md](robust_mixed_recipe_taxonomy.md). For now the
robust SEM paper is cleanest scoped all-ordinal, where no mixing question arises.

## Simulation protocol

All designs should emit one tidy results table with columns:

```text
design, rep, n, contamination, method, tuning, converged, elapsed_ms,
iterations, min_eigen_r, r_repair, objective, pair, parameter,
truth, estimate, se, ci_low, ci_high
```

Add SEM-specific columns when fitting a CFA:

```text
loading, factor_cov, residual_var, chi2, df, cfi, tli, rmsea, srmr
```

### Design 1: Welz bivariate corner contamination

Purpose: sanity-check the h family against the canonical paper story.

- Two five-category ordinal variables.
- Clean latent distribution: bivariate normal, rho = 0.5.
- Thresholds: Welz-style asymmetric or balanced five-category thresholds.
- Contamination: small bivariate normal blob inflating an opposite corner cell.
- Vary contamination: `0, .01, .05, .10, .15, .20, .30`.
- Metrics: bias, RMSE, convergence failures, estimated SE calibration.

This design should be tiny in the paper: one figure or one small table.

### Design 2: Welz five-variable robust polychoric matrix

Purpose: compare full robust correlation matrices, not isolated pairs.

- Five ordinal variables.
- Latent correlation matrix from Welz/Foldnes-Gronneberg:
  loadings approximately `(0.8, 0.7, 0.6, 0.5, 0.4)`.
- Five categories with thresholds around cumulative probabilities
  `.10, .30, .70, .90`.
- Contamination: independent Gumbel margins with location 0 and scale 3,
  discretized with the same thresholds.
- Metrics: pairwise bias/RMSE, minimum eigenvalue of `R`, repair frequency,
  Gamma conditioning, runtime.

This is the main "robust polychoric matrix" simulation.

### Design 3: Ordinal CFA downstream behavior

Purpose: show that robust moments matter for SEM estimates.

- One-factor CFA using the same five-variable loading pattern.
- Fit DWLS and optionally WLS from each moment builder.
- Contamination mechanisms:
  - diffuse careless responses as in Design 2;
  - straightlining/acquiescence pattern to stress shared thresholds.
- Metrics: loading bias/RMSE, SE calibration, robust ordinal test statistic
  stability, fit-measure distortion, convergence and runtime.

This should be the main SEM-facing result.

### Design 4: Copula distributional misspecification stress test

Purpose: show boundaries. This is not the core claim.

- Use VITA/covsim-style latent data with standard normal margins and nonnormal
  copula.
- Main text: Clayton copula only, matching Welz's distributional
  misspecification stress test.
- Optional supplement: Gumbel or t-copula.
- Metrics: point-estimate bias and failures. Be cautious with sandwich
  inference because the contamination theory is not the data-generating model.

Interpretation: robust h methods may help when nonnormality behaves like tail
contamination, but they are not a universal copula SEM estimator.

### Design 5: computation benchmark

Purpose: justify smooth h as more than aesthetic.

Grid over:

- method: ML, hard cap, smooth cap, Huber residual;
- `n`: maybe `250, 500, 1000, 2500`;
- categories: `3, 5, 7`;
- contamination: `0, .05, .15`;
- correlation strength: moderate and high.

Report:

- median elapsed time;
- iterations/function evaluations if available;
- convergence failure rate;
- nonfinite Gamma rate;
- `R` positive-definiteness failures/repairs;
- worst or median condition number of Gamma/WLS.

## Manuscript skeleton

1. Introduction: ordinal SEM uses polychoric moments; ML polychorics are
   fragile; bivariate robust estimators do not by themselves define a coherent
   SEM moment vector.
2. Robust ordinal moments: define cells, h-score, shared-threshold joint
   objective, Huber residual comparator, Gamma/influence contract.
3. Implementation: moment order, DWLS/WLS weights, positive-definiteness
   diagnostics/repair, tuning defaults.
4. Simulations: Designs 1--4, with Design 3 carrying the SEM story.
5. Benchmarks: hard cap versus smooth h stability and cost.
6. Discussion: use robust moments as methods-developer inputs; limitations
   under pure distributional misspecification; direct robust pairwise SEM as
   future work.

## Next concrete tasks

1. Use `docs/research/sims/r/robust_ordinal_sem_paper.R` to run enough
   replications for Design 1 and Design 2; verify ML/hard-cap results reproduce
   the known Welz qualitative pattern.
2. Add first plotting/summary helpers for bias, RMSE, convergence, runtime,
   and minimum eigenvalue of `R`.
3. Add the CFA downstream fit only after the moment-level plots are sane.
4. Decide from early plots whether DPD earns a main-text line or stays in the
   code/supplement.
