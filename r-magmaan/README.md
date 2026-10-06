# magmaan: the ordinary-user R package

One call estimates a structural equation model and computes inference under a
single policy that magmaan chooses and justifies. The model is written in
lavaan syntax; structural options use lavaan's names where the concept is the
same.

```r
library(magmaan)
fit <- magmaan("visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6",
               lavaan::HolzingerSwineford1939)
summary(fit)
coef(fit)            # free estimates, matching vcov(fit)
coef(summary(fit))   # the parameter table: estimates, robust SEs, z, p, intervals
```

Grouped, ordinal and repeatedly fitted models are constructed once with
`magmaan_model()`, whose `prototype` declares groups and categories:

```r
skeleton <- data.frame(y1 = ordered(character(), levels = 1:5),
                       y2 = ordered(character(), levels = 1:5),
                       y3 = ordered(character(), levels = 1:5),
                       site = factor(character(), levels = c("A", "B")))
m <- magmaan_model("f =~ y1 + y2 + y3", prototype = skeleton,
                   ordered = c("y1", "y2", "y3"), group = "site")
fits <- lapply(datasets, function(d) magmaan(m, d, estimator = "DWLS"))
```

`magmaan(model, data, estimator, covariance, inference, options)` takes the
choices that define the estimate. `covariance` is `"unrestricted"`, `"psd"`
or `barrier(lambda)` (experimental); `options` holds optimization details such
as `start` and the `preset = "lavaan-0.7.2"` fitting conventions. Every model
has a mean structure and treats observed covariates as random. Saved
specifications from `magmaanlab::model_spec()` must carry lavaan syntax; EQS
remains in `magmaanlab`.

This package is pure R. Every computation happens in
[`magmaanlab`](../r-package/), the methods-development package over the
magmaan C++ core, which also offers every alternative convention.
`as_lab_fit(fit)` returns the underlying magmaanlab fit.

Reporting can select a named lavaan inference bundle from the same estimates:

```r
vcov(fit, convention = "MLM")
confint(fit, convention = "MLR")
summary(fit, convention = "MLR")
anova(restricted_fit, fit, convention = "MLM")
```

The default is `convention = "magmaan"`. Complete-data ML accepts `"ML"`,
`"MLM"` and `"MLR"`; ordinal DWLS accepts `"DWLS"` and `"WLSMV"`, and
ordinal ULS accepts `"ULS"` and `"ULSMV"`. Ordinal WLS accepts `"WLS"`.
Bundles compute on demand without refitting; `infer(fit, convention = "MLR")`
caches an additional bundle while preserving the default policy. FIML and
ordinal nested compatibility currently report unavailable inference. Comparisons
with lavaan need matching model/fitting settings, including ordinary models'
mean structure and `fixed.x = FALSE`. See the
[capability inventory](../project/validation/capabilities.md) for checked slices.

Status: single-level ML (complete data) and FIML (complete or incomplete
continuous data) compute the full inference policy: observed-information casewise-score sandwich covariance (standard
errors, Wald tests, intervals, defined parameters), global score and LR tests,
and nested score and LR tests, each calibrated with SB and PEBA4. FIML score
uses observed sensitivity and an expected metric; nested LR uses empirical
scores at the larger fit. FIML calibration remains limited; see the capability
inventory. All-ordinal DWLS computes the estimated-weight IJ covariance, a
global fit-function test with the exact spectrum (All) tail, and a nested
fit-function difference with SB and PEBA4; nested score is unavailable. ML2S, GLS,
ULS and WLS fit with typed unavailable policy components. `vcov()` and
`confint()` raise `magmaan_inference_unavailable` when covariance is unavailable.
The design is
[`project/design/r-interface-vision.md`](../project/design/r-interface-vision.md).
Barrier fits report every inference component unavailable with reason
`penalized`.

Install `magmaanlab` first (`just r-dev` or `just r-install`), then this
package (`just r-magmaan`); `just r-magmaan-test` runs the tests.

The 0.1.0 simulation prerelease pairs `magmaan` 0.1.0 with `magmaanlab` 0.1.0;
the development version carries the 0.2.0 API described here (see NEWS).
Development is unfinished. Remaining bugs need fixing, and coverage of the main
estimators and inferential procedures needs completing and validating before
a finished release. Version 0.1.0 is a snapshot for supported simulations;
the inference limits above still apply. See the
[remaining work](../project/backlog/todo.md#release-plan).

Install the two source archives from the same release snapshot, compiled package
first. The ordinary package declares the minimum compatible lab version;
pin both versions for a simulation run.

For repeatable extraction:

- `coef(fit)` is the named free-estimate vector in covariance-matrix order.
  Equality constraints can give repeated labels; use indices to distinguish
  entries. `confint(fit, parm = ...)` accepts names or positive integer indices.
- `coef(summary(fit))` contains `lhs`, `op`, `rhs`, `label`, logical `free`,
  `est`, `se`, `z`, `pvalue`, `ci.lower`, and `ci.upper`, plus `group` for
  multiple groups. Defined estimates are retained with `inference = FALSE`
  and when inference is unsupported; unavailable uncertainty is `NA`.
- `fit$rows` contains `group`, `rows`, `used`, and `deleted` per group.
  `nobs(fit)` sums used observations.
- `as_lab_fit(fit)$converged` is `TRUE`, `FALSE`, or `NA` (unchecked).
  Count successes with `isTRUE()`; estimation and inference availability are
  separate outcomes.
- `fit$inference` is `NULL` before inference. Otherwise its `$status` table
  has `component`, `available`, `reason`, and `detail` for `covariance`,
  `global_score`, and `global_lr`. Unavailable covariance raises a
  `magmaan_inference_unavailable` condition from `vcov()`/`confint()`, with
  machine-readable `component` and `reason`.
- `summary(fit)$tests` is `NULL` before inference is computed, otherwise a
  table of `test`, `statistic`, `df`, `reference`, `pvalue`, `recommended`, and `reason`.
  Nested comparisons have the same base columns and an `unavailable` attribute.
  Unavailable tests keep one row with a typed reason. Test codes are `score`,
  `lr`, `fit_function`, `fit_function_difference`; print methods show readable
  labels. For simulation comparisons, use `summary(fit, references = c("sb", "peba4", "all"))` or the same argument
  to `anova()`; select p-values by `test` and `reference`.

Record provenance once per simulation run:

```r
versions <- vapply(c("magmaan", "magmaanlab"),
                   function(p) as.character(packageVersion(p)), character(1))
# Store versions, the release commit, and RNG settings alongside results.
```

Fits may be saved with `saveRDS()` and reused under the same matched package
versions. Their internal lab fields are a development interface; cross-version
saved-fit compatibility is not promised.
