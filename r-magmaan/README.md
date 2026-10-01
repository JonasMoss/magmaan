# magmaan: the ordinary-user R package

One call estimates a structural equation model and computes inference under a
single policy that magmaan chooses and justifies. The model is written in
lavaan syntax or supplied as a model specification. The options use lavaan's
names where the concept is the same.

```r
library(magmaan)
fit <- magmaan("visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6",
               lavaan::HolzingerSwineford1939)
summary(fit)
coef(fit)            # free estimates, matching vcov(fit)
coef(summary(fit))   # the parameter table: estimates, robust SEs, z, p, intervals
```

Saved specifications from `magmaanlab::model_spec()` must carry lavaan syntax.
Identification, mean-structure and equality choices belong in that constructor.
Ordered variables, parameterization and grouping are inherited from the
specification; explicit conflicting options error. EQS remains in `magmaanlab`.

This package is pure R. Every computation happens in
[`magmaanlab`](../r-package/), the methods-development package over the
magmaan C++ core, which also offers every alternative convention.
`as_lab_fit(fit)` returns the underlying magmaanlab fit.

Status: for single-level complete-data ML, `magmaan()` computes the full
inference policy: the observed-information sandwich covariance (standard
errors, Wald tests, intervals, defined parameters) and the global score and
likelihood-ratio tests, each calibrated with SB and PEBA4. The other estimators
fit, but their inference components report `unsupported_model`, and `vcov()`
and `confint()` raise a `magmaan_inference_unavailable` condition. The design is
[`project/design/r-interface-vision.md`](../project/design/r-interface-vision.md).
Models with fixed observed covariates also report unsupported policy inference;
the fitting default `fixed.x = TRUE` does not imply inference support for those
models. A joint random-X model with `fixed.x = FALSE` is a different model and
must be selected explicitly.

Install `magmaanlab` first (`just r-dev` or `just r-install`), then this
package (`just r-magmaan`); `just r-magmaan-test` runs the tests.

This simulation prerelease pairs `magmaan` 0.1.0 with `magmaanlab` 0.1.0.
Development is unfinished. Remaining bugs need fixing, and coverage of the main
estimators and inferential procedures needs completing and validating before
a finished release. Version 0.1.0 is a snapshot for supported simulations;
the inference limits above still apply. See the
[remaining work](../project/backlog/todo.md#r-simulation-prerelease).

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
- `summary(fit)$tests` is `NULL` when no global test is available, otherwise a
  table of `test`, `statistic`, `df`, `p.sb`, `p.peba4`, and `sb.scale`.
  Nested comparisons have the same columns and an `unavailable` attribute.

Record provenance once per simulation run:

```r
versions <- vapply(c("magmaan", "magmaanlab"),
                   function(p) as.character(packageVersion(p)), character(1))
# Store versions, the release commit, and RNG settings alongside results.
```

Fits may be saved with `saveRDS()` and reused under the same matched package
versions. Their internal lab fields are a development interface; cross-version
saved-fit compatibility is not promised.
