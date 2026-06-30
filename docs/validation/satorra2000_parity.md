# Satorra-2000 nested-test lavaan parity

**Status:** resolved 2026-05-17; missing-data lavaan convention added
2026-06-30. This was not a confirmed lavaan bug.

The apparent parity gap between magmaan's nested robust LRT helper
(`robust_nested_lrt()`, historically `nestedTest()`) and lavaan's
`lavTestLRT(method = "satorra.2000")` came from lavaan's `A.method` default,
not from the Satorra-2000 scaling formula itself.

For parameter-nested models, magmaan's restriction matrix matches lavaan's
`A.method = "exact"` path. magmaan's `T_scaled` is the mean-scaled statistic
`T / c`, so the matching lavaan call also sets `scaled.shifted = FALSE`.
lavaan's documented default is `A.method = "delta"`, which constructs the
restriction Jacobian from the column spaces of the two models' moment
Jacobians. That default is useful when models are nested only in the
covariance/moment sense, but it is not the same restriction Jacobian as the
explicit parameter equality in simple parameter-nested comparisons.

## Bottom line

Use this as the lavaan parity target for magmaan's default
`robust_nested_lrt(method = "restriction_map", convention = "magmaan")`
on complete-data parameter-nested equality constraints:

```r
lavTestLRT(fit_h1, fit_h0, method = "satorra.2000",
           A.method = "exact", scaled.shifted = FALSE)
```

Do not use lavaan's bare default as an oracle for parameter-nested equality
constraints:

```r
lavTestLRT(fit_h1, fit_h0, method = "satorra.2000")
```

That call uses `A.method = "delta"`. The difference can be tiny for weakly
binding restrictions and visible for strongly binding restrictions. It also
uses `scaled.shifted = TRUE`, so for `m > 1` it reports lavaan's
scaled-shifted statistic rather than the mean-scaled `T / c` statistic.

For a deliberate lavaan-default comparison, including the FIML/ML2S missing-data
paper parity gate, use magmaan's compatibility flag:

```r
nestedTest(fit_h1, fit_h0, method = "satorra.2000",
           convention = "lavaan")
```

The R wrapper then defaults the restriction map to `A.method = "delta"` and
returns the lavaan scaled-shifted p-value in `p_scaled_shifted`.

## Evidence

For three single-group Holzinger-Swineford CFA restrictions, the original
report compared magmaan to lavaan's default `A.method = "delta"`:

| restriction | lavaan `A.method = "delta"` scale | lavaan `A.method = "exact"` scale | magmaan scale |
|---|---:|---:|---:|
| `visual =~ x2 == x3` | 0.749899281 | 0.747094573 | 0.7471 |
| `speed =~ x8 == x9` | 0.839290405 | 0.865142793 | 0.8651 |
| `textual =~ x5 == x6` | 1.280963205 | 1.446237750 | 1.4462 |

The strongly binding `textual =~ x5 == x6` case produced the alarming 12.9%
gap only because it compared magmaan's exact parameter-restriction matrix with
lavaan's delta-method covariance-nesting matrix.

Reconstructing lavaan's formula in R confirms this diagnosis. With
`Delta = lavInspect(fit_h1, "delta")`, `V = lavInspect(fit_h1, "WLS.V")`,
`Gamma = lavInspect(fit_h1, "gamma")`, and
`P = t(Delta) %*% V %*% Delta`, the only switch needed to move between the two
answers is the restriction matrix `A`:

```r
calc_scale <- function(fit_h1, A) {
  D <- as.matrix(lavInspect(fit_h1, "delta"))
  V <- as.matrix(lavInspect(fit_h1, "WLS.V"))
  G <- as.matrix(lavInspect(fit_h1, "gamma"))
  P_inv <- solve(t(D) %*% V %*% D)
  C_inv <- MASS::ginv(A %*% P_inv %*% t(A))
  PAAPAAP <- P_inv %*% t(A) %*% C_inv %*% A %*% P_inv
  sum(diag(V %*% G %*% V %*% D %*% PAAPAAP %*% t(D))) / nrow(A)
}
```

For `textual =~ x5 == x6`, lavaan's two `A` matrices differ materially:

```r
A_delta <- lavaan:::lav_test_diff_a(fit_h1, fit_h0,
                                    method = "delta", reference = "H1")
A_exact <- lavaan:::lav_test_diff_a(fit_h1, fit_h0,
                                    method = "exact", reference = "H1")
```

`A_exact` has the expected two non-zero entries for the loading equality
(`L5 - L6`). `A_delta` is dense in several H1 free parameters because it is a
moment-Jacobian column-space construction. Plugging `A_delta` into equation 23
reproduces lavaan's default scale 1.280963205; plugging `A_exact` into the same
equation reproduces magmaan and lavaan's explicit exact scale 1.446237750.

## Relation to Satorra (2000)

Satorra's setup defines restrictions through a parameter-space constraint
function `a(delta) = a0` with full-row-rank Jacobian `A`. In the special case
where the restriction is equating parameters to specific values, the paper
spells out that `A` can be taken as a selector for the restricted parameter
components. The scaled restricted-test statistic uses

```text
U = V * Delta * P^-1 * A' * (A * P^-1 * A')^-1 * A * P^-1 * Delta' * V
c = tr(U * Gamma) / m
```

with the multi-sample trace written equivalently as a weighted group sum. This
is the formula in lavaan's `lav_test_diff_satorra2000` and in magmaan's
`compute_satorra2000`; the parity issue was the choice of `A`, not the trace
formula.

Primary sources checked:

- Satorra, A. (1999/2000), *Scaled and adjusted restricted tests in
  multi-sample analysis of moment structures*, especially equations 22-23 and
  the parameter-restriction special case.
- lavaan `lavTestLRT` documentation: `A.method = "exact"` requires
  parameter-nested models; `A.method = "delta"` only requires covariance-matrix
  nesting. The same documentation also notes that `scaled.shifted` controls the
  Satorra-2000 scaled-shifted statistic.
- lavaan `R/lav_test_diff.R`: `lav_test_diff_satorra2000` uses the same
  equation-23 trace form after computing `A`.

## Reproduction

```r
suppressMessages(library(lavaan))

df <- as.data.frame(HolzingerSwineford1939[paste0("x", 1:9)])
h1 <- "visual=~x1+x2+x3
       textual=~x4+L5*x5+L6*x6
       speed=~x7+x8+x9"
h0 <- "visual=~x1+x2+x3
       textual=~x4+Lt*x5+Lt*x6
       speed=~x7+x8+x9"

l1 <- cfa(h1, data = df, estimator = "MLM")
l0 <- cfa(h0, data = df, estimator = "MLM")

lr_delta <- lavTestLRT(l1, l0, method = "satorra.2000",
                       scaled.shifted = FALSE)
lr_exact <- lavTestLRT(l1, l0, method = "satorra.2000",
                       A.method = "exact", scaled.shifted = FALSE)
Tdiff <- fitMeasures(l0, "chisq") - fitMeasures(l1, "chisq")

c(delta = Tdiff / lr_delta[2, "Chisq diff"],
  exact = Tdiff / lr_exact[2, "Chisq diff"])
#    delta    exact
# 1.280963 1.446238
```

## Project decision

magmaan should continue using the exact parameter-nesting restriction matrix by
default for `robust_nested_lrt(method = "restriction_map")`, because that is
the natural contract for two lavaanified models differing by shared labels or
explicit linear `==` constraints. `A.method = "delta"` is available as an
explicit compatibility mode for lavaan-style covariance/moment nesting checks.
The result reports the mean-scaled `T / c`, mean/variance-adjusted,
scaled/shifted, and mixture distributional summaries from the same
Satorra-2000 eigenvalues.

For lavaan-facing examples and parity checks, compare against
`lavTestLRT(..., method = "satorra.2000", A.method = "exact",
scaled.shifted = FALSE)`. It is still reasonable to mention lavaan's
delta-method option in examples, but only as a documented alternative for
covariance-nested comparisons, not as the default magmaan oracle.

## Mean structure (2026-06-17)

The complete-data ML restriction map now handles mean-structured models. The
per-group moment vector becomes the augmented `[μ_g; vech(Σ_g)]` (mean rows on
top): `lr_test_satorra2000_from_data` stacks `dmu_dtheta` over `dsigma_dtheta`
for the per-group `Pi_alpha` and the delta restriction, and
`compute_satorra2000` carries it through `SatorraGroup.mu_dim` (the μ-block of
`V` is `Σ⁻¹`; the empirical meat picks up the full μ×σ cross-block). Previously
any `meanstructure = TRUE` complete-data pair errored with a rank-deficient
pooled `P`, because the moment Jacobian omitted the mean block and free
intercepts/latent means produced zero columns.

`r-package/examples/nested_test_satorra2000.R` gates this against lavaan with
two `meanstructure = TRUE` Holzinger-Swineford pairs: configural vs metric
(a covariance restriction in the augmented moment space) and intercept
invariance (free vs cross-group-tied indicator intercepts, latent means fixed
at 0 in both groups — a genuine restriction on the mean block). Both reproduce
lavaan's `lavTestLRT(method = "satorra.2000", A.method = "exact",
scaled.shifted = FALSE)` to 1e-3 (intercept invariance: `Δχ²_scaled = 80.5597`,
`df = 9`).

Caveat (separate, pre-existing): magmaan's convenience keyword
`group_equal = "intercepts"` ties the indicator intercepts but does *not*
auto-free the non-reference-group latent means the way lavaan does, so a full
metric→scalar comparison built that way is over-restricted (df too high) and is
not a pure parameter-nesting the exact map accepts. Specify the scalar step with
explicit per-group intercept labels plus `factor ~ c(0, NA)*1` latent-mean rows
when an exact metric→scalar test is needed; the fits then match lavaan exactly.
This is a `spec::build` lavaanify gap, not a nested-test gap.

## Missing-data FIML/ML2S lavaan convention (2026-06-30)

The FIML empirical missing-data route now exposes two finite-sample moment
conventions.

`convention = "magmaan"` is the independent saturated-EM eta-space convention:

```text
A1 = K1' I_obs(theta_H1) K1
B1 = (V Delta K1)' Gamma (V Delta K1)
V = SaturatedMoments::H
Gamma = SaturatedMoments::acov
```

`convention = "lavaan"` mirrors the public
`lavTestLRT(method = "satorra.2000")` construction. The parameter-space bread is
the per-observation observed FIML information from the H1 model. The `WLS.V`
side is lavaan's H1 **expected** information in the saturated
`[mean; vech(cov)]` metric: for FIML missing data it averages, over observed
missingness patterns, the embedded inverse saturated H1 covariance and its
normal-theory covariance block. The `Gamma` side is the model-based raw-moment
Gamma centered at the H1 model-implied mean/covariance and pairwise skipped
over missing entries. ML2S uses the selected Stage-2 `WLS.V` with the same
lavaan model-based raw-moment Gamma.

The earlier 2026-06-28 probe localized the original strict-rung failure to the
Satorra-2000 ingredients, not to `T_diff`, `df`, model specification, partables,
or constraints. Replacing only the parameter bread with observed information
moved the strict replicate in the right direction but did not fully match
lavaan. The remaining gap was lavaan's FIML `WLS.V` convention: complete data
reduces to `SaturatedMoments::H / n_g`, while missing data uses the
pattern-expected H1 information above, not the saturated EM observed Hessian.

The paper parity gate (`papers/fiml-fmg/analysis/parity_lavaan.R`) now checks
complete, MCAR, and MAR-linear cells for FIML and ML2S against lavaan
0.7.1.2691. With `convention = "lavaan"` it reports 90/90 passing checks,
including every nested scaled-shifted statistic and p-value. The `gamma = "NT"`
sanity path still keeps the saturated eta-space bread so every difference
eigenvalue collapses to one under the normal-theory reference.

The strict `A.method = "exact"` discrepancy remains a row-space convention, not
the moment-convention bug. lavaan's internal exact helper for that invariance
step carries earlier equality rows into the exact restriction body before
projection, whereas magmaan's exact route uses the clean `K_H1` versus `K_H0`
complement. lavaan's public default is `A.method = "delta", scaled.shifted =
TRUE`; use magmaan `convention = "lavaan"` when that is the oracle.
