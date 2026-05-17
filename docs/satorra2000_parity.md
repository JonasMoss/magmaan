# Satorra-2000 nested-test lavaan parity

**Status:** resolved 2026-05-17. This was not a confirmed lavaan bug.

The apparent parity gap between magmaan's `nestedTest()` and lavaan's
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

Use this as the lavaan parity target for magmaan's current `nestedTest()`:

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

magmaan should continue using the exact parameter-nesting restriction matrix for
`nestedTest()`, because that is the natural contract for two lavaanified models
differing by shared labels or explicit linear `==` constraints. Its reported
`T_scaled` is the mean-scaled `T / c` statistic; the separate adjusted and
mixture p-values cover the other Satorra-2000 distributional summaries.

For lavaan-facing examples and parity checks, compare against
`lavTestLRT(..., method = "satorra.2000", A.method = "exact",
scaled.shifted = FALSE)`. It is still reasonable to mention lavaan's
delta-method default in examples, but only as a documented alternative for
covariance-nested comparisons, not as a magmaan oracle.
