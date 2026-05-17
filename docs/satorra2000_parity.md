# Satorra-2000 nested-test parity gap

**Status:** open — under investigation. Recorded 2026-05-17.

This document records a discrepancy between magmaan's Satorra-2000 scaled
nested-model difference test and lavaan's `lavTestLRT(method = "satorra.2000")`.
It is *not* a confirmed implementation bug in magmaan; the evidence below shows
magmaan faithfully implements one standard, well-defined form of the test. The
open question is which variant is canonical and whether magmaan should match
lavaan's specific construction. Treat the numbers here as a starting point for a
careful re-derivation, not as a settled conclusion.

## Summary

`nestedTest()` (C++ `robust::lr_test_satorra2000_from_data` →
`compute_satorra2000`) and lavaan's `lavTestLRT(method = "satorra.2000")` agree
on the *unscaled* difference statistic exactly, but disagree on the *scaled*
difference statistic — i.e. on the scaling factor `c` (equivalently, the
generalised eigenvalues `λ`). The disagreement grows with how strongly the
restriction binds:

| restriction (single-group HS 3-factor CFA) | binding? | magmaan `c` | lavaan s2000 `c` | gap |
|---|---|---|---|---|
| `visual =~ x2 == x3` (loadings ≈ equal in H1) | weak | 0.7471 | 0.7499 | 0.4% |
| `speed =~ x8 == x9` | moderate | 0.8651 | 0.8393 | 3.1% |
| `textual =~ x5 == x6` (loadings differ ≈ 0.19 in H1) | strong | 1.4462 | 1.2810 | 12.9% |
| 2-group metric invariance (m = 6, ≈ holds on HS) | weak | 1.0667 | 1.0604 | 0.6% |

The naive unscaled difference (`χ²_H0 − χ²_H1`) matches lavaan exactly in every
case, and the normal-theory (`gamma = "NT"`) path correctly collapses every
eigenvalue to 1. The discrepancy is confined to the empirical-Γ scaling factor.

Because the existing example `nested_test_satorra2000.R` exercises only the
2-group metric-invariance case — where metric invariance nearly holds on the HS
data, so the restriction barely binds — the gap there is 0.6% and the example's
lavaan cross-check passed. That masked the issue. The cross-check has since been
removed from that example (see "Examples" below).

## What magmaan computes

`compute_satorra2000` implements the reduced `m × m` generalised-eigenvalue form
of the Satorra-2000 difference test, with **every quantity evaluated at the H1
fit** `θ̂_H1`. For a nested pair H1 ⊃ H0 with restriction Jacobian `A` (m × r1,
the orthonormal basis of the H0 restriction in H1's parameter space):

```
P = Δᵀ V Δ                         (r1 × r1, expected information at θ̂_H1)
C = A P⁻¹ Aᵀ                       (m × m)
S = A P⁻¹ Δᵀ V Γ V Δ P⁻¹ Aᵀ        (m × m)
λ = generalised eigenvalues of (S, C);   c = mean(λ);   T_scaled = T_diff / c
```

where `Δ = ∂σ/∂θ` at `θ̂_H1`, `V = Γ_NT(Σ̂_H1)⁻¹` the normal-theory weight at the
H1 model-implied moments, and `Γ` the empirical fourth-moment ACOV of the
sample moments (n divisor). This is the standard reduced form of the
H1-anchored difference test: `λ` are exactly the non-zero eigenvalues of `Uᵈ Γ`
with `Uᵈ = V Δ P⁻¹ Aᵀ C⁻¹ A P⁻¹ Δᵀ V`, the H1-anchored difference of the two
models' residual-weight matrices. See `include/magmaan/robust/satorra2000.hpp`.

## Evidence that magmaan faithfully implements that form

The reduced form above was reconstructed **independently in R** from lavaan's
*own* `delta`, `WLS.V` and `gamma` matrices for the H1 fit, with no magmaan code
in the path:

```r
indep_scale <- function(fit_h1, free_i, free_j) {
  D <- as.matrix(lavInspect(fit_h1, "delta"))   # ∂σ/∂θ at θ̂_H1
  V <- as.matrix(lavInspect(fit_h1, "WLS.V"))   # normal-theory weight
  G <- as.matrix(lavInspect(fit_h1, "gamma"))   # empirical Γ
  A <- matrix(0, 1, ncol(D))
  A[1, free_i] <-  1 / sqrt(2)
  A[1, free_j] <- -1 / sqrt(2)
  Pinv <- solve(t(D) %*% V %*% D)
  C    <- A %*% Pinv %*% t(A)
  S    <- A %*% Pinv %*% (t(D) %*% V %*% G %*% V %*% D) %*% Pinv %*% t(A)
  as.numeric(S / C)
}
```

This reproduces magmaan's `nestedTest()` scaling factor to all printed digits
(0.7471, 0.8651, 1.4462 for the three single-group cases). So magmaan's C++ is a
faithful implementation of the reduced-form formula — the magmaan-vs-lavaan gap
is a difference in the *definition of the test*, not a coding error in
`compute_satorra2000`.

## What this is *not*

Ruled out as the cause of the gap:

- **The n vs n−1 divisor on Γ̂.** That is an O(1/N) ≈ 0.3% effect on HS
  (N = 301); the observed gap is ~13%. (The comment previously in
  `nested_test_satorra2000.R` blaming the divisor was a misdiagnosis and has
  been corrected.)
- **The `:=` defined-parameter row.** Removing it from H1 leaves the scaling
  factor unchanged.
- **Single- vs multi-group per se.** A single-group weak restriction
  (`visual x2 == x3`) matches lavaan to 0.4%; the gap tracks how strongly the
  restriction binds, not the group count.
- **The `auto.cov.lv.x` duplication bug.** That was a separate, real defect
  (now fixed); it does not touch the Satorra-2000 path.

## What lavaan does (partially characterised)

lavaan's `lavTestLRT(method = "satorra.2000")` dispatches to
`lav_test_diff_Satorra2000`. It is **not** the H1-anchored reduced form:

- A naive two-model difference `Uᵈ = U₀ − U₁`, with each `U` built from its own
  fit's `V` and `Δ` (so `U₀` uses `Σ̂_H0`, `U₁` uses `Σ̂_H1`), gave values
  inconsistent with both magmaan *and* lavaan (≈ 0.18 / 0.31 / 0.13) in a quick
  reconstruction — so either that is not lavaan's construction or the
  eigenvalue extraction in that reconstruction was wrong. To be pinned down.
- A *common-V* two-model difference `U₀ − U₁` (both with `V` from the same fit)
  collapses algebraically back to the H1-anchored reduced form — i.e. it equals
  magmaan. So lavaan's variant is neither of those simple forms.

For context, lavaan's three difference-test methods on the strongly-binding
`textual x5 == x6` case bracket magmaan's value:

| method | scaling factor `c` |
|---|---|
| lavaan `satorra.bentler.2001` | 1.2320 |
| lavaan `satorra.2000` | 1.2810 |
| lavaan `satorra.bentler.2010` | 1.3721 |
| magmaan `nestedTest` (H1-anchored reduced form) | 1.4462 |

magmaan's per-model Satorra-Bentler scaling factors are themselves correct:
`c_H1 = 1.05482`, `c_H0 = 1.06191` match lavaan, and the hand-computed
Satorra-Bentler-2001 difference scale `(df₁·c₁ − df₀·c₀)/(df₁ − df₀) = 1.2320`
reproduces lavaan's `satorra.bentler.2001` exactly. The gap is specific to the
*eigenvalue-based* difference test.

## Open questions / next steps

1. Read `lav_test_diff_Satorra2000` in the lavaan source and characterise its
   construction exactly (which model anchors `Δ`, `V`, `Σ̂`; which `Γ`; whether
   it uses a full `p* × p*` `Uᵈ Γ` spectrum or a reduced form).
2. Re-derive the Satorra (2000) difference test from the paper and decide which
   construction is canonical — the H1-anchored reduced form, lavaan's variant,
   or both as asymptotically-equivalent estimators of the same limit.
3. Decide whether magmaan should match lavaan's `satorra.2000` for oracle
   parity, or document the difference as an intentional, defensible choice.
4. If the math says lavaan's `satorra.2000` is the outlier, prepare a minimal
   reproducible example and file a lavaan issue.
5. Once resolved, re-enable a nested-test parity check in
   `r-package/examples/nested_test_satorra2000.R`.

## Reproduction

```r
suppressMessages({ library(magmaan); library(lavaan) })
df <- as.data.frame(HolzingerSwineford1939[paste0("x", 1:9)])
h1 <- "visual=~x1+x2+x3\ntextual=~x4+L5*x5+L6*x6\nspeed=~x7+x8+x9"
h0 <- "visual=~x1+x2+x3\ntextual=~x4+Lt*x5+Lt*x6\nspeed=~x7+x8+x9"

f1 <- magmaan(h1, df, estimator = "ML", se = "none", test = "none")
f0 <- magmaan(h0, df, estimator = "ML", se = "none", test = "none")
nt <- nestedTest(fit_H1 = f1, fit_H0 = f0, data = df)

l1 <- cfa(h1, data = df, estimator = "MLM")
l0 <- cfa(h0, data = df, estimator = "MLM")
lr <- lavTestLRT(l1, l0, method = "satorra.2000")

cat("magmaan scaling c :", nt$scale_c, "\n")                       # ≈ 1.4462
cat("lavaan  scaling c :",
    (fitMeasures(l0,"chisq") - fitMeasures(l1,"chisq")) /
    lr[2, "Chisq diff"], "\n")                                     # ≈ 1.2810
```
