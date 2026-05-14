# Nested-model χ² difference tests — Satorra (2000)

A nested-model likelihood-ratio test compares a restricted model `H0` against
a less-restricted superset `H1`. Under multivariate normality the difference

```
T_diff = T_H0 − T_H1    where  T_M = N · F_ML(θ̂_M)
```

is asymptotically `χ²(m)` with `m = df_H0 − df_H1`. Under arbitrary continuous
distributions (the SEM "robust" regime, lavaan's `estimator = "MLR"` / `"MLM"`)
the distribution becomes a *weighted* mixture

```
T_diff ⤳ Σⱼ λⱼ · χ²₁,ⱼ ,
```

where `λ` are the `m` non-zero eigenvalues of the (potentially huge) `UΓ`
matrix Satorra-Bentler theory associates with the restriction.  Satorra
(2000) — and the GPT-5.5 derivation in `notes.md` — show those eigenvalues are
exactly the `m` generalised eigenvalues of an `m × m` problem, and that the
empirical Γ̂ can be folded into the accumulator without ever being
materialised.

This is the test magmaan implements via:

* `compute_satorra2000(...)` — `include/magmaan/fit/satorra2000.hpp`. The pure
  core: takes per-group `Π_α`, `Σ̂_g`, raw `X`, and `A_α`; returns `C`, `S`,
  eigenvalues, and the two scalar traces.
* `lr_test_satorra2000(...)` / `lr_test_satorra2000_from_data(...)` —
  `include/magmaan/fit/lr_test_satorra.hpp`. Wraps eigenvalues into the
  scaled / adjusted / mixture p-values.
* `restriction_alpha_from_K(K_H1, K_H0)` —
  `include/magmaan/fit/restriction.hpp`. Derives the `m × r1` restriction
  matrix from the two fits' constraint reparameterisations.
* `imhof_upper(λ, x)` — `include/magmaan/fit/weighted_chisq.hpp`. The exact
  mixture-χ² tail via Imhof's quadrature.

## The math, abridged

Let `Π = ∂σ/∂θ` (`p* × k`), `V` the moment-structure weight (`p* × p*`,
`V = Γ_NT(Σ̂)⁻¹` for the SB family), and `Γ` the 4th-moment ACOV of the
sample moments (`p* × p*`, either `Γ_NT(Σ̂)` for the normal-theory case or
the empirical `Γ̂` for MLR).  Then with the H0 restriction `A` (`m × k`),
Satorra-Bentler's reduced matrix has eigenvalues equal to those of

```
C = A · P⁻¹ · Aᵀ                    (m × m)
S = A · P⁻¹ · Πᵀ · V · Γ · V · Π · P⁻¹ · Aᵀ
P = Πᵀ · V · Π                       (k × k, expected info at H1)
```

via the generalised eigenproblem `S·v = λ·C·v`.  The streaming trick: with
`Y = P⁻¹·Aᵀ` (`k × m`), `D_g = V_g · Π_g · Y` (`p*_g × m`), and casewise
residuals `d_gi − s_g`,

```
u_gi = D_gᵀ · (d_gi − s_g)              (m-vector)
S    = Σ_g (n_g/N) · (1/n_g) · Σᵢ u_gi · u_giᵀ
```

— Γ_g never enters the algebra.  The only group-sized object touched is `D_g`
(p*_g × m, with m typically 1–10).

## The three p-values

Given `λ = (λ₁, …, λ_m)`:

| Variant | Statistic | Reference distribution | p-value |
|---|---|---|---|
| **Scaled** (Satorra-Bentler) | `T_scaled = T_diff / ĉ`, `ĉ = (Σ λ) / m` | `χ²(m)` | `Pr(χ²(m) > T_scaled)` |
| **Adjusted** (mean + variance) | `T_adj = T_diff · d̂₀ / (Σ λ)`, `d̂₀ = (Σ λ)² / (Σ λ²)` | `χ²(d̂₀)` (real-valued df) | `Pr(χ²(d̂₀) > T_adj)` via the noncentral CDF |
| **Mixture** (exact, Imhof) | `T_diff` | `Σⱼ λⱼ · χ²₁,ⱼ` | `imhof_upper(λ, T_diff)` |

A fourth column reports `Pr(χ²(m) > T_diff)` — the *unscaled* p-value, which
is wrong under non-normality but useful to compare against the corrections.

## How to call it from C++

```cpp
auto restr = magmaan::fit::restriction_alpha_from_K(K_H1, K_H0);
auto res   = magmaan::fit::lr_test_satorra2000_from_data(
    pt, rep, theta_H1_full, K_H1, K_H0,
    X_per_group, mean_per_group, n_per_group, weight_per_group,
    T_H0, T_H1, df_H0, df_H1,
    magmaan::fit::GammaSource::Empirical);
// res->p_scaled, res->p_adjusted, res->p_mixture, res->eigenvalues, …
```

`weight_per_group[g] = n_g / N_total` — the same per-group weight as in the
pooled ML objective `F = Σ_g f_g · F_g`.

## How to call it from R

```r
fit_H1 <- magmaan::cfa(...)            # less-restricted
fit_H0 <- magmaan::cfa(..., constraints = "...")
res    <- magmaan::nestedTest(fit_H1, fit_H0)
# res$p_scaled, res$p_adjusted, res$p_mixture, res$eigenvalues, …
print(res)                           # lavaan-style LRT table
```

The R wrapper extracts the raw data from the call frame (`data` argument on
each fit) and the two constraint matrices from each fit's stored partable;
the user does not have to hand-pass anything.

## Cross-validation

* `tests/unit/satorra2000_test.cpp` — eigenvalues from the streaming m × m
  path agree with an inline form-the-full-`UΓ` reference to ≤ `1e-10` on a
  synthetic 2-variable saturated fixture.  NT-Γ sanity (all `λⱼ → 1`) and
  the degenerate `m = 0` case are also checked.
* `tests/unit/weighted_chisq_test.cpp` — Imhof matches closed-form `χ²`
  cases (single `λ`, all-equal `λ`) to 1e-6, and Monte-Carlo cross-checks
  unequal-spectrum cases within 4·σ.
* `r-package/examples/nested_test_satorra2000.R` — end-to-end against
  lavaan's `lavTestLRT(method = "satorra.2000")` on Holzinger-Swineford
  configural vs metric invariance.
