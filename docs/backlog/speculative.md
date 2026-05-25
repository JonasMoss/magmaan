# Speculative backlog

Items we may never need but want to keep findable. Each entry names the gap,
the cheaper alternative that already covers the practical case, and the
specific condition under which we'd actually build the item. Unlike
[`todo.md`](todo.md), nothing here is committed work — promote to `todo.md`
only when a concrete downstream consumer (paper row, user request, parity
failure) appears.

## Pairwise covariance / missing-data side

### ~~Inference-side pairwise NT reducer~~ — landed

`robust::reduced_gamma_nt_pairwise(uf, raw, pw)` is now in
`include/magmaan/robust/robust.hpp`, with a sibling `WeightMoments::Pairwise`
value and the `build_u_factor(pt, rep, samp, est, raw, pw, spec)` overload
that builds the U-factor with `(Γ_NT^pw)⁻¹` as the bread. R surface:
`magmaan_core$robust_build_u_factor_pairwise(fit, X, mask, bread)` and
`magmaan_core$robust_reduced_gamma_nt_pairwise(uf, X, mask)`. Tests in
`tests/unit/pairwise_inference_test.cpp`; R demo at
`r-package/examples/pairwise_robust_se.R`.

### Pairwise μ ACOV for `fit_gls_pairwise` mean-structure models

The μ-block of the GLS weight currently uses the Σ-only `Ŝ_pw⁻¹`
convention. The asymptotically efficient choice under MAR is
`(Γ_NT^μ-pw)⁻¹` with `Γ_NT^μ-pw_{j,k} = σ_{jk} · π_{jk}/(π_j π_k)` — the
same Hadamard structure as the σ-side, applied to the p × p mean ACOV.
Same pattern-grouped trick: enumerate distinct missingness patterns,
average masked-and-rescaled `Σ̂_pw` per pattern.

**Alternative already available.** The current Σ-only μ-block is
consistent under MAR; only asymptotic efficiency on the mean side is
sacrificed.

**Build if.** A pairwise-GLS fit with `meanstructure = TRUE` shows up as
a downstream consumer AND the means side carries non-trivial information
about θ AND the missingness is heavy enough that the μ-block weight
matters. Most CFA / SEM work fits `meanstructure = FALSE` so this is a
double-edge case.

### Pairwise Browne-unbiased

The paper (`papers/ugamma-fast/ugamma-fast.tex`, eq. 19) flags that the
literal finite-sample-unbiased pairwise Γ_u^pw has entrywise-dependent
overlap-set coefficients and does not collapse cleanly to the projected
form. Two paths exist, neither is core machinery:

- **Literal Γ_u^pw**: entrywise overlap-set bookkeeping, p*² entries
  each with its own (a_{ab}, b_{ab}) coefficients and overlap-restricted
  Γ_NT / rank-1 pieces. Worst-case O(p*² · n + p*² · df) per block.
  Expensive and intricate. This is the canonical estimator.

- **Pragmatic fallback**: global Browne coefficients with
  Hadamard-adjusted NT and rank-1 pieces. Cheap. Asymptotic behaviour
  of the substitution is not established literature.

**Alternative already available.** Complete-data Browne unbiased is in
`robust::reduced_gamma_unbiased`. Pairwise data with empirical meat
(Ψ̂'Ψ̂/n) is in `pairwise_casewise_contributions`. The model-implied NT
pairwise meat is the inference-side reducer above.

**Build if.** A downstream project needs a finite-sample-unbiased
pairwise meat AND the empirical alternative is demonstrably inadequate.
Path: literal first (so it's a credible reference); then pragmatic as
opt-in with a bias study comparing the two — see
[[feedback-shortcut-variants]] memo: pragmatic-only is research-tier
work, not a free addition.
