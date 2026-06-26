# Calculating and Interpreting Maximal Reliability in Bifactor Models

**Cite.** Li, S., & Savalei, V. (2026). *Multivariate Behavioral Research*, 61(3):469-490. DOI: 10.1080/00273171.2025.2612035. OSF: osf.io/6k5gq.
**PDF.** `external/refs/Calculating and Interpreting Maximal Reliability in Bifactor Models.pdf`
**Read.** 2026-06-26  ·  **Verdict.** → speculative

## TL;DR
Coefficient H, as routinely applied to bifactor models (Rodriguez et al. 2016 plug
standardized bifactor loadings into the one-factor H formula), is wrong: it is not
the reliability of any weighted composite in a bifactor model. The paper gives the
*correct* maximal-reliability formulas, in closed form, on quantities a fitted
bifactor model already produces, with a clean lavaan parity oracle
(`lavInspect(fit, "fs.reliability")` / `"fs.determinacy"`). For magmaan this is a
thin `measures::frontier` reliability layer and the maximal sibling of the
already-recorded model-based-omega entry; but the paper's own conclusion is
discouraging (group-factor composites are not reliable at realistic sizes), so it
is a low-demand build with a tight trigger.

## Contribution
- **The error it fixes.** Coefficient H (Hancock 2001) is the maximal reliability
  of an optimal linear composite (OLC) under a *one-factor* model. Substituting
  bifactor loadings into the H equation does not yield the reliability of any
  composite. Despite this, H is widely computed in bifactor applications.
- **The key identity.** The one-factor maximal-reliability derivation never assumes
  the residual covariance Ψ is diagonal, so a bifactor model is rewritten as a
  one-factor model with structured "error" Σ − λ_G λ_G' (for the general factor) or
  Σ − λ_grp,j λ_grp,j' (for group factor j). Everything else follows by reusing the
  one-factor algebra.
- **Two correct coefficients.** (1) OLC: weights on *all* items maximizing
  reliability for one target factor: `w_gen = (Σ − λ_G λ_G')⁻¹ λ_G`, `ρ*_gen = 1 /
  (1 + 1/(λ_G' (Σ − λ_G λ_G')⁻¹ λ_G))`, and the group-factor analog (eqs. 14-17).
  (2) OLSC: weights on only that group factor's items (eqs. 19-20); this is the
  *correct* generalization of coefficient H. Ordering: ρ_gen ≥ ω_H, and ρ*_grp,j ≥
  ρ*_gj ≥ ω_HS,j (OLC ≥ OLSC ≥ unit-weighted).
- **Unification.** Thurstone regression factor scores *are* OLCs (weights ∝ optimal
  weights), and factor determinacy FD = sqrt(maximal reliability). So magmaan's
  existing regression-score path already contains these objects.
- **Negative result (the real message).** Population sim, 9 to 1500 indicators: OLC
  for the general factor reaches reliability 0.8 by ~15 indicators, but group-factor
  OLC needs ~200 and OLSC ~1000. Optimal weights are frequently negative
  (asymptotically ~45-55% of items), including positively-loading items, making the
  composites uninterpretable. Recommendation: do not use composites/sub-composites
  as proxies for group factors; report maximal reliability only as a *model
  diagnostic*. No SEs/CIs (explicitly future work: Aguirre-Urreta finite-sample bias,
  Raykov delta method).

## Relevance to magmaan
**A `measures::frontier` reliability layer, closed-form, with a lavaan oracle; the
maximal sibling of the model-based-omega entry.** Every input is already produced:
model-implied Σ̂ and the loading matrix from `model::MatrixRep` /
`ModelEvaluator`, and `measures::factor_scores` already computes the regression
(Thurstone) weights `(AΨAᵀ)ΛᵀΣ̂⁻¹` whose rows *are* the OLC weights. So `ρ*_gen`,
`ρ*_grp,j`, `ρ*_gj` (OLSC), the OLC/OLSC weight vectors, and FD = sqrt(ρ*) are each
a few lines over machinery in hand. The omega family (ω_H, ω_HS) is the
unit-weighted floor and already lives in the
[Model-based omega](../../backlog/speculative.md) speculative entry.

**The parity story is clean, which is what makes it magmaan-shaped.** lavaan exposes
`lavInspect(fit, "fs.reliability")` and `"fs.determinacy"`, which equal per-factor
maximal reliability and FD for the regression-score OLC (the paper cross-checks
against exactly these: Table 1 fs.reliability 0.871/0.178/0.497/0.661). So the
maximal-reliability-of-OLC numbers gate directly against lavaan; the unit-weighted
ω against semTools on a refit. Note the distinction from the existing in-flight
`measures::frontier::reliability` module, which is *S-based* (alpha, Guttman λ6,
Spearman-Guttman omega off the sample covariance): this paper's objects are
*model-based* (functions of θ̂, like model-based omega), a different object class.

**Caveats that gate the build.** The paper's bottom line is a discouragement: at
empirically realistic sizes (Li & Savalei 2025 found 82% of bifactor apps use < 30
indicators) group-factor composites are not reliable and are riddled with negative
weights, so the practical deliverable shrinks to "report ρ* as a model diagnostic."
That is real but narrow, and magmaan's audience is methods developers, not end users
scoring scales. There is no new estimator, no new inference (the paper leaves SEs/CIs
open), and no parser/partable surface. The genuinely novel build cell, as with the
Bell omega entry, would be the ordinal/categorical extension, which the paper does
not touch (continuous normal only). Connects to the exp-20 alpha-omega thread
([[exp20-deng-chan-alpha-omega]]) only loosely (that is a reliability-difference
test, not bifactor maximal reliability).

## Verdict
→ speculative — recorded as a new `measures::frontier` entry,
[Maximal reliability / corrected coefficient H for bifactor models](../../backlog/speculative.md),
cross-linked to the Model-based omega entry (the unit-weighted floor of the same
ordering). Build trigger: a bifactor/hierarchical reliability experiment or paper
that needs maximal reliability as a first-class output, OR the continuous
factor-score path graduating to expose `fs.reliability` / `fs.determinacy` (at which
point ρ* and FD are nearly free and gate against lavInspect), OR an ordinal
extension of the omega/reliability cluster. Cite as the correct-coefficient-H
reference and for the OLC = regression-score = sqrt(FD) unification regardless of
whether the layer is built.
