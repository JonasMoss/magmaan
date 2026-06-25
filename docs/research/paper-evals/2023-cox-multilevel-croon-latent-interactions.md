# Croon's Bias-corrected Estimation for Multilevel Structural Equation Models with Latent Interactions

**Cite.** Cox, K., Kelcey, B., & Bai, F. (2023). *Structural Equation Modeling: A Multidisciplinary Journal*, 30(3):467-480. DOI: 10.1080/10705511.2022.2140290.
**PDF.** `external/refs/Croon s Bias-corrected Estimation for Multilevel Structural Equation Models with Latent Interactions.pdf`
**Read.** 2026-06-25  ·  **Verdict.** background

## TL;DR
Extends Croon's method-of-moments bias correction (the SAM/factor-score-regression
family) to two-level SEMs with latent interactions, as a convergence-robust
alternative to LMS/ML for latent moderation. Doubly outside magmaan's contract:
multilevel and latent interactions, validated against Mplus, not lavaan.

## Contribution
- Croon's estimation = a SAM/FSR sequential recipe: fit each measurement model,
  predict factor scores, estimate the structural model against a factor-score
  covariance that has been *corrected* for latent-variable unreliability via a
  method of moments (Croon 2002). The correction scales each covariance by the
  reliabilities implied by Λ, Θ.
- New piece: corrections for **latent interactions in two-level MLSEMs**. They
  enumerate seven moderation designs across within/between/cross levels
  (1×(1-1), 2×(1-1), 2×(2-1), 1×(2-1)) and derive six distinct Croon corrections
  for the interaction covariances. The multilevel-specific term is `R^L2`, the
  vector of between-level factor-score reliabilities, needed because between-level
  indicator components are predicted (cluster means) rather than observed.
- Latent-interaction variance via Bohrnstedt-Goldberger: var(ηZ ηX) =
  var(ηZ)var(ηX) + cov(ηZ,ηX)², computed from the corrected components since the
  interaction is not itself a measured factor.
- Simulations vs Mplus 8.6 (LMS-style ML and plain factor-score regression):
  Croon's gives near-unbiased structural coefficients and far fewer convergence
  failures than ML at small N, at FSR's computational cost. Assumes no random
  slopes, cross-level loading invariance, uncorrelated normal residuals.

## Relevance to magmaan
Background only, nothing to build. magmaan's contract is single-level, linear,
complete-data normal-theory SEM with lavaan as oracle; this paper is multilevel
+ latent interactions + Mplus-validated, three steps outside it. The one
connection is conceptual: the underlying **single-level** Croon correction is the
same Croon/Wall-Amemiya factor-covariance correction already named in the SAM/LSAM
speculative entry (the FSR-with-Croon = local-SAM-with-ML-mapping identity from
the [Dhaene & Rosseel 2023 eval](2023-dhaene-noniterative-sam.md)). This paper
neither sharpens that entry nor opens a new one: the actionable SAM design notes
(reuse ML measurement fit, skip the closed-form estimator zoo, parity vs lavaan
`sam()`) already live there, and the multilevel/interaction machinery has no
lavaan oracle and no place in the matrix-rep contract. File it as a pointer to
where the latent-moderation and multilevel-SEM frontiers begin, both firmly
beyond v1.

## Verdict
background — single-level Croon is the SAM entry's existing territory; the
multilevel + latent-interaction extension is out of scope with no lavaan oracle.
No graduation.
