# Investigating Structural Model Fit Evaluation

**Cite.** Zhang, X., & Wu, H. (2024). *Structural Equation Modeling: A Multidisciplinary Journal*, 31(5):863-881. DOI: 10.1080/10705511.2024.2350023.
**PDF.** `external/refs/Investigating Structural Model Fit Evaluation.pdf`
**Read.** 2026-06-25  ·  **Verdict.** → speculative

## TL;DR
Small-sample corrections and confidence intervals for the chi-square test and
fit indices (RMSEA, CFI, SRMR) of *just the structural model* of a full SEM,
computed via the Hancock-Mueller two-step procedure. The naive structural
chi-square has Type-I error above 80% in small low-reliability samples; a
Satorra-Bentler-type mean correction fixes it, and the paper supplies the
missing CIs for the structural indices.

## Contribution
- **Setup, not new:** the two-step procedure (Hancock & Mueller 2011). Step 1
  fits the measurement model with a saturated latent covariance, yielding a
  K×K saturated factor covariance Φ̂. Step 2 fits the hypothesized structural
  model treating the K latents as observed with Φ̂ as input; the Step-2 fit
  stats are the "structural" fit indices, isolated from measurement misfit and
  (per McNeish & Hancock 2018) immune to the reliability paradox. This two-step
  is a special case of joint-measurement local SAM.
- **The problem they fix:** the original structural chi-square T_HS and the HS
  indices ignore that Φ̂ in Step 1 came from raw data, so Φ̂ is not Wishart and
  T_HS is a weighted mixture of 1-df chi-squares, not χ²_df. Small-sample
  biased; Type-I grossly inflated (>80% low-reliability, >30% high).
- **Fix 1:** a mean-corrected structural chi-square T_HS,C = T_HS / ĉ_HS, with
  ĉ_HS an SB-type scaling constant estimating the average mixture weight
  (eq. 24).
- **Fix 2:** corrected structural RMSEA/CFI/SRMR (eqs. 25-27) that fold ĉ plus
  proper small-sample corrections accounting for Step-1 variability (the
  original RMSEA/CFI carried a naive correction, the original SRMR none).
- **Fix 3 (the genuine gap-fill):** CIs for the structural indices. RMSEA by
  noncentral-χ² inversion with the scaling; CFI and SRMR by Welch-type /
  delta-method CIs adapting Lai (2019) and Maydeu-Olivares (2017). No structural
  fit-index CIs existed before.
- **Six versions V1-V6** of the scaling differ by (a) which information matrix
  for the Step-2 weight Ŵ_HS (observed / expected / sandwich) and (b) the
  Γ̂_SS estimate of the latent-covariance asymptotic covariance (expected-info
  block W_SS⁻¹ vs a sandwich triple product). **V2 and V5 win** (both use the
  expected information of the saturated structural model in Step 2 evaluated at
  HS estimates); V5's triple-product Γ̂_SS is the non-normal-robust sandwich and
  ≈ V2 under normality. Both give near-nominal Type-I, low bias, good CI
  coverage.
- Sim: mediation model (4 exogenous + 3 endogenous latents), low/high
  reliability, n = 150-1000, six increasing-misfit structural models. Side
  result refining McNeish & Hancock 2018: structural-index bias *does* decrease
  as measurement reliability rises (their invariance finding was an n=1000,
  misspecified-only artifact).

## Relevance to magmaan
Squarely in-domain but two-deep. The capability is `measures::frontier`:
structural fit indices + their CIs as a post-fit surface. It is built from
machinery magmaan already has (`inference` observed/expected/first-order
information; `robust` Satorra-Bentler scaling family; `measures` full-SEM
RMSEA/CFI/SRMR and baseline), but it is **gated on the two-step / SAM
scaffolding** that magmaan does not have: the Step-1 saturated-latent
measurement fit producing Φ̂ and its asymptotic covariance Γ̂_SS, then a
structural Step-2 fit. That scaffolding is the
[SAM `estimate::frontier` entry](../../backlog/speculative.md) the Bogaert 2025
eval just recorded; this paper is the fit-evaluation layer on top of it.

What magmaan would add beyond SAM itself: the structural scaling constant ĉ_HS
(eq. 24, Table 1), the corrected indices (eqs. 25-27), and the Welch/delta CIs
(Table 2). Not pure transcription: ĉ_HS needs Γ̂_SS = the W_SS⁻¹ block (expected
information of the latent variances/covariances from Step 1), and the authors
explicitly flag that "specific algorithms are needed to compute the inverse of
the expected information matrix ... in Step 1" as unspecified future work, so
this is real machinery. Oracle split: the *original* HS versions (RMSEA_HS,
CFI_HS, naive-BC SRMR_HS) and their CIs are in lavaan; the *corrected* versions
are the paper's contribution with code only on OSF (osf.io/dcj2z), so gate the
corrected ones transitively or first-principles, not against lavaan.

Two hooks worth recording. (1) Structural CFI needs a *structural* baseline
(latents uncorrelated); magmaan's known baseline-df bug under fixed.x would bite
here, see [[fixedx-baseline-df-gap]]. (2) The structural chi-square is a mixture
of 1-df chi-squares, and ĉ_HS matches only its first moment (SB style); the FMG
/ pEBA spectrum machinery in `papers/fiml-fmg/` matches the *whole* spectrum, so
a "structural-FMG" test is the natural over-the-top extension if the structural
two-step ever lands and a paper wants a sharper structural GOF than the
mean-corrected one.

## Verdict
→ speculative — recorded as the `measures::frontier` structural-fit-evaluation
entry in [backlog/speculative.md](../../backlog/speculative.md)
("Structural-model fit indices, tests, and CIs (two-step / SAM)"), explicitly
downstream of the SAM estimator entry. Build trigger: the SAM two-step lands
*and* a methods workflow or paper needs isolated structural GOF with CIs (the
small-N inference or SAM-under-misspecification studies are the natural home).
Until then, lavaan's original two-step HS indices cover the uncorrected case and
the authors' OSF code covers the corrected one.
