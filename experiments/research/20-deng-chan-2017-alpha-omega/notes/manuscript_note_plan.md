# Manuscript note plan: alpha, omega, and the non-regular Deng-Chan Wald test

## Working thesis

Testing alpha = omega in a one-factor model is testing tau-equivalence. Deng and
Chan's scalar Wald test for omega - alpha is not a valid null test of that
hypothesis because the reliability gap has zero first derivative at the
tau-equivalent point; the statistic is second-order and needs a quadratic-form
reference law.

The note should be short and corrective: two sentences on alpha/omega, one
paragraph on why equality is tau-equivalence, one table showing the failure, and
an appendix for the Taylor expansion and Imhof construction.

## Target and framing

- First target: *Educational and Psychological Measurement*. The note corrects
  a method published there and should be easy for the journal's audience to
  evaluate.
- Backup targets: *Psychological Methods* if the paper is expanded around
  singular inference, or *Structural Equation Modeling* / *Measurement* if EPM
  does not fit the article type.
- Tone: do not spend Deng and Chan's intro length re-arguing alpha versus omega.
  The core claim is mathematical and visible in one simulation table.

## Main text outline

1. Define alpha and omega briefly. Alpha is the covariance-based reliability
   coefficient that is exact for tau-equivalent item sums. Omega is the
   one-factor/congeneric reliability coefficient for the same total score.
2. State the equality problem. In a one-factor model, omega is at least alpha,
   with equality exactly when loadings are equal, so alpha = omega is
   tau-equivalence.
3. Separate two Wald tests. Deng and Chan's Wald test is a one-df Wald test on
   the derived scalar omega - alpha; the regular structural Wald test is a
   (p - 1)-df test of equal loadings.
4. Show the non-regularity. At tau-equivalence the reliability gap is at an
   interior minimum, the gradient is zero, and the first-order delta method
   collapses.
5. Present the comparison table from `report.qmd`: Deng Wald, Imhof gap test,
   equal-loading Wald, Satorra LR, and score across tau, congeneric, and
   misspecified populations.
6. Conclude with the recommendation. For tau-equivalence, use the regular
   equal-loading Wald/LR/score tests. If the scalar reliability gap is the
   estimand, calibrate it with the second-order quadratic-form law or a null
   bootstrap, not a normal z.

## Appendix contents

- Algebra: omega - alpha is proportional to loading dispersion divided by total
  variance, so the tau-equivalent point is an interior minimum.
- Asymptotics: if sqrt(n)(s_hat - s0) converges to N(0, Gamma) and the gradient
  of d(s) = omega(s) - alpha(s) is zero, then
  n d(s_hat) converges to 0.5 Z' H Z.
- Nonnormality diagnostics: `alpha_delta_nonnormality.qmd` records the
  first-order delta-method decomposition for coefficient alpha, separating the
  normal-theory, latent-kurtosis, and residual-kurtosis contributions under an
  independent-source congeneric model.
- Deng statistic: the estimated first-order SE is also O(1 / n), producing a
  non-pivotal ratio rather than N(0, 1).
- Computation: estimate H and Gamma under the null, eigendecompose the
  quadratic form, and evaluate the weighted chi-square tail by Imhof.
- Validation: alpha equals the omega from a ULS tau-equivalent fit; the
  first-order sandwich SE agrees with a nonparametric bootstrap away from the
  null; the null diagnostic shows SD(omega - alpha) scales as 1 / N.
  `alpha_uls_identity.md` records the exact statement, the two-line proof
  (tau-equivalent and parallel), why it is ULS and not ML, and the citation
  provenance for the identity.

## Citation audit status

Quick web audit, 2026-06-02:

- Deng and Chan (2017) is the source being corrected: it proposes SEs,
  confidence intervals, and a test for alpha and omega differences
  ([SAGE abstract](https://journals.sagepub.com/doi/abs/10.1177/0013164416658325),
  [PubMed record](https://pubmed.ncbi.nlm.nih.gov/29795909/)).
- Indexed search for `Deng Chan alpha omega nonregular`, `Deng Chan alpha omega
  Imhof`, and `Deng Chan second-order delta method` did not surface an obvious
  prior correction of the null-law problem. This is not enough for a novelty
  claim; it only clears the first pass.
- Later reliability discussions cite Deng and Chan as an applied reference or as
  evidence that empirical alpha-omega differences are often tiny, without
  flagging the singular Wald issue in the snippets inspected
  ([Psicothema/Scielo review](https://scielo.isciii.es/scielo.php?lng=es&nrm=iso&pid=S1886-144X2023000100002&script=sci_arttext)).
- Applied papers continue to cite Deng and Chan for reporting alpha and omega,
  including broad psychometric and scale-validation work visible in PubMed,
  Frontiers, ScienceDirect, and Cambridge-indexed snippets.

Needed before submission:

- Pull the citing-paper set from Google Scholar plus at least one reproducible
  source such as OpenAlex/Semantic Scholar/Scopus.
- Screen titles/abstracts for: `alpha omega difference`, `Deng Chan`, `Wald`,
  `delta method`, `bootstrap`, `tau-equivalence`, `nonregular`, `singular`,
  `Imhof`, and `quadratic form`.
- Read any candidate methodological papers, not only snippets.
- If a prior correction exists, reframe the note as an accessible derivation,
  empirical demonstration, and implementation comparison.

## Current experiment status

- `run_experiment.R` now reports five tests: Deng scalar Wald, Imhof-calibrated
  reliability gap, regular equal-loading Wald, Satorra-2000 LR, and score.
- `report.qmd` is the manuscript evidence source: it contains the diagnostic,
  main rejection-rate table, implementation checks, caveats, and reproduction
  commands.
- Development runs currently use magmaan for ML fits and Satorra-2000 LR. The
  pre-publication reproducibility target is to replace as much as possible with
  lavaan or clearly document the magmaan-only pieces.
