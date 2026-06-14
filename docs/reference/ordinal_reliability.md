# Ordinal Reliability References

Local PDFs are kept outside Git under:

```text
external/refs/ordinal_reliability/
```

This file is the tracked manifest. Do not make builds, tests, fixtures, or
examples depend on files in `external/refs/`.

## Downloaded Local PDFs

Downloaded on 2026-06-14:

- Liu, Pek, and Maydeu-Olivares. "Understanding Measurement Precision from a
  Regression Perspective." arXiv:2404.16709, latest arXiv revision 2025.
  Local file:
  `liu_pek_maydeu_olivares_2025_measurement_precision_regression_perspective.pdf`
  Links: https://arxiv.org/abs/2404.16709,
  https://doi.org/10.48550/arXiv.2404.16709,
  related DOI https://doi.org/10.1037/met0000763.
- Liu, Pek, and Maydeu-Olivares. "On a General Theoretical Framework of
  Reliability." arXiv:2407.00716, latest arXiv revision 2024.
  Local file:
  `liu_pek_maydeu_olivares_2025_general_theoretical_framework_reliability.pdf`
  Links: https://arxiv.org/abs/2407.00716,
  https://doi.org/10.48550/arXiv.2407.00716,
  related DOI https://doi.org/10.1111/bmsp.12360.
- Sung and Liu. "Asymptotic Standard Errors for Reliability Coefficients in
  Item Response Theory." arXiv:2503.22924, latest arXiv revision 2026.
  Local file:
  `sung_liu_2026_asymptotic_standard_errors_reliability_irt.pdf`
  Links: https://arxiv.org/abs/2503.22924,
  https://doi.org/10.48550/arXiv.2503.22924.
- Beauducel, Hilger, and Weide. "Bias of Determinacy Coefficients in
  Confirmatory Factor Analysis Based on Categorical Variables."
  arXiv:2305.06903.
  Local file:
  `beauducel_hilger_weide_2023_bias_determinacy_categorical_cfa.pdf`
  Links: https://arxiv.org/abs/2305.06903,
  https://doi.org/10.48550/arXiv.2305.06903.

## Background Pointers

These are useful context papers for the distinction between CTT reliability,
factor score determinacy, and PRMSE. They are listed here as bibliographic
pointers; local PDFs were not downloaded in this pass.

- McDonald (2011). "Measuring Latent Quantities." Psychometrika 76, 511-536.
- Kim (2012). "A Note on the Reliability Coefficients for Item Response
  Model-Based Ability Estimates." Psychometrika 77, 153-162.
- Haberman and Sinharay (2010). "Reporting of Subscores Using Multidimensional
  Item Response Theory." Psychometrika 75, 209-227.
- Andersson and Xin (2018). "Large Sample Confidence Intervals for Item
  Response Theory Reliability Coefficients." Educational and Psychological
  Measurement 78, 32-45.

## Current magmaan Use

The active research note is
`docs/research/notes/ordinal_factor_score_reliability.md`.

The implementation currently exposes one-factor ordinal/mixed-ordinal EAP
posterior moments, sample-normalized PRMSE, and concrete ordinal reliability.
Inference with theta-hat uncertainty is documented as future work; the Sung and
Liu arXiv paper is the closest derivation template, with the IRT item-parameter
influence function replaced by magmaan's ordinal SEM moment influence path.

## Working Notes

Terminology:

- Keep "factor-score reliability" out of public names unless a specific
  coefficient is named. It is overloaded across squared determinacy, CTT
  reliability of an observed score, PRMSE for a latent score, and posterior-MSE
  coefficients.
- The EAP/posterior mean is the only ordinal score currently implemented in
  magmaan that has the exact posterior-MSE identity. EBM and ML are modes, so
  they can have useful determinacy-like summaries but should not be called
  PRMSE without defining a different target.
- Current result fields deliberately separate:
  - sample-normalized PRMSE:
    `Var_n(E[Z | y]) / {Var_n(E[Z | y]) + mean_n Var(Z | y)}`;
  - concrete ordinal reliability:
    `1 - mean_n Var(Z | y) / Var_theta(Z)`, reducing to
    `1 - mean_n Var(Z | y)` when `Var_theta(Z) = 1`.

Inference plan:

- Bootstrap first. Parametric bootstrap is closest to model reliability:
  simulate ordinal/mixed data at `theta_hat`, rebuild stats, refit, recompute
  both coefficients. Nonparametric bootstrap is also useful for robustness but
  answers a slightly different question.
- Analytic SEs need the same shape as Sung and Liu but with magmaan's ordinal
  SEM estimator influence function: casewise threshold/correlation moments,
  NACOV/Gamma, the LS/GMM bread, and the derivative of posterior moments wrt
  free SEM parameters.
- For sample PRMSE, differentiate the three sample moments
  `[m_i, m_i^2, v_i]`. For concrete reliability, differentiate only `v_i` plus
  the fitted latent variance. If the latent variance is fixed to one, the
  concrete coefficient is the simpler first analytic target.
- Finite-difference derivatives of the posterior moments are acceptable for a
  research surface. Analytic/AD derivatives can wait until there is evidence
  this becomes core.

Likely next slices:

1. Add a small simulation/fixture check for a one-factor ordinal GRM-like CFA:
   EAP scores match current scorer, `corr(Z, E[Z | Y])^2` tracks PRMSE, and
   concrete reliability tracks `1 - mean Var(Z | Y)` under `std.lv`.
2. Add a `measures::frontier` or R-only bootstrap helper for
   `factor_score_precision`, returning percentile/basic intervals for both
   coefficients.
3. Expose or factor stable casewise ordinal moment influence plumbing before
   attempting analytic SEs.
4. Leave CTT reliability of the EAP score, multi-factor EAP reliability, and
   EBM/ML determinacy summaries explicit and separate until there is a concrete
   consumer.
