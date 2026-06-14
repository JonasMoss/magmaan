# Ordinal Factor-Score Reliability / PRMSE

Status: research note, 2026-06-14.

## Short Answer

The quantity we want is known in the recent reliability literature as
proportional reduction in mean squared error (PRMSE), not always as
"reliability" in the classical-test-theory sense.

For a scalar latent score `Z`,

```text
PRMSE(Z) = Var(E[Z | Y]) / Var(Z)
         = 1 - E[Var(Z | Y)] / Var(Z).
```

If `Var(Z) = 1`, this is the user's heuristic
`1 - E[(Zhat - Z)^2]` when `Zhat = E[Z | Y]`, the EAP/posterior-mean
score. The equality is not automatic for EBM or ML factor scores: those are
modes, not the MSE-minimizing posterior mean. For EBM/ML we can report a
squared-correlation or determinacy-like score, but calling it exact PRMSE
would be wrong unless we deliberately define a different coefficient.

There are two sample plug-ins worth keeping distinct:

```text
sample-normalized PRMSE =
  Var_n(m_i) / {Var_n(m_i) + mean_n(v_i)}

concrete ordinal reliability =
  1 - mean_n(v_i) / Var_theta_hat(Z)
```

where `m_i = E[Z | y_i; theta_hat]` and
`v_i = Var(Z | y_i; theta_hat)`. The second coefficient is the direct
`1 - E[(Zhat - Z)^2]` idea. When the model fixes `Var(Z) = 1`, it reduces to
`1 - mean_n(v_i)`.

## Literature Map

Core thread:

- McDonald (2011), "Measuring latent quantities", Psychometrika 76, 511-536:
  regression framing for latent quantities and reliability.
- Liu, Pek, and Maydeu-Olivares (2025), "Understanding measurement precision
  from a regression perspective", Psychological Methods; arXiv version:
  https://arxiv.org/abs/2404.16709. They separate two decompositions:
  CTT reliability is about an observed score `s(Y)`, while PRMSE is about
  predicting a latent score from manifest variables. They explicitly note that
  confidence intervals are future work in that first paper.
- Liu, Pek, and Maydeu-Olivares (2025), "On a general theoretical framework of
  reliability", British Journal of Mathematical and Statistical Psychology 78,
  286-302; arXiv version: https://arxiv.org/abs/2407.00716. This broadens the
  target from R^2 to association measures and lays out desiderata:
  estimability, normalization, symmetry, and invariance.
- Sung and Liu (2025), "Asymptotic Standard Errors for Reliability Coefficients
  in Item Response Theory", arXiv: https://arxiv.org/abs/2503.22924. This is
  the important inference follow-up. It derives asymptotic SEs for CTT
  reliability of the EAP score and PRMSE of the latent variable under a
  unidimensional graded response model. It handles both item-parameter
  estimation uncertainty and the use of sample moments.

Nearby factor-score determinacy literature:

- Kim (2012), "A note on the reliability coefficients for item response
  model-based ability estimates", Psychometrika 77, 153-162.
- Haberman and Sinharay (2010), "Reporting of subscores using multidimensional
  item response theory", Psychometrika 75, 209-227; source of the PRMSE term in
  this thread.
- Andersson and Xin (2018), "Large sample confidence intervals for item
  response theory reliability coefficients", Educational and Psychological
  Measurement 78, 32-45. Earlier delta-method SEs for narrower IRT reliability
  forms.
- Beauducel, Hilger, and Weide (2023), "Bias of determinacy coefficients in
  confirmatory factor analysis based on categorical variables",
  https://arxiv.org/abs/2305.06903. Relevant warning: parameter-based
  determinacy coefficients for categorical CFA can be positively biased,
  especially with binary indicators, correlated factors, and unmodeled
  cross-loadings.

Lavaan context:

- `lavPredict(..., rel = TRUE)` documents factor reliabilities as squared
  factor determinacies, i.e. squared correlations between factor scores and
  latent variables. Its categorical `lavPredict()` methods expose EBM/ML
  scores, not EAP in the installed method set we currently test against.
- magmaan already implements ordinal/mixed EBM, ML, and one-factor EAP scores;
  the EAP path is the one that lines up exactly with PRMSE.

## Point Estimation

For one scalar latent `Z`, at fitted parameters `theta_hat`, compute for each
case/pattern:

```text
m_i = E[Z | y_i; theta_hat]
v_i = Var(Z | y_i; theta_hat)
```

Then form the sample-moment PRMSE estimate:

```text
h1 = mean(m_i)
h2 = mean(m_i^2)
h3 = mean(v_i)

prmse_hat = (h2 - h1^2) / (h2 - h1^2 + h3)
```

This is a self-normalizing empirical total-variance version of PRMSE. It uses
`h2 - h1^2 + h3` as the empirical analogue of `Var(Z)`, so it does not force
the denominator to be the fitted model's latent variance.

The direct concrete ordinal reliability instead uses the fitted latent prior
variance:

```text
concrete_hat = 1 - h3 / Var_theta_hat(Z).
```

For a single group with `Var_theta_hat(Z) = 1`, this is exactly the concrete
`1 - mean(v_i)` estimator. For multiple groups, the pooled denominator should
be the sample-weighted fitted latent variance, including between-group latent
mean differences:

```text
Var_pooled(Z) = mean_g[w_g {Var_g(Z) + E_g(Z)^2}]
                - {mean_g[w_g E_g(Z)]}^2.
```

### What We Already Have

`src/measures/factor_scores.cpp` already has the all-important one-factor EAP
numerical integration:

- `evaluate_pattern(...)` computes the log posterior kernel, gradient, and
  Hessian for ordinal/mixed response patterns.
- `eap_score(...)` integrates moment 0 and moment 1 by QUADPACK.
- response patterns are cached.

To compute PRMSE, the scoring primitive only needs one more posterior moment:

```text
E[Z^2 | y_i; theta_hat]
v_i = E[Z^2 | y_i; theta_hat] - m_i^2
```

That is a small extension of the existing `moment` switch in the EAP integrand.
The public result type should not overload `FactorScores`; a cleaner shape is a
new `FactorScorePrecision` or `FactorScoreReliability` result containing:

```text
method = EAP
targets = sample_prmse, concrete_ordinal_reliability
scores_by_group
posterior_var_by_group
prmse_by_group
pooled_prmse
concrete_ordinal_reliability_by_group
pooled_concrete_ordinal_reliability
```

Start scalar/componentwise. For multi-factor EAP the project already defers
adaptive Gauss-Hermite; PRMSE can stay deferred with it.

### Population vs Sample Moment Target

There are two possible estimands:

1. Population plug-in:
   average `m(Y)` and `v(Y)` under the fitted model distribution `P_theta(Y)`.
   This is clean but explodes over ordinal response patterns and is awkward for
   mixed continuous variables.
2. Sample-moment plug-in:
   average `m_i`, `m_i^2`, and `v_i` over the observed cases. This is what the
   Sung/Liu SE paper targets for long IRT tests, and it is the right first
   implementation for magmaan.

Use sample-moment PRMSE and concrete ordinal reliability first. If a paper
needs population PRMSE later, add it as an explicit `population = "model"`
option backed by response-pattern enumeration for small all-ordinal models and
model-based Monte Carlo otherwise.

## Inference

There are three viable layers.

### 1. Bootstrap, First

For a first methods-developer surface, use bootstrap CIs:

- parametric bootstrap: draw ordinal/mixed data from the fitted latent-response
  model, rebuild ordinal stats, refit, recompute PRMSE;
- nonparametric bootstrap: resample rows, rebuild stats, refit, recompute PRMSE.

Parametric bootstrap is closer to the model-based estimand. Nonparametric
bootstrap is more robust to mild misspecification but less aligned with "model
reliability". Both are expensive but simple and defensible.

### 2. Analytic Delta Method

The Sung/Liu derivation has the exact shape we need, but their parameter
estimator is IRT ML under a unidimensional GRM. For magmaan ordinal DWLS/WLS,
replace their item-parameter influence function with the ordinal SEM
GMM/polychoric influence function.

For sample-normalized PRMSE define:

```text
H_i(theta) = [m_i(theta), m_i(theta)^2, v_i(theta)]'
eta_hat(theta_hat) = mean_i H_i(theta_hat)
phi(x1, x2, x3) = (x2 - x1^2) / (x2 - x1^2 + x3)
```

The asymptotic linearization is:

```text
IF_eta,i = H_i(theta_0) - E[H(theta_0)]
           + E[dH(theta_0) / dtheta] IF_theta,i

IF_prmse,i = grad_phi' IF_eta,i
Var(prmse_hat) = Var(IF_prmse,i) / n
```

For concrete ordinal reliability define:

```text
K_i(theta) = v_i(theta)
tau(theta) = Var_theta(Z)
psi(k, tau) = 1 - k / tau
```

The linearization is:

```text
IF_k,i = K_i(theta_0) - E[K(theta_0)]
         + E[dK(theta_0) / dtheta] IF_theta,i

IF_tau,i = grad_tau(theta_0)' IF_theta,i

IF_concrete,i = -(1 / tau) IF_k,i
                + (E[K] / tau^2) IF_tau,i
```

If `Var(Z)` is fixed by identification, `IF_tau,i = 0`. If it is fixed to one,
the coefficient and its linearization collapse to:

```text
concrete_hat = 1 - mean_i v_i(theta_hat)
IF_concrete,i = -{v_i(theta_0) - E[v(theta_0)]}
                - E[dv(theta_0) / dtheta] IF_theta,i
```

In magmaan terms, `IF_theta,i` should come from the same ordinal moment stack
used by robust ordinal SEs:

```text
casewise threshold/correlation psi_i
  -> NACOV/Gamma
  -> theta influence through Delta' W Delta bread
  -> H(theta) derivative through posterior-moment quadrature
```

This means analytic inference is not conceptually hard, but it does need a
small casewise-influence plumbing project. The tricky pieces are:

- exposing or recomputing the casewise moment influence used to build
  `OrdinalStats::NACOV` / `MixedOrdinalStats::NACOV`;
- differentiating posterior moments with respect to the free SEM parameters;
  finite differences are acceptable for a first research surface, AD can come
  later if this becomes core;
- multi-group pooling: compute block-specific `H` and `IF_theta` with the same
  `n_g / N` convention as the robust ordinal sandwich, then report group and
  pooled PRMSE explicitly;
- boundary behavior: PRMSE is in `[0, 1]`, so Wald intervals can misbehave near
  the boundary. The concrete sample plug-in can also drift slightly outside
  `[0, 1]` when the observed response distribution is far from the fitted model
  distribution. Fisher/logit transforms or bootstrap percentile intervals are
  worth exposing.

### 3. CTT Reliability of the EAP Score

This is a different coefficient:

```text
Rel(s) = Var(E[s(Y) | Z]) / Var(s(Y))
```

where `s(Y) = E[Z | Y]` if the observed score is the EAP score. It asks whether
the observed score itself is true-score-like. Sung/Liu derive an SE for this
too, but the computation is more involved because `E[s(Y) | Z = z]` needs
conditional response probabilities or likelihood-ratio weighting over
quadrature nodes.

Do not implement this first. It is useful to document because it explains why
"reliability of factor scores" is an overloaded phrase.

## Proposed magmaan Milestones

1. Done: add a private one-factor posterior-moment helper returning
   `(mean, second_moment, variance)` per response pattern. Reuse the current EAP
   cache.
2. Done: add `measures::factor_score_precision_ordinal(...)` and
   `factor_score_precision_mixed_ordinal(...)` for one latent dimension, EAP
   only, complete data only, sample-moment plug-ins only.
3. Done: add the R wrapper `factor_score_precision(fit, data)`, returning
   scores, posterior variances, sample PRMSE, and concrete ordinal reliability.
4. Validate further against a small GRM/ordinal one-factor fixture:
   - compare `m_i` to current EAP scores;
   - compare `prmse_hat` to an R/mirt empirical-reliability calculation when
     the model class is close enough;
   - simulation check: generated `corr(Z, E[Z|Y])^2` matches PRMSE.
5. Add bootstrap CIs as the first inference surface for both sample PRMSE and
   concrete ordinal reliability.
6. Add analytic delta SEs only after we expose a stable casewise ordinal moment
   influence path for downstream statistics.

## Decision

This is worth doing, but it should land under `measures` / maybe
`measures::frontier` first, not as a generic reliability module. The reliable
core target is scalar ordinal EAP posterior precision, sample PRMSE, and
concrete ordinal reliability. Everything else -- EBM/ML determinacy, CTT
reliability of the EAP score, multi-factor EAP, and analytic SEs -- should stay
explicit so we do not collapse different coefficients under one overloaded
"factor-score reliability" label.
