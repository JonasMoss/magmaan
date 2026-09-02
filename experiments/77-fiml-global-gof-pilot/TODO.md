# Paper-design TODO

This file is a temporary design ledger for a possible *Psychological Methods*
paper. It is not a preregistration. The current experiment remains an
exploratory pilot until the three gates below are resolved and the final design
is frozen.

## Scope decision

- [x] Make projected-score pEBA(4), with score SB as the simple benchmark, the
  main method story.
- [x] Treat LR corrections as a separate statistic/correction interaction, not
  as interchangeable versions of the score test.
- [x] Keep ML2S as a secondary negative result unless a distinct two-stage
  question emerges.
- [x] Drop the multiplier from the planned paper comparison. Preserve existing
  multiplier results and diagnostics, but do not spend the publication design
  on further multiplier variants.
- [x] Smoke-audit the opt-in full-observed global-score construction, which uses the
  realized H0 information for the nuisance projection, score quadratic, and
  robust-spectrum bread. In 300 `n=120` fits across the five pilot models,
  normal/VM/IG generators, and complete/MAR data, 74 calls failed because the
  projected information was not positive definite. Among the 226 usable calls,
  score pEBA(4) rejected 58%, compared with 2.7% for the observed-sensitivity
  score retaining the Fisher metric. Keep the implementation as a diagnostic,
  but do not put this variant in the planned paper battery.
- [x] Add a statistic-by-spectrum crossing diagnostic for FIML LRT/FMG versus
  projected-score pEBA(4). In a 6,000-fit one-factor pilot, neither the base
  statistic nor the spectrum alone reproduced the expected-score calibration;
  the native score pairing was required. The observed-sensitivity crossing
  showed much stronger opposing movements and 63 non-positive-definite tangent
  failures. See `notes/iteration-10-statistic-spectrum-crossing.md`.
- [x] Approximate the H0 and saturated H1 Gaussian-FIML pseudo-targets under
  VM2/IG2 MAR with 200,000-case draws, and run a Gaussian shadow null at the H0
  pseudo-true moments. The original arms retain positive per-case LR
  discrepancy and reject; the artificial Gaussian shadows return near nominal.
  See `notes/iteration-11-pseudotrue-shadow.md`.
- [x] Express the H1 pseudo-target outside H0 using zero-under-H0 directions.
  For the one-factor pilot these are descriptive off-diagonal shadow residual
  covariances plus parameterization-free tetrad violations; literal
  cross-loadings require a multi-factor follow-up.
- [x] Retire the incomplete `power-v1-20260821` Modal continuation without
  repairing its three partial focus cells. Collect and combine the available
  rows only as an explicitly incomplete exploratory snapshot. The next
  production run will use Slurm after the design gates are frozen.

## Gate A: five broad, sourceable SEMs

The provisional panel is documented in
[`notes/iteration-08-paper-design.md`](notes/iteration-08-paper-design.md). The
working set is:

1. six-indicator one-factor CFA;
2. 15-indicator correlated three-factor CFA;
3. 12-indicator orthogonal bifactor model;
4. nine-indicator latent mediation SEM; and
5. five-wave linear growth model.

Before changing the simulation code:

- [ ] Confirm that these five cover distinct applied uses without wasting a
  slot on a second near-duplicate CFA. Keep the Bollen political-democracy SEM
  as the named substitute if the generic mediation model is judged too tidy.
- [ ] Give every model a provenance record containing the source, exact source
  specification, every adaptation, and the final population parameter table.
  A citation to a similar model is not enough.
- [ ] Freeze the deterministic reduction of Savalei and Falk's 21-indicator
  model to at most 15 indicators; check identification, conditioning, degrees
  of freedom, and standardized residual variances after reduction.
- [ ] Decide whether the bifactor null sets the source paper's cross-loadings
  to zero and reserves them for power. Label that explicitly as an adaptation.
- [ ] Find a published numeric population for the five-wave growth model, or
  derive and archive a population fit from a named public data example. The
  current convenient growth parameters are not sufficiently sourced.
- [ ] Run a small complete-normal fit gate after the source-based populations
  replace the pilot populations. Require positive-definite covariance matrices,
  expected numerical rank/df, and stable FIML fits at the smallest planned
  sample size.
- [ ] Freeze the five models before tuning power. Do not select models based on
  which method wins their pilot cells.

## Gate B: literature-grounded MAR mechanisms

- [ ] Reproduce Savalei and Falk's MCAR, MAR-L, MAR-L2, and MAR-NL rules,
  including their distinction between few and many missing-data patterns.
  Adapt the variable blocks deterministically to each model and document the
  mapping.
- [ ] Decide which missing rates are primary. Their 15% and 30% conditions are
  the default candidates; a smaller paper grid may retain 30% as the main
  stress and use 15% only for sensitivity.
- [ ] Keep complete and MCAR controls. Separate normal-MAR cells, where the
  Gaussian observed-data likelihood is correctly specified, from nonnormal-MAR
  estimand stress.
- [ ] Add one deliberately adverse but still MAR diagnostic based on the
  Yuan--Bentler/Savalei warning: selection on an always-observed, skewed tail
  that is strongly related to the variables made missing. Calculate the
  Gaussian-FIML pseudo-true discrepancy from the generating SEM before running
  Monte Carlo.
- [ ] Do not call nonnormal-MAR rejection ordinary Type-I error when Gaussian
  FIML targets a different pseudo-true moment structure. Report it as estimand
  drift/model-target failure, with parameter and population-discrepancy
  diagnostics.
- [ ] Calibrate the achieved marginal missing rate and pattern distribution for
  every model/mechanism pair; save these population or very-large-sample checks
  before the production run.
- [ ] Freeze mechanism formulas, thresholds, conditioning variables, and
  missing-variable blocks before examining method rankings.

Primary sources are Savalei and Falk (2014),
<https://doi.org/10.1080/10705511.2014.882692>; Savalei (2008),
<https://doi.org/10.1080/10705510701758091>; and Yuan and Bentler (2000),
<https://doi.org/10.1111/0081-1750.00078>.

## Gate C: one or two empirical disagreement examples

- [ ] Define a small, public, reproducible candidate corpus before screening.
  Each entry needs a substantive model source, data source, redistribution or
  access terms, missing-data description, and a reason that treating its
  indicators as continuous is defensible.
- [ ] Predeclare the screening rule. The target pattern is MLR global rejection
  at 5% with projected-score pEBA(4) nonrejection, but selection must also
  require stable convergence, adequate effective sample/pattern counts, and a
  substantively interpretable fitted model.
- [ ] Report the entire candidate-corpus screen, not only the one or two
  examples used in the text. This keeps the examples illustrative rather than
  post-hoc evidence of prevalence.
- [ ] For selected examples, diagnose the disagreement: raw statistics, df,
  correction eigenvalues/moments, score contributions, influential missingness
  patterns, and sensitivity to SB and the viable LR correction.
- [ ] Prefer naturally incomplete public data. If missingness is imposed on a
  complete public data set, label it a semi-empirical illustration rather than
  a real-data discovery.
- [ ] Keep discovery and final illustration separate where possible: use one
  corpus to locate candidates and a held-out or independently motivated example
  to show transportability.

## Production decision after the gates

- [ ] Build the replacement production harness for Slurm. Do not resume the
  retired Modal run or treat its completed stress phase as publication data.
- [ ] Choose the final sample-size, generator, MAR, and power grid only after
  Gates A and B are frozen.
- [ ] Recalibrate sparse and diffuse alternatives for the source-based model
  populations. Use method/cell-specific empirical-null cutoffs for the primary
  power comparison and retain nominal power as a diagnostic.
- [ ] Write a short simulation protocol stating primary estimands, primary
  cells, exclusion/failure handling, Monte Carlo precision, and multiplicity of
  descriptive comparisons before launching the publication run.
