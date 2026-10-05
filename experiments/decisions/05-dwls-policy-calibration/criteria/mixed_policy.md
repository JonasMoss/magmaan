# Mixed DWLS calibration registration draft

Registered 2026-10-05 before smoke or pilot; merger review is required before
production. This lane calibrates TASK-71's mixed policy. It does not select a
new recipe: the default uses exact first-stage sampling rows, estimated-weight
IJ covariance, All for the global statistic, and the parameter-space nested
law with SB/PEBA4. Flags call for finite-sample work, never recipe switching.
Earlier all-ordinal criteria, seeds and evidence are unchanged.

## Population and design

Gaussian latent responses, two correlated factors (.3), six indicators per
factor. Loadings cycle .65, .70, .75, .80, .70, .75; residual variances give
unit response variance. Within each factor the first three indicators are
continuous and the last three ordinal (x4:x6 and x10:x12). This keeps both
measurement types on each factor. Continuous means are zero. Ordinal designs
are binary symmetric/skewed and five-category symmetric/skewed: probabilities
(.5,.5), (.85,.15), (.2,.2,.2,.2,.2), (.45,.25,.15,.10,.05).
N is total, split equally in two-group cells. Identification is marker-variable.

- Global null: all four ordinal designs, N {300,500,1000,2000}, one group delta
  and two identical groups delta/theta. Policy All; comparators SB/PEBA4 on the
  same spectrum and installed lavaan's default mixed WLSMV scaled-shifted test.
- Nested null: two identical groups, all four designs, all four N, delta/theta,
  metric versus configural; larger model correct or omitting a .3 cross-loading
  from x7 on f1 in both groups. Additionally five-category theta cells compare
  thresholds equal versus configural (Wu–Estabrook moment nesting). Binary
  threshold equality has no independent threshold-spacing restriction and is
  excluded from that family. Policy SB/PEBA4; installed lavaan lavTestLRT where
  available. Typed unavailability is reported, not substituted.
- Coverage: single-group delta, four ordinal designs, N {300,1000}, omitted
  x7-on-f1 cross-loading {0,.2,.4}. Targets: continuous loading x2, ordinal
  loading x5, factor correlation, x5 first threshold, continuous intercept x2.
  Policy IJ and lavaan robust SEs. An OPG-first-stage IJ comparator is unavailable
  for mixed data in the current lab interface; fixed-weight covariance is not
  an equivalent substitute. Pseudo-true targets use one fixed 100,000-row draw
  per group and a population DWLS fit, including its estimated population
  weight; this is a Monte Carlo approximation, not analytic truth.
- Power (reported, not gating): global cross-loading .3 and nested group-b x2
  loading increase .15, symmetric five-category, all four N, matched nulls;
  single-group global and two-group theta metric. Arm-specific empirical null
  95th percentiles give size-adjusted power.

Production uses 2,000 draws per null/coverage cell, 1,000 per power cell. Smoke
uses one draw in a documented representative subset; pilot uses two draws per
full-grid cell or an explicitly recorded pricing subset to stay about ten
minutes. Pilot is mechanics/pricing evidence, never a calibration decision.

## Seeds and reporting

Bases: population 1017000001, smoke 1217000001, pilot 1417000001,
production 1617000001. Each differs by at least 100,000,000 from every
817xxxxxx all-ordinal base and from one another. Draw seed = base +
10000 * cell_id + replicate; population uses the stable coverage-design key ID.
The same fixed population draw is reused across sample sizes and fan-out cells.

Library fit$converged is the acceptance verdict. For lavaan estimates, rebuild
and audit at that estimate using the library verdict; if that adapter is not
available, record a comparator limitation and do not claim identical judging.
Errors, nonconvergence and unavailable arms remain in attempted denominators.
Report attempted-denominator rates and Wilson intervals, plus successful-arm
rates for diagnosis, per cell/target/arm without pooling. Null policy flags:
rejection outside [3%,7%] at N >= 500; policy coverage below 93% at N >= 300.
Missing arms are separately flagged unavailable. Production review must consider
failure rates and population-target approximation before interpreting flags.

## Compute boundary

No production or Modal launch by this lane. Commit this draft before smoke.
Freeze pilot cells, metadata, population targets, summaries, failures and timing;
raw rows stay ignored. Price production in CPU-hours from representative pilot
strata, identifying unpriced cells. Local two-cell fan-out must reproduce the
single-process summary and reject mismatched source/package/seed provenance.
