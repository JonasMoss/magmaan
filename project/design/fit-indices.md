# Ordinary fit indices

Adopted 2026-10-06 (TASK-100). It plans `fit_measures()` for the ordinary
package: RMSEA, CFI, TLI, SRMR (CRMR for ordinal data) and, for likelihood
estimators, the log-likelihood, AIC and BIC. Standardized estimates (TASK-98)
and modification indices (TASK-99) are separate cards.

**User decisions (2026-10-06).** Misspecification-robust point estimators;
every estimator's indices use its own discrepancy and its own robust bias
correction ("DWLS and all other discrepancies in the natural way");
no intervals until an evaluation study (TASK-103); the interface below.
Implementation: TASK-101 (C++ composer), TASK-102 (ordinary interface).

## Targets

Let $F$ be the estimator's discrepancy: the ML discrepancy for ML and FIML,
the diagonally weighted least-squares discrepancy for DWLS. Under the scope's
joint-sampling regime the population misfit is

$$
F_0=\min_\theta \sum_g \frac{n_g}{N}\,F_g\bigl(\sigma_{0g},\sigma_g(\theta)\bigr),
$$

and the indices are functions of it:

$$
\mathrm{RMSEA}_0=\sqrt{F_0/\mathrm{df}},\qquad
\mathrm{CFI}_0=1-\frac{F_0}{F_{0,b}},\qquad
\mathrm{TLI}_0=1-\frac{F_0/\mathrm{df}}{F_{0,b}/\mathrm{df}_b},
$$

with $F_{0,b}$ the population misfit of the independence baseline. With $G$
groups RMSEA uses Steiger's (1998) multigroup definition
$\sqrt{G\,F_0/\mathrm{df}}$, as lavaan does: the pooled discrepancy averages
over groups while df adds over them, so without the factor $G$ identical
groups would shrink RMSEA by $\sqrt{G}$ (found by the TASK-101 lane;
corrected 2026-10-06). CFI and TLI need no such factor. SRMR and CRMR are
residual summaries and need no test statistic.

DWLS indices measure misfit in the DWLS metric. Their population values depend
on the weight and are not comparable to ML values or to ML-derived cutoffs
(Savalei 2021; Xia and Yang 2019). This is a property of the target, not of
the estimator below.

## Point estimators

The fitted discrepancy is biased upward. To first order

$$
E[\hat F]\approx F_0+\frac{\operatorname{tr}(Q\Gamma)}{N},
$$

where $Q$ is the profile Hessian of $F$ at the pseudo-true parameter in
observed geometry and $\Gamma$ the sampling covariance of the moments,
including the estimated-weight channel for DWLS. Three estimators differ only
in the trace:

| Correction | Trace | Valid when |
| --- | --- | --- |
| Naive | df | Correct model, normal data, ML |
| lavaan "robust" (Brosseau-Liard, Savalei and Li 2012; Savalei 2018, 2021) | $\operatorname{tr}(U\Gamma)$ in expected geometry at the fit | Correct model, any data |
| Misspecification-robust | $\operatorname{tr}(\hat Q\hat\Gamma)$ in observed geometry, estimated weight included | Also under fixed misspecification |

All three are consistent for $F_0$, because the correction is $O(1/N)$. They
differ in finite-sample bias, which matters most near good fit, where RMSEA is
truncated at zero, and they reduce to one another at correct specification
(observed and expected geometry coincide) and, for ML, under normality.
Interval width and coverage are where the geometry matters to first order:
under fixed misspecification $\hat F$ has a first-order variance term that the
noncentral chi-square intervals ignore.

## What exists

- **lavaan-style indices**: standard, scaled and robust RMSEA/CFI/TLI with
  intervals and close-fit p-values, SRMR, log-likelihood, AIC, BIC; gated
  against lavaan for ML, MLR, FIML (corrected robust family) and all-ordinal
  DWLS/WLS (`fit_measures()`, `measures/fit_measures.cpp`; see the area files).
  Continuous ML robust dispatch takes caller-supplied scalars; FIML robust is
  automatic.
- **Misspecification-robust family for ordinal and mixed DWLS**:
  `fit_measures_misspec()` and `fit_measures_misspec_mixed_ordinal()` give
  RMSEA, CRMR/SRMR, CFI and TLI with the observed-geometry trace and
  estimated-weight intervals. The Monte Carlo harness
  (`cpp/tests/checks/ordinal_cfi_inference`) reports the CFI interval
  calibrated, the TLI interval conservative at strong misfit, and RMSEA
  intervals sensitive to weight estimation. The R help still says
  single-group only while the area file records multi-group pooling; reconcile.
- **Continuous moment-quadratic tier** (GLS/ULS/WLS): profile-RMSEA primitives
  with the observed-geometry trace.
- **Missing**: the misspecification-robust trace for ML and FIML (the ML
  discrepancy is not a moment quadratic; the observed profile Hessian and the
  casewise score rows already exist in the policy nested machinery), and an
  ordinary composer that selects one recipe per estimator.

## Decisions (adopted as recommended, 2026-10-06)

1. **Point estimator.** Recommended: the misspecification-robust correction for
   every estimator, gated to reduce to lavaan's robust indices at correct
   specification. Alternative: lavaan's robust correction (cheaper; no new
   ML/FIML work).
2. **Intervals and close-fit p-values.** Recommended: none in the first
   release; add them after a registered decision study. The existing ordinal
   evidence supports a CFI interval at most.
3. **DWLS indices.** Recommended: report them, labelled as DWLS-metric
   indices, with the comparability caveat in the help; CRMR as the residual
   index. Alternative: report only CRMR for ordinal data.
4. **Interface.** Recommended: `fit_measures(fit, lavaan_compat = NULL)`
   returning one row per index (`index`, `estimate`, `reason`; interval
   columns later), computed on request because the baseline is a second fit.
   `summary(fit, fit_measures = TRUE)` prints them. `lavaan_compat = "MLR"`
   (and the other bundles) returns lavaan's fitMeasures conventions from the
   existing gated helpers, for simulations that compare with lavaan.
5. **Baseline.** The independence model fitted with the same estimator and
   data treatment (FIML for incomplete data; free thresholds for ordinal
   data), per group and pooled, unconstrained across groups as in lavaan.

## Validation

- Reductions: at a correct model the observed-geometry trace equals the
  expected-geometry trace, so the indices equal lavaan's robust indices
  (`lavaan_compat`); under normal ML they equal the naive ones.
- Point-estimate bias study (registered, simbox): misspecified populations
  with exactly computed $F_0$ (fit the population), ML, FIML (MAR) and DWLS,
  several N; compare the three corrections by bias and RMSE of RMSEA and CFI.
  The decisions/07 frozen textbook populations can be reused for DWLS.
- Interval study later, if decision 2 adds intervals.

## Work

1. C++: misspecification-robust trace for ML and FIML; policy fit-index
   composer over ML, FIML, all-ordinal and mixed DWLS reusing the existing
   ordinal/mixed family, baseline fits, SRMR/CRMR and likelihood criteria.
   Medium to large.
2. Ordinary `fit_measures()`, `summary(fit_measures = TRUE)` and the
   `lavaan_compat` route, with reduction and lavaan gates. Medium.
3. The registered bias study. Medium; about a simbox evening.
