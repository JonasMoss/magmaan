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
\mathrm{RMSEA}_0=\sqrt{G\,F_0/\mathrm{df}},\qquad
\mathrm{CFI}_0=1-\frac{F_0}{F_{0,b}},\qquad
\mathrm{TLI}_0=1-\frac{F_0/\mathrm{df}}{F_{0,b}/\mathrm{df}_b},
$$

with $F_{0,b}$ the population misfit of the independence baseline. With $G$
groups RMSEA uses Steiger's (1998) multigroup definition
$\sqrt{G\,F_0/\mathrm{df}}$, as lavaan does: the pooled discrepancy averages
over groups while df adds over them, so without the factor $G$ identical
groups would shrink RMSEA by $\sqrt{G}$ (found by the TASK-101 lane;
corrected 2026-10-06). CFI and TLI need no such factor. TLI uses the nominal
df, as in its population definition; the lab's ordinal interval family uses
traces as generalized df, which targets a distribution-dependent quantity and
stays a lab comparator. SRMR and CRMR pool squared residuals with weights
$n_g/N$ before the root, like the discrepancy, and get the same
misspecification-robust bias correction,
$\sqrt{\max(\lVert\hat r\rVert^2-\operatorname{tr}(\widehat{\mathrm{Var}}\,\hat r),0)/k}$
in the pooled metric; lavaan's average of per-group roots stays in the
compatibility route (TASK-101 lane questions, answered 2026-10-06).

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
including the estimated-weight channel for DWLS. Under misspecification a
second first-order term adds $\nabla_s F^\top b/N$, where $b$ is the
first-order bias of the moment estimator (for example $-\Sigma$ for the
$N$-divisor covariance); it vanishes at exact fit. The likelihood (Takeuchi)
trace difference used for ML and FIML contains it automatically (found by the
TASK-101 lane). The least-squares tiers add it where $b$ is known in closed
form (the sample covariance); for polychoric and polyserial moments $b$ is not
derived, so it is omitted and documented, and TASK-103 measures the remaining
bias. Three estimators differ in the trace:

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
  intervals sensitive to weight estimation. R help now describes multi-group
  pooling; the Exact selector preserves OPG defaults.
- **Continuous moment-quadratic tier** (GLS/ULS/WLS): profile-RMSEA primitives
  with the observed-geometry trace.
- **Policy point composer (TASK-101)**: `api::policy_fit_measures()` and
  `magmaanlab::policy_fit_measures()` compose ML/FIML Takeuchi trace differences,
  continuous ULS profile traces plus N-divisor covariance bias, and ordinal /
  mixed DWLS observed-geometry corrections with exact first-stage rows.
  ML2S-NT composes its Stage-1-Gamma profile points; its residual adapter
  and non-NT recipes remain typed unavailable. Independence baselines are
  unconstrained across groups and use the same
  estimator/data treatment. Per-index unavailable reasons preserve fit-state
  gating. The lab result has index/estimate/reason columns and a details
  attribute with discrepancies, traces, corrected values and nominal df.
  Residual points subtract the residual influence trace in the pooled metric.
  GLS/WLS estimated-weight profile corrections, ordinal ULS/WLS and non-NT ML2S
  policy composition remain unavailable with named missing ingredients.
  Numerical reductions and influence checks are component evidence; point
  bias and interval calibration remain TASK-103 work.
- **Exact lab comparator**: the ordinal/mixed misspecification primitives and
  lab wrappers accept an explicit Exact first stage. OPG remains the default;
  the consolidated family's residual fields remain uncorrected and its TLI
  retains generalized df. Policy residuals instead use the primitive's
  `point_bias_corrected`; policy TLI uses nominal df and truncation.

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
   `summary(fit, fit_measures = TRUE)` prints them. TASK-102 implements the
   policy route. The compatibility composer is split into TASK-109 (decision
   2026-10-06): non-NULL `lavaan_compat` errors clearly until it is available.
   It will return lavaan's fitMeasures conventions for simulation comparisons.
5. **Baseline.** The independence model fitted with the same estimator and
   data treatment (FIML for incomplete data; free thresholds for ordinal
   data), per group and pooled, unconstrained across groups as in lavaan.

## Validation

- Reductions: at a correct model the observed-geometry trace equals the
  expected-geometry trace, so exact-fit RMSEA/CFI equal lavaan's robust points and policy TLI equals
  its truncated comparator; no baseline trace equality is asserted under
  independence-model misspecification.
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

## Component validation conventions

The ML/FIML correction uses observed likelihood Hessians and raw casewise
score cross-products. Covariance-only complete ML profiles out unrestricted
means; their nuisance trace is included in the model and saturated traces.
The independent complete-ML calculation includes both the profile-Hessian
trace and the first-order N-divisor covariance bias. Plain profile equality
is used at exact fit, where the discrepancy gradient and bias term vanish.
At exact fit RMSEA/CFI reduce to lavaan robust points; truncated policy TLI
reduces to `min(lavaan_tli_robust, 1)`. The interval family's generalized-df
TLI remains unchanged. Mixed DWLS's Exact comparator also includes the known N-divisor bias of
its continuous variance and continuous-continuous covariance moments. The
OPG default is unchanged. Polychoric/polyserial and EM-moment first-order
bias terms remain omitted because those terms have not been derived.

The Exact residual primitives use the full pooled residual influence map,
including sqrt(n_g/N) weights and cross-group parameter influence. The legacy
OPG calculation omitted those terms; its default output is preserved. This
is an identified missing-term correction under the TASK-101 standing rule.

Numerical gates pass for the independent ML profile Hessian plus moment bias,
exact-fit MLM reductions (TLI truncated), Exact DWLS primitive points,
unconstrained group duplication and likelihood helpers. Delete-one residual
trace comparisons give relative gaps 0.0344136 (ML) and 0.0210847 (FIML),
both within the specified 0.05 gate. These are component gates rather than a
point-bias calibration. `details$srmr_trace` is normalized to the pooled mean
squared residual metric, so SRMR is
`sqrt(max(srmr_uncorrected^2 - srmr_trace/N, 0))` on every supported route.
