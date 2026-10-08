# Fit-index evaluation study (DRAFT, not registered)

Draft 2026-10-08 for TASK-103 and the planned `experiments/decisions/09-fit-indices`.
**Status: design draft only.** Registration waits for design principles the user
and planner will set together, including the shared, tiered simulation setup
the user wants for all studies (see [simulation backlog](../backlog/simulation.md)).
Nothing here is a registered rule, and no run is authorized.

**User direction (2026-10-08).** Use representative models as the basis, as
in the sem-score study, with bespoke handling for further models. Keep MAR
simple, as in common practice (Savalei and Falk style), and reason about it
properly later. Prefer a unified simulation setup with complexity tiers, since
these designs recur across studies.

## Questions

1. **Points.** Do the adopted misspecification-robust point estimators
   ([fit indices](fit-indices.md)) have the smallest bias and RMSE among naive,
   lavaan-robust and misspecification-robust corrections, per index (RMSEA,
   CFI, TLI, SRMR/CRMR), estimator (ML, FIML, DWLS) and population family?
2. **Intervals.** Which interval, if any, should the ordinary package report per
   index and estimator? Candidates: lavaan's noncentral chi-square intervals
   (naive and scaled); a misspecification-robust Wald interval on $\hat F$ with
   the observed-geometry variance, including the first-order variance term
   under fixed misspecification, on a transformed scale (for example
   $\sqrt{\cdot}$ for RMSEA); and the existing ordinal estimated-weight
   intervals as comparators. Close-fit p-values are a later, separate question.

## Targets

Population values come from fitting population moments exactly, as in
decisions/07 and 08: $F_0$, $F_{0,b}$ and the indices defined in
[fit indices](fit-indices.md) (TLI with nominal df). The DWLS indices measure
misfit in their own metric and are compared only with their own targets.

**Open (FIML under MAR).** Under misspecification, FIML converges to the
minimizer of the expected observed-data log-likelihood under the missingness
mechanism, which generally differs from the complete-data pseudo-true value.
FIML indices therefore estimate a mechanism-dependent population discrepancy,
not the complete-data $F_0$, and the gap is $O(1)$, not a finite-sample bias.
The draft proposes reporting both targets: the complete-data $F_0$ (what users
mean by model misfit) and the FIML population discrepancy (what the estimator
is consistent for). For normal data under threshold MAR the latter has a
closed form via truncated-normal moments; non-normal generators use large
population draws as in decisions/08. This is part of the deferred MAR
reasoning.

## Populations

**Tier 1, representative basis.** The sem-score representative models (ptsd8,
mtmm9, second12, bifactor15, long18, long30, mimic7, worland11, growth6,
mg_path5, mgcfa_mdd9), each with ONE natural misspecification in the spirit of
that study's one-restriction rule: the generating model has an extra parameter
(an omitted cross-loading, residual correlation or cross-lag, whichever is
natural for the model) that the fitted model omits. Its size is set to hit
population RMSEA levels, for example 0 (correct), .03, .05 and .08. The model
definitions move into `experiments/_support` as shared harness data, because
the layering rules forbid importing from a paper repository.

**Tier 2, bespoke additions.** Model-specific extensions where a question needs
them: the frozen decisions/07 textbook populations for DWLS, multigroup cells
for pooled-index conventions, and small-df models where RMSEA truncation
dominates.

**Generators.** Normal, Vale-Maurelli, independent-generator and discretized
data for continuous estimators, matching the sem-score generators; thresholded
latent responses with Gaussian and skewed latents for DWLS, as in decisions/08.

**Missing data (FIML).** Base tier: simple single-cause MAR in the style of
Savalei and Falk, with missingness on a subset of variables driven by fully
observed variables and two missingness rates plus complete data. Richer
mechanisms belong to a higher tier, after the MAR reasoning.

**Sample sizes.** For example N = 200, 500, 1000 and 4000, keeping a small N
where truncation near zero matters.

## Arms and judge (to be fixed at registration)

- Point arms: naive, lavaan-robust and misspecification-robust corrections,
  same draws. Judge: bias and RMSE against the exact target per index,
  estimator, family and N, never pooled; a pre-specified margin decides "smallest".
- Interval arms: the candidates above at the conventional 90% level for RMSEA
  (to be decided for CFI and TLI). Judge: coverage, with flags outside a
  pre-specified band, plus width and availability. Truncation at zero needs an
  explicit rule.
- Default fitting options and `fit$converged`; failures stay in attempted
  denominators.

## Open questions for the design-principles session

1. The FIML target under MAR (complete-data versus mechanism-dependent) and the
   MAR base tier.
2. Misfit construction: one natural extra parameter per model, versus a common
   recipe; misfit levels set by RMSEA or by $F_0$.
3. Reuse of the sem-score basis as is, or a trimmed tier-1 subset to price.
4. Interval candidates, confidence levels and truncation conventions.
5. The margins that decide "smallest bias" and acceptable coverage.
6. How this study instantiates the shared tiered simulation setup.

## Cost

Unknown until a pilot. Production runs on simbox and needs explicit
authorization.
