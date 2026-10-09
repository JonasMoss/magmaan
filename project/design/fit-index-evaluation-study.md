# Fit-index evaluation study (design settled, registration pending)

Draft 2026-10-08 for TASK-103; design settled with the user 2026-10-09 under the
shared [simulation study principles](simulation-study-principles.md). The study
is `experiments/decisions/09-fit-indices`. Its criteria are committed before
the decision run, after the harness and a pricing pilot exist; no run is
authorized yet.

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

## Decisions (user, 2026-10-09)

1. **FIML target under MAR.** Bias is judged against the estimator's own
   target, the missingness-dependent FIML population discrepancy; the
   complete-data $F_0$ is reported beside it as what users mean by misfit. The
   base MAR tier: missingness on half of the variables, driven by one fully
   observed variable, at 0, 15% and 30%.
2. **Misfit construction.** One natural omitted parameter per model, sized to
   population RMSEA 0, .03, .05 and .08 (ML metric). Many-small-misfit
   populations are a later tier.
3. **Basis.** All tier-1 models for the point arms, priced by a pilot; if
   intervals are expensive, a 4-5 model subset for the interval arms, chosen
   before any results. Tier-2 additions as listed under Populations.
4. **Intervals.** 90% for RMSEA, CFI and TLI. RMSEA intervals are clipped at
   0, and a population RMSEA of 0 counts as covered when the lower bound is 0.
   Candidates: lavaan's naive and scaled noncentral chi-square intervals; the
   misspecification-robust Wald interval below; the existing ordinal
   estimated-weight intervals for DWLS.
5. **Margins.** "Smallest" means an RMSE more than 10% (relative) below every
   competitor, otherwise a tie; 90% coverage is flagged outside 87.5-92.5%,
   with width and availability reported.
6. **Shared setup.** Tier 1 is the representative basis held as data in
   `experiments/_support`; misfit sizing, targets, arms and the judge belong to
   the study.

## The misspecification-robust Wald interval (to build)

For likelihood estimators the sample discrepancy is a likelihood-ratio
contrast against the saturated model, so by the envelope theorem its first-order
influence under fixed misspecification uses casewise log-likelihoods at the two
fitted points only:

$$
\hat F=\frac{2}{N}\sum_i\bigl(\ell_i^{\mathrm{sat}}-\ell_i^{H_0}\bigr),\qquad
\widehat{\mathrm{Var}}_1(\hat F)=\frac{4}{N}\,\widehat{\mathrm{Var}}_i\bigl(\ell_i^{\mathrm{sat}}-\ell_i^{H_0}\bigr).
$$

This term vanishes at correct specification, where the second-order term
$2\operatorname{tr}\bigl((\hat Q\hat\Gamma)^2\bigr)/N^2$ of the robust trace
machinery dominates, so the candidate uses their sum. The interval is formed
for $\sqrt{F}$ around the bias-corrected point and mapped to RMSEA, CFI and
TLI (CFI and TLI by the delta method with the baseline discrepancy's own
influence and their covariance), clipping at the parameter boundary. It is a
lab-only candidate until the study decides.

## Next steps

1. Tier-1 basis as data in `experiments/_support`, with a proposed natural
   misspecification per model (lane).
2. The continuous misspecification-robust Wald interval for ML and FIML as a
   lab function (lane).
3. The decisions/09 harness: population construction and misfit sizing,
   generators, MAR, targets, arms, per-draw outputs and a smoke run (lane).
4. Criteria committed (planner, user review), pricing pilot, then production
   on simbox with the user's go.

## Cost

Unknown until a pilot. Production runs on simbox and needs explicit
authorization.
