# DWLS spectral reference law: registered criteria

Registered before smoke or pilot. TASK-86 prepares a study; it does not
change a default or authorize production. Smoke and pilot are development
controls. Confirming evidence requires fresh production draws and merger review.

## Population construction

Eight model families use the textbook corpus parameter summaries, not paper
code or data. Keep the published covariance/regression structure, remove mean
structure and derived quantities, reconstruct the full implied covariance,
and standardize every variable (including factors). Impose the following
exact null on the standardized coefficients, then adjust observed residual
diagonals to retain unit response variances. Preserve all other parameters.
Do not repair a nonpositive residual covariance or population correlation:
stop and request a design decision. Record source hashes and minimum eigenvalues.

| Model | Null restriction |
|---|---|
| Brown PTSD, 8 items | Equal second and third standardized loadings within each factor (2 df) |
| Brown MTMM correlated uniqueness, 9 items | Equal inventory and interview loadings per trait (3 df) |
| Brown higher-order coping, 12 items | Equal second and third loadings on the first three first-order factors (3 df) |
| Brown bifactor perceived control, 15 items | Equal standardized specific loadings within each specific factor (9 df) |
| Little longitudinal 2 constructs x 3 waves, 18 items | No lagged residual covariances (18 df) |
| Little longitudinal 2 constructs x 5 waves, 30 items | Equal standardized item loadings across waves per construct; marker rescaling gives weak invariance (16 df) |
| Kline Worland structural regression, 11 items | Equal second and third Cognitive loadings, preserving structural paths (1 df) |
| Brown MDD two-group CFA, 9 items | Identical replicated group-1 latent-response population; configural-to-metric step and configural-to-Wu–Estabrook threshold step |

The MDD replication deliberately removes published group mean and variance
contrasts: after separate response standardization these need not define the
ordinal metric null. This is a controlled invariance population anchored to
the published first-group solution, not a replication of the book's group contrast.
Use marker identification; retain published zero covariances. Within-factor
equalities translate to marker coordinates after population standardization.
All-ordinal theta parameterization is fixed for every fit. Binary threshold
steps may be equivalent models and must retain that typed unavailability.

## Design

Gaussian latent responses only. Cross all models with categories 2, 5, 7,
symmetric or skewed probabilities and total N 250, 500, 1000, 2000. Split
MDD N equally. Symmetric probabilities are uniform. Skewed probabilities:
2: (.85,.15); 5: (.45,.25,.15,.10,.05);
7: (.35,.25,.15,.10,.07,.05,.03). All items use the cell's same shape;
thresholds are standard-normal cumulative quantiles, identical across groups.
Each cell uses 1000 production draws. Global H1 and every applicable nested
H0 are evaluated on the same draw. Library fit$converged is the judge; use
ordinary DWLS defaults without optimizer overrides, exact first-stage policy
inference and current policy nested composers. Never substitute another law
when a policy test is unavailable. Save statistic, df, full spectrum, convergence,
seed and typed failure for each draw/test, including unavailable tests.

References, all evaluated on each saved spectrum through robust_fmg_test:
All, SB, PEBA4, scaled-shifted, mean-variance, scaled-F, EBA2/4/6,
pEBA2/4/6, pOLS and pAll. All truncates negative eigenvalues, matching the
current global policy. Other references use their library definitions.

## Comparison rule

Per cell/test/reference report attempted, available and failed counts,
successful-denominator rejection at .05 with Wilson 95% intervals, and
attempted-denominator rejection. Never conceal failures or pool away model
losses. At N >= 500 flag rates outside [3%,7%]. Summarize outside-band counts
and RMSE of rejection minus .05 per model and overall, separately for
N >= 500 and N < 500, global and each nested step. Compare on jointly
available reference draws; retain marginal availability separately.

For TASK-81 prefer the spectral reference with fewest outside-band cells
at N >= 500, ties broken by RMSE. Report every per-model loss relative to
All (global) and SB/PEBA4 (nested). Penalized or approximate references must
still use the identical retained policy spectrum; no reference ranking
licenses changing the statistic, first stage, meat or restriction map.
Missing cells, unavailable tests, unresolved numerical failures or severe
availability differences leave the affected choice open. Pilot flags do not
select a reference. The user decides adoption; no automatic default change.

## Seeds, execution and cost

Bases: smoke 210000001, pilot 410000001, production 1910000001.
Seed = base + 10000*cell_id + replicate. 192 stable cells give disjoint
phase ranges, below the R integer limit and distinct from decisions/05–06.
One smoke draw per selected cell; pilot 2 draws per selected pricing cell.
Pilot is capped at two workers, one math thread each, nice priority and
approximately ten minutes; a representative subset is allowed. No production.

Per-cell production invocation: --production --cell ID --workers 1
--out-dir ATTEMPT. Write COMPLETE last; combine rejects source, binary,
population, design, seed, count or completion discrepancies. Selected IDs
and seeds remain stable under fan-out. Pricing reports observed cells and
unpriced cells, scaling each measured model/category/shape/N cell to 1000
replicates; never claim a full-grid estimate from unmatched timing. If a
credible grid estimate exceeds about 45 CPU-hours, propose dropping N=2000
or seven categories for the largest model in a dated amendment before
production. Simbox remains unavailable; do not launch there or on Modal.

## Amendment 2026-10-05, before smoke

Approved in TASK-86 comment 5 after population validation, before simulation:
the averaged Risk-path population is not positive definite (standardized paths
−1.59/−2.31; response-correlation minimum eigenvalue −0.217). Replace that
null with equal second/third Cognitive loadings, preserving published paths.
Apply the same validity gate; if it fails, retain Worland global cells only.

## Cost amendment 2026-10-05, after development pilot, before production

The complete 192-cell, two-draw-per-cell development pilot projects 72.56
CPU-hours for 1,000 draws per cell. Under the registered cost rule, omit the
seven-category cells of long30 from production (IDs 22, 46, 70, 94, 118,
142, 166, 190). Retain all other cells and their original IDs/seeds: 184 cells,
184,000 draws, projected 44.54 CPU-hours (ideal 3.71 h on 12 cores or 2.78 h
on 16 cores). This is a cost-driven amendment after seeing development timing
and availability, not a reference selection. Comparator evaluation, I/O and
startup add overhead; two timing draws per cell leave substantial uncertainty.
The pilot still reports all 192 cells. Seven-category long30 conclusions stay
open; no pooling may imply they were confirmed. No production is authorized.
Pilot source and criteria are recoverable at commit 021d31bc. Combining those
old pilot attempts requires that version; current provenance checks deliberately
reject it after the production amendment. Reproduction under current sources
requires fresh attempts, and production uses the new fingerprint and fresh base.
