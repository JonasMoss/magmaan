# Structured-mean lane: registration draft, 2026-10-05 (TASK-75)

Decision: calibrate the adopted observed nested ML policy after TASK-66 under
restricted-mean misspecification. No default change. Merger reviews/amends this
registration before held-out production on Modal; no production is authorized
by this lane task. Smoke and pilot are development/pricing, never evidence for
adoption. Baseline is ordinary magmaan() ML without fitting overrides, with
library convergence verdicts. Score-first reporting remains the adopted policy.

## Problems and arms

Two groups, one factor with six indicators or two factors with six each.
Standardized loadings .7, .8, .75, .65, .7, .8; factor variances one,
correlation .4 for two factors; residual variances 1-loading squared. Larger
model has equal observed intercepts across groups and both latent means fixed
zero. Null adds equality of nonmarker loadings (5 or 10 restrictions).

Correct-mean control: zero means. Mild: group means -.1 and +.1 on item 1
(.2 SD difference). Strong: -.2 and +.2 on items 1 and 2 (.4 SD differences).
At the symmetric pseudo-true solution, shared intercepts are zero and each
group contributes the same covariance plus mean-shift outer product. The
loading-equality null therefore holds there. Verify the symmetric population
solution and its local observed curvature before production; symmetry alone
does not prove global uniqueness. Power lowers group b's x3 loading by .15,
with residual variance adjusted to preserve unit variance; report its actual
pseudo-true restriction departure before production.

Normal or the parent's Vale-Maurelli skew generator (skew 2, excess kurtosis 7).
N per group 100, 300, 1000. Null and matched power: 72 cells. Held-out production:
2000 null and 1000 power draws/cell (108000 draws). Correct controls do not gate.

Arms: observed policy LR and score, SB and PEBA4, plus explicit pre-66 LR
with centered sample covariance/mean influence transported through fitted NT
weights and the model Jacobian. All LR arms share statistic, observed bread,
and restriction map. The old arm omits the fitted-mean linear/constant
likelihood-row terms. A numerical-Jacobian exact-row reconstruction must match
the policy LR spectrum (relative maximum discrepancy <=1e-6), independently
checking projection/scaling. Failures are saved against all requested draws;
no study-specific optimizer acceptance rule.

## Registered flag rule

As in nested_geometry.md, flag any misspecified null cell where adopted LR's
absolute nominal 5% size error exceeds the comparator's by >1 percentage point
and a paired 95% percentile bootstrap interval excludes zero (10000 resamples).
This is a reporting flag for finite-sample work, never a return to the
inconsistent old meat. Report every family/cell/calibration and every loss;
Wilson intervals condition on available arms. Score SB/PEBA4 size and
size-adjusted power are reported, not gated by a LR-comparator rule. Power
uses its matching null p-value 5% quantile (type 1). No pooling hides losses.
Pilot decision always stays open_development_only.

## Seeds and pricing

Earlier metadata bases: 526100021 (smoke), 626100031 (pilot and pilot-strong),
726100041 (production and score replays); earlier summary bootstrap 826100051.
New bases: 1326100021 smoke, 1526100031 pilot, 1726100041 production,
1926100061 summary bootstrap; each is at least 100000000 from prior bases and
from one another. Draw seed = base + 10000*stable cell ID + replicate.
Smoke two draws/cell. Pilot twenty draws/cell, two workers, nice, one math
thread/worker, <=approximately ten minutes; a documented subset may price
production without claiming full-grid calibration. Summaries, failures,
cell design, cost and metadata are frozen; raw rows stay ignored. Fingerprints
cover runner/helpers/criteria and native binary. Production needs fresh seeds
and a committed, merger-approved registration; any amendment after pilot must
be dated and explicitly post-results.
