# Regular-interior Newton audit and L-BFGS controls

2026-09-21. Advisory implementation and control study. No production default
or public audit contract changes. Boundary audits and practical verification
by restarts/independent optimizers remain deferred, as requested.

## Candidate choices

The agreed regular-interior accuracy target is **maximum Newton distance d = 0.01**. In regular ML,
this measures predicted remaining displacement in joint information-standard-
error units. Equivalently, require predicted improvement in twice the total
log likelihood below 1e-4, or total-negative-log-likelihood EDM below 5e-5.
For our per-observation half-discrepancy f, the raw decrement is
`g' H^{-1}g`; multiply by N before comparing with 1e-4. Do not forget that
factor when reusing information matrices or importing a package's tolerance.

This is a proposed accuracy budget, not a universal constant or a proven error
bound. It controls the predicted standardized displacement of every linear
contrast under the same local quadratic/information approximation, without
dividing by parameter count. It does not assert valid Wald inference under
misspecification, at a boundary, or under weak identification. Report the
continuous distance and curvature status. Sensitivity targets are .003 and
.03. Do not gate an indefinite or numerically singular Hessian by a
pseudoinverse or a repaired positive-definite matrix.

The original conservative **candidate L-BFGS stopping profile** in this study is
`ftol_rel=1e-12`, `xtol_rel=1e-10`, retaining NLopt's internal gradient default
and automatic memory setting. This is not yet the production default. The
line-search/domain-recovery defect must be addressed independently; the
extended-backtracking diagnostic is not a portable exposed solver option.

## What the current controls actually do

Inspected NLopt source: the opt build's v2.10.1 dependency.

| Control | Actual role | Candidate action |
|---|---|---|
| magmaan `ftol` | NLopt relative objective-change stopping | Try 1e-12; tighter objective stopping alone cannot override a step stop |
| magmaan `gtol` | NLopt `xtol_rel`, **not** a gradient tolerance | Try 1e-10; expose an accurately named control with compatibility handling |
| NLopt `tolg` parameter | Luksan internal gradient stopping; nonpositive input selects its 1e-8 default | Keep default initially; expose distinctly if needed |
| magmaan `history` | Currently not forwarded by this adapter | Fix/control explicitly; do not assume runs used memory 10 |
| NLopt vector storage | Automatic memory when not supplied; explicitly setting 10/20 changes behavior | Retain automatic setting pending a separate reason to change |
| magmaan `max_iter` | Adapter passes to NLopt `maxeval` | Keep a safety budget; not a convergence tolerance |
| Luksan `mred` | Ten line-search reductions hard-coded internally | Separate domain-recovery job; increasing maxeval or tightening tolerances cannot fix exhausted backtracking |

The internal gradient condition can itself stop L-BFGS independently of the
function and step conditions. Disabling f/x tests and driving only a very
small gradient tolerance increased work and negative return codes without
improving the useful accuracy category here. Tighter is not automatically a
better operational default.

## Experiment

Implementation: `tests/checks/interior_newton/`, using canonical C++ SEM
objective, analytic observed information, start values and equality reduction.
Direct NLopt calls retain terminal points even on negative return codes and
expose all control profiles. This is a diagnostic harness, not a comparison
of production wrapper convergence flags.

Seven settings: one-factor CFA (4 and 12 indicators), two-factor CFA,
weak-loadings/high-correlation CFA, equality-constrained loadings with means,
a latent structural regression with means, and a misspecified one-factor model
with an omitted residual covariance. N=100,1000,100000; three replications;
measurement multipliers .1,1,10. Nine profiles per dataset: 189 paired cases,
1,701 fits per line-search version, 3,402 total. Starts are recomputed by
FABIN for the scaled data, so unit changes test the whole numerical pipeline,
not solely fixed-point invariance. All data are Gaussian. No multistart,
independent optimizer, SE calculation or hypothesis test is performed.

Version 1 uses unmodified NLopt. Version 2 links a diagnostic copy with only
`mred=10` changed to `mred=60`; installed libraries and source are unchanged.
Both executable/library hashes and exact seeds are retained locally.

With stock backtracking, all profiles had only 123 eligible interior endpoints.
With extended backtracking, all profiles had 180. Nine endpoints were improper
weak-model fits at N=100 (three samples in three unit systems); they are not
included in the regular PSD-interior assessment. Missing/indefinite curvature
is not treated as a Newton-audit pass.

Results below use extended backtracking. Pass counts are among 180 eligible
interior endpoints; median evaluation counts use all 189 attempted cases.

| Profile | Pass d <= .01 | Pass d <= .003 | Maximum d | Median evaluations |
|---|---:|---:|---:|---:|
| Current: f=1e-10, x=1e-7 | 179 | 174 | .2542 | 68 |
| f=1e-12 only | 179 | 176 | .2542 | 70 |
| f=1e-14 only | 179 | 176 | .2542 | 70 |
| x=1e-10 only | 180 | 178 | .00503 | 72 |
| **f=1e-12, x=1e-10** | **180** | **180** | **.00231** | **77** |
| Only internal gradient stopping, tolg=1e-10 | 180 | 180 | .00231 | 108 |
| Only internal gradient stopping, tolg=1e-12 | 180 | 180 | .00231 | 119 |
| Candidate f/x, memory 10 | 180 | 180 | .00233 | 138 |
| Candidate f/x, memory 20 | 180 | 180 | .00229 | 89 |

Every eligible endpoint passed the old model-Frobenius cutoff under the current
controls. The exceptional Newton-audit failure was the 12-indicator CFA,
N=100000, replication 3, units=.1. Its old residual was 7.47e-4, but its
Newton distance was .25418. NLopt returned XTOL_REACHED. Tightening only ftol
left this point unchanged; the candidate f/x profile reduced d to 2.86e-5.
The predicted twice-log-likelihood improvement was .0646054, compared with
.0646836 between the two control-profile results. This directly illustrates
the difference between an absolute gradient gate and estimated local accuracy.
It is a paired stopping-control comparison, not an added practical-refit study.

The candidate profile's median evaluations increased from 68 to 77 (~13%).
Median fitting times were .283 and .289 ms; candidate analytic Hessian plus
Newton audit cost .030 ms. These single-run, fixed-order timings are descriptive
and too small for a strong speed claim. Observed-information inference can
reuse that matrix. Eight current-profile and 22 candidate-profile interior
endpoints had negative raw NLopt codes but passed d <= .01. Retaining a
candidate for independent assessment matters; do not equate those counts with
errors returned by magmaan's production adapter, which already salvages some
soft failures.

## Implementation and validation details

The prototype computes the total score G=N*g and total observed information I,
then reduces both by the same linear-equality basis K. Diagonal equilibration
precedes a positive-definiteness/conditioning check and Cholesky solve. No
inverse is explicitly formed. It records d, total EDM, raw maximum predicted
step, equilibrated condition number and linear-system residual. Eligibility
requires positive-definite primitive covariance blocks; no PSD-boundary
interpretation is attempted. Nonlinear equality constraints and multigroup
models were not included in this first study.

Provisional numerical guards are equilibrated condition number <=1e12 and
relative solve residual <=1e-10. These are engineering safeguards rather than
statistical identification thresholds; their validation is separate from the
meaningful d accuracy budget. Diagonal equilibration helps with diagonal unit
changes but does not make the conditioning guard invariant to arbitrary
ill-conditioned mixing of coordinates.

Exact checks verify quadratic error, affine invariance of the decrement,
objective/sample-size scaling, and rejection of singular/indefinite curvature.
Seven directional finite-difference comparisons against analytic SEM gradients
verify Hessian normalization and equality reduction; maximum relative error
was 5.08e-10 in the stock run. Coverage and paired seeds match exactly. The
local corrected summaries and paired control differences are under
`tests/checks/interior_newton/results/extended/`.

## Literature collected and source status (2026-09-21)

Local reference files and URL/SHA-256 receipts are in the ignored
`external/refs/interior-newton/` directory. They are not experiment inputs.

- Dennis, J. E., Jr., Gay, D. M., and Welsch, R. E. (1981).
  *An Adaptive Nonlinear Least-Squares Algorithm*. ACM TOMS 7(3), 348–368.
  DOI: [10.1145/355958.355965](https://doi.org/10.1145/355958.355965).
  Section 6, equations (6.8)–(6.9), motivates small steps relative to SEs
  and then maximizes the standardized displacement over linear contrasts.
  This is the direct methodological precedent for our accuracy scale.
  The published PDF returned HTTP 403. The public article text was inspected
  through the browser's [indexed full-text rendering](https://www.scribd.com/document/45401126/An-Adaptive-Nonlinear-Least-squares-Algorithm);
  the locally downloaded HTML contains only a preview, not the complete paper.
  A scanned [1977 NBER working-paper version](https://www.nber.org/system/files/working_papers/w0196/w0196.pdf)
  was downloaded as `dennis-gay-welsch-1977-nber.pdf`; it is not assumed to
  contain the published version's equation numbering or all of its discussion.
- Pratt, J. W. (1977). *When to stop a quasi-Newton search for a maximum
  likelihood estimate*. Harvard School of Business Working Paper 77-16.
  Not obtained or read directly. Dennis et al. attribute their all-contrasts
  criterion to this paper; cite that attribution as secondary evidence only.
- Belsley, D. A. (1980). *On the efficient computation of the nonlinear
  full-information maximum-likelihood estimator*. Journal of Econometrics
  14(2), 203–225. DOI: [10.1016/0304-4076(80)90091-3](https://doi.org/10.1016/0304-4076(80)90091-3).
  Abstract verified; published PDF returned HTTP 403 and no open full copy
  was located. Its reported weighted-gradient criterion is relevant, but we
  do not attribute a numerical threshold to it without the full text.
- James, F., and Roos, M. (1975). *Minuit—a system for function minimization
  and analysis of the parameter errors and correlations*. Computer Physics
  Communications 10(6), 343–367. DOI:
  [10.1016/0010-4655(75)90039-9](https://doi.org/10.1016/0010-4655(75)90039-9).
  The [public CERN record](https://repository.cern/records/g64tm-4s433)
  has a full-text link, but this environment's download received HTTP 403.
  Instead, obtained James's [1994 MINUIT reference manual, version 94.1](https://root.cern/download/minuit.pdf),
  `minuit-reference-manual.pdf`. This is a distinct source, not the 1975 paper.
- Also saved the [Stata maximize manual](https://www.stata.com/manuals/rmaximize.pdf)
  and [iminuit reference](https://scikit-hep.org/iminuit/reference.html#iminuit.Minuit.tol).
  Software thresholds below are attributed to these particular documents,
  not retroactively to the original papers.

For a total negative log likelihood, our conversions are:

| Source | Documented nominal criterion | Equivalent distance d |
|---|---|---:|
| MINUIT manual 94.1, MIGRAD command | EDM < .001 × tolerance × UP, default tolerance .1, UP=.5 for negative log likelihood | .01 |
| iminuit 2.33.0 documentation, MIGRAD | EDM < .002 × tolerance × errordef, default tolerance .1, errordef=.5 | .01414 |
| Stata maximize manual, nrtolerance | Hessian-scaled gradient quadratic form < 1e-5 | .003162 |

The historical MINUIT criterion therefore matches the proposed .01 exactly
under this normalization. This is a precedent, not an independent proof of
adequacy. Modern iminuit documents possible factor-ten EDM exceptions; Stata
also requires a step/function test and checks actual curvature after a
quasi-Newton criterion passes. Their complete stopping policies differ.

## Choosing a budget rather than fitting a cutoff

The following derivations are our accuracy interpretation, not numerical
recommendations claimed from the papers. A tolerance is an accuracy contract:
mathematics can translate a declared error budget into a cutoff, but cannot
select the acceptable budget without a scientific or numerical convention.

Let F be the total negative log likelihood, G its gradient, and I its positive
observed Hessian in identified, equality-reduced coordinates. Write

$$
 s=-I^{-1}G,\qquad V=I^{-1},\qquad
 d=\sqrt{G^\top I^{-1}G}.
$$

Then an exact algebraic identity for the *predicted Newton step* is

$$
 \sup_{a\ne0}\frac{|a^\top s|}{\sqrt{a^\top Va}}
 =\sqrt{s^\top V^{-1}s}=d.
$$

Thus this is already relative error: relative to information-based sampling
uncertainty, uniformly over linear contrasts. Dividing by a coefficient's
magnitude instead is unstable near zero and depends on the parameter origin.
The identity is invariant under nonsingular linear changes of coordinates;
nonlinear reparameterizations supply only a local interpretation.

### A conservative RMSE contract

For a fixed contrast, let T be its exact estimator, theta its target, and
sigma a fixed reference sampling-error scale. Suppose the *actual* numerical
error e satisfies root-mean-square(e) <= epsilon sigma. The L2 triangle
inequality gives, without independence or zero mean of numerical error,

$$
 \operatorname{RMSE}(T+e)
 \le \operatorname{RMSE}(T)+\epsilon\sigma.
$$

When sigma equals the exact estimator's RMSE, epsilon=.01 permits at most
1% inflation of RMSE, or 2.01% inflation of MSE. More generally it bounds
added RMSE by .01 of the declared scale; identification of that scale with
an SE requires the usual regular, approximately unbiased model-based regime.
If the desired contract is at most r relative MSE inflation, it suffices to
choose epsilon <= sqrt(1+r)-1; r=.02 gives epsilon approximately .00995.

A per-fit actual-error bound in the fixed covariance metric implies the
contrastwise RMS premise. Replacing that fixed covariance with estimated
information and replacing actual error with the Newton prediction makes
this a local/asymptotic design rationale, **not an established finite-sample
risk guarantee**. In particular, d=.01 does not imply merely .01% extra MSE:
the cross term between numerical and sampling errors need not vanish.
Do not apply this unconditional argument to a selected subset of converged
replications without separately examining selection.

### A distributional interpretation, without choosing one downstream test

For Gaussian reference distributions with identical covariance V and centres
separated by an *actual* displacement e, set D=sqrt(e' V^-1 e). Whitening and
projection onto the displacement direction yield the exact identity

$$
 \operatorname{TV}\{N(\theta,V),N(\theta+e,V)\}
 =2\Phi(D/2)-1.
$$

D=.01 corresponds to .003989 total variation: probabilities of any fixed
measurable event differ by at most about .4 percentage points between these
two Gaussian references. A chosen probability budget eta gives
D <= 2 Phi^-1((1+eta)/2). This is an optional interpretation of approximation
accuracy, not a guarantee about actual p-values, repeated-sampling coverage,
a changing covariance estimate, or the true likelihood's non-Gaussian shape.
Using predicted d in place of D introduces the same Newton-accuracy question.

### What would turn the prediction into a bound?

Positive curvature at one point alone is insufficient. Suppose a stationary
point exists nearby, and every Hessian along the segment from the current
point to that stationary point lies between (1-rho)I and (1+rho)I in Loewner
order, with rho < 1. Integrating the gradient along the segment then gives

$$
 \|\theta_{\mathrm{current}}-\theta_{\mathrm{stationary}}\|_I
 \le \frac{d}{1-\rho}.
$$

Consequently, a *verified* rho bound would allow d <= (1-rho)epsilon to
certify the declared displacement budget. We do not currently have such a
bound for general SEM. An observed Hessian at the terminal point, or a few
extra evaluations, does not by itself establish it. Boundary extensions,
verified neighbourhood analysis, and practical refined-solution comparisons
remain deferred; self-concordant convex guarantees cannot simply be assumed
for the nonlinear SEM objective.

Recommendation: retain .01 as a proposed one-percent sampling-error-scale
budget, supported by the exact historical MINUIT precedent. Keep .003 and
.03 sensitivity and continuous distance reporting. Validate the Newton
prediction separately from selecting the acceptable error budget. No default
or existing success classification changes in this literature update.


## Precedent and scope of the recommendation

[MINUIT/MIGRAD's EDM](https://scikit-hep.org/iminuit/reference.html#iminuit.Minuit.tol)
and [Stata's terminal Hessian-scaled gradient check](https://www.stata.com/manuals/rmaximize.pdf)
are direct practical precedents. A distance of .01 corresponds to total EDM
5e-5; this supplies an accuracy interpretation rather than copying a raw
constant from differently normalized objectives. [NLopt's documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
also cautions that matching nominal stopping tolerances need not match
attained accuracy.

The proposed profile worked across these controlled settings, not a broad
empirical corpus. Immediate next implementation jobs are (1) an opt-in
regular-interior Newton diagnostic with explicit objective normalization,
(2) effective-control and stopping-reason reporting (explicit backend controls
are now implemented; see `docs/reference/optimizer-controls.md`),
and (3) portable domain-recovery/backtracking handling. Neither this check nor
the candidate settings silently replaces the authoritative existing audit.
Boundary extensions and practical verification remain deferred.

Validation status: standalone exact/derivative checks, paired-run coverage,
R summary execution, shell syntax and whitespace checks passed. The repository-
wide layering checker reported existing references in unchanged paper trees
(including the ignored pinned checkout); no finding named this new harness.
Those unrelated paper paths were not changed by this investigation.

## Refined-reference smoke (2026-09-21)

At the author's request, performed a bounded first check of predicted versus
measured numerical error; broader practical calibration remains deferred.
Reproduction and complete design are in the advisory check's README, using
its `refine` mode and `summarize_refinement.py`. Results are local to
`tests/checks/interior_newton/results/refinement-smoke-2026-09-21-v2/`.
No production optimizer/audit default or manuscript success rate changed.

The check uses 90 natural L-BFGS endpoints (45 per control profile), across
seven settings, N=100/1000, three measurement units and one replication,
plus the previously problematic cfa12 N=100000 rep=3 fixture. Six endpoints
are improper weak8 N=100 fits, three units under each profile. All 84 eligible
endpoints refined to d <=1e-8 in at most two accepted Newton steps. References
use recomputed analytic curvature and gradient and safeguarded Newton steps;
they are numerical reference solutions, not exact/global optima. The fitting
stage uses the diagnostic extended-backtracking build, not stock NLopt.

Most natural endpoints were already very accurate. Distance ratios are
therefore reported for 25 legacy and 14 candidate endpoints with enough
separation from the reference tolerance; the others are not used to claim
relative prediction accuracy. The actual/predicted distance ranges were
0.999925–1.001817 for legacy and indistinguishable from 1 at six decimals
for the candidate profile. There were no .003/.01/.03 pass/fail disagreements
among the qualified natural endpoints. Their scarcity near the thresholds
motivated deliberately placed local probes.

There are 56 probes around the seven N=1000, unit-scale-1 reference fits:
two directions and four nominal information-distance targets. Each was
independently refined, and all returned within 5.9e-9 information-distance
of the originating anchor. Direction selection and the single replication
are exploratory, not a representative model or direction sample.

| Nominal probe distance | Probes | Measured / predicted distance | Largest relative vector error |
|---|---:|---:|---:|
| .003 | 14 | .998749–1.000061 | .181% |
| .01 | 14 | .995880–1.000203 | .603% |
| .03 | 14 | .988064–1.000608 | 1.82% |
| .1 | 14 | .965600–1.002030 | 6.13% |

Both distances use observed information at the initial probe. Vector error
is the information norm of (measured correction minus Newton correction),
divided by the norm of the measured correction. At the proposed .01 scale,
the length prediction is within .42% in these probes and the vector prediction
within .61%. Predicted twice-loglik improvement agrees to within .28% there.
Natural-endpoint likelihood-gain ratios are assessable in only 12 legacy and
two candidate cases; subtracting nearly equal objectives is unreliable below
the declared floating-point floor.

This is still an approximation: five probes centred on the .01 target fall
on different sides of the exact .01 pass boundary under predicted and measured
distance. Four are conservatively rejected by prediction; one predicts a pass
while measured distance exceeds .01 by 1.8e-6 SE units. Corresponding effects
occur at .003/.03. Reporting continuous distance matters; the results do not
turn a hard cutoff into a mathematical certificate.

The previous conspicuous legacy endpoint has predicted d=.254176 and measured
distance .254638 after refinement, a .182% length discrepancy. Predicted
remaining twice-loglik improvement is .0646054; measured improvement is
.0646836. This supports the diagnostic's substantive warning for that endpoint.

Interpretation: the local Newton approximation is encouraging at .01 in this
small interior check. It supports trying the accuracy budget, not claiming
that .01 is uniquely optimal or validating finite-sample RMSE/coverage bounds.
Remaining questions are heterogeneous/weakly identified interiors, additional
replications and directions, and independently qualified references. PSD
boundaries remain outside this experiment.

Validation: exact quadratic/refinement checks passed; all 146 output rows have
unique design keys, the prescribed 90 natural endpoints and 56 probes, and
references/precision flags are preserved. Source/library hashes and executed
source snapshots accompany the local results. No Monte Carlo error or timing
precision claim is made from this smoke.


## Intermediate L-BFGS options after agreeing the .01 target

The author has accepted .01 as the regular-interior accuracy budget. It remains
an approximate statistical-scale accuracy target, not a universal bound. The
next question is search efficiency at that fixed target, not selecting a cutoff
from success rates.

A further paired run uses the same 189 datasets/unit combinations and six
profiles (1,134 fits). It retains automatic memory, internal gradient tolerance
1e-8, and maxeval=5000. As before, it uses diagnostic extended backtracking
(mred=60); this is not available through the production options interface.
Nine weak-model endpoints per profile are improper and excluded from interior
interpretation. All profiles have the same 180 eligible cases.

| Relative function tolerance | Relative step tolerance | Pass .01 | Maximum d | Median evaluations | Total evaluations |
|---|---|---:|---:|---:|---:|
| 1e-10 | 1e-7 (current) | 179/180 | .2542 | 68 | 14575 |
| 1e-10 | 1e-8 | 180/180 | .00503 | 72 | 15122 |
| 1e-10 | 1e-9 | 180/180 | .00503 | 72 | 15245 |
| 1e-10 | 1e-10 | 180/180 | .00503 | 72 | 15245 |
| 1e-11 | 1e-9 | 180/180 | .00231 | 75 | 16257 |
| 1e-12 | 1e-10 | 180/180 | .00231 | 77 | 17054 |

The cheapest successful tested profile is ftol_rel=1e-10, xtol_rel=1e-8:
3.8% more total evaluations than current controls, versus 17.0% for the original
conservative candidate. The latter consumes 12.8% more evaluations than the
cheaper profile. These comparisons count all attempted fits. Single-run median
fit times were .298, .300, .288, .287, .309 and .319 ms respectively; fixed order
and very short fits make these descriptive only. Evaluation counts support the
cost comparison more clearly.

Working recommendation: evaluate the cheaper profile as a first-pass candidate,
with the separate .01 Newton audit deciding achieved accuracy. Retain the
conservative profile as a candidate polishing setting when the audit fails;
that conditional continuation policy has not been implemented or tested here.
Do not infer that either search tolerance guarantees an audit pass. This panel
reuses the exploratory datasets, so it does not establish held-out reliability
or justify a production default change by itself. Internal gradient tightening
and fixed memory had no useful benefit in the earlier comparison and remain
unchanged. The evaluation cap was held fixed, not calibrated by this run.

The domain/backtracking issue must be resolved separately before deployment.
A positive solver code alone is insufficient (the current profile's failing
case returns XTOL_REACHED), and a negative code alone must not discard an
accurate candidate. No production controls, public audit, manuscript results,
or boundary treatment changed.

Reproduction and local results: `tests/checks/interior_newton/README.md`,
`results/options-2026-09-21/` within that check. The summary validates the paired
1,134-row design and seven analytic-Hessian directional checks. Executable
quadratic/affine/scaling checks also passed. Source snapshots, hashes and the
raw terminal candidates are retained in that local results directory.

## Model-coverage review and fixed-control validation

The original seven cases only span p=4–12 and one or two factors. Despite
variation in means, equality constraints, weak signal and misspecification,
this is too narrow to settle optimizer defaults. The expanded panel was fixed
before inspecting results, with the three current/cheap/conservative profiles
unchanged. It contains ten new model/settings, p=6–48, one to twelve factors,
and 12–162 free parameters after equality reduction. N=50/200/1000/10000,
two independent samples per model/N and three global unit scales yield 240
paired cases, 720 fits. These are not 240 independent samples: scale variants
share the same data. The original panel retains the very-large-N=100000 check.

Added structure: larger correlated-factor CFAs; a three-factor regression
chain with means; cross-loadings; correlated residuals; a larger equality-linked
model with means; weak factors; and unequal indicator scales spanning a factor
of 100. All populations have positive-definite covariance. N=50,p=48 deliberately
approaches the sample covariance rank limit. Exact generating parameters and
reproduction are in the check README; raw output records p, factors and reduced
parameter count. Backtracking is still the diagnostic mred=60 version.

| New model | p | Free parameters | Current pass/eligible | Cheap pass/eligible | Conservative pass/eligible |
|---|---:|---:|---:|---:|---:|
| One-factor CFA | 6 | 12 | 24/24 | 24/24 | 24/24 |
| Four-factor CFA | 16 | 38 | 23/24 | 24/24 | 24/24 |
| Eight-factor CFA | 32 | 92 | 23/24 | 23/24 | 24/24 |
| Twelve-factor CFA | 48 | 162 | 13/21 | 13/21 | 21/21 |
| Regression chain, means | 12 | 38 | 23/24 | 24/24 | 24/24 |
| Cross-loadings | 12 | 29 | 24/24 | 24/24 | 24/24 |
| Correlated residuals | 12 | 30 | 18/18 | 18/18 | 18/18 |
| Equality-linked loadings, means | 24 | 72 | 24/24 | 24/24 | 24/24 |
| Weak factors | 12 | 27 | 10/13 | 10/13 | 13/13 |
| Unequal indicator scales, means | 16 | 54 | 16/24 | 17/24 | 22/24 |
| **Total** | | | **198/220** | **201/220** | **218/220** |

Each model/profile has 24 attempts. Twenty endpoints per profile are outside
the primitive covariance interior and are not counted as audit successes.
Some of those also have nonpositive curvature. None hit the 5000-evaluation
budget. Total evaluations are 33295/34550/46041; medians are 119/121/137.
Thus the conservative profile costs about 33% more total evaluations than the
cheap one on this panel, but its accuracy advantage is substantial. A median
alone understates the cost difference on difficult fits.

The conservative profile's two eligible failures both involve unequal scales
at global multiplier .1: N=200, replication 2 (d=.01510, FTOL_REACHED), and
N=1000, replication 1 (d=.01327, generic negative return). Cheap-profile
failures also occur at N=10000 (8 of 60 eligible fits): large samples do not
remove the need to calibrate numerical accuracy relative to sampling precision.
No threshold was relaxed and no settings were changed to remove these failures.

**Updated conclusion:** withdraw the cheap profile as a near-settled default.
Retain it as a possible first stage of an audit-and-polish policy; that policy
still requires testing. The conservative profile is the stronger standalone
candidate, but does not guarantee .01 accuracy either. The larger models and
mixed scales were consequential omissions, and a separate accuracy audit remains
necessary. Production defaults are unchanged.

Coverage is now useful for continued single-group ML development, not sufficient
for a package-wide default claim. Before deployment, add multi-group equality/
invariance, a growth model with fixed loadings, a regular feedback model, and
published-data/corpus fixtures owned by this check. Non-ML and missing-data
objectives require their own scaling/coverage work. More replications would
estimate frequencies better; they do not replace these missing model classes.
The new panel was held out from the preceding tolerance comparison; once used
for further tuning it is development data, and another frozen validation set
will be needed for that policy.

Checks: complete unique paired design; ten analytic-Hessian directional checks;
positive-definite generating covariances; exact quadratic/affine/normalization
checks. Raw data, executable/source hashes and model-level summaries are in
`tests/checks/interior_newton/results/validation-2026-09-21/` locally.
