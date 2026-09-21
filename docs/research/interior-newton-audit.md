# Regular-interior Newton audit and L-BFGS controls

2026-09-21. Advisory implementation and control study. No production default
or public audit contract changes. Boundary audits and practical verification
by restarts/independent optimizers remain deferred, as requested.

## Candidate choices

Use a candidate default **maximum Newton distance d = 0.01**. In regular ML,
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

The best balanced **candidate L-BFGS stopping profile** in this study is
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
(2) accurately named/exposed L-BFGS controls and a documented candidate profile,
and (3) portable domain-recovery/backtracking handling. Neither this check nor
the candidate settings silently replaces the authoritative existing audit.
Boundary extensions and practical verification remain deferred.

Validation status: standalone exact/derivative checks, paired-run coverage,
R summary execution, shell syntax and whitespace checks passed. The repository-
wide layering checker reported existing references in unchanged paper trees
(including the ignored pinned checkout); no finding named this new harness.
Those unrelated paper paths were not changed by this investigation.
