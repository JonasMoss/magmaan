# Interior Newton audit and NLopt L-BFGS controls

Advisory, standalone diagnostic; no production audit or defaults changed.
Consumes the canonical model, objective, constraints and observed-information
APIs. Direct NLopt calls expose actual controls and preserve every terminal
candidate, independently of the production adapter's return policy.

From the repository root:

```sh
bash tests/checks/interior_newton/run.sh tests/checks/interior_newton/results/stock-new 10
bash tests/checks/interior_newton/run.sh tests/checks/interior_newton/results/extended-new 60
TZ=Europe/Oslo Rscript tests/checks/interior_newton/summarize.R tests/checks/interior_newton/results/stock-new tests/checks/interior_newton/results/extended-new
```

Requires the existing opt CMake configuration and its Eigen/NLopt dependency
sources, clang++, cc and R. `run.sh` builds the core, rejects existing output
directories, and records source/library hashes. The second run compiles a
local copy of NLopt `plis.c` with only `mred=10` changed to `mred=60`, then links
that object into this executable. Neither dependency source nor installed
library is edited. The extended version is a diagnostic, not stock NLopt.

Design is fixed in `check.cpp`: seven model/settings, N=100/1000/100000,
three replications, three global measurement-unit multipliers, nine control
profiles: 1,701 fits per backtracking version. The same underlying dataset is
used across profiles and units. Population covariance generators include
one-factor CFA with 4 or 12 indicators; two-factor CFA with 8 indicators;
a weak-loading/high-correlation variant; equality-linked loadings and means;
a structural regression with means; and a misspecified one-factor model
with an omitted residual covariance. These are diagnostic simulations, not
published-data examples or a prevalence sample.

Fit starts use canonical FABIN. Linear equalities use canonical reduction K.
No bounds, PSD fitting, optimizer restarts, alternative optimizers, SEs, or
tests are requested. Observed information is computed for the audit itself.
No library default is altered. A candidate outside the positive-definite
primitive covariance interior is ineligible for the proposed statistical
interpretation even if its ambient Newton calculation is available.

`assess` is an experimental interior diagnostic, not a public API. It uses
G=N*gradient(f), I=observed total information, after linear equality reduction.
Diagonal equilibration and LLT solve give d=sqrt(G'I^-1G), total EDM=d^2/2,
and the predicted Newton step. Indefinite/singular or ill-conditioned
curvature is unavailable, not repaired. The condition cap (1e12) and solve
residual cap (1e-10) are explicit provisional numerical guards. Their behavior
under arbitrary badly conditioned coordinate transforms is not established.

Executable checks cover exact quadratic error, nonsingular affine changes,
objective/sample-size scaling, and singular/indefinite rejection. Seven
independent directional finite differences check the analytic Hessian's
normalization and equality reduction in the SEM cases. The summary verifies
coverage, exact paired design, and finite-difference error. Raw negative
NLopt return codes are retained; they are not automatically numerical failures.

Timings are single executions in fixed profile order and describe this pilot
only. Evaluation counts are the more reliable cost comparison. The Hessian
cost includes construction plus the experimental audit, excludes the older
geometric audit, and can partly be reused for observed-information inference.

Results and design interpretation: `docs/research/interior-newton-audit.md`.
The stock and extended local runs are under this directory's ignored
`results/`; use a new directory for every subsequent execution.

Methodological references and the threshold rationale are recorded in the
study note: Dennis, Gay & Welsch (1981), DOI 10.1145/355958.355965, section 6;
James's MINUIT manual 94.1; and the documented Stata/iminuit stopping rules.
The note distinguishes retrieved versions from unavailable published PDFs.
The .01 candidate is an accuracy budget relative to sampling uncertainty,
not a cutoff calibrated from this experiment's pass rate. Its interpretation
as actual remaining error still depends on the local quadratic approximation.

## Refined-reference smoke

```sh
bash tests/checks/interior_newton/run.sh tests/checks/interior_newton/results/refinement-new 60 refine
python3 tests/checks/interior_newton/summarize_refinement.py tests/checks/interior_newton/results/refinement-new
```

This mode evaluates 90 natural endpoints: seven models, N=100/1000, rep=1,
three measurement units, legacy/current and candidate L-BFGS settings, plus
cfa12 N=100000 rep=3 in all three units under both settings. It also evaluates
56 controlled local probes: seven models at N=1000 and unit scale 1, two
directions (a fixed sine vector and the smallest-information-eigenvalue
vector), and distances .003/.01/.03/.1 from a refined anchor. Directions are
normalized in the anchor information metric; their choice is exploratory.

References use analytic-Hessian Newton refinement with backtracking and
recomputed objective/gradient/curvature. They must remain eligible interior
points and reach distance <=1e-8 within 20 accepted steps. Trials must reduce
distance and may increase per-observation objective by no more than the
explicit roundoff allowance 64*machine_epsilon*(1+abs(f)). This handles
objective cancellation near the minimum; it is not evidence of ascent at
statistically meaningful scale. Unqualified references are retained, not
silently dropped. The reference is a much more accurate numerical solution,
not an exact optimum or independent derivative implementation.

Measured displacement and Newton prediction use the same **initial-point**
information metric. The vector-error column also checks direction, not merely
length. Distance ratios are suppressed unless initial d exceeds 100 times
max(1e-8, reference d). Objective-gain ratios require predicted twice-loglik
improvement to exceed 100 times the declared floating-point subtraction floor.
Probe refinements additionally report agreement with their originating anchor.
Exact quadratic checks exercise the refinement and Newton correction.

First completed run: `results/refinement-smoke-2026-09-21-v2/`; `v2` adds
quadratic checks and anchor-agreement telemetry to the initial smoke. Both
runs are retained. This checks the local error prediction; it does not select
an accuracy budget or establish uniform guarantees, coverage or globality.


## Intermediate stopping options

After fixing the desired audit budget at .01, compare cheaper intermediate
function/step tolerances on the original paired panel:

```sh
bash tests/checks/interior_newton/run.sh tests/checks/interior_newton/results/options-new 60 options
python3 tests/checks/interior_newton/summarize_options.py tests/checks/interior_newton/results/options-new
```

Six profiles, 189 cases each (1,134 fits); unchanged datasets, starts, internal
gradient tolerance, automatic memory and evaluation cap. This is exploratory
reuse, not held-out validation. The standard-library Python summary checks
paired design coverage and finite-difference checks, retains failures, and
reports evaluation counts alongside descriptive single-run timings. The first
run is `results/options-2026-09-21/`. Full interpretation is in the study note.

## Broader validation panel

```sh
bash tests/checks/interior_newton/run.sh tests/checks/interior_newton/results/validation-new 60 validate
python3 tests/checks/interior_newton/summarize_validation.py tests/checks/interior_newton/results/validation-new
```

Ten new settings, fixed before inspecting results: CFAs with p=6/16/32/48
and 1/4/8/12 factors; a three-factor regression chain with means; a CFA with
two cross-loadings; a CFA with three correlated residual pairs; a six-factor
model with equality-linked loadings and means (p=24); a weak-factor CFA;
and p=16 with means and unequal measurement scales. The latter multiplies
indicator scales by a log-spaced sequence from .1 to 10. It is additional to
the global .1/1/10 multipliers applied to every setting.

N=50/200/1000/10000, two replications, three global unit scales: 240 paired
cases, fitted under current, x8 and f12_x10 settings (720 fits). Fresh seed
base 760921; populations are explicitly positive definite. Ordinary CFAs have
population factor correlations .3; the weak model has variances .12 and
covariances .10; chain paths are .3 with unit disturbance variances;
cross-loadings are .25; residual covariances are .15. Other loadings and
residual variances follow the original generator. Actual equality-reduced
parameter counts are recorded. N=50,p=48 intentionally approaches the
sample-covariance rank limit without crossing it.

The summary checks design coverage, ten directional Hessian checks and reports
all attempts, eligibility, audit failures and evaluation counts. Exclusions
are not successes. Scale variants share a dataset and are not independent
replications. First results: `results/validation-2026-09-21/`.

This is an expanded synthetic single-group complete-data ML panel, not broad
validation of every estimator. It still lacks multi-group invariance, growth,
feedback, non-Gaussian/missing-data and published-data models. It is held out
from the earlier option choice, but becomes development evidence once used
for subsequent tuning. Do not tune and describe these same cases as held out.

## Growth, feedback, multi-group and published-data panel

```sh
bash tests/checks/interior_newton/run.sh tests/checks/interior_newton/results/advanced-new 60 advanced
python3 tests/checks/interior_newton/summarize_advanced.py tests/checks/interior_newton/results/advanced-new
```

`advanced.cpp` shares this check's Newton helpers, independently builds each
model and uses the same three fixed validation profiles. Synthetic designs:

- Linear growth at 4 and 8 occasions, fixed intercept/slope loadings 1 and
  0,...,p-1; latent means (1,.2), covariance [[1,.15],[.15,.2]], residuals .7.
  Observed intercepts are fixed zero and both latent means are free.
- Four-factor feedback with three indicators per factor. f3 <- .2*f4+.5*f1,
  f4 <- .25*f3+.4*f2, independent unit disturbances and exogenous factors;
  exclusion restrictions provide separate predictors of the feedback pair.
  Marker loadings 1, other loadings .8, residual variances .7. I-B is invertible.
- Two-group CFA with configural, metric and scalar restrictions, plus a
  five-group metric model with unequal group sizes. Three factors, four
  indicators each, loadings 1/.8, latent correlations .3, residuals .7.
  Group covariance g (zero-based) is (1+.2*g) times the base covariance;
  common observed means are .1,.2,...,1.2. These populations satisfy all the
  fitted loading/intercept equalities.

For each: base n=50/200/1000/10000, two replications, global scales .1/1/10.
Balanced groups each have base n observations. Unbalanced group g has
max(p+2,floor(n/(g+1))); the floor avoids singular empirical covariance in
this regular-ML panel. Both total N and every group size are recorded.
Independent group samples use a continuous RNG stream from the recorded seed.

Published-data fits read existing test-owned summary-statistic fixtures:
Bollen political democracy and Holzinger–Swineford from `tests/fixtures/parity/`,
and all four Kline/Guo invariance exports from
`tests/fixtures/textbook_corpus/case_exports.json`. No fitted parameters or
oracle starts are used. Covariances are used exactly as exported, with no
additional n/(n-1) conversion. Models/options follow those fixtures. Published
fits use original sample sizes and three unit scales, not simulated replications.
The four Guo specifications share the same observations. No private corpus or
sibling experiment/paper dependency is introduced.

There are 504 synthetic and 54 published-data fits: 558 total, 186 per profile.
`advanced-summary.json` retains failures/exclusions and `case-metadata.json`
records pre-fit descriptors for later paired option comparisons. Group counts
are not independent model samples, and no model-specific optimizer rules have
been selected. First complete run: `results/advanced-2026-09-21-v2/`.
The initial run stopped at the singular smallest unbalanced-group covariance;
that incomplete run is retained separately, not included in the results.

Observed information uses total N across groups. Thirty-four directional
finite differences independently check the normalization and equality reduction
at unit scale, including all corpus models. Build inputs, source snapshots and
fixture SHA-256 hashes are saved with the completed local run.

## Feedback starts and parameter coordinates

```sh
bash tests/checks/interior_newton/run.sh tests/checks/interior_newton/results/feedback-starts-new 60 starts
python3 tests/checks/interior_newton/summarize_starts.py tests/checks/interior_newton/results/feedback-starts-new
```

This reproduces the eight feedback datasets from the advanced panel (same seed,
RNG order and population). Three unit scales and three arms give 72 fits:
canonical FABIN starts; correctly transformed unit-1 FABIN starts; and those
transformed starts with the optimizer working in unit-1 parameter coordinates.
All use conservative controls and diagnostic mred=60. This is an isolation
experiment, not an implemented general-purpose scaling policy.

For this marker-identified, covariance-only model, multiplying observations by
u leaves Lambda and Beta unchanged and multiplies Psi and Theta by u^2. There
are no equality constraints in this model. Mapped coordinates use theta=D*z,
where D has u^2 on covariance parameters and 1 elsewhere; the gradient passed
to NLopt is D times the theta gradient. Initial objective and gradient
transformation identities are checked. Final audits use original coordinates.

`raw.csv.starts.csv` records all 29 parameter starts, their matrix cells, and
values transformed back to unit-1 scale. `starts-summary.json` verifies design,
which starts differ, accuracy and objective agreement with the unit-1 results.
The only material native/mapped differences are four latent variance starts.
The 24 native control fits exactly reproduce the preceding advanced run.
First completed run: `results/feedback-starts-2026-09-21/`.
