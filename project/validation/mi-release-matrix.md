# MI and equality-release validation matrix

This inventory covers the lab/C++ one-parameter score surface for 0.2.0.
An ordinary MI here is the unscaled score statistic, not the ordinary R
package's inference policy. Each robust cell covers both MI and one-at-a-time
equality releases unless its gate explicitly says MI only. Fixed weight uses
the fitted W; estimated weight adds the influence of that same recorded recipe.
This is component evidence, not calibration or an ordinary-user recommendation.

## Gates

The labels below give a file and the exact test name (or a stated group of
names); a cell lists its C++ gate first and its R gate second.

- **C-ML**: `cpp/tests/unit/score_test.cpp`, `modification_indices: complete ML reports finite fixed-row tests`, `score_tests: equality releases are reported in constrained ML models`, and `modification_indices: absent-row MI / EPC match lavaan modindices`.
- **C-GOLD**: `cpp/tests/golden/score_golden_test.cpp`, `score/modification-index goldens match lavaan fixed-row and equality-release targets` (ML, observed FIML, ULS, ordinal/mixed DWLS, GLS and ADF/WLS).
- **C-ROB-ML**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust MI: model-implied Expected bread reduces to NT`, `frontier robust MI: observed bread is finite and positive`, `frontier robust score test: equality release reduces to NT`, `frontier robust MI multi-group: model-implied Γ_NT meat reduces to NT`, and `frontier robust MI multi-group: empirical raw-data path scales, finite`. `cpp/tests/golden/score_robust_golden_test.cpp`, `robust release-score matches the lavaan-internals oracle (c != 1)`, pins the robust MLM release.
- **C-LS**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust LS MI: GLS weight + Gamma_NT(S) meat reduces to NT`, `frontier robust LS score test: GLS equality release reduces to NT`, `frontier robust LS MI: DWLS raw path scales; sandwich matches primitives`, and `frontier robust LS score test: raw and supplied Gamma agree`.
- **C-RECIPE**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust LS MI: an estimated-weight recipe must reproduce the fitting weight` and `estimated-weight IJ mode follows the recorded weight recipe`.
- **C-ORD**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust ordinal MI: WLS + NACOV meat reduces to ordinary`, `frontier robust ordinal score test: WLS equality release reduces`, `frontier robust ordinal MI: DWLS scales against the NACOV meat`, `frontier robust ordinal MI: estimated-weight DWLS shifts the scaling`, and `ordinal score rank: latent units and nearby points preserve candidates`. The primary ordinal rank and weight-scale R gates are R-ORD.
- **C-ORD-REFUSE**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust ordinal inference: estimated weight refuses NT, DLS and supplied weights`.
- **C-MIX**: `cpp/tests/unit/score_robust_test.cpp`, `frontier robust mixed ordinal: WLS reduces, DWLS finite, ULS rejected` and `frontier robust mixed ordinal multi-group: WLS reductions cover MI and score`.
- **C-2S**: `cpp/tests/unit/score_robust_test.cpp`, `frontier ML2S MI and releases reduce to complete-data robust tests` and `frontier ML2S MI: naive statistic plus Stage-1 scaling under missing data`.
- **R-ML**: `r-package/tests/testthat/test-score-rank.R`, `MI tables exclude identification releases across indicator units`; `test_wls_robust_covariance.R` supplies related weight and covariance guards. `test_weight_recipe_inference.R`, `complete ML MI and equality releases share ordinary and robust coordinates` gates complete-data ordinary/fixed-robust releases and the estimated-weight R refusal.
- **C-FIML-ROB**: `cpp/tests/unit/score_robust_test.cpp`, `frontier FIML robust MI: unscaled mi matches the non-robust FIML MI`, `frontier FIML robust score test: equality release runs and is finite`, and `frontier FIML robust MI and score tests support multi-group raw blocks`; `cpp/tests/golden/score_robust_golden_test.cpp`, `robust FIML release-score matches the lavaan-internals oracle (MLR)`.
- **R-SUPPLIED**: `r-package/tests/testthat/test_wls_robust_covariance.R`, `WLS robust MI and releases preserve empirical covariance and fitting W` and `WLS covariance choices fail explicitly when unavailable` (single/unequal groups, means, shared labels and supplied diagonal/full W).
- **R-FIML**: `r-package/tests/testthat/test_fiml_robust_score.R`, `FIML robust MI and releases agree across retained and explicit data` and `FIML robust score wrappers reject incompatible conventions`.
- **R-RECIPE**: `r-package/tests/testthat/test_weight_recipe_inference.R`, `estimated-weight inference reads the recorded continuous recipe`, `release score tests read the recorded recipe`, and `supplied weights are used as fitted but have no estimated-weight recipe`.
- **R-LS-MEAN**: same file, `continuous LS MI and releases preserve means and equality constraints`.
- **R-ORD**: `r-package/tests/testthat/test_ordinal_score_rank.R`, `ordinal ordinary and robust ranks agree across weight scales` and `ordinal fixed rows and equality releases use the same rank and moment scale`.
- **R-ORD-RECIPE**: `r-package/tests/testthat/test_weight_recipe_inference.R`, `ordinal NT, DLS and supplied weights refuse the weight influence`.
- **R-2S**: same file, `two-stage MI and releases reduce to complete-data robust tests`, `two-stage MI under missing data uses the recorded Stage-2 weight`, `two-stage unequal groups reduce to complete-data robust tests`, and `two-stage MAR MI and releases retain positive Stage-1 scaling`.

- **C-LS-MATRIX**: `cpp/tests/unit/score_robust_test.cpp`, `continuous LS recipe matrix gates means constraints and ordinary score reductions` (ULS, NT, DWLS, ADF and DLS; ordinary and fixed/estimated robust MI and releases).
- **C-ORD-MATRIX**: same file, `ordinal MI recipe matrix gates ordinary releases and typed refusals` (ULS, DWLS, WLS, NT, DLS and supplied W; both ordinary statistics and both robust channels). The association-tagged estimates assert `UnsupportedInference` in all four workers; the genuine association fit is also gated by `cpp/tests/unit/api_sem_test.cpp`, `api ML dispatches ordinal associations without granting Gaussian inference`.
- **C-MIX-MATRIX**: same file, `mixed ordinal MI recipe matrix gates ordinary cells and typed robust refusals` (ULS, DWLS and WLS; ordinary MI/releases, fixed robust DWLS/WLS, ULS and estimated robust refusals).
- **R-ORD-MATRIX**: `r-package/tests/testthat/test_weight_recipe_inference.R`, `ordinal recipe matrix gates releases and estimated-weight refusals` (all six recipes, both MI and releases; association-ML refusals).
- **R-MIX-MATRIX**: same file, `mixed ordinal MI matrix gates fixed weights and explicit refusals` (DWLS/WLS MI and releases; R ULS fitting refusal).

## Cells

`UnsupportedInference` names the core error kind; the R binding includes it in
its error message where available. ML has no second-stage estimated weight.
Ordinal ULS has no weight influence, so its estimated channel reduces to fixed.
Each cell names an exercised component gate or the actual refusal. ML/FIML
have no estimated quadratic-weight core operation; their R-level argument
guards reject attempts to request one. Mixed ordinal currently reports
`PostError::NumericIssue` for unavailable robust choices, not
`UnsupportedInference`; the matrix records that behavior without relabeling it.

| Fitting estimator / weight | Ordinary MI | Ordinary release | Robust fixed weight | Robust estimated weight |
| --- | --- | --- | --- | --- |
| Complete ML / likelihood | C-ML, C-GOLD / R-ML | C-ML, C-GOLD / R-ML | C-ROB-ML / R-ML | Inapplicable: no Stage-2 recipe; R-ML rejects estimated weight |
| Direct FIML / likelihood | C-GOLD / R-FIML | C-GOLD / R-FIML | observed statistic/bread and pattern-score meat: C-FIML-ROB / R-FIML | Inapplicable: no second-stage weight; R-FIML rejects it |
| ML2S / NT-ML | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S; equals fixed (no quadratic-weight influence) |
| ML2S / ULS | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S; equals fixed (identity) |
| ML2S / DWLS | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S |
| ML2S / ADF | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S |
| ML2S / DLS(a) | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S | C-2S / R-2S |
| Continuous ULS / identity | C-GOLD, C-LS-MATRIX / R-LS-MEAN | C-GOLD, C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | Identity has zero influence: C-LS-MATRIX / R-LS-MEAN |
| Continuous GLS / NT(S) | C-GOLD (transported), C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX, C-RECIPE / R-RECIPE, R-LS-MEAN |
| Continuous WLS / ADF | C-GOLD (transported), C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX / R-RECIPE, R-LS-MEAN | C-LS-MATRIX, C-RECIPE / R-RECIPE, R-LS-MEAN |
| Continuous DWLS / diag ADF | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX, C-RECIPE / R-LS-MEAN |
| Continuous DLS(a) / NT-ADF mixture | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX / R-LS-MEAN | C-LS-MATRIX, C-RECIPE / R-LS-MEAN |
| Continuous LS / supplied W | C-LS / R-RECIPE | C-LS / R-SUPPLIED | C-LS / R-SUPPLIED | UnsupportedInference: unknown influence of supplied W; C-RECIPE / R-RECIPE |
| All-ordinal ULS / identity | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | Zero influence: C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal DWLS / diag NACOV | C-GOLD, C-ORD-MATRIX / R-ORD-MATRIX | C-GOLD, C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal WLS / inverse NACOV | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal GLS / NT | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | UnsupportedInference: NT weight influence not derived; C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal DLS(a) | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | UnsupportedInference: DLS weight influence not derived; C-ORD-MATRIX / R-ORD-MATRIX |
| All-ordinal LS / supplied W | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | C-ORD-MATRIX / R-ORD-MATRIX | UnsupportedInference: unknown influence of supplied W; C-ORD-MATRIX / R-ORD-MATRIX |
| Mixed ordinal / DWLS | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | NumericIssue: mixed weight influence not implemented; C-MIX-MATRIX / R-MIX-MATRIX |
| Mixed ordinal / WLS | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | C-MIX-MATRIX / R-MIX-MATRIX | NumericIssue: mixed weight influence not implemented; C-MIX-MATRIX / R-MIX-MATRIX |
| Mixed ordinal / ULS | C-MIX-MATRIX / R-MIX-MATRIX rejects fitting ULS | C-MIX-MATRIX / R-MIX-MATRIX rejects fitting ULS | NumericIssue: mixed robust score supports DWLS/WLS only; C-MIX-MATRIX / R fitting refusal | Same core NumericIssue / R fitting refusal |
| Prepared ordinal association ML | UnsupportedInference: LS score is not the ML-target score; C-ORD-MATRIX / R-ORD-MATRIX | Same rejection and gates | Same rejection and gates | Same rejection and gates |

## Conventions and limits

FIML uses analytic observed information, with expected information only for
identification rank. Complete ML exposes expected and observed geometry.
The complete ML R adapter refuses model-implied robust equality releases
(the core accepts caller Gamma_NT); empirical robust releases are gated in
R-ML. R mixed ULS is refused at fit_model(), while its C++ ordinary score
workers exist. These adapter guards currently provide explicit R error text,
not a PostError enum.
Continuous LS uses the expected residual-Jacobian geometry and refuses observed
bread; estimated weights require empirical moments. Caller-Gamma R adapters and
a separately derived observed/estimated-weight score projection remain task-4.
Pairwise moment sources remain MCAR; no MAR law is inferred for those routes.
Mixed ordinal expansion and association-ML score contracts remain 0.3.0 work.

ML2S distinguishes sampling groups from missingness patterns. Its ordinary MI
is explicitly `naive_stage2`, evaluated on Stage-1 EM moments with the fitted
Stage-2 objective. Robust scaling uses Stage-1 moment covariance. lavaan's
ordinary two-stage `modindices()` uses unstructured information; it is not a
structured naive-Stage-2 reference fixture.

The grouped reductions check the same raw-data covariance construction on both
sides, with unequal n/N weights. They do not establish reference-law calibration
under a misspecified multi-group mean restriction. Group-allocation and score
centering follow the [sampling-law scope](../scope.md#group-allocation-and-likelihood-score-covariance);
random missingness patterns are not sampling groups. A separate fixed-allocation
extension or a choice to center between-group score means is not validated by
these algebraic gates.

GLS/WLS golden parity is **transported**, not raw. For the fitted covariance
residual r, nuisance moment Jacobian J and candidate derivative d, define
v = d - J(J'WJ)^(-1)J'Wd. The core fixed-W statistic is
MI = N(v'Wr)^2/(v'Wv), and EPC = (v'Wr)/(v'Wv).
The new generator independently constructs the one-factor covariance and its
analytic derivatives, using column-major lower-triangle moments including the
diagonal. Both fits set `sample.cov.rescale=FALSE`; the generator verifies that
the frozen sample covariance equals the unbiased N-1 covariance. Installed lavaan 0.7-2's continuous GLS/WLS output uses
k = (N-1)/N in the candidate score, while its expected information remains
J'WJ: MI_lavaan = N(k d'Wr)^2/(v'Wv), EPC_lavaan = k d'Wr/(v'Wv).
The inspected pinned `lav_model_grad` sets GLS/WLS group score weights to
(nobs-1)/ntotal, while `modindices` combines that candidate gradient with N and
the expected-information Schur complement. The generator checks both latter formulas to 1e-8 and preserves raw oracle
MI/EPC in JSON, alongside k and the independent efficient-score result. The
C++ consumer explicitly multiplies its MI by k^2 and EPC by k for the oracle
comparison. This scalar score transport leaves no hidden change of fitted W,
sample covariance, n or oracle values. Small remaining differences reflect the
nonzero fitted nuisance gradient: the core retains J'Wr in the Schur score,
whereas the stationary oracle form assumes it is zero. This is a stated
comparison convention, not an oracle-defect exemption.

## Measurements

Measured on the pinned oracle theta, with no refitting in the C++ consumer.
The table lists maximum absolute differences over the checked target rows.
Raw comparisons preserve the existing fixture's sample-size conventions;
GLS/WLS explicitly use the recorded score transport described above.

| Fixture | MI error | EPC error | MI / EPC gate | Release MI / p error | Release MI / p gate |
| --- | ---: | ---: | --- | --- | --- |
| ML | 3.450044e-5 | 5.133190e-7 | 1e-4 / 2e-6 | 5.231415e-11 / 4.194478e-12 | 1e-7 / 1e-8 |
| Observed FIML | 7.532795e-6 | 1.116287e-7 | 2e-5 / 5e-7 | 4.804921e-10 / 1.864403e-11 | 1e-7 / 1e-8 |
| Raw ULS | 0.059270984 | 0.000940361 | 0.065 / 0.0012 | 0.023441073 / 0.000856438 | 0.03 / 0.0012 |
| Raw ordinal DWLS | 0.013473879 | 0.000601743 | 0.02 / 0.001 | 0.002135256 / 0.001134891 | 0.003 / 0.002 |
| Raw mixed DWLS | 0.829736594 | 0.000195575 | 1.1 retained / 0.0005 | 0.005409003 / 0.001345888 | 0.01 / 0.002 |
| Transported GLS | 3.013589e-6 | 4.491212e-8 | 1e-5 / 5e-7 | No equality targets | — |
| Transported ADF/WLS | 7.960787e-6 | 1.534900e-7 | 1e-5 / 5e-7 | No equality targets | — |

The FIML MI limit is tightened from 0.2 to 2e-5; ordinal DWLS from 0.1
to 0.02; ULS MI from 0.07 to 0.065 and release from 0.05 to 0.03.
ULS and ordinal DWLS cannot use machine-size **raw** tolerances: their leading
MI ratio is (N/(N-1))^2 and EPC ratio N/(N-1), with N=301 and N=360,
respectively. The small residual after that transport is optimizer/polychoric
precision. Their frozen raw oracle values have not been rewritten.

Mixed ordinary MI is **limited validation**, not raw lavaan parity. Its current
worker uses score=2N J'Wr and information=2N J'WJ, while its equality-release
worker uses N. The measured MI/oracle ratio is approximately
2(N/(N-1))^2. The existing 1.1 absolute MI tolerance is retained solely as the
pre-existing regression gate; it is not a justified precision tolerance and
must not be used to claim oracle agreement. Resolving the mixed MI target/scale
is `TASK-33.4` in the local work board (0.3.0); this lane changes no numerical convention or default.
The new mixed matrix gates establish the retained ordinary/robust reduction
and precise refusals, not a solved oracle discrepancy.

The GLS/WLS generator checks the stationary candidate-score oracle form to
1e-8, then checks the independently reconstructed efficient MI/EPC to 1e-5.
The C++ consumer compares the independent results to 1e-7 relative tolerance
and the standardized EPCs to 1e-6. The remaining 3e-6/8e-6 transported MI
errors are explained by recorded nonzero nuisance scores, not finite-difference
steps (the generator uses analytic derivatives). No tolerance was widened.

## Verification

On 2026-10-02, the full `fast` C++ suite passed all 1,430 tests with
`just jobs=1 test`. The full installed `magmaanlab` testthat suite completed
without test failures or errors; its existing two-level tests emitted two
admissibility warnings and skipped the unavailable multi-group lavaan oracle.
The lane package was installed by `just jobs=1 r-dev fast` into
`/tmp/rlib-task-3`, reusing the fast core and its normal O1 development glue.
The pinned generator's independent guards, focused recipe gates, layering
guard and tracked-file checks also passed.
