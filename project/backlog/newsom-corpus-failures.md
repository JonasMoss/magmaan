# Newsom corpus: survey failure classes

## Current reproduction (2026-09-25)

The corrected-corpus optimizer experiment (`engineering/active/17-corpus-optimizer-recovery`)
reran `newsom_2015_ex5_4` and `newsom_2015_ex5_4c` on a pinned engine including the
start-policy and parameter-reporting fixes (`35ed8471`). Ordinary L-BFGS, PORT,
and SLSQP all return accepted GLS fits at matching objectives. L-BFGS's terminal
gradient infinity norms are approximately 1.7e-9 and 2.7e-7 respectively; the
full-model stationarity residuals are also small. Tighter L-BFGS controls reduce
the gradients further. The historical failure does not reproduce on these
current inputs and numerical paths.

The cancellation diagnosis below is therefore **historical and unconfirmed**.
PORT's noisy-objective status and a non-small terminal gradient do not, by
themselves, locate floating-point error in the evaluator. Do not schedule a
compensated-summation rewrite from this note alone. A current same-point
objective/derivative discrepancy against an independent calculation is the
reopening criterion. This does not establish accuracy on every GLS model, nor
does it revisit the second-edition `ex12_3` case absent from the current corpus.

## Historical survey

**Case ids** refer to the corpus before 2026-09-25, which mixed Newsom's
first-edition (2015) and second-edition (2024) scripts. The rebuilt
`newsom_2015` corpus is first-edition only, so `ex12_*` and some other ids now
exist only in the second edition. The numerical findings are unaffected.

Found when the cross-corpus speed survey
(`papers/snlls-constrained/scripts/run_corpus_speed_survey.R`) ran the Newsom
corpus under fixed-weight GLS. None of the failures are separability
rejections — the SNLLS column classifier rejected **0 of 290** corpus models.

## 1. Two distinct failure mechanisms, lumped together until the audit landed

The original diagnosis here was "magmaan optimizer line-search robustness
gap" covering all three of `ex5_4`, `ex5_4c`, `ex12_3`. The terminal audit
(see [`project/design/terminal-audit.md`](../design/terminal-audit.md))
recomputes `f` and `∇f` at the returned iterate and reveals that these are
two genuinely different problems:

### 1a. `ex5_4`, `ex5_4c` — near-perfect fit, not first-order stationary

The pre-audit framing was "magmaan reaches `f = 0.00301331` (lavaan's
optimum) then the line search fails." The audit shows the *objective value*
is at the optimum but the projected gradient at the returned iterate is
**0.0015 / 0.0073** in driven (constraint-reduced α) coordinates — not
machine zero, not noise-floor tiny. The objective is near-flat in some α
directions; the NLopt line search stopped because no further `f` decrease
was measurable, but the iterate is **not** geometrically stationary.

| model  | lavaan GLS                          | magmaan GLS Full (post-audit verdict)                 |
|--------|-------------------------------------|-------------------------------------------------------|
| ex5_4  | converged, 28 iter, chisq 3.5/2     | LineSearchFailed (audit: gnorm 1.5e-3 ≫ 1e-6)         |
| ex5_4c | converged, 33 iter, chisq 10.8/9    | LineSearchFailed (audit: gnorm 7.3e-3 ≫ 1e-6)         |

The previous ad-hoc salvage at `max(1e-3, 1e3·gtol)` would not have
caught these either — `0.0015 > 1e-3` is borderline and `0.0073 ≫ 1e-3`.
The audit's verdict ("genuinely non-stationary") is honest, and points to the
real next investigation rather than masking it with a looser tolerance:

- **Suspected cancellation noise near a near-perfect fit.** PORT's `IV(1)=8` ("noisy gradient detected") flagged
  a numerical difficulty; neither that flag nor stationarity alone proves
  cancellation in the evaluator. lavaan's own
  `nlminb` (same `drmngb` algorithm) does not trip on the same models, which
  motivated a comparison of evaluators, starts, and solver settings.
- A demonstrated evaluator discrepancy would motivate work on residual /
  Jacobian assembly in `cpp/src/estimate/gmm/`. The location and remedy were
  not established by the historical solver statuses.

**Status:** not reproduced on the corrected corpus; reopen on current
independent evaluator evidence, as described above.

### 1b. `ex12_3` — NLopt L-BFGS stuck early, distinct issue

| model  | lavaan GLS                          | magmaan GLS Full (post-audit verdict)                 |
|--------|-------------------------------------|-------------------------------------------------------|
| ex12_3 | converged, 59 iter, chisq 784.5/297 | LineSearchFailed at f=982 (audit: gnorm large)        |

`ex12_3` is genuinely different. NLopt L-BFGS gets stuck early at `f = 982`
(true optimum `3.06`); PORT (`nlminb` family) and SNLLS both converge it
cleanly. The audit correctly reports non-stationary at `f = 982` — gradient
is large there. Tolerance tuning won't fix this; the fix is either a more
robust starting-value path or harness-level fallback (PORT for this model).

**Status:** open, magmaan core, starting values / NLopt L-BFGS robustness
track. **Not** an evaluator-accuracy issue.

## 2. FIML missing-data models, out of fixed-weight scope

`ex14_4a`, `ex14_4b` fail magmaan with `NonPositiveDefiniteSample` — and lavaan
fails identically (`lav_samplestats_icov(): sample covariance`).

They are Newsom's outcome-dependent pattern-mixture models. The source data
(`health missing.dat`) is genuinely 25-30% missing in `bmi2..bmi6` (panel
dropout), and Newsom fits them with `missing = "fiml"`. The models regress the
BMI indicators on dropout indicators `m1..m5`; under listwise deletion the
survivors' dropout indicators are near-constant, so the complete-data
covariance is singular. Fixed-weight GLS/ULS/ADF require a positive-definite
complete-data covariance, so these models are genuinely out of scope —
correctly, lavaan rejects them the same way.

**The data extraction is correct** — it is Newsom's real panel data. The fix
is not in extraction: `build_newsom_corpus.R` now gates each model on a
positive-definite listwise complete-data covariance; models that fail it carry
`status = degenerate_covariance` in the catalogue and are skipped by the
harness (no model/data files written).

FIML alone is not the test — most of Newsom's 14 `missing = "fiml"` scripts
have a perfectly usable listwise covariance and fit fine under listwise GLS.
The positive-definiteness gate catches exactly the genuinely-degenerate cases
regardless of cause; it flags **5** Newsom models: `ex14_4a`, `ex14_4b`,
`ex7_5a` (FIML pattern-mixture / dropout) plus `ex3_7g` and `ex5_3` (collinear
listwise covariance).

## 3. Second-edition ex6.1c: recoverable finite local minima

TASK-140 re-evaluated the TASK-138 endpoints with the read-only lane-a
`magmaanlab 0.2.0` build (source `8b4f180c`, lavaan 0.7.2, N = 5,335).
The model is generically identified (45/45). Each estimator received one PORT
continuation from its lowest-objective retained endpoint (5,000 iterations /
20,000 evaluations), plus one canonical `frontier_fit_sphere()` PORT fit
(2,000 iterations / 8,000 evaluations, default polishing). The bounded probes
completed within the 15-minute allowance. No library or corpus inputs changed.

| Estimator / endpoint | Objective | Reduced gradient infinity norm | Three smallest Hessian eigenvalues | Admissible | Largest absolute parameter |
|---|---:|---:|---|---|---:|
| ML / retained PORT | 0.09091966926 | 13.90 | 1.877e-7, 562.5, 603.1 | no | 341.06 |
| ML / retained L-BFGS | 0.09092194976 | 0.003396 | 5.427e-4, 561.9, 602.6 | no | 6.837 |
| ML / PORT continuation | 0.09091966926 | 13.90 | 1.918e-7, 562.5, 603.1 | no | 341.06 |
| ML / sphere | 0.09002994972 | 0.02284 | 1.062, 2.114, 7.360 | yes | 3.759 |
| GLS / retained PORT | 0.08513800244 | 1.829e7 | -1543, -585.5, -46.66 | yes | 2192.48 |
| GLS / retained L-BFGS | 0.08472941947 | 1.220 | -0.003576, 0.06424, 0.4598 | no | 18.641 |
| GLS / PORT continuation | 0.08428706355 | 2.646 | 5.056e-8, 458.0, 542.0 | no | 328.85 |
| GLS / sphere | 0.08394354034 | 0.003651 | 1.052, 3.129, 7.521 | yes | 3.594 |

Derivative columns are the analytic `retain_newton_artifacts` reduced gradient
and Hessian at identical caller-coordinate points, in the library's retained
audit scaling; they are not the optimizer's gradient tolerance or
coordinate-invariant eigenvalues. Smallest-eigenvector loadings are mapped
back with `derivative_basis`. For retained ML and its continuation, that
vector is dominated by `cesdna2 ~~ cesdna2` (absolute coefficient about 1)
and `etana ~~ etana` (0.0278); the continuation retains their values 341.06
and -9.474. GLS continuation has the analogous `cesdna4` residual-variance /
`etana` direction (1 and 0.0278), with values 328.85 and -9.130. Retained GLS
PORT's most negative direction instead mixes `etapa =~ cesdpa3` (0.351)
and the repeated `eta* =~ cesdso*` loading (0.262 per row); GLS L-BFGS mixes
`etana =~ cesdna4` (0.751) and `etana =~ cesdna3` (0.610). At both sphere
endpoints the softest direction mixes finite `etana` loadings, chiefly
`cesdna4` and `cesdna5` (ML: 0.744 / 0.544; GLS: 0.783 / 0.483).

**Classification:** recoverable search/coordinate failure with certified
finite, interior local minima for both objectives. Sphere ML and GLS pass
objective consistency, identification and ambient stationarity; Newton
distances are 2.733e-5 and 1.905e-5 against budget 0.01. All variances are
positive (ML range 0.002423–0.22736; GLS 0.002769–0.21860). The retained
paths show severe ill-conditioning and variance cancellation: ML continuation
makes no meaningful progress; GLS lowers its objective while moving to large
oppositely signed variances. These paths do not establish nonattainment,
since sphere finds lower-objective finite admissible endpoints with positive
reduced curvature. Global optimality is unproved. Shared default lavaan
nonconvergence is therefore not evidence of an evaluator defect or an
unattainable optimum. Scratch measurements and raw fits are retained under
`~/.cache/magmaan-logs/task-140/`.

## 4. Second-edition ex8.5c: structural rotation gauge

TASK-138 finds structural rank 18/19 (27 moments), with one rotation gauge
between `lin` and `nonlin` and no deficit directions. Magmaan refuses to
certify even the admissible ML PORT endpoint whose Newton distance is
1.523e-5 against budget 0.01: identification fails despite passed
stationarity. Fresh lavaan ML and GLS defaults report convergence but warn
that the information matrix cannot be inverted and the model may be
unidentified. This is a model identification problem; impose independent
loading restrictions and recheck identification before treating it as an
optimizer failure. TASK-138's raw identification and endpoint records are
retained under `~/.cache/magmaan-logs/task-138/`.
