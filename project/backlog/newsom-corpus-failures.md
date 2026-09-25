# Newsom corpus: survey failure classes

## Current reproduction (2026-09-25)

The corrected-corpus optimizer experiment (`engineering/17-corpus-optimizer-recovery`)
reran `newsom_2015_ex5_4` and `newsom_2015_ex5_4c` on a pinned engine including the
start-policy and parameter-reporting fixes (`0a0671ce`). Ordinary L-BFGS, PORT,
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
