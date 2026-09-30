# Source audit of six corpus accuracy rejections

**Update:** the Geiser marker default-start failure described below was traced
to Reduced-representation start construction and repaired. See the
"Resolution" section; the earlier diagnostics remain a record of the pre-fix
behavior.

**Update (2026-09-25):** the whole textbook corpus has since been
source-audited and rebuilt; the Little translation was replaced and every
Little case is verified against LISREL's own output. See
[Full corpus source-fidelity pass](#full-corpus-source-fidelity-pass-2026-09-25).
The sections before it describe the corpus as it was on 2026-09-24.

Audited 2026-09-24 against original companion inputs, not just lavaan output
from the translated syntax. Fitting probes used magmaan `5bacca25dded`,
R 4.6.1 and lavaan 0.7-2. This is an input-fidelity audit; it does not alter
the existing fixture snapshots or certify the other corpus cases.

## Little: five incorrect translations

Affected IDs in `cpp/tests/fixtures/little/continuous_reference.json`:

- `ch8_fig3a_unconstrained_4wave_negaff`
- `ch8_fig3b_unconstrained_4wave_negaff`
- `ch8_fig4a_gc_linear_4wave_negaff`
- `ch8_fig4b_gc_linear_4wave_negaff`
- `ch8_fig4d_gc_linear_4wave_negaff`

Evidence: the corresponding original `.LS8` files under the ignored
`external/textbook-corpus/raw/little/models_lisrel/`, also retained as
`cases/little_2013/little_2013_<id>/source/original.LS8` in that corpus.
All five originals specify four waves, unit intercept-factor loadings at
every wave, equal residual variances, zero observed intercepts and free
growth-factor means. The translations retain only the first intercept-factor
loading, omit the residual equalities and omit the mean structure. The
unidentified single-indicator factor is therefore a translation defect,
not a property of these published examples.

The intended slope loadings are:

| Figure | Loadings at waves 1–4 |
|---|---|
| 3a | 0, free, free, 1 |
| 3b | 0, 1, free, free |
| 4a | 0, .333333, .666666, 1 |
| 4b | 0, 1, 2, 3 |
| 4d | -3, -2, -1, 0 |

Figure 4a additionally loses the fractional loadings. Its fixture consequently
contains only two observed variables, despite the original selecting four.

The mechanism is visible in `cpp/tests/tools/build_little_corpus.R`:
`fixed_values()` recognizes only the first matrix reference after each `VA`
and its numeric pattern cannot recognize a leading-dot decimal. The
translation does not implement `EQ` residual constraints or the `TY`/`AL`
mean specification. The raw-data manifest derives `meanstructure` from an
explicit mean-data section, which these raw-data inputs do not have.
The retained-case gate does not reject these unsupported constructs.

Local source-faithful probes used all four raw-data columns, listwise deletion
(N=1684), the loadings above, one shared residual-variance label, and
`growth` mean conventions. Results:

| Figure | lavaan df | lavaan ML discrepancy f | Default ordinary-first | Default direct PSD |
|---|---:|---:|---|---|
| 3a | 6 | .008794812 | accepted ordinary | accepted |
| 3b | 6 | .008794812 | rejected; PSD budget exhausted | budget exhausted |
| 4a | 8 | .009700416 | accepted ordinary | accepted |
| 4b | 8 | .009700401 | accepted ordinary | accepted |
| 4d | 8 | .009700401 | accepted ordinary | accepted |

Accepted magmaan fits agree with the displayed lavaan discrepancies. For 3b,
PSD reaches and certifies the lavaan solution when the lavaan estimates are
explicitly supplied as `ustart`. That diagnostic changes the start policy;
it is not a default-start benchmark success. Rounded fractional loadings in
4a explain why its objective is not exactly that of 4b/4d.

## Geiser: correct structural translation, default-start failure

`latent_path` in `cpp/tests/fixtures/geiser/gls_reference.json` matches the
measurement and regression specification in Geiser's Figure 3.15 companion
Mplus input. The source is `Chapter 3 (SEM)/4_Path_Analysis/2_latent_path_analysis.inp`
and its accompanying `.out`, downloaded from the publisher's
[Chapter 3 archive](https://www.guilford.com/add/geiser/Chapter_3_SEM.zip?t=1).
Archive SHA-256:
`8fbdaaa009520e28c250522f44b9d8ed03072c1fdeefcd24307dfe4c134f4a70`.
Original files remain outside tracked source.

The six selected raw-data columns agree exactly with the corpus copy. The
fixture has the same free-loading pattern, three directed latent regressions,
variance structure and free observed means. `fixed_x=TRUE` is immaterial here:
there are no observed exogenous predictors. The separate corpus case metadata
says `meanstructure=false`; that metadata does not match the Mplus analysis,
although the GLS fixture used for the fitting probe correctly says `true`.

The fixture covariance uses divisor N-1, matching the raw sample covariance
to 3.5e-13. The source ML analysis uses divisor N. Its published output reports
chi-square 5.509 on 6 df; lavaan on the original data gives 5.509076 and
f=.001671443, with proper covariance matrices. Free intercepts allow the
covariance divisor change to be absorbed by rescaling fitted covariances;
it does not explain the large default-start objective.

Default magmaan L-BFGS fails its line search; default PSD terminates near
f=.246792170 and fails the curvature check. With the lavaan estimates explicitly
installed in the partable's `ustart` column, PSD passes at f=.001671443.
The default result is therefore demonstrably inferior to an available proper
solution; a small projected gradient did not establish a minimum.

## Consequences

Agreement with lavaan on mistranslated syntax does not validate fidelity to a
textbook. The six original accuracy rejections must not be described as six
failures on faithful published specifications. Five have input errors; the
sixth has a correct structural specification and a default-start failure.
Correcting the five inputs does not guarantee universal default-start success:
3b remains a separate numerical problem.

Repair and source-audit the Little converter/corpus upstream, then regenerate
the derived fixtures and dependent corpus geometry snapshots. Audit other
retained cases for the same converter limitations before renewed aggregate
timing or reliability claims. Preserve old results as records of their actual
inputs; new scientific claims require a new pinned run. Investigate Geiser and
corrected Little 3b as start/solver regressions without substituting oracle
starts into a default-start performance comparison.

## Follow-up: ordinary optimizers and initialization

On the Geiser fixture statistics, both ordinary optimizers fail with omitted
start controls: L-BFGS reports a line-search failure after 779 evaluations;
SLSQP exhausts 5000 evaluations near f=.0515458. Neither is a PSD-specific
failure. With covariance and means rescaled to unit observed variances, both
return accurate, proper fits at f=.001671443: L-BFGS uses 243 evaluations and
SLSQP 118. This is a diagnostic on an equivalent rescaled model, not a new
accepted timing run.

The initialization/scaling policy is partly shared but not uniform in effect:

- R complete-data `fit_ml`, direct PSD and ordinary-first PSD all request
  `scaled-fabin` when start controls are omitted. C++ `api::ml()` also selects
  `ml_starts()`. Low-level fitting functions instead take an explicit vector.
- `ml_start_values()` attempts FABIN in a unit-latent-variance representation
  and transports it to the original marker identification when its eligibility
  checks pass; otherwise it returns native FABIN. This is not the same as
  standardizing observed data before deriving starts.
- Geiser takes the `native-fabin-fallback` branch. Inspection through
  `fit_start_values` shows identical starts for `scaled-fabin`, `fabin3` and
  `simple`: free loadings .7; structural slopes zero; observed residual
  variances equal to sample variances; latent variances .05; free intercepts
  zero. Alternative named loading-start methods also give the same observed
  fit failures. The structural representation prevents the intended FABIN
  improvement from reaching this case; investigate its matrix-coordinate
  eligibility rather than treating the label as evidence of scaled starts.
- Ordinary L-BFGS and SLSQP both enable sample-based coordinate scaling in
  `ml_optim_options()` and `drive_ml_scalar`. Scaling optimizer coordinates
  does not replace the initial point with a scale-equivariant initialization.
  PSD uses a separate information-based diagonal preconditioner.
- R's explicit `start="default"` currently resolves to `fabin3`, unlike an
  omitted ML start control. `fit_start_values()` itself defaults to `simple`.
  The helper's comment describing ordinary defaults as FABIN3 is stale.

As a narrower diagnostic, computing starts on standardized statistics and
transporting only those parameter values back to original units rescues
ordinary SLSQP (406 evaluations), but L-BFGS still fails. This does not yet
establish whether initialization alone can resolve L-BFGS's scale sensitivity.
The comparison uses the same model, explicit starts, and existing coordinate
scaling; no oracle estimates are supplied in this diagnostic.

## Resolution: Reduced representation bypassed measurement start heuristics

The numerical evaluator correctly lowers observed measurement loadings into
Beta, observed residual variances into Psi, and observed intercepts into Alpha
when building the Reduced representation. The start producers were still
looking for those quantities in Lambda, Theta and Nu respectively. Consequently
FABIN skipped the loadings, residual variances used full sample variances, and
observed intercepts used zero instead of sample means.

The correction gives simple/FABIN start construction a semantic view of these
cells without changing the evaluator, parameterization, optimizer or objective.
A regression property checks identical parameter starts for a CFA model and
the equivalent model with a fixed-zero latent regression, under both marker
and std.lv identification and with nonzero means and unequal observed scales.

On the source-verified Geiser fixture, the corrected marker starting vector
matches lavaan's *initial* values exactly (not its final estimates). With
single-threaded numerical libraries, default ordinary L-BFGS, ordinary SLSQP
and direct PSD-SLSQP all pass at f=.00167144309659, in 199, 187 and 93 objective
evaluations. The ordinary-first policy can accept the ordinary fit.

A controlled ablation retains the original optimizer settings and changes
only three groups of initial values. Repairing residual variances and observed
means together already recovers all three routes (172/200/92 evaluations);
repairing all groups gives the native corrected result above. Fixing loadings
alone does not rescue L-BFGS or direct PSD. This is a start-vector defect,
not evidence that PSD itself necessarily causes the earlier failure.

The reduction of observed loadings into Beta dates to commit `7be42999`
(2026-05-25); the mismatch is not demonstrated to be a regression introduced
in the last few days. Recent default/policy changes and reliance on different
start routes can expose it. This diagnosis does not settle every historical
performance difference or the separate std.lv L-BFGS difficulty.

## Full corpus source-fidelity pass (2026-09-25)

Every book in the optional corpus (`external/textbook-corpus`, now v3.0.0 and
its own local git repository) was checked against the book software's own
output or the author's own lavaan call, not against lavaan run on our
translation. Per-case evidence lives in the corpus's `docs/audit/`. A case
is retained only if it reproduces its source.

| Book | Before | After | Evidence |
|---|---|---|---|
| Little 2013 (LISREL) | 38 cases, none exact | 106 `book_verified` | the `.OUT` LISREL wrote for each input |
| Newsom 2015 (lavaan R scripts) | 137, 117 from the 2024 2nd edition | 88 first-edition cases | the author's own lavaan call, rerun in a sandbox |
| Kline 2023 (lavaan R scripts) | 37 | 37 (34 `book_verified`) | the author's call and companion `.out` transcripts |
| Geiser 2013 (Mplus) | 28 | 28 `book_verified` | companion `.out` files |
| Brown 2015 (Mplus) | 19 | 19 `book_verified` | the book's printed output, Mplus Demo reruns |
| Mplus User's Guide v8 | 29 | 26 `book_verified` | shipped `.out` files |
| Muthén et al. 2017 (Mplus) | 9 | 19 (18 `book_verified`) | shipped `.out` files |

**Little.** `cpp/tests/tools/lisrel_translate.R` interprets the full LISREL 8
command stream (sequential FR/FI/VA/ST/MA, EQ with LISREL's leader rule, CO,
IR, vector ranges, `FI=` data streams, PRELIS system files, multi-group
IN/PS/SP inheritance including replicated within-group ties and detachment of
IN elements). Each translation is verified against the `.OUT`: the free, fixed,
equality and `Constr'd` pattern, every printed estimate (fixed values
included), df, the minimum-fit-function chi-square, N and the input moments,
refitting with `likelihood = "wishart"` because LISREL uses divisor N-1. 106 of
108 inputs verify; the two others ship fatal-error outputs. The old converter
produced no exact case: it lost every `CO` effects-coding constraint and every
`EQ` invariance constraint (the "weak"/"strong" cases were configural), read
only the first `VA` element and no leading-dot values, ignored `TY`/`AL`, and
stored the correlation matrix for `MA=CM` inputs that supply `KM` and `SD`.

**The other books.** Their corrections were substantive too: Newsom's
extractor let second-edition scripts overwrite first-edition ones and dropped
`meanstructure`, `missing`, `group` and `theta` options; Kline's lost
`sample.cov.rescale = FALSE` and `:=` rows; Geiser's Chapter 3 lacked mean
structures and listwise samples and three growth fixtures had free intercepts
with zero growth means; Brown had two misidentified models and a wrongly fixed
formative loading; the shared Mplus translator fitted censored/count outcomes
as normal, freed x variables, missed growth defaults and ignored MISSING codes
and MLR.

**Fixtures regenerated.** `little/` now holds the 48 verified single-group
cases with at most 18 observed variables (the 1 MB file limit); 21 multi-group
and 37 wider or bounded cases are verified in the corpus and listed as
retained-not-tested. `newsom/`, `mplus_sem/`, `geiser/` (by the new
`regen_geiser_fixtures.R`), `textbook_corpus/` and `psd_ml/corpus_geometries.json`
were regenerated; the overlap graph now links cluster members by a spanning
star so it grows linearly. The `mplus_sem` golden now resolves fixed-x moments
before its implied-moment check and fails on theta/df/implied mismatches that
were previously only messaged.

**Consequences for earlier claims.**

- The advisory PSD-ML corpus audit, rerun on the corrected fixtures (102
  continuous cases), finds ordinary ML inadmissible in two cases, both Geiser
  (`cfa_second_order`, `growth_quadratic`). The four "inadmissible Little
  cases" of the earlier 97-case audit were artifacts of the mistranslations.
  `psd_ml/corpus_geometries.json` keeps its two Little-data geometries as
  explicitly labelled synthetic specifications (`set = "synthetic"`), not as
  Little's models.
- The sem-psd supplement's corpus timing bank used the mistranslated inputs
  and needs a new pinned run on the corrected corpus.
- magmaan's default ordinary fit (NLopt L-BFGS from default starts, the
  skipped Little/Newsom continuous golden run with `--no-skip`) fails with a
  generic solver failure on 12 corrected Little cases (the 2-by-3 invariance
  models, the Chapter 6 card-sorting simplex and Figure 3b) and stops at a
  worse local optimum on the Chapter 3.11 phantom model. lavaan's default
  starts also miss the printed optimum on 14 Little inputs, which carry the
  verified solution as `start_values`. These are solver/start findings on
  faithful models, not translation defects.


## Recovered cases (corpus v3.1.0, 2026-09-25)

A review of the cases v3.0.0 dropped found that most lacked only a corpus
option or translator scope. Four groups were recovered.

**Categorical Mplus models.** The shared Mplus translator now writes out
Mplus's categorical (WLSMV) defaults: thresholds, scale factors or theta
residual variances, threshold invariance, and the categorical growth
conventions. It reproduces the WLSMV test with lavaan's
`information.expected.mplus`, not `mimic = "Mplus"`, because that mimic also
imposes `group.equal` in categorical multi-group models. This added 14 User's
Guide cases and 1 Muthén case. Two are `book_partial`, each with a proof that
the difference is Mplus precision: ex5.17's theta run stops short of the
optimum on the same data as the exact-matching delta ex5.16, and ex8.29_A2's
first stage is off the probit MLE.

**Newsom theta and group.equal fits.** The 11 fits were retained through new
`model_options` fields, and all 99 Newsom cases reproduce the author's fits.

**Kline's rescale.** Kline's `sample.cov.rescale = FALSE` cases now store the
matrix verbatim.

**Brown tab9.4.** The case declares `mimic = "Mplus"` for the book's MLM
statistic.

Two-level models remain open (see `project/backlog/todo.md`). Their H0 models
reproduce, but both programs under-converge the saturated model at default
settings, and Mplus's two-level MLR scaling is not reproduced.

## Newsom second edition (corpus v3.2.0, 2026-09-25)

The second edition's R scripts became their own book, `newsom_2024`. All 125
fits reproduce the author's calls. Seven needed documented patches for script
defects, and the extractor now keeps FIML fits whose listwise covariance is
singular.

The book stores only the 49 fits that differ from every `newsom_2015` case:
- same N, df, free parameters, objective and estimates counts as a repeat;
- labels, start values and `:=` rows are ignored.

The comparison exposed a first-edition case at a local optimum. The second
edition's ex8.5b is the same model with start values, and it reaches fmin
0.084 where the first-edition call stops at 0.372. `newsom_2015_ex8_5b` now
carries those starts.

The second edition adds seven all-ordinal cases to the WLSMV lane:
- four longitudinal invariance models match exactly;
- two latent-mean models hit the known mean-structure gap;
- one saturated theta model stalls in L-BFGS from lavaan's starts.

## Categorical fits against lavaan (2026-09-25)

Evidence behind the "Categorical models" items in
[`todo.md`](../backlog/todo.md). The faithful categorical cases expose
magmaan gaps, not translation defects. Sources:
- **C++ lane.** `textbook_ordinal_golden_test.cpp` (21 all-ordinal,
  covariate-free cases from lavaan's partable, run at `6e1968c8`): 13 match,
  8 known gaps.
- **R path.** engineering/active/19-newton-verdict-migration `results/current/` (`fits.csv`,
  `rejections.csv`): `fit_model(model, data = raw, estimator = "DWLS",
  ordered, parameterization)` on the single-group cases; corpus `3f5b838`,
  magmaan `aa04285b` plus the ordinal Newton check of `bf64ee3a`, lavaan
  0.7.2. The call passes every option these cases set; none has missing data.
- **lavaan.** `fitMeasures(fit, "fmin")` of the corpus call
  (`textbookcorpus:::.lavaan_args(case, "WLSMV")`). Objectives below are fmin
  in each program's convention; on Mplus ex5.3, where both reach the same
  minimum, they agree to 0.2%.

**The nine Newton-check rejections are model-setup gaps.** Each endpoint is a
saddle of magmaan's own objective, far above lavaan's minimum.

| case | R path | C++ lane | lavaan | cause |
|---|---|---|---|---|
| Mplus ex3.4 | 0.927 | not exported | 4.0e-22 | covariates |
| Mplus ex3.12 | 2.638 | not exported | 0.00082 | covariates |
| Mplus ex3.13 | 2.600 | not exported | 0.0034 | covariates |
| Mplus ex3.14 | 1.566 | not exported | 0.0012 | covariates |
| Mplus ex6.4 | 0.192393 | 0.192393 | 0.00079 | scale factors, means |
| Mplus ex6.15 | 0.862093 | 0.862093 | 0.055 | scale factors, means |
| Newsom 2015 ex9.2 | 1.42593 | 1.42593 | 5.7e-5 | scale factors, means |
| Newsom 2024 ex7.2a | 1.4474 | 1.4474 | 0.022 | means |
| Newsom 2024 ex9.2 | 1.42593 | 1.42593 | 0.00055 | means |

- The five growth and latent change models reach the C++ lane's known-gap
  objectives to six digits, so both paths fit the same wrong model.
- Newsom's 2024 ex9.2 isolates the mean-structure gap. It is the 2015 model
  with the scale factors fixed at 1 (2015 frees them) and start values added.
  magmaan forces the scale factors to 1 anyway, so it fits both editions as
  the same model and reaches the same objective. Its latent change means
  stay at their start value 1.1 (lavaan: 0.09 to 0.11): they do not enter
  magmaan's objective.
- The four User's Guide probit regressions have covariates, which lavaan
  treats conditionally (`conditional.x`) and magmaan has no path for. ex3.4
  is just-identified, so any consistent setup fits it exactly; magmaan's R
  path stops at 0.93. The two other single-group covariate cases (Muthén
  ex8.29_2, Newsom 2024 ex4.2b) fail in stage 1 instead ("mixed ordinal
  stage-1 information matrix is not positive definite").

**Mixed ordinal/continuous fits are accepted above lavaan's minimum.** All
five mixed cases without covariates converge and pass the Newton check.

| case | parameterization | ordinal / continuous | R path | lavaan | ratio | d |
|---|---|---|---|---|---|---|
| Mplus ex5.3 | delta | 3 / 3 | 0.000947 | 0.000946 | 1.00 | 9.0e-8 |
| Newsom 2015 ex5.3a | theta | 8 / 6 | 0.1207 | 0.0744 | 1.62 | 4.9e-5 |
| Newsom 2015 ex5.3b | theta | 8 / 11 | 0.1849 | 0.1456 | 1.27 | 9.9e-5 |
| Newsom 2015 ex5.7a | delta | 8 / 16 | 0.3848 | 0.3027 | 1.27 | 2.4e-4 |
| Newsom 2024 ex5.8b | delta | 8 / 28 | 1.2163 | 1.0817 | 1.12 | 3.6e-4 |

Minima this far above lavaan's, well inside the accuracy budget, mean magmaan
minimizes a different objective. Both parameterizations are affected and the
small Mplus model is not, so the mixed path works in the simple case.
Candidates, in the order to check:
1. The stage-1 statistics magmaan estimates from raw data (polyserial
   correlations, continuous means and variances). Compare them with
   `lavInspect(fit, "sampstat")`.
2. The DWLS weight's continuous block (`lavInspect(fit, "wls.v")`).
3. Scale and residual-variance semantics for mixed longitudinal models.
Evaluating magmaan's objective at lavaan's estimates separates the model
from the data side.

**Newsom 2024 ex1.3c is a flat ridge, not a failure.** In this saturated theta
model, the C++ lane's L-BFGS stops at fmin 5.8e-9 from lavaan's starts, with
the factor variance at 83.2 against lavaan's 88.0. lavaan reaches 1e-16, and
magmaan started there stays there. The R path ends at fmin 4.5e-9 and passes
the Newton check (d = 0.0023, condition number 3.8e5). Under magmaan's
accuracy budget the fit has converged. The known gap is the golden's
parameter tolerance on a poorly determined factor variance.
