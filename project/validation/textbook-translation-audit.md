# Source audit of six corpus accuracy rejections

Audited 2026-09-24 against original companion inputs, not just lavaan output
from the translated syntax. Fitting probes used magmaan `99c5d6dbc07a`,
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
