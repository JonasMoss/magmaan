# EQS model frontend

The [EQS grammar](eqs_grammar.ebnf) is normative for `parse::EqsParser`.
This frontend accepts model sections, not complete EQS analysis jobs. It adds
an input language to the existing model triple and numerical implementation;
it does not run EQS or promise reproduction of its estimator defaults.

The [source inventory](eqs_source_inventory.md) extracts the documented SEM
language rules, page references, adapter requirements and unresolved checks
for extending this subset. It is evidence for future grammar changes, not a
claim that the additional syntax is implemented.

## Supported subset

Single-group continuous covariance models using `/EQUATIONS`, `/VARIANCES`
and `/COVARIANCES`, optional `/END`, case-insensitive variable IDs/keywords,
unambiguous section prefixes of three or more letters (`/EQU`, `/VAR`,
`/COV`), `!` comments, signed numeric coefficients, fixed/free values and
starts. Lists use commas, and same-family ascending ranges use `-` or `TO`.
Later variance/covariance specifications override earlier ones. Each range
expands at most 10001 entries. Scientific exponents need a sign (`1e-3`),
which distinguishes them from `1E1`, a fixed unit path on error `E1`.

Every equation has exactly one distinct unit error: E for an observed V,
D for a factor F. Error numbering is arbitrary. Every factor has an observed
indicator. Unspecified independent variances are free; unspecified covariances
are zero. Fixed coefficient one can be written `F1` or `1F1`; `1*F1` is a free
coefficient starting at one. Without an explicit start, magmaan's existing
start policy is used; EQS's automatic starting algorithm is not emulated.

Rejected explicitly: `/MODEL`, `/LABELS`, `/SPECIFICATIONS`, data/output
sections, means/intercepts (`V999`), multiple groups, constraints, shared
errors, nonunit error paths, error/predictor covariances, E/D
cross-covariances and observed indicators participating in structural
regressions. The latter need additional work in the shared model machinery;
separate observed predictors and purely observed path models are supported.
Unsupported job instructions are never skipped.

## Lowering and ownership

`parse::EqsParser::parse(source, observed_names)` first parses equations and
assignments, then resolves variable roles and error ownership. It emits the
same `FlatPartable` consumed by `spec::build`. Original source bytes and spans
are retained; normalized/mapped names live in the flat model's owning
`symbol_text` buffer, so moving a parsed model does not invalidate its views.

- V equations with F predictors become measurement rows; V predictors become
  structural regression rows. F equations become latent regression rows.
- E variances/covariances become their owning V residual rows; D
  variances/covariances become their owning F disturbance rows.
- Independent V/F variances and specified covariances remain model rows.
- Fixed values belong to `LatentStructure`; free-parameter hints to `Starts`;
  canonical factor names and optional mapped observed names to `LatentNames`.
- `compat::eqs::build_options()` disables automatic covariance insertion,
  marker fixes, single-indicator residual fixes and fixed.x handling. All
  independent variances are materialized by the frontend.

The existing `matrix_rep`, evaluation, fit and inference paths consume the
triple. `api::Model::from_eqs` / `api::model_from_eqs` compose those stages.
`compat::eqs::to_lavaan_syntax` supplies a rebuildable textual projection of
already-lowered rows; it is not an independent semantic translator.

## R usage

The methods-development package exposes `eqs_model()`:

```r
model <- magmaanlab::eqs_model('
/EQUATIONS
V1 = 1*F1 + E1;
V2 = .7*F1 + E2;
V3 = .8*F1 + E3;
/VARIANCES
F1 = 1;
E1-E3 = .5*;
', observed_names = c('x1', 'x2', 'x3'))
fit <- magmaanlab::fit_model(model, data, estimator = 'ML')
```

`observed_names` is in EQS data-column order: Vn maps to its nth entry.
Without that argument, data columns retain V-number names. Supply the actual
EQS column order; latent names remain F-number names. The returned
model specification retains `eqs_source` and an explicit lavaan projection in
`syntax` for existing rebuild/refit helpers. Prepared-model construction and
lab fitting accept the same spec. EQS is excluded from the ordinary-user
`magmaan` simulation prerelease; future adoption there remains undecided.

## Evidence and limits

Language semantics follow Bentler (2006), *EQS 6 Structural Equations Program
Manual*, Chapter 3, especially pp. 71–74. The public manual is available at
<https://www3.nd.edu/~kyuan/courses/sem/EQS-Manual6.pdf>. Third-party manuals
remain outside the tracked tree.

`unit/eqs_parser_test.cpp` checks fixed/free distinctions, unit factor variance,
error ownership, covariance defaults, structural disturbances, mapping/move
lifetimes, range overrides and rejection diagnostics.
`golden/eqs_golden_test.cpp` consumes the frozen `eqs.json` fixture, generated
with the pinned installed lavaan by:

```sh
Rscript cpp/tests/tools/regen_oracle_eqs.R
just test-area spec EQS
```

The generator independently specifies paired EQS/lavaan models and synthetic
covariance summaries. It does not call magmaan to derive the lavaan model.
Gates compare semantic parameter rows and starts, ML estimates, implied
covariances, expected-information SEs, df and chi-square. Tolerances are 1e-12
for fixed values/starts and 1e-5 for fitted numeric quantities. Marker and
unit-variance examples share sample moments; the two-factor default-zero
example deliberately has nonzero sample cross-factor covariances. A structural
example includes an observed predictor, a disturbance, predictor/factor
covariance and correlated measurement errors.

Binding integration tests additionally compare installed lavaan. These
establish lowering and
shared-engine correctness for the supported subset. They are not a live EQS
parser or numerical oracle, and no full EQS compatibility claim is made.
