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
`magmaan` simulation prerelease; ordinary-user integration is deferred until
the C++ and lab gates below are complete.

## Implementation sequence (planned)

Direction adopted 2026-10-01: implement the full documented linear SEM model
language in **C++ first**, then expose each validated increment through
**magmaanlab**. Integration into **r-magmaan follows later** and is tracked in
the [active backlog](../backlog/todo.md#eqs-language-extension).
This section is a work plan; the supported subset above remains the implemented
contract. The [source inventory](eqs_source_inventory.md) owns manual evidence
and unresolved behavior.

### C++ dependency order

1. **Resolve declarations before lowering.** Extend the normative
   `eqs_grammar.ebnf` first, then introduce an internal parsed-document stage
   in the EQS frontend. Retain numeric variable identities, aliases, declared
   dependent/independent roles, equation order, segments, parameter references
   and source spans before emitting `FlatPartable`. Resolve labels/ranges and
   documented overrides by identity, not by displayed names. Keep the existing
   single-group entry point working; specify any new schema/options input in
   `parse/eqs_parser.hpp`, `compat/eqs/model.hpp` and `api/sem.hpp` together.
   Bound expansion sizes and diagnose unsupported headers at their source spans.
   **Gate:** independent named-row/start expectations for labels, ID limits,
   duplicate RHS recovery and lexical boundaries, including move lifetimes.

2. **Prove and implement general-equation representation.** Use a small
   independently specified linear system before broad shorthand expansion.
   Cover an observed indicator used as a structural predictor, a nonunit/free
   error path with fixed error variance, and error/predictor covariances.
   Design an exact augmented LISREL mapping through the shared `spec::build`
   and `model::build_matrix_rep` machinery. Preserve original estimands in
   `LatentStructure`, reference/name mappings in `LatentNames`, and hints in
   `Starts`; distinguish generated coordinates from user parameters. Do not
   fold a free error coefficient into its variance. Shared/multiple/errorless
   error graphs remain derived cases until language acceptance is resolved.
   **Gate:** direct linear-system covariance/mean calculations, parameter
   perturbations and analytic-versus-finite-difference derivatives. Existing
   lavaan models must retain their rows and numeric behavior. Identification
   failures belong to evaluation/fitting, not an observed-indicator syntax rule.

3. **Expand MODEL into the same resolved document.** Add ON, VAR, COV, PCOV
   and EQM recognition, retaining assignment order and source provenance.
   Apply shorthand residual/marker rules before shared building; disable
   competing lavaan automatic fixes. Keep explicit equations separate from
   shorthand defaults. Add RELIABILITY only after resolving its contradictory
   printed expansion. **Gate:** independent expansions for Cartesian/paired
   paths, range overrides, bare-VAR fixed-one versus omitted free variances,
   residual generation and explicit unit-variance identification. Keep ambiguous
   marker cases unsupported until their source/runtime question is settled.

4. **Add means, segment-local models and minimal schema.** Recognize V999
   paths as intercepts and map them to Nu/Alpha without a constant data column.
   Read GROUPS and segment boundaries; adapt existing group blocks to unequal
   segment models rather than blindly replicating one model with `n_groups`.
   Retain segment-local starts and explicit fixed-zero/free latent intercepts.
   Specify ANALYSIS and CATEGORY metadata independently of data payloads and
   estimation requests. Group count/order must agree with the supplied schema.
   Levels remain distinct from groups; HLM/DEFINE needs additional language
   evidence and a separately supported level contract.
   **Gate:** direct/indirect means, asymmetric group models and starts, stable
   observed-column mappings, and explicit refusal of unavailable fit targets.

5. **Resolve restrictions after all expansions and groups.** Map directed,
   variance, symmetric covariance and group-qualified references to original
   parameters, using existing expression/equality/linear-constraint machinery.
   Implement equality chains, constants, SET role classification and forced-free
   exceptions. Preserve free/fixed semantics and validate supplied starts.
   Parse explicit inequalities; route simple parameter bounds through supported
   shared bounds where exact, and retain general restrictions with explicit
   unsupported-fit diagnostics where no enforcing backend exists. The language
   goal does not require a new optimizer or active-bound inference.
   **Gate:** named restrictions and free-parameter counts, constrained implied
   moments, SET exclusions and cross-group chains, plus a fit-path check that
   no accepted restriction is silently ignored.

6. **Close coverage and runtime uncertainties.** Add growth examples through
   ordinary paths/means/restrictions and classify remaining model-envelope
   directives. Build a supported/unsupported rule matrix keyed to inventory
   IDs. Run the focused probes when EQS access or authoritative expansions
   become available; keep unverified repairs identified explicitly. Full EQS
   job execution, numerical defaults and estimator parity remain outside scope.

### Lab exposure and rebuild contract

After each C++ gate, extend `eqs_model()` and its thin compiled adapter; the lab
may expose capabilities incrementally with explicit unsupported-fit results.
It inherits numerical/inference capability checks from C++, with no R parser or
second SEM implementation. Lab freedom does not relax parameter preservation.

Before exposing aliases, means, groups or restrictions, remove the assumption
that EQS specs can always be rebuilt from `to_lavaan_syntax(flat)`. The current
row-only projection cannot carry the full new contract. Retain source language,
original EQS text, resolved column/alias mappings, schema and construction
settings. Rebuild through the EQS C++ entry point or a lossless portable triple
with schema. A lavaan string is an optional interoperable projection only where
it preserves all supported rows, starts and restrictions. Structural overrides
must be checked against explicit EQS choices; do not reapply lavaan defaults.

Lab gates cover construction/partable inspection, fresh versus prepared fits,
rebuild/refit and save/reload into a fresh worker. Compare names, groups,
constraints, explicit starts and implied covariance/means as well as estimates.
Run installed-lavaan integration comparisons only for independently equivalent
models and conventions. Refresh vendors after canonical C++ changes; validate
generated exports, the fast binding loop and portable installation as applicable.
Do not edit generated core mirrors directly.

### Completion and later ordinary-user integration

An increment is complete when its normative grammar, C++ lowering, model-triple
contract, component fixtures and lab round trips agree. Parsing and fitting
capability must be reported separately. Markers or ambiguous shorthand awaiting
evidence cannot be advertised as full EQS compatibility. Document the rule
matrix and remaining source gaps before declaring the language target complete.

Only after these gates, extend the ordinary reusable-model constructor to
accept validated EQS specs, preserve schema during reconstruction, and apply
the existing ordinary inference policy. The later task must cover simulations,
worker reconstruction and explicit refusals without changing ordinary defaults.
No changes in `r-magmaan/` belong to the C++/lab implementation milestones.

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
