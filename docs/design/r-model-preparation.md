# R model preparation and repeated estimation: interface audit

Status: source audit and proposed contract, 16 September 2026. No prepared-model
API has been implemented by this audit. Function names below describe roles,
not a new exported API. The existing staged C++ types should be assessed before
adding another implementation of model preparation in R.

## Finding

Both the interface and the Oslo benchmark contributed to the ordinal timing
problem. `model_spec()` stores an R partable plus syntax/options; it does not
retain a compiled native model. The benchmark excluded that constructor but
still timed `augment_ordinal_partable()` inside every ordinal fit. Preparing
that table outside the timer produces identical complete fitted objects on the
two measured one-factor CFAs.

The installed R package's sampled fit time was approximately 89% augmentation
at p=9, N=500 and 83% at p=12, N=1000. Independently timed medians (ms) were:

| Operation | p=9, N=500 | p=12, N=1000 |
|---|---:|---:|
| Existing R fit wrapper | 5.60 | 8.19 |
| Ordinal table augmentation | 4.76 | 6.54 |
| Native entry with augmented table supplied | 0.51 | 1.42 |
| Post-fit spectrum helper | 0.48 | 1.29 |
| pEBA4 tail | 0.023 | 0.039 |

These are separately timed operations, not an exact additive decomposition.
The native entry still performs conversion, internal construction, optimization,
diagnostics and result construction. Rprof sampled at its platform minimum of
10 ms. Timings describe the installed 9 September package, not a rebuild of the
working tree. The source audit below includes current working-tree changes;
it does not infer their runtime from those older installed-package measurements.

## Estimator-family inventory

| Family | Existing reusable R input | Work still reconstructed by the fit interface |
|---|---|---|
| Continuous ML / ULS / GLS / WLS | model spec + sample moments; explicit W for WLS | Native structure/names/starts from partable, matrix representation and data-dependent starts |
| Continuous SNLLS, Fisher/IRLS, PSD/GMM variants | Same specification/moment convention, separate method entry points | Native context and method-specific preparation; fit inputs are not retained native model handles |
| FIML | Model spec + `df_to_fiml_data()` | Native matrix representation, missingness pack and starts; current implementation additionally constructs saturated H1 moments |
| ML2S / pattern NTML | Model spec + missing-data object; explicit reusable `stage1` on low-level helpers | Model conversion; Stage 1 unless supplied; Stage-2 data/weight work remains dataset-specific |
| Ordinal ULS / DWLS / WLS, ordinal PSD / CatML | Model spec + ordinal statistics | R threshold/scaling augmentation and renumbering, then native ordinal preparation and matrix representation |
| Mixed ordinal DWLS / WLS and PSD | Model spec + mixed statistics | Parallel R augmentation and native preparation; uniform staging must not imply support for currently rejected estimator/parameterization combinations |
| Two-level ML | Model spec + data/cluster IDs | Matrix representation and cluster summaries; wrapper also computes observed-information SEs and H1/fit statistics |
| Native FC-SEM ML | Separate FC-SEM spec and sample data | Syntax passed back to C++; reparsing/building the native FC-SEM structure |
| SAM / noniterative CFA | Specs or partables plus sample moments, dedicated entry points | Separate orchestration rather than the main `magmaan()` dispatcher; SAM defaults to two-step SEs |

Evidence entry points:

- [model_data.R](../../r-package/R/model_data.R): `model_spec`,
  `augment_ordinal_partable`, `augment_mixed_ordinal_partable`, the `fit_*`
  wrappers and `magmaan` dispatcher.
- [fit.cpp](../../r-package/src/fit.cpp): `fit_ml_impl`, `fit_fiml_impl`,
  `fit_twolevel_impl`, `fit_ml_fcsem_impl`, ordinal/mixed and frontier fit entries.
- [internal.hpp](../../r-package/src/internal.hpp): `ctx_from_parts` and
  `ctx_from_sample_stats` construct native contexts from partable projections.
- [noniterative.R](../../r-package/R/noniterative.R): dedicated noniterative
  partable/sample-statistic interface.
- [sem.hpp](../../include/magmaan/api/sem.hpp) and
  [sem.cpp](../../src/api/sem.cpp): `api::Model` already owns the model triple
  and matrix representation, and staged estimation consumes model/data objects.
  This is a foundation, not proof that all R variants can already be bound
  through one C++ dispatcher unchanged.

The high-level `magmaan()` accepts an existing model spec, but may rebuild it
when group metadata changes or FIML/ML2S requires adding a mean structure.
The `magmaan_core$estimate_*` aliases can bypass R augmentation when given an
already augmented partable; they still reconstruct native objects.

## Proposed uniform contract

Use the same four stages across estimator families, with family-specific
capabilities and explicit errors for unsupported combinations:

1. **Prepare the model once per structural specification.** Own the model
   triple, compiled mappings/representation, constraints, names and declared
   observation schema. Ordinal schema includes ordered variables, category
   counts/levels, groups and parameterization. Resolve threshold/scaling row
   layout here. Accept explicit schema or derive it from a prototype once.
2. **Prepare data once per dataset.** Own sample moments, thresholds,
   polychorics, weights, missingness patterns or clustered summaries as needed.
   Choose expensive data ingredients explicitly. These can be reused for
   several fits to the same data but change for each simulation replication.
3. **Estimate from the prepared model and data.** Select estimator, optimizer,
   bounds and starts explicitly. Preserve numerical audits and admissibility
   diagnostics. Refresh data-derived starts; do not freeze threshold estimates
   from the prototype. Do not silently warm-start from a prior fitted result.
4. **Request inference explicitly.** Expected/observed information, SEs,
   goodness of fit and nested tests are separate consumers of the model/data/fit.
   Put reusable H1/Stage-1 results under explicit caller ownership. Computing
   inference that is intrinsic to a named procedure must be explicit in its
   contract, not hidden by an estimate-only label.

This does not require one giant fitted object or one overloaded estimator
implementation. Existing estimator-specific entry points can retain their
names while accepting the same kind of prepared model/data inputs. Native
FC-SEM and ordinary latent-variable models may keep distinct validated model
families with the same staging pattern.

A reusable model must not silently rebuild itself because the caller changed
`ordered`, meanstructure, identification, groups or category schema. Such changes
require explicit preparation (the one-shot convenience API can still do this).
Schema validation must cover variable identity/order, levels, group order and
parameterization. Caller-supplied starts and constraints must be distinguished
from data-derived defaults. Persistence/worker handling must have an explicit
reconstruction policy; mutable optimizer workspaces must not be shared across
concurrent fits through a supposedly immutable model handle.

## Acceptance evidence for implementation

- Fresh and prepared paths agree on parameters, objective, partable, convergence
  and admissibility diagnostics for each supported estimator family.
- Structural preparation counters stay zero inside repeated prepared fits;
  timing alone is not evidence that construction has moved.
- Two different datasets with the same schema reuse one model correctly,
  including changed thresholds, N, missingness and cluster membership.
- Explicit user starts/threshold restrictions survive; prototype-derived starts
  never leak into a new dataset. Unsupported schema changes fail explicitly.
- Choosing a different supported estimator or information matrix does not
  mutate the model or data or recompute unrelated inference.
- Benchmarks separately measure model preparation, data preparation, estimation
  and requested inference. Compare matching timing boundaries on both engines.

The implementation priority and remaining work belong in
[the active backlog](../backlog/todo.md), not in a second roadmap here.
