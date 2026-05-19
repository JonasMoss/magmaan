# Non-Binding Ideas

This note is a scratchpad for possible philosophical and scope changes. It is
not the roadmap, not the backlog, and not a commitment. Concrete accepted work
still belongs in [roadmap.md](roadmap.md) or [todo.md](todo.md).

## Public Identity

The current direction is strong: magmaan should remain a C++ SEM engine and
methods-development workbench, not an end-user lavaan replacement.

A sharper public framing could be:

> magmaan is a verifiable SEM engine and research workbench for lavaan-compatible
> linear SEM slices, plus explicit experimental methods primitives.

That wording preserves the useful pieces:

- lavaan remains the oracle where compatibility is claimed;
- methods developers can inspect and compose the primitive graph;
- the package does not promise lavaan-style one-shot ergonomics;
- experimental methods work can live in the repo without pretending to be
  ordinary supported SEM functionality.

## A Practical Status Model

The main question is how to distinguish supported work from experimental work
in a way that is visible in code, docs, tests, and R bindings.

This should separate two questions that are easy to conflate:

- **API status**: what compatibility promise does magmaan make for this
  function, type, header, wrapper, or result shape?
- **Statistical status**: what evidence supports the method being computed:
  lavaan parity, theory, simulation, diagnostic invariants, published work, or
  local exploratory use?

A method can be statistically important without having a stable software API.
Likewise, a small diagnostic primitive can have a stable API if its contract is
simple, useful, and unlikely to change. Published work, informal use, or
simulation value are evidence for a method, not automatic source-compatibility
promises.

One possible status vocabulary:

- **Core**: fixture-backed, lavaan-compatible behavior. These paths can be used
  in public examples, benchmarks, and compatibility claims.
- **Experimental**: public methods-development primitives with explicit
  diagnostics and tests, but unsettled calibration, naming, result shape,
  supported scope, or source-compatibility promise. These may be statistically
  useful, citable, and intentionally exposed, but should be described as
  experimental.
- **Compat**: projection, naming, and parity helpers whose purpose is matching
  lavaan's surface or oracle outputs, not defining magmaan's internal ontology.
- **Rejected**: combinations that fail explicitly because the statistical or
  semantic contract is not designed.
- **Undesigned**: plausible future work with no current contract.

The important rule is that status should not be only prose. It should be
visible at the boundary where users and contributors meet the project.

## Public Surface Categories

magmaan should probably expose two main public API layers over one
implementation, plus explicit status-scoped surfaces:

- **Friendly API**: ergonomic staged entry points for ordinary use, currently
  centered on `magmaan::api` and the exported R helpers. These compose
  lower-level primitives while keeping statistical choices explicit.
- **Compositional API**: public power-user primitives in domain namespaces such
  as `spec`, `model`, `data`, `estimate`, `inference`, `robust`, `measures`,
  and `optim`. This is the methods-development, simulation, and LLM-readable
  surface.
- **Compatibility API**: public surfaces under `compat::*`, especially
  `compat::lavaan`, whose purpose is matching external naming, projection, or
  oracle behavior rather than defining magmaan's internal ontology.
- **Experimental API**: public research primitives whose contract,
  calibration, naming, result shape, or supported scope may still change.
  Experimental APIs should stay in their domain namespace, for example
  `data::experimental` or `optim::experimental`, rather than moving all
  experiments to one top-level namespace.

Internal implementation details should live in uninstalled headers, `src/`, or
explicit `detail` / `internal` namespaces. They carry no compatibility promise.

In C++ terms, the practical public boundary is not only symbol visibility. It
is "installed public header plus documented support status." Namespace names
communicate intent, but the support matrix and API documentation carry the
actual compatibility claim. Unless magmaan later decides otherwise, C++ ABI
stability should not be promised; source-level behavior for documented public
surfaces is the realistic contract.

## Should Status Be Namespaces?

Maybe partly, but not as a top-level taxonomy that replaces domain namespaces.

Namespaces are attractive because they make status visible at the call site:

```cpp
magmaan::estimate::fit_ml(...)
magmaan::data::experimental::pairwise_joint_composite_objective(...)
magmaan::compat::lavaan::to_lavaan_partable(...)
```

This has real advantages:

- experimental APIs are harder to accidentally treat as stable;
- compatibility code is not confused with the internal model contract;
- future deprecation or promotion can be done by moving APIs deliberately;
- R and Python bindings can mirror the same grouping.

But namespaces alone are too blunt. Some modules contain both stable and
experimental entry points, and moving every experimental helper into a
top-level `lab::...` namespace could make the implementation noisier than the
distinction is worth.
For example, an ordinal data module may have one lavaan-compatible moment
builder and several robust experimental builders that share low-level kernels.
Likewise, a new optimizer should remain conceptually in `optim`; if its public
contract is unsettled, it can live in `optim::experimental` or an experimental
optimizer header rather than `lab::optim`.

A hybrid policy may work better:

- Use top-level namespaces for durable conceptual boundaries:
  `parse`, `spec`, `model`, `data`, `estimate`, `inference`, `robust`,
  `measures`, `optim`, `api`, and `compat::lavaan`.
- Put experimental public surfaces inside the relevant domain namespace, such
  as `data::experimental`, `estimate::experimental`, or
  `optim::experimental`.
- Optionally add a curated top-level `lab` facade later if it is useful as a
  research-workbench index, but do not make it the owner of every experiment.
- Keep private shared kernels in ordinary implementation files when they are
  not a public boundary.
- Require docs and tests to carry the final status claim; the namespace is a
  signal, not the whole contract.

Possible C++ shape:

```cpp
namespace magmaan::data {
// Supported lavaan-compatible ordinal statistics.
std::expected<OrdinalStats, Error> ordinal_stats_from_integer_data(...);
}

namespace magmaan::data::experimental {
// Experimental methods primitives; tested, but not lavaan compatibility claims.
std::expected<OrdinalStats, Error>
pairwise_ordinal_stats_h_weighted_from_integer_data(...);
}

namespace magmaan::compat::lavaan {
// Compatibility projections and lavaan-shaped views.
std::expected<LavaanParTable, Error> to_lavaan_partable(...);
}
```

Possible R shape:

```r
magmaan_core$data_ordinal_stats_from_raw(...)
magmaan_core$data_experimental_pairwise_ordinal_h_weighted(...)
magmaan_core$compat_lavaan_lavaanify(...)
```

This would keep the friendly namespace small, keep `magmaan_core` broad but
mostly compatibility-backed, and make explicitly experimental work reachable
without making it look like an ordinary sibling of the stable compatibility
path.

The model is similar to the broad pattern used by established C++ and
scientific libraries: documented public entry points receive compatibility
discipline; private or underscored/internal implementation details do not; and
experimental features require a visible opt-in or status marker. For magmaan,
the closest analogue is: installed public header plus support-matrix entry
defines public API; `detail` / `internal` / uninstalled headers are private;
`experimental` namespaces or headers mark public research APIs whose software
contract may still change.

## Stable vs Experimental

A public function, type, or header can be treated as stable when:

- its statistical and software contract is written down;
- input ordering, output scaling, group/block conventions, and missing-data
  behavior are clear;
- the name, signature, and result shape are considered settled;
- unsupported combinations fail explicitly;
- tests support the claim being made: lavaan fixtures for compatibility
  claims, or invariants, finite-difference checks, diagnostic fixtures,
  simulation checks, or independent references for non-lavaan behavior;
- it appears in the support matrix or API documentation;
- maintainers are willing to deprecate before breaking it.

A public API should remain experimental when any of these are still fluid:
contract, calibration, argument shape, result fields, supported model/data
slice, or intended downstream use.

Experimental does not mean untested or uncitable. It means no ordinary source
compatibility promise yet. A simulation paper should be able to cite magmaan
and name an experimental surface precisely, but readers should be able to see
that it was a versioned research primitive rather than a default supported SEM
workflow.

## Support Matrix

The project could benefit from a single support matrix that records the status
of model/estimator/data/post-fit combinations. This would reduce the burden on
roadmap prose and make scope decisions easier to audit.

The matrix could live in `docs/support.md` or as a generated table sourced from
YAML. It might have rows like:

| Model/data slice | Estimator | Feature | Status | Evidence |
|---|---|---|---|---|
| continuous complete-data CFA | ML | estimates, SE, chi-square | Core | lavaan fixtures |
| continuous raw missing data | FIML | MLR robust report | Core | lavaan fixtures |
| all-ordinal complete/listwise | DWLS/WLS | estimates, fit measures | Core | lavaan fixtures |
| all-ordinal pairwise observed missing | composite likelihood | objective diagnostics | Experimental | unit fixtures |
| mixed categorical | h-weighted polyserial | moment builder | Undesigned | design needed |
| inequality constraints | any | active-bound inference | Rejected | out of scope |

The support matrix should answer:

- Is this lavaan-compatible, experimental, rejected, or undesigned?
- Which tests or fixtures support the claim?
- Which public API exposes it?
- Which combinations fail explicitly?
- What would be required to promote it?

## Promotion Rules

Experimental work should be promotable, but promotion should require more than
passing tests.

An experimental feature could become core only when:

- the statistical contract is written down;
- unsupported combinations fail explicitly;
- lavaan parity exists where lavaan has matching behavior;
- non-lavaan behavior has its own oracle, simulation check, or documented
  diagnostic contract;
- the C++ API has a stable owner namespace;
- the R surface is thin and decomposable into visible primitives;
- benchmarks exist if performance is part of the reason for the feature.

This avoids a common failure mode: experimental methods code accumulates until
users cannot tell which parts are polished and which parts are research
scaffolding.

## Sprawl As Design Reconnaissance

Some of the current breadth is not accidental scope creep. It is design work.

If magmaan only supported clean complete-data ML, expected information,
standard SEs, and a small set of fit measures, the design could be much tidier.
It might also be falsely tidy. Robust Satorra-Bentler variants, observed versus
expected bread, biased versus unbiased Gamma, FIML robust traces, ordinal
NACOV, DLS weights, h-weighted moments, DPD comparators, and composite
likelihood prototypes all put different pressure on the same abstractions.
They reveal which boundaries are real and which only looked good because the
easy case was too simple.

This suggests a more generous rule:

> Sprawl is acceptable while it is producing architectural information. It
> becomes a problem when it makes support claims unclear.

The job is not to remove breadth too early. The job is to label what the
breadth is doing.

Useful labels might be:

- **Architecture probe**: exists to test whether a namespace, result type,
  matrix contract, or estimator interface generalizes.
- **Research surface**: an experimental but coherent methods API, useful in
  its own right.
- **Diagnostic kernel**: a small callable primitive used to inspect or validate
  one statistical subproblem.
- **Compatibility slice**: exists because lavaan exposes or implies a behavior
  that magmaan wants to match.
- **Promotion candidate**: experimental today, but intended to become core once
  the contract and evidence are complete.

This is different from treating everything outside the stable core as generic
`lab` material. A robust categorical moment builder that is testing future SEM
estimator boundaries, a diagnostic bivariate ordinal kernel, and a compatibility
helper for lavaan-shaped partables are not the same kind of thing. They may all
be non-core, but they serve different design purposes.

The key discipline is to keep exploratory breadth from becoming ambiguous
public scope. A design probe can be messy internally. A support claim cannot be.

## Scope Bias

When deciding between breadth and depth, prefer deep completed slices.

Good magmaan scope:

- explicit model/data construction;
- verifiable point estimation;
- inspectable post-fit inference;
- reproducible robust corrections;
- fixture-backed lavaan parity;
- performance claims gated by correctness.

Risky magmaan scope:

- broad lavaan tutorial parity as a goal by itself;
- end-user convenience that hides statistical choices;
- new syntax before the model ontology is settled;
- experimental estimators exposed as if they were ordinary compatibility paths;
- public performance claims before workload equivalence is documented.

## Bootstrap As Simulation Scope

Bootstrap support is intentionally not on the active lavaan tutorial parity
checklist. It may belong in magmaan later, but only if the project grows an
explicit simulation/resampling layer rather than adding one-off
`se = "bootstrap"` plumbing to the core inference namespace.

A reasonable future shape would be:

- keep the public C++/R/Python contract seed-based: bindings pass one integer
  seed, and C++ owns all random draws after that;
- make the low-level resample/refit loop consume a C++ RNG object by reference,
  so tests and future simulation drivers can provide a deterministic stream;
- keep the RNG type project-owned rather than exposing R, NumPy, Boost, or
  standard-library distribution details across bindings;
- resample raw rows within group/block, carrying FIML masks with the selected
  rows when missing-data bootstrap support is designed;
- treat Bollen-Stine as a C++ data transform plus the same resample/refit loop;
- evaluate `:=` defined parameters on each successful refit if bootstrap
  intervals are part of the accepted simulation surface;
- return the seed, RNG algorithm/version, requested draws, successful draws,
  failed-draw diagnostics, and bootstrap estimates so runs are auditable and
  replayable from any binding.

This is deliberately a future scope note, not a backlog item. If simulation
support becomes a first-class direction, bootstrap can be promoted as one
resampling primitive within that broader design.

## Composite Models

Composites deserve extra caution. The `<~` operator is not just another parser
feature; it changes what kind of object the model contains. Composites should
not be treated as ordinary reflective latent factors.

A conservative policy:

- do not promote composites until their place in `LatentStructure`,
  `LatentNames`, `Starts`, matrix representation, partable projection, and
  implied moments is designed;
- support only the smallest complete-data ML slice first, if support is
  accepted at all;
- keep ordinal, FIML, robust corrections, and LS composite support out of scope
  until the complete-data ML contract is stable;
- require explicit rejected errors for unsupported composite combinations.

## README And Docs Drift

The README should describe the current public contract rather than an older
snapshot. In particular, it should not say a feature is out of scope if it is
already implemented for a supported slice.

A possible docs split:

- `README.md`: short identity, current supported headline slices, build loop,
  and strong caveats.
- `docs/roadmap.md`: current state and architecture contracts.
- `docs/todo.md`: accepted remaining work only.
- `docs/support.md`: status matrix for supported/lab/rejected combinations.
- `docs/ideas.md`: non-binding sketches like this file.

## One-Sentence Philosophy

If forced into one sentence:

> Make magmaan less ambitious as a package, and more ambitious as an engine.

That means fewer broad promises, more completed slices, and sharper boundaries
between compatibility, research, and future ideas.
