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

One possible status vocabulary:

- **Core**: fixture-backed, lavaan-compatible behavior. These paths can be used
  in public examples, benchmarks, and compatibility claims.
- **Lab**: methods-development primitives with explicit diagnostics and tests,
  but no broad lavaan-compatibility promise. These may be statistically useful
  and intentionally exposed, but should be described as experimental.
- **Compat**: projection, naming, and parity helpers whose purpose is matching
  lavaan's surface or oracle outputs, not defining magmaan's internal ontology.
- **Rejected**: combinations that fail explicitly because the statistical or
  semantic contract is not designed.
- **Undesigned**: plausible future work with no current contract.

The important rule is that status should not be only prose. It should be
visible at the boundary where users and contributors meet the project.

## Should Status Be Namespaces?

Maybe partly, but probably not only.

Namespaces are attractive because they make status visible at the call site:

```cpp
magmaan::estimate::fit_ml(...)
magmaan::lab::ordinal::pairwise_joint_composite_objective(...)
magmaan::compat::lavaan::to_lavaan_partable(...)
```

This has real advantages:

- experimental APIs are harder to accidentally treat as stable;
- compatibility code is not confused with the internal model contract;
- future deprecation or promotion can be done by moving APIs deliberately;
- R and Python bindings can mirror the same grouping.

But namespaces alone are too blunt. Some modules contain both stable and
experimental entry points, and moving every experimental helper under
`lab::...` could make the implementation noisier than the distinction is worth.
For example, an ordinal data module may have one lavaan-compatible moment
builder and several robust experimental builders that share low-level kernels.

A hybrid policy may work better:

- Use top-level namespaces for durable conceptual boundaries:
  `parse`, `spec`, `model`, `data`, `estimate`, `inference`, `robust`,
  `measures`, `optim`, `api`, and `compat::lavaan`.
- Add `lab` namespaces for public experimental surfaces that users may call
  directly but should not mistake for supported compatibility behavior.
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

namespace magmaan::lab::data {
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
magmaan_lab$data_pairwise_ordinal_h_weighted(...)
magmaan_core$compat_lavaan_lavaanify(...)
```

This would keep the friendly namespace small, keep `magmaan_core` broad but
mostly compatibility-backed, and move explicitly experimental work to a place
where methods developers can still reach it.

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
| all-ordinal pairwise observed missing | composite likelihood | objective diagnostics | Lab | unit fixtures |
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

A lab feature could become core only when:

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
