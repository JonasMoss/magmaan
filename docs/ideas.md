# Non-Binding Ideas

This note is a scratchpad for philosophical and scope changes. Most of it is
not the roadmap, not the backlog, and not a commitment; concrete accepted work
belongs in [roadmap.md](roadmap.md) or [todo.md](todo.md).

Two parts are further along. The API-tier sections reflect a May 2026 design
pass that reached real decisions — most firmly the `frontier` tier name and the
core/frontier split. The **User-Facing API** section is the opposite: it is
explicitly *unsettled*, and is written down only so the reasoning is not lost
when that thread resumes.

## Public Identity

The current direction is strong: magmaan should remain a C++ SEM engine and
methods-development workbench, not an end-user lavaan replacement.

A sharper public framing could be:

> magmaan is a verifiable SEM engine and research workbench for
> lavaan-compatible linear SEM slices, plus explicit frontier methods
> primitives.

That wording preserves the useful pieces:

- lavaan remains the oracle where compatibility is claimed;
- methods developers can inspect and compose the primitive graph;
- the package does not promise lavaan-style one-shot ergonomics;
- frontier methods work can live in the repo without pretending to be ordinary
  supported SEM functionality.

## API Status: Two Axes

The recurring question is how to distinguish supported work from
methods-development work so the distinction is visible in code, docs, tests,
and bindings. Two questions are easy to conflate and must be kept apart:

- **API status** — the source-compatibility promise magmaan makes for a
  function, type, or header.
- **Statistical status** — the evidence behind the method: lavaan parity, an
  independent oracle, simulation, diagnostic invariants, published work, or
  local exploratory use.

A method can be statistically strong with an unsettled software API, and a
small primitive can have a settled API while being statistically modest. The
key consequence: **`core` is an API-status word, not a statistical one.** A
faithfully ported, paper-validated estimator unrelated to lavaan can still be
`core` if its software contract is settled and it has its own committed oracle.
`core` does not mean "lavaan-compatible."

Status vocabulary:

- **Core** — settled software contract: stable name, signature, and result
  shape; unsupported combinations fail explicitly; documented; the maintainer
  will deprecate before breaking it. Evidence may be lavaan parity *or* an
  independent oracle.
- **Frontier** — the methods-development tier: public, tested, citable research
  primitives whose API carries no deprecation-cycle promise. Not a waiting
  room — a permanent destination (see below).
- **Compat** — projection and naming helpers under `compat::lavaan` whose job
  is matching lavaan's surface, not defining magmaan's ontology.
- **Rejected** — combinations that fail explicitly because the contract is not
  designed.
- **Undesigned** — plausible future work with no current contract.

Status must be visible where users and contributors meet the project, not only
in prose.

## Public Surface Layers

Over one implementation, magmaan exposes:

- **Friendly API** — ergonomic entry points for ordinary use (`magmaan::api`
  and the exported bindings), composing lower-level primitives while keeping
  statistical choices explicit. Its concrete shape is unsettled — see
  *User-Facing API* below.
- **Compositional API** — public power-user primitives in domain namespaces
  (`spec`, `model`, `data`, `estimate`, `inference`, `robust`, `measures`,
  `optim`): free functions, no hidden state — the methods-development,
  simulation, and LLM-readable surface.
- **Compatibility API** — `compat::lavaan`: lavaan-shaped projection and oracle
  matching.
- **Frontier API** — research primitives whose contract may still change.

Internal details live in uninstalled headers, `src/`, or `detail` namespaces
and carry no promise. magmaan does not promise C++ ABI stability; source-level
behavior for documented public surfaces is the realistic contract — the Abseil
stance of API-yes, ABI-no.

## The `frontier` Tier

Decided: the non-core methods tier is named **`frontier`** — not "experimental"
and not "lab" — and it **nests per domain** (`estimate::frontier`,
`optim::frontier`, `robust::frontier`, …) rather than collecting every
experiment in one top-level namespace.

Per-domain nesting runs *against* the nearest precedents — Eigen's top-level
`unsupported/`, scikit-learn's top-level `sklearn.experimental`. It is still
right for magmaan for a specific reason: those projects put experiments at the
top because the experiments were usually whole new *domains* with no existing
home. magmaan's frontier methods almost always do have a home domain — a
strange optimizer is still an optimizer; a robust estimator is still
`estimate`. When the frontier thing is a citizen of an existing namespace,
nesting it there beats exiling it.

Naming rationale: `core` / `frontier` is a matched metaphor — both territory
words — where `core` / `experimental` mixes a place with a process. `frontier`
also *invites* use where "experimental" and Eigen's "unsupported" repel, and a
methods repository wants users in. The cost of deviating from precedent here is
near zero: the *mechanism* — separate namespace, no deprecation promise,
catalog-as-contract — follows precedent; only the label is local.

Friction model: in statistics, weak stability expectations are normal —
convergence failures and contested asymptotics are routine. A hard opt-in gate
(Eigen's separate include path, scikit-learn's enabling import, Rust's feature
flag) is therefore overkill. The `frontier` namespace label, plus the evidence
column of the methods catalog, is proportionate friction.

## Frontier as a Methods Repository

The intent is for `frontier` to grow *large* — a common repository for the many
published SEM and psychometrika methods that will never be lavaan-parity
"core", but are worth a verified, composable implementation. Frontier is a
destination, not a transit lounge.

This is sustainable only under one discipline: **every frontier method must be
a model of an existing extension concept** — `Discrepancy`, `Optimizer`,
`StandardErrorMethod`, `FitIndex`, a moment/data builder — not a parallel
re-implementation of the fit machinery. A method that cannot be expressed
against the concept set is not a failure; it is the most valuable signal the
frontier produces, because it shows the concept set has a missing slot. The
failure mode is a silo'd method that forks the engine.

Sprawl is acceptable while it produces architectural information; it becomes a
problem when it makes support claims unclear. So the job is not to remove
breadth — it is to label what a given piece of breadth is doing:

- **Architecture probe** — tests whether a namespace, result type, or estimator
  interface generalizes.
- **Research surface** — a coherent methods API, useful in its own right.
- **Diagnostic kernel** — a small primitive that inspects one subproblem.
- **Compatibility slice** — exists because lavaan implies a behavior to match.
- **Promotion candidate** — intended for `core` once contract and evidence are
  complete.

Entry bar for a method to land in `<domain>::frontier`. Because the maintainer
is realistically the main contributor for the foreseeable future, this is a
self-imposed checklist, not a review gate:

- a citation;
- at least one oracle test — the source paper's numbers, a simulation, or
  finite-difference invariants;
- explicit `Rejected` errors for the combinations it does not support;
- a row in the methods catalog.

## Two Maturation Arrows

"Promotion" conflates two orthogonal moves; separate them.

- **Arrow 1 — reach.** A method lands compositional-only in
  `<domain>::frontier`. Later, a maintainer writes a thin friendly wrapper
  (`api::frontier::…`) and a binding. This is about *ergonomics*: routine,
  encouraged, cheap. The method stays `frontier` throughout.
- **Arrow 2 — contract.** `frontier` → `core`. This is about *evidence and
  promise*: a committed oracle, a stable owner namespace, and a willingness to
  deprecate before breaking. Rare. Most methods never take it — and that is the
  model working, not failing.

A surface can be treated as `core` when its statistical and software contract
is written down; input ordering, output scaling, group/block conventions, and
missing-data behavior are clear; name, signature, and result shape are settled;
unsupported combinations fail explicitly; tests support the specific claim
(lavaan fixtures for compatibility claims, or invariants / finite-difference /
simulation / an independent oracle for non-lavaan behavior); it appears in the
catalog; and the maintainer will deprecate before breaking it.

`frontier` does not mean untested or uncitable. It means no deprecation-cycle
promise yet. A paper should be able to cite magmaan and name a frontier surface
precisely.

## Methods Catalog

The project should keep one catalog recording the status of
model/estimator/data/post-fit combinations — replacing roadmap prose and making
scope auditable. At frontier scale it becomes the real discovery tool; the
namespace is only a coarse signpost.

One row per method or slice: model/data slice, estimator, feature, **API
status** (core/frontier/compat/rejected/undesigned), **statistical evidence**
(lavaan fixtures / validated vs source paper / simulation / invariants /
exploratory), the public entry point, and the header.

It should answer: is this core, frontier, rejected, or undesigned? Which tests
support the claim? Which public API exposes it? Which combinations fail
explicitly? What would promotion (Arrow 2) require?

It should be **generated and checked** from per-method metadata, not
hand-maintained prose — SciPy's lesson is that the explicit list *is* the
contract, and Stan's is that an ungenerated support list drifts. Likely home:
`docs/support.md`, or a table generated from YAML.

## User-Facing API (Unsettled)

**This section records an in-progress discussion. None of it is decided.** It
is written down so the reasoning survives until the thread resumes.

Motivation: avoid lavaan's monster fitted object — a god-object of precomputed
slots plus a stringly-typed `lavInspect(fit, "…")`. That object is bad for
legibility and for methods research, even though it is convenient for end users
with good LSP tooling. magmaan's user-facing layer should keep every
statistical choice — which vcov, which SE, which test — *explicit*; a user who
wants those choices made for them should use lavaan.

Current thinking, all tentative:

- **Two shapes for two layers.** The compositional layer stays free functions
  (composable, no hidden state, legible). The friendly layer is *methods on
  small objects* — methods are LSP-discoverable (type `fit.` and the surface
  appears); free functions are not. Methods are not the monster: lavaan's
  monster is a *storage* problem, not a *methods* problem.
- **`Fit` is small and immutable** — it holds only {model, data, estimates}.
  Post-fit "poke" methods (`fit.standard_errors(spec)`) compute on demand,
  return `Result<T>`, and fail fast. This is the methods-research surface.
- **`Report` is the chain accumulator** — entered explicitly (`fit.report()`);
  its methods store-and-return the report; `.summary()` renders. It accumulates
  an append-only *list* per category, not a single slot. Each slot holds
  `Result<T>`, so the chain is `std::expected`-free and `.summary()` can render
  a partial report with per-section errors. `.summary()` itself computes
  nothing — it is a pure renderer of explicitly chosen results, so its content
  is a pure function of the chain. This rehabilitates, rather than keeps, the
  current `api::Analysis`.
- **Results carry their own config.** A result embeds the full configuration
  that produced it; no API path yields a bare number. "Satorra-Bentler" is not
  a function but a correction *form* parameterized by a cube (base × Γ × U,
  plus the form axis); friendly presets are named points in that cube, and
  unnamed cells display by coordinates. The summary's "ingredient grouping" is
  then emergent: the renderer de-duplicates results by their embedded config.
- **No implicit cache.** Sub-ingredients with choice axes (Γ, U) are explicit
  *values* — computed once, named, passed. Choice-free intrinsics (the Jacobian
  Δ at θ̂, the implied Σ) *may* be memoized on `Fit` if benchmarks demand it,
  but never the choice-bearing quantities.
- **Bindings mirror the C++ shape** — R via R6 (method syntax,
  autocomplete-discoverable, in the spirit of mlr3), not S4 generics; a future
  Python binding as a plain class.

Open questions: the whole shape is unconfirmed; the parameter-estimates table
wants SE/z/p columns, and multiple SE computations make it uncomfortably wide;
whether the friendly facade needs a generic templated escape hatch for frontier
methods or whether `api::frontier::` free functions suffice; and the fate of
the pre-fit `analyze()` entry point.

Precedents consulted: Stan (the `stan::services` facade over `stan::math` and
`stan::mcmc`); statsmodels (a `Results` object, post-estimation as
explicit-argument methods); Eigen `unsupported/`; the tidyverse `lifecycle`
stages; Abseil (API-yes / ABI-no); Rust stability attributes; the SciPy
public/private convention; scikit-learn `experimental`; mlr3 (R6).

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
- frontier estimators exposed as if they were ordinary compatibility paths;
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

## Open Refactor Questions

Minor and unresolved, noted so they are not lost. The next working pass is a
small refactor alongside the core/frontier split.

- `estimate/` may hold things that belong elsewhere — worth an audit.
- `Starts` and the start-value producers could become a sub-namespace.
- The role of `model/` is unclear and worth pinning down.
- Whether an `nt` namespace should regroup FIML and complete-data covariance
  estimation — probably not: complete-data normal theory is as closely related
  to the GMM core as it is to FIML, so the current split may already be right.

## README And Docs Drift

The README should describe the current public contract, not an older snapshot;
in particular it should not call a feature out of scope once it is implemented
for a supported slice.

A possible docs split:

- `README.md` — short identity, current supported headline slices, build loop,
  and caveats.
- `docs/roadmap.md` — current state and architecture contracts.
- `docs/todo.md` — accepted remaining work only.
- `docs/support.md` — the methods catalog (core/frontier/rejected combinations
  with evidence).
- `docs/ideas.md` — non-binding sketches like this file.

## One-Sentence Philosophy

If forced into one sentence:

> Make magmaan less ambitious as a package, and more ambitious as an engine.

That means fewer broad promises, more completed slices, and sharper boundaries
between compatibility, research, and future ideas.
