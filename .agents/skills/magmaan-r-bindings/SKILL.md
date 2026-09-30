---
name: magmaan-r-bindings
description: Change magmaanlab Rcpp bindings, R wrappers, generated exports or vendored C++ packaging, and coordinate adapters used by the pure-R magmaan package. Use for binding/API/ABI and portable-install work in magmaan, not unrelated R analysis or routine experiment scripts.
---

# magmaan R bindings

Read `r-package/AGENTS.md`; also read `r-magmaan/AGENTS.md` when the ordinary-user
surface changes. Identify the entry point and owning layer before editing:
canonical numerics in `cpp/`, hand-written glue at `r-package/src/` top level,
thin R wrappers in `r-package/R/`, user-facing composition in `r-magmaan/R/`.
Inspect current signatures and export machinery; old examples are not authority.

## Change the correct layer

- Keep C++ failures/unsupported cases visible at the R boundary. Validate
  R-shaped inputs and preserve names/groups without introducing duplicate SEM
  calculations. Ordinary-user inference policy stays in C++.
- Partables are model projections. Fix row/name/mean mismatches in the triple,
  projection or reconstruction, rather than formatting them away.
- `magmaanlab::fit_model()` estimates only, with explicit post-fit inference.
  `magmaan()` composes the ordinary-user policy. Do not export the same name
  from both packages with different meanings.
- Edit canonical C++ and run `just vendor` when needed. Never hand-edit
  `r-package/src/{core,magmaan,third_party}/`. When annotated glue signatures
  change, inspect the package's existing Rcpp export-generation recipe and
  regenerate both R/C++ exports with that mechanism. Do not assume every
  adapter is automatically exported or add an R alias for missing numerics.

## Build and check

Read [references/builds.md](references/builds.md) for fast versus portable
builds, export generation and ABI diagnosis. Select verification for the change:
focused wrapper/boundary tests for an adapter, relevant C++ numerical gates for
core changes, ordinary-package tests for policy/inspection behavior, and the
portable install for changes to vendor/Makevars/dependency resolution. Preserve
unsupported semantics and test meaningful edge cases rather than just export
presence.

## Deliver

Provide the changed entry point and resulting behavior, synchronized generated
files when applicable, and the checks/build mode actually used. Run structural
guards for new files; review generated diffs for unrelated changes. Update the
roadmap/backlog for changed contracts or concrete adapter gaps. Commit explicit
completed paths while preserving unrelated work.
