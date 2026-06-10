# magmaan Maintainer Notes

This directory is for maintainer-facing project knowledge, not polished user
manuals. Public package-facing documentation can grow separately once the API
is stable enough to describe without rewriting the story every week.

## Layout

- `architecture/` - current implementation state and design contracts.
- `backlog/` - accepted remaining work and known failures:
  - `todo.md` - the active SEM/parser/estimation backlog (open work only).
  - `simulation.md` - the `magmaan::sim` work queue and decision log.
  - `speculative.md` - may-never-build ideas, each with the cheaper alternative
    and the build-if trigger.
  - `newsom-corpus-failures.md` - open optimizer / evaluator-accuracy cases
    surfaced by the Newsom corpus.
- `grammar/` - normative parser grammar and lexer notes.
- `validation/` - parity, benchmark, and diagnostic validation plans.
- `design/` - proposals, audits, and non-binding sketches.
- `research/` - tracked research notes and simulation scripts.
- `reference/` - policies for local-only resources and external source mirrors.
- `assets/` - small checked-in assets used by repository docs.

Local reference PDFs, package tarballs, downloaded data, and cloned upstream
source trees are intentionally not tracked here. See
[`reference/external_resources.md`](reference/external_resources.md).
