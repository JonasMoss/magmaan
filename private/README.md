# private/ — work that is not magmaan

magmaan's repository holds magmaan: the library, its bindings, tests,
fixtures, experiments, benchmarks and maintainer docs. Everything else that
grows up around it lives here, one project per folder, each folder its **own
independent git repository** (the same pattern as `papers/`). This README is the
only tracked file; `private/*` is gitignored.

Typical residents:

- talks and slide decks;
- collaborator handoffs, proposals, and correspondence drafts;
- funding and compute-allocation applications;
- evaluations of other people's papers;
- anything else that is yours but not the magmaan product.

Rules:

- **Never referenced from magmaan.** No tracked file may source, include, or
  read `private/**` (enforced for code by `tests/tools/check_layering.sh`).
  A private project may use magmaan freely.
- **Never tracked by magmaan.** `tests/tools/check_tracked_files.sh` fails if
  anything under `private/` other than this README is staged.
- **Its own remote, or none.** Push a project to its own private remote if it
  needs a backup or collaborators; it never travels with magmaan.

When a project is split out of magmaan, extract it with its history
(`git filter-repo --subdirectory-filter <path>` on a scratch clone) and move the
resulting `.git` into `private/<name>/`.
