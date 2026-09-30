# Experiments

Each experiment answers one research or engineering question. A large grid is
fine when every cell serves that question; a new question gets a new folder.
Experiments are advisory tooling, not component parity gates.

Use `magmaan-experiment` when creating, extending, running or reporting a study.
Its [runner reference](../.agents/skills/magmaan-experiment/references/runners.md)
and [report reference](../.agents/skills/magmaan-experiment/references/reports.md)
hold the detailed procedures; read the relevant one before that work. Read any
study-local `AGENTS.md` and its report/design before changing an existing study.

## Dependencies and ownership

Each numbered study is an independent leaf, including studies in the same
category or engineering activity. No paper/sibling imports or result reads.
Allowed inputs are core, bindings, `_support`, benchmarks and the optional
textbook corpus. `_support` owns path/seed/metadata/I/O/formatting mechanics,
with no SEM logic or study-specific statistical decisions. Shared statistical
primitives go into core or compiled bindings. `just check-layering` checks code;
review report inputs manually. Reports read only their own `results/`.

## Index and lifecycle

[README.md](README.md) indexes every study: category-local number, slug, kind,
lifecycle and question. Update it when adding or changing a study's status.
Categories are `decisions/`, `showcases/`, `replications/`, `research/` and
`engineering/`. Navigation categories do not create dependency tiers.

Kind is `parity`, `replication`, `paper-sim`, `benchmark` or `probe`.
Lifecycle is `active` (rerun/extended), `banked` (no queued run; named consumer
or reopening trigger), `complete` (retained reference/evidence), or `archived`
(inert). A complete investigation may remain load-bearing paper evidence.
Preserve paper-supporting pipelines; classification does not authorize deletion
or an archive move.

Engineering uses `active/` (unresolved implementation choice and next check),
`banked/` (explicit reopening trigger), and `evidence/` (completed paper evidence).
Record current default, alternatives, acceptance criteria, latest evidence and
reopening trigger. Research can remain exploratory; do not impose the engineering
decision format on it. Review inherited lifecycle labels before treating them
as current. Once a decision settles, preserve its contract in tests/project docs
before archiving; delete a redundant probe only after naming its maintained
replacement checks and recording the deletion in the index.

Number live studies from `01` independently per category; engineering uses one
sequence across all three activities. Use the next free number. Activity moves
keep the number; deliberate renumbering updates the index and paths. Cite the
category-qualified slug, since numbers are navigation, not stable IDs.
Archives use `_archive/<slug>/` without numbers. Preserve sources/results and
update reproduction paths; resolve slug collisions descriptively. Frozen
metadata and remote app/volume IDs may retain legacy numbers; do not rewrite
evidence or rename cloud storage as navigation cleanup.

## Decisions set library defaults

Default-setting studies live in `decisions/`, one lane per estimation route.
Exploratory studies that inform them stay in engineering. Before a decision run:

- Commit `criteria/<lane>.md`: decision, configurations, problems, judge,
  rules and the outcome they imply. Preserve it after the run; later amendments
  are dated and explicitly marked as written after seeing results.
- Use the actual library default without options as baseline. Judge success
  with the library's `fit$converged`, including other engines at their estimates;
  no study-specific acceptance tolerance.
- Use held-out problems. Development populations are labeled controls,
  reported but never gating. Report rates per family, never pooled, and list
  every loss against baseline.
- Use a fresh seed base, distinct from smoke runs and previous decision runs.
  A draw that prompted a fix is development data; confirmation needs fresh draws.
- Commit per-run summary CSVs and `metadata.csv` under
  `results/<lane>/<run-id>/`; raw per-fit rows stay ignored.

The report opens with the default register: covered choices, current values,
settled/provisional/open status and evidence. This is the explicit exception to
the ordinary Question/Short answer opening; keep the answer immediately readable.

## Required artifacts

A live study contains `report.qmd`, `run_experiment.R`, `.gitignore` and
`results/.gitignore`. Optional `R/` holds local helpers; `scripts/` holds secondary
runners for the same question; ignored `resources/` holds local external inputs.
Literature replications use `NN-author-year[-topic]` slugs.

The report is the sole reader-facing narrative. No per-study README; merge its
useful content into the report before removing it. The collection index is the
exception. Runner `--help` is the command reference. Generated Quarto output and
raw results stay ignored. Tiny frozen artifacts need a documented reason; a
GitHub-facing study may track `report.md` rendered from the QMD with GFM.

Runners do expensive work, expose help and a cheap smoke path, use deterministic
seeds, record command/versions/design metadata, save failures, and print progress
and output paths. Estimate cost with a pilot for runs longer than a few minutes.
Reports read results and fail helpfully when absent; they never fit, simulate,
install, download or regenerate fixtures while rendering.

Open ordinary reports with Question and Short answer, then Evidence, Caveats and
Reproduce. Keep the first screen complete and honest. Use concise summary
tables/figures, with full grids and precision in CSV. Explain scientific
conventions (including information, sandwich or reference-distribution choices)
when needed to interpret the result; detailed methods may follow the short
opening. Keep code-location and implementation bookkeeping in code/metadata.
Separate statistical conclusions from engineering diagnostics and report cost,
uncertainty, failures and scope limits. Update roadmap/backlog when findings
change implementation state, validation expectations or remaining work.
