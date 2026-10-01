# Experiments

Each experiment answers one focused question. A large grid is fine when every
cell serves that question. Extend an existing study when the question matches;
a genuinely different question gets a new folder after checking scope and overlap.
Experiments are advisory tooling, not component parity gates.

Use `magmaan-experiment` when creating, extending, running or reporting a study.
Its [runner reference](../.agents/skills/magmaan-experiment/references/runners.md)
and [report reference](../.agents/skills/magmaan-experiment/references/reports.md)
hold the detailed procedures; read the relevant one before that work. Read any
study-local `AGENTS.md` and its report/design before changing an existing study.

## Dependencies and ownership

Each numbered study is an independent leaf, including studies in the same
category or activity directory. No paper/sibling imports or result reads.
Allowed inputs are core, bindings, `_support`, benchmarks and the optional
textbook corpus. `_support` owns path/seed/metadata/I/O/formatting mechanics,
with no SEM logic or study-specific statistical decisions. Shared statistical
primitives go into core or compiled bindings. `just check-layering` checks code;
review report inputs manually. Reports read only their own `results/`.

## Where new work belongs

Choose the home by the decision or question, not the statistical topic or kind
label. Read the [index](README.md), [scope](../project/scope.md) and relevant
[active backlog](../project/backlog/todo.md) or
[trigger register](../project/backlog/speculative.md) before creating a leaf.

| Primary purpose | Home |
|-----------------|------|
| Select or change an ordinary-user or implementation default using prespecified criteria and held-out evidence | `experiments/decisions/NN-slug/` |
| Demonstrate an implemented capability, oracle agreement or measured performance | `experiments/showcases/NN-slug/` |
| Reproduce a published result or reconstruct a reference method | `experiments/replications/NN-author-year-topic/` |
| Resolve statistical behavior, an estimand or inferential validity | `experiments/research/active/NN-slug/` |
| Resolve an implementation choice, numerical failure, correctness mechanism or cost | `experiments/engineering/active/NN-slug/` |

Before making a new folder, search current and retained studies for the same
question. A new run belongs under the existing leaf's `results/<run-id>/`;
another method or population serving that question can be a lane in the same
leaf. Keep separate targets, designs, seeds, failures and metadata. A lane is
part of its parent leaf, not a new numbered experiment. One parent runner and
report select and explain the lanes; do not create sibling studies just to
share code or copy an existing study's scaffolding and narrative wholesale.

Start research/engineering work in `active/` only when it has a concrete next
check or an ongoing paper-supporting pipeline. Retained prototypes or evidence
without a queued run belong in `banked/`, with a named reopening trigger.
Completed answers belong in `evidence/`. Move between activities as the work
changes; no fresh number or duplicate result tree is needed. A speculative idea
without a run or reusable evidence belongs in the trigger register first, with
an available cheaper alternative and a build-if condition.

Component regression/parity gates belong in the existing C++/R tests and fixture
workflows. Shared benchmark machinery belongs in `benchmarks/`; a focused
comparison may consume it from an experiment. Studies serving only an independent
manuscript belong in that paper's `code/studies/` under
[`papers/<slug>/`](../papers/AGENTS.md). Non-magmaan notes and research belong in
[`private/<slug>/`](../private/README.md). Preserve existing paper-supporting
experiments; these placement rules do not relocate protected pipelines.

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
reopening trigger. Research uses the same activity directories: `active/` for
concrete remaining research checks and ongoing paper pipelines, `banked/` for
ideas with named reopening triggers, and `evidence/` for completed reference
answers and frozen results. Preserve different estimands when merging arms.
Research can remain exploratory; do not impose the engineering
decision format on it. Review inherited lifecycle labels before treating them
as current. Once a decision settles, preserve its contract in tests/project docs
before archiving; delete a redundant probe only after naming its maintained
replacement checks and recording the deletion in the index.

Number studies from `01` independently per category; research and engineering
each use one sequence across their three activities. Allocate a new number
above the highest already assigned in that category, checking current folders
and the index's cleanup record. Do not fill holes left by moves, merges or
deletions, or reuse a retired category-qualified number. Activity moves keep
the number; deliberate renumbering updates the index and paths. Cite the
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

Before the first run, register the leaf in the index with its question, kind,
lifecycle and concrete next check or reopening trigger. Add the required ignore
files before generating outputs. Further runs use fresh output directories;
preserve frozen evidence and keep source-fingerprint resume checks strict.

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
