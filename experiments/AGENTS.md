# Experiments

Repo-local rules for magmaan experiments. These rules apply under
`experiments/` and are meant to make "write an experiment about ..." prompts
produce one focused, reproducible folder instead of a pile of notes.

## Purpose

Each experiment folder answers one research or engineering question. It may
have a large design grid, many backends, or many simulation cells, but those
pieces must serve the same question. If the question changes, start a new
numbered folder.

Experiments are advisory research tooling. They can motivate core work, docs,
or fixtures, but they are not themselves parity gates.

**Experiments are endpoints** (see "Dependency layering" in the root
`AGENTS.md`). Nothing outside an experiment references it, and an experiment
references no paper and no other experiment. The only shared experiment sibling
is `experiments/_support` (path / metadata / IO helpers, no SEM logic); an
experiment may also consume the core library, `r-package`, `benchmarks`, and the
optional textbook corpus. To share statistical code, lift it into core, the
`r-package`, or `experiments/_support` - never `source()`/`sys.source()` another
experiment or a paper. Enforced by `cpp/tests/tools/check_layering.sh`
(`just check-layering`).

## Index And Lifecycle

`experiments/README.md` is the tracked index of every experiment: category-local number (if live), slug,
kind, lifecycle, and the one-line question. Adding an experiment means adding a
row. This collection-level index is the one allowed exception to the "no README"
rule below, which still holds for individual `NN-` folders.

Store experiments under the matching purpose folder: `decisions/`, `showcases/`,
`replications/` (including reference studies), `research/`, or `engineering/`.
`_archive/` holds retired studies and `_support/` remains the shared harness.
Engineering adds one activity directory: `engineering/active/`,
`engineering/banked/`, or `engineering/evidence/`. Each individual experiment
is the dependency leaf, not its category or activity directory; two studies
in the same activity may not reference each other.

Group the index by primary purpose: **showcase**, **replication/reference study**,
**research**, or **engineering check**. This is navigation metadata, independent
of lifecycle and of the existing kind tags. A published design can support either
a showcase or research; classify by the report's main question. Keep uncertain
cases explicit. Classification alone does not authorize an archive move.

Also retain these two axes in the index:

- **Kind** - `parity` (audits magmaan against lavaan), `replication` (reproduces a
  published author-year simulation), `paper-sim` (a pipeline a paper's results
  depend on; never delete), `benchmark` (speed or statistical efficiency), `probe`
  (a one-off engineering diagnostic).
- **Lifecycle** - `active` (still rerun or extended), `banked` (no run queued;
  a named consumer or reopening trigger is required), `complete` (finished,
  retained for reference or paper evidence), `archived` (inert). Paper evidence
  can remain load-bearing while its investigation is complete.

## Decisions set library defaults

A `decisions/` study holds the evidence and the reasoning for a group of
library defaults, one **lane** per estimation route (for example `ml-gls`,
`psd-ml`, later `ordinal`). It is held to a stricter standard than the other
categories, because its results change what every user gets:

- **Criteria first.** Before a lane's decision run, `criteria/<lane>.md` is
  committed: the decision, the configurations, the problems, the judge, the
  rules and the outcome they imply. It is not edited after the run; a later
  amendment gets a dated section marked as written after seeing results.
- **The baseline is the library default**, called without options, so the
  study tests what users get.
- **One judge.** Success is the library verdict (`fit$converged`), never a
  study-specific tolerance. Other engines are judged by the same verdict at
  their estimates.
- **Held-out problems.** Test problems were not used to develop any
  candidate; populations that were are labelled controls, reported and never
  gating. Rates are reported per family, never pooled across families, and
  every loss against the baseline is listed.
- **Fresh draws.** Each decision run uses a seed base not used before; smoke
  runs use a different base. Draws that prompted a fix become development
  data, so the confirming run after the fix uses a new base.
- **Tracked evidence.** The per-run summary CSVs and `metadata.csv` under
  `results/<lane>/<run-id>/` are committed (an exception to the ignored-results
  rule below); raw per-fit rows stay local.
- **The report opens with the register**: every default the study covers, its
  current value, its status (settled, provisional, open) and the evidence.

Exploratory checks that inform a decision stay in `engineering/`.

## Engineering is a live decision workspace

Engineering is organized by activity, with stable study numbers across the
three directories:

- `active/`: an unresolved implementation choice and a concrete next check.
- `banked/`: a retained idea with an explicit reopening trigger, no queued run.
- `evidence/`: completed evidence required by a paper; preserve its design,
  results and reproduction path. Do not merge it into a development sample.

The collection index records the current question, next check or reopening
trigger. Moving a study between activities keeps its number.

Engineering studies should support current implementation choices: default
estimators, optimizers, starts, scaling, convergence rules, or performance
tradeoffs. Keep each study focused on one decision and record the current default,
alternatives, acceptance criteria, latest evidence, and a reopening trigger.
A recurring benchmark belongs here when it can change an active decision.

When the decision is settled, preserve its contract in maintained tests and
project docs, then archive the study, or retain paper-supporting work in
`evidence/`. Delete a redundant probe only after naming the maintained tests
that cover its findings; record that deletion in the collection index. Do not infer that an inherited `active`
label already meets this standard; review the inventory during the next cleanup.
Research may remain exploratory, with evolving questions and trial scripts;
do not force it into the engineering decision format. Keep its evidence and
reproduction paths, and never discard paper-supporting pipelines as cleanup.

## Numbering and archives

Number live studies from `01` independently within each purpose category.
Use the next free number in that category when adding a study; engineering
uses one sequence across its activity directories. Numbers are local
navigation, not stable identifiers: cite a category-qualified slug, and update
paths and the index whenever a deliberate renumbering is requested.

Archived studies live at `experiments/_archive/<slug>/`, without numbers.
Keep their sources and existing results; update reproduction paths and references
when archiving. If a slug would collide, choose a descriptive unique slug.
Frozen result metadata and external cloud app/volume identifiers may retain legacy
numbers; do not rewrite evidence or rename remote storage as part of navigation.

## Directory Shape

Use a category-local two-digit prefix and a short kebab-case slug. Literature
replications use `author-year[-topic]`, for example `02-rhemtulla-2012`.

```text
experiments/
  <category>/NN-topic-slug/              # engineering: engineering/<activity>/NN-topic-slug/
    report.qmd
    run_experiment.R
    .gitignore
    results/
      .gitignore
    R/              # optional experiment-local helpers
    scripts/        # optional secondary runners for the same question
    resources/      # optional local-only external inputs; always ignored
```

Required files:

- `report.qmd`: the standalone narrative artifact.
- `run_experiment.R`: the main compute entry point.
- `.gitignore`: ignore generated Quarto output and local resources.
- `results/.gitignore`: keep `results/` present while ignoring generated data.

Per-experiment `README.md` files are forbidden. The report is the single
reader-facing source of truth, and `Rscript run_experiment.R --help` is the
command reference. Do not split design rationale, findings, reproduction
instructions, or output descriptions into a README; if the report is not
self-contained enough to replace one, fix the report. Remove a per-experiment
README when encountered. The sole exception is the collection-level
`experiments/README.md` index (see "Index And Lifecycle").

## Generated And Local Files

Generated outputs live under `results/` and are ignored by git unless there is
a deliberate, documented reason to commit a tiny frozen artifact.

Persistent simulation calibration caches also live under `results/cache/` by
default and are ignored by git. Use the `_support` helpers
`calibration_cache_key()`, `calibration_cache_write()`,
`calibration_cache_read()`, `calibration_cache_list()`, and
`calibration_cache_clear()` for expensive two-stage simulation calibration
objects that should be reused across separate runs. Keys must include the
population target, generator name, generator options, and the magmaan package /
git reference captured by `magmaan_cache_ref()`. Runners must opt in
explicitly; the core library and R package do not own a global on-disk cache,
and invalidation is an explicit delete via `calibration_cache_clear()` or
removing `results/cache/`.

External papers, downloaded files, local notes, cached datasets, and bulky
scratch inputs live under `resources/`. `resources/` is always ignored by git.
Do not make a report silently depend on a local-only resource. If a resource is
needed, the runner should give a clear error or document the acquisition step
in `--help`.

Quarto outputs (`.quarto/`, `report.html`, `report.pdf`, `report_files/`) are
generated and ignored by default. A tracked `report.md` is allowed when an
experiment is linked from the repository README or another GitHub-facing index;
render it from `report.qmd` with Quarto's GFM format so it is readable in
GitHub's normal Markdown viewer.

## Runners

`run_experiment.R` does the expensive work. Rendering the report must be fast:
read result files, reshape modest summaries, make small plots, and stop with a
clear message when results are missing. Do not simulate, fit models, install
packages, download resources, or regenerate fixtures from `report.qmd`.

Runners should:

- support `--help`;
- support a cheap smoke path such as `--smoke`, `--reps 1`, `--cases`, or
  `--cells`;
- set or accept a deterministic seed (`--seed-base` for simulations);
- for runs expected to take more than a few minutes, support either a small
  timing trial (`--smoke`, `--reps`, or `--dry-run-plan`) that can estimate wall
  time for the requested grid, or print that estimate after a pilot subset;
- print progress while expensive loops run. For serial loops this can be every
  fixed number of replications or cells. For parallel loops, report progress
  from the parent process at chunk boundaries, or write a small
  `results/progress.csv` / `results/progress.log` heartbeat that records total
  tasks, completed tasks, elapsed time, and rough ETA. Do not leave a
  30-minute runner silent inside one large estimator block;
- when using `parallel::mclapply`, choose the scheduling deliberately:
  `mc.preschedule = TRUE` is usually best for many roughly equal-cost
  replications because it chunks work per core and avoids thousands of forks;
  `mc.preschedule = FALSE` is for uneven task costs, but then the runner still
  needs parent-visible chunk/progress reporting. For large grids, prefer
  explicit chunks of deterministic replicate IDs over one fork per replicate;
- create `results/` as needed;
- write `results/metadata.csv` with command arguments, seed base when relevant,
  session/package versions, and the design dimensions needed to interpret the
  report;
- write rectangular CSV outputs with stable column names;
- print the paths written at the end.

Keep reusable harness mechanics out of individual experiments when they recur.
Shared path, metadata, seed, result I/O, and formatting helpers may live in a
small support package under `experiments/_support/` (`magmaan.experiments`).
Runners may source `experiments/_support/R/helpers.R` directly so they work
without a separate package install, but the package itself must remain
installable with `R CMD INSTALL experiments/_support`. Do not put SEM logic or
experiment-specific statistical decisions in that package.

## Reports

A report is a short talk, not a lab notebook. Write it for a colleague who has
sixty seconds with you on a Zoom call. They should leave knowing three things:
the question, the answer, and how much to trust it. Everything else is optional
reading.

The four rules, not negotiable per experiment:

- **The first screen is the whole story.** The report opens with the question
  and the answer, in prose, before any table or figure. A reader who stops after
  the first screen still gets the finding. If the answer only emerges by reading
  a table, the report has failed.
- **Lead with the conclusion, never the setup.** No metadata dump, threshold
  table, or design grid before the answer. Setup, if shown at all, comes after
  the finding; most of it belongs in `results/`.
- **One summary artifact per finding.** Each genuine outcome gets at most one
  table or one plot, showing a summary slice, not the full design grid. A
  replication may carry several outcomes; that is fine. A report's length should
  track the number of real findings, never the size of the design grid. If a
  table has a row per design cell, it belongs in `results/`, not the report.
- **No implementation internals in a report, ever.** Caveats are for the reader:
  "reps=10 is noisy", "only the symmetric regime", "asymmetric case deferred".
  If you are naming a C++ primitive, a sandwich variant, or a df
  parameterization, you are writing a code comment in the wrong file.

Required shape, always, in this order:

1. **Question** - one or two sentences. What did we want to know, and why does
   it matter?
2. **Short answer** - the finding in prose, naming the numbers that matter. The
   sentence you would say out loud. For a multi-outcome replication, one sentence
   per outcome.
3. **Evidence** - one summary table or plot per finding, rounded for reading;
   optionally a few sentences on the key slice.
4. **Caveats** - what this run can and cannot establish. Reader-facing only.
5. **Reproduce** - the commands.

An experiment with unusual content may add sections after these, but it always
opens with Question and Short answer.

Self-check before committing a report: read only the Question and Short answer
aloud. If that is not a complete, honest account of the result, fix those two
sections before touching anything else.

Recommended Quarto defaults:

```yaml
format:
  html:
    toc: true
    number-sections: false
execute:
  echo: false
  warning: false
  message: false
```

Reports should read from `results/`, not from hidden local state. If the
required result files are absent, stop with a command the reader can run.

## Tables And Figures

Tables in the report are presentation objects, not data dumps.

- Use properly capitalized, human-readable headers such as `Median Time (ms)`,
  not raw names like `median_elapsed_ms`.
- Keep tables short enough to fit on one screen. Use summaries, top-N rows, or
  grouped slices instead of printing a full design grid or all replicates.
- Put long tables in `results/*.csv` and mention the file in prose.
- Round numbers for interpretation. Keep raw precision in the CSV.
- Prefer one clear plot over several near-duplicates.

## Style

Use the repo's methods-developer voice: direct, specific, and modest about
claims. Say what the experiment can and cannot establish. Keep comparisons
cost-aware when optimizer or simulation work is involved, and separate
statistical conclusions from engineering diagnostics.

When a finding changes implementation state, validation expectations, or the
active backlog, update `project/architecture/roadmap.md` or
`project/backlog/todo.md` as appropriate.
