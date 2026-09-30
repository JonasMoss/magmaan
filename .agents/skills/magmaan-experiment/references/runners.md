# Runner and artifact workflow

Paths below are relative to the magmaan repository or the selected study.

## Generated and local files

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
