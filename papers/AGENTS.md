# Papers

Repo-local rules for magmaan paper subprojects. Sibling to
`experiments/AGENTS.md` but distinct: experiments are short, single-question
folders rendered as one Quarto report; papers are larger LaTeX manuscripts
with their own simulation pipelines, helper packages, and online supplements.

Each paper folder is **its own independent git repository**, nested inside
the magmaan working copy. The outer magmaan repo ignores `papers/*` except
this convention doc. Submodules are not used; a paper that is ready to
archive can be pushed to its own remote (Zenodo, OSF mirror, GitHub) without
touching magmaan.

## Purpose

One paper folder holds one manuscript-in-progress aimed at a specific
journal. It owns its bibliography, its simulation scripts, its helper R
package, its online supplement, and a curated set of run outputs that the
manuscript cites. It does not hold scratch that belongs in a separate
experiments folder, nor library changes that belong in `src/`.

## Directory Shape

A short kebab-case slug names the folder. No numeric prefix; papers are not
ordered.

```text
papers/<paper-slug>/
  # ---- archive-bound: ships to the journal / OSF / Zenodo ----
  <paper-slug>.tex            # manuscript
  <paper-slug>.bib            # bibliography
  figures/                    # final figures (tracked)
  tables/                     # final .tex tables + stats macros (tracked)
  supplement/                 # online supplement (Quarto sources)
    *.qmd                     # tracked
    *.html / *_files/         # generated, ignored
  scripts/                    # thin runners, public-facing
  r-package/                  # paper helper package, public-facing
  results/                    # curated CSVs that the manuscript cites (tracked)
    raw/                      # bulky run dumps (ignored)
    .gitignore                # keeps results/ present; ignores raw/

  # ---- dev-only: never ships ----
  dev/
    notes/                    # derivation / design notes (tracked)
    inspect/                  # .qmd reports that read run data for dev eyes
    audits/                   # investigation write-ups + raw audit data
    status/                   # dated status snapshots
    todo.md                   # current paper TODO

  # ---- always ignored ----
  extern/                     # downloaded papers, code mirrors
  resources/                  # local data, scratch

  # ---- meta ----
  README.md                   # one-screen overview for an outside reader
  AGENTS.md                   # paper-specific direction (voice, scope)
  justfile                    # pipeline driver
  .gitignore
```

The split is **archive-bound vs dev-only**, with `extern/` and `resources/`
as permanently-ignored sinks. The `just archive` recipe bundles only the
archive-bound parts; nothing from `dev/`, `extern/`, or `resources/`
appears in the OSF zip.

## Naming

- Folder slug is the manuscript's working short name: `snlls-continuous`,
  `composite-ml`, `ugamma-fast`. Avoid numeric prefixes; paper slugs are
  meaningful identifiers and there is no "next paper" sequence.
- The main TeX file matches the folder slug (`<slug>.tex`, `<slug>.bib`).
  Do not rename to `paper.tex`; ambiguous filenames break grep across
  paper boundaries.
- The R helper package directory is always `r-package/`. Its `Package:`
  name in `DESCRIPTION` is the folder slug with dashes removed
  (`snllscontinuous`, `compositeml`).

## Run Outputs

Run output is partitioned by intent, not by run id:

- **`results/`** — curated CSVs (and other small artifacts) that the
  manuscript, the supplement, or a public reader actually depends on.
  Tracked. Small enough to live in git.
- **`results/raw/`** — everything else. Raw per-replicate simulation
  output, timing logs, per-run pilot directories. Ignored. Bulky and
  unstable.

Runners write to `results/raw/<run-id>/` by default. A summarising step
(usually invoked via `just tables` or a `make_*` script) reads
`results/raw/` and writes the curated CSVs into `results/`. If a runner
writes directly to `results/` it must be small, stable, and cited in
the manuscript or supplement.

`LATEST_*` pointer files used to find the most recent run live next to the
runs themselves under `results/raw/` and are git-ignored.

## Supplement

`supplement/` is the home for the online appendix. Sources are tracked
`.qmd` files; generated HTML/PDF outputs are ignored. Supplements read
from `results/` and `results/raw/`, never from `dev/`.

Each `supplement/*.qmd` should declare itself a supplement in the YAML
title (`title: "Online Supplement: ..."`) and should be self-contained
enough that an OSF reader can render it from a clean clone given a
populated `results/` directory.

## Dev Folder

`dev/` collects work intended for the author and coding agents, never for
the OSF archive.

- `dev/notes/` — derivation notes, design notes, implementation maps. May
  feed paper text later. Tracked.
- `dev/inspect/` — `.qmd` files that read run data and produce inspection
  HTML for the author. Differs from `supplement/` in audience and polish;
  these may break, change, or be deleted freely. Tracked sources, ignored
  HTML.
- `dev/audits/` — investigation write-ups for specific anomalies
  (`lcs-objective-gap.md`, `convergence-audit-notes.md`) and any small
  audit data they need. Tracked.
- `dev/status/` — dated status snapshots
  (`status-2026-05-23.md`). Tracked.
- `dev/todo.md` — current paper TODO. Tracked.

The rule of thumb: if a reader of the published paper might want it,
it belongs in `supplement/` or `results/`; otherwise it belongs in `dev/`.

## Build Driver

Papers use a `justfile`, not a `Makefile`. Recipes:

```
just                    # alias for `just --list`
just pdf                # build the manuscript PDF
just sim <name>         # run one simulation by short name
just sims               # run every simulation (hours)
just tables             # rebuild tables/ from results/
just figures            # rebuild figures/ from results/
just supplement         # render supplement/*.qmd to HTML
just paper              # tables + figures + pdf (no sims)
just archive            # zip the archive-bound subset for OSF
just clean              # remove LaTeX build products
```

Heavy work goes in `sim` / `sims`. The cheap recipes (`tables`, `figures`,
`supplement`, `pdf`, `paper`) must reuse existing `results/` and complete
in seconds-to-minutes, so manuscript iteration does not require
re-simulation.

OSF reviewers may not have `just` installed. The `archive` bundle should
include a `scripts/run-all.sh` mirror at submission time. Day-to-day
work does not need it.

## R Helper Package

Each paper carries a paper-local R helper package under `r-package/`. It
holds reusable benchmark cases, simulation generators, and runners that
would otherwise bloat individual scripts. Scripts load it with
`pkgload::load_all("r-package")`; the package itself must be installable
with `R CMD INSTALL r-package`.

Package code is organised by tier inside `R/`:

- `core-*.R` — reusable infrastructure (env helpers, I/O, timing, stats,
  optimizer specs). Expected to survive into future papers.
- `harness-*.R` — this paper's apparatus (its benchmark runner,
  simulation designs, table toolkit). Expected to be rewritten per paper.

The tiering is a reader convention; R sources every file in `R/`
regardless of name.

## Manuscript

LaTeX, plain `natbib`. `latexmk` when available, `pdflatex`/`bibtex` cycle
otherwise; the `justfile` handles both. Manuscript prose numbers come from
a single generated file under `tables/` (e.g. `tables/<slug>_stats.tex`),
so every cited number is reproducible from `results/` and the prose itself
carries `??` fallbacks until the tables are built.

Bibliographies are paper-local. Do not share `.bib` files across paper
folders; copy what you need.

## Style

Prose, tables, figures, citations, math notation, and audience profiles
live in `papers/STYLE.md`. Every paper inherits those defaults. The
per-paper AGENTS.md declares one `Profile:` line (Math / Applied / Tool)
that picks the matching length and tone defaults from `STYLE.md`, plus
paper-specific direction (target journal, scope, what to cite or skip).
It should not re-state the defaults.

## Reproducibility and Versioning

magmaan has no releases, so the honest version is a commit SHA. Near
submission:

1. Pick a magmaan commit.
2. Tag it (e.g. `<paper-slug>-v1`).
3. Rerun every simulation against that single state and regenerate
   every table, figure, and supplement from one consistent build.
4. Cite the commit/tag in the manuscript ("benchmarks were run against
   magmaan at commit `abc1234`").

Bit-exact reproducibility is not the goal; a recoverable pin plus robust
reporting is.

Report ratios where possible (a speedup factor survives hardware
changes), disclose CPU/OS/compiler/BLAS, and keep correctness validation
(against lavaan / OpenMx to several digits) separate from speed
benchmarks.

## Git Hygiene

- Generated outputs (LaTeX build products, Quarto HTML, raw run data,
  rendered PDFs) are ignored. Curated tables/figures/results are tracked.
- Bulky binaries, downloaded papers, vendor source mirrors, and local
  scratch all live under `extern/` or `resources/` and never enter the
  index.
- Each paper folder carries its own `.gitignore`. The patterns below are
  the recommended baseline; copy and trim per paper.

```gitignore
# Local resources and source mirrors
extern/
resources/

# Run output
results/raw/
results/**/LATEST*

# Supplement build
supplement/*.html
supplement/*_files/

# Dev build artifacts
dev/inspect/*.html
dev/inspect/*_files/

# R artifacts
.Rproj.user/
.Rhistory
.RData
.Ruserdata
*.Rcheck/
*.tar.gz
r-package/src/*.o
r-package/src/*.so
r-package/src/*.dll

# LaTeX build products
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.log
*.lof
*.lot
*.nav
*.out
*.pdf
*.run.xml
*.snm
*.synctex.gz
*.toc
*.vrb
*.xdv
_minted-*/
build/
```

Placeholder `.gitkeep` files and per-folder `.gitignore` files may be
tracked so the directory layout survives cloning.
