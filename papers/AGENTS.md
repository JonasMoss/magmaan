# Papers

Repo-local rules for magmaan paper subprojects. Sibling to
`experiments/AGENTS.md` but distinct: experiments are short, single-question
folders rendered as one Quarto report; papers are LaTeX manuscripts with their
own studies, code, supplement, and kept results.

Each paper folder is **its own independent git repository**, nested inside
the magmaan working copy. The outer magmaan repo ignores `papers/*` except
this convention doc, `CLAUDE.md`, and `STYLE.md`. Submodules are not used; a
paper that is ready to archive can be pushed to its own remote (Zenodo, OSF
mirror, GitHub) without touching magmaan.

**Papers are private** (see "Dependency layering" in the root `AGENTS.md`).
Nothing outside a paper references it: no experiment, test, benchmark, core
file, or other paper may `source`, `load`, `include`, or read
`papers/<this>/**`. A paper depends only on the core library and `r-package`;
shared infrastructure lives there or in `experiments/_support`, never across a
paper boundary. Enforced (best-effort, since `papers/*` is gitignored) by
`cpp/tests/tools/check_layering.sh`.

Retired paper working trees live frozen under `papers/_archive/`. They are
historical material rather than active paper leaves, so the layering checker
does not scan their internal code. Active core, orchestration, experiment,
test, benchmark, and paper code still must not depend on the archive.

## Layout

Papers use the collection-wide layout `paper-project-structure/0.2`, defined in
`00_paper-writing/docs/contracts/paper-project-structure.md` next to this
checkout. Create a paper from the magmaan root with the scaffold, which also
gives it its own repository:

```sh
../00_paper-writing/tools/scaffold-paper.rb \
  --slug <slug> --title "Working Title" --output papers/<slug>
```

```text
papers/<slug>/
  README.md  AGENTS.md  STATUS.md (optional)  justfile  .gitignore
  manuscript/  <slug>.tex, <slug>-supplement.tex, preamble.tex, <slug>.bib,
               appendices/, tables/, figures/; compiles on its own        ships
  code/        studies/<study>/run.R, tables.R, modal/, slurm/,
               r-package/, environment/magmaan.json; README.md             ships
  results/     curated CSVs and kept runs in runs/<study>/<run-id>/        ships
  dev/         notes, probes, checks, drafts, talks                        never
  local/       git-ignored: literature, raw runs, builds, magmaan checkouts never
```

The three rules, the study runner protocol, `just sim`/`keep`/`tables`, and
the `just ship` bundle are in the contract; the paper's `README.md` repeats
them. Single-question explorations that are about magmaan rather than the
paper belong in `experiments/`, not in the paper's `dev/`.

Papers still on an older layout (sem-psd and sem-sphere on 0.1;
closed-form-omega, guttman-inference, sem-misspecification, snlls-continuous,
and snlls-ordinal on earlier flat layouts) keep working as they are. Move one
at a time with the `paper-init` skill, which moves nothing until the author
approves the complete mapping.

## magmaan specifics

- **Pin magmaan.** Record the exact magmaan commit the paper's code uses in
  `code/environment/magmaan.json`. Runners read a clean checkout of that commit
  through `MAGMAAN_SOURCE_ROOT` (and a matching build through
  `MAGMAAN_BUILD_ROOT` when paper code compiles against the core). Checkouts and
  builds live under the paper's `local/`, never in Git.
- **Paper-local R package.** `code/r-package/`, with `Package:` equal to the
  slug with dashes removed (`snllscontinuous`). The layering check reads its
  `DESCRIPTION`, so core code cannot reference the paper's namespace. Inside
  `R/`, `core-*.R` holds infrastructure expected to outlive the paper and
  `harness-*.R` holds this paper's apparatus; R sources every file regardless.
- **Modal.** The template's `code/modal/app.py` image must also build magmaan at
  the pinned commit before running the study; closed-form-omega and sem-psd
  show two ways to do that.

## Style

Prose, tables, figures, citations, math notation, and audience profiles live
in `papers/STYLE.md`. Every paper inherits those defaults. The per-paper
`AGENTS.md` declares one `Profile:` line (Math, Applied, or Tool) plus
paper-specific direction (target journal, scope, what to cite or skip). It
should not restate the defaults.

## Reproducibility and versioning

magmaan has no releases, so the honest version is a commit SHA. Near
submission:

1. Pick a magmaan commit, tag it (for example `<slug>-v1`), and record it in
   `code/environment/magmaan.json`.
2. Rerun every study against that single state, keep the runs, and regenerate
   tables and figures with `just tables`.
3. Commit, run `just ship`, and cite the commit or tag in the manuscript
   ("benchmarks were run against magmaan at commit `abc1234`").

Bit-exact reproducibility is not the goal; a recoverable pin plus robust
reporting is. Report ratios where possible (a speedup factor survives hardware
changes), disclose CPU, OS, compiler, and BLAS, and keep correctness
validation (against lavaan or OpenMx to several digits) separate from speed
benchmarks.

## Git hygiene

The scaffold's `.gitignore` and self-ignoring `local/` are the baseline.
Downloaded papers, source mirrors, magmaan checkouts and builds, raw runs, and
ship bundles stay in `local/`. Kept runs, curated results, and generated tables
and figures are committed; build products and rendered PDFs are not.
