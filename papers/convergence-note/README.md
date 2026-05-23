# convergence-note

Working title: Fast start-value portfolios for hard small-sample SEM.

This paper is a convergence and starting-value note built around magmaan's
strengths: fast SEM fitting, explicit start-value producers, optimizer
diagnostics, and growing corpora of real and simulated models. The first target
is not a universal starting-value theorem. It is a practical comparison of
start policies under fixed compute budgets, with special attention to cases
where modern optimizers still end at nonconvergence, non-PD paths, boundary
solutions, or ugly local basins.

## Priority

1. Reproduce the small-sample designs from De Jonckere and Rosseel (2022,
   2025) and Ludtke, Ulitzsch, and Robitzsch (2021) as magmaan/lavaan-ready
   data generators.
2. Mine genuinely hard cases by filtering for default failures, improper
   solutions, high final gradients, and disagreement across start policies.
3. Compare deterministic starts, bounded random starts, and small portfolios
   under equal evaluation budgets.
4. Use magmaan's speed to separate optimizer failure from data/model hardness:
   objective value, PD margin, final gradient norm, evaluation count, wall time,
   admissibility, and basin agreement.

## Layout

- `convergence-note.tex` - manuscript skeleton.
- `convergence-note.bib` - paper bibliography.
- `notes/` - design notes and paper-reading summaries.
- `resources/` - local paper PDFs and external reference material.
- `scripts/` - simulation runners and analysis scripts.
- `tables/` - generated or curated tables.
- `figures/` - generated or curated figures.
- `r-package/` - optional paper-local helpers if the root R package surface
  gets too broad.

## Current R Helpers

The root `r-package/` exports the initial simulation factories:

```r
convergence_sim("dejonckere_simple_2025")
convergence_sim("dejonckere_shrinkage_2023_study2_14")
convergence_sim_dejonckere_simple()
convergence_sim_dejonckere_crossloading()
convergence_sim_dejonckere_shrinkage()
convergence_sim_dejonckere_msst()
convergence_sim_ludtke_cfa()
convergence_sim_catalog()
```

Each factory returns raw data, sample statistics, analysis syntax, population
syntax, the population covariance, and a truth bundle.
