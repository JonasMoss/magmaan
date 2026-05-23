# Resources

Local PDFs and reference material for the convergence-note paper. Keep this
directory curated: paper PDFs, small replication-code archives, and short
indexes are welcome; generated simulation output and broad source mirrors are
not.

## Papers

- `De Jonckere and Rosseel 2022 - Using bounded estimation to avoid nonconvergence in small sample SEM - published.pdf`:
  published bounded-estimation paper.
- `De Jonckere and Rosseel 2023 - A model-based shrinkage target to avoid nonconvergence in small sample SEM.pdf`:
  published model-based shrinkage-target paper.
- `Jonckere and Rosseel 2025 - A note on using random starting values in small sample SEM.pdf`:
  published bounded-random-start paper.
- `Ludtke et al. 2021 - A comparison of penalized maximum likelihood estima ... stimating confirmatory factor analysis models with small sample sizes.pdf`:
  penalized/Bayesian CFA comparison in small
  samples.

## Expanded OSF resources

- `dejonckere_rosseel_2022_bounded/`: replication archive for the
  bounded-estimation paper. It contains appendix examples and the two
  simulation-study scripts comparing unbounded ML, observed-variance bounds,
  standard bounds, and wider bounds.
- `dejonckere_rosseel_2023_model_based_shrinkage_osf/`: replication archive for
  the model-based shrinkage-target note. This is start-adjacent rather than a
  start-value paper: it stabilizes small-sample covariance input by replacing
  `S` with `(1 - lambda) S + lambda T`, where `T` is a model-based target.
- `dejonckere_rosseel_2025_random_starts_osf/`: replication archive for the
  bounded-random-start paper. The simulation scripts define the population
  models and bounded-random-start reruns used in the 2025 note.
- `ludtke_2021_penalized/`: replication material for the penalized/Bayesian CFA
  comparison, including Stan models, an empirical example, and simulation data
  generation/analysis scripts.

The raw `osfstorage-archive*.zip` downloads and duplicate standalone OSF
preprint PDFs were removed after expansion. The older bounded-estimation
preprint was replaced by the published 2022 PDF.
