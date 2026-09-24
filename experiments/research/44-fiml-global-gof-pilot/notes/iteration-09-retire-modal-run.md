# Iteration 09: retire the Modal continuation

Date: 2026-08-21

The `power-v1-20260821` Modal continuation is retired and will not be repaired
or interpreted as the paper's power study.

- The stress phase completed and combined all 60 cells, producing 30,000
  estimator rows.
- The focus phase completed 162 of 165 shards. The three partial 15-indicator
  cells stopped after 725, 650, and 900 of 1,000 replications, respectively.
- Their restarts failed because the resumability guard read the blank
  `max_cells` CSV field as `NA` and compared it with the in-memory empty string.
  This was an orchestration/configuration failure, not an SEM fit failure.
- Repairing the remaining 725 replications is not worthwhile because the model
  and missingness setup will change. The 164,275 available focus rows were
  nevertheless collected and combined with an explicit coverage ledger for an
  exploratory report snapshot.

The persistent Modal volume is left intact as provenance; no files are deleted.
The next production run will be built for Slurm after the five source-based
models, MAR mechanisms, adverse estimand-stress arm, and power alternatives are
frozen. Results from this retired run may inform implementation diagnostics but
must not enter the publication tables. They may appear in the preliminary
experiment report only when labeled incomplete and retired.
