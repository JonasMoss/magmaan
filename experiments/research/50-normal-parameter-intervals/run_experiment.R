#!/usr/bin/env Rscript
# Design scaffold only. Statistical definitions: report.qmd.
# Implementation and validation requirements: local AGENTS.md.
args <- commandArgs(trailingOnly = TRUE)
if (identical(args, "--help") || length(args) == 0L) {
  cat("Planned experiment: normal parameter intervals with a Bartlett feasibility pilot.\n",
      "Status: design only; simulation is not implemented.\n",
      "Usage: Rscript run_experiment.R --help | --plan\n",
      "--plan prints the fixed cell design; writes no results and fits no models.\n",
      "Read report.qmd and AGENTS.md before implementing computation.\n", sep = "")
  quit(status = 0L)
}
if (!identical(args, "--plan")) {
  stop("Design scaffold only: supported options are --help and --plan; simulation is not implemented.",
       call. = FALSE)
}
cells <- expand.grid(n = c(100L, 300L, 1000L), distribution = "normal",
                     stringsAsFactors = FALSE)
cells$replications <- 200L
cells$targets <- 2L
cells$seed_base <- 2026092750L
write.table(cells, stdout(), sep = ",", row.names = FALSE, quote = TRUE)
cat("Targets: second loading = 0.7; factor correlation = 0.4.\n")
cat("Methods: wald_expected, score_expected, lr; secondary wald_observed.\n")
cat("Total datasets:", sum(cells$replications), "\n")
cat("Bartlett pilot: first 50 N=100 datasets, loading only, B=399; first 10 repeated at B=1999 and an independent stream.\n")
