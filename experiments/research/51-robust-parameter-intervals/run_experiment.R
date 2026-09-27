#!/usr/bin/env Rscript
# Design scaffold only. Statistical definitions: report.qmd.
# Implementation and validation requirements: local AGENTS.md.
args <- commandArgs(trailingOnly = TRUE)
if (identical(args, "--help") || length(args) == 0L) {
  cat("Planned experiment: robust Wald, score and profile-LR parameter intervals.\n",
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
cells <- expand.grid(n = c(100L, 300L), distribution = c("normal", "t10", "heterogeneous_gamma"),
                     stringsAsFactors = FALSE)
cells$replications <- 500L
cells$targets <- 2L
cells$seed_base <- 2026092751L
write.table(cells, stdout(), sep = ",", row.names = FALSE, quote = TRUE)
cat("Targets: second loading = 0.7; factor correlation = 0.4.\n")
cat("Methods: wald_O, score_E, lr_E; secondary wald_E, score_O, lr_O; 3 uncorrected diagnostics.\n")
cat("Total datasets:", sum(cells$replications), "\n")
