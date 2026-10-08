#!/usr/bin/env Rscript
# Compact frozen evidence for a combined production directory: the per-draw
# failure and timing rows stay ignored (they exceed the tracked-file limit),
# while their counts and per-cell timing quantiles are committed beside the
# summary. Usage: Rscript scripts/compact_production.R results/<run-id>
args <- commandArgs(TRUE)
if (length(args) != 1 || args[1] %in% c("-h", "--help")) {
  cat("Usage: Rscript scripts/compact_production.R RESULTS_DIR\n",
      "Writes failure_counts.csv and timing_summary.csv from failures.csv and timing.csv.\n")
  quit(status = if (length(args) == 1) 0 else 1)
}
dir <- args[1]
failures <- read.csv(file.path(dir, "failures.csv"))
timing <- read.csv(file.path(dir, "timing.csv"))

# Error class: the message without draw-specific detail (iteration counts).
failures$error_class <- sub(" after [0-9]+ iters.*$", "", sub("\n.*$", "", failures$error))
key <- c("cell_id", "arm", "target", "error_class")
counts <- aggregate(list(count = rep(1L, nrow(failures))), failures[key], sum)
counts <- counts[do.call(order, counts[key]), ]
write.csv(counts, file.path(dir, "failure_counts.csv"), row.names = FALSE)

q <- function(x, p) unname(stats::quantile(x, p, names = FALSE))
cells <- sort(unique(timing$cell_id))
summary <- do.call(rbind, lapply(cells, function(id) {
  s <- timing$seconds[timing$cell_id == id]
  data.frame(cell_id = id, draws = length(s), total_seconds = sum(s),
             median_seconds = q(s, 0.5), p90_seconds = q(s, 0.9),
             p99_seconds = q(s, 0.99), max_seconds = max(s))
}))
write.csv(summary, file.path(dir, "timing_summary.csv"), row.names = FALSE)
cat("Wrote failure_counts.csv (", nrow(counts), " rows) and timing_summary.csv (",
    nrow(summary), " cells) in ", dir, "\n", sep = "")
