#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
root <- dirname(normalizePath(script, mustWork = TRUE))
lanes <- c("coefficients", "bell")
usage <- function() cat("Usage: Rscript run_experiment.R --lane NAME [lane options]\n",
  "Lanes: ", paste(lanes, collapse = ", "), "\n",
  "Use --lane NAME --help for the original lane's options.\n",
  "Results and method contracts remain separate for each lane.\n", sep = "")
if (!length(args) || identical(args, "--help")) { usage(); quit(status = 0L) }
lane <- lanes[[1L]]
i <- which(args == "--lane")
if (length(i)) {
  if (length(i) != 1L || i == length(args)) stop("--lane needs one name")
  lane <- args[[i + 1L]]
  args <- args[-c(i, i + 1L)]
}
if (!lane %in% lanes) stop("Unknown lane: ", lane)
child <- file.path(root, "lanes", lane, "run_experiment.R")
if (!length(args)) args <- "--help"
status <- system2(file.path(R.home("bin"), "Rscript"), c(shQuote(child), shQuote(args)))
quit(save = "no", status = status)
