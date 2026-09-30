#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
usage <- function() cat(
  "Usage: Rscript run_experiment.R [--lane all|target|sampling] [options]\n\n",
  "Compare ordinal covariance omega with direct true-score reliability, and\n",
  "check delta intervals for the covariance target. No default decision.\n\n",
  "--lane all|target|sampling  Select calculation (default all).\n",
  "--smoke                     Four target cells; two sampling replicates.\n",
  "--tol X                     Target integration tolerance (default 1e-10).\n",
  "--reps N --n N --n-pop N     Sampling sizes (defaults 50, 500, 200000).\n",
  "--seed-base N               Recorded/base seed (default 20260702).\n",
  "--help                      Show help without loading packages.\n\n",
  "Outputs: results/target/ and results/sampling/, with separate metadata.\n",
  sep = "")
if ("--help" %in% args) { usage(); quit(save = "no") }
i <- match("--lane", args)
lane <- if (is.na(i)) "all" else {
  if (i == length(args)) stop("--lane needs a value", call. = FALSE)
  args[[i + 1L]]
}
if (!lane %in% c("all", "target", "sampling")) {
  stop("--lane must be all, target or sampling", call. = FALSE)
}
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]))
here <- dirname(script)
setwd(here)
for (name in if (lane == "all") c("target", "sampling") else lane) {
  message("lane=", name)
  source(file.path(here, "R", paste0(name, ".R")), local = new.env(parent = globalenv()))
}
