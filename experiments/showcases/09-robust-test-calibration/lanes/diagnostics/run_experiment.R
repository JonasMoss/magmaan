#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
root <- dirname(normalizePath(script, mustWork = TRUE))
checks <- c(trace = "diagnose_trace_sb_nested.R", runtime = "diagnose_runtime.R",
  setup = "diagnose_lavaan_setup.R", calibration = "diagnose_calibration_cost.R",
  validation = "validate.R")
if (!length(args) || identical(args, "--help")) {
  cat("Usage: --check trace|runtime|setup|calibration|validation [check options]\n",
      "Trace smoke: --check trace --smoke --no-time\n",
      "Other diagnostics: --check NAME --smoke (validation has no options).\n",
      "Original correctness/timing diagnostics; the comparison simulation is retired.\n")
  quit(status = 0L)
}
check <- "trace"
i <- which(args == "--check")
if (length(i)) {
  if (length(i) != 1L || i == length(args)) stop("--check needs one name")
  check <- args[[i + 1L]]; args <- args[-c(i, i + 1L)]
}
if (!check %in% names(checks)) stop("Unknown check: ", check)
if (check == "validation" && length(args)) stop("validation accepts no options")
status <- system2(file.path(R.home("bin"), "Rscript"),
  c(shQuote(file.path(root, checks[[check]])), shQuote(args)))
quit(save = "no", status = status)
