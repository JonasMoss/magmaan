#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
root <- dirname(normalizePath(script, mustWork = TRUE))
lanes <- c("calibration" = "calibration/run_experiment.R", "score-flips" = "score-flips/run_experiment.R", "oracle" = "oracle/run_experiment.R", "pilot" = "pilot/run_experiment.R")
checks <- c()
usage <- function() cat("Usage: Rscript run_experiment.R --lane NAME [lane options]\n",
  "Lanes: ", paste(names(lanes), collapse = ", "), "\n",
  "Use --lane NAME --help for the original compute path's options.\n",
  if (length(checks)) paste0("Diagnostics: --check ", paste(names(checks),collapse="|"), " [options]\n") else "",
  "Use fresh absolute output directories; do not overwrite historical results.\n", sep = "")
if (!length(args) || identical(args, "--help")) { usage(); quit(status = 0L) }
selectors <- which(args %in% c("--lane", "--check"))
if (length(selectors) != 1L || selectors == length(args)) stop("Choose one --lane or --check and its name")
i <- selectors[[1L]]; kind <- args[[i]]; name <- args[[i+1L]]; args <- args[-c(i,i+1L)]
choices <- if (kind == "--lane") lanes else checks
if (!name %in% names(choices)) stop("Unknown ", kind, ": ", name)
child <- file.path(root,"lanes",choices[[name]])

if (!length(args)) args <- "--help"
# Old runners that use relative paths run inside their own lane. Their results
# symlinks keep all output within this experiment's results tree.
setwd(dirname(child))
status <- system2(file.path(R.home("bin"),"Rscript"),c(shQuote(child),shQuote(args)))
quit(save="no",status=status)
