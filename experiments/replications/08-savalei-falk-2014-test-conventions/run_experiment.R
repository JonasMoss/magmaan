#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
root <- dirname(normalizePath(script, mustWork = TRUE))
lanes <- list("definitions" = c("definitions/run_experiment.R"),"matrices" = c("matrices/run_experiment.R"))
checks <- list()
usage <- function() cat("Usage: Rscript run_experiment.R --lane NAME [options]\n",
  "Lanes: ", paste(names(lanes), collapse = ", "), "\n",
  "Diagnostics (--check NAME): ", paste(names(checks), collapse = ", "), "\n",
  "Use --lane NAME --help for options and a fresh absolute --results-dir for verification.\n", sep = "")
if (!length(args) || identical(args, "--help")) { usage(); quit(status = 0L) }
selector <- which(args %in% c("--lane", "--check"))
if (length(selector) != 1L || selector == length(args)) stop("Choose one lane or diagnostic")
i <- selector[[1L]]; kind <- args[[i]]; name <- args[[i+1L]]; args <- args[-c(i,i+1L)]
choices <- if (kind == "--lane") lanes else checks
if (!name %in% names(choices)) stop("Unknown ", kind, ": ", name)
selected <- choices[[name]]
child <- file.path(root,"lanes",selected[[1L]])
if (!length(args)) args <- "--help"
setwd(dirname(child))
status <- system2(file.path(R.home("bin"),"Rscript"),
  shQuote(c(child, selected[-1L], args)))
quit(save = "no", status = status)
