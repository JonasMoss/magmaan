#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
root <- dirname(normalizePath(script, mustWork = TRUE))
lanes <- c("sem" = "global/run_sem_models.R", "screen" = "global/run_experiment.R", "two-stage" = "two-stage/run_experiment.R", "legacy-mlr" = "legacy-mlr/run_experiment.R")
checks <- c("crossing" = "global/check_lrt_score_crossing.R", "pseudo-null" = "global/investigate_pseudonull_information.R", "pseudo-target" = "global/estimate_pseudotrue_shadow.R", "h1-quadratic" = "global/smoke_fiml_h1_quadratic.R", "ml2s-rls" = "global/smoke_ml2s_rls.R", "mlr-parity" = "global/validate_mlr_lavaan.R", "power-calibration" = "global/calibrate_power.R")
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
if (kind == "--lane" && name == "sem" && "--smoke" %in% args) {
  args <- args[args != "--smoke"]
  defaults <- c("--reps"="1","--n"="120","--flips"="9","--cores"="1",
    "--max-cells"="1","--seed-base"="2026102044")
  for (key in names(defaults)) if (!key %in% args) args <- c(args,key,defaults[[key]])
}
if (!length(args)) args <- "--help"
# Old runners that use relative paths run inside their own lane. Their results
# symlinks keep all output within this experiment's results tree.
setwd(dirname(child))
status <- system2(file.path(R.home("bin"),"Rscript"),c(shQuote(child),shQuote(args)))
quit(save="no",status=status)
