#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "sem_models.R"))
source(file.path(script_dir, "R", "sem_power.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript calibrate_power.R [options]\n\n",
  "Calibrate sparse and diffuse SEM misspecifications to 50% asymptotic\n",
  "normal-theory LRT power at n=200.\n\n",
  "  --n N          Sample size defining local power (default 200).\n",
  "  --target X     Target normal-theory power (default .50).\n",
  "  --models CSV   Model ids (default all five).\n",
  "  --output P     Calibration CSV path.\n",
  "  --help         Show this help.\n", sep = "")

opts <- list(
  n = 200L, target = 0.50, models = NULL,
  output = file.path(script_dir, "results", "calibration",
                     "power_calibration_n200.csv"))
args <- commandArgs(TRUE)
i <- 1L
take <- function() {
  i <<- i + 1L
  if (i > length(args)) stop("missing option value", call. = FALSE)
  args[[i]]
}
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
  else if (arg == "--n") opts$n <- as.integer(take())
  else if (arg == "--target") opts$target <- as.numeric(take())
  else if (arg == "--models") opts$models <- parse_csv_arg(take())
  else if (arg == "--output") opts$output <- take()
  else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}
stopifnot(opts$n >= 80L, opts$target > 0, opts$target < 1)
models <- sem_model_catalog()
if (!is.null(opts$models)) {
  unknown <- setdiff(opts$models, names(models))
  if (length(unknown)) stop("unknown models: ", paste(unknown, collapse = ", "))
  models <- models[opts$models]
}
dir.create(dirname(opts$output), recursive = TRUE, showWarnings = FALSE)
calibration <- sem_calibrate_power_design(
  models, n = opts$n, target_power = opts$target)
write_csv(calibration, opts$output)
write_metadata(sub("\\.csv$", "_metadata.csv", opts$output), list(
  n = opts$n,
  target_power = opts$target,
  alpha = 0.05,
  reference = "population normal-theory ML chi-square",
  noncentrality = "(n-1) times population FML",
  alternatives = sem_power_alternatives,
  models = names(models)), packages = "magmaan")
print(calibration, row.names = FALSE, digits = 5)
cat("wrote ", opts$output, "\n", sep = "")
