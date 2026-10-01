#!/usr/bin/env Rscript
# Curate retained evidence only. No fitting or regeneration of historical runs.
a <- commandArgs(FALSE)
f <- sub("^--file=", "", a[grepl("^--file=", a)][1L])
root <- dirname(dirname(normalizePath(f)))
if ("--help" %in% commandArgs(TRUE)) {
  cat("Usage: Rscript scripts/summarize.R\nRead this study's retained CSVs and refresh results/overview with source hashes.\n")
  quit(save = "no")
}
inputs <- character()
read_result <- function(path) {
  full <- file.path(root, "results", path)
  if (!file.exists(full)) stop("Missing retained evidence: ", full)
  inputs <<- union(inputs, path)
  read.csv(full, stringsAsFactors = FALSE)
}
out <- file.path(root, "results", "overview")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write_result <- function(x, name) write.csv(x, file.path(out, name), row.names = FALSE, na = "")
key_value <- function(path, key) {
  x <- read_result(path)
  x$value[match(key, x$key)]
}
x <- read_result("bifactor/coverage.csv")
write_result(x, "bifactor_intervals.csv")
x <- read_result("bifactor/lr_coverage.csv")
x$attempts <- as.integer(key_value("bifactor/lr_metadata.csv", "reps"))
write_result(x[, c("p", "coef", "dist", "n", "attempts", "reps_ok", "mean_LR",
                   "cov_chi2", "cov_bartlett", "cov_robust")], "bifactor_pointwise.csv")
p <- read_result("profile/semlbci_parity.csv")
engine_ok <- is.finite(p$tdev_eng_lo) & is.finite(p$tdev_eng_hi) &
  abs(p$tdev_eng_lo) <= .02 & abs(p$tdev_eng_hi) <= .02
oracle_ok <- is.finite(p$tdev_sl_lo) & is.finite(p$tdev_sl_hi) &
  abs(p$tdev_sl_lo) <= .02 & abs(p$tdev_sl_hi) <= .02
both <- engine_ok & oracle_ok
write_result(data.frame(datasets_functionals = nrow(p), engine_bad_roots = sum(!engine_ok),
                        oracle_bad_roots = sum(!oracle_ok), both_valid = sum(both),
                        max_bound_difference_both_valid = if (any(both))
                          max(abs(c(p$d_lo[both], p$d_hi[both]))) else NA_real_), "parity.csv")
x <- read_result("profile/coverage_H.csv")
x$functional <- "H"
x$attempts <- as.integer(key_value("profile/coverage_metadata_H.csv", "reps"))
y <- read_result("profile/coverage_omega.csv")
y$functional <- "omega_total"
y$attempts <- as.integer(key_value("profile/coverage_metadata_omega.csv", "reps"))
fields <- c("functional", "pop", "n", "attempts", "reps", "mean_T", "cov_nt", "cov_bart_oracle", "cov_bart_null")
write_result(rbind(x[, fields], y[, fields]), "profile_pointwise.csv")
inputs <- sort(inputs)
write_result(data.frame(source = paste0("results/", inputs),
                        bytes = file.info(file.path(root, "results", inputs))$size,
                        md5 = unname(tools::md5sum(file.path(root, "results", inputs)))), "sources.csv")
cat("Curated retained evidence:", out, "\n")
