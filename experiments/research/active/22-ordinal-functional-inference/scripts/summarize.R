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
x <- read_result("delta/coverage/simulation_summary.csv")
write_result(x, "delta_gaussian.csv")
x <- read_result("delta/stress/simulation_summary.csv")
write_result(x, "delta_stress.csv")
x <- read_result("profile/simulation_summary.csv")
x$usable <- x$reps - x$failures
x$ci_successful <- x$ci_attempted - x$ci_failures
fields <- c("dgp", "regime", "n", "reference", "reps", "usable", "failures",
            "target", "sample_polychoric_target", "reference_coverage_95",
            "ci_attempted", "ci_successful", "ci_failures", "ci_coverage_95",
            "ci_lrt_disagreements", "max_abs_ci_endpoint_gap")
write_result(x[, fields], "profile_inversion.csv")
x <- read_result("profile/stress_pointwise/simulation_summary.csv")
x$usable <- x$reps - x$failures
write_result(x[, c("dgp", "regime", "n", "reference", "reps", "usable", "failures",
                   "reference_coverage_95")], "profile_pointwise.csv")
for (p in c("delta/coverage/metadata.csv", "delta/stress/metadata.csv",
            "profile/metadata.csv", "profile/stress_pointwise/metadata.csv")) invisible(read_result(p))
inputs <- sort(inputs)
write_result(data.frame(source = paste0("results/", inputs),
                        bytes = file.info(file.path(root, "results", inputs))$size,
                        md5 = unname(tools::md5sum(file.path(root, "results", inputs)))), "sources.csv")
cat("Curated retained evidence:", out, "\n")
