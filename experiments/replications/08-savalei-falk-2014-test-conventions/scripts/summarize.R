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
fp <- read_result("matrices/fingerprint/summary.csv")
d <- read_result("matrices/fingerprint/design.csv")
rp <- read_result("matrices/replication/summary.csv")
rd <- read_result("matrices/replication/design.csv")
select_route <- function(x, route) {
  if (route == "fiml_expected_hybrid") {
    keep <- x$method == "fiml" & x$residual_information == "structured_expected_h1" &
      x$omega_bread_point == "saturated" & x$omega_bread_kind == "expected" & x$omega_meat_point == "saturated"
  } else if (route == "fiml_saturated_expected") {
    keep <- x$method == "fiml" & x$residual_information == "saturated_expected_h1" &
      x$omega_bread_point == "saturated" & x$omega_bread_kind == "expected" & x$omega_meat_point == "saturated"
  } else if (route == "fiml_literal_observed_proxy") {
    keep <- x$method == "fiml" & x$residual_information == "structured_observed_h1" &
      x$omega_bread_point == "structured" & x$omega_bread_kind == "observed" & x$omega_meat_point == "structured"
  } else {
    keep <- x$method == "ml2s" & x$omega_bread_point == "saturated" &
      x$omega_bread_kind == "observed" & x$omega_meat_point == "saturated" &
      x$stage2_information == "structured_observed"
  }
  x[which(keep), , drop = FALSE]
}
rows <- lapply(c("fiml_expected_hybrid", "fiml_saturated_expected", "fiml_literal_observed_proxy",
                 "ml2s_source_n_minus_1", "ml2s_source_n"), function(route) {
  x <- select_route(fp, route)
  stopifnot(nrow(x) == 6L, !anyDuplicated(x$cell_id))
  target <- if (startsWith(route, "fiml")) d$paper_fiml_rejection else d$paper_ts_rejection
  target <- target[match(x$cell_id, d$cell_id)]
  rate <- if (route == "ml2s_source_n_minus_1") x$rejection_rate_n_minus_1 else x$rejection_rate
  data.frame(route, cells = nrow(x), min_usable = min(x$replications), max_usable = max(x$replications),
             rmse_percentage_points = 100 * sqrt(mean((rate - target)^2)))
})
write_result(do.call(rbind, rows), "fingerprint.csv")
rows <- lapply(c("fiml_expected_hybrid", "fiml_saturated_expected", "fiml_literal_observed_proxy",
                 "ml2s_source_n_minus_1", "ml2s_source_n"), function(route) {
  x <- select_route(rp, route)
  x <- x[x$cell_id == "paper-mar-l-k7", , drop = FALSE]
  stopifnot(nrow(x) == 1L)
  rate <- if (route == "ml2s_source_n_minus_1") x$rejection_rate_n_minus_1 else x$rejection_rate
  target <- if (startsWith(route, "fiml")) .633 else .100
  data.frame(route, attempted = 1000L, usable = x$replications, rejection = rate, paper_rejection = target)
})
write_result(do.call(rbind, rows), "mar_comparison.csv")
x <- read_result("definitions/configuration_probe.csv")
write_result(x, "routing_witness.csv")
for (p in c("definitions/metadata.csv", "matrices/fingerprint/metadata.csv", "matrices/replication/metadata.csv"))
  invisible(read_result(p))
for (name in c("published_parameters.csv", "published_design.csv", "missingness_sets.csv", "implementation_map.csv"))
  write_result(read_result(file.path("definitions", name)), name)
inputs <- sort(inputs)
write_result(data.frame(source = paste0("results/", inputs),
                        bytes = file.info(file.path(root, "results", inputs))$size,
                        md5 = unname(tools::md5sum(file.path(root, "results", inputs)))), "sources.csv")
cat("Curated retained evidence:", out, "\n")
