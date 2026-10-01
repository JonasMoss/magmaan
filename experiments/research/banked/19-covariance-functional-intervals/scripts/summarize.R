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
write_result(read_result("coverage.csv"), "coverage.csv")
write_result(read_result("metadata.csv"), "metadata.csv")
inputs <- sort(inputs)
write_result(data.frame(source = paste0("results/", inputs),
                        bytes = file.info(file.path(root, "results", inputs))$size,
                        md5 = unname(tools::md5sum(file.path(root, "results", inputs)))), "sources.csv")
cat("Curated retained evidence:", out, "\n")
