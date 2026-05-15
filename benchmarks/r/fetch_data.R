#!/usr/bin/env Rscript
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "common.R"))
source(file.path(bench_script_dir(), "cases.R"))

args <- commandArgs(trailingOnly = TRUE)
ids <- if (length(args) == 0 || identical(args, "all")) case_ids() else args

for (id in ids) {
  case <- get_case(id)
  if (is.null(case$source_url) || is.null(case$raw_file)) {
    cat("skip", id, "- no direct raw-data URL in registry\n")
    next
  }

  ensure_dir(raw_dir(id))
  dest <- file.path(raw_dir(id), case$raw_file)
  if (file.exists(dest)) {
    cat("exists", dest, "\n")
    next
  }

  cat("download", id, "\n  from:", case$source_url, "\n  to:  ", dest, "\n")
  utils::download.file(case$source_url, destfile = dest, mode = "wb", quiet = FALSE)
}
