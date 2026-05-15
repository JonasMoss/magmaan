#!/usr/bin/env Rscript
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "common.R"))
source(file.path(bench_script_dir(), "cases.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("usage: Rscript benchmarks/r/prepare_case.R <case_id> [<case_id> ...] | all", call. = FALSE)
}
ids <- if (length(args) == 1 && identical(args, "all")) {
  names(Filter(function(x) isTRUE(x$supported_now), benchmark_cases))
} else {
  args
}

for (id in ids) {
  case <- get_case(id)
  data <- load_case_source_data(case)
  meta <- write_prepared_data(
    id,
    data,
    sprintf("%s::%s", case$package %||% case$source_type, case$dataset %||% case$raw_file %||% "manual")
  )
  cat(sprintf("prepared %-28s n=%d p=%d -> %s\n", id, meta$n_obs, length(meta$variables), prepared_data_path(id)))
}
