#!/usr/bin/env Rscript
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "common.R"))
source(file.path(bench_script_dir(), "cases.R"))

required <- c("lavaan", "jsonlite", "bench", "callr", "haven")
optional <- c("psychTools", "semTools", "OpenMx")

cat("benchmark root:", bench_root(), "\n\n")
cat("R packages\n")
for (pkg in c(required, optional)) {
  installed <- requireNamespace(pkg, quietly = TRUE)
  marker <- if (pkg %in% required) "required" else "optional"
  version <- if (installed) as.character(utils::packageVersion(pkg)) else "-"
  cat(sprintf("  %-10s %-8s installed=%-5s version=%s\n", pkg, marker, installed, version))
}

cat("\nCases\n")
for (id in case_ids()) {
  case <- get_case(id)
  prepared <- file.exists(prepared_data_path(id))
  reference <- file.exists(reference_path(id))
  raw <- if (!is.null(case$raw_file)) file.exists(file.path(raw_dir(id), case$raw_file)) else NA
  cat(sprintf(
    "  %-28s tier=%s status=%-9s source=%-16s prepared=%-5s raw=%-5s lavaan_ref=%-5s\n",
    id, case$tier %||% "-", case$status, case$source_type, prepared, raw, reference
  ))
}
