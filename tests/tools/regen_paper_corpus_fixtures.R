#!/usr/bin/env Rscript

# Copy magmaan-facing paper-corpus exports from the ignored nested corpus repo.
#
# Raw ingestion, derived minimal data/models, validation, and oracle generation
# live in external/paper_corpus/. This bridge only refreshes checked-in magmaan
# fixture snapshots from exported JSON.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))

export_root <- Sys.getenv("PAPER_CORPUS_MAGMAAN_EXPORTS", unset = "")
if (!nzchar(export_root)) {
  export_root <- file.path(repo_root, "external", "paper_corpus", "exports",
                           "magmaan")
}

exports <- list(
  list(name = "zxqvn_reference.json",
       corpus_id = "magmaan_paper_corpus_zxqvn_v1")
)

out_dir <- file.path(repo_root, "tests", "fixtures", "paper_corpus")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

for (item in exports) {
  source_path <- file.path(export_root, item$name)
  if (!file.exists(source_path)) {
    stop("Missing paper-corpus export: ", source_path,
         "\nRun external/paper_corpus/scripts/export_magmaan.R first.",
         call. = FALSE)
  }
  payload <- jsonlite::fromJSON(source_path, simplifyVector = FALSE)
  if (!identical(payload$`_meta`$corpus_id, item$corpus_id)) {
    stop("Unexpected corpus id in ", source_path, call. = FALSE)
  }
  file.copy(source_path, file.path(out_dir, item$name), overwrite = TRUE)
  cat("Copied ", source_path, " to tests/fixtures/paper_corpus/", item$name,
      "\n", sep = "")
}
