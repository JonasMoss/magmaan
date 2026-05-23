#!/usr/bin/env Rscript

# Copy the paper-corpus scout export from the ignored nested corpus repository.
#
# The actual OSF scout lives in external/paper_corpus/scripts/scout_osf.R.
# magmaan keeps only the exported JSON snapshot used by C++ tests.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))

source_path <- Sys.getenv("PAPER_CORPUS_SCOUT_EXPORT", unset = "")
if (!nzchar(source_path)) {
  source_path <- file.path(repo_root, "external", "paper_corpus", "exports",
                           "magmaan", "scout_manifest.json")
}
if (!file.exists(source_path)) {
  stop("Missing paper-corpus scout export: ", source_path,
       "\nRun external/paper_corpus/scripts/scout_osf.R first.",
       call. = FALSE)
}

payload <- jsonlite::fromJSON(source_path, simplifyVector = FALSE)
if (!identical(payload$`_meta`$corpus_id, "magmaan_paper_corpus_scout_v1")) {
  stop("Unexpected paper-corpus scout export: ", source_path, call. = FALSE)
}

out_dir <- file.path(repo_root, "tests", "fixtures", "paper_corpus")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
invisible(file.copy(source_path, file.path(out_dir, "scout_manifest.json"),
                    overwrite = TRUE))
cat("Copied ", source_path, " to tests/fixtures/paper_corpus/scout_manifest.json\n",
    sep = "")
