#!/usr/bin/env Rscript

# Copy magmaan-facing paper-corpus exports from the ignored nested corpus repo.
#
# Raw ingestion, derived minimal data/models, validation, and oracle generation
# live in external/paper-corpus/. This bridge only refreshes checked-in magmaan
# fixture snapshots from exported JSON.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", "..", ".."))

export_root <- Sys.getenv("PAPER_CORPUS_MAGMAAN_EXPORTS", unset = "")
if (!nzchar(export_root)) {
  export_root <- file.path(repo_root, "external", "paper-corpus", "exports",
                           "magmaan")
}

exports <- list(
  list(name = "zxqvn_reference.json",
       corpus_id = "magmaan_paper_corpus_zxqvn_v1")
)

out_dir <- file.path(repo_root, "cpp", "tests", "fixtures", "paper_corpus")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

for (item in exports) {
  source_path <- file.path(export_root, item$name)
  if (!file.exists(source_path)) {
    stop("Missing paper-corpus export: ", source_path,
         "\nRun external/paper-corpus/scripts/export_magmaan.R first.",
         call. = FALSE)
  }
  payload <- jsonlite::fromJSON(source_path, simplifyVector = FALSE)
  if (!identical(payload$`_meta`$corpus_id, item$corpus_id)) {
    stop("Unexpected corpus id in ", source_path, call. = FALSE)
  }
  file.copy(source_path, file.path(out_dir, item$name), overwrite = TRUE)
  cat("Copied ", source_path, " to cpp/tests/fixtures/paper_corpus/", item$name,
      "\n", sep = "")
}

# The discovery batch contains aggregate inputs, never participant-level rows.
example_ids <- c(
  'boivin_hierarchical', 'boivin_two_factor', 'boivin_crossloadings',
  'boivin_equal_crossloadings', 'boivin_equal_primary_crossloading',
  'pregnancy_mediation', 'kievit_ulcs', 'kievit_milcs', 'kievit_blcs',
  'kievit_bdcs', 'kievit_mg_ulcs', 'kievit_milcs_missing',
  'jiwani_dass_1f_cfa', 'jiwani_dass_3f_cfa')
example_dir <- file.path(out_dir, 'examples')
dir.create(example_dir, showWarnings = FALSE)
for (id in example_ids) {
  source_path <- file.path(export_root, 'examples', paste0(id, '.json'))
  if (!file.exists(source_path)) stop('Missing ', source_path,
    '; run external/paper-corpus/scripts/export_examples.R first.')
  payload <- fromJSON(source_path, simplifyVector = FALSE)
  stopifnot(identical(payload$id, id),
    identical(payload$`_meta`$corpus_id, 'magmaan_paper_examples_v1'),
    isTRUE(payload$`_meta`$aggregate_only))
  stopifnot(file.copy(source_path, file.path(example_dir, paste0(id, '.json')), overwrite = TRUE))
}
cat('Copied 14 aggregate paper examples\n')
