#!/usr/bin/env Rscript

# Regenerate compact, checked-in textbook-corpus case fixtures from the
# textbook-corpus submodule. This exports only curated cases that magmaan's
# C++ tests consume directly; the full corpus remains in the submodule.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))

corpus_root <- file.path(repo_root, "corpus", "textbook-corpus")
out_dir <- file.path(repo_root, "tests", "fixtures", "textbook_corpus")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

kline_guo_ids <- c(
  "kline_2023_ch22_guo_mi_configural",
  "kline_2023_ch22_guo_mi_weak",
  "kline_2023_ch22_guo_mi_strong",
  "kline_2023_ch22_guo_mi_partial_strong"
)

read_json <- function(path) {
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

read_text <- function(path) {
  paste(readLines(path, warn = FALSE), collapse = "\n")
}

read_matrix_csv <- function(path) {
  x <- utils::read.csv(path, row.names = 1L, check.names = FALSE)
  unname(as.matrix(x))
}

read_vector_csv <- function(path) {
  x <- utils::read.csv(path, row.names = 1L, check.names = FALSE)
  unname(as.numeric(x[[1L]]))
}

read_case <- function(book, id) {
  case_dir <- file.path(corpus_root, "cases", book, id)
  meta <- read_json(file.path(case_dir, "meta.json"))
  lav <- read_json(file.path(case_dir, "expected", "lavaan_ml.json"))
  cov_files <- unlist(meta$data$files$sample_cov)
  mean_files <- unlist(meta$data$files$sample_mean)
  list(
    case_id = meta$case_id,
    book = meta$book,
    label = meta$label,
    tags = meta$tags,
    estimator = "ML",
    lavaan_status = lav$lavaan_status,
    lavaan_version = lav$lavaan_version,
    lavaan_function = meta$lavaan_function,
    model_options = meta$model_options,
    model = read_text(file.path(case_dir, "model.lav")),
    data = list(
      kind = meta$data$kind,
      n_groups = meta$data$n_groups,
      n_obs = meta$data$n_obs,
      sample_cov = lapply(cov_files, function(p) {
        read_matrix_csv(file.path(case_dir, p))
      }),
      sample_mean = lapply(mean_files, function(p) {
        read_vector_csv(file.path(case_dir, p))
      })
    ),
    lavaan = list(
      fit = lav$fit[c("fmin", "chisq", "df", "npar")],
      theta = lav$theta,
      implied = lav$implied
    )
  )
}

cases <- lapply(kline_guo_ids, function(id) read_case("kline_2023", id))

payload <- list(
  `_meta` = list(
    format_version = 1L,
    fixture_kind = "textbook_corpus.case_export",
    corpus_id = "magmaan_textbook_corpus_v1",
    tool = "tests/tools/regen_textbook_case_fixtures.R",
    generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    source_root = "corpus/textbook-corpus",
    selection = "kline_guo_measurement_invariance_ml"
  ),
  counts = list(
    cases = length(cases),
    estimators = list(ML = length(cases)),
    fiml = 0L,
    wlsmv = 0L
  ),
  cases = cases,
  gaps = list(
    fiml = paste(
      "No canonical textbook-corpus cases currently export FIML snapshots;",
      "add raw continuous missing-data cases with expected/lavaan_fiml.json",
      "before enabling a hard FIML corpus lane."
    ),
    ordinal_wlsmv = paste(
      "Canonical Newsom WLSMV snapshots exist, but this export keeps them out",
      "of the checked C++ lane until raw ordered/missing-policy handling is",
      "classified case-by-case."
    )
  )
)

jsonlite::write_json(payload,
                     file.path(out_dir, "case_exports.json"),
                     pretty = TRUE, auto_unbox = TRUE, null = "null",
                     na = "null", digits = NA)
cat("Wrote ", file.path(out_dir, "case_exports.json"), "\n", sep = "")
