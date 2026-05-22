#!/usr/bin/env Rscript

# Regenerate the consolidated textbook-corpus v1 manifest.
#
# This is an index layer over source-specific corpora; it deliberately does not
# duplicate the heavy oracle payloads from Geiser, Mplus SEM, Little, or Newsom.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))
fixtures <- file.path(repo_root, "tests", "fixtures")
out_dir <- file.path(fixtures, "textbook_corpus")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

`%||%` <- function(x, y) if (is.null(x)) y else x

read_fixture <- function(...) {
  jsonlite::fromJSON(file.path(fixtures, ...), simplifyVector = FALSE)
}

boolish <- function(x) {
  if (is.null(x) || length(x) != 1L || is.na(x)) return(FALSE)
  if (is.logical(x)) return(isTRUE(x))
  toupper(trimws(as.character(x))) %in% c("TRUE", "T", "1", "YES")
}

has_latent <- function(model) {
  is.character(model) && length(model) == 1L && grepl("=~", model, fixed = TRUE)
}

case_row <- function(source, id, label, family, data_kind, measurement_kind,
                     observed_only, status, strict_parity, oracle_files,
                     source_input = "", note = "") {
  list(
    id = paste(source, id, sep = "::"),
    source = source,
    source_case_id = id,
    label = label %||% "",
    family = family %||% "",
    data_kind = data_kind %||% "",
    measurement_kind = measurement_kind %||% data_kind %||% "",
    observed_only = isTRUE(observed_only),
    status = status %||% "",
    strict_parity = isTRUE(strict_parity),
    oracle_files = oracle_files,
    source_input = source_input %||% "",
    note = note %||% ""
  )
}

geiser_cases <- function() {
  gls <- read_fixture("geiser", "gls_reference.json")
  ids <- vapply(gls$cases, function(x) x$id, character(1L))
  lapply(gls$cases, function(c) {
    id <- c$id
    known_smoke_only <- grepl("^manifest_", id) ||
      grepl("^latent_ar_cross_lagged$", id)
    case_row(
      source = "geiser",
      id = id,
      label = c$label,
      family = c$family,
      data_kind = c$data_kind,
      measurement_kind = "continuous",
      observed_only = !has_latent(c$model),
      status = "retained",
      strict_parity = !known_smoke_only,
      oracle_files = c("geiser/gls_reference.json", "geiser/uls_reference.json"),
      source_input = "",
      note = if (known_smoke_only) {
        "smoke-checked by source golden; not strict implied-moment parity"
      } else {
        "strict Geiser source golden"
      }
    )
  })
}

mplus_cases <- function() {
  manifest <- read_fixture("mplus_sem", "manifest.json")
  cont <- read_fixture("mplus_sem", "continuous_reference.json")
  strict_ids <- vapply(cont$cases, function(x) x$id, character(1L))
  lapply(manifest$cases, function(c) {
    id <- c$case_id
    case_row(
      source = "mplus_sem",
      id = id,
      label = c$title,
      family = c$model_kind,
      data_kind = c$data_kind,
      measurement_kind = c$data_kind,
      observed_only = identical(c$model_kind, "observed_path"),
      status = c$status,
      strict_parity = id %in% strict_ids,
      oracle_files = c("mplus_sem/manifest.json",
                       if (id %in% strict_ids) "mplus_sem/continuous_reference.json" else NULL),
      source_input = c$source_input,
      note = c$note
    )
  })
}

manifest_cases <- function(source) {
  manifest <- read_fixture(source, "manifest.json")
  lapply(manifest$cases, function(c) {
    kind <- c$measurement_kind %||% c$data_kind %||% ""
    case_row(
      source = source,
      id = c$id,
      label = c$name,
      family = c$family,
      data_kind = c$data_kind,
      measurement_kind = kind,
      observed_only = boolish(c$observed_only),
      status = c$status,
      strict_parity = boolish(c$strict_parity),
      oracle_files = c(file.path(source, "manifest.json"),
                       if (boolish(c$strict_parity) && identical(kind, "continuous")) {
                         file.path(source, "continuous_reference.json")
                       } else NULL),
      source_input = c$source_input,
      note = c$note
    )
  })
}

cases <- c(geiser_cases(), mplus_cases(), manifest_cases("little"),
           manifest_cases("newsom"))

count <- function(pred) sum(vapply(cases, pred, logical(1L)))
source_summary <- function(source, label, fixture_files, source_root) {
  src_cases <- cases[vapply(cases, function(x) identical(x$source, source),
                            logical(1L))]
  list(
    id = source,
    label = label,
    source_root = source_root,
    fixture_files = fixture_files,
    counts = list(
      total = length(src_cases),
      retained = sum(vapply(src_cases, function(x) identical(x$status, "retained"),
                            logical(1L))),
      strict_parity = sum(vapply(src_cases, function(x) isTRUE(x$strict_parity),
                                 logical(1L))),
      continuous = sum(vapply(src_cases, function(x) identical(x$measurement_kind, "continuous"),
                              logical(1L))),
      ordinal = sum(vapply(src_cases, function(x) identical(x$measurement_kind, "ordinal"),
                           logical(1L))),
      mixed = sum(vapply(src_cases, function(x) identical(x$measurement_kind, "mixed"),
                         logical(1L))),
      observed_only = sum(vapply(src_cases, function(x) isTRUE(x$observed_only),
                                 logical(1L)))
    )
  )
}

version_sources <- list(
  read_fixture("geiser", "gls_reference.json"),
  read_fixture("mplus_sem", "manifest.json"),
  read_fixture("little", "manifest.json"),
  read_fixture("newsom", "manifest.json")
)
versions <- unique(unlist(lapply(version_sources, function(x) {
  x[["_meta"]][["lavaan_version"]]
}), use.names = FALSE))

payload <- list(
  `_meta` = list(
    format_version = 1L,
    fixture_kind = "textbook_corpus.manifest",
    corpus_id = "magmaan_textbook_corpus_v1",
    tool = "tests/tools/regen_textbook_corpus_manifest.R",
    generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    lavaan_versions = versions
  ),
  counts = list(
    sources = 4L,
    total = length(cases),
    retained = count(function(x) identical(x$status, "retained")),
    strict_parity = count(function(x) isTRUE(x$strict_parity)),
    continuous = count(function(x) identical(x$measurement_kind, "continuous")),
    ordinal = count(function(x) identical(x$measurement_kind, "ordinal")),
    mixed = count(function(x) identical(x$measurement_kind, "mixed")),
    observed_only = count(function(x) isTRUE(x$observed_only))
  ),
  sources = list(
    source_summary("geiser", "Geiser textbook corpus",
                   c("geiser/gls_reference.json", "geiser/uls_reference.json"),
                   "external/geiser"),
    source_summary("mplus_sem", "Mplus SEM textbook corpus",
                   c("mplus_sem/manifest.json",
                     "mplus_sem/continuous_reference.json",
                     "mplus_sem/ordinal_reference.json",
                     "mplus_sem/mixed_reference.json"),
                   "external/mplus_sem"),
    source_summary("little", "Little LISREL textbook corpus",
                   c("little/manifest.json",
                     "little/continuous_reference.json",
                     "little/ordinal_reference.json",
                     "little/mixed_reference.json",
                     "little/observed_reference.json"),
                   "external/little"),
    source_summary("newsom", "Newsom longitudinal SEM textbook corpus",
                   c("newsom/manifest.json",
                     "newsom/continuous_reference.json",
                     "newsom/ordinal_reference.json",
                     "newsom/mixed_reference.json",
                     "newsom/observed_reference.json"),
                   "external/newsom")
  ),
  cases = cases
)

jsonlite::write_json(payload, file.path(out_dir, "manifest.json"),
                     pretty = TRUE, auto_unbox = TRUE, null = "null",
                     na = "null", digits = NA)
cat("Wrote ", file.path(out_dir, "manifest.json"), "\n", sep = "")
