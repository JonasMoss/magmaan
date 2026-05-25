#!/usr/bin/env Rscript

# Build the ignored corpus/textbook-corpus/raw/mplus_sem corpus from raw Mplus
# zip archives.
#
# This script is a maintainer convenience: it extracts the raw .inp examples,
# deduplicates by input-file content, classifies scope, translates the MODEL
# block to lavaan syntax, and writes a stable per-case folder. The resulting
# raw/mplus_sem tree is intentionally not tracked; checked-in tests consume
# only derived fixtures generated from it.

suppressPackageStartupMessages({
  library(lavaan)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))

zip_root <- Sys.getenv("MPLUS_SEM_ZIP_ROOT", unset = "")
if (!nzchar(zip_root)) {
  zip_root <- file.path(repo_root, "corpus", "textbook-corpus", "raw", "MPLUS")
}
zip_root <- normalizePath(zip_root, mustWork = TRUE)

out_root <- Sys.getenv("MPLUS_SEM_ROOT", unset = "")
if (!nzchar(out_root)) {
  out_root <- file.path(repo_root, "corpus", "textbook-corpus", "raw", "mplus_sem")
}

zips <- sort(Sys.glob(file.path(zip_root, "*.zip")))
if (!length(zips)) stop("No .zip archives found under ", zip_root, call. = FALSE)

unlink(out_root, recursive = TRUE, force = TRUE)
dir.create(file.path(out_root, "cases"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_root, "scripts"), recursive = TRUE, showWarnings = FALSE)

tmp_root <- tempfile("mplus-sem-raw-")
dir.create(tmp_root)
on.exit(unlink(tmp_root, recursive = TRUE, force = TRUE), add = TRUE)
for (z in zips) utils::unzip(z, exdir = tmp_root)

read_zip_text <- function(zip, name) {
  x <- readLines(unz(zip, name), warn = FALSE)
  x <- iconv(x, from = "", to = "UTF-8", sub = "byte")
  paste(x, collapse = "\n")
}

md5_text <- function(x) {
  path <- tempfile()
  on.exit(unlink(path, force = TRUE), add = TRUE)
  writeLines(x, path, useBytes = TRUE)
  unname(tools::md5sum(path))
}

strip_comments <- function(x) {
  lines <- strsplit(x, "\n", fixed = TRUE)[[1L]]
  lines <- sub("!.*$", "", lines)
  paste(lines, collapse = "\n")
}

section_map <- function(x) {
  lines <- strsplit(strip_comments(x), "\n", fixed = TRUE)[[1L]]
  sections <- list()
  current <- NULL
  for (line in lines) {
    m <- regexec("^\\s*([A-Za-z][A-Za-z0-9 _-]*)\\s*:(.*)$", line,
                 perl = TRUE)
    mm <- regmatches(line, m)[[1L]]
    if (length(mm)) {
      current <- tolower(gsub("\\s+", " ", trimws(mm[[2L]])))
      rest <- trimws(mm[[3L]])
      if (is.null(sections[[current]])) sections[[current]] <- character()
      if (nzchar(rest)) sections[[current]] <- c(sections[[current]], rest)
    } else if (!is.null(current)) {
      sections[[current]] <- c(sections[[current]], line)
    }
  }
  lapply(sections, function(v) paste(v, collapse = "\n"))
}

section_value <- function(block, key) {
  if (is.null(block) || !nzchar(block)) return("")
  pat <- paste0("(?is)\\b", key, "\\b\\s*(?:are\\s+|is\\s+)?",
                "(?:=\\s*)?([^;]+)")
  m <- regexec(pat, block, perl = TRUE)
  mm <- regmatches(block, m)[[1L]]
  if (!length(mm)) "" else trimws(gsub("\\s+", " ", mm[[2L]]))
}

expand_names <- function(x) {
  if (!nzchar(x)) return(character())
  x <- gsub("[(),]", " ", x)
  toks <- unlist(strsplit(trimws(x), "\\s+"))
  toks <- toks[nzchar(toks)]
  out <- character()
  for (tok in toks) {
    m <- regexec("^([A-Za-z_.]+)([0-9]+)-(?:(?:([A-Za-z_.]+))?([0-9]+))$",
                 tok, perl = TRUE)
    mm <- regmatches(tok, m)[[1L]]
    if (length(mm)) {
      lhs <- mm[[2L]]
      a <- as.integer(mm[[3L]])
      rhs <- if (nzchar(mm[[4L]])) mm[[4L]] else lhs
      b <- as.integer(mm[[5L]])
      if (identical(lhs, rhs) && !is.na(a) && !is.na(b)) {
        rng <- if (a <= b) seq.int(a, b) else seq.int(a, b)
        out <- c(out, paste0(lhs, rng))
        next
      }
    }
    out <- c(out, tok)
  }
  unique(out)
}

source_file_ref <- function(sections) {
  data <- sections[["data"]]
  if (is.null(data)) return("")
  ref <- section_value(data, "file")
  ref <- sub("^['\"]", "", sub("['\"]$", "", ref))
  ref
}

canonical_id <- function(zip, name) {
  chapter <- sub("[.]zip$", "", basename(zip))
  chapter <- sub("_+$", "", tolower(chapter))
  stem <- sub("[.][^.]+$", "", basename(name))
  id <- tolower(paste(chapter, stem, sep = "_"))
  id <- gsub("[^a-z0-9]+", "_", id)
  gsub("^_|_$", "", id)
}

find_extracted <- function(path_in_zip) {
  direct <- file.path(tmp_root, path_in_zip)
  if (file.exists(direct)) return(direct)
  hits <- list.files(tmp_root, pattern = paste0("^", basename(path_in_zip), "$"),
                     recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
  if (length(hits)) hits[[1L]] else ""
}

find_related <- function(source_name, ref) {
  if (!nzchar(ref)) return("")
  candidates <- c(
    file.path(dirname(file.path(tmp_root, source_name)), ref),
    file.path(tmp_root, ref)
  )
  hit <- candidates[file.exists(candidates)][1L]
  if (!is.na(hit)) return(hit)
  hits <- list.files(tmp_root, pattern = paste0("^", basename(ref), "$"),
                     recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
  if (length(hits)) hits[[1L]] else ""
}

copy_related <- function(source_name, case_dir, ext, dest) {
  stem <- sub("[.][^.]+$", "", basename(source_name))
  rel <- file.path(dirname(source_name), paste0(stem, ext))
  hit <- find_extracted(rel)
  if (nzchar(hit)) {
    file.copy(hit, file.path(case_dir, dest), overwrite = TRUE)
    return(TRUE)
  }
  FALSE
}

translate_model <- function(model_block) {
  if (is.null(model_block) || !nzchar(trimws(model_block))) return("")
  out <- tryCatch(lavaan::lav_mplus_syntax_model(model_block),
                  error = function(e) structure(conditionMessage(e),
                                                class = "mplus_translate_error"))
  if (inherits(out, "mplus_translate_error")) return("")
  out <- gsub("[ \t]+$", "", out)
  paste(Filter(nzchar, strsplit(out, "\n", fixed = TRUE)[[1L]]),
        collapse = "\n")
}

contains <- function(x, pat) grepl(pat, x, ignore.case = TRUE, perl = TRUE)

classify_exclusion <- function(raw, sections) {
  low <- tolower(raw)
  model <- sections[["model"]] %||% ""
  reasons <- character()
  if (is.null(sections[["model"]])) reasons <- c(reasons, "no_model")
  if (contains(low, "(^|\\n)\\s*montecarlo\\s*:")) reasons <- c(reasons, "monte_carlo")
  if (contains(low, "\\bclasses\\s*=|\\btype\\s*=\\s*[^;]*mixture")) reasons <- c(reasons, "mixture")
  if (contains(low, "\\btype\\s*=\\s*[^;]*(twolevel|complex)|\\bcluster\\s*=|\\bwithin\\s*=|\\bbetween\\s*=")) {
    reasons <- c(reasons, "multilevel_or_clustered")
  }
  if (contains(low, "\\btype\\s*=\\s*efa|\\befa\\b")) reasons <- c(reasons, "efa")
  if (contains(model, "\\|[^;\\n]*\\bon\\b")) reasons <- c(reasons, "random_effect")
  if (contains(low, "\\bimputation\\b")) reasons <- c(reasons, "imputation")
  if (contains(low, "\\bbayes\\b")) reasons <- c(reasons, "bayes")
  if (contains(low, "\\bnominal\\s*=")) reasons <- c(reasons, "nominal")
  if (contains(low, "\\bcount\\s*=")) reasons <- c(reasons, "count")
  if (contains(low, "\\bcensored\\s*=")) reasons <- c(reasons, "censored")
  if (contains(low, "\\bsurvival\\b|\\btimecensored\\b")) reasons <- c(reasons, "survival")
  if (contains(low, "\\btwopart\\b|\\bdata\\s+twopart\\s*:")) reasons <- c(reasons, "two_part")
  unique(reasons)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

yaml_quote <- function(x) {
  if (is.null(x) || length(x) == 0L || is.na(x)) x <- ""
  x <- gsub("\\\\", "\\\\\\\\", as.character(x))
  x <- gsub("\"", "\\\\\"", x)
  paste0("\"", x, "\"")
}

write_case_yml <- function(path, row) {
  lines <- c(
    paste("case_id:", yaml_quote(row$case_id)),
    paste("status:", yaml_quote(row$status)),
    paste("title:", yaml_quote(row$title)),
    paste("source_zip:", yaml_quote(row$source_zip)),
    paste("source_input:", yaml_quote(row$source_input)),
    paste("source_data:", yaml_quote(row$source_data)),
    paste("data_kind:", yaml_quote(row$data_kind)),
    paste("model_kind:", yaml_quote(row$model_kind)),
    paste("test_candidate:", tolower(as.character(row$test_candidate))),
    paste("snlls_candidate:", tolower(as.character(row$snlls_candidate))),
    paste("exclude_reason:", yaml_quote(row$exclude_reason)),
    paste("notes:", yaml_quote(row$note))
  )
  writeLines(lines, path, useBytes = TRUE)
}

raw_inputs <- list()
for (z in zips) {
  names <- utils::unzip(z, list = TRUE)$Name
  inps <- names[grepl("[.]inp$", names, ignore.case = TRUE)]
  for (nm in inps) {
    txt <- read_zip_text(z, nm)
    hash <- md5_text(gsub("\r\n?", "\n", txt))
    raw_inputs[[length(raw_inputs) + 1L]] <- list(zip = z, name = nm,
                                                  text = txt, hash = hash)
  }
}

seen <- character()
used_case_ids <- character()
rows <- list()
retained <- 0L
for (item in raw_inputs) {
  if (item$hash %in% seen) next
  seen <- c(seen, item$hash)

  sections <- section_map(item$text)
  vars <- expand_names(section_value(sections[["variable"]] %||% "", "names"))
  usev <- expand_names(section_value(sections[["variable"]] %||% "", "usevariables"))
  if (!length(usev)) usev <- expand_names(section_value(sections[["variable"]] %||% "", "usev"))
  if (!length(usev)) usev <- vars
  categorical <- expand_names(section_value(sections[["variable"]] %||% "", "categorical"))
  ordered_used <- intersect(categorical, usev)

  reasons <- classify_exclusion(item$text, sections)
  data_ref <- source_file_ref(sections)
  data_path <- find_related(item$name, data_ref)
  lav <- ""
  if (!length(reasons)) lav <- translate_model(sections[["model"]])
  if (!length(reasons) && !nzchar(lav)) reasons <- c(reasons, "translation_failed")

  data_kind <- "continuous"
  if (length(ordered_used) > 0L) {
    data_kind <- if (length(setdiff(usev, ordered_used)) == 0L) "ordinal" else "mixed"
  }

  has_regression <- contains(lav, "(^|\\n)\\s*[^\\n~]+\\s+~\\s+[^~]")
  has_measurement <- contains(lav, "=~")
  model_kind <- if (contains(sections[["model"]] %||% "", "\\|")) {
    "growth"
  } else if (has_measurement && has_regression) {
    "latent_sem"
  } else if (has_measurement) {
    "cfa"
  } else if (has_regression) {
    "observed_path"
  } else {
    "other"
  }

  has_define <- !is.null(sections[["define"]]) && nzchar(trimws(sections[["define"]]))
  has_weight <- contains(sections[["variable"]] %||% "", "\\b(freqweight|weight)\\b")
  has_useobs <- contains(sections[["variable"]] %||% "", "\\buseobservations\\b|\\buseobs\\b")
  test_candidate <- !length(reasons) && identical(data_kind, "continuous") &&
    nzchar(data_path) && !has_define && !has_weight && !has_useobs
  snlls_candidate <- test_candidate && model_kind %in% c("cfa", "latent_sem", "growth")

  status <- if (length(reasons)) "excluded" else "retained"
  case_id <- canonical_id(item$zip, item$name)
  if (case_id %in% used_case_ids) {
    base_id <- case_id
    suffix <- 2L
    while (paste0(base_id, "_", suffix) %in% used_case_ids) suffix <- suffix + 1L
    case_id <- paste0(base_id, "_", suffix)
  }
  used_case_ids <- c(used_case_ids, case_id)
  title <- trimws(strsplit(sections[["title"]] %||% "", "\n", fixed = TRUE)[[1L]][1L] %||% "")

  case_dir <- ""
  if (identical(status, "retained")) {
    retained <- retained + 1L
    case_dir <- file.path("cases", case_id)
    abs_case_dir <- file.path(out_root, case_dir)
    dir.create(abs_case_dir, recursive = TRUE, showWarnings = FALSE)
    writeLines(item$text, file.path(abs_case_dir, "source.inp"), useBytes = TRUE)
    writeLines(lav, file.path(abs_case_dir, "model.lav"), useBytes = TRUE)
    if (nzchar(data_path)) file.copy(data_path, file.path(abs_case_dir, "data.dat"), overwrite = TRUE)
    copy_related(item$name, abs_case_dir, ".out", "source.out")
  }

  row <- data.frame(
    case_id = case_id,
    title = title,
    status = status,
    source_zip = basename(item$zip),
    source_input = item$name,
    source_data = data_ref,
    case_dir = case_dir,
    has_data = nzchar(data_path),
    data_kind = data_kind,
    model_kind = model_kind,
    snlls_candidate = snlls_candidate,
    test_candidate = test_candidate,
    has_define = has_define,
    has_constraints = any(!vapply(c("model indirect", "model constraint", "model test"),
                                  function(k) is.null(sections[[k]]), logical(1L))),
    exclude_reason = if (length(reasons)) paste(reasons, collapse = ";") else "",
    note = if (length(reasons)) "excluded by automated first-pass screen"
           else "retained first-pass lavaan translation",
    stringsAsFactors = FALSE
  )
  rows[[length(rows) + 1L]] <- row
  if (identical(status, "retained")) {
    write_case_yml(file.path(out_root, case_dir, "case.yml"), row)
  }
}

manifest <- do.call(rbind, rows)
manifest <- manifest[order(manifest$status, manifest$case_id), ]
utils::write.csv(manifest, file.path(out_root, "manifest.csv"), row.names = FALSE)

# A retained case is only a usable catalogue entry when its referenced data
# file was actually found and copied; otherwise it lives in the manifest only.
catalogue <- manifest[manifest$status == "retained" & manifest$has_data,
                      , drop = FALSE]
if (nrow(catalogue)) {
  catalogue <- transform(
    catalogue,
    id = case_id,
    name = ifelse(nzchar(title), title, case_id),
    family = model_kind,
    provenance = "Mplus User's Guide zip examples",
    generated_model = file.path(case_dir, "model.lav"),
    generated_data = file.path(case_dir, "data.dat"),
    generated_script = "",
    data_kind = ifelse(data_kind == "continuous", "raw", data_kind)
  )
  keep <- c("id", "name", "family", "provenance", "source_input",
            "source_data", "data_kind", "generated_data", "generated_model",
            "generated_script", "status", "note", "test_candidate",
            "snlls_candidate", "model_kind")
  utils::write.csv(catalogue[, keep], file.path(out_root, "catalogue.csv"),
                   row.names = FALSE)
}

readme <- c(
  "# Mplus SEM corpus",
  "",
  "Ignored local corpus built from raw `corpus/textbook-corpus/raw/MPLUS/*.zip` archives.",
  "The checked-in test suite does not commit Mplus raw data; it commits only",
  "derived sample statistics and lavaan oracle outputs under",
  "`tests/fixtures/mplus_sem/`.",
  "",
  "## Inclusion screen",
  "",
  "The first pass keeps translated lavaan SEM/path/growth examples and excludes",
  "Monte Carlo, mixture, multilevel/clustered, EFA, random-effect, imputation,",
  "Bayes, nominal, count, censored, survival, and two-part models. `MODEL",
  "INDIRECT`, `MODEL CONSTRAINT`, and `MODEL TEST` sections are not translated",
  "for v1 fixtures.",
  "",
  "## Files",
  "",
  "- `manifest.csv`: every unique `.inp`, retained or excluded.",
  "- `catalogue.csv`: retained cases in the paper-corpus loader shape.",
  "- `cases/<case_id>/source.inp`: original Mplus input.",
  "- `cases/<case_id>/model.lav`: translated lavaan model block.",
  "- `cases/<case_id>/data.dat`: local raw data when the source input needs it.",
  "- `cases/<case_id>/source.out`: source output when present.",
  "- `cases/<case_id>/case.yml`: per-case classification metadata."
)
writeLines(readme, file.path(out_root, "README.md"), useBytes = TRUE)

script_copy <- file.path(out_root, "scripts", "build_corpus.R")
invisible(file.copy(normalizePath(sub("^--file=", "", script_arg)),
                    script_copy, overwrite = TRUE))

cat("Mplus SEM corpus built at ", out_root, "\n", sep = "")
cat("raw .inp files: ", length(raw_inputs), "\n", sep = "")
cat("unique .inp files: ", length(seen), "\n", sep = "")
cat("retained: ", sum(manifest$status == "retained"), "\n", sep = "")
cat("excluded: ", sum(manifest$status == "excluded"), "\n", sep = "")
cat("test candidates: ", sum(manifest$test_candidate), "\n", sep = "")
cat("SNLLS candidates: ", sum(manifest$snlls_candidate), "\n", sep = "")
print(table(manifest$data_kind, manifest$status))
print(table(manifest$model_kind, manifest$status))
