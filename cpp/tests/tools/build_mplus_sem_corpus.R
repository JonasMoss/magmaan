#!/usr/bin/env Rscript

# Build the ignored external/textbook-corpus/raw/mplus_sem corpus from raw Mplus
# zip archives (Mplus User's Guide chapters and Muthén, Muthén & Asparouhov
# 2017 chapters mixed in external/textbook-corpus/raw/MPLUS).
#
# Translation, data preparation and verification come from the textbook
# corpus's shared Mplus translator, external/textbook-corpus/ingest/
# _mplus_helpers.R, so this bundle and the corpus's mplus_users_guide_v8 /
# muthen_2017 books cannot drift apart. A case is retained only when
#   * it is expressible as a linear normal-theory lavaan model (no mixtures,
#     multilevel/complex data, categorical or count outcomes, ESEM rotation,
#     random slopes, Bayes, imputation, data-dependent constraints, ...), and
#   * lavaan's fit of the translation reproduces the shipped Mplus .out:
#     analysed N, free parameters, df, chi-square, H0 log-likelihood and every
#     printed MODEL RESULTS estimate.
# Everything else is kept in manifest.csv with its exclusion reason.
#
# Each retained case directory holds the Mplus source (.inp/.out), the lavaan
# translation (model.lav), the analysed data exactly as Mplus analysed it
# (data.csv after MISSING, USEOBSERVATIONS, DEFINE and case exclusion; or
# sample_cov.csv/sample_mean.csv for summary-data inputs) and case.yml with
# the lavaan options. The raw/mplus_sem tree is intentionally not tracked;
# checked-in tests consume only derived fixtures generated from it by
# regen_mplus_sem_fixtures.R.

suppressPackageStartupMessages({
  library(lavaan)
  library(digest)
  library(jsonlite)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", "..", ".."))

corpus_root <- file.path(repo_root, "external", "textbook-corpus")
helper <- file.path(corpus_root, "ingest", "_mplus_helpers.R")
if (!file.exists(helper)) {
  stop("The Mplus translator is part of the optional textbook corpus; mount it at ",
       corpus_root, " (see project/reference/textbook-corpus.md).", call. = FALSE)
}
source(helper)

zip_root <- Sys.getenv("MPLUS_SEM_ZIP_ROOT", unset = "")
if (!nzchar(zip_root)) zip_root <- file.path(corpus_root, "raw", "MPLUS")
zip_root <- normalizePath(zip_root, mustWork = TRUE)

out_root <- Sys.getenv("MPLUS_SEM_ROOT", unset = "")
if (!nzchar(out_root)) out_root <- file.path(corpus_root, "raw", "mplus_sem")

# Documented per-case translation overrides, pinned to the md5 of the
# newline-normalised .inp. Only MODEL CONSTRAINT blocks whose NEW parameters
# are free parameters of the Mplus model need one; the constraint is
# inverted algebraically into the same parameter space. Mirrors the
# corpus's ingest/build_mplus_users_guide.R table.
OVERRIDES <- list(
  "ex6.17.inp" = list(
    md5 = "b219964157f3a80a95e330dc968e5bf9",
    why = paste("the NEW autocorrelation parameter scales the lag-1, lag-2 and lag-3",
                "residual covariances by its first to third power; expressing it as",
                "p1/resvar removes the free NEW parameter without changing the parameter space."),
    constraint_lines = c("p2 == p1^2/resvar", "p3 == p1^3/resvar^2", "corr := p1/resvar"))
)

# Documented differences a retained case may show against the Mplus .out
# (none in this bundle).
ACCEPTED <- list()

# Corpus books that carry the same example, for overlap bookkeeping.
CORPUS_BOOKS <- c(
  "chapter4.zip" = "mplus_users_guide_v8", "chapter6.zip" = "mplus_users_guide_v8",
  "chapter7.zip" = "mplus_users_guide_v8", "chapter8.zip" = "mplus_users_guide_v8",
  "chapter9.zip" = "mplus_users_guide_v8", "chapter10.zip" = "mplus_users_guide_v8",
  "chapter11.zip" = "mplus_users_guide_v8",
  "chapter2.zip" = "muthen_2017", "chapter2_.zip" = "muthen_2017",
  "chapter3.zip" = "muthen_2017", "chapter3_.zip" = "muthen_2017",
  "chapter5.zip" = "muthen_2017", "chapter8_.zip" = "muthen_2017",
  "chapter10_.zip" = "muthen_2017")

book_of <- function(zip) {
  b <- unname(CORPUS_BOOKS[basename(zip)])
  if (is.na(b)) "" else b
}

canonical_id <- function(zip, name) {
  chapter <- sub("[.]zip$", "", basename(zip))
  chapter <- sub("_+$", "", tolower(chapter))
  stem <- sub("[.][^.]+$", "", basename(name))
  id <- tolower(paste(chapter, stem, sep = "_"))
  id <- gsub("[^a-z0-9]+", "_", id)
  gsub("^_|_$", "", id)
}

corpus_case_id <- function(zip, name) {
  book <- book_of(zip)
  if (!nzchar(book)) return("")
  id <- default_case_id(book, zip, name)
  if (dir.exists(file.path(corpus_root, "cases", book, id))) id else ""
}

measurement_kind <- function(inp) {
  usev <- tryCatch(analysis_variables(inp), error = function(e) character())
  cat_vars <- expand_varlist(inp$variable$categorical %||% "", inp$names)
  used <- intersect(cat_vars, usev)
  if (!length(used)) "continuous" else if (length(setdiff(usev, used))) "mixed" else "ordinal"
}

model_kind_of <- function(res) {
  r <- res$built$roles
  pt <- res$built$pt
  if (length(r$growth_factors)) return("growth")
  has_meas <- any(pt$op == "=~")
  has_reg <- any(pt$op == "~")
  if (has_meas && has_reg) "latent_sem" else if (has_meas) "cfa" else
    if (has_reg) "observed_path" else "other"
}

yaml_quote <- function(x) {
  if (is.null(x) || length(x) == 0L || is.na(x)) x <- ""
  x <- gsub("\\\\", "\\\\\\\\", as.character(x))
  x <- gsub("\"", "\\\\\"", x)
  paste0("\"", x, "\"")
}

write_case_yml <- function(path, row) {
  keys <- c("case_id", "status", "title", "source_zip", "source_input", "source_data",
            "data_kind", "model_kind", "estimator", "meanstructure", "fixed_x", "missing",
            "n_groups", "group_var", "n_obs", "test_candidate", "snlls_candidate",
            "verified", "verification", "corpus_case_id", "note")
  lines <- vapply(keys, function(k) {
    v <- row[[k]]
    if (is.logical(v)) paste0(k, ": ", tolower(as.character(v)))
    else if (is.numeric(v)) paste0(k, ": ", v)
    else paste0(k, ": ", yaml_quote(v))
  }, character(1L))
  writeLines(lines, path, useBytes = TRUE)
}

unlink(out_root, recursive = TRUE, force = TRUE)
dir.create(file.path(out_root, "cases"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_root, "scripts"), recursive = TRUE, showWarnings = FALSE)

items <- discover_inputs(zip_root)
seen <- character()
used_case_ids <- character()
rows <- list()
for (it in items) {
  if (it$hash %in% seen) next
  seen <- c(seen, it$hash)
  inp <- mplus_parse_input(it$text)
  case_id <- canonical_id(it$zip, it$name)
  if (case_id %in% used_case_ids) {
    base_id <- case_id; suffix <- 2L
    while (paste0(base_id, "_", suffix) %in% used_case_ids) suffix <- suffix + 1L
    case_id <- paste0(base_id, "_", suffix)
  }
  used_case_ids <- c(used_case_ids, case_id)
  # Our own label, not the Mplus TITLE text (tracked fixtures carry no input text).
  src_book <- book_of(it$zip)
  title <- paste(if (identical(src_book, "muthen_2017")) "Muthen et al. (2017)" else
                   "Mplus User's Guide", sub("[.]inp$", "", basename(it$name), ignore.case = TRUE))
  dat <- mplus_find_data(it, inp, zip_root)
  on <- companion_in_zip(it$zip, it$name, ".out")
  out_txt <- if (nzchar(on)) read_zip_text(it$zip, on) else ""
  ov <- mplus_lookup(OVERRIDES, it)
  acc <- mplus_lookup(ACCEPTED, it)
  res <- tryCatch(mplus_translate_case(it$text, dat$text, out_txt, overrides = ov),
                  error = function(e) list(status = "excluded",
                                           reasons = paste0("translator_error: ", conditionMessage(e))))
  row <- list(case_id = case_id, title = title, status = "excluded",
              source_zip = basename(it$zip), source_input = it$name,
              source_data = sub("^['\"]|['\"]$", "", inp$data$file %||% ""),
              case_dir = "", has_data = nzchar(dat$text),
              data_kind = measurement_kind(inp), model_kind = "", estimator = "",
              meanstructure = NA, fixed_x = NA, missing = "", n_groups = NA_integer_,
              group_var = "", n_obs = NA_integer_, has_define = nzchar(trimws(inp$define)),
              has_constraints = FALSE, verified = FALSE, verification = "",
              snlls_candidate = FALSE, test_candidate = FALSE,
              corpus_case_id = corpus_case_id(it$zip, it$name),
              exclude_reason = "", note = "")
  if (res$status != "translated") {
    row$exclude_reason <- paste(res$reasons, collapse = ";")
  } else if (is.null(res$cmp) || is.null(res$cmp$summary)) {
    row$exclude_reason <- if (!nzchar(out_txt)) "unverifiable: no Mplus .out" else
      paste0("unverifiable: ", res$cmp$message %||% "no MODEL RESULTS")
  } else {
    failed <- names(res$cmp$checks)[!res$cmp$checks]
    accepted <- length(failed) && !is.null(acc) && all(failed %in% acc$allow)
    row$model_kind <- model_kind_of(res)
    row$verification <- mplus_verification_text(res)
    if (length(failed) && !accepted) {
      row$exclude_reason <- paste0("unverified: ", paste(failed, collapse = ","))
    } else {
      row$status <- "retained"
      row$verified <- !length(failed)
      row$estimator <- res$opts$estimator
      row$meanstructure <- isTRUE(res$opts$meanstructure)
      row$fixed_x <- TRUE
      row$missing <- res$prep$missing %||% "none"
      row$n_groups <- length(res$built$groups)
      row$group_var <- res$prep$group_var %||% ""
      row$n_obs <- if (is.null(res$prep$data)) as.integer(res$prep$n) else nrow(res$prep$data)
      row$has_constraints <- length(res$constraint_lines) > 0L
      row$note <- paste(c(res$notes, if (accepted) paste0("Accepted difference: ", acc$why)),
                        collapse = " | ")
      row$test_candidate <- identical(row$missing, "none") && row$n_groups == 1L &&
        !row$has_constraints && res$opts$estimator %in% c("ML", "MLR")
      row$snlls_candidate <- row$test_candidate && row$model_kind %in% c("cfa", "latent_sem", "growth")
      row$case_dir <- file.path("cases", case_id)
      cdir <- file.path(out_root, row$case_dir)
      dir.create(cdir, recursive = TRUE, showWarnings = FALSE)
      writeLines(gsub("\r\n?", "\n", it$text), file.path(cdir, "source.inp"), useBytes = TRUE)
      if (nzchar(out_txt)) writeLines(gsub("\r\n?", "\n", out_txt), file.path(cdir, "source.out"),
                                      useBytes = TRUE)
      writeLines(res$syntax, file.path(cdir, "model.lav"), useBytes = TRUE)
      if (!is.null(res$prep$data)) {
        utils::write.csv(res$prep$data, file.path(cdir, "data.csv"), row.names = FALSE)
      } else {
        utils::write.csv(res$prep$cov, file.path(cdir, "sample_cov.csv"))
        if (!is.null(res$prep$mean))
          utils::write.csv(data.frame(mean = res$prep$mean, row.names = names(res$prep$mean)),
                           file.path(cdir, "sample_mean.csv"))
      }
      write_case_yml(file.path(cdir, "case.yml"), row)
    }
  }
  if (!nzchar(row$note))
    row$note <- if (identical(row$status, "retained")) "verified lavaan translation" else
      "excluded by the source-fidelity screen"
  rows[[length(rows) + 1L]] <- as.data.frame(row, stringsAsFactors = FALSE)
}

manifest <- do.call(rbind, rows)
manifest <- manifest[order(manifest$status, manifest$case_id), ]
utils::write.csv(manifest, file.path(out_root, "manifest.csv"), row.names = FALSE)

catalogue <- manifest[manifest$status == "retained", , drop = FALSE]
if (nrow(catalogue)) {
  catalogue <- transform(
    catalogue,
    id = case_id,
    name = ifelse(nzchar(title), title, case_id),
    family = model_kind,
    provenance = "Mplus User's Guide / Muthen et al. (2017) zip examples",
    generated_model = file.path(case_dir, "model.lav"),
    generated_data = ifelse(file.exists(file.path(out_root, case_dir, "data.csv")),
                            file.path(case_dir, "data.csv"), file.path(case_dir, "sample_cov.csv")),
    generated_script = "",
    data_kind = ifelse(data_kind == "continuous", "raw", data_kind)
  )
  keep <- c("id", "name", "family", "provenance", "source_input",
            "source_data", "data_kind", "generated_data", "generated_model",
            "generated_script", "status", "note", "test_candidate",
            "snlls_candidate", "model_kind")
  utils::write.csv(catalogue[, keep], file.path(out_root, "catalogue.csv"), row.names = FALSE)
}

readme <- c(
  "# Mplus SEM corpus",
  "",
  "Ignored local corpus built from raw `external/textbook-corpus/raw/MPLUS/*.zip` archives",
  "by `cpp/tests/tools/build_mplus_sem_corpus.R`, using the textbook corpus's shared",
  "translator `external/textbook-corpus/ingest/_mplus_helpers.R`. The checked-in test",
  "suite commits only derived sample statistics and lavaan oracle outputs under",
  "`cpp/tests/fixtures/mplus_sem/`.",
  "",
  "## Retention",
  "",
  "A case is retained only if it is expressible as a linear normal-theory lavaan model",
  "and lavaan's fit of the translation reproduces the shipped Mplus `.out` (N, free",
  "parameters, df, chi-square, H0 log-likelihood, every printed estimate). Exclusion",
  "reasons are listed in `manifest.csv`.",
  "",
  "## Files",
  "",
  "- `manifest.csv`: every unique `.inp`, retained or excluded, with the verification line.",
  "- `catalogue.csv`: retained cases in the paper-corpus loader shape.",
  "- `cases/<case_id>/source.inp`, `source.out`: original Mplus input and output.",
  "- `cases/<case_id>/model.lav`: lavaan translation (fit with `lavaan::sem`).",
  "- `cases/<case_id>/data.csv`: Mplus's analysis sample (or `sample_cov.csv`/`sample_mean.csv`).",
  "- `cases/<case_id>/case.yml`: lavaan options and classification metadata."
)
writeLines(readme, file.path(out_root, "README.md"), useBytes = TRUE)
invisible(file.copy(normalizePath(sub("^--file=", "", script_arg)),
                    file.path(out_root, "scripts", "build_corpus.R"), overwrite = TRUE))

cat("Mplus SEM corpus built at ", out_root, "\n", sep = "")
cat("unique .inp files: ", nrow(manifest), "\n", sep = "")
cat("retained: ", sum(manifest$status == "retained"), "\n", sep = "")
cat("excluded: ", sum(manifest$status == "excluded"), "\n", sep = "")
cat("test candidates: ", sum(manifest$test_candidate), "\n", sep = "")
cat("SNLLS candidates: ", sum(manifest$snlls_candidate), "\n", sep = "")
print(table(manifest$model_kind[manifest$status == "retained"]))
