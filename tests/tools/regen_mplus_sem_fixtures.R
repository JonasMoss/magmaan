#!/usr/bin/env Rscript

# Regenerate Mplus SEM corpus parity fixtures from external/mplus_sem.
#
# The external corpus is local/ignored and may contain raw Mplus data. This
# script fits retained test-candidate cases with lavaan and writes only derived
# sample statistics plus lavaan oracle quantities to tests/fixtures/mplus_sem/.

suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
  library(magmaan)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))

source(file.path(repo_root, "benchmarks", "r", "fixture_json.R"))

corpus_root <- Sys.getenv("MPLUS_SEM_ROOT", unset = "")
if (!nzchar(corpus_root)) corpus_root <- file.path(repo_root, "external", "mplus_sem")
corpus_root <- normalizePath(corpus_root, mustWork = TRUE)

out_dir <- Sys.getenv("MAGMAAN_MPLUS_SEM_FIXTURE_DIR", unset = "")
if (!nzchar(out_dir)) {
  out_dir <- file.path(repo_root, "tests", "fixtures", "mplus_sem")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

manifest <- utils::read.csv(file.path(corpus_root, "manifest.csv"),
                            stringsAsFactors = FALSE, check.names = FALSE)
lavaan_version <- as.character(utils::packageVersion("lavaan"))

`%||%` <- function(x, y) if (is.null(x)) y else x

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
        out <- c(out, paste0(lhs, seq.int(a, b)))
        next
      }
    }
    out <- c(out, tok)
  }
  unique(out)
}

missing_codes <- function(sections) {
  txt <- section_value(sections[["variable"]] %||% "", "missing")
  vals <- regmatches(txt, gregexpr("-?[0-9]+(?:[.][0-9]+)?", txt, perl = TRUE))[[1L]]
  if (!length(vals)) character() else vals
}

case_paths <- function(row) {
  root <- file.path(corpus_root, row$case_dir)
  list(root = root,
       source = file.path(root, "source.inp"),
       model = file.path(root, "model.lav"),
       data = file.path(root, "data.dat"))
}

case_data <- function(row, sections) {
  p <- case_paths(row)
  vars <- expand_names(section_value(sections[["variable"]] %||% "", "names"))
  if (!length(vars)) stop("no VARIABLE:NAMES found", call. = FALSE)
  miss <- missing_codes(sections)
  dat <- utils::read.table(p$data, header = FALSE, col.names = vars,
                           na.strings = miss, check.names = FALSE)
  usev <- expand_names(section_value(sections[["variable"]] %||% "", "usevariables"))
  if (!length(usev)) usev <- expand_names(section_value(sections[["variable"]] %||% "", "usev"))
  if (length(usev)) {
    keep <- intersect(usev, names(dat))
    dat <- dat[, keep, drop = FALSE]
  }
  as.data.frame(dat, check.names = FALSE)
}

as_plain_matrix <- function(x) unname(as.matrix(x))
as_plain_vector <- function(x, names_ref = NULL) {
  if (!is.null(names_ref)) x <- x[names_ref]
  unname(as.numeric(x))
}

align_magmaan_free <- function(model, lavaan_free, meanstructure,
                               model_type = "sem") {
  mspec <- magmaan::model_spec(model, model_type = model_type,
                               meanstructure = meanstructure)
  mfree <- mspec$partable[mspec$partable$free > 0, , drop = FALSE]
  mfree <- mfree[order(mfree$free), , drop = FALSE]
  key <- function(d) paste(d$group, d$op, d$lhs, d$rhs, sep = "\r")
  idx <- match(key(mfree), key(lavaan_free))
  list(mfree = mfree, idx = idx,
       aligned = nrow(mfree) == nrow(lavaan_free) && !anyNA(idx))
}

fit_function <- function(model_kind) {
  if (identical(model_kind, "growth")) lavaan::growth else lavaan::sem
}

fit_args <- function(row, model, data, estimator) {
  meanstructure <- identical(row$model_kind, "growth") ||
    grepl("(^|\\n)\\s*[^\\n]+~\\s*1\\b", model, perl = TRUE)
  args <- list(model = model, data = data, estimator = estimator,
               meanstructure = meanstructure, fixed.x = TRUE, warn = FALSE,
               missing = "listwise")
  if (identical(row$model_kind, "growth")) args$fixed.x <- FALSE
  args
}

implied_json <- function(fit, ov) {
  im <- lavaan::lavInspect(fit, "implied")
  cov <- im$cov
  mean <- im$mean
  if (is.null(mean) || anyNA(mean)) mean <- rep(0, ncol(cov))
  list(sigma = as_plain_matrix(cov[ov, ov, drop = FALSE]),
       mu = as_plain_vector(mean, ov))
}

sample_json <- function(fit) {
  ss <- lavaan::lavInspect(fit, "sampstat")
  nobs <- as.integer(lavaan::lavInspect(fit, "nobs"))
  if (is.list(ss[[1L]]) && !is.matrix(ss[[1L]])) {
    stop("Mplus SEM fixture generator currently expects single-group fits",
         call. = FALSE)
  }
  list(cov = as_plain_matrix(ss$cov),
       mean = if (is.null(ss$mean)) NULL else as_plain_vector(ss$mean),
       n_obs = as.integer(nobs))
}

continuous_fit_payload <- function(row, model, data, estimator, align = NULL) {
  fun <- fit_function(row$model_kind)
  fit <- suppressWarnings(do.call(fun, fit_args(row, model, data, estimator)))
  if (!isTRUE(lavaan::lavInspect(fit, "converged"))) {
    stop(estimator, " did not converge", call. = FALSE)
  }

  pt <- lavaan::parTable(fit)
  free <- pt[pt$free > 0L, , drop = FALSE]
  free <- free[order(free$free), , drop = FALSE]
  meanstructure <- identical(row$model_kind, "growth") ||
    grepl("(^|\\n)\\s*[^\\n]+~\\s*1\\b", model, perl = TRUE)
  model_type <- if (identical(row$model_kind, "growth")) "growth" else "sem"
  if (is.null(align)) align <- align_magmaan_free(model, free, meanstructure, model_type)
  if (!isTRUE(align$aligned)) {
    stop("magmaan/lavaan free-parameter sets do not align", call. = FALSE)
  }

  fm <- lavaan::fitMeasures(fit)
  ov <- lavaan::lavNames(fit, type = "ov")
  samp <- sample_json(fit)
  im <- implied_json(fit, ov)
  out <- list(
    converged = TRUE,
    theta_hat = as.numeric(free$est[align$idx]),
    free_rows = lapply(seq_len(nrow(align$mfree)), function(i) {
      list(lhs = as.character(align$mfree$lhs[i]),
           op = as.character(align$mfree$op[i]),
           rhs = as.character(align$mfree$rhs[i]),
           group = as.integer(align$mfree$group[i]),
           free = as.integer(align$mfree$free[i]))
    }),
    fmin = as.numeric(fm["fmin"]),
    chisq = as.numeric(fm["chisq"]),
    df = as.integer(fm["df"]),
    npar = as.integer(fm["npar"]),
    sample_cov = samp$cov,
    sample_mean = samp$mean,
    n_obs = samp$n_obs,
    sigma = im$sigma,
    mu = im$mu
  )
  if (estimator %in% c("WLS")) {
    out$WLS.V <- matrix_list_json(lavaan::lavInspect(fit, "WLS.V"))
  }
  out
}

emit_continuous_case <- function(row) {
  paths <- case_paths(row)
  source <- paste(readLines(paths$source, warn = FALSE), collapse = "\n")
  sections <- section_map(source)
  model <- paste(readLines(paths$model, warn = FALSE), collapse = "\n")
  data <- case_data(row, sections)

  estimators <- "ML"
  if (isTRUE(row$snlls_candidate)) estimators <- c(estimators, "ULS", "GLS", "WLS")

  fits <- list()
  align <- NULL
  for (est in estimators) {
    payload <- continuous_fit_payload(row, model, data, est, align)
    if (is.null(align)) {
      # Recompute from the successful ML fit so all estimators share the same
      # magmaan-order free-row metadata.
      fun <- fit_function(row$model_kind)
      fit <- suppressWarnings(do.call(fun, fit_args(row, model, data, est)))
      free <- lavaan::parTable(fit)
      free <- free[free$free > 0L, , drop = FALSE]
      free <- free[order(free$free), , drop = FALSE]
      meanstructure <- identical(row$model_kind, "growth") ||
        grepl("(^|\\n)\\s*[^\\n]+~\\s*1\\b", model, perl = TRUE)
      model_type <- if (identical(row$model_kind, "growth")) "growth" else "sem"
      align <- align_magmaan_free(model, free, meanstructure, model_type)
    }
    fits[[est]] <- payload
  }

  first_fit <- fits[[1L]]
  list(
    id = row$case_id,
    title = row$title,
    source_input = row$source_input,
    source_data = row$source_data,
    data_kind = row$data_kind,
    model_kind = row$model_kind,
    snlls_candidate = isTRUE(row$snlls_candidate),
    lavaan_function = if (identical(row$model_kind, "growth")) "growth" else "sem",
    meanstructure = identical(row$model_kind, "growth") ||
      grepl("(^|\\n)\\s*[^\\n]+~\\s*1\\b", model, perl = TRUE),
    fixed_x = !identical(row$model_kind, "growth"),
    model = model,
    ov_names = lavaan::lavNames(suppressWarnings(do.call(
      fit_function(row$model_kind), fit_args(row, model, data, "ML"))), type = "ov"),
    estimators = names(fits),
    n_obs = first_fit$n_obs,
    fits = fits
  )
}

manifest_json <- list(
  `_meta` = list(format_version = 1L,
                 fixture_kind = "mplus_sem.manifest",
                 tool = "tests/tools/regen_mplus_sem_fixtures.R",
                 generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
                 mplus_sem_root = if (startsWith(corpus_root, paste0(repo_root, .Platform$file.sep))) {
                   sub(paste0("^", repo_root, .Platform$file.sep), "", corpus_root)
                 } else corpus_root,
                 lavaan_version = lavaan_version),
  counts = list(
    total = nrow(manifest),
    retained = sum(manifest$status == "retained"),
    excluded = sum(manifest$status == "excluded"),
    continuous_retained = sum(manifest$status == "retained" & manifest$data_kind == "continuous"),
    ordinal_retained = sum(manifest$status == "retained" & manifest$data_kind == "ordinal"),
    mixed_retained = sum(manifest$status == "retained" & manifest$data_kind == "mixed"),
    prescreen_test_candidates = sum(manifest$test_candidate),
    prescreen_snlls_candidates = sum(manifest$snlls_candidate)
  ),
  cases = lapply(seq_len(nrow(manifest)), function(i) as.list(manifest[i, , drop = FALSE]))
)
jsonlite::write_json(manifest_json, file.path(out_dir, "manifest.json"),
                     pretty = TRUE, auto_unbox = TRUE, null = "null",
                     na = "null", digits = NA)

continuous_rows <- manifest[manifest$test_candidate &
                              manifest$data_kind == "continuous" &
                              manifest$status == "retained" &
                              manifest$model_kind != "observed_path", , drop = FALSE]

continuous_cases <- list()
skipped <- list()
for (i in seq_len(nrow(continuous_rows))) {
  row <- continuous_rows[i, , drop = FALSE]
  message("mplus_sem fixture: ", row$case_id)
  res <- tryCatch(emit_continuous_case(row), error = function(e) e)
  if (inherits(res, "error")) {
    skipped[[length(skipped) + 1L]] <- list(
      id = row$case_id,
      reason = conditionMessage(res)
    )
    message("  skipped: ", conditionMessage(res))
  } else {
    continuous_cases[[length(continuous_cases) + 1L]] <- res
  }
}

continuous_payload <- list(
  `_meta` = list(format_version = 1L,
                 fixture_kind = "mplus_sem.continuous",
                 tool = "tests/tools/regen_mplus_sem_fixtures.R",
                 generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
                 lavaan_version = lavaan_version),
  cases = continuous_cases,
  skipped = skipped
)
jsonlite::write_json(continuous_payload,
                     file.path(out_dir, "continuous_reference.json"),
                     pretty = TRUE, auto_unbox = TRUE, null = "null",
                     na = "null", digits = NA)

write_empty_categorical <- function(kind) {
  rows <- manifest[manifest$status == "retained" &
                     manifest$data_kind == kind, , drop = FALSE]
  payload <- list(
    `_meta` = list(format_version = 1L,
                   fixture_kind = paste0("mplus_sem.", kind),
                   tool = "tests/tools/regen_mplus_sem_fixtures.R",
                   generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
                   lavaan_version = lavaan_version,
                   note = "No v1 test candidates: retained categorical Mplus examples are observed-response models or require features outside magmaan's ordinal LS surface."),
    cases = list(),
    retained_not_tested = lapply(seq_len(nrow(rows)), function(i) {
      as.list(rows[i, , drop = FALSE])
    })
  )
  jsonlite::write_json(payload,
                       file.path(out_dir, paste0(kind, "_reference.json")),
                       pretty = TRUE, auto_unbox = TRUE, null = "null",
                       na = "null", digits = NA)
}

write_empty_categorical("ordinal")
write_empty_categorical("mixed")

cat("Wrote Mplus SEM fixtures under ", out_dir, "\n", sep = "")
cat("continuous cases: ", length(continuous_cases), "\n", sep = "")
cat("continuous skipped: ", length(skipped), "\n", sep = "")
