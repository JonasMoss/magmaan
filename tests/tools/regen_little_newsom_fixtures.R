#!/usr/bin/env Rscript

# Regenerate checked-in Little/Newsom corpus fixtures from the ignored curated
# corpora at corpus/textbook-corpus/raw/little and
# corpus/textbook-corpus/raw/newsom. The builders create the corpora; this
# script fits strict lavaan-backed cases and writes compact JSON oracles.

suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", ".."))

lavaan_version <- as.character(utils::packageVersion("lavaan"))
out_base <- file.path(repo_root, "tests", "fixtures")

as_plain_matrix <- function(x) unname(as.matrix(x))
as_plain_vector <- function(x, names_ref = NULL) {
  if (!is.null(names_ref)) x <- x[names_ref]
  unname(as.numeric(x))
}

read_named_vector <- function(path) {
  x <- utils::read.csv(path, check.names = FALSE)
  stats::setNames(as.numeric(x[[2L]]), x[[1L]])
}

read_cov <- function(path) {
  as.matrix(utils::read.csv(path, row.names = 1L, check.names = FALSE))
}

row_bool <- function(row, name, default = FALSE) {
  if (!name %in% names(row)) return(default)
  v <- row[[name]]
  if (is.logical(v)) return(isTRUE(v))
  toupper(trimws(as.character(v))) %in% c("TRUE", "T", "1", "YES")
}

row_int <- function(row, name, default = NA_integer_) {
  if (!name %in% names(row)) return(default)
  v <- suppressWarnings(as.integer(row[[name]]))
  if (is.na(v)) default else v
}

fit_fun <- function(name) {
  switch(tolower(name), cfa = lavaan::cfa, growth = lavaan::growth, lavaan::sem)
}

fit_case <- function(root, row) {
  model_path <- file.path(root, row$generated_model)
  if (!file.exists(model_path)) stop("missing model file", call. = FALSE)
  model <- paste(readLines(model_path, warn = FALSE), collapse = "\n")
  fun <- fit_fun(row$lavaan_function)
  meanstructure <- row_bool(row, "meanstructure", TRUE)
  fixed_x <- row_bool(row, "fixed_x", TRUE)

  if (identical(row$data_kind, "raw")) {
    data_path <- file.path(root, row$generated_data)
    if (!file.exists(data_path)) stop("missing data file", call. = FALSE)
    data <- utils::read.csv(data_path, check.names = FALSE)
    fit <- suppressWarnings(fun(model = model, data = data, estimator = "ML",
                                meanstructure = meanstructure,
                                fixed.x = fixed_x, missing = "listwise",
                                warn = FALSE))
  } else if (identical(row$data_kind, "summary")) {
    cov_path <- file.path(root, row$generated_data)
    if (!file.exists(cov_path)) stop("missing covariance file", call. = FALSE)
    sample_cov <- read_cov(cov_path)
    mean_path <- sub("_cov\\.csv$", "_mean.csv", cov_path)
    sample_mean <- if (file.exists(mean_path)) {
      read_named_vector(mean_path)[colnames(sample_cov)]
    } else {
      NULL
    }
    fit <- suppressWarnings(fun(model = model, sample.cov = sample_cov,
                                sample.mean = sample_mean,
                                sample.nobs = row_int(row, "nobs", 100L),
                                estimator = "ML",
                                meanstructure = meanstructure,
                                fixed.x = fixed_x, warn = FALSE))
  } else {
    stop("no lavaan data for row", call. = FALSE)
  }
  if (!isTRUE(lavaan::lavInspect(fit, "converged"))) {
    stop("lavaan did not converge", call. = FALSE)
  }

  pt <- lavaan::parTable(fit)
  free <- pt[pt$free > 0L, , drop = FALSE]
  free <- free[order(free$free), , drop = FALSE]
  ss <- lavaan::lavInspect(fit, "sampstat")
  im <- lavaan::lavInspect(fit, "implied")
  ov <- lavaan::lavNames(fit, type = "ov")
  fm <- lavaan::fitMeasures(fit)
  mean <- ss$mean
  if (is.null(mean)) mean <- rep(0, ncol(ss$cov))
  mu <- im$mean
  if (is.null(mu)) mu <- rep(0, ncol(im$cov))

  list(
    id = row$id,
    name = row$name,
    family = row$family,
    source_input = row$source_input,
    source_data = row$source_data,
    data_kind = row$data_kind,
    measurement_kind = row$measurement_kind,
    observed_only = row_bool(row, "observed_only", FALSE),
    lavaan_function = row$lavaan_function,
    meanstructure = meanstructure,
    fixed_x = fixed_x,
    model = model,
    ov_names = unname(ov),
    n_obs = as.integer(lavaan::lavInspect(fit, "nobs")),
    sample_cov = as_plain_matrix(ss$cov[ov, ov, drop = FALSE]),
    sample_mean = as_plain_vector(mean, ov),
    lavaan = list(
      version = lavaan_version,
      fmin = as.numeric(fm["fmin"]),
      chisq = as.numeric(fm["chisq"]),
      df = as.integer(fm["df"]),
      npar = as.integer(fm["npar"]),
      sigma = as_plain_matrix(im$cov[ov, ov, drop = FALSE]),
      mu = as_plain_vector(mu, ov),
      theta = as.numeric(free$est),
      free_rows = lapply(seq_len(nrow(free)), function(i) {
        list(lhs = as.character(free$lhs[i]),
             op = as.character(free$op[i]),
             rhs = as.character(free$rhs[i]),
             group = as.integer(free$group[i]),
             free = as.integer(free$free[i]))
      })
    )
  )
}

write_json_file <- function(payload, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(payload, path, pretty = TRUE, auto_unbox = TRUE,
                       null = "null", na = "null", digits = NA)
}

build_payloads <- function(corpus) {
  root <- file.path(repo_root, "corpus", "textbook-corpus", "raw", corpus)
  catalogue <- utils::read.csv(file.path(root, "catalogue.csv"),
                               stringsAsFactors = FALSE, check.names = FALSE)
  out_dir <- file.path(out_base, corpus)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  root_meta <- if (startsWith(root, paste0(repo_root, .Platform$file.sep))) {
    sub(paste0("^", repo_root, .Platform$file.sep), "", root)
  } else {
    root
  }

  manifest <- list(
    `_meta` = list(format_version = 1L,
                   fixture_kind = paste0(corpus, ".manifest"),
                   tool = "tests/tools/regen_little_newsom_fixtures.R",
                   generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
                   corpus_root = root_meta,
                   lavaan_version = lavaan_version),
    counts = list(
      total = nrow(catalogue),
      continuous = sum(catalogue$measurement_kind == "continuous"),
      ordinal = sum(catalogue$measurement_kind == "ordinal"),
      mixed = sum(catalogue$measurement_kind == "mixed"),
      observed_only = sum(catalogue$observed_only),
      strict_parity = sum(catalogue$strict_parity)
    ),
    cases = lapply(seq_len(nrow(catalogue)), function(i) {
      as.list(catalogue[i, , drop = FALSE])
    })
  )
  write_json_file(manifest, file.path(out_dir, "manifest.json"))

  fitted <- list()
  skipped <- list()
  strict_rows <- catalogue[catalogue$strict_parity, , drop = FALSE]
  for (i in seq_len(nrow(strict_rows))) {
    row <- strict_rows[i, , drop = FALSE]
    message(corpus, " fixture: ", row$id)
    res <- tryCatch(fit_case(root, row), error = function(e) e)
    if (inherits(res, "error")) {
      skipped[[length(skipped) + 1L]] <- list(
        id = row$id, measurement_kind = row$measurement_kind,
        observed_only = row_bool(row, "observed_only", FALSE),
        reason = conditionMessage(res)
      )
      message("  skipped: ", conditionMessage(res))
    } else {
      fitted[[length(fitted) + 1L]] <- res
    }
  }

  kinds <- c("continuous", "ordinal", "mixed")
  for (kind in kinds) {
    rows <- catalogue[catalogue$measurement_kind == kind, , drop = FALSE]
    payload <- list(
      `_meta` = list(format_version = 1L,
                     fixture_kind = paste0(corpus, ".", kind),
                     tool = "tests/tools/regen_little_newsom_fixtures.R",
                     generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
                     lavaan_version = lavaan_version),
      cases = fitted[vapply(fitted, function(x) identical(x$measurement_kind, kind),
                            logical(1L))],
      skipped = skipped[vapply(skipped, function(x) identical(x$measurement_kind, kind),
                               logical(1L))],
      retained_not_tested = lapply(seq_len(nrow(rows[!rows$strict_parity, , drop = FALSE])),
                                   function(i) {
        as.list(rows[!rows$strict_parity, , drop = FALSE][i, , drop = FALSE])
      })
    )
    write_json_file(payload, file.path(out_dir, paste0(kind, "_reference.json")))
  }

  observed_rows <- catalogue[catalogue$observed_only, , drop = FALSE]
  observed_payload <- list(
    `_meta` = list(format_version = 1L,
                   fixture_kind = paste0(corpus, ".observed"),
                   tool = "tests/tools/regen_little_newsom_fixtures.R",
                   generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
                   lavaan_version = lavaan_version),
    cases = fitted[vapply(fitted, function(x) isTRUE(x$observed_only),
                          logical(1L))],
    retained_not_tested = lapply(seq_len(nrow(observed_rows[!observed_rows$strict_parity, , drop = FALSE])),
                                 function(i) {
      as.list(observed_rows[!observed_rows$strict_parity, , drop = FALSE][i, , drop = FALSE])
    })
  )
  write_json_file(observed_payload, file.path(out_dir, "observed_reference.json"))
}

build_payloads("newsom")
build_payloads("little")
cat("Wrote Little/Newsom fixtures under ", out_base, "\n", sep = "")
