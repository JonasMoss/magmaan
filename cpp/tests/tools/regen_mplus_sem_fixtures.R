#!/usr/bin/env Rscript

# Regenerate Mplus SEM corpus parity fixtures from
# external/textbook-corpus/raw/mplus_sem.
#
# The raw corpus is local/ignored and may contain raw Mplus data. This script
# fits retained test-candidate cases with lavaan and writes only derived
# sample statistics plus lavaan oracle quantities to cpp/tests/fixtures/mplus_sem/.

suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
  library(magmaanlab)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", "..", ".."))

source(file.path(repo_root, "benchmarks", "r", "fixture_json.R"))

corpus_root <- Sys.getenv("MPLUS_SEM_ROOT", unset = "")
if (!nzchar(corpus_root)) {
  corpus_root <- file.path(repo_root, "external", "textbook-corpus", "raw", "mplus_sem")
}
corpus_root <- normalizePath(corpus_root, mustWork = TRUE)

out_dir <- Sys.getenv("MAGMAAN_MPLUS_SEM_FIXTURE_DIR", unset = "")
if (!nzchar(out_dir)) {
  out_dir <- file.path(repo_root, "cpp", "tests", "fixtures", "mplus_sem")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

manifest <- utils::read.csv(file.path(corpus_root, "manifest.csv"),
                            stringsAsFactors = FALSE, check.names = FALSE)
lavaan_version <- as.character(utils::packageVersion("lavaan"))

`%||%` <- function(x, y) if (is.null(x)) y else x

case_paths <- function(row) {
  root <- file.path(corpus_root, row$case_dir)
  list(root = root,
       model = file.path(root, "model.lav"),
       data = file.path(root, "data.csv"),
       sample_cov = file.path(root, "sample_cov.csv"),
       sample_mean = file.path(root, "sample_mean.csv"))
}

# The builder stores Mplus's analysis sample (after MISSING, USEOBSERVATIONS,
# DEFINE and case exclusion) or the summary statistics of summary-data inputs.
case_data <- function(row) {
  p <- case_paths(row)
  if (file.exists(p$data)) {
    return(list(data = utils::read.csv(p$data, check.names = FALSE)))
  }
  S <- as.matrix(utils::read.csv(p$sample_cov, row.names = 1, check.names = FALSE))
  out <- list(sample.cov = S, sample.nobs = as.integer(row$n_obs))
  if (file.exists(p$sample_mean)) {
    m <- utils::read.csv(p$sample_mean, row.names = 1, check.names = FALSE)
    out$sample.mean <- setNames(m[[1L]], rownames(m))
  }
  out
}

as_plain_matrix <- function(x) unname(as.matrix(x))
as_plain_vector <- function(x, names_ref = NULL) {
  if (!is.null(names_ref)) x <- x[names_ref]
  unname(as.numeric(x))
}

align_magmaan_free <- function(model, lavaan_free, meanstructure,
                               model_type = "sem") {
  mspec <- magmaanlab::model_spec(model, model_type = model_type,
                               meanstructure = meanstructure)
  mfree <- mspec$partable[mspec$partable$free > 0, , drop = FALSE]
  mfree <- mfree[order(mfree$free), , drop = FALSE]
  key <- function(d) paste(d$group, d$op, d$lhs, d$rhs, sep = "\r")
  idx <- match(key(mfree), key(lavaan_free))
  list(mfree = mfree, idx = idx,
       aligned = nrow(mfree) == nrow(lavaan_free) && !anyNA(idx))
}

# The translation is explicit about every Mplus default (growth intercepts,
# latent means, x variables as fixed covariates), so every case is fit with
# lavaan::sem under the options recorded by the builder.
fit_function <- function(model_kind) lavaan::sem

fit_args <- function(row, model, data, estimator) {
  args <- c(list(model = model, estimator = estimator,
                 meanstructure = isTRUE(row$meanstructure),
                 fixed.x = isTRUE(row$fixed_x), warn = FALSE), data)
  if (!is.na(row$missing) && !identical(row$missing, "none")) args$missing <- row$missing
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
  if (is.null(align)) align <- align_magmaan_free(model, free, isTRUE(row$meanstructure), "sem")
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
  model <- paste(readLines(paths$model, warn = FALSE), collapse = "\n")
  data <- case_data(row)

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
      align <- align_magmaan_free(model, free, isTRUE(row$meanstructure), "sem")
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
    lavaan_function = "sem",
    meanstructure = isTRUE(row$meanstructure),
    fixed_x = isTRUE(row$fixed_x),
    mplus_estimator = row$estimator,
    mplus_verification = row$verification,
    corpus_case_id = row$corpus_case_id,
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
                 tool = "cpp/tests/tools/regen_mplus_sem_fixtures.R",
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
                 tool = "cpp/tests/tools/regen_mplus_sem_fixtures.R",
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
                   tool = "cpp/tests/tools/regen_mplus_sem_fixtures.R",
                   generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
                   lavaan_version = lavaan_version,
                   note = "No retained categorical cases: the source-fidelity screen keeps only examples lavaan reproduces as linear normal-theory models verified against the Mplus output; categorical examples are listed with their exclusion reason in manifest.json."),
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
