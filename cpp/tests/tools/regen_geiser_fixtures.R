#!/usr/bin/env Rscript

# Regenerate the Geiser-corpus parity fixtures consumed by
# cpp/tests/golden/geiser_golden_test.cpp. One fixture file per estimator:
#
#   cpp/tests/fixtures/geiser/gls_reference.json   GLS  (normal-theory weight)
#   cpp/tests/fixtures/geiser/uls_reference.json   ULS  (identity weight)
#
# Input: the Geiser (2013) cases of the ignored textbook corpus at
# external/textbook-corpus/cases/geiser_2013/ (model.lav, meta.json, data/).
# Those cases are translated from Geiser's companion Mplus inputs and audited
# against the companion Mplus output (external/textbook-corpus/docs/audit/
# geiser_brown.md): the model, mean structure, fixed-x treatment, variable
# subset and listwise N are the book's. The model strings written here are
# fully explicit, so lavaan's sem() conventions and magmaan's build give the
# book's parameter set.
#
# Each case carries the model, sample statistics (unbiased covariance of the
# listwise-complete data, or the published summary statistics), and the
# lavaan oracle (objective, implied moments, free estimates). The C++ golden
# test fits magmaan's own estimator end to end and checks it against the
# oracle. This is a manual developer step; CI never runs R.
#
# Dependencies: lavaan, jsonlite.
#
# Usage: Rscript cpp/tests/tools/regen_geiser_fixtures.R
#   env MAGMAAN_TEXTBOOK_CORPUS  corpus root (default external/textbook-corpus)
#   env MAGMAAN_GEISER_FIXTURE_DIR  output dir (default cpp/tests/fixtures/geiser)

suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(
  sub("^--file=", "", script_arg))), "..", "..", ".."))

corpus_root <- Sys.getenv("MAGMAAN_TEXTBOOK_CORPUS", unset = "")
if (!nzchar(corpus_root)) {
  corpus_root <- file.path(repo_root, "external", "textbook-corpus")
}
corpus_root <- normalizePath(corpus_root, mustWork = TRUE)
geiser_dir <- file.path(corpus_root, "cases", "geiser_2013")
geiser_root_meta <- if (startsWith(geiser_dir, paste0(repo_root, .Platform$file.sep))) {
  sub(paste0("^", repo_root, .Platform$file.sep), "", geiser_dir)
} else {
  geiser_dir
}

out_dir <- Sys.getenv("MAGMAAN_GEISER_FIXTURE_DIR", unset = "")
if (!nzchar(out_dir)) out_dir <- file.path(repo_root, "cpp", "tests", "fixtures", "geiser")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Fixture ids, order and family labels (Geiser's curated catalogue).
CASES <- data.frame(
  id = c(
    "manifest_regression", "latent_regression", "latent_regression_free_loadings",
    "cfa_one_factor", "cfa_three_factor", "cfa_second_order",
    "manifest_path", "manifest_path_non_saturated", "latent_path",
    "latent_state_basic", "latent_state_correlated_residuals",
    "latent_state_indicator_specific", "latent_state_weak_invariance",
    "latent_state_strong_invariance", "latent_state_equal_means",
    "latent_state_strict_invariance", "latent_state_trait",
    "manifest_ar_cross_lagged", "manifest_ar_cross_lagged_extended",
    "latent_ar_cross_lagged", "latent_ar_cross_lagged_extended",
    "latent_ar_cross_lagged_strong_invariance",
    "latent_change_baseline", "latent_change_neighbor",
    "growth_linear", "growth_intercept_only", "growth_quadratic",
    "growth_second_order_linear"),
  family = c(
    "manifest regression", "latent regression", "latent regression",
    "CFA", "CFA", "CFA",
    "path analysis", "path analysis", "path analysis",
    rep("latent state", 7), "latent state-trait",
    rep("autoregressive cross-lagged", 5),
    "latent change", "latent change",
    rep("latent growth", 4)),
  stringsAsFactors = FALSE)

as_plain_matrix <- function(x) unname(as.matrix(x))
as_plain_vector <- function(x, names_ref = NULL) {
  if (!is.null(names_ref)) x <- x[names_ref]
  unname(as.numeric(x))
}

case_dir_for <- function(id) {
  dirs <- list.dirs(geiser_dir, recursive = FALSE, full.names = TRUE)
  hit <- dirs[sub("^geiser_2013_ch[0-9]+_", "", basename(dirs)) == id]
  if (length(hit) != 1L) stop("no unique corpus case for fixture id ", id, call. = FALSE)
  hit
}

# Sample statistics as the C++ test consumes them: the unbiased (N - 1)
# covariance and mean of the listwise-complete raw data over the case's
# variables, or the published summary statistics.
case_data <- function(case_dir, meta) {
  files <- meta$data$files
  if (identical(meta$data$kind, "raw")) {
    raw <- utils::read.csv(file.path(case_dir, files$raw), check.names = FALSE)
    if (!identical(meta$model_options$missing, "listwise") && anyNA(raw)) {
      stop(basename(case_dir), ": missing data without listwise deletion", call. = FALSE)
    }
    raw <- raw[stats::complete.cases(raw), , drop = FALSE]
    X <- as.matrix(raw)
    list(S = stats::cov(X), mean = colMeans(X), n_obs = nrow(X))
  } else {
    S <- as.matrix(utils::read.csv(file.path(case_dir, files$sample_cov[[1]]),
                                   row.names = 1L, check.names = FALSE))
    mv <- utils::read.csv(file.path(case_dir, files$sample_mean[[1]]), check.names = FALSE)
    mean <- stats::setNames(as.numeric(mv[[2L]]), mv[[1L]])[colnames(S)]
    list(S = S, mean = mean, n_obs = as.integer(meta$data$n_obs[[1]]))
  }
}

case_payload <- function(id, family, estimator) {
  case_dir <- case_dir_for(id)
  meta <- jsonlite::fromJSON(file.path(case_dir, "meta.json"), simplifyVector = FALSE)
  if (!identical(meta$lavaan_function, "sem")) {
    stop(id, ": fixture expects lavaan_function = sem (explicit syntax)", call. = FALSE)
  }
  model <- paste(readLines(file.path(case_dir, "model.lav"), warn = FALSE), collapse = "\n")
  dat <- case_data(case_dir, meta)
  meanstructure <- isTRUE(meta$model_options$meanstructure)
  fixed_x <- isTRUE(meta$model_options$fixed_x)

  # GLS/ULS take the supplied covariance as is (sample.cov.rescale defaults to
  # FALSE for these estimators), matching the C++ test's SampleStats.
  fit <- lavaan::sem(model, sample.cov = dat$S, sample.mean = dat$mean,
                     sample.nobs = dat$n_obs, estimator = estimator,
                     meanstructure = meanstructure, fixed.x = fixed_x,
                     se = "none", test = "none", baseline = FALSE)
  ov <- lavaan::lavNames(fit, "ov")
  # Keep the data column order for the fixture (the C++ test permutes by name).
  ov_names <- colnames(dat$S)[colnames(dat$S) %in% ov]
  if (length(ov_names) != length(ov)) stop(id, ": data lacks model variables", call. = FALSE)
  implied <- lavaan::lavInspect(fit, "implied")
  sigma <- implied$cov[ov_names, ov_names, drop = FALSE]
  mu <- if (!is.null(implied$mean)) implied$mean[ov_names] else rep(0, length(ov_names))
  pt <- lavaan::parameterTable(fit)
  free <- pt[pt$free > 0L, , drop = FALSE]
  theta <- lapply(seq_len(nrow(free)), function(i) {
    list(lhs = free$lhs[i], op = free$op[i], rhs = free$rhs[i],
         group = as.integer(free$group[i]), est = unname(free$est[i]))
  })

  list(
    id = id,
    label = sub(" \\(Geiser 2013\\)$", "", meta$label),
    family = family,
    data_kind = meta$data$kind,
    estimator = estimator,
    meanstructure = meanstructure,
    fixed_x = fixed_x,
    model = model,
    ov_names = ov_names,
    n_obs = as.integer(dat$n_obs),
    sample_cov = as_plain_matrix(dat$S[ov_names, ov_names, drop = FALSE]),
    sample_mean = as_plain_vector(dat$mean, ov_names),
    lavaan = list(
      version = as.character(utils::packageDescription("lavaan")$Version),
      converged = isTRUE(lavaan::lavInspect(fit, "converged")),
      fx = unname(lavaan::lavInspect(fit, "optim")$fx),
      sigma = as_plain_matrix(sigma),
      mu = as_plain_vector(mu),
      theta = theta
    )
  )
}

generate <- function(estimator) {
  payload_cases <- lapply(seq_len(nrow(CASES)), function(i) {
    message("geiser fixture (", estimator, "): ", CASES$id[i])
    case_payload(CASES$id[i], CASES$family[i], estimator)
  })
  payload <- list(
    `_meta` = list(
      format_version = 1L,
      fixture_kind = paste0("geiser.", tolower(estimator)),
      tool = "cpp/tests/tools/regen_geiser_fixtures.R",
      generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
      geiser_root = geiser_root_meta,
      lavaan_version = as.character(utils::packageDescription("lavaan")$Version)
    ),
    cases = payload_cases
  )
  out_path <- file.path(out_dir, paste0(tolower(estimator), "_reference.json"))
  jsonlite::write_json(payload, out_path, pretty = TRUE, auto_unbox = TRUE,
                       digits = NA, null = "null")
  message("Wrote ", out_path)
}

for (estimator in c("GLS", "ULS")) generate(estimator)
