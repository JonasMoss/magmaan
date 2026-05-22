#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
})

args <- commandArgs(FALSE)
script_arg <- args[grepl("^--file=", args)][1]
if (is.na(script_arg)) stop("Run this script with Rscript.", call. = FALSE)
repo_root <- normalizePath(file.path(dirname(normalizePath(sub("^--file=", "", script_arg))), "..", ".."))
paper_dir <- file.path(repo_root, "papers", "snlls-constrained")
pkg_dir <- file.path(paper_dir, "r-package")

if (!requireNamespace("pkgload", quietly = TRUE)) {
  stop("Install pkgload before running this script.", call. = FALSE)
}
pkgload::load_all(pkg_dir, quiet = TRUE)

geiser_root <- Sys.getenv("SNLLS_GEISER_ROOT", unset = "")
if (!nzchar(geiser_root)) geiser_root <- file.path(repo_root, "external", "geiser")
geiser_root <- normalizePath(geiser_root, mustWork = TRUE)
geiser_root_meta <- if (startsWith(geiser_root, paste0(repo_root, .Platform$file.sep))) {
  sub(paste0("^", repo_root, .Platform$file.sep), "", geiser_root)
} else {
  geiser_root
}

out_path <- Sys.getenv("MAGMAAN_GEISER_FIXTURE", unset = "")
if (!nzchar(out_path)) {
  out_path <- file.path(repo_root, "tests", "fixtures", "geiser", "gls_reference.json")
}
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

as_plain_matrix <- function(x) {
  unname(as.matrix(x))
}

as_plain_vector <- function(x, names_ref = NULL) {
  if (!is.null(names_ref)) x <- x[names_ref]
  unname(as.numeric(x))
}

lavaan_fit_for_case <- function(case) {
  fit_once <- snllsconstrained:::.snlls_lavaan_build(case, optim_method = "nlminb")
  fit_once()
}

lavaan_implied_moments <- function(fit, ov) {
  est <- lavaan::lavInspect(fit, "est")
  one_block <- function(mats) {
    lambda <- as.matrix(mats$lambda)
    beta <- if (is.null(mats$beta)) {
      matrix(0, ncol(lambda), ncol(lambda))
    } else {
      as.matrix(mats$beta)
    }
    psi <- as.matrix(mats$psi)
    theta <- as.matrix(mats$theta)
    nu <- if (is.null(mats$nu)) {
      matrix(0, nrow(lambda), 1L)
    } else {
      as.matrix(mats$nu)
    }
    alpha <- if (is.null(mats$alpha)) {
      matrix(0, ncol(lambda), 1L)
    } else {
      as.matrix(mats$alpha)
    }
    inv <- solve(diag(nrow(beta)) - beta)
    sigma <- lambda %*% inv %*% psi %*% t(inv) %*% t(lambda) + theta
    mu <- nu + lambda %*% inv %*% alpha
    rownames(sigma) <- rownames(lambda)
    colnames(sigma) <- rownames(lambda)
    names_mu <- rownames(lambda)
    mu <- as.numeric(mu[, 1L])
    names(mu) <- names_mu
    list(cov = sigma[ov, ov, drop = FALSE], mean = mu[ov])
  }
  if (is.list(est[[1L]]) && !is.matrix(est[[1L]])) {
    return(lapply(est, one_block))
  }
  one_block(est)
}

case_payload <- function(case) {
  problem <- snlls_make_problem(case)
  dat <- problem$dat
  S <- dat$S[[1L]]
  ov <- colnames(S)
  mean <- if (!is.null(dat$mean)) dat$mean[[1L]] else rep(0, ncol(S))
  nobs <- dat$nobs
  if (is.list(nobs) || length(nobs) > 1L) nobs <- nobs[[1L]]

  fit <- lavaan_fit_for_case(case)
  implied <- lavaan_implied_moments(fit, ov)
  pt <- lavaan::parameterTable(fit)
  free <- pt[pt$free > 0L, , drop = FALSE]
  group <- if ("group" %in% names(free)) free$group else rep(1L, nrow(free))
  theta <- lapply(seq_len(nrow(free)), function(i) {
    list(lhs = free$lhs[i], op = free$op[i], rhs = free$rhs[i],
         group = as.integer(group[i]), est = unname(free$est[i]))
  })

  list(
    id = case$geiser_id,
    label = case$label,
    family = case$family,
    data_kind = case$data_kind,
    estimator = "GLS",
    meanstructure = TRUE,
    fixed_x = TRUE,
    model = case$model,
    ov_names = unname(ov),
    n_obs = as.integer(nobs),
    sample_cov = as_plain_matrix(S),
    sample_mean = as_plain_vector(mean, ov),
    lavaan = list(
      version = as.character(utils::packageDescription("lavaan")$Version),
      converged = isTRUE(lavaan::lavInspect(fit, "converged")),
      fx = unname(lavaan::lavInspect(fit, "optim")$fx),
      sigma = as_plain_matrix(implied$cov),
      mu = as_plain_vector(implied$mean, ov),
      theta = theta
    )
  )
}

cases <- geiser_corpus_cases(geiser_root, weights = "GLS")
payload_cases <- list()
for (nm in names(cases)) {
  case <- cases[[nm]]
  message("geiser fixture: ", case$geiser_id)
  payload_cases[[length(payload_cases) + 1L]] <- case_payload(case)
}

payload <- list(
  `_meta` = list(
    format_version = 1L,
    fixture_kind = "geiser.gls",
    tool = "tests/tools/regen_geiser_fixtures.R",
    generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    geiser_root = geiser_root_meta,
    lavaan_version = as.character(utils::packageDescription("lavaan")$Version)
  ),
  cases = payload_cases
)

jsonlite::write_json(payload, out_path, pretty = TRUE, auto_unbox = TRUE,
                     digits = NA, null = "null")
message("Wrote ", out_path)
