#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list(cases = "400:0.30:1000008,200:0.50:1000008,120:0.60:1000008")
  for (a in args) {
    kv <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1L]]
    if (length(kv) == 2L && identical(kv[[1L]], "cases")) out$cases <- kv[[2L]]
  }
  out
}

parse_cases <- function(x) {
  parts <- strsplit(x, ",", fixed = TRUE)[[1L]]
  do.call(rbind, lapply(parts, function(part) {
    fields <- strsplit(part, ":", fixed = TRUE)[[1L]]
    if (length(fields) != 3L) stop("case must be n:missing:seed: ", part)
    data.frame(
      n = as.integer(fields[[1L]]),
      missing = as.numeric(fields[[2L]]),
      seed = as.integer(fields[[3L]])
    )
  }))
}

big_five_phi <- function() {
  matrix(c(
    1.0,  0.2,  0.3,  0.1,  0.0,
    0.2,  1.0,  0.1,  0.3, -0.3,
    0.3,  0.1,  1.0,  0.1, -0.3,
    0.1,  0.3,  0.1,  1.0, -0.2,
    0.0, -0.3, -0.3, -0.2,  1.0
  ), nrow = 5L, byrow = TRUE)
}

population <- function(indicators_per_factor = 6L) {
  p <- 5L * indicators_per_factor
  set.seed(1000L + p)
  loadings <- unlist(lapply(seq_len(5L), function(f) {
    round(stats::runif(indicators_per_factor, 0.55, 0.85), 2L)
  }))
  lambda <- matrix(0, nrow = p, ncol = 5L)
  for (f in seq_len(5L)) {
    rows <- ((f - 1L) * indicators_per_factor + 1L):(f * indicators_per_factor)
    lambda[rows, f] <- loadings[rows]
  }
  theta <- diag(1 - rowSums(lambda^2), p)
  sigma <- lambda %*% big_five_phi() %*% t(lambda) + theta
  colnames(sigma) <- rownames(sigma) <- paste0("x", seq_len(p))
  syntax <- paste(vapply(seq_len(5L), function(f) {
    rhs <- paste0("x", ((f - 1L) * indicators_per_factor + 1L):
                    (f * indicators_per_factor), collapse = " + ")
    paste0("F", f, " =~ ", rhs)
  }, character(1L)), collapse = "\n")
  list(sigma = sigma, syntax = syntax)
}

draw_mcar <- function(pop, n, missing, seed) {
  set.seed(seed)
  p <- ncol(pop$sigma)
  X <- matrix(rnorm(n * p), nrow = n) %*% chol(pop$sigma)
  colnames(X) <- colnames(pop$sigma)
  mask <- matrix(stats::runif(n * p) >= missing, nrow = n, ncol = p)
  empty <- rowSums(mask) == 0L
  if (any(empty)) mask[cbind(which(empty), 1L)] <- TRUE
  X[!mask] <- NA_real_
  as.data.frame(X)
}

min_eig <- function(x) {
  min(eigen((x + t(x)) / 2, symmetric = TRUE, only.values = TRUE)$values)
}

one_lavaan <- function(pop, df) {
  if (!requireNamespace("lavaan", quietly = TRUE)) return(NULL)
  warnings <- character()
  fit <- withCallingHandlers(
    tryCatch(lavaan::cfa(pop$syntax, data = df, missing = "fiml",
                         std.lv = TRUE, meanstructure = TRUE),
             error = function(e) e),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  if (inherits(fit, "error")) {
    return(data.frame(engine = "lavaan", status = "error",
                      converged = NA, iter = NA, cov_min = NA, info_min = NA,
                      warnings = conditionMessage(fit)))
  }
  h1 <- lavaan::lavInspect(fit, "h1")
  h1_cov <- if (!is.null(h1$cov)) h1$cov else h1[[1L]]$cov
  info <- tryCatch(lavaan::lavInspect(fit, "h1.information"),
                   error = function(e) NULL)
  data.frame(
    engine = "lavaan",
    status = "ok",
    converged = lavaan::lavInspect(fit, "converged"),
    iter = lavaan::lavInspect(fit, "iterations"),
    cov_min = min_eig(h1_cov),
    info_min = if (is.null(info)) NA_real_ else min_eig(info),
    warnings = paste(unique(warnings), collapse = " | ")
  )
}

one_magmaan <- function(pop, df) {
  if (!requireNamespace("magmaan", quietly = TRUE)) {
    stop("Install magmaan before running this check.")
  }
  raw <- list(X = list(as.matrix(df)))
  stage1 <- magmaan::magmaan_core$estimate_saturated_em_moments(raw)
  data.frame(
    engine = "magmaan",
    status = "ok",
    converged = NA,
    iter = NA,
    cov_min = min(vapply(stage1$cov, min_eig, numeric(1L))),
    info_min = min_eig(stage1$H),
    warnings = paste(stage1$warnings %||% character(), collapse = " | ")
  )
}

`%||%` <- function(x, y) if (is.null(x)) y else x

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
cases <- parse_cases(cfg$cases)
pop <- population()

rows <- list()
for (i in seq_len(nrow(cases))) {
  df <- draw_mcar(pop, cases$n[[i]], cases$missing[[i]], cases$seed[[i]])
  lav <- one_lavaan(pop, df)
  mag <- one_magmaan(pop, df)
  block <- rbind(if (is.null(lav)) NULL else lav, mag)
  block$n <- cases$n[[i]]
  block$missing <- cases$missing[[i]]
  block$seed <- cases$seed[[i]]
  rows[[i]] <- block
}

out <- do.call(rbind, rows)
out <- out[, c("n", "missing", "seed", "engine", "status", "converged",
               "iter", "cov_min", "info_min", "warnings")]
print(out, row.names = FALSE, digits = 4, right = FALSE)
