#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(magmaan))

parse_args <- function(args) {
  out <- list(
    n = c(500L, 2000L),
    reps = 6L,
    missing = 0.15,
    dist = "normal",
    t_df = 5.0,
    alpha = 0.05,
    a_method = "delta",
    seed = 20260609L,
    max_iter = 10000L
  )
  for (a in args) {
    kv <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1L]]
    if (length(kv) != 2L) next
    key <- kv[[1L]]
    val <- kv[[2L]]
    if (identical(key, "n")) {
      out$n <- as.integer(strsplit(val, ",", fixed = TRUE)[[1L]])
    } else if (identical(key, "reps")) {
      out$reps <- as.integer(val)
    } else if (identical(key, "missing")) {
      out$missing <- as.numeric(val)
    } else if (identical(key, "dist")) {
      out$dist <- val
    } else if (identical(key, "t-df")) {
      out$t_df <- as.numeric(val)
    } else if (identical(key, "alpha")) {
      out$alpha <- as.numeric(val)
    } else if (identical(key, "a-method")) {
      out$a_method <- val
    } else if (identical(key, "seed")) {
      out$seed <- as.integer(val)
    } else if (identical(key, "max-iter")) {
      out$max_iter <- as.integer(val)
    }
  }
  out
}

population <- function() {
  lambda <- c(1.00, 0.80, 0.80, 0.80, 0.85)
  theta <- c(0.60, 0.70, 0.50, 0.80, 0.65)
  mu <- c(0.20, -0.10, 0.40, 0.00, 0.15)
  Sigma <- tcrossprod(lambda) + diag(theta)
  list(mu = mu, Sigma = Sigma)
}

draw_null_data <- function(n, pop, missing_prob, dist, t_df) {
  p <- length(pop$mu)
  Z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  X0 <- Z %*% chol(pop$Sigma)
  if (identical(dist, "t")) {
    w <- stats::rchisq(n, df = t_df) / t_df
    row_scale <- sqrt((t_df - 2.0) / t_df) / sqrt(w)
    X0 <- X0 * matrix(row_scale, nrow = n, ncol = p)
  }
  X <- sweep(X0, 2L, pop$mu, "+")
  colnames(X) <- paste0("x", seq_len(p))

  M <- matrix(runif(n * p) >= missing_prob, nrow = n, ncol = p)
  empty <- which(rowSums(M) == 0L)
  if (length(empty)) {
    keep <- ((empty - 1L) %% p) + 1L
    M[cbind(empty, keep)] <- TRUE
  }
  X[!M] <- NA_real_
  as.data.frame(X)
}

fit_pair <- function(df, spec_h1, spec_h0, max_iter, alpha, a_method) {
  raw_h1 <- df_to_fiml_data(df, spec_h1)
  raw_h0 <- df_to_fiml_data(df, spec_h0)
  fit_h1 <- magmaan_core$fit_fiml(
    spec_h1, raw_h1,
    control = list(max_iter = max_iter, ftol = 1e-12, gtol = 1e-8)
  )
  fit_h0 <- magmaan_core$fit_fiml(
    spec_h0, raw_h0,
    control = list(max_iter = max_iter, ftol = 1e-12, gtol = 1e-8)
  )
  nt <- nestedTest(fit_h1, fit_h0, method = "restriction_map",
                   A.method = a_method)
  data.frame(
    T_diff = nt$T_diff,
    df_diff = nt$df_diff,
    scale_c = nt$scale_c,
    adjust_d0 = nt$adjust_d0,
    min_lambda = min(nt$eigenvalues),
    max_lambda = max(nt$eigenvalues),
    mean_lambda = mean(nt$eigenvalues),
    p_unscaled = nt$p_unscaled,
    p_scaled = nt$p_scaled,
    p_adjusted = nt$p_adjusted,
    p_scaled_shifted = nt$p_scaled_shifted,
    p_mixture = nt$p_mixture,
    reject_unscaled = nt$p_unscaled < alpha,
    reject_scaled = nt$p_scaled < alpha,
    reject_adjusted = nt$p_adjusted < alpha,
    reject_scaled_shifted = nt$p_scaled_shifted < alpha,
    reject_mixture = nt$p_mixture < alpha
  )
}

summarise_n <- function(rows) {
  data.frame(
    n = rows$n[[1L]],
    reps_ok = nrow(rows),
    df_diff = rows$df_diff[[1L]],
    mean_T_diff = mean(rows$T_diff),
    mean_scale_c = mean(rows$scale_c),
    mean_adjust_d0 = mean(rows$adjust_d0),
    min_lambda = min(rows$min_lambda),
    max_lambda = max(rows$max_lambda),
    mean_lambda = mean(rows$mean_lambda),
    reject_unscaled = mean(rows$reject_unscaled),
    reject_scaled = mean(rows$reject_scaled),
    reject_adjusted = mean(rows$reject_adjusted),
    reject_scaled_shifted = mean(rows$reject_scaled_shifted),
    reject_mixture = mean(rows$reject_mixture),
    mean_p_mixture = mean(rows$p_mixture)
  )
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
if (any(!is.finite(cfg$n)) || any(cfg$n <= 0L)) stop("--n must be positive")
if (!is.finite(cfg$reps) || cfg$reps <= 0L) stop("--reps must be positive")
if (!is.finite(cfg$missing) || cfg$missing < 0 || cfg$missing >= 1) {
  stop("--missing must be in [0, 1)")
}
if (!cfg$dist %in% c("normal", "t")) stop("--dist must be 'normal' or 't'")
if (!cfg$a_method %in% c("exact", "delta")) {
  stop("--a-method must be 'exact' or 'delta'")
}
if (identical(cfg$dist, "t") && (!is.finite(cfg$t_df) || cfg$t_df <= 2.0)) {
  stop("--t-df must be finite and > 2")
}
if (!is.finite(cfg$alpha) || cfg$alpha <= 0 || cfg$alpha >= 1) {
  stop("--alpha must be in (0, 1)")
}

set.seed(cfg$seed)

model_h1 <- "
  f =~ x1 + x2 + x3 + x4 + x5
"
model_h0 <- "
  f =~ x1 + a*x2 + a*x3 + a*x4 + x5
"
spec_h1 <- model_spec(model_h1, meanstructure = TRUE)
spec_h0 <- model_spec(model_h0, meanstructure = TRUE)
pop <- population()

cat("FIML nested FMG nominal smoke under the null\n")
cat("H0 restriction: x2, x3, x4 loadings equal\n")
cat("data: ", cfg$dist, ", MCAR missingness\n", sep = "")
if (identical(cfg$dist, "t")) cat("t_df=", cfg$t_df, "\n", sep = "")
cat("A.method=", cfg$a_method, "\n", sep = "")
cat("missing=", cfg$missing, " reps=", cfg$reps,
    " alpha=", cfg$alpha, " seed=", cfg$seed, "\n\n", sep = "")

all_rows <- list()
idx <- 1L
for (n in cfg$n) {
  for (rep in seq_len(cfg$reps)) {
    df <- draw_null_data(n, pop, cfg$missing, cfg$dist, cfg$t_df)
    row <- tryCatch(
      fit_pair(df, spec_h1, spec_h0, cfg$max_iter, cfg$alpha, cfg$a_method),
      error = function(e) {
        warning("n=", n, " rep=", rep, " failed: ", conditionMessage(e),
                call. = FALSE)
        NULL
      }
    )
    if (is.null(row)) next
    row$n <- n
    row$rep <- rep
    all_rows[[idx]] <- row
    idx <- idx + 1L
  }
}

if (!length(all_rows)) stop("all nested FIML fits failed")

rows <- do.call(rbind, all_rows)
summary <- do.call(rbind, lapply(split(rows, rows$n), summarise_n))
rownames(summary) <- NULL

print(summary, digits = 4, row.names = FALSE)

cat("\nInterpretation: rejection columns are stochastic smoke checks against ",
    "the nominal alpha. Heavy-tailed H0-true data should penalize the unscaled ",
    "normal-theory difference more than the robust restriction-map p-values. ",
    "Use --a-method=exact only for exact row-space diagnostics; delta is the ",
    "lavaan-default nested-test convention. This script is not a standalone ",
    "calibration study.\n",
    sep = "")
