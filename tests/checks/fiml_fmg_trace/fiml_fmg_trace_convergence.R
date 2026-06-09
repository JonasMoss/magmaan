#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(magmaan))

parse_args <- function(args) {
  out <- list(
    n = c(250L, 1000L, 4000L),
    reps = 6L,
    missing = 0.15,
    seed = 20260609L,
    max_iter = 8000L
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
    } else if (identical(key, "seed")) {
      out$seed <- as.integer(val)
    } else if (identical(key, "max-iter")) {
      out$max_iter <- as.integer(val)
    }
  }
  out
}

population <- function() {
  lambda <- c(1.00, 0.80, 0.90, 0.70, 0.85)
  theta <- c(0.60, 0.70, 0.50, 0.80, 0.65)
  mu <- c(0.20, -0.10, 0.40, 0.00, 0.15)
  Sigma <- tcrossprod(lambda) + diag(theta)
  list(mu = mu, Sigma = Sigma)
}

draw_null_data <- function(n, pop, missing_prob) {
  p <- length(pop$mu)
  Z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  X <- sweep(Z %*% chol(pop$Sigma), 2L, pop$mu, "+")
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

fit_one <- function(df, spec, max_iter) {
  fit <- magmaan_core$fit_fiml(
    spec,
    df_to_fiml_data(df, spec),
    control = list(max_iter = max_iter, ftol = 1e-12, gtol = 1e-8)
  )
  sp <- magmaan:::infer_fiml_fmg_spectrum(fit)
  mlr <- magmaan:::estimate_fiml_robust_mlr(fit)
  gap <- sp$trace_xcheck - mlr$trace_ugamma
  data.frame(
    df = sp$df,
    chi2 = sp$chi2_lrt,
    trace_fmg = sp$trace_xcheck,
    trace_mlr = mlr$trace_ugamma,
    gap = gap,
    rel_gap = abs(gap) / max(1.0, abs(mlr$trace_ugamma)),
    min_lambda = min(sp$biased),
    max_lambda = max(sp$biased)
  )
}

summarise_n <- function(rows) {
  data.frame(
    n = rows$n[[1L]],
    reps_ok = nrow(rows),
    df = rows$df[[1L]],
    mean_trace_fmg = mean(rows$trace_fmg),
    mean_trace_mlr = mean(rows$trace_mlr),
    mean_gap = mean(rows$gap),
    mean_abs_gap = mean(abs(rows$gap)),
    mean_rel_gap = mean(rows$rel_gap),
    p90_rel_gap = unname(stats::quantile(rows$rel_gap, 0.90, names = FALSE)),
    min_lambda = min(rows$min_lambda),
    max_lambda = max(rows$max_lambda)
  )
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
if (any(!is.finite(cfg$n)) || any(cfg$n <= 0L)) stop("--n must be positive")
if (!is.finite(cfg$reps) || cfg$reps <= 0L) stop("--reps must be positive")
if (!is.finite(cfg$missing) || cfg$missing < 0 || cfg$missing >= 1) {
  stop("--missing must be in [0, 1)")
}

set.seed(cfg$seed)

model <- "
  f =~ x1 + x2 + x3 + x4 + x5
"
spec <- model_spec(model, meanstructure = TRUE)
pop <- population()

cat("FIML FMG trace convergence under the null\n")
cat("model: one-factor CFA, normal data, MCAR missingness\n")
cat("missing=", cfg$missing, " reps=", cfg$reps,
    " seed=", cfg$seed, "\n\n", sep = "")

all_rows <- list()
idx <- 1L
for (n in cfg$n) {
  for (rep in seq_len(cfg$reps)) {
    df <- draw_null_data(n, pop, cfg$missing)
    row <- tryCatch(
      fit_one(df, spec, cfg$max_iter),
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

if (!length(all_rows)) stop("all FIML fits failed")

rows <- do.call(rbind, all_rows)
summary <- do.call(rbind, lapply(split(rows, rows$n), summarise_n))
rownames(summary) <- NULL

print(summary, digits = 4, row.names = FALSE)

cat("\nInterpretation: the null/local theory predicts the relative trace gap ",
    "should shrink with n on average, not necessarily monotonically in every ",
    "finite-sample replicate.\n", sep = "")
