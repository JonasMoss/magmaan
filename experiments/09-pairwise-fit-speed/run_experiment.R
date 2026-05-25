#!/usr/bin/env Rscript

# Experiment 09 — pairwise fit-speed bench.
# Time the three pairwise estimators studied in experiment 08:
#   GLSpw_Sigma   — fit_gls(samp.S = Ŝ_pw)
#   PairwiseML    — fit_ml(samp.S = Ŝ_pw)
#   GLSpw_Gamma   — fit_gls_pairwise(raw, pw)
# Two textbook-style CFA designs, two missingness mechanisms, microbenchmark.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

parse_args <- function(args) {
  out <- list(
    times = 30L,        # microbenchmark replications per cell
    n     = 500L,       # fixed sample size; speed isn't strongly n-dependent
    rate  = 0.20,       # one missing rate; speed isn't strongly rate-dependent
    seed_base = 20260612L,
    smoke = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--times K] [--n N] [--rate R]\n",
        "                                [--smoke] [--seed-base S]\n",
        "\n",
        "  --times K        microbenchmark replications per cell (default 30).\n",
        "  --n N            sample size (default 500).\n",
        "  --rate R         missing rate, both mechanisms (default 0.20).\n",
        "  --smoke          K = 5, n = 200.\n",
        "  --seed-base S    deterministic seed base (default 20260612).\n",
        "  --help           show this message.\n",
        "\n",
        "Design: two textbook-style CFA models —\n",
        "  brown_ch5_1f5i — 1 factor, 5 indicators, std loadings 0.7\n",
        "  hs1939_3f9i    — 3 correlated factors × 3 indicators (9 total)\n",
        "  × {MCAR, MAR}  × {Σ-only GLS, pairwise ML, Γ_NT^pw GLS}\n",
        "Outcome: median wall-time per fit and ratio to the fastest estimator.\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    } else if (arg == "--times")     { i <- i + 1L; out$times <- as.integer(args[[i]]) }
    else if (arg == "--n")           { i <- i + 1L; out$n <- as.integer(args[[i]]) }
    else if (arg == "--rate")        { i <- i + 1L; out$rate <- as.numeric(args[[i]]) }
    else if (arg == "--seed-base")   { i <- i + 1L; out$seed_base <- as.integer(args[[i]]) }
    else if (arg == "--smoke")       { out$smoke <- TRUE }
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  if (out$smoke) { out$times <- 5L; out$n <- 200L }
  out
}

# -- Textbook-style designs ------------------------------------------------

# Brown 2015 ch. 5 1-factor CFA archetype: 5 indicators on one factor,
# standardized loadings ~ 0.7.
model_brown_1f5i <- function() {
  lam <- c(0.7, 0.75, 0.65, 0.7, 0.8)
  psi <- 1.0
  theta_res <- 1.0 - lam^2 * psi
  Sigma <- psi * tcrossprod(lam); diag(Sigma) <- diag(Sigma) + theta_res
  list(name = "brown_1f5i", p = 5L, Sigma = Sigma,
       model_syntax = "f =~ x1 + x2 + x3 + x4 + x5")
}

# Holzinger-Swineford 1939-style 3-factor correlated CFA: 3 indicators per
# factor, 9 indicators total. Standardized loadings around 0.7, factor
# correlations ~0.4.
model_hs_3f9i <- function() {
  Lam <- matrix(0, 9, 3)
  Lam[1:3, 1] <- c(0.8, 0.7, 0.75)
  Lam[4:6, 2] <- c(0.75, 0.7, 0.8)
  Lam[7:9, 3] <- c(0.7, 0.75, 0.7)
  Psi <- matrix(c(1.0, 0.4, 0.3,
                  0.4, 1.0, 0.4,
                  0.3, 0.4, 1.0), 3, 3)
  theta_res <- 1.0 - diag(Lam %*% Psi %*% t(Lam))
  Sigma <- Lam %*% Psi %*% t(Lam); diag(Sigma) <- diag(Sigma) + theta_res
  list(name = "hs1939_3f9i", p = 9L, Sigma = Sigma,
       model_syntax = paste(
         "f1 =~ x1 + x2 + x3",
         "f2 =~ x4 + x5 + x6",
         "f3 =~ x7 + x8 + x9",
         sep = "\n"))
}

models <- list(model_brown_1f5i(), model_hs_3f9i())

# -- Missingness (sb2005_mcar + mar from the paper's r-package) ------------

.sb2005_path <- function() {
  candidates <- c(
    file.path("..", "..", "papers", "pairwise-robust-sem", "r-package",
              "R", "missingness.R"),
    file.path("..", "..", "..", "papers", "pairwise-robust-sem", "r-package",
              "R", "missingness.R")
  )
  hits <- candidates[file.exists(candidates)]
  if (!length(hits)) {
    stop("Cannot locate papers/pairwise-robust-sem/r-package/R/missingness.R",
         call. = FALSE)
  }
  hits[[1L]]
}
sys.source(.sb2005_path(), envir = environment())

simulate_data <- function(n, Sigma, seed) {
  set.seed(seed)
  p <- nrow(Sigma)
  X <- matrix(rnorm(n * p), n, p) %*% chol(Sigma)
  colnames(X) <- paste0("x", seq_len(p))
  X
}

apply_missingness <- function(X, mechanism, rate, seed) {
  if (rate <= 0) return(list(X = X, mask = matrix(TRUE, nrow(X), ncol(X))))
  df <- as.data.frame(X)
  res <- if (mechanism == "mcar") {
    sb2005_mcar(df, rate = rate, intact = 1L, seed = seed)
  } else {
    sb2005_mar(df, rate = rate, predictors = 1:2, seed = seed, calibrate = TRUE)
  }
  list(X = as.matrix(res$data), mask = !res$mask)
}

# -- Main driver -----------------------------------------------------------

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  require_pkg("magmaan")
  require_pkg("lavaan")
  require_pkg("microbenchmark")
  ensure_results_dir()

  cat(sprintf("[exp09] n = %d, missing rate = %.2f, microbench times = %d\n",
              opts$n, opts$rate, opts$times))

  rows <- list()
  mechanisms <- c("mcar", "mar")
  for (m in models) {
    partable <- lavaan::lavaanify(m$model_syntax, fixed.x = FALSE,
                                  auto.var = TRUE, auto.fix.first = TRUE,
                                  auto.cov.lv.x = TRUE)
    for (mech in mechanisms) {
      seed <- opts$seed_base + 1000L * which(sapply(models, `[[`, "name") == m$name) +
              (if (mech == "mar") 1L else 0L)
      X <- simulate_data(opts$n, m$Sigma, seed)
      miss <- apply_missingness(X, mech, opts$rate, seed + 1L)
      mask <- miss$mask
      keep <- rowSums(mask) > 0L
      Xk <- miss$X[keep, , drop = FALSE]
      mk <- mask[keep, , drop = FALSE]; storage.mode(mk) <- "logical"

      pw <- magmaan::magmaan_core$data_pairwise_sample_stats(Xk, mk)
      sample_pw <- list(S = pw$S, mean = pw$mean, nobs = pw$nobs)

      cat(sprintf("[exp09] %s + %s: timing %d reps × 3 estimators\n",
                  m$name, toupper(mech), opts$times))
      bench <- microbenchmark::microbenchmark(
        sigma = magmaan::magmaan_core$estimate_gls(partable, sample_pw),
        ml    = magmaan::magmaan_core$estimate_ml(partable, sample_pw),
        gamma = magmaan::magmaan_core$estimate_gls_pairwise(partable, Xk, mk),
        times = opts$times, unit = "ms"
      )
      # microbenchmark returns nanoseconds in $time; convert to milliseconds.
      bench_df <- as.data.frame(bench)
      bench_df$time_ms <- bench_df$time / 1e6
      bench_df$model <- m$name
      bench_df$p <- m$p
      bench_df$mechanism <- mech
      bench_df$n_eff <- nrow(Xk)
      rows[[length(rows) + 1L]] <- bench_df
    }
  }
  fits <- do.call(rbind, rows)

  # Per-cell summary: median, MAD, ratio to fastest.
  summary_rows <- list()
  for (mdl in unique(fits$model)) {
    for (mech in unique(fits$mechanism)) {
      cell <- fits[fits$model == mdl & fits$mechanism == mech, ]
      med <- tapply(cell$time_ms, cell$expr, stats::median)
      mad <- tapply(cell$time_ms, cell$expr, stats::mad)
      fastest <- min(med)
      summary_rows[[length(summary_rows) + 1L]] <- data.frame(
        model = mdl, p = cell$p[[1L]], mechanism = mech, n_eff = cell$n_eff[[1L]],
        estimator = names(med),
        median_ms = as.numeric(med),
        mad_ms = as.numeric(mad),
        ratio_to_fastest = as.numeric(med) / fastest
      )
    }
  }
  summary_df <- do.call(rbind, summary_rows)
  rownames(summary_df) <- NULL

  fits_path    <- experiment_path("results", "fits.csv")
  summary_path <- experiment_path("results", "summary.csv")
  meta_path    <- experiment_path("results", "metadata.csv")
  write_csv(fits[, c("model", "p", "mechanism", "n_eff",
                     "expr", "time_ms")], fits_path)
  write_csv(summary_df, summary_path)
  write_metadata(
    meta_path,
    values = list(
      times      = opts$times,
      n          = opts$n,
      rate       = opts$rate,
      seed_base  = opts$seed_base,
      smoke      = opts$smoke,
      models     = paste(sapply(models, `[[`, "name"), collapse = ","),
      mechanisms = paste(mechanisms, collapse = ",")
    ),
    packages = c("magmaan", "lavaan", "microbenchmark")
  )
  cat(sprintf("\nWrote: %s\n       %s\n       %s\n",
              fits_path, summary_path, meta_path))
}

main()
