#!/usr/bin/env Rscript

# Experiment 08 — pairwise GLS efficiency.
# Two estimators on the Savalei-Bentler 2005 Study-0 CFA design under MCAR:
#   Σ-only weight   — fit_gls(samp.S = Ŝ_pw)        (literature default)
#   Γ_NT^pw weight  — fit_gls_pairwise(raw, pw)     (asymptotically efficient)
# Outcome: trace of empirical θ̂ MSE summed over free parameters, per cell.

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

# -- CLI parsing -----------------------------------------------------------

parse_args <- function(args) {
  out <- list(
    reps = 200L,
    smoke = FALSE,
    seed_base = 20260610L,
    n_grid = c(200L, 500L, 1000L),
    miss_grid = c(0.0, 0.15, 0.30)
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--reps N] [--smoke] [--seed-base S]\n",
        "\n",
        "  --reps N         Monte Carlo replications per cell (default 200).\n",
        "  --smoke          Tiny run: 5 reps, n grid {100, 300}, missing grid {0, 0.2}.\n",
        "  --seed-base S    Deterministic seed base (default 20260610).\n",
        "  --help           Show this message.\n",
        "\n",
        "Design: 1-factor CFA, 9 indicators (Study-0 calibration);\n",
        "  n   ∈ {200, 500, 1000}\n",
        "  rate ∈ {0%, 15%, 30%}   (Savalei-Bentler 2005 MCAR; intact = 1)\n",
        "  estimators: GLSpw_Sigma (Σ-only weight), GLSpw_Gamma (Γ_NT^pw weight)\n",
        "  outcome: trace of per-parameter empirical MSE\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    } else if (arg == "--reps") {
      i <- i + 1L
      out$reps <- as.integer(args[[i]])
    } else if (startsWith(arg, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", arg))
    } else if (arg == "--smoke") {
      out$smoke <- TRUE
    } else if (arg == "--seed-base") {
      i <- i + 1L
      out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(arg, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", arg))
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  if (out$smoke) {
    out$reps <- 5L
    out$n_grid <- c(100L, 300L)
    out$miss_grid <- c(0.0, 0.20)
  }
  out
}

# -- True model: 1-factor CFA, 9 indicators (Study-0 calibration) ----------
# Standardized loadings clustered around 0.7. Residuals chosen so each
# indicator has unit total variance. Factor variance = 1, no means.
true_model <- function() {
  p <- 9L
  lam <- c(0.7, 0.75, 0.65, 0.7, 0.8, 0.6, 0.75, 0.7, 0.7)
  psi <- 1.0
  theta_res <- 1.0 - lam^2 * psi
  list(p = p, lam = lam, psi = psi, theta_res = theta_res)
}

# Population Σ; sample mvn from it.
population_sigma <- function(m) {
  Sigma <- m$psi * tcrossprod(m$lam)
  diag(Sigma) <- diag(Sigma) + m$theta_res
  Sigma
}

simulate_cfa <- function(n, Sigma, seed) {
  set.seed(seed)
  p <- nrow(Sigma)
  L <- chol(Sigma)
  X <- matrix(rnorm(n * p), n, p) %*% L
  colnames(X) <- paste0("x", seq_len(p))
  X
}

# Savalei-Bentler 2005 MCAR: keep `intact` columns observed, apply MCAR at
# `rate` to the rest. Mirrors papers/pairwise-robust-sem/r-package/R/missingness.R
# but inlined to keep this experiment self-contained.
apply_sb2005_mcar <- function(X, rate, intact = 1L, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  if (rate <= 0) return(list(X = X, mask = matrix(TRUE, nrow(X), ncol(X))))
  n <- nrow(X); p <- ncol(X)
  target <- setdiff(seq_len(p), intact)
  drop <- matrix(FALSE, n, p)
  drop[, target] <- runif(n * length(target)) < rate
  # Guarantee no fully-missing row (intact column already observed, so safe).
  Xm <- X
  Xm[drop] <- NA
  list(X = Xm, mask = !drop)
}

# Build a magmaan partable for "f =~ x1 + ... + xp" via lavaanify.
build_partable <- function(p) {
  rhs <- paste(paste0("x", seq_len(p)), collapse = " + ")
  model <- paste0("f =~ ", rhs)
  lavaan::lavaanify(model, fixed.x = FALSE,
                    auto.var = TRUE, auto.fix.first = TRUE,
                    auto.cov.lv.x = TRUE)
}

# True θ in the (lavaanify) free-parameter order.
true_theta <- function(partable, m) {
  is_free <- partable$free > 0L
  ord <- partable$free[is_free]
  rows <- partable[is_free, , drop = FALSE]
  vals <- numeric(nrow(rows))
  for (i in seq_len(nrow(rows))) {
    r <- rows[i, ]
    if (r$op == "=~") {
      idx <- as.integer(sub("^x", "", r$rhs))
      vals[i] <- m$lam[idx]
    } else if (r$op == "~~" && r$lhs == "f" && r$rhs == "f") {
      vals[i] <- m$psi
    } else if (r$op == "~~" && r$lhs == r$rhs && r$lhs != "f") {
      idx <- as.integer(sub("^x", "", r$lhs))
      vals[i] <- m$theta_res[idx]
    } else {
      vals[i] <- NA_real_
    }
  }
  vals[order(ord)]
}

# Fit both estimators on one (X, mask) pair. Returns named list of two
# numeric vectors (θ̂ in free-parameter order) plus per-fit wall times.
fit_both <- function(X, mask, partable) {
  pw <- magmaan::magmaan_core$data_pairwise_sample_stats(X, mask)
  sample_pw <- list(S = pw$S, mean = pw$mean, nobs = pw$nobs)

  t1 <- Sys.time()
  fit_sigma <- magmaan::magmaan_core$estimate_gls(partable, sample_pw)
  t2 <- Sys.time()
  fit_gamma <- magmaan::magmaan_core$estimate_gls_pairwise(partable, X, mask)
  t3 <- Sys.time()

  pt_s <- fit_sigma$partable
  pt_g <- fit_gamma$partable
  theta_sigma <- pt_s$est[pt_s$free > 0][order(pt_s$free[pt_s$free > 0])]
  theta_gamma <- pt_g$est[pt_g$free > 0][order(pt_g$free[pt_g$free > 0])]

  list(
    sigma = theta_sigma,
    gamma = theta_gamma,
    t_sigma = as.numeric(t2 - t1, units = "secs"),
    t_gamma = as.numeric(t3 - t2, units = "secs")
  )
}

# -- Main driver -----------------------------------------------------------

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  require_pkg("magmaan")
  require_pkg("lavaan")

  ensure_results_dir()

  m <- true_model()
  Sigma <- population_sigma(m)
  partable <- build_partable(m$p)
  theta_star <- true_theta(partable, m)
  n_par <- length(theta_star)

  cat(sprintf("[exp08] p=%d, %d free params, %d reps, seed_base=%d\n",
              m$p, n_par, opts$reps, opts$seed_base))
  cat(sprintf("[exp08] n grid:    %s\n",    paste(opts$n_grid, collapse = ", ")))
  cat(sprintf("[exp08] miss grid: %s\n",
              paste(sprintf("%g", opts$miss_grid), collapse = ", ")))

  fits_rows <- list()
  row_id <- 0L
  for (n in opts$n_grid) {
    for (rate in opts$miss_grid) {
      for (rep in seq_len(opts$reps)) {
        seed <- opts$seed_base + 1000L * n + as.integer(round(rate * 100)) * 100L + rep
        X <- simulate_cfa(n, Sigma, seed)
        miss <- apply_sb2005_mcar(X, rate, intact = 1L, seed = seed + 1L)
        # Guarantee no fully-missing row (already true with intact = 1, but
        # belt-and-braces).
        keep <- rowSums(miss$mask) > 0L
        Xk <- miss$X[keep, , drop = FALSE]
        mk <- miss$mask[keep, , drop = FALSE]
        storage.mode(mk) <- "logical"

        res <- tryCatch(
          fit_both(Xk, mk, partable),
          error = function(e) {
            warning(sprintf("fit failed: n=%d rate=%.2f rep=%d: %s",
                            n, rate, rep, conditionMessage(e)),
                    call. = FALSE)
            NULL
          }
        )
        if (is.null(res)) next
        row_id <- row_id + 1L
        fits_rows[[length(fits_rows) + 1L]] <- data.frame(
          row_id = row_id,
          n = n, miss_rate = rate, rep = rep,
          par_idx = seq_len(n_par),
          theta_true = theta_star,
          est_sigma  = res$sigma,
          est_gamma  = res$gamma,
          t_sigma    = res$t_sigma,
          t_gamma    = res$t_gamma
        )
      }
      cat(sprintf("[exp08] cell n=%d rate=%.2f done (%d reps)\n",
                  n, rate, opts$reps))
    }
  }

  fits <- do.call(rbind, fits_rows)

  # Per-cell summary: empirical variance, bias², MSE, trace.
  summary_rows <- list()
  cells <- unique(fits[, c("n", "miss_rate")])
  for (i in seq_len(nrow(cells))) {
    n  <- cells$n[i]
    rt <- cells$miss_rate[i]
    cell <- fits[fits$n == n & fits$miss_rate == rt, , drop = FALSE]
    var_sigma <- numeric(n_par); var_gamma <- numeric(n_par)
    mse_sigma <- numeric(n_par); mse_gamma <- numeric(n_par)
    for (k in seq_len(n_par)) {
      cl <- cell[cell$par_idx == k, ]
      var_sigma[k] <- stats::var(cl$est_sigma)
      var_gamma[k] <- stats::var(cl$est_gamma)
      bias_s <- mean(cl$est_sigma) - cl$theta_true[[1L]]
      bias_g <- mean(cl$est_gamma) - cl$theta_true[[1L]]
      mse_sigma[k] <- var_sigma[k] + bias_s^2
      mse_gamma[k] <- var_gamma[k] + bias_g^2
    }
    summary_rows[[i]] <- data.frame(
      n = n, miss_rate = rt,
      reps = length(unique(cell$rep)),
      trace_var_sigma = sum(var_sigma),
      trace_var_gamma = sum(var_gamma),
      trace_mse_sigma = sum(mse_sigma),
      trace_mse_gamma = sum(mse_gamma),
      eff_ratio_var = sum(var_sigma) / sum(var_gamma),
      eff_ratio_mse = sum(mse_sigma) / sum(mse_gamma),
      mean_t_sigma = mean(cell$t_sigma, na.rm = TRUE),
      mean_t_gamma = mean(cell$t_gamma, na.rm = TRUE)
    )
  }
  summary_df <- do.call(rbind, summary_rows)
  summary_df <- summary_df[order(summary_df$n, summary_df$miss_rate), ]

  fits_path    <- experiment_path("results", "fits.csv")
  summary_path <- experiment_path("results", "summary.csv")
  meta_path    <- experiment_path("results", "metadata.csv")
  write_csv(fits, fits_path)
  write_csv(summary_df, summary_path)
  write_metadata(
    meta_path,
    values = list(
      reps      = opts$reps,
      smoke     = opts$smoke,
      seed_base = opts$seed_base,
      n_grid    = paste(opts$n_grid, collapse = ","),
      miss_grid = paste(sprintf("%g", opts$miss_grid), collapse = ","),
      p         = m$p,
      n_free    = n_par
    ),
    packages = c("magmaan", "lavaan")
  )
  cat(sprintf("\nWrote: %s\n       %s\n       %s\n",
              fits_path, summary_path, meta_path))
}

main()
