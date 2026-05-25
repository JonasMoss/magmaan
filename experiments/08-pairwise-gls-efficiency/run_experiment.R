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
    miss_grid = c(0.0, 0.15, 0.30),
    mechanisms = c("mcar", "mar")
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--reps N] [--smoke] [--seed-base S]\n",
        "                                [--mcar-only] [--mar-only]\n",
        "\n",
        "  --reps N         Monte Carlo replications per cell (default 200).\n",
        "  --smoke          Tiny run: 5 reps, n grid {100, 300}, rates {0, 0.2}.\n",
        "  --seed-base S    Deterministic seed base (default 20260610).\n",
        "  --mcar-only      Skip MAR cells.\n",
        "  --mar-only       Skip MCAR cells (rate=0 still included for the\n",
        "                   degeneracy reference, evaluated as MCAR).\n",
        "  --help           Show this message.\n",
        "\n",
        "Design: 1-factor CFA, 9 indicators (Study-0 calibration);\n",
        "  n         ∈ {200, 500, 1000}\n",
        "  rate      ∈ {0%, 15%, 30%}\n",
        "  mechanism ∈ {MCAR, MAR}  (Savalei-Bentler 2005)\n",
        "    MCAR: intact = x1, MCAR rate on x2..x9.\n",
        "    MAR:  predictors = x1, x2 (intact); 3-rule design from\n",
        "          papers/pairwise-robust-sem missingness.R.\n",
        "  estimators: GLSpw_Sigma (Σ-only weight), GLSpw_Gamma (Γ_NT^pw weight)\n",
        "  outcome: trace of per-parameter empirical MSE; conditioning of\n",
        "          Γ_NT(Ŝ_pw) and Γ_NT^pw per fit.\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    } else if (arg == "--mcar-only") {
      out$mechanisms <- "mcar"
    } else if (arg == "--mar-only") {
      out$mechanisms <- "mar"
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

# Savalei-Bentler 2005 MAR: sourced from the paper's r-package. The 3-rule
# design (calibrated to hit `rate` on average) ties missingness in target
# columns to two intact predictor columns. Different Ω structure than MCAR —
# overlaps are not uniform across pairs, so this is the test of whether the
# Γ_NT^pw conditioning advantage from MCAR-with-intact is a generic effect
# or a happenstance of uniform-Ω geometry.
.sb2005_mar_path <- function() {
  candidates <- c(
    file.path("..", "..", "papers", "pairwise-robust-sem", "r-package",
              "R", "missingness.R"),
    file.path("..", "..", "..", "papers", "pairwise-robust-sem", "r-package",
              "R", "missingness.R")
  )
  hits <- candidates[file.exists(candidates)]
  if (!length(hits)) {
    stop("Cannot locate papers/pairwise-robust-sem/r-package/R/missingness.R; ",
         "checked: ", paste(candidates, collapse = "; "), call. = FALSE)
  }
  hits[[1L]]
}
sys.source(.sb2005_mar_path(), envir = environment())

apply_sb2005_mar <- function(X, rate, predictors = 1:2, seed = NULL) {
  if (rate <= 0) return(list(X = X, mask = matrix(TRUE, nrow(X), ncol(X))))
  df <- as.data.frame(X)
  res <- sb2005_mar(df, rate = rate, predictors = predictors, seed = seed,
                   calibrate = TRUE)
  list(X = as.matrix(res$data), mask = !res$mask)
}

apply_missingness <- function(X, mechanism, rate, seed) {
  if (rate <= 0 || identical(mechanism, "mcar"))
    return(apply_sb2005_mcar(X, rate, intact = 1L, seed = seed))
  apply_sb2005_mar(X, rate, predictors = 1:2, seed = seed)
}

# Build a magmaan partable for "f =~ x1 + ... + xp" via lavaanify, with
# mean structure included — required for FIML, and harmless for ML/GLS
# (intercepts enter as additional free parameters with true value 0). This
# keeps the partable identical across all four estimators so their
# per-parameter estimates are directly comparable.
build_partable <- function(p) {
  rhs <- paste(paste0("x", seq_len(p)), collapse = " + ")
  model <- paste0("f =~ ", rhs)
  lavaan::lavaanify(model, fixed.x = FALSE,
                    auto.var = TRUE, auto.fix.first = TRUE,
                    auto.cov.lv.x = TRUE,
                    meanstructure = TRUE,
                    int.lv.free = FALSE, int.ov.free = TRUE)
}

# True θ in the (lavaanify) free-parameter order. Intercepts have true
# value 0 because the simulated data is mean-centred.
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
    } else if (r$op == "~1") {
      vals[i] <- 0.0
    } else {
      vals[i] <- NA_real_
    }
  }
  vals[order(ord)]
}

# Fit all four estimators on one (X, mask) pair:
#   sigma — fit_gls with samp.S = Ŝ_pw                  (Σ-only weight)
#   gamma — fit_gls_pairwise(raw, pw)                   (Γ_NT^pw weight)
#   ml    — fit_ml      with samp.S = Ŝ_pw              (Savalei-Bentler 2005
#                                                        pairwise covariance ML;
#                                                        asymptotically first-order
#                                                        equivalent to sigma)
#   fiml  — fit_fiml(partable, raw_data)                (oracle: direct
#                                                        missing-data ML; uses
#                                                        the full incomplete-data
#                                                        likelihood, not a
#                                                        pairwise summary)
# Also records cond(Γ_NT(Ŝ_pw)) and cond(Γ_NT^pw) — the conditioning
# diagnostic the story turns on. Returns NA estimates for any path that
# errored (e.g. ML when Ŝ_pw is non-PD).
fit_triple <- function(X, mask, partable, n_par) {
  pw <- magmaan::magmaan_core$data_pairwise_sample_stats(X, mask)
  sample_pw <- list(S = pw$S, mean = pw$mean, nobs = pw$nobs)

  S_pw <- pw$S[[1L]]
  G_sigma <- magmaan::magmaan_core$robust_gamma_nt(S_pw)
  G_pw    <- magmaan::magmaan_core$data_gamma_nt_pairwise(X, mask)[[1L]]
  cond_of <- function(M) {
    e <- eigen(M, symmetric = TRUE, only.values = TRUE)$values
    max(e) / max(min(e), .Machine$double.eps)
  }
  c_sigma <- cond_of(G_sigma)
  c_gamma <- cond_of(G_pw)

  pull_theta <- function(fit) {
    pt <- fit$partable
    pt$est[pt$free > 0][order(pt$free[pt$free > 0])]
  }
  na_theta <- rep(NA_real_, n_par)

  t1 <- Sys.time()
  sigma_or <- tryCatch(pull_theta(
    magmaan::magmaan_core$estimate_gls(partable, sample_pw)),
    error = function(e) na_theta)
  t2 <- Sys.time()
  gamma_or <- tryCatch(pull_theta(
    magmaan::magmaan_core$estimate_gls_pairwise(partable, X, mask)),
    error = function(e) na_theta)
  t3 <- Sys.time()
  ml_or <- tryCatch(pull_theta(
    magmaan::magmaan_core$estimate_ml(partable, sample_pw)),
    error = function(e) na_theta)
  t4 <- Sys.time()
  fiml_or <- tryCatch(pull_theta(
    magmaan::magmaan_core$estimate_fiml(partable, list(X = X, mask = mask))),
    error = function(e) na_theta)
  t5 <- Sys.time()

  list(
    sigma = sigma_or, gamma = gamma_or, ml = ml_or, fiml = fiml_or,
    t_sigma = as.numeric(t2 - t1, units = "secs"),
    t_gamma = as.numeric(t3 - t2, units = "secs"),
    t_ml    = as.numeric(t4 - t3, units = "secs"),
    t_fiml  = as.numeric(t5 - t4, units = "secs"),
    cond_sigma = c_sigma,
    cond_gamma = c_gamma
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
  cat(sprintf("[exp08] n grid:        %s\n",
              paste(opts$n_grid, collapse = ", ")))
  cat(sprintf("[exp08] rate grid:     %s\n",
              paste(sprintf("%g", opts$miss_grid), collapse = ", ")))
  cat(sprintf("[exp08] mechanisms:    %s\n",
              paste(opts$mechanisms, collapse = ", ")))

  # rate = 0 is always evaluated as MCAR (degeneracy reference); MAR at
  # rate = 0 is identical and skipped to avoid duplicate rows.
  cells <- expand.grid(
    n = opts$n_grid, rate = opts$miss_grid,
    mechanism = opts$mechanisms, stringsAsFactors = FALSE
  )
  cells <- cells[!(cells$mechanism == "mar" & cells$rate == 0), , drop = FALSE]

  fits_rows <- list()
  row_id <- 0L
  for (i in seq_len(nrow(cells))) {
    n <- cells$n[i]
    rate <- cells$rate[i]
    mech <- cells$mechanism[i]
    for (rep in seq_len(opts$reps)) {
      seed <- opts$seed_base + 1000L * n +
              as.integer(round(rate * 100)) * 100L + rep +
              (if (mech == "mar") 7L else 0L)  # decorrelate MAR/MCAR seeds
      X <- simulate_cfa(n, Sigma, seed)
      miss <- apply_missingness(X, mech, rate, seed + 1L)
      keep <- rowSums(miss$mask) > 0L
      Xk <- miss$X[keep, , drop = FALSE]
      mk <- miss$mask[keep, , drop = FALSE]
      storage.mode(mk) <- "logical"

      res <- tryCatch(
        fit_triple(Xk, mk, partable, n_par),
        error = function(e) {
          warning(sprintf("fit failed: n=%d rate=%.2f mech=%s rep=%d: %s",
                          n, rate, mech, rep, conditionMessage(e)),
                  call. = FALSE)
          NULL
        }
      )
      if (is.null(res)) next
      row_id <- row_id + 1L
      fits_rows[[length(fits_rows) + 1L]] <- data.frame(
        row_id = row_id,
        n = n, miss_rate = rate, mechanism = mech, rep = rep,
        par_idx = seq_len(n_par),
        theta_true = theta_star,
        est_sigma  = res$sigma,
        est_gamma  = res$gamma,
        est_ml     = res$ml,
        est_fiml   = res$fiml,
        t_sigma    = res$t_sigma,
        t_gamma    = res$t_gamma,
        t_ml       = res$t_ml,
        t_fiml     = res$t_fiml,
        cond_sigma = res$cond_sigma,
        cond_gamma = res$cond_gamma
      )
    }
    cat(sprintf("[exp08] cell n=%d rate=%.2f mech=%s done (%d reps)\n",
                n, rate, mech, opts$reps))
  }

  fits <- do.call(rbind, fits_rows)

  # Per-cell summary: empirical variance, bias², MSE, trace; plus conditioning
  # statistics so the report can talk about Γ_NT(Ŝ_pw) vs Γ_NT^pw geometry.
  summary_rows <- list()
  cells_uniq <- unique(fits[, c("n", "miss_rate", "mechanism")])
  for (i in seq_len(nrow(cells_uniq))) {
    n   <- cells_uniq$n[i]
    rt  <- cells_uniq$miss_rate[i]
    mch <- cells_uniq$mechanism[i]
    cell <- fits[fits$n == n & fits$miss_rate == rt &
                 fits$mechanism == mch, , drop = FALSE]
    var_sigma <- numeric(n_par); var_gamma <- numeric(n_par)
    var_ml    <- numeric(n_par); var_fiml  <- numeric(n_par)
    mse_sigma <- numeric(n_par); mse_gamma <- numeric(n_par)
    mse_ml    <- numeric(n_par); mse_fiml  <- numeric(n_par)
    for (k in seq_len(n_par)) {
      cl <- cell[cell$par_idx == k, ]
      var_sigma[k] <- stats::var(cl$est_sigma, na.rm = TRUE)
      var_gamma[k] <- stats::var(cl$est_gamma, na.rm = TRUE)
      var_ml[k]    <- stats::var(cl$est_ml,    na.rm = TRUE)
      var_fiml[k]  <- stats::var(cl$est_fiml,  na.rm = TRUE)
      bias_s <- mean(cl$est_sigma, na.rm = TRUE) - cl$theta_true[[1L]]
      bias_g <- mean(cl$est_gamma, na.rm = TRUE) - cl$theta_true[[1L]]
      bias_m <- mean(cl$est_ml,    na.rm = TRUE) - cl$theta_true[[1L]]
      bias_f <- mean(cl$est_fiml,  na.rm = TRUE) - cl$theta_true[[1L]]
      mse_sigma[k] <- var_sigma[k] + bias_s^2
      mse_gamma[k] <- var_gamma[k] + bias_g^2
      mse_ml[k]    <- var_ml[k]    + bias_m^2
      mse_fiml[k]  <- var_fiml[k]  + bias_f^2
    }
    cond_cells <- cell[!duplicated(cell$rep), c("cond_sigma", "cond_gamma")]
    # Count reps where each estimator returned all-finite θ̂ (per-param NA
    # in any row of a rep is treated as a failed rep for that estimator).
    rep_ids <- unique(cell$rep)
    fin_ok <- function(col) {
      vapply(rep_ids, function(r) {
        v <- cell[[col]][cell$rep == r]
        all(is.finite(v))
      }, logical(1L))
    }
    summary_rows[[i]] <- data.frame(
      n = n, miss_rate = rt, mechanism = mch,
      reps = length(rep_ids),
      reps_ml_ok    = sum(fin_ok("est_ml")),
      reps_sigma_ok = sum(fin_ok("est_sigma")),
      reps_gamma_ok = sum(fin_ok("est_gamma")),
      reps_fiml_ok  = sum(fin_ok("est_fiml")),
      trace_var_sigma = sum(var_sigma),
      trace_var_gamma = sum(var_gamma),
      trace_var_ml    = sum(var_ml),
      trace_var_fiml  = sum(var_fiml),
      trace_mse_sigma = sum(mse_sigma),
      trace_mse_gamma = sum(mse_gamma),
      trace_mse_ml    = sum(mse_ml),
      trace_mse_fiml  = sum(mse_fiml),
      eff_ratio_mse_sigma_vs_gamma = sum(mse_sigma) / sum(mse_gamma),
      eff_ratio_mse_ml_vs_gamma    = sum(mse_ml)    / sum(mse_gamma),
      eff_ratio_mse_ml_vs_sigma    = sum(mse_ml)    / sum(mse_sigma),
      # Ratios > 1 mean FIML is more efficient than the named pairwise
      # estimator (so it's the "FIML / pairwise" loss ratio).
      eff_ratio_mse_sigma_vs_fiml = sum(mse_sigma) / sum(mse_fiml),
      eff_ratio_mse_ml_vs_fiml    = sum(mse_ml)    / sum(mse_fiml),
      eff_ratio_mse_gamma_vs_fiml = sum(mse_gamma) / sum(mse_fiml),
      mean_t_sigma = mean(cell$t_sigma, na.rm = TRUE),
      mean_t_gamma = mean(cell$t_gamma, na.rm = TRUE),
      mean_t_ml    = mean(cell$t_ml,    na.rm = TRUE),
      mean_t_fiml  = mean(cell$t_fiml,  na.rm = TRUE),
      median_cond_sigma = stats::median(cond_cells$cond_sigma),
      median_cond_gamma = stats::median(cond_cells$cond_gamma),
      q95_cond_sigma    = stats::quantile(cond_cells$cond_sigma, 0.95, names = FALSE),
      q95_cond_gamma    = stats::quantile(cond_cells$cond_gamma, 0.95, names = FALSE)
    )
  }
  summary_df <- do.call(rbind, summary_rows)
  summary_df <- summary_df[order(summary_df$n, summary_df$miss_rate,
                                 summary_df$mechanism), ]

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
