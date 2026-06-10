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

ALL_DESIGNS    <- c("cfa_1f9i", "cfa_1f5i", "cfa_3f9i")
ALL_MECHANISMS <- c("mcar_intact", "mcar_strat",
                    "mar_sb2005", "mar_obs_value")

parse_args <- function(args) {
  out <- list(
    reps = 200L,
    smoke = FALSE,
    seed_base = 20260610L,
    n_grid = c(100L, 200L, 500L, 1000L),
    miss_grid = c(0.0, 0.15, 0.30),
    designs = ALL_DESIGNS,
    mechanisms = ALL_MECHANISMS
  )
  parse_csv <- function(s, valid, what) {
    items <- strsplit(s, ",", fixed = TRUE)[[1L]]
    items <- trimws(items)
    bad <- setdiff(items, valid)
    if (length(bad))
      stop(sprintf("unknown %s: %s (valid: %s)",
                   what, paste(bad, collapse = ","),
                   paste(valid, collapse = ",")),
           call. = FALSE)
    items
  }
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--reps N] [--smoke] [--seed-base S]\n",
        "                                [--designs LIST] [--mechanisms LIST]\n",
        "                                [--n-grid LIST] [--miss-grid LIST]\n",
        "\n",
        "  --reps N         Monte Carlo replications per cell (default 200).\n",
        "  --smoke          Tiny run: 3 reps × {cfa_1f9i} × {mcar_intact,mar_sb2005}\n",
        "                   × n {100,300} × rate {0, 0.2}.\n",
        "  --seed-base S    Deterministic seed base (default 20260610).\n",
        "  --designs LIST   Comma-separated design names. Default: all four.\n",
        "  --mechanisms LIST  Comma-separated mechanism names. Default: all four.\n",
        "  --n-grid LIST    Comma-separated sample sizes (default 100,200,500,1000).\n",
        "  --miss-grid LIST Comma-separated missing rates (default 0,0.15,0.30).\n",
        "  --help           Show this message.\n",
        "\n",
        "Designs (std.lv parameterization throughout):\n",
        "  cfa_1f9i — 1 factor, 9 indicators (Study-0 calibration; loadings ~0.7).\n",
        "  cfa_1f5i — Brown 2015 ch.5 archetype: 1 factor, 5 indicators (heterog. loadings).\n",
        "  cfa_3f9i — Holzinger-Swineford 1939 shape: 3 correlated factors × 3 indicators.\n",
        "\n",
        "Missingness mechanisms:\n",
        "  mcar_intact   — SB-2005: x1 intact, MCAR-rate on rest. Uniform Ω over pairs.\n",
        "  mcar_strat    — Stratified MCAR: variable per-column rates, no intact column.\n",
        "                  Tests whether Γ_NT^pw advantage needs uniform Ω.\n",
        "  mar_sb2005    — SB-2005 3-rule MAR; predictors x1,x2 intact.\n",
        "  mar_obs_value — Standard textbook MAR: selection on observed column x_k\n",
        "                  (k randomly chosen per replication).\n",
        "\n",
        "Estimators: Σ-only GLS, Γ_NT^pw GLS, pairwise ML, two-stage ML, FIML.\n",
        "Outcome: trace of per-parameter empirical MSE; conditioning of\n",
        "         Γ_NT(Ŝ_pw) and Γ_NT^pw per fit.\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    } else if (arg == "--designs") {
      i <- i + 1L
      out$designs <- parse_csv(args[[i]], ALL_DESIGNS, "design")
    } else if (startsWith(arg, "--designs=")) {
      out$designs <- parse_csv(sub("^--designs=", "", arg), ALL_DESIGNS, "design")
    } else if (arg == "--mechanisms") {
      i <- i + 1L
      out$mechanisms <- parse_csv(args[[i]], ALL_MECHANISMS, "mechanism")
    } else if (startsWith(arg, "--mechanisms=")) {
      out$mechanisms <- parse_csv(sub("^--mechanisms=", "", arg),
                                  ALL_MECHANISMS, "mechanism")
    } else if (arg == "--n-grid") {
      i <- i + 1L
      out$n_grid <- as.integer(strsplit(args[[i]], ",", fixed = TRUE)[[1L]])
    } else if (startsWith(arg, "--n-grid=")) {
      out$n_grid <- as.integer(strsplit(sub("^--n-grid=", "", arg),
                                        ",", fixed = TRUE)[[1L]])
    } else if (arg == "--miss-grid") {
      i <- i + 1L
      out$miss_grid <- as.numeric(strsplit(args[[i]], ",", fixed = TRUE)[[1L]])
    } else if (startsWith(arg, "--miss-grid=")) {
      out$miss_grid <- as.numeric(strsplit(sub("^--miss-grid=", "", arg),
                                           ",", fixed = TRUE)[[1L]])
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
    out$reps <- 3L
    out$n_grid <- c(100L, 300L)
    out$miss_grid <- c(0.0, 0.20)
    out$designs <- "cfa_1f9i"
    out$mechanisms <- c("mcar_intact", "mar_sb2005")
  }
  out
}

# -- Design factories ------------------------------------------------------
# All designs use the std.lv parameterization (factor variances fixed = 1,
# all loadings free, factor covariances free for multi-factor designs).
# Under this convention the simulated population parameters (lam, factor
# correlations, residual variances) match the free-parameter truth directly,
# so per-parameter MSE is dominated by sampling variance + bias from missing
# data, not by a parameterization mismatch.
#
# Each factory returns:
#   name        — short identifier (used in CSV `design` column).
#   p           — number of observed variables.
#   syntax      — lavaan model syntax (passed to lavaanify with std.lv=TRUE).
#   Sigma       — population covariance matrix (used to simulate data).
#   truth       — named list of "true" free parameters indexed by the
#                 partable row signature `(op, lhs, rhs)`. Built once and
#                 looked up by `true_theta(partable, design)`.

design_cfa_1f9i <- function() {
  p <- 9L
  lam <- c(0.7, 0.75, 0.65, 0.7, 0.8, 0.6, 0.75, 0.7, 0.7)
  psi <- 1.0
  theta_res <- 1.0 - lam^2 * psi
  Sigma <- psi * tcrossprod(lam); diag(Sigma) <- diag(Sigma) + theta_res
  syntax <- paste0("f =~ ", paste(paste0("x", seq_len(p)), collapse = " + "))
  truth <- list()
  for (j in seq_len(p)) truth[[paste0("=~|f|x", j)]] <- lam[[j]]
  for (j in seq_len(p)) truth[[paste0("~~|x", j, "|x", j)]] <- theta_res[[j]]
  for (j in seq_len(p)) truth[[paste0("~1|x", j, "|")]] <- 0.0
  list(name = "cfa_1f9i", p = p, syntax = syntax, Sigma = Sigma, truth = truth)
}

design_cfa_1f5i <- function() {
  p <- 5L
  # Brown 2015 ch.5 archetype with heterogeneous loadings (0.5 .. 0.9 spread).
  lam <- c(0.9, 0.5, 0.85, 0.6, 0.75)
  psi <- 1.0
  theta_res <- 1.0 - lam^2 * psi
  Sigma <- psi * tcrossprod(lam); diag(Sigma) <- diag(Sigma) + theta_res
  syntax <- paste0("f =~ ", paste(paste0("x", seq_len(p)), collapse = " + "))
  truth <- list()
  for (j in seq_len(p)) truth[[paste0("=~|f|x", j)]] <- lam[[j]]
  for (j in seq_len(p)) truth[[paste0("~~|x", j, "|x", j)]] <- theta_res[[j]]
  for (j in seq_len(p)) truth[[paste0("~1|x", j, "|")]] <- 0.0
  list(name = "cfa_1f5i", p = p, syntax = syntax, Sigma = Sigma, truth = truth)
}

design_cfa_3f9i <- function() {
  # Holzinger-Swineford 1939 shape: 3 correlated factors × 3 indicators.
  # Factor variances = 1; loadings 0.7 within each factor block; factor
  # correlations 0.3.
  p <- 9L
  lam <- rep(0.7, p)
  factor_idx <- rep(1:3, each = 3L)
  Psi <- matrix(c(1.0, 0.3, 0.3,
                  0.3, 1.0, 0.3,
                  0.3, 0.3, 1.0), 3L, 3L)
  Lam <- matrix(0.0, p, 3L)
  for (j in seq_len(p)) Lam[j, factor_idx[[j]]] <- lam[[j]]
  Sigma <- Lam %*% Psi %*% t(Lam)
  theta_res <- 1.0 - lam^2  # ψ_ff = 1 for each factor in std.lv
  diag(Sigma) <- diag(Sigma) + theta_res
  syntax <- paste(
    "f1 =~ x1 + x2 + x3",
    "f2 =~ x4 + x5 + x6",
    "f3 =~ x7 + x8 + x9",
    sep = "\n")
  truth <- list()
  for (j in seq_len(p)) {
    fj <- paste0("f", factor_idx[[j]])
    truth[[paste0("=~|", fj, "|x", j)]] <- lam[[j]]
  }
  for (j in seq_len(p)) truth[[paste0("~~|x", j, "|x", j)]] <- theta_res[[j]]
  truth[["~~|f1|f2"]] <- Psi[1L, 2L]
  truth[["~~|f1|f3"]] <- Psi[1L, 3L]
  truth[["~~|f2|f3"]] <- Psi[2L, 3L]
  for (j in seq_len(p)) truth[[paste0("~1|x", j, "|")]] <- 0.0
  list(name = "cfa_3f9i", p = p, syntax = syntax, Sigma = Sigma, truth = truth)
}

DESIGN_FACTORIES <- list(
  cfa_1f9i = design_cfa_1f9i,
  cfa_1f5i = design_cfa_1f5i,
  cfa_3f9i = design_cfa_3f9i
)

simulate_data <- function(n, Sigma, seed) {
  set.seed(seed)
  p <- nrow(Sigma)
  L <- chol(Sigma)
  X <- matrix(rnorm(n * p), n, p) %*% L
  colnames(X) <- paste0("x", seq_len(p))
  X
}

# -- Missingness mechanisms ------------------------------------------------
# All return list(X = matrix-with-NA, mask = logical-matrix-of-observed).

# MCAR-intact (SB-2005 calibration): x1 stays observed; MCAR at `rate` on
# x2..xp. Uniform Ω over (j, k) pairs not involving x1.
apply_mcar_intact <- function(X, rate, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  if (rate <= 0) return(list(X = X, mask = matrix(TRUE, nrow(X), ncol(X))))
  n <- nrow(X); p <- ncol(X)
  target <- 2:p
  drop <- matrix(FALSE, n, p)
  drop[, target] <- runif(n * length(target)) < rate
  Xm <- X; Xm[drop] <- NA
  list(X = Xm, mask = !drop)
}

# MCAR-stratified: per-column rates spread around `rate` to break the
# uniform-Ω geometry that mcar_intact creates. No intact column; the rate
# at column j is `rate * (0.5 + j/p)` clamped to [0.05, 0.55].
apply_mcar_strat <- function(X, rate, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  if (rate <= 0) return(list(X = X, mask = matrix(TRUE, nrow(X), ncol(X))))
  n <- nrow(X); p <- ncol(X)
  per_col <- pmin(pmax(rate * (0.5 + seq_len(p) / p), 0.05), 0.55)
  drop <- matrix(FALSE, n, p)
  for (j in seq_len(p)) drop[, j] <- runif(n) < per_col[[j]]
  # Guarantee no fully-missing row: re-observe a random column for any row
  # that lost every variable (rare at sub-50% column rates but possible).
  fully_missing <- which(rowSums(!drop) == 0L)
  for (r in fully_missing) drop[r, sample.int(p, 1L)] <- FALSE
  Xm <- X; Xm[drop] <- NA
  list(X = Xm, mask = !drop)
}

# SB-2005 MAR mechanism from the shared experiment harness. The 3-rule design
# (calibrated to hit `rate` on average) ties missingness in target columns to
# two intact predictor columns.
source(support_path("R", "missingness.R"))

apply_mar_sb2005 <- function(X, rate, predictors = 1:2, seed = NULL) {
  if (rate <= 0) return(list(X = X, mask = matrix(TRUE, nrow(X), ncol(X))))
  df <- as.data.frame(X)
  res <- sb2005_mar(df, rate = rate, predictors = predictors, seed = seed,
                   calibrate = TRUE)
  list(X = as.matrix(res$data), mask = !res$mask)
}

# MAR-by-observed-value: standard textbook MAR. For each target column k,
# the per-row missingness logit is `α + β · z`, where z = standardized x_1
# (universal anchor that stays observed). The slope β is fixed at 1.5 (a
# moderate-to-strong dependence) and the intercept α is calibrated by binary
# search so the marginal missing rate ≈ `rate`. Different Ω geometry than
# the SB-2005 3-rule design — selection here uses a continuous anchor, not
# discrete rules.
apply_mar_obs_value <- function(X, rate, slope = 1.5, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  if (rate <= 0) return(list(X = X, mask = matrix(TRUE, nrow(X), ncol(X))))
  n <- nrow(X); p <- ncol(X)
  if (p < 2L) stop("apply_mar_obs_value needs p >= 2")
  z <- scale(X[, 1L])[, 1L]
  f <- function(alpha) mean(plogis(alpha + slope * z)) - rate
  alpha <- uniroot(f, c(-15, 15), tol = 1e-7)$root
  target_cols <- 2:p
  drop <- matrix(FALSE, n, p)
  for (k in target_cols) drop[, k] <- runif(n) < plogis(alpha + slope * z)
  Xm <- X; Xm[drop] <- NA
  list(X = Xm, mask = !drop)
}

apply_missingness <- function(X, mechanism, rate, seed) {
  if (rate <= 0) return(apply_mcar_intact(X, 0, seed = seed))
  switch(mechanism,
    mcar_intact   = apply_mcar_intact(X, rate, seed = seed),
    mcar_strat    = apply_mcar_strat(X, rate, seed = seed),
    mar_sb2005    = apply_mar_sb2005(X, rate, predictors = 1:2,
                                     seed = seed),
    mar_obs_value = apply_mar_obs_value(X, rate, seed = seed),
    stop("unknown mechanism: ", mechanism, call. = FALSE))
}

# Build a magmaan partable for a design using std.lv parameterization
# (factor variances fixed = 1, all loadings free). With std.lv the truth
# in simulation = truth in the partable's free-param order.
build_partable <- function(design) {
  lavaan::lavaanify(design$syntax, fixed.x = FALSE,
                    auto.var = TRUE, auto.fix.first = FALSE,
                    auto.cov.lv.x = TRUE,
                    std.lv = TRUE,
                    meanstructure = TRUE,
                    int.lv.free = FALSE, int.ov.free = TRUE)
}

# True θ in partable's free-parameter order, using the design's `truth`
# lookup keyed on the row signature `(op, lhs, rhs)`. With std.lv all
# loadings/variances/covariances are free at their simulated values; the
# intercepts are simulated to 0.
true_theta <- function(partable, design) {
  is_free <- partable$free > 0L
  ord <- partable$free[is_free]
  rows <- partable[is_free, , drop = FALSE]
  vals <- numeric(nrow(rows))
  for (i in seq_len(nrow(rows))) {
    r <- rows[i, ]
    key <- paste(r$op, r$lhs, r$rhs, sep = "|")
    # Symmetric off-diagonal covariances: try both orderings.
    key_swap <- paste(r$op, r$rhs, r$lhs, sep = "|")
    v <- design$truth[[key]]
    if (is.null(v)) v <- design$truth[[key_swap]]
    vals[i] <- if (is.null(v)) NA_real_ else v
  }
  vals[order(ord)]
}

# Fit all five estimators on one (X, mask) pair:
#   sigma — fit_gls with samp.S = Ŝ_pw                  (Σ-only weight)
#   gamma — fit_gls_pairwise(raw, pw)                   (Γ_NT^pw weight)
#   ml    — fit_ml      with samp.S = Ŝ_pw              (Savalei-Bentler 2005
#                                                        pairwise covariance ML;
#                                                        asymptotically first-order
#                                                        equivalent to sigma)
#   ts_ml — fit_ml on the saturated EM (μ̂_em, Σ̂_em)    (Savalei-Bentler 2009
#                                                        two-stage point estimate;
#                                                        first stage = saturated
#                                                        EM, second stage = ML
#                                                        on the EM moments)
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
  ts_ml_or <- tryCatch(pull_theta(
    magmaan::magmaan_core$estimate_two_stage_em(partable,
      list(X = X, mask = mask), kind = "ml")),
    error = function(e) na_theta)
  t5 <- Sys.time()
  fiml_or <- tryCatch(pull_theta(
    magmaan::magmaan_core$estimate_fiml(partable, list(X = X, mask = mask))),
    error = function(e) na_theta)
  t6 <- Sys.time()

  list(
    sigma = sigma_or, gamma = gamma_or, ml = ml_or,
    ts_ml = ts_ml_or, fiml = fiml_or,
    t_sigma = as.numeric(t2 - t1, units = "secs"),
    t_gamma = as.numeric(t3 - t2, units = "secs"),
    t_ml    = as.numeric(t4 - t3, units = "secs"),
    t_ts_ml = as.numeric(t5 - t4, units = "secs"),
    t_fiml  = as.numeric(t6 - t5, units = "secs"),
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

  # Pre-build designs & partables once.
  design_objs <- lapply(opts$designs, function(name) DESIGN_FACTORIES[[name]]())
  names(design_objs) <- opts$designs
  partables <- lapply(design_objs, build_partable)
  truths <- Map(true_theta, partables, design_objs)
  for (nm in names(truths)) {
    if (anyNA(truths[[nm]]))
      stop(sprintf("design %s: true_theta returned NA for some free params — ",
                   nm), "check design$truth keys vs partable rows",
           call. = FALSE)
  }
  n_par_per <- vapply(truths, length, integer(1L))

  cat(sprintf("[exp08] designs:       %s\n",
              paste(opts$designs, collapse = ", ")))
  cat(sprintf("[exp08] # free params: %s\n",
              paste(sprintf("%s=%d", names(n_par_per), n_par_per),
                    collapse = ", ")))
  cat(sprintf("[exp08] reps=%d, seed_base=%d\n",
              opts$reps, opts$seed_base))
  cat(sprintf("[exp08] n grid:        %s\n",
              paste(opts$n_grid, collapse = ", ")))
  cat(sprintf("[exp08] rate grid:     %s\n",
              paste(sprintf("%g", opts$miss_grid), collapse = ", ")))
  cat(sprintf("[exp08] mechanisms:    %s\n",
              paste(opts$mechanisms, collapse = ", ")))

  # rate = 0 cells are evaluated under a single (mcar_intact) representative —
  # all mechanisms collapse to complete data, so duplicate rows would add no
  # information.
  cells <- expand.grid(
    design = opts$designs, n = opts$n_grid, rate = opts$miss_grid,
    mechanism = opts$mechanisms, stringsAsFactors = FALSE
  )
  rep_mech <- opts$mechanisms[[1L]]
  cells <- cells[!(cells$rate == 0 & cells$mechanism != rep_mech), ,
                 drop = FALSE]

  fits_rows <- list()
  row_id <- 0L
  for (i in seq_len(nrow(cells))) {
    design_name <- cells$design[i]
    n <- cells$n[i]
    rate <- cells$rate[i]
    mech <- cells$mechanism[i]
    design <- design_objs[[design_name]]
    partable <- partables[[design_name]]
    theta_star <- truths[[design_name]]
    n_par <- n_par_per[[design_name]]
    mech_seed_offset <- match(mech, ALL_MECHANISMS) * 7L
    design_seed_offset <- match(design_name, ALL_DESIGNS) * 1009L

    for (rep in seq_len(opts$reps)) {
      seed <- opts$seed_base + 1000L * n +
              as.integer(round(rate * 100)) * 100L + rep +
              mech_seed_offset + design_seed_offset
      X <- simulate_data(n, design$Sigma, seed)
      miss <- apply_missingness(X, mech, rate, seed + 1L)
      keep <- rowSums(miss$mask) > 0L
      Xk <- miss$X[keep, , drop = FALSE]
      mk <- miss$mask[keep, , drop = FALSE]
      storage.mode(mk) <- "logical"

      res <- tryCatch(
        fit_triple(Xk, mk, partable, n_par),
        error = function(e) {
          warning(sprintf(
            "fit failed: design=%s n=%d rate=%.2f mech=%s rep=%d: %s",
            design_name, n, rate, mech, rep, conditionMessage(e)),
            call. = FALSE)
          NULL
        }
      )
      if (is.null(res)) next
      row_id <- row_id + 1L
      fits_rows[[length(fits_rows) + 1L]] <- data.frame(
        row_id = row_id,
        design = design_name,
        n = n, miss_rate = rate, mechanism = mech, rep = rep,
        par_idx = seq_len(n_par),
        theta_true = theta_star,
        est_sigma  = res$sigma,
        est_gamma  = res$gamma,
        est_ml     = res$ml,
        est_ts_ml  = res$ts_ml,
        est_fiml   = res$fiml,
        t_sigma    = res$t_sigma,
        t_gamma    = res$t_gamma,
        t_ml       = res$t_ml,
        t_ts_ml    = res$t_ts_ml,
        t_fiml     = res$t_fiml,
        cond_sigma = res$cond_sigma,
        cond_gamma = res$cond_gamma
      )
    }
    cat(sprintf("[exp08] cell design=%s n=%d rate=%.2f mech=%s done (%d reps)\n",
                design_name, n, rate, mech, opts$reps))
  }

  fits <- do.call(rbind, fits_rows)

  # Per-cell summary: empirical variance, bias², MSE, trace; plus conditioning
  # statistics so the report can talk about Γ_NT(Ŝ_pw) vs Γ_NT^pw geometry.
  summary_rows <- list()
  cells_uniq <- unique(fits[, c("design", "n", "miss_rate", "mechanism")])
  for (i in seq_len(nrow(cells_uniq))) {
    dn  <- cells_uniq$design[i]
    n   <- cells_uniq$n[i]
    rt  <- cells_uniq$miss_rate[i]
    mch <- cells_uniq$mechanism[i]
    cell <- fits[fits$design == dn & fits$n == n & fits$miss_rate == rt &
                 fits$mechanism == mch, , drop = FALSE]
    n_par <- n_par_per[[dn]]
    var_sigma <- numeric(n_par); var_gamma <- numeric(n_par)
    var_ml    <- numeric(n_par); var_ts_ml <- numeric(n_par)
    var_fiml  <- numeric(n_par)
    mse_sigma <- numeric(n_par); mse_gamma <- numeric(n_par)
    mse_ml    <- numeric(n_par); mse_ts_ml <- numeric(n_par)
    mse_fiml  <- numeric(n_par)
    for (k in seq_len(n_par)) {
      cl <- cell[cell$par_idx == k, ]
      var_sigma[k] <- stats::var(cl$est_sigma, na.rm = TRUE)
      var_gamma[k] <- stats::var(cl$est_gamma, na.rm = TRUE)
      var_ml[k]    <- stats::var(cl$est_ml,    na.rm = TRUE)
      var_ts_ml[k] <- stats::var(cl$est_ts_ml, na.rm = TRUE)
      var_fiml[k]  <- stats::var(cl$est_fiml,  na.rm = TRUE)
      bias_s <- mean(cl$est_sigma, na.rm = TRUE) - cl$theta_true[[1L]]
      bias_g <- mean(cl$est_gamma, na.rm = TRUE) - cl$theta_true[[1L]]
      bias_m <- mean(cl$est_ml,    na.rm = TRUE) - cl$theta_true[[1L]]
      bias_ts<- mean(cl$est_ts_ml, na.rm = TRUE) - cl$theta_true[[1L]]
      bias_f <- mean(cl$est_fiml,  na.rm = TRUE) - cl$theta_true[[1L]]
      mse_sigma[k] <- var_sigma[k] + bias_s^2
      mse_gamma[k] <- var_gamma[k] + bias_g^2
      mse_ml[k]    <- var_ml[k]    + bias_m^2
      mse_ts_ml[k] <- var_ts_ml[k] + bias_ts^2
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
      design = dn,
      n = n, miss_rate = rt, mechanism = mch,
      reps = length(rep_ids),
      reps_ml_ok    = sum(fin_ok("est_ml")),
      reps_sigma_ok = sum(fin_ok("est_sigma")),
      reps_gamma_ok = sum(fin_ok("est_gamma")),
      reps_ts_ml_ok = sum(fin_ok("est_ts_ml")),
      reps_fiml_ok  = sum(fin_ok("est_fiml")),
      trace_var_sigma = sum(var_sigma),
      trace_var_gamma = sum(var_gamma),
      trace_var_ml    = sum(var_ml),
      trace_var_ts_ml = sum(var_ts_ml),
      trace_var_fiml  = sum(var_fiml),
      trace_mse_sigma = sum(mse_sigma),
      trace_mse_gamma = sum(mse_gamma),
      trace_mse_ml    = sum(mse_ml),
      trace_mse_ts_ml = sum(mse_ts_ml),
      trace_mse_fiml  = sum(mse_fiml),
      eff_ratio_mse_sigma_vs_gamma = sum(mse_sigma) / sum(mse_gamma),
      eff_ratio_mse_ml_vs_gamma    = sum(mse_ml)    / sum(mse_gamma),
      eff_ratio_mse_ml_vs_sigma    = sum(mse_ml)    / sum(mse_sigma),
      # Ratios > 1 mean FIML is more efficient than the named pairwise
      # estimator (so it's the "FIML / pairwise" loss ratio).
      eff_ratio_mse_sigma_vs_fiml = sum(mse_sigma) / sum(mse_fiml),
      eff_ratio_mse_ml_vs_fiml    = sum(mse_ml)    / sum(mse_fiml),
      eff_ratio_mse_gamma_vs_fiml = sum(mse_gamma) / sum(mse_fiml),
      eff_ratio_mse_ts_ml_vs_fiml = sum(mse_ts_ml) / sum(mse_fiml),
      mean_t_sigma = mean(cell$t_sigma, na.rm = TRUE),
      mean_t_gamma = mean(cell$t_gamma, na.rm = TRUE),
      mean_t_ml    = mean(cell$t_ml,    na.rm = TRUE),
      mean_t_ts_ml = mean(cell$t_ts_ml, na.rm = TRUE),
      mean_t_fiml  = mean(cell$t_fiml,  na.rm = TRUE),
      median_cond_sigma = stats::median(cond_cells$cond_sigma),
      median_cond_gamma = stats::median(cond_cells$cond_gamma),
      q95_cond_sigma    = stats::quantile(cond_cells$cond_sigma, 0.95, names = FALSE),
      q95_cond_gamma    = stats::quantile(cond_cells$cond_gamma, 0.95, names = FALSE)
    )
  }
  summary_df <- do.call(rbind, summary_rows)
  summary_df <- summary_df[order(summary_df$design, summary_df$n,
                                 summary_df$miss_rate,
                                 summary_df$mechanism), ]

  fits_path    <- experiment_path("results", "fits.csv")
  summary_path <- experiment_path("results", "summary.csv")
  meta_path    <- experiment_path("results", "metadata.csv")
  write_csv(fits, fits_path)
  write_csv(summary_df, summary_path)
  write_metadata(
    meta_path,
    values = list(
      reps        = opts$reps,
      smoke       = opts$smoke,
      seed_base   = opts$seed_base,
      n_grid      = paste(opts$n_grid, collapse = ","),
      miss_grid   = paste(sprintf("%g", opts$miss_grid), collapse = ","),
      designs     = paste(opts$designs, collapse = ","),
      mechanisms  = paste(opts$mechanisms, collapse = ","),
      n_free_per_design = paste(sprintf("%s=%d",
                                        names(n_par_per), n_par_per),
                                collapse = ",")
    ),
    packages = c("magmaan", "lavaan")
  )
  cat(sprintf("\nWrote: %s\n       %s\n       %s\n",
              fits_path, summary_path, meta_path))
}

main()
