#!/usr/bin/env Rscript
# Maydeu-Olivares (2017) replication.
#
# Two-factor CFA, fit unidimensional. Continuous-data ML with seven SE methods
# and five χ² adjustments — all from magmaan's own ML inference, with an
# optional lavaan parity check on the first replicate of each cell.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--cells FILTER] [--seed-base S]
#                            [--lavaan-parity]
#
# `--cells FILTER` is a comma-separated list of `key=value` pairs (e.g.
# `n_items=16,N=200`) that restrict the 54-cell crossed design to a subset.

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x
}

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE))
  }
  normalizePath(sys.frames()[[1L]]$ofile %||% "run_experiment.R",
                mustWork = FALSE)
}

parse_args <- function(args) {
  out <- list(reps = 10L, cells_filter = NULL, seed_base = 20260524L,
              lavaan_parity = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--cells FILTER] ",
          "[--seed-base S] [--lavaan-parity]\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--cells") {
      i <- i + 1L; out$cells_filter <- args[[i]]
    } else if (startsWith(a, "--cells=")) {
      out$cells_filter <- sub("^--cells=", "", a)
    } else if (a == "--seed-base") {
      i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--lavaan-parity") {
      out$lavaan_parity <- TRUE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(out$reps) || out$reps < 1L) {
    stop("--reps must be a positive integer", call. = FALSE)
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

experiment_dir <- dirname(script_path())
project_root <- normalizePath(file.path(experiment_dir, "..", ".."),
                              mustWork = TRUE)
results_dir <- file.path(experiment_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

for (v in c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")) {
  do.call(Sys.setenv, stats::setNames(list("1"), v))
}

require_pkg <- function(p) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop("required R package not installed: ", p, call. = FALSE)
  }
  invisible(TRUE)
}
require_pkg("magmaan")
core <- magmaan::magmaan_core

# ── Population covariance ───────────────────────────────────────────────────
# Two-factor CFA. Loadings cycle through {0.5, 0.6, 0.7, 0.8}, n_items/2 per
# factor. Residual variances chosen so each indicator has unit variance.
# Caches the Cholesky factor so the per-rep sampler doesn't re-factorise.
build_population <- function(n_items, rho) {
  stopifnot(n_items %% 2 == 0L)
  loadings_cycle <- c(0.5, 0.6, 0.7, 0.8)
  half <- as.integer(n_items / 2L)
  per_factor <- rep(loadings_cycle, length.out = half)
  lambdas <- c(per_factor, per_factor)
  Lambda <- matrix(0.0, n_items, 2)
  Lambda[seq_len(half), 1L] <- per_factor
  Lambda[(half + 1L):n_items, 2L] <- per_factor
  Phi <- matrix(c(1, rho, rho, 1), 2, 2)
  resid_vars <- 1 - lambdas^2
  Theta <- diag(resid_vars, n_items)
  Sigma <- Lambda %*% Phi %*% t(Lambda) + Theta
  list(Sigma = Sigma, L = chol(Sigma), lambdas = lambdas,
       n_items = n_items, rho = rho)
}

# Cache of magmaan_model_spec objects keyed by n_items. Building the spec
# parses the lavaan syntax and runs lavaanify — keeping it out of the rep
# loop avoids ~1.5% repeated parsing cost (worse at larger n_items).
.spec_cache <- new.env(parent = emptyenv())
get_spec <- function(n_items) {
  key <- as.character(n_items)
  if (!is.null(.spec_cache[[key]])) return(.spec_cache[[key]])
  spec <- magmaan::model_spec(build_unidim_syntax(n_items))
  .spec_cache[[key]] <- spec
  spec
}

.varname_cache <- new.env(parent = emptyenv())
get_varnames <- function(n_items) {
  key <- as.character(n_items)
  if (!is.null(.varname_cache[[key]])) return(.varname_cache[[key]])
  vars <- paste0("x", seq_len(n_items))
  .varname_cache[[key]] <- vars
  vars
}

# ── Population RMSEA ────────────────────────────────────────────────────────
# F_ML at population for the unidim analysis model fit to the two-factor
# population Σ. Used as the reference for the sample-RMSEA-vs-population
# comparison (Maydeu Table 4).
population_rmsea <- function(Sigma_pop, n_items) {
  # Fit a unidim CFA to a large MVN sample drawn from Σ_pop so the sample F_ML
  # approximates the population F_ML. We don't need exact closed-form here —
  # a single large draw is fine for the reference value.
  set.seed(0xBEEF)
  L <- chol(Sigma_pop)
  N <- 100000L
  y <- (matrix(stats::rnorm(N * n_items), N, n_items) %*% L)
  colnames(y) <- paste0("x", seq_len(n_items))
  df <- as.data.frame(y)
  syntax <- build_unidim_syntax(n_items)
  fit <- tryCatch(magmaan::magmaan(syntax, df, estimator = "ML"),
                  error = function(e) e)
  if (inherits(fit, "error") || !isTRUE(fit$converged)) return(NA_real_)
  fmin <- as.numeric(fit$fmin)
  df_val <- n_items * (n_items + 1L) / 2L - (n_items + 1L)
  if (df_val <= 0) return(0.0)
  sqrt(max(fmin / df_val, 0))
}

# ── Threshold finder ────────────────────────────────────────────────────────
# 4 thresholds split a standard normal into 5 categories. Codes 1..5.
# We search numerically for thresholds that make the 5-category integer
# variable hit the target (skewness, excess-kurtosis) pair.
five_cat_moments <- function(taus) {
  cuts <- c(-Inf, sort(taus), Inf)
  p <- diff(stats::pnorm(cuts))
  if (any(p <= 0)) return(c(NA_real_, NA_real_))
  k <- 1:5
  mu <- sum(k * p)
  s2 <- sum((k - mu)^2 * p)
  m3 <- sum((k - mu)^3 * p)
  m4 <- sum((k - mu)^4 * p)
  if (s2 <= 0) return(c(NA_real_, NA_real_))
  c(m3 / s2^1.5, m4 / s2^2 - 3)
}

find_thresholds <- function(skew_target, kurt_target) {
  obj <- function(taus) {
    m <- five_cat_moments(taus)
    if (any(is.na(m))) return(1e9)
    sum((m - c(skew_target, kurt_target))^2)
  }
  if (abs(skew_target) < 1e-9 && abs(kurt_target) < 1e-9) {
    # Equal-spaced symmetric init → 5 equal-prob categories.
    init <- stats::qnorm(c(0.2, 0.4, 0.6, 0.8))
  } else if (abs(skew_target) < 1e-9) {
    # Symmetric high-kurtosis: thin middle category.
    init <- c(-1.3, -0.3, 0.3, 1.3)
  } else if (skew_target < 0) {
    # Left-skew: thresholds shifted left, long left tail, mass in the right.
    init <- c(-1.0, 0.5, 1.0, 1.5)
  } else {
    init <- c(-1.5, -1.0, -0.5, 1.0)
  }
  best <- list(par = init, value = obj(init))
  # Deterministic restarts so the threshold table is reproducible.
  set.seed(0xACEBEE)
  for (jitter in seq_len(8L)) {
    init_j <- init + stats::runif(4L, -0.3, 0.3)
    opt <- stats::optim(init_j, obj, method = "Nelder-Mead",
                        control = list(reltol = 1e-12, maxit = 8000))
    if (opt$value < best$value) best <- opt
  }
  taus <- sort(best$par)
  list(taus = taus, achieved = five_cat_moments(taus),
       loss = best$value)
}

build_unidim_syntax <- function(n_items) {
  vars <- get_varnames(n_items)
  paste("f =~", paste(vars, collapse = " + "))
}

discretize_five_cat <- function(y_cont, threshold_taus) {
  matrix(findInterval(y_cont, threshold_taus, left.open = TRUE) + 1L,
         nrow = nrow(y_cont), ncol = ncol(y_cont))
}

sample_data_from_matrix <- function(X) {
  ss <- core$data_sample_stats_from_raw(list(X))
  nm <- colnames(X)
  dimnames(ss$S[[1L]]) <- list(nm, nm)
  if (!is.null(ss$mean) && length(ss$mean)) names(ss$mean[[1L]]) <- nm
  ss$X <- list(X)
  ss$ov_names <- list(nm)
  ss$group_var <- ""
  ss$group_labels <- character()
  ss$scaling <- "n"
  class(ss) <- c("magmaan_data", "list")
  ss
}

# Per-cell context — everything that's constant across reps. Built once in the
# cell loop and passed straight into `fit_one_rep` to avoid recomputation.
build_cell_ctx <- function(cell) {
  list(
    pop      = build_population(cell$n_items, cell$rho),
    spec     = get_spec(cell$n_items),
    varnames = get_varnames(cell$n_items)
  )
}

# ── χ² reduced-matrix pipeline ──────────────────────────────────────────────
# Build UFactor, reduce sample Γ̂ through B, return the df × df symmetric
# matrix M = B'Γ̂B itself. The SB / MV-adj / SS adjustments need only
# `tr(M)` and `tr(M²)` — eigenvalues are wasted work — so we feed M directly
# into the `_M`-suffixed χ² primitives. `Zc` and `denom` are passed in so we
# don't re-compute them across bread variants.
ugamma_M <- function(fit, Zc, denom, bread) {
  uf <- core$robust_build_u_factor(fit, bread = bread, moments = "structured")
  M <- core$robust_reduced_gamma_sample(uf, Zc, denom)
  list(M = M, df = as.integer(uf$df))
}

# ── Per-replicate fit pipeline ──────────────────────────────────────────────
# `cell_ctx` carries everything that's constant across the reps of a cell:
# the population covariance (with cached chol), the pre-built magmaan_model_spec,
# and the variable-name vector. Built once per cell in the main loop.
fit_one_rep <- function(cell_ctx, threshold_taus, N, seed,
                        keep_parity = FALSE) {
  pop <- cell_ctx$pop
  p <- nrow(pop$Sigma)
  set.seed(seed)
  z <- matrix(stats::rnorm(N * p), N, p)
  y_cont <- z %*% pop$L
  X <- discretize_five_cat(y_cont, threshold_taus)
  storage.mode(X) <- "double"
  colnames(X) <- cell_ctx$varnames
  dat <- sample_data_from_matrix(X)

  fit <- tryCatch(core$fit_ml(cell_ctx$spec, dat),
                  error = function(e) e)
  if (inherits(fit, "error")) {
    return(list(ok = FALSE, error = conditionMessage(fit)))
  }
  if (!isTRUE(fit$converged)) {
    return(list(ok = FALSE, error = paste0("did not converge: ",
                                           as.character(fit$optimizer_status %||% ""))))
  }

  # Identify free factor-loading parameters in the partable.
  pt <- fit$partable
  loading_idx <- which(pt$op == "=~" & pt$free > 0L)
  loading_free_idx <- pt$free[loading_idx]
  loading_names <- paste0(pt$rhs[loading_idx])

  # Shared casewise contributions. Whether the cheapest sandwich form is
  # (Zc·WΔ)ᵀ(Zc·WΔ) (Zc-overload of robust_se) or WΔᵀ·Γ̂·WΔ (gamma_hat
  # overload) depends on the shape: the former is O(N · p* · q), the latter
  # is O(p*² · q + p*·q²). For Maydeu's design N ∈ {200, 500, 1000} dominates
  # p* ∈ {136, 528}, so the γ̂ path wins; for N ≤ p* (e.g. very small samples
  # or larger models) prefer `core$robust_se_zc(fit, Zc, N_total, ...)`.
  Zc <- core$robust_casewise_contributions(fit$partable, X)
  N_total <- nrow(X)
  denom <- as.numeric(unlist(fit$nobs, use.names = FALSE))
  if (!length(denom)) denom <- N_total
  gamma_hat <- crossprod(Zc) / N_total
  samp <- core$fit_sample_stats(fit)

  # ── Six SE methods ──────────────────────────────────────────────────────
  info_E_or <- tryCatch(core$inference_information_expected(fit),
                       error = function(e) e)
  info_O_or <- tryCatch(core$inference_information_observed_analytic(fit),
                       error = function(e) e)
  info_XP_or <- tryCatch(
    core$inference_information_cross_products(fit, list(X)),
    error = function(e) e
  )

  se_from_info <- function(info_or) {
    if (inherits(info_or, "error")) return(NULL)
    vc <- tryCatch(core$inference_vcov_fit(info_or, fit),
                   error = function(e) NULL)
    if (is.null(vc)) return(NULL)
    core$inference_se(vc)
  }
  se_ML_obs <- se_from_info(info_O_or)
  se_ML_exp <- se_from_info(info_E_or)
  se_MLF   <- se_from_info(info_XP_or)

  # MLM/MLMV/MLR-exp share the sandwich SE (bread=expected, cov=empirical);
  # MLR uses bread=observed. The both_breads call runs robust_setup once and
  # returns both SE vectors, skipping the duplicated Δ-stacking / Γ_NT
  # Cholesky / WΔ work that two single-bread calls would pay.
  pair <- tryCatch(
    core$robust_se_both_breads(fit, gamma_hat,
                               moments = "structured", cov = "empirical"),
    error = function(e) e
  )
  se_MLM <- if (inherits(pair, "error")) NULL else pair$expected$se
  se_MLR <- if (inherits(pair, "error")) NULL else pair$observed$se

  # ── Five χ² statistics ──────────────────────────────────────────────────
  T_ML <- core$inference_chi2_stat(samp, fit$fmin)
  df_ML <- core$inference_df_stat(fit$partable, samp)

  # Build the reduced df × df symmetric matrix M for each bread; the
  # adjustments below take M directly (tr(M), tr(M²)) and skip the O(k³)
  # eigendecomposition entirely.
  mat_E <- tryCatch(ugamma_M(fit, Zc, denom, "expected"),
                    error = function(e) e)
  mat_O <- tryCatch(ugamma_M(fit, Zc, denom, "observed"),
                    error = function(e) e)
  # Field names differ by adjustment:
  #   satorra_bentler_M   → list(chi2_scaled, scale_c, df)
  #   mean_var_adjusted_M → list(chi2_adj,    df_adj)
  #   scaled_shifted_M    → list(chi2_adj,    df, scale_a, shift_b)
  pick_chi2 <- function(fun, mat_pack, stat_field, df_field) {
    if (inherits(mat_pack, "error")) return(list(stat = NA_real_, df = NA_integer_))
    res <- tryCatch(fun(T_ML, as.integer(mat_pack$df), mat_pack$M),
                    error = function(e) e)
    if (inherits(res, "error")) return(list(stat = NA_real_, df = NA_integer_))
    list(stat = as.numeric(res[[stat_field]] %||% NA_real_),
         df = as.integer(res[[df_field]] %||% mat_pack$df))
  }
  T_MLM    <- pick_chi2(core$robust_satorra_bentler_M,    mat_E,
                        "chi2_scaled", "df")
  T_MLMV   <- pick_chi2(core$robust_mean_var_adjusted_M,  mat_E,
                        "chi2_adj", "df_adj")
  T_MLR    <- pick_chi2(core$robust_scaled_shifted_M,     mat_O,
                        "chi2_adj", "df")
  T_MLR_e  <- pick_chi2(core$robust_scaled_shifted_M,     mat_E,
                        "chi2_adj", "df")

  # ── RMSEA / SRMR ────────────────────────────────────────────────────────
  rmsea_from <- function(T, df, N) {
    if (!is.finite(T) || !is.finite(df) || df <= 0 || N <= 0) return(NA_real_)
    sqrt(max((T - df) / (df * (N - 1L)), 0))
  }
  rmsea_ML <- rmsea_from(T_ML, df_ML, N_total)
  rmsea_MLM <- rmsea_from(T_MLM$stat, T_MLM$df, N_total)
  rmsea_MLMV <- rmsea_from(T_MLMV$stat, T_MLMV$df, N_total)
  rmsea_MLR <- rmsea_from(T_MLR$stat, T_MLR$df, N_total)
  rmsea_MLR_e <- rmsea_from(T_MLR_e$stat, T_MLR_e$df, N_total)

  std_resid <- core$measures_standardized_residuals(fit)
  srmr <- as.numeric(std_resid$srmr %||% NA_real_)

  # ── Long-format rows ────────────────────────────────────────────────────
  theta <- as.numeric(fit$theta)
  loading_theta <- theta[loading_free_idx]

  se_methods <- list(
    ML_obs = se_ML_obs, ML_exp = se_ML_exp, MLF = se_MLF,
    MLM = se_MLM, MLR = se_MLR
  )
  se_rows <- do.call(rbind, lapply(seq_along(se_methods), function(k) {
    nm <- names(se_methods)[[k]]
    se_vec <- se_methods[[k]]
    if (is.null(se_vec)) {
      return(data.frame(method = nm, loading = loading_names,
                        theta = loading_theta, se = NA_real_,
                        stringsAsFactors = FALSE))
    }
    data.frame(method = nm, loading = loading_names,
               theta = loading_theta,
               se = as.numeric(se_vec)[loading_free_idx],
               stringsAsFactors = FALSE)
  }))

  chi2_rows <- data.frame(
    stat_name = c("ML", "MLM", "MLMV", "MLR", "MLR_exp"),
    statistic = c(T_ML, T_MLM$stat, T_MLMV$stat, T_MLR$stat, T_MLR_e$stat),
    df = c(df_ML, T_MLM$df, T_MLMV$df, T_MLR$df, T_MLR_e$df),
    rmsea = c(rmsea_ML, rmsea_MLM, rmsea_MLMV, rmsea_MLR, rmsea_MLR_e),
    stringsAsFactors = FALSE
  )

  parity <- if (isTRUE(keep_parity)) {
    list(data = as.data.frame(X),
         loading_theta = loading_theta,
         chi2_rows = chi2_rows)
  } else NULL

  list(ok = TRUE, se_rows = se_rows, chi2_rows = chi2_rows,
       srmr = srmr, n_loadings = length(loading_free_idx),
       converged = TRUE, fmin = as.numeric(fit$fmin),
       iterations = as.integer(fit$iterations %||% NA),
       parity = parity)
}

# ── Cell grid ───────────────────────────────────────────────────────────────
parse_cells_filter <- function(s) {
  if (is.null(s) || !nzchar(s)) return(list())
  pairs <- strsplit(s, ",", fixed = TRUE)[[1L]]
  out <- list()
  for (p in pairs) {
    kv <- strsplit(p, "=", fixed = TRUE)[[1L]]
    if (length(kv) != 2L) stop("bad --cells entry: ", p, call. = FALSE)
    out[[kv[[1L]]]] <- kv[[2L]]
  }
  out
}

cell_grid <- expand.grid(
  n_items = c(16L, 32L),
  rho     = c(0.4, 0.8, 1.0),
  dist    = c("norm", "kurt2", "skew_kurt"),
  N       = c(200L, 500L, 1000L),
  stringsAsFactors = FALSE
)

filter <- parse_cells_filter(args$cells_filter)
for (key in names(filter)) {
  if (!(key %in% names(cell_grid))) {
    stop("--cells: unknown key '", key, "'", call. = FALSE)
  }
  target <- if (is.numeric(cell_grid[[key]])) as.numeric(filter[[key]]) else filter[[key]]
  cell_grid <- cell_grid[cell_grid[[key]] == target, , drop = FALSE]
}
if (!nrow(cell_grid)) stop("no cells selected", call. = FALSE)

skew_kurt_lookup <- list(
  norm      = c(0, 0),
  kurt2     = c(0, 2),
  skew_kurt = c(-2, 3.18)
)

threshold_sets <- setNames(lapply(unique(cell_grid$dist), function(nm) {
  target <- skew_kurt_lookup[[nm]]
  find_thresholds(target[[1L]], target[[2L]])
}), unique(cell_grid$dist))

# ── Run ─────────────────────────────────────────────────────────────────────
cat(sprintf(
  "maydeu-olivares-2017: magmaan %s, %s, reps=%d, cells=%d\n",
  as.character(utils::packageVersion("magmaan")),
  R.version.string, args$reps, nrow(cell_grid)
))

# Write the threshold table once.
threshold_rows <- do.call(rbind, lapply(names(threshold_sets), function(nm) {
  ts <- threshold_sets[[nm]]
  data.frame(dist = nm,
             skew_target = skew_kurt_lookup[[nm]][[1L]],
             kurt_target = skew_kurt_lookup[[nm]][[2L]],
             skew_achieved = ts$achieved[[1L]],
             kurt_achieved = ts$achieved[[2L]],
             tau1 = ts$taus[[1L]], tau2 = ts$taus[[2L]],
             tau3 = ts$taus[[3L]], tau4 = ts$taus[[4L]],
             loss = ts$loss,
             stringsAsFactors = FALSE)
}))
utils::write.csv(threshold_rows,
                 file.path(results_dir, "thresholds.csv"),
                 row.names = FALSE)

# Write the cell grid.
utils::write.csv(cell_grid, file.path(results_dir, "cells.csv"),
                 row.names = FALSE)

max_rep_rows <- nrow(cell_grid) * args$reps
all_se <- vector("list", max_rep_rows)
all_chi2 <- vector("list", max_rep_rows)
all_meta <- vector("list", nrow(cell_grid))
all_parity_inputs <- vector("list", nrow(cell_grid))
se_pos <- 0L
chi2_pos <- 0L

t_global0 <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(cell_grid))) {
  cell <- as.list(cell_grid[ci, , drop = FALSE])
  taus <- threshold_sets[[cell$dist]]$taus
  ctx <- build_cell_ctx(cell)  # spec + pop + chol cached for the whole cell
  cat(sprintf("  cell %3d/%3d: n_items=%2d rho=%.1f dist=%-9s N=%4d ",
              ci, nrow(cell_grid), cell$n_items, cell$rho, cell$dist, cell$N))
  rep_ok <- 0L
  rep_fail <- 0L
  t_cell0 <- proc.time()[["elapsed"]]
  for (rep_idx in seq_len(args$reps)) {
    seed <- args$seed_base + ci * 10000L + rep_idx
    keep_parity <- isTRUE(args$lavaan_parity) && rep_idx == 1L
    out <- fit_one_rep(ctx, taus, cell$N, seed, keep_parity = keep_parity)
    if (!isTRUE(out$ok)) {
      rep_fail <- rep_fail + 1L
      next
    }
    rep_ok <- rep_ok + 1L
    se_df <- cbind(out$se_rows,
                   data.frame(cell_idx = ci, rep = rep_idx, seed = seed,
                              n_items = cell$n_items, rho = cell$rho,
                              dist = cell$dist, N = cell$N,
                              stringsAsFactors = FALSE))
    chi2_df <- cbind(out$chi2_rows,
                     data.frame(cell_idx = ci, rep = rep_idx, seed = seed,
                                n_items = cell$n_items, rho = cell$rho,
                                dist = cell$dist, N = cell$N,
                                srmr = out$srmr, fmin = out$fmin,
                                stringsAsFactors = FALSE))
    se_pos <- se_pos + 1L
    chi2_pos <- chi2_pos + 1L
    all_se[[se_pos]] <- se_df
    all_chi2[[chi2_pos]] <- chi2_df
    if (keep_parity) all_parity_inputs[[ci]] <- out$parity
  }
  t_cell1 <- proc.time()[["elapsed"]]
  cat(sprintf("ok=%d fail=%d  (%.1fs)\n", rep_ok, rep_fail, t_cell1 - t_cell0))
  all_meta[[ci]] <- data.frame(
    cell_idx = ci, n_items = cell$n_items, rho = cell$rho,
    dist = cell$dist, N = cell$N,
    rep_ok = rep_ok, rep_fail = rep_fail,
    seconds = t_cell1 - t_cell0,
    stringsAsFactors = FALSE
  )
}
t_global1 <- proc.time()[["elapsed"]]

all_se <- all_se[seq_len(se_pos)]
all_chi2 <- all_chi2[seq_len(chi2_pos)]

se_long <- if (length(all_se)) do.call(rbind, all_se) else
  data.frame(method = character(), loading = character(),
             theta = double(), se = double(), cell_idx = integer(),
             rep = integer(), seed = integer(), n_items = integer(),
             rho = double(), dist = character(), N = integer(),
             stringsAsFactors = FALSE)
chi2_long <- if (length(all_chi2)) do.call(rbind, all_chi2) else
  data.frame(stat_name = character(), statistic = double(), df = integer(),
             rmsea = double(), cell_idx = integer(), rep = integer(),
             seed = integer(), n_items = integer(), rho = double(),
             dist = character(), N = integer(), srmr = double(),
             fmin = double(), stringsAsFactors = FALSE)
meta_long <- if (length(all_meta)) do.call(rbind, all_meta) else
  data.frame(cell_idx = integer(), n_items = integer(), rho = double(),
             dist = character(), N = integer(), rep_ok = integer(),
             rep_fail = integer(), seconds = double(),
             stringsAsFactors = FALSE)

utils::write.csv(se_long, file.path(results_dir, "fits_se.csv"),
                 row.names = FALSE)
utils::write.csv(chi2_long, file.path(results_dir, "fits_chi2.csv"),
                 row.names = FALSE)
utils::write.csv(meta_long, file.path(results_dir, "cell_meta.csv"),
                 row.names = FALSE)

# ── Aggregated summary (SE relative bias, χ² rejection rate) ────────────────
summary_se <- if (nrow(se_long) > 0L) {
  by_cell_method <- split(se_long, list(se_long$cell_idx, se_long$method),
                          drop = TRUE)
  do.call(rbind, lapply(by_cell_method, function(g) {
    # Per-loading: mean(SE) vs sd(theta) across reps.
    per_loading <- split(g, g$loading)
    rel_bias <- vapply(per_loading, function(h) {
      sd_t <- stats::sd(h$theta)
      mn_se <- mean(h$se, na.rm = TRUE)
      if (!is.finite(sd_t) || sd_t < 1e-12) return(NA_real_)
      (mn_se - sd_t) / sd_t
    }, double(1))
    data.frame(cell_idx = g$cell_idx[[1L]],
               n_items = g$n_items[[1L]], rho = g$rho[[1L]],
               dist = g$dist[[1L]], N = g$N[[1L]],
               method = g$method[[1L]],
               n_reps = length(unique(g$rep)),
               mean_rel_bias_se = mean(rel_bias, na.rm = TRUE),
               stringsAsFactors = FALSE)
  }))
} else NULL
if (!is.null(summary_se)) {
  utils::write.csv(summary_se, file.path(results_dir, "summary_se.csv"),
                   row.names = FALSE)
}

summary_chi2 <- if (nrow(chi2_long) > 0L) {
  by_cell_stat <- split(chi2_long, list(chi2_long$cell_idx,
                                        chi2_long$stat_name), drop = TRUE)
  do.call(rbind, lapply(by_cell_stat, function(g) {
    p_vals <- stats::pchisq(g$statistic, df = g$df, lower.tail = FALSE)
    rej_05 <- mean(p_vals < 0.05, na.rm = TRUE)
    data.frame(cell_idx = g$cell_idx[[1L]],
               n_items = g$n_items[[1L]], rho = g$rho[[1L]],
               dist = g$dist[[1L]], N = g$N[[1L]],
               stat_name = g$stat_name[[1L]],
               n_reps = nrow(g),
               mean_chi2 = mean(g$statistic, na.rm = TRUE),
               mean_df = mean(g$df, na.rm = TRUE),
               reject_05 = rej_05,
               mean_rmsea = mean(g$rmsea, na.rm = TRUE),
               mean_srmr = mean(g$srmr, na.rm = TRUE),
               stringsAsFactors = FALSE)
  }))
} else NULL
if (!is.null(summary_chi2)) {
  utils::write.csv(summary_chi2, file.path(results_dir, "summary_chi2.csv"),
                   row.names = FALSE)
}

# ── Metadata ────────────────────────────────────────────────────────────────
meta <- data.frame(
  key = c("magmaan_version", "R_version", "reps", "seed_base",
          "cells_filter", "lavaan_parity",
          "n_cells", "n_se_rows", "n_chi2_rows", "total_seconds"),
  value = c(as.character(utils::packageVersion("magmaan")),
            R.version.string, args$reps, args$seed_base,
            args$cells_filter %||% "",
            isTRUE(args$lavaan_parity),
            nrow(cell_grid), nrow(se_long), nrow(chi2_long),
            sprintf("%.2f", t_global1 - t_global0)),
  stringsAsFactors = FALSE
)
utils::write.csv(meta, file.path(results_dir, "metadata.csv"),
                 row.names = FALSE)
cat(sprintf("\ndone in %.1fs — wrote results to %s\n",
            t_global1 - t_global0, results_dir))

# ── Lavaan parity (one fit per cell, rep 1 only) ────────────────────────────
if (isTRUE(args$lavaan_parity)) {
  if (!requireNamespace("lavaan", quietly = TRUE)) {
    cat("--lavaan-parity requested but lavaan is not installed; skipping.\n")
  } else {
    cat("running lavaan parity check (one fit per cell)...\n")
    parity_rows <- list()
    for (ci in seq_len(nrow(cell_grid))) {
      cell <- as.list(cell_grid[ci, , drop = FALSE])
      cached <- all_parity_inputs[[ci]]
      if (is.null(cached)) {
        parity_rows[[length(parity_rows) + 1L]] <- data.frame(
          cell_idx = ci, n_items = cell$n_items, rho = cell$rho,
          dist = cell$dist, N = cell$N,
          metric = "magmaan_error",
          magmaan = NA_real_, lavaan = NA_real_,
          stringsAsFactors = FALSE
        )
        next
      }
      df <- cached$data
      syntax <- build_unidim_syntax(cell$n_items)
      # lavaan fit with multiple tests at once.
      lv_or <- tryCatch(lavaan::sem(
          syntax, data = df, estimator = "ML",
          test = c("standard", "satorra.bentler",
                   "scaled.shifted", "yuan.bentler.mplus")),
        error = function(e) e)
      if (inherits(lv_or, "error")) {
        parity_rows[[length(parity_rows) + 1L]] <- data.frame(
          cell_idx = ci, n_items = cell$n_items, rho = cell$rho,
          dist = cell$dist, N = cell$N,
          metric = "lavaan_error",
          magmaan = NA_real_, lavaan = NA_real_,
          stringsAsFactors = FALSE
        )
        next
      }
      # Point estimates.
      lv_pe <- lavaan::parameterEstimates(lv_or)
      lv_loadings <- lv_pe[lv_pe$op == "=~" & lv_pe$rhs != lv_pe$rhs[1L], ]
      mg_theta_loadings <- cached$loading_theta
      max_abs_theta_diff <- if (length(lv_loadings$est) == length(mg_theta_loadings))
        max(abs(lv_loadings$est - mg_theta_loadings), na.rm = TRUE)
      else NA_real_
      # χ² statistics.
      lv_stats <- lavaan::lavInspect(lv_or, "test")
      lv_chi2_std <- lv_stats[["standard"]]$stat %||% NA_real_
      lv_chi2_sb  <- lv_stats[["satorra.bentler"]]$stat %||% NA_real_
      lv_chi2_ss  <- lv_stats[["scaled.shifted"]]$stat %||% NA_real_
      lv_chi2_yb  <- lv_stats[["yuan.bentler.mplus"]]$stat %||% NA_real_
      mg_chi2 <- cached$chi2_rows
      mg_stat <- function(name) mg_chi2$statistic[match(name, mg_chi2$stat_name)]
      mg_df <- function(name) mg_chi2$df[match(name, mg_chi2$stat_name)]
      append_row <- function(metric, mag, lv) {
        parity_rows[[length(parity_rows) + 1L]] <<- data.frame(
          cell_idx = ci, n_items = cell$n_items, rho = cell$rho,
          dist = cell$dist, N = cell$N,
          metric = metric,
          magmaan = mag, lavaan = lv,
          stringsAsFactors = FALSE
        )
      }
      append_row("max_abs_theta_diff_loadings", max_abs_theta_diff, 0)
      # ML and MLM compare statistics directly (same df, same parameterization).
      append_row("chi2_ML", mg_stat("ML"), lv_chi2_std)
      append_row("chi2_MLM_SB", mg_stat("MLM"), lv_chi2_sb)
      # MLMV (mean+var) parameterizations differ — magmaan uses Satterthwaite-
      # df, lavaan's scaled.shifted is a linear chi² transform on the original
      # df. Compare the upper-tail p-values instead of statistic values.
      lv_df_std  <- lv_stats[["standard"]]$df %||% NA_real_
      lv_df_ss   <- lv_stats[["scaled.shifted"]]$df %||% lv_df_std
      lv_df_yb   <- lv_stats[["yuan.bentler.mplus"]]$df %||% lv_df_std
      mg_mlmv_p <- stats::pchisq(as.numeric(mg_stat("MLMV")),
                                 df = as.numeric(mg_df("MLMV")),
                                 lower.tail = FALSE)
      lv_mlmv_p <- stats::pchisq(lv_chi2_ss, df = lv_df_ss,
                                 lower.tail = FALSE)
      append_row("p_value_MLMV", mg_mlmv_p, lv_mlmv_p)
      mg_mlr_p <- stats::pchisq(as.numeric(mg_stat("MLR")),
                                df = as.numeric(mg_df("MLR")),
                                lower.tail = FALSE)
      lv_mlr_p <- stats::pchisq(lv_chi2_yb, df = lv_df_yb, lower.tail = FALSE)
      append_row("p_value_MLR", mg_mlr_p, lv_mlr_p)
    }
    parity_df <- do.call(rbind, parity_rows)
    parity_df$abs_diff <- abs(parity_df$magmaan - parity_df$lavaan)
    utils::write.csv(parity_df,
                     file.path(results_dir, "lavaan_parity.csv"),
                     row.names = FALSE)
    cat(sprintf("lavaan parity: %d rows written\n", nrow(parity_df)))
  }
}
