#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "..", "_support", "R", "missingness.R"))
source(file.path(script_dir, "R", "sem_models.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript smoke_fiml_h1_quadratic.R [options]\n\n",
  "Compares the FIML LR statistic with residual quadratics using saturated-\n",
  "coordinate curvature at H1 and observed/expected curvature at H0. This is\n",
  "a one-stage diagnostic; saturated EM supplies the unrestricted point.\n\n",
  "  --reps N             Replications per cell (default 25).\n",
  "  --n CSV              Sample sizes (default 120,500).\n",
  "  --cores N            Parallel cell workers (default up to 4).\n",
  "  --model ID           SEM model id (default one_factor_6).\n",
  "  --distributions CSV  Default normal,vm2,ig2.\n",
  "  --missingness CSV    Default complete,mcar_30,mar_30.\n",
  "  --seed-base N        Deterministic seed base.\n",
  "  --results-dir P      Output directory.\n",
  "  --help               Show this help.\n", sep = "")

opts <- list(
  reps = 25L,
  n = c(120L, 500L),
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  model = "one_factor_6",
  distributions = c("normal", "vm2", "ig2"),
  missingness = c("complete", "mcar_30", "mar_30"),
  seed_base = 20260905L,
  results_dir = NULL)
args <- commandArgs(TRUE)
i <- 1L
take <- function() {
  i <<- i + 1L
  if (i > length(args)) stop("missing option value", call. = FALSE)
  args[[i]]
}
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg %in% c("-h", "--help")) {
    usage()
    quit(save = "no", status = 0L)
  } else if (arg == "--reps") opts$reps <- as.integer(take())
  else if (arg == "--n") opts$n <- as.integer(parse_csv_arg(take()))
  else if (arg == "--cores") opts$cores <- as.integer(take())
  else if (arg == "--model") opts$model <- take()
  else if (arg == "--distributions") {
    opts$distributions <- parse_csv_arg(take())
  } else if (arg == "--missingness") {
    opts$missingness <- parse_csv_arg(take())
  } else if (arg == "--seed-base") opts$seed_base <- as.integer(take())
  else if (arg == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}

stopifnot(
  opts$reps > 0L,
  all(opts$n >= 80L),
  opts$cores > 0L,
  all(opts$distributions %in% c("normal", "vm1", "ig1", "vm2", "ig2")),
  all(opts$missingness %in% c("complete", "mcar_30", "mar_30")))

models <- sem_model_catalog()
if (!opts$model %in% names(models)) {
  stop("unknown model: ", opts$model, call. = FALSE)
}
model <- models[[opts$model]]
results <- opts$results_dir %||% file.path(
  script_dir, "results", "fiml-h1-quadratic-smoke")
dir.create(results, recursive = TRUE, showWarnings = FALSE)

grid <- expand.grid(
  n = opts$n,
  distribution = opts$distributions,
  missingness = opts$missingness,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE)
grid$cell_id <- seq_len(nrow(grid))
grid$null_contract <- ifelse(
  grid$missingness != "mar_30" | grid$distribution == "normal",
  "pseudo-null", "nonnormal-MAR estimand stress")

samplers <- setNames(lapply(opts$distributions, function(distribution) {
  sem_calibrate_sampler(model, distribution)
}), opts$distributions)

vech_lower <- function(S) S[lower.tri(S, diag = TRUE)]

# Saturated-coordinate negative log-likelihood Hessian for a single continuous
# FIML block, in [mu; lower-vech(Sigma)] order. `expected = TRUE` replaces each
# realized pattern second moment by its working-normal expectation and zeros
# the mean/covariance cross block. This is ambient eta curvature: it deliberately
# excludes the second-derivative term introduced by composing eta(theta).
fiml_saturated_information_at <- function(X, mu, Sigma, expected = FALSE) {
  X <- as.matrix(X)
  p <- ncol(X)
  stopifnot(length(mu) == p, identical(dim(Sigma), c(p, p)))
  pairs <- which(lower.tri(Sigma, diag = TRUE), arr.ind = TRUE)
  q_cov <- nrow(pairs)
  H <- matrix(0, p + q_cov, p + q_cov)
  keys <- apply(!is.na(X), 1L, paste0, collapse = "")

  for (rows in split(seq_len(nrow(X)), keys)) {
    observed <- which(!is.na(X[rows[[1L]], ]))
    if (!length(observed)) next
    Xo <- X[rows, observed, drop = FALSE]
    n_pattern <- nrow(Xo)
    mu_o <- mu[observed]
    Sigma_o <- Sigma[observed, observed, drop = FALSE]
    P <- solve(Sigma_o)
    H[observed, observed] <- H[observed, observed, drop = FALSE] +
      n_pattern * P

    derivatives <- lapply(seq_len(q_cov), function(j) {
      a <- pairs[j, 1L]
      b <- pairs[j, 2L]
      A <- matrix(0, length(observed), length(observed))
      ao <- match(a, observed, nomatch = 0L)
      bo <- match(b, observed, nomatch = 0L)
      if (ao > 0L && bo > 0L) {
        A[ao, bo] <- 1
        A[bo, ao] <- 1
      }
      A
    })
    active <- which(vapply(derivatives, function(A) any(A != 0), logical(1L)))
    if (!length(active)) next

    if (expected) {
      for (jj in seq_along(active)) {
        j <- active[[jj]]
        PAj <- P %*% derivatives[[j]]
        for (kk in seq_len(jj)) {
          k <- active[[kk]]
          value <- n_pattern / 2 * sum(diag(
            PAj %*% P %*% derivatives[[k]]))
          H[p + j, p + k] <- H[p + j, p + k] + value
          if (j != k) H[p + k, p + j] <- H[p + k, p + j] + value
        }
      }
    } else {
      centered <- sweep(Xo, 2L, mu_o, "-")
      C <- crossprod(centered) / n_pattern
      d <- colMeans(Xo) - mu_o
      for (j in active) {
        A <- derivatives[[j]]
        cross <- n_pattern * P %*% A %*% P %*% d
        H[observed, p + j] <- H[observed, p + j] + cross
        H[p + j, observed] <- H[p + j, observed] + as.vector(cross)
      }
      for (jj in seq_along(active)) {
        j <- active[[jj]]
        A <- derivatives[[j]]
        for (kk in seq_len(jj)) {
          k <- active[[kk]]
          B <- derivatives[[k]]
          value <- n_pattern / 2 * (
            -sum(diag(P %*% B %*% P %*% A)) +
              sum(diag(P %*% B %*% P %*% C %*% P %*% A)) +
              sum(diag(P %*% A %*% P %*% B %*% P %*% C)))
          H[p + j, p + k] <- H[p + j, p + k] + value
          if (j != k) H[p + k, p + j] <- H[p + k, p + j] + value
        }
      }
    }
  }
  0.5 * (H + t(H))
}

stack_h1_displacement <- function(stage1, implied) {
  if (length(stage1$mean) != length(implied$mu) ||
      length(stage1$cov) != length(implied$sigma)) {
    stop("saturated and restricted moment blocks disagree", call. = FALSE)
  }
  unlist(lapply(seq_along(stage1$cov), function(b) {
    c(stage1$mean[[b]] - implied$mu[[b]],
      vech_lower(stage1$cov[[b]] - implied$sigma[[b]]))
  }), use.names = FALSE)
}

fmg_p <- function(statistic, df, eigenvalues, method) {
  magmaan:::infer_fmg_test(
    statistic, df, eigenvalues,
    method = method, param = 4,
    truncate_negative = TRUE)$p_value
}

one_rep <- function(cell, rep_id) {
  begin <- proc.time()[["elapsed"]]
  seed <- sem_seed(opts$seed_base + cell$cell_id * 100003L + rep_id)
  out <- data.frame(
    cell_id = cell$cell_id,
    model_id = model$model_id,
    distribution = cell$distribution,
    missingness = cell$missingness,
    null_contract = cell$null_contract,
    n = cell$n,
    rep = rep_id,
    seed = seed,
    ok = FALSE,
    error = "",
    df = NA_integer_,
    statistic_lrt = NA_real_,
    statistic_h1_quadratic = NA_real_,
    statistic_h0_observed_quadratic = NA_real_,
    statistic_h0_expected_quadratic = NA_real_,
    statistic_expected_score = NA_real_,
    statistic_rls_h0 = NA_real_,
    difference_h1q_minus_lrt = NA_real_,
    ratio_h1q_to_lrt = NA_real_,
    p_peba4_lrt = NA_real_,
    p_peba4_h1q = NA_real_,
    p_peba4_h0_observed = NA_real_,
    p_peba4_h0_expected = NA_real_,
    p_peba4_expected_score = NA_real_,
    p_sb_lrt = NA_real_,
    p_sb_h1q = NA_real_,
    p_sb_h0_observed = NA_real_,
    p_sb_h0_expected = NA_real_,
    h1_min_eigenvalue = NA_real_,
    h0_observed_min_eigenvalue = NA_real_,
    h0_expected_min_eigenvalue = NA_real_,
    h1_reconstruction_max_abs = NA_real_,
    realized_missing = NA_real_,
    seconds = NA_real_,
    stringsAsFactors = FALSE)

  ans <- tryCatch({
    X <- sem_draw(model, samplers[[cell$distribution]], cell$n, seed)
    set.seed(seed + 700001L)
    X <- sem_apply_missingness(X, cell$missingness)
    realized_missing <- mean(is.na(X))
    data_frame <- as.data.frame(X)
    fd <- magmaan::df_to_fiml_data(data_frame, model$spec)
    control <- list(max_iter = 8000L, ftol = 1e-11, gtol = 1e-8)
    stage1 <- magmaan::magmaan_core$estimate_saturated_em_moments(
      fd, control = control)
    fit <- magmaan::magmaan_core$fit_fiml(
      model$spec, fd,
      optimizer = "nlopt-lbfgs-slsqp-fallback",
      control = control)
    fit$stage1 <- stage1
    if (!isTRUE(fit$converged)) stop("FIML fit did not converge")

    implied <- magmaan:::model_implied(fit)
    displacement <- stack_h1_displacement(stage1, implied)
    if (length(displacement) != nrow(stage1$H)) {
      stop("moment displacement and H1 information disagree")
    }
    statistic_h1q <- drop(crossprod(
      displacement, stage1$H %*% displacement))

    # The smoke is deliberately one-block. Reconstructing H1 observed
    # information from the pattern formulas is an internal correctness check
    # before evaluating the same ambient Hessian at the restricted moments.
    stopifnot(length(stage1$mean) == 1L, length(implied$mu) == 1L)
    H1_reconstructed <- fiml_saturated_information_at(
      X, stage1$mean[[1L]], stage1$cov[[1L]], expected = FALSE)
    h1_reconstruction_max_abs <- max(abs(H1_reconstructed - stage1$H))
    reconstruction_scale <- max(1, max(abs(stage1$H)))
    if (h1_reconstruction_max_abs > 1e-8 * reconstruction_scale) {
      stop("R saturated observed-Hessian reconstruction disagrees with magmaan")
    }
    H0_observed <- fiml_saturated_information_at(
      X, implied$mu[[1L]], implied$sigma[[1L]], expected = FALSE)
    H0_expected <- fiml_saturated_information_at(
      X, implied$mu[[1L]], implied$sigma[[1L]], expected = TRUE)
    statistic_h0_observed <- drop(crossprod(
      displacement, H0_observed %*% displacement))
    statistic_h0_expected <- drop(crossprod(
      displacement, H0_expected %*% displacement))

    lrt <- magmaan::fmg_tests(fit, tests = "all")
    statistic_lrt <- lrt$base_statistic[[1L]]
    df <- as.integer(lrt$df[[1L]])
    eigenvalues <- lrt$eigenvalues[[1L]]
    score <- magmaan::global_score_flip_test(
      fit,
      n_flips = 1L,
      seed = seed + 900001L,
      multiplier = "rademacher",
      sensitivity = "expected")
    if (as.integer(score$df) != df) stop("LR and score df differ")

    statistic_rls <- NA_real_
    if (identical(cell$missingness, "complete")) {
      fit_ml <- magmaan::magmaan(
        model$spec, data_frame, estimator = "ML",
        optimizer = "nlopt-lbfgs-slsqp-fallback")
      if (!isTRUE(fit_ml$converged)) stop("complete-data ML fit did not converge")
      statistic_rls <- magmaan:::infer_rls_chi2_fit(
        fit_ml, magmaan:::model_implied(fit_ml))$statistic
    }

    list(
      df = df,
      statistic_lrt = statistic_lrt,
      statistic_h1_quadratic = statistic_h1q,
      statistic_h0_observed_quadratic = statistic_h0_observed,
      statistic_h0_expected_quadratic = statistic_h0_expected,
      statistic_expected_score = score$statistic_effective,
      statistic_rls_h0 = statistic_rls,
      difference_h1q_minus_lrt = statistic_h1q - statistic_lrt,
      ratio_h1q_to_lrt = statistic_h1q / statistic_lrt,
      p_peba4_lrt = fmg_p(statistic_lrt, df, eigenvalues, "peba"),
      p_peba4_h1q = fmg_p(statistic_h1q, df, eigenvalues, "peba"),
      p_peba4_h0_observed = fmg_p(
        statistic_h0_observed, df, eigenvalues, "peba"),
      p_peba4_h0_expected = fmg_p(
        statistic_h0_expected, df, eigenvalues, "peba"),
      p_peba4_expected_score = fmg_p(
        score$statistic_effective, df, score$eigenvalues, "peba"),
      p_sb_lrt = fmg_p(statistic_lrt, df, eigenvalues, "sb"),
      p_sb_h1q = fmg_p(statistic_h1q, df, eigenvalues, "sb"),
      p_sb_h0_observed = fmg_p(
        statistic_h0_observed, df, eigenvalues, "sb"),
      p_sb_h0_expected = fmg_p(
        statistic_h0_expected, df, eigenvalues, "sb"),
      h1_min_eigenvalue = min(eigen(stage1$H, symmetric = TRUE,
                                    only.values = TRUE)$values),
      h0_observed_min_eigenvalue = min(eigen(
        H0_observed, symmetric = TRUE, only.values = TRUE)$values),
      h0_expected_min_eigenvalue = min(eigen(
        H0_expected, symmetric = TRUE, only.values = TRUE)$values),
      h1_reconstruction_max_abs = h1_reconstruction_max_abs,
      realized_missing = realized_missing)
  }, error = function(e) e)

  if (inherits(ans, "error")) {
    out$error <- conditionMessage(ans)
  } else {
    for (name in names(ans)) out[[name]] <- ans[[name]]
    finite_required <- unlist(ans[setdiff(names(ans), "statistic_rls_h0")])
    out$ok <- all(is.finite(finite_required)) &&
      (is.finite(ans$statistic_rls_h0) || !identical(cell$missingness, "complete"))
    if (!out$ok) out$error <- "non-finite statistic or diagnostic"
  }
  out$seconds <- proc.time()[["elapsed"]] - begin
  out
}

one_cell <- function(index) {
  cell <- as.list(grid[index, , drop = FALSE])
  message(sprintf(
    "cell %d/%d: n=%d / %s / %s",
    index, nrow(grid), cell$n, cell$distribution, cell$missingness))
  do.call(rbind, lapply(seq_len(opts$reps), function(rep_id) {
    one_rep(cell, rep_id)
  }))
}

cat(sprintf(
  "cells=%d reps=%d model=%s cores=%d\n",
  nrow(grid), opts$reps, model$model_id, opts$cores))
begin <- proc.time()[["elapsed"]]
pieces <- if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  parallel::mclapply(
    seq_len(nrow(grid)), one_cell,
    mc.cores = min(opts$cores, nrow(grid)), mc.preschedule = FALSE)
} else lapply(seq_len(nrow(grid)), one_cell)
raw <- do.call(rbind, pieces)
raw <- raw[order(raw$cell_id, raw$rep), ]
row.names(raw) <- NULL
write_csv(raw, file.path(results, "replications.csv"))

summarize_cell <- function(z) {
  attempted <- nrow(z)
  z <- z[z$ok, , drop = FALSE]
  if (!nrow(z)) return(data.frame())
  ratio <- z$ratio_h1q_to_lrt
  complete <- is.finite(z$statistic_rls_h0)
  data.frame(
    z[1L, c("model_id", "distribution", "missingness", "null_contract", "n"),
      drop = FALSE],
    attempted = attempted,
    usable = nrow(z),
    mean_lrt = mean(z$statistic_lrt),
    mean_h1q = mean(z$statistic_h1_quadratic),
    mean_h0_observed = mean(z$statistic_h0_observed_quadratic),
    mean_h0_expected = mean(z$statistic_h0_expected_quadratic),
    mean_expected_score = mean(z$statistic_expected_score),
    mean_difference = mean(z$difference_h1q_minus_lrt),
    mean_abs_difference = mean(abs(z$difference_h1q_minus_lrt)),
    median_ratio = stats::median(ratio),
    ratio_q10 = unname(stats::quantile(ratio, 0.10)),
    ratio_q90 = unname(stats::quantile(ratio, 0.90)),
    correlation = if (nrow(z) > 1L) {
      stats::cor(z$statistic_lrt, z$statistic_h1_quadratic)
    } else NA_real_,
    rejection_peba4_lrt = mean(z$p_peba4_lrt <= 0.05),
    rejection_peba4_h1q = mean(z$p_peba4_h1q <= 0.05),
    rejection_peba4_h0_observed = mean(z$p_peba4_h0_observed <= 0.05),
    rejection_peba4_h0_expected = mean(z$p_peba4_h0_expected <= 0.05),
    rejection_peba4_expected_score = mean(z$p_peba4_expected_score <= 0.05),
    rejection_sb_lrt = mean(z$p_sb_lrt <= 0.05),
    rejection_sb_h1q = mean(z$p_sb_h1q <= 0.05),
    rejection_sb_h0_observed = mean(z$p_sb_h0_observed <= 0.05),
    rejection_sb_h0_expected = mean(z$p_sb_h0_expected <= 0.05),
    mean_rls_h0 = if (any(complete)) mean(z$statistic_rls_h0[complete]) else NA_real_,
    mean_abs_h1q_minus_rls = if (any(complete)) {
      mean(abs(z$statistic_h1_quadratic[complete] -
                 z$statistic_rls_h0[complete]))
    } else NA_real_,
    mean_abs_h0_expected_minus_rls = if (any(complete)) {
      mean(abs(z$statistic_h0_expected_quadratic[complete] -
                 z$statistic_rls_h0[complete]))
    } else NA_real_,
    fraction_h0_observed_indefinite = mean(z$h0_observed_min_eigenvalue <= 0),
    max_h1_reconstruction_error = max(z$h1_reconstruction_max_abs),
    mean_realized_missing = mean(z$realized_missing),
    stringsAsFactors = FALSE)
}

groups <- split(seq_len(nrow(raw)), interaction(
  raw[c("n", "distribution", "missingness")],
  drop = TRUE, lex.order = TRUE))
summary <- do.call(rbind, lapply(groups, function(ii) summarize_cell(raw[ii, ])))
row.names(summary) <- NULL
summary <- summary[order(summary$n, summary$distribution, summary$missingness), ]
write_csv(summary, file.path(results, "summary.csv"))
write_metadata(file.path(results, "metadata.csv"), list(
  reps = opts$reps,
  sample_sizes = paste(opts$n, collapse = ","),
  model = model$model_id,
  distributions = paste(opts$distributions, collapse = ","),
  missingness = paste(opts$missingness, collapse = ","),
  seed_base = opts$seed_base,
  failures = sum(!raw$ok),
  runtime_wall_seconds = proc.time()[["elapsed"]] - begin,
  h1_information_scale = "summed log-likelihood information",
  quadratics = paste(
    "(eta_H1-eta_H0)' H (eta_H1-eta_H0), with H equal to",
    "H1 observed, H0 observed, or H0 working-normal expected curvature")),
  packages = "magmaan")

cat(sprintf(
  "runtime_wall=%.1fs failures=%d/%d\n\n",
  proc.time()[["elapsed"]] - begin, sum(!raw$ok), nrow(raw)))
print(summary, row.names = FALSE, digits = 3)
