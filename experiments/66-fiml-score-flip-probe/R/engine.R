fiml_flip_nested_tests <- c(
  "std_ml", "sb_ml", "mv_ml", "ss_ml", "peba4_ml", "all_ml")

fiml_flip_p_columns <- c(
  "p_flip_basic", "p_flip_effective", "p_flip_standardized",
  "p_score_chisq", "p_score_sb", "p_score_mv", "p_score_ss",
  "p_score_peba4", "p_score_all", "p_score_sandwich",
  paste0("p_nested_", fiml_flip_nested_tests))

fiml_flip_empty_p <- function() {
  stats::setNames(rep(NA_real_, length(fiml_flip_p_columns)),
                  fiml_flip_p_columns)
}

.fiml_flip_finish_block <- function(X, mu, ov, group) {
  X <- sweep(X, 2L, mu, "+")
  colnames(X) <- ov
  data.frame(X, school = if (group == 1L) "A" else "B",
             check.names = FALSE)
}

fiml_flip_make_sampler <- function(pop, distribution, n_values, reps,
                                   seed_base) {
  begin <- proc.time()[["elapsed"]]
  distributions <- c("normal", "pl")
  if (!distribution %in% distributions) {
    stop("unknown distribution: ", distribution, call. = FALSE)
  }
  states <- if (distribution == "normal") {
    lapply(pop$Sigma, function(Sigma) list(L = chol(Sigma), Sigma = Sigma))
  } else {
    lapply(pop$Sigma, function(Sigma) list(
      Sigma = Sigma,
      calibration = magmaan::magmaan_core$sim_plsim_calibrate(
        stats::cov2cor(Sigma), rep(3, 6), rep(21, 6),
        method = "hermite_then_rectangle", num_segments = 12L,
        quadrature_points = 31L, hermite_order = 24L)))
  }

  batches <- list()
  for (n1 in as.integer(n_values)) {
    sizes <- fiml_flip_group_sizes(n1)
    blocks <- vector("list", 2L)
    for (g in seq_len(2L)) {
      seed <- seed_base + n1 * 10007L + g * 1000003L
      if (distribution == "normal") {
        set.seed(seed)
        blocks[[g]] <- lapply(seq_len(reps), function(rep_id) {
          matrix(stats::rnorm(sizes[[g]] * 6L), sizes[[g]], 6L) %*%
            states[[g]]$L
        })
      } else {
        batch <- magmaan::magmaan_core$sim_plsim_draw(
          states[[g]]$calibration, n = sizes[[g]], reps = reps,
          seed_base = seed)
        scale <- sqrt(diag(states[[g]]$Sigma))
        blocks[[g]] <- lapply(batch$draws, function(X)
          sweep(X, 2L, scale, "*"))
      }
    }
    batches[[as.character(n1)]] <- blocks
  }
  draw <- function(n_group1, rep_id) {
    blocks <- batches[[as.character(n_group1)]]
    do.call(rbind, lapply(seq_len(2L), function(g)
      .fiml_flip_finish_block(blocks[[g]][[rep_id]], pop$mu[[g]], pop$ov, g)))
  }
  list(draw = draw,
       setup_seconds = proc.time()[["elapsed"]] - begin,
       unique_group_calibrations = length(states))
}

fiml_flip_apply_mcar <- function(data, ov, rate, seed) {
  if (rate <= 0) {
    return(list(data = data, realized_rate = 0, pattern_count = 2L))
  }
  masked <- sb2005_mcar(data[, ov, drop = FALSE], rate = rate,
                        intact = 1:2, seed = seed)
  out <- data
  out[, ov] <- masked$data
  pattern_count <- sum(vapply(split(out[, ov, drop = FALSE], out$school),
    function(block) nrow(unique(is.na(block))), integer(1L)))
  list(data = out, realized_rate = masked$summary$overall_rate,
       pattern_count = pattern_count)
}

fiml_flip_fit_pair <- function(data, specs) {
  fits <- lapply(specs, function(spec) magmaan::magmaan(
    spec, data, estimator = "FIML",
    optimizer = "nlopt-lbfgs-slsqp-fallback",
    control = list(max_iter = 16000L, ftol = 1e-13, gtol = 1e-9),
    se = "none", test = "none"))
  names(fits) <- names(specs)
  fits
}

fiml_flip_empty_replication <- function(rep_id, error = "") {
  base <- data.frame(
    rep = rep_id, fit_ok = FALSE, flip_ok = FALSE, nested_ok = FALSE,
    fit_error = error, flip_error = "", nested_error = "",
    realized_rate = NA_real_, pattern_count = NA_integer_,
    missing_seconds = NA_real_, fit_seconds = NA_real_,
    flip_seconds = NA_real_, nested_seconds = NA_real_, total_seconds = NA_real_,
    flip_setup_seconds = NA_real_, flip_score_seconds = NA_real_,
    flip_standardization_seconds = NA_real_, flip_asymptotic_seconds = NA_real_,
    nuisance_stationarity_norm = NA_real_,
    mean_variance_relative_shift = NA_real_,
    max_variance_relative_shift = NA_real_,
    min_variance_eigenvalue = NA_real_, max_variance_condition = NA_real_,
    sandwich_available = NA, sandwich_condition = NA_real_,
    score_eigen_mean = NA_real_, score_eigen_cv = NA_real_,
    score_eigen_ratio = NA_real_, stringsAsFactors = FALSE)
  cbind(base, as.data.frame(as.list(fiml_flip_empty_p())))
}

fiml_flip_one_rep <- function(cell, rep_id, complete_data, specs, flips,
                              seed_base) {
  total_begin <- proc.time()[["elapsed"]]
  out <- fiml_flip_empty_replication(rep_id)
  missing_begin <- proc.time()[["elapsed"]]
  masked <- fiml_flip_apply_mcar(
    complete_data, paste0("x", 1:6), cell$missing_rate,
    seed_base + cell$cell_id * 100003L + rep_id)
  out$missing_seconds <- proc.time()[["elapsed"]] - missing_begin
  out$realized_rate <- masked$realized_rate
  out$pattern_count <- masked$pattern_count

  fit_begin <- proc.time()[["elapsed"]]
  fits <- tryCatch(fiml_flip_fit_pair(masked$data, specs), error = function(e) e)
  out$fit_seconds <- proc.time()[["elapsed"]] - fit_begin
  if (inherits(fits, "error")) {
    out$fit_error <- conditionMessage(fits)
    out$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(out)
  }
  if (!all(vapply(fits, function(fit) isTRUE(fit$converged), logical(1L)))) {
    out$fit_error <- "one or both FIML fits did not converge"
    out$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(out)
  }
  out$fit_ok <- TRUE
  values <- fiml_flip_empty_p()

  flip_begin <- proc.time()[["elapsed"]]
  flip <- tryCatch(magmaan::score_flip_test(
    fits$H1, fits$H0, n_flips = flips,
    seed = seed_base + cell$cell_id * 1000003L + rep_id),
    error = function(e) e)
  out$flip_seconds <- proc.time()[["elapsed"]] - flip_begin
  if (inherits(flip, "error")) {
    out$flip_error <- conditionMessage(flip)
  } else {
    if (!identical(as.integer(flip$df), as.integer(cell$df))) {
      stop("FIML flip rank was ", flip$df, ", expected ", cell$df,
           call. = FALSE)
    }
    infer_score <- function(method, param = 4) tryCatch(
      magmaan:::infer_fmg_test(
        flip$statistic_effective, flip$df, flip$eigenvalues,
        method = method, param = param)$p_value,
      error = function(e) NA_real_)
    values[c("p_flip_basic", "p_flip_effective", "p_flip_standardized",
             "p_score_chisq", "p_score_sb", "p_score_all",
             "p_score_sandwich")] <-
      c(flip$p_basic, flip$p_effective, flip$p_standardized,
        flip$p_chisq, flip$p_mean_scaled, flip$p_mixture, flip$p_sandwich)
    values[["p_score_mv"]] <- infer_score("mv")
    values[["p_score_ss"]] <- infer_score("ss")
    values[["p_score_peba4"]] <- infer_score("peba", 4)
    out$flip_ok <- TRUE
    out$flip_setup_seconds <- flip$setup_seconds
    out$flip_score_seconds <- flip$resampling_score_seconds
    out$flip_standardization_seconds <-
      flip$resampling_standardization_seconds
    out$flip_asymptotic_seconds <- flip$asymptotic_seconds
    out$nuisance_stationarity_norm <- flip$nuisance_stationarity_norm
    out$mean_variance_relative_shift <- flip$mean_variance_relative_shift
    out$max_variance_relative_shift <- flip$max_variance_relative_shift
    out$min_variance_eigenvalue <- flip$min_variance_eigenvalue
    out$max_variance_condition <- flip$max_variance_condition
    out$sandwich_available <- flip$sandwich_available
    out$sandwich_condition <- flip$sandwich_condition
    out$score_eigen_mean <- mean(flip$eigenvalues)
    out$score_eigen_cv <- stats::sd(flip$eigenvalues) /
      mean(flip$eigenvalues)
    out$score_eigen_ratio <- max(flip$eigenvalues) /
      min(flip$eigenvalues)
  }

  nested_begin <- proc.time()[["elapsed"]]
  nested <- tryCatch(magmaan::fmg_nested(
    fits$H1, fits$H0, tests = fiml_flip_nested_tests, A.method = "exact"),
    error = function(e) e)
  out$nested_seconds <- proc.time()[["elapsed"]] - nested_begin
  if (inherits(nested, "error")) {
    out$nested_error <- conditionMessage(nested)
  } else {
    if (!all(nested$df == cell$df) ||
        !identical(as.character(nested$label), fiml_flip_nested_tests)) {
      stop("FIML nested battery did not preserve the five-df design",
           call. = FALSE)
    }
    values[paste0("p_nested_", nested$label)] <- nested$p_value
    out$nested_ok <- TRUE
  }
  out[names(values)] <- as.list(values)
  out$total_seconds <- proc.time()[["elapsed"]] - total_begin
  out
}

fiml_flip_wilson <- function(x, n, z = 1.95996398454005) {
  if (!is.finite(n) || n <= 0L) return(c(lower = NA_real_, upper = NA_real_))
  phat <- x / n
  den <- 1 + z^2 / n
  centre <- (phat + z^2 / (2 * n)) / den
  half <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) / den
  c(lower = max(0, centre - half), upper = min(1, centre + half))
}
