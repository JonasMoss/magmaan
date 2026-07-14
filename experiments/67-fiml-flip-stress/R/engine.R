stress_nested_tests <- c("sb_ml", "peba4_ml", "all_ml")

stress_p_columns <- c(
  "p_flip_basic", "p_flip_effective", "p_flip_standardized",
  "p_score_sb", "p_score_peba4", "p_score_all", "p_score_sandwich",
  paste0("p_nested_", stress_nested_tests))

stress_design_columns <- c(
  "cell_id", "base_id", "distribution", "n_group1", "n_group2",
  "n_total", "missingness", "missing_rate", "rank", "df")

stress_method_table <- function() {
  method <- sub("^p_", "", stress_p_columns)
  data.frame(
    method = method,
    family = ifelse(startsWith(method, "flip_"), "flip",
                    ifelse(startsWith(method, "score_"), "score", "nested")),
    primary = method %in% c("flip_effective", "flip_standardized",
                            "score_peba4", "nested_peba4_ml"),
    stringsAsFactors = FALSE)
}

stress_empty_p <- function() {
  stats::setNames(rep(NA_real_, length(stress_p_columns)), stress_p_columns)
}

.stress_calibrate_group <- function(Sigma, distribution) {
  p <- ncol(Sigma)
  skew <- rep(3, p)
  exkurt <- rep(21, p)
  if (distribution == "normal") {
    return(list(distribution = distribution, L = chol(Sigma), Sigma = Sigma))
  }
  if (distribution == "vm") {
    calibration <- magmaan:::sim_vm_calibrate_impl(
      stats::cov2cor(Sigma), skew, exkurt)
  } else if (distribution == "ig") {
    calibration <- magmaan:::sim_ig_calibrate_impl(
      Sigma, skew, exkurt, root = "symmetric", generator_family = "pearson",
      quadrature_points = 81L)
  } else if (distribution == "pl") {
    calibration <- magmaan:::sim_plsim_calibrate_impl(
      stats::cov2cor(Sigma), skew, exkurt,
      method = "hermite_then_rectangle", num_segments = 12L,
      quadrature_points = 31L, hermite_order = 24L)
  } else stop("unknown distribution: ", distribution, call. = FALSE)
  list(distribution = distribution, calibration = calibration, Sigma = Sigma)
}

stress_calibrate_sampler <- function(pop, distribution) {
  begin <- proc.time()[["elapsed"]]
  states <- lapply(pop$Sigma, .stress_calibrate_group,
                   distribution = distribution)
  list(
    distribution = distribution, states = states, mu = pop$mu, ov = pop$ov,
    setup_seconds = proc.time()[["elapsed"]] - begin,
    unique_group_calibrations = length(states))
}

.stress_draw_group <- function(state, n, seed) {
  p <- ncol(state$Sigma)
  if (state$distribution == "normal") {
    set.seed(seed + 1)
    return(matrix(stats::rnorm(n * p), n, p) %*% state$L)
  }
  if (state$distribution == "vm") {
    batch <- magmaan:::sim_vm_draw_impl(
      state$calibration, n = n, reps = 1L, seed_base = seed)
  } else if (state$distribution == "ig") {
    batch <- magmaan:::sim_ig_draw_impl(
      state$calibration, n = n, reps = 1L, seed_base = seed,
      quadrature_points = 81L)
  } else {
    batch <- magmaan:::sim_plsim_draw_impl(
      state$calibration, n = n, reps = 1L, seed_base = seed)
  }
  X <- batch$draws[[1L]]
  if (state$distribution %in% c("vm", "pl")) {
    X <- sweep(X, 2L, sqrt(diag(state$Sigma)), "*")
  }
  X
}

stress_draw_replication <- function(sampler, group_sizes, seed) {
  blocks <- lapply(seq_along(group_sizes), function(g) {
    X <- .stress_draw_group(sampler$states[[g]], group_sizes[[g]],
                            seed + g * 1000003)
    X <- sweep(X, 2L, sampler$mu[[g]], "+")
    colnames(X) <- sampler$ov
    X
  })
  data.frame(
    do.call(rbind, blocks),
    school = factor(rep(c("A", "B"), times = group_sizes),
                    levels = c("A", "B")),
    check.names = FALSE)
}

.stress_pattern_count <- function(data, ov) {
  sum(vapply(split(data[, ov, drop = FALSE], data$school), function(block)
    nrow(unique(is.na(block))), integer(1L)))
}

.stress_standardize <- function(x) {
  scale <- stats::sd(x)
  if (!is.finite(scale) || scale <= 0) return(rep(0, length(x)))
  (x - mean(x)) / scale
}

.stress_logistic_intercept <- function(eta, target) {
  lo <- -30
  hi <- 30
  for (iteration in seq_len(80L)) {
    mid <- (lo + hi) / 2
    if (mean(stats::plogis(mid + eta)) < target) lo <- mid else hi <- mid
  }
  (lo + hi) / 2
}

stress_strong_mar <- function(data, ov, rate, seed) {
  X <- data[, ov, drop = FALSE]
  z1 <- .stress_standardize(X[[1L]])
  z2 <- .stress_standardize(X[[2L]])
  slopes <- rbind(c(1.6, -1.2), c(-1.3, 1.5),
                  c(1.8, 0.6), c(0.6, -1.8))
  eta <- vapply(seq_len(nrow(slopes)), function(j)
    slopes[j, 1L] * z1 + slopes[j, 2L] * z2, numeric(nrow(X)))
  intercept <- vapply(seq_len(ncol(eta)), function(j)
    .stress_logistic_intercept(eta[, j], rate), numeric(1L))
  probabilities <- sweep(eta, 2L, intercept, "+")
  probabilities <- stats::plogis(probabilities)
  set.seed(seed)
  target_mask <- matrix(stats::runif(length(probabilities)), nrow(X), 4L) <
    probabilities
  out <- data
  for (j in seq_len(4L)) out[target_mask[, j], ov[[j + 2L]]] <- NA_real_
  q <- stats::quantile(probabilities, c(.1, .9), names = FALSE, type = 8)
  list(
    data = out, realized_rate = mean(target_mask),
    pattern_count = .stress_pattern_count(out, ov),
    selection_contrast = q[[2L]] / max(q[[1L]], .Machine$double.eps))
}

stress_apply_missingness <- function(data, ov, mechanism, seed) {
  if (mechanism == "complete") {
    return(list(data = data, realized_rate = 0,
                pattern_count = .stress_pattern_count(data, ov),
                selection_contrast = 1))
  }
  if (mechanism == "strong_mar30") {
    return(stress_strong_mar(data, ov, .30, seed))
  }
  if (mechanism == "mcar15") {
    masked <- sb2005_mcar(data[, ov, drop = FALSE], rate = .15,
                          intact = 1:2, seed = seed)
  } else if (mechanism == "mcar30") {
    masked <- sb2005_mcar(data[, ov, drop = FALSE], rate = .30,
                          intact = 1:2, seed = seed)
  } else if (mechanism == "mar30") {
    masked <- sb2005_mar(data[, ov, drop = FALSE], rate = .30,
                         predictors = 1:2, seed = seed, calibrate = TRUE)
  } else stop("unknown missingness mechanism: ", mechanism, call. = FALSE)
  out <- data
  out[, ov] <- masked$data
  list(data = out, realized_rate = masked$summary$overall_rate,
       pattern_count = .stress_pattern_count(out, ov),
       selection_contrast = NA_real_)
}

stress_fit <- function(spec, data) {
  magmaan::magmaan(
    spec, data, estimator = "FIML",
    optimizer = "nlopt-lbfgs-slsqp-fallback",
    control = list(max_iter = 16000L, ftol = 1e-13, gtol = 1e-9),
    se = "none", test = "none")
}

stress_empty_replication <- function(cell, rep_id, error = "") {
  base <- data.frame(
    as.data.frame(cell[stress_design_columns], stringsAsFactors = FALSE),
    rep = rep_id, fit_ok = FALSE, flip_ok = FALSE, nested_ok = FALSE,
    fit_error = error, flip_error = "", nested_error = "",
    realized_rate = NA_real_, pattern_count = NA_integer_,
    selection_contrast = NA_real_, draw_seconds = NA_real_,
    missing_seconds = NA_real_, fit_h1_seconds = NA_real_,
    fit_h0_seconds = NA_real_, fit_seconds = NA_real_,
    flip_seconds = NA_real_, nested_seconds = NA_real_,
    replication_seconds = NA_real_, h1_fit_count = NA_integer_,
    flip_setup_seconds = NA_real_, flip_score_seconds = NA_real_,
    flip_standardization_seconds = NA_real_, flip_asymptotic_seconds = NA_real_,
    nuisance_stationarity_norm = NA_real_,
    mean_variance_relative_shift = NA_real_,
    max_variance_relative_shift = NA_real_,
    min_variance_eigenvalue = NA_real_, max_variance_condition = NA_real_,
    sandwich_available = NA, sandwich_condition = NA_real_,
    score_eigen_mean = NA_real_, score_eigen_cv = NA_real_,
    score_eigen_ratio = NA_real_, generator_setup_seconds = NA_real_,
    unique_group_calibrations = NA_integer_, stringsAsFactors = FALSE)
  cbind(base, as.data.frame(as.list(stress_empty_p())))
}

.stress_add_flip <- function(out, flip, cell) {
  if (!identical(as.integer(flip$df), as.integer(cell$df))) {
    stop("score-flip rank was ", flip$df, ", expected ", cell$df,
         call. = FALSE)
  }
  peba4 <- tryCatch(magmaan:::infer_fmg_test(
    flip$statistic_effective, flip$df, flip$eigenvalues,
    method = "peba", param = 4)$p_value, error = function(e) NA_real_)
  out[c("p_flip_basic", "p_flip_effective", "p_flip_standardized",
        "p_score_sb", "p_score_peba4", "p_score_all",
        "p_score_sandwich")] <- list(
    flip$p_basic, flip$p_effective, flip$p_standardized,
    flip$p_mean_scaled, peba4, flip$p_mixture, flip$p_sandwich)
  out$flip_ok <- TRUE
  out$flip_setup_seconds <- flip$setup_seconds
  out$flip_score_seconds <- flip$resampling_score_seconds
  out$flip_standardization_seconds <- flip$resampling_standardization_seconds
  out$flip_asymptotic_seconds <- flip$asymptotic_seconds
  out$nuisance_stationarity_norm <- flip$nuisance_stationarity_norm
  out$mean_variance_relative_shift <- flip$mean_variance_relative_shift
  out$max_variance_relative_shift <- flip$max_variance_relative_shift
  out$min_variance_eigenvalue <- flip$min_variance_eigenvalue
  out$max_variance_condition <- flip$max_variance_condition
  out$sandwich_available <- flip$sandwich_available
  out$sandwich_condition <- flip$sandwich_condition
  out$score_eigen_mean <- mean(flip$eigenvalues)
  out$score_eigen_cv <- if (length(flip$eigenvalues) > 1L)
    stats::sd(flip$eigenvalues) / mean(flip$eigenvalues) else 0
  out$score_eigen_ratio <- max(flip$eigenvalues) / min(flip$eigenvalues)
  out
}

stress_one_rep <- function(base, cells, rep_id, sampler, specs, flips,
                           draw_seed, missing_seed, flip_seed) {
  replication_begin <- proc.time()[["elapsed"]]
  rows <- lapply(seq_len(nrow(cells)), function(k)
    stress_empty_replication(as.list(cells[k, , drop = FALSE]), rep_id))

  draw_begin <- proc.time()[["elapsed"]]
  complete <- tryCatch(stress_draw_replication(
    sampler, c(base$n_group1, base$n_group2), draw_seed), error = function(e) e)
  draw_seconds <- proc.time()[["elapsed"]] - draw_begin
  if (inherits(complete, "error")) {
    return(lapply(seq_along(rows), function(k) {
      out <- stress_empty_replication(as.list(cells[k, , drop = FALSE]), rep_id,
        paste0("simulation draw: ", conditionMessage(complete)))
      out$draw_seconds <- draw_seconds
      out$replication_seconds <- proc.time()[["elapsed"]] - replication_begin
      out
    }))
  }

  missing_begin <- proc.time()[["elapsed"]]
  masked <- tryCatch(stress_apply_missingness(
    complete, sampler$ov, base$missingness, missing_seed), error = function(e) e)
  missing_seconds <- proc.time()[["elapsed"]] - missing_begin
  if (inherits(masked, "error")) {
    return(lapply(seq_along(rows), function(k) {
      out <- stress_empty_replication(as.list(cells[k, , drop = FALSE]), rep_id,
        paste0("missingness: ", conditionMessage(masked)))
      out$draw_seconds <- draw_seconds
      out$missing_seconds <- missing_seconds
      out$replication_seconds <- proc.time()[["elapsed"]] - replication_begin
      out
    }))
  }
  for (k in seq_along(rows)) {
    rows[[k]]$draw_seconds <- draw_seconds
    rows[[k]]$missing_seconds <- missing_seconds
    rows[[k]]$realized_rate <- masked$realized_rate
    rows[[k]]$pattern_count <- masked$pattern_count
    rows[[k]]$selection_contrast <- masked$selection_contrast
    rows[[k]]$h1_fit_count <- 1L
  }

  h1_begin <- proc.time()[["elapsed"]]
  H1 <- tryCatch(stress_fit(specs$H1, masked$data), error = function(e) e)
  h1_seconds <- proc.time()[["elapsed"]] - h1_begin
  if (inherits(H1, "error") || !isTRUE(H1$converged)) {
    message <- if (inherits(H1, "error")) conditionMessage(H1) else
      "configural FIML fit did not converge"
    return(lapply(rows, function(out) {
      out$fit_error <- message
      out$fit_h1_seconds <- h1_seconds
      out$fit_seconds <- h1_seconds
      out$replication_seconds <- proc.time()[["elapsed"]] - replication_begin
      out
    }))
  }

  for (k in seq_len(nrow(cells))) {
    cell <- as.list(cells[k, , drop = FALSE])
    out <- rows[[k]]
    out$fit_h1_seconds <- h1_seconds
    h0_begin <- proc.time()[["elapsed"]]
    H0 <- tryCatch(stress_fit(specs$H0[[as.character(cell$rank)]], masked$data),
                   error = function(e) e)
    out$fit_h0_seconds <- proc.time()[["elapsed"]] - h0_begin
    out$fit_seconds <- h1_seconds + out$fit_h0_seconds
    if (inherits(H0, "error") || !isTRUE(H0$converged)) {
      out$fit_error <- if (inherits(H0, "error")) conditionMessage(H0) else
        "restricted FIML fit did not converge"
      rows[[k]] <- out
      next
    }
    out$fit_ok <- TRUE

    flip_begin <- proc.time()[["elapsed"]]
    flip <- tryCatch(magmaan::score_flip_test(
      H1, H0, n_flips = flips,
      seed = flip_seed + cell$cell_id * 1009), error = function(e) e)
    out$flip_seconds <- proc.time()[["elapsed"]] - flip_begin
    if (inherits(flip, "error")) out$flip_error <- conditionMessage(flip)
    else out <- .stress_add_flip(out, flip, cell)

    nested_begin <- proc.time()[["elapsed"]]
    nested <- tryCatch(magmaan::fmg_nested(
      H1, H0, tests = stress_nested_tests, A.method = "exact"),
      error = function(e) e)
    out$nested_seconds <- proc.time()[["elapsed"]] - nested_begin
    if (inherits(nested, "error")) {
      out$nested_error <- conditionMessage(nested)
    } else if (!all(nested$df == cell$df) ||
               !identical(as.character(nested$label), stress_nested_tests)) {
      out$nested_error <- "nested FMG battery returned unexpected labels or df"
    } else {
      out$nested_ok <- TRUE
      out[paste0("p_nested_", nested$label)] <- as.list(nested$p_value)
    }
    rows[[k]] <- out
  }
  elapsed <- proc.time()[["elapsed"]] - replication_begin
  lapply(rows, function(out) { out$replication_seconds <- elapsed; out })
}

stress_wilson <- function(x, n, z = 1.95996398454005) {
  if (!is.finite(n) || n <= 0L) return(c(lower = NA_real_, upper = NA_real_))
  phat <- x / n
  den <- 1 + z^2 / n
  centre <- (phat + z^2 / (2 * n)) / den
  half <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) / den
  c(lower = max(0, centre - half), upper = min(1, centre + half))
}
