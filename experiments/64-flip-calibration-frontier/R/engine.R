frontier_nested_tests <- c(
  "std_ml", "sb_ml", "ss_ml", "mv_ml", "peba4_ml", "pall_ml", "all_ml",
  "ss_rls", "peba4_rls")

frontier_p_columns <- c(
  "p_flip_basic", "p_flip_effective", "p_flip_standardized",
  "p_score_chisq", "p_score_sb", "p_score_mv", "p_score_ss",
  "p_score_peba2", "p_score_peba4", "p_score_peba6",
  "p_score_pall", "p_score_all", "p_score_sandwich",
  paste0("p_nested_", frontier_nested_tests))

frontier_method_table <- function() {
  method <- sub("^p_", "", frontier_p_columns)
  data.frame(
    method = method,
    family = ifelse(startsWith(method, "flip_"), "flip",
                    ifelse(startsWith(method, "score_"), "score", "nested")),
    primary = method %in% c("flip_effective", "flip_standardized",
                            "score_peba4"),
    stringsAsFactors = FALSE)
}

frontier_empty_p <- function() {
  stats::setNames(rep(NA_real_, length(frontier_p_columns)), frontier_p_columns)
}

.frontier_calibrate_group <- function(Sigma, distribution) {
  p <- ncol(Sigma)
  skew <- rep(3, p)
  exkurt <- rep(21, p)
  if (distribution == "normal") {
    return(list(distribution = distribution, L = chol(Sigma), Sigma = Sigma))
  }
  if (distribution == "vm") {
    cal <- magmaan:::sim_vm_calibrate_impl(stats::cov2cor(Sigma), skew, exkurt)
  } else if (distribution == "ig") {
    cal <- magmaan:::sim_ig_calibrate_impl(
      Sigma, skew, exkurt, root = "symmetric", generator_family = "pearson",
      quadrature_points = 81L)
  } else if (distribution == "pl") {
    cal <- magmaan:::sim_plsim_calibrate_impl(
      stats::cov2cor(Sigma), skew, exkurt,
      method = "hermite_then_rectangle", num_segments = 12L,
      quadrature_points = 31L, hermite_order = 24L)
  } else stop("unknown distribution: ", distribution, call. = FALSE)
  list(distribution = distribution, calibration = cal, Sigma = Sigma)
}

frontier_calibrate_sampler <- function(pop, distribution) {
  begin <- proc.time()[["elapsed"]]
  group_cache <- new.env(parent = emptyenv())
  states <- lapply(pop$Sigma, function(Sigma) {
    key <- paste(c(nrow(Sigma), formatC(as.vector(Sigma), digits = 17,
                                        format = "fg", flag = "#")),
                 collapse = "|")
    if (!exists(key, envir = group_cache, inherits = FALSE)) {
      assign(key, .frontier_calibrate_group(Sigma, distribution),
             envir = group_cache)
    }
    get(key, envir = group_cache, inherits = FALSE)
  })
  list(p = pop$p, groups = pop$groups, distribution = distribution,
       states = states, unique_group_calibrations = length(ls(group_cache)),
       setup_seconds = proc.time()[["elapsed"]] - begin)
}

.frontier_draw_group <- function(state, n, seed) {
  p <- ncol(state$Sigma)
  distribution <- state$distribution
  if (distribution == "normal") {
    set.seed(seed + 1)
    return(matrix(stats::rnorm(n * p), n, p) %*% state$L)
  }
  if (distribution == "vm") {
    batch <- magmaan:::sim_vm_draw_impl(state$calibration, n = n, reps = 1L,
                                        seed_base = seed)
  } else if (distribution == "ig") {
    batch <- magmaan:::sim_ig_draw_impl(
      state$calibration, n = n, reps = 1L, seed_base = seed,
      quadrature_points = 81L)
  } else {
    batch <- magmaan:::sim_plsim_draw_impl(
      state$calibration, n = n, reps = 1L, seed_base = seed)
  }
  X <- batch$draws[[1L]]
  if (distribution %in% c("vm", "pl")) {
    X <- sweep(X, 2L, sqrt(diag(state$Sigma)), "*")
  }
  X
}

frontier_draw_replication <- function(sampler, group_sizes, seed) {
  stopifnot(length(group_sizes) == sampler$groups)
  varnames <- paste0("x", seq_len(sampler$p))
  labels <- paste0("g", seq_len(sampler$groups))
  blocks <- lapply(seq_len(sampler$groups), function(g) {
    X <- .frontier_draw_group(sampler$states[[g]], group_sizes[[g]],
                              seed + g * 1000003)
    colnames(X) <- varnames
    storage.mode(X) <- "double"
    X
  })
  data <- data.frame(
    do.call(rbind, blocks),
    group = factor(rep(labels, times = group_sizes), levels = labels),
    check.names = FALSE)
  list(data = data, blocks = blocks)
}

.frontier_empty_replication <- function(rep_id, error = "") {
  base <- data.frame(
    rep = rep_id, fit_ok = FALSE, flip_ok = FALSE, nested_ok = FALSE,
    fit_error = error, flip_error = "", nested_error = "",
    draw_seconds = NA_real_, fit_seconds = NA_real_, flip_seconds = NA_real_,
    nested_seconds = NA_real_, total_seconds = NA_real_,
    flip_setup_seconds = NA_real_, flip_score_seconds = NA_real_,
    flip_standardization_seconds = NA_real_, flip_asymptotic_seconds = NA_real_,
    nuisance_stationarity_norm = NA_real_,
    mean_variance_relative_shift = NA_real_,
    max_variance_relative_shift = NA_real_, min_variance_eigenvalue = NA_real_,
    max_variance_condition = NA_real_, sandwich_available = NA,
    sandwich_condition = NA_real_, score_eigen_mean = NA_real_,
    score_eigen_cv = NA_real_, score_eigen_ratio = NA_real_,
    stringsAsFactors = FALSE)
  cbind(base, as.data.frame(as.list(frontier_empty_p())))
}

frontier_one_rep <- function(cell, rep_id, sampler, specs, group_sizes,
                             flips, dgp_seed, flip_seed) {
  total_begin <- proc.time()[["elapsed"]]
  draw_begin <- proc.time()[["elapsed"]]
  sample <- tryCatch(frontier_draw_replication(sampler, group_sizes, dgp_seed),
                     error = function(e) e)
  draw_seconds <- proc.time()[["elapsed"]] - draw_begin
  if (inherits(sample, "error")) {
    out <- .frontier_empty_replication(
      rep_id, paste0("simulation draw: ", conditionMessage(sample)))
    out$draw_seconds <- draw_seconds
    out$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(out)
  }

  fit_begin <- proc.time()[["elapsed"]]
  fits <- tryCatch(lapply(specs, function(spec) magmaan::magmaan(
    spec, sample$data, estimator = "ML",
    optimizer = "nlopt-lbfgs-slsqp-fallback", se = "none", test = "none")),
    error = function(e) e)
  fit_seconds <- proc.time()[["elapsed"]] - fit_begin
  out <- .frontier_empty_replication(rep_id)
  out$draw_seconds <- draw_seconds
  out$fit_seconds <- fit_seconds
  if (inherits(fits, "error")) {
    out$fit_error <- conditionMessage(fits)
    out$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(out)
  }
  names(fits) <- names(specs)
  if (!all(vapply(fits, function(x) isTRUE(x$converged), logical(1)))) {
    out$fit_error <- "one or both ML fits did not converge"
    out$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(out)
  }
  out$fit_ok <- TRUE

  flip_begin <- proc.time()[["elapsed"]]
  flip <- tryCatch(magmaan::score_flip_test(
    fits$H1, fits$H0, sample$blocks, n_flips = flips, seed = flip_seed),
    error = function(e) e)
  out$flip_seconds <- proc.time()[["elapsed"]] - flip_begin
  if (inherits(flip, "error")) {
    out$flip_error <- conditionMessage(flip)
  } else {
    if (!identical(as.integer(flip$df), as.integer(cell$df))) {
      stop("score flip rank was ", flip$df, ", expected ", cell$df,
           call. = FALSE)
    }
    out$flip_ok <- TRUE
    infer_score <- function(method, param = 4) tryCatch(
      magmaan:::infer_fmg_test(
        flip$statistic_effective, flip$df, flip$eigenvalues,
        method = method, param = param)$p_value,
      error = function(e) NA_real_)
    values <- frontier_empty_p()
    values[c("p_flip_basic", "p_flip_effective", "p_flip_standardized",
             "p_score_chisq", "p_score_sb", "p_score_all",
             "p_score_sandwich")] <-
      c(flip$p_basic, flip$p_effective, flip$p_standardized,
        flip$p_chisq, flip$p_mean_scaled, flip$p_mixture, flip$p_sandwich)
    values["p_score_mv"] <- infer_score("mv")
    values["p_score_ss"] <- infer_score("ss")
    values["p_score_peba2"] <- infer_score("peba", 2)
    values["p_score_peba4"] <- infer_score("peba", 4)
    values["p_score_peba6"] <- infer_score("peba", 6)
    values["p_score_pall"] <- infer_score("penalized_all")
    out[names(values)] <- as.list(values)
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
    out$score_eigen_cv <- if (length(flip$eigenvalues) > 1L)
      stats::sd(flip$eigenvalues) / mean(flip$eigenvalues) else 0
    out$score_eigen_ratio <- max(flip$eigenvalues) / min(flip$eigenvalues)
  }

  nested_begin <- proc.time()[["elapsed"]]
  nested <- tryCatch(magmaan::fmg_nested(
    fits$H1, fits$H0, data = sample$blocks, tests = frontier_nested_tests,
    A.method = "exact"), error = function(e) e)
  out$nested_seconds <- proc.time()[["elapsed"]] - nested_begin
  if (inherits(nested, "error")) {
    out$nested_error <- conditionMessage(nested)
  } else {
    if (!all(nested$df == cell$df) ||
        !identical(as.character(nested$label), frontier_nested_tests)) {
      stop("nested FMG result did not preserve the fixed 28-df design",
           call. = FALSE)
    }
    out$nested_ok <- TRUE
    out[paste0("p_nested_", nested$label)] <- as.list(nested$p_value)
  }
  out$total_seconds <- proc.time()[["elapsed"]] - total_begin
  out
}

frontier_wilson <- function(x, n, z = 1.95996398454005) {
  if (!is.finite(n) || n <= 0L) return(c(lower = NA_real_, upper = NA_real_))
  phat <- x / n
  den <- 1 + z^2 / n
  centre <- (phat + z^2 / (2 * n)) / den
  half <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) / den
  c(lower = max(0, centre - half), upper = min(1, centre + half))
}
