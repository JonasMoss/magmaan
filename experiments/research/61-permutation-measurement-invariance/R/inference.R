# Adjacent invariance tests and permutation calibration for experiment 61.

stack_groups <- function(X1, X2) {
  out <- as.data.frame(rbind(X1, X2))
  out$group <- factor(rep(c("g1", "g2"), c(nrow(X1), nrow(X2))),
                      levels = c("g1", "g2"))
  out
}

fit_level <- function(X1, X2, pop, level) {
  fit <- magmaan::magmaan(
    invariance_syntax(pop, level), stack_groups(X1, X2), estimator = "ML",
    groups = "group",
    control = list(max_iter = 1000L, ftol = 1e-10, gtol = 1e-7))
  if (!isTRUE(fit$converged)) {
    stop("nonconverged ", level, " fit: ", fit$optimizer_status %||% "unknown",
         call. = FALSE)
  }
  fit
}

contrast_parameters <- function(pop, step) {
  if (step == "metric") {
    items <- setdiff(seq_len(pop$p), pop$marker)
    return(data.frame(lhs = pop$factor_names[pop$factor_of[items]], op = "=~",
                      rhs = pop$ov[items], stringsAsFactors = FALSE))
  }
  if (step == "scalar") {
    items <- setdiff(seq_len(pop$p), pop$marker)
    return(data.frame(lhs = pop$ov[items], op = "~1", rhs = "",
                      stringsAsFactors = FALSE))
  }
  if (step == "strict") {
    return(data.frame(lhs = pop$ov, op = "~~", rhs = pop$ov,
                      stringsAsFactors = FALSE))
  }
  stop("unknown invariance step: ", step, call. = FALSE)
}

invariance_contrast <- function(fit, pop, step) {
  pt <- fit$partable
  target <- contrast_parameters(pop, step)
  R <- matrix(0, nrow = nrow(target), ncol = length(fit$theta))
  for (j in seq_len(nrow(target))) {
    rows <- lapply(1:2, function(g) which(
      pt$group == g & pt$lhs == target$lhs[[j]] & pt$op == target$op[[j]] &
        pt$rhs == target$rhs[[j]] & pt$free > 0L))
    if (length(rows[[1L]]) != 1L || length(rows[[2L]]) != 1L) {
      stop("could not locate group-specific ", step, " contrast for ",
           target$lhs[[j]], " ", target$op[[j]], " ", target$rhs[[j]],
           call. = FALSE)
    }
    R[j, pt$free[rows[[2L]]]] <- 1
    R[j, pt$free[rows[[1L]]]] <- -1
  }
  R
}

wald_statistics <- function(X1, X2, pop, step) {
  levels <- step_models(step)
  fit <- fit_level(X1, X2, pop, levels[["h1"]])
  R <- invariance_contrast(fit, pop, step)
  model_vcov <- {
    info <- magmaan::magmaan_core$inference_information_expected(fit)
    magmaan::magmaan_core$inference_vcov_fit(info, fit)
  }
  sandwich_vcov <- magmaan::magmaan_core$infer_robust_se_raw_fit(
    fit, list(X1, X2), bread = "expected", moments = "structured",
    cov = "empirical")$vcov
  w_model <- magmaan::magmaan_core$infer_wald_test_fit(fit, R, model_vcov)$chi2
  w_sandwich <- magmaan::magmaan_core$infer_wald_test_fit(
    fit, R, sandwich_vcov)$chi2
  d <- drop(R %*% fit$theta)
  if (!all(is.finite(c(w_model, w_sandwich, d)))) {
    stop("non-finite Wald statistic", call. = FALSE)
  }
  list(fit = fit, R = R, df = nrow(R), w_model = w_model,
       w_sandwich = w_sandwich,
       raw = (nrow(X1) + nrow(X2)) * sum(d^2))
}

observed_statistics <- function(X1, X2, pop, step) {
  out <- wald_statistics(X1, X2, pop, step)
  levels <- step_models(step)
  fit0 <- fit_level(X1, X2, pop, levels[["h0"]])
  n_total <- nrow(X1) + nrow(X2)
  t_diff <- max(0, 2 * n_total * (fit0$fmin - out$fit$fmin))
  out$lrt <- t_diff
  out$lrt_p <- stats::pchisq(t_diff, df = out$df, lower.tail = FALSE)
  robust <- magmaan::robust_nested_lrt(
    out$fit, fit0, data = list(X1, X2), gamma = "empirical",
    method = "restriction_map")
  out$robust_lrt_p <- robust$p_scaled
  out
}

permutation_statistics <- function(X1, X2, pop, step, include_lrt = TRUE) {
  out <- wald_statistics(X1, X2, pop, step)
  out$lrt <- NA_real_
  if (include_lrt) {
    levels <- step_models(step)
    fit0 <- tryCatch(fit_level(X1, X2, pop, levels[["h0"]]),
                     error = function(e) NULL)
    if (!is.null(fit0)) {
      n_total <- nrow(X1) + nrow(X2)
      out$lrt <- max(0, 2 * n_total * (fit0$fmin - out$fit$fmin))
    }
  }
  out
}

permutation_pvalues <- function(X1, X2, pop, step, observed, permutations, seed,
                                include_lrt = TRUE) {
  n1 <- nrow(X1)
  pooled <- rbind(X1, X2)
  set.seed(as.integer(seed %% (.Machine$integer.max - 1L)))
  indices <- replicate(permutations, sample.int(nrow(pooled)), simplify = FALSE)
  draws <- lapply(indices, function(index) {
    tryCatch(permutation_statistics(
      pooled[index[seq_len(n1)], , drop = FALSE],
      pooled[index[-seq_len(n1)], , drop = FALSE], pop, step, include_lrt),
      error = function(e) NULL)
  })
  w <- vapply(draws, function(x) if (is.null(x)) NA_real_ else x$w_sandwich,
              numeric(1))
  raw <- vapply(draws, function(x) if (is.null(x)) NA_real_ else x$raw,
                numeric(1))
  lrt <- vapply(draws, function(x) if (is.null(x)) NA_real_ else x$lrt,
                numeric(1))
  valid_w <- w[is.finite(w)]
  valid_raw <- raw[is.finite(raw)]
  valid_lrt <- lrt[is.finite(lrt)]
  list(
    p_studentized = if (length(valid_w))
      (1 + sum(valid_w >= observed$w_sandwich)) / (1 + length(valid_w)) else NA_real_,
    p_raw = if (length(valid_raw))
      (1 + sum(valid_raw >= observed$raw)) / (1 + length(valid_raw)) else NA_real_,
    p_lrt = if (include_lrt && length(valid_lrt))
      (1 + sum(valid_lrt >= observed$lrt)) / (1 + length(valid_lrt)) else NA_real_,
    n_studentized = length(valid_w), n_raw = length(valid_raw),
    n_lrt = length(valid_lrt))
}

empty_replication <- function(rep_id, error = NA_character_) {
  data.frame(
    rep = rep_id, observed_ok = FALSE, error = error,
    p_lrt = NA_real_, p_robust_lrt = NA_real_, p_model_wald = NA_real_,
    p_sandwich_wald = NA_real_, p_permutation_studentized = NA_real_,
    p_permutation_lrt = NA_real_, p_permutation_raw = NA_real_,
    w_model = NA_real_, w_sandwich = NA_real_,
    n_perm_studentized = 0L, n_perm_lrt = 0L, n_perm_raw = 0L,
    simulation_seconds = NA_real_, observed_seconds = NA_real_,
    permutation_seconds = NA_real_, total_seconds = NA_real_,
    stringsAsFactors = FALSE)
}

run_replication <- function(rep_id, X, pop, step, permutations, permutation_seed,
                            simulation_seconds) {
  started <- proc.time()[["elapsed"]]
  observed_started <- proc.time()[["elapsed"]]
  observed <- tryCatch(observed_statistics(X[[1L]], X[[2L]], pop, step),
                       error = function(e) e)
  observed_seconds <- proc.time()[["elapsed"]] - observed_started
  if (inherits(observed, "error")) {
    out <- empty_replication(rep_id, conditionMessage(observed))
    out$simulation_seconds <- simulation_seconds
    out$observed_seconds <- observed_seconds
    out$total_seconds <- proc.time()[["elapsed"]] - started + simulation_seconds
    return(out)
  }
  permutation_started <- proc.time()[["elapsed"]]
  perm <- permutation_pvalues(X[[1L]], X[[2L]], pop, step, observed,
                              permutations, permutation_seed)
  permutation_seconds <- proc.time()[["elapsed"]] - permutation_started
  out <- empty_replication(rep_id)
  out$observed_ok <- TRUE
  out$error <- NA_character_
  out$p_lrt <- observed$lrt_p
  out$p_robust_lrt <- observed$robust_lrt_p
  out$p_model_wald <- stats::pchisq(observed$w_model, df = observed$df,
                                    lower.tail = FALSE)
  out$p_sandwich_wald <- stats::pchisq(observed$w_sandwich, df = observed$df,
                                       lower.tail = FALSE)
  out$p_permutation_studentized <- perm$p_studentized
  out$p_permutation_lrt <- perm$p_lrt
  out$p_permutation_raw <- perm$p_raw
  out$w_model <- observed$w_model
  out$w_sandwich <- observed$w_sandwich
  out$n_perm_studentized <- perm$n_studentized
  out$n_perm_lrt <- perm$n_lrt
  out$n_perm_raw <- perm$n_raw
  out$simulation_seconds <- simulation_seconds
  out$observed_seconds <- observed_seconds
  out$permutation_seconds <- permutation_seconds
  out$total_seconds <- proc.time()[["elapsed"]] - started + simulation_seconds
  out
}
