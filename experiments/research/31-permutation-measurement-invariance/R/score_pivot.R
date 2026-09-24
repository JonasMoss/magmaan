# Focused fully recomputed permutation probe for pivotal robust nested scores.

score_pivot_population <- function(p = 9L, q = 0L, delta = 0) {
  p <- as.integer(p)
  if (p != 9L) stop("score-pivot probe is fixed at p = 9", call. = FALSE)
  q <- as.integer(q)
  delta <- as.numeric(delta)
  if (q < 0L || q > p - 1L || !is.finite(delta)) {
    stop("alternative q/delta is invalid", call. = FALSE)
  }
  lambda_1 <- c(1, .78, .72, .83, .68, .75, .80, .70, .76)
  lambda_2 <- lambda_1
  if (q > 0L && delta != 0) {
    selected <- seq.int(2L, q + 1L)
    lambda_2[selected] <- lambda_2[selected] + delta
  }
  theta_1 <- c(.55, .62, .70, .58, .76, .67, .61, .73, .65)
  theta_2 <- theta_1 * c(1.35, .80, 1.25, .75, 1.40, .85, 1.20, .90, 1.30)
  phi <- c(1, 1.45)
  sigma_1 <- phi[[1L]] * tcrossprod(lambda_1) + diag(theta_1)
  sigma_2 <- phi[[2L]] * tcrossprod(lambda_2) + diag(theta_2)
  list(
    p = p, ov = paste0("x", seq_len(p)),
    lambda = list(lambda_1, lambda_2),
    theta = list(theta_1, theta_2), phi = phi,
    Sigma = list(sigma_1, sigma_2), alternative_q = q, delta = delta)
}

score_pivot_syntax <- function(p, q = 0L) {
  p <- as.integer(p)
  q <- as.integer(q)
  if (q < 0L || q > p - 1L) {
    stop("q must be between zero and p - 1", call. = FALSE)
  }
  rhs <- paste0("x", seq_len(p))
  if (q > 0L) {
    selected <- seq.int(2L, q + 1L)
    rhs[selected] <- paste0("l", selected, "*", rhs[selected])
  }
  paste0("f =~ ", paste(rhs, collapse = " + "))
}

score_pivot_specs <- function(q, p = 9L) {
  group_options <- list(
    group = "group", group_labels = c("g1", "g2"),
    std_lv = FALSE, meanstructure = FALSE)
  h1 <- do.call(
    magmaan::model_spec,
    c(list(syntax = score_pivot_syntax(p, q = 0L)), group_options))
  h0 <- do.call(
    magmaan::model_spec,
    c(list(syntax = score_pivot_syntax(p, q = q)), group_options))
  list(H1 = h1, H0 = h0, q = as.integer(q), p = as.integer(p))
}

score_pivot_calibrate_samplers <- function(pop, skew = 2, exkurt = 7) {
  normal <- list(
    regime = "exchangeable_normal",
    kind = "normal",
    L = list(chol(pop$Sigma[[1L]]), chol(pop$Sigma[[1L]])))
  vm_skew <- list(rep(skew / 2, pop$p), rep(skew, pop$p))
  vm_exkurt <- list(rep(exkurt / 2, pop$p), rep(exkurt, pop$p))
  vm <- list(
    regime = "heterogeneous_vm",
    kind = "vm",
    calibration = lapply(seq_len(2L), function(g) {
      magmaan:::sim_vm_calibrate_impl(
        stats::cov2cor(pop$Sigma[[g]]), vm_skew[[g]], vm_exkurt[[g]])
    }),
    sds = lapply(pop$Sigma, function(Sigma) sqrt(diag(Sigma)))
  )
  list(exchangeable_normal = normal, heterogeneous_vm = vm)
}

score_pivot_draw <- function(sampler, group_sizes, seed, ov) {
  group_sizes <- as.integer(group_sizes)
  if (length(group_sizes) != 2L || anyNA(group_sizes) ||
      any(group_sizes < 2L)) {
    stop("group_sizes must contain two integers of at least two", call. = FALSE)
  }
  blocks <- lapply(seq_len(2L), function(g) {
    n_group <- group_sizes[[g]]
    block_seed <- as.integer((seed + g * 1000003) %%
                               (.Machine$integer.max - 1L))
    if (sampler$kind == "normal") {
      set.seed(block_seed)
      X <- matrix(
        stats::rnorm(n_group * length(ov)), nrow = n_group) %*%
        sampler$L[[g]]
    } else {
      draw <- magmaan:::sim_vm_draw_impl(
        sampler$calibration[[g]], n = n_group, reps = 1L,
        seed_base = block_seed)
      X <- sweep(draw$draws[[1L]], 2L, sampler$sds[[g]], "*")
    }
    colnames(X) <- ov
    storage.mode(X) <- "double"
    X
  })
  blocks
}

score_pivot_stack <- function(X1, X2) {
  out <- as.data.frame(rbind(X1, X2))
  out$group <- factor(
    rep(c("g1", "g2"), c(nrow(X1), nrow(X2))),
    levels = c("g1", "g2"))
  out
}

score_pivot_fit <- function(spec, X1, X2) {
  fit <- magmaan::magmaan(
    spec, score_pivot_stack(X1, X2), estimator = "ML", groups = "group",
    optimizer = "nlopt-lbfgs-slsqp-fallback",
    control = list(max_iter = 1000L, ftol = 1e-10, gtol = 1e-7),
    se = "none", test = "none")
  if (!isTRUE(fit$converged)) {
    stop("nonconverged fit: ", fit$optimizer_status %||% "unknown",
         call. = FALSE)
  }
  fit
}

score_pivot_hotelling_p <- function(statistic, q, n_total) {
  if (!is.finite(statistic) || n_total <= q ||
      statistic < 0 || statistic >= n_total) {
    return(NA_real_)
  }
  # The nested-score core uses the raw OPG. If Qr is that quadratic and Qc
  # uses the unbiased centered covariance, then
  #   Qr = n Qc / (n - 1 + Qc).
  # Substitution in the ordinary Hotelling reference gives this direct map.
  transformed <- (n_total - q) * statistic /
    (q * (n_total - statistic))
  stats::pf(transformed, df1 = q, df2 = n_total - q, lower.tail = FALSE)
}

score_pivot_contrast <- function(fit, q) {
  pt <- fit$partable
  R <- matrix(0, nrow = q, ncol = length(fit$theta))
  for (j in seq_len(q)) {
    rhs <- paste0("x", j + 1L)
    rows <- lapply(seq_len(2L), function(g) which(
      pt$group == g & pt$lhs == "f" & pt$op == "=~" &
        pt$rhs == rhs & pt$free > 0L))
    if (any(lengths(rows) != 1L)) {
      stop("could not locate group-specific loading contrast for ", rhs,
           call. = FALSE)
    }
    R[j, pt$free[rows[[2L]]]] <- 1
    R[j, pt$free[rows[[1L]]]] <- -1
  }
  R
}

score_pivot_statistic <- function(X1, X2, specs, include_wald = TRUE) {
  n_total <- nrow(X1) + nrow(X2)
  score_started <- proc.time()[["elapsed"]]
  score <- tryCatch({
    fit0 <- score_pivot_fit(specs$H0, X1, X2)
    test <- magmaan::nested_score_test(
      specs$H1, fit0, data = list(X1, X2))
    if (!isTRUE(test$sandwich_available) ||
        !identical(as.integer(test$df), specs$q)) {
      stop("sandwich score unavailable or rank mismatch", call. = FALSE)
    }
    p_fmg <- function(method, param = 4) {
      magmaan:::infer_fmg_test(
        test$statistic_effective, test$df, test$eigenvalues,
        method = method, param = param)$p_value
    }
    list(
      ok = TRUE, error = "",
      statistic = test$statistic_sandwich,
      p_chisq = test$p_sandwich,
      p_hotelling = score_pivot_hotelling_p(
        test$statistic_sandwich, specs$q, n_total),
      p_sb = test$p_mean_scaled,
      p_mv = p_fmg("mv"),
      p_peba4 = test$p_peba4,
      p_all = test$p_mixture,
      eigen_mean = mean(test$eigenvalues),
      eigen_cv = if (length(test$eigenvalues) > 1L) {
        stats::sd(test$eigenvalues) / mean(test$eigenvalues)
      } else 0,
      eigen_ratio = max(test$eigenvalues) / min(test$eigenvalues))
  }, error = function(e) {
    list(ok = FALSE, error = conditionMessage(e), statistic = NA_real_,
         p_chisq = NA_real_, p_hotelling = NA_real_, p_sb = NA_real_,
         p_mv = NA_real_, p_peba4 = NA_real_, p_all = NA_real_,
         eigen_mean = NA_real_, eigen_cv = NA_real_,
         eigen_ratio = NA_real_)
  })
  score_seconds <- proc.time()[["elapsed"]] - score_started

  wald <- list(ok = NA, error = "", statistic = NA_real_, p_chisq = NA_real_)
  wald_seconds <- 0
  if (isTRUE(include_wald)) {
    wald_started <- proc.time()[["elapsed"]]
    wald <- tryCatch({
      fit1 <- score_pivot_fit(specs$H1, X1, X2)
      R <- score_pivot_contrast(fit1, specs$q)
      vcov <- magmaan::magmaan_core$infer_robust_se_raw_fit(
        fit1, list(X1, X2), bread = "expected", moments = "structured",
        cov = "empirical")$vcov
      statistic <- magmaan::magmaan_core$infer_wald_test_fit(
        fit1, R, vcov)$chi2
      list(
        ok = is.finite(statistic), error = "", statistic = statistic,
        p_chisq = stats::pchisq(
          statistic, df = specs$q, lower.tail = FALSE))
    }, error = function(e) {
      list(ok = FALSE, error = conditionMessage(e),
           statistic = NA_real_, p_chisq = NA_real_)
    })
    wald_seconds <- proc.time()[["elapsed"]] - wald_started
  }
  list(
    score = score, wald = wald,
    score_seconds = score_seconds, wald_seconds = wald_seconds)
}

score_pivot_permutation <- function(X1, X2, specs, observed, permutations,
                                    seed, include_wald = TRUE) {
  n1 <- nrow(X1)
  pooled <- rbind(X1, X2)
  set.seed(as.integer(seed %% (.Machine$integer.max - 1L)))
  indices <- replicate(
    as.integer(permutations), sample.int(nrow(pooled)), simplify = FALSE)
  draws <- lapply(indices, function(index) {
    score_pivot_statistic(
      pooled[index[seq_len(n1)], , drop = FALSE],
      pooled[index[-seq_len(n1)], , drop = FALSE],
      specs, include_wald = include_wald)
  })

  score_stat <- vapply(
    draws, function(x) x$score$statistic, numeric(1))
  score_hotelling <- vapply(
    draws, function(x) x$score$p_hotelling, numeric(1))
  valid_score <- is.finite(score_stat) & is.finite(score_hotelling)
  score_stat <- score_stat[valid_score]
  score_hotelling <- score_hotelling[valid_score]
  p_score <- if (length(score_stat)) {
    (1 + sum(score_stat >= observed$score$statistic)) /
      (1 + length(score_stat))
  } else NA_real_
  p_hotelling <- if (length(score_hotelling)) {
    (1 + sum(score_hotelling <= observed$score$p_hotelling)) /
      (1 + length(score_hotelling))
  } else NA_real_
  rank_identity_error <- if (is.finite(p_score) && is.finite(p_hotelling)) {
    abs(p_score - p_hotelling)
  } else NA_real_
  if (is.finite(rank_identity_error) && rank_identity_error > 1e-15) {
    stop("permutation score and Hotelling ranks differ", call. = FALSE)
  }

  wald_stat <- vapply(draws, function(x) x$wald$statistic, numeric(1))
  wald_stat <- wald_stat[is.finite(wald_stat)]
  p_wald <- if (isTRUE(include_wald) && length(wald_stat) &&
                is.finite(observed$wald$statistic)) {
    (1 + sum(wald_stat >= observed$wald$statistic)) /
      (1 + length(wald_stat))
  } else NA_real_
  list(
    p_score = p_score, p_hotelling = p_hotelling, p_wald = p_wald,
    n_score = length(score_stat), n_wald = length(wald_stat),
    rank_identity_error = rank_identity_error,
    score_seconds = sum(vapply(
      draws, function(x) x$score_seconds, numeric(1))),
    wald_seconds = sum(vapply(
      draws, function(x) x$wald_seconds, numeric(1))))
}

score_pivot_empty_replication <- function(rep_id, seed, error = "") {
  data.frame(
    rep = rep_id, seed = seed, observed_score_ok = FALSE,
    observed_wald_ok = FALSE, error_score = error, error_wald = "",
    statistic_score_sandwich = NA_real_,
    statistic_wald_sandwich = NA_real_,
    p_score_chisq = NA_real_, p_score_hotelling = NA_real_,
    p_score_sb = NA_real_, p_score_mv = NA_real_,
    p_score_peba4 = NA_real_, p_score_all = NA_real_,
    score_eigen_mean = NA_real_, score_eigen_cv = NA_real_,
    score_eigen_ratio = NA_real_,
    p_wald_chisq = NA_real_,
    p_permutation_score = NA_real_,
    p_permutation_hotelling = NA_real_,
    p_permutation_wald = NA_real_,
    n_permutation_score = 0L, n_permutation_wald = 0L,
    permutation_rank_identity_error = NA_real_,
    observed_score_seconds = NA_real_, observed_wald_seconds = NA_real_,
    permutation_score_seconds = NA_real_,
    permutation_wald_seconds = NA_real_,
    simulation_seconds = NA_real_, observed_seconds = NA_real_,
    permutation_seconds = NA_real_, total_seconds = NA_real_,
    stringsAsFactors = FALSE)
}

score_pivot_replication <- function(rep_id, cell, specs, sampler, pop,
                                    permutations, seed_base,
                                    include_wald = TRUE) {
  seed <- as.numeric(seed_base + cell$cell_id * 1000003 + rep_id * 1009)
  started <- proc.time()[["elapsed"]]
  simulation_started <- proc.time()[["elapsed"]]
  X <- tryCatch(
    score_pivot_draw(sampler, c(cell$n1, cell$n2), seed, pop$ov),
    error = function(e) e)
  simulation_seconds <- proc.time()[["elapsed"]] - simulation_started
  if (inherits(X, "error")) {
    out <- score_pivot_empty_replication(
      rep_id, seed, paste0("simulation: ", conditionMessage(X)))
    out$simulation_seconds <- simulation_seconds
    out$total_seconds <- proc.time()[["elapsed"]] - started
    return(out)
  }

  observed_started <- proc.time()[["elapsed"]]
  observed <- score_pivot_statistic(
    X[[1L]], X[[2L]], specs, include_wald = include_wald)
  observed_seconds <- proc.time()[["elapsed"]] - observed_started
  out <- score_pivot_empty_replication(rep_id, seed)
  out$observed_score_ok <- observed$score$ok
  out$observed_wald_ok <- observed$wald$ok
  out$error_score <- observed$score$error
  out$error_wald <- observed$wald$error
  out$observed_score_seconds <- observed$score_seconds
  out$observed_wald_seconds <- observed$wald_seconds
  out$simulation_seconds <- simulation_seconds
  out$observed_seconds <- observed_seconds
  if (!observed$score$ok) {
    out$total_seconds <- proc.time()[["elapsed"]] - started
    return(out)
  }

  out$statistic_score_sandwich <- observed$score$statistic
  out$statistic_wald_sandwich <- observed$wald$statistic
  out$p_score_chisq <- observed$score$p_chisq
  out$p_score_hotelling <- observed$score$p_hotelling
  out$p_score_sb <- observed$score$p_sb
  out$p_score_mv <- observed$score$p_mv
  out$p_score_peba4 <- observed$score$p_peba4
  out$p_score_all <- observed$score$p_all
  out$score_eigen_mean <- observed$score$eigen_mean
  out$score_eigen_cv <- observed$score$eigen_cv
  out$score_eigen_ratio <- observed$score$eigen_ratio
  out$p_wald_chisq <- observed$wald$p_chisq

  if (permutations < 1L) {
    out$permutation_seconds <- 0
    out$permutation_score_seconds <- 0
    out$permutation_wald_seconds <- 0
    out$total_seconds <- proc.time()[["elapsed"]] - started
    return(out)
  }

  permutation_started <- proc.time()[["elapsed"]]
  perm <- tryCatch(
    score_pivot_permutation(
      X[[1L]], X[[2L]], specs, observed, permutations,
      seed = seed + 700000001, include_wald = include_wald),
    error = function(e) e)
  out$permutation_seconds <- proc.time()[["elapsed"]] - permutation_started
  if (inherits(perm, "error")) {
    out$error_score <- paste0("permutation: ", conditionMessage(perm))
  } else {
    out$p_permutation_score <- perm$p_score
    out$p_permutation_hotelling <- perm$p_hotelling
    out$p_permutation_wald <- perm$p_wald
    out$n_permutation_score <- perm$n_score
    out$n_permutation_wald <- perm$n_wald
    out$permutation_rank_identity_error <- perm$rank_identity_error
    out$permutation_score_seconds <- perm$score_seconds
    out$permutation_wald_seconds <- perm$wald_seconds
  }
  out$total_seconds <- proc.time()[["elapsed"]] - started
  out
}
