# Self-contained projected-score machinery for the ordinary-Hotelling GOF grid.
# This experiment is a leaf and consumes only magmaan plus _support harness code.

hrg_loadings <- c(.8, .9, 1.1, 1.4, .7, 1.4, 1.4, 1.2)

hrg_population <- function(p) {
  stopifnot(p %in% 4:8)
  lambda <- hrg_loadings[seq_len(p)]
  theta <- diag(p)
  Sigma <- tcrossprod(lambda) + theta
  list(
    p = p, lambda = lambda, theta = theta, Sigma = Sigma,
    q = as.integer(p * (p - 3L) / 2L))
}

hrg_model_spec <- function(p) {
  magmaan::model_spec(
    paste0("f =~ ", paste0("x", seq_len(p), collapse = " + ")),
    std_lv = FALSE, meanstructure = FALSE)
}

hrg_calibrate_sampler <- function(pop, distribution,
                                  skew = 3, exkurt = 21) {
  if (distribution == "normal") {
    return(list(kind = "normal", L = chol(pop$Sigma)))
  }
  if (distribution != "vm") {
    stop("unknown distribution: ", distribution, call. = FALSE)
  }
  list(
    kind = "vm",
    calibration = magmaan:::sim_vm_calibrate_impl(
      stats::cov2cor(pop$Sigma),
      rep(skew, pop$p), rep(exkurt, pop$p)),
    sds = sqrt(diag(pop$Sigma)))
}

hrg_draw <- function(sampler, n, seed) {
  if (sampler$kind == "normal") {
    set.seed(as.integer(seed))
    X <- matrix(
      stats::rnorm(n * ncol(sampler$L)), nrow = n) %*% sampler$L
  } else {
    batch <- magmaan:::sim_vm_draw_impl(
      sampler$calibration, n = as.integer(n), reps = 1L,
      seed_base = as.integer(seed))
    X <- sweep(batch$draws[[1L]], 2L, sampler$sds, "*")
  }
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  storage.mode(X) <- "double"
  X
}

hrg_fit <- function(spec, X) {
  magmaan::magmaan(
    spec, as.data.frame(X), estimator = "ML",
    optimizer = "nlopt-lbfgs-slsqp-fallback",
    se = "none", test = "none")
}

# Model-centred saturated covariance contributions, projected onto the
# expected-information complement of the fitted model tangent.
hrg_projected_scores <- function(fit, X) {
  core <- magmaan::magmaan_core
  uf <- core$infer_build_u_factor(
    fit, bread = "expected", moments = "structured")
  if (!identical(uf$kind, "ProjectionExpected") ||
      length(uf$blocks) != 1L || isTRUE(uf$has_means)) {
    stop("expected one covariance-only ML block", call. = FALSE)
  }
  centred <- core$infer_casewise_contributions(fit$partable, X)
  residual_matrix <- uf$blocks[[1L]]$S - uf$blocks[[1L]]$Sigma_hat
  residual <- residual_matrix[lower.tri(residual_matrix, diag = TRUE)]
  if (ncol(centred) != length(residual) ||
      nrow(uf$B) != length(residual)) {
    stop("covariance contribution layout mismatch", call. = FALSE)
  }
  psi <- sweep(centred, 2L, residual, "+") %*% uf$B
  if (ncol(psi) < 1L || any(!is.finite(psi))) {
    stop("projected score matrix is empty or non-finite", call. = FALSE)
  }
  psi
}

hrg_inverse_quadratic <- function(u, matrix, label) {
  eig <- eigen(0.5 * (matrix + t(matrix)), symmetric = TRUE)
  tol <- 1e-10 * max(1, max(eig$values))
  available <- all(is.finite(eig$values)) && min(eig$values) > tol
  if (!available) {
    return(list(
      statistic = NA_real_, available = FALSE,
      min_eigenvalue = min(eig$values), condition = Inf,
      error = paste0(label, " is not positive definite")))
  }
  coordinates <- crossprod(eig$vectors, u)
  list(
    statistic = sum(coordinates^2 / eig$values),
    available = TRUE,
    min_eigenvalue = min(eig$values),
    condition = max(eig$values) / min(eig$values),
    error = "")
}

hrg_hotelling_p <- function(statistic, q, n) {
  if (!is.finite(statistic) || n <= q) return(NA_real_)
  transformed <- (n - q) * statistic / (q * (n - 1))
  stats::pf(transformed, df1 = q, df2 = n - q, lower.tail = FALSE)
}

hrg_p_names <- c(
  "p_score_chisq_centered",
  "p_score_hotelling",
  "p_score_chisq_raw",
  "p_rls_chisq",
  "p_fmg_sb_rls",
  "p_fmg_mv_rls",
  "p_fmg_peba4_rls")

hrg_empty_rep <- function(cell, rep_id, seed) {
  base <- data.frame(
    cell_id = cell$cell_id, p = cell$p, q = cell$q, n = cell$n,
    n_over_q = cell$n_over_q, distribution = cell$distribution,
    rep = rep_id, seed = seed,
    fit_ok = FALSE, score_ok = FALSE, fmg_ok = FALSE,
    fit_error = "", score_error = "", fmg_error = "",
    q_centered = NA_real_, q_raw = NA_real_, rls_statistic = NA_real_,
    raw_opg_identity_error = NA_real_,
    centered_meat_min_eigenvalue = NA_real_,
    centered_meat_condition = NA_real_,
    fit_seconds = NA_real_, score_seconds = NA_real_,
    fmg_seconds = NA_real_,
    stringsAsFactors = FALSE)
  p_values <- stats::setNames(
    rep(NA_real_, length(hrg_p_names)), hrg_p_names)
  cbind(base, as.data.frame(as.list(p_values)))
}

hrg_one_rep <- function(cell, rep_id, spec, sampler, seed_base) {
  seed <- as.integer(seed_base + cell$cell_id * 100000L + rep_id)
  out <- hrg_empty_rep(cell, rep_id, seed)

  X <- tryCatch(hrg_draw(sampler, cell$n, seed), error = function(e) e)
  if (inherits(X, "error")) {
    out$fit_error <- paste0("draw: ", conditionMessage(X))
    return(out)
  }

  fit_begin <- proc.time()[["elapsed"]]
  fit <- tryCatch(hrg_fit(spec, X), error = function(e) e)
  out$fit_seconds <- proc.time()[["elapsed"]] - fit_begin
  if (inherits(fit, "error")) {
    out$fit_error <- conditionMessage(fit)
    return(out)
  }
  if (!isTRUE(fit$converged)) {
    out$fit_error <- "ML fit did not converge"
    return(out)
  }
  out$fit_ok <- TRUE

  score_begin <- proc.time()[["elapsed"]]
  score <- tryCatch({
    psi <- hrg_projected_scores(fit, X)
    n <- nrow(psi)
    q <- ncol(psi)
    if (q != cell$q) {
      stop("projected score rank does not match design", call. = FALSE)
    }
    u <- sqrt(n) * colMeans(psi)
    xc <- sweep(psi, 2L, colMeans(psi), "-")
    centered <- hrg_inverse_quadratic(
      u, crossprod(xc) / (n - 1), "centered meat")
    raw <- hrg_inverse_quadratic(
      u, crossprod(psi) / n, "raw meat")
    if (!centered$available) stop(centered$error, call. = FALSE)
    identity_error <- if (raw$available) {
      abs(
        raw$statistic -
          n * centered$statistic / (n - 1 + centered$statistic))
    } else {
      NA_real_
    }
    if (is.finite(identity_error) &&
        identity_error > 1e-8 * max(1, abs(raw$statistic))) {
      stop("raw-OPG rank-one identity failed", call. = FALSE)
    }
    list(
      q = q, n = n, centered = centered, raw = raw,
      rls = sum(colSums(psi)^2) / n,
      identity_error = identity_error)
  }, error = function(e) e)
  out$score_seconds <- proc.time()[["elapsed"]] - score_begin

  if (inherits(score, "error")) {
    out$score_error <- conditionMessage(score)
  } else {
    out$score_ok <- TRUE
    out$q_centered <- score$centered$statistic
    out$q_raw <- score$raw$statistic
    out$rls_statistic <- score$rls
    out$raw_opg_identity_error <- score$identity_error
    out$centered_meat_min_eigenvalue <-
      score$centered$min_eigenvalue
    out$centered_meat_condition <- score$centered$condition
    out$p_score_chisq_centered <- stats::pchisq(
      score$centered$statistic, score$q, lower.tail = FALSE)
    out$p_score_hotelling <- hrg_hotelling_p(
      score$centered$statistic, score$q, score$n)
    out$p_score_chisq_raw <- stats::pchisq(
      score$raw$statistic, score$q, lower.tail = FALSE)
    out$p_rls_chisq <- stats::pchisq(
      score$rls, score$q, lower.tail = FALSE)
  }

  fmg_begin <- proc.time()[["elapsed"]]
  fmg <- tryCatch(
    magmaan::fmg_tests(
      fit, tests = c("sb_rls", "mv_rls", "peba4_rls"), data = X),
    error = function(e) e)
  out$fmg_seconds <- proc.time()[["elapsed"]] - fmg_begin
  if (inherits(fmg, "error")) {
    out$fmg_error <- conditionMessage(fmg)
  } else {
    out$fmg_ok <- TRUE
    for (label in c("sb_rls", "mv_rls", "peba4_rls")) {
      row <- which(fmg$label == label)
      if (length(row) == 1L) {
        out[[paste0("p_fmg_", label)]] <- fmg$p_value[[row]]
      }
    }
  }
  out
}

hrg_wilson <- function(rejected, n, z = 1.95996398454005) {
  if (n <= 0L) return(c(lower = NA_real_, upper = NA_real_))
  phat <- rejected / n
  denominator <- 1 + z^2 / n
  centre <- (phat + z^2 / (2 * n)) / denominator
  half <- z * sqrt(
    phat * (1 - phat) / n + z^2 / (4 * n^2)) / denominator
  c(lower = max(0, centre - half), upper = min(1, centre + half))
}

hrg_summarize_methods <- function(raw) {
  long <- do.call(rbind, lapply(hrg_p_names, function(column) {
    data.frame(
      raw[c(
        "cell_id", "p", "q", "n", "n_over_q", "distribution", "rep")],
      method = sub("^p_", "", column),
      p_value = raw[[column]],
      stringsAsFactors = FALSE)
  }))
  pieces <- split(long, interaction(long$cell_id, long$method, drop = TRUE))
  out <- do.call(rbind, lapply(pieces, function(x) {
    valid <- is.finite(x$p_value)
    n_valid <- sum(valid)
    rejected <- sum(x$p_value[valid] <= .05)
    interval <- hrg_wilson(rejected, n_valid)
    data.frame(
      x[1L, c(
        "cell_id", "p", "q", "n", "n_over_q",
        "distribution", "method")],
      n_valid = n_valid,
      rejected = rejected,
      rejection_rate = if (n_valid) rejected / n_valid else NA_real_,
      ci_lower = interval[[1L]], ci_upper = interval[[2L]],
      mean_p = if (n_valid) mean(x$p_value[valid]) else NA_real_,
      stringsAsFactors = FALSE)
  }))
  row.names(out) <- NULL
  out[order(out$cell_id, out$method), , drop = FALSE]
}

hrg_summarize_diagnostics <- function(raw) {
  pieces <- split(raw, raw$cell_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    finite_median <- function(column) {
      values <- x[[column]]
      values <- values[is.finite(values)]
      if (length(values)) stats::median(values) else NA_real_
    }
    base <- x[1L, c(
      "cell_id", "p", "q", "n", "n_over_q", "distribution")]
    base$fit_success <- mean(x$fit_ok)
    base$score_success <- mean(x$score_ok)
    base$fmg_success <- mean(x$fmg_ok)
    base$median_centered_meat_condition <-
      finite_median("centered_meat_condition")
    base$median_centered_meat_min_eigenvalue <-
      finite_median("centered_meat_min_eigenvalue")
    base$max_raw_opg_identity_error <- {
      values <- x$raw_opg_identity_error
      values <- values[is.finite(values)]
      if (length(values)) max(values) else NA_real_
    }
    base$median_fit_seconds <- finite_median("fit_seconds")
    base$median_score_seconds <- finite_median("score_seconds")
    base$median_fmg_seconds <- finite_median("fmg_seconds")
    base
  }))
  row.names(out) <- NULL
  out
}
