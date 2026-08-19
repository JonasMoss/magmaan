logdet_pd <- function(x) {
  ch <- chol(x)
  2 * sum(log(diag(ch)))
}

vech_lower <- function(x) {
  x[lower.tri(x, diag = TRUE)]
}

strict_lower <- function(x) {
  x[lower.tri(x, diag = FALSE)]
}

continuous_delta <- function(sample_stats, implied) {
  pieces <- vector("list", length(sample_stats$S))
  for (b in seq_along(sample_stats$S)) {
    d <- numeric()
    if (length(sample_stats$mean) && length(sample_stats$mean[[b]]) &&
        length(implied$mu) && length(implied$mu[[b]])) {
      d <- c(d, implied$mu[[b]] - sample_stats$mean[[b]])
    }
    pieces[[b]] <- c(d, vech_lower(implied$sigma[[b]] - sample_stats$S[[b]]))
  }
  pieces
}

symmetric_vech_gls_weight <- function(sigma_inverse) {
  p <- nrow(sigma_inverse)
  pairs <- which(lower.tri(matrix(0, p, p), diag = TRUE), arr.ind = TRUE)
  pairs <- pairs[order(pairs[, "col"], pairs[, "row"]), , drop = FALSE]
  basis <- lapply(seq_len(nrow(pairs)), function(k) {
    e <- matrix(0, p, p)
    i <- pairs[k, "row"]
    j <- pairs[k, "col"]
    e[i, j] <- 1
    e[j, i] <- 1
    e
  })
  out <- matrix(0, length(basis), length(basis))
  for (i in seq_along(basis)) {
    a <- sigma_inverse %*% basis[[i]] %*% sigma_inverse
    for (j in seq_along(basis)) out[i, j] <- sum(diag(a %*% basis[[j]]))
  }
  out
}

normal_theory_weight <- function(sigma, include_mean) {
  inverse <- solve(sigma)
  wcov <- 0.5 * symmetric_vech_gls_weight(inverse)
  if (!include_mean) return(wcov)
  out <- matrix(0, nrow(sigma) + nrow(wcov), nrow(sigma) + nrow(wcov))
  out[seq_len(nrow(sigma)), seq_len(nrow(sigma))] <- inverse
  idx <- nrow(sigma) + seq_len(nrow(wcov))
  out[idx, idx] <- wcov
  out
}

quadratic_objective <- function(sample_stats, implied, weights = NULL) {
  deltas <- continuous_delta(sample_stats, implied)
  total_n <- sum(sample_stats$nobs)
  value <- 0
  for (b in seq_along(deltas)) {
    w <- if (is.null(weights)) diag(length(deltas[[b]])) else weights[[b]]
    share <- sample_stats$nobs[[b]] / total_n
    value <- value + 0.5 * share * drop(
      crossprod(deltas[[b]], w %*% deltas[[b]])
    )
  }
  value
}

ml_objective <- function(sample_stats, implied) {
  total_n <- sum(sample_stats$nobs)
  value <- 0
  for (b in seq_along(sample_stats$S)) {
    s <- sample_stats$S[[b]]
    sigma <- implied$sigma[[b]]
    p <- nrow(s)
    delta_mu <- numeric()
    mean_term <- 0
    if (length(sample_stats$mean) && length(sample_stats$mean[[b]]) &&
        length(implied$mu) && length(implied$mu[[b]])) {
      delta_mu <- implied$mu[[b]] - sample_stats$mean[[b]]
      mean_term <- drop(crossprod(delta_mu, solve(sigma, delta_mu)))
    }
    block <- logdet_pd(sigma) + sum(diag(solve(sigma, s))) + mean_term -
      logdet_pd(s) - p
    value <- value + 0.5 * (sample_stats$nobs[[b]] / total_n) * block
  }
  value
}

fiml_objective <- function(raw, implied) {
  x <- as.matrix(raw)
  sigma <- implied$sigma[[1L]]
  mu <- implied$mu[[1L]]
  kernels <- vapply(seq_len(nrow(x)), function(i) {
    observed <- which(is.finite(x[i, ]))
    block <- sigma[observed, observed, drop = FALSE]
    delta <- x[i, observed] - mu[observed]
    logdet_pd(block) + drop(crossprod(delta, solve(block, delta)))
  }, numeric(1))
  0.5 * mean(kernels)
}

threshold_estimates <- function(fit) {
  fit$partable$est[fit$partable$op == "|"]
}

ordinal_model_moments <- function(fit, stats, parameterization) {
  implied <- magmaan_core$model_implied(fit)
  sigma <- implied$sigma[[1L]]
  thresholds <- threshold_estimates(fit)
  if (identical(parameterization, "theta")) {
    thresholds <- thresholds /
      sqrt(diag(sigma)[stats$threshold_ov[[1L]]])
    associations <- strict_lower(stats::cov2cor(sigma))
  } else {
    associations <- strict_lower(sigma)
  }
  c(thresholds, associations)
}

ordinal_objective <- function(fit, stats, kind, parameterization) {
  delta <- ordinal_model_moments(fit, stats, parameterization) -
    stats$moments[[1L]]
  weight <- switch(
    kind,
    uls = diag(length(delta)),
    dwls = stats$W_dwls[[1L]],
    wls = stats$W_wls[[1L]],
    stop("unknown ordinal weight: ", kind, call. = FALSE)
  )
  0.5 * drop(crossprod(delta, weight %*% delta))
}

mixed_model_moments <- function(fit, stats, parameterization) {
  implied <- magmaan_core$model_implied(fit)
  sigma <- implied$sigma[[1L]]
  mu <- implied$mu[[1L]]
  ordered <- as.logical(stats$ordered_mask[[1L]])
  thresholds <- threshold_estimates(fit)
  if (identical(parameterization, "theta")) {
    thresholds <- thresholds /
      sqrt(diag(sigma)[stats$threshold_ov[[1L]]])
  }
  out <- thresholds
  out <- c(out, -mu[!ordered])
  out <- c(out, diag(sigma)[!ordered])
  for (j in seq_len(nrow(sigma))) {
    if (j == nrow(sigma)) next
    for (i in seq.int(j + 1L, nrow(sigma))) {
      value <- sigma[i, j]
      if (identical(parameterization, "theta")) {
        if (ordered[[i]] && ordered[[j]]) {
          value <- value / sqrt(sigma[i, i] * sigma[j, j])
        } else if (xor(ordered[[i]], ordered[[j]])) {
          o <- if (ordered[[i]]) i else j
          value <- value / sqrt(sigma[o, o])
        }
      }
      out <- c(out, value)
    }
  }
  out
}

mixed_objective <- function(fit, stats, kind, parameterization) {
  delta <- mixed_model_moments(fit, stats, parameterization) -
    stats$moments[[1L]]
  weight <- switch(
    kind,
    uls = diag(length(delta)),
    dwls = stats$W_dwls[[1L]],
    wls = stats$W_wls[[1L]],
    stop("unknown mixed-ordinal weight: ", kind, call. = FALSE)
  )
  0.5 * drop(crossprod(delta, weight %*% delta))
}

catml_objective <- function(fit, stats) {
  sigma <- magmaan_core$model_implied(fit)$sigma[[1L]]
  implied_r <- stats::cov2cor(sigma)
  sample_r <- stats$R[[1L]]
  p <- nrow(sample_r)
  0.5 * (
    logdet_pd(implied_r) + sum(diag(solve(implied_r, sample_r))) -
      logdet_pd(sample_r) - p
  )
}
