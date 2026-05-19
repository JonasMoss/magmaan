# Shared helpers for paper-adjacent categorical SEM simulation scaffolds.

`%||%` <- function(x, y) if (is.null(x)) y else x

sim_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package `", pkg, "` is required for this operation", call. = FALSE)
  }
}

sim_mvn <- function(n, sigma, mean = rep(0, ncol(sigma))) {
  z <- matrix(stats::rnorm(n * ncol(sigma)), nrow = n)
  sweep(z %*% chol(sigma), 2L, mean, "+")
}

sim_thresholds_from_probs <- function(probs) {
  probs <- probs / sum(probs)
  stats::qnorm(cumsum(probs)[-length(probs)])
}

sim_ordinalize <- function(x, thresholds, ordered = TRUE) {
  y <- cut(x, breaks = c(-Inf, thresholds, Inf), labels = FALSE)
  if (ordered) ordered(y, levels = seq_len(length(thresholds) + 1L)) else y
}

sim_ordinalize_df <- function(z, thresholds_by_var) {
  out <- data.frame(row.names = seq_len(nrow(z)))
  for (j in seq_len(ncol(z))) {
    nm <- colnames(z)[j] %||% paste0("y", j)
    th <- thresholds_by_var[[nm]] %||% thresholds_by_var[[j]]
    out[[nm]] <- sim_ordinalize(z[, j], th)
  }
  out
}

sim_latent_cov_from_sem <- function(B, Gamma, Phi, Psi) {
  inv <- solve(diag(nrow(B)) - B)
  top_xi <- diag(nrow(Phi))
  top_zeta <- matrix(0, nrow(Phi), nrow(Psi))
  bot_xi <- inv %*% Gamma
  bot_zeta <- inv
  M_xi <- rbind(top_xi, bot_xi)
  M_zeta <- rbind(top_zeta, bot_zeta)
  M_xi %*% Phi %*% t(M_xi) + M_zeta %*% Psi %*% t(M_zeta)
}

sim_li_structural_latent_cov <- function() {
  B <- matrix(c(0, 0, 0,
                .3, 0, 0,
                .2, .5, 0), 3, 3, byrow = TRUE)
  Gamma <- matrix(c(.4, .6,
                    .4, .2,
                    .1, .1), 3, 2, byrow = TRUE)
  Phi <- matrix(c(1, .3, .3, 1), 2, 2)
  Psi <- diag(c(.336, .436, .379))
  out <- sim_latent_cov_from_sem(B, Gamma, Phi, Psi)
  dimnames(out) <- list(c("xi1", "xi2", "eta1", "eta2", "eta3"),
                        c("xi1", "xi2", "eta1", "eta2", "eta3"))
  out
}

sim_generate_indicators <- function(n, loadings_by_factor, factor_cov,
                                    variable_names = NULL) {
  factor_names <- names(loadings_by_factor)
  eta <- sim_mvn(n, factor_cov)
  colnames(eta) <- factor_names

  p <- sum(lengths(loadings_by_factor))
  y <- matrix(NA_real_, n, p)
  lambda <- numeric(p)
  factor_index <- integer(p)
  if (is.null(variable_names)) variable_names <- paste0("y", seq_len(p))

  col <- 0L
  for (f in seq_along(loadings_by_factor)) {
    for (loading in loadings_by_factor[[f]]) {
      col <- col + 1L
      lambda[col] <- loading
      factor_index[col] <- f
      y[, col] <- loading * eta[, f] +
        sqrt(max(0, 1 - loading^2)) * stats::rnorm(n)
    }
  }

  colnames(y) <- variable_names
  attr(y, "loadings") <- lambda
  attr(y, "factor_index") <- factor_index
  attr(y, "factor_scores") <- eta
  y
}

sim_fit_lavaan <- function(model, data, ordered = character(),
                           estimator = "DWLS", meanstructure = FALSE,
                           parameterization = "delta", ...) {
  sim_require("lavaan")
  lavaan::sem(model = model, data = data, ordered = ordered,
              estimator = estimator, meanstructure = meanstructure,
              parameterization = parameterization, ...)
}

sim_fit_magmaan <- function(model, data, ordered = character(),
                            estimator = "DWLS", ...) {
  sim_require("magmaan")
  magmaan::magmaan(model, data, ordered = ordered, estimator = estimator,
                   parameterization = "delta", ...)
}

sim_free_estimates <- function(fit) {
  if (inherits(fit, "lavaan")) {
    pt <- lavaan::parTable(fit)
  } else if (is.list(fit) && "partable" %in% names(fit)) {
    pt <- fit$partable
  } else {
    stop("Unsupported fit object", call. = FALSE)
  }
  pt <- pt[pt$free > 0L, , drop = FALSE]
  pt$key <- paste(pt$group %||% 1L, pt$lhs, pt$op, pt$rhs, sep = "\r")
  pt[, c("key", "lhs", "op", "rhs", "est"), drop = FALSE]
}

sim_estimate_mse <- function(estimates, truth) {
  keys <- intersect(names(truth), names(estimates))
  if (!length(keys)) stop("No common parameter keys", call. = FALSE)
  err <- estimates[keys] - truth[keys]
  data.frame(
    key = keys,
    truth = unname(truth[keys]),
    estimate = unname(estimates[keys]),
    error = unname(err),
    squared_error = unname(err^2),
    row.names = NULL
  )
}

sim_replication_summary <- function(rows) {
  x <- do.call(rbind, rows)
  stats::aggregate(
    x[, c("error", "squared_error")],
    by = list(key = x$key),
    FUN = function(v) c(mean = mean(v), sd = stats::sd(v))
  )
}
