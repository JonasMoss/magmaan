#!/usr/bin/env Rscript

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x)) y else x

script_path <- {
  cmd <- commandArgs(FALSE)
  hit <- grep("^--file=", cmd, value = TRUE)
  if (length(hit)) sub("^--file=", "", hit[[1]]) else NA_character_
}
script_dir <- if (!is.na(script_path)) dirname(normalizePath(script_path)) else getwd()

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n",
    "\n",
    "Population-level asymptotic comparison for Guttman's multiple-group CFA\n",
    "estimator against NT, ULS, GLS, and oracle WLS in a three-factor CFA.\n",
    "\n",
    "Options:\n",
    "  --smoke              Run one low-reliability / low-correlation slice.\n",
    "                       This is the default.\n",
    "  --full               Run the full 2 x 2 x 3 grid.\n",
    "  --fd-step X          Relative finite-difference step. Default: 1e-5.\n",
    "  --n-risk LIST        Comma-separated sample sizes for risk summaries.\n",
    "                       Default: 50,100,500,1000.\n",
    "  --results-dir PATH   Output directory. Default: results.\n",
    "  --help               Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  opts <- list(
    smoke = TRUE,
    full = FALSE,
    fd_step = 1e-5,
    n_risk = c(50L, 100L, 500L, 1000L),
    results_dir = file.path(script_dir, "results")
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg == "--help") {
      usage()
      quit(status = 0)
    } else if (arg == "--smoke") {
      opts$smoke <- TRUE
      opts$full <- FALSE
    } else if (arg == "--full") {
      opts$full <- TRUE
      opts$smoke <- FALSE
    } else if (arg == "--fd-step") {
      i <- i + 1L
      if (i > length(args)) stop("--fd-step needs a value", call. = FALSE)
      opts$fd_step <- as.numeric(args[[i]])
    } else if (arg == "--n-risk") {
      i <- i + 1L
      if (i > length(args)) stop("--n-risk needs a value", call. = FALSE)
      opts$n_risk <- as.integer(strsplit(args[[i]], ",", fixed = TRUE)[[1]])
    } else if (arg == "--results-dir") {
      i <- i + 1L
      if (i > length(args)) stop("--results-dir needs a value", call. = FALSE)
      opts$results_dir <- args[[i]]
    } else {
      stop("Unknown option: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(opts$fd_step) || opts$fd_step <= 0) {
    stop("--fd-step must be positive", call. = FALSE)
  }
  if (!length(opts$n_risk) || any(!is.finite(opts$n_risk)) || any(opts$n_risk <= 0)) {
    stop("--n-risk must contain positive integers", call. = FALSE)
  }
  opts
}

opts <- parse_args(commandArgs(TRUE))
dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)

vech_pairs <- function(p) which(lower.tri(matrix(0, p, p), diag = TRUE), arr.ind = TRUE)

vech <- function(M) {
  M[lower.tri(M, diag = TRUE)]
}

unvech <- function(v, p) {
  M <- matrix(0, p, p)
  M[lower.tri(M, diag = TRUE)] <- v
  M + t(M) - diag(diag(M), nrow = p)
}

trace_prod <- function(A, B) sum(A * t(B))

solve_sym <- function(A) {
  A <- 0.5 * (A + t(A))
  R <- try(chol(A), silent = TRUE)
  if (!inherits(R, "try-error")) return(chol2inv(R))
  qr.solve(A)
}

safe_logdet_inv <- function(A) {
  R <- try(chol(A), silent = TRUE)
  if (inherits(R, "try-error")) return(NULL)
  list(logdet = 2 * sum(log(diag(R))), inv = chol2inv(R))
}

gamma_normal <- function(Sigma) {
  p <- nrow(Sigma)
  pairs <- vech_pairs(p)
  m <- nrow(pairs)
  G <- matrix(0, m, m)
  for (u in seq_len(m)) {
    a <- pairs[u, 1]
    b <- pairs[u, 2]
    for (v in seq_len(m)) {
      c <- pairs[v, 1]
      d <- pairs[v, 2]
      G[u, v] <- Sigma[a, c] * Sigma[b, d] + Sigma[a, d] * Sigma[b, c]
    }
  }
  0.5 * (G + t(G))
}

gls_weight <- function(S) {
  si <- safe_logdet_inv(S)
  if (is.null(si)) stop("S is not positive definite", call. = FALSE)
  Sinv <- si$inv
  p <- nrow(S)
  pairs <- vech_pairs(p)
  m <- nrow(pairs)
  W <- matrix(0, m, m)
  for (u in seq_len(m)) {
    E1 <- matrix(0, p, p)
    E1[pairs[u, 1], pairs[u, 2]] <- 1
    E1[pairs[u, 2], pairs[u, 1]] <- 1
    A1 <- Sinv %*% E1 %*% Sinv
    for (v in seq_len(m)) {
      E2 <- matrix(0, p, p)
      E2[pairs[v, 1], pairs[v, 2]] <- 1
      E2[pairs[v, 2], pairs[v, 1]] <- 1
      W[u, v] <- trace_prod(A1, E2)
    }
  }
  0.5 * (W + t(W))
}

factor_of_row <- rep(seq_len(3), each = 3)
marker_rows <- c(1L, 4L, 7L)
free_rows <- c(2L, 3L, 5L, 6L, 8L, 9L)
free_factors <- factor_of_row[free_rows]
lambda_names <- paste0("lambda_y", free_rows, "_f", free_factors)
phi_pairs <- vech_pairs(3)
phi_names <- paste0("phi_f", phi_pairs[, 1], "_f", phi_pairs[, 2])
theta_names <- paste0("theta_y", seq_len(9))
tau_names <- c(lambda_names, phi_names, theta_names)

lambda_from_free <- function(loadings) {
  Lambda <- matrix(0, 9, 3)
  Lambda[cbind(marker_rows, seq_len(3))] <- 1
  Lambda[cbind(free_rows, free_factors)] <- loadings
  Lambda
}

tau_from_mats <- function(Lambda, Phi, theta) {
  out <- c(Lambda[cbind(free_rows, free_factors)], vech(Phi), theta)
  names(out) <- tau_names
  out
}

mats_from_alpha <- function(alpha) {
  Lambda <- lambda_from_free(alpha[1:6])
  L <- matrix(0, 3, 3)
  L[1, 1] <- exp(alpha[7])
  L[2, 1] <- alpha[8]
  L[2, 2] <- exp(alpha[9])
  L[3, 1] <- alpha[10]
  L[3, 2] <- alpha[11]
  L[3, 3] <- exp(alpha[12])
  Phi <- L %*% t(L)
  theta <- exp(alpha[13:21])
  list(Lambda = Lambda, Phi = Phi, theta = theta)
}

sigma_from_alpha <- function(alpha) {
  m <- mats_from_alpha(alpha)
  m$Lambda %*% m$Phi %*% t(m$Lambda) + diag(m$theta, 9)
}

tau_from_alpha <- function(alpha) {
  m <- mats_from_alpha(alpha)
  tau_from_mats(m$Lambda, m$Phi, m$theta)
}

alpha_from_tau <- function(tau) {
  load <- unname(tau[1:6])
  Phi <- unvech(unname(tau[7:12]), 3)
  R <- try(chol(Phi), silent = TRUE)
  if (inherits(R, "try-error")) stop("Phi is not positive definite", call. = FALSE)
  L <- t(R)
  theta <- pmax(unname(tau[13:21]), .Machine$double.eps)
  c(
    load,
    log(L[1, 1]),
    L[2, 1],
    log(L[2, 2]),
    L[3, 1],
    L[3, 2],
    log(L[3, 3]),
    log(theta)
  )
}

make_population <- function(reliability, factor_cor, misspec) {
  main <- rep(c(1.0, 0.9, 0.8), 3)
  Lambda_analysis <- lambda_from_free(main[free_rows])
  Phi <- matrix(factor_cor, 3, 3)
  diag(Phi) <- 1
  theta <- main^2 * (1 - reliability) / reliability

  Lambda_dgp <- Lambda_analysis
  Theta_dgp <- diag(theta, 9)
  if (misspec == "cross_loading") {
    Lambda_dgp[4, 1] <- 0.20
  } else if (misspec == "residual_covariance") {
    Theta_dgp[2, 5] <- Theta_dgp[5, 2] <- 0.20 * sqrt(theta[2] * theta[5])
  } else if (misspec != "none") {
    stop("Unknown misspecification: ", misspec, call. = FALSE)
  }

  Sigma <- Lambda_dgp %*% Phi %*% t(Lambda_dgp) + Theta_dgp
  eig <- eigen(Sigma, symmetric = TRUE, only.values = TRUE)$values
  if (min(eig) <= 1e-8) stop("Population covariance is not positive definite", call. = FALSE)

  list(
    Sigma = Sigma,
    sigma = vech(Sigma),
    target_tau = tau_from_mats(Lambda_analysis, Phi, theta),
    analysis_Lambda = Lambda_analysis,
    analysis_Phi = Phi,
    analysis_theta = theta
  )
}

spearman_communalities <- function(S) {
  p <- nrow(S)
  if (p < 3) return(0.5 * diag(S))
  sd <- sqrt(pmax(diag(S), .Machine$double.eps))
  R <- S / tcrossprod(sd)
  h <- numeric(p)
  for (i in seq_len(p)) {
    others <- setdiff(seq_len(p), i)
    pairs <- combn(others, 2)
    vals <- apply(pairs, 2, function(idx) {
      den <- R[idx[1], idx[2]]
      if (abs(den) < 1e-12) return(NA_real_)
      R[i, idx[1]] * R[i, idx[2]] / den
    })
    h[i] <- mean(vals, na.rm = TRUE) * S[i, i]
  }
  h
}

guttman_tau <- function(sigma) {
  S <- unvech(sigma, 9)
  X <- matrix(0, 9, 3)
  X[cbind(seq_len(9), factor_of_row)] <- 1
  h <- numeric(9)
  for (f in seq_len(3)) {
    rows <- which(factor_of_row == f)
    h[rows] <- spearman_communalities(S[rows, rows, drop = FALSE])
  }
  C <- S
  diag(C) <- h
  A <- C %*% X
  B <- t(X) %*% C %*% X
  d <- diag(B)
  if (any(!is.finite(d)) || any(d <= 0)) stop("Guttman diagonal scaling failed", call. = FALSE)
  Q <- diag(1 / sqrt(d), 3)
  L <- A %*% Q
  P <- Q %*% B %*% Q
  Kstar <- L %*% solve(P)
  marker <- Kstar[cbind(marker_rows, seq_len(3))]
  if (any(abs(marker) < 1e-12)) stop("Guttman marker scaling failed", call. = FALSE)
  M <- diag(marker, 3)
  Lambda <- Kstar %*% diag(1 / marker, 3)
  Phi <- M %*% P %*% M
  theta <- diag(S - Lambda %*% Phi %*% t(Lambda))
  tau_from_mats(Lambda, Phi, theta)
}

objective <- function(alpha, sigma, estimator, W_oracle = NULL) {
  S <- unvech(sigma, 9)
  Sigma <- sigma_from_alpha(alpha)
  if (any(!is.finite(Sigma))) return(1e100)

  if (estimator == "NT") {
    si <- safe_logdet_inv(Sigma)
    if (is.null(si)) return(1e100)
    return(si$logdet + trace_prod(S, si$inv))
  }

  d <- vech(Sigma - S)
  W <- switch(
    estimator,
    ULS = diag(length(d)),
    GLS = tryCatch(gls_weight(S), error = function(e) NULL),
    WLS = W_oracle,
    stop("Unknown estimator: ", estimator, call. = FALSE)
  )
  if (is.null(W) || any(!is.finite(W))) return(1e100)
  0.5 * drop(crossprod(d, W %*% d))
}

fit_estimator <- function(sigma, estimator, alpha_start, W_oracle) {
  if (estimator == "Guttman") {
    return(list(tau = guttman_tau(sigma), alpha = NA, value = NA_real_, convergence = 0L))
  }

  fn <- function(a) objective(a, sigma, estimator, W_oracle)
  opt <- optim(
    alpha_start, fn, method = "BFGS",
    control = list(maxit = 2000, reltol = 1e-12)
  )
  if (opt$convergence != 0) {
    opt2 <- optim(
      opt$par, fn, method = "BFGS",
      control = list(maxit = 2000, reltol = 1e-12)
    )
    if (is.finite(opt2$value) && opt2$value <= opt$value) opt <- opt2
  }
  list(tau = tau_from_alpha(opt$par), alpha = opt$par, value = opt$value, convergence = opt$convergence)
}

num_grad <- function(fn, x, rel_step) {
  g <- numeric(length(x))
  for (j in seq_along(x)) {
    h <- rel_step * max(1, abs(x[j]))
    xp <- xm <- x
    xp[j] <- xp[j] + h
    xm[j] <- xm[j] - h
    g[j] <- (fn(xp) - fn(xm)) / (2 * h)
  }
  g
}

jacobian <- function(fn, x, rel_step) {
  y0 <- fn(x)
  J <- matrix(NA_real_, length(y0), length(x))
  for (j in seq_along(x)) {
    h <- rel_step * max(1, abs(x[j]))
    xp <- xm <- x
    xp[j] <- xp[j] + h
    xm[j] <- xm[j] - h
    J[, j] <- (fn(xp) - fn(xm)) / (2 * h)
  }
  J
}

estimator_jacobian <- function(estimator, fit, sigma0, W_oracle, rel_step) {
  if (estimator == "Guttman") {
    return(jacobian(guttman_tau, sigma0, rel_step))
  }

  obj <- function(a, s) objective(a, s, estimator, W_oracle)
  grad_alpha <- function(a, s) num_grad(function(aa) obj(aa, s), a, rel_step)
  H <- jacobian(function(a) grad_alpha(a, sigma0), fit$alpha, rel_step)
  H <- 0.5 * (H + t(H))
  G <- jacobian(function(s) grad_alpha(fit$alpha, s), sigma0, rel_step)
  Dalpha <- -qr.solve(H, G)
  Jtau <- jacobian(tau_from_alpha, fit$alpha, rel_step)
  Jtau %*% Dalpha
}

design_grid <- if (isTRUE(opts$smoke)) {
  expand.grid(
    reliability = 0.5,
    factor_cor = 0.1,
    misspec = c("none", "cross_loading", "residual_covariance"),
    stringsAsFactors = FALSE
  )
} else {
  expand.grid(
    reliability = c(0.5, 0.8),
    factor_cor = c(0.1, 0.4),
    misspec = c("none", "cross_loading", "residual_covariance"),
    stringsAsFactors = FALSE
  )
}

estimators <- c("Guttman", "NT", "ULS", "GLS", "WLS")
groups <- list(
  loadings = seq_len(6),
  factor_cov = 7:12,
  residual_var = 13:21,
  all = seq_len(21)
)

summary_rows <- list()
point_rows <- list()
diag_rows <- list()
row_i <- 0L
point_i <- 0L
diag_i <- 0L

for (cell in seq_len(nrow(design_grid))) {
  rel <- design_grid$reliability[cell]
  cor <- design_grid$factor_cor[cell]
  misspec <- design_grid$misspec[cell]
  message(sprintf("Cell %d/%d: reliability=%.1f factor_cor=%.1f misspec=%s",
                  cell, nrow(design_grid), rel, cor, misspec))

  pop <- make_population(rel, cor, misspec)
  Gamma <- gamma_normal(pop$Sigma)
  W_oracle <- solve_sym(Gamma)
  alpha_start <- alpha_from_tau(pop$target_tau)

  fits <- list()
  for (est in estimators) {
    message("  fitting ", est)
    fits[[est]] <- fit_estimator(pop$sigma, est, alpha_start, W_oracle)
    if (est != "Guttman" && is.finite(fits[[est]]$value)) {
      alpha_start <- fits[[est]]$alpha
    }
  }

  for (est in estimators) {
    message("  differentiating ", est)
    J <- estimator_jacobian(est, fits[[est]], pop$sigma, W_oracle, opts$fd_step)
    Omega <- J %*% Gamma %*% t(J)
    Omega <- 0.5 * (Omega + t(Omega))
    bias <- fits[[est]]$tau - pop$target_tau

    diag_i <- diag_i + 1L
    diag_rows[[diag_i]] <- data.frame(
      reliability = rel,
      factor_cor = cor,
      misspec = misspec,
      estimator = est,
      convergence = fits[[est]]$convergence %||% NA_integer_,
      objective = fits[[est]]$value %||% NA_real_,
      min_omega_eigen = min(eigen(Omega, symmetric = TRUE, only.values = TRUE)$values),
      max_abs_bias = max(abs(bias)),
      stringsAsFactors = FALSE
    )

    for (j in seq_along(fits[[est]]$tau)) {
      point_i <- point_i + 1L
      point_rows[[point_i]] <- data.frame(
        reliability = rel,
        factor_cor = cor,
        misspec = misspec,
        estimator = est,
        parameter = names(fits[[est]]$tau)[j],
        estimate = unname(fits[[est]]$tau[j]),
        target = unname(pop$target_tau[j]),
        bias = unname(bias[j]),
        asymp_var = Omega[j, j],
        stringsAsFactors = FALSE
      )
    }

    for (grp in names(groups)) {
      idx <- groups[[grp]]
      bias2 <- sum(bias[idx]^2)
      var_trace <- sum(diag(Omega)[idx])
      for (n_ref in opts$n_risk) {
        row_i <- row_i + 1L
        summary_rows[[row_i]] <- data.frame(
          reliability = rel,
          factor_cor = cor,
          misspec = misspec,
          estimator = est,
          group = grp,
          n_ref = n_ref,
          bias_norm = sqrt(bias2),
          asymp_sd_norm = sqrt(max(var_trace, 0)),
          var_trace = var_trace,
          risk = bias2 + var_trace / n_ref,
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

summary <- do.call(rbind, summary_rows)
key_cols <- c("reliability", "factor_cor", "misspec", "group", "n_ref")
nt <- summary[summary$estimator == "NT", c(key_cols, "risk", "var_trace")]
names(nt)[names(nt) == "risk"] <- "nt_risk"
names(nt)[names(nt) == "var_trace"] <- "nt_var_trace"
gut <- summary[summary$estimator == "Guttman", c(key_cols, "risk", "var_trace")]
names(gut)[names(gut) == "risk"] <- "guttman_risk"
names(gut)[names(gut) == "var_trace"] <- "guttman_var_trace"
summary <- merge(summary, nt, by = key_cols, all.x = TRUE)
summary <- merge(summary, gut, by = key_cols, all.x = TRUE)
summary$risk_ratio_to_nt <- summary$risk / summary$nt_risk
summary$var_ratio_to_nt <- summary$var_trace / summary$nt_var_trace
summary$risk_ratio_to_guttman <- summary$risk / summary$guttman_risk
summary$var_ratio_to_guttman <- summary$var_trace / summary$guttman_var_trace
summary <- summary[order(summary$reliability, summary$factor_cor, summary$misspec,
                         summary$group, summary$n_ref, summary$estimator), ]

points <- do.call(rbind, point_rows)
diagnostics <- do.call(rbind, diag_rows)
metadata <- data.frame(
  key = c("timestamp", "command", "smoke", "full", "fd_step", "n_risk", "R"),
  value = c(
    format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    paste(commandArgs(FALSE), collapse = " "),
    as.character(opts$smoke),
    as.character(opts$full),
    format(opts$fd_step, scientific = TRUE),
    paste(opts$n_risk, collapse = ","),
    R.version.string
  )
)

summary_path <- file.path(opts$results_dir, "asymptotic_summary.csv")
points_path <- file.path(opts$results_dir, "point_estimates.csv")
diag_path <- file.path(opts$results_dir, "diagnostics.csv")
meta_path <- file.path(opts$results_dir, "metadata.csv")
write.csv(summary, summary_path, row.names = FALSE)
write.csv(points, points_path, row.names = FALSE)
write.csv(diagnostics, diag_path, row.names = FALSE)
write.csv(metadata, meta_path, row.names = FALSE)

cat("Wrote:\n")
cat("  ", summary_path, "\n", sep = "")
cat("  ", points_path, "\n", sep = "")
cat("  ", diag_path, "\n", sep = "")
cat("  ", meta_path, "\n", sep = "")
