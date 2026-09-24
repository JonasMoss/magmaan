# Experiment-local covariance maps. No public estimator implementation.
# All charts use the same objective, gradient, optimizer and positive-definite
# component domain; only independent coordinates differ.
implied <- function(m) {
  A <- solve(diag(nrow(m$psi)) - m$beta)
  C <- A %*% m$psi %*% t(A)
  list(A = A, C = C, Sigma = m$lambda %*% C %*% t(m$lambda) + m$theta)
}
rescale <- function(m, d) {
  m$lambda <- sweep(m$lambda, 2, d, `*`)
  m$beta <- sweep(sweep(m$beta, 1, d, `/`), 2, d, `*`)
  m$psi <- m$psi / outer(d, d)
  m
}
chart_map <- function(base, mask, chart) {
  q <- nrow(base$psi)
  anchors <- lapply(seq_len(q), function(j) {
    ii <- which(base$lambda[, j] != 0)
    if (length(ii)) list(matrix = "lambda", i = ii[1], j = j)
    else {
      ii <- which(base$beta[, j] != 0)
      if (!length(ii)) stop("Latent has no measurement anchor")
      list(matrix = "beta", i = ii[1], j = j)
    }
  })
  marker_scale <- function(m) {
    K <- diag(q); rhs <- numeric(q)
    for (j in seq_len(q)) {
      a <- anchors[[j]]; value <- m[[a$matrix]][a$i, a$j]
      if (value <= 0) stop("Pilot anchor orientation requires positive loadings")
      rhs[j] <- -log(value)
      if (a$matrix == "beta") K[j, a$i] <- K[j, a$i] - 1
    }
    exp(solve(K, rhs))
  }
  to_chart <- function(m) {
    d <- switch(chart,
      marker = marker_scale(m),
      disturbance = sqrt(diag(m$psi)),
      total = sqrt(diag(implied(m)$C)))
    rescale(m, d)
  }
  base <- to_chart(base)
  for (a in anchors) mask[[a$matrix]][a$i, a$j] <- chart != "marker"
  diag(mask$psi) <- chart == "marker"
  slots <- do.call(rbind, lapply(names(mask), function(nm) {
    z <- mask[[nm]]
    if (nm %in% c("psi", "theta")) z[upper.tri(z)] <- FALSE
    ij <- which(z, arr.ind = TRUE)
    data.frame(matrix = rep(nm, nrow(ij)), i = ij[, 1], j = ij[, 2])
  }))
  pack <- function(m) {
    m <- to_chart(m)
    vapply(seq_len(nrow(slots)), function(k) {
      s <- slots[k, ]; m[[s$matrix]][s$i, s$j]
    }, 0.0)
  }
  unpack <- function(x) {
    m <- base
    for (k in seq_along(x)) {
      s <- slots[k, ]; m[[s$matrix]][s$i, s$j] <- x[k]
      if (s$matrix %in% c("psi", "theta")) m[[s$matrix]][s$j, s$i] <- x[k]
    }
    A <- solve(diag(q) - m$beta)
    H <- A * A
    if (chart == "total") {
      diag(m$psi) <- 0
      diag(m$psi) <- solve(H, 1 - diag(A %*% m$psi %*% t(A)))
    }
    c(list(m = m, H = H), implied(m))
  }
  # Exact directional derivatives, including the eliminated disturbance diagonal.
  derivatives <- function(u) {
    lapply(seq_len(nrow(slots)), function(k) {
      s <- slots[k, ]; dm <- lapply(u$m, function(z) z * 0)
      dm[[s$matrix]][s$i, s$j] <- 1
      if (s$matrix %in% c("psi", "theta")) dm[[s$matrix]][s$j, s$i] <- 1
      dA <- u$A %*% dm$beta %*% u$A
      dC <- dA %*% u$m$psi %*% t(u$A) + u$A %*% u$m$psi %*% t(dA) +
        u$A %*% dm$psi %*% t(u$A)
      if (chart == "total") {
        dd <- solve(u$H, -diag(dC))
        dC <- dC + u$A %*% diag(dd, q) %*% t(u$A)
      }
      dm$lambda %*% u$C %*% t(u$m$lambda) +
        u$m$lambda %*% dC %*% t(u$m$lambda) +
        u$m$lambda %*% u$C %*% t(dm$lambda) + dm$theta
    })
  }
  list(pack = pack, unpack = unpack, derivatives = derivatives)
}
min_eigen <- function(x) min(eigen(x, symmetric = TRUE, only.values = TRUE)$values)
fit_chart <- function(map, start, S, maxit) {
  constant <- as.numeric(determinant(S, logarithm = TRUE)$modulus) + nrow(S)
  calls <- invalid <- 0L
  last_x <- last <- NULL
  evaluate <- function(x) {
    if (identical(x, last_x)) return(last)
    calls <<- calls + 1L
    value <- tryCatch({
      u <- map$unpack(x)
      # This arm compares the common interior, never unconstrained Heywood fits.
      chol(u$m$psi); chol(u$m$theta)
      R <- chol(u$Sigma); W <- chol2inv(R)
      G <- W - W %*% S %*% W
      list(value = 2 * sum(log(diag(R))) + sum(W * S) - constant,
           gradient = vapply(map$derivatives(u), function(d) sum(G * d), 0.0))
    }, error = function(e) NULL)
    if (is.null(value)) {
      invalid <<- invalid + 1L
      value <- list(value = 1e30, gradient = numeric(length(x)))
    }
    last_x <<- x; last <<- value
    value
  }
  t0 <- proc.time()[[3]]
  fit <- optim(start, function(x) evaluate(x)$value,
               function(x) evaluate(x)$gradient, method = "BFGS",
               control = list(maxit = maxit, reltol = 1e-10))
  elapsed <- proc.time()[[3]] - t0
  terminal <- map$unpack(fit$par)
  list(fit = fit, terminal = terminal, seconds = elapsed,
       evaluations = calls, invalid = invalid,
       gradient = evaluate(fit$par)$gradient)
}
geometry <- function(map, x) {
  u <- map$unpack(x); W <- chol2inv(chol(u$Sigma))
  ds <- map$derivatives(u)
  Z <- vapply(ds, function(d) as.vector(W %*% d), numeric(length(W)))
  # trace(W dSigma_i W dSigma_j), retaining both symmetric matrix halves.
  Zt <- vapply(ds, function(d) as.vector(t(W %*% d)), numeric(length(W)))
  info <- 0.5 * crossprod(Zt, Z)
  ee <- eigen(info, symmetric = TRUE, only.values = TRUE)$values
  rank <- sum(ee > max(ee) * 1e-10)
  list(information = info, rank = rank, npar = length(x),
       condition = if (rank == length(x)) max(ee) / min(ee) else Inf)
}
validate_chart <- function(map, m) {
  x <- map$pack(m); u <- map$unpack(x)
  covariance_error <- max(abs(u$Sigma - implied(m)$Sigma))
  ds <- map$derivatives(u)
  derivative_error <- max(vapply(seq_along(x), function(k) {
    h <- 1e-5 * max(1, abs(x[k])); e <- numeric(length(x)); e[k] <- h
    fd <- (map$unpack(x + e)$Sigma - map$unpack(x - e)$Sigma) / (2 * h)
    max(abs(fd - ds[[k]])) / max(1, max(abs(fd)))
  }, 0.0))
  if (covariance_error > 1e-8 || derivative_error > 1e-6)
    stop("Chart covariance/derivative validation failed")
  c(covariance_error = covariance_error, derivative_error = derivative_error)
}
