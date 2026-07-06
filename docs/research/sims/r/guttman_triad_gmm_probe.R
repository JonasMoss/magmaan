# Informal probe for Guttman communality estimators.
#
# This is not a support claim or a benchmark. It compares closed-form
# correlation-scale communality rules in small continuous CFA designs:
# average-of-ratios Spearman, ratio-of-sums Spearman, identity linear moments,
# oracle-weight triad GMM, and plug-in one-step triad GMM.

parse_args <- function(args) {
  out <- list(reps = 300L, n = c(250L), seed = 20260706L, out = "")
  for (arg in args) {
    if (grepl("^--reps=", arg)) out$reps <- as.integer(sub("^--reps=", "", arg))
    if (grepl("^--n=", arg)) {
      out$n <- as.integer(strsplit(sub("^--n=", "", arg), ",", fixed = TRUE)[[1L]])
    }
    if (grepl("^--seed=", arg)) out$seed <- as.integer(sub("^--seed=", "", arg))
    if (grepl("^--out=", arg)) out$out <- sub("^--out=", "", arg)
  }
  out
}

pinv <- function(x, tol = sqrt(.Machine$double.eps)) {
  if (!all(is.finite(x))) return(matrix(NA_real_, ncol(x), nrow(x)))
  ee <- eigen((x + t(x)) / 2, symmetric = TRUE)
  keep <- ee$values > tol * max(1, max(abs(ee$values)))
  if (!any(keep)) return(matrix(0, ncol(x), nrow(x)))
  ee$vectors[, keep, drop = FALSE] %*%
    (diag(1 / ee$values[keep], nrow = sum(keep)) %*%
       t(ee$vectors[, keep, drop = FALSE]))
}

sim_mvn <- function(n, sigma) {
  matrix(stats::rnorm(n * ncol(sigma)), n, ncol(sigma)) %*% chol(sigma)
}

make_sigma <- function(loadings_by_factor, phi) {
  lambda <- unlist(loadings_by_factor, use.names = FALSE)
  blocks <- rep(seq_along(loadings_by_factor), lengths(loadings_by_factor))
  Lambda <- matrix(0, length(lambda), length(loadings_by_factor))
  for (i in seq_along(lambda)) Lambda[i, blocks[i]] <- lambda[i]
  common <- Lambda %*% phi %*% t(Lambda)
  theta <- diag(1 - diag(common))
  sigma <- common + theta
  list(sigma = sigma, common = common, lambda = Lambda, blocks = blocks)
}

designs <- function() {
  list(
    one_congeneric = make_sigma(
      list(f1 = c(.45, .55, .65, .75, .85, .70)),
      matrix(1, 1, 1)
    ),
    two_orthogonal = make_sigma(
      list(f1 = c(.45, .55, .70, .80, .65),
           f2 = c(.50, .60, .75, .85, .70)),
      diag(2)
    ),
    two_correlated = make_sigma(
      list(f1 = c(.45, .55, .70, .80, .65),
           f2 = c(.50, .60, .75, .85, .70)),
      matrix(c(1, .45, .45, 1), 2, 2)
    )
  )
}

offdiag_index <- function(p) {
  which(upper.tri(matrix(NA, p, p)), arr.ind = TRUE)
}

normal_cor_gamma <- function(r) {
  p <- nrow(r)
  pairs <- which(upper.tri(matrix(NA, p, p), diag = TRUE), arr.ind = TRUE)
  off <- offdiag_index(p)
  gamma_s <- matrix(0, nrow(pairs), nrow(pairs))
  for (a in seq_len(nrow(pairs))) {
    i <- pairs[a, 1L]
    j <- pairs[a, 2L]
    for (b in seq_len(nrow(pairs))) {
      k <- pairs[b, 1L]
      l <- pairs[b, 2L]
      gamma_s[a, b] <- r[i, k] * r[j, l] + r[i, l] * r[j, k]
    }
  }
  jac <- matrix(0, nrow(off), nrow(pairs))
  pair_key <- paste(pairs[, 1L], pairs[, 2L])
  for (a in seq_len(nrow(off))) {
    i <- off[a, 1L]
    j <- off[a, 2L]
    rho <- r[i, j]
    jac[a, match(paste(i, j), pair_key)] <- 1
    jac[a, match(paste(i, i), pair_key)] <- -.5 * rho
    jac[a, match(paste(j, j), pair_key)] <- -.5 * rho
  }
  jac %*% gamma_s %*% t(jac)
}

triad_system <- function(r, blocks, gamma_for_jac = NULL) {
  p <- nrow(r)
  off <- offdiag_index(p)
  off_key <- paste(pmin(off[, 1L], off[, 2L]), pmax(off[, 1L], off[, 2L]))
  row_list <- list()
  for (f in sort(unique(blocks))) {
    idx <- which(blocks == f)
    for (i in idx) {
      others <- setdiff(idx, i)
      if (length(others) < 2L) next
      cmb <- utils::combn(others, 2L)
      for (cc in seq_len(ncol(cmb))) {
        row_list[[length(row_list) + 1L]] <- c(i, cmb[, cc])
      }
    }
  }
  rows <- do.call(rbind, row_list)
  A <- matrix(0, nrow(rows), p)
  b <- numeric(nrow(rows))
  M <- matrix(0, nrow(rows), nrow(off))
  for (rr in seq_len(nrow(rows))) {
    i <- rows[rr, 1L]
    j <- rows[rr, 2L]
    k <- rows[rr, 3L]
    A[rr, i] <- r[j, k]
    b[rr] <- r[i, j] * r[i, k]
    if (!is.null(gamma_for_jac)) {
      M[rr, match(paste(min(j, k), max(j, k)), off_key)] <-
        M[rr, match(paste(min(j, k), max(j, k)), off_key)] + gamma_for_jac[i]
      M[rr, match(paste(min(i, j), max(i, j)), off_key)] <-
        M[rr, match(paste(min(i, j), max(i, j)), off_key)] - r[i, k]
      M[rr, match(paste(min(i, k), max(i, k)), off_key)] <-
        M[rr, match(paste(min(i, k), max(i, k)), off_key)] - r[i, j]
    }
  }
  list(rows = rows, A = A, b = b, M = M)
}

estimate_gamma <- function(s, blocks, method, gamma_true = NULL, r_true = NULL) {
  r <- stats::cov2cor(s)
  p <- nrow(r)
  gamma <- rep(NA_real_, p)
  if (method %in% c("ar", "rs", "ilm")) {
    for (i in seq_len(p)) {
      idx <- which(blocks == blocks[i] & seq_len(p) != i)
      cmb <- utils::combn(idx, 2L)
      rij <- r[i, cmb[1L, ]]
      rik <- r[i, cmb[2L, ]]
      rjk <- r[cbind(cmb[1L, ], cmb[2L, ])]
      gamma[i] <- switch(method,
        ar = mean(rij * rik / rjk),
        rs = sum(rij * rik) / sum(rjk),
        ilm = sum(rjk * rij * rik) / sum(rjk^2)
      )
    }
    return(gamma)
  }

  if (method == "gmm_oracle") {
    sys <- triad_system(r, blocks)
    sys_w <- triad_system(r_true, blocks, gamma_true)
    omega <- sys_w$M %*% normal_cor_gamma(r_true) %*% t(sys_w$M)
    W <- pinv(omega)
  } else if (method == "gmm_plugin") {
    gamma0 <- estimate_gamma(s, blocks, "ilm")
    sys <- triad_system(r, blocks, gamma0)
    omega <- sys$M %*% normal_cor_gamma(r) %*% t(sys$M)
    W <- pinv(omega)
  } else {
    stop("unknown method: ", method, call. = FALSE)
  }
  lhs <- t(sys$A) %*% W %*% sys$A
  rhs <- t(sys$A) %*% W %*% sys$b
  drop(pinv(lhs) %*% rhs)
}

fill_h <- function(s, gamma) {
  h <- diag(s) * gamma
  hmat <- s
  diag(hmat) <- h
  hmat
}

guttman_common <- function(hmat, blocks, score = c("incidence", "aligned")) {
  score <- match.arg(score)
  p <- nrow(hmat)
  q <- length(unique(blocks))
  A <- matrix(0, p, q)
  if (score == "incidence") {
    for (f in seq_len(q)) A[blocks == f, f] <- 1
  } else {
    hdiag <- pmax(diag(hmat), 0)
    for (f in seq_len(q)) {
      idx <- which(blocks == f)
      lam <- sqrt(hdiag[idx])
      denom <- sum(lam^2)
      if (!is.finite(denom) || denom <= 1e-12) return(NULL)
      A[idx, f] <- lam / denom
    }
  }
  d <- diag(t(A) %*% hmat %*% A)
  if (any(!is.finite(d)) || any(d <= 1e-10)) return(NULL)
  Dm <- diag(1 / sqrt(d), q)
  phi <- Dm %*% t(A) %*% hmat %*% A %*% Dm
  if (any(!is.finite(phi))) return(NULL)
  inv_phi <- tryCatch(solve(phi), error = function(e) NULL)
  if (is.null(inv_phi)) return(NULL)
  lambda <- hmat %*% A %*% Dm %*% inv_phi
  lambda %*% phi %*% t(lambda)
}

one_rep <- function(design, n, methods, scores) {
  y <- sim_mvn(n, design$sigma)
  s <- stats::cov(y)
  gamma_true <- diag(design$common) / diag(design$sigma)
  r_true <- stats::cov2cor(design$sigma)
  out <- list()
  for (method in methods) {
    gamma_hat <- tryCatch(
      estimate_gamma(s, design$blocks, method, gamma_true, r_true),
      error = function(e) rep(NA_real_, length(gamma_true))
    )
    diag_mise <- mean((diag(design$sigma) * gamma_hat - diag(design$common))^2,
                      na.rm = TRUE)
    hmat <- fill_h(s, gamma_hat)
    for (score in scores) {
      common_hat <- guttman_common(hmat, design$blocks, score)
      common_mise <- if (is.null(common_hat)) NA_real_ else {
        mean((common_hat - design$common)^2)
      }
      out[[length(out) + 1L]] <- data.frame(
        method = method,
        score = score,
        diag_mise = diag_mise,
        common_mise = common_mise,
        failed_common = is.null(common_hat)
      )
    }
  }
  do.call(rbind, out)
}

summarise_probe <- function(rows) {
  agg <- aggregate(
    cbind(diag_mise, common_mise, failed_common) ~ design + n + method + score,
    rows,
    function(x) c(mean = mean(x, na.rm = TRUE), se = stats::sd(x, na.rm = TRUE) / sqrt(length(x)))
  )
  data.frame(
    design = agg$design,
    n = agg$n,
    method = agg$method,
    score = agg$score,
    diag_mise = agg$diag_mise[, "mean"],
    diag_se = agg$diag_mise[, "se"],
    common_mise = agg$common_mise[, "mean"],
    common_se = agg$common_mise[, "se"],
    fail_rate = agg$failed_common[, "mean"],
    row.names = NULL
  )
}

run_probe <- function(args = parse_args(commandArgs(trailingOnly = TRUE))) {
  set.seed(args$seed)
  ds <- designs()
  methods <- c("ar", "rs", "ilm", "gmm_plugin", "gmm_oracle")
  scores <- c("incidence", "aligned")
  rows <- list()
  for (design_name in names(ds)) {
    for (n in args$n) {
      for (rep in seq_len(args$reps)) {
        rr <- one_rep(ds[[design_name]], n, methods, scores)
        rr$design <- design_name
        rr$n <- n
        rr$rep <- rep
        rows[[length(rows) + 1L]] <- rr
      }
    }
  }
  rows <- do.call(rbind, rows)
  summary <- summarise_probe(rows)
  summary <- summary[order(summary$design, summary$n, summary$score,
                           summary$common_mise, summary$diag_mise), ]
  print(summary, digits = 4, row.names = FALSE)
  if (nzchar(args$out)) {
    utils::write.csv(summary, args$out, row.names = FALSE)
    message("wrote ", args$out)
  }
  invisible(summary)
}

if (sys.nframe() == 0L) run_probe()
