# Informal Guttman communality experiment.
#
# This expands the quick probe into a small factorial simulation over sample
# size, indicators per factor, number of factors, loading spread, and latent
# correlations. It stays in continuous normal CFA so the plug-in normal
# correlation covariance is the relevant ADF reference.

parse_csv_int <- function(x) as.integer(strsplit(x, ",", fixed = TRUE)[[1L]])
parse_csv_num <- function(x) as.numeric(strsplit(x, ",", fixed = TRUE)[[1L]])
parse_csv_chr <- function(x) strsplit(x, ",", fixed = TRUE)[[1L]]

parse_args <- function(args) {
  out <- list(
    reps = 100L,
    n = c(100L, 250L),
    factors = c(1L, 2L, 4L),
    indicators = c(3L, 5L),
    rho = c(0, .4),
    loading = c("mild", "wide"),
    seed = 20260706L,
    out = ""
  )
  for (arg in args) {
    if (grepl("^--reps=", arg)) out$reps <- as.integer(sub("^--reps=", "", arg))
    if (grepl("^--n=", arg)) out$n <- parse_csv_int(sub("^--n=", "", arg))
    if (grepl("^--factors=", arg)) out$factors <- parse_csv_int(sub("^--factors=", "", arg))
    if (grepl("^--indicators=", arg)) out$indicators <- parse_csv_int(sub("^--indicators=", "", arg))
    if (grepl("^--rho=", arg)) out$rho <- parse_csv_num(sub("^--rho=", "", arg))
    if (grepl("^--loading=", arg)) out$loading <- parse_csv_chr(sub("^--loading=", "", arg))
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

loading_vector <- function(m, kind) {
  if (kind == "mild") return(seq(.55, .80, length.out = m))
  if (kind == "wide") return(seq(.35, .85, length.out = m))
  stop("unknown loading pattern: ", kind, call. = FALSE)
}

make_design <- function(q, m, rho, loading) {
  if (q == 1L) rho <- 0
  phi <- matrix(rho, q, q)
  diag(phi) <- 1
  base <- loading_vector(m, loading)
  lambda <- rep(base, q)
  blocks <- rep(seq_len(q), each = m)
  Lambda <- matrix(0, q * m, q)
  for (i in seq_along(lambda)) Lambda[i, blocks[i]] <- lambda[i]
  common <- Lambda %*% phi %*% t(Lambda)
  theta <- diag(1 - diag(common))
  sigma <- common + theta
  list(sigma = sigma, common = common, lambda = Lambda, blocks = blocks,
       q = q, m = m, rho = rho, loading = loading)
}

normal_cor_gamma_pairs <- function(r, off) {
  n_pairs <- nrow(off)
  out <- matrix(0, n_pairs, n_pairs)
  deriv_terms <- function(i, j) {
    rho <- r[i, j]
    data.frame(a = c(i, i, j), b = c(j, i, j), d = c(1, -.5 * rho, -.5 * rho))
  }
  terms <- vector("list", n_pairs)
  for (a in seq_len(n_pairs)) terms[[a]] <- deriv_terms(off[a, 1L], off[a, 2L])
  for (a in seq_len(n_pairs)) {
    ta <- terms[[a]]
    for (b in seq_len(a)) {
      tb <- terms[[b]]
      val <- 0
      for (u in seq_len(nrow(ta))) {
        for (v in seq_len(nrow(tb))) {
          i <- ta$a[u]; j <- ta$b[u]
          k <- tb$a[v]; l <- tb$b[v]
          cov_s <- r[i, k] * r[j, l] + r[i, l] * r[j, k]
          val <- val + ta$d[u] * tb$d[v] * cov_s
        }
      }
      out[a, b] <- val
      out[b, a] <- val
    }
  }
  out
}

within_off_pairs <- function(blocks) {
  rows <- list()
  for (f in sort(unique(blocks))) {
    idx <- which(blocks == f)
    cmb <- utils::combn(idx, 2L)
    rows[[length(rows) + 1L]] <- t(cmb)
  }
  do.call(rbind, rows)
}

triad_system <- function(r, blocks, gamma_for_jac = NULL, selected_off = NULL) {
  if (is.null(selected_off)) selected_off <- within_off_pairs(blocks)
  off_key <- paste(pmin(selected_off[, 1L], selected_off[, 2L]),
                   pmax(selected_off[, 1L], selected_off[, 2L]))
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
  A <- matrix(0, nrow(rows), nrow(r))
  b <- numeric(nrow(rows))
  M <- matrix(0, nrow(rows), nrow(selected_off))
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
  list(rows = rows, A = A, b = b, M = M, off = selected_off)
}

solve_gmm <- function(A, b, W) {
  if (nrow(A) == ncol(A)) {
    direct <- tryCatch(solve(A, b), error = function(e) NULL)
    if (!is.null(direct) && all(is.finite(direct))) return(drop(direct))
  }
  lhs <- t(A) %*% W %*% A
  rhs <- t(A) %*% W %*% b
  drop(pinv(lhs) %*% rhs)
}

estimate_itemwise <- function(r, blocks, method) {
  p <- nrow(r)
  gamma <- rep(NA_real_, p)
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
  gamma
}

estimate_gmm_block <- function(r, blocks, oracle = NULL) {
  gamma <- rep(NA_real_, nrow(r))
  for (f in sort(unique(blocks))) {
    idx <- which(blocks == f)
    rb <- r[idx, idx, drop = FALSE]
    local_blocks <- rep(1L, length(idx))
    gamma0 <- if (is.null(oracle)) estimate_itemwise(rb, local_blocks, "ilm") else oracle$gamma[idx]
    rw <- if (is.null(oracle)) rb else oracle$r[idx, idx, drop = FALSE]
    sys <- triad_system(rb, local_blocks)
    sys_w <- triad_system(rw, local_blocks, gamma0)
    omega <- sys_w$M %*% normal_cor_gamma_pairs(rw, sys_w$off) %*% t(sys_w$M)
    gamma[idx] <- solve_gmm(sys$A, sys$b, pinv(omega))
  }
  gamma
}

estimate_gmm_full <- function(r, blocks, oracle = NULL) {
  gamma0 <- if (is.null(oracle)) estimate_itemwise(r, blocks, "ilm") else oracle$gamma
  rw <- if (is.null(oracle)) r else oracle$r
  off <- within_off_pairs(blocks)
  sys <- triad_system(r, blocks, selected_off = off)
  sys_w <- triad_system(rw, blocks, gamma0, selected_off = off)
  omega <- sys_w$M %*% normal_cor_gamma_pairs(rw, off) %*% t(sys_w$M)
  solve_gmm(sys$A, sys$b, pinv(omega))
}

estimate_gamma <- function(s, blocks, method, gamma_true, r_true) {
  r <- stats::cov2cor(s)
  if (method %in% c("ar", "rs", "ilm")) return(estimate_itemwise(r, blocks, method))
  if (method == "gmm_block") return(estimate_gmm_block(r, blocks))
  if (method == "gmm_full") return(estimate_gmm_full(r, blocks))
  if (method == "oracle_block") {
    return(estimate_gmm_block(r, blocks, list(gamma = gamma_true, r = r_true)))
  }
  if (method == "oracle_full") {
    return(estimate_gmm_full(r, blocks, list(gamma = gamma_true, r = r_true)))
  }
  stop("unknown method: ", method, call. = FALSE)
}

fill_h <- function(s, gamma) {
  hmat <- s
  diag(hmat) <- diag(s) * gamma
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
  inv_phi <- tryCatch(solve(phi), error = function(e) NULL)
  if (is.null(inv_phi)) return(NULL)
  lambda <- hmat %*% A %*% Dm %*% inv_phi
  lambda %*% phi %*% t(lambda)
}

complexity_row <- function(q, m) {
  p <- q * m
  d_block <- m * (m - 1) / 2
  g_block <- m * (m - 1) * (m - 2) / 2
  data.frame(
    factors = q,
    indicators = m,
    p = p,
    corr_pairs_per_block = d_block,
    triad_rows_per_block = g_block,
    within_corr_pairs_total = q * d_block,
    triad_rows_total = q * g_block,
    full_sem_corr_pairs = p * (p - 1) / 2
  )
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

summarise_rows <- function(rows) {
  agg <- aggregate(
    cbind(diag_mise, common_mise, failed_common) ~
      factors + indicators + loading + rho + n + method + score,
    rows,
    function(x) c(mean = mean(x, na.rm = TRUE),
                  se = stats::sd(x, na.rm = TRUE) / sqrt(length(x)))
  )
  out <- data.frame(
    factors = agg$factors,
    indicators = agg$indicators,
    loading = agg$loading,
    rho = agg$rho,
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
  key <- c("factors", "indicators", "loading", "rho", "n", "score")
  ar <- out[out$method == "ar", c(key, "diag_mise", "common_mise")]
  names(ar)[(length(key) + 1L):(length(key) + 2L)] <- c("diag_ar", "common_ar")
  out <- merge(out, ar, by = key, all.x = TRUE)
  out$diag_rel_ar <- out$diag_mise / out$diag_ar
  out$common_rel_ar <- out$common_mise / out$common_ar
  out[order(out$factors, out$indicators, out$loading, out$rho, out$n,
            out$score, out$diag_rel_ar), ]
}

condition_grid <- function(args) {
  rows <- expand.grid(
    factors = args$factors,
    indicators = args$indicators,
    rho = args$rho,
    loading = args$loading,
    n = args$n,
    stringsAsFactors = FALSE
  )
  rows <- rows[!(rows$factors == 1L & rows$rho != 0), , drop = FALSE]
  rows[order(rows$factors, rows$indicators, rows$loading, rows$rho, rows$n), ]
}

run_experiment <- function(args = parse_args(commandArgs(trailingOnly = TRUE))) {
  set.seed(args$seed)
  methods <- c("ar", "rs", "ilm", "gmm_block", "gmm_full",
               "oracle_block", "oracle_full")
  scores <- c("incidence", "aligned")
  grid <- condition_grid(args)
  message("conditions: ", nrow(grid), "; reps per condition: ", args$reps)
  complexity <- do.call(rbind, Map(complexity_row, grid$factors, grid$indicators))
  complexity <- unique(complexity)
  message("complexity rows:")
  print(complexity[order(complexity$factors, complexity$indicators), ],
        row.names = FALSE)

  rows <- list()
  tick <- 0L
  for (cc in seq_len(nrow(grid))) {
    g <- grid[cc, ]
    design <- make_design(g$factors, g$indicators, g$rho, g$loading)
    for (rep in seq_len(args$reps)) {
      rr <- one_rep(design, g$n, methods, scores)
      rr$factors <- g$factors
      rr$indicators <- g$indicators
      rr$loading <- g$loading
      rr$rho <- g$rho
      rr$n <- g$n
      rr$rep <- rep
      rows[[length(rows) + 1L]] <- rr
    }
    tick <- tick + 1L
    if (tick %% 10L == 0L || tick == nrow(grid)) {
      message("finished condition ", tick, "/", nrow(grid))
    }
  }
  rows <- do.call(rbind, rows)
  summary <- summarise_rows(rows)
  print(summary, digits = 4, row.names = FALSE)
  if (nzchar(args$out)) {
    utils::write.csv(summary, args$out, row.names = FALSE)
    message("wrote ", args$out)
  }
  invisible(summary)
}

if (sys.nframe() == 0L) run_experiment()
