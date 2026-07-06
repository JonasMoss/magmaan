#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

source(file.path(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]])),
  "..", "_support", "R", "helpers.R"
))

core <- magmaan::magmaan_core

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Finite-sample comparison of Guttman communality estimators, with magmaan\n",
    "normal-theory ML as the iterative benchmark and a categorical Pearson-code\n",
    "simulation arm from magmaan's simulation scaffold.\n\n",
    "Options:\n",
    "  --smoke              Quick normal + ordinal slice. Default.\n",
    "  --full               Larger default grid.\n",
    "  --reps N             Replications per cell. Full default: 200.\n",
    "  --n LIST             Sample sizes. Full default: 100,250,500.\n",
    "  --factors LIST       Factor counts. Full default: 1,2,4.\n",
    "  --indicators LIST    Indicators per factor. Full default: 3,5.\n",
    "  --rho LIST           Exchangeable latent correlations. Full default: 0,.4.\n",
    "  --loading LIST       Loading patterns: mild,wide. Full default: mild,wide.\n",
    "  --generators LIST    Data generators: normal,ordinal. Default: both.\n",
    "  --methods LIST       Methods: ar,rs,ilm,gmm_block,gmm_full,nt_ml.\n",
    "  --seed-base N        Base RNG seed. Default: 20260706.\n",
    "  --ml-max-iter N      magmaan ML max iterations. Default: 2000.\n",
    "  --results-dir PATH   Output directory. Default: results.\n",
    "  --help               Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  opts <- list(
    smoke = TRUE,
    full = FALSE,
    reps = 200L,
    n = c(100L, 250L, 500L),
    factors = c(1L, 2L, 4L),
    indicators = c(3L, 5L),
    rho = c(0, .4),
    loading = c("mild", "wide"),
    generators = c("normal", "ordinal"),
    methods = c("ar", "rs", "ilm", "gmm_block", "gmm_full", "nt_ml"),
    seed_base = 20260706L,
    ml_max_iter = 2000L,
    results_dir = experiment_path("results")
  )
  explicit <- character()
  i <- 1L
  take <- function(name) {
    i <<- i + 1L
    if (i > length(args)) stop(name, " needs a value", call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--help") {
      usage()
      quit(status = 0)
    } else if (a == "--smoke") {
      opts$smoke <- TRUE
      opts$full <- FALSE
    } else if (a == "--full") {
      opts$full <- TRUE
      opts$smoke <- FALSE
    } else if (a == "--reps") {
      opts$reps <- as.integer(take(a)); explicit <- c(explicit, "reps")
    } else if (grepl("^--reps=", a)) {
      opts$reps <- as.integer(sub("^--reps=", "", a)); explicit <- c(explicit, "reps")
    } else if (a == "--n") {
      opts$n <- as.integer(parse_csv_arg(take(a))); explicit <- c(explicit, "n")
    } else if (grepl("^--n=", a)) {
      opts$n <- as.integer(parse_csv_arg(sub("^--n=", "", a))); explicit <- c(explicit, "n")
    } else if (a == "--factors") {
      opts$factors <- as.integer(parse_csv_arg(take(a))); explicit <- c(explicit, "factors")
    } else if (grepl("^--factors=", a)) {
      opts$factors <- as.integer(parse_csv_arg(sub("^--factors=", "", a))); explicit <- c(explicit, "factors")
    } else if (a == "--indicators") {
      opts$indicators <- as.integer(parse_csv_arg(take(a))); explicit <- c(explicit, "indicators")
    } else if (grepl("^--indicators=", a)) {
      opts$indicators <- as.integer(parse_csv_arg(sub("^--indicators=", "", a))); explicit <- c(explicit, "indicators")
    } else if (a == "--rho") {
      opts$rho <- as.numeric(parse_csv_arg(take(a))); explicit <- c(explicit, "rho")
    } else if (grepl("^--rho=", a)) {
      opts$rho <- as.numeric(parse_csv_arg(sub("^--rho=", "", a))); explicit <- c(explicit, "rho")
    } else if (a == "--loading") {
      opts$loading <- parse_csv_arg(take(a)); explicit <- c(explicit, "loading")
    } else if (grepl("^--loading=", a)) {
      opts$loading <- parse_csv_arg(sub("^--loading=", "", a)); explicit <- c(explicit, "loading")
    } else if (a == "--generators") {
      opts$generators <- parse_csv_arg(take(a)); explicit <- c(explicit, "generators")
    } else if (grepl("^--generators=", a)) {
      opts$generators <- parse_csv_arg(sub("^--generators=", "", a)); explicit <- c(explicit, "generators")
    } else if (a == "--methods") {
      opts$methods <- parse_csv_arg(take(a)); explicit <- c(explicit, "methods")
    } else if (grepl("^--methods=", a)) {
      opts$methods <- parse_csv_arg(sub("^--methods=", "", a)); explicit <- c(explicit, "methods")
    } else if (a == "--seed-base") {
      opts$seed_base <- as.integer(take(a))
    } else if (grepl("^--seed-base=", a)) {
      opts$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--ml-max-iter") {
      opts$ml_max_iter <- as.integer(take(a))
    } else if (grepl("^--ml-max-iter=", a)) {
      opts$ml_max_iter <- as.integer(sub("^--ml-max-iter=", "", a))
    } else if (a == "--results-dir") {
      opts$results_dir <- take(a)
    } else if (grepl("^--results-dir=", a)) {
      opts$results_dir <- sub("^--results-dir=", "", a)
    } else {
      stop("unknown option: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (isTRUE(opts$smoke)) {
    if (!"reps" %in% explicit) opts$reps <- 8L
    if (!"n" %in% explicit) opts$n <- c(100L, 250L)
    if (!"factors" %in% explicit) opts$factors <- 2L
    if (!"indicators" %in% explicit) opts$indicators <- 5L
    if (!"rho" %in% explicit) opts$rho <- .4
    if (!"loading" %in% explicit) opts$loading <- c("mild", "wide")
    if (!"generators" %in% explicit) opts$generators <- c("normal", "ordinal")
  }
  bad <- setdiff(opts$loading, c("mild", "wide"))
  if (length(bad)) stop("unknown loading patterns: ", paste(bad, collapse = ","), call. = FALSE)
  bad <- setdiff(opts$generators, c("normal", "ordinal"))
  if (length(bad)) stop("unknown generators: ", paste(bad, collapse = ","), call. = FALSE)
  bad <- setdiff(opts$methods, c("ar", "rs", "ilm", "gmm_block", "gmm_full", "nt_ml"))
  if (length(bad)) stop("unknown methods: ", paste(bad, collapse = ","), call. = FALSE)
  opts
}

opts <- parse_args(commandArgs(TRUE))
set_single_threaded_math()
dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)

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
  Lambda[cbind(seq_along(lambda), blocks)] <- lambda
  common <- Lambda %*% phi %*% t(Lambda)
  theta <- diag(1 - diag(common))
  sigma <- common + theta
  vars <- paste0("x", seq_len(q * m))
  dimnames(sigma) <- dimnames(common) <- list(vars, vars)
  list(sigma = sigma, common = common, lambda = Lambda, blocks = blocks,
       q = q, m = m, rho = rho, loading = loading, vars = vars)
}

model_syntax <- function(q, m) {
  lines <- character(q)
  for (f in seq_len(q)) {
    idx <- ((f - 1L) * m + 1L):(f * m)
    lines[[f]] <- paste0("f", f, " =~ ", paste0("x", idx, collapse = " + "))
  }
  paste(lines, collapse = "\n")
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
    rows[[length(rows) + 1L]] <- t(utils::combn(idx, 2L))
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

estimate_gmm_block <- function(r, blocks) {
  gamma <- rep(NA_real_, nrow(r))
  for (f in sort(unique(blocks))) {
    idx <- which(blocks == f)
    rb <- r[idx, idx, drop = FALSE]
    local_blocks <- rep(1L, length(idx))
    gamma0 <- estimate_itemwise(rb, local_blocks, "ilm")
    sys <- triad_system(rb, local_blocks)
    sys_w <- triad_system(rb, local_blocks, gamma0)
    omega <- sys_w$M %*% normal_cor_gamma_pairs(rb, sys_w$off) %*% t(sys_w$M)
    gamma[idx] <- solve_gmm(sys$A, sys$b, pinv(omega))
  }
  gamma
}

estimate_gmm_full <- function(r, blocks) {
  gamma0 <- estimate_itemwise(r, blocks, "ilm")
  off <- within_off_pairs(blocks)
  sys <- triad_system(r, blocks, selected_off = off)
  sys_w <- triad_system(r, blocks, gamma0, selected_off = off)
  omega <- sys_w$M %*% normal_cor_gamma_pairs(r, off) %*% t(sys_w$M)
  solve_gmm(sys$A, sys$b, pinv(omega))
}

estimate_gamma <- function(s, blocks, method) {
  r <- stats::cov2cor(s)
  if (method %in% c("ar", "rs", "ilm")) return(estimate_itemwise(r, blocks, method))
  if (method == "gmm_block") return(estimate_gmm_block(r, blocks))
  if (method == "gmm_full") return(estimate_gmm_full(r, blocks))
  stop("unknown communality method: ", method, call. = FALSE)
}

fill_h <- function(s, gamma) {
  hmat <- s
  diag(hmat) <- diag(s) * gamma
  hmat
}

guttman_common <- function(hmat, blocks) {
  p <- nrow(hmat)
  q <- length(unique(blocks))
  A <- matrix(0, p, q)
  hdiag <- pmax(diag(hmat), 0)
  for (f in seq_len(q)) {
    idx <- which(blocks == f)
    lam <- sqrt(hdiag[idx])
    denom <- sum(lam^2)
    if (!is.finite(denom) || denom <= 1e-12) return(NULL)
    A[idx, f] <- lam / denom
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

pt_value <- function(pt, row) {
  est <- suppressWarnings(as.numeric(pt$est[[row]]))
  if (is.finite(est)) return(est)
  start <- suppressWarnings(as.numeric(pt$ustart[[row]]))
  if (is.finite(start)) return(start)
  0
}

ml_common_from_fit <- function(fit, vars, q) {
  pt <- fit$partable
  p <- length(vars)
  lvs <- paste0("f", seq_len(q))
  Lambda <- matrix(0, p, q, dimnames = list(vars, lvs))
  Phi <- matrix(0, q, q, dimnames = list(lvs, lvs))
  resid <- rep(NA_real_, p)
  for (r in seq_len(nrow(pt))) {
    op <- as.character(pt$op[[r]])
    lhs <- as.character(pt$lhs[[r]])
    rhs <- as.character(pt$rhs[[r]])
    val <- pt_value(pt, r)
    if (op == "=~" && lhs %in% lvs && rhs %in% vars) {
      Lambda[rhs, lhs] <- val
    } else if (op == "~~" && lhs %in% lvs && rhs %in% lvs) {
      Phi[lhs, rhs] <- Phi[rhs, lhs] <- val
    } else if (op == "~~" && lhs == rhs && lhs %in% vars) {
      resid[match(lhs, vars)] <- val
    }
  }
  list(common = Lambda %*% Phi %*% t(Lambda),
       improper = any(is.finite(resid) & resid < 0))
}

fit_nt_ml <- function(x, design, control) {
  df <- as.data.frame(x)
  names(df) <- design$vars
  syntax <- model_syntax(design$q, design$m)
  fit <- magmaan::magmaan(syntax, df, estimator = "ML", se = "none", test = "none",
                          control = control)
  if (!isTRUE(fit$converged)) stop("magmaan ML did not converge", call. = FALSE)
  ml_common_from_fit(fit, design$vars, design$q)
}

time_call <- function(expr) {
  start <- Sys.time()
  value <- tryCatch(
    list(ok = TRUE, value = force(expr), message = ""),
    error = function(e) list(ok = FALSE, value = NULL, message = conditionMessage(e))
  )
  value$elapsed_ms <- 1000 * as.numeric(difftime(Sys.time(), start, units = "secs"))
  value
}

one_method <- function(x, design, target_common, method, ml_control) {
  if (method == "nt_ml") {
    timed <- time_call(fit_nt_ml(x, design, ml_control))
    failed <- !timed$ok || is.null(timed$value)
    common <- if (failed) NULL else timed$value$common
    improper <- if (failed) NA else isTRUE(timed$value$improper)
  } else {
    timed <- time_call({
      s <- stats::cov(x)
      gamma <- estimate_gamma(s, design$blocks, method)
      if (any(!is.finite(gamma))) stop("non-finite communality", call. = FALSE)
      common <- guttman_common(fill_h(s, gamma), design$blocks)
      if (is.null(common)) stop("Guttman reconstruction failed", call. = FALSE)
      common
    })
    failed <- !timed$ok || is.null(timed$value)
    common <- if (failed) NULL else timed$value
    improper <- if (failed) NA else any(diag(stats::cov(x) - common) < 0)
  }
  common_mise <- if (failed) NA_real_ else mean((common - target_common)^2)
  diag_mise <- if (failed) NA_real_ else {
    mean((diag(common) - diag(target_common))^2)
  }
  data.frame(
    method = method,
    diag_mise = diag_mise,
    common_mise = common_mise,
    elapsed_ms = timed$elapsed_ms,
    failed = failed,
    improper = improper,
    message = if (failed) timed$message else "",
    stringsAsFactors = FALSE
  )
}

code_variance <- function(prob) {
  x <- seq_along(prob)
  mu <- sum(prob * x)
  sum(prob * x^2) - mu^2
}

ordinal_prob <- c(.08, .17, .35, .25, .15)

target_for_generator <- function(design, generator) {
  if (generator == "normal") {
    return(list(sigma = design$sigma, common = design$common))
  }
  sd_code <- sqrt(code_variance(ordinal_prob))
  D <- diag(rep(sd_code, nrow(design$sigma)))
  list(sigma = D %*% design$sigma %*% D,
       common = D %*% design$common %*% D)
}

draws_for_condition <- function(design, generator, n, reps, seed_base) {
  if (generator == "normal") {
    set.seed(seed_base)
    draws <- vector("list", reps)
    for (r in seq_len(reps)) {
      draws[[r]] <- sim_mvn(n, design$sigma)
      colnames(draws[[r]]) <- design$vars
    }
    return(draws)
  }
  marginals <- stats::setNames(rep(list(ordinal_prob), nrow(design$sigma)), design$vars)
  cal <- magmaan::sim_ordcorr_calibrate(
    design$sigma, marginals, metric = "pearson_codes",
    matrix_repair = "none")
  sim <- magmaan::sim_ordcorr_draw(cal, n = n, reps = reps, seed_base = seed_base)
  lapply(sim$draws, function(d) {
    x <- d$X
    colnames(x) <- design$vars
    x
  })
}

condition_grid <- function(opts) {
  out <- expand.grid(
    generator = opts$generators,
    factors = opts$factors,
    indicators = opts$indicators,
    rho = opts$rho,
    loading = opts$loading,
    n = opts$n,
    stringsAsFactors = FALSE
  )
  out <- out[!(out$factors == 1L & out$rho != 0), , drop = FALSE]
  out[order(out$generator, out$factors, out$indicators, out$loading, out$rho, out$n), ]
}

complexity_row <- function(q, m) {
  p <- q * m
  corr_block <- m * (m - 1) / 2
  triad_block <- m * (m - 1) * (m - 2) / 2
  data.frame(
    factors = q,
    indicators = m,
    p = p,
    corr_pairs_per_block = corr_block,
    triad_rows_per_block = triad_block,
    within_corr_pairs_total = q * corr_block,
    triad_rows_total = q * triad_block,
    full_sem_corr_pairs = p * (p - 1) / 2,
    block_weight_dim = triad_block,
    full_weight_dim = q * triad_block,
    stringsAsFactors = FALSE
  )
}

summarise_results <- function(rows) {
  agg <- aggregate(
    cbind(diag_mise, common_mise, elapsed_ms, failed, improper) ~
      generator + factors + indicators + loading + rho + n + method,
    rows,
    function(x) c(mean = mean(x, na.rm = TRUE),
                  se = stats::sd(x, na.rm = TRUE) / sqrt(length(x)),
                  median = stats::median(x, na.rm = TRUE),
                  p90 = stats::quantile(x, .9, na.rm = TRUE, names = FALSE))
  )
  out <- data.frame(
    generator = agg$generator,
    factors = agg$factors,
    indicators = agg$indicators,
    loading = agg$loading,
    rho = agg$rho,
    n = agg$n,
    method = agg$method,
    diag_mise = agg$diag_mise[, "mean"],
    diag_se = agg$diag_mise[, "se"],
    common_mise = agg$common_mise[, "mean"],
    common_se = agg$common_mise[, "se"],
    elapsed_mean_ms = agg$elapsed_ms[, "mean"],
    elapsed_median_ms = agg$elapsed_ms[, "median"],
    elapsed_p90_ms = agg$elapsed_ms[, "p90"],
    fail_rate = agg$failed[, "mean"],
    improper_rate = agg$improper[, "mean"],
    stringsAsFactors = FALSE
  )
  key <- c("generator", "factors", "indicators", "loading", "rho", "n")
  rs <- out[out$method == "rs", c(key, "common_mise", "diag_mise")]
  names(rs)[(length(key) + 1L):(length(key) + 2L)] <- c("common_rs", "diag_rs")
  nt <- out[out$method == "nt_ml", c(key, "common_mise", "diag_mise")]
  names(nt)[(length(key) + 1L):(length(key) + 2L)] <- c("common_nt", "diag_nt")
  out <- merge(out, rs, by = key, all.x = TRUE)
  out <- merge(out, nt, by = key, all.x = TRUE)
  out$common_rel_rs <- out$common_mise / out$common_rs
  out$diag_rel_rs <- out$diag_mise / out$diag_rs
  out$common_rel_nt <- out$common_mise / out$common_nt
  out$diag_rel_nt <- out$diag_mise / out$diag_nt
  out[order(out$generator, out$factors, out$indicators, out$loading, out$rho,
            out$n, out$method), ]
}

write_all <- function(reps, summary, complexity, metadata) {
  write_csv(reps, file.path(opts$results_dir, "replicates.csv"))
  write_csv(summary, file.path(opts$results_dir, "summary.csv"))
  write_csv(complexity, file.path(opts$results_dir, "complexity.csv"))
  write_csv(metadata, file.path(opts$results_dir, "metadata.csv"))
}

grid <- condition_grid(opts)
complexity <- unique(do.call(rbind, Map(complexity_row, grid$factors, grid$indicators)))
complexity <- complexity[order(complexity$factors, complexity$indicators), ]

message("conditions: ", nrow(grid), "; reps per condition: ", opts$reps)
message("methods: ", paste(opts$methods, collapse = ","))
print(complexity, row.names = FALSE)

ml_control <- list(max_iter = opts$ml_max_iter, ftol = 1e-10, gtol = 1e-8)
rows <- list()
row_i <- 0L
for (cc in seq_len(nrow(grid))) {
  g <- grid[cc, ]
  design <- make_design(g$factors, g$indicators, g$rho, g$loading)
  target <- target_for_generator(design, g$generator)
  seed <- opts$seed_base + cc * 100000L
  draws <- draws_for_condition(design, g$generator, g$n, opts$reps, seed)
  for (rr in seq_along(draws)) {
    x <- draws[[rr]]
    for (method in opts$methods) {
      row_i <- row_i + 1L
      out <- one_method(x, design, target$common, method, ml_control)
      out$generator <- g$generator
      out$factors <- g$factors
      out$indicators <- g$indicators
      out$loading <- g$loading
      out$rho <- g$rho
      out$n <- g$n
      out$rep <- rr
      rows[[row_i]] <- out
    }
  }
  message("finished condition ", cc, "/", nrow(grid))
}

replicates <- do.call(rbind, rows)
summary <- summarise_results(replicates)
metadata <- metadata_frame(
  list(
    reps = opts$reps,
    n = opts$n,
    factors = opts$factors,
    indicators = opts$indicators,
    rho = opts$rho,
    loading = opts$loading,
    generators = opts$generators,
    methods = opts$methods,
    seed_base = opts$seed_base,
    ml_max_iter = opts$ml_max_iter,
    ordinal_prob = ordinal_prob,
    smoke = opts$smoke
  ),
  packages = c("magmaan")
)

write_all(replicates, summary, complexity, metadata)
message("wrote ", normalizePath(opts$results_dir, mustWork = FALSE))
