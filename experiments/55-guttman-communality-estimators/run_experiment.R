#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

source(file.path(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]])),
  "..", "_support", "R", "helpers.R"
))

core <- magmaan::magmaan_core

canonical_method <- function(method) {
  out <- method
  out[out == "gmm_block"] <- "nt_gls_block"
  out[out == "gmm_full"] <- "nt_gls_full"
  out
}

guttman_backend_method <- function(method) {
  if (method == "nt_gls_block") return("gmm_block")
  if (method == "nt_gls_full") return("gmm_full")
  method
}

accepted_methods <- c(
  "ar", "rs", "ilm", "anchor_ilm", "nt_gls_block", "nt_gls_full",
  "gmm_block", "gmm_full", "nt_ml"
)

accepted_maps <- c("incidence", "aligned", "diag_aligned", "block_aligned")

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
    "  --n LIST             Sample sizes. Full default: 12,20,50,100,250,500.\n",
    "  --factors LIST       Factor counts. Full default: 1,2,3,4.\n",
    "  --indicators LIST    Indicators per factor. Full default: 3,5.\n",
    "  --rho LIST           Exchangeable latent correlations. Default: 0,.4.\n",
    "  --loading LIST       Loading patterns: mild,wide. Full default: mild,wide.\n",
    "  --generators LIST    Data generators: normal,ordinal. Default: both.\n",
    "  --methods LIST       Methods. Default: rs,ilm,anchor_ilm,nt_gls_block,nt_ml.\n",
    "                       Also accepted: ar,nt_gls_full,gmm_block,gmm_full.\n",
    "  --maps LIST          Guttman maps: incidence,aligned,block_aligned.\n",
    "                       Optional: diag_aligned. Default: first three.\n",
    "  --timing-reps N      Dedicated timing reps per cell. Full default: 20.\n",
    "  --timing-warmup N    Unrecorded timing warmups per cell. Default: 2.\n",
    "  --timing-inner N     Repeated calls per timing component. Full default: 10.\n",
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
    n = c(12L, 20L, 50L, 100L, 250L, 500L),
    factors = c(1L, 2L, 3L, 4L),
    indicators = c(3L, 5L),
    rho = c(0, .4),
    loading = c("mild", "wide"),
    generators = c("normal", "ordinal"),
    methods = c("rs", "ilm", "anchor_ilm", "nt_gls_block", "nt_ml"),
    maps = c("incidence", "aligned", "block_aligned"),
    timing_reps = 20L,
    timing_warmup = 2L,
    timing_inner = 10L,
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
    } else if (a == "--maps") {
      opts$maps <- parse_csv_arg(take(a)); explicit <- c(explicit, "maps")
    } else if (grepl("^--maps=", a)) {
      opts$maps <- parse_csv_arg(sub("^--maps=", "", a)); explicit <- c(explicit, "maps")
    } else if (a == "--timing-reps") {
      opts$timing_reps <- as.integer(take(a)); explicit <- c(explicit, "timing_reps")
    } else if (grepl("^--timing-reps=", a)) {
      opts$timing_reps <- as.integer(sub("^--timing-reps=", "", a))
      explicit <- c(explicit, "timing_reps")
    } else if (a == "--timing-warmup") {
      opts$timing_warmup <- as.integer(take(a)); explicit <- c(explicit, "timing_warmup")
    } else if (grepl("^--timing-warmup=", a)) {
      opts$timing_warmup <- as.integer(sub("^--timing-warmup=", "", a))
      explicit <- c(explicit, "timing_warmup")
    } else if (a == "--timing-inner") {
      opts$timing_inner <- as.integer(take(a)); explicit <- c(explicit, "timing_inner")
    } else if (grepl("^--timing-inner=", a)) {
      opts$timing_inner <- as.integer(sub("^--timing-inner=", "", a))
      explicit <- c(explicit, "timing_inner")
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
    if (!"n" %in% explicit) opts$n <- c(12L, 20L, 50L, 100L)
    if (!"factors" %in% explicit) opts$factors <- 3L
    if (!"indicators" %in% explicit) opts$indicators <- c(3L, 5L)
    if (!"rho" %in% explicit) opts$rho <- c(0, .4)
    if (!"loading" %in% explicit) opts$loading <- c("mild", "wide")
    if (!"generators" %in% explicit) opts$generators <- c("normal", "ordinal")
    if (!"timing_reps" %in% explicit) opts$timing_reps <- 6L
    if (!"timing_inner" %in% explicit) opts$timing_inner <- 25L
  }
  bad <- setdiff(opts$loading, c("mild", "wide"))
  if (length(bad)) stop("unknown loading patterns: ", paste(bad, collapse = ","), call. = FALSE)
  bad <- setdiff(opts$generators, c("normal", "ordinal"))
  if (length(bad)) stop("unknown generators: ", paste(bad, collapse = ","), call. = FALSE)
  bad <- setdiff(opts$methods, accepted_methods)
  if (length(bad)) stop("unknown methods: ", paste(bad, collapse = ","), call. = FALSE)
  opts$methods <- unique(canonical_method(opts$methods))
  bad <- setdiff(opts$maps, accepted_maps)
  if (length(bad)) stop("unknown maps: ", paste(bad, collapse = ","), call. = FALSE)
  opts$maps <- unique(opts$maps)
  if (!is.finite(opts$timing_reps) || opts$timing_reps < 0) {
    stop("--timing-reps must be non-negative", call. = FALSE)
  }
  if (!is.finite(opts$timing_warmup) || opts$timing_warmup < 0) {
    stop("--timing-warmup must be non-negative", call. = FALSE)
  }
  if (!is.finite(opts$timing_inner) || opts$timing_inner < 1) {
    stop("--timing-inner must be at least 1", call. = FALSE)
  }
  opts
}

opts <- parse_args(commandArgs(TRUE))
set_single_threaded_math()
dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)

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

sample_cov_n <- function(x, vars = NULL) {
  xc <- sweep(x, 2L, colMeans(x), check.margin = FALSE)
  s <- crossprod(xc) / nrow(x)
  if (!is.null(vars)) dimnames(s) <- list(vars, vars)
  s
}

sample_stats_n <- function(x, vars) {
  s <- sample_cov_n(x, vars)
  mu <- colMeans(x)
  names(mu) <- vars
  list(S = list(s), mean = list(mu), nobs = as.integer(nrow(x)))
}

min_eigen_sym <- function(x) {
  if (is.null(x) || !all(is.finite(x))) return(NA_real_)
  out <- tryCatch(
    eigen((x + t(x)) / 2, symmetric = TRUE, only.values = TRUE)$values,
    error = function(e) NA_real_
  )
  if (all(is.na(out))) NA_real_ else min(out)
}

make_Z <- function(blocks) {
  p <- length(blocks)
  q <- length(unique(blocks))
  Z <- matrix(0, p, q)
  for (f in seq_len(q)) {
    Z[blocks == f, f] <- 1
  }
  Z
}

correlation_from_cov <- function(s) {
  d <- diag(s)
  if (any(!is.finite(d) | d <= 0)) {
    stop("covariance diagonal must be positive", call. = FALSE)
  }
  r <- s / sqrt(outer(d, d))
  diag(r) <- 1
  dimnames(r) <- dimnames(s)
  r
}

anchor_row_cache <- new.env(parent = emptyenv())

anchor_triad_rows <- function(blocks, include_cross = TRUE) {
  key <- paste(include_cross, paste(blocks, collapse = ","), sep = "|")
  if (exists(key, envir = anchor_row_cache, inherits = FALSE)) {
    return(get(key, envir = anchor_row_cache, inherits = FALSE))
  }
  p <- length(blocks)
  target <- integer()
  own_anchor <- integer()
  other_anchor <- integer()
  for (i in seq_len(p)) {
    same <- setdiff(which(blocks == blocks[[i]]), i)
    if (length(same) >= 2L) {
      cmb <- utils::combn(same, 2L)
      target <- c(target, rep(i, ncol(cmb)))
      own_anchor <- c(own_anchor, cmb[1L, ])
      other_anchor <- c(other_anchor, cmb[2L, ])
    }
    if (include_cross && length(same)) {
      cross <- which(blocks != blocks[[i]])
      if (length(cross)) {
        grid <- expand.grid(j = same, k = cross)
        target <- c(target, rep(i, nrow(grid)))
        own_anchor <- c(own_anchor, grid$j)
        other_anchor <- c(other_anchor, grid$k)
      }
    }
  }
  out <- list(i = target, j = own_anchor, k = other_anchor,
              n = length(target))
  assign(key, out, envir = anchor_row_cache)
  out
}

anchor_gamma_identity <- function(r, blocks, include_cross = TRUE,
                                  tol = sqrt(.Machine$double.eps)) {
  rows <- anchor_triad_rows(blocks, include_cross = include_cross)
  p <- nrow(r)
  a <- r[cbind(rows$j, rows$k)]
  b <- r[cbind(rows$i, rows$j)] * r[cbind(rows$i, rows$k)]
  num <- as.numeric(rowsum(a * b, rows$i, reorder = FALSE))
  den <- as.numeric(rowsum(a * a, rows$i, reorder = FALSE))
  gamma <- rep(NA_real_, p)
  ok <- is.finite(den) & den > tol
  gamma[ok] <- num[ok] / den[ok]
  list(gamma = gamma, rows = rows$n)
}

anchor_guttman_h <- function(s, blocks) {
  r <- correlation_from_cov(s)
  fit <- anchor_gamma_identity(r, blocks, include_cross = TRUE)
  h2 <- diag(s) * fit$gamma
  H <- s
  diag(H) <- h2
  dimnames(H) <- dimnames(s)
  list(h2 = h2, H = H, gamma = fit$gamma, rows = fit$rows)
}

estimate_guttman_h <- function(s, blocks, method) {
  if (method == "anchor_ilm") return(anchor_guttman_h(s, blocks))
  magmaan::guttman_h(s, blocks, method = guttman_backend_method(method))
}

sym_power <- function(x, power, label, tol = 1e-10) {
  x <- (x + t(x)) / 2
  es <- tryCatch(eigen(x, symmetric = TRUE), error = function(e) NULL)
  if (is.null(es) || any(!is.finite(es$values))) {
    stop(label, ": eigendecomposition failed", call. = FALSE)
  }
  scale <- max(1, max(abs(es$values)))
  if (min(es$values) <= tol * scale) {
    stop(label, ": covariance block is not positive definite", call. = FALSE)
  }
  es$vectors %*% diag(es$values^power, nrow = length(es$values)) %*%
    t(es$vectors)
}

canonical_transform <- function(s, blocks, map) {
  p <- nrow(s)
  W <- diag(1, p)
  Ghalf <- diag(1, p)
  if (map %in% c("incidence", "aligned")) {
    return(list(W = W, Ghalf = Ghalf))
  }
  if (map == "diag_aligned") {
    d <- diag(s)
    if (any(!is.finite(d) | d <= 0)) {
      stop("diag_aligned map: covariance diagonal must be positive",
           call. = FALSE)
    }
    W <- diag(1 / sqrt(d), p)
    Ghalf <- diag(sqrt(d), p)
    return(list(W = W, Ghalf = Ghalf))
  }
  if (map == "block_aligned") {
    for (b in sort(unique(blocks))) {
      idx <- which(blocks == b)
      sb <- s[idx, idx, drop = FALSE]
      W[idx, idx] <- sym_power(sb, -0.5, "block_aligned map")
      Ghalf[idx, idx] <- sym_power(sb, 0.5, "block_aligned map")
    }
    return(list(W = W, Ghalf = Ghalf))
  }
  stop("unknown Guttman map: ", map, call. = FALSE)
}

common_from_scores <- function(hmat, B) {
  score_cov <- crossprod(B, hmat %*% B)
  inv_score_cov <- tryCatch(solve(score_cov), error = function(e) NULL)
  if (is.null(inv_score_cov)) return(list(common = NULL, score_cov = score_cov))
  hz <- hmat %*% B
  common <- hz %*% inv_score_cov %*% t(hz)
  dimnames(common) <- dimnames(hmat)
  list(common = common, score_cov = score_cov)
}

aligned_scores <- function(hmat, Z) {
  pre <- common_from_scores(hmat, Z)
  if (is.null(pre$common)) {
    stop("aligned map: preliminary incidence regression failed", call. = FALSE)
  }
  Gamma <- hmat %*% Z %*% solve(pre$score_cov)
  gram <- crossprod(Gamma)
  inv_gram <- tryCatch(solve(gram), error = function(e) NULL)
  if (is.null(inv_gram)) {
    stop("aligned map: preliminary loading matrix is rank deficient",
         call. = FALSE)
  }
  Gamma %*% inv_gram
}

guttman_common <- function(hmat, blocks, s, map) {
  trans <- canonical_transform(s, blocks, map)
  h_work <- trans$W %*% hmat %*% trans$W
  h_work <- (h_work + t(h_work)) / 2
  dimnames(h_work) <- dimnames(hmat)
  Z <- make_Z(blocks)
  B <- if (map == "incidence") Z else aligned_scores(h_work, Z)
  fit <- common_from_scores(h_work, B)
  if (is.null(fit$common)) return(fit)
  common <- trans$Ghalf %*% fit$common %*% trans$Ghalf
  common <- (common + t(common)) / 2
  dimnames(common) <- dimnames(hmat)
  list(common = common, score_cov = fit$score_cov)
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
  common <- Lambda %*% Phi %*% t(Lambda)
  dimnames(common) <- list(vars, vars)
  list(common = common,
       resid = resid,
       improper = any(is.finite(resid) & resid < 0))
}

fit_nt_ml <- function(sample_stats, spec, design, control) {
  fit <- core$estimate_ml(spec$partable, sample_stats, optimizer = "nlopt-lbfgs",
                          control = control, bounds = NULL)
  if (!isTRUE(fit$converged)) stop("magmaan ML did not converge", call. = FALSE)
  ml_common_from_fit(fit, design$vars, design$q)
}

time_call <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- tryCatch(
    list(ok = TRUE, value = force(expr), message = ""),
    error = function(e) list(ok = FALSE, value = NULL, message = conditionMessage(e))
  )
  value$elapsed_ms <- 1000 * (proc.time()[["elapsed"]] - start)
  value
}

one_method <- function(x, design, spec, target_common, method, map, ml_control) {
  if (method == "nt_ml") {
    timed <- time_call({
      sample_stats <- sample_stats_n(x, design$vars)
      out <- fit_nt_ml(sample_stats, spec, design, ml_control)
      out$s <- sample_stats$S[[1]]
      out$failed <- FALSE
      out$message <- ""
      out
    })
  } else {
    timed <- time_call({
      s <- sample_cov_n(x, design$vars)
      h <- tryCatch(
        estimate_guttman_h(s, design$blocks, method),
        error = function(e) e
      )
      if (inherits(h, "error")) {
        list(common = NULL, resid = NULL, s = s, hmat = NULL,
             score_cov = NULL, failed = TRUE, message = conditionMessage(h))
      } else if (any(!is.finite(h$h2))) {
        list(common = NULL, resid = NULL, s = s, hmat = h$H,
             score_cov = NULL, failed = TRUE,
             message = "non-finite communality")
      } else {
        recon <- guttman_common(h$H, design$blocks, s, map)
        if (is.null(recon$common)) {
          list(common = NULL, resid = NULL, s = s, hmat = h$H,
               score_cov = recon$score_cov, failed = TRUE,
               message = "Guttman reconstruction failed")
        } else {
          resid <- diag(s - recon$common)
          list(common = recon$common, resid = resid, s = s, hmat = h$H,
               score_cov = recon$score_cov, failed = FALSE, message = "",
               improper = any(resid < 0))
        }
      }
    })
  }
  failed <- !timed$ok || is.null(timed$value) || isTRUE(timed$value$failed)
  common <- if (failed) NULL else timed$value$common
  resid <- if (is.null(timed$value$resid)) NA_real_ else timed$value$resid
  s <- if (is.null(timed$value$s)) NULL else timed$value$s
  hmat <- if (is.null(timed$value$hmat)) NULL else timed$value$hmat
  score_cov <- if (is.null(timed$value$score_cov)) NULL else timed$value$score_cov
  improper <- if (failed && all(is.na(resid))) NA else any(is.finite(resid) & resid < 0)
  sample_min_eig <- min_eigen_sym(s)
  h_min_eig <- min_eigen_sym(hmat)
  score_min_eig <- min_eigen_sym(score_cov)
  common_min_eig <- min_eigen_sym(common)
  min_resid <- if (all(is.na(resid))) NA_real_ else min(resid, na.rm = TRUE)
  h_diag_bad <- if (is.null(hmat) || is.null(s)) {
    NA
  } else {
    any(!is.finite(diag(hmat)) | diag(hmat) < -1e-8 |
          diag(hmat) - diag(s) > 1e-8)
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
    min_resid = min_resid,
    sample_min_eig = sample_min_eig,
    h_min_eig = h_min_eig,
    score_min_eig = score_min_eig,
    common_min_eig = common_min_eig,
    sample_singular = if (is.finite(sample_min_eig)) sample_min_eig <= 1e-8 else NA,
    h_diag_bad = h_diag_bad,
    h_indef = if (is.finite(h_min_eig)) h_min_eig < -1e-8 else NA,
    score_nonpos = if (is.finite(score_min_eig)) score_min_eig <= 1e-8 else NA,
    common_indef = if (is.finite(common_min_eig)) common_min_eig < -1e-8 else NA,
    message = if (failed) {
      if (!is.null(timed$value$message)) timed$value$message else timed$message
    } else "",
    stringsAsFactors = FALSE
  )
}

timing_record <- function(method, component, elapsed_ms, failed, message) {
  data.frame(
    method = method,
    component = component,
    elapsed_ms = elapsed_ms,
    failed = failed,
    message = message,
    stringsAsFactors = FALSE
  )
}

time_repeat <- function(times, fn) {
  start <- proc.time()[["elapsed"]]
  value <- NULL
  ok <- TRUE
  message <- ""
  completed <- 0L
  for (ii in seq_len(times)) {
    result <- tryCatch(
      list(ok = TRUE, value = fn(), message = ""),
      error = function(e) list(ok = FALSE, value = NULL,
                               message = conditionMessage(e))
    )
    completed <- ii
    if (!result$ok) {
      ok <- FALSE
      message <- result$message
      value <- NULL
      break
    }
    value <- result$value
  }
  elapsed_ms <- 1000 * (proc.time()[["elapsed"]] - start) / max(completed, 1L)
  list(ok = ok, value = value, message = message, elapsed_ms = elapsed_ms)
}

time_one_method_detail <- function(x, design, spec, method, map, ml_control,
                                   timing_inner) {
  rows <- list()
  add_row <- function(component, timed, failed = !timed$ok,
                      message = timed$message) {
    rows[[length(rows) + 1L]] <<- timing_record(
      method, component, timed$elapsed_ms, failed, if (failed) message else "")
  }

  if (method == "nt_ml") {
    stats_t <- time_repeat(timing_inner, function() sample_stats_n(x, design$vars))
    add_row("sample_stats", stats_t)
    fit_failed <- TRUE
    fit_t <- NULL
    if (stats_t$ok) {
      fit_t <- time_repeat(timing_inner, function() core$estimate_ml(
        spec$partable, stats_t$value, optimizer = "nlopt-lbfgs",
        control = ml_control, bounds = NULL))
      fit_failed <- !fit_t$ok || !isTRUE(fit_t$value$converged)
      fit_message <- if (!fit_t$ok) {
        fit_t$message
      } else if (!isTRUE(fit_t$value$converged)) {
        "magmaan ML did not converge"
      } else {
        ""
      }
      add_row("estimate_ml", fit_t, fit_failed, fit_message)
    }
    if (!fit_failed) {
      extract_t <- time_repeat(
        timing_inner,
        function() ml_common_from_fit(fit_t$value, design$vars, design$q))
      add_row("extract_common", extract_t)
    }
  } else {
    cov_t <- time_repeat(timing_inner, function() sample_cov_n(x, design$vars))
    add_row("sample_cov", cov_t)
    h_failed <- TRUE
    h_t <- NULL
    if (cov_t$ok) {
      h_t <- time_repeat(
        timing_inner,
        function() estimate_guttman_h(cov_t$value, design$blocks, method))
      h_failed <- !h_t$ok || any(!is.finite(h_t$value$h2))
      h_message <- if (!h_t$ok) {
        h_t$message
      } else if (any(!is.finite(h_t$value$h2))) {
        "non-finite communality"
      } else {
        ""
      }
      add_row("guttman_h", h_t, h_failed, h_message)
    }
    if (!h_failed) {
      map_t <- time_repeat(
        timing_inner,
        function() guttman_common(h_t$value$H, design$blocks, cov_t$value, map))
      map_failed <- !map_t$ok || is.null(map_t$value$common)
      map_message <- if (!map_t$ok) {
        map_t$message
      } else if (is.null(map_t$value$common)) {
        "Guttman reconstruction failed"
      } else {
        ""
      }
      add_row("guttman_map", map_t, map_failed, map_message)
    }
  }

  out <- do.call(rbind, rows)
  total_failed <- any(out$failed)
  total_message <- paste(out$message[out$failed & nzchar(out$message)], collapse = "; ")
  rbind(out, timing_record(method, "total", sum(out$elapsed_ms), total_failed,
                           total_message))
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
  anchor_rows_per_item <- (m - 1) * (m - 2) / 2 + (m - 1) * (p - m)
  anchor_rows_total <- p * anchor_rows_per_item
  data.frame(
    factors = q,
    indicators = m,
    p = p,
    corr_pairs_per_block = corr_block,
    triad_rows_per_block = triad_block,
    anchor_rows_per_item = anchor_rows_per_item,
    within_corr_pairs_total = q * corr_block,
    triad_rows_total = q * triad_block,
    anchor_rows_total = anchor_rows_total,
    full_sem_corr_pairs = p * (p - 1) / 2,
    block_weight_dim = triad_block,
    full_weight_dim = q * triad_block,
    stringsAsFactors = FALSE
  )
}

summarise_vec <- function(x) {
  x <- as.numeric(x[is.finite(x)])
  if (!length(x)) {
    return(c(mean = NA_real_, se = NA_real_, median = NA_real_,
             p10 = NA_real_, p90 = NA_real_, min = NA_real_))
  }
  c(mean = mean(x),
    se = stats::sd(x) / sqrt(length(x)),
    median = stats::median(x),
    p10 = stats::quantile(x, .1, names = FALSE),
    p90 = stats::quantile(x, .9, names = FALSE),
    min = min(x))
}

summarise_results <- function(rows) {
  agg <- aggregate(
    cbind(diag_mise, common_mise, elapsed_ms, failed, improper,
          min_resid, sample_min_eig, h_min_eig, score_min_eig,
          common_min_eig, sample_singular, h_diag_bad, h_indef,
          score_nonpos, common_indef) ~
      generator + factors + indicators + loading + rho + n + method + map,
    rows,
    summarise_vec,
    na.action = stats::na.pass
  )
  out <- data.frame(
    generator = agg$generator,
    factors = agg$factors,
    indicators = agg$indicators,
    loading = agg$loading,
    rho = agg$rho,
    n = agg$n,
    method = agg$method,
    map = agg$map,
    diag_mise = agg$diag_mise[, "mean"],
    diag_se = agg$diag_mise[, "se"],
    common_mise = agg$common_mise[, "mean"],
    common_se = agg$common_mise[, "se"],
    elapsed_mean_ms = agg$elapsed_ms[, "mean"],
    elapsed_median_ms = agg$elapsed_ms[, "median"],
    elapsed_p90_ms = agg$elapsed_ms[, "p90"],
    fail_rate = agg$failed[, "mean"],
    improper_rate = agg$improper[, "mean"],
    min_resid_median = agg$min_resid[, "median"],
    min_resid_p10 = agg$min_resid[, "p10"],
    min_resid_min = agg$min_resid[, "min"],
    sample_min_eig_median = agg$sample_min_eig[, "median"],
    sample_min_eig_min = agg$sample_min_eig[, "min"],
    h_min_eig_median = agg$h_min_eig[, "median"],
    h_min_eig_min = agg$h_min_eig[, "min"],
    score_min_eig_median = agg$score_min_eig[, "median"],
    score_min_eig_min = agg$score_min_eig[, "min"],
    common_min_eig_median = agg$common_min_eig[, "median"],
    common_min_eig_min = agg$common_min_eig[, "min"],
    sample_singular_rate = agg$sample_singular[, "mean"],
    h_diag_bad_rate = agg$h_diag_bad[, "mean"],
    h_indef_rate = agg$h_indef[, "mean"],
    score_nonpos_rate = agg$score_nonpos[, "mean"],
    common_indef_rate = agg$common_indef[, "mean"],
    stringsAsFactors = FALSE
  )
  key <- c("generator", "factors", "indicators", "loading", "rho", "n")
  key_map <- c(key, "map")
  rs <- out[out$method == "rs", c(key_map, "common_mise", "diag_mise")]
  names(rs)[(length(key_map) + 1L):(length(key_map) + 2L)] <- c("common_rs", "diag_rs")
  nt <- out[out$method == "nt_ml", c(key, "common_mise", "diag_mise")]
  names(nt)[(length(key) + 1L):(length(key) + 2L)] <- c("common_nt", "diag_nt")
  out <- merge(out, rs, by = key_map, all.x = TRUE)
  out <- merge(out, nt, by = key, all.x = TRUE)
  out$common_rel_rs <- out$common_mise / out$common_rs
  out$diag_rel_rs <- out$diag_mise / out$diag_rs
  out$common_rel_nt_marginal <- out$common_mise / out$common_nt
  out$diag_rel_nt_marginal <- out$diag_mise / out$diag_nt

  nt_rep <- rows[rows$method == "nt_ml",
                 c(key, "rep", "common_mise", "diag_mise", "failed")]
  names(nt_rep)[(length(key) + 2L):ncol(nt_rep)] <-
    c("common_mise_nt_rep", "diag_mise_nt_rep", "failed_nt_rep")
  joint <- merge(rows, nt_rep, by = c(key, "rep"), all.x = TRUE)
  joint$joint_success <- !joint$failed & !joint$failed_nt_rep
  joint$common_mise_joint <- ifelse(joint$joint_success,
                                    joint$common_mise, NA_real_)
  joint$diag_mise_joint <- ifelse(joint$joint_success,
                                  joint$diag_mise, NA_real_)
  joint$common_nt_joint <- ifelse(joint$joint_success,
                                  joint$common_mise_nt_rep, NA_real_)
  joint$diag_nt_joint <- ifelse(joint$joint_success,
                                joint$diag_mise_nt_rep, NA_real_)
  joint_agg <- aggregate(
    cbind(common_mise_joint, diag_mise_joint, common_nt_joint, diag_nt_joint,
          joint_success) ~
      generator + factors + indicators + loading + rho + n + method + map,
    joint,
    function(x) mean(x, na.rm = TRUE),
    na.action = stats::na.pass
  )
  names(joint_agg)[(ncol(joint_agg) - 4L):ncol(joint_agg)] <-
    c("common_mise_nt_joint", "diag_mise_nt_joint", "common_nt_joint",
      "diag_nt_joint", "nt_joint_success_rate")
  joint_n <- aggregate(
    joint_success ~ generator + factors + indicators + loading + rho + n +
      method + map,
    joint,
    sum,
    na.action = stats::na.pass
  )
  names(joint_n)[ncol(joint_n)] <- "nt_joint_success_n"
  joint_agg <- merge(joint_agg, joint_n,
                     by = c(key, "method", "map"), all.x = TRUE)
  out <- merge(out, joint_agg, by = c(key, "method", "map"), all.x = TRUE)
  out$common_rel_nt <- out$common_mise_nt_joint / out$common_nt_joint
  out$diag_rel_nt <- out$diag_mise_nt_joint / out$diag_nt_joint
  out[order(out$generator, out$factors, out$indicators, out$loading, out$rho,
            out$n, out$map, out$method), ]
}

summarise_timing <- function(rows) {
  if (!nrow(rows)) return(data.frame())
  agg <- aggregate(
    cbind(elapsed_ms, failed) ~
      generator + factors + indicators + loading + rho + n + method + map +
        component,
    rows,
    summarise_vec,
    na.action = stats::na.pass
  )
  out <- data.frame(
    generator = agg$generator,
    factors = agg$factors,
    indicators = agg$indicators,
    loading = agg$loading,
    rho = agg$rho,
    n = agg$n,
    method = agg$method,
    map = agg$map,
    component = agg$component,
    elapsed_mean_ms = agg$elapsed_ms[, "mean"],
    elapsed_median_ms = agg$elapsed_ms[, "median"],
    elapsed_p10_ms = agg$elapsed_ms[, "p10"],
    elapsed_p90_ms = agg$elapsed_ms[, "p90"],
    fail_rate = agg$failed[, "mean"],
    stringsAsFactors = FALSE
  )
  out[order(out$generator, out$factors, out$indicators, out$loading, out$rho,
            out$n, out$map, out$method, out$component), ]
}

run_timing_study <- function(grid, opts, ml_control) {
  if (opts$timing_reps == 0L) return(data.frame())
  rows <- list()
  row_i <- 0L
  for (cc in seq_len(nrow(grid))) {
    g <- grid[cc, ]
    design <- make_design(g$factors, g$indicators, g$rho, g$loading)
    spec <- magmaan::model_spec(model_syntax(design$q, design$m))
    n_draws <- opts$timing_warmup + opts$timing_reps
    seed <- opts$seed_base + 700000000L + cc * 100000L
    draws <- draws_for_condition(design, g$generator, g$n, n_draws, seed)
    for (rr in seq_along(draws)) {
      x <- draws[[rr]]
      for (method in opts$methods) {
        maps <- if (method == "nt_ml") "ml" else opts$maps
        for (map in maps) {
          out <- time_one_method_detail(
            x, design, spec, method, map, ml_control, opts$timing_inner)
          if (rr > opts$timing_warmup) {
            out$generator <- g$generator
            out$factors <- g$factors
            out$indicators <- g$indicators
            out$loading <- g$loading
            out$rho <- g$rho
            out$n <- g$n
            out$map <- map
            out$timing_rep <- rr - opts$timing_warmup
            row_i <- row_i + 1L
            rows[[row_i]] <- out
          }
        }
      }
    }
    message("timed condition ", cc, "/", nrow(grid))
  }
  if (!length(rows)) return(data.frame())
  do.call(rbind, rows)
}

write_all <- function(reps, summary, complexity, metadata,
                      timing_reps, timing_summary) {
  write_csv(reps, file.path(opts$results_dir, "replicates.csv"))
  write_csv(summary, file.path(opts$results_dir, "summary.csv"))
  write_csv(complexity, file.path(opts$results_dir, "complexity.csv"))
  write_csv(metadata, file.path(opts$results_dir, "metadata.csv"))
  write_csv(timing_reps, file.path(opts$results_dir, "timing_replicates.csv"))
  write_csv(timing_summary, file.path(opts$results_dir, "timing_summary.csv"))
}

grid <- condition_grid(opts)
complexity <- unique(do.call(rbind, Map(complexity_row, grid$factors, grid$indicators)))
complexity <- complexity[order(complexity$factors, complexity$indicators), ]

message("conditions: ", nrow(grid), "; reps per condition: ", opts$reps)
message("methods: ", paste(opts$methods, collapse = ","))
message("maps: ", paste(opts$maps, collapse = ","))
print(complexity, row.names = FALSE)

ml_control <- list(max_iter = opts$ml_max_iter, ftol = 1e-10, gtol = 1e-8)
rows <- list()
row_i <- 0L
for (cc in seq_len(nrow(grid))) {
  g <- grid[cc, ]
  design <- make_design(g$factors, g$indicators, g$rho, g$loading)
  spec <- magmaan::model_spec(model_syntax(design$q, design$m))
  target <- target_for_generator(design, g$generator)
  seed <- opts$seed_base + cc * 100000L
  draws <- draws_for_condition(design, g$generator, g$n, opts$reps, seed)
  for (rr in seq_along(draws)) {
    x <- draws[[rr]]
    for (method in opts$methods) {
      maps <- if (method == "nt_ml") "ml" else opts$maps
      for (map in maps) {
        row_i <- row_i + 1L
        out <- one_method(x, design, spec, target$common, method, map,
                          ml_control)
        out$generator <- g$generator
        out$factors <- g$factors
        out$indicators <- g$indicators
        out$loading <- g$loading
        out$rho <- g$rho
        out$n <- g$n
        out$map <- map
        out$rep <- rr
        rows[[row_i]] <- out
      }
    }
  }
  message("finished condition ", cc, "/", nrow(grid))
}

replicates <- do.call(rbind, rows)
summary <- summarise_results(replicates)
message("running dedicated timing pass: reps=", opts$timing_reps,
        "; warmup=", opts$timing_warmup,
        "; inner=", opts$timing_inner)
timing_replicates <- run_timing_study(grid, opts, ml_control)
timing_summary <- summarise_timing(timing_replicates)
metadata <- metadata_frame(
  list(
    reps = opts$reps,
    timing_reps = opts$timing_reps,
    timing_warmup = opts$timing_warmup,
    timing_inner = opts$timing_inner,
    n = opts$n,
    factors = opts$factors,
    indicators = opts$indicators,
    rho = opts$rho,
    loading = opts$loading,
    generators = opts$generators,
    methods = opts$methods,
    maps = opts$maps,
    seed_base = opts$seed_base,
    ml_max_iter = opts$ml_max_iter,
    ordinal_prob = ordinal_prob,
    smoke = opts$smoke,
    covariance_scaling = "n",
    timing_scope = paste(
      "per-rep clocks include ML-scale covariance/sample-stat construction,",
      "the estimator, and common-covariance reconstruction;",
      "model syntax/lavaanification is precomputed per condition"
    ),
    guttman_reconstruction = "traditional incidence-composite map H Z (Z' H Z)^-1 Z' H"
  ),
  packages = c("magmaan")
)

write_all(replicates, summary, complexity, metadata,
          timing_replicates, timing_summary)
message("wrote ", normalizePath(opts$results_dir, mustWork = FALSE))
