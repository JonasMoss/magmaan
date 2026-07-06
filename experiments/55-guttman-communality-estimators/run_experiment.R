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
    "  --n LIST             Sample sizes. Full default: 12,20,50,100,250,500.\n",
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
    n = c(12L, 20L, 50L, 100L, 250L, 500L),
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
    if (!"n" %in% explicit) opts$n <- c(12L, 20L, 50L, 100L)
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

guttman_common <- function(hmat, blocks) {
  p <- nrow(hmat)
  q <- length(unique(blocks))
  Z <- matrix(0, p, q)
  for (f in seq_len(q)) {
    Z[blocks == f, f] <- 1
  }
  score_cov <- crossprod(Z, hmat %*% Z)
  inv_score_cov <- tryCatch(solve(score_cov), error = function(e) NULL)
  if (is.null(inv_score_cov)) return(list(common = NULL, score_cov = score_cov))
  hz <- hmat %*% Z
  common <- hz %*% inv_score_cov %*% t(hz)
  dimnames(common) <- dimnames(hmat)
  list(common = common, score_cov = score_cov)
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
  start <- Sys.time()
  value <- tryCatch(
    list(ok = TRUE, value = force(expr), message = ""),
    error = function(e) list(ok = FALSE, value = NULL, message = conditionMessage(e))
  )
  value$elapsed_ms <- 1000 * as.numeric(difftime(Sys.time(), start, units = "secs"))
  value
}

one_method <- function(x, design, spec, target_common, method, ml_control) {
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
        magmaan::guttman_h(s, design$blocks, method = method),
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
        recon <- guttman_common(h$H, design$blocks)
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
  agg <- aggregate(
    cbind(diag_mise, common_mise, elapsed_ms, failed, improper,
          min_resid, sample_min_eig, h_min_eig, score_min_eig,
          common_min_eig, sample_singular, h_diag_bad, h_indef,
          score_nonpos, common_indef) ~
      generator + factors + indicators + loading + rho + n + method,
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
  spec <- magmaan::model_spec(model_syntax(design$q, design$m))
  target <- target_for_generator(design, g$generator)
  seed <- opts$seed_base + cc * 100000L
  draws <- draws_for_condition(design, g$generator, g$n, opts$reps, seed)
  for (rr in seq_along(draws)) {
    x <- draws[[rr]]
    for (method in opts$methods) {
      row_i <- row_i + 1L
      out <- one_method(x, design, spec, target$common, method, ml_control)
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

write_all(replicates, summary, complexity, metadata)
message("wrote ", normalizePath(opts$results_dir, mustWork = FALSE))
