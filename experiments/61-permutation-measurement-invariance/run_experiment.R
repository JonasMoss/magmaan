#!/usr/bin/env Rscript
# Experiment 61: studentized permutation tests for continuous metric invariance.
#
# This is the feasibility gate for a measurement-invariance permutation paper.
# It compares group-label permutation distributions of a fully recomputed
# sandwich-Wald statistic with ML LRT and chi-square Wald calibrations. The
# continuous two-group metric test is deliberately first: its pooled covariance
# preserves the common-loading CFA form when factor and residual variances differ.

suppressWarnings(suppressMessages(library(magmaan)))
source(file.path(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]])),
  "..", "_support", "R", "helpers.R"
))
set_single_threaded_math()
core <- magmaan::magmaan_core

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Tests continuous two-group metric invariance with an ML LRT, model and\n",
    "sandwich Wald tests, and fully recomputed raw-label permutation tests.\n\n",
    "Options:\n",
    "  --smoke             Cheap H0 check (10 reps, 19 permutations). Default.\n",
    "  --full              Full distribution x sample-size x scenario grid.\n",
    "  --reps N            Replications per cell in full mode. Default: 1000.\n",
    "  --permutations B    Random label permutations per replication. Default: 499.\n",
    "  --n LIST            Group-size pairs, e.g. 100:250,250:250.\n",
    "                      Default: 100:250,250:250.\n",
    "  --models L          Subset of one_factor,three_factor. Default: one_factor.\n",
    "  --heterogeneity L   Subset of standard,strong. Default: standard.\n",
    "  --distributions L   Subset of normal,t5. Default: both.\n",
    "  --scenarios L       Subset of exchangeable,invariant,loading_violation.\n",
    "                      Default: invariant,loading_violation.\n",
    "  --alpha X           Rejection level. Default: 0.05.\n",
    "  --cores N           mclapply cores. Default: detectCores()-1.\n",
    "  --seed-base N       Base RNG seed. Default: 20260710.\n",
    "  --results-dir PATH  Output directory. Default: results.\n",
    "  --help              Show this help.\n",
    sep = ""
  )
}

parse_group_sizes <- function(x) {
  out <- strsplit(parse_csv_arg(x), ":", fixed = TRUE)
  ans <- lapply(out, function(z) {
    if (length(z) != 2L) stop("each --n entry must be n1:n2", call. = FALSE)
    n <- as.integer(z)
    if (anyNA(n) || any(n < 20L)) stop("group sizes must be integers >= 20", call. = FALSE)
    n
  })
  if (!length(ans)) stop("--n must contain at least one n1:n2 pair", call. = FALSE)
  ans
}

opts <- list(
  mode = "smoke", reps = 1000L, permutations = 499L,
  sizes = list(c(100L, 250L), c(250L, 250L)),
  models = "one_factor", heterogeneity = "standard",
  distributions = c("normal", "t5"),
  scenarios = c("invariant", "loading_violation"),
  alpha = 0.05,
  cores = max(1L, parallel::detectCores() - 1L),
  seed_base = 20260710L, results_dir = NULL, explicit = character()
)

args <- commandArgs(trailingOnly = TRUE)
i <- 1L
take <- function() { i <<- i + 1L; args[[i]] }
while (i <= length(args)) {
  a <- args[[i]]
  if (a == "--help") { usage(); quit(status = 0L) }
  else if (a == "--smoke") opts$mode <- "smoke"
  else if (a == "--full") opts$mode <- "full"
  else if (a == "--reps") { opts$reps <- as.integer(take()); opts$explicit <- c(opts$explicit, "reps") }
  else if (a == "--permutations") { opts$permutations <- as.integer(take()); opts$explicit <- c(opts$explicit, "permutations") }
  else if (a == "--n") { opts$sizes <- parse_group_sizes(take()); opts$explicit <- c(opts$explicit, "sizes") }
  else if (a == "--models") { opts$models <- parse_csv_arg(take()); opts$explicit <- c(opts$explicit, "models") }
  else if (a == "--heterogeneity") { opts$heterogeneity <- parse_csv_arg(take()); opts$explicit <- c(opts$explicit, "heterogeneity") }
  else if (a == "--distributions") { opts$distributions <- parse_csv_arg(take()); opts$explicit <- c(opts$explicit, "distributions") }
  else if (a == "--scenarios") { opts$scenarios <- parse_csv_arg(take()); opts$explicit <- c(opts$explicit, "scenarios") }
  else if (a == "--alpha") opts$alpha <- as.numeric(take())
  else if (a == "--cores") opts$cores <- as.integer(take())
  else if (a == "--seed-base") opts$seed_base <- as.integer(take())
  else if (a == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
if (is.na(opts$reps) || opts$reps < 1L) stop("--reps must be positive", call. = FALSE)
if (is.na(opts$permutations) || opts$permutations < 1L) stop("--permutations must be positive", call. = FALSE)
if (!is.finite(opts$alpha) || opts$alpha <= 0 || opts$alpha >= 1) stop("--alpha must be in (0, 1)", call. = FALSE)
bad <- setdiff(opts$models, c("one_factor", "three_factor"))
if (length(bad)) stop("unknown models: ", paste(bad, collapse = ","), call. = FALSE)
bad <- setdiff(opts$heterogeneity, c("standard", "strong"))
if (length(bad)) stop("unknown heterogeneity levels: ", paste(bad, collapse = ","), call. = FALSE)
bad <- setdiff(opts$distributions, c("normal", "t5"))
if (length(bad)) stop("unknown distributions: ", paste(bad, collapse = ","), call. = FALSE)
bad <- setdiff(opts$scenarios, c("exchangeable", "invariant", "loading_violation"))
if (length(bad)) stop("unknown scenarios: ", paste(bad, collapse = ","), call. = FALSE)

if (opts$mode == "smoke") {
  if (!"reps" %in% opts$explicit) opts$reps <- 10L
  if (!"permutations" %in% opts$explicit) opts$permutations <- 19L
  if (!"sizes" %in% opts$explicit) opts$sizes <- list(c(60L, 90L))
  if (!"models" %in% opts$explicit) opts$models <- "one_factor"
  if (!"heterogeneity" %in% opts$explicit) opts$heterogeneity <- "standard"
  if (!"distributions" %in% opts$explicit) opts$distributions <- "normal"
  if (!"scenarios" %in% opts$explicit) opts$scenarios <- "invariant"
}
results_path <- opts$results_dir %||% ensure_results_dir()
dir.create(results_path, recursive = TRUE, showWarnings = FALSE)

# Population builders. The exchangeable control makes both group distributions
# identical, so its random label permutation is finite-sample valid for every
# statistic. The invariant scenario instead holds the loadings fixed while
# factor and residual variances differ between groups.
make_phi <- function(variances, correlations) {
  D <- diag(sqrt(variances))
  D %*% correlations %*% D
}

population_spec <- function(model_name, heterogeneity, scenario) {
  if (identical(model_name, "one_factor")) {
    ov <- paste0("x", 1:6)
    model <- "f =~ x1 + x2 + x3 + x4 + x5 + x6"
    Lambda1 <- matrix(c(1.00, 0.82, 0.74, 0.93, 0.68, 0.88), ncol = 1L)
    Theta1 <- c(0.42, 0.48, 0.56, 0.44, 0.60, 0.50)
    loading_rows <- data.frame(lhs = "f", rhs = ov[-1L], stringsAsFactors = FALSE)
    if (identical(heterogeneity, "strong")) {
      Phi2 <- matrix(2.40, 1L, 1L)
      Theta2 <- c(0.15, 0.95, 0.25, 1.05, 0.30, 0.90)
    } else {
      Phi2 <- matrix(1.65, 1L, 1L)
      Theta2 <- c(0.25, 0.73, 0.37, 0.78, 0.45, 0.66)
    }
    Phi1 <- matrix(1.00, 1L, 1L)
  } else {
    ov <- paste0("x", 1:9)
    model <- paste(
      "f1 =~ x1 + x2 + x3",
      "f2 =~ x4 + x5 + x6",
      "f3 =~ x7 + x8 + x9",
      "f1 ~~ f2",
      "f1 ~~ f3",
      "f2 ~~ f3",
      sep = "\n"
    )
    Lambda1 <- matrix(0, 9L, 3L)
    Lambda1[1:3, 1] <- c(1.00, 0.82, 0.74)
    Lambda1[4:6, 2] <- c(1.00, 0.78, 0.69)
    Lambda1[7:9, 3] <- c(1.00, 0.86, 0.72)
    Theta1 <- c(0.42, 0.48, 0.56, 0.46, 0.52, 0.60, 0.40, 0.50, 0.58)
    loading_rows <- data.frame(
      lhs = c("f1", "f1", "f2", "f2", "f3", "f3"),
      rhs = c("x2", "x3", "x5", "x6", "x8", "x9"),
      stringsAsFactors = FALSE
    )
    Phi1 <- make_phi(c(1.00, 1.10, 0.95),
                     matrix(c(1, .30, .25, .30, 1, .35, .25, .35, 1), 3L, 3L))
    if (identical(heterogeneity, "strong")) {
      Phi2 <- make_phi(c(2.40, 0.80, 2.80),
                       matrix(c(1, .45, .40, .45, 1, .30, .40, .30, 1), 3L, 3L))
      Theta2 <- c(0.14, 0.95, 0.22, 0.18, 1.05, 0.30, 0.12, 0.92, 0.24)
    } else {
      Phi2 <- make_phi(c(1.65, 1.30, 1.90),
                       matrix(c(1, .20, .35, .20, 1, .18, .35, .18, 1), 3L, 3L))
      Theta2 <- c(0.25, 0.73, 0.37, 0.35, 0.82, 0.46, 0.20, 0.76, 0.43)
    }
  }

  Lambda2 <- Lambda1
  if (identical(scenario, "loading_violation")) Lambda2[3L, 1L] <- Lambda2[3L, 1L] + 0.28
  if (identical(scenario, "exchangeable")) {
    Phi2 <- Phi1
    Theta2 <- Theta1
  }
  S1 <- Lambda1 %*% Phi1 %*% t(Lambda1) + diag(Theta1)
  S2 <- Lambda2 %*% Phi2 %*% t(Lambda2) + diag(Theta2)
  if (min(eigen(S1, symmetric = TRUE, only.values = TRUE)$values) <= 1e-8 ||
      min(eigen(S2, symmetric = TRUE, only.values = TRUE)$values) <= 1e-8) {
    stop("population covariance is not positive definite", call. = FALSE)
  }
  dimnames(S1) <- dimnames(S2) <- list(ov, ov)
  list(name = model_name, heterogeneity = heterogeneity, model = model,
       ov = ov, loading_rows = loading_rows, S1 = S1, S2 = S2)
}

draw_group <- function(n, Sigma, distribution, seed) {
  set.seed(seed)
  Z <- matrix(stats::rnorm(n * ncol(Sigma)), nrow = n) %*% chol(Sigma)
  if (identical(distribution, "t5")) {
    Z <- Z * sqrt(3 / stats::rchisq(n, df = 5))
  }
  colnames(Z) <- colnames(Sigma)
  Z
}

stack_groups <- function(X1, X2) {
  out <- as.data.frame(rbind(X1, X2))
  out$group <- factor(rep(c("g1", "g2"), c(nrow(X1), nrow(X2))),
                      levels = c("g1", "g2"))
  out
}

fit_cfa <- function(X1, X2, population, metric = FALSE) {
  magmaan::magmaan(
    population$model, stack_groups(X1, X2), estimator = "ML", groups = "group",
    group_equal = if (metric) "loadings" else NULL,
    control = list(max_iter = 1000L, ftol = 1e-10, gtol = 1e-7)
  )
}

metric_contrast <- function(fit, population) {
  pt <- fit$partable
  npar <- length(fit$theta)
  R <- matrix(0, nrow = nrow(population$loading_rows), ncol = npar)
  for (j in seq_len(nrow(population$loading_rows))) {
    lhs <- population$loading_rows$lhs[[j]]
    rhs <- population$loading_rows$rhs[[j]]
    i1 <- which(pt$group == 1L & pt$lhs == lhs & pt$op == "=~" &
                  pt$rhs == rhs & pt$free > 0L)
    i2 <- which(pt$group == 2L & pt$lhs == lhs & pt$op == "=~" &
                  pt$rhs == rhs & pt$free > 0L)
    if (length(i1) != 1L || length(i2) != 1L) {
      stop("could not locate free group-specific loading contrast for ", rhs,
           call. = FALSE)
    }
    R[j, pt$free[[i2]]] <- 1
    R[j, pt$free[[i1]]] <- -1
  }
  R
}

fit_statistics <- function(X1, X2, population, include_lrt = FALSE) {
  fit <- tryCatch(fit_cfa(X1, X2, population), error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  R <- tryCatch(metric_contrast(fit, population), error = function(e) NULL)
  if (is.null(R)) return(NULL)
  model_vcov <- tryCatch({
    info <- core$inference_information_expected(fit)
    core$inference_vcov_fit(info, fit)
  }, error = function(e) NULL)
  sandwich_vcov <- tryCatch(
    core$infer_robust_se_raw_fit(fit, list(X1, X2), bread = "expected",
                                  moments = "structured", cov = "empirical")$vcov,
    error = function(e) NULL
  )
  if (is.null(model_vcov) || is.null(sandwich_vcov)) return(NULL)
  w_model <- tryCatch(core$infer_wald_test_fit(fit, R, model_vcov)$chi2,
                      error = function(e) NA_real_)
  w_sandwich <- tryCatch(core$infer_wald_test_fit(fit, R, sandwich_vcov)$chi2,
                         error = function(e) NA_real_)
  d <- drop(R %*% fit$theta)
  out <- list(
    fit = fit, w_model = w_model, w_sandwich = w_sandwich,
    raw = length(c(X1, X2)) * sum(d^2), df = nrow(R)
  )
  if (!include_lrt) return(out)

  metric <- tryCatch(fit_cfa(X1, X2, population, metric = TRUE), error = function(e) NULL)
  if (is.null(metric)) return(out)
  n_total <- nrow(X1) + nrow(X2)
  t_diff <- 2 * n_total * (metric$fmin - fit$fmin)
  out$lrt_p <- stats::pchisq(t_diff, df = out$df, lower.tail = FALSE)
  robust <- tryCatch(
    magmaan::robust_nested_lrt(fit, metric, data = list(X1, X2),
                               gamma = "empirical", method = "restriction_map"),
    error = function(e) NULL
  )
  out$robust_lrt_p <- if (is.null(robust)) NA_real_ else robust$p_scaled
  out
}

permutation_pvalues <- function(X1, X2, population, observed, permutations, seed) {
  n1 <- nrow(X1)
  pooled <- rbind(X1, X2)
  set.seed(seed)
  permuted <- replicate(permutations, sample.int(nrow(pooled)), simplify = FALSE)
  draws <- lapply(permuted, function(index) {
    fit_statistics(pooled[index[seq_len(n1)], , drop = FALSE],
                   pooled[index[-seq_len(n1)], , drop = FALSE], population)
  })
  w <- vapply(draws, function(x) if (is.null(x)) NA_real_ else x$w_sandwich, numeric(1))
  raw <- vapply(draws, function(x) if (is.null(x)) NA_real_ else x$raw, numeric(1))
  valid_w <- w[is.finite(w)]
  valid_raw <- raw[is.finite(raw)]
  list(
    p_studentized = if (length(valid_w)) (1 + sum(valid_w >= observed$w_sandwich)) / (1 + length(valid_w)) else NA_real_,
    p_raw = if (length(valid_raw)) (1 + sum(valid_raw >= observed$raw)) / (1 + length(valid_raw)) else NA_real_,
    n_studentized = length(valid_w), n_raw = length(valid_raw)
  )
}

replication_record <- function(rep_id, converged = FALSE) {
  data.frame(
    rep = rep_id, converged = converged,
    p_lrt = NA_real_, p_robust_lrt = NA_real_,
    p_model_wald = NA_real_, p_sandwich_wald = NA_real_,
    p_permutation_studentized = NA_real_, p_permutation_raw = NA_real_,
    w_model = NA_real_, w_sandwich = NA_real_,
    n_perm_studentized = NA_real_, n_perm_raw = NA_real_,
    stringsAsFactors = FALSE
  )
}

one_replication <- function(rep_id, population, n, distribution, seed) {
  X1 <- draw_group(n[[1]], population$S1, distribution, seed + 11L)
  X2 <- draw_group(n[[2]], population$S2, distribution, seed + 29L)
  observed <- fit_statistics(X1, X2, population, include_lrt = TRUE)
  if (is.null(observed)) {
    return(replication_record(rep_id))
  }
  perm <- permutation_pvalues(X1, X2, population, observed, opts$permutations, seed + 101L)
  out <- replication_record(rep_id, converged = TRUE)
  scalar_or_na <- function(x) if (length(x) == 1L) as.numeric(x) else NA_real_
  out$p_lrt <- scalar_or_na(observed$lrt_p)
  out$p_robust_lrt <- scalar_or_na(observed$robust_lrt_p)
  out$p_model_wald <- stats::pchisq(observed$w_model, df = observed$df, lower.tail = FALSE)
  out$p_sandwich_wald <- stats::pchisq(observed$w_sandwich, df = observed$df, lower.tail = FALSE)
  out$p_permutation_studentized <- perm$p_studentized
  out$p_permutation_raw <- perm$p_raw
  out$w_model <- observed$w_model
  out$w_sandwich <- observed$w_sandwich
  out$n_perm_studentized <- perm$n_studentized
  out$n_perm_raw <- perm$n_raw
  out
}

design <- do.call(rbind, lapply(opts$sizes, function(n) {
  expand.grid(model = opts$models, heterogeneity = opts$heterogeneity,
              distribution = opts$distributions, scenario = opts$scenarios,
              n1 = n[[1]], n2 = n[[2]], stringsAsFactors = FALSE)
}))
# Heterogeneity has no meaning when the two populations are identical, so keep
# one exact-exchangeability control per model/distribution/sample-size cell.
if ("exchangeable" %in% opts$scenarios && length(opts$heterogeneity) > 1L) {
  design <- design[design$scenario != "exchangeable" |
                     design$heterogeneity == opts$heterogeneity[[1L]], , drop = FALSE]
}
design <- design[order(design$model, design$heterogeneity, design$distribution,
                       design$scenario, design$n1, design$n2), ]
message(sprintf("Grid: %d cells; reps=%d, permutations=%d, cores=%d",
                nrow(design), opts$reps, opts$permutations, opts$cores))

all_results <- vector("list", nrow(design))
t0 <- proc.time()[["elapsed"]]
for (cell in seq_len(nrow(design))) {
  d <- design[cell, ]
  population <- population_spec(d$model, d$heterogeneity, d$scenario)
  seed_cell <- opts$seed_base + cell * 1000003L
  cell_t0 <- proc.time()[["elapsed"]]
  chunks <- split(seq_len(opts$reps),
                  ceiling(seq_len(opts$reps) / min(50L, opts$reps)))
  chunk_rows <- vector("list", length(chunks))
  for (chunk_id in seq_along(chunks)) {
    rep_ids <- chunks[[chunk_id]]
    rows <- parallel::mclapply(rep_ids, function(rep_id) {
      one_replication(rep_id, population, c(d$n1, d$n2), d$distribution,
                      seed_cell + rep_id * 1009L)
    }, mc.cores = opts$cores, mc.preschedule = TRUE)
    chunk_rows[[chunk_id]] <- do.call(rbind, rows)
    done <- sum(vapply(chunks[seq_len(chunk_id)], length, integer(1)))
    cell_elapsed <- proc.time()[["elapsed"]] - cell_t0
    cell_eta <- cell_elapsed / done * (opts$reps - done)
    message(sprintf("    cell %d/%d: %d/%d reps (%.1fs elapsed, %.1fs cell ETA)",
                    cell, nrow(design), done, opts$reps, cell_elapsed, cell_eta))
  }
  out <- do.call(rbind, chunk_rows)
  out$model <- d$model
  out$heterogeneity <- d$heterogeneity
  out$distribution <- d$distribution
  out$scenario <- d$scenario
  out$n1 <- d$n1
  out$n2 <- d$n2
  all_results[[cell]] <- out
  elapsed <- proc.time()[["elapsed"]] - t0
  eta <- elapsed / cell * (nrow(design) - cell)
  message(sprintf("  [%d/%d] %s/%s/%s/%s n=(%d,%d): %.1fs elapsed, %.1fs ETA",
                  cell, nrow(design), d$model, d$heterogeneity,
                  d$distribution, d$scenario, d$n1, d$n2,
                  elapsed, eta))
}
replicate_results <- do.call(rbind, all_results)

p_columns <- grep("^p_", names(replicate_results), value = TRUE)
summary_rows <- lapply(split(replicate_results,
                             interaction(replicate_results$model,
                                         replicate_results$heterogeneity,
                                         replicate_results$distribution,
                                         replicate_results$scenario,
                                         replicate_results$n1,
                                         replicate_results$n2, drop = TRUE)), function(x) {
  ans <- x[1L, c("model", "heterogeneity", "distribution", "scenario", "n1", "n2"), drop = FALSE]
  ans$reps <- nrow(x)
  ans$converged <- sum(x$converged)
  for (p_name in p_columns) {
    usable <- x[[p_name]][is.finite(x[[p_name]])]
    suffix <- sub("^p_", "", p_name)
    ans[[paste0("reject_", suffix)]] <- if (length(usable)) mean(usable < opts$alpha) else NA_real_
    ans[[paste0("n_", suffix)]] <- length(usable)
  }
  ans$median_valid_permutations <- stats::median(x$n_perm_studentized, na.rm = TRUE)
  ans
})
summary_results <- do.call(rbind, summary_rows)

raw_dir <- file.path(results_path, "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
raw_path <- file.path(raw_dir, "replicate_results.csv")
summary_path <- file.path(results_path, "summary.csv")
write_csv(replicate_results, raw_path)
write_csv(summary_results, summary_path)
write_metadata(
  file.path(results_path, "metadata.csv"),
  values = list(
    experiment = "61-permutation-measurement-invariance", mode = opts$mode,
    reps = opts$reps, permutations = opts$permutations,
    group_sizes = vapply(opts$sizes, function(x) paste(x, collapse = ":"), character(1)),
    models = opts$models, heterogeneity = opts$heterogeneity,
    distributions = opts$distributions, scenarios = opts$scenarios,
    alpha = opts$alpha, seed_base = opts$seed_base, cores = opts$cores
  ), packages = "magmaan"
)

message("Wrote:\n  ", raw_path, "\n  ", summary_path, "\n  ",
        file.path(results_path, "metadata.csv"))
print(summary_results, row.names = FALSE)
