#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--probe] [options]\n\n",
  "Small-N calibration of basic, effective, and standardized score flips\n",
  "for the configural-to-metric continuous ML invariance test.\n\n",
  "Options:\n",
  "  --smoke          One cell, 4 reps, 39 flips (default).\n",
  "  --probe          Full 12-null-cell grid plus one power cell.\n",
  "  --reps N         Replications per cell (probe default: 300).\n",
  "  --flips N        Random flips per replication (probe default: 499).\n",
  "  --cores N        Parallel cell workers.\n",
  "  --seed-base N    Deterministic seed base.\n",
  "  --results-dir P  Output directory.\n",
  "  --help           Show this help.\n", sep = "")

opts <- list(mode = "smoke", reps = NULL, flips = NULL,
             cores = max(1L, parallel::detectCores() - 2L),
             seed_base = 20260712L, results_dir = NULL)
args <- commandArgs(TRUE); i <- 1L
take <- function() { i <<- i + 1L; args[[i]] }
while (i <= length(args)) {
  a <- args[[i]]
  if (a == "--help") { usage(); quit(status = 0L) }
  else if (a == "--smoke") opts$mode <- "smoke"
  else if (a == "--probe") opts$mode <- "probe"
  else if (a == "--reps") opts$reps <- as.integer(take())
  else if (a == "--flips") opts$flips <- as.integer(take())
  else if (a == "--cores") opts$cores <- as.integer(take())
  else if (a == "--seed-base") opts$seed_base <- as.integer(take())
  else if (a == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
if (is.null(opts$reps)) opts$reps <- if (opts$mode == "smoke") 4L else 300L
if (is.null(opts$flips)) opts$flips <- if (opts$mode == "smoke") 39L else 499L
stopifnot(opts$reps > 0L, opts$flips > 0L, opts$cores > 0L)
results_dir <- opts$results_dir %||% file.path(script_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

ov <- paste0("x", 1:6)
model <- paste(
  "f1 =~ x1 + x2 + x3",
  "f2 =~ x4 + x5 + x6",
  "f1 ~~ f2", sep = "\n")
specs <- list(
  configural = model_spec(model, meanstructure = TRUE, group = "school",
                          group_labels = c("A", "B")),
  metric = model_spec(model, meanstructure = TRUE, group = "school",
                      group_labels = c("A", "B"), group_equal = "loadings"))
Lambda <- matrix(0, 6, 2)
Lambda[1:3, 1] <- c(1, .8, 1.2)
Lambda[4:6, 2] <- c(1, .7, 1.3)
Theta <- diag(c(.55, .65, .50, .60, .70, .45))
nu <- c(.2, .5, .8, 1.1, 1.4, 1.7)
alpha2 <- c(.4, -.25)

population <- function(phi_ratio, alternative) {
  L1 <- Lambda; L2 <- Lambda
  if (alternative) L2[2, 1] <- L2[2, 1] + .25
  Phi1 <- matrix(c(1, .3, .3, 1), 2, 2)
  Phi2 <- phi_ratio * Phi1
  list(
    mu = list(nu, nu + as.vector(L2 %*% alpha2)),
    Sigma = list(L1 %*% Phi1 %*% t(L1) + Theta,
                 L2 %*% Phi2 %*% t(L2) + Theta))
}

draw_block <- function(n, mu, Sigma, distribution) {
  Z <- matrix(rnorm(n * length(mu)), n) %*% chol(Sigma)
  if (distribution == "t5") {
    Z <- Z * sqrt(3 / 5) / sqrt(rchisq(n, 5) / 5)
  }
  sweep(Z, 2, mu, "+")
}

draw_data <- function(cell, rep_id) {
  set.seed(opts$seed_base + cell$cell_id * 100000L + rep_id)
  pop <- population(cell$phi_ratio, cell$alternative)
  blocks <- lapply(1:2, function(g)
    draw_block(cell$n, pop$mu[[g]], pop$Sigma[[g]], cell$distribution))
  blocks <- lapply(blocks, function(x) { colnames(x) <- ov; x })
  data.frame(rbind(blocks[[1]], blocks[[2]]),
             school = rep(c("A", "B"), each = cell$n),
             check.names = FALSE)
}

null_grid <- expand.grid(
  n = c(30L, 50L, 100L), distribution = c("normal", "t5"),
  phi_ratio = c(1, 3), stringsAsFactors = FALSE)
null_grid$alternative <- FALSE
power <- data.frame(n = 50L, distribution = "t5", phi_ratio = 3,
                    alternative = TRUE)
grid <- rbind(null_grid, power)
if (opts$mode == "smoke") grid <- grid[grid$n == 30 &
  grid$distribution == "normal" & grid$phi_ratio == 3 &
  !grid$alternative, , drop = FALSE]
grid$cell_id <- seq_len(nrow(grid))

failed_rep <- function(rep_id, error) data.frame(
  rep = rep_id, ok = FALSE, error = error, method = NA_character_,
  p_value = NA_real_, statistic = NA_real_, nuisance_norm = NA_real_,
  min_variance_eigenvalue = NA_real_, max_variance_condition = NA_real_,
  stringsAsFactors = FALSE)

one_rep <- function(cell, rep_id) {
  dat <- draw_data(cell, rep_id)
  fit <- tryCatch(lapply(specs, magmaan, data = dat, estimator = "ML"),
                  error = function(e) e)
  if (inherits(fit, "error")) {
    return(failed_rep(rep_id, conditionMessage(fit)))
  }
  flip <- tryCatch(score_flip_test(
    fit$configural, fit$metric, dat, n_flips = opts$flips,
    seed = opts$seed_base + cell$cell_id * 1000000L + rep_id),
    error = function(e) e)
  if (inherits(flip, "error")) {
    return(failed_rep(rep_id, conditionMessage(flip)))
  }
  raw_blocks <- lapply(c("A", "B"), function(g)
    as.matrix(dat[dat$school == g, ov, drop = FALSE]))
  nt <- tryCatch(nestedTest(fit$configural, fit$metric, data = raw_blocks,
                            method = "restriction_map", A.method = "exact"),
                 error = function(e) NULL)
  p <- c(
    flip_basic = flip$p_basic,
    flip_effective = flip$p_effective,
    flip_standardized = flip$p_standardized,
    score_chisq = flip$p_chisq,
    score_mean_scaled = flip$p_mean_scaled,
    score_mixture = flip$p_mixture,
    score_sandwich = flip$p_sandwich,
    satorra2000_scaled = if (is.null(nt)) NA_real_ else nt$p_scaled,
    satorra2000_mixture = if (is.null(nt)) NA_real_ else nt$p_mixture)
  data.frame(rep = rep_id, ok = TRUE, error = "", method = names(p),
             p_value = unname(p), statistic = flip$statistic_effective,
             nuisance_norm = flip$nuisance_stationarity_norm,
             min_variance_eigenvalue = flip$min_variance_eigenvalue,
             max_variance_condition = flip$max_variance_condition,
             stringsAsFactors = FALSE)
}

run_cell <- function(k) {
  cell <- as.list(grid[k, , drop = FALSE])
  rows <- lapply(seq_len(opts$reps), function(r) one_rep(cell, r))
  out <- do.call(rbind, rows)
  out$cell_id <- cell$cell_id
  out$n <- cell$n
  out$distribution <- cell$distribution
  out$phi_ratio <- cell$phi_ratio
  out$alternative <- cell$alternative
  out
}

cat(sprintf("mode=%s cells=%d reps=%d flips=%d cores=%d\n",
            opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores))
t0 <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L && nrow(grid) > 1L) {
  pieces <- parallel::mclapply(seq_len(nrow(grid)), run_cell,
                               mc.cores = min(opts$cores, nrow(grid)),
                               mc.preschedule = TRUE)
} else {
  pieces <- lapply(seq_len(nrow(grid)), function(k) {
    cat(sprintf("  cell %d/%d\n", k, nrow(grid))); flush.console()
    run_cell(k)
  })
}
raw <- do.call(rbind, pieces)
write.csv(raw, file.path(results_dir, "replications.csv"), row.names = FALSE)

ok <- raw[raw$ok & !is.na(raw$p_value), , drop = FALSE]
summary <- aggregate(p_value ~ cell_id + n + distribution + phi_ratio +
                       alternative + method, ok,
                     function(p) c(rejection = mean(p < .05), mean_p = mean(p),
                                   reps = length(p)))
vals <- if (is.list(summary$p_value)) {
  do.call(rbind, summary$p_value)
} else {
  as.matrix(summary$p_value)
}
summary$p_value <- NULL
summary$rejection <- vals[, "rejection"]
summary$mean_p <- vals[, "mean_p"]
summary$reps_ok <- as.integer(vals[, "reps"])
write.csv(summary, file.path(results_dir, "summary.csv"), row.names = FALSE)

rep_status <- unique(raw[, c("cell_id", "n", "distribution", "phi_ratio",
                             "alternative", "rep", "ok")])
fail <- aggregate(ok ~ cell_id + n + distribution + phi_ratio + alternative,
                  rep_status, function(x) mean(!x))
names(fail)[names(fail) == "ok"] <- "failure_rate"
write.csv(fail, file.path(results_dir, "failures.csv"), row.names = FALSE)

metadata <- data.frame(
  key = c("mode", "reps", "flips", "cores", "seed_base", "elapsed_seconds",
          "magmaan_version", "R_version"),
  value = c(opts$mode, opts$reps, opts$flips, opts$cores, opts$seed_base,
            proc.time()[["elapsed"]] - t0, as.character(packageVersion("magmaan")),
            R.version.string))
write.csv(metadata, file.path(results_dir, "metadata.csv"), row.names = FALSE)
cat(sprintf("wrote results to %s (%.1fs)\n", results_dir,
            proc.time()[["elapsed"]] - t0))
