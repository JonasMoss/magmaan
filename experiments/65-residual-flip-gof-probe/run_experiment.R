#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "design.R"))
source(file.path(script_dir, "R", "gof_flip.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--probe] [options]\n\n",
  "Residual-score sign-flip GOF probe.\n\n",
  "  --smoke          16 cells, 2 reps, 19 flips (default).\n",
  "  --probe          16 cells, 200 reps, 199 flips.\n",
  "  --reps N         Override replications per cell.\n",
  "  --flips N        Override random flips per replication.\n",
  "  --cores N        Parallel DGP workers.\n",
  "  --seed-base N    Deterministic seed base.\n",
  "  --results-dir P  Output directory.\n",
  "  --help           Show this help.\n", sep = "")

opts <- list(mode = "smoke", reps = NULL, flips = NULL,
             cores = min(4L, max(1L, parallel::detectCores() - 2L)),
             seed_base = 20260713L, results_dir = NULL)
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
if (is.null(opts$reps)) opts$reps <- if (opts$mode == "smoke") 2L else 200L
if (is.null(opts$flips)) opts$flips <- if (opts$mode == "smoke") 19L else 199L
stopifnot(opts$reps > 0L, opts$flips > 0L, opts$cores > 0L,
          opts$seed_base >= 0L)
results_dir <- opts$results_dir %||% file.path(script_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

gof_validate_design()
grid <- gof_design_grid()
fmg_tests_requested <- c(
  "std_ml", "sb_ml", "mv_ml", "ss_ml", "peba4_ml", "all_ml",
  "std_rls", "sb_rls", "mv_rls", "ss_rls", "peba4_rls", "all_rls")
p_columns <- c(
  "p_flip_effective", "p_flip_standardized", "p_flip_standardized_chisq",
  paste0("p_gof_", fmg_tests_requested))
empty_p <- function() stats::setNames(rep(NA_real_, length(p_columns)), p_columns)

one_rep <- function(cell, rep_id, sampler, spec) {
  total_begin <- proc.time()[["elapsed"]]
  draw_begin <- proc.time()[["elapsed"]]
  X <- sampler$draw(rep_id, cell$n)
  draw_seconds <- proc.time()[["elapsed"]] - draw_begin
  p <- empty_p()
  base <- data.frame(
    rep = rep_id, fit_ok = FALSE, flip_ok = FALSE, fmg_ok = FALSE,
    fit_error = "", flip_error = "", fmg_error = "",
    draw_seconds = draw_seconds, fit_seconds = NA_real_, flip_seconds = NA_real_,
    fmg_seconds = NA_real_, total_seconds = NA_real_,
    flip_setup_seconds = NA_real_, flip_score_seconds = NA_real_,
    standardization_setup_seconds = NA_real_,
    flip_standardization_seconds = NA_real_,
    standardization_available = NA, empirical_eigen_min = NA_real_,
    empirical_eigen_mean = NA_real_, empirical_eigen_cv = NA_real_,
    empirical_eigen_ratio = NA_real_, empirical_identity_distance = NA_real_,
    rls_statistic = NA_real_, flip_rls_abs_difference = NA_real_,
    stringsAsFactors = FALSE)

  fit_begin <- proc.time()[["elapsed"]]
  fit <- tryCatch(magmaan(
    spec, as.data.frame(X), estimator = "ML",
    optimizer = "nlopt-lbfgs-slsqp-fallback", se = "none", test = "none"),
    error = function(e) e)
  base$fit_seconds <- proc.time()[["elapsed"]] - fit_begin
  if (inherits(fit, "error")) {
    base$fit_error <- conditionMessage(fit)
    base$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(cbind(base, as.data.frame(as.list(p))))
  }
  if (!isTRUE(fit$converged)) {
    base$fit_error <- "ML fit did not converge"
    base$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(cbind(base, as.data.frame(as.list(p))))
  }
  base$fit_ok <- TRUE

  flip_begin <- proc.time()[["elapsed"]]
  flip <- tryCatch(residual_flip_gof(
    fit, X, n_flips = opts$flips,
    seed = opts$seed_base + cell$cell_id * 100000L + rep_id),
    error = function(e) e)
  base$flip_seconds <- proc.time()[["elapsed"]] - flip_begin
  if (inherits(flip, "error")) {
    base$flip_error <- conditionMessage(flip)
  } else {
    stopifnot(flip$df == cell$df)
    base$flip_ok <- TRUE
    p[c("p_flip_effective", "p_flip_standardized",
        "p_flip_standardized_chisq")] <-
      c(flip$p_effective, flip$p_standardized, flip$p_standardized_chisq)
    base$flip_setup_seconds <- flip$setup_seconds
    base$flip_score_seconds <- flip$resampling_score_seconds
    base$standardization_setup_seconds <- flip$standardization_setup_seconds
    base$flip_standardization_seconds <-
      flip$resampling_standardization_seconds
    base$standardization_available <- flip$standardization_available
    base$empirical_eigen_min <- flip$empirical_eigen_min
    base$empirical_eigen_mean <- flip$empirical_eigen_mean
    base$empirical_eigen_cv <- flip$empirical_eigen_cv
    base$empirical_eigen_ratio <- flip$empirical_eigen_ratio
    base$empirical_identity_distance <- flip$empirical_identity_distance
  }

  fmg_begin <- proc.time()[["elapsed"]]
  fmg <- tryCatch(fmg_tests(fit, tests = fmg_tests_requested, data = X),
                  error = function(e) e)
  base$fmg_seconds <- proc.time()[["elapsed"]] - fmg_begin
  if (inherits(fmg, "error")) {
    base$fmg_error <- conditionMessage(fmg)
  } else {
    base$fmg_ok <- TRUE
    p[paste0("p_gof_", fmg$label)] <- fmg$p_value
    rls_row <- which(fmg$label == "std_rls")
    if (length(rls_row) == 1L) {
      base$rls_statistic <- fmg$base_statistic[[rls_row]]
      if (!inherits(flip, "error")) {
        base$flip_rls_abs_difference <-
          abs(flip$statistic_effective - base$rls_statistic)
        if (base$flip_rls_abs_difference >
            1e-8 * max(1, abs(base$rls_statistic))) {
          stop("projected flip statistic failed the RLS identity", call. = FALSE)
        }
      }
    }
  }
  base$total_seconds <- proc.time()[["elapsed"]] - total_begin
  cbind(base, as.data.frame(as.list(p)))
}

run_dgp <- function(k) {
  key <- dgp_grid[k, , drop = FALSE]
  pop <- gof_population(key$p, key$truth)
  dist_id <- match(key$distribution, c("normal", "pl"))
  sampler <- tryCatch(gof_make_sampler(
    pop, n_max = max(grid$n), reps = opts$reps,
    distribution = key$distribution,
    seed_base = opts$seed_base + key$p * 10000L + dist_id * 100L),
    error = function(e) e)
  cells <- grid[grid$p == key$p & grid$distribution == key$distribution &
                  grid$truth == key$truth, , drop = FALSE]
  if (inherits(sampler, "error")) {
    rows <- lapply(seq_len(nrow(cells)), function(j) {
      cell <- cells[j, ]
      do.call(rbind, lapply(seq_len(opts$reps), function(rep_id) {
        p <- empty_p()
        base <- data.frame(
          rep = rep_id, fit_ok = FALSE, flip_ok = FALSE, fmg_ok = FALSE,
          fit_error = paste0("generator: ", conditionMessage(sampler)),
          flip_error = "", fmg_error = "", draw_seconds = NA_real_,
          fit_seconds = NA_real_, flip_seconds = NA_real_, fmg_seconds = NA_real_,
          total_seconds = NA_real_, flip_setup_seconds = NA_real_,
          flip_score_seconds = NA_real_, standardization_setup_seconds = NA_real_,
          flip_standardization_seconds = NA_real_,
          standardization_available = NA, empirical_eigen_min = NA_real_,
          empirical_eigen_mean = NA_real_, empirical_eigen_cv = NA_real_,
          empirical_eigen_ratio = NA_real_, empirical_identity_distance = NA_real_,
          rls_statistic = NA_real_, flip_rls_abs_difference = NA_real_)
        out <- cbind(base, as.data.frame(as.list(p)))
        for (name in names(cell)) out[[name]] <- cell[[name]]
        out
      }))
    })
    timing <- data.frame(key, setup_seconds = NA_real_, error = conditionMessage(sampler))
  } else {
    spec <- gof_model_spec(key$p)
    rows <- lapply(seq_len(nrow(cells)), function(j) {
      cell <- cells[j, ]
      out <- do.call(rbind, lapply(seq_len(opts$reps), function(rep_id) {
        tryCatch(one_rep(cell, rep_id, sampler, spec), error = function(e) {
          p <- empty_p()
          base <- data.frame(
            rep = rep_id, fit_ok = FALSE, flip_ok = FALSE, fmg_ok = FALSE,
            fit_error = conditionMessage(e), flip_error = "", fmg_error = "",
            draw_seconds = NA_real_, fit_seconds = NA_real_, flip_seconds = NA_real_,
            fmg_seconds = NA_real_, total_seconds = NA_real_,
            flip_setup_seconds = NA_real_, flip_score_seconds = NA_real_,
            standardization_setup_seconds = NA_real_,
            flip_standardization_seconds = NA_real_,
            standardization_available = NA, empirical_eigen_min = NA_real_,
            empirical_eigen_mean = NA_real_, empirical_eigen_cv = NA_real_,
            empirical_eigen_ratio = NA_real_, empirical_identity_distance = NA_real_,
            rls_statistic = NA_real_, flip_rls_abs_difference = NA_real_)
          cbind(base, as.data.frame(as.list(p)))
        })
      }))
      for (name in names(cell)) out[[name]] <- cell[[name]]
      out
    })
    timing <- data.frame(key, setup_seconds = sampler$setup_seconds, error = "")
  }
  list(rows = do.call(rbind, rows), timing = timing)
}

wilson <- function(x, n, z = 1.95996398454005) {
  if (!is.finite(n) || n <= 0L) return(c(lower = NA_real_, upper = NA_real_))
  phat <- x / n; den <- 1 + z^2 / n
  centre <- (phat + z^2 / (2 * n)) / den
  half <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) / den
  c(lower = max(0, centre - half), upper = min(1, centre + half))
}

cat(sprintf("mode=%s cells=%d reps=%d flips=%d cores=%d\n",
            opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores))
dgp_grid <- unique(grid[c("p", "distribution", "truth")])
wall_begin <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  pieces <- parallel::mclapply(
    seq_len(nrow(dgp_grid)), run_dgp,
    mc.cores = min(opts$cores, nrow(dgp_grid)), mc.preschedule = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(dgp_grid)), function(k) {
    cat(sprintf("  DGP %d/%d\n", k, nrow(dgp_grid))); flush.console()
    run_dgp(k)
  })
}
raw <- do.call(rbind, lapply(pieces, `[[`, "rows"))
dgp_timing <- do.call(rbind, lapply(pieces, `[[`, "timing"))
raw <- raw[order(raw$cell_id, raw$rep), ]
row.names(raw) <- NULL
write.csv(raw, file.path(results_dir, "replications.csv"), row.names = FALSE)
write.csv(grid, file.path(results_dir, "design.csv"), row.names = FALSE)
write.csv(dgp_timing, file.path(results_dir, "dgp_timing.csv"), row.names = FALSE)

method_rows <- do.call(rbind, lapply(p_columns, function(column) {
  data.frame(raw[c("cell_id", "p", "n", "df", "n_over_df", "distribution",
                   "truth", "rep")],
             method = sub("^p_", "", column), p_value = raw[[column]],
             stringsAsFactors = FALSE)
}))
method_split <- split(method_rows, interaction(method_rows$cell_id,
                                                method_rows$method, drop = TRUE))
method_summary <- do.call(rbind, lapply(method_split, function(x) {
  valid <- is.finite(x$p_value); n <- sum(valid)
  rejected <- sum(x$p_value[valid] <= .05)
  ci <- wilson(rejected, n)
  data.frame(x[1L, c("cell_id", "p", "n", "df", "n_over_df",
                     "distribution", "truth", "method")],
             n_valid = n, rejected = rejected,
             rejection_rate = if (n) rejected / n else NA_real_,
             ci_lower = ci[[1L]], ci_upper = ci[[2L]])
}))
method_summary <- method_summary[order(method_summary$cell_id,
                                       method_summary$method), ]
row.names(method_summary) <- NULL
write.csv(method_summary, file.path(results_dir, "method_summary.csv"),
          row.names = FALSE)

cell_split <- split(raw, raw$cell_id)
paired_summary <- do.call(rbind, lapply(cell_split, function(x) {
  keep <- is.finite(x$p_flip_effective) & is.finite(x$p_flip_standardized)
  y <- x[keep, , drop = FALSE]
  data.frame(x[1L, c("cell_id", "p", "n", "df", "n_over_df",
                     "distribution", "truth")],
             n_paired = nrow(y),
             standardization_availability = mean(x$standardization_available,
                                                 na.rm = TRUE),
             rejection_effective = if (nrow(y))
               mean(y$p_flip_effective <= .05) else NA_real_,
             rejection_standardized = if (nrow(y))
               mean(y$p_flip_standardized <= .05) else NA_real_,
             disagreement_rate = if (nrow(y)) mean(
               (y$p_flip_effective <= .05) !=
                 (y$p_flip_standardized <= .05)) else NA_real_,
             mean_abs_delta_p = if (nrow(y)) mean(abs(
               y$p_flip_standardized - y$p_flip_effective)) else NA_real_,
             median_eigen_ratio = stats::median(x$empirical_eigen_ratio,
                                                na.rm = TRUE),
             median_identity_distance = stats::median(
               x$empirical_identity_distance, na.rm = TRUE))
}))
row.names(paired_summary) <- NULL
write.csv(paired_summary, file.path(results_dir, "paired_summary.csv"),
          row.names = FALSE)

power_rows <- method_rows[method_rows$truth == "power" &
                            is.finite(method_rows$p_value), , drop = FALSE]
matched_split <- split(power_rows, interaction(
  power_rows$p, power_rows$n, power_rows$distribution,
  power_rows$method, drop = TRUE))
matched_power <- do.call(rbind, lapply(matched_split, function(x) {
  null <- method_rows[
    method_rows$truth == "null" & method_rows$p == x$p[[1L]] &
      method_rows$n == x$n[[1L]] &
      method_rows$distribution == x$distribution[[1L]] &
      method_rows$method == x$method[[1L]] & is.finite(method_rows$p_value),
    "p_value"]
  empirical_p <- vapply(x$p_value, function(pv)
    (1 + sum(null <= pv)) / (length(null) + 1), numeric(1L))
  rejected <- sum(empirical_p <= .05); n <- length(empirical_p)
  ci <- wilson(rejected, n)
  data.frame(x[1L, c("p", "n", "df", "n_over_df", "distribution", "method")],
             n_null = length(null), n_power = n,
             nominal_power = mean(x$p_value <= .05),
             matched_power = if (n) rejected / n else NA_real_,
             ci_lower = ci[[1L]], ci_upper = ci[[2L]])
}))
row.names(matched_power) <- NULL
write.csv(matched_power, file.path(results_dir, "matched_power.csv"),
          row.names = FALSE)

timing_columns <- c(
  draw = "draw_seconds", fit = "fit_seconds", flip_total = "flip_seconds",
  flip_setup = "flip_setup_seconds", flip_score = "flip_score_seconds",
  standardization_setup = "standardization_setup_seconds",
  flip_standardization = "flip_standardization_seconds",
  fmg_battery = "fmg_seconds", replication_total = "total_seconds")
timing_summary <- do.call(rbind, lapply(cell_split, function(x) {
  do.call(rbind, lapply(names(timing_columns), function(phase) {
    values <- x[[timing_columns[[phase]]]]
    values <- values[is.finite(values)]
    data.frame(x[1L, c("cell_id", "p", "n", "df", "n_over_df",
                       "distribution", "truth")], phase = phase,
               n_timed = length(values),
               median_seconds = if (length(values)) stats::median(values) else NA_real_,
               p90_seconds = if (length(values)) unname(stats::quantile(
                 values, .9, type = 8)) else NA_real_)
  }))
}))
row.names(timing_summary) <- NULL
write.csv(timing_summary, file.path(results_dir, "timing_summary.csv"),
          row.names = FALSE)

failures <- raw[!raw$fit_ok | !raw$flip_ok | !raw$fmg_ok,
                c("cell_id", "rep", "p", "n", "distribution", "truth",
                  "fit_ok", "flip_ok", "fmg_ok", "fit_error", "flip_error",
                  "fmg_error")]
write.csv(failures, file.path(results_dir, "failures.csv"), row.names = FALSE)
methods <- data.frame(
  method = sub("^p_", "", p_columns),
  family = c(rep("flip", 3L), rep("FMG-ML", 6L), rep("FMG-RLS", 6L)),
  stringsAsFactors = FALSE)
write.csv(methods, file.path(results_dir, "methods.csv"), row.names = FALSE)

elapsed <- proc.time()[["elapsed"]] - wall_begin
metadata <- data.frame(
  key = c("mode", "cells", "reps", "flips", "cores", "seed_base",
          "elapsed_seconds", "magmaan_version", "R_version", "dgp_source",
          "alternative"),
  value = c(opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores,
            opts$seed_base, elapsed, as.character(packageVersion("magmaan")),
            R.version.string, "Foldnes-Moss-Gronneberg marginal targets",
            "one omitted residual covariance rho=.25"))
write.csv(metadata, file.path(results_dir, "metadata.csv"), row.names = FALSE)
cat(sprintf("wrote results to %s (%.1fs, %d failures)\n",
            results_dir, elapsed, nrow(failures)))
