#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))
suppressWarnings(suppressMessages(library(lavaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "pilot.R"))
set_single_threaded_math()

opts <- list(
  reps = 25L,
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  n = 120L, p = 5L, seed_base = 20260818L, results_dir = NULL
)
args <- commandArgs(TRUE)
i <- 1L
take <- function() {
  i <<- i + 1L
  if (i > length(args)) stop("missing option value", call. = FALSE)
  args[[i]]
}
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg == "--reps") opts$reps <- as.integer(take())
  else if (arg == "--cores") opts$cores <- as.integer(take())
  else if (arg == "--n") opts$n <- as.integer(take())
  else if (arg == "--p") opts$p <- as.integer(take())
  else if (arg == "--seed-base") opts$seed_base <- as.integer(take())
  else if (arg == "--results-dir") opts$results_dir <- take()
  else if (arg %in% c("-h", "--help")) {
    cat(paste(
      "Usage: Rscript validate_mlr_lavaan.R [options]",
      "  --reps N --cores N --n N --p P --seed-base N --results-dir P",
      sep = "\n"), "\n")
    quit(save = "no", status = 0L)
  } else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}
stopifnot(opts$reps > 0L, opts$cores > 0L, opts$n >= 40L,
          opts$p >= 3L, opts$p <= 8L)

results <- opts$results_dir %||% file.path(script_dir, "results", "pilot")
dir.create(results, recursive = TRUE, showWarnings = FALSE)
grid <- pilot_design(opts$n, opts$p)
grid <- grid[grid$truth == "null", , drop = FALSE]
specs <- pilot_specs(opts$p)

one_rep <- function(cell, rep_id) {
  seed <- opts$seed_base + cell$cell_id * 100003L + rep_id
  data <- pilot_draw(cell, seed)
  fit_magmaan <- magmaan::magmaan(
    specs$H0, data, estimator = "FIML", se = "none", test = "none",
    optimizer = "nlopt-lbfgs-slsqp-fallback",
    control = list(max_iter = 8000L, ftol = 1e-11, gtol = 1e-8))
  if (!isTRUE(fit_magmaan$converged)) {
    stop("magmaan FIML fit did not converge", call. = FALSE)
  }
  mag <- magmaan::magmaan_core$estimate_fiml_robust_mlr(fit_magmaan)

  missing <- if (cell$missing_rate > 0) "ml" else "listwise"
  fit_lavaan <- lavaan::sem(
    pilot_model_syntax(cell$p, restricted = TRUE), data = data,
    estimator = "MLR", missing = missing, meanstructure = TRUE,
    fixed.x = FALSE)
  if (!lavaan::lavInspect(fit_lavaan, "converged")) {
    stop("lavaan MLR fit did not converge", call. = FALSE)
  }
  lav_test <- lavaan::lavInspect(fit_lavaan, "test")$yuan.bentler.mplus
  if (is.null(lav_test)) stop("lavaan returned no yuan.bentler.mplus test",
                              call. = FALSE)
  lav_chisq <- as.numeric(lav_test$scaled.test.stat)
  lav_scaled <- as.numeric(lav_test$stat)
  lav_p <- as.numeric(lav_test$pvalue)
  lav_scale <- as.numeric(lav_test$scaling.factor)
  lav_trace <- as.numeric(lav_test$trace.UGamma)
  lav_h1 <- as.numeric(attr(lav_test$stat, "h1"))
  lav_h0 <- as.numeric(attr(lav_test$stat, "h0"))
  mag_p <- stats::pchisq(mag$chisq_scaled, mag$df, lower.tail = FALSE)

  data.frame(
    cell_id = cell$cell_id, distribution = cell$distribution,
    missing_rate = cell$missing_rate, rep = rep_id,
    magmaan_chisq = mag$chisq, lavaan_chisq = lav_chisq,
    diff_chisq = mag$chisq - lav_chisq,
    magmaan_scaled = mag$chisq_scaled, lavaan_scaled = lav_scaled,
    diff_scaled = mag$chisq_scaled - lav_scaled,
    magmaan_p = mag_p, lavaan_p = lav_p, diff_p = mag_p - lav_p,
    magmaan_scale = mag$scaling_factor, lavaan_scale = lav_scale,
    diff_scale = mag$scaling_factor - lav_scale,
    magmaan_trace = mag$trace_ugamma, lavaan_trace = lav_trace,
    diff_trace = mag$trace_ugamma - lav_trace,
    magmaan_h1 = mag$trace_ugamma_h1, lavaan_h1 = lav_h1,
    diff_h1 = mag$trace_ugamma_h1 - lav_h1,
    magmaan_h0 = mag$trace_ugamma_h0, lavaan_h0 = lav_h0,
    diff_h0 = mag$trace_ugamma_h0 - lav_h0,
    magmaan_df = mag$df, lavaan_df = as.integer(lav_test$df),
    decision_mismatch = (mag_p <= 0.05) != (lav_p <= 0.05),
    stringsAsFactors = FALSE)
}

run_cell <- function(index) {
  cell <- as.list(grid[index, , drop = FALSE])
  message(sprintf("  %s, missing %.0f%%", cell$distribution,
                  100 * cell$missing_rate))
  do.call(rbind, lapply(seq_len(opts$reps), function(rep_id) {
    one_rep(cell, rep_id)
  }))
}

wall_begin <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  pieces <- parallel::mclapply(
    seq_len(nrow(grid)), run_cell,
    mc.cores = min(opts$cores, nrow(grid)), mc.preschedule = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(grid)), run_cell)
}
if (any(vapply(pieces, inherits, logical(1L), what = "try-error"))) {
  stop("one or more parallel lavaan parity workers failed", call. = FALSE)
}
rows <- do.call(rbind, pieces)
row.names(rows) <- NULL
wall_seconds <- proc.time()[["elapsed"]] - wall_begin

metrics <- c("chisq", "scaled", "p", "scale", "trace", "h1", "h0")
summary <- do.call(rbind, lapply(metrics, function(metric) {
  difference <- rows[[paste0("diff_", metric)]]
  data.frame(
    metric = metric, n = sum(is.finite(difference)),
    max_abs_difference = max(abs(difference), na.rm = TRUE),
    mean_abs_difference = mean(abs(difference), na.rm = TRUE),
    stringsAsFactors = FALSE)
}))
summary$decision_mismatches <- sum(rows$decision_mismatch)
summary$df_mismatches <- sum(rows$magmaan_df != rows$lavaan_df)
summary$wall_seconds <- wall_seconds

write_csv(rows, file.path(results, "mlr_lavaan_parity.csv"))
write_csv(summary, file.path(results, "mlr_lavaan_parity_summary.csv"))
write_metadata(file.path(results, "mlr_lavaan_parity_metadata.csv"), list(
  reps_per_cell = opts$reps, cells = nrow(grid), n = opts$n, p = opts$p,
  seed_base = opts$seed_base, wall_seconds = wall_seconds
), packages = c("magmaan", "lavaan"))

print(summary, row.names = FALSE, digits = 4)
tolerance <- c(chisq = 1e-4, scaled = 5e-3, p = 1e-4, scale = 1e-3,
               trace = 1e-3, h1 = 1e-3, h0 = 1e-3)
within_tolerance <- summary$max_abs_difference <= tolerance[summary$metric]
if (any(rows$magmaan_df != rows$lavaan_df) || any(rows$decision_mismatch) ||
    !all(within_tolerance)) {
  stop("MLR lavaan parity gate failed", call. = FALSE)
}
cat(sprintf("PASS: %d same-data magmaan/lavaan MLR comparisons in %.1f s\n",
            nrow(rows), wall_seconds))
