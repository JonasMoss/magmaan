#!/usr/bin/env Rscript
# magmaan vs lavaan speed bench (README-facing headline).
#
# Answers one question: on the same fitted model, how much faster is magmaan
# than lavaan? Three one-factor CFA populations (small/medium/large) are each
# fit three ways (continuous ML, FIML on 10% MCAR data, ordinal DWLS). Both
# engines are timed from their own *prepared* inputs: magmaan via
# prepare_model()/prepare_data()/prepare_weight(), lavaan by re-running a
# lavaan() call from its own unfitted slots (do.fit = FALSE, then flipped
# back on). That excludes model/data setup from both sides and leaves the
# fit call itself as the timed unit. Every call still reoptimizes from
# data-dependent starting values; no fitted solution is reused.
#
# A rigorous, fully attributed pipeline-cost study (raw-to-report timing,
# stage decomposition, matched inference workloads) is tracked separately;
# see project/validation/benchmark_plan.md. This experiment stays small and
# is the one the README links to.
#
# Usage:
#   Rscript run_experiment.R [--batches N] [--target-ms N] [--smoke]

if ("--help" %in% commandArgs(TRUE)) {
  cat("Usage: Rscript run_experiment.R [--batches N] [--target-ms N] [--smoke]\n",
      "  --batches N    timing batches per (task, case) pair (default 7)\n",
      "  --target-ms N  per-batch calibration target in ms (default 100)\n",
      "  --smoke        fast check: 1 batch, 20ms target; not publication numbers\n",
      sep = "")
  quit(save = "no", status = 0L)
}

.here <- dirname(normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])))
source(file.path(.here, "..", "..", "_support", "R", "helpers.R"))
source(file.path(repo_root(), "benchmarks", "r", "timing.R"))

parse_args <- function(args) {
  smoke <- "--smoke" %in% args
  flag_value <- function(flag, default) {
    i <- match(flag, args)
    if (is.na(i)) return(default)
    if (i == length(args)) stop("missing value for ", flag, call. = FALSE)
    args[[i + 1L]]
  }
  list(
    batches = as.integer(flag_value("--batches", if (smoke) 1L else 7L)),
    target_ms = as.numeric(flag_value("--target-ms", if (smoke) 20 else 100))
  )
}
args <- parse_args(commandArgs(TRUE))
stopifnot(is.finite(args$batches), args$batches > 0,
          is.finite(args$target_ms), args$target_ms > 0)

set_single_threaded_math()
require_pkg("lavaan")
require_pkg("magmaanlab")
suppressPackageStartupMessages(library(magmaanlab))

results_dir <- ensure_results_dir()

# --- Three one-factor populations, small to large --------------------------
# Loading 0.7, residual variance 0.51: standardized indicators (0.7^2+0.51=1).
seed_base <- 20260929L
cases <- list(
  list(n = 200L,  p = 6L,  seed = seed_base + 1L),
  list(n = 500L,  p = 9L,  seed = seed_base + 2L),
  list(n = 1000L, p = 12L, seed = seed_base + 3L)
)

make_population <- function(n, p, seed) {
  set.seed(seed)
  latent <- rnorm(n)
  df <- as.data.frame(0.7 * matrix(rep(latent, p), n, p) +
                       sqrt(0.51) * matrix(rnorm(n * p), n, p))
  names(df) <- paste0("x", seq_len(p))
  df
}

make_fiml_data <- function(df, seed, frac = 0.10) {
  set.seed(seed)
  n <- nrow(df)
  for (j in seq_len(ncol(df))) df[sample(n, round(frac * n)), j] <- NA_real_
  df
}

ordinal_cuts <- c(-0.70, 0, 0.70)
make_ordinal_data <- function(df) {
  df[] <- lapply(df, function(x) ordered(cut(x, c(-Inf, ordinal_cuts, Inf), labels = FALSE)))
  df
}

# --- Prepared fit closures for each engine ----------------------------------
# Each returns list(run = function() fit, fit = a first fit for the gate).

magmaan_continuous <- function(syntax, df) {
  nm <- prepare_model(syntax, meanstructure = TRUE)
  nd <- prepare_data(nm, df)
  run <- function() estimate(nm, nd, estimator = "ML")
  list(run = run, fit = run())
}

lavaan_continuous <- function(syntax, df) {
  unfitted <- lavaan::cfa(syntax, df, meanstructure = TRUE,
    se = "none", test = "none", baseline = FALSE, h1 = FALSE, do.fit = FALSE)
  opts <- unfitted@Options
  opts$do.fit <- TRUE
  run <- function() lavaan::lavaan(slotOptions = opts, slotParTable = unfitted@ParTable,
    slotSampleStats = unfitted@SampleStats, slotData = unfitted@Data,
    slotModel = unfitted@Model, slotCache = unfitted@Cache)
  list(run = run, fit = run())
}

magmaan_fiml <- function(syntax, df) {
  nm <- prepare_model(syntax, meanstructure = TRUE)
  nd <- prepare_data(nm, df, kind = "raw")
  run <- function() estimate(nm, nd, estimator = "FIML")
  list(run = run, fit = run())
}

lavaan_fiml <- function(syntax, df) {
  unfitted <- lavaan::cfa(syntax, df, meanstructure = TRUE, missing = "fiml",
    se = "none", test = "none", baseline = FALSE, h1 = FALSE, do.fit = FALSE)
  opts <- unfitted@Options
  opts$do.fit <- TRUE
  run <- function() lavaan::lavaan(slotOptions = opts, slotParTable = unfitted@ParTable,
    slotSampleStats = unfitted@SampleStats, slotData = unfitted@Data,
    slotModel = unfitted@Model, slotCache = unfitted@Cache)
  list(run = run, fit = run())
}

magmaan_ordinal <- function(syntax, df, ordered_names) {
  nm <- prepare_model(syntax, ordered = ordered_names, meanstructure = TRUE, prototype = df)
  nd <- prepare_data(nm, df)
  nw <- prepare_weight(nd, "DWLS", full = FALSE)
  run <- function() estimate(nm, nd, weight = nw)
  list(run = run, fit = run())
}

lavaan_ordinal <- function(syntax, df, ordered_names) {
  unfitted <- lavaan::cfa(syntax, df, ordered = ordered_names, estimator = "DWLS",
    meanstructure = TRUE, se = "none", test = "none", baseline = FALSE, h1 = FALSE,
    do.fit = FALSE)
  opts <- unfitted@Options
  opts$do.fit <- TRUE
  run <- function() lavaan::lavaan(slotOptions = opts, slotParTable = unfitted@ParTable,
    slotSampleStats = unfitted@SampleStats, slotData = unfitted@Data,
    slotModel = unfitted@Model, slotCache = unfitted@Cache)
  list(run = run, fit = run())
}

tasks <- list(
  continuous = list(label = "Continuous ML",
    magmaan = magmaan_continuous, lavaan = lavaan_continuous),
  fiml = list(label = "FIML (10% MCAR)",
    magmaan = magmaan_fiml, lavaan = lavaan_fiml),
  ordinal = list(label = "Ordinal DWLS (4 categories)",
    magmaan = magmaan_ordinal, lavaan = lavaan_ordinal)
)

# --- Correctness gate --------------------------------------------------------
# Every free parameter must agree between engines before a fit is timed.
# Comparing free parameters (not the raw partable) sidesteps how the ordinal
# partables represent fixed/derived residual variances differently.
max_free_param_diff <- function(magmaan_fit, lavaan_fit) {
  mp <- magmaan_fit$partable
  lp <- lavaan::parTable(lavaan_fit)
  key <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group, sep = "|")
  ii <- match(key(mp), key(lp))
  stopifnot(!anyNA(ii))
  free <- mp$free > 0
  max(abs(mp$est[free] - lp$est[ii][free]))
}

# --- Run every (task, case) --------------------------------------------------
case_rows <- list()
sample_rows <- list()

for (task_id in names(tasks)) {
  task <- tasks[[task_id]]
  for (case in cases) {
    df <- make_population(case$n, case$p, case$seed)
    syntax <- paste("f =~", paste(names(df), collapse = " + "))

    if (task_id == "fiml") {
      df <- make_fiml_data(df, case$seed)
    } else if (task_id == "ordinal") {
      df <- make_ordinal_data(df)
    }
    ordered_names <- if (task_id == "ordinal") names(df) else NULL

    magmaan_ready <- if (task_id == "ordinal") task$magmaan(syntax, df, ordered_names) else task$magmaan(syntax, df)
    lavaan_ready <- if (task_id == "ordinal") task$lavaan(syntax, df, ordered_names) else task$lavaan(syntax, df)

    stopifnot(isTRUE(magmaan_ready$fit$converged),
              isTRUE(lavaan::lavInspect(lavaan_ready$fit, "converged")))
    diff <- max_free_param_diff(magmaan_ready$fit, lavaan_ready$fit)
    if (!is.finite(diff) || diff > 1e-3) {
      stop(sprintf("%s at n=%d, p=%d: estimate drift %.3g exceeds 1e-3",
                    task_id, case$n, case$p, diff), call. = FALSE)
    }

    message(sprintf("%-10s n=%-4d p=%-3d  max|delta est|=%.2g", task_id, case$n, case$p, diff))
    timed <- time_paired(list(magmaan = magmaan_ready$run, lavaan = lavaan_ready$run),
                          batches = args$batches, target_ms = args$target_ms)
    timed$task <- task_id
    timed$n <- case$n
    timed$p <- case$p
    sample_rows[[length(sample_rows) + 1L]] <- timed

    magmaan_ms <- stats::median(timed$ms[timed$engine == "magmaan"])
    lavaan_ms <- stats::median(timed$ms[timed$engine == "lavaan"])
    case_rows[[length(case_rows) + 1L]] <- data.frame(
      task = task_id, label = task$label, n = case$n, p = case$p,
      magmaan_ms = magmaan_ms, lavaan_ms = lavaan_ms,
      speedup = lavaan_ms / magmaan_ms, check_max_abs_diff = diff
    )
  }
}

cases_out <- do.call(rbind, case_rows)
samples_out <- do.call(rbind, sample_rows)

# One number per task: geometric mean speedup across the three population sizes.
summary_out <- do.call(rbind, lapply(names(tasks), function(task_id) {
  rows <- cases_out[cases_out$task == task_id, ]
  data.frame(task = task_id, label = tasks[[task_id]]$label,
             speedup_geomean = exp(mean(log(rows$speedup))))
}))

write_csv(cases_out, file.path(results_dir, "cases.csv"))
write_csv(samples_out, file.path(results_dir, "samples.csv"))
write_csv(summary_out, file.path(results_dir, "summary.csv"))

write_metadata(
  file.path(results_dir, "metadata.csv"),
  values = list(
    batches = args$batches, target_ms = args$target_ms,
    seed_base = seed_base,
    cases = vapply(cases, function(c) sprintf("n=%d,p=%d", c$n, c$p), character(1)),
    tasks = names(tasks),
    single_threaded_math = TRUE,
    check_tolerance = 1e-3
  ),
  packages = c("lavaan", "magmaanlab")
)

print(summary_out, row.names = FALSE)
cat(sprintf("wrote %s\n", file.path(results_dir, "cases.csv")))
cat(sprintf("wrote %s\n", file.path(results_dir, "summary.csv")))
