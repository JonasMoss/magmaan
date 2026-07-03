#!/usr/bin/env Rscript

bootstrap_script <- {
  args0 <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args0, value = TRUE)
  if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    normalizePath("run_experiment.R", mustWork = FALSE)
  }
}
source(file.path(dirname(dirname(bootstrap_script)), "_support", "R", "helpers.R"))

usage <- function() {
  cat(
"SAM efficiency/stability probe

Usage:
  Rscript run_experiment.R [options]

Options:
  --help                 Show this help.
  --smoke                Cheap run: n=80, reps=2, ref_n=1000.
  --budget-min VALUE     Wall-clock budget in minutes [default: 30].
  --reps-target VALUE    Target replications per generator x n cell [default: 120].
  --n-values CSV         Comma-separated sample sizes [default: 80,160,320].
  --generators CSV       normal,ig_johnson,ordinal5_pseudocontinuous [default: all].
  --cores VALUE          Number of forked workers, or auto [default: 1].
  --seed-base VALUE      Base seed [default: 51000].
  --ref-n VALUE          Large-N pseudo-true reference for ordinal scores [default: 20000].
  --results-dir PATH     Output directory [default: ./results].

Outputs:
  results/replicates.csv, results/summary.csv, results/metadata.csv,
  and results/progress.csv.
", sep = "")
}

arg_value <- function(args, name, default = NULL) {
  key <- paste0("--", name)
  eq <- grep(paste0("^", key, "="), args, value = TRUE)
  if (length(eq)) return(sub(paste0("^", key, "="), "", eq[[1L]]))
  hit <- which(args == key)
  if (length(hit) && hit[[1L]] < length(args)) return(args[[hit[[1L]] + 1L]])
  default
}

args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  usage()
  quit(save = "no", status = 0)
}

script <- script_path()
set_single_threaded_math()
require_pkg("magmaan", "run `just r-dev` from the repository root first")

smoke <- "--smoke" %in% args
budget_min <- as.numeric(arg_value(args, "budget-min", "30"))
reps_target <- as.integer(arg_value(args, "reps-target", "120"))
n_values <- as.integer(parse_csv_arg(arg_value(args, "n-values", "80,160,320")))
generators <- parse_csv_arg(arg_value(
  args, "generators", "normal,ig_johnson,ordinal5_pseudocontinuous"))
cores_arg <- arg_value(args, "cores", "1")
seed_base <- as.integer(arg_value(args, "seed-base", "51000"))
ref_n <- as.integer(arg_value(args, "ref-n", "20000"))
out_dir <- normalizePath(arg_value(args, "results-dir",
                                   results_dir(script = script, create = TRUE)),
                         mustWork = FALSE)

if (smoke) {
  budget_min <- 2
  reps_target <- 2L
  n_values <- 80L
  ref_n <- 1000L
}
if (!length(n_values) || any(!is.finite(n_values)) || any(n_values < 20L)) {
  stop("n-values must be positive integers >= 20", call. = FALSE)
}
allowed_generators <- c("normal", "ig_johnson", "ordinal5_pseudocontinuous")
if (!all(generators %in% allowed_generators)) {
  stop("unsupported generator(s): ",
       paste(setdiff(generators, allowed_generators), collapse = ", "),
       call. = FALSE)
}
cores <- if (identical(cores_arg, "auto")) {
  max(1L, parallel::detectCores(logical = FALSE) - 1L)
} else {
  as.integer(cores_arg)
}
cores <- max(1L, cores)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

core <- magmaan::magmaan_core
ov_names <- c(paste0("x", 1:4), paste0("y", 1:4))
model <- "
  f1 =~ x1 + x2 + x3 + x4
  f2 =~ y1 + y2 + y3 + y4
  f2 ~ f1
"

population_cov <- function() {
  loadings <- c(.80, .75, .70, .65, .78, .74, .68, .62)
  Lambda <- matrix(0, 8, 2)
  Lambda[1:4, 1] <- loadings[1:4]
  Lambda[5:8, 2] <- loadings[5:8]
  Phi <- matrix(c(1, .5, .5, 1), 2, 2)
  Theta <- diag(1 - loadings^2)
  out <- Lambda %*% Phi %*% t(Lambda) + Theta
  dimnames(out) <- list(ov_names, ov_names)
  out
}

Sigma <- population_cov()
p <- ncol(Sigma)
chol_sigma <- chol(Sigma)

ig_calibration <- NULL
ordinal_calibration <- NULL

calibrate_generator <- function(generator) {
  if (identical(generator, "ig_johnson") && is.null(ig_calibration)) {
    message("Calibrating Johnson independent-generator margins...")
    ig_calibration <<- core$sim_ig_calibrate(
      Sigma,
      target_skewness = rep(1, p),
      target_excess_kurtosis = rep(3, p),
      root = "symmetric",
      generator_family = "johnson"
    )
  }
  if (identical(generator, "ordinal5_pseudocontinuous") &&
      is.null(ordinal_calibration)) {
    message("Calibrating five-category ordinal latent-response generator...")
    ordinal_calibration <<- core$sim_ordcorr_summary_calibrate(
      latent_corr = Sigma,
      kinds = rep(1L, p),
      thresholds = rep(list(c(-1.2, -0.4, 0.4, 1.2)), p)
    )
  }
}

draw_data <- function(generator, n, seed) {
  calibrate_generator(generator)
  if (identical(generator, "normal")) {
    set.seed(seed)
    X <- matrix(stats::rnorm(n * p), n, p) %*% chol_sigma
  } else if (identical(generator, "ig_johnson")) {
    X <- core$sim_ig_draw(ig_calibration, n = n, reps = 1L,
                          seed_base = seed)$draws[[1L]]
  } else {
    draw <- core$sim_ordcorr_draw(ordinal_calibration, n = n, reps = 1L,
                                  seed_base = seed)$draws[[1L]]
    X <- matrix(as.numeric(draw$X), nrow = n)
  }
  colnames(X) <- ov_names
  as.data.frame(X)
}

focal_from_fit <- function(fit, vc = NULL) {
  pt <- fit$partable
  row <- which(pt$lhs == "f2" & pt$op == "~" & pt$rhs == "f1")
  if (length(row) != 1L) stop("focal parameter f2 ~ f1 not found")
  free <- pt$free[[row]]
  se <- NA_real_
  if (!is.null(fit$se) && length(fit$se) >= free) {
    se <- fit$se[[free]]
  } else if (!is.null(vc) && free > 0L && free <= nrow(vc)) {
    se <- sqrt(max(0, vc[free, free]))
  }
  c(estimate = pt$est[[row]], se = se)
}

fit_one <- function(X, estimator, compute_se = TRUE) {
  if (identical(estimator, "SAM")) {
    fit <- magmaan::sam(
      model, X,
      method = "local",
      mapping = "ml",
      se = if (compute_se) "twostep.robust" else "none"
    )
    return(focal_from_fit(fit))
  }
  fit <- magmaan::magmaan(model, X, estimator = "ML",
                          se = "none", test = "none")
  vc <- if (compute_se) stats::vcov(fit, data = X, regime = "model") else NULL
  focal_from_fit(fit, vc)
}

reference_cache <- new.env(parent = emptyenv())
reference_value <- function(generator, estimator) {
  if (!identical(generator, "ordinal5_pseudocontinuous")) return(0.5)
  key <- paste(generator, estimator, sep = "::")
  if (exists(key, reference_cache, inherits = FALSE)) {
    return(get(key, reference_cache, inherits = FALSE))
  }
  Xref <- draw_data(generator, ref_n,
                    seed_base + 900000L + match(estimator, c("SAM", "ML")))
  val <- fit_one(Xref, estimator, compute_se = FALSE)[["estimate"]]
  assign(key, val, reference_cache)
  val
}

safe_rep <- function(generator, n, rep_id, estimator) {
  seed <- seed_base +
    match(generator, allowed_generators) * 100000L +
    n * 100L + rep_id
  ref <- reference_value(generator, estimator)
  t0 <- proc.time()[["elapsed"]]
  out <- tryCatch({
    X <- draw_data(generator, n, seed)
    ans <- fit_one(X, estimator, compute_se = TRUE)
    elapsed <- 1000 * (proc.time()[["elapsed"]] - t0)
    data.frame(
      generator = generator,
      n = n,
      rep = rep_id,
      estimator = estimator,
      ok = TRUE,
      error = "",
      estimate = unname(ans[["estimate"]]),
      se = unname(ans[["se"]]),
      reference = ref,
      covered = is.finite(ans[["se"]]) &&
        abs(ans[["estimate"]] - ref) <= 1.96 * ans[["se"]],
      elapsed_ms = elapsed,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    elapsed <- 1000 * (proc.time()[["elapsed"]] - t0)
    data.frame(
      generator = generator,
      n = n,
      rep = rep_id,
      estimator = estimator,
      ok = FALSE,
      error = conditionMessage(e),
      estimate = NA_real_,
      se = NA_real_,
      reference = ref,
      covered = NA,
      elapsed_ms = elapsed,
      stringsAsFactors = FALSE
    )
  })
  out
}

summarize_rows <- function(rows) {
  if (!nrow(rows)) return(data.frame())
  pieces <- split(rows, paste(rows$generator, rows$n, rows$estimator, sep = "\r"))
  do.call(rbind, lapply(pieces, function(d) {
    ok <- d[d$ok & is.finite(d$estimate), , drop = FALSE]
    emp_sd <- if (nrow(ok) > 1L) stats::sd(ok$estimate) else NA_real_
    mean_se <- if (nrow(ok)) mean(ok$se, na.rm = TRUE) else NA_real_
    data.frame(
      generator = d$generator[[1L]],
      n = d$n[[1L]],
      estimator = d$estimator[[1L]],
      reps = nrow(d),
      ok_rate = mean(d$ok),
      failure_rate = mean(!d$ok),
      bias = if (nrow(ok)) mean(ok$estimate - ok$reference, na.rm = TRUE) else NA_real_,
      emp_sd = emp_sd,
      mean_se = mean_se,
      se_to_sd = mean_se / emp_sd,
      coverage95 = if (nrow(ok)) mean(ok$covered, na.rm = TRUE) else NA_real_,
      median_ms = stats::median(d$elapsed_ms, na.rm = TRUE),
      p90_ms = unname(stats::quantile(d$elapsed_ms, 0.90, na.rm = TRUE)),
      stringsAsFactors = FALSE
    )
  }))
}

write_progress <- function(completed, total, start_time, last = "") {
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  rate <- if (completed > 0L) elapsed / completed else NA_real_
  eta <- if (is.finite(rate)) rate * (total - completed) else NA_real_
  write_csv(data.frame(
    completed = completed,
    total = total,
    elapsed_seconds = elapsed,
    eta_seconds = eta,
    last = last,
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    stringsAsFactors = FALSE
  ), file.path(out_dir, "progress.csv"))
}

estimators <- c("SAM", "ML")
tasks <- expand.grid(generator = generators, n = n_values,
                     estimator = estimators, stringsAsFactors = FALSE)
total_reps <- nrow(tasks) * reps_target
start_time <- Sys.time()
deadline <- start_time + budget_min * 60
all_rows <- list()
completed <- 0L
write_progress(completed, total_reps, start_time)

message("Running ", nrow(tasks), " cells with target reps=", reps_target,
        ", cores=", cores, ", budget_min=", budget_min)
for (i in seq_len(nrow(tasks))) {
  task <- tasks[i, ]
  if (Sys.time() >= deadline && length(all_rows)) {
    message("Budget reached before scheduling remaining cells.")
    break
  }
  label <- paste(task$generator, "n", task$n, task$estimator)
  message("[", i, "/", nrow(tasks), "] ", label)
  reps <- seq_len(reps_target)
  run_rep <- function(r) safe_rep(task$generator, task$n, r, task$estimator)
  rows <- if (cores > 1L) {
    parallel::mclapply(reps, run_rep, mc.cores = cores,
                       mc.preschedule = TRUE)
  } else {
    lapply(reps, run_rep)
  }
  all_rows <- c(all_rows, rows)
  completed <- completed + length(rows)
  rep_df <- do.call(rbind, all_rows)
  write_csv(rep_df, file.path(out_dir, "replicates.csv"))
  write_csv(summarize_rows(rep_df), file.path(out_dir, "summary.csv"))
  write_progress(completed, total_reps, start_time, label)
}

rep_df <- if (length(all_rows)) do.call(rbind, all_rows) else data.frame()
write_csv(rep_df, file.path(out_dir, "replicates.csv"))
write_csv(summarize_rows(rep_df), file.path(out_dir, "summary.csv"))
write_metadata(
  file.path(out_dir, "metadata.csv"),
  values = list(
    experiment = "51-sam-efficiency-stability",
    smoke = smoke,
    budget_min = budget_min,
    reps_target = reps_target,
    n_values = n_values,
    generators = generators,
    estimators = estimators,
    cores = cores,
    seed_base = seed_base,
    ref_n = ref_n,
    completed_reps = nrow(rep_df)
  ),
  packages = c("magmaan")
)

message("Wrote:")
message("  ", file.path(out_dir, "replicates.csv"))
message("  ", file.path(out_dir, "summary.csv"))
message("  ", file.path(out_dir, "metadata.csv"))
