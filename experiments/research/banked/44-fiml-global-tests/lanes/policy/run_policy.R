#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(magmaan))
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
lane_dir <- dirname(normalizePath(script))
study_dir <- normalizePath(file.path(lane_dir, "../.."))
support_dir <- normalizePath(file.path(study_dir, "../../../_support/R"))
source(file.path(support_dir, "helpers.R"))
source(file.path(support_dir, "missingness.R"))
source(file.path(study_dir, "lanes/global/R/sem_models.R"))
set_single_threaded_math()
launch <- proc.time()
usage <- function() cat(
  "Usage: Rscript run_experiment.R --lane policy [options]\n",
  "Ordinary FIML policy null calibration with paired diagnostic geometries.\n",
  "  --models CSV          Default one_factor_6,two_factor_fmg_10,three_factor_15;\n",
  "                        also bifactor_12,linear_growth_5.\n",
  "  --distributions CSV   Default normal,ig2; also vm1,ig1,vm2.\n",
  "  --missingness CSV     Default mcar_30,mar_30; also complete.\n",
  "  --n CSV               Default 200,1000.\n",
  "  --reps CSV            Default 400,200, matched to n; one applies to all n.\n",
  "  --seed-base N         Default 2026102901.\n",
  "  --chunk-size N        Default 25.\n",
  "  --workers N           Default 4 (single-threaded math).\n",
  "  --time-budget-minutes M  Stop starting chunks after M minutes.\n",
  "  --results-dir PATH    Absolute output directory, or\n",
  "  --run-id ID           Output under results/policy/ID.\n",
  "  --summarize-only      Read checkpoints without fitting.\n",
  "  --smoke               Two reps, N=200, normal, three default models,\n",
  "                        both missingness mechanisms, two workers, fresh seed;\n",
  "                        output /tmp/fiml-policy-smoke.\n",
  "  --help                Show help.\n", sep = "")
opts <- list(models = c("one_factor_6", "two_factor_fmg_10", "three_factor_15"),
             distributions = c("normal", "ig2"), missingness = c("mcar_30", "mar_30"),
             n = c(200L, 1000L), reps = c(400L, 200L), seed_base = 2026102901,
             chunk_size = 25L, workers = 4L, time_budget_minutes = Inf,
             results_dir = NULL, run_id = NULL, summarize_only = FALSE)
args <- commandArgs(TRUE)
i <- 1L
while (i <= length(args)) {
  key <- args[[i]]
  if (key %in% c("--help", "-h")) { usage(); quit(status = 0L) }
  if (key == "--summarize-only") opts$summarize_only <- TRUE
  else if (key == "--smoke") {
    opts$n <- 200L; opts$reps <- 2L; opts$distributions <- "normal"
    opts$workers <- 2L; opts$seed_base <- 2026102999
    opts$results_dir <- "/tmp/fiml-policy-smoke"
  } else {
    name <- gsub("-", "_", sub("^--", "", key))
    if (!startsWith(key, "--") || !name %in% names(opts) || i == length(args))
      stop("Unknown option or missing value: ", key)
    i <- i + 1L
    value <- args[[i]]
    opts[[name]] <- if (name %in% c("models", "distributions", "missingness"))
      parse_csv_arg(value) else if (name %in% c("n", "reps"))
        as.numeric(parse_csv_arg(value)) else if (is.numeric(opts[[name]]))
          as.numeric(value) else value
  }
  i <- i + 1L
}
positive_int <- function(x) length(x) > 0L && all(is.finite(x) & x > 0 & x == floor(x))
stopifnot(positive_int(opts$n), positive_int(opts$reps), positive_int(opts$workers),
          positive_int(opts$chunk_size), length(opts$workers) == 1L,
          length(opts$chunk_size) == 1L, !anyDuplicated(opts$n),
          is.finite(opts$seed_base), opts$seed_base >= 0,
          opts$seed_base == floor(opts$seed_base), opts$time_budget_minutes > 0)
if (length(opts$reps) == 1L) opts$reps <- rep(opts$reps, length(opts$n))
stopifnot(length(opts$reps) == length(opts$n))
if (!is.null(opts$results_dir) && !is.null(opts$run_id)) stop("Choose results-dir or run-id")
if (!is.null(opts$run_id)) {
  stopifnot(grepl("^[A-Za-z0-9][A-Za-z0-9_.-]*$", opts$run_id))
  opts$results_dir <- file.path(study_dir, "results/policy", opts$run_id)
}
if (is.null(opts$results_dir) || !startsWith(opts$results_dir, "/"))
  stop("An absolute --results-dir or --run-id is required")
models <- sem_model_catalog()
all_models <- names(models)
all_distributions <- c("normal", "vm1", "ig1", "vm2", "ig2")
all_missingness <- c("complete", "mcar_30", "mar_30")
for (key in c("models", "distributions", "missingness")) {
  allowed <- switch(key, models = all_models, distributions = all_distributions,
                    missingness = all_missingness)
  if (!length(opts[[key]]) || anyDuplicated(opts[[key]]) ||
      !all(opts[[key]] %in% allowed)) stop("Invalid --", key)
}
grid <- expand.grid(model_id = all_models, distribution = all_distributions,
                    missingness = all_missingness, n = opts$n,
                    stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
grid$cell_id <- seq_len(nrow(grid))
grid <- grid[grid$model_id %in% opts$models & grid$distribution %in% opts$distributions &
               grid$missingness %in% opts$missingness, ]
grid$expected_df <- vapply(grid$model_id, function(id) models[[id]]$expected_df, integer(1))
grid$analysis_region <- ifelse(grid$distribution != "normal" & grid$missingness == "mar_30",
                               "nonnormal_mar_stress", "identified_null")
grid$reps <- opts$reps[match(grid$n, opts$n)]
results <- opts$results_dir
raw_dir <- file.path(results, "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
code_files <- sort(c(list.files(lane_dir, recursive = TRUE, full.names = TRUE),
                     file.path(study_dir, "run_experiment.R"),
                     file.path(study_dir, "lanes/global/R/sem_models.R"),
                     file.path(support_dir, "missingness.R")))
code_hash <- paste(unname(tools::md5sum(code_files)), collapse = ":")
config <- data.frame(models = paste(opts$models, collapse = ","),
  distributions = paste(opts$distributions, collapse = ","),
  missingness = paste(opts$missingness, collapse = ","), n = paste(opts$n, collapse = ","),
  reps = paste(opts$reps, collapse = ","), seed_base = opts$seed_base,
  chunk_size = opts$chunk_size, code_hash = code_hash)
config_path <- file.path(results, "run_config.csv")
if (file.exists(config_path)) {
  old <- read.csv(config_path, colClasses = "character", check.names = FALSE)
  if (!identical(names(old), names(config)) || nrow(old) != 1L ||
      !identical(unname(unlist(old)), unname(as.character(config[1L, ]))))
    stop("Existing result configuration differs; choose a new output directory")
} else write_csv(config, config_path)

loading <- function(factor, ov) paste0(factor, " =~ ", paste(ov, collapse = " + "))
syntax_table <- list(
  one_factor_6 = loading("f", paste0("x", 1:6)),
  two_factor_fmg_10 = paste(loading("f1", paste0("x", 1:5)),
                             loading("f2", paste0("x", 6:10)), sep = "\n"),
  three_factor_15 = paste(vapply(1:3, function(j)
    loading(paste0("f", j), paste0("x", ((j-1)*5+1):(j*5))), character(1)), collapse = "\n"),
  bifactor_12 = paste(loading("G", paste0("x", 1:12)),
    loading("g1", paste0("x", 1:4)), loading("g2", paste0("x", 5:8)),
    loading("g3", paste0("x", 9:12)), "G ~~ 0*g1 + 0*g2 + 0*g3",
    "g1 ~~ 0*g2 + 0*g3", "g2 ~~ 0*g3", sep = "\n"),
  linear_growth_5 = paste("i =~ 1*y1 + 1*y2 + 1*y3 + 1*y4 + 1*y5",
    "s =~ 0*y1 + 1*y2 + 2*y3 + 3*y4 + 4*y5", "i ~ 1", "s ~ 1",
    paste0("y", 1:5, " ~ 0*1", collapse = "\n"), sep = "\n"))
arms <- list(obsH0_exp = c("observed", "expected"), expH0_exp = c("expected", "expected"),
             obsH1_obsH1 = c("observed-h1", "observed-h1"), obsH1_exp = c("observed-h1", "expected"))
ordinary_models <- samplers <- list()
prepare_models <- function() {
  for (id in opts$models) {
    model <- models[[id]]
    prototype <- as.data.frame(matrix(numeric(), 0, model$p, dimnames = list(NULL, model$ov)))
    ordinary_models[[id]] <<- magmaan_model(syntax_table[[id]], prototype = prototype,
      identification = if (id %in% c("three_factor_15", "bifactor_12")) "std.lv" else "marker")
    normal <- sem_calibrate_sampler(model, "normal")
    fit <- magmaan(ordinary_models[[id]], as.data.frame(sem_draw(model, normal, 2000L,
                       sem_seed(opts$seed_base + match(id, all_models)))), estimator = "FIML")
    dfs <- summary(fit, references = "std")$tests$df
    if (!length(dfs) || anyNA(dfs) || any(dfs != model$expected_df))
      stop("Startup df mismatch for ", id, ": ", paste(dfs, collapse = ","))
    for (distribution in opts$distributions)
      samplers[[paste(id, distribution)]] <<- sem_calibrate_sampler(model, distribution)
  }
}
empty_rep <- function(cell, rep_id, seed) {
  out <- as.list(cell[c("cell_id", "model_id", "expected_df", "distribution", "missingness",
                         "analysis_region", "n")])
  out <- c(out, list(rep = rep_id, seed = seed, fit_ok = FALSE, nobs = NA_real_,
    realized_missing_fraction = NA_real_, fit_error = "", policy_error = "", mlr_error = "",
    global_score_available = FALSE, global_lr_available = FALSE,
    global_score_reason = "", global_lr_reason = "", mlr_reference = "",
    policy_matches_obsH0 = NA))
  for (arm in c("score", "lr", "mlr", names(arms))) {
    out[[paste0(arm, "_stat")]] <- NA_real_
    out[[paste0(arm, "_df")]] <- NA_real_
    if (arm %in% names(arms)) {
      out[[paste0(arm, "_rank")]] <- NA_real_
      out[[paste0(arm, "_eig_mean")]] <- NA_real_
      out[[paste0(arm, "_error")]] <- ""
    }
    refs <- if (arm == "mlr") "" else if (arm %in% names(arms)) c("sb", "peba4") else c("std", "sb", "peba4")
    for (ref in refs) {
      suffix <- if (ref == "") "" else paste0("_", ref)
      out[[paste0("p_", arm, suffix)]] <- NA_real_
      out[[paste0(arm, suffix, "_reason")]] <- ""
    }
  }
  for (stage in c("fit", "policy", "mlr", names(arms), "total")) out[[paste0(stage, "_seconds")]] <- NA_real_
  out
}
one_rep <- function(cell, rep_id) {
  begin <- proc.time()[["elapsed"]]
  seed <- sem_seed(opts$seed_base + cell$cell_id * 100003 + rep_id)
  out <- empty_rep(cell, rep_id, seed)
  # Each component is isolated so unavailable inference never suppresses another arm.
  attempt <- function(component, expr) {
    start <- proc.time()[["elapsed"]]
    value <- tryCatch(force(expr), error = function(e) {
      out[[paste0(component, "_error")]] <<- conditionMessage(e)
      NULL
    })
    out[[paste0(component, "_seconds")]] <<- proc.time()[["elapsed"]] - start
    value
  }
  fit <- attempt("fit", {
    X <- sem_draw(models[[cell$model_id]], samplers[[paste(cell$model_id, cell$distribution)]], cell$n, seed)
    set.seed(seed + 700001L)
    X <- sem_apply_missingness(X, cell$missingness)
    out$realized_missing_fraction <- mean(is.na(X))
    magmaan(ordinary_models[[cell$model_id]], as.data.frame(X), estimator = "FIML")
  })
  if (!is.null(fit)) {
    out$fit_ok <- isTRUE(as_lab_fit(fit)$converged)
    out$nobs <- nobs(fit)
    for (component in c("global_score", "global_lr")) {
      status <- fit$inference$status
      row <- status[status$component == component, , drop = FALSE]
      out[[paste0(component, "_available")]] <- nrow(row) == 1L && isTRUE(row$available)
      out[[paste0(component, "_reason")]] <- paste(c(row$reason, row$detail), collapse = "; ")
    }
    attempt("policy", {
      tests <- summary(fit, references = c("std", "sb", "peba4"))$tests
      for (arm in c("score", "lr")) {
        rows <- tests[tests$test == arm, , drop = FALSE]
        if (!nrow(rows)) stop("Missing policy test: ", arm)
        out[[paste0(arm, "_stat")]] <- rows$statistic[[1L]]
        out[[paste0(arm, "_df")]] <- rows$df[[1L]]
        for (ref in c("std", "sb", "peba4")) {
          row <- rows[rows$reference == ref, , drop = FALSE]
          if (nrow(row) != 1L) stop("Missing policy reference: ", arm, "/", ref)
          out[[paste0("p_", arm, "_", ref)]] <- row$pvalue
          out[[paste0(arm, "_", ref, "_reason")]] <- row$reason
        }
      }
    })
    attempt("mlr", {
      z <- summary(fit, lavaan_compat = "MLR")$tests
      if (nrow(z) != 1L) stop("Expected one MLR test")
      out$mlr_stat <- z$statistic; out$mlr_df <- z$df
      out$p_mlr <- z$pvalue; out$mlr_reference <- z$reference; out$mlr_reason <- z$reason
    })
    for (arm in names(arms)) attempt(arm, {
      z <- magmaanlab::global_score_flip_test(as_lab_fit(fit), n_flips = 1L,
        seed = seed + 900001L, sensitivity = arms[[arm]][1L], metric = arms[[arm]][2L])
      out[[paste0(arm, "_stat")]] <- z$statistic_effective
      out[[paste0(arm, "_df")]] <- z$df
      out[[paste0(arm, "_rank")]] <- z$tangent_rank
      out[[paste0("p_", arm, "_sb")]] <- z$p_mean_scaled
      out[[paste0("p_", arm, "_peba4")]] <- magmaanlab:::infer_fmg_test(
        z$statistic_effective, z$df, z$eigenvalues, method = "peba", param = 4)$p_value
      out[[paste0(arm, "_eig_mean")]] <- mean(z$eigenvalues)
    })
    out$policy_matches_obsH0 <- if (all(is.finite(c(out$score_stat, out$obsH0_exp_stat))))
      abs(out$score_stat - out$obsH0_exp_stat) <= 1e-6 * max(1, abs(out$score_stat)) else NA
  }
  out$total_seconds <- proc.time()[["elapsed"]] - begin
  as.data.frame(out, stringsAsFactors = FALSE)
}

jobs <- list()
for (j in seq_len(nrow(grid))) {
  for (a in seq(1L, grid$reps[j], by = opts$chunk_size)) {
    b <- min(a + opts$chunk_size - 1L, grid$reps[j])
    jobs[[length(jobs) + 1L]] <- list(cell = grid[j, ], a = a, b = b,
      round = (a - 1L) %/% opts$chunk_size,
      path = file.path(raw_dir, sprintf("cell_%d_reps_%d_%d.csv", grid$cell_id[j], a, b)))
  }
}
jobs <- jobs[order(vapply(jobs, `[[`, numeric(1), "round"))]
valid_chunk <- function(job) {
  z <- read.csv(job$path, stringsAsFactors = FALSE)
  if (nrow(z) != job$b - job$a + 1L || !identical(z$rep, seq.int(job$a, job$b)) ||
      anyNA(z$cell_id) || any(z$cell_id != job$cell$cell_id))
    stop("Invalid existing chunk: ", job$path)
  z
}
existing <- vapply(jobs, function(job) file.exists(job$path), logical(1))
for (job in jobs[existing]) valid_chunk(job)
cat(sum(existing), "/", length(jobs), " chunks reused\n", sep = "")
if (!opts$summarize_only && any(!existing)) {
  prepare_models()
  pending <- which(!existing)
  for (batch in split(pending, ceiling(seq_along(pending) / opts$workers))) {
    parallel::mclapply(batch, function(index) {
      job <- jobs[[index]]
      elapsed <- proc.time()[["elapsed"]] - launch[["elapsed"]]
      if (elapsed > 60 * opts$time_budget_minutes) return(NULL)
      cat(sprintf("chunk %d/%d cell %d elapsed %.1fs\n", index, length(jobs), job$cell$cell_id, elapsed))
      rows <- do.call(rbind, lapply(seq.int(job$a, job$b), function(rep_id) one_rep(job$cell, rep_id)))
      temporary <- paste0(job$path, ".tmp")
      write_csv(rows, temporary)
      if (!file.rename(temporary, job$path)) stop("Cannot finalize chunk: ", job$path)
      TRUE
    }, mc.cores = opts$workers, mc.preschedule = FALSE) -> completed
    if (any(vapply(completed, inherits, logical(1), "try-error"))) stop("Chunk worker failed")
    if (proc.time()[["elapsed"]] - launch[["elapsed"]] > 60 * opts$time_budget_minutes) break
  }
}
chunks <- lapply(jobs[vapply(jobs, function(job) file.exists(job$path), logical(1))], valid_chunk)
rows <- if (length(chunks)) do.call(rbind, chunks) else {
  empty <- as.data.frame(empty_rep(grid[1L, ], 1L, 1L)); empty[FALSE, ]
}
write_csv(rows, file.path(results, "replications.csv"))
failures <- data.frame(cell_id = integer(), rep = integer(), seed = integer(),
                       component = character(), message = character())
for (column in grep("_error$", names(rows), value = TRUE)) {
  take <- !is.na(rows[[column]]) & nzchar(rows[[column]])
  failures <- rbind(failures, data.frame(rows[take, c("cell_id", "rep", "seed")],
    component = rep(sub("_error$", "", column), sum(take)), message = rows[[column]][take]))
}
write_csv(failures, file.path(results, "failures.csv"))
mean_finite <- function(x) if (any(is.finite(x))) mean(x[is.finite(x)]) else NA_real_
median_finite <- function(x) if (any(is.finite(x))) median(x[is.finite(x)]) else NA_real_
rates <- statistics <- timing <- list()
for (j in seq_len(nrow(grid))) {
  cell <- grid[j, ]; z <- rows[rows$cell_id == cell$cell_id, , drop = FALSE]
  descriptors <- cell[setdiff(names(cell), "reps")]
  for (arm in c("score", "lr", "mlr", names(arms))) {
    refs <- if (arm == "mlr") "yuan.bentler.mplus" else if (arm %in% names(arms)) c("sb", "peba4") else c("std", "sb", "peba4")
    for (ref in refs) {
      p <- z[[if (arm == "mlr") "p_mlr" else paste0("p_", arm, "_", ref)]]
      rates[[length(rates) + 1L]] <- cbind(descriptors, arm = arm, reference = ref,
        attempted = nrow(z), fit_ok = sum(z$fit_ok), usable = sum(is.finite(p)),
        rejection_rate = mean_finite(ifelse(is.finite(p), as.numeric(p <= .05), NA_real_)))
    }
  }
  s <- descriptors
  for (column in grep("_(stat|df)$", names(z), value = TRUE)) {
    s[[paste0("mean_", column)]] <- mean_finite(z[[column]])
    s[[paste0("median_", column)]] <- median_finite(z[[column]])
  }
  s$mean_score_over_lr <- mean_finite(z$score_stat / z$lr_stat)
  s$mean_obsH0_over_expH0 <- mean_finite(z$obsH0_exp_stat / z$expH0_exp_stat)
  s$policy_matches_obsH0 <- mean_finite(as.numeric(z$policy_matches_obsH0))
  statistics[[j]] <- s
  t <- descriptors
  for (column in grep("_seconds$", names(z), value = TRUE)) t[[paste0("mean_", column)]] <- mean_finite(z[[column]])
  timing[[j]] <- t
}
write_csv(do.call(rbind, rates), file.path(results, "rates.csv"))
write_csv(do.call(rbind, statistics), file.path(results, "statistics.csv"))
write_csv(do.call(rbind, timing), file.path(results, "timing.csv"))
commit <- tryCatch(system2("git", c("-C", shQuote(study_dir), "rev-parse", "HEAD"),
                           stdout = TRUE, stderr = FALSE), error = function(e) character())
if (!length(commit) || !is.null(attr(commit, "status"))) commit <- Sys.getenv("MAGMAAN_RUN_COMMIT", "unknown")
counts <- table(factor(failures$component, levels = c("fit", "policy", "mlr", names(arms))))
complete <- vapply(seq_len(nrow(grid)), function(j)
  sum(rows$cell_id == grid$cell_id[j]) == grid$reps[j], logical(1))
cpu <- proc.time() - launch
write_metadata(file.path(results, "metadata.csv"), values = c(list(
  command = paste(commandArgs(), collapse = " "), git_commit = commit[[1L]],
  seed_base = opts$seed_base, models = paste(opts$models, collapse = ","),
  distributions = paste(opts$distributions, collapse = ","),
  missingness = paste(opts$missingness, collapse = ","), n = paste(opts$n, collapse = ","),
  reps = paste(opts$reps, collapse = ","), workers = opts$workers,
  time_budget_minutes = opts$time_budget_minutes, wall_seconds = cpu[["elapsed"]],
  cpu_seconds = sum(cpu[c("user.self", "sys.self", "user.child", "sys.child")]),
  cells_complete = sum(complete), cells_partial = sum(!complete), code_hash = code_hash),
  setNames(as.list(as.integer(counts)), paste0("failures_", names(counts)))),
  packages = c("magmaan", "magmaanlab"))
cat("Outputs: ", results, "\n", sep = "")
