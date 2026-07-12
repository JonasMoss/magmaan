#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "expansion_design.R"))
set_single_threaded_math()

opts <- list(repeats = 20L, flips = c(99L, 499L, 1999L),
             seed_base = 20260714L, results_dir = NULL)
args <- commandArgs(TRUE); i <- 1L
take <- function() { i <<- i + 1L; args[[i]] }
while (i <= length(args)) {
  a <- args[[i]]
  if (a == "--repeats") opts$repeats <- as.integer(take())
  else if (a == "--flips") {
    opts$flips <- as.integer(strsplit(take(), ",", fixed = TRUE)[[1L]])
  } else if (a == "--seed-base") opts$seed_base <- as.integer(take())
  else if (a == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
results_dir <- opts$results_dir %||% file.path(script_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

scenarios <- data.frame(
  scenario = c("easy", "hard"), n_total = c(200L, 60L),
  balance = c("1:1", "1:3"),
  heterogeneity = c("homogeneous", "geometry"),
  distribution = c("normal", "skew"), stringsAsFactors = FALSE)

warm_data <- flip_expansion_draw_data(
  60L, "1:1", "homogeneous", "normal", opts$seed_base - 1L)
warm_pair <- flip_expansion_specs(1L)
invisible(lapply(warm_pair, magmaan, data = warm_data, estimator = "ML",
                 optimizer = "nlopt-lbfgs-slsqp-fallback"))

rows <- list(); row_id <- 1L
for (s in seq_len(nrow(scenarios))) {
  scenario <- scenarios[s, ]
  dat <- flip_expansion_draw_data(
    scenario$n_total, scenario$balance, scenario$heterogeneity,
    scenario$distribution, opts$seed_base + s)
  for (df in c(1L, 4L, 8L)) {
    pair <- flip_expansion_specs(df)
    fit_begin <- proc.time()[["elapsed"]]
    fit <- lapply(pair, magmaan, data = dat, estimator = "ML",
                  optimizer = "nlopt-lbfgs-slsqp-fallback")
    fit_seconds <- proc.time()[["elapsed"]] - fit_begin
    invisible(score_flip_test(fit$configural, fit$restricted, dat,
                              n_flips = 19L, seed = opts$seed_base))
    for (n_flips in opts$flips) {
      for (r in seq_len(opts$repeats)) {
        begin <- proc.time()[["elapsed"]]
        out <- score_flip_test(
          fit$configural, fit$restricted, dat, n_flips = n_flips,
          seed = opts$seed_base + 10000L * s + 100L * df + r)
        elapsed <- proc.time()[["elapsed"]] - begin
        rows[[row_id]] <- data.frame(
          scenario = scenario$scenario, n_total = scenario$n_total,
          balance = scenario$balance, heterogeneity = scenario$heterogeneity,
          distribution = scenario$distribution, df = df,
          n_flips = n_flips, repetition = r, fit_seconds = fit_seconds,
          elapsed_seconds = elapsed, setup_seconds = out$setup_seconds,
          resampling_score_seconds = out$resampling_score_seconds,
          standardization_seconds = out$resampling_standardization_seconds,
          asymptotic_seconds = out$asymptotic_seconds,
          core_total_seconds = out$total_seconds)
        row_id <- row_id + 1L
      }
    }
  }
}
raw <- do.call(rbind, rows)
write.csv(raw, file.path(results_dir, "timing_replications.csv"), row.names = FALSE)

timing_columns <- c("elapsed_seconds", "setup_seconds",
                    "resampling_score_seconds", "standardization_seconds",
                    "asymptotic_seconds", "core_total_seconds")
keys <- c("scenario", "n_total", "balance", "heterogeneity",
          "distribution", "df", "n_flips")
med <- aggregate(raw[timing_columns], raw[keys], median)
names(med)[match(timing_columns, names(med))] <- paste0("median_", timing_columns)
write.csv(med, file.path(results_dir, "timing_summary.csv"), row.names = FALSE)
cat(sprintf("wrote %d timing runs to %s\n", nrow(raw), results_dir))
