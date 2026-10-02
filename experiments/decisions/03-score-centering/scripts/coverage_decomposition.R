#!/usr/bin/env Rscript
# Decompose Wald coverage of the known target into point bias and reported
# variance error, per cell, from the frozen raw rows. No fitting.
args <- commandArgs(trailingOnly = TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/coverage_decomposition.R [RUN_ID ...]\n',
      'Default: ml_confirmation fiml_confirmation group_confirmation.\n',
      'Columns: coverage; bias/sd = mean error over the empirical SD of the\n',
      'estimate; var_ratio and med_var_ratio = mean and median reported\n',
      'variance over the empirical variance; se_cv = coefficient of variation of\n',
      'the reported SE; cover_true_sd, cover_const_se and cover_debiased =\n',
      'coverage with the empirical SD, with a constant SE equal to the root mean\n',
      'reported variance, and with the mean error removed.\n')
  quit(save = 'no')
}
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(dirname(script))
runs <- if (length(args)) args else c('ml_confirmation', 'fiml_confirmation', 'group_confirmation')
z <- qnorm(0.975)
out <- list()
for (run in runs) {
  if (!grepl('^[a-zA-Z0-9_-]+$', run)) stop('Invalid run ID')
  dir <- file.path(here, 'results', 'score-centering', run)
  files <- list.files(file.path(dir, 'raw'), pattern = '^cell_.*\\.csv$', full.names = TRUE)
  if (!length(files)) stop('Run the study first: no raw rows for ', run)
  cells <- read.csv(file.path(dir, 'cells.csv'), stringsAsFactors = FALSE)
  keep <- c('cell_id', 'rep', 'arm', 'fit_success', 'covariance_available',
            'estimate', 'target', 'variance')
  d <- do.call(rbind, lapply(files, function(f) read.csv(f, stringsAsFactors = FALSE)[, keep]))
  # The raw and centered arms share one estimate and covariance per replicate
  # (stationary equivalence), so one arm per replicate suffices.
  d <- d[d$arm %in% c('raw', 'global') & d$fit_success & d$covariance_available &
           is.finite(d$estimate) & is.finite(d$target) & is.finite(d$variance), ]
  d <- d[!duplicated(d[, c('cell_id', 'rep')]), ]
  for (id in sort(unique(d$cell_id))) {
    x <- d[d$cell_id == id, ]
    cl <- cells[cells$cell_id == id, ]
    e <- x$estimate - x$target
    se <- sqrt(x$variance)
    v <- var(x$estimate)
    out[[length(out) + 1L]] <- data.frame(
      run = run, cell = id, family = cl$family, n = cl$n, role = cl$role, reps = nrow(x),
      coverage = mean(abs(e) <= z * se), bias_sd = mean(e) / sd(x$estimate),
      var_ratio = mean(x$variance) / v, med_var_ratio = median(x$variance) / v,
      se_cv = sd(se) / mean(se), cover_true_sd = mean(abs(e) <= z * sd(x$estimate)),
      cover_const_se = mean(abs(e) <= z * sqrt(mean(x$variance))),
      cover_debiased = mean(abs(e - mean(e)) <= z * se))
  }
}
r <- do.call(rbind, out)
num <- vapply(r, is.double, logical(1))
r[num] <- lapply(r[num], round, 3)
options(width = 200)
print(r, row.names = FALSE)
