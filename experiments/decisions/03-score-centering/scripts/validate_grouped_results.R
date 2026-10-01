#!/usr/bin/env Rscript
# Independently replay the declared sampling law; no fitting or SEM evaluation.
args <- commandArgs(trailingOnly = TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/validate_grouped_results.R [RUN_ID]\n',
      'Default group_confirmation. Replays seeds and checks pooled-mean estimates,\n',
      'scalar covariance and score p-values against closed-form normal-model formulas.\n')
  quit(save = 'no')
}
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here, '..', '..', '_support', 'R', 'helpers.R'))
source(file.path(here, 'R', 'cells.R'))
run_id <- if (length(args)) args[1] else 'group_confirmation'
if (!grepl('^[a-zA-Z0-9_-]+$', run_id)) stop('Invalid run ID')
out <- file.path(here, 'results', 'score-centering', run_id)
files <- list.files(file.path(out, 'raw'), pattern = '^cell_.*\\.csv$', full.names = TRUE)
if (!length(files)) stop('Run the study first: no raw rows for ', run_id)
rows <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE))
rows <- rows[rows$family == 'grouped_mean', ]
cells <- centering_cells()
checked <- list()
for (at in split(seq_len(nrow(rows)), paste(rows$cell_id, rows$rep))) {
  r <- rows[at, ]; c <- cells[cells$cell_id == r$cell_id[1], ]
  pop <- centering_population(c, r$seed[1])
  x <- pop$data$x1; n <- length(x); mean_x <- mean(x)
  within <- x - ave(x, pop$data$group, FUN = mean)
  for (i in seq_len(nrow(r))) {
    v <- if (r$arm[i] == 'group') sum(within^2) / n^2 else sum((x - mean_x)^2) / n^2
    denom <- switch(r$arm[i], raw = sum(x^2), global = sum((x - mean_x)^2), group = sum(within^2))
    p <- pchisq(n^2 * mean_x^2 / denom, 1, lower.tail = FALSE)
    checked[[length(checked) + 1L]] <- data.frame(n = n, role = r$role[i], arm = r$arm[i],
      available = r$covariance_available[i] && r$test_available[i],
      estimate_gap = abs(r$estimate[i] - mean_x), variance_gap = abs(r$variance[i] - v),
      statistic_gap = abs(r$statistic[i] - n * mean_x^2),
      sb_gap = abs(r$p_sb[i] - p), peba4_gap = abs(r$p_peba4[i] - p))
  }
}
x <- do.call(rbind, checked)
summary <- do.call(rbind, lapply(split(seq_len(nrow(x)), paste(x$n, x$role, x$arm)), function(at) {
  r <- x[at, ]
  data.frame(r[1, c('n', 'role', 'arm')], draws = nrow(r), unavailable = sum(!r$available),
    as.list(vapply(r[c('estimate_gap', 'variance_gap', 'statistic_gap', 'sb_gap', 'peba4_gap')],
                  function(v) max(v, na.rm = TRUE), numeric(1))))
}))
write_csv(summary, file.path(out, 'oracle_checks.csv'))
meta <- read.csv(file.path(out, 'metadata.csv'), stringsAsFactors = FALSE)
meta <- meta[!meta$key %in% c('oracle_validator_md5', 'oracle_validation_at_utc'), ]
meta <- rbind(meta, data.frame(key = c('oracle_validator_md5', 'oracle_validation_at_utc'),
  value = c(unname(tools::md5sum(script)), format(Sys.time(), tz = 'UTC', usetz = TRUE))))
write_csv(meta, file.path(out, 'metadata.csv'))
if (any(!x$available) || any(as.matrix(x[, c('estimate_gap', 'variance_gap', 'statistic_gap', 'sb_gap', 'peba4_gap')]) > 1e-7))
  stop('Grouped closed-form checks failed; see oracle_checks.csv')
cat('Grouped closed-form estimate/covariance/statistic/SB/PEBA4 checks passed: ', nrow(x), ' rows.\n', sep = '')
