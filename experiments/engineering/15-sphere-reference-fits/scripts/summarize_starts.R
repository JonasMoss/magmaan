#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value = TRUE)[1]))
here <- dirname(dirname(script))
if ('--help' %in% commandArgs(TRUE)) {
  cat('Usage: Rscript scripts/summarize_starts.R\nReads start-development and start-fresh raw comparisons; writes compact paired evidence.\n')
  quit(save = 'no')
}
for (phase in c('development', 'fresh')) {
  out <- file.path(here, 'results', paste0('start-', phase))
  f <- read.csv(file.path(out, 'comparisons.csv'))
  portfolios <- list(canonical = 'canonical',
    canonical_single_negative = c('canonical', 'spectral_negative_x', 'spectral_negative_y'),
    canonical_signed = c('canonical', 'spectral_negative_x', 'spectral_negative_y', 'spectral_negative_xy'),
    all_five = unique(f$start_id))
  rows <- list()
  for (backend in c('both', 'port', 'nlopt-lbfgs')) for (p in names(portfolios)) {
    d <- f[f$start_id %in% portfolios[[p]] & (backend == 'both' | f$backend == backend), ]
    groups <- split(d, interaction(d$design, d$n, d$rep, drop = TRUE))
    for (z in groups) rows[[length(rows) + 1L]] <- data.frame(
      phase = phase, design = z$design[1], n = z$n[1], rep = z$rep[1], seed = z$seed[1],
      backend = backend, portfolio = p, attempts = nrow(z),
      has_candidate = any(z$screened), matches_best_observed = any(z$comparison == 'matches_sphere_reference'),
      seconds = sum(z$seconds))
  }
  paired <- do.call(rbind, rows)
  write.csv(paired, file.path(out, 'coverage.csv'), row.names = FALSE)
  summary <- aggregate(paired[c('attempts', 'has_candidate', 'matches_best_observed', 'seconds')],
                       paired[c('phase', 'design', 'backend', 'portfolio')], sum)
  write.csv(summary, file.path(out, 'coverage_summary.csv'), row.names = FALSE)
}
