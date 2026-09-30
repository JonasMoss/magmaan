#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here, '..', '..', '..', '_support', 'R', 'helpers.R'))
set_single_threaded_math()
for (f in c('designs.R', 'fit.R', 'start_design.R')) source(file.path(here, 'R', f))
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/direct_starts.R [--smoke] [--run-id NAME]\n',
      'Fits every recipe in the ordinary marker chart using sample moments only.\n',
      '60 development + 60 previously retained fresh draws; smoke uses 6 per batch.\n',
      'ML: layered native and four spectral starts, L-BFGS and PORT.\n',
      'PSD: FABIN3 auto transport and positive spectral, SLSQP, none/diagonal scaling.\n',
      'Saved reference CSVs are read only AFTER all fitting; no sphere optimization.\n', sep='')
  quit(save='no')
}
smoke <- '--smoke' %in% args
run <- if ('--run-id' %in% args) args[match('--run-id', args)+1L] else
  if (smoke) 'direct-starts-smoke' else 'direct-starts-information'
stopifnot(!is.na(run), grepl('^[a-zA-Z0-9_-]+$', run))
out <- file.path(here, 'results', run)
if (file.exists(file.path(out, 'fits.csv'))) stop('choose a fresh --run-id')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
write_out <- function(x, name) write.csv(x, file.path(out, paste0(name, '.csv')), row.names=FALSE)
spec <- magmaanlab::model_spec(model_syntax)
rows <- starts <- list(); t0 <- proc.time()[['elapsed']]
for (batch in c('development', 'fresh')) {
  tasks <- expand.grid(design=names(designs_all()), n=c(20L, 100L), rep=1:10, stringsAsFactors=FALSE)
  tasks$seed <- (if (batch=='development') 202609281L else 902609281L) +
    match(tasks$design, names(designs_all()))*100000L + tasks$n*100L + tasks$rep
  if (smoke) tasks <- head(tasks, 6)
  for (i in seq_len(nrow(tasks))) {
    task <- tasks[i, ]
    data <- draw_data(design_sigma(designs_all()[[task$design]]), task$n, task$seed)
    sample <- sample_moments(data)
    recipes <- start_recipes(spec, sample)
    for (name in names(recipes)) {
      theta <- recipes[[name]]
      ev <- magmaanlab::magmaan_core$evaluate_at(spec$partable, sample, theta, estimator='ML')
      stopifnot(is.finite(ev$fmin), ev$diagnostics$admissibility$implied_sigma_pd)
      starts[[length(starts)+1L]] <- cbind(batch=batch, task[rep(1L, length(theta)), ],
        start_id=name, parameter=seq_along(theta), value=theta, start_objective=ev$fmin)
    }
    for (domain in c('ML', 'PSD')) {
      constructor <- if (domain=='ML') 'layered_native' else 'fabin3_auto'
      arms <- c(constructor, if (domain=='ML') names(recipes) else 'spectral_positive')
      for (backend in if (domain=='ML') c('nlopt-lbfgs', 'port') else 'nlopt-slsqp') {
        for (scaling in if (domain=='ML') 'information' else c('none', 'diagonal')) {
          for (arm in arms) {
            theta <- recipes[[arm]]
            ctl <- if (arm=='layered_native') list(start='layered', start_transport='native') else
              if (arm=='fabin3_auto') list(start='fabin3', start_transport='auto') else list(start_transport='native')
            if (domain=='ML') ctl$coordinate_scaling <- 'information'
            z <- run_fit(spec, data, sample, domain, 'ordinary', backend, arm, theta,
                         preconditioning=if (domain=='PSD') scaling else 'none', control=ctl)
            id <- length(rows)+1L
            rows[[id]] <- cbind(fit_id=id, batch=batch, task, transform='native', domain=domain,
              route='ordinary', backend=backend, preconditioning=scaling, start_id=arm, z$record)
          }
        }
      }
    }
    cat(sprintf('%s [%d/%d] %.1fs\n', batch, i, nrow(tasks), proc.time()[['elapsed']]-t0))
  }
}
fits <- do.call(rbind, rows)
write_out(fits, 'fits'); write_out(do.call(rbind, starts), 'starts')
# The reference is evaluation-only: no endpoint or reference flag enters a fit.
refs <- lapply(c('development', 'fresh'), function(batch) {
  x <- read.csv(file.path(here, 'results', paste0('start-', batch), 'references.csv'))
  if (batch=='development') {
    psd <- read.csv(file.path(here, 'results', 'reference-pilot', 'references.csv'))
    x <- rbind(x, psd[psd$domain=='PSD', names(x)])
  }
  cbind(batch=batch, x)
})
refs <- do.call(rbind, refs)
write_out(refs, 'references')
cmp <- do.call(rbind, lapply(c('development', 'fresh'), function(b)
  compare_rows(fits[fits$batch==b, ], refs[refs$batch==b, setdiff(names(refs), 'batch')])))
write_out(cmp, 'comparisons')
keys <- c('batch','design','n','rep','domain','backend','preconditioning')
paired <- lapply(split(cmp, interaction(cmp[keys], drop=TRUE)), function(z) {
  base <- z[z$start_id %in% c('layered_native','fabin3_auto'), ]
  good <- z$screened & is.finite(z$best_objective) & z$reference_gap <= 1e-6*(1+abs(z$best_objective))
  spectral <- grepl('^spectral_', z$start_id)
  cbind(base[c(keys,'seed','comparison','reference_label','chart_status')],
    reference_available=is.finite(base$best_objective),
    constructor_hit=any(good & !spectral), spectral_hit=any(good & spectral), portfolio_hit=any(good),
    any_screened=any(z$screened), best_screened=if(any(z$screened)) min(z$objective[z$screened]) else NA_real_)
})
paired <- do.call(rbind, paired); write_out(paired, 'paired')
summary <- aggregate(paired[c('reference_available','constructor_hit','spectral_hit','portfolio_hit','any_screened')],
  paired[c('batch','domain','backend','preconditioning')], sum)
summary[summary$reference_available==0, c('constructor_hit','spectral_hit','portfolio_hit')] <- NA_integer_
write_out(summary, 'summary')
write_out(aggregate(list(fits=cmp$fit_id), cmp[c('batch','domain','backend','preconditioning','start_id','label','comparison')], length), 'labels')
ref <- magmaan_cache_ref()
write_metadata(file.path(out,'metadata.csv'), values=list(tasks=nrow(fits)/14, fits=nrow(fits),
  seed_rule='development 202609281; fresh 902609281; +design_index*100000+N*100+rep',
  selection='all recipes on all draws; sample moments only; no reference-dependent fitting',
  references='read after fitting; development ML/PSD, fresh ML only; missing fresh PSD target is not failure',
  strength=.5, ml_start='layered native; four spectral marker vectors',
  psd_start='FABIN3 auto std.lv transport; positive spectral marker vector',
  scaling='ML information; PSD none and diagonal',
  git_head=ref$git_head, git_dirty=ref$git_dirty,
  magmaanlab_built=utils::packageDescription('magmaanlab')$Built), packages='magmaanlab')
cat('Wrote ', out, '\n', sep='')
