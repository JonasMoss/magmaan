#!/usr/bin/env Rscript
# Retrospective single-fit policies: selectors see sample moments/start values only.
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here, '..', '..', '_support', 'R', 'helpers.R'))
set_single_threaded_math()
for (f in c('designs.R', 'fit.R', 'start_design.R')) source(file.path(here, 'R', f))
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/select_single_start.R [--smoke] [--run-id NAME]\n',
      'Select one start from sample moments, then look up its previously saved fit.\n',
      'Requires direct-starts-information and start-development/start-fresh raw results.\n',
      'Policies: largest absolute off-diagonal eigenvalue per factor; retain constructor\n',
      'when both selected signs are positive; smallest initial ML objective among four\n',
      'spectral vectors. No fitted objective enters selection. ML only.\n', sep='')
  quit(save='no')
}
smoke <- '--smoke' %in% args
run <- if ('--run-id' %in% args) args[match('--run-id', args)+1L] else
  if (smoke) 'single-start-selection-smoke' else 'single-start-selection'
stopifnot(!is.na(run), grepl('^[a-zA-Z0-9_-]+$', run))
out <- file.path(here, 'results', run)
if (file.exists(file.path(out, 'summary.csv'))) stop('choose a fresh --run-id')
spec <- magmaanlab::model_spec(model_syntax)
selections <- list()
for (batch in c('development', 'fresh')) {
  tasks <- expand.grid(design=names(designs_all()), n=c(20L,100L), rep=1:10, stringsAsFactors=FALSE)
  tasks$seed <- (if (batch=='development') 202609281L else 902609281L) +
    match(tasks$design, names(designs_all()))*100000L + tasks$n*100L + tasks$rep
  if (smoke) tasks <- head(tasks, 6)
  for (i in seq_len(nrow(tasks))) {
    task <- tasks[i, ]
    sample <- sample_moments(draw_data(design_sigma(designs_all()[[task$design]]), task$n, task$seed))
    R <- cov2cor(sample$S[[1]])
    negative <- vapply(list(1:3, 4:6), function(idx) {
      A <- R[idx,idx]; diag(A) <- 0
      ev <- eigen(A, symmetric=TRUE, only.values=TRUE)$values
      abs(tail(ev,1)) > ev[1] # Exact ties choose positive.
    }, logical(1))
    signed <- c('spectral_positive','spectral_negative_x','spectral_negative_y','spectral_negative_xy')[1+sum(negative*c(1,2))]
    recipes <- start_recipes(spec, sample)
    objectives <- vapply(recipes, function(theta) {
      ev <- magmaanlab::magmaan_core$evaluate_at(spec$partable, sample, theta, estimator='ML')
      stopifnot(ev$diagnostics$admissibility$implied_sigma_pd, is.finite(ev$fmin))
      ev$fmin
    }, numeric(1))
    selections[[length(selections)+1L]] <- cbind(batch=batch, task,
      dominant_signed=signed, signed_override=if(any(negative)) signed else 'constructor',
      lowest_initial=names(which.min(objectives)),
      as.data.frame(as.list(setNames(objectives,paste0('initial_',names(objectives))))))
  }
}
selections <- do.call(rbind,selections)
# Outcome files are first read after every selection has been fixed.
ordinary <- read.csv(file.path(here,'results','direct-starts-information','comparisons.csv'))
ordinary <- ordinary[ordinary$domain=='ML',]
rows <- list()
for (batch in c('development','fresh')) {
  sphere <- read.csv(file.path(here,'results',paste0('start-',batch),'comparisons.csv'))
  for (route in c('ordinary','sphere')) {
    fits <- if(route=='ordinary') ordinary[ordinary$batch==batch,] else sphere
    base_name <- if(route=='ordinary') 'layered_native' else 'canonical'
    for (i in which(selections$batch==batch)) {
      sel <- selections[i,]
      for (backend in c('nlopt-lbfgs','port')) {
        z <- fits[fits$design==sel$design & fits$n==sel$n & fits$rep==sel$rep & fits$backend==backend,]
        base <- z[z$start_id==base_name,]
        stopifnot(nrow(base)==1)
        hit <- function(x) isTRUE(x$screened & is.finite(x$best_objective) &
          x$reference_gap <= 1e-6*(1+abs(x$best_objective)))
        for (policy in c('dominant_signed','signed_override','lowest_initial')) {
          arm <- sel[[policy]]; if(arm=='constructor') arm <- base_name
          candidate <- z[z$start_id==arm,]; stopifnot(nrow(candidate)==1)
          bh <- hit(base); ch <- hit(candidate)
          rows[[length(rows)+1L]] <- cbind(sel[c('batch','design','n','rep','seed')],
            route=route, backend=backend, policy=policy, selected_start=arm,
            reference_available=is.finite(base$best_objective), baseline_hit=bh, selected_hit=ch,
            gain=!bh & ch, loss=bh & !ch, baseline_verdict=isTRUE(base$original_verdict),
            selected_verdict=isTRUE(candidate$original_verdict),
            baseline_label=base$label, selected_label=candidate$label,
            baseline_objective=base$objective, selected_objective=candidate$objective)
        }
      }
    }
  }
}
paired <- do.call(rbind,rows)
metrics <- c('reference_available','baseline_hit','selected_hit','gain','loss','baseline_verdict','selected_verdict')
summary <- aggregate(paired[metrics],paired[c('batch','route','backend','policy')],sum)
by_family <- aggregate(paired[metrics],paired[c('batch','design','route','backend','policy')],sum)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
for (name in c('selections','paired','summary','by_family'))
  write.csv(get(name),file.path(out,paste0(name,'.csv')),row.names=FALSE)
write_metadata(file.path(out,'metadata.csv'), values=list(
  tasks=nrow(selections), smoke=smoke, domain='unrestricted ML',
  selection='sample-only; deterministic ties; no outcome-based tuning; no new optimization',
  seed_rule='development 202609281; fresh 902609281; +design_index*100000+N*100+rep',
  status='retrospective exploratory; both batches previously inspected; not held out',
  judge='frozen reference-hit screen includes historical extent cutoff; verdict also retained',
  ordinary='layered native / information scaling; sphere canonical / diagonal scaling; polish off',
  strength=.5, git_head=magmaan_cache_ref()$git_head),packages='magmaanlab')
print(summary,row.names=FALSE)
cat('Wrote ',out,'\n',sep='')
