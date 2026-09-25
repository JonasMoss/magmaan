#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/native_psd_probe.R [--smoke] [--reps N] [--budget-sec N] [--results-dir PATH]\n',
      'Compare native PSD ML without/with diagonal preconditioning, from common/native starts.\n',
      'Default: original Bollen data plus ten matched draws, 44 fits, 30s soft budget.\n',
      'Smoke: original data plus draws 1 and 2. No repeated timing batches.\n', sep='')
  quit()
}
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1]))
here <- dirname(dirname(script)); repo <- normalizePath(file.path(here, '../../..'))
o <- list(reps=10L, budget_sec=30, results_dir=file.path(here, 'results/native'))
i <- 1L
while (i <= length(args)) {
  a <- args[i]
  if (a == '--smoke') o$reps <- 2L else {
    if (i == length(args)) stop('Missing value: ', a)
    i <- i + 1L; key <- gsub('-', '_', sub('^--', '', a))
    if (!key %in% names(o)) stop('Unknown option: ', a)
    o[[key]] <- if (key == 'results_dir') args[i] else as.numeric(args[i])
  }
  i <- i + 1L
}
stopifnot(is.finite(o$reps), o$reps >= 0, o$reps == as.integer(o$reps),
          is.finite(o$budget_sec), o$budget_sec > 0)
source(file.path(repo, 'experiments/_support/R/helpers.R')); set_single_threaded_math()
source(file.path(here, 'R/charts.R')); source(file.path(here, 'R/models.R'))
if (!'preconditioning' %in% names(formals(magmaanlab::frontier_fit_ml_psd)))
  stop('Reinstall magmaan with just r-dev; native preconditioning is missing')
dir.create(o$results_dir, recursive=TRUE, showWarnings=FALSE)
unlink(file.path(o$results_dir, c('fits.csv', 'metadata.csv')))
t0 <- Sys.time(); elapsed <- function() as.numeric(difftime(Sys.time(), t0, units='secs'))
case <- empirical_case('bollen_democracy_sem', repo)
seed <- 20260919L + 60000L; start <- common_start(case, seed)
map <- chart_map(case$m, case$mask, 'marker')
m <- map$unpack(map$pack(start))$m
syntax <- paste(readLines(file.path(repo, 'benchmarks/cases/bollen_democracy_sem/model.lav')), collapse='\n')
spec <- magmaanlab::model_spec(syntax, fixed_x=FALSE); warm <- spec
warm$partable$ustart <- vapply(seq_len(nrow(warm$partable)), function(k) {
  p <- warm$partable[k, ]
  if (p$op == '=~') return(m$lambda[p$rhs, p$lhs])
  if (p$op == '~') return(m$beta[p$lhs, p$rhs])
  if (p$op != '~~') stop('Unsupported model row')
  V <- if (p$lhs %in% rownames(m$psi)) m$psi else m$theta
  V[p$lhs, p$rhs]
}, 0.0)
rows <- list(); exhausted <- FALSE
for (r in 0:o$reps) {
  S <- case$S
  if (r > 0) { set.seed(seed+r); S <- rWishart(1, case$n-1, implied(case$m)$Sigma)[,,1]/case$n }
  dimnames(S) <- dimnames(case$S)
  for (st in c('common', 'native')) for (method in if (r %% 2 == 0) c('none','diagonal') else c('diagonal','none')) {
    if (elapsed() > o$budget_sec) { exhausted <- TRUE; break }
    tick <- elapsed()
    f <- tryCatch(magmaanlab::frontier_fit_ml_psd(if (st == 'common') warm else spec,
      list(S=list(S), mean=list(rep(0,nrow(S))), nobs=case$n),
      preconditioning=method, control=list(max_iter=1000L, gtol=1e-8, ftol=1e-12)),
      error=function(e)e)
    seconds <- elapsed()-tick; err <- inherits(f,'error')
    if (!err) stopifnot(identical(f$psd_preconditioning, method))
    g <- if (err) NULL else f$diagnostics$geometric_stationarity
    rows[[length(rows)+1L]] <- data.frame(replicate=r, start=st, preconditioning=method,
      objective=if(err) NA else 2*f$fmin, seconds=seconds,
      evaluations=if(err) NA else f$f_evals, iterations=if(err) NA else f$iterations,
      accepted=!err && isTRUE(f$converged) && isTRUE(f$diagnostics$admissibility$admissible) && isTRUE(g$cone_stationary),
      nullity=if(err) NA_integer_ else g$covariance_nullity,
      constraint_violation=if(err) NA else f$audit$constraint_violation_inf,
      cone_residual=if(err) NA else g$cone_residual_l2,
      error=if(err) conditionMessage(f) else '')
    write.csv(do.call(rbind, rows), file.path(o$results_dir,'fits.csv'), row.names=FALSE)
    cat(sprintf('%d/%d draw=%d %s %s %.3fs elapsed\n', length(rows),4*(o$reps+1),r,st,method,elapsed()));flush.console()
  }
  if (exhausted) break
}
write_metadata(file.path(o$results_dir,'metadata.csv'), values=c(o,list(
  complete=!exhausted, elapsed_seconds=elapsed(), seed_base=seed,
  planned_fits=4*(o$reps+1), completed_fits=length(rows),
  git_head=git_scalar(c('rev-parse','HEAD'),root=repo),
  source_md5=paste(tools::md5sum(c(script,file.path(here,'R/charts.R'),file.path(here,'R/models.R'))),collapse=','))),
  packages=c('magmaanlab','lavaan'))
cat('Results: ',o$results_dir,'\n',sep='')
if(exhausted) quit(status=2L)
