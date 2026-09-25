#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/native_restart_probe.R [results-directory]\n',
      'Cross-restart both methods from each discrepant endpoint saved by native_structure_probe.R.\n',
      'Four fits per discrepancy, 30s soft budget, 1000 evaluations per fit.\n',sep='')
  quit()
}
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script)); repo <- normalizePath(file.path(here,'../../..'))
source(file.path(repo,'experiments/_support/R/helpers.R')); set_single_threaded_math()
results <- if(length(args)) args[1] else file.path(here,'results/native-structures')
path <- file.path(results,'discrepancies.rds')
if (!file.exists(path)) stop('No saved discrepancies; run native_structure_probe.R first')
cases <- readRDS(path); rows <- list(); t0 <- Sys.time()
elapsed <- function() as.numeric(difftime(Sys.time(),t0,units='secs'))
unlink(file.path(results,'restarts.csv'))
for (d in cases) for (from in c('none','diagonal')) for (method in c('none','diagonal')) {
  if (elapsed() > 30) stop('Restart soft budget exceeded; partial results saved')
  spec <- d$spec; k <- spec$partable$free; free <- k > 0
  spec$partable$ustart[free] <- d$fits[[from]]$theta[k[free]]
  tick <- elapsed()
  f <- tryCatch(magmaanlab::frontier_fit_ml_psd(spec,d$stats,preconditioning=method,
    control=list(max_iter=1000L,gtol=1e-8,ftol=1e-12)),error=function(e)e)
  seconds <- elapsed()-tick; err <- inherits(f,'error')
  g <- if(err) NULL else f$diagnostics$geometric_stationarity
  rows[[length(rows)+1L]] <- data.frame(case=d$case,setting=d$setting,replicate=d$replicate,
    from=from,preconditioning=method,initial_objective=2*d$fits[[from]]$fmin,
    objective=if(err) NA_real_ else 2*f$fmin,seconds=seconds,
    evaluations=if(err) NA_integer_ else f$f_evals,
    accepted=!err && isTRUE(f$converged) && isTRUE(f$diagnostics$admissibility$admissible) && isTRUE(g$cone_stationary),
    nullity=if(err) NA_integer_ else g$covariance_nullity,
    error=if(err) conditionMessage(f) else '')
  write.csv(do.call(rbind,rows),file.path(results,'restarts.csv'),row.names=FALSE)
}
write_metadata(file.path(results,'restart_metadata.csv'),values=list(elapsed_seconds=elapsed(),
  completed_fits=length(rows),git_head=git_scalar(c('rev-parse','HEAD'),root=repo),
  source_md5=unname(tools::md5sum(script)),input_md5=unname(tools::md5sum(path))),packages='magmaanlab')
print(do.call(rbind,rows)); cat('Results: ',file.path(results,'restarts.csv'),'\n',sep='')
