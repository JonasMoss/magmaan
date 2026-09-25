#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/native_structure_probe.R [--smoke] [--reps N] [--budget-sec N] [--seed-base N] [--results-dir PATH] [--dry-run-plan]\n',
      'Six structures; ordinary/stressed synthetic settings; native starts only.\n',
      'Default: 10 settings x (reference + 10 Gaussian draws) x 2 methods = 220 fits.\n',
      'Smoke: one draw per setting (40 fits). Soft budget 30s, checked between fits.\n', sep='')
  quit()
}
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1]))
here <- dirname(dirname(script)); repo <- normalizePath(file.path(here, '../../..'))
o <- list(reps=10L, budget_sec=30, seed_base=20260919L,
          results_dir=file.path(here,'results/native-structures'), plan=FALSE)
i <- 1L
while (i <= length(args)) {
  a <- args[i]
  if (a == '--smoke') o$reps <- 1L else if (a == '--dry-run-plan') o$plan <- TRUE else {
    if (i == length(args)) stop('Missing value: ',a)
    i <- i+1L; key <- gsub('-','_',sub('^--','',a))
    if (!key %in% c('reps','budget_sec','seed_base','results_dir')) stop('Unknown option: ',a)
    o[[key]] <- if (key == 'results_dir') args[i] else as.numeric(args[i])
  }
  i <- i+1L
}
stopifnot(is.finite(o$reps), o$reps >= 0, o$reps == as.integer(o$reps),
          is.finite(o$budget_sec), o$budget_sec > 0,
          is.finite(o$seed_base), o$seed_base >= 0, o$seed_base < 2e9,
          o$seed_base == as.integer(o$seed_base))
ids <- c('hs_3factor_cfa','mediation','correlated_predictors','higher_order',
         'correlated_disturbances','bollen_democracy_sem')
grid <- do.call(rbind,lapply(ids,function(id) data.frame(case=id,
  setting=if (id %in% ids[c(1,6)]) 'observed' else c('ordinary','stress'))))
planned <- nrow(grid)*(o$reps+1)*2
cat(sprintf('%d settings, %d planned fits, %.0fs soft budget\n',nrow(grid),planned,o$budget_sec))
if (o$plan) quit()
source(file.path(repo,'experiments/_support/R/helpers.R')); set_single_threaded_math()
source(file.path(here,'R/charts.R')); source(file.path(here,'R/models.R'))
stopifnot('preconditioning' %in% names(formals(magmaanlab::frontier_fit_ml_psd)))
dir.create(o$results_dir,recursive=TRUE,showWarnings=FALSE)
unlink(file.path(o$results_dir,c('fits.csv','pairs.csv','validation.csv','metadata.csv','discrepancies.rds','restarts.csv','restart_metadata.csv')))
t0 <- Sys.time(); elapsed <- function() as.numeric(difftime(Sys.time(),t0,units='secs'))
rows <- validations <- discrepancies <- list(); exhausted <- FALSE
for (cell in seq_len(nrow(grid))) {
  if (elapsed() > o$budget_sec) { exhausted <- TRUE; break }
  id <- grid$case[cell]; setting <- grid$setting[cell]
  if (setting == 'observed') {
    case <- empirical_case(id,repo)
    syntax <- paste(readLines(file.path(repo,'benchmarks/cases',id,'model.lav')),collapse='\n')
  } else {
    case <- synthetic_case(id,setting == 'stress')
    syntax <- paste('f1 =~ x1 + x2 + x3','f2 =~ x4 + x5 + x6','f3 =~ x7 + x8 + x9',
      switch(id, mediation='f2 ~ f1\nf3 ~ f1 + f2',
        correlated_predictors='f3 ~ f1 + f2\nf1 ~~ f2',
        correlated_disturbances='f2 ~ f1\nf3 ~ f1\nf2 ~~ f3\nx2 ~~ x5',
        higher_order='g =~ f1 + f2 + f3'),sep='\n')
    ov <- paste0('x',1:9); lv <- c('f1','f2','f3',if(id == 'higher_order') 'g')
    dimnames(case$m$lambda) <- list(ov,lv)
    dimnames(case$m$beta) <- dimnames(case$m$psi) <- list(lv,lv)
    dimnames(case$m$theta) <- list(ov,ov); dimnames(case$S) <- list(ov,ov)
  }
  spec <- magmaanlab::model_spec(syntax,fixed_x=FALSE,auto_cov_lv_x=(setting == 'observed'))
  # Validate the syntax and matrix-to-partable mapping at known model moments.
  map <- chart_map(case$m,case$mask,'marker'); m <- map$unpack(map$pack(case$m))$m
  pt <- spec$partable
  values <- vapply(seq_len(nrow(pt)),function(k) {
    p <- pt[k,]
    if (p$op == '=~') return(if(p$rhs %in% rownames(m$lambda)) m$lambda[p$rhs,p$lhs] else m$beta[p$rhs,p$lhs])
    if (p$op == '~') return(m$beta[p$lhs,p$rhs])
    if (p$op != '~~') stop('Unsupported row')
    V <- if(p$lhs %in% rownames(m$psi)) m$psi else m$theta
    V[p$lhs,p$rhs]
  },0.0)
  theta <- values[match(seq_len(max(pt$free)),pt$free)]
  population <- implied(case$m)$Sigma
  dimnames(population) <- dimnames(case$S)
  stats <- function(S) list(S=list(S),mean=list(rep(0,nrow(S))),nobs=case$n)
  check <- magmaanlab::magmaan_core$evaluate_at(spec,stats(population),theta,estimator='ULS')
  stopifnot(is.finite(check$fmin),abs(check$fmin)<1e-18)
  validations[[cell]] <- data.frame(case=id,setting=setting,p=nrow(population),
    free_parameters=length(theta),population_uls=check$fmin)
  write.csv(do.call(rbind,validations),file.path(o$results_dir,'validation.csv'),row.names=FALSE)
  for (r in 0:o$reps) {
    S <- case$S
    seed <- o$seed_base + match(id,ids)*10000L + r
    if (r > 0) { set.seed(seed); S <- rWishart(1,case$n-1,population)[,,1]/case$n }
    dimnames(S) <- dimnames(case$S)
    fitted <- list()
    for (method in if ((r+cell) %% 2 == 0) c('none','diagonal') else c('diagonal','none')) {
      if (elapsed() > o$budget_sec) { exhausted <- TRUE; break }
      tick <- elapsed()
      f <- tryCatch(magmaanlab::frontier_fit_ml_psd(spec,stats(S),preconditioning=method,
        control=list(max_iter=1000L,gtol=1e-8,ftol=1e-12)),error=function(e)e)
      seconds <- elapsed()-tick; err <- inherits(f,'error')
      fitted[[method]] <- f
      g <- if(err) NULL else f$diagnostics$geometric_stationarity
      rows[[length(rows)+1L]] <- data.frame(case=id,setting=setting,replicate=r,seed=seed,
        preconditioning=method,objective=if(err) NA_real_ else 2*f$fmin,seconds=seconds,
        evaluations=if(err) NA_integer_ else f$f_evals,
        accepted=!err && isTRUE(f$converged) && isTRUE(f$diagnostics$admissibility$admissible) && isTRUE(g$cone_stationary),
        nullity=if(err) NA_integer_ else g$covariance_nullity,
        cone_residual=if(err) NA_real_ else g$cone_residual_l2,
        constraint_violation=if(err) NA_real_ else f$audit$constraint_violation_inf,
        error=if(err) conditionMessage(f) else '')
      write.csv(do.call(rbind,rows),file.path(o$results_dir,'fits.csv'),row.names=FALSE)
    }
    if (length(fitted) == 2 && all(vapply(fitted,function(f) !inherits(f,'error'),TRUE)) &&
        abs(fitted$none$fmin-fitted$diagonal$fmin) > 5e-8) {
      discrepancies[[length(discrepancies)+1L]] <- list(case=id,setting=setting,replicate=r,
        spec=spec,stats=stats(S),fits=fitted)
      saveRDS(discrepancies,file.path(o$results_dir,'discrepancies.rds'))
    }
    if (exhausted) break
  }
  cat(sprintf('%s %s: %d/%d fits, %.2fs elapsed\n',id,setting,length(rows),planned,elapsed())); flush.console()
  if (exhausted) break
}
if (length(rows)) {
  d <- do.call(rbind,rows)
  pairs <- reshape(d[c('case','setting','replicate','preconditioning','objective','accepted','nullity','evaluations','seconds')],
    idvar=c('case','setting','replicate'),timevar='preconditioning',direction='wide')
  if (all(c('objective.none','objective.diagonal') %in% names(pairs))) {
    pairs$objective_gap <- abs(pairs$objective.none-pairs$objective.diagonal)
    pairs$comparable <- with(pairs,accepted.none & accepted.diagonal &
      objective_gap < 1e-7 & nullity.none == nullity.diagonal)
  }
  write.csv(pairs,file.path(o$results_dir,'pairs.csv'),row.names=FALSE)
}
write_metadata(file.path(o$results_dir,'metadata.csv'),values=c(o,list(
  complete=!exhausted,elapsed_seconds=elapsed(),planned_fits=planned,completed_fits=length(rows),
  git_head=git_scalar(c('rev-parse','HEAD'),root=repo),
  source_md5=paste(tools::md5sum(c(script,file.path(here,'R/charts.R'),file.path(here,'R/models.R'))),collapse=','))),
  packages=c('magmaanlab','lavaan'))
cat('Results: ',o$results_dir,'\n',sep='')
if (exhausted) quit(status=2L)
