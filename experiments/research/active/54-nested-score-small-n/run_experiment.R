#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Development score bootstrap: --reps 40 --bootstrap 199 --workers 2 --seed-base 1126100091 --run-id pilot\n--smoke uses 2 draws/cell and 9 resamples. --max-seconds 1100 caps batch starts. Fresh directories only.\nRequires installed magmaan, magmaanlab, lavaan; no package installation.\n')
  quit(save='no')
}
opt <- function(k,d) { i <- match(k,args); if(is.na(i)) d else args[i+1] }
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(script)
source(file.path(here,'../../../_support/R/helpers.R'))
set_single_threaded_math()
smoke <- '--smoke' %in% args
reps <- as.integer(opt('--reps',if(smoke) '2' else '40'))
boot <- as.integer(opt('--bootstrap',if(smoke) '9' else '199'))
workers <- as.integer(opt('--workers','2')); base <- as.integer(opt('--seed-base',if(smoke) '1026100091' else '1126100091'))
limit <- as.numeric(opt('--max-seconds','1100'))
stopifnot(reps>=2,boot>=9,workers %in% 1:2,limit>0,limit<=1100)
out <- file.path(here,'results',opt('--run-id',if(smoke) 'smoke' else 'pilot'))
if(dir.exists(out)) stop('Choose a fresh run-id')
dir.create(file.path(out,'raw'),recursive=TRUE)
# Design constants from decisions/04 registration; no sibling code/result reads.
cells <- data.frame(cell_id=c(1,5,9,13),distribution=rep(c('normal','skewed'),2),larger=rep(c('correct','mild'),each=2),n=100L)
syntax <- 'f1 =~ x1+x2+x3+x4\nf2 =~ x5+x6+x7+x8'
population <- function(cell) {
  l <- rep(c(.7,.8,.75,.65),2); v <- 1-l^2
  paste(c(paste0('f1 =~ ',paste0(l[1:4],'*x',1:4,collapse=' + ')),
    paste0('f2 =~ ',paste0(l[5:8],'*x',5:8,collapse=' + ')),
    'f1 ~~ 1*f1','f2 ~~ 1*f2','f1 ~~ .4*f2',paste0('x',1:8,' ~~ ',v,'*x',1:8),
    if(cell$larger=='mild') paste0('x1 ~~ ',.3*sqrt(v[1]*v[5]),'*x5')),collapse='\n')
}
draw <- function(cell,seed) {
  set.seed(seed)
  do.call(rbind,lapply(c('a','b'),function(g) {
    d <- lavaan::simulateData(population(cell),sample.nobs=100,
      skewness=if(cell$distribution=='skewed') rep(2,8) else NULL,
      kurtosis=if(cell$distribution=='skewed') rep(7,8) else NULL)
    d$group <- g; d
  }))
}
components <- function(d) {
  fit <- function(eq=NULL) magmaan::as_lab_fit(magmaan::magmaan(
    magmaan::magmaan_model(syntax,prototype=d,group='group',group.equal=eq),d,estimator='ML'))
  f0 <- fit('loadings'); f1 <- fit()
  if(!isTRUE(f0$converged)||!isTRUE(f1$converged)) stop('library convergence verdict failed')
  shared <- magmaanlab::prepare_inference_data(f1)
  i0 <- magmaanlab::prepare_inference(f0,shared); i1 <- magmaanlab::prepare_inference(f1,shared)
  list(c=magmaanlab::score_components(i0,H1=f1),q=magmaanlab::inference_quadratic(magmaanlab::prepare_hypothesis(i0,i1),'score','observed'))
}
geometry <- function(c) {
  k <- c$nuisance; d <- c$directions
  g <- d-k%*%solve(crossprod(k,c$sensitivity%*%k),crossprod(k,c$sensitivity%*%d))
  list(g=g,m=crossprod(g,c$metric%*%g))
}
quadratic <- function(s,a) as.numeric(crossprod(crossprod(a$g,s),solve(a$m,crossprod(a$g,s))))
started <- proc.time()[['elapsed']]
one <- function(i,r) {
  cell <- cells[i,]; seed <- base+10000L*cell$cell_id+r
  errors <- data.frame(cell_id=integer(),rep=integer(),bootstrap=integer(),seed=integer(),error=character())
  row <- data.frame(cell_id=cell$cell_id,rep=r,seed=seed,statistic=NA_real_,policy_gap=NA_real_,p_current_sb=NA_real_,p_current_peba4=NA_real_,p_fixed=NA_real_,p_refit=NA_real_,bootstrap_ok=0L,elapsed_s=NA_real_)
  t0 <- proc.time()[['elapsed']]
  fail <- function(b,e) errors <<- rbind(errors,data.frame(cell_id=cell$cell_id,rep=r,bootstrap=b,seed=seed,error=conditionMessage(e)))
  tryCatch({
    d <- draw(cell,seed); z <- components(d); a <- geometry(z$c)
    row$statistic <- quadratic(z$c$score,a); row$policy_gap <- abs(row$statistic-z$q$statistic)
    if(row$policy_gap>1e-7) stop('independent quadratic differs from policy')
    p <- magmaanlab::calibrate_quadratic(z$q,c('sb','peba4'))
    row$p_current_sb <- p$p_value[1]; row$p_current_peba4 <- p$p_value[2]
    fixed <- refit <- rep(NA_real_,boot)
    for(b in seq_len(boot)) {
      # Reset before each index draw: failed fits cannot shift later resamples.
      set.seed(seed+1000000L+b)
      idx <- c(sample.int(100,100,TRUE),100L+sample.int(100,100,TRUE))
      # Within-group centering preserves the fixed group allocation law.
      delta <- colSums(z$c$rows[idx,,drop=FALSE])-colSums(z$c$rows)
      fixed[b] <- quadratic(delta,a)
      tryCatch({
        zz <- components(d[idx,,drop=FALSE]); aa <- geometry(zz$c)
        if(max(abs(z$c$nuisance-zz$c$nuisance))>1e-12 || max(abs(z$c$directions-zz$c$directions))>1e-12) stop('bootstrap coordinate mismatch')
        refit[b] <- quadratic(zz$c$score-z$c$score,aa)
      },error=function(e) fail(b,e))
    }
    row$bootstrap_ok <- sum(is.finite(refit))
    row$p_fixed <- (1+sum(fixed>=row$statistic))/(boot+1)
    # No deletion/replacement of failed bootstrap draws: whole arm unavailable.
    if(row$bootstrap_ok==boot) row$p_refit <- (1+sum(refit>=row$statistic))/(boot+1)
    saveRDS(list(fixed=fixed,refit=refit),file.path(out,'raw',paste0(cell$cell_id,'-',r,'.rds')))
  },error=function(e) fail(0L,e))
  row$elapsed_s <- proc.time()[['elapsed']]-t0
  list(row=row,errors=errors)
}
all <- list()
for(r in seq_len(reps)) {
  if(proc.time()[['elapsed']]-started>limit) break
  v <- parallel::mclapply(seq_len(4),function(i) one(i,r),mc.cores=workers,mc.preschedule=TRUE)
  all <- c(all,v)
  saveRDS(v,file.path(out,'raw',paste0('rep-',r,'.rds')))
  cat('Completed',r,'draws/cell;',round(proc.time()[['elapsed']]-started,1),'seconds\n');flush.console()
  # Predict one next balanced batch conservatively; stop before budget overrun.
  if((proc.time()[['elapsed']]-started)*(1+1/r)>limit) break
}
rows <- do.call(rbind,lapply(all,`[[`,'row')); errors <- do.call(rbind,lapply(all,`[[`,'errors'))
write_csv(rows,file.path(out,'raw','draws.csv')); write_csv(errors,file.path(out,'failures.csv'))
rates <- do.call(rbind,lapply(cells$cell_id,function(id) do.call(rbind,lapply(c('current_sb','current_peba4','fixed','refit'),function(arm) {
  p <- rows[[paste0('p_',arm)]][rows$cell_id==id]; n <- sum(is.finite(p)); k <- sum(p<.05,na.rm=TRUE)
  if(n) { ph <- k/n; z <- qnorm(.975); mid <- (ph+z^2/(2*n))/(1+z^2/n); half <- z*sqrt(ph*(1-ph)/n+z^2/(4*n^2))/(1+z^2/n) } else mid <- half <- NA_real_
  data.frame(cell_id=id,arm=arm,attempted=length(p),available=n,rejected=k,rejection=if(n) k/n else NA_real_,lower=mid-half,upper=mid+half)
}))))
write_csv(rates,file.path(out,'rates.csv'));write_csv(cells,file.path(out,'cells.csv'))
write_metadata(file.path(out,'metadata.csv'),list(seed_base=base,bootstrap_seed='draw_seed + 1000000 + bootstrap_id',requested_reps=reps,completed_reps=nrow(rows)/4,bootstrap=boot,workers=workers,max_seconds=limit,elapsed_s=proc.time()[['elapsed']]-started,command=paste(args,collapse=' '),git_head=git_scalar(c('rev-parse','HEAD')),git_dirty=git_dirty(),source_md5=unname(tools::md5sum(script)),native_md5=unname(tools::md5sum(list.files(file.path(find.package('magmaanlab'),'libs'),'\\.so$',full.names=TRUE))),max_policy_gap=max(rows$policy_gap,na.rm=TRUE),failed_bootstraps=nrow(errors),status='development_only'),packages=c('magmaan','magmaanlab','lavaan'))
cat('Results:',out,'\n')
if(nrow(errors)) stop('Failures saved; inspect before interpreting')
