#!/usr/bin/env Rscript
# Run from the repository root; the original draw function owns production seeds.
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Rscript diagnostics/nested_profile_law.R [--reps 100] [--workers 2]\nRun from repo root with R_LIBS pointing to the lane library. Output: diagnostics/results/.\n')
  quit()
}
option <- function(key, default) { i <- match(key,args); if(is.na(i)) default else as.integer(args[i+1]) }
reps <- option('--reps',100L); workers <- option('--workers',2L)
stopifnot(reps>=1,reps<=100,workers>=1,workers<=2)
Sys.setenv(OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1)
library(magmaanlab)
study <- 'experiments/decisions/05-dwls-policy-calibration'
source(file.path(study,'R/compute.R'))
out <- file.path(study,'diagnostics/results')
if(file.exists(file.path(out,'metadata.csv'))) stop('Output exists; preserve it before a new run.')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
all_cells <- read.csv(file.path(study,'results/dwls-policy/production-2026-10-03/cells.csv'))
sibling <- subset(all_cells,family=='nested' & factors==2 & categories==5 & n==1000 & nesting=='thresholds' & cross==.3 & role=='null')
binary <- head(subset(all_cells,family=='nested' & factors==2 & categories==2 & n==400 & nesting=='metric' & cross==0 & role=='null'),1)
cells <- rbind(all_cells[all_cells$cell_id==75,],sibling,binary)
stopifnot(nrow(cells)==3)
write.csv(cells,file.path(out,'cells.csv'),row.names=FALSE)
core <- magmaan_core
methods <- c(sb='sb',scaled_shifted='ss',mean_variance='mv',scaled_f='scaled_f',all='all',penalized_all='pall',eba2='eba',eba4='eba',eba6='eba',peba2='peba',peba4='peba',peba6='peba',pols='pols')
run_one <- function(job) {
 c <- cells[job$cell,]; id <- job$rep; seed <- 817150001L+10000L*c$cell_id+id
 tryCatch({
  d <- dwls_draw_data(c,seed)
  h1 <- dwls_fit(c,d,if(c$nesting=='thresholds') 'thresholds' else NULL)
  h0 <- dwls_fit(c,d,if(c$nesting=='thresholds') c('thresholds','loadings') else 'loadings')
  if(!h1$converged || !h0$converged) stop('Nonconverged fit')
  profile <- core$ordinal_profile_lrt(h1,h0,h1$ordinal_stats)
  fixed <- robust_nested_lrt(h1,h0,data=h1$ordinal_stats,gamma='empirical',method='restriction_map',A.method='exact',weight='DWLS')
  parts <- magmaanlab:::ordinal_nested_diagnostic_impl(h1,h0)
  K <- parts$K; A <- parts$A
  H <- crossprod(K,parts$hessian_total%*%K)/h1$ntotal
  # Covariance is returned in full theta; pure-merge K need not be orthonormal.
  L <- solve(crossprod(K),t(K))
  V <- L%*%core$robust_ordinal_ij(h1,h1$ordinal_stats)$vcov%*%t(L)*h1$ntotal
  C <- A%*%solve(H,t(A)); S <- A%*%V%*%t(A)
  R <- chol(C); Ri <- solve(R)
  parameter <- eigen(t(Ri)%*%S%*%Ri,symmetric=TRUE,only.values=TRUE)$values
  laws <- list(profile=profile$eigvals,fixed=fixed$eigenvalues,parameter=parameter,common=parts$common$eigvals)
  T <- profile$T_diff; df <- profile$df_diff
  rows <- lapply(names(laws),function(law) {
   e <- sort(laws[[law]][laws[[law]]>0],decreasing=TRUE)
   p <- vapply(names(methods),function(m) {
    param <- if(grepl('eba',m)) as.numeric(sub('.*eba','',m)) else 4
    # FMG keeps only its df largest terms. Pass the spectrum dimension
    # for the full-law family, as production pEBA does; SB uses nominal df.
    tryCatch(if(m=='sb') pchisq(core$robust_satorra_bentler(T,df,e)$chi2_scaled,
      df,lower.tail=FALSE) else core$robust_fmg_test(T,max(df,length(e)),e,
      methods[[m]],param)$p_value,error=function(e) NA_real_)
   },numeric(1))
   p <- c(p,exact=magmaanlab:::weighted_chisq_diagnostic_impl(e,T))
   data.frame(cell_id=c$cell_id,replicate=id,seed=seed,law=law,T=T,df=df,
    trace=sum(e),variance=2*sum(e^2),terms=length(e),nonnegligible=sum(e>1e-3*max(e)),
    top_r_relative_error=if(law %in% c('profile','common')) sqrt(sum((head(e,df)-sort(parameter,decreasing=TRUE))^2)/sum(parameter^2)) else NA_real_,
    signed_trace=if(law=='profile') profile$trace_signed else NA_real_,
    negative_trace=if(law=='profile') profile$negative_trace_abs else NA_real_,
    as.list(setNames(p,paste0('p_',names(p)))),check.names=FALSE)
  })
  list(rows=do.call(rbind,rows),spectra=laws,error=NULL)
 },error=function(e) list(error=data.frame(cell_id=c$cell_id,replicate=id,seed=seed,error=conditionMessage(e))))
}
start <- proc.time()[3]; results <- list()
# Small batches preserve completed draws and keep progress visible.
for (first in seq(1,reps,by=5)) {
 jobs <- expand.grid(cell=1:3,rep=first:min(first+4,reps))
 batch <- parallel::mclapply(split(jobs,seq_len(nrow(jobs))),run_one,mc.cores=workers,mc.preschedule=TRUE)
 results <- c(results,batch); saveRDS(results,file.path(out,'raw.rds'))
 cat('Completed attempts:',length(results),'elapsed:',round(proc.time()[3]-start,1),'seconds\n')
 if(proc.time()[3]-start>270) {cat('Bounded run stopping before next batch.\n'); break}
}
rows <- do.call(rbind,lapply(results,`[[`,'rows')); errors <- do.call(rbind,lapply(results,`[[`,'error'))
write.csv(rows,file.path(out,'raw.csv'),row.names=FALSE)
if(!is.null(errors)) write.csv(errors,file.path(out,'failures.csv'),row.names=FALSE)
if(!is.null(rows)) {
 summary <- do.call(rbind,lapply(split(rows,paste(rows$cell_id,rows$law)),function(z) {
  ps <- z[startsWith(names(z),'p_')]
  data.frame(cell_id=z$cell_id[1],law=z$law[1],successful=nrow(z),mc_mean=mean(z$T),mc_variance=var(z$T),
   mean_trace=mean(z$trace),mean_law_variance=mean(z$variance),mean_terms=mean(z$terms),
   mean_nonnegligible=mean(z$nonnegligible),mean_top_r_relative_error=mean(z$top_r_relative_error),
   mean_signed_trace=mean(z$signed_trace),mean_negative_trace=mean(z$negative_trace),
   as.list(vapply(ps,function(p) mean(p<.05,na.rm=TRUE),numeric(1))),
   as.list(setNames(vapply(ps,function(p) sum(is.finite(p)),integer(1)),paste0('available_',names(ps)))))
 }))
 write.csv(summary,file.path(out,'summary.csv'),row.names=FALSE)
}
write.csv(data.frame(seed_base=817150001,reps_requested=reps,attempts=length(results),workers=workers,
 elapsed_seconds=proc.time()[3]-start,package=as.character(packageVersion('magmaanlab')),
 git_ref=system('git rev-parse HEAD',intern=TRUE),
 runner_md5=unname(tools::md5sum(file.path(study,'diagnostics/nested_profile_law.R'))),
 compute_md5=unname(tools::md5sum(file.path(study,'R/compute.R'))),
 native_md5=unname(tools::md5sum(system.file('libs/magmaanlab.so',package='magmaanlab'))),
 command=paste(commandArgs(),collapse=' ')),file.path(out,'metadata.csv'),row.names=FALSE)
cat('Outputs:',out,'\n')
