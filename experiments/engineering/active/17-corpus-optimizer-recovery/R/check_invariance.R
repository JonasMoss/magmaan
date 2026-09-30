#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/check_invariance.R CORPUS RESULTS [--workers N]
Classifies each case/estimator pair as unit-invariant under the rescaling of test_coordinate_scaling.R
--rescaled: the native optimum, carried to the new units by the ratio of magmaan parameter units and
refined by a constrained local PORT fit, must attain the native objective on the rescaled data. The native optimum
is the accepted layered-start PORT fit in sample units, else the reference lavaan fit. Writes
RESULTS/invariance.csv. A few minutes.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2]);script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),'inputs.R'))
d<-read.csv(file.path(out,'comparison.csv'),stringsAsFactors=FALSE);jobs<-subset(d,arm=='port_default')
manifest<-read.csv(file.path(root,'manifest.csv'),stringsAsFactors=FALSE)
unit_factors<-function(names){k<-match(names,sort(unique(names)));setNames(10^(((5*k)%%9-4)/2),names)}
rescale_sample<-function(sample){ov<-unique(unlist(lapply(sample$S,rownames)));f<-unit_factors(ov)
  sample$S<-lapply(sample$S,function(S){g<-f[rownames(S)];S*outer(g,g)})
  if(!is.null(sample$mean))sample$mean<-lapply(sample$mean,function(m)m*f[names(m)]);sample}
one<-function(i){
  j<-jobs[i,];row<-data.frame(case=j$case,estimator=j$estimator,target=j$best_observed,native_f=NA_real_,
    transported_f=NA_real_,refit_f=NA_real_,source='',invariant=NA,message='',stringsAsFactors=FALSE)
  r<-tryCatch({
    c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
    fitter<-if(j$estimator=='ML')magmaan_core$fit_ml else magmaan_core$fit_gls
    x<-as.numeric(magmaan_core$estimate_start_values(c$model$partable,c$sample,start='layered'))
    f<-fitter(c$model,c$sample,optimizer='port',control=list(start=x,coordinate_scaling='sample_units'))
    tol<-1e-6*(1+abs(j$best_observed))
    if(isTRUE(f$converged) && f$fmin<=j$best_observed+tol){theta<-f$theta;row$source<-'magmaan'} else {
      theta<-map_theta(c$model,reference_fit(c));row$source<-'lavaan reference'}
    row$native_f<-magmaan_core$evaluate_at(c$model,c$sample,theta,j$estimator)$fmin
    sd<-rescale_sample(c$sample)
    ua<-magmaan_core$estimate_coordinate_map(c$model$partable,c$sample,theta,'sample_units')$units
    ub<-magmaan_core$estimate_coordinate_map(c$model$partable,sd,theta*0+1,'sample_units')$units
    moved<-theta*ub/ua
    row$transported_f<-magmaan_core$evaluate_at(c$model,sd,moved,j$estimator)$fmin
    # The transported point can violate equality constraints (equal variances
    # of latents whose markers change by different factors) and effect-coded
    # latents transform only approximately with their units, so only a
    # constrained local refit from it decides.
    g<-tryCatch(fitter(c$model,sd,optimizer='port',control=list(start=moved,coordinate_scaling='sample_units')),error=function(e)NULL)
    if(!is.null(g) && isTRUE(g$converged))row$refit_f<-g$fmin
    row$invariant<-is.finite(row$refit_f) && abs(row$refit_f-row$native_f)<=1e-6*(1+abs(row$native_f));row},error=function(e){row$message<-conditionMessage(e);row})
  r
}
workers<-if('--workers' %in% args) as.integer(args[match('--workers',args)+1L]) else 3L
z<-do.call(rbind,parallel::mclapply(seq_len(nrow(jobs)),one,mc.cores=workers,mc.preschedule=TRUE))
write.csv(z,file.path(out,'invariance.csv'),row.names=FALSE)
cat(sprintf('pairs %d, invariant %d, not invariant %d, unresolved %d\n',nrow(z),sum(z$invariant %in% TRUE),
  sum(z$invariant %in% FALSE),sum(is.na(z$invariant))))
