#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/test_constrained_lane.R CORPUS RESULTS
Fits the nonlinear-constraints cases of RESULTS/problem_classes.csv (R/classify_problems.R) under ML and
GLS with SLSQP, which honors nonlinear equalities, from the current and the layered start in information
coordinates. PORT and L-BFGS reject these models. Writes RESULTS/constrained_lane.csv. Seconds.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2]);script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),'inputs.R'))
classes<-read.csv(file.path(out,'problem_classes.csv'),stringsAsFactors=FALSE)
d<-read.csv(file.path(out,'comparison.csv'),stringsAsFactors=FALSE);jobs<-subset(d,arm=='port_default')
jobs<-jobs[jobs$case %in% classes$case[classes$class=='nonlinear-constraints'],]
manifest<-read.csv(file.path(root,'manifest.csv'),stringsAsFactors=FALSE)
rows<-list()
for(i in seq_len(nrow(jobs))) {
  j<-jobs[i,];c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
  fitter<-if(j$estimator=='ML')magmaan_core$fit_ml else magmaan_core$fit_gls
  starts<-list(current=function()magmaan_core$estimate_start_values(c$model$partable,c$sample,
      start=if(j$estimator=='ML')'scaled-fabin' else 'fabin3',transport=if(j$estimator=='ML')'auto' else 'native'),
    layered=function()magmaan_core$estimate_start_values(c$model$partable,c$sample,start='layered'))
  for(name in names(starts)) {
    x<-as.numeric(starts[[name]]())
    fit<-tryCatch(fitter(c$model,c$sample,optimizer='nlopt-slsqp',control=list(start=x,coordinate_scaling='information')),error=function(e)e)
    rows[[length(rows)+1L]]<-data.frame(case=j$case,estimator=j$estimator,start=name,optimizer='nlopt-slsqp',
      scaling='information',target=j$best_observed,accepted=!inherits(fit,'error') && isTRUE(fit$converged),
      f=if(inherits(fit,'error'))NA_real_ else fit$fmin,message=if(inherits(fit,'error'))conditionMessage(fit) else '',
      stringsAsFactors=FALSE)
  }
}
z<-do.call(rbind,rows)
z$best<-z$accepted & is.finite(z$f) & z$f<=z$target+1e-6*(1+abs(z$target))
write.csv(z,file.path(out,'constrained_lane.csv'),row.names=FALSE)
print(z[,c('case','estimator','start','accepted','best','f','target')],row.names=FALSE)
