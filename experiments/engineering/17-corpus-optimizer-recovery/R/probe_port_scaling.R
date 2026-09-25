#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/probe_port_scaling.R CORPUS RESULTS\nRuns R nlminb (the same PORT routine lavaan and magmaan call) on the installed magmaan objective\nfrom the same start, with and without lavaan\'s start-based PORT scale vector, and compares with\nlavaan. Writes RESULTS/port_scaling.csv. Finite-difference gradients; a few minutes.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2]);script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),'inputs.R'))
manifest<-read.csv(file.path(root,'manifest.csv'),stringsAsFactors=FALSE)
# lavaan's nlminb controls, and its scale rule: 1/|start| for |start| > 1, else 1.
lav_ctrl<-list(eval.max=20000L,iter.max=10000L,abs.tol=.Machine$double.eps*10,rel.tol=1e-10,x.tol=1.5e-8,xf.tol=2.2e-14,step.min=1,step.max=1)
cases<-data.frame(case=c('kline_2023_ch9_roth_illness_path','kline_2023_ch9_roth_illness_mean','geiser_2013_ch4_latent_ar_cross_lagged',
  'kline_2023_ch14_sabatelli_heywood','kline_2023_ch15_worland_sr_step2b'),estimator=c('ML','ML','GLS','GLS','GLS'),
  start=c('layered','layered','fabin3','fabin3','fabin3'))
rows<-list()
for(i in seq_len(nrow(cases))) {
  j<-cases[i,];c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
  x<-as.numeric(magmaan_core$estimate_start_values(c$model$partable,c$sample,start=j$start,transport='native'))
  obj<-function(t){v<-magmaan_core$evaluate_at(c$model,c$sample,t,j$estimator)$fmin;if(!is.finite(v))1e20 else v}
  gr<-function(t)vapply(seq_along(t),function(k){h<-1e-6*max(1,abs(t[k]));a<-t;b<-t;a[k]<-a[k]+h;b[k]<-b[k]-h;(obj(a)-obj(b))/(2*h)},0)
  sc<-ifelse(abs(x)>1,1/abs(x),1)
  mp<-c$model$partable;lp<-lavaan::parTable(c$pre);ix<-which(lp$free>0);mx<-match(row_key(lp[ix,]),row_key(mp))
  a<-c$args;a$start<-x[mp$free[mx]];lf<-suppressWarnings(do.call(c$fun,a))
  lav_f<-magmaan_core$evaluate_at(c$model,c$sample,map_theta(c$model,lf),j$estimator)$fmin
  for(arm in c('nlminb unscaled','nlminb with lavaan scale')) {
    r<-nlminb(x,obj,gr,control=lav_ctrl,scale=if(arm=='nlminb unscaled')rep(1,length(x)) else sc)
    rows[[length(rows)+1L]]<-data.frame(case=j$case,estimator=j$estimator,start=j$start,run=arm,f=r$objective,
      message=r$message,iterations=r$iterations,target=lav_f)
  }
  rows[[length(rows)+1L]]<-data.frame(case=j$case,estimator=j$estimator,start=j$start,run='lavaan',f=lav_f,
    message=if(isTRUE(lavInspect(lf,'converged')))'converged' else 'not converged',iterations=lavInspect(lf,'iterations'),target=lav_f)
  cat(j$case,j$estimator,'done\n')
}
z<-do.call(rbind,rows);write.csv(z,file.path(out,'port_scaling.csv'),row.names=FALSE);print(z,row.names=FALSE)
