#!/usr/bin/env Rscript
# TASK-54.2: null calibration of fitted-tangent ordinal nested inference.
suppressPackageStartupMessages({library(lavaan); library(magmaanlab)})
args <- commandArgs(TRUE)
reps <- if(length(args)) as.integer(args[1]) else 500L
base <- 'f =~ y1+a*y2+b*y3+y4'
null <- paste(base,'a == b^2',sep='\n')
results <- vector('list',reps)
started <- proc.time()[3]
for(i in seq_len(reps)) {
 set.seed(542200+i); n<-800L; eta<-rnorm(n)
 # Marker .8 gives relative b=.875 and a=b^2 exactly.
 d<-as.data.frame(sapply(c(.8,.8*.875^2,.7,.65),function(l)
  as.integer(cut(l*eta+rnorm(n,sd=.8),c(-Inf,-.5,.5,Inf)))))
 names(d)<-paste0('y',1:4)
 results[[i]]<-tryCatch({
  lv0<-cfa(null,d,ordered=names(d),estimator='WLSMV')
  lv1<-cfa(base,d,ordered=names(d),estimator='WLSMV')
  fitted<-cfa(null,d,ordered=names(d),estimator='WLSMV',start=parTable(lv0))
  stopifnot(lavInspect(lv0,'converged'),lavInspect(lv1,'converged'),lavInspect(fitted,'converged'))
  f0<-fit_model(null,d,ordered=names(d),estimator='DWLS')
  f1<-fit_model(base,d,ordered=names(d),estimator='DWLS')
  stopifnot(f0$converged,f1$converged)
  mg<-convention_nested(f1,f0,'WLSMV')$test
  stopifnot(mg$available)
  lt<-lavTestLRT(lv1,fitted,method='satorra.2000')
  data.frame(rep=i,magmaan=mg$statistic,lavaan_fitted=lt[2,'Chisq diff'],p_magmaan=mg$pvalue,p_lavaan=lt[2,'Pr(>Chisq)'])
 },error=function(e) data.frame(rep=i,error=conditionMessage(e)))
 if(i%%10==0) cat('completed',i,'elapsed',proc.time()[3]-started,'\n')
 if(proc.time()[3]-started>570) stop('Authorized 10 minute wall budget exhausted')
}
valid<-vapply(results,function(r) !'error'%in%names(r),logical(1))
r<-do.call(rbind,results[valid]);print(r)
cat('valid',sum(valid),'failures',sum(!valid),'elapsed',proc.time()[3]-started,'\n')
if(any(!valid)) print(results[!valid])
if(sum(valid)>0) {
 cat('max_statistic_difference',max(abs(r$magmaan-r$lavaan_fitted)),'\n')
 print(c(magmaan_rejections=sum(r$p_magmaan<.05),lavaan_rejections=sum(r$p_lavaan<.05)))
 print(binom.test(sum(r$p_magmaan<.05),nrow(r),p=.05))
}
stopifnot(sum(valid)==reps)
