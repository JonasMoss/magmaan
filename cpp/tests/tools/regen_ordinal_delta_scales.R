#!/usr/bin/env Rscript
# Synthetic summaries only; lavaan is the independently fitted component oracle.
suppressPackageStartupMessages(library(lavaan))
stopifnot(as.character(packageVersion('lavaan')) == gsub('-', '.', trimws(readLines('cpp/tests/fixtures/lavaan_version.txt')), fixed=TRUE))
set.seed(531072)
n <- 600L
eta <- matrix(rnorm(2*n),n,2); eta[,2] <- .4*eta[,1]+sqrt(.84)*eta[,2]
x <- cbind(outer(eta[,1],c(.8,.7,.65)),outer(eta[,2],c(.85,.75,.6)))+matrix(rnorm(6*n,sd=.7),n,6)
d <- as.data.frame(apply(x,2,function(z) as.integer(cut(z,c(-Inf,-.5,.5,Inf)))))
names(d)<-paste0('u',1:6); d$g<-rep(1:2,each=n/2)
base <- c('f1 =~ 1*u1+u2+u3','f2 =~ 1*u4+u5+u6','f1 ~~ f1; f2 ~~ f2; f1 ~~ f2','f1 ~ 0*1; f2 ~ 0*1')
model <- function(scales=rep('1',6), fixed_thresholds=integer(), extra='') {
 paste(c(base,vapply(1:6,function(j) paste(sprintf('u%d | %s',j,if(j %in% fixed_thresholds) '-0.5*t1+0.5*t2' else 't1+t2'),sprintf('u%d ~ 0*1',j),sprintf('u%d ~*~ %s*u%d',j,scales[j],j),sep='\n'),character(1)),extra),collapse='\n')
}
cases <- list(
 fixed_nonunit=list(model=model(c('.8',rep('1',5)))),
 equal_scales=list(model=model(c('shared','shared',rep('1',4)),1:2)),
 released_scale=list(model=model(c('NA',rep('1',5)),1)),
 threshold_loading_invariance=list(model='f1 =~ u1+u2+u3; f2 =~ u4+u5+u6',group_equal=c('thresholds','loadings'),group=TRUE),
 scalar_groups=list(group=TRUE),
 longitudinal_equal_scales=list(model=model(c('shared','1','1','shared','1','1'),c(1,4))),
 fixed_scale_linear_constraint=list(model=model(c('.8','a','b','1','1','1'),2:3,'a == b')))
s <- c('f1 =~ c(1,1)*u1+c(l2,l2)*u2+c(l3,l3)*u3','f2 =~ c(1,1)*u4+c(l5,l5)*u5+c(l6,l6)*u6','f1 ~~ c(NA,NA)*f1; f2 ~~ c(NA,NA)*f2; f1 ~~ c(NA,NA)*f2','f1 ~ c(0,NA)*1; f2 ~ c(0,NA)*1')
for(j in 1:6) s<-c(s,sprintf('u%d | c(t%d1,t%d1)*t1+c(t%d2,t%d2)*t2',j,j,j,j,j),sprintf('u%d ~ c(0,0)*1',j),sprintf('u%d ~*~ c(1,NA)*u%d',j,j))
cases$scalar_groups$model<-paste(s,collapse='\n')
out<-list(version=as.character(packageVersion('lavaan')),seed=531072,cases=list())
for(id in names(cases)) {
 c<-cases[[id]]; args<-list(model=c$model,data=d,ordered=paste0('u',1:6),parameterization='delta',auto.var=FALSE,auto.fix.first=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,meanstructure=TRUE)
 if(isTRUE(c$group)) args$group<-'g'
 if(length(c$group_equal)) args$group.equal<-c$group_equal
 if(id=='threshold_loading_invariance') args<-args[!names(args) %in% c('auto.var','auto.fix.first','auto.cov.lv.x','auto.cov.y')]
 fitter<-if(id=='threshold_loading_invariance') cfa else lavaan
 lv<-do.call(fitter,c(args,list(estimator='WLSMV')))
 stopifnot(lavInspect(lv,'converged'),all(vapply(if(isTRUE(c$group)) lavInspect(lv,'theta') else list(lavInspect(lv,'theta')),function(x) all(diag(x)>0),logical(1))))
 pt<-parTable(lv)
 dw<-do.call(fitter,c(args,list(estimator='DWLS')))
 ul<-do.call(fitter,c(args,list(estimator='ULS')))
 stopifnot(lavInspect(dw,'converged'),lavInspect(ul,'converged'))
 pt$ustart[pt$free==0 & pt$op %in% c('~~','~*~','~1')]<-pt$est[pt$free==0 & pt$op %in% c('~~','~*~','~1')]
 # DELTA residual rows are derived; retain the evaluator's fixed-unit placeholder.
 pt$ustart[pt$op=='~~' & pt$lhs==pt$rhs & pt$lhs %in% paste0('u',1:6)]<-1
 out$cases[[id]]<-list(model=c$model,group_equal=c$group_equal,
  partable=pt[,c('id','lhs','op','rhs','user','block','group','free','exo','ustart','label','plabel','est','se')],
  R=lapply(lv@SampleStats@cov,unname),thresholds=lapply(lv@SampleStats@th,as.numeric),
  n_obs=as.list(as.integer(unlist(lv@SampleStats@nobs))),weight=lapply(lv@SampleStats@WLS.VD,as.numeric),
  nacov=lapply(lv@SampleStats@NACOV,unname),dwls_se=parTable(dw)$se,uls_est=parTable(ul)$est,
  test=as.list(fitMeasures(lv,c('chisq','chisq.scaled','chisq.scaling.factor','df'))),
  shift=as.numeric(lavInspect(lv,'test')$scaled.shifted$shift.parameter))
 cat(id,'df',fitMeasures(lv,'df'),'scales',pt$est[pt$op=='~*~'],'\n')
}
jsonlite::write_json(out,'cpp/tests/fixtures/ordinal/delta_scales.json',auto_unbox=TRUE,pretty=TRUE,digits=17,na='null')
