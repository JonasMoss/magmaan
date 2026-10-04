#!/usr/bin/env Rscript
# Synthetic ordinal summaries, independently fitted by the pinned lavaan oracle.
suppressPackageStartupMessages(library(lavaan))
stopifnot(as.character(packageVersion('lavaan')) == gsub('-', '.', trimws(readLines('cpp/tests/fixtures/lavaan_version.txt')), fixed=TRUE))
set.seed(542072)
n <- 800L
eta <- matrix(rnorm(2*n),n,2); eta[,2] <- .35*eta[,1]+sqrt(1-.35^2)*eta[,2]
x <- cbind(outer(eta[,1],c(.85,.75,.7,.65)),outer(eta[,2],c(.8,.75,.65,.7))) + matrix(rnorm(8*n,sd=.8),n,8)
data_for <- function(binary=FALSE) {
 d <- as.data.frame(apply(x,2,function(z) as.integer(cut(z,if(binary) c(-Inf,0,Inf) else c(-Inf,-.5,.5,Inf)))))
 names(d) <- paste0('y',1:8); d$g<-rep(1:2,each=n/2); d
}
square <- 'f =~ y1+l2*y2+l3*y3+y4\nl2 == l3^2'
product <- 'f1 =~ y1+a*y2+b*y3+y4\nf2 =~ y5+c*y6+y7+y8\na == b*c'
cases <- list(square_delta=list(model=square,par='delta'),square_theta=list(model=square,par='theta'),
 binary=list(model=square,par='delta',binary=TRUE),product=list(model=product,par='delta'),
 mixed_equalities=list(model=paste(product,'b == c',sep='\n'),par='theta'),
 cross_group=list(model='f =~ y1+c(a1,a2)*y2+c(b1,b2)*y3+y4\na1 == b2^2',par='delta',group=TRUE))
out<-list(version=as.character(packageVersion('lavaan')),seed=542072,cases=list())
for(id in names(cases)) {
 c<-cases[[id]]; d<-data_for(isTRUE(c$binary)); ov<-paste0('y',seq_len(if(id %in% c('product','mixed_equalities')) 8 else 4))
 args<-list(model=c$model,data=d,ordered=ov,parameterization=c$par)
 if(isTRUE(c$group)) args$group<-'g'
 fits<-list()
 for(estimator in c('WLSMV','DWLS','ULSMV','ULS','WLS')) {
  lv<-do.call(cfa,c(args,list(estimator=estimator)))
  stopifnot(lavInspect(lv,'converged'))
  pt<-parTable(lv)
  # DELTA residual variances are derived; use the evaluator's unit placeholder.
  if(c$par=='delta') pt$ustart[pt$op=='~~' & pt$lhs==pt$rhs & pt$lhs %in% ov]<-1
  fits[[estimator]]<-list(partable=pt[,c('id','lhs','op','rhs','user','block','group','free','exo','ustart','label','plabel','est','se')],
   test=as.list(fitMeasures(lv,c('chisq','df','chisq.scaled','chisq.scaling.factor'))),
   shift=if(estimator %in% c('WLSMV','ULSMV')) lavInspect(lv,'test')$scaled.shifted$shift.parameter else NULL)
  if(estimator=='WLSMV') {
   stats<-list(R=lapply(lv@SampleStats@cov,unname),thresholds=lapply(lv@SampleStats@th,as.numeric),
    n_obs=as.list(as.integer(unlist(lv@SampleStats@nobs))),weight=lapply(lv@SampleStats@WLS.VD,as.numeric),
    nacov=lapply(lv@SampleStats@NACOV,unname),category_counts=rep(list(rep(if(isTRUE(c$binary)) 2L else 3L,length(ov))),if(isTRUE(c$group)) 2 else 1))
   fitted_start<-do.call(cfa,c(args,list(estimator=estimator,start=parTable(lv))))
   score<-lavTestScore(fitted_start)$uni; fits[[estimator]]$score<-as.list(score$X2)
   fits[[estimator]]$default_start_score<-as.list(lavTestScore(lv)$uni$X2)
   fits[[estimator]]$reference_tangent<-'Both models refitted with start=parTable(fit); fitted equality Jacobian (oracle-defects.md TASK-54.2)'
   stopifnot(max(abs(coef(fitted_start)-coef(lv)))<1e-5)
   un<-sub('\n[^\n]*==.*','',c$model)
   {
    if(id=='mixed_equalities') un<-product
    h1<-do.call(cfa,c(args[names(args)!='model'],list(model=un,estimator='WLSMV')))
    fits[[estimator]]$null_partable<-parTable(h1)[,c('id','lhs','op','rhs','user','block','group','free','exo','ustart','label','plabel','est','se')]
    if(c$par=='delta') { q<-fits[[estimator]]$null_partable; q$ustart[q$op=='~~' & q$lhs==q$rhs & q$lhs %in% ov]<-1; fits[[estimator]]$null_partable<-q }
    h1_fitted_start<-do.call(cfa,c(args[names(args)!='model'],list(model=un,estimator='WLSMV',start=parTable(h1))))
    nested<-lavTestLRT(h1_fitted_start,fitted_start,method='satorra.2000')
    fits[[estimator]]$nested<-as.list(nested[2,c('Chisq diff','Df diff','Pr(>Chisq)')])
    fits[[estimator]]$default_start_nested<-as.list(lavTestLRT(h1,lv,method='satorra.2000')[2,c('Chisq diff','Df diff','Pr(>Chisq)')])
   }
  }
 }
 out$cases[[id]]<-c(list(model=c$model,parameterization=c$par),stats,list(fits=fits))
 cat(id,'df',fits$WLSMV$test$df,'\n')
}
# Mplus 9.1 Demo meaning probe (local /tmp/task542-demo): npar=11, df=3.
# This same four-indicator syntax has an independent lavaan numerical gate.
out$cases$square_delta$mplus<-paste('DATA: FILE=synthetic.dat;','VARIABLE: NAMES=y1-y4; CATEGORICAL=y1-y4;','ANALYSIS: ESTIMATOR=WLSMV;','MODEL: f BY y1; f BY y2 (l2); f BY y3 (l3); f BY y4;','MODEL CONSTRAINT: l2 = l3*l3;',sep='\n')
out$cases$square_delta$demo<-list(version='Mplus 9.1 Demo',npar=11L,df=3L)
jsonlite::write_json(out,'cpp/tests/fixtures/ordinal/nonlinear_equalities.json',auto_unbox=TRUE,pretty=TRUE,digits=17,na='null')
