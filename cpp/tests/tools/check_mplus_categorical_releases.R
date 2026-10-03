#!/usr/bin/env Rscript
# Independent fixed-threshold references for explicit single-group releases.
# Generate data with regen_mplus_probes.R --categorical before running.
suppressPackageStartupMessages(library(lavaan))
suppressPackageStartupMessages(library(magmaanlab))
scale_equality <- '--scale-equality' %in% commandArgs(trailingOnly=TRUE)
failures <- character()
d <- read.table(path.expand('~/.cache/magmaan-logs/mplus-probes-categorical/P-IV2/delta_ordinal_scalar/probe.dat'))
names(d)<-c(paste0('u',1:6),'g'); d<-d[d$g==1,]
for(par in c('delta','theta')) {
s <- c('f1 =~ 1*u1 + u2 + u3','f2 =~ 1*u4 + u5 + u6','f1 ~~ f1; f2 ~~ f2; f1 ~~ f2','f1 ~ 0*1; f2 ~ 0*1')
for(j in 1:6) s<-c(s,if(j==1 || (scale_equality && j==2)) sprintf('u%d | -0.5*t1 + 0.5*t2',j) else sprintf('u%d | t1+t2',j),sprintf('u%d ~ 0*1',j), if(par=='delta') sprintf('u%d ~*~ %s*u%d',j,if(scale_equality && j %in% 1:2) 'shared' else if(j==1) 'NA' else '1',j) else sprintf('u%d ~~ %s*u%d',j,if(scale_equality && j %in% 1:2) 'shared' else if(j==1) 'NA' else '1',j))
s<-paste(s,collapse='\n')
l <- lavaan(s,data=d,ordered=paste0('u',1:6),parameterization=par,estimator='WLSMV',auto.var=FALSE,auto.fix.first=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,meanstructure=TRUE)
spec <- model_spec(s,ordered=paste0('u',1:6),parameterization=par,auto_var=FALSE,auto_fix_first=FALSE,auto_cov_lv_x=FALSE,auto_cov_y=FALSE,meanstructure=TRUE)
if (scale_equality && par == 'delta') {
  actual <- tryCatch(fit_model(spec,d,estimator='DWLS'),error=identity)
  stopifnot(inherits(actual,'error'),
    grepl('unsupported DELTA response scale',conditionMessage(actual),fixed=TRUE),
    grepl('theta',conditionMessage(actual),fixed=TRUE))
  cat('delta scale equality: explicit unsupported error, theta alternative\n')
  next
}
m <- fit_model(spec,d,estimator='DWLS')
p<-m$partable; q<-parTable(l); k<-function(p)paste(p$lhs,p$op,p$rhs,p$group); i<-match(k(p),k(q)); use<-!is.na(i)&p$op%in%c('=~','|','~1','~*~'); err<-abs(p$est[use]-q$est[i[use]])
r<-convention_inference(m,'WLSMV'); free<-!is.na(i)&p$free>0&p$op%in%c('=~','|','~1'); se<-sqrt(diag(r$covariance))[p$free[free]]; ref<-q$se[i[free]]
cat(par,'converged',m$converged,lavInspect(l,'converged'),'est',max(err),'se',max(abs(se-ref)),'test',r$test$statistic-fitMeasures(l,'chisq.scaled'),'df',r$test$df,fitMeasures(l,'df'),'available',r$covariance_available,r$test$available,'\n')
if(scale_equality) {
  print(p[p$op=='~*~',]); print(q[q$op=='~*~',])
}
if(!isTRUE(m$converged) || !isTRUE(lavInspect(l,'converged')) ||
   any(err > 1e-5*(1+pmax(abs(p$est[use]),abs(q$est[i[use]])))))
  failures <- c(failures,paste(par,'explicit release estimates disagree'))
if(r$covariance_available && r$test$available &&
   (any(abs(se-ref)>1e-5*(1+pmax(abs(se),abs(ref)))) ||
    abs(r$test$statistic-fitMeasures(l,'chisq.scaled')) > 1e-5*(1+abs(fitMeasures(l,'chisq.scaled'))) ||
    r$test$df!=fitMeasures(l,'df')))
  failures <- c(failures,paste(par,'available WLSMV convention disagrees with lavaan'))
}

if(length(failures)) stop(paste(failures,collapse='\n'),call.=FALSE)
