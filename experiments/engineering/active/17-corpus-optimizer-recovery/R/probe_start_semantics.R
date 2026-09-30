#!/usr/bin/env Rscript
# Compare two equivalent spellings of the phantom scale paths.
args<-commandArgs(TRUE);root<-normalizePath(args[1]);out<-normalizePath(args[2])
script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),'../../../../_support/R/helpers.R'));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),'inputs.R'));source(file.path(dirname(script),'arms.R'))
id<-'little_2013_ch3_fig_3_11_longitudinal_cfa_phantom'
m<-read.csv(file.path(root,'manifest.csv'));c<-read_case(root,m$case_dir[match(id,m$case_id)],'ML')
syntax<-c$args$model
syntax<-sub('ETA1 ~ PosAffT1','PosAffT1 =~ NA*ETA1',syntax,fixed=TRUE)
syntax<-sub('ETA2 ~ PosAffT2','PosAffT2 =~ NA*ETA2',syntax,fixed=TRUE)
alternative<-model_spec(syntax,meanstructure=TRUE,fixed_x=FALSE,auto_cov_y=TRUE)
canonical<-function(p) {
  lv<-unique(p$lhs[p$op=='=~']);ix<-p$op=='=~' & p$rhs %in% lv
  lhs<-p$lhs[ix];p$lhs[ix]<-p$rhs[ix];p$rhs[ix]<-lhs;p$op[ix]<-'~';p
}
p<-c$model$partable;q<-alternative$partable
idx<-match(row_key(canonical(p)),row_key(canonical(q)))
stopifnot(!anyNA(idx),nrow(p)==nrow(q),all((p$free>0)==(q$free[idx]>0)))
fix<-p$free==0 & is.finite(p$ustart);stopifnot(all(p$ustart[fix]==q$ustart[idx[fix]]))
xi<-which(p$free>0);map<-function(x){y<-numeric(max(q$free));y[q$free[idx[xi]]]<-x[p$free[xi]];y}
x<-magmaan_core$estimate_start_values(p,c$sample,start='simple',transport='native')
y<-magmaan_core$estimate_start_values(q,c$sample,start='simple',transport='native')
a<-magmaan_core$evaluate_at(c$model,c$sample,x,'ML')
b<-magmaan_core$evaluate_at(alternative,c$sample,map(x),'ML')
# Also compare a nonzero-path point, not only the singular zero-path point.
xnz<-x;paths<-which(p$op=='~' & p$lhs %in% c('ETA1','ETA2') & p$rhs %in% c('PosAffT1','PosAffT2'))
xnz[p$free[paths]]<-0.5
an<-magmaan_core$evaluate_at(c$model,c$sample,xnz,'ML')
bn<-magmaan_core$evaluate_at(alternative,c$sample,map(xnz),'ML')
stopifnot(abs(a$fmin-b$fmin)<1e-10,abs(an$fmin-bn$fmin)<1e-10)
z<-data.frame(path=paste(p$lhs[paths],p$op[paths],p$rhs[paths]),
  regression_start=x[p$free[paths]],measurement_start=y[q$free[idx[paths]]],
  shared_zero_objective=a$fmin,shared_nonzero_objective=an$fmin)
write.csv(z,file.path(out,'start_semantics.csv'),row.names=FALSE);print(z,row.names=FALSE)
