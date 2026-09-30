#!/usr/bin/env Rscript
args<-commandArgs(TRUE);out<-normalizePath(args[1]);script<-normalizePath(sub('--file=','',grep('--file=',commandArgs(),value=TRUE)[1]))
suppressPackageStartupMessages(library(magmaanlab));source(file.path(dirname(script),'structural_start.R'))
S<-tcrossprod(c(1,.8,.6))*2+diag(c(.4,.6,.8));dimnames(S)<-list(paste0('y',1:3),paste0('y',1:3))
base<-'f =~ 1*y1 + y2 + y3\ng =~ 0*y1\nf ~ g\nf ~~ 0*f\ng ~~ 1*g'
make<-function(syntax=base,sample=S){m<-model_spec(syntax,fixed_x=FALSE);list(model=m,sample=list(S=list(sample),nobs=500L,mean=NULL))}
start<-function(c)magmaan_core$estimate_start_values(c$model$partable,c$sample,start=method,transport='native')
path<-function(c,x){p<-c$model$partable;i<-which((p$op=='~' & p$lhs=='f' & p$rhs=='g') | (p$op=='=~' & p$lhs=='g' & p$rhs=='f'));x[p$free[i]]}
checks<-list();check<-function(name,ok){checks[[length(checks)+1]]<<-data.frame(constructor=method,check=name,passed=isTRUE(ok))}
for(method in c('simple','fabin3')) {
c<-make();x<-start(c);a<-structural_start(c,x);check('supported single-parent path initialized',a$paths==1 && path(c,a$theta)>0)
q<-make(sub('f ~ g','g =~ NA*f',base,fixed=TRUE));b<-structural_start(q,start(q));check('equivalent syntax yields same scale',abs(path(c,a$theta)-path(q,b$theta))<1e-10)
f<-function(c,x)magmaan_core$evaluate_at(c$model,c$sample,x,'ML')$fmin
check('equivalent syntax yields same objective',abs(f(c,a$theta)-f(q,b$theta))<1e-10)
u<-make(sample=100*S);b<-structural_start(u,start(u));check('observed units rescale path magnitude',abs(path(u,b$theta)/path(c,a$theta)-10)<1e-10)
u<-make(sub('g ~~ 1*g','g ~~ 4*g',base,fixed=TRUE));b<-structural_start(u,start(u));check('parent variance rescales path inversely',abs(path(u,b$theta)/path(c,a$theta)-.5)<1e-10)
u<-make(sub('f ~ g','f ~ start(0)*g',base,fixed=TRUE));b<-structural_start(u,start(u));check('explicit zero hint preserved',path(u,b$theta)==0 && b$paths==0)
u<-make(sub('f ~ g','f ~ start(-0.3)*g',base,fixed=TRUE));b<-structural_start(u,start(u));check('explicit negative hint preserved',path(u,b$theta)==-.3 && b$paths==0)
u<-make(sub('f ~~ 0*f','f ~~ f',base,fixed=TRUE));xx<-start(u);b<-structural_start(u,xx);check('ordinary regression left unchanged',identical(as.numeric(xx),b$theta))
u<-make(sub('y2','start(-0.9)*y2',base,fixed=TRUE));b<-structural_start(u,start(u));p<-u$model$partable;i<-which(p$op=='=~' & p$rhs=='y2');check('loading hint preserved',abs(b$theta[p$free[i]]+.9)<1e-12)
# Effect-coded loadings: the constructor must first honor the affine scale.
syn<-'f =~ NA*l1*y1 + l2*y2 + l3*y3\ng =~ 0*y1\nf ~ g\nf ~~ 0*f\ng ~~ 1*g\nl1 == 3-l2-l3'
u<-make(syn);b<-structural_start(u,start(u));p<-u$model$partable;i<-which(p$op=='=~' & p$lhs=='f');check('effect-coding equality preserved',abs(sum(b$theta[p$free[i]])-3)<1e-10 && b$paths==1)
# Inconsistent starts and equalities are a reported fallback, not overwritten hints.
u<-make(paste(syn,'l1 == 1','l2 == 1','l3 == 1',sep='\n'));p<-u$model$partable;i<-which(p$label=='l1' & p$free>0);u$model$partable$ustart[i]<-2;b<-structural_start(u,start(u));check('conflicting hints reported',b$reason=='infeasible constraints/hints')
negative<-S;negative[3,]<--negative[3,];negative[,3]<--negative[,3]
u<-make(sample=negative);b<-structural_start(u,start(u));check('reverse-keyed indicator handled',b$paths==1 && path(u,b$theta)>0)
# Isolate the moment rule from constructor changes in measurement coordinates.
u<-make();xx<-start(u);pp<-u$model$partable;ll<-which(pp$op=='=~' & pp$lhs=='f' & pp$free>0);xx[pp$free[ll]]<-c(.8,.6)
b0<-structural_start(u,xx)
v<-make(sample=100*S);yy<-start(v);yy[pp$free[ll]]<-c(.8,.6);b1<-structural_start(v,yy)
check('moment rule respects consistent measurement units',abs(path(v,b1$theta)/path(u,b0$theta)-10)<1e-10)
v<-make(sample=negative);yy<-start(v);yy[pp$free[ll]]<-c(.8,-.6);b1<-structural_start(v,yy)
check('moment rule respects consistent loading signs',b1$paths==1 && abs(path(v,b1$theta)-path(u,b0$theta))<1e-10)
}
d<-do.call(rbind,checks);write.csv(d,file.path(out,'structural_properties.csv'),row.names=FALSE);print(d,row.names=FALSE)
