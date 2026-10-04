#!/usr/bin/env Rscript
# Independent public-component reconstruction; no upstream implementation used.
suppressPackageStartupMessages(library(lavaan))
set.seed(542072); n<-800L
eta<-matrix(rnorm(2*n),n,2);eta[,2]<-.35*eta[,1]+sqrt(1-.35^2)*eta[,2]
x<-cbind(outer(eta[,1],c(.85,.75,.7,.65)),outer(eta[,2],c(.8,.75,.65,.7)))+matrix(rnorm(8*n,sd=.8),n,8)
d<-as.data.frame(apply(x,2,function(z) as.integer(cut(z,c(-Inf,-.5,.5,Inf))))); names(d)<-paste0('y',1:8)
base<-'f =~ y1+l2*y2+l3*y3+y4'; null<-paste(base,'l2 == l3^2',sep='\n')
fit<-function(model,start=NULL) cfa(model,d,ordered=paste0('y',1:4),estimator='WLSMV',start=start)
f1<-fit(base);f0<-fit(null); fitted<-fit(null,parTable(f0))
D1<-lavInspect(f1,'delta');D0<-lavInspect(f0,'delta');pt<-parTable(f0)
H<-matrix(0,1,ncol(D0));H[pt$free[pt$label=='l2']]<-1;H[pt$free[pt$label=='l3']]<--2*pt$est[pt$label=='l3']
K<-qr.Q(qr(t(H)),complete=TRUE)[,-1,drop=FALSE]
stopifnot(max(abs(H%*%K))<1e-12)
M<-solve(crossprod(D1),crossprod(D1,D0%*%K))
A<-t(svd(M,nu=nrow(M))$u[,nrow(M),drop=FALSE])
W<-lavInspect(f1,'wls.v');G<-lavInspect(f1,'gamma')
Iinv<-solve(crossprod(D1,W%*%D1));B<-crossprod(W%*%D1,G%*%W%*%D1)
scale<-drop(A%*%Iinv%*%B%*%Iinv%*%t(A))/drop(A%*%Iinv%*%t(A))
independent<-(fitMeasures(f0,'chisq')-fitMeasures(f1,'chisq'))/scale
old<-lavTestLRT(f1,f0,method='satorra.2000')[2,'Chisq diff']
new<-lavTestLRT(f1,fitted,method='satorra.2000')[2,'Chisq diff']
print(c(independent=independent,default_start=old,fitted_start=new,max_est_diff=max(abs(coef(f0)-coef(fitted)))))
stopifnot(abs(independent-new)<1e-5,max(abs(coef(f0)-coef(fitted)))<1e-5,abs(old-new)>1e-3)
