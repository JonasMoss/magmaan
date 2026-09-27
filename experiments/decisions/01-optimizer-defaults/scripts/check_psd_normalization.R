#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
source(file.path(here,'..','..','_support','R','helpers.R'));set_single_threaded_math()
for(f in c('families.R','fits.R','psd_normalization.R'))source(file.path(here,'R',f))
S<-matrix(c(1.8,.72,.55,.72,1.5,.46,.55,.46,1.3),3)
dimnames(S)<-list(c('x1','x2','x3'),c('x1','x2','x3'))
sample<-list(S=list(S),nobs=100L)
reject<-function(syntax,std_lv=FALSE){
 model<-magmaanlab::model_spec(syntax,std_lv=std_lv)
 z<-try(normalize_unconstrained_sample(model,sample,std_lv),silent=TRUE)
 stopifnot(inherits(z,'try-error'))
}
reject('f =~ x1 + a*x2 + a*x3')
reject('f =~ x1 + 2*x2 + x3')
reject('f =~ x1 + x2 + x3\nf ~~ 2*f')
# Mean structures obey the same units without centering the observations.
model<-magmaanlab::model_spec('f =~ x1 + x2 + x3',meanstructure=TRUE)
sample$mean<-list(c(4,-2,7))
z<-normalize_unconstrained_sample(model,sample,FALSE)
fit<-normalization_fit(model,z$sample,'direct')$fit
back<-magmaanlab::magmaan_core$evaluate_at(model,sample,fit$theta*z$theta_units,estimator='ML')
a<-magmaanlab::magmaan_core$model_implied(fit)
b<-magmaanlab::magmaan_core$model_implied(back)
stopifnot(max(abs(b$mu[[1]]-a$mu[[1]]*z$sd))<1e-10,
 abs(fit$fmin-back$fmin)<1e-10)
root<-file.path(here,'results','psd-normalization','pilot')
x<-read.csv(file.path(root,'fits.csv'))
stopifnot(nrow(x)==9600,sum(x$returned & !x$transport_ok)==0,
 all(x$success== (x$fit_success & x$transport_ok & x$original_psd)),
 all(x$output_success==(x$success & x$original_accuracy_passed)))
t<-read.csv(file.path(root,'transport.csv'));stopifnot(sum(t$failures)==0)
i<-read.csv(file.path(root,'input_invariance.csv'));stopifnot(max(i$normalized_input_error)<1e-14)
cat('Unsupported constraints rejected; mean transport verified; 9600 fit records and normalized inputs checked.\n')
