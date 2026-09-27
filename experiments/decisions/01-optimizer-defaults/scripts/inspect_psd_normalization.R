#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script));args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript scripts/inspect_psd_normalization.R\nInspect saved pilot exceptions without refitting.\n');quit(save='no')}
source(file.path(here,'..','..','_support','R','helpers.R'));set_single_threaded_math()
for(f in c('families.R','fits.R','psd_normalization.R'))source(file.path(here,'R',f))
out<-file.path(here,'results','psd-normalization','pilot')
x<-read.csv(file.path(out,'fits.csv'));pops<-all_populations()
tasks<-unique(x[c('pop','n','rep','seed')]);rows<-list()
for(i in seq_len(nrow(tasks))){
 t<-tasks[i,];p<-pops[[t$pop]];moments<-draw_moments(p,t$n,t$seed)
 S<-moments$S;R<-S/outer(sqrt(diag(S)),sqrt(diag(S)))
 for(tr in c('x100','x0.01','mixed')){
  T<-S*outer(transform_factors(tr,p$p),transform_factors(tr,p$p))
  Q<-T/outer(sqrt(diag(T)),sqrt(diag(T)))
  rows[[length(rows)+1L]]<-cbind(t,transform=tr,normalized_input_error=max(abs(Q-R)))
 }
}
input_invariance<-do.call(rbind,rows)
write.csv(input_invariance,file.path(out,'input_invariance.csv'),row.names=FALSE)
fail<-read.csv(file.path(out,'transport_failures.csv'))
files<-list.files(file.path(out,'raw'),pattern='^parameters_.*csv$',full.names=TRUE)
parameters<-do.call(rbind,lapply(files,read.csv));checks<-list()
keys<-c('pop','model','n','rep','chart','transform','route','arm')
for(i in seq_len(nrow(fail))){
 f<-fail[i,];p<-pops[[f$pop]];fm<-p$models[[which(vapply(p$models,`[[`,'','key')==f$model)]]
 fm$std_lv<-f$chart=='std_lv';model<-build_model(fm)
 moments<-draw_moments(p,f$n,f$seed)
 native<-sample_in_units(moments,rep(1,p$p),fm$meanstructure)
 original<-sample_in_units(moments,transform_factors(f$transform,p$p),fm$meanstructure)
 setup<-normalize_unconstrained_sample(model,original,fm$std_lv)
 native_setup<-normalize_unconstrained_sample(model,native,fm$std_lv)
 keep<-rep(TRUE,nrow(parameters));for(k in keys)keep<-keep & parameters[[k]]==f[[k]]
 z<-parameters[keep,];theta<-z$value[order(z$parameter)]
 stopifnot(length(theta)==max(model$partable$free))
 normalized_theta<-theta/setup$theta_units
 for(space in c('normalized','original','native')){
  sample<-switch(space,normalized=setup$sample,original=original,native=native)
  point<-switch(space,normalized=normalized_theta,original=theta,native=normalized_theta*native_setup$theta_units)
  ev<-magmaanlab::magmaan_core$evaluate_at(model,sample,point,estimator='ML')
  a<-magmaanlab::frontier_newton_accuracy(ev,psd=TRUE)
  checks[[length(checks)+1L]]<-cbind(f[keys],space=space,fmin=ev$fmin,status=a$status,
   passed=a$passed,condition=a$condition,distance=a$distance,null_directions=a$null_directions,
   constrained_directions=a$constrained_directions,min_multiplier=a$min_multiplier)
 }
}
accuracy_checks<-if(length(checks))do.call(rbind,checks)else data.frame()
write.csv(accuracy_checks,file.path(out,'accuracy_checks.csv'),row.names=FALSE)
cat('Maximum normalized-input discrepancy: ',max(input_invariance$normalized_input_error),'\n',sep='')
print(accuracy_checks,row.names=FALSE)
