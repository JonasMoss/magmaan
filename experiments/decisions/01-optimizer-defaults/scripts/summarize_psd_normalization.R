#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script));args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript scripts/summarize_psd_normalization.R [--run-id pilot|smoke]\n');quit(save='no')}
run<-if('--run-id' %in% args)args[match('--run-id',args)+1L]else'pilot'
out<-file.path(here,'results','psd-normalization',run)
x<-read.csv(file.path(out,'fits.csv'));x$message[is.na(x$message)]<-''
key<-c('pop','model','n','rep','chart','transform','route')
stopifnot(!anyDuplicated(x[c(key,'arm')]))
problem<-setdiff(key,'route')
x$best<-ave(ifelse(x$success,x$fmin,Inf),interaction(x[problem],drop=TRUE),FUN=min)
x$hit<-x$success & is.finite(x$best) & x$fmin<=x$best+1e-6*(1+abs(x$best))
x$fits<-1L
metrics<-c('fits','fit_success','success','output_success','hit','seconds')
summary<-aggregate(x[metrics],x[c('chart','transform','route','arm')],sum)
family<-aggregate(x[metrics],x[c('family','chart','transform','route','arm')],sum)
fields<-c('success','output_success','hit','fmin','original_accuracy','message')
paired<-merge(x[x$arm=='current',c(key,'family',fields)],x[x$arm=='normalized',c(key,fields)],by=key,suffixes=c('_current','_normalized'))
paired$gain<-!paired$success_current & paired$success_normalized
paired$loss<-paired$success_current & !paired$success_normalized
paired$objective_gain<-!paired$hit_current & paired$hit_normalized
paired$objective_loss<-paired$hit_current & !paired$hit_normalized
changes<-aggregate(paired[c('gain','loss','objective_gain','objective_loss')],paired[c('chart','transform','route')],sum)
paired_changes<-paired[paired$gain | paired$loss | paired$objective_gain | paired$objective_loss,]
# Implied covariances are already divided by the original sample SD products,
# so their elements are directly comparable across observed-unit transforms.
files<-list.files(file.path(out,'raw'),pattern='^covariances_.*csv$',full.names=TRUE)
cov<-do.call(rbind,lapply(files,read.csv))
basekeys<-c('pop','model','n','rep','chart','route','arm')
v<-merge(cov[cov$transform=='native',c(basekeys,'element','value')],
         cov[cov$transform!='native',c(basekeys,'element','transform','value')],
         by=c(basekeys,'element'),suffixes=c('_native','_changed'))
v$error<-abs(v$value_native-v$value_changed)
covdiff<-aggregate(v['error'],v[c(basekeys,'transform')],max)
native<-x[x$transform=='native',c(basekeys,'success','output_success','fmin')]
other<-x[x$transform!='native',c(basekeys,'transform','success','output_success','fmin')]
unit_pairs<-merge(native,other,by=basekeys,suffixes=c('_native','_changed'))
unit_pairs<-merge(unit_pairs,covdiff,by=c(basekeys,'transform'),all.x=TRUE)
unit_pairs$both_success<-unit_pairs$success_native & unit_pairs$success_changed
unit_pairs$verdict_changed<-unit_pairs$success_native != unit_pairs$success_changed
unit_pairs$objective_gap<-abs(unit_pairs$fmin_native-unit_pairs$fmin_changed)
unit_pairs$objective_mismatch<-unit_pairs$both_success & unit_pairs$objective_gap>1e-6*(1+abs(unit_pairs$fmin_native))
unit_pairs$covariance_mismatch<-unit_pairs$both_success & unit_pairs$error>1e-5
unit_pairs$pairs<-1L
invariance<-aggregate(unit_pairs[c('pairs','both_success','verdict_changed','objective_mismatch','covariance_mismatch')],unit_pairs[c('chart','route','arm')],sum)
unit_exceptions<-unit_pairs[unit_pairs$verdict_changed | unit_pairs$objective_mismatch | unit_pairs$covariance_mismatch,]
transport<-do.call(rbind,lapply(split(x,interaction(x$chart,x$arm,drop=TRUE)),function(z)data.frame(
 chart=z$chart[1],arm=z$arm[1],returned=sum(z$returned),checked=sum(is.finite(z$covariance_error)),
 failures=sum(z$returned & !z$transport_ok),max_covariance_error=max(z$covariance_error,na.rm=TRUE),
 max_objective_error=max(z$objective_error,na.rm=TRUE),
 accuracy_lost_after_transport=sum(z$success & !z$original_accuracy_passed))))
failures<-x[!x$fit_success | !x$output_success,]
transport_failures<-x[x$returned & (!x$transport_ok | (x$fit_success & !x$original_psd) | (x$success & !x$original_accuracy_passed)),]
for(name in c('summary','family','paired_changes','changes','invariance','unit_exceptions','transport','transport_failures','failures'))
 write.csv(get(name),file.path(out,paste0(name,'.csv')),row.names=FALSE)
print(aggregate(x[metrics],x[c('chart','route','arm')],sum),row.names=FALSE)
print(invariance,row.names=FALSE);print(transport,row.names=FALSE)
cat('Wrote summaries in ',out,'\n',sep='')
