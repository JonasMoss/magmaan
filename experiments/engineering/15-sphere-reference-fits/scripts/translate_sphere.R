#!/usr/bin/env Rscript
# Inspect saved endpoints only. No model fitting or optimizer call occurs here.
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
for(f in c('designs.R','fit.R'))source(file.path(here,'R',f))
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript scripts/translate_sphere.R [--run-id NAME]\n',
 'Reads raw unresolved-identifications fits/parameters (run inspect_unresolved.R first).\n',
 'For each of the nine cases, selects its lowest-objective evaluable sphere endpoint.\n',
 'Translates without refitting to first marker, strongest marker and positive std.lv.\n',
 'Retains all parameters, translation errors, local status and covariance checks.\n',sep='');quit(save='no')}
run<-if('--run-id' %in% args)args[match('--run-id',args)+1L]else 'sphere-translations-inspected'
stopifnot(!is.na(run),grepl('^[a-zA-Z0-9_-]+$',run))
out<-file.path(here,'results',run);if(file.exists(file.path(out,'summary.csv')))stop('choose a fresh run-id')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_out<-function(x,name)write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE)
input<-file.path(here,'results','unresolved-identifications')
f<-read.csv(file.path(input,'fits.csv'));params<-read.csv(file.path(input,'parameters.csv'))
f<-f[f$chart=='sphere' & is.finite(f$objective),]
selected<-do.call(rbind,lapply(split(f,interaction(f$batch,f$design,f$n,f$rep,drop=TRUE)),function(z)z[which.min(z$objective),]))
write_out(selected,'source_endpoints')
base<-magmaanlab::model_spec(model_syntax);rows<-tables<-sources<-list()
for(i in seq_len(nrow(selected))){
 x<-selected[i,];sample<-sample_moments(draw_data(design_sigma(designs_all()[[x$design]]),x$n,x$seed))
 z<-params[params$fit_id==x$fit_id,];p<-base$partable;key<-function(x)paste(x$lhs,x$op,x$rhs)
 idx<-match(key(p),key(z));stopifnot(!anyNA(idx))
 p$est<-z$est[idx];p$free<-seq_len(nrow(p));p$ustart<-p$est
 sources[[i]]<-cbind(case_id=i,source_fit_id=x$fit_id,p[c('lhs','op','rhs','est')])
 targets<-list(first_marker=base,strongest_marker=strong_marker_model(p,sample),
   std_lv=magmaanlab::model_spec(model_syntax,std_lv=TRUE))
 # Independent evaluation of the saved point in a well-loaded marker chart.
 tr<-magmaanlab::frontier_reidentify(p,targets$strongest_marker,pole_tol=0)
 ev<-magmaanlab::magmaan_core$evaluate_at(targets$strongest_marker$partable,sample,tr$theta,estimator='ML')
 reference_sigma<-magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
 stopifnot(abs(ev$fmin-x$objective)<1e-8)
 for(chart in names(targets)){
  result<-tryCatch({
   tr<-magmaanlab::frontier_reidentify(p,targets[[chart]],pole_tol=1e-6)
   ev<-magmaanlab::magmaan_core$evaluate_at(targets[[chart]]$partable,sample,tr$theta,estimator='ML')
   sigma<-magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
   covariance_gap<-max(abs(sigma-reference_sigma)/sqrt(outer(diag(sample$S[[1]]),diag(sample$S[[1]]))))
   stopifnot(covariance_gap<1e-8,abs(ev$fmin-x$objective)<1e-8)
   list(pt=ev$partable,gap=covariance_gap,objective=ev$fmin)
  },error=function(e)e)
  rec<-data.frame(case_id=i,batch=x$batch,design=x$design,n=x$n,rep=x$rep,source_fit_id=x$fit_id,
   source_label=x$label,source_local_checked=x$local_checked,chart=chart,translated=!inherits(result,'error'),
   marker_X='',marker_Y='',max_abs_loading=NA_real_,var_X=NA_real_,disturbance_Y=NA_real_,
   total_var_Y=NA_real_,beta=NA_real_,negative_variances='',max_residual_ratio=NA_real_,
   objective=NA_real_,covariance_gap=NA_real_,message='')
  if(inherits(result,'error'))rec$message<-conditionMessage(result)else{
   q<-result$pt;value<-function(a,op,b)q$est[q$lhs==a&q$op==op&q$rhs==b][1]
   for(factor in c('X','Y')){
    marker<-targets[[chart]]$partable
    rec[[paste0('marker_',factor)]]<-paste(marker$rhs[marker$op=='=~'&marker$lhs==factor&marker$free==0],collapse=';')
   }
   rec$max_abs_loading<-max(abs(q$est[q$op=='=~']))
   rec$var_X<-value('X','~~','X');rec$disturbance_Y<-value('Y','~~','Y');rec$beta<-value('Y','~','X')
   rec$total_var_Y<-rec$beta^2*rec$var_X+rec$disturbance_Y
   variance<-q[q$op=='~~'&q$lhs==q$rhs,]
   # Judge sign away from rounding noise on a scale preserved by reidentification.
   energy<-vapply(c('X','Y'),function(f){l<-q[q$op=='=~'&q$lhs==f,];sum(l$est^2/diag(sample$S[[1]])[l$rhs])},0)
   scale<-ifelse(variance$lhs %in% c('X','Y'),energy[variance$lhs],1/diag(sample$S[[1]])[variance$lhs])
   rec$negative_variances<-paste(variance$lhs[variance$est*scale < -1e-8],collapse=';')
   residual<-variance[variance$lhs %in% ov_names,]
   rec$max_residual_ratio<-max(abs(residual$est)/diag(sample$S[[1]])[residual$lhs])
   rec$objective<-result$objective;rec$covariance_gap<-result$gap
   tables[[length(tables)+1L]]<-cbind(case_id=i,chart=chart,q[c('lhs','op','rhs','est')])
  }
  rows[[length(rows)+1L]]<-rec
 }
}
write_out(do.call(rbind,rows),'summary');write_out(do.call(rbind,tables),'parameters');write_out(do.call(rbind,sources),'sphere_parameters')
write_out(data.frame(key=c('source','selection','method','pole_tol','negative_scaled_variance_report_tol','interpretation'),
 value=c('unresolved-identifications','lowest objective evaluable sphere endpoint per case, regardless of magnitude/local-check status',
 'reidentify + evaluate only; no optimization','1e-6','-1e-8','translation success does not establish local accuracy or global optimality')),'metadata')
cat('Translated ',nrow(selected),' saved sphere endpoints without refitting.\nWrote ',out,'\n',sep='')
