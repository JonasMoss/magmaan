#!/usr/bin/env Rscript
# Re-evaluate saved points, without optimization or replacing requested markers.
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script));args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript scripts/check_normalization_witnesses.R [RUN_ID=normalization-revisit]\n');quit(save='no')}
run<-if(length(args))args[1]else'normalization-revisit';stopifnot(grepl('^[a-zA-Z0-9_-]+$',run))
for(f in c('designs.R','fit.R'))source(file.path(here,'R',f))
base<-magmaanlab::model_spec(model_syntax)
w<-read.csv(file.path(here,'results','open-case-conclusions','finite_witnesses.csv'))
vectors<-read.csv(file.path(here,'results','open-case-conclusions','parameters.csv'))
rows<-list()
for(i in seq_len(nrow(w))){
 c<-w[i,];v<-vectors$value[vectors$case_id==c$case_id];pt<-base$partable;values<-pt$ustart
 values[pt$op=='=~']<-c(1,v[1:2],1,v[3:4]);values[pt$op=='~']<-v[6]/v[5]
 values[pt$op=='~~'&pt$lhs=='X']<-v[5];values[pt$op=='~~'&pt$lhs=='Y']<-v[7]-v[6]^2/v[5]
 for(j in 1:6)values[pt$op=='~~'&pt$lhs==ov_names[j]]<-v[7+j]
 theta<-numeric(max(pt$free));theta[pt$free[pt$free>0]]<-values[pt$free>0]
 seed<-(if(c$batch=='development')202609281L else 902609281L)+match(c$design,names(designs_all()))*100000L+c$n*100L+c$rep
 sample<-sample_moments(draw_data(design_sigma(designs_all()[[c$design]]),c$n,seed))
 ev<-magmaanlab::magmaan_core$evaluate_at(pt,sample,theta,estimator='ML')
 audit<-magmaanlab::frontier_newton_accuracy(ev)
 tr<-tryCatch(magmaanlab::frontier_reidentify(ev$partable,base,pole_tol=1e-4),error=function(e)e)
 stopifnot(abs(ev$fmin-c$objective)<1e-8,!inherits(tr,'error'))
 rows[[i]]<-cbind(c[c('case_id','batch','design','n','rep')],objective=ev$fmin,accuracy_status=audit$status,
  accuracy_passed=isTRUE(audit$passed),newton_distance=audit$distance,condition=audit$condition,
  chart_pass=TRUE,admissible=isTRUE(ev$diagnostics$admissibility$admissible))
}
out<-file.path(here,'results',run);stopifnot(dir.exists(out))
write.csv(do.call(rbind,rows),file.path(out,'witness_audits.csv'),row.names=FALSE)
src<-file.path(here,'results','sphere-translations-inspected')
x<-read.csv(file.path(src,'source_endpoints.csv'));pts<-read.csv(file.path(src,'sphere_parameters.csv'))
checks<-lapply(seq_len(nrow(x)),function(i){
 point<-pts[pts$case_id==i,];pt<-base$partable;key<-function(z)paste(z$lhs,z$op,z$rhs)
 idx<-match(key(pt),key(point));stopifnot(!anyNA(idx))
 pt$est<-point$est[idx];pt$free<-seq_len(nrow(pt));pt$ustart<-pt$est
 accepted<-function(tol)!inherits(tryCatch(magmaanlab::frontier_reidentify(pt,base,pole_tol=tol),error=function(e)e),'error')
 cbind(x[i,c('batch','design','n','rep','chart_level')],chart_pass_1e4=accepted(1e-4),chart_pass_1e6=accepted(1e-6))
})
checks<-do.call(rbind,checks);stopifnot(sum(!checks$chart_pass_1e4)==2)
write.csv(checks,file.path(out,'chart_witnesses.csv'),row.names=FALSE)
print(do.call(rbind,rows),row.names=FALSE);print(checks[!checks$chart_pass_1e4,],row.names=FALSE)
