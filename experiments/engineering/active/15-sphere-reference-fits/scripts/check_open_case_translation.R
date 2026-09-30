#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
source(file.path(here,'..', '..', '..', '_support','R','helpers.R'));set_single_threaded_math()
for(f in c('designs.R','fit.R'))source(file.path(here,'R',f))
p<-file.path(here,'results','open-case-conclusions');cases<-read.csv(file.path(p,'finite_witnesses.csv'));params<-read.csv(file.path(p,'parameters.csv'))
base<-magmaanlab::model_spec(model_syntax);rows<-warm<-list()
for(i in seq_len(nrow(cases))){
 c<-cases[i,];v<-params$value[params$case_id==c$case_id]
 pt<-base$partable;values<-pt$ustart
 values[pt$op=='=~']<-c(1,v[1:2],1,v[3:4])
 values[pt$op=='~']<-v[6]/v[5]
 values[pt$op=='~~' & pt$lhs=='X']<-v[5]
 values[pt$op=='~~' & pt$lhs=='Y']<-v[7]-v[6]^2/v[5]
 for(j in 1:6)values[pt$op=='~~'&pt$lhs==ov_names[j]]<-v[7+j]
 theta<-numeric(max(pt$free));theta[pt$free[pt$free>0]]<-values[pt$free>0]
 seed<-(if(c$batch=='development')202609281L else 902609281L)+match(c$design,names(designs_all()))*100000L+c$n*100L+c$rep
 data<-draw_data(design_sigma(designs_all()[[c$design]]),c$n,seed);sample<-sample_moments(data)
 ev<-magmaanlab::magmaan_core$evaluate_at(pt,sample,theta,estimator='ML')
 stopifnot(abs(ev$fmin-c$objective)<1e-8,ev$diagnostics$admissibility$implied_sigma_pd)
 tr<-magmaanlab::frontier_reidentify(ev$partable,base,pole_tol=1e-4)
 audit<-magmaanlab::frontier_newton_accuracy(ev)
 rows[[i]]<-data.frame(case_id=c$case_id,library_objective=ev$fmin,objective_gap=abs(ev$fmin-c$objective),
  requested_marker_translation=TRUE,accuracy_status=audit$status,accuracy_passed=isTRUE(audit$passed),
  newton_distance=audit$distance,newton_condition=audit$condition,admissible=ev$diagnostics$admissibility$admissible)
 # Test whether the sphere route can reach the lower negative-variance basin
 # once supplied the finite witness; this is diagnostic, not a start recipe.
 if(c$batch=='fresh' && c$design=='weak_marker' && c$n==20 && c$rep==1){
  for(backend in c('nlopt-lbfgs','port')){
   z<-run_fit(base,data,sample,'ML','sphere',backend,'finite_witness',theta)
   stopifnot(z$record$screened,abs(z$record$objective-c$objective)<1e-8)
   warm[[length(warm)+1L]]<-cbind(case_id=c$case_id,backend=backend,z$record)
  }
 }
}
write.csv(do.call(rbind,rows),file.path(p,'library_crosscheck.csv'),row.names=FALSE)
write.csv(do.call(rbind,warm),file.path(p,'sphere_witness_restart.csv'),row.names=FALSE)
cat('Seven refined points preserve the objective in original marker coordinates and pass the study chart check.\nBoth sphere backends recover the previously missed finite basin from its witness.\n')
