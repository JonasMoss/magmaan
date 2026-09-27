#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
for(f in c('designs.R','fit.R','start_design.R','escape_diagnostics.R'))source(file.path(here,'R',f))
if('--help' %in% commandArgs(TRUE)){cat('Usage: Rscript scripts/export_open_cases.R\nExports the seven non-near-pole cases, saved sphere and best-checked endpoints, in original marker coordinates. No fitting.\n');quit(save='no')}
root<-file.path(here,'results');out<-file.path(root,'open-case-inputs');dir.create(out,recursive=TRUE,showWarnings=FALSE)
checks<-read.csv(file.path(root,'requested-chart-policy','endpoint_checks.csv'))
checks<-checks[checks$translates_at_1e4,]
sphere<-read.csv(file.path(root,'sphere-translations-inspected','summary.csv'))
sphere<-sphere[sphere$chart=='first_marker',]
sphere_pt<-read.csv(file.path(root,'sphere-translations-inspected','parameters.csv'))
best<-read.csv(file.path(root,'unresolved-identifications','candidate_evidence.csv'))
best_pt<-read.csv(file.path(root,'unresolved-identifications','candidate_parameters.csv'))
charts<-inspection_charts();base<-charts$marker_11
match_case<-function(df,c)df$batch==c$batch & df$design==c$design & df$n==c$n & df$rep==c$rep
rows<-moments<-vectors<-list()
for(i in seq_len(nrow(checks))){
 c<-checks[i,];seed<-(if(c$batch=='development')202609281L else 902609281L)+match(c$design,names(designs_all()))*100000L+c$n*100L+c$rep
 sample<-sample_moments(draw_data(design_sigma(designs_all()[[c$design]]),c$n,seed));S<-sample$S[[1]]
 moments[[i]]<-data.frame(case_id=i,row=rep(1:6,6),column=rep(1:6,each=6),value=sprintf('%.17g',as.vector(S)))
 for(source in c('sphere','best_checked')){
  if(source=='sphere'){
   r<-sphere[match_case(sphere,c),];z<-sphere_pt[sphere_pt$case_id==r$case_id & sphere_pt$chart=='first_marker',];pt<-base$partable
  }else{
   r<-best[match_case(best,c),];z<-best_pt[best_pt$fit_id==r$fit_id,];pt<-charts[[r$chart]]$partable
  }
  key<-function(x)paste(x$lhs,x$op,x$rhs);idx<-match(key(pt),key(z));stopifnot(!anyNA(idx));pt$est<-z$est[idx]
  tr<-magmaanlab::frontier_reidentify(pt,base,pole_tol=0)
  ev<-magmaanlab::magmaan_core$evaluate_at(base$partable,sample,tr$theta,estimator='ML');p<-ev$partable
  value<-function(a,op,b)p$est[p$lhs==a & p$op==op & p$rhs==b][1]
  vx<-value('X','~~','X');beta<-value('Y','~','X');psi<-value('Y','~~','Y')
  v<-c(value('X','=~','x2'),value('X','=~','x3'),value('Y','=~','y2'),value('Y','=~','y3'),
       vx,beta*vx,beta^2*vx+psi,vapply(ov_names,function(n)value(n,'~~',n),0))
  L<-matrix(0,6,2);L[1:3,1]<-c(1,v[1:2]);L[4:6,2]<-c(1,v[3:4]);Phi<-matrix(c(v[5],v[6],v[6],v[7]),2)
  Sigma<-L%*%Phi%*%t(L)+diag(v[8:13])
  stopifnot(max(abs(Sigma-magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]))<1e-8)
  id<-length(rows)+1L
  rows[[id]]<-data.frame(start_id=id,case_id=i,batch=c$batch,design=c$design,n=c$n,rep=c$rep,seed=seed,source=source,
   library_objective=sprintf('%.17g',ev$fmin))
  vectors[[id]]<-data.frame(start_id=id,parameter=seq_along(v),value=sprintf('%.17g',v))
 }
}
for(name in c('cases','moments','starts'))write.csv(switch(name,cases=do.call(rbind,rows),moments=do.call(rbind,moments),starts=do.call(rbind,vectors)),file.path(out,paste0(name,'.csv')),row.names=FALSE)
cat('Exported seven cases and fourteen endpoints; original markers x1 and y1 preserved.\n')
