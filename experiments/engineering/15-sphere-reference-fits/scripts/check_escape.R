#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
for(f in c('designs.R','fit.R','start_design.R','escape_diagnostics.R'))source(file.path(here,'R',f))
charts<-inspection_charts();base<-charts$marker_11
sample<-sample_moments(draw_data(design_sigma(designs_all()$weak_marker),100,91842))
recipes<-start_recipes(base,sample)
for(name in names(recipes)){
 ev<-magmaanlab::magmaan_core$evaluate_at(base$partable,sample,recipes[[name]],estimator='ML')
 expected<-escape_metrics(ev$partable,sample)
 for(chart in setdiff(names(charts),'std_lv_positive')){
  tr<-magmaanlab::frontier_reidentify(ev$partable,charts[[chart]],pole_tol=0)
  other<-magmaanlab::magmaan_core$evaluate_at(charts[[chart]]$partable,sample,tr$theta,estimator='ML')
  actual<-escape_metrics(other$partable,sample)
  stopifnot(max(abs(expected$sigma-actual$sigma))<1e-10,
    max(abs(as.matrix(expected$record)-as.matrix(actual$record)))<1e-9)
 }
 tr<-tryCatch(magmaanlab::frontier_reidentify(ev$partable,charts$std_lv_positive,pole_tol=0),error=function(e)e)
 if(name=='spectral_positive')stopifnot(!inherits(tr,'error'))else stopifnot(inherits(tr,'error'))
}
# Stored selected candidates are evaluable finite points, not rejected solely
# for magnitude by this verification; the old extent labels remain untouched.
p<-file.path(here,'results','unresolved-identifications')
e<-read.csv(file.path(p,'candidate_evidence.csv'));pars<-read.csv(file.path(p,'candidate_parameters.csv'))
covs<-read.csv(file.path(p,'candidate_covariances.csv'))
stopifnot(nrow(e)==9,all(is.finite(e$objective)),all(is.finite(e$component_extent)))
for(i in seq_len(nrow(e))){
 x<-e[i,];spec<-charts[[x$chart]];pt<-spec$partable
 z<-pars[pars$fit_id==x$fit_id,];key<-function(p)paste(p$lhs,p$op,p$rhs)
 idx<-match(key(pt),key(z));stopifnot(!anyNA(idx));pt$est<-z$est[idx]
 # A saved sphere endpoint has released marker restrictions.
 if(x$chart=='sphere'){pt$free<-seq_len(nrow(pt));pt$ustart<-pt$est}
 seed<-(if(x$batch=='development')202609281L else 902609281L)+match(x$design,names(designs_all()))*100000L+x$n*100L+x$rep
 sample<-sample_moments(draw_data(design_sigma(designs_all()[[x$design]]),x$n,seed))
 m<-escape_metrics(pt,sample);z<-covs[covs$fit_id==x$fit_id,]
 stopifnot(!is.null(m$sigma),max(abs(as.vector(m$sigma)-z$value))<1e-8,
   abs(m$record$component_extent-x$component_extent)<1e-7)
}
cat('Common-point covariance, chart-invariant components, std.lv sign restriction and retained candidate checks passed.\n')
