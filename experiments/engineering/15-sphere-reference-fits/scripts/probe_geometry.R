#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here,'..','..','_support','R','helpers.R')); set_single_threaded_math()
for(f in c('designs.R','fit.R','start_design.R','expanded_designs.R','geometry_designs.R')) source(file.path(here,'R',f))
args <- commandArgs(TRUE)
if('--help' %in% args) {
 cat('Usage: Rscript scripts/probe_geometry.R [--smoke] [--run-id NAME]\n',
 'Five exact covariance witnesses: interior, marker pole, PSD face, std.lv pole, ML nonattainment.\n',
 'ML L-BFGS/PORT; PSD SLSQP none/diagonal. Requested chart, fixed alternate marker, sphere.\n',
 'Analytic classes are design facts, never inferred from numerical failure. Smoke: interior and marker pole.\n',sep='');quit(save='no')
}
smoke <- '--smoke' %in% args
run <- if('--run-id' %in% args) args[match('--run-id',args)+1L] else if(smoke)'geometry-smoke' else 'geometry-witnesses'
stopifnot(!is.na(run),grepl('^[a-zA-Z0-9_-]+$',run))
out<-file.path(here,'results',run);if(file.exists(file.path(out,'fits.csv')))stop('choose a fresh run-id')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_out<-function(x,name)write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE)
designs<-geometry_designs();if(smoke)designs<-designs[c('interior','marker_pole')]
rows<-parameters<-list();t0<-proc.time()[['elapsed']]
for(name in names(designs)) {
 d<-designs[[name]]; data<-exact_data(d$sigma); sample<-sample_moments(data)
 original<-magmaanlab::model_spec(d$syntax,std_lv=d$std_lv)
 for(domain in c('ML','PSD')) for(route in c('requested','alternate_marker','sphere')) {
  spec<-if(route=='alternate_marker')geometry_marker(d) else original
  recipes<-if(route=='sphere')list(sphere_canonical=NULL) else
    if(domain=='ML')list(layered_native=NULL) else list(fabin3_auto=NULL)
  # This spectral constructor is marker-specific. The std.lv route gets its
  # constructor; its fixed alternative marker gets the spectral portfolio.
  if(route!='sphere' && !(d$std_lv && route=='requested')) {
   lv<-names(d$blocks)
   subsets<-if(domain=='PSD')list(character()) else lapply(0:(2^length(lv)-1),function(mask)lv[as.logical(intToBits(mask)[seq_along(lv)])])
   for(negative in subsets) {
    id<-if(length(negative))paste0('spectral_negative_',paste(negative,collapse='')) else 'spectral_positive'
    recipes[id]<-list(tryCatch(spectral_start(spec,sample,negative),error=function(e)e))
   }
  }
  for(backend in if(domain=='ML')c('nlopt-lbfgs','port') else 'nlopt-slsqp')
   for(scaling in if(domain=='ML')'information' else c('none','diagonal')) for(arm in names(recipes)) {
    theta<-recipes[[arm]]
    if(inherits(theta,'error')) {
     rec<-endpoint_record();rec$label<-'start_unavailable';rec$message<-conditionMessage(theta)
     z<-list(record=rec,partable=NULL)
    } else {
     ctl<-if(domain=='ML')list(start='layered',start_transport='native',coordinate_scaling='information') else list(start='fabin3',start_transport='auto')
     if(!is.null(theta))ctl<-list(start_transport='native',coordinate_scaling='information')
     z<-run_fit(spec,data,sample,domain,if(route=='sphere')'sphere' else 'ordinary',backend,arm,theta,
       preconditioning=if(domain=='PSD')scaling else 'none',control=ctl,
       assessment=list(marker_model=function(pt,sample)geometry_marker(d),extent=expanded_extent))
    }
    id<-length(rows)+1L
    rows[[id]]<-cbind(fit_id=id,design=name,analytic_class=d$kind,domain=domain,route=route,
      backend=backend,preconditioning=if(route=='sphere' && domain=='ML')'sphere_internal' else scaling,
      start_id=arm,z$record,geometry_endpoint(z$partable,d,sample,domain))
    if(!is.null(z$partable))parameters[[length(parameters)+1L]]<-cbind(fit_id=id,z$partable[c('lhs','op','rhs','est')])
   }
 }
 cat(sprintf('%s: %d attempts, %.1fs\n',name,length(rows),proc.time()[['elapsed']]-t0))
}
fits<-do.call(rbind,rows)
# Numerical reproduction is separate from local screening and analytic truth.
fits$matches_covariance<-is.finite(fits$covariance_gap)&fits$covariance_gap<=1e-6
fits$interpretation<-ifelse(fits$analytic_class=='ml_nonattainment' & fits$domain=='ML',
 'analytic_nonattainment_do_not_certify',ifelse(fits$screened,'screened_candidate','unresolved'))
write_out(fits,'fits');write_out(do.call(rbind,parameters),'parameters')
write_out(aggregate(fits[c('returned','screened','matches_covariance')],fits[c('design','domain','route')],sum),'summary')
# Explicit independent sequence for the nonattainment witness; F >= 0 with
# equality iff Sigma=S. Nonzero s12 and s13 force nonzero s23 at finite values.
S<-geometry_designs()$nonattainment$sigma
sequence<-do.call(rbind,lapply(c(1,2,4,8,16,32,64),function(t){
 l<-c(t,.3/t,.4/t);theta<-1-l^2;Sigma<-tcrossprod(l)+diag(theta)
 data.frame(t=t,F=as.numeric(determinant(Sigma,logarithm=TRUE)$modulus)-
   as.numeric(determinant(S,logarithm=TRUE)$modulus)+sum(diag(solve(Sigma,S)))-3,
   covariance_gap=max(abs(Sigma-S)),min_residual=min(theta),
   min_observed_eigen=min(eigen(Sigma,symmetric=TRUE,only.values=TRUE)$values))
}))
stopifnot(all(diff(sequence$F)<0),all(sequence$min_observed_eigen>0))
write_out(sequence,'nonattainment_sequence')
ref<-magmaan_cache_ref();write_metadata(file.path(out,'metadata.csv'),values=list(
 data='deterministic exact sample covariances; n=100; no random sampling',
 classes='analytically specified; numerical fits never establish nonattainment',
 sphere='canonical FABIN gauge-free start; polish FALSE; existing sphere-internal scaling',
 ml='layered native + signed spectral; information scaling',psd='FABIN3 auto + positive spectral; SLSQP none/diagonal',
 git_head=ref$git_head,git_dirty=ref$git_dirty),packages='magmaanlab')
cat('Wrote ',out,'\n',sep='')
