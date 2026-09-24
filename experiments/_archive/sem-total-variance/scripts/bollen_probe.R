#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/bollen_probe.R [--smoke] [--reps N] [--budget-sec N] [--results-dir PATH]\n',
      'Default: original Bollen data + ten matched Gaussian draws, nine coordinate choices,\n',
      'lavaan unrestricted and two native magmaan PSD starts per moment set; 60s soft budget.\n',
      'Smoke: original data + draws 1 and 2. Outputs only under selected results directory.\n',sep='')
  quit()
}
script <- normalizePath(sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script)); repo <- normalizePath(file.path(here,'../../..'))
o <- list(reps=10L,budget_sec=60,results_dir=file.path(here,'results/bollen'),seed_base=20260919L)
i<-1L
while(i<=length(args)) {
  a<-args[i]
  if(a=='--smoke') o$reps<-2L else {
    if(i==length(args)) stop('Missing value: ',a)
    i<-i+1L; key<-gsub('-','_',sub('^--','',a))
    if(!key %in% c('reps','budget_sec','results_dir')) stop('Unknown option: ',a)
    o[[key]]<-if(key=='results_dir') args[i] else as.numeric(args[i])
  }
  i<-i+1L
}
stopifnot(is.finite(o$reps),o$reps>=0,o$reps==as.integer(o$reps),
          is.finite(o$budget_sec),o$budget_sec>0)
source(file.path(repo,'experiments/_support/R/helpers.R')); set_single_threaded_math()
source(file.path(here,'R/charts.R')); source(file.path(here,'R/models.R'))
source(file.path(here,'R/precondition.R'))
if(!requireNamespace('magmaan',quietly=TRUE)) stop('Install magmaan for the PSD reference')
dir.create(o$results_dir,recursive=TRUE,showWarnings=FALSE)
unlink(file.path(o$results_dir,c('fits.csv','references.csv','selection.csv','validation.csv','routing.csv','metadata.csv')))
t0<-Sys.time(); elapsed<-function() as.numeric(difftime(Sys.time(),t0,units='secs'))
case<-empirical_case('bollen_democracy_sem',repo)
syntax<-paste(readLines(file.path(repo,'benchmarks/cases/bollen_democracy_sem/model.lav')),collapse='\n')
seed<-o$seed_base+60000L; start<-common_start(case,seed)
algorithm<-select_preconditioner(case$m,case$mask,start)
charts<-c('marker','disturbance','total'); methods<-c('none','diagonal','full')
maps<-list(); validation<-selection<-list()
for(ch in charts) for(method in methods) {
  label<-paste(ch,method,sep='_')
  map<-precondition_map(chart_map(case$m,case$mask,ch),start,method)
  maps[[label]]<-map
  v<-validate_chart(map,start); g<-geometry(map,map$pack(start))
  stopifnot(g$rank==g$npar)
  if(method=='full' && abs(g$condition-1)>1e-7) stop('Whitening gate failed')
  validation[[label]]<-data.frame(chart=ch,method=method,covariance_error=v[1],
    derivative_error=v[2],start_condition=g$condition)
}
validation<-do.call(rbind,validation)
# Candidate choice uses only information at the common start, never fit outcomes.
diag_rows<-validation[validation$method=='diagonal',]
chosen<-algorithm$chart
stopifnot(chosen==diag_rows$chart[which.min(diag_rows$start_condition)])
selection<-data.frame(chart=diag_rows$chart,score=diag_rows$start_condition,
                      selected=diag_rows$chart==chosen,selector_seconds=algorithm$seconds,setup_seconds=elapsed())
write.csv(selection,file.path(o$results_dir,'selection.csv'),row.names=FALSE)
write.csv(validation,file.path(o$results_dir,'validation.csv'),row.names=FALSE)
cat('Selected ',chosen,' from diagonal-equilibrated information; setup ',round(elapsed(),3),'s\n',sep='')
# Bollen-only named projection; this is not a generic partable compiler.
row_value<-function(m,lhs,op,rhs) {
  if(op=='=~') return(m$lambda[rhs,lhs])
  if(op=='~') return(m$beta[lhs,rhs])
  if(op=='~~') {
    V<-if(lhs %in% rownames(m$psi)) m$psi else m$theta
    return(V[lhs,rhs])
  }
  stop('Unsupported Bollen row: ',op)
}
from_partable<-function(pt) {
  m<-case$m
  for(k in seq_len(nrow(pt))) {
    a<-pt$lhs[k]; b<-pt$rhs[k]; op<-pt$op[k]; value<-pt$est[k]
    if(op=='=~') m$lambda[b,a]<-value else if(op=='~') m$beta[a,b]<-value else if(op=='~~') {
      nm<-if(a %in% rownames(m$psi)) 'psi' else 'theta'
      m[[nm]][a,b]<-m[[nm]][b,a]<-value
    } else stop('Unsupported Bollen output row: ',op)
  }
  implied_result<-implied(m); c(list(m=m),implied_result)
}
objective<-function(Sigma,S) as.numeric(determinant(Sigma,logarithm=TRUE)$modulus)+
  sum(solve(Sigma)*S)-as.numeric(determinant(S,logarithm=TRUE)$modulus)-nrow(S)
spec<-magmaan::model_spec(syntax,fixed_x=FALSE)
warm<-spec; marker_start<-maps$marker_none$unpack(maps$marker_none$pack(start))$m
warm$partable$ustart<-vapply(seq_len(nrow(warm$partable)),function(k) {
  p<-warm$partable[k,];row_value(marker_start,p$lhs,p$op,p$rhs)
},0.0)
fits<-refs<-list(); exhausted<-FALSE
for(r in 0:o$reps) {
  if(elapsed()>o$budget_sec) {exhausted<-TRUE;break}
  S<-case$S
  if(r>0) {set.seed(seed+r);S<-rWishart(1,case$n-1,implied(case$m)$Sigma)[,,1]/case$n}
  dimnames(S)<-dimnames(case$S)
  tt<-elapsed()
  lf<-suppressWarnings(lavaan::sem(syntax,sample.cov=S,sample.nobs=case$n,
          sample.cov.rescale=FALSE,meanstructure=FALSE,fixed.x=FALSE))
  lm<-lavaan::lavInspect(lf,'est'); lc<-lavaan::lavInspect(lf,'converged')
  lv_min<-min(min_eigen(lm$psi),min_eigen(lm$theta))
  row<-data.frame(replicate=r,engine='lavaan_unrestricted',start='native',
    objective=2*as.numeric(lavaan::fitMeasures(lf,'fmin')),seconds=elapsed()-tt,
    converged=lc,admissible=lv_min>=-1e-8,cone_stationary=NA,
    nullity=NA_integer_,raw_gradient=NA_real_,min_fraction=NA_real_,error='')
  refs[[length(refs)+1L]]<-row
  for(st in c('common','native')) {
    if(elapsed()>o$budget_sec) {exhausted<-TRUE;break}
    tt<-elapsed()
    pf<-tryCatch(magmaan::frontier_fit_ml_psd(if(st=='common') warm else spec,
      list(S=list(S),mean=list(rep(0,nrow(S))),nobs=case$n),
      control=list(max_iter=1000L,gtol=1e-8,ftol=1e-12)),error=function(e)e)
    err<-inherits(pf,'error'); u<-if(err) NULL else from_partable(pf$partable)
    if(!err && abs(objective(u$Sigma,S)-2*pf$fmin)>1e-8) stop('Native PSD projection mismatch')
    gs<-if(err) NULL else pf$diagnostics$geometric_stationarity
    refs[[length(refs)+1L]]<-data.frame(replicate=r,engine='magmaan_psd',start=st,
      objective=if(err) NA else 2*pf$fmin,seconds=elapsed()-tt,
      converged=if(err) FALSE else isTRUE(pf$converged),
      admissible=if(err) FALSE else isTRUE(pf$diagnostics$admissibility$admissible),
      cone_stationary=if(err) FALSE else isTRUE(gs$cone_stationary),
      nullity=if(err) NA_integer_ else gs$covariance_nullity,
      raw_gradient=if(err) NA_real_ else gs$raw_gradient_inf,
      min_fraction=if(err) NA_real_ else min(boundary_diagnostics(u)),
      error=if(err) conditionMessage(pf) else '')
  }
  write.csv(do.call(rbind,refs),file.path(o$results_dir,'references.csv'),row.names=FALSE)
  if(exhausted) break
  # Rotation prevents one chart always paying process warmup/order effects.
  order<-names(maps)[((seq_along(maps)+r-1L)%%length(maps))+1L]
  for(label in order) {
    if(elapsed()>o$budget_sec) {exhausted<-TRUE;break}
    map<-maps[[label]]; parts<-strsplit(label,'_',fixed=TRUE)[[1]]
    ft<-tryCatch(fit_chart(map,map$pack(start),S,200L),error=function(e)e)
    err<-inherits(ft,'error')
    dual<-if(err) Inf else tryCatch({
      g<-geometry(map,ft$fit$par)
      if(g$rank<g$npar) Inf else sqrt(max(0,sum(ft$gradient*solve(g$information,ft$gradient))))
    },error=function(e)Inf)
    boundary<-if(err) NA_real_ else min(boundary_diagnostics(ft$terminal))
    good<-!err && min_eigen(ft$terminal$m$psi)>0 && min_eigen(ft$terminal$m$theta)>0
    fits[[length(fits)+1L]]<-data.frame(replicate=r,chart=parts[1],method=parts[2],
      objective=if(err) NA else ft$fit$value,seconds=if(err) NA else ft$seconds,
      evaluations=if(err) NA else ft$evaluations,invalid=if(err) NA else ft$invalid,
      stationary=is.finite(dual)&&dual<1e-4,gradient_norm=dual,admissible=good,
      min_fraction=boundary,selected=parts[1]==chosen && parts[2]=='full',
      error=if(err) conditionMessage(ft) else '')
    write.csv(do.call(rbind,fits),file.path(o$results_dir,'fits.csv'),row.names=FALSE)
    cat(sprintf('draw %d/%d %-22s %.2fs elapsed\n',r,o$reps,label,elapsed()));flush.console()
  }
  if(exhausted) break
}
refs<-if(length(refs)) do.call(rbind,refs) else data.frame()
if(length(fits)) {
  fits<-do.call(rbind,fits)
  fits$reference_objective<-NA_real_;fits$reference_class<-'unresolved_reference'
  for(r in unique(fits$replicate)) {
    rr<-refs[refs$replicate==r & refs$engine=='magmaan_psd',]
    accepted<-rr$converged & rr$admissible & rr$cone_stationary
    if(nrow(rr)==2 && all(accepted) && diff(range(rr$objective))<1e-7) {
      ii<-fits$replicate==r
      fits$reference_objective[ii]<-min(rr$objective)
      fits$reference_class[ii]<-if(all(rr$nullity>0)) 'psd_boundary_candidate' else
        if(all(rr$nullity==0)) 'interior_candidate' else 'unresolved_activity'
    }
  }
  fits$gap_to_psd<-fits$objective-fits$reference_objective
  fits$classification<-ifelse(!fits$admissible,'invalid',
    ifelse(is.finite(fits$gap_to_psd)&fits$gap_to_psd< -1e-7,'better_than_reference',
      ifelse(fits$reference_class=='psd_boundary_candidate','boundary_needs_cone_audit',
        ifelse(fits$reference_class=='interior_candidate' & fits$stationary & is.finite(fits$gap_to_psd)&abs(fits$gap_to_psd)<1e-7,
          'matched_interior',ifelse(is.finite(fits$min_fraction)&fits$min_fraction<1e-6,
            'near_boundary_stall','unresolved_interior')))))
  fits$paired_complete<-ave(fits$chart,fits$replicate,FUN=function(x) as.character(length(x)==9))=='TRUE'
  write.csv(fits,file.path(o$results_dir,'fits.csv'),row.names=FALSE)
  # Replay the prespecified fast-path rule; reference class/objective are never
  # used to choose the route. Both routes start from the same original model.
  routing<-lapply(which(fits$selected),function(k) {
    z<-fits[k,]
    fallback<-!z$admissible || !z$stationary || !is.finite(z$min_fraction) || z$min_fraction<1e-6
    rr<-refs[refs$replicate==z$replicate & refs$engine=='magmaan_psd' & refs$start=='common',]
    accepted<-if(fallback) nrow(rr)==1 && rr$converged && rr$admissible && rr$cone_stationary else TRUE
    data.frame(replicate=z$replicate,route=if(fallback) 'psd' else 'interior',
      accepted=accepted,objective=if(fallback && nrow(rr)==1) rr$objective else z$objective,
      fit_seconds=z$seconds+if(fallback && nrow(rr)==1) rr$seconds else 0,
      selector_seconds=algorithm$seconds)
  })
  write.csv(do.call(rbind,routing),file.path(o$results_dir,'routing.csv'),row.names=FALSE)
}
write_metadata(file.path(o$results_dir,'metadata.csv'),values=c(o,list(complete=!exhausted,
  elapsed_seconds=elapsed(),selected_chart=chosen,selected_method='full',
  selection_rule='minimum diagonal-equilibrated start information condition',
  setup_seconds=selection$setup_seconds[1],selector_seconds=algorithm$seconds,planned_fits=9*(o$reps+1),
  git_head=git_scalar(c('rev-parse','HEAD'),root=repo),
  source_md5=paste(tools::md5sum(c(script,file.path(here,'R/precondition.R'),
    file.path(here,'R/charts.R'),file.path(here,'R/models.R'))),collapse=','))),
  packages=c('magmaan','lavaan'))
cat('Results: ',o$results_dir,'\n',sep='')
if(exhausted) quit(status=2L)
