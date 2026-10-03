# A focused search diagnosis: the ULS target, domain and audit budget stay fixed.
run_uls_search <- function(args,here) {
  value <- function(k,default) {i<-match(k,args);if(is.na(i))return(default);if(i==length(args)||startsWith(args[i+1L],'--'))stop('missing ',k);args[i+1L]}
  if(any(args %in% c('--help','-h'))) {
    cat('Usage: Rscript run_experiment.R --ordinary --uls-search --run-id NAME [--phase retained|fresh] [--smoke] [--reps N] [--seed-base N] [--controls stock|tight] [--skip-reference] [--python PATH]\n',
      'Retained: ten weak/mixed cases from audit-terminal-final-v1; smoke takes one of each.\n',
      'Fresh: regular, weak-marker and mixed-unit draws; default ten per family, smoke one.\n',
      'Default-start API calls, shared default physical start, layered/native, four signed starts.\n',
      'Marker/sphere, PORT-NLS and sample_units fixed; infinite bounds; no sphere polish.\n',
      'Portfolios select the lowest original objective among interval-qualified endpoints only.\n',
      '90-digit checks cover default, layered and selected portfolio endpoints; no default adoption.\n',sep='');return(invisible(NULL))
  }
  known<-c('--ordinary','--uls-search','--run-id','--phase','--smoke','--reps','--seed-base','--controls','--skip-reference','--python')
  if(any(startsWith(args,'--') & !args %in% known))stop('unknown search option')
  phase<-value('--phase','retained');smoke<-'--smoke'%in%args
  reps<-as.integer(value('--reps',if(smoke)'1' else '10'));seed<-as.integer(value('--seed-base',if(smoke)'974261101' else '974261201'))
  controls<-value('--controls','stock');run<-value('--run-id','uls-search')
  if(!phase%in%c('retained','fresh') || !controls%in%c('stock','tight') || is.na(reps)||reps<1||reps>100||is.na(seed)||seed<1||!grepl('^[A-Za-z0-9_-]+$',run))stop('invalid search options')
  out<-file.path(here,'results',run);if(dir.exists(out))stop('choose a fresh run ID');dir.create(out,recursive=TRUE)
  require_pkg('magmaanlab')
  source(file.path(here,'R/uls_reliability_study.R'))
  write_out<-function(x,name) {
    for(k in names(x))if(is.double(x[[k]]))x[[k]]<-sprintf('%.17g',x[[k]])
    write_csv(x,file.path(out,paste0(name,'.csv')))
  }
  spec<-magmaanlab::model_spec(model_syntax,fixed_x=FALSE,meanstructure=FALSE)
  q<-max(spec$partable$free);unbounded<-list(lower=rep(-Inf,q),upper=rep(Inf,q))
  tasks<-samples<-list();old_dir<-file.path(here,'results/audit-terminal-final-v1')
  if(phase=='retained') {
    old<-read.csv(file.path(old_dir,'covariances.csv'));cases<-read.csv(file.path(old_dir,'cases.csv'))
    for(id in if(smoke)c(6L,11L) else 6:15) {
      z<-old[old$case_id==id,];S<-matrix(0,6,6,dimnames=list(ov_names,ov_names));S[cbind(z$row,z$col)]<-z$value
      samples[[length(samples)+1L]]<-list(S=list(S),nobs=100L)
      tasks[[length(tasks)+1L]]<-data.frame(case_id=length(tasks)+1L,source_case=id,family=cases$family[cases$case_id==id],rep=cases$rep[cases$case_id==id],seed=NA_integer_,phase=phase)
    }
  } else for(family in c('regular','weak','mixed'))for(rep in seq_len(reps)) {
    case_seed<-seed+match(family,c('regular','weak','mixed'))*10000L+rep
    d<-designs_all()[[if(family=='weak')'weak_marker' else 'ernst']]
    units<-if(family=='mixed')c(.01,100,2,.3,10,.1) else rep(1,6)
    data<-as.matrix(draw_data(design_sigma(d),100L,case_seed))*rep(units,each=100L)
    samples[[length(samples)+1L]]<-list(S=list(stats::cov(data)*99/100),nobs=100L)
    tasks[[length(tasks)+1L]]<-data.frame(case_id=length(tasks)+1L,source_case=NA_integer_,family=family,rep=rep,seed=case_seed,phase=phase)
  }
  tasks<-do.call(rbind,tasks);write_out(tasks,'cases')
  rows<-covariances<-starts_rows<-points<-matrices<-intervals<-portfolios<-list();clock<-proc.time()[['elapsed']]
  matrix_rows<-function(x,id,name){x<-as.matrix(x);if(!length(x))return(NULL);grid<-expand.grid(row=seq_len(nrow(x)),col=seq_len(ncol(x)));cbind(point_id=id,name=name,grid,value=sprintf('%.17g',as.vector(x)))}
  retain<-function(endpoint,id,case,chart,row) {
    a<-endpoint$a;pt<-endpoint$pt
    pp<-pt[c('lhs','op','rhs','group','free','est')];pp$est<-sprintf('%.17g',pp$est);points[[length(points)+1L]]<<-cbind(point_id=id,pp)
    e<-a$derived_interval_input_errors;z<-a$distance_interval_derived_inputs
    if(is.null(z))z<-list(status=e$status,decision='unresolved',lower=0,upper=Inf,distance=NA_real_)
    intervals[[length(intervals)+1L]]<<-data.frame(point_id=id,case_id=case,estimator='ULS',chart=chart,target=0,
      legacy_passed=row$legacy_passed,selected_passed=row$qualified,selected_status=row$selected_status,
      source_status=e$status,matrix_bound=e$matrix,vector_bound=e$vector,curvature_bound=e$curvature,
      curvature_lower=e$curvature_lower_bound,interval_status=z$status,decision=z$decision,distance=z$distance,lower=z$lower,upper=z$upper,
      seconds=row$seconds,recomputed_objective=row$objective)
    if(chart=='marker')for(name in c('derivative_basis','gradient','equilibrated_factor','factor_scale','metric_score_residual','curvature_equilibrated_hessian','curvature_scale','curvature_coordinate_map'))
      matrices[[length(matrices)+1L]]<<-matrix_rows(a[[name]],id,name)
    else {
      for(name in c('point','tangent_basis','reduced_gradient','equilibrated_factor','factor_scale','metric_score_residual','curvature_scale'))matrices[[length(matrices)+1L]]<<-matrix_rows(a[[name]],id,name)
      matrices[[length(matrices)+1L]]<<-matrix_rows(a$curvature_system$equilibrated_hessian,id,'curvature_equilibrated_hessian')
      matrices[[length(matrices)+1L]]<<-matrix_rows(a$curvature_system$coordinate_map,id,'curvature_coordinate_map')
      for(name in c('offset','rest_basis','rounded_point'))matrices[[length(matrices)+1L]]<<-matrix_rows(a$input_map[[name]],id,paste0('map_',name))
      for(k in seq_along(a$input_map$spheres))for(name in c('basis','units','parameters','offset'))matrices[[length(matrices)+1L]]<<-matrix_rows(a$input_map$spheres[[k]][[name]],id,paste0('sphere_',k,'_',name))
    }
  }
  for(case in seq_along(samples)) {
    sample<-samples[[case]];S<-sample$S[[1]]
    grid<-expand.grid(row=1:6,col=1:6);covariances[[case]]<-cbind(case_id=case,block=1L,nobs=100L,grid,value=sprintf('%.17g',as.vector(S)))
    default<-magmaanlab::magmaan_core$estimate_start_values(spec$partable,sample,start='fabin3',transport='native')
    starts<-list(shared_default=default,layered=magmaanlab::magmaan_core$estimate_start_values(spec$partable,sample,start='layered',transport='native'))
    for(sx in c(1,-1))for(sy in c(1,-1))starts[[paste0('signed_',sx,'_',sy)]]<-uls_study_transport(uls_signed_moment_point(sample,sx,sy),spec)$theta
    for(k in names(starts))starts_rows[[length(starts_rows)+1L]]<-data.frame(case_id=case,start_id=k,parameter=seq_len(q),value=as.numeric(starts[[k]]))
    arms<-rbind(data.frame(chart=c('marker','sphere'),start_id='api_default'),
      data.frame(chart='sphere',start_id='shared_default'),expand.grid(chart=c('marker','sphere'),start_id=setdiff(names(starts),'shared_default'),stringsAsFactors=FALSE))
    endpoints<-list();case_rows<-list()
    for(j in seq_len(nrow(arms))) {
      chart<-arms$chart[j];start_id<-arms$start_id[j];target<-spec;ctl<-list(coordinate_scaling='sample_units')
      if(controls=='tight')ctl$port<-list(rel_f_tol=1e-14,abs_f_tol=0,x_tol=0,false_conv_tol=0,max_iter=10000L,max_eval=100000L)
      if(start_id!='api_default') {
        target$partable$ustart[target$partable$free>0]<-starts[[start_id]][target$partable$free[target$partable$free>0]]
        if(chart=='marker')ctl$start<-as.numeric(starts[[start_id]])
      }
      begin<-proc.time()[['elapsed']];warnings<-character()
      fit<-tryCatch(withCallingHandlers({
        if(chart=='sphere') {ctl$verified_newton<-TRUE;magmaanlab::frontier_fit_sphere(target,sample,estimator='ULS',optimizer='port-nls',control=ctl,
          bounds=unbounded,start=if(start_id=='api_default')'canonical' else 'user',polish=FALSE)}
        else magmaanlab::fit_model(target,sample,estimator='ULS',optimizer='port-nls',control=ctl,bounds=unbounded)
      },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=identity)
      native<-if(chart=='sphere')fit$gauge$native_audit else NULL
      returned<-!inherits(fit,'error')||!is.null(native);a<-pt<-ev<-NULL
      if(returned) {
        if(!inherits(fit,'error')) {
          expected<-if(start_id=='api_default')default else starts[[start_id]]
          if(!isTRUE(all.equal(as.numeric(fit$start$theta),as.numeric(expected),tolerance=0)))stop('API start provenance differs')
        }
        if(chart=='sphere'){a<-native;pt<-fit$gauge$sphere_partable}
        else {
          ev<-magmaanlab::magmaan_core$estimate_evaluate_at(spec$partable,sample,fit$theta,estimator='ULS',bounds=unbounded,
            audit_options=list(verified_newton=TRUE,reported_objective=fit$fmin))
          a<-ev$newton_audit;pt<-ev$partable
        }
      }
      seconds<-proc.time()[['elapsed']]-begin
      assessment<-if(!returned)NULL else if(chart=='sphere')a else ev$verified_convergence
      source<-a$derived_interval_input_errors;z<-a$distance_interval_derived_inputs
      id<-length(rows)+1L;objective<-if(!returned)NA_real_ else if(chart=='sphere')a$fmin else ev$fmin
      consistent<-returned && assessment$objective_consistency$status=='passed'
      qualified<-returned && assessment$status=='passed'
      point<-if(returned)uls_study_point(pt) else NULL
      phi_min<-if(returned)min(eigen(point$phi,symmetric=TRUE,only.values=TRUE)$values) else NA_real_
      row<-cbind(tasks[case,],point_id=id,chart=chart,start_id=start_id,controls=controls,returned=returned,chart_available=!inherits(fit,'error'),
        objective=objective,reported_objective=if(is.null(native))fit$fmin%||%NA_real_ else a$reported_fmin,
        objective_consistent=consistent,qualified=qualified,selected_status=assessment$status%||%'unavailable',
        legacy_passed=if(!returned)NA else if(chart=='sphere')a$compatibility_assessment$converged else ev$converged_compatibility,
        source_status=source$status%||%'unavailable',decision=z$decision%||%'unresolved',distance=z$distance%||%NA_real_,
        lower=z$lower%||%0,upper=z$upper%||%Inf,curvature_lower=source$curvature_lower_bound%||%0,
        residual_variance_min=if(returned)min(point$residual) else NA_real_,phi_min=phi_min,
        optimizer_status=if(!returned)'error' else if(chart=='sphere')fit$gauge$optimizer_status else fit$optimizer_status,
        iterations=if(chart=='sphere')fit$gauge$iterations%||%NA_integer_ else fit$iterations%||%NA_integer_,
        f_evals=fit$f_evals%||%NA_integer_,seconds=seconds,message=if(inherits(fit,'error'))conditionMessage(fit) else '',warnings=paste(unique(warnings),collapse=' | '))
      rows[[id]]<-row;case_rows[[j]]<-row;endpoints[[j]]<-list(a=a,pt=pt,point=point)
    }
    d<-do.call(rbind,case_rows);chosen<-which(d$start_id %in%c('api_default','layered') & d$returned)
    for(chart in c('marker','sphere'))for(portfolio in c('api_default','signed_only','default_signed','all')) {
      ids<-switch(portfolio,api_default='api_default',signed_only=grep('^signed_',d$start_id,value=TRUE),
        default_signed=c('api_default',grep('^signed_',d$start_id,value=TRUE)),all=d$start_id)
      candidates<-which(d$chart==chart & d$start_id%in%ids)
      qualified_rows<-candidates[d$qualified[candidates]]
      best<-if(length(qualified_rows))qualified_rows[which.min(d$objective[qualified_rows])] else NA_integer_
      finite_rows<-candidates[d$objective_consistent[candidates] & is.finite(d$objective[candidates])]
      low<-if(length(finite_rows))finite_rows[which.min(d$objective[finite_rows])] else NA_integer_
      if(!is.na(best))chosen<-c(chosen,best)
      if(portfolio=='all' && !is.na(low))chosen<-c(chosen,low)
      portfolios[[length(portfolios)+1L]]<-cbind(tasks[case,],chart=chart,portfolio=portfolio,controls=controls,calls=length(candidates),
        qualified=!is.na(best),selected_id=if(is.na(best))NA_integer_ else d$point_id[best],selected_start=if(is.na(best))'' else d$start_id[best],
        objective=if(is.na(best))NA_real_ else d$objective[best],lowest_finite_id=if(is.na(low))NA_integer_ else d$point_id[low],
        lowest_finite_objective=if(is.na(low))NA_real_ else d$objective[low],seconds=sum(d$seconds[candidates]))
    }
    for(j in unique(chosen))retain(endpoints[[j]],d$point_id[j],case,d$chart[j],d[j,])
    saveRDS(list(rows=d,endpoints=endpoints,sample=sample,starts=starts),file.path(out,sprintf('case_%03d.rds',case)))
    write_out(do.call(rbind,rows),'fits');write_out(do.call(rbind,portfolios),'portfolios')
    cat(sprintf('Search case %d/%d (%s): %d/%d endpoints qualified; %.1fs\n',case,nrow(tasks),tasks$family[case],sum(d$qualified),nrow(d),proc.time()[['elapsed']]-clock))
  }
  write_out(do.call(rbind,starts_rows),'starts');write_out(do.call(rbind,covariances),'covariances')
  write_out(do.call(rbind,points),'points');write_out(do.call(rbind,matrices),'artifacts');write_out(do.call(rbind,intervals),'intervals')
  ref<-magmaan_cache_ref();sources<-file.path(here,c('run_experiment.R','R/uls_search.R','R/uls_reliability_study.R','R/designs.R','scripts/audit_terminal_reference.py'))
  write_metadata(file.path(out,'metadata.csv'),values=list(lane='uls_search',phase=phase,controls=controls,seed_base=seed,reps=reps,cases=nrow(tasks),fits=length(rows),reference_points=length(intervals),
    fit_elapsed_s=proc.time()[['elapsed']]-clock,threads=1,source_md5=paste(tools::md5sum(sources),collapse=';'),
    package_dll_md5=unname(tools::md5sum(getLoadedDLLs()[['magmaanlab']][['path']])),git_head=ref$git_head,git_dirty=ref$git_dirty,
    retained_input_md5=if(phase=='retained')unname(tools::md5sum(file.path(old_dir,'covariances.csv'))) else '',
    baseline='API no start override; ULS fabin3/native; PORT-NLS/sample_units fixed, not the full no-options optimizer default',
    portfolio='lowest original objective among interval-qualified, original-objective-consistent endpoints; no default adoption',
    reference_subset='API defaults, layered controls, qualified portfolio winners and lowest finite endpoints'),packages='magmaanlab')
  if(!'--skip-reference'%in%args) {
    status<-system2(value('--python','python3'),c(shQuote(file.path(here,'scripts/audit_terminal_reference.py')),'--run-dir',shQuote(normalizePath(out))))
    if(status!=0)stop('search reference failed; raw fits retained')
  }
  cat('Search evidence saved: ',out,'\n',sep='')
}
