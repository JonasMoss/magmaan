# Fresh sampled numerical confirmation, holding the search/start recipe fixed.
run_audit_terminal <- function(args,here) {
  value <- function(k,default) {
    i <- match(k,args); if(is.na(i)) return(default)
    if(i==length(args) || startsWith(args[i+1L],'--')) stop('missing ',k)
    args[i+1L]
  }
  if(any(args %in% c('--help','-h'))) {
    cat('Usage: Rscript run_experiment.R --ordinary --audit-terminal --run-id NAME [--smoke] [--reps N] [--seed-base N] [--python PATH]\n',
      'Fresh sampled regular, weak-marker, mixed-unit, shared-loading, feedback and two-group cases.\n',
      'ULS PORT-NLS and ML PORT, marker and native sphere; common sample-only layered starts.\n',
      'Marker endpoints plus 1%/5% variance perturbations; sphere endpoints include the exact normalization map.\n',
      'Independent 90-digit derivatives/bounds/decisions. No default adoption or global-optimum claim.\n',sep='')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--audit-terminal','--run-id','--smoke','--reps','--seed-base','--python')
  if(any(startsWith(args,'--') & !args %in% known)) stop('unknown terminal option')
  smoke <- '--smoke' %in% args; reps <- as.integer(value('--reps',if(smoke) '1' else '5'))
  seed <- as.integer(value('--seed-base',if(smoke) '863261101' else '863261201'))
  run <- value('--run-id','audit-terminal')
  if(is.na(reps) || reps<1 || reps>50 || is.na(seed) || seed<1 || !grepl('^[A-Za-z0-9_-]+$',run)) stop('invalid options')
  out <- file.path(here,'results',run); if(dir.exists(out)) stop('choose a fresh run ID')
  require_pkg('magmaanlab'); dir.create(out,recursive=TRUE)
  write_out <- function(x,name) {
    for(k in names(x)) if(is.double(x[[k]])) x[[k]] <- sprintf('%.17g',x[[k]])
    write_csv(x,file.path(out,paste0(name,'.csv')))
  }
  matrices <- points <- rows <- fits <- covariances <- list()
  matrix_rows <- function(x,id,name) {
    x <- as.matrix(x); if(!length(x)) return(NULL)
    grid <- expand.grid(row=seq_len(nrow(x)),col=seq_len(ncol(x)))
    cbind(point_id=id,name=name,grid,value=sprintf('%.17g',as.vector(x)))
  }
  retain <- function(a,pt,case,estimator,chart,target,old,new,assessment,seconds,objective) {
    id <- length(rows)+1L; e <- a$derived_interval_input_errors
    z <- a$distance_interval_derived_inputs
    if(is.null(z)) z <- list(status=e$status,decision='unresolved',lower=0,upper=Inf,distance=NA_real_)
    rows[[id]] <<- data.frame(point_id=id,case_id=case,estimator=estimator,chart=chart,target=target,
      legacy_passed=old,selected_passed=new,selected_status=assessment,
      source_status=e$status,matrix_bound=e$matrix,vector_bound=e$vector,curvature_bound=e$curvature,
      curvature_lower=e$curvature_lower_bound,interval_status=z$status,decision=z$decision,
      distance=z$distance,lower=z$lower,upper=z$upper,seconds=seconds,recomputed_objective=objective)
    pp <- pt[c('lhs','op','rhs','group','free','est')]; pp$est <- sprintf('%.17g',pp$est)
    points[[id]] <<- cbind(point_id=id,pp)
    if(chart=='marker') {
      names <- c('derivative_basis','gradient','equilibrated_factor','factor_scale','metric_score_residual',
        'curvature_equilibrated_hessian','curvature_scale','curvature_coordinate_map')
      for(name in names) matrices[[length(matrices)+1L]] <<- matrix_rows(a[[name]],id,name)
    } else {
      for(name in c('point','tangent_basis','reduced_gradient','equilibrated_factor','factor_scale','metric_score_residual','curvature_scale'))
        matrices[[length(matrices)+1L]] <<- matrix_rows(a[[name]],id,name)
      matrices[[length(matrices)+1L]] <<- matrix_rows(a$curvature_system$equilibrated_hessian,id,'curvature_equilibrated_hessian')
      matrices[[length(matrices)+1L]] <<- matrix_rows(a$curvature_system$coordinate_map,id,'curvature_coordinate_map')
      map <- a$input_map
      for(name in c('offset','rest_basis','rounded_point'))
        matrices[[length(matrices)+1L]] <<- matrix_rows(map[[name]],id,paste0('map_',name))
      for(k in seq_along(map$spheres)) for(name in c('basis','units','parameters','offset'))
        matrices[[length(matrices)+1L]] <<- matrix_rows(map$spheres[[k]][[name]],id,paste0('sphere_',k,'_',name))
    }
  }
  families <- c('regular','weak','mixed','shared','feedback','groups')
  cases <- expand.grid(rep=seq_len(reps),family=families,stringsAsFactors=FALSE)
  start <- proc.time()[['elapsed']]
  for(case in seq_len(nrow(cases))) {
    family <- cases$family[case]; case_seed <- seed+match(family,families)*10000L+cases$rep[case]
    d <- designs_all()[[if(family=='weak') 'weak_marker' else 'ernst']]
    syntax <- model_syntax
    if(family=='shared') {
      syntax <- 'X =~ x1 + a*x2 + a*x3\nY =~ y1 + b*y2 + b*y3\nY ~ X'
      d$lx[3] <- d$lx[2]; d$ly[3] <- d$ly[2]
    }
    Sigma <- design_sigma(d)
    if(family=='feedback') {
      syntax <- 'X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nX ~ .2*Y\nY ~ .3*X'
      L <- matrix(0,6,2); L[1:3,1] <- d$lx; L[4:6,2] <- d$ly
      A <- solve(diag(2)-matrix(c(0,.3,.2,0),2)); Sigma <- L%*%A%*%t(A)%*%t(L)+diag(6)
      dimnames(Sigma) <- list(ov_names,ov_names)
    }
    group_count <- if(family=='groups') 2L else 1L
    spec <- magmaanlab::model_spec(syntax,fixed_x=FALSE,meanstructure=FALSE,
      group_labels=if(group_count==2L) c('g1','g2') else NULL)
    units <- if(family=='mixed') c(.01,100,2,.3,10,.1) else rep(1,6)
    samples <- lapply(seq_len(group_count),function(g) {
      population <- Sigma*(if(g==2L) 1.4 else 1)
      data <- as.matrix(draw_data(population,if(g==2L) 160L else 100L,case_seed+g*100L))*rep(units,each=if(g==2L)160L else 100L)
      stats::cov(data)*(nrow(data)-1)/nrow(data)
    })
    sample <- list(S=samples,nobs=if(group_count==2L) c(100L,160L) else 100L)
    for(g in seq_along(samples)) {
      grid <- expand.grid(row=1:6,col=1:6)
      covariances[[length(covariances)+1L]] <- cbind(case_id=case,block=g,nobs=sample$nobs[g],grid,value=sprintf('%.17g',as.vector(samples[[g]])))
    }
    initial <- magmaanlab::magmaan_core$estimate_start_values(spec$partable,sample,start='layered',transport='native')
    spec$partable$ustart[spec$partable$free>0] <- initial[spec$partable$free[spec$partable$free>0]]
    unbounded <- list(lower=rep(-Inf,length(initial)),upper=rep(Inf,length(initial)))
    for(estimator in c('ULS','ML')) for(chart in c('marker','sphere')) {
      clock <- proc.time()[['elapsed']]
      fit <- tryCatch(suppressWarnings({
        ctl <- list(start=initial,coordinate_scaling='sample_units')
        optimizer <- if(estimator=='ULS') 'port-nls' else 'port'
        if(chart=='sphere') {
          ctl$start <- NULL # user starts are already in the spec hints
          ctl$verified_newton <- TRUE
          magmaanlab::frontier_fit_sphere(spec,sample,estimator=estimator,optimizer=optimizer,
            control=ctl,bounds=unbounded,start='user',polish=FALSE)
        } else magmaanlab::fit_model(spec,sample,estimator=estimator,optimizer=optimizer,control=ctl,bounds=unbounded)
      }),error=identity)
      seconds <- proc.time()[['elapsed']]-clock
      native <- if(chart=='sphere' && !is.null(fit$gauge)) fit$gauge$native_audit else NULL
      returned <- !inherits(fit,'error') || !is.null(native)
      fits[[length(fits)+1L]] <- data.frame(case_id=case,family=family,rep=cases$rep[case],seed=case_seed,estimator=estimator,chart=chart,
        returned=returned,chart_available=!inherits(fit,'error'),seconds=seconds,
        fmin=if(is.null(native)) fit$fmin %||% NA_real_ else native$fmin,
        legacy_passed=if(is.null(native)) fit$converged %||% NA else native$compatibility_assessment$converged,
        message=if(inherits(fit,'error'))conditionMessage(fit) else '')
      if(!returned) {saveRDS(fit,file.path(out,paste0('failure_',case,'_',estimator,'_',chart,'.rds'))); next}
      if(chart=='sphere') {
        pt <- fit$gauge$sphere_partable
        retain(native,pt,case,estimator,chart,0,native$compatibility_assessment$converged,
          native$converged,native$status,seconds,native$fmin)
      } else for(target in c(0,.01,.05)) {
        theta <- fit$theta
        row <- which(spec$partable$lhs=='x1' & spec$partable$op=='~~' & spec$partable$rhs=='x1' & spec$partable$group==1L)[1]
        theta[spec$partable$free[row]] <- theta[spec$partable$free[row]]+target*samples[[1]][1,1]
        audit <- list(retain_newton_artifacts=TRUE,verified_newton=TRUE)
        if(target==0) audit$reported_objective <- fit$fmin
        ev <- magmaanlab::magmaan_core$estimate_evaluate_at(spec$partable,sample,theta,estimator=estimator,bounds=unbounded,
          audit_options=audit)
        pt <- ev$partable
        retain(ev$newton_audit,pt,case,estimator,chart,target,ev$converged_compatibility,
          ev$converged,ev$verified_convergence$status,seconds,ev$fmin)
      }
    }
    cat(sprintf('Terminal case %d/%d (%s); %.1fs\n',case,nrow(cases),family,proc.time()[['elapsed']]-start))
  }
  write_out(do.call(rbind,rows),'intervals'); write_out(do.call(rbind,points),'points')
  write_out(do.call(rbind,matrices),'artifacts'); write_out(do.call(rbind,covariances),'covariances')
  write_out(do.call(rbind,fits),'fits'); write_out(cbind(case_id=seq_len(nrow(cases)),cases),'cases')
  ref <- magmaan_cache_ref()
  sources <- file.path(here,c('run_experiment.R','R/audit_terminal.R','scripts/audit_terminal_reference.py',
    '../../../../cpp/src/estimate/frontier/newton_input_bounds.cpp','../../../../cpp/src/estimate/frontier/sphere.cpp',
    '../../../../cpp/src/estimate/frontier/convergence_policy.cpp','../../../../r-package/src/fit.cpp'))
  write_metadata(file.path(out,'metadata.csv'),values=list(lane='audit_terminal',seed_base=seed,reps=reps,cases=nrow(cases),points=length(rows),
    command=paste(commandArgs(),collapse=' '),evaluation_elapsed_s=proc.time()[['elapsed']]-start,threads=1,
    package_dll_md5=unname(tools::md5sum(getLoadedDLLs()[['magmaanlab']][['path']])),
    source_md5=paste(tools::md5sum(sources),collapse=';'),git_head=ref$git_head,git_dirty=ref$git_dirty,
    start='common sample-only layered/native start; sphere user start; no polish',
    control='ULS PORT-NLS and ML PORT, sample_units; audit changes no search controls',
    interpretation='fresh sampled numerical confirmation, no default adoption or global-optimality statement'),packages='magmaanlab')
  status <- system2(value('--python','python3'),c(shQuote(file.path(here,'scripts/audit_terminal_reference.py')),'--run-dir',shQuote(normalizePath(out))))
  if(status!=0) stop('terminal reference failed; inputs retained')
  cat('Terminal evidence saved: ',out,'\n',sep='')
}
