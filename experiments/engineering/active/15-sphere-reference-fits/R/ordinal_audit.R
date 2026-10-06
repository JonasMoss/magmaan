# Numerical integration study; no sampling-law or default-adoption decision.
run_ordinal_audit <- function(args, here) {
  value <- function(k, default) {
    i <- match(k,args); if(is.na(i)) return(default)
    if(i==length(args)) stop('missing ',k)
    args[i+1L]
  }
  if(any(args %in% c('--help','-h'))) {
    cat('Usage: --ordinary --ordinal-audit --run-id NAME [--smoke] [--reps N] [--seed-base N] [--python PATH]\n',
        '--sparse-search restricts to regular/sparse controls and adds bounded tight L-BFGS. --retained replays saved case 9 from ordinal-audit-confirm-v1.\n',
        'All-ordinal ULS/DWLS/WLS and mixed DWLS/WLS; delta/theta, actual default starts, default and PORT-NLS controls.\n',
        'Retains original prepared maps and actual weights; independent 90-digit point checks. Construction bounds remain unsupported.\n',sep='')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--ordinal-audit','--run-id','--smoke','--reps','--seed-base','--python','--sparse-search','--retained')
  if(any(startsWith(args,'--') & !args %in% known)) stop('unknown ordinal option')
  seed <- as.integer(value('--seed-base','865051201'))
  reps <- as.integer(value('--reps','1'))
  run <- value('--run-id','ordinal-audit')
  if(is.na(seed)||seed<1||is.na(reps)||reps<1||reps>10||!grepl('^[A-Za-z0-9_-]+$',run))stop('invalid options')
  out <- file.path(here,'results',run); if(dir.exists(out))stop('choose fresh run ID')
  require_pkg('magmaanlab'); dir.create(out,recursive=TRUE)
  core <- magmaanlab::magmaan_core
  write_out <- function(x,name) write_csv(x,file.path(out,paste0(name,'.csv')))
  matrix_rows <- function(x,id,name) {
    x <- as.matrix(x); if(!length(x))return(NULL)
    cbind(id=id,name=name,expand.grid(row=seq_len(nrow(x)),col=seq_len(ncol(x))),value=sprintf('%.17g',as.vector(x)))
  }
  cases <- fits <- points <- artifacts <- samples <- audits <- controls <- list()
  capture_audit <- function(fit,theta,fid,cid,family,param,method,arm,target) {
    begin <- proc.time()[['elapsed']]
    a <- core$frontier_ordinal_newton_audit(fit,theta)
    id <- length(audits)+1L
    audits[[id]] <<- data.frame(point_id=id,fit_id=fid,case_id=cid,family=family,parameterization=param,method=method,arm=arm,target=target,
      objective=sprintf('%.17g',a$objective),distance=sprintf('%.17g',a$diagnostics$distance),status=a$diagnostics$status,passed=a$diagnostics$passed,
      construction_status=a$construction_status,audit_seconds=proc.time()[['elapsed']]-begin)
    pt <- a$partable[c('lhs','op','rhs','group','free','est')]; pt$est <- sprintf('%.17g',pt$est)
    points[[id]] <<- cbind(point_id=id,pt)
    for(name in c('theta','gradient','hessian','metric','metric_factor','metric_score_residual','curvature_correction','derivative_basis','whitened_jacobian','whitened_residual'))
      artifacts[[length(artifacts)+1L]] <<- matrix_rows(a[[name]],id,name)
    for(g in seq_along(a$weight_factors))artifacts[[length(artifacts)+1L]] <<- matrix_rows(a$weight_factors[[g]],id,paste0('F_',g))
  }
  sparse_search <- '--sparse-search' %in% args
  retained <- '--retained' %in% args
  if(retained && !sparse_search)stop('--retained requires --sparse-search')
  if(retained)reps <- 1L
  families <- if(retained)'sparse' else if(sparse_search)c('regular','sparse') else if('--smoke' %in% args)c('regular','mixed') else c('regular','shared','groups','mixed','sparse','extreme','saddle')
  t0 <- proc.time()[['elapsed']]
  for(family in families) for(rep in seq_len(reps)) {
    cid <- length(cases)+1L; mixed <- family=='mixed'; groups <- if(family=='groups')2L else 1L
    draw_seed <- seed+match(family,c('regular','shared','groups','mixed','sparse','extreme','saddle'))*10000L+rep
    set.seed(draw_seed)
    X <- lapply(seq_len(groups),function(g) {
      n <- if(g==2)220L else 180L
      latent <- rnorm(n); residual_sd <- if(family=='extreme').2 else 1
      x <- outer(latent,c(.85,.8,.7,.9))+matrix(rnorm(n*4,sd=residual_sd),n,4)
      for(j in if(mixed)1:2 else 1:4) x[,j] <- 1L+(x[,j]>if(family=='sparse')1.7 else -.5)+(x[,j]>if(family=='sparse')2.1 else .6)
      if(mixed) x[,3:4] <- sweep(x[,3:4],2,c(.2,10),'*')
      colnames(x) <- paste0('x',1:4); x
    })
    if(retained) {
      saved <- readRDS(file.path(here,'results/ordinal-audit-confirm-v1/case_9.rds'))
      X <- saved$X; producer <- saved$stats; draw_seed <- 865101202L
    } else producer <- tryCatch(if(mixed)core$data_mixed_ordinal_stats_from_raw(X,c(TRUE,TRUE,FALSE,FALSE)) else core$data_ordinal_stats_from_raw(X),error=identity)
    cases[[cid]] <- data.frame(case_id=cid,family=family,rep=rep,seed=draw_seed,mixed=mixed,
      producer_returned=!inherits(producer,'error'),producer_error=if(inherits(producer,'error'))conditionMessage(producer) else '')
    saveRDS(list(X=X,stats=producer),file.path(out,paste0('case_',cid,'.rds')))
    write_out(do.call(rbind,cases),'cases')
    if(inherits(producer,'error'))next
    # The raw producer is name-free; attach the same observation layout that
    # the dataframe constructors retain before invoking model wrappers.
    producer$ov_names <- lapply(X,colnames)
    producer$ordered <- if(mixed)c('x1','x2') else paste0('x',1:4)
    for(g in seq_len(groups)) {
      for(name in c('R','thresholds','threshold_ov','threshold_level','moments','NACOV','W_dwls','W_wls'))
        if(!is.null(producer[[name]]))samples[[length(samples)+1L]] <- matrix_rows(producer[[name]][[g]],cid,paste0(name,'_',g))
      samples[[length(samples)+1L]] <- matrix_rows(producer$nobs[g],cid,paste0('n_',g))
      samples[[length(samples)+1L]] <- matrix_rows(if(mixed)c(1,1,0,0) else rep(1,4),cid,paste0('ordered_',g))
      samples[[length(samples)+1L]] <- matrix_rows(X[[g]],cid,paste0('X_',g))
    }
    for(param in c('delta','theta')) {
      syntax <- if(family=='shared')'f =~ x1 + a*x2 + a*x3 + x4' else 'f =~ x1 + x2 + x3 + x4'
      spec <- magmaanlab::model_spec(syntax,ordered=if(mixed)c('x1','x2') else paste0('x',1:4),parameterization=param,
        fixed_x=FALSE,std_lv=family=='saddle',group_labels=if(groups==2)c('g1','g2') else NULL,group_equal=if(groups==2)'thresholds' else NULL)
      if(cid==1L && param=='delta') {
        bad <- producer; bad$W_wls <- lapply(bad$W_wls,function(x)matrix(numeric(),0,0))
        check <- tryCatch(core$fit_wls_ordinal(spec,bad),error=identity)
        controls[[length(controls)+1L]] <- data.frame(control='unavailable_wls_weight',
          rejected=inherits(check,'error'),message=if(inherits(check,'error'))conditionMessage(check) else '')
        constant <- X; constant[[1]][,1] <- 1
        check <- tryCatch(core$data_ordinal_stats_from_raw(constant),error=identity)
        controls[[length(controls)+1L]] <- data.frame(control='constant_category',
          rejected=inherits(check,'error'),message=if(inherits(check,'error'))conditionMessage(check) else '')
      }
      for(method in if(mixed)c('DWLS','WLS') else c('ULS','DWLS','WLS')) for(arm in if(sparse_search)c('default','port_nls','tight_lbfgs') else c('default','port_nls')) {
        fit_fun <- if(mixed)core[[paste0('fit_',tolower(method),'_mixed_ordinal')]] else core[[paste0('fit_',tolower(method),'_ordinal')]]
        begin <- proc.time()[['elapsed']]
        ctl <- switch(arm,default=NULL,port_nls=list(max_iter=1000),
          tight_lbfgs=list(nlopt=list(ftol_rel=1e-14,xtol_rel=1e-12,max_eval=3000)))
        optimizer <- if(arm=='port_nls')'port-nls' else 'nlopt-lbfgs'
        fit <- tryCatch(suppressWarnings(if(arm=='default')fit_fun(spec,producer) else
          fit_fun(spec,producer,optimizer=optimizer,control=ctl)),error=identity)
        fid <- length(fits)+1L
        saveRDS(fit,file.path(out,paste0('fit_',fid,'.rds')))
        fits[[fid]] <- data.frame(fit_id=fid,case_id=cid,family=family,parameterization=param,method=method,arm=arm,
          optimizer=optimizer,max_iter=if(arm=='tight_lbfgs')3000L else 1000L,
          returned=!inherits(fit,'error'),fit_passed=if(inherits(fit,'error'))NA else fit$converged,
          fmin=if(inherits(fit,'error'))NA else fit$fmin,seconds=proc.time()[['elapsed']]-begin,
          optimizer_status=if(inherits(fit,'error'))'' else fit$optimizer_status,
          objective_consistent=if(inherits(fit,'error'))NA else fit$audit$f_consistent,
          iterations=if(inherits(fit,'error'))NA else fit$iterations,
          f_evals=if(inherits(fit,'error'))NA else fit$f_evals,
          message=if(inherits(fit,'error'))conditionMessage(fit) else '')
        write_out(do.call(rbind,fits),'fits')
        if(inherits(fit,'error'))next
        targets <- if(sparse_search)if(retained)c('terminal','start') else 'terminal' else if(family=='saddle' && arm=='default')c('terminal','displaced','saddle') else if(arm=='default')c('terminal','displaced') else 'terminal'
        for(target in targets) {
          theta <- fit$theta
          if(target=='start')theta <- fit$start$theta
          if(target=='displaced')theta <- theta+.025*sin(seq_along(theta))
          if(target=='saddle')theta[fit$partable$free[fit$partable$op=='=~']] <- 0
          capture_audit(fit,theta,fid,cid,family,param,method,arm,target)
        }
      }
    }
    elapsed <- proc.time()[['elapsed']]-t0
    cat(sprintf('Ordinal case %d: %s; %d fits, %d points; %.1fs\n',cid,family,length(fits),length(audits),elapsed))
  }
  for(name in c('cases','fits','points','artifacts','samples','audits','controls'))write_out(do.call(rbind,get(name)),name)
  if(sparse_search) {
    rows <- do.call(rbind,fits); checks <- list()
    old <- if(retained)read.csv(file.path(here,'results/ordinal-audit-confirm-v1/fits.csv')) else NULL
    for(i in which(rows$returned)) {
      r <- rows[i,]; fit <- readRDS(file.path(out,paste0('fit_',r$fit_id,'.rds')))
      baseline <- rows[rows$case_id==r$case_id & rows$parameterization==r$parameterization & rows$method==r$method & rows$arm=='default',]
      stock <- readRDS(file.path(out,paste0('fit_',baseline$fit_id,'.rds')))
      error <- NA_real_; old_id <- NA_integer_
      if(retained && r$arm!='tight_lbfgs') {
        origin <- old[old$case_id==9 & old$parameterization==r$parameterization & old$method==r$method & old$arm==r$arm,]
        old_id <- origin$fit_id
        before <- readRDS(file.path(here,'results/ordinal-audit-confirm-v1',paste0('fit_',old_id,'.rds')))
        error <- max(abs(before$theta-fit$theta))
      }
      checks[[length(checks)+1L]] <- data.frame(fit_id=r$fit_id,original_fit_id=old_id,
        actual_start_equal=identical(fit$start$theta,stock$start$theta),actual_start_method=fit$start$method,
        replay_theta_max_error=error,backend_objective_consistent=fit$audit$f_consistent,
        point_objective_error=abs(core$frontier_ordinal_newton_audit(fit)$objective-fit$fmin))
    }
    check_rows <- do.call(rbind,checks); write_out(check_rows,'replay_checks')
    # PORT-NLS has no scalar backend recomputation (NaN/false telemetry).
    # Check the actual full-threshold objective explicitly and retain that gap.
    if(any(!check_rows$actual_start_equal)||any(check_rows$point_objective_error>1e-10)||any(check_rows$replay_theta_max_error>1e-10,na.rm=TRUE))stop('replay/start/consistency gate failed; evidence retained')
  }
  write_metadata(file.path(out,'metadata.csv'),values=list(seed_base=seed,reps=reps,families=families,
    package_dll_md5=unname(tools::md5sum(getLoadedDLLs()[['magmaanlab']][['path']])),
    source_md5=paste(tools::md5sum(file.path(here,c('R/ordinal_audit.R','scripts/ordinal_audit_reference.py','scripts/weighted_audit_reference.py',if(sparse_search)'scripts/ordinal_boundary_reference.py'))),collapse=';'),
    elapsed_s=proc.time()[['elapsed']]-t0,starts='unmodified ordinal/mixed default starts',
    sparse_search=sparse_search,retained=retained,
    retained_case_md5=if(retained)unname(tools::md5sum(file.path(here,'results/ordinal-audit-confirm-v1/case_9.rds'))) else '',
    effective_controls='default: NLopt max_eval=1000 ftol_rel=1e-10 xtol_rel=1e-7 tolg=backend default; PORT max_iter=1000 max_eval=10000 rel_f_tol=1e-10 (other PORT controls native defaults); tight: NLopt max_eval=3000 ftol_rel=1e-14 xtol_rel=1e-12 tolg=backend default; ordinal route does not use complete-data coordinate normalization',
    metric='fitting-weight working metric; no sampling/inference claim',scope='ambient original full-threshold objective; no barrier/PSD/sphere certificate',
    construction='unsupported: ordinal moment-map construction bounds not implemented',git_head=system('git rev-parse HEAD',intern=TRUE),
    git_dirty=length(system('git status --porcelain',intern=TRUE))>0),packages='magmaanlab')
  python <- value('--python','python3')
  if(sparse_search) {
    status <- system2(python,c(shQuote(file.path(here,'scripts/ordinal_boundary_reference.py')),'--run-dir',shQuote(out)))
    if(status!=0)stop('ordinal boundary reference failed; evidence retained')
    path_file <- file.path(out,'path_parameters.csv')
    if(file.exists(path_file)) {
      paths <- read.csv(path_file)
      fit_rows <- do.call(rbind,fits)
      for(path in unique(paths$path_id)) {
        p <- paths[paths$path_id==path,]; f <- fit_rows[fit_rows$fit_id==p$fit_id[1],]
        fit <- readRDS(file.path(out,paste0('fit_',f$fit_id,'.rds')))
        capture_audit(fit,p$value[order(p$parameter)],f$fit_id,f$case_id,f$family,f$parameterization,f$method,f$arm,'boundary_path')
      }
      for(name in c('points','artifacts','audits'))write_out(do.call(rbind,get(name)),name)
    }
  }
  status <- system2(python,c(shQuote(file.path(here,'scripts/ordinal_audit_reference.py')),'--run-dir',shQuote(out)))
  if(status!=0)stop('ordinal independent reference failed; evidence retained')
  cat('Wrote ',out,'\n',sep='')
}
