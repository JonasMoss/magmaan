# Numerical integration study; no sampling-law or default-adoption decision.
run_ordinal_audit <- function(args, here) {
  value <- function(k, default) {
    i <- match(k,args); if(is.na(i)) return(default)
    if(i==length(args)) stop('missing ',k)
    args[i+1L]
  }
  if(any(args %in% c('--help','-h'))) {
    cat('Usage: --ordinary --ordinal-audit --run-id NAME [--smoke] [--reps N] [--seed-base N] [--python PATH]\n',
        'All-ordinal ULS/DWLS/WLS and mixed DWLS/WLS; delta/theta, actual default starts, default and PORT-NLS controls.\n',
        'Retains original prepared maps and actual weights; independent 90-digit point checks. Construction bounds remain unsupported.\n',sep='')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--ordinal-audit','--run-id','--smoke','--reps','--seed-base','--python')
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
  families <- if('--smoke' %in% args)c('regular','mixed') else c('regular','shared','groups','mixed','sparse','extreme','saddle')
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
    producer <- tryCatch(if(mixed)core$data_mixed_ordinal_stats_from_raw(X,c(TRUE,TRUE,FALSE,FALSE)) else core$data_ordinal_stats_from_raw(X),error=identity)
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
      if(family=='regular' && rep==1L && param=='delta') {
        bad <- producer; bad$W_wls <- lapply(bad$W_wls,function(x)matrix(numeric(),0,0))
        check <- tryCatch(core$fit_wls_ordinal(spec,bad),error=identity)
        controls[[length(controls)+1L]] <- data.frame(control='unavailable_wls_weight',
          rejected=inherits(check,'error'),message=if(inherits(check,'error'))conditionMessage(check) else '')
        constant <- X; constant[[1]][,1] <- 1
        check <- tryCatch(core$data_ordinal_stats_from_raw(constant),error=identity)
        controls[[length(controls)+1L]] <- data.frame(control='constant_category',
          rejected=inherits(check,'error'),message=if(inherits(check,'error'))conditionMessage(check) else '')
      }
      for(method in if(mixed)c('DWLS','WLS') else c('ULS','DWLS','WLS')) for(arm in c('default','port_nls')) {
        fit_fun <- if(mixed)core[[paste0('fit_',tolower(method),'_mixed_ordinal')]] else core[[paste0('fit_',tolower(method),'_ordinal')]]
        begin <- proc.time()[['elapsed']]
        fit <- tryCatch(suppressWarnings(if(arm=='default')fit_fun(spec,producer) else
          fit_fun(spec,producer,optimizer='port-nls',control=list(max_iter=1000))),error=identity)
        fid <- length(fits)+1L
        saveRDS(fit,file.path(out,paste0('fit_',fid,'.rds')))
        fits[[fid]] <- data.frame(fit_id=fid,case_id=cid,family=family,parameterization=param,method=method,arm=arm,
          optimizer=if(arm=='default')'nlopt-lbfgs' else 'port-nls',max_iter=if(arm=='default')NA_integer_ else 1000L,
          returned=!inherits(fit,'error'),fit_passed=if(inherits(fit,'error'))NA else fit$converged,
          fmin=if(inherits(fit,'error'))NA else fit$fmin,seconds=proc.time()[['elapsed']]-begin,
          message=if(inherits(fit,'error'))conditionMessage(fit) else '')
        write_out(do.call(rbind,fits),'fits')
        if(inherits(fit,'error'))next
        for(target in if(family=='saddle' && arm=='default')c('terminal','displaced','saddle') else if(arm=='default')c('terminal','displaced') else 'terminal') {
          theta <- fit$theta
          if(target=='displaced')theta <- theta+.025*sin(seq_along(theta))
          if(target=='saddle')theta[fit$partable$free[fit$partable$op=='=~']] <- 0
          begin <- proc.time()[['elapsed']]
          a <- core$frontier_ordinal_newton_audit(fit,theta)
          id <- length(audits)+1L
          audits[[id]] <- data.frame(point_id=id,fit_id=fid,case_id=cid,family=family,parameterization=param,method=method,arm=arm,target=target,
            objective=sprintf('%.17g',a$objective),distance=sprintf('%.17g',a$diagnostics$distance),status=a$diagnostics$status,passed=a$diagnostics$passed,
            construction_status=a$construction_status,audit_seconds=proc.time()[['elapsed']]-begin)
          pt <- a$partable[c('lhs','op','rhs','group','free','est')]; pt$est <- sprintf('%.17g',pt$est)
          points[[id]] <- cbind(point_id=id,pt)
          for(name in c('theta','gradient','hessian','metric','metric_factor','metric_score_residual','curvature_correction','derivative_basis','whitened_jacobian','whitened_residual'))
            artifacts[[length(artifacts)+1L]] <- matrix_rows(a[[name]],id,name)
          for(g in seq_along(a$weight_factors))artifacts[[length(artifacts)+1L]] <- matrix_rows(a$weight_factors[[g]],id,paste0('F_',g))
        }
      }
    }
    elapsed <- proc.time()[['elapsed']]-t0
    cat(sprintf('Ordinal case %d: %s; %d fits, %d points; %.1fs\n',cid,family,length(fits),length(audits),elapsed))
  }
  for(name in c('cases','fits','points','artifacts','samples','audits','controls'))write_out(do.call(rbind,get(name)),name)
  write_metadata(file.path(out,'metadata.csv'),values=list(seed_base=seed,reps=reps,families=families,
    package_dll_md5=unname(tools::md5sum(getLoadedDLLs()[['magmaanlab']][['path']])),
    source_md5=paste(tools::md5sum(file.path(here,c('R/ordinal_audit.R','scripts/ordinal_audit_reference.py','scripts/weighted_audit_reference.py'))),collapse=';'),
    elapsed_s=proc.time()[['elapsed']]-t0,starts='unmodified ordinal/mixed default starts',
    metric='fitting-weight working metric; no sampling/inference claim',scope='ambient original full-threshold objective; no barrier/PSD/sphere certificate',
    construction='unsupported: ordinal moment-map construction bounds not implemented',git_head=system('git rev-parse HEAD',intern=TRUE),
    git_dirty=length(system('git status --porcelain',intern=TRUE))>0),packages='magmaanlab')
  python <- value('--python','python3')
  status <- system2(python,c(shQuote(file.path(here,'scripts/ordinal_audit_reference.py')),'--run-dir',shQuote(out)))
  if(status!=0)stop('ordinal independent reference failed; evidence retained')
  cat('Wrote ',out,'\n',sep='')
}
