# Direct observed-pattern numerical integration; no inference/default decision.
run_fiml_audit <- function(args,here) {
  value <- function(k,default) {
    i <- match(k,args); if(is.na(i))return(default)
    if(i==length(args))stop('missing ',k)
    args[i+1L]
  }
  if(any(args %in% c('--help','-h'))) {
    cat('Usage: --ordinary --fiml-audit --run-id NAME [--smoke] [--reps N] [--seed-base N] [--python PATH]\n',
      'Actual default FABIN3/fallback and SLSQP fits; early-stop, displaced, sparse-information and invalid-input controls. PORT is explicitly unsupported.\n',
      'Complete, MCAR, MAR, heterogeneous units, unequal groups, shared loadings, weak marker and saddle families.\n',
      'Independent 90-digit observed-pattern objective, gradient, observed Hessian and conditional Newton audit. Construction bounds unsupported.\n',sep='')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--fiml-audit','--run-id','--smoke','--reps','--seed-base','--python')
  if(any(startsWith(args,'--') & !args %in% known))stop('unknown FIML option')
  seed <- as.integer(value('--seed-base','865061201')); reps <- as.integer(value('--reps','1'))
  run <- value('--run-id','fiml-audit')
  if(is.na(seed)||seed<1||is.na(reps)||reps<1||reps>10||!grepl('^[A-Za-z0-9_-]+$',run))stop('invalid options')
  out <- file.path(here,'results',run); if(dir.exists(out))stop('choose fresh run ID')
  require_pkg('magmaanlab');dir.create(out,recursive=TRUE);core <- magmaanlab::magmaan_core
  write_out <- function(x,name)write_csv(x,file.path(out,paste0(name,'.csv')))
  matrix_rows <- function(x,id,name) {
    x <- as.matrix(x); if(!length(x))return(NULL)
    cbind(id=id,name=name,expand.grid(row=seq_len(nrow(x)),col=seq_len(ncol(x))),value=sprintf('%.17g',as.vector(x)))
  }
  cases <- fits <- audits <- points <- artifacts <- samples <- pattern_rows <- controls <- reductions <- starts <- list()
  retain_sample <- function(raw,cid) {
    for(g in seq_along(raw$X))for(name in c('X','mask'))
      samples[[length(samples)+1L]] <<- matrix_rows(raw[[name]][[g]],cid,paste0(name,'_',g))
  }
  retain <- function(fit,theta,cid,fid,family,arm,target) {
    a <- core$frontier_fiml_newton_audit(fit,theta);id <- length(audits)+1L
    audits[[id]] <<- data.frame(point_id=id,fit_id=fid,case_id=cid,family=family,arm=arm,target=target,
      objective=sprintf('%.17g',a$objective),n_obs=a$n_obs,distance=sprintf('%.17g',a$diagnostics$distance),
      status=a$diagnostics$status,passed=a$diagnostics$passed,construction_status=a$construction_status)
    pt <- a$partable[c('lhs','op','rhs','group','free','est')];pt$est <- sprintf('%.17g',pt$est)
    points[[id]] <<- cbind(point_id=id,pt)
    for(name in c('theta','gradient','hessian','derivative_basis'))
      artifacts[[length(artifacts)+1L]] <<- matrix_rows(a[[name]],id,name)
    for(k in seq_along(a$patterns)) {
      p <- a$patterns[[k]]
      pattern_rows[[length(pattern_rows)+1L]] <<- data.frame(point_id=id,pattern_id=k,block=p$block,n_obs=p$n_obs,
        observed=paste(p$observed,collapse=':'))
      for(name in c('mean','cov'))artifacts[[length(artifacts)+1L]] <<- matrix_rows(p[[name]],id,paste0('pattern_',k,'_',name))
    }
    a
  }
  families_all <- c('complete','mcar','mar','units','groups','shared','weak','saddle')
  families <- if('--smoke' %in% args)c('complete','mcar') else families_all
  t0 <- proc.time()[['elapsed']]
  for(family in families)for(rep in seq_len(reps)) {
    cid <- length(cases)+1L; ng <- if(family=='groups')2L else 1L
    draw_seed <- seed+match(family,families_all)*10000L+rep;set.seed(draw_seed)
    raw <- list(X=list(),mask=list())
    for(g in seq_len(ng)) {
      n <- if(g==2L)210L else 130L
      loading <- if(family=='weak')c(.05,.8,.7,.9) else if(family=='shared')c(.85,.8,.8,.9) else c(.85,.8,.7,.9)
      x <- outer(rnorm(n),loading)+matrix(rnorm(n*4),n,4)
      x <- sweep(x,2,c(2,-1,.5,4)+.3*(g-1),'+')
      units <- if(family=='units')c(.01,100,.2,3) else rep(1,4)
      x <- sweep(x,2,units,'*');colnames(x) <- paste0('x',1:4)
      mask <- matrix(TRUE,n,4)
      if(family!='complete') {
        prob <- if(family=='mar')plogis(-1+.5*(x[,1]-mean(x[,1]))/sd(x[,1])) else rep(.25,n)
        mask <- matrix(runif(n*4),n,4)>prob
        if(family=='mar')mask[,1] <- TRUE
        mask[rowSums(mask)==0,1] <- TRUE
        # MCAR-only singleton observations exercise covariance-zero patterns.
        if(family!='mar')for(j in 1:4){mask[j,] <- FALSE;mask[j,j] <- TRUE}
      }
      raw$X[[g]] <- x;raw$mask[[g]] <- mask
    }
    cases[[cid]] <- data.frame(case_id=cid,family=family,rep=rep,seed=draw_seed,groups=ng,
      rows=sum(vapply(raw$X,nrow,integer(1))),scope='single-level joint observed-pattern normal likelihood')
    retain_sample(raw,cid);saveRDS(raw,file.path(out,paste0('case_',cid,'.rds')))
    syntax <- if(family=='shared')'f =~ x1 + a*x2 + a*x3 + x4' else 'f =~ x1 + x2 + x3 + x4'
    spec <- magmaanlab::model_spec(syntax,fixed_x=FALSE,meanstructure=TRUE,std_lv=family=='saddle',
      group_labels=if(ng==2)c('g1','g2') else NULL)
    default_start <- NULL
    for(arm in c('default','slsqp',if(rep==1L)'early')) {
      begin <- proc.time()[['elapsed']]
      fit <- tryCatch(suppressWarnings(switch(arm,
        default=core$fit_fiml(spec,raw),slsqp=core$fit_fiml(spec,raw,optimizer='nlopt-slsqp'),
        early=core$fit_fiml(spec,raw,control=list(max_iter=1)))),error=identity)
      fid <- length(fits)+1L;returned <- !inherits(fit,'error')
      fits[[fid]] <- data.frame(fit_id=fid,case_id=cid,family=family,arm=arm,returned=returned,
        fit_passed=if(returned)fit$converged else NA,fmin=if(returned)fit$fmin else NA,
        seconds=proc.time()[['elapsed']]-begin,start_method=if(returned)fit$start$method else '',
        coordinate_scaling=if(returned)fit$coordinate_scaling else '',
        optimizer_status=if(returned)fit$optimizer_status else '',
        f_evals=if(returned)fit$f_evals else NA,iterations=if(returned)fit$iterations else NA,
        message=if(returned)'' else conditionMessage(fit))
      saveRDS(fit,file.path(out,paste0('fit_',fid,'.rds')))
      write_out(do.call(rbind,fits),'fits');if(!returned)next
      if(arm=='default')default_start <- fit$start$theta
      starts[[length(starts)+1L]] <- data.frame(fit_id=fid,case_id=cid,arm=arm,
        same_actual_start=if(is.null(default_start))NA else identical(default_start,fit$start$theta),method=fit$start$method)
      targets <- if(arm=='default')c('terminal','displaced','start',if(family=='saddle')'saddle') else 'terminal'
      for(target in targets) {
        theta <- fit$theta
        if(target=='start')theta <- fit$start$theta
        if(target=='displaced') {
          row <- which(fit$partable$lhs=='x2' & fit$partable$op=='~~' & fit$partable$rhs=='x2' & fit$partable$group==1)[1]
          theta[fit$partable$free[row]] <- theta[fit$partable$free[row]]+.05*fit$S[[1]][2,2]
          row <- which(fit$partable$lhs=='x1' & fit$partable$op=='~1' & fit$partable$group==1)[1]
          theta[fit$partable$free[row]] <- theta[fit$partable$free[row]]+.1*sqrt(fit$S[[1]][1,1])
        }
        if(target=='saddle')theta[fit$partable$free[fit$partable$op=='=~']] <- 0
        a <- retain(fit,theta,cid,fid,family,arm,target)
        if(family=='complete') {
          ml <- core$estimate_evaluate_at(spec$partable,list(S=fit$S,nobs=fit$nobs,mean=fit$sample_mean),theta,estimator='ML',
            bounds=list(lower=rep(-Inf,length(theta)),upper=rep(Inf,length(theta))),
            audit_options=list(retain_newton_artifacts=TRUE))
          reductions[[length(reductions)+1L]] <- data.frame(point_id=length(audits),
            gradient_error=max(abs(crossprod(a$derivative_basis,a$gradient)-ml$newton_audit$gradient)),
            hessian_error=max(abs(crossprod(a$derivative_basis,a$hessian%*%a$derivative_basis)-ml$newton_audit$hessian)),ml_objective=ml$fmin)
        }
      }
      if(cid==1L && arm=='default') {
        result <- tryCatch(core$fit_fiml(spec,raw,optimizer='port'),error=identity)
        controls[[length(controls)+1L]] <- data.frame(control='unsupported_port_backend',rejected=inherits(result,'error'),message=if(inherits(result,'error'))conditionMessage(result) else '')
        bad <- fit;bad$raw_data$mask[[1]][1,] <- FALSE
        result <- tryCatch(core$frontier_fiml_newton_audit(bad),error=identity)
        controls[[length(controls)+1L]] <- data.frame(control='empty_observed_row',rejected=inherits(result,'error'),message=if(inherits(result,'error'))conditionMessage(result) else '')
        bad <- fit;bad$raw_data$X[[1]][1,1] <- NA_real_;bad$raw_data$mask[[1]][1,1] <- TRUE
        result <- tryCatch(core$frontier_fiml_newton_audit(bad),error=identity)
        controls[[length(controls)+1L]] <- data.frame(control='nonfinite_observed_value',rejected=inherits(result,'error'),message=if(inherits(result,'error'))conditionMessage(result) else '')
        shuffled <- fit;shuffled$raw_data$X[[1]] <- shuffled$raw_data$X[[1]][,4:1]
        shuffled$raw_data$mask[[1]] <- shuffled$raw_data$mask[[1]][,4:1]
        a <- retain(shuffled,fit$theta,cid,fid,family,arm,'permuted_columns')
        theta <- fit$theta;theta[fit$partable$free[fit$partable$op=='~~' & fit$partable$free>0]] <- -100
        retain(fit,theta,cid,fid,family,arm,'invalid_covariance')
        sparse <- fit;sparse$raw_data$mask[[1]][,] <- FALSE
        for(i in seq_len(nrow(sparse$raw_data$mask[[1]])))sparse$raw_data$mask[[1]][i,1+(i-1)%%4] <- TRUE
        result <- tryCatch(core$frontier_fiml_newton_audit(sparse),error=identity)
        controls[[length(controls)+1L]] <- data.frame(control='no_joint_observations',rejected=inherits(result,'error'),message=if(inherits(result,'error'))conditionMessage(result) else '')
        # The current pack requires each pair to have a joint observation.
        # One row per pair retains that contract while leaving very weak
        # off-diagonal pattern information (and mostly singleton rows).
        pairs <- combn(4,2)
        for(i in seq_len(ncol(pairs))){sparse$raw_data$mask[[1]][i,] <- FALSE;sparse$raw_data$mask[[1]][i,pairs[,i]] <- TRUE}
        sid <- length(cases)+1L
        cases[[sid]] <- data.frame(case_id=sid,family='sparse_pattern_information',rep=0L,seed=draw_seed,groups=1L,
          rows=nrow(sparse$raw_data$X[[1]]),scope='point-only sparse-information control')
        retain_sample(sparse$raw_data,sid)
        retain(sparse,fit$theta,sid,fid,'sparse_pattern_information',arm,'weak_information')
        ridge <- fit;ridge$partable$free[1] <- max(ridge$partable$free)+1L
        ridge$theta <- c(fit$theta,1);ridge$npar <- length(ridge$theta)
        retain(ridge,ridge$theta,cid,fid,family,arm,'rank_control')
      }
    }
    cat(sprintf('FIML case %d: %s; %d fits/%d points; %.1fs\n',cid,family,length(fits),length(audits),proc.time()[['elapsed']]-t0))
  }
  for(name in c('cases','fits','audits','points','artifacts','samples','pattern_rows','controls','reductions','starts'))write_out(do.call(rbind,get(name)),name)
  if(any(!do.call(rbind,starts)$same_actual_start,na.rm=TRUE))stop('actual-start equality failed; evidence retained')
  write_metadata(file.path(out,'metadata.csv'),values=list(seed_base=seed,reps=reps,families=families,
    source_md5=paste(tools::md5sum(file.path(here,c('R/fiml_audit.R','scripts/fiml_audit_reference.py','scripts/weighted_audit_reference.py'))),collapse=';'),
    package_dll_md5=unname(tools::md5sum(getLoadedDLLs()[['magmaanlab']][['path']])),elapsed_s=proc.time()[['elapsed']]-t0,
    starts='actual public FIML FABIN3 default; same starts for SLSQP; no start tuning',
    controls='default fallback and SLSQP: max_eval=1000 ftol_rel=1e-10 xtol_rel=1e-7; early max_iter=1 per fallback stage; PORT unsupported; no sample-normalization extension',
    metric='observed total negative-log-likelihood Hessian',construction='unsupported: no propagated pattern likelihood construction bound',
    identification='bank declares identified model scope; known scale ridge remains unchecked, raw local accuracy retained; TASK-33.3 owns identification guard',
    git_head=system('git rev-parse HEAD',intern=TRUE),git_dirty=length(system('git status --porcelain',intern=TRUE))>0),packages='magmaanlab')
  status <- system2(value('--python','python3'),c(shQuote(file.path(here,'scripts/fiml_audit_reference.py')),'--run-dir',shQuote(out)))
  if(status!=0)stop('independent FIML reference failed; evidence retained')
  cat('Wrote ',out,'\n',sep='')
}
