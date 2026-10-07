# Conditional Stage-1/handoff/Stage-2 composition; no inference decision.
run_ml2s_audit <- function(args,here) {
  value <- function(k,default) {
    i <- match(k,args);if(is.na(i))return(default)
    if(i==length(args)||startsWith(args[i+1L],'--'))stop('missing ',k)
    args[i+1L]
  }
  if(any(args %in% c('--help','-h'))) {
    cat('Usage: --ordinary --ml2s-audit --run-id NAME [--smoke] [--reps N] [--seed-base N] [--python PATH]\n',
      'One retained saturated build per draw; actual ML2S NT/ULS/DWLS/ADF/DLS default fits and separate audits.\n',
      'Complete, MCAR, MAR, unequal groups, mixed units and shared loadings; fixed diagonal transformation controls.\n',
      'Early EM stops, raw/repaired information, missing evidence and mismatched handoffs remain visible.\n',
      'Independent 90-digit derivatives/ACOV/weight checks; no propagated construction interval or inference calibration.\n',sep='')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--ml2s-audit','--run-id','--smoke','--reps','--seed-base','--python')
  if(any(startsWith(args,'--') & !args %in% known))stop('unknown ML2S option')
  seed <- as.integer(value('--seed-base','865071201'));reps <- as.integer(value('--reps','1'))
  run <- value('--run-id','ml2s-audit')
  if(is.na(seed)||seed<1||is.na(reps)||reps<1||reps>10||!grepl('^[A-Za-z0-9_-]+$',run))stop('invalid options')
  out <- file.path(here,'results',run);if(dir.exists(out))stop('choose fresh run ID')
  require_pkg('magmaanlab');dir.create(out,recursive=TRUE);core <- magmaanlab::magmaan_core
  write_out <- function(x,name)write_csv(x,file.path(out,paste0(name,'.csv')))
  matrix_rows <- function(x,id,name) {
    if(is.null(x))return(NULL)
    x <- as.matrix(x);if(!length(x))return(NULL)
    cbind(id=id,name=name,expand.grid(row=seq_len(nrow(x)),col=seq_len(ncol(x))),value=sprintf('%.17g',as.vector(x)))
  }
  cases <- fits <- audits <- artifacts <- samples <- points <- controls <- stages <- list()
  retain <- function(fit,cid,fid,family,kind,target,theta=NULL,stage1_point=NULL) {
    a <- core$frontier_ml2s_convergence_audit(fit,theta,stage1_point);id <- length(audits)+1L
    audits[[id]] <<- data.frame(point_id=id,fit_id=fid,case_id=cid,family=family,kind=kind,target=target,
      status=a$status,stage1_status=a$stage1_assessment$status,handoff_status=a$handoff$status,
      solver_status=a$solver_stop$status,stage2_status=a$stage2_assessment$status,
      stage1_objective=sprintf('%.17g',a$stage1$objective),stage2_objective=sprintf('%.17g',a$stage2$objective),
      stage1_distance=sprintf('%.17g',a$stage1$diagnostics$distance),stage2_distance=sprintf('%.17g',a$stage2$diagnostics$distance),
      stage1_raw_passed=a$stage1$diagnostics$passed,stage2_raw_passed=a$stage2$diagnostics$passed,
      stage1_curvature=a$stage1$diagnostics$status,stage2_curvature=a$stage2$diagnostics$status,
      information_repaired=a$source$information_repaired,information_ridge=sprintf('%.17g',a$source$information_ridge),
      value_recorded=a$stage1$value_recorded,dls_a=fit$stage2_dls_a,
      transform_intensity=if(is.null(fit$stage2_input$transformation))0 else fit$stage2_input$transformation$intensity)
    pt <- a$partable[c('lhs','op','rhs','group','free','est')];pt$est <- sprintf('%.17g',pt$est)
    points[[id]] <<- cbind(point_id=id,pt)
    for(stage in c('stage1','stage2'))for(name in c('theta','gradient','hessian','metric_factor','derivative_basis'))
      if(!is.null(a[[stage]][[name]]))artifacts[[length(artifacts)+1L]] <<- matrix_rows(a[[stage]][[name]],id,paste0(stage,'_',name))
    for(stage in c('source','stage2_input')) {
      sm <- a[[stage]]
      for(g in seq_along(sm$cov))for(name in c('cov','mean'))
        artifacts[[length(artifacts)+1L]] <<- matrix_rows(sm[[name]][[g]],id,paste0(stage,'_',name,'_',g))
      for(name in c('n_obs','H','J','acov','raw_H','raw_gradient'))
        artifacts[[length(artifacts)+1L]] <<- matrix_rows(sm[[name]],id,paste0(stage,'_',name))
    }
    record <- fit$stage2_input$moments
    if(!is.null(record)) {
      for(g in seq_along(record$cov))for(name in c('cov','mean'))
        artifacts[[length(artifacts)+1L]] <<- matrix_rows(record[[name]][[g]],id,paste0('recorded_',name,'_',g))
      for(name in c('n_obs','acov'))artifacts[[length(artifacts)+1L]] <<- matrix_rows(record[[name]],id,paste0('recorded_',name))
      artifacts[[length(artifacts)+1L]] <<- matrix_rows(match(fit$stage2_input$stage2_weight,c('nt','uls','dwls','adf','dls')),id,'recorded_kind')
      artifacts[[length(artifacts)+1L]] <<- matrix_rows(fit$stage2_input$dls_a,id,'recorded_dls_a')
    }
    for(g in seq_along(fit$S)) {
      artifacts[[length(artifacts)+1L]] <<- matrix_rows(fit$S[[g]],id,paste0('fit_cov_',g))
      artifacts[[length(artifacts)+1L]] <<- matrix_rows(fit$sample_mean[[g]],id,paste0('fit_mean_',g))
    }
    artifacts[[length(artifacts)+1L]] <<- matrix_rows(fit$nobs,id,'fit_n_obs')
    for(g in seq_along(a$stage2$retained_ls_weights)) {
      w <- a$stage2$retained_ls_weights[[g]]
      for(name in c('diagonal','factor','root'))
        artifacts[[length(artifacts)+1L]] <<- matrix_rows(w[[name]],id,paste0('weight_',g,'_',name))
      artifacts[[length(artifacts)+1L]] <<- matrix_rows(match(w$kind,c('identity','diagonal','dense_factor','normal_theory_root')),id,paste0('weight_',g,'_kind'))
    }
    if(!is.null(fit$stage2_input$weight_blocks))for(g in seq_along(fit$stage2_input$weight_blocks))
      artifacts[[length(artifacts)+1L]] <<- matrix_rows(fit$stage2_input$weight_blocks[[g]],id,paste0('supplied_weight_',g))
    saveRDS(a,file.path(out,paste0('audit_',id,'.rds')));a
  }
  all_families <- c('complete','mcar','mar','groups','units','shared')
  families <- if('--smoke' %in% args)c('complete','mcar') else all_families
  begin <- proc.time()[['elapsed']]
  for(family in families)for(rep in seq_len(reps)) {
    cid <- length(cases)+1L;draw_seed <- seed+match(family,all_families)*10000L+rep;set.seed(draw_seed)
    raw <- list(X=list(),mask=list());ng <- if(family=='groups')2L else 1L
    for(g in seq_len(ng)) {
      n <- if(g==2L)210L else 130L
      loading <- if(family=='shared')c(.85,.8,.8,.9) else c(.85,.8,.7,.9)
      x <- outer(rnorm(n),loading)+matrix(rnorm(n*4),n,4)
      x <- sweep(x,2,c(2,-1,.5,4)+.3*(g-1),'+')
      if(family=='units')x <- sweep(x,2,c(.01,100,.2,3),'*')
      colnames(x) <- paste0('x',1:4);mask <- matrix(TRUE,n,4)
      if(family!='complete') {
        prob <- if(family=='mar')plogis(-1+.5*(x[,1]-mean(x[,1]))/sd(x[,1])) else rep(.25,n)
        mask <- matrix(runif(n*4),n,4)>prob
        if(family=='mar')mask[,1] <- TRUE
        mask[rowSums(mask)==0,1] <- TRUE
      }
      raw$X[[g]] <- x;raw$mask[[g]] <- mask
      for(name in c('X','mask'))samples[[length(samples)+1L]] <- matrix_rows(raw[[name]][[g]],cid,paste0(name,'_',g))
    }
    cases[[cid]] <- data.frame(case_id=cid,family=family,rep=rep,seed=draw_seed,groups=ng,rows=sum(vapply(raw$X,nrow,integer(1))))
    spec <- magmaanlab::model_spec(if(family=='shared')'f =~ x1 + a*x2 + a*x3 + x4' else 'f =~ x1 + x2 + x3 + x4',
      fixed_x=FALSE,meanstructure=TRUE,group_labels=if(ng==2L)c('g1','g2') else NULL)
    t0 <- proc.time()[['elapsed']]
    sm <- tryCatch(suppressWarnings(core$estimate_saturated_em_moments(raw)),error=identity)
    stages[[cid]] <- data.frame(case_id=cid,family=family,returned=!inherits(sm,'error'),seconds=proc.time()[['elapsed']]-t0,
      message=if(inherits(sm,'error'))conditionMessage(sm) else '')
    saveRDS(list(raw=raw,spec=spec,stage1=sm),file.path(out,paste0('case_',cid,'.rds')))
    if(inherits(sm,'error'))next
    baseline <- NULL
    arms <- c('nt','uls','dwls','adf','dls',if(rep==1L && family %in% c('complete','mcar'))'transformed_nt')
    for(arm in arms) {
      kind <- if(arm=='transformed_nt')'nt' else arm
      t0 <- proc.time()[['elapsed']]
      fit <- tryCatch(suppressWarnings(core$fit_ml2s(spec,raw,stage1=sm,stage2_weight=kind,
        stage1_regularization=if(arm=='transformed_nt')list(target='diagonal',intensity=.25) else NULL)),error=identity)
      fid <- length(fits)+1L;returned <- !inherits(fit,'error')
      fits[[fid]] <- data.frame(fit_id=fid,case_id=cid,family=family,arm=arm,kind=kind,returned=returned,
        fit_passed=if(returned)fit$converged else NA,fmin=if(returned)fit$fmin else NA,
        seconds=proc.time()[['elapsed']]-t0,start_method=if(returned)fit$start$method else '',
        optimizer_status=if(returned)fit$optimizer_status else '',f_evals=if(returned)fit$f_evals else NA,
        message=if(returned)'' else conditionMessage(fit))
      saveRDS(fit,file.path(out,paste0('fit_',fid,'.rds')));write_out(do.call(rbind,fits),'fits')
      if(!returned)next
      a <- retain(fit,cid,fid,family,kind,'terminal')
      if(arm=='nt')baseline <- fit
      if(arm=='nt') {
        theta <- fit$theta;row <- which(fit$partable$lhs=='x2' & fit$partable$op=='~~' & fit$partable$rhs=='x2' & fit$partable$group==1)[1]
        theta[fit$partable$free[row]] <- theta[fit$partable$free[row]]+.05*fit$S[[1]][2,2]
        retain(fit,cid,fid,family,kind,'stage2_displaced',theta)
      }
      if(cid==1L && arm=='nt') {
        bad <- fit;bad$stage2_input <- NULL;retain(bad,cid,fid,family,kind,'missing_input')
        bad <- fit;bad$stage1$solver_recorded <- FALSE;retain(bad,cid,fid,family,kind,'missing_stop')
        bad <- fit;bad$stage1 <- bad$stage1[c('mean','cov','n_obs','H','J','acov','warnings')]
        retain(bad,cid,fid,family,kind,'legacy_stage1')
        bad <- fit;bad$stage2_input$moments$cov[[1]][1,1] <- bad$stage2_input$moments$cov[[1]][1,1]+.1
        retain(bad,cid,fid,family,kind,'mismatched_moments')
        point <- sm;point$cov[[1]] <- 10*point$cov[[1]]
        retain(fit,cid,fid,family,kind,'repaired_raw_curvature',stage1_point=point)
      }
      if(cid==1L && arm=='dls') {
        bad <- fit;bad$stage2_input$dls_a <- .25;retain(bad,cid,fid,family,kind,'mismatched_mix')
        bad <- fit;bad$stage2_input$weight_blocks[[1]][1,1] <- 2*bad$stage2_input$weight_blocks[[1]][1,1]
        retain(bad,cid,fid,family,kind,'mismatched_weight')
        bad <- fit;bad$stage2_input$moments$acov[1,1] <- 2*bad$stage2_input$moments$acov[1,1]
        retain(bad,cid,fid,family,kind,'mismatched_acov')
        bad <- fit;bad$stage2_input$moments$acov <- NULL
        retain(bad,cid,fid,family,kind,'missing_required_acov')
      }
    }
    if(family=='mcar' && rep==1L) {
      ctl <- list(h1_em_max_iter=1L,h1_em_param_tol=1e-14,h1_em_cov_floor=2)
      e <- tryCatch(core$estimate_saturated_em_moments(raw,control=ctl),error=identity)
      controls[[length(controls)+1L]] <- data.frame(control='strict_early_EM',case_id=cid,rejected=inherits(e,'error'),message=if(inherits(e,'error'))conditionMessage(e) else '')
      ctl$h1_em_error_on_nonconvergence <- FALSE
      early <- suppressWarnings(core$estimate_saturated_em_moments(raw,control=ctl));saveRDS(early,file.path(out,'early_stage1.rds'))
      bad <- baseline;bad$stage1 <- early
      retain(bad,cid,0L,family,'nt','early_source_with_original_fit')
    }
    cat(sprintf('ML2S case %d: %s; %d fits/%d audits; %.1fs\n',cid,family,length(fits),length(audits),proc.time()[['elapsed']]-begin))
  }
  for(name in c('cases','fits','audits','artifacts','samples','points','controls','stages'))write_out(do.call(rbind,get(name)),name)
  write_metadata(file.path(out,'metadata.csv'),values=list(seed_base=seed,reps=reps,families=families,
    source_md5=paste(tools::md5sum(file.path(here,c('R/ml2s_audit.R','scripts/ml2s_audit_reference.py','scripts/fiml_audit_reference.py','scripts/weighted_audit_reference.py'))),collapse=';'),
    package_dll_md5=unname(tools::md5sum(getLoadedDLLs()[['magmaanlab']][['path']])),elapsed_s=proc.time()[['elapsed']]-begin,
    controls='actual fit_ml2s default L-BFGS/starts/stopping; one retained Stage-1 per draw; DLS a=0.5; transformed controls diagonal intensity=0.25',
    convention='Stage-1 raw likelihood; Nt Stage-2 ML, other weights original quadratic; separate local distances, no propagated error or inference calibration',
    git_head=system('git rev-parse HEAD',intern=TRUE),git_dirty=length(system('git status --porcelain',intern=TRUE))>0),packages='magmaanlab')
  status <- system2(value('--python','python3'),c(shQuote(file.path(here,'scripts/ml2s_audit_reference.py')),'--run-dir',shQuote(out)))
  if(status!=0)stop('independent ML2S check failed; evidence retained')
  cat('Wrote ',out,'\n',sep='')
}
