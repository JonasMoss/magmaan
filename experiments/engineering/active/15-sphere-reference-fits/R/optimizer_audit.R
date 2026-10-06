# Shared start/adapter checks. Independent references assess the same objective.
run_optimizer_audit <- function(args, here) {
  value <- function(k, default) {
    i <- match(k,args); if(is.na(i))return(default)
    if(i==length(args))stop('missing ',k)
    args[i+1L]
  }
  if(any(args %in% c('--help','-h'))) {
    cat('Usage: --ordinary --optimizer-audit --run-id NAME [--smoke] [--reps N] [--seed-base N]\n',
      '[--textbook-root PATH] [--retained-run ml2s-audit-confirm-v1]\n',
      'Pinned equality starts against explicitly fixed models and lavaan; fresh rescaled draws.\n',
      'ML/LS/FIML/ordinal early stops and retained mixed-unit ML2S losses; no recovery/default proposal.\n',sep='');return(invisible(NULL))
  }
  known <- c('--ordinary','--optimizer-audit','--run-id','--smoke','--reps','--seed-base','--textbook-root','--retained-run')
  if(any(startsWith(args,'--') & !args %in% known))stop('unknown optimizer option')
  run <- value('--run-id','optimizer-audit');seed <- as.integer(value('--seed-base','865081201'))
  reps <- as.integer(value('--reps','1'))
  if(is.na(seed)||seed<1||is.na(reps)||reps<1||reps>10||!grepl('^[A-Za-z0-9_-]+$',run))stop('invalid options')
  out <- file.path(here,'results',run);if(dir.exists(out))stop('choose fresh run ID')
  require_pkg('magmaanlab');require_pkg('lavaan');core <- magmaanlab::magmaan_core
  dir.create(out,recursive=TRUE);rows <- controls <- references <- oracle_rows <- list()
  write_out <- function(x,name)write_csv(x,file.path(out,paste0(name,'.csv')))
  begin <- proc.time()[['elapsed']]
  capture <- function(fun,family,arm,method,expected='observe',reference=NA_real_) {
    t <- proc.time()[['elapsed']];fit <- tryCatch(suppressWarnings(fun()),error=identity)
    id <- length(rows)+1L;ok <- !inherits(fit,'error');a <- if(ok)fit$audit else list()
    ctl <- a$nlopt_controls
    distance <- if(ok)fit$diagnostics$newton_accuracy$distance else NA_real_
    if(is.null(distance))distance <- NA_real_
    row <- data.frame(fit_id=id,family=family,arm=arm,method=method,expected=expected,returned=ok,
      fit_passed=if(ok)fit$converged else NA,optimizer_status=if(ok)fit$optimizer_status else '',
      raw_status=if(is.null(a$raw_backend_status))NA_integer_ else a$raw_backend_status,
      fmin=if(ok)fit$fmin else NA_real_,reference=reference,
      objective_gap=if(ok && is.finite(reference))abs(fit$fmin-reference) else NA_real_,
      distance=distance,f_evals=if(ok)fit$f_evals else NA_integer_,
      g_evals=if(ok)fit$g_evals else NA_integer_,
      stationary=if(ok)a$stationary else NA,f_consistent=if(ok)a$f_consistent else NA,
      max_eval=if(is.null(ctl$max_eval))NA_integer_ else ctl$max_eval,
      ftol_rel=if(is.null(ctl$ftol_rel))NA_real_ else ctl$ftol_rel,
      xtol_rel=if(is.null(ctl$xtol_rel))NA_real_ else ctl$xtol_rel,
      ftol_abs=if(is.null(ctl$ftol_abs))NA_real_ else ctl$ftol_abs,
      xtol_abs=if(is.null(ctl$xtol_abs))NA_real_ else ctl$xtol_abs,
      tolg=if(is.null(ctl$tolg))NA_real_ else ctl$tolg,
      vector_storage=if(is.null(ctl$vector_storage))NA_integer_ else ctl$vector_storage,
      constraint_tol=if(is.null(ctl$constraint_tol))NA_real_ else ctl$constraint_tol,
      start_method=if(ok)fit$start$method else '',seconds=proc.time()[['elapsed']]-t,
      message=if(ok)'' else conditionMessage(fit))
    rows[[id]] <<- row
    write_out(do.call(rbind,rows),'fits');print(row[c('family','arm','method','returned','fit_passed','objective_gap','raw_status')])
    saveRDS(fit,file.path(out,paste0('fit_',id,'.rds')))
    if(ok && method %in% c('ML','ULS') && is.finite(reference)) {
      moments <- core$model_implied(fit)
      references[[length(references)+1L]] <<- data.frame(fit_id=id,
        objective_recomputed=if(method=='ML') {
          S <- fit$S[[1]];Sigma <- moments$sigma[[1]];p <- nrow(S)
          mu <- moments$mu[[1]];delta <- fit$sample_mean[[1]]-mu
          .5*(as.numeric(determinant(Sigma,logarithm=TRUE)$modulus)-as.numeric(determinant(S,logarithm=TRUE)$modulus)+
            sum(diag(solve(Sigma,S)))+(if(length(delta))drop(crossprod(delta,solve(Sigma,delta))) else 0)-p)
        } else NA_real_)
    }
    fit
  }
  ml_objective <- function(x,Sigma,mu) {
    S <- stats::cov(x)*(nrow(x)-1)/nrow(x);delta <- colMeans(x)-mu;p <- ncol(x)
    .5*(as.numeric(determinant(Sigma,logarithm=TRUE)$modulus)-
      as.numeric(determinant(S,logarithm=TRUE)$modulus)+sum(diag(solve(Sigma,S)))+
      drop(crossprod(delta,solve(Sigma,delta)))-p)
  }
  little_reference <- function(x,s,tag,scale=1) {
    groups <- lapply(s[grepl('=~',s,fixed=TRUE)],function(z)regmatches(z,gregexpr('ly_[0-9]+_[0-9]+',z))[[1]])
    eq <- unlist(lapply(groups,function(z)paste(z[1],'==',z[-1])))
    models <- list(equal=paste(c(s,eq),collapse='\n'),
      fixed=paste(gsub('(NA\\*)?ly_[0-9]+_[0-9]+\\*','1*',s[!grepl('^ly_.*==',s)]),collapse='\n'))
    raw <- suppressWarnings(lavaan::sem(models$fixed,data=x,meanstructure=TRUE,fixed.x=FALSE))
    # Uniform units preserve fixed unit loadings and zero-RHS mean constraints.
    # Solve the oracle in controlled units, then transport its moments back to
    # check the caller's original likelihood. Keep raw-unit stops as evidence.
    l <- if(scale==1)raw else suppressWarnings(lavaan::sem(models$fixed,
      data=x/scale,meanstructure=TRUE,fixed.x=FALSE))
    ref <- unname(lavaan::fitMeasures(l,'fmin'))
    if(!isTRUE(lavaan::lavInspect(l,'converged')))stop('independent fixed reference failed')
    implied <- lavaan::lavInspect(l,'implied')
    oracle_rows[[length(oracle_rows)+1L]] <<- data.frame(tag=tag,scale=scale,
      raw_converged=lavaan::lavInspect(raw,'converged'),raw_fmin=unname(lavaan::fitMeasures(raw,'fmin')),
      reference_converged=lavaan::lavInspect(l,'converged'),reference_fmin=ref,
      transported_objective=ml_objective(x,implied$cov*scale^2,implied$mean*scale),
      raw_iterations=lavaan::lavInspect(raw,'optim')$iterations,
      reference_iterations=lavaan::lavInspect(l,'optim')$iterations)
    write_out(do.call(rbind,oracle_rows),'oracle_controls')
    saveRDS(raw,file.path(out,paste0('lavaan_raw_',tag,'.rds')))
    saveRDS(l,file.path(out,paste0('lavaan_',tag,'.rds')))
    for(m in names(models))for(b in c('nlopt-lbfgs','port')) {
      spec <- magmaanlab::model_spec(models[[m]],meanstructure=TRUE,fixed_x=FALSE)
      capture(function()magmaanlab::fit_model(spec,x,optimizer=b),tag,paste(m,b,sep='_'),'ML','pass',ref)
    }
  }
  if(!'--smoke' %in% args) {
    root <- value('--textbook-root',file.path(repo_root(),'external/textbook-corpus'))
    case <- file.path(root,'cases/little_2013/little_2013_ch10_tab3_correlation')
    if(!file.exists(file.path(case,'data/raw.csv')))stop('provide --textbook-root with the optional Little corpus')
    x <- read.csv(file.path(case,'data/raw.csv'),check.names=FALSE);s <- readLines(file.path(case,'model.lav'))
    little_reference(x,s,'little_table_10_3')
    write_out(data.frame(file=c('model.lav','data/raw.csv'),md5=unname(tools::md5sum(file.path(case,c('model.lav','data/raw.csv'))))),'corpus_provenance')
  }
  # Fresh samples: the paired formulations constrain each loading to the same
  # constant. Scales change the data, never the target within a paired draw.
  for(rep in seq_len(reps))for(scale in if('--smoke' %in% args)1 else c(1,.01,100)) {
    draw_seed <- seed+rep+match(scale,c(1,.01,100))*10000L;set.seed(draw_seed)
    n <- 240L;f <- rnorm(n);g <- .35*f+rnorm(n)
    x <- scale*cbind(outer(f,rep(1,3)),outer(g,rep(1,3)))+scale*matrix(rnorm(n*6,sd=.6),n,6)
    x <- sweep(x,2,scale*c(2,-1,.5,3,0,-2),'+');colnames(x) <- paste0('y',1:6)
    s <- c('f =~ NA*ly_1_1*y1 + ly_2_1*y2 + ly_3_1*y3',
      'g =~ NA*ly_4_2*y4 + ly_5_2*y5 + ly_6_2*y6',
      'ly_1_1 == 3 - ly_2_1 - ly_3_1','ly_4_2 == 3 - ly_5_2 - ly_6_2')
    little_reference(as.data.frame(x),s,paste0('fresh_',rep,'_scale_',scale),scale)
  }
  set.seed(seed+90001L);x <- outer(rnorm(160),c(1,.8,.7,.9))+matrix(rnorm(640),160,4)
  colnames(x) <- paste0('x',1:4);spec <- magmaanlab::model_spec('f =~ x1 + x2 + x3 + x4',meanstructure=TRUE,fixed_x=FALSE)
  early <- list(nlopt=list(max_eval=1,ftol_rel=1e-13,xtol_rel=1e-11))
  for(method in c('ML','ULS','DWLS','WLS'))for(arm in c('default','early'))
    capture(function()magmaanlab::fit_model(spec,as.data.frame(x),estimator=method,
      control=if(arm=='early')early else NULL),'complete',arm,method,if(arm=='early')'failed_candidate' else 'observe')
  mask <- matrix(runif(length(x))>.25,nrow(x),4);mask[rowSums(mask)==0,1] <- TRUE
  raw <- list(X=list(x),mask=list(mask))
  for(arm in c('default','early','slsqp_early'))capture(function()core$fit_fiml(spec,raw,
    optimizer=if(arm=='slsqp_early')'nlopt-slsqp' else 'nlopt-lbfgs-slsqp-fallback',
    control=if(arm=='default')NULL else early),'missing',arm,'FIML',if(arm=='default')'observe' else 'failed_candidate')
  ordinal <- matrix(as.integer(cut(x,c(-Inf,-.5,.5,Inf),labels=FALSE)),nrow(x),4,dimnames=dimnames(x))
  producer <- core$data_ordinal_stats_from_raw(list(ordinal))
  producer$ov_names <- list(colnames(x));producer$ordered <- colnames(x)
  ospec <- magmaanlab::model_spec('f =~ x1 + x2 + x3 + x4',ordered=colnames(x),fixed_x=FALSE)
  for(arm in c('default','early'))capture(function()core$fit_dwls_ordinal(ospec,producer,
    control=if(arm=='early')early else NULL),'ordinal',arm,'ordinal_DWLS',if(arm=='early')'failed_candidate' else 'observe')
  if(!'--smoke' %in% args) {
    retained <- value('--retained-run','ml2s-audit-confirm-v1')
    for(cid in c(9L,10L)) {
      case_file <- file.path(here,'results',retained,paste0('case_',cid,'.rds'))
      if(!file.exists(case_file))stop('missing owning-leaf retained case: ',case_file)
      c <- readRDS(case_file)
      for(kind in c('nt','uls','dwls','adf','dls')) {
        fit <- capture(function()core$fit_ml2s(c$spec,c$raw,stage1=c$stage1,stage2_weight=kind),
          paste0('retained_ml2s_',cid),'default',paste0('ML2S_',kind),'retained_candidate')
        if(!inherits(fit,'error')) {
          a <- core$frontier_ml2s_convergence_audit(fit)
          controls[[length(controls)+1L]] <- data.frame(fit_id=length(rows),control='independent_composed_reaudit',
            status=a$status,stage1=a$stage1_assessment$status,handoff=a$handoff$status,
            stage2=a$stage2_assessment$status,
            distance=a$stage2$diagnostics$distance)
          saveRDS(a,file.path(out,paste0('ml2s_audit_',length(rows),'.rds')))
        }
      }
    }
  }
  fits <- do.call(rbind,rows)
  original <- if(length(references))do.call(rbind,references) else NULL
  checks <- data.frame(check=c('required_endpoints_return','pinned_reference_optima','early_candidates_fail','early_controls_exact','original_ml_objectives','oracle_transport_objectives','ordinary_defaults_pass','early_evaluation_costs','retained_composed_verdicts'),
    passed=c(all(fits$returned[fits$expected %in% c('pass','failed_candidate','retained_candidate')]),
      all(fits$fit_passed[fits$expected=='pass'] & fits$objective_gap[fits$expected=='pass']<1e-7),
      all(!fits$fit_passed[fits$expected=='failed_candidate']),
      all(fits$max_eval[fits$expected=='failed_candidate']==1 & fits$ftol_rel[fits$expected=='failed_candidate']==1e-13 & fits$xtol_rel[fits$expected=='failed_candidate']==1e-11),
      !is.null(original) && all(abs(original$objective_recomputed-fits$fmin[original$fit_id])<1e-8),
      all(vapply(oracle_rows,function(z)abs(z$transported_objective-z$reference_fmin)<1e-8,logical(1))),
      all(fits$returned[fits$expected=='observe'] & fits$fit_passed[fits$expected=='observe']),
      all(fits$f_evals[fits$expected=='failed_candidate']==
        ifelse(fits$family[fits$expected=='failed_candidate']=='missing' &
          fits$arm[fits$expected=='failed_candidate']=='early',2L,1L)),
      !length(controls) || all(vapply(controls,function(z)
        identical(z$stage1,'passed') && identical(z$handoff,'passed') &&
        identical(z$status=='passed',fits$fit_passed[z$fit_id]),logical(1)))))
  write_out(checks,'verification');if(length(controls))write_out(do.call(rbind,controls),'controls')
  if(length(references))write_out(do.call(rbind,references),'references')
  write_metadata(file.path(out,'metadata.csv'),values=list(run_id=run,seed_base=seed,reps=reps,
    seconds=proc.time()[['elapsed']]-begin,
    runner_md5=unname(tools::md5sum(file.path(here,'R/optimizer_audit.R'))),
    git_head=git_scalar(c('rev-parse','HEAD')),git_dirty=length(system2('git',c('status','--porcelain'),stdout=TRUE))>0L,
    binding_md5=unname(tools::md5sum(getLoadedDLLs()[['magmaanlab']][['path']])),
    scope='start mechanism and candidate retention; no domain recovery or default adoption'),
    packages=c('magmaanlab','lavaan'))
  if(anyNA(checks$passed)||!all(checks$passed))stop('optimizer mechanism gate failed; preserve run')
  cat('Wrote ',out,'\n',sep='')
}
