# Fixed-point numerical validation; interval decisions never replace fit verdicts.
run_audit_uncertainty <- function(args, here) {
  derive <- "--derive-input-errors" %in% args
  value <- function(k, default) {
    i <- match(k,args); if (is.na(i)) return(default)
    if (i==length(args) || startsWith(args[i+1L],'--')) stop('missing ',k)
    args[i+1L]
  }
  if (any(args %in% c('--help','-h'))) {
    cat('Usage: Rscript run_experiment.R --ordinary --audit-uncertainty --run-id NAME [--smoke] [--seed-base N] [--python PATH] [--derive-input-errors]\n',
      'Retained ULS bank, 15 fresh exact-population numerical controls, and seven retained finite NTML witnesses.\n',
      'Seven points per case; no optimization. Smoke selects one case of each type, including the flat NTML witness.\n',
      'Compare retained-input arithmetic intervals and the previously declared dimensional construction sensitivity.\n',
      '--derive-input-errors: add independently constructed outward-interval bounds and their audited distance.\n',
      '90-digit independent derivatives; neither arm replaces the stored fit verdict. Fresh output IDs required.\n',sep='')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--audit-uncertainty','--run-id','--smoke','--seed-base','--python','--derive-input-errors')
  if (any(startsWith(args,'--') & !args %in% known)) stop('unknown uncertainty option')
  run <- value('--run-id','audit-uncertainty'); seed <- as.integer(value('--seed-base','863261003'))
  if (!grepl('^[A-Za-z0-9_-]+$',run) || is.na(seed) || seed<1) stop('invalid run ID/seed')
  out <- file.path(here,'results',run); if (dir.exists(out)) stop('choose a fresh run ID')
  require_pkg('magmaanlab'); spec <- magmaanlab::model_spec(model_syntax)
  key <- function(p) paste(p$lhs,p$op,p$rhs)
  theta_of <- function(p) p$est[p$free>0][order(p$free[p$free>0])]
  evaluate <- function(p,s,n,estimator,errors=NULL) {
    ctl <- list(retain_newton_artifacts=TRUE,derive_interval_input_errors=derive && is.null(errors))
    if (!is.null(errors)) ctl$interval_input_errors <- errors
    magmaanlab::magmaan_core$estimate_evaluate_at(spec$partable,
      list(S=list(s),nobs=as.integer(n),ov_names=list(ov_names)),theta_of(p),estimator=estimator,
      bounds=list(lower=rep(-Inf,length(theta_of(p))),upper=rep(Inf,length(theta_of(p)))),audit_options=ctl)
  }
  plans <- list()
  input <- file.path(here,'results/uls-reliability-final-v2')
  covariance <- read.csv(file.path(input,'covariances.csv'))
  parameters <- read.csv(file.path(input,'reference_parameters.csv'))
  slopes <- read.csv(file.path(here,'results/uls-factor-square-root-v2/comparisons.csv'))
  ids <- sort(unique(parameters$case_id)); if ('--smoke' %in% args) ids <- head(ids,1)
  for (id in ids) {
    s <- matrix(0,6,6); d <- covariance[covariance$case_id==id,]
    s[cbind(d$row,d$col)] <- d$value; dimnames(s) <- list(ov_names,ov_names)
    pt <- spec$partable; ref <- parameters[parameters$case_id==id,]
    pt$est <- ref$est[match(key(pt),key(ref))]
    slope <- slopes$reference_distance[slopes$case_id==id & slopes$epsilon==.1]/.1
    plans[[length(plans)+1L]] <- list(pt=pt,s=s,n=100,estimator='ULS',role='retained_ULS',source_case=id,slope=slope)
  }
  # New exact-population controls validate the audit, not an optimizer policy.
  # Their binary64 sample matrices and points are judged as actually supplied.
  set.seed(seed)
  for (i in seq_len(if ('--smoke' %in% args) 1L else 15L)) {
    weak <- i>5 && i<=10; units <- if (i>10) c(.01,100,2,.3,10,.1) else rep(1,6)
    loading <- matrix(0,6,2)
    loading[1:3,1] <- c(if(weak) .03 else 1,runif(2,.4,1.2))
    loading[4:6,2] <- c(if(weak) .05 else 1,runif(2,.4,1.2))
    vx <- runif(1,.5,1.5); beta <- runif(1,-.6,.6); psi <- runif(1,.5,1.5)
    phi <- matrix(c(vx,beta*vx,beta*vx,psi+beta^2*vx),2)
    residual <- runif(6,.5,1.5)
    s <- (loading%*%phi%*%t(loading)+diag(residual))*outer(units,units)
    loading <- loading*units; anchors <- c(loading[1,1],loading[4,2])
    loading <- sweep(loading,2,anchors,'/'); phi <- phi*outer(anchors,anchors)
    pt <- spec$partable
    pt$est <- vapply(seq_len(nrow(pt)),function(j) {
      a <- pt$lhs[j]; b <- pt$rhs[j]
      if (pt$op[j]=='=~') return(loading[match(b,ov_names),match(a,c('X','Y'))])
      if (pt$op[j]=='~') return(phi[2,1]/phi[1,1])
      if (a %in% ov_names) return(residual[match(a,ov_names)]*units[match(a,ov_names)]^2)
      if (a=='X') return(phi[1,1])
      phi[2,2]-phi[1,2]^2/phi[1,1]
    },0.0)
    dimnames(s) <- list(ov_names,ov_names)
    p <- pt; j <- p$lhs=='x1' & p$op=='~~' & p$rhs=='x1'; p$est[j] <- p$est[j]+.001*s[1,1]
    slope <- evaluate(p,s,100,'ULS')$diagnostics$newton_accuracy$distance/.001
    stopifnot(is.finite(slope),slope>0)
    plans[[length(plans)+1L]] <- list(pt=pt,s=s,n=100,estimator='ULS',role=if(weak) 'fresh_weak' else if(i>10) 'fresh_mixed_units' else 'fresh_regular',source_case=i,slope=slope)
  }
  finite <- read.csv(file.path(here,'results/open-case-conclusions/finite_witnesses.csv'))
  values <- read.csv(file.path(here,'results/open-case-conclusions/parameters.csv'))
  ids <- if ('--smoke' %in% args) 4L else finite$case_id
  for (id in ids) {
    a <- finite[finite$case_id==id,]; v <- values$value[values$case_id==id]
    pt <- spec$partable; pt$est <- pt$ustart
    pt$est[pt$op=='=~'] <- c(1,v[1:2],1,v[3:4]); pt$est[pt$op=='~'] <- v[6]/v[5]
    pt$est[pt$lhs=='X' & pt$op=='~~'] <- v[5]
    pt$est[pt$lhs=='Y' & pt$op=='~~'] <- v[7]-v[6]^2/v[5]
    for(j in 1:6) pt$est[pt$lhs==ov_names[j] & pt$op=='~~'] <- v[7+j]
    old_seed <- (if(a$batch=='development') 202609281L else 902609281L)+match(a$design,names(designs_all()))*100000L+a$n*100L+a$rep
    s <- as.matrix(sample_moments(draw_data(design_sigma(designs_all()[[a$design]]),a$n,old_seed))$S[[1]])
    plans[[length(plans)+1L]] <- list(pt=pt,s=s,n=a$n,estimator='ML',role='retained_NTML',source_case=id,slope=NA_real_)
  }
  dir.create(out,recursive=TRUE)
  write_out <- function(d,name) {
    # Preserve the exact binary64 interval endpoints; generic CSV formatting
    # can round a lower bound upward or an upper bound downward.
    for (column in names(d)) if (is.double(d[[column]])) d[[column]] <- sprintf('%.17g',d[[column]])
    write_csv(d,file.path(out,paste0(name,'.csv')))
  }
  rows <- points <- matrices <- covariances <- list(); start <- proc.time()[['elapsed']]
  matrix_row <- function(x,id,name) {
    x <- as.matrix(x); if (!length(x)) return(NULL)
    grid <- expand.grid(row=seq_len(nrow(x)),col=seq_len(ncol(x)))
    cbind(point_id=id,name=name,grid,value=sprintf('%.17g',as.vector(x)))
  }
  for(cid in seq_along(plans)) {
    plan <- plans[[cid]]; s <- plan$s
    grid <- expand.grid(row=1:6,col=1:6)
    covariances[[cid]] <- cbind(case_id=cid,grid,value=sprintf('%.17g',as.vector(s)))
    targets <- if(plan$estimator=='ULS') c(0,.009,.00999,.01,.01001,.011,.02) else c(0,1e-7,1e-5,.001,.01,.1,.5)
    for(target in targets) {
      p <- plan$pt; j <- p$lhs=='x1' & p$op=='~~' & p$rhs=='x1'
      p$est[j] <- p$est[j]+(if(plan$estimator=='ULS') target/plan$slope else target)*s[1,1]
      fit <- evaluate(p,s,plan$n,plan$estimator); a <- fit$newton_audit
      if (is.null(a)) { saveRDS(fit,file.path(out,'failed_fit.rds')); stop('artifacts unavailable; failure retained') }
      id <- length(rows)+1L
      input <- if(plan$estimator=='ULS') a$equilibrated_factor else a$curvature_equilibrated_hessian
      vector <- if(plan$estimator=='ULS') a$metric_score_residual else a$curvature_scale*a$gradient
      k <- 8*nrow(input)*ncol(input); t <- k*.Machine$double.eps; allowance <- t/(1-t)
      errors <- list(matrix=allowance*sqrt(sum(input^2)),vector=allowance*max(sqrt(sum(vector^2)),1))
      conditional <- evaluate(p,s,plan$n,plan$estimator,errors)$newton_audit$distance_interval_conditional
      z <- a$distance_interval_retained_inputs
      rows[[id]] <- data.frame(point_id=id,case_id=cid,source_case=plan$source_case,role=plan$role,estimator=plan$estimator,nobs=plan$n,target=target,
        actual_passed=isTRUE(fit$diagnostics$newton_accuracy$passed),actual_status=fit$diagnostics$newton_accuracy$status,
        curvature_status=a$curvature_status,condition=fit$diagnostics$newton_accuracy$condition,
        retained_status=z$status,retained_decision=z$decision,retained_distance=z$distance,retained_lower=z$lower,retained_upper=z$upper,
        rank_margin=z$rank_margin,verification_matrix_error=z$factor_error_bound,
        matrix_allowance=errors$matrix,vector_allowance=errors$vector,construction_multiplier=k,
        conditional_status=conditional$status,conditional_decision=conditional$decision,conditional_lower=conditional$lower,conditional_upper=conditional$upper)
      if(derive) {
        e <- a$derived_interval_input_errors; z <- a$distance_interval_derived_inputs
        if(is.null(z)) z <- list(status=e$status,decision='unresolved',lower=0,upper=Inf)
        rows[[id]] <- cbind(rows[[id]],derived_status=e$status,derived_matrix=e$matrix,derived_vector=e$vector,
          derived_curvature=e$curvature,derived_curvature_lower_bound=e$curvature_lower_bound,
          derived_interval_status=z$status,derived_decision=z$decision,derived_lower=z$lower,derived_upper=z$upper)
      }
      pp <- p[c('lhs','op','rhs','free','est')]; pp$est <- sprintf('%.17g',pp$est)
      points[[id]] <- cbind(point_id=id,case_id=cid,pp)
      for(name in c('equilibrated_factor','factor_scale','metric_score_residual','curvature_equilibrated_hessian','curvature_scale','gradient','derivative_basis','curvature_coordinate_map'))
        matrices[[length(matrices)+1L]] <- matrix_row(a[[name]],id,name)
    }
    cat(sprintf('Uncertainty case %d/%d (%s); %.1fs\n',cid,length(plans),plan$role,proc.time()[['elapsed']]-start))
  }
  write_out(do.call(rbind,rows),'intervals'); write_out(do.call(rbind,points),'points')
  write_out(do.call(rbind,matrices),'artifacts'); write_out(do.call(rbind,covariances),'covariances')
  files <- file.path(here,c('run_experiment.R','R/audit_uncertainty.R','R/designs.R','R/fit.R','scripts/audit_uncertainty_reference.py','scripts/uls_audit_reference.py','scripts/refine_open_cases.py'))
  files <- c(files,file.path(here,'../../../../cpp/src/estimate/frontier/newton_uncertainty.cpp'),file.path(here,'../../../../cpp/include/magmaan/estimate/frontier/newton_accuracy.hpp'),file.path(here,'../../../../cpp/src/estimate/frontier/newton_input_bounds.cpp'),file.path(here,'../../../../cpp/src/estimate/frontier/detail_newton_interval.hpp'))
  pkg <- list.files(find.package('magmaanlab'),recursive=TRUE,full.names=TRUE); ref <- magmaan_cache_ref()
  write_metadata(file.path(out,'metadata.csv'),values=list(lane=if(derive) 'audit_construction' else 'audit_uncertainty',seed_base=seed,cases=length(plans),points=length(rows),threads=1,
    evaluation_elapsed_s=proc.time()[['elapsed']]-start,
    command= paste(commandArgs(),collapse=' '),source_md5=paste(tools::md5sum(files),collapse=';'),
    input_md5=paste(tools::md5sum(file.path(here,c('results/uls-reliability-final-v2/covariances.csv','results/uls-reliability-final-v2/reference_parameters.csv','results/uls-factor-square-root-v2/comparisons.csv','results/open-case-conclusions/finite_witnesses.csv','results/open-case-conclusions/parameters.csv'))),collapse=';'),
    package_md5=paste(tools::md5sum(pkg[grepl('\\.(so|rdb|rdx)$',pkg)]),collapse=';'),git_head=ref$git_head,git_dirty=ref$git_dirty,
    construction_sensitivity='gamma(8*rows*columns), binary64 epsilon; inherited dimensional assumption, not a proved SEM construction bound',
    fresh_scope='new exact-population numerical controls; no sampled fitting/default or optimizer confirmation',
    derived_inputs=derive,interpretation='conditional numerical intervals and actual production verdict separate; zero bounds concern retained inputs only'),packages='magmaanlab')
  status <- system2(value('--python','python3'),c(shQuote(file.path(here,'scripts/audit_uncertainty_reference.py')),'--run-dir',shQuote(normalizePath(out))))
  if(status!=0) stop('independent reference failed; inputs retained')
  cat('Uncertainty evidence saved: ',out,'\n',sep='')
}
