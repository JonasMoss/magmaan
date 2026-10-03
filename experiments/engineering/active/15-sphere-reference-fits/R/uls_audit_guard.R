# Numerical calibration at fixed points, including controls near the budget.
# This lane records conditional intervals; it never replaces a fit verdict.
run_uls_audit_guard <- function(args, here) {
  curvature <- '--uls-audit-curvature' %in% args
  value <- function(k, default) {
    i <- match(k,args); if (is.na(i)) return(default)
    if (i==length(args) || startsWith(args[i+1L],'--')) stop('missing ',k)
    args[i+1L]
  }
  if (any(args %in% c('--help','-h'))) {
    cat(paste('Usage: Rscript run_experiment.R --ordinary --uls-audit-guard [--smoke]',
      '--run-id NAME --source-run uls-reliability-final-v2 --factor-run uls-factor-square-root-v2 --python python3',
      'Seven fixed points per dataset: reference and distances near .009, .00999, .01, .01001, .011, .02.',
      '90-digit forward-error comparisons and conditional projection intervals; no optimization or default change.',
      'Inspect diagonal curvature and a reference-only Jacobian QR change of coordinates.',
      'Requires mpmath and the square-root artifact API; matrices remain local.',sep='\n'),'\n')
    cat('Use --uls-audit-curvature instead of --uls-audit-guard to validate the implemented QR curvature and Newton steps.\n')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--uls-audit-guard','--uls-audit-curvature','--smoke','--run-id','--source-run','--factor-run','--python')
  if (any(startsWith(args,'--') & !args %in% known)) stop('unknown guard option')
  run <- value('--run-id','uls-audit-guard'); source_run <- value('--source-run','uls-reliability-final-v2')
  factor_run <- value('--factor-run','uls-factor-square-root-v2')
  if (!all(grepl('^[A-Za-z0-9_-]+$',c(run,source_run,factor_run)))) stop('invalid run ID')
  out <- file.path(here,'results',run); input <- file.path(here,'results',source_run)
  if (dir.exists(out)) stop('choose a fresh run ID')
  require_pkg('magmaanlab')
  covariance <- read.csv(file.path(input,'covariances.csv'))
  parameters <- read.csv(file.path(input,'reference_parameters.csv'))
  calibration_file <- file.path(here,'results',factor_run,'comparisons.csv')
  calibration <- read.csv(calibration_file)
  cases <- sort(unique(parameters$case_id)); if ('--smoke' %in% args) cases <- head(cases,1)
  targets <- c(0,.009,.00999,.01,.01001,.011,.02)
  spec <- magmaanlab::model_spec(model_syntax); dir.create(out,recursive=TRUE)
  write_out <- function(d,name) write_csv(d,file.path(out,paste0(name,'.csv')))
  cov <- covariance[covariance$case_id %in% cases,]; cov$value <- sprintf('%.17g',cov$value)
  write_out(cov,'covariances')
  rows <- points <- matrices <- list(); t0 <- proc.time()[['elapsed']]
  for (case_id in cases) {
    s <- matrix(0,6,6); d <- covariance[covariance$case_id==case_id,]
    s[cbind(d$row,d$col)] <- d$value; dimnames(s) <- list(ov_names,ov_names)
    sample <- list(S=list(s),nobs=100L,ov_names=list(ov_names))
    ref <- parameters[parameters$case_id==case_id,]; pt <- spec$partable
    key <- function(p) paste(p$lhs,p$op,p$rhs)
    pt$est <- ref$est[match(key(pt),key(ref))]; if (anyNA(pt$est)) stop('incomplete reference')
    # The diagonal perturbation leaves J and Gamma fixed. Historical exact
    # distances give a slope for nomination only; the new exact judge labels it.
    d0 <- calibration[calibration$case_id==case_id & calibration$epsilon==.1,]
    stopifnot(nrow(d0)==1L,d0$reference_distance>0)
    for (target in targets) {
      epsilon <- target*.1/d0$reference_distance
      p <- pt; i <- p$lhs=='x1' & p$op=='~~' & p$rhs=='x1'
      p$est[i] <- p$est[i]+epsilon*s[1,1]
      theta <- p$est[p$free>0][order(p$free[p$free>0])]
      fit <- magmaanlab::magmaan_core$estimate_evaluate_at(spec$partable,sample,theta,estimator='ULS',
        bounds=list(lower=rep(-Inf,length(theta)),upper=rep(Inf,length(theta))),
        audit_options=list(retain_newton_artifacts=TRUE))
      a <- fit$newton_audit; audit <- fit$diagnostics$newton_accuracy
      if (is.null(a) || a$factor_status!='available') {
        write_out(cbind(case_id=case_id,target_distance=target,p),'failed_point')
        saveRDS(fit,file.path(out,'failed_fit.rds'))
        stop('factor artifacts unavailable; failed point retained')
      }
      id <- length(rows)+1L
      rows[[id]] <- data.frame(point_id=id,case_id=case_id,target_distance=target,epsilon=epsilon,
        passed=isTRUE(audit$passed),distance=audit$distance,audit_status=audit$status,
        curvature_condition=a$curvature_condition,factor_condition=a$factor_condition,
        factor_residual=a$factor_residual,solve_residual=audit$solve_residual)
      pp <- p[c('lhs','op','rhs','free','est')]; pp$est <- sprintf('%.17g',pp$est)
      points[[id]] <- cbind(point_id=id,case_id=case_id,pp)
      names <- c('metric_factor','metric_score_residual','hessian')
      if (curvature) names <- c(names,'curvature_coordinate_map','curvature_equilibrated_hessian',
        'newton_step','ls_curvature_correction','whitened_jacobian','whitened_residual')
      for (name in names) {
        x <- as.matrix(a[[name]]); grid <- expand.grid(row=seq_len(nrow(x)),col=seq_len(ncol(x)))
        matrices[[length(matrices)+1L]] <- cbind(point_id=id,name=name,grid,value=sprintf('%.17g',as.vector(x)))
      }
    }
    cat(sprintf('Guard controls %d/%d; %.1fs\n',match(case_id,cases),length(cases),proc.time()[['elapsed']]-t0))
  }
  write_out(do.call(rbind,rows),'audits'); write_out(do.call(rbind,points),'points')
  write_out(do.call(rbind,matrices),'artifacts')
  files <- file.path(here,c('run_experiment.R','R/uls_audit_guard.R','scripts/uls_guard_reference.py',
    'scripts/uls_audit_reference.py','scripts/uls_unit_reference.py'))
  package_files <- list.files(find.package('magmaanlab'),recursive=TRUE,full.names=TRUE)
  ref <- magmaan_cache_ref()
  write_metadata(file.path(out,'metadata.csv'),values=list(lane=if(curvature) 'uls_audit_curvature' else 'uls_audit_guard',source_run=source_run,
    factor_run=factor_run,cases=length(cases),points_per_case=length(targets),threads=1,
    targets=paste(targets,collapse=';'),target='original mixed-unit unrestricted ULS; explicit infinite bounds',
    git_head=ref$git_head,git_dirty=ref$git_dirty,
    source_md5=paste(tools::md5sum(files),collapse=';'),
    input_md5=paste(tools::md5sum(c(file.path(input,c('covariances.csv','reference_parameters.csv')),calibration_file)),collapse=';'),
    package_md5=paste(tools::md5sum(package_files[grepl('\\.(so|rdb|rdx)$',package_files)]),collapse=';'),
    interpretation='development numerical calibration; QR curvature when requested; conditional perturbation allowances; acceptance thresholds unchanged',
    audit_elapsed_s=proc.time()[['elapsed']]-t0),packages='magmaanlab')
  status <- system2(value('--python','python3'),c(shQuote(file.path(here,'scripts/uls_guard_reference.py')),
    '--run-dir',shQuote(normalizePath(out))))
  if (status!=0) stop('guard reference failed; inputs retained')
  cat('Guard calibration saved: ',out,'\n',sep='')
}
