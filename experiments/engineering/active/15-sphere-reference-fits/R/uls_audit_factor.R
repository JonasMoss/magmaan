# Fixed-point controls from this leaf's reference bank. References are used
# deliberately here: this tests numerical auditing, never cold-start search.
run_uls_audit_factor <- function(args, here) {
  value <- function(k, default) {
    i <- match(k, args); if (is.na(i)) return(default)
    if (i == length(args) || startsWith(args[i+1L], '--')) stop('missing ', k)
    args[i+1L]
  }
  if (any(args %in% c('--help', '-h'))) {
    cat(paste('Usage: Rscript run_experiment.R --ordinary --uls-audit-factor [--smoke]',
      '--run-id NAME --source-run uls-reliability-final-v2 --python python3',
      'Audits 25 retained reference points and x1-variance perturbations; no optimization.',
      'Compare original fixed-condition guards with diagnostic distances against 90-digit calculations.',
      'Use separate run IDs/install libraries for the baseline and square-root implementation.',
      'Requires mpmath; raw artifacts are local, compact comparisons and metadata are frozen.', sep='\n'), '\n')
    return(invisible(NULL))
  }
  known <- c('--ordinary','--uls-audit-factor','--smoke','--run-id','--source-run','--python')
  if (any(startsWith(args,'--') & !args %in% known)) stop('unknown factor-audit option')
  run <- value('--run-id','uls-audit-factor'); source_run <- value('--source-run','uls-reliability-final-v2')
  if (!all(grepl('^[A-Za-z0-9_-]+$',c(run, source_run)))) stop('invalid run ID')
  out <- file.path(here,'results',run); input <- file.path(here,'results',source_run)
  if (dir.exists(out)) stop('choose a fresh run ID')
  require_pkg('magmaanlab')
  covariance <- read.csv(file.path(input,'covariances.csv')); parameters <- read.csv(file.path(input,'reference_parameters.csv'))
  cases <- sort(unique(parameters$case_id)); if ('--smoke' %in% args) cases <- head(cases,1)
  spec <- magmaanlab::model_spec(model_syntax)
  dir.create(out, recursive=TRUE)
  write_out <- function(d,name) write_csv(d,file.path(out,paste0(name,'.csv')))
  input_covariance <- covariance[covariance$case_id %in% cases,]
  input_covariance$value <- sprintf('%.17g',input_covariance$value)
  write_out(input_covariance, 'covariances')
  sources <- file.path(here,c('run_experiment.R','R/uls_audit_factor.R','scripts/uls_audit_reference.py','scripts/uls_unit_reference.py'))
  package_files <- list.files(find.package('magmaanlab'),recursive=TRUE,full.names=TRUE)
  meta <- list(lane='uls_audit_factor', source_run=source_run, cases=length(cases),
    points_per_case=4, source_md5=paste(tools::md5sum(sources),collapse=';'),
    input_md5=paste(tools::md5sum(file.path(input,c('covariances.csv','reference_parameters.csv'))),collapse=';'),
    package_md5=paste(tools::md5sum(package_files[grepl('\\.(so|rdb|rdx)$',package_files)]),collapse=';'),
    target='original mixed-unit unrestricted ULS, explicit infinite bounds',
    judge='90-digit exact derivatives at each returned floating-point input; d<=.01; fixed condition cap 1e12 unchanged',
    controls='reference point; x1 variance + sample variance * (1e-4,1e-3,.1); no optimizer',threads=1)
  rows <- points <- artifacts <- list(); t0 <- proc.time()[['elapsed']]
  for (case_id in cases) {
    s <- matrix(0,6,6); d <- covariance[covariance$case_id==case_id,]
    s[cbind(d$row,d$col)] <- d$value; dimnames(s) <- list(ov_names,ov_names)
    sample <- list(S=list(s),nobs=100L,ov_names=list(ov_names))
    ref <- parameters[parameters$case_id==case_id,]; pt <- spec$partable
    key <- function(p) paste(p$lhs,p$op,p$rhs)
    pt$est <- ref$est[match(key(pt),key(ref))]
    if (anyNA(pt$est)) stop('reference does not cover model rows')
    for (epsilon in c(0,1e-4,1e-3,.1)) {
      p <- pt; i <- p$lhs=='x1' & p$op=='~~' & p$rhs=='x1'
      p$est[i] <- p$est[i]+epsilon*s[1,1]
      theta <- p$est[p$free>0][order(p$free[p$free>0])]
      fit <- magmaanlab::magmaan_core$estimate_evaluate_at(spec$partable,sample,theta,estimator='ULS',
        bounds=list(lower=rep(-Inf,length(theta)),upper=rep(Inf,length(theta))),
        audit_options=list(retain_newton_artifacts=TRUE))
      audit <- fit$diagnostics$newton_accuracy; a <- fit$newton_audit
      id <- length(rows)+1L
      rows[[id]] <- data.frame(point_id=id,case_id=case_id,epsilon=epsilon,
        audit_status=audit$status,passed=isTRUE(audit$passed),distance=audit$distance,
        condition=audit$condition,solve_residual=audit$solve_residual,
        curvature_status=if(is.null(a)) NA_character_ else a$curvature_status,
        curvature_condition=if(is.null(a)) NA_real_ else a$curvature_condition,
        factor_status=if(is.null(a)) NA_character_ else a$factor_status,
        factor_condition=if(is.null(a)) NA_real_ else a$factor_condition,
        factor_rank=if(is.null(a)) NA_integer_ else a$factor_rank)
      input_point <- p[c('lhs','op','rhs','est')]
      input_point$est <- sprintf('%.17g',input_point$est)
      points[[id]] <- cbind(point_id=id,case_id=case_id,input_point)
      artifacts[[id]] <- a
    }
    cat(sprintf('Factor audit case %d/%d; %.1fs\n',case_id,length(cases),proc.time()[['elapsed']]-t0))
  }
  write_out(do.call(rbind,rows),'audits'); write_out(do.call(rbind,points),'points')
  saveRDS(artifacts,file.path(out,'artifacts.rds'))
  meta$audit_elapsed_s <- proc.time()[['elapsed']]-t0
  write_metadata(file.path(out,'metadata.csv'),values=meta,packages='magmaanlab')
  status <- system2(value('--python','python3'),c(shQuote(file.path(here,'scripts/uls_audit_reference.py')),
    '--run-dir',shQuote(normalizePath(out))))
  if (status!=0) stop('independent point calculation failed; audit inputs retained')
  reference <- read.csv(file.path(out,'point_references.csv'))
  reference$reference_accurate <- tolower(reference$reference_accurate)=='true'
  reference$reference_hessian_positive <- tolower(reference$reference_hessian_positive)=='true'
  comparison <- merge(do.call(rbind,rows),reference,by=c('point_id','case_id'),sort=TRUE)
  comparison$absolute_distance_error <- abs(comparison$distance-comparison$reference_distance)
  comparison$distance_decision_match <- (comparison$distance<=.01)==comparison$reference_accurate
  # Exploratory candidate only: apply the numerical cap to the QR factor's
  # condition, retaining the Hessian cap, rank, residual and distance checks.
  # This does not replace the actual production verdict above.
  comparison$factor_guard_candidate <- with(comparison,
    factor_status=='available' & curvature_status=='available' &
    sqrt(factor_condition)<=1e12 & curvature_condition<=1e12 &
    solve_residual<=1e-10 & distance<=.01)
  write_out(comparison,'comparisons')
  cat('Factor comparison saved: ',out,'\n',sep='')
}
