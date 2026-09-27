# Pilot only: no repeated labels, equality constraints or arbitrary fixed values.
# The original model is valid in both spaces because its only nonzero fixed
# values are marker loadings or std.lv unit variances.
normalize_unconstrained_sample <- function(model, sample, std_lv) {
  p <- model$partable
  stopifnot(length(sample$S)==1L,all(p$op %in% c('=~','~','~~','~1')),
            length(unique(p$block))==1L,
            !anyDuplicated(p$free[p$free>0]))
  labels <- p$label[p$free>0 & !is.na(p$label) & nzchar(p$label)]
  stopifnot(!anyDuplicated(labels))
  ov <- colnames(sample$S[[1]]);lv <- unique(p$lhs[p$op=='=~'])
  sd <- sqrt(diag(sample$S[[1]]));names(sd)<-ov
  stopifnot(all(is.finite(sd)),all(sd>0))
  latent <- setNames(rep(1,length(lv)),lv)
  if(!std_lv) for(f in lv) {
    marker <- which(p$op=='=~' & p$lhs==f & p$free==0 & is.finite(p$ustart) & p$ustart!=0)
    stopifnot(length(marker)==1,p$ustart[marker]==1)
    latent[f] <- sd[p$rhs[marker]]
  }
  nonzero <- p$free==0 & is.finite(p$ustart) & p$ustart!=0
  permitted <- if(std_lv) p$op=='~~' & p$lhs==p$rhs & p$lhs %in% lv & p$ustart==1 else
    p$op=='=~' & p$ustart==1
  stopifnot(all(!nonzero | permitted))
  units <- c(sd,latent); row_units <- rep(NA_real_,nrow(p))
  for(i in seq_len(nrow(p))) row_units[i] <- switch(p$op[i],
    '=~'=units[p$rhs[i]]/units[p$lhs[i]],
    '~'=units[p$lhs[i]]/units[p$rhs[i]],
    '~~'=units[p$lhs[i]]*units[p$rhs[i]],
    '~1'=units[p$lhs[i]])
  stopifnot(all(is.finite(row_units)),all(row_units>0))
  theta_units <- numeric(max(p$free));free <- p$free>0
  theta_units[p$free[free]] <- row_units[free]
  normalized <- sample
  normalized$S[[1]] <- sample$S[[1]]/outer(sd,sd)
  if(length(sample$mean)) normalized$mean[[1]] <- sample$mean[[1]]/sd
  list(sample=normalized,theta_units=theta_units,sd=sd)
}

normalization_fit <- function(model,sample,route) {
  # This pilot controls normalization itself; keep the original-unit arm explicit.
  ctl <- list(start='fabin3',start_transport='auto',normalize_sample=FALSE,max_iter=5000L,
    nlopt=list(max_eval=5000L,ftol_rel=1e-12,xtol_rel=1e-10),coordinate_scaling='information')
  if(route=='direct') {
    fit <- magmaanlab::frontier_fit_ml_psd(model,sample,optimizer='nlopt-slsqp',control=ctl,
      preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6)
    return(list(fit=fit,converged=isTRUE(fit$converged),stage='direct'))
  }
  x <- magmaanlab::frontier_fit_ml_psd_fallback(model,sample,
    ordinary_optimizer='nlopt-lbfgs',psd_optimizer='nlopt-slsqp',ordinary_control=ctl,
    psd_control=ctl[setdiff(names(ctl),c('start','start_transport'))],
    preconditioning='diagonal',start_eigen_floor=1e-6,feasibility_tol=1e-6)
  used <- if(isTRUE(x$fallback_used)) x$psd else x$ordinary
  if(is.null(used$fit)) stop(paste(used$error$kind,used$error$detail))
  list(fit=used$fit,converged=isTRUE(x$converged),
    stage=if(isTRUE(x$fallback_used)) paste0('psd:',x$fallback_reason) else 'ordinary')
}

normalization_attempt <- function(model,sample,std_lv,route,normalize) {
  rec <- data.frame(returned=FALSE,fit_success=FALSE,success=FALSE,output_success=FALSE,
    fmin=NA_real_,seconds=NA_real_,stage='',fit_accuracy='',original_accuracy='',
    original_accuracy_passed=FALSE,original_psd=FALSE,transport_ok=FALSE,
    covariance_error=NA_real_,objective_error=NA_real_,message='')
  setup <- normalize_unconstrained_sample(model,sample,std_lv)
  t0 <- proc.time()[['elapsed']]
  z <- tryCatch(normalization_fit(model,if(normalize)setup$sample else sample,route),error=function(e)e)
  rec$seconds <- proc.time()[['elapsed']]-t0
  if(inherits(z,'error')) {rec$message<-one_line(conditionMessage(z));return(list(record=rec))}
  fit<-z$fit;rec$returned<-TRUE;rec$fmin<-fit$fmin;rec$stage<-z$stage
  rec$fit_success<-z$converged && isTRUE(fit$diagnostics$admissibility$admissible)
  rec$fit_accuracy<-fit$diagnostics$newton_accuracy$status %||% ''
  theta<-fit$theta * if(normalize)setup$theta_units else 1
  check<-tryCatch({
    ev<-magmaanlab::magmaan_core$evaluate_at(model,sample,theta,estimator='ML')
    implied<-magmaanlab::magmaan_core$model_implied(ev)$sigma[[1]]
    fitted<-magmaanlab::magmaan_core$model_implied(fit)$sigma[[1]]
    expected<-fitted * if(normalize)outer(setup$sd,setup$sd) else 1
    accuracy<-magmaanlab::frontier_newton_accuracy(ev,psd=TRUE)
    rec$covariance_error<-max(abs(implied-expected)/outer(setup$sd,setup$sd))
    rec$objective_error<-abs(ev$fmin-fit$fmin)
    rec$original_psd<-isTRUE(ev$diagnostics$admissibility$admissible)
    rec$original_accuracy<-accuracy$status;rec$original_accuracy_passed<-isTRUE(accuracy$passed)
    rec$transport_ok<-rec$covariance_error<=1e-10 && rec$objective_error<=1e-8*(1+abs(fit$fmin))
    rec$success<-rec$fit_success && rec$transport_ok && rec$original_psd
    rec$output_success<-rec$success && rec$original_accuracy_passed
    list(record=rec,theta=theta,covariance=implied/outer(setup$sd,setup$sd))
  },error=function(e)e)
  if(inherits(check,'error')) {rec$message<-one_line(conditionMessage(check));return(list(record=rec))}
  check
}


normalization_populations <- function() {
  pops <- all_populations()
  pops <- pops[vapply(pops,function(p)p$role=='test' && p$family!='constrained',logical(1))]
  pops[['r47_mis_f2_resid']]$models[[2]] <- fitted_model('correlated_residuals',
    'f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6\nx1 ~~ x4\nx2 ~~ x5\nx3 ~~ x6')
  pops
}
