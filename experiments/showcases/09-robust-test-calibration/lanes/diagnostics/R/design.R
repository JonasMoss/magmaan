# Published two-factor population parameters, doi:10.1080/10705511.2024.2372028.
# Formulas below construct the covariance; no third-party source is included.
# Unit factor variances, correlation .5, residual variances 1 - loading^2.
population <- function(p) {
  a <- c(.4,.5,.6,.8,.4,.7,.8,.6,.6,.3)[seq_len(p/2)]
  b <- c(.4,.7,.6,.4,.8,.8,.4,.7,.5,.6)[seq_len(p/2)]
  L <- matrix(0, p, 2)
  L[seq_len(p/2),1] <- a; L[p/2 + seq_len(p/2),2] <- b
  sigma <- L %*% matrix(c(1,.5,.5,1),2) %*% t(L) + diag(1-rowSums(L^2))
  list(sigma=sigma, root=chol(sigma), syntax=paste(
    paste('f1 =~',paste0('x',seq_len(p/2),collapse=' + ')),
    paste('f2 =~',paste0('x',p/2+seq_len(p/2),collapse=' + ')),sep='\n'))
}
prepare <- function(p, distribution) {
  pop <- population(p)
  pop$cal <- if(distribution == 'vm2') magmaanlab::magmaan_core$sim_vm_calibrate(
    pop$sigma, rep(3,p), rep(21,p)) else NULL
  pop
}
draw_sample <- function(ctx, n, distribution, seed) {
  p <- nrow(ctx$sigma)
  if(distribution == 'vm2') {
    X <- magmaanlab::magmaan_core$sim_vm_draw(ctx$cal,n=n,reps=1L,seed_base=seed)$draws[[1L]]
  } else {
    set.seed(seed)
    X <- matrix(rnorm(n*p),n,p) %*% ctx$root
    # Symmetric finite-eighth-moment stress extension, not a source-paper cell.
    if(distribution == 't10') X <- X * sqrt(8/rchisq(n,df=10))
  }
  colnames(X) <- paste0('x',seq_len(p)); as.data.frame(X)
}
# One batched call shares the fitted-model spectrum across these comparators.
lr_tests <- c('std_ml','sb_ml','peba2_ml','peba4_ml',
              'std_rls','sb_ug_rls','peba2_ug_rls','peba4_rls')
score_tests <- c('score_sb','score_peba2','score_peba4','score_sandwich')
clock_seconds <- function() proc.time()[['elapsed']]
measure <- function(expr) {
  start <- clock_seconds()
  ans <- tryCatch(withCallingHandlers(force(expr),warning=function(w) invokeRestart('muffleWarning')),
                  error=function(e)e)
  list(value=ans, seconds=clock_seconds()-start)
}
fit_sample <- function(ctx, d) magmaanlab::fit_model(ctx$syntax,d,estimator='ML',
  control=list(max_iter=4000L,ftol=1e-12,gtol=1e-8))
one_rep <- function(cell, rep_id, ctx, seed_base) {
  seed <- seed_base + cell$cell_id*100000L + rep_id
  start <- clock_seconds()
  out <- data.frame(cell_id=cell$cell_id,p=cell$p,n=cell$n,
    distribution=cell$distribution,rep=rep_id,seed=seed,
    method=c(lr_tests,score_tests),p_value=NA_real_,df=NA_real_,statistic=NA_real_,
    error='',admissible=NA,score_rank=NA_integer_,generation_seconds=NA_real_,fit_seconds=NA_real_,
    postfit_seconds=NA_real_,postfit_batch_seconds=NA_real_,
    pipeline_seconds=NA_real_,score_rls_abs_diff=NA_real_,stationarity=NA_real_,
    total_rep_seconds=NA_real_,stringsAsFactors=FALSE)
  finish <- function() {out$total_rep_seconds <- clock_seconds()-start; out}
  gen <- measure(draw_sample(ctx,cell$n,cell$distribution,seed));out$generation_seconds <- gen$seconds
  if(inherits(gen$value,'error')) {out$error <- conditionMessage(gen$value);return(finish())}
  ft <- measure(fit_sample(ctx,gen$value));out$fit_seconds <- ft$seconds
  if(inherits(ft$value,'error') || !isTRUE(ft$value$converged)) {
    out$error <- if(inherits(ft$value,'error')) conditionMessage(ft$value) else 'fit not converged'
    return(finish())
  }
  f <- ft$value
  out$admissible <- isTRUE(f$diagnostics$admissibility$admissible)
  if(!out$admissible[1]) {out$error <- 'fit covariance inadmissible';return(finish())}
  # Snapshot preparation is included in total_rep_seconds / simulation wall
  # time, but excluded from the historical per-method pipeline_seconds field.
  context <- magmaanlab::prepare_inference(f, gen$value)
  lr <- measure(magmaanlab::fmg_tests(context,tests=lr_tests))
  ix <- seq_along(lr_tests)
  out$postfit_batch_seconds[ix] <- lr$seconds
  if(inherits(lr$value,'error')) out$error[ix] <- conditionMessage(lr$value) else {
    out$p_value[ix] <- lr$value$p_value;out$df[ix] <- lr$value$df
    out$statistic[ix] <- lr$value$base_statistic
  }
  # Explicit asymptotic primitives: no multiplier draws or exact-mixture tail.
  # Projection/spectrum are shared by the requested score calibrations.
  sc <- measure({
    projected <- magmaanlab::project_scores(magmaanlab::score_components(context,
      sensitivity='expected',metric='expected'))
    reference <- magmaanlab::score_spectrum(projected)
    list(projected=projected, reference=reference,
         statistic_effective=projected$statistic)
  })
  ix <- length(lr_tests)+seq_along(score_tests)
  out$postfit_batch_seconds[ix] <- sc$seconds
  if(inherits(sc$value,'error')) out$error[ix] <- conditionMessage(sc$value) else {
    z <- sc$value$reference
    out$score_rank <- sum(z$eigenvalues > 1e-10 * max(z$eigenvalues))
    out$df[ix] <- z$df;out$statistic[ix] <- z$statistic
    for(k in 1:3) {
      tr <- measure(magmaanlab::calibrate_quadratic(z,c('sb','peba2','peba4')[k]))
      out$postfit_seconds[ix[k]] <- sc$seconds + tr$seconds
      if(inherits(tr$value,'error')) out$error[ix[k]] <- conditionMessage(tr$value) else
        out$p_value[ix[k]] <- tr$value$p_value
    }
    sandwich <- measure(magmaanlab::calibrate_quadratic(
      magmaanlab::score_sandwich(sc$value$projected),'std'))
    out$postfit_seconds[ix[4]] <- sc$seconds + sandwich$seconds
    if(inherits(sandwich$value,'error')) out$error[ix[4]] <- conditionMessage(sandwich$value) else {
      out$statistic[ix[4]] <- sandwich$value$statistic
      out$p_value[ix[4]] <- sandwich$value$p_value
    }
    if(!inherits(lr$value,'error')) out$score_rls_abs_diff <- abs(
      z$statistic-lr$value$base_statistic[lr$value$label=='std_rls'])
  }
  # At an interior optimum expected-information score equals fitted-weight
  # RLS. Reject the shared fit if this necessary stationarity check fails.
  if(any(is.finite(out$score_rls_abs_diff)) &&
     out$score_rls_abs_diff[1] > 1e-6*(1+abs(sc$value$statistic_effective))) {
    out$error[] <- 'null-fit score/RLS identity check failed'
  }
  bad <- !is.finite(out$p_value) | out$p_value < 0 | out$p_value > 1
  out$error[bad & !nzchar(out$error)] <- 'invalid p-value'
  # LR times are deliberately labeled batch times; do not pretend to have
  # measured eight separate method calls. Score rows include shared setup.
  out$pipeline_seconds <- out$generation_seconds+out$fit_seconds+
    ifelse(is.na(out$postfit_seconds),out$postfit_batch_seconds,out$postfit_seconds)
  finish()
}
