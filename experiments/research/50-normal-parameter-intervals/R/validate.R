validate_normal <- function(output) {
  checks <- list()
  check <- function(name, error, tolerance) {
    checks[[length(checks)+1L]] <<- data.frame(check=name,error=error,
      tolerance=tolerance,passed=is.finite(error)&&abs(error)<=tolerance)
  }
  X <- draw_sample(300L,2026112750L); u <- fit_sample(X)
  id <- free_id(u,'loading'); b <- u$theta[id]+.1
  r <- fit_sample(X,'loading',b,u$theta)
  prof <- core$frontier_profile_lrt_parameter_ml(u,id,b,optimizer='nlopt-slsqp',control=control_primary)
  lr <- ordinary_lr(u,r,nrow(X))
  check('profile_binding_vs_equality_fit',lr-prof$T,1e-5)
  lu <- lavaan::cfa(model_text,data=as.data.frame(X),std.lv=TRUE,meanstructure=FALSE,
                    se='none',test='standard')
  lrfit <- lavaan::cfa(paste(model_text,sprintf('b == %.17g',b),sep='\n'),
    data=as.data.frame(X),std.lv=TRUE,meanstructure=FALSE,se='none',test='standard')
  check('lavaan_convergence',as.numeric(!lavaan::lavInspect(lu,'converged') ||
    !lavaan::lavInspect(lrfit,'converged')),0)
  check('profile_lr_vs_lavaan',lr-as.numeric(2*(lavaan::fitMeasures(lu,'logl')-lavaan::fitMeasures(lrfit,'logl'))),1e-5)
  engine <- new_engine(u,X,'loading','score_expected')
  score <- engine$evaluate(b)$statistic
  nested <- magmaanlab::nested_score_test(u,r,X,sensitivity='expected')
  check('score_vs_nested_surface',score-nested$statistic_effective,1e-8)
  # Independent likelihood formula evaluated from library-implied covariance.
  S <- crossprod(X)/nrow(X)
  objective <- function(theta) {
    f <- r; f$theta <- theta; V <- core$model_implied(f)$sigma[[1]]
    .5*(as.numeric(determinant(V,logarithm=TRUE)$modulus)+sum(diag(solve(V,S))))
  }
  eps <- 1e-5; th <- r$theta; a <- th; a[id] <- a[id]+eps
  z <- th; z[id] <- z[id]-eps
  derivative <- (objective(a)-objective(z))/(2*eps)
  I <- core$infer_information_expected(r)
  score_fd <- (nrow(X)*derivative)^2 * solve(I)[id,id]
  check('score_vs_finite_difference',score-score_fd,1e-4)
  # Analytically known normal variance information, after estimating the mean.
  set.seed(2026112751); xx <- data.frame(x=rnorm(200,sd=1.4)); xx$x <- xx$x-mean(xx$x)
  one <- magmaanlab::fit_model('x ~~ v*x',xx,meanstructure=FALSE,
                               optimizer='nlopt-slsqp',control=control_primary)
  iv <- one$partable$free[one$partable$label=='v']
  info <- core$infer_information_expected(one)
  check('normal_variance_information',info[iv,iv]-200/(2*one$theta[iv]^2),1e-7)
  ctx <- magmaanlab::prepare_inference(u,X)
  H <- magmaanlab::inference_information(ctx,'expected')
  check('information_covariance_scaling',max(abs(magmaanlab::parameter_covariance(ctx,H)-solve(H))),1e-10)
  # Independent large draw: normal sample covariance entry variance is known.
  big <- draw_sample(100000L,2026112752L); Sig <- population_sigma()
  mcse <- sqrt((outer(diag(Sig),diag(Sig))+Sig^2)/nrow(big))
  check('population_covariance_max_mc_z',max(abs(crossprod(big)/nrow(big)-Sig)/mcse),5)
  for(method in c('score_expected','lr')) {
    eng <- new_engine(u,X,'loading',method)
    se <- sqrt(solve(H)[id,id]); ci <- invert(eng,u$theta[id],se)
    gap <- max(abs(c(eng$evaluate(ci$lower)$statistic,eng$evaluate(ci$upper)$statistic)-cutoff))
    check(paste0(method,'_endpoint'),gap,1e-4)
    check(paste0(method,'_truth_inversion'),as.numeric(
      (ci$lower<=.7 && ci$upper>=.7)!=(eng$evaluate(.7)$statistic<=cutoff)),0)
  }
  # Exercise bootstrap null refits, reproducibility, and B-prefix convention.
  be <- new_engine(u,X,'loading','lr_bartlett',B=9L,boot_seed=123L)
  bv <- be$evaluate(b); bv2 <- be$evaluate(b,refresh=TRUE)
  check('bootstrap_candidate_reproducible',bv$statistic-bv2$statistic,1e-10)
  check('bootstrap_mean_identity',bv$statistic-bv$lr/bv$correction,1e-12)
  # Low B here validates mechanics only, never calibration.
  out <- do.call(rbind,checks)
  dir.create(output,recursive=TRUE,showWarnings=FALSE)
  write.csv(out,file.path(output,'validation.csv'),row.names=FALSE)
  print(out,row.names=FALSE)
  if(!all(out$passed)) stop('validation failed; do not launch the main simulation')
  invisible(out)
}
