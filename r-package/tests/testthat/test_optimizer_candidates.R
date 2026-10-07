library(magmaanlab)

testthat::test_that('budget candidates preserve controls and fail independent audits', {
  set.seed(865081001)
  x <- outer(rnorm(140),c(1,.8,.7,.9))+matrix(rnorm(560),140,4)
  colnames(x) <- paste0('x',1:4)
  spec <- model_spec('f =~ x1 + x2 + x3 + x4',meanstructure=TRUE,fixed_x=FALSE)
  ctl <- list(nlopt=list(max_eval=1,ftol_rel=1e-13,xtol_rel=1e-11))
  for(method in c('ML','ULS')) {
    fit <- suppressWarnings(fit_model(spec,as.data.frame(x),estimator=method,control=ctl))
    testthat::expect_false(fit$converged)
    testthat::expect_identical(fit$optimizer_status,'budget_exhausted')
    testthat::expect_identical(fit$audit$raw_backend_status,5L)
    testthat::expect_identical(fit$audit$nlopt_controls$max_eval,1L)
    testthat::expect_equal(fit$audit$nlopt_controls$ftol_rel,1e-13,tolerance=0)
    testthat::expect_equal(fit$audit$nlopt_controls$xtol_rel,1e-11,tolerance=0)
    testthat::expect_true(all(is.finite(fit$theta)))
    testthat::expect_true(fit$audit$f_consistent)
    evaluated <- magmaan_core$estimate_evaluate_at(fit$partable,
      list(S=fit$S,nobs=fit$nobs,mean=fit$sample_mean),fit$theta,estimator=method,
      audit_options=list(retain_newton_artifacts=TRUE))
    testthat::expect_false(evaluated$newton_audit$diagnostics$passed)
  }
  mask <- matrix(runif(length(x))>.25,nrow(x),4);mask[rowSums(mask)==0,1] <- TRUE
  fit <- suppressWarnings(magmaan_core$fit_fiml(spec,list(X=list(x),mask=list(mask)),control=ctl))
  testthat::expect_false(fit$converged)
  testthat::expect_identical(fit$f_evals,2L)
  testthat::expect_identical(fit$audit$raw_backend_status,5L)
  audit <- magmaan_core$frontier_fiml_newton_audit(fit)
  testthat::expect_false(audit$diagnostics$passed)
  testthat::expect_true(all(is.finite(audit$gradient)))
})

testthat::test_that('pinned equality loadings fit the explicitly fixed model', {
  set.seed(865081002)
  x <- matrix(rnorm(480,sd=.7),160,3)+rnorm(160)
  colnames(x) <- paste0('x',1:3)
  syntax <- 'f =~ NA*a*x1 + b*x2 + c*x3\na+b+c == 3\na == b\nb == c'
  equal <- suppressWarnings(fit_model(syntax,as.data.frame(x)))
  fixed <- suppressWarnings(fit_model('f =~ 1*x1 + 1*x2 + 1*x3',as.data.frame(x)))
  testthat::expect_true(equal$converged)
  testthat::expect_equal(equal$fmin,fixed$fmin,tolerance=1e-9)
  testthat::expect_equal(magmaan_core$model_implied(equal)$sigma,magmaan_core$model_implied(fixed)$sigma,tolerance=1e-7)
})

testthat::test_that('fitted sample means remain owned across GC and serialization', {
  set.seed(865081003)
  x <- matrix(rnorm(560),140,4)+rnorm(140)
  x <- sweep(x,2,c(2,-1,.5,4),'+');colnames(x) <- paste0('x',1:4)
  spec <- model_spec('f =~ x1 + x2 + x3 + x4',meanstructure=TRUE,fixed_x=FALSE)
  invisible(fit_model(spec,as.data.frame(x))) # load wrapper paths before torture
  checked <- function() {
    prior <- gctorture2(1000)
    on.exit(gctorture2(prior),add=TRUE)
    fit <- fit_model(spec,as.data.frame(x))
    bytes <- serialize(fit,NULL)
    invisible(gc())
    unserialize(bytes)
  }
  fit <- checked()
  testthat::expect_equal(as.numeric(fit$sample_mean[[1]]),as.numeric(colMeans(x)),tolerance=1e-12)
  testthat::expect_true(fit$converged)
  testthat::expect_identical(fit$audit$nlopt_controls$max_eval,5000L)
})

testthat::test_that('failed ML2S candidates return before automatic inference', {
  set.seed(865081004)
  x <- outer(rnorm(130),c(1,.8,.7,.9))+matrix(rnorm(520),130,4)
  colnames(x) <- paste0('x',1:4)
  mask <- matrix(runif(520)>.25,130,4);mask[rowSums(mask)==0,1] <- TRUE
  spec <- model_spec('f =~ x1 + x2 + x3 + x4',meanstructure=TRUE,fixed_x=FALSE)
  fit <- suppressWarnings(magmaan_core$fit_ml2s(spec,list(X=list(x),mask=list(mask)),
    stage2_weight='uls',control=list(nlopt=list(max_eval=1))))
  testthat::expect_false(fit$converged)
  testthat::expect_null(fit$ml2s)
  testthat::expect_null(fit$vcov)
  audit <- magmaan_core$frontier_ml2s_convergence_audit(fit)
  testthat::expect_identical(audit$stage1_assessment$status,'passed')
  testthat::expect_identical(audit$handoff$status,'passed')
  testthat::expect_false(identical(audit$status,'passed'))
})
