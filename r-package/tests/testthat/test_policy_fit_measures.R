policy_value <- function(out, name) out$estimate[match(name, out$index)]
policy_data <- function(n = 256L) {
  set.seed(101)
  f <- rnorm(n); nuisance <- rnorm(n)
  d <- as.data.frame(sapply(1:4, function(j) .7*f + (if(j<3) .4*nuisance else 0) + rnorm(n)))
  names(d) <- paste0('x',1:4)
  d
}
policy_cfa <- 'f =~ x1+x2+x3+x4'

test_that('policy fit measures reduce to robust MLM at exact fit', {
  skip_if_not_installed('lavaan')
  set.seed(101)
  n <- 128L
  z <- scale(matrix(rnorm(n*4),n,4),center=TRUE,scale=FALSE)
  z <- z %*% solve(chol(crossprod(z)/n))
  sigma <- matrix(.49,4,4); diag(sigma) <- 1
  d <- as.data.frame(z %*% chol(sigma)); names(d) <- paste0('x',1:4)
  fit <- fit_model(policy_cfa,d)
  got <- policy_fit_measures(fit)
  expect_true(all(is.na(got$reason)))
  ref <- lavaan::cfa(policy_cfa,d,estimator='MLM')
  fm <- lavaan::fitMeasures(ref)
  details <- attr(got,'details')
  expect_equal(details$user$trace,unname(fm['chisq.scaling.factor']*fm['df']),tolerance=1e-8)
  # Independent exact-fit profile Hessian in saturated covariance moments:
  # Q = W - W D (D' W D)^-1 D' W. Construct D directly from the
  # one-factor covariance formula, without the model evaluator or policy bread.
  positions <- which(lower.tri(sigma,diag=TRUE),arr.ind=TRUE)
  duplication <- matrix(0,16,10)
  tangent <- matrix(0,10,8)
  moment_rows <- matrix(0,n,10)
  for (r in seq_len(10)) {
    i <- positions[r,1]; j <- positions[r,2]
    duplication[i+4*(j-1),r] <- 1
    duplication[j+4*(i-1),r] <- 1
    tangent[r,i] <- tangent[r,i]+.7
    tangent[r,j] <- tangent[r,j]+.7
    if(i==j) tangent[r,4+i] <- 1
    moment_rows[,r] <- as.matrix(d)[,i]*as.matrix(d)[,j]-sigma[i,j]
  }
  weight <- .5*t(duplication)%*%kronecker(solve(sigma),solve(sigma))%*%duplication
  profile <- weight-weight%*%tangent%*%solve(t(tangent)%*%weight%*%tangent)%*%t(tangent)%*%weight
  plain_trace <- sum(diag(profile%*%(crossprod(moment_rows)/n)))
  expect_equal(details$user$trace,plain_trace,tolerance=1e-6)

  for (name in c('rmsea','cfi')) expect_equal(policy_value(got,name),unname(fm[paste0(name,'.robust')]),tolerance=1e-10)
  expect_equal(policy_value(got,'tli'),min(unname(fm['tli.robust']),1),tolerance=1e-10)
})

test_that('policy likelihood accounting and multigroup pooling preserve their contracts', {
  skip_if_not_installed('lavaan')
  d <- policy_data()
  for (estimator in c('ML','FIML','ULS')) {
    fit <- fit_model(policy_cfa,d,estimator=estimator)
    got <- policy_fit_measures(fit)
    expect_true(all(is.na(got$reason)))
    df <- rbind(transform(d,group='A'),transform(d,group='B'))
    spec <- model_spec(policy_cfa,group='group',group_labels=c('A','B'),meanstructure=estimator=='FIML')
    grouped <- policy_fit_measures(fit_model(spec,df,estimator=estimator))
    for (name in c('rmsea','cfi','tli','srmr')) expect_equal(policy_value(got,name),policy_value(grouped,name),tolerance=1e-6)
    if (estimator!='ULS') {
      ref <- lavaan::cfa(policy_cfa,d,missing=if(estimator=='FIML') 'ml' else 'listwise',meanstructure=estimator=='FIML')
      fm <- lavaan::fitMeasures(ref)
      for (name in c('logl','unrestricted.logl','aic','bic')) expect_equal(policy_value(got,name),unname(fm[name]),tolerance=1e-7)
      expect_equal(attr(got,'details')$srmr_uncorrected,unname(fm['srmr']),tolerance=1e-7)
    }
  }
})

test_that('policy fit measure reasons are typed for failed and penalized fits', {
  fit <- fit_model(policy_cfa,policy_data())
  fit$converged <- FALSE
  got <- policy_fit_measures(fit)
  expect_true(all(is.na(got$estimate)))
  expect_true(all(grepl('^not_converged:',got$reason)))
  fit$penalty_inference <- 'not_validated'
  got <- policy_fit_measures(fit)
  expect_true(all(grepl('^penalized:',got$reason)))
  unsupported <- fit_model(policy_cfa,policy_data(),estimator='GLS')
  got <- policy_fit_measures(unsupported)
  expect_true(all(grepl('^unsupported_model:',got$reason)))
})

test_that('the Exact comparator preserves OPG lab defaults', {
  d <- policy_data(240L)
  for (v in names(d)) d[[v]] <- as.integer(cut(d[[v]],c(-Inf,-.4,.6,Inf)))
  spec <- model_spec(policy_cfa,ordered=names(d))
  fit <- fit_model(spec,d,estimator='DWLS')
  default <- fit_measures_misspec(fit,fit$ordinal_stats)
  opg <- fit_measures_misspec(fit,fit$ordinal_stats,first_stage='opg')
  expect_identical(default,opg)
  exact <- fit_measures_misspec(fit,fit$ordinal_stats,first_stage='exact')
  got <- policy_fit_measures(fit)
  expect_equal(policy_value(got,'rmsea'),exact$rmsea,tolerance=0)
  expect_equal(policy_value(got,'cfi'),exact$cfi,tolerance=0)
})

# Independent delete-one refits validate the entire residual influence chain,
# including its observed-standardization denominator and FIML EM-moment map.
test_that('ML and FIML residual trace agrees with delete-one jackknife', {
  d <- policy_data(384L)
  for (estimator in c('ML','FIML')) {
    data <- d
    if (estimator=='FIML') {
      data$x2[seq(2,nrow(data),by=7)] <- NA
      data$x4[seq(3,nrow(data),by=9)] <- NA
    }
    residual <- function(fit, observed) {
      im <- model_implied(fit)
      if (estimator=='FIML') {
        sat <- saturated_em_moments_impl(fit$raw_data)
        s <- sat$cov[[1]]; m <- sat$mean[[1]]
      } else {
        x <- as.matrix(observed)
        s <- cov(x)*(nrow(x)-1)/nrow(x); m <- colMeans(x)
      }
      r <- (s-im$sigma[[1]])/sqrt(outer(diag(s),diag(s)))
      out <- r[lower.tri(r,diag=TRUE)]
      if (estimator=='FIML') out <- c(out,(m-im$mu[[1]])/sqrt(diag(s)))
      out
    }
    fit <- fit_model(policy_cfa,data,estimator=estimator)
    got <- policy_fit_measures(fit)
    expect_true(all(is.na(got$reason)))
    deleted <- vapply(seq_len(nrow(data)),function(i) residual(fit_model(policy_cfa,data[-i,],estimator=estimator),data[-i,]), numeric(if(estimator=='FIML') 14L else 10L))
    centered <- deleted-rowMeans(deleted)
    jackknife <- (nrow(data)-1)*sum(centered^2)/nrow(deleted)
    trace <- attr(got,'details')$srmr_trace
    cat(sprintf('\n%s residual trace %.12g jackknife %.12g relative gap %.6g\n',estimator,trace,jackknife,abs(trace-jackknife)/jackknife))
    expect_true(abs(trace-jackknife)/jackknife < .05,info=paste(estimator,'trace',trace,'jackknife',jackknife))
  }
})

test_that('categorical policy pooling and Exact lab comparators agree across groups', {
  for (mixed in c(FALSE,TRUE)) {
    d <- policy_data(240L)
    ordered <- if (mixed) names(d)[1:2] else names(d)
    for (v in ordered) d[[v]] <- as.integer(cut(d[[v]],c(-Inf,-.4,.6,Inf)))
    spec <- model_spec(policy_cfa,ordered=ordered,meanstructure=mixed)
    fit <- fit_model(spec,d,estimator='DWLS')
    got <- policy_fit_measures(fit)
    df <- rbind(transform(d,group='A'),transform(d,group='B'))
    group_spec <- model_spec(policy_cfa,ordered=ordered,meanstructure=mixed,group='group',group_labels=c('A','B'))
    group_fit <- fit_model(group_spec,df,estimator='DWLS')
    pooled <- policy_fit_measures(group_fit)
    expect_true(all(is.na(got$reason)))
    expect_true(all(is.na(pooled$reason)))
    for (name in c('rmsea','cfi','tli','srmr','crmr')) expect_equal(policy_value(got,name),policy_value(pooled,name),tolerance=1e-7)
    if (mixed) {
      opg <- fit_measures_misspec_mixed_ordinal(fit,fit$mixed_ordinal_stats)
      expect_identical(opg,fit_measures_misspec_mixed_ordinal(fit,fit$mixed_ordinal_stats,first_stage='opg'))
      exact <- fit_measures_misspec_mixed_ordinal(fit,fit$mixed_ordinal_stats,first_stage='exact')
      expect_equal(policy_value(got,'rmsea'),exact$rmsea,tolerance=0)
      expect_equal(policy_value(got,'cfi'),exact$cfi,tolerance=0)
    }
  }
})

test_that('ML2S-NT composes its profile while reporting the missing residual adapter', {
  d <- policy_data()
  d$x2[seq(2,nrow(d),by=7)] <- NA
  got <- policy_fit_measures(fit_model(policy_cfa,d,estimator='ML2S'))
  expect_true(all(is.finite(got$estimate[got$index %in% c('rmsea','cfi','tli')])))
  expect_match(got$reason[got$index=='srmr'],'ML2S Stage-2 residual influence')
})
