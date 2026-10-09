test_that('continuous Wald intervals gate casewise likelihood and corrected points', {
  set.seed(142)
  n <- 500L; f <- rnorm(n); extra <- rnorm(n)
  d0 <- as.data.frame(sapply(1:4,function(j) .7*f+rnorm(n)+if(j<=2) extra else 0))
  names(d0) <- paste0('x',1:4)
  for(groups in 1:2) for(route in c('covariance','means','fiml')) {
    d <- d0
    if(route=='fiml') d$x2[seq(2,n,by=7)] <- NA_real_
    if(groups==2) d <- rbind(transform(d,group='A'),transform(d,group='B'))
    spec <- model_spec('f =~ x1+x2+x3+x4',meanstructure=route!='covariance',
      group=if(groups==2) 'group' else NULL,group_labels=if(groups==2) c('A','B') else NULL)
    fit <- fit_model(spec,d,estimator=if(route=='fiml') 'FIML' else 'ML')
    out <- fit_index_intervals_misspec(fit)
    p <- attr(policy_fit_measures(fit),'details')
    info <- attr(out,'details')
    expect_equal(nrow(out),6L)
    expect_true(all(is.na(out$reason)))
    expect_equal(info$F_hat,p$user$discrepancy,tolerance=1e-10)
    expect_equal(info$F_tilde,p$user$corrected,tolerance=1e-12)
    expect_equal(info$F_tilde,max(info$F_hat-p$user$trace/info$ntotal,0),tolerance=1e-10)
    expect_true(all(out$lower<=out$estimate & out$upper>=out$estimate))
    expect_true(all(out$V1>=0 & out$V2>=0))
    row <- out[out$index=='rmsea' & out$variant=='wald_f',]
    width <- qnorm(.95)*sqrt(row$V1+row$V2)
    expect_equal(row$lower,sqrt(groups*max(info$F_tilde-width,0)/p$user$df),tolerance=1e-12)
    expect_equal(row$upper,sqrt(groups*(info$F_tilde+width)/p$user$df),tolerance=1e-12)
    root_row <- out[out$index=='rmsea' & out$variant=='wald_sqrt',]
    root_width <- width/(2*sqrt(info$F_tilde))
    expect_equal(root_row$lower,sqrt(groups/p$user$df)*max(sqrt(info$F_tilde)-root_width,0),tolerance=1e-12)
    expect_equal(root_row$upper,sqrt(groups/p$user$df)*(sqrt(info$F_tilde)+root_width),tolerance=1e-12)
    if(route!='fiml') {
      # Independent normal likelihood rows, not the candidate's casewise helper.
      im <- model_implied(fit)
      contrast <- unlist(lapply(seq_len(groups),function(g) {
        x <- as.matrix(if(groups==1) d else d[d$group==c('A','B')[g],1:4])
        mu <- colMeans(x); xc <- sweep(x,2,mu); s <- crossprod(xc)/nrow(x)
        sm <- im$sigma[[g]]
        xm <- if(route=='covariance') xc else sweep(x,2,im$mu[[g]])
        .5*(as.numeric(determinant(sm,logarithm=TRUE)$modulus-determinant(s,logarithm=TRUE)$modulus)+
          rowSums((xm%*%solve(sm))*xm)-rowSums((xc%*%solve(s))*xc))
      }))
      expect_equal(row$V1,4*var(contrast)/info$ntotal,tolerance=1e-10)
      baseline_contrast <- unlist(lapply(seq_len(groups),function(g) {
        x <- as.matrix(if(groups==1) d else d[d$group==c('A','B')[g],1:4])
        xc <- sweep(x,2,colMeans(x)); s <- crossprod(xc)/nrow(x)
        sb <- diag(diag(s))
        .5*(as.numeric(determinant(sb,logarithm=TRUE)$modulus-determinant(s,logarithm=TRUE)$modulus)+
          rowSums((xc%*%solve(sb))*xc)-rowSums((xc%*%solve(s))*xc))
      }))
      vb <- 4*var(baseline_contrast)/info$ntotal
      cross <- 4*cov(contrast,baseline_contrast)/info$ntotal
      for(name in c('cfi','tli')) {
        incremental <- out[out$index==name & out$variant=='wald_f',]
        factor <- if(name=='tli') p$baseline$df/p$user$df else 1
        gu <- -factor/p$baseline$corrected
        gb <- factor*info$F_tilde/p$baseline$corrected^2
        width_index <- qnorm(.95)*sqrt(gu^2*(row$V1+row$V2)+gb^2*vb+2*gu*gb*cross)
        bounds <- incremental$estimate+c(-1,1)*width_index
        if(name=='cfi') bounds <- pmax(pmin(bounds,1),0)
        expect_equal(c(incremental$lower,incremental$upper),bounds,tolerance=1e-10)
      }
    }
  }
})

test_that('interval refusals and the sqrt boundary are typed', {
  set.seed(11)
  n <- 240L; f <- rnorm(n)
  d <- as.data.frame(sapply(1:4,function(j) .7*f+rnorm(n)))
  names(d) <- paste0('x',1:4)
  syntax <- 'f =~ x1+x2+x3+x4'
  for(estimator in c('ULS','ML2S')) {
    out <- fit_index_intervals_misspec(fit_model(syntax,d,estimator=estimator))
    expect_true(all(grepl('^unsupported_model:',out$reason)))
    expect_true(all(is.na(out$lower)))
  }
  ordinal <- d
  ordinal[] <- lapply(d,function(x) as.integer(cut(x,c(-Inf,-.4,.6,Inf))))
  fit <- fit_model(model_spec(syntax,ordered=names(d)),ordinal,estimator='DWLS')
  expect_true(all(grepl('^unsupported_model:',fit_index_intervals_misspec(fit)$reason)))
  fit <- fit_model(syntax,d)
  fit$converged <- FALSE
  expect_true(all(grepl('^not_converged:',fit_index_intervals_misspec(fit)$reason)))
  fit$penalty_inference <- 'not_validated'
  expect_true(all(grepl('^penalized:',fit_index_intervals_misspec(fit)$reason)))
  expect_error(fit_index_intervals_misspec(fit,0),'strictly between')
  expect_error(fit_index_intervals_misspec(fit,NA_real_),'strictly between')

  x <- scale(as.matrix(d),center=TRUE,scale=FALSE)
  x <- x%*%solve(chol(crossprod(x)/n))
  sigma <- matrix(.49,4,4); diag(sigma)<-1
  exact <- as.data.frame(x%*%chol(sigma)); names(exact)<-names(d)
  out <- fit_index_intervals_misspec(fit_model(syntax,exact))
  expect_true(all(grepl('^inapplicable:',out$reason[out$variant=='wald_sqrt'])))
  expect_true(all(is.na(out$reason[out$variant=='wald_f'])))
  expect_lt(out$V1[1],1e-15)
  expect_gt(out$V2[1],0)
})
