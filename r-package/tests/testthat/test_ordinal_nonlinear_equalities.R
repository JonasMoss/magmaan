.nonlinear_ordinal_data <- function() {
  set.seed(542072)
  n <- 600L; eta <- rnorm(n)
  d <- as.data.frame(sapply(c(.85,.75,.7,.65), function(l) as.integer(cut(l*eta+rnorm(n,sd=.8),c(-Inf,-.5,.5,Inf)))))
  names(d) <- paste0('y',1:4); d$g <- rep(c('a','b'),each=n/2); d
}
.nonlinear_ordinal_key <- function(pt) paste(pt$lhs,pt$op,pt$rhs,pt$group)

test_that('ordinal nonlinear equalities match live lavaan covariance tests scores and nested tests', {
  skip_if_not_installed('lavaan')
  d <- .nonlinear_ordinal_data()
  for (grouped in c(FALSE,TRUE)) for (par in c('delta','theta')) {
    base <- if (grouped) 'f =~ y1+c(a1,a2)*y2+c(b1,b2)*y3+y4' else 'f =~ y1+a*y2+b*y3+y4'
    model <- paste(base,if(grouped) 'a1 == b2^2' else 'a == b^2',sep='\n')
    group <- if(grouped) 'g' else NULL
    fit <- fit_model(model,d,ordered=paste0('y',1:4),groups=group,parameterization=par,estimator='DWLS')
    ref <- lavaan::cfa(model,d,ordered=paste0('y',1:4),group=group,parameterization=par,estimator='WLSMV')
    expect_true(fit$converged)
    rows <- fit$partable[fit$partable$free>0,]; rows<-rows[order(rows$free),]
    oracle<-lavaan::parTable(ref); hit<-match(.nonlinear_ordinal_key(rows),.nonlinear_ordinal_key(oracle))
    expect_equal(as.numeric(fit$theta),oracle$est[hit],tolerance=1e-5)
    inf<-convention_inference(fit,'WLSMV')
    expect_true(inf$covariance_available,info=inf$covariance_detail)
    expect_equal(sqrt(diag(inf$covariance)),oracle$se[hit],tolerance=1e-5)
    expect_equal(inf$test$df,as.numeric(lavaan::fitMeasures(ref,'df')))
    expect_equal(infer_fit_df_stat(fit),inf$test$df)
    expect_equal(inf$test$statistic,as.numeric(lavaan::fitMeasures(ref,'chisq.scaled')),tolerance=1e-5)
    ref_fitted<-lavaan::cfa(model,d,ordered=paste0('y',1:4),group=group,parameterization=par,estimator='WLSMV',start=lavaan::parTable(ref))
    score<-score_tests(fit,bread='expected',estimated_weight=FALSE,cov='model_implied')
    # Preserve the core N*F/2 score; transport lavaan's N/(N-G) score divisor.
    divisor<-(nrow(d)-if(grouped) 2 else 1)/nrow(d)
    expect_equal(score$mi*divisor^2,lavaan::lavTestScore(ref_fitted)$uni$X2,tolerance=1e-5)
    h1<-fit_model(base,d,ordered=paste0('y',1:4),groups=group,parameterization=par,estimator='DWLS')
    lv1<-lavaan::cfa(base,d,ordered=paste0('y',1:4),group=group,parameterization=par,estimator='WLSMV')
    nested<-convention_nested(h1,fit,'WLSMV')$test
    expected<-lavaan::lavTestLRT(lv1,ref_fitted,method='satorra.2000')
    own_start<-lavaan::parTable(ref)
    own_start$est[hit]<-as.numeric(fit$theta)
    ref_own<-lavaan::cfa(model,d,ordered=paste0('y',1:4),group=group,parameterization=par,estimator='WLSMV',start=own_start)
    own_expected<-lavaan::lavTestLRT(lv1,ref_own,method='satorra.2000')
    expect_equal(own_expected[2,'Chisq diff'],expected[2,'Chisq diff'],tolerance=1e-5)
    expect_true(nested$available,info=nested$detail)
    expect_equal(nested$statistic,expected[2,'Chisq diff'],tolerance=1e-5)
    expect_error(score_tests(fit,bread='observed',estimated_weight=FALSE),'Lagrangian curvature')
    policy<-policy_inference(fit)
    expect_identical(policy$covariance_reason,'unsupported_model')
    expect_match(policy$covariance_detail,'Lagrangian curvature')
  }
})

test_that('ordinal nonlinear configured and categorical Mplus fits preserve restrictions', {
  d <- .nonlinear_ordinal_data()
  fit<-fit_model('f =~ y1+a*y2+b*y3+y4\na == b^2',d,ordered=paste0('y',1:4),estimator='DWLS',options=list(preset='lavaan-0.7.2'))
  expect_true(fit$converged)
  input<-paste('DATA: FILE=synthetic.dat;','VARIABLE: NAMES=y1-y4; CATEGORICAL=y1-y4;',
    'MODEL: f BY y1; f BY y2 (a); f BY y3 (b); f BY y4;',
    'MODEL CONSTRAINT: a=b*b;',sep='\n')
  m<-mplus_model(input)
  mp<-fit_model(m,d,estimator='DWLS')
  expect_true(mp$converged)
  a<-mp$theta[mp$partable$free[mp$partable$label=='a' & mp$partable$free>0]]
  b<-mp$theta[mp$partable$free[mp$partable$label=='b' & mp$partable$free>0]]
  expect_equal(a,b^2,tolerance=1e-8)
})
