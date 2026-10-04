test_that('ordinary ordinal nonlinear fits retain lavaan compatibility and refuse the native policy', {
  skip_if_not_installed('lavaan')
  set.seed(542074)
  eta<-rnorm(600)
  d<-as.data.frame(sapply(c(.85,.75,.7,.65),function(l) as.integer(cut(l*eta+rnorm(600,sd=.8),c(-Inf,-.5,.5,Inf)))))
  names(d)<-paste0('y',1:4)
  syntax<-'f =~ y1+a*y2+b*y3+y4\na == b^2'
  for (par in c('delta','theta')) {
    m<-magmaan_model(syntax,d,ordered=names(d),parameterization=par)
    fit<-magmaan(m,d,estimator='DWLS',inference=FALSE)
    lv<-lavaan::cfa(syntax,d,ordered=names(d),parameterization=par,estimator='WLSMV')
    lab<-as_lab_fit(fit); rows<-lab$partable[lab$partable$free>0,]; rows<-rows[order(rows$free),]
    key<-function(pt) paste(pt$lhs,pt$op,pt$rhs,pt$group)
    reference<-lavaan::parTable(lv); hit<-match(key(rows),key(reference))
    expect_equal(as.numeric(lab$theta),reference$est[hit],tolerance=1e-5)
    expect_equal(as.numeric(sqrt(diag(vcov(fit,lavaan_compat='WLSMV')))),reference$se[hit],tolerance=1e-5)
    reported<-summary(fit,lavaan_compat='WLSMV')$tests
    expect_equal(reported$df,as.numeric(lavaan::fitMeasures(lv,'df')))
    expect_equal(reported$statistic,as.numeric(lavaan::fitMeasures(lv,'chisq.scaled')),tolerance=1e-5)
    policy<-magmaanlab::policy_inference(as_lab_fit(fit))
    expect_identical(policy$covariance_reason,'unsupported_model')
    expect_match(policy$covariance_detail,'Lagrangian curvature')
  }
})
