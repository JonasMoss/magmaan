test_that("auxiliary NEW coordinates preserve fitted moments across continuous routes", {
  set.seed(540075)
  x <- rnorm(500)
  d <- data.frame(y1 = x+rnorm(500), y2 = .6*x+rnorm(500),
                  y3 = .3*x+rnorm(500), y4 = .8*x+rnorm(500))
  base <- paste("DATA: FILE=x;\nVARIABLE: NAMES=y1-y4;",
                "MODEL: f BY y1; f BY y2 (p1);\nf BY y3 (p2); f BY y4 (p3);",sep="\n")
  reference <- mplus_model(base)
  for (restriction in c("NEW(c); p2=p1+c;", "NEW(c); 0=p1-p2-c**3;")) {
    spec <- mplus_model(paste(base, "MODEL CONSTRAINT:", restriction,sep="\n"))
    nonlinear <- grepl("**", restriction, fixed=TRUE)
    for (estimator in c("ML", "GLS", "ULS", "FIML")) {
      data <- d
      if (estimator == "FIML") data$y2[seq(5,500,by=13)] <- NA_real_
      ref <- fit_model(reference, data, estimator=estimator)
      fit <- fit_model(spec, data, estimator=estimator,
                       optimizer="nlopt-slsqp")
      expect_true(fit$converged, info=paste(restriction,estimator))
      keep <- !fit$partable$op %in% c("new","==")
      expect_equal(fit$partable$est[keep], ref$partable$est, tolerance=2e-5)
      if (nonlinear) {
        expect_error(vcov(fit), "nonlinear equality constraints")
        V <- magmaan_core$inference_vcov(
          magmaan_core$inference_information_expected(fit), fit)
      } else {
        V <- vcov(fit)
      }
      expect_true(all(is.finite(V)))
      ids <- fit$partable$free[keep & fit$partable$free > 0]
      ref_ids <- ref$partable$free[ref$partable$free > 0]
      ref_v <- if (nonlinear) magmaan_core$inference_vcov(
        magmaan_core$inference_information_expected(ref), ref) else vcov(ref)
      expect_equal(unname(V[ids,ids]),unname(ref_v[ref_ids,ref_ids]),tolerance=2e-5)
      for (type in c("all", "lv")) {
        std <- standardized(fit, V, type=type)
        expect_length(std$theta,length(fit$theta))
        expect_true(all(is.finite(std$theta)))
        ref_std <- standardized(ref, ref_v, type=type)
        expect_equal(std$theta[ids],ref_std$theta[ref_ids],tolerance=2e-5)
      }
      policy <- policy_inference(fit)
      expect_type(policy,"list")
    }
    for (covariance in c("psd", "barrier")) {
      ref <- fit_model(reference,d,covariance=covariance)
      fit <- fit_model(spec,d,covariance=covariance,optimizer="nlopt-slsqp")
      expect_true(fit$converged)
      expect_equal(fit$partable$est[!fit$partable$op %in% c("new","==")],ref$partable$est,tolerance=2e-5)
    }
    for (fit_fun in list(fit_noniterative_cfa, fit_noniterative_cfa_metric,
                         fit_noniterative_cfa_restricted)) {
      expect_error(fit_fun(spec$partable, list(S=cov(d),nobs=nrow(d))),
                   "auxiliary NEW coordinates")
    }
  }
})

test_that("DWLS linear NEW coordinates retain ordinal moments and inference", {
  set.seed(540076)
  f <- rnorm(600)
  d <- as.data.frame(sapply(c(1,.8,.7,.9),function(a)
    ordered(cut(a*f+rnorm(600),c(-Inf,-.5,.5,Inf),labels=FALSE),levels=1:3)))
  names(d) <- paste0("y",1:4)
  base <- paste("DATA: FILE=x;\nVARIABLE: NAMES=y1-y4; CATEGORICAL=y1-y4;",
                "MODEL: f BY y1; f BY y2 (a);\nf BY y3 (b); f BY y4;",sep="\n")
  ref <- fit_model(mplus_model(base),d,estimator="DWLS")
  spec <- mplus_model(paste(base,"MODEL CONSTRAINT: NEW(c); b=a+c;",sep="\n"))
  fit <- fit_model(spec,d,estimator="DWLS")
  expect_true(fit$converged)
  expect_equal(fit$partable$est[!fit$partable$op %in% c("new","==")],ref$partable$est,tolerance=2e-5)
  V <- vcov(fit)
  expect_true(all(is.finite(V)))
  expect_type(standardized(fit,V),"list")
  expect_type(policy_inference(fit),"list")
})
