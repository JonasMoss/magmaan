test_that("association ML lab IJ retains joint thresholds and policy refusal", {
  set.seed(322)
  n <- 500L
  f <- (rchisq(n,3)-3)/sqrt(6)
  d <- as.data.frame(sapply(c(.8,.73,.66,.59), function(l) {
    z <- l*f+sqrt(1-l*l)*(rchisq(n,5)-5)/sqrt(10)
    1L+(z > -.45)+(z > .55)
  }))
  names(d) <- paste0("x",1:4)
  for (grouped in c(FALSE,TRUE)) {
    d$g <- rep(c("a","b"),each=n/2)
    fit <- fit_model("f =~ x1 + l2*x2 + l3*x3 + x4", d,
                     estimator="ML", ordered=paste0("x",1:4),
                     groups=if(grouped) "g" else NULL)
    ij <- association_ml_ij(fit,fit$ordinal_stats)
    expect_true(ij$available, info=ij$detail)
    if (!isTRUE(ij$available)) next
    expect_equal(ij$value,fit$fmin,tolerance=1e-10)
    expect_equal(ij$vcov,Reduce(`+`,lapply(ij$influence,crossprod))/n^2,tolerance=1e-12)
    expect_equal(ij$vcov_active,Reduce(`+`,lapply(ij$influence_active,crossprod))/n^2,tolerance=1e-12)
    expect_equal(ij$vcov_active,solve(ij$H)%*%ij$B%*%t(solve(ij$H))/n,tolerance=1e-12)
    for (rows in ij$influence) expect_lt(max(abs(colMeans(rows))),1e-9)
    th <- unique(fit$partable$free[fit$partable$op=="|"])
    other <- setdiff(seq_along(fit$theta),th)
    expect_true(all(diag(ij$vcov)[th]>0))
    expect_gt(max(abs(ij$vcov[th,other,drop=FALSE])),1e-7)
    expect_error(vcov(fit),"sampling/inference contract")
    penalized <- fit; penalized$penalty <- list()
    expect_identical(association_ml_ij(penalized,fit$ordinal_stats)$reason,"penalty")
    missing <- fit$ordinal_stats; missing$int_data[[1]][1,1] <- -1L
    expect_false(association_ml_ij(fit,missing)$available)
  }
})
