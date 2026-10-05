test_that("lab likelihood meat retains joint group-score mean variation", {
  X <- list(a = matrix(rep(c(-1, 1), 50), ncol = 1,
                       dimnames = list(NULL, "y")),
            b = matrix(rep(c(-3, 3), 50), ncol = 1,
                       dimnames = list(NULL, "y")))
  d <- data.frame(y = c(X$a, X$b), g = rep(c("a", "b"), each = 100))
  m <- model_spec("y ~ 1\ny ~~ v*y", meanstructure = TRUE, group = "g", group_labels = c("a","b"))
  f <- fit_model(m, d)
  expect_true(f$converged)
  exact <- magmaan_core$infer_robust_se_raw(f, X)$vcov
  old <- magmaan_core$infer_robust_se_raw(f, X, moments = "structured")$vcov
  vars <- which(f$partable$op == "~~")
  # Each group's variance-score mean is +/- .08. Total observed information
  # is 200/(2*5^2)=4: (1/4)^2 * 200*.08^2=.08. Within-group centered rows
  # omit this joint-sampling term, despite both fitted means being saturated.
  expect_equal(exact[vars, vars], matrix(.08, 2, 2), tolerance = 1e-7)
  expect_equal(old[vars, vars], matrix(0, 2, 2), tolerance = 1e-12)
  expect_equal(exact, policy_inference(f)$covariance, tolerance = 1e-8)
  covariance_rows <- magmaan_core$infer_casewise_contributions(f$partable,X)
  # The public legacy builder is covariance-only. Supply the documented
  # [mean; covariance] group layout explicitly for this mean-structure fit.
  z <- matrix(0,200,4)
  z[,c(2,4)] <- covariance_rows
  z[1:100,1] <- X$a - mean(X$a)
  z[101:200,3] <- X$b - mean(X$b)
  G <- crossprod(z)/200
  expect_equal(magmaan_core$infer_robust_se_zc(f,z,200)$vcov,exact,tolerance=1e-12)
  expect_equal(magmaan_core$infer_robust_se(f,G)$vcov,exact,tolerance=1e-12)
  pair <- magmaan_core$infer_robust_se_both_breads_raw(f,X)
  expect_equal(pair$observed$vcov,exact,tolerance=1e-12)
  expect_error(magmaan_core$infer_robust_se_zc(f,z[-1,],200),"fitting group counts")
  expect_error(magmaan_core$infer_robust_se_zc(f,z,199),"fitting group counts")
  expect_error(magmaan_core$infer_robust_se_raw(f,lapply(X,function(x)x[-1,,drop=FALSE])),
               "fitting counts")

})

test_that("saturated moments and zero projected score means preserve comparators", {
  X <- matrix(c(-2,-1,0,1,2), ncol = 1, dimnames = list(NULL,"y"))
  saturated <- fit_model(model_spec("y ~ 1\ny ~~ y", meanstructure = TRUE), as.data.frame(X))
  get <- function(f, moments) magmaan_core$infer_robust_se_raw(f,X,moments=moments)$vcov
  expect_equal(get(saturated,"likelihood"),get(saturated,"structured"),tolerance=1e-9)
  expect_equal(get(saturated,"structured"),get(saturated,"unstructured"),tolerance=1e-9)
  fixed <- fit_model(model_spec("y ~ 1\ny ~~ 1*y", meanstructure = TRUE), as.data.frame(X))
  expect_equal(get(fixed,"likelihood"),get(fixed,"structured"),tolerance=1e-12)
  expect_gt(abs(get(fixed,"structured")[1,1]-get(fixed,"unstructured")[1,1]),.1)
  expect_error(magmaan_core$infer_build_u_factor(fixed,"expected",moments="likelihood"),
               "SE/score meat")
})

test_that("continuous LS centered meat matches fixed-allocation moment influence", {
  X <- list(matrix(c(-2,-1,1,2),ncol=1,dimnames=list(NULL,"y")),
            matrix(c(0,1,3,4),ncol=1,dimnames=list(NULL,"y")))
  d <- data.frame(y=c(X[[1]],X[[2]]),g=rep(c("a","b"),each=4))
  f <- fit_model(model_spec("y ~ c(m,m)*1\ny ~~ c(v,v)*y",
                           meanstructure=TRUE,group="g",group_labels=c("a","b")),d,estimator="ULS")
  # Differentiate the exact fixed-allocation sample-moment estimator while
  # perturbing one case weight. Linear implied moments make observed and
  # expected LS sensitivity identical; no estimated fitting weight is involved.
  estimate <- function(group, case, eps) {
    moments <- lapply(seq_along(X),function(g) {
      w <- rep(1,4); if(g==group) w[case] <- w[case]+eps
      y <- as.numeric(X[[g]]); mu <- sum(w*y)/sum(w)
      c(mu,sum(w*(y-mu)^2)/sum(w))
    })
    (moments[[1]]+moments[[2]])/2
  }
  h <- 1e-5
  rows <- do.call(rbind,lapply(1:2,function(g) do.call(rbind,lapply(1:4,function(i)
    (estimate(g,i,h)-estimate(g,i,-h))/(2*h)))))
  target <- crossprod(rows)
  ans <- magmaan_core$infer_continuous_ls_robust(f,X,bread="expected",fixed_weight=TRUE)$vcov
  pt <- f$partable
  indices <- c(which(pt$op=="~1")[1],which(pt$op=="~~")[1])
  expect_equal(ans[indices,indices],target,tolerance=1e-7,ignore_attr=TRUE)
  # Empirical LS score meat uses its fitting weight, irrespective of NT metric.
  a <- score_tests_robust(f, data=d, bread="expected", moments="structured",
                         estimated_weight=FALSE)
  b <- score_tests_robust(f, data=d, bread="expected", moments="unstructured",
                         estimated_weight=FALSE)
  expect_gt(nrow(a),0L)
  expect_equal(a,b,tolerance=1e-12)
})
