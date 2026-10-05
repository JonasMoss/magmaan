test_that("lab association MI and equality releases reconstruct and retain refusals", {
  set.seed(3244)
  n <- 500L
  f <- rnorm(n)
  d <- as.data.frame(sapply(c(.8,.73,.66,.59),function(l) {
    z <- l*f+sqrt(1-l*l)*rnorm(n)
    1L+(z > -.45)+(z > .55)
  }))
  names(d) <- paste0("x",1:4)
  d$g <- rep(c("a","b"),each=n/2)
  for (grouped in c(FALSE,TRUE)) {
    fit <- function(syntax) fit_model(syntax,d,estimator="ML",ordered=paste0("x",1:4),
                                     groups=if(grouped) "g" else NULL)
    null <- fit("f =~ x1+x2+x3+x4\nx1 ~~ 0*x4")
    mi <- association_ml_modification_indices(null,null$ordinal_stats)
    expect_true(mi$available,info=mi$detail)
    if (!isTRUE(mi$available)) next
    expect_identical(mi$metric,"observed")
    expect_true(any(vapply(mi$rows,function(x) !x$available,logical(1))))
    available <- Filter(function(x) x$available,mi$rows)
    expect_gt(length(available),0)
    for (x in available) {
      v <- x$efficient_direction
      s <- as.numeric(crossprod(v,x$augmented_score))
      h <- as.numeric(crossprod(v,x$H%*%v))
      b <- as.numeric(crossprod(v,x$B%*%v))
      expect_equal(x$statistic,n*s*s/b,tolerance=1e-10)
      expect_equal(x$epc,-s/h,tolerance=1e-10)
      expect_equal(x$p_value,pchisq(x$statistic,1,lower.tail=FALSE),tolerance=1e-12)
      expect_equal(as.numeric(crossprod(x$nuisance,x$H%*%v)),
                   rep(0,ncol(x$nuisance)),tolerance=1e-10)
    }
    eq <- fit(if(grouped) "f =~ x1+l*x2+x3+x4" else "f =~ x1+l*x2+l*x3+x4")
    release <- association_ml_score_tests(eq,eq$ordinal_stats)
    expect_true(release$available,info=release$detail)
    expect_length(release$rows,1)
    expect_true(release$rows[[1]]$available,info=release$rows[[1]]$detail)
    penalized <- null; penalized$penalty <- list()
    expect_identical(association_ml_modification_indices(penalized,null$ordinal_stats)$reason,"penalty")
    expect_identical(association_ml_score_tests(penalized,null$ordinal_stats)$reason,"penalty")
    other <- null$ordinal_stats; other$int_data[[1]][1,1] <- -1L
    expect_identical(association_ml_modification_indices(null,other)$reason,"incompatible_stage1")
    untagged <- null; untagged$association <- NULL
    expect_identical(association_ml_score_tests(untagged,null$ordinal_stats)$reason,"unsupported_estimator")
    expect_error(modification_indices(null),"contract")
    expect_error(vcov(null),"sampling/inference contract")
  }
})
