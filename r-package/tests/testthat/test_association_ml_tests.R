test_that("association ML spectral laws reconstruct in the lab and preserve policy guards", {
  set.seed(3233)
  n <- 500L
  f <- rnorm(n)
  d <- as.data.frame(sapply(c(.8,.73,.66,.59), function(l) {
    z <- l*f+sqrt(1-l*l)*rnorm(n)
    1L+(z > -.45)+(z > .55)
  }))
  names(d) <- paste0("x",1:4)
  for (grouped in c(FALSE,TRUE)) {
    d$g <- rep(c("a","b"),each=n/2)
    fit <- function(model) fit_model(model,d,estimator="ML",ordered=paste0("x",1:4),
                                     groups=if(grouped) "g" else NULL)
    alt <- fit("f =~ x1 + x2 + x3 + x4")
    null <- fit(if(grouped) "f =~ x1 + l2*x2 + x3 + x4" else "f =~ x1 + l*x2 + l*x3 + x4")
    stats <- alt$ordinal_stats
    ij <- association_ml_ij(alt,stats)
    global <- association_ml_global_test(alt,stats)
    expect_true(global$available,info=global$detail)
    if (!isTRUE(global$available)) next
    expect_equal(global$statistic,2*n*ij$value,tolerance=1e-10)
    reconstructed <- sort(Re(eigen(global$U %*% global$Gamma)$values),decreasing=TRUE)[seq_len(global$df)]
    expect_equal(sort(global$spectrum,decreasing=TRUE),reconstructed,tolerance=1e-10)
    for (r in c("All","SB","PEBA4")) {
      expect_equal(global[[r]]$p_value,
                   infer_fmg_test(global$statistic,global$df,global$spectrum,
                                  method=switch(r,All="all",SB="sb",PEBA4="peba"))$p_value,
                   tolerance=1e-12)
      expect_equal(sum(global[[r]]$lambdas_reference),sum(global$spectrum),tolerance=1e-12)
    }
    expect_equal(global$U,global$V-global$V%*%global$Delta%*%
      solve(crossprod(global$Delta,global$V%*%global$Delta))%*%t(global$Delta)%*%global$V,tolerance=1e-12)
    nested <- association_ml_nested_test(alt,null,stats)
    expect_true(nested$available,info=nested$detail)
    if (!isTRUE(nested$available)) next
    expect_equal(nested$df,1L)
    Hinv <- solve(ij$H)
    C <- nested$A %*% Hinv %*% t(nested$A)
    S <- nested$A %*% Hinv %*% ij$B %*% Hinv %*% t(nested$A)
    expect_equal(nested$C,C,tolerance=1e-10)
    expect_equal(nested$S,S,tolerance=1e-10)
    expect_equal(as.numeric(nested$spectrum),as.numeric(S/C),tolerance=1e-10)
    for (r in c("All","SB","PEBA4")) {
      expect_equal(nested[[r]]$p_value,
                   infer_fmg_test(nested$statistic,1L,nested$spectrum,
                                  method=switch(r,All="all",SB="sb",PEBA4="peba"))$p_value,
                   tolerance=1e-12)
    }
    zero <- association_ml_nested_test(alt,alt,stats)
    expect_false(zero$available)
    expect_identical(zero$reason,"zero_df")
    expect_equal(zero$statistic,0)
    expect_true(is.nan(zero$All$p_value))
    expect_error(vcov(alt),"sampling/inference contract")
    untagged <- alt; untagged$association <- NULL
    expect_identical(association_ml_global_test(untagged,stats)$reason,"unsupported_estimator")
    expect_identical(association_ml_nested_test(alt,untagged,stats)$reason,"unsupported_estimator")
    penalized <- alt; penalized$penalty <- list()
    expect_identical(association_ml_global_test(penalized,stats)$reason,"penalty")
    expect_identical(association_ml_nested_test(alt,penalized,stats)$reason,"penalty")
    other <- stats; other$int_data[[1]][1,1] <- (other$int_data[[1]][1,1]+1L) %% 3L
    expect_identical(association_ml_nested_test(alt,null,other)$reason,"incompatible_stage1")
    marker <- fit("g =~ x1 + x2 + x3 + x4")
    moment <- association_ml_nested_test(alt,marker,stats)
    expect_false(moment$available)
    expect_true(moment$reason %in% c("not_nested","unsupported_nesting"))
    missing <- stats; missing$int_data[[1]][1,1] <- -1L
    expect_false(association_ml_global_test(alt,missing)$available)
  }
})
