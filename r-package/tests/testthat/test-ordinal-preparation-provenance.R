test_that("released response scales retain preparation through post-fit reconstruction", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ordered <- paste0("x", 1:4)
  for (v in ordered) d[[v]] <- as.integer(cut(d[[v]], 3))
  syntax <- "f =~ x1+x2+x3+x4\nx1 ~*~ c(1, NA)*x1"
  fit_one <- function(extra = "") {
    spec <- model_spec(paste(syntax, extra, sep = "\n"), ordered = ordered,
      parameterization = "delta", group = "school",
      group_labels = unique(as.character(d$school)), group_equal = "thresholds")
    fit_model(spec, d, estimator = "DWLS")
  }
  fit <- fit_one()
  expect_true(fit$converged)
  stamp <- attr(fit$partable, "magmaan.ordinal_preparation")
  expect_identical(stamp, rep(list(rep(2L, 4)), 2))
  n <- max(fit$partable$free)
  released <- with(fit$partable, op == "~~" & lhs == "x1" & rhs == "x1" & group == 2)
  expect_true(fit$partable$free[released] > 0L)
  robust <- robust_ordinal(fit, fit$ordinal_stats)
  ij <- vcov(fit, regime = "sandwich_ij")
  expect_equal(nrow(robust$vcov), n)
  expect_equal(nrow(ij), n)
  expect_true(all(is.finite(ij)))
  policy <- policy_inference(fit)
  expect_true(policy$covariance_available)
  expect_equal(nrow(policy$covariance), n)
  null <- fit_one("x2 ~~ c(1, 1)*x2")
  expect_true(null$converged)
  nested <- robust_nested_lrt(fit, null, data = fit$ordinal_stats, A.method = "delta", method = "restriction_map")
  expect_equal(nested$df_diff, 1)
  expect_true(is.finite(nested$T_scaled))
  expect_identical(attr(fit$partable, "magmaan.ordinal_preparation"), stamp)
})

test_that("explicit grouped theta residuals survive lab preparation", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ordered <- paste0("x", 1:4)
  for (v in ordered) d[[v]] <- as.integer(cut(d[[v]], 3))
  syntax <- paste(c(
    "f =~ c(1,1)*x1+c(l2,l2)*x2+c(l3,l3)*x3+c(l4,l4)*x4",
    "f ~~ c(NA,NA)*f", "f ~ c(0,NA)*1",
    vapply(seq_len(4), function(j) paste0(
      "x",j," | c(t",j,"1,t",j,"1)*t1+c(t",j,"2,t",j,"2)*t2\n",
      "x",j," ~ c(0,0)*1\nx",j," ~~ c(1,NA)*x",j), character(1))), collapse="\n")
  spec <- model_spec(syntax, ordered=ordered, parameterization="theta",
    group="school", group_labels=unique(as.character(d$school)),
    auto_var=FALSE, auto_fix_first=FALSE, auto_cov_lv_x=FALSE,
    auto_cov_y=FALSE, meanstructure=TRUE)
  actual <- fit_model(spec,d,estimator="DWLS",
    options=list(preset="lavaan-0.7.2"))
  oracle <- lavaan::lavaan(syntax,data=d,ordered=ordered,parameterization="theta",
    group="school", group.label=unique(as.character(d$school)),
    estimator="WLSMV", auto.var=FALSE,auto.fix.first=FALSE,
    auto.cov.lv.x=FALSE,auto.cov.y=FALSE,meanstructure=TRUE)
  expect_true(actual$converged)
  expect_true(lavaan::lavInspect(oracle,"converged"))
  mp <- actual$partable
  lp <- lavaan::parTable(oracle)
  key <- function(pt) paste(pt$lhs,pt$op,pt$rhs,pt$group)
  index <- match(key(mp),key(lp))
  use <- mp$op %in% c("=~","|","~1","~~") & !is.na(index)
  expect_equal(mp$est[use],lp$est[index[use]],tolerance=1e-5)
  residuals <- with(mp, group==2 & op=="~~" & lhs==rhs & lhs %in% ordered)
  expect_equal(sum(mp$free[residuals]>0L),4L)
})
