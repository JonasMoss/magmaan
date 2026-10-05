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
  released <- with(fit$partable, op == "~*~" & lhs == "x1" & rhs == "x1" & group == 2)
  expect_true(fit$partable$free[released] > 0L)
  robust <- robust_ordinal(fit, fit$ordinal_stats)
  ij <- vcov(fit, regime = "sandwich_ij")
  expect_equal(nrow(robust$vcov), n)
  expect_equal(nrow(ij), n)
  expect_true(all(is.finite(ij)))
  policy <- policy_inference(fit)
  expect_true(policy$covariance_available)
  expect_equal(nrow(policy$covariance), n)
  null <- fit_one("x2 ~*~ c(1, 1)*x2")
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


test_that("live DELTA scale restrictions and invariance match lavaan", {
  skip_if_not_installed("lavaan")
  skip_if_not_installed("jsonlite")
  # Keep the live-oracle model declarations in the source package as well as the checkout.
  fixture <- test_path("fixtures", "delta_scale_models.json")
  reference <- jsonlite::fromJSON(fixture, simplifyVector=FALSE)
  set.seed(531072)
  n <- 600L
  eta <- matrix(rnorm(2*n),n,2)
  eta[,2] <- .4*eta[,1]+sqrt(.84)*eta[,2]
  x <- cbind(outer(eta[,1],c(.8,.7,.65)),outer(eta[,2],c(.85,.75,.6))) + matrix(rnorm(6*n,sd=.7),n,6)
  d <- as.data.frame(apply(x,2,function(z) as.integer(cut(z,c(-Inf,-.5,.5,Inf)))))
  ordered <- paste0("u",1:6)
  names(d) <- ordered
  d$g <- rep(1:2,each=n/2)
  for (id in names(reference)) {
    case <- reference[[id]]
    grouped <- id %in% c("threshold_loading_invariance", "scalar_groups")
    automatic <- id == "threshold_loading_invariance"
    args <- list(syntax=case$model, ordered=ordered, parameterization="delta",
      meanstructure=TRUE)
    if (grouped) { args$group <- "g"; args$group_labels <- c("1","2") }
    if (length(case$group_equal)) args$group_equal <- unlist(case$group_equal)
    if (!automatic) args <- c(args,list(auto_var=FALSE,auto_fix_first=FALSE,
      auto_cov_lv_x=FALSE,auto_cov_y=FALSE))
    spec <- do.call(model_spec,args)
    fit <- fit_model(spec,d,estimator="DWLS")
    lavaan_args <- args
    names(lavaan_args) <- gsub("_",".",names(lavaan_args),fixed=TRUE)
    names(lavaan_args)[names(lavaan_args)=="syntax"] <- "model"
    names(lavaan_args)[names(lavaan_args)=="group.labels"] <- "group.label"
    oracle <- do.call(if (automatic) lavaan::cfa else lavaan::lavaan,
      c(lavaan_args,list(data=d,estimator="WLSMV")))
    expect_true(fit$converged, info=id)
    expect_true(lavaan::lavInspect(oracle,"converged"), info=id)
    mp <- fit$partable; lp <- lavaan::parTable(oracle)
    key <- function(p) paste(p$lhs,p$op,p$rhs,p$group)
    index <- match(key(mp),key(lp))
    use <- !is.na(index) & mp$group>0
    expect_equal(mp$free[use]>0L,lp$free[index[use]]>0L,info=id)
    expect_equal(mp$est[use],lp$est[index[use]],tolerance=1e-5,info=id)
    free <- use & mp$free>0L
    bundle <- convention_inference(fit,"WLSMV")
    expect_true(bundle$covariance_available,info=id)
    expect_true(bundle$test$available,info=id)
    expect_equal(sqrt(diag(bundle$covariance))[mp$free[free]],lp$se[index[free]],tolerance=1e-5,info=id)
    expect_equal(bundle$test$statistic,unname(lavaan::fitMeasures(oracle,"chisq.scaled")),tolerance=1e-5,info=id)
    expect_equal(bundle$test$df,unname(lavaan::fitMeasures(oracle,"df")),info=id)
    std <- standardized(fit,bundle$covariance,type="all")
    ls <- lavaan::standardizedSolution(oracle,type="std.all")
    si <- match(key(mp),key(ls))
    loadings <- mp$op=="=~" & mp$free>0L & !is.na(si)
    expect_equal(as.numeric(std$theta)[mp$free[loadings]],ls$est.std[si[loadings]],tolerance=1e-5,info=id)
    expect_equal(as.numeric(std$se)[mp$free[loadings]],ls$se[si[loadings]],tolerance=1e-5,info=id)
    if (id %in% c("fixed_nonunit","equal_scales","released_scale")) {
      scores <- factor_scores(fit,d,method="EBM")$scores[[1]]
      reference_scores <- lavaan::lavPredict(oracle,method="EBM")
      expect_equal(unname(scores),unname(reference_scores),tolerance=5e-4,info=id)
    }
  }
})
