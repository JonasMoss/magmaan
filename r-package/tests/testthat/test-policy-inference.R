cfa <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"

test_that("policy_inference() equals the explicit composition of primitives", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  fit <- fit_model(cfa, d)
  res <- policy_inference(fit)
  expect_true(res$covariance_available)
  # Without a mean structure the observed-bread sandwich of centered moment
  # contributions is the exact-score sandwich.
  expect_equal(res$covariance, vcov(fit, regime = "robust", data = d),
               tolerance = 1e-7, ignore_attr = TRUE)
  ctx <- prepare_inference(fit)
  for (test in c("score", "lr")) {
    explicit <- calibrate_quadratic(inference_quadratic(ctx, test), c("sb", "peba4"))
    got <- res[[test]]
    expect_true(got$available)
    expect_identical(got$reference, "peba4")
    expect_equal(got$statistic, explicit$statistic[1])
    expect_equal(got$df, explicit$df[1])
    expect_equal(got$p_sb, explicit$p_value[explicit$method == "sb"], tolerance = 1e-10)
    expect_equal(got$p_peba4, explicit$p_value[explicit$method == "peba4"], tolerance = 1e-10)
  }
})

test_that("policy_inference() reports estimators outside its scope", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  uls <- policy_inference(fit_model(cfa, d, estimator = "ULS"))
  expect_equal(uls$lr$reason, "unsupported_model")
})

test_that("penalized fits get no policy inference, for every estimator", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  for (estimator in c("ML", "FIML", "ULS")) {
    fit <- fit_model(cfa, d, estimator = estimator, covariance = "barrier")
    res <- policy_inference(fit)
    expect_false(res$covariance_available)
    expect_null(res$covariance)
    expect_equal(c(res$covariance_reason, res$score$reason, res$lr$reason),
                 rep("penalized", 3), label = estimator)
  }
  barrier <- fit_model(cfa, d, covariance = "barrier")
  plain <- fit_model(cfa, d)
  for (nested in list(policy_nested(plain, barrier), policy_nested(barrier, plain))) {
    expect_equal(c(nested$lr$reason, nested$score$reason), rep("penalized", 2))
  }
})

test_that("inference_rows() rebuilds the score statistic and its spectrum", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m1 <- "visual =~ x1 + a*x2 + b*x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9"
  f1 <- fit_model(m1, d)
  f0 <- fit_model(paste(m1, "a == b", sep = "\n"), d)
  shared <- prepare_inference_data(f1, d, storage = "casewise")
  c1 <- prepare_inference(f1, shared)
  c0 <- prepare_inference(f0, shared)
  for (q in list(inference_quadratic(c1, "score"),
                 inference_quadratic(prepare_hypothesis(c0, c1), "score"))) {
    rows <- inference_rows(q)
    expect_equal(nrow(rows), nrow(d))
    expect_equal(sum(colSums(rows)^2), q$statistic, tolerance = 1e-10)
    expect_equal(sort(eigen(crossprod(rows), symmetric = TRUE, only.values = TRUE)$values),
                 sort(score_spectrum(q)$eigenvalues), tolerance = 1e-8)
  }
})


test_that("FIML policy uses retained missing-pattern scores for global and nested tests", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  set.seed(13)
  d$x2[runif(nrow(d)) < plogis(-1.5 + scale(d$x1)[, 1])] <- NA
  restricted <- "visual =~ x1 + a*x2 + a*x3\ntextual =~ x4 + x5 + x6"
  for (group in list(NULL, "school")) {
    f1 <- fit_model(cfa, d, estimator = "FIML", group = group, meanstructure = TRUE)
    f0 <- fit_model(restricted, d, estimator = "FIML", group = group, meanstructure = TRUE)
    p <- policy_inference(f1)
    expect_true(p$covariance_available)
    expect_equal(p$covariance, vcov(f1, regime = "robust"), tolerance = 1e-7, ignore_attr = TRUE)
    for (t in list(p$score, p$lr, policy_nested(f1, f0)$score, policy_nested(f1, f0)$lr)) {
      expect_true(t$available, info = t$detail)
      expect_identical(t$reference, "peba4")
      expect_true(all(is.finite(c(t$statistic, t$p_sb, t$p_peba4))))
      expect_true(all(t$eigenvalues >= 0))
    }
    failed <- f1; failed$converged <- FALSE
    expect_identical(policy_inference(failed)$score$reason, "not_converged")
  }
})


test_that("DWLS nested policy equals the exact parameter-IJ construction", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ord <- paste0("x",1:6)
  for(v in ord) d[[v]] <- ordered(cut(d[[v]],
    quantile(d[[v]],c(0,1/3,2/3,1)),include.lowest=TRUE,labels=FALSE),levels=1:3)
  f1 <- fit_model(model_spec(cfa,ordered=ord,parameterization="theta",
    group="school",group_labels=levels(d$school),group_equal="thresholds"),d,estimator="DWLS")
  f0 <- fit_model(model_spec(cfa,ordered=ord,parameterization="theta",
    group="school",group_labels=levels(d$school),group_equal=c("thresholds","loadings")),d,estimator="DWLS")
  expect_true(f1$converged); expect_true(f0$converged)
  policy <- policy_nested(f1,f0)$lr
  expect_true(policy$available,info=policy$detail)
  parts <- magmaanlab:::ordinal_nested_diagnostic_impl(f1,f0)
  K <- parts$K; A <- parts$A
  H <- crossprod(K,parts$hessian_total%*%K)/f1$ntotal
  L <- solve(crossprod(K),t(K))
  V <- f1$ntotal*L%*%magmaan_core$robust_ordinal_ij(f1,f1$ordinal_stats, first_stage = "exact")$vcov%*%t(L)
  Ri <- solve(chol(A%*%solve(H,t(A))))
  e <- sort(eigen(t(Ri)%*%A%*%V%*%t(A)%*%Ri,symmetric=TRUE,only.values=TRUE)$values)
  expect_length(policy$eigenvalues,policy$df)
  expect_equal(policy$eigenvalues,e,tolerance=1e-10)
  # parts$common is the explicit OPG lab comparator, whose sampling law differs.
  profile <- magmaan_core$ordinal_profile_lrt(f1,f0,f1$ordinal_stats)
  expect_equal(policy$statistic,profile$T_diff,tolerance=1e-10)
  expect_equal(robust_nested_lrt(f1,f0)$eigenvalues,policy$eigenvalues,tolerance=1e-10)
  expect_true(all(is.finite(e) & e>=0))
  failed <- f1; failed$converged <- FALSE
  expect_identical(policy_nested(failed,f0)$lr$reason,"not_converged")
})

test_that("DWLS global policy exposes the exact spectrum All reference", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ord <- paste0("x", 1:6)
  for (v in ord) d[[v]] <- ordered(cut(d[[v]],
    quantile(d[[v]], c(0, 1/3, 2/3, 1)), include.lowest = TRUE, labels = FALSE))
  for (parameterization in c("delta", "theta")) {
    for (group in list(NULL, "school")) {
      spec <- model_spec(cfa, ordered = ord, parameterization = parameterization,
        group = group, group_labels = if (is.null(group)) NULL else levels(d$school))
      fit <- fit_model(spec, d, estimator = "DWLS")
      expect_true(fit$converged)
      policy <- policy_inference(fit)
      t <- policy$score
      expect_true(t$available, info = t$detail)
      expect_identical(t$reference, "all")
      expect_identical(policy$lr$reference, "sb_peba4")
      fixed <- robust_ordinal(fit, fit$ordinal_stats, bread = "expected")
      expect_equal(t$statistic, fixed$chisq_standard)
      exact <- magmaan_core$robust_ordinal_ij(fit, fit$ordinal_stats, first_stage = "exact")
      sampling_stats <- fit$ordinal_stats
      sampling_stats$NACOV <- exact$sampling_gamma
      exact_global <- robust_ordinal(fit, sampling_stats, bread = "expected")
      expect_equal(t$eigenvalues, sort(exact_global$eigvals), tolerance = 1e-10)
      explicit <- robust_fmg_test(t$statistic, t$df, t$eigenvalues, "all", 0,
        truncate_negative = TRUE)
      expect_equal(t$p_all, explicit$p_value, tolerance = 1e-7)
      expect_true(is.nan(t$p_sb) && is.nan(t$p_peba4) && is.nan(t$sb_scale))
      expect_equal(t$peba_blocks, 0L)
    }
  }
})
