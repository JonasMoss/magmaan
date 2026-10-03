# Compare cached calls with fresh handles, including process-local rebuilding.
fresh_policy_fit <- function(fit) {
  attr(fit, "policy_cache") <- NULL
  fit
}

test_that("FIML policy snapshots reuse ingredients and rebuild after restoration", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  d$x2[d$x1 < median(d$x1) & seq_len(nrow(d)) %% 3 == 0] <- NA_real_
  a <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  b <- "visual =~ x1 + a*x2 + a*x3\ntextual =~ x4 + x5 + x6"
  f1 <- fit_model(a, d, estimator = "FIML", group = "school", meanstructure = TRUE)
  f0 <- fit_model(b, d, estimator = "FIML", group = "school", meanstructure = TRUE)
  expect_true(f1$converged); expect_true(f0$converged)
  global <- policy_inference(f1)
  nested <- policy_nested(f1, f0)
  expect_identical(global, policy_inference(fresh_policy_fit(f1)))
  expect_identical(nested, policy_nested(fresh_policy_fit(f1), fresh_policy_fit(f0)))
  expect_identical(policy_nested(f1, f0), nested)
  expect_identical(inference_reuse(prepare_inference(f1))$ingredient_builds, 1)
  expect_identical(prepare_inference(f1), prepare_inference(f1))
  restored <- unserialize(serialize(list(f1, f0), NULL))
  expect_identical(policy_inference(restored[[1]]), global)
  expect_identical(policy_nested(restored[[1]], restored[[2]]), nested)
  ctx <- unserialize(serialize(prepare_inference(f1), NULL))
  expect_identical(scores(ctx), scores(prepare_inference(f1)))
  cache <- attr(f1, "policy_cache")
  cache$fiml$pid <- -1L
  before <- cache$fiml$value
  expect_identical(policy_inference(f1), global)
  expect_false(identical(before, prepare_inference(f1)))
  changed <- f1; changed$theta[1] <- changed$theta[1] + 0.01
  expect_identical(policy_inference(changed), policy_inference(fresh_policy_fit(changed)))
})

test_that("DWLS theta policy snapshots retain IJ and Hessian across nested calls", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ord <- paste0("x", 1:6)
  for (v in ord) d[[v]] <- ordered(cut(d[[v]], quantile(d[[v]], c(0, 1/3, 2/3, 1)),
                                     include.lowest = TRUE, labels = FALSE), levels = 1:3)
  syntax <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  fit <- function(equal) fit_model(model_spec(syntax, ordered = ord, parameterization = "theta",
    group = "school", group_labels = levels(d$school), group_equal = equal), d, estimator = "DWLS")
  f1 <- fit("thresholds"); f0 <- fit(c("thresholds", "loadings"))
  expect_true(f1$converged); expect_true(f0$converged)
  global <- policy_inference(f1); nested <- policy_nested(f1, f0)
  expect_true(global$covariance_available); expect_true(nested$lr$available)
  expect_identical(global, policy_inference(fresh_policy_fit(f1)))
  expect_identical(nested, policy_nested(fresh_policy_fit(f1), fresh_policy_fit(f0)))
  expect_identical(policy_inference(f1), global)
  expect_identical(policy_nested(f1, f0), nested)
  expect_identical(inference_reuse(f1)$ingredient_builds, 2)
  restored <- unserialize(serialize(list(f1, f0), NULL))
  expect_identical(policy_inference(restored[[1]]), global)
  expect_identical(policy_nested(restored[[1]], restored[[2]]), nested)
  cache <- attr(f1, "policy_cache")
  cache$dwls$pid <- -1L
  old <- cache$dwls$value
  expect_identical(policy_inference(f1), global)
  expect_false(identical(cache$dwls$value, old))
  changed <- f1; changed$ordinal_stats$W_dwls[[1]][1,1] <- changed$ordinal_stats$W_dwls[[1]][1,1] + 0.1
  expect_identical(policy_inference(changed), policy_inference(fresh_policy_fit(changed)))
  expect_error(policy_nested(changed, f0), "same observations")
})
