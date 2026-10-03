test_that("generic ML inference uses observed empirical covariance and nested geometry", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  syntax <- "visual =~ x1 + a*x2 + b*x3\ntextual =~ x4+x5+x6"
  h1 <- fit_model(syntax, d)
  h0 <- fit_model(paste(syntax, "a == b", sep = "\n"), d)
  shared <- prepare_inference_data(h1, d)
  c1 <- prepare_inference(h1, shared)
  c0 <- prepare_inference(h0, shared)
  hyp <- prepare_hypothesis(c0, c1)
  expect_equal(inference_information(c1), inference_information(c1, "observed"))
  expect_equal(inference_covariance(c1), inference_covariance(c1, TRUE))
  expect_equal(as.numeric(inference_covariance(c1)),
    as.numeric(vcov(h1, "sandwich_observed")), tolerance = 1e-8)
  for (kind in c("score", "lr")) {
    expect_equal(inference_quadratic(hyp, kind)$statistic,
      inference_quadratic(hyp, kind, "observed")$statistic)
    expect_true(all(is.finite(score_spectrum(inference_quadratic(hyp, kind))$eigenvalues)))
    expect_equal(inference_quadratic(c1, kind)$statistic,
      inference_quadratic(c1, kind, "expected")$statistic)
  }
  expect_true(all(is.finite(modification_indices(h1, d)$mi.scaled)))
  expect_true(all(is.finite(score_tests(h0, d)$mi.scaled)))
  expect_identical(nested_score_test(h1, h0, d)$sensitivity, "observed")
})

test_that("Q-route decisions use explicit meats, breads and supported policy laws", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939[paste0("x", 1:4)]
  syntax <- "f =~ x1 + a*x2 + b*x3 + x4"
  h1 <- fit_model(syntax, d)
  h0 <- fit_model(paste(syntax, "a == b", sep = "\n"), d)
  context <- prepare_inference(h1, d)
  info <- inference_information(context)
  meat <- crossprod(magmaan_core$infer_casewise_scores_fit(h1, as.matrix(d)))
  expect_equal(parameter_covariance(context, info),
    parameter_covariance(context, info, meat), tolerance = 1e-12)
  expect_equal(as.numeric(parameter_covariance(context, info)),
    as.numeric(vcov(h1)), tolerance = 1e-7)
  expect_true(all(is.finite(parameter_covariance(context, info, "model"))))
  expect_error(score_quadratic(c(1, 2), diag(2)), "explicit meat")
  expect_true(is.finite(score_quadratic(c(1, 2), diag(2), diag(2))$statistic))
  expect_error(magmaan_core$robust_build_u_factor(h1), "bread")
  expect_true(is.list(magmaan_core$robust_build_u_factor(h1, bread = "expected")))
  policy <- policy_nested(h1, h0)
  nested <- robust_nested_lrt(h1, h0)
  expect_equal(nested$eigenvalues, policy$lr$eigenvalues)
  expect_equal(nested$p_scaled, policy$lr$p_sb)
  expect_identical(nested$method, "policy")
  missing <- d
  missing$x2[seq(1L, nrow(d), 7L)] <- NA_real_
  f1 <- fit_model(syntax, missing, estimator = "FIML")
  f0 <- fit_model(paste(syntax, "a == b", sep = "\n"), missing, estimator = "FIML")
  fiml_policy <- policy_nested(f1, f0)
  fiml_nested <- robust_nested_lrt(f1, f0)
  expect_equal(fiml_nested$eigenvalues, fiml_policy$lr$eigenvalues)
  expect_equal(fiml_nested$p_scaled, fiml_policy$lr$p_sb)
  expect_identical(fiml_nested$method, "policy")
  expect_identical(global_score_flip_test(f1, n_flips = 3L)$sensitivity, "observed")
  expect_identical(global_score_flip_test(h1, d, n_flips = 3L)$sensitivity, "expected")
  fit <- fit_model(syntax, d, estimator = "GLS")
  expect_error(magmaan_core$infer_continuous_ls_robust(fit, as.matrix(d)),
    "infer_casewise_influence_ij_fit")
  expect_true(all(is.finite(magmaan_core$infer_continuous_ls_robust(
    fit, as.matrix(d), fixed_weight = TRUE)$vcov)))
})

test_that("reliability needs empirical data or explicit Gamma", {
  set.seed(61)
  eta <- rnorm(60)
  X <- outer(eta, c(1,.8,.9,.7)) + matrix(rnorm(240),60,4)
  S <- cov(X) * 59 / 60
  G <- magmaan_core$robust_empirical_gamma(X)
  for (omega in c(FALSE, TRUE)) {
    run <- function(...) {
      if (omega) magmaan_core$measures_reliability_omega_multidim(S, rep(1L, 4L), ...)
      else magmaan_core$measures_reliability_cov(S, ...)
    }
    expect_error(run(), "requires raw_data")
    expect_equal(run(raw_data = X), run(gamma = G, n = 60L), tolerance = 1e-12)
  }
})

test_that("scalar profile defaults to the misspecification-scaled reference", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939[paste0("x", 1:4)]
  fit <- fit_model("f =~ x1+x2+x3+x4", d)
  run <- magmaan_core$frontier_profile_lrt_parameter_ml
  index <- fit$partable$free[fit$partable$op == "=~" & fit$partable$rhs == "x2"]
  target <- fit$theta[index] * .95
  expect_error(run(fit, index, target), "ordinary.*explicitly")
  out <- run(fit, index, target, raw_data = as.matrix(d))
  explicit <- run(fit, index, target, raw_data = as.matrix(d), reference = "misspec_scaled")
  expect_equal(out, explicit)
  expect_true(is.finite(out$p_value))
  expect_true(is.list(run(fit, index, target, reference = "ordinary")))
})
