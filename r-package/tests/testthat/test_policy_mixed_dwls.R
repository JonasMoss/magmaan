test_that("mixed DWLS policy composes IJ covariance and nested differences", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  ordered <- paste0("x", 1:3)
  for (v in ordered) d[[v]] <- as.integer(cut(d[[v]], 3))
  syntax <- "f =~ x1 + a*x2 + b*x3\ng =~ x4 + x5 + x6\ng ~ f"
  for (parameterization in c("delta", "theta")) {
    for (group in list(NULL, "school")) {
      group_labels <- if (is.null(group)) NULL else unique(as.character(d[[group]]))
      spec <- model_spec(syntax, ordered = ordered, group = group, group_labels = group_labels,
                         parameterization = parameterization, fixed_x = FALSE,
                         auto_cov_y = TRUE)
      null <- model_spec(paste(syntax, "a == b", sep = "\n"), ordered = ordered,
                         group = group, group_labels = group_labels, parameterization = parameterization,
                         fixed_x = FALSE, auto_cov_y = TRUE)
      f1 <- fit_model(spec, d, estimator = "DWLS")
      f0 <- fit_model(null, d, estimator = "DWLS")
      expect_true(f1$converged)
      expect_true(f0$converged)
      out <- policy_inference(f1)
      expect_true(out$covariance_available, info = out$covariance_detail)
      expect_true(out$score$available, info = out$score$detail)
      expect_equal(out$covariance, vcov(f1, regime = "sandwich_ij"), tolerance = 1e-10, ignore_attr = TRUE)
      expect_identical(out$score$reference, "all")
      expect_true(is.finite(out$score$p_all))
      expect_identical(out$lr$reason, "inapplicable")
      nested <- policy_nested(f1, f0)
      expect_true(nested$lr$available, info = nested$lr$detail)
      expect_identical(nested$lr$label, "fit_function_difference")
      expect_true(is.finite(nested$lr$p_sb))
      expect_true(is.finite(nested$lr$p_peba4))
      expect_identical(nested$score$reason, "unsupported_model")
      expect_identical(inference_reuse(f1)$ingredient_builds, 2)
      expect_identical(policy_inference(f1), out)
      expect_identical(policy_nested(f1, f0), nested)
      expect_identical(inference_reuse(f1)$ingredient_builds, 2)
      wls <- fit_model(spec, d, estimator = "WLS")
      expect_identical(policy_inference(wls)$covariance_reason, "unsupported_model")
      failed <- f1; failed$converged <- FALSE
      expect_identical(policy_inference(failed)$score$reason, "not_converged")
    }
  }
})
