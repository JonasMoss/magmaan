test_that("SNLLS rejects bounds at both R entry levels and exposes the full audit", {
  model <- model_spec("f =~ x1 + x2 + x3 + x4")
  set.seed(120926)
  z <- rnorm(300)
  x <- as.data.frame(sapply(1:4, function(j) (0.6 + j / 10) * z + rnorm(300)))
  names(x) <- paste0("x", 1:4)
  stats <- df_to_data(x, model)
  for (kind in c("uls", "gls", "wls")) {
    extra <- if (kind == "wls") list(W = list(diag(10))) else list()
    friendly <- magmaan_core[[paste0("fit_", kind, "_snlls")]]
    primitive <- magmaan_core[[paste0("estimate_", kind, "_snlls")]]
    expect_error(do.call(friendly, c(list(model, stats, bounds = "pos.var"), extra)),
                 "SNLLS does not support bounds")
    # Even an empty supplied list must not be silently ignored by the primitive.
    primitive_args <- list(model$partable,
                           list(S = stats$S, mean = stats$mean, nobs = stats$nobs),
                           bounds = list())
    expect_error(do.call(primitive, c(primitive_args, extra)),
                 "SNLLS does not support bounds")
    fit <- do.call(friendly, c(list(model, stats, bounds = "none"), extra))
    expect_true(fit$converged)
    expect_true(fit$diagnostics$geometric_stationarity$checked)
    expect_true(fit$diagnostics$geometric_stationarity$gradient_finite)
    expect_true(fit$diagnostics$geometric_stationarity$ambient_stationary)
  }
})
