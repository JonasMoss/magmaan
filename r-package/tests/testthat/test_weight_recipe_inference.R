# Estimated-weight inference follows the weight recipe a fit records, never its
# computational label, and two-stage (ML2S) fits dispatch to the ML2S score
# tests. DLS with a = 0 and a = 1 rebuilds the NT and ADF weights, which gives
# exact cross-recipe reductions without an oracle.

recipe_data <- function(n = 400L, seed = 20261002L) {
  set.seed(seed)
  eta <- rt(n, df = 6) * sqrt(4 / 6)
  noise <- matrix(rt(n * 5L, df = 6), n, 5L) * 0.6
  X <- outer(eta, c(1, 0.8, 0.7, 0.6, 0.5)) + noise
  X[, 5L] <- X[, 5L] + 0.35 * X[, 4L]
  colnames(X) <- paste0("x", 1:5)
  as.data.frame(X)
}

recipe_fits <- function(spec, d) {
  list(
    nt = fit_model(spec, d, estimator = "GLS"),
    adf = fit_model(spec, d, estimator = "WLS"),
    dwls = fit_model(spec, d, estimator = "DWLS"),
    dls0 = fit_model(spec, d, estimator = "DLS", dls_a = 0),
    dls1 = fit_model(spec, d, estimator = "DLS", dls_a = 1),
    dls3 = fit_model(spec, d, estimator = "DLS", dls_a = 0.3)
  )
}

loading_parameter <- function(fit, rhs) {
  fit$partable$free[fit$partable$op == "=~" & fit$partable$rhs == rhs]
}

test_that("estimated-weight inference reads the recorded continuous recipe", {
  d <- recipe_data()
  fits <- recipe_fits(model_spec("f =~ x1 + x2 + x3 + x4 + x5",
                                 meanstructure = FALSE), d)
  for (nm in names(fits)) expect_false(is.null(fits[[nm]]$composition$weight), info = nm)

  mi <- lapply(fits, modification_indices_robust, data = d,
               estimated_weight = TRUE)
  for (nm in names(mi)) {
    expect_true(all(is.finite(mi[[nm]]$mi.scaled)), info = nm)
  }
  # DLS(1) rebuilds the ADF weight exactly; DLS(0) and GLS are separate
  # optimizations of the same objective, so they agree to optimizer precision.
  expect_equal(mi$dls0$mi.scaled, mi$nt$mi.scaled, tolerance = 1e-5)
  expect_equal(mi$dls1$mi.scaled, mi$adf$mi.scaled, tolerance = 1e-10)
  expect_gt(max(abs(mi$dls3$mi.scaled - mi$dls0$mi.scaled)), 1e-3)
  expect_gt(max(abs(mi$dls3$mi.scaled - mi$dls1$mi.scaled)), 1e-3)
  # The weight influence is not zero for the estimated recipes.
  for (nm in c("dwls", "dls3")) {
    fixed <- modification_indices_robust(fits[[nm]], data = d)
    expect_gt(max(abs(fixed$mi.scaled - mi[[nm]]$mi.scaled)), 1e-4)
  }

  res <- lapply(fits[c("nt", "adf", "dls0", "dls1", "dwls")], residuals,
                standardized = TRUE, estimated_weight = TRUE, data = d)
  expect_equal(res$dls0, res$nt, tolerance = 1e-5)
  expect_equal(res$dls1, res$adf, tolerance = 1e-10)

  influence <- lapply(fits[c("nt", "adf", "dls0", "dls1", "dls3")],
                      est_change_raw_approx, type = "estimated.weight")
  expect_equal(unclass(influence$dls0)[, ], unclass(influence$nt)[, ],
               tolerance = 1e-5)
  expect_equal(unclass(influence$dls1)[, ], unclass(influence$adf)[, ],
               tolerance = 1e-10)

  X <- as.matrix(d)
  profile <- lapply(fits[c("nt", "dls0", "dls3")], function(fit) {
    k <- loading_parameter(fit, "x3")
    magmaan_core$frontier_profile_lrt_parameter_gmm(
      fit, k, 0.95 * fit$theta[k], raw_data = X, robust = TRUE,
      estimated_weight = TRUE)
  })
  expect_equal(profile$dls0$scaling_factor, profile$nt$scaling_factor,
               tolerance = 1e-5)
  expect_true(is.finite(profile$dls3$scaling_factor))

  rbm <- lapply(fits[c("nt", "dls0")], magmaan_core$frontier_rbm, raw_data = X)
  expect_equal(rbm$dls0$theta, rbm$nt$theta, tolerance = 1e-5)
})

test_that("release score tests read the recorded recipe", {
  d <- recipe_data()
  spec <- model_spec("f =~ x1 + a*x2 + a*x3 + x4 + x5", meanstructure = FALSE)
  fits <- recipe_fits(spec, d)
  st <- lapply(fits, score_tests_robust, data = d, estimated_weight = TRUE)
  for (nm in names(st)) expect_true(all(is.finite(st[[nm]]$mi.scaled)), info = nm)
  expect_equal(st$dls0$mi.scaled, st$nt$mi.scaled, tolerance = 1e-5)
  expect_equal(st$dls1$mi.scaled, st$adf$mi.scaled, tolerance = 1e-10)
  # The ordinary release tests use the recorded fitting weight.
  expect_true(all(is.finite(score_tests(fits$dwls)$mi)))
})

test_that("supplied weights are used as fitted but have no estimated-weight recipe", {
  d <- recipe_data()
  spec <- model_spec("f =~ x1 + x2 + x3 + x4 + x5", meanstructure = FALSE)
  adf <- fit_model(spec, d, estimator = "WLS")
  supplied <- fit_model(spec, d, estimator = "WLS", W = adf$W)
  expect_identical(supplied$composition$weight, "custom")

  fixed <- modification_indices_robust(supplied, data = d)
  expect_equal(fixed$mi.scaled, modification_indices_robust(adf, data = d)$mi.scaled,
               tolerance = 1e-6)
  expect_equal(modification_indices(supplied)$mi, modification_indices(adf)$mi,
               tolerance = 1e-6)
  expect_equal(modification_indices_robust(adf, data = d, weight = adf$W)$mi.scaled,
               fixed$mi.scaled, tolerance = 1e-6)
  expect_error(modification_indices_robust(adf, data = d, weight = diag(nrow(adf$W))),
               "fit\\$W")

  expect_error(modification_indices_robust(supplied, data = d, estimated_weight = TRUE),
               "UnsupportedInference")
  expect_error(residuals(supplied, standardized = TRUE, estimated_weight = TRUE, data = d),
               "UnsupportedInference")
  expect_error(est_change_raw_approx(supplied, type = "estimated.weight"),
               "UnsupportedInference")
  k <- loading_parameter(supplied, "x3")
  expect_error(magmaan_core$frontier_profile_lrt_parameter_gmm(
    supplied, k, 0.95 * supplied$theta[k], raw_data = as.matrix(d), robust = TRUE,
    estimated_weight = TRUE), "UnsupportedInference")
})

test_that("ordinal NT, DLS and supplied weights refuse the weight influence", {
  cont <- recipe_data(n = 600L)
  ord <- as.data.frame(lapply(cont[1:4], function(x)
    ordered(cut(x, c(-Inf, -0.5, 0.5, Inf)))))
  spec <- model_spec("f =~ x1 + x2 + x3 + x4", ordered = names(ord),
                     meanstructure = FALSE)
  dwls <- fit_model(spec, ord, estimator = "DWLS")
  expect_true(all(is.finite(
    modification_indices_robust(dwls, estimated_weight = TRUE)$mi.scaled)))
  supplied <- fit_model(spec, ord, estimator = "DWLS",
                        W = diag(seq(0.7, 1.3, length.out = 14L)))
  fits <- list(nt = fit_model(spec, ord, estimator = "GLS"),
               dls = fit_model(spec, ord, estimator = "DLS", dls_a = 0.3),
               supplied = supplied)
  for (nm in names(fits)) {
    expect_true(all(is.finite(modification_indices_robust(fits[[nm]])$mi.scaled)),
                info = nm)
    expect_error(modification_indices_robust(fits[[nm]], estimated_weight = TRUE),
                 "UnsupportedInference", info = nm)
  }
  # Association ML has no MI contract until 0.3.0.
  ml <- fit_model(spec, ord, estimator = "ML")
  expect_error(modification_indices(ml), "association")
  expect_error(modification_indices_robust(ml), "association")
})

test_that("two-stage MI and releases reduce to complete-data robust tests", {
  d <- recipe_data(n = 300L)
  spec <- model_spec("f =~ x1 + x2 + x3 + x4 + x5", meanstructure = TRUE)
  spec_eq <- model_spec("f =~ x1 + a*x2 + a*x3 + x4 + x5", meanstructure = TRUE)
  complete <- c(nt = "ML", uls = "ULS", dwls = "DWLS", adf = "WLS", dls = "DLS")
  for (w in names(complete)) {
    for (s in list(spec, spec_eq)) {
      two <- fit_model(s, d, estimator = "ML2S", stage2_weight = w, dls_a = 0.3)
      one <- fit_model(s, d, estimator = complete[[w]], dls_a = 0.3)
      expect_equal(two$theta, one$theta, tolerance = 1e-10, info = w)
      run <- if (identical(s, spec)) modification_indices_robust else score_tests_robust
      for (ew in c(FALSE, TRUE)) {
        if (ew && w %in% c("nt", "uls")) next
        a <- run(two, estimated_weight = ew)
        b <- run(one, data = d, estimated_weight = ew)
        expect_identical(attr(a, "mi_type"), "naive_stage2")
        expect_equal(a$mi, b$mi, tolerance = 1e-10, info = paste(w, ew))
        expect_equal(a$mi.scaled, b$mi.scaled, tolerance = 1e-10, info = paste(w, ew))
      }
      naive <- if (identical(s, spec)) modification_indices(two) else score_tests(two)
      expect_identical(attr(naive, "mi_type"), "naive_stage2")
      expect_equal(naive$mi, run(two)$mi, tolerance = 1e-8, info = w)
    }
  }
})

test_that("two-stage MI under missing data uses the recorded Stage-2 weight", {
  d <- recipe_data(n = 300L)
  set.seed(7L)
  d$x2[sample.int(nrow(d), 40L)] <- NA_real_
  d$x4[sample.int(nrow(d), 40L)] <- NA_real_
  spec <- model_spec("f =~ x1 + x2 + x3 + x4 + x5", meanstructure = TRUE)
  nt <- fit_model(spec, d, estimator = "ML2S")
  dls <- fit_model(spec, d, estimator = "ML2S", stage2_weight = "dls", dls_a = 0.3)

  nt_mi <- modification_indices_robust(nt)
  expect_true(all(is.finite(nt_mi$mi.scaled) & nt_mi$scaling.factor > 0))
  expect_equal(modification_indices_robust(nt, estimated_weight = TRUE)$mi.scaled,
               nt_mi$mi.scaled, tolerance = 1e-10)
  fixed <- modification_indices_robust(dls)
  estimated <- modification_indices_robust(dls, estimated_weight = TRUE)
  expect_true(all(is.finite(estimated$mi.scaled)))
  expect_equal(estimated$mi, fixed$mi, tolerance = 1e-10)
  expect_gt(max(abs(estimated$mi.scaled - fixed$mi.scaled)), 1e-4)
  expect_equal(modification_indices(dls)$mi, fixed$mi, tolerance = 1e-8)

  expect_error(modification_indices_robust(nt, information = "observed"),
               "UnsupportedInference")
  expect_error(modification_indices_robust(nt, data = d), "omit `data`")
  expect_error(modification_indices_robust(nt, bread = "observed"), "bread")
  expect_error(modification_indices_robust(dls, weight = diag(20)), "omit `weight`")

  # Refits and case influence use the recorded Stage-2 weight.
  expect_error(magmaan_core$frontier_rbm(dls, stage2_weight = "nt"), "recorded")
  expect_error(magmaan_core$frontier_rbm(dls, dls_a = 0.5), "recorded")
  expect_true(all(is.finite(est_change_raw_approx(dls, type = "estimated.weight"))))
})
