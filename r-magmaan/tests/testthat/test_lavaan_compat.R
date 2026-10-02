# Whole-bundle parity, with matching mean structure, group order and random X.
lavaan_compat_reference <- function(fit, lav, lavaan_compat, tolerance = 2e-4) {
  ours <- fit$lab$partable
  ours <- ours[ours$free > 0L & !duplicated(ours$free), , drop = FALSE]
  ours <- ours[order(ours$free), , drop = FALSE]
  theirs <- lavaan::parTable(lav)
  theirs <- theirs[theirs$free > 0L & !duplicated(theirs$free), , drop = FALSE]
  theirs <- theirs[order(theirs$free), , drop = FALSE]
  index <- match(.key(ours, fit$lab$group_labels), .key(theirs, .lav_labels(lav)))
  expect_false(anyNA(index))
  actual <- vcov(fit, lavaan_compat = lavaan_compat)
  expect_equal(as.vector(actual), as.vector(lavaan::lavInspect(lav, "vcov")[index, index]), tolerance = tolerance)
  s <- summary(fit, lavaan_compat = lavaan_compat)
  rows <- coef(s)
  lr <- lavaan::lavInspect(lav, "test")
  t <- lr[[length(lr)]]
  expect_equal(s$tests$statistic, as.numeric(t$stat), tolerance = tolerance)
  expect_equal(s$tests$df, t$df)
  if (is.na(t$pvalue)) expect_true(is.na(s$tests$pvalue))
  else expect_equal(s$tests$pvalue, as.numeric(t$pvalue), tolerance = tolerance)
  expect_equal(s$tests$scale, if (is.null(t$scaling.factor)) 1 else as.numeric(t$scaling.factor), tolerance = tolerance)
  expect_equal(s$tests$shift, if (is.null(t$shift.parameter)) 0 else t$shift.parameter, tolerance = tolerance)
  free <- rows$free
  expect_equal(rows$se[free], sqrt(diag(actual))[fit$lab$partable$free[free]], ignore_attr = TRUE)
  ci <- confint(fit, lavaan_compat = lavaan_compat)
  expect_equal(as.vector(ci[, 2] - ci[, 1]), as.vector(2 * qnorm(.975) * sqrt(diag(actual))))
  expect_identical(attr(actual, "lavaan_compat"), lavaan_compat)
  expect_identical(attr(ci, "lavaan_compat"), lavaan_compat)
  expect_output(print(s), paste0("lavaan compatibility: ", lavaan_compat))
}

test_that("ML bundles match lavaan on one retained fit", {
  d <- hs()
  fit <- magmaan(cfa, d)
  original <- fit$inference
  theta <- coef(fit)
  for (lavaan_compat in c("ML", "MLM", "MLR")) {
    lav <- lav_cfa(cfa, d, estimator = lavaan_compat, fixed.x = FALSE)
    lavaan_compat_reference(fit, lav, lavaan_compat)
  }
  expect_identical(fit$inference, original)
  expect_identical(summary(fit, lavaan_compat = "MLM")$fit$inference, original)
  expect_identical(coef(fit), theta)
  cached <- infer(fit, lavaan_compat = "MLM")
  expect_identical(cached$inference, original)
  expect_identical(cached$lavaan_compat[["MLM"]]$lavaan_compat, "MLM")
  expect_equal(vcov(cached, lavaan_compat = NULL), vcov(fit))
  expect_equal(vcov(cached, lavaan_compat = "MLM"), vcov(fit, lavaan_compat = "MLM"))
  expect_equal(vcov(cached), vcov(fit))
  deferred <- magmaan(cfa, d, inference = FALSE)
  expect_equal(vcov(deferred, lavaan_compat = "ML"), vcov(fit, lavaan_compat = "ML"))
  expect_null(deferred$inference)
})

test_that("grouped bundles match with constrained loadings and means", {
  d <- hs()
  eq <- c("loadings", "intercepts")
  model <- magmaan_model(cfa, prototype = d, group = "school", group.equal = eq)
  fit <- magmaan(model, d)
  for (lavaan_compat in c("ML", "MLM", "MLR")) {
    lav <- lavaan::cfa(cfa, d, estimator = lavaan_compat, meanstructure = TRUE,
      fixed.x = FALSE, group = "school", group.equal = eq, group.label = levels(d$school))
    lavaan_compat_reference(fit, lav, lavaan_compat)
  }
})

test_that("random covariates compare against lavaan with fixed.x FALSE", {
  d <- hs()
  # The original integer grade contains only a few distinct values, producing
  # lavaan's near-singular fourth-moment covariance warning. A continuous
  # covariate keeps the joint random-X regression gate away from that corner.
  model <- paste(cfa, "visual ~ ageyr", sep = "\n")
  fit <- magmaan(model, d)
  for (lavaan_compat in c("ML", "MLM", "MLR")) {
    lav <- lav_cfa(model, d, estimator = lavaan_compat, fixed.x = FALSE)
    lavaan_compat_reference(fit, lav, lavaan_compat)
  }
})

test_that("ordinal bundles retain the estimator and match lavaan", {
  d <- ordinal_hs()
  ord <- paste0("x", 1:6)
  for (parameterization in c("delta", "theta")) {
    model <- magmaan_model(cfa, prototype = d, ordered = ord, parameterization = parameterization)
    for (estimator in c("DWLS", "ULS", "WLS")) {
      fit <- magmaan(model, d, estimator = estimator)
      lavaan_bundles <- switch(estimator, DWLS = c("DWLS", "WLSMV"),
                            ULS = c("ULS", "ULSMV"), WLS = "WLS")
      for (lavaan_compat in lavaan_bundles) {
        lav <- lavaan::cfa(cfa, d, ordered = ord, estimator = lavaan_compat,
                          parameterization = parameterization)
        lavaan_compat_reference(fit, lav, lavaan_compat, tolerance = 2e-3)
      }
      s <- fit$inference$status
      if (estimator == "DWLS") {
        expect_equal(s$available, c(TRUE, TRUE, FALSE))
        expect_equal(s$reason[3L], "inapplicable")
      } else expect_true(all(s$reason == "unsupported_model"))
    }
  }
})

test_that("nested ML lavaan_bundles match lavTestLRT defaults in either order", {
  d <- hs()
  restricted <- paste(cfa, "visual ~~ 0*textual", sep = "\n")
  h1 <- magmaan(cfa, d)
  h0 <- magmaan(restricted, d)
  for (lavaan_compat in c("ML", "MLM", "MLR")) {
    lav1 <- lav_cfa(cfa, d, estimator = lavaan_compat)
    lav0 <- lav_cfa(restricted, d, estimator = lavaan_compat)
    ref <- lavaan::lavTestLRT(lav1, lav0)
    a <- anova(h1, h0, lavaan_compat = lavaan_compat)
    b <- anova(h0, h1, lavaan_compat = lavaan_compat)
    expect_equal(a$statistic, ref[["Chisq diff"]][2], tolerance = 2e-4)
    expect_equal(a$df, ref[["Df diff"]][2])
    expect_equal(a$pvalue, ref[["Pr(>Chisq)"]][2], tolerance = 2e-4)
    expect_equal(a$statistic, b$statistic)
    expect_identical(attr(a, "lavaan_compat"), lavaan_compat)
    expect_output(print(a), paste0("lavaan compatibility: ", lavaan_compat))
  }
})

test_that("grouped ML nested defaults gate loading intercept and mean restrictions", {
  d <- hs()
  restrictions <- list(character(), "loadings", c("loadings", "intercepts"),
                       c("loadings", "intercepts", "means"))
  fits <- lapply(restrictions, function(eq) {
    model <- magmaan_model(cfa, prototype = d, group = "school", group.equal = eq)
    magmaan(model, d)
  })
  for (lavaan_compat in c("ML", "MLM", "MLR")) {
    refs <- lapply(restrictions, function(eq) lavaan::cfa(cfa, d,
      estimator = lavaan_compat, meanstructure = TRUE, fixed.x = FALSE,
      group = "school", group.equal = eq, group.label = levels(d$school)))
    for (i in 2:4) {
      # lavaan 0.7.2 defaults: standard for ML, satorra.bentler.2001
      # for MLM/MLR. The last pair releases the second group's latent means.
      ref <- lavaan::lavTestLRT(refs[[i - 1L]], refs[[i]])
      method <- if (lavaan_compat == "ML") "standard" else "satorra.bentler.2001"
      explicit <- lavaan::lavTestLRT(refs[[i - 1L]], refs[[i]], method = method)
      expect_equal(ref, explicit)
      for (order in list(c(i - 1L, i), c(i, i - 1L))) {
        a <- anova(fits[[order[1]]], fits[[order[2]]], lavaan_compat = lavaan_compat)
        expect_length(attr(a, "unavailable"), 0L)
        expect_equal(a$statistic, ref[["Chisq diff"]][2], tolerance = 2e-4)
        expect_equal(a$df, ref[["Df diff"]][2])
        expect_equal(a$pvalue, ref[["Pr(>Chisq)"]][2], tolerance = 2e-4)
      }
    }
  }
})

test_that("grouped ordinal reporting uses each group's n minus one", {
  d <- ordinal_hs()
  ord <- paste0("x", 1:6)
  expect_length(unique(as.integer(table(d$school))), 2L)
  for (parameterization in c("delta", "theta")) {
    model <- magmaan_model(cfa, prototype = d, ordered = ord, group = "school",
                          group.equal = "loadings", parameterization = parameterization)
    for (estimator in c("DWLS", "ULS", "WLS")) {
      fit <- magmaan(model, d, estimator = estimator)
      lavaan_bundles <- switch(estimator, DWLS = c("DWLS", "WLSMV"),
                            ULS = c("ULS", "ULSMV"), WLS = "WLS")
      for (lavaan_compat in lavaan_bundles) {
        # lavInspect("vcov") and lavInspect("test"): robust.sem and
        # scaled.shifted for MV bundles; robust.sem/standard for plain
        # DWLS/ULS (no p-value); standard/standard for WLS. lavaan's
        # categorical reporting uses each unequal group's n_g - 1.
        lav <- lavaan::cfa(cfa, d, ordered = ord, estimator = lavaan_compat,
          parameterization = parameterization, group = "school", group.equal = "loadings",
          group.label = levels(d$school))
        lavaan_compat_reference(fit, lav, lavaan_compat, tolerance = 2e-3)
        options <- lavaan::lavInspect(lav, "options")
        expect_identical(options$se, if (estimator == "WLS") "standard" else "robust.sem")
        test <- if (lavaan_compat %in% c("WLSMV", "ULSMV")) "scaled.shifted" else "standard"
        expect_true(test %in% options$test)
      }
    }
  }
})

test_that("grouped theta threshold equalities match lavaan reporting bundles", {
  # Theta only: under delta lavaan's released ~*~ scale is not identified
  # (see project/architecture/capabilities/ordinal_and_mixed.md).
  d <- ordinal_hs()
  ord <- paste0("x", 1:6)
  for (eq in list("thresholds", c("loadings", "thresholds"),
                  c("loadings", "thresholds", "intercepts"))) {
    model <- magmaan_model(cfa, prototype = d, ordered = ord, group = "school",
                          group.equal = eq, parameterization = "theta")
    for (estimator in c("DWLS", "ULS", "WLS")) {
      fit <- magmaan(model, d, estimator = estimator)
      lavaan_bundles <- switch(estimator, DWLS = c("DWLS", "WLSMV"),
                            ULS = c("ULS", "ULSMV"), WLS = "WLS")
      for (lavaan_compat in lavaan_bundles) {
        lav <- lavaan::cfa(cfa, d, ordered = ord, estimator = lavaan_compat,
          parameterization = "theta", group = "school", group.equal = eq,
          group.label = levels(d$school))
        lavaan_compat_reference(fit, lav, lavaan_compat, tolerance = 2e-3)
      }
    }
  }
})

test_that("nested robust comparisons retain a saturated alternative", {
  d <- hs()
  h1 <- magmaan("x1 ~ x2 + x3", d)
  h0 <- magmaan("x1 ~ 0*x2 + x3", d)
  for (lavaan_compat in c("ML", "MLM", "MLR")) {
    lav1 <- lavaan::sem("x1 ~ x2 + x3", d, meanstructure = TRUE, fixed.x = FALSE,
                        estimator = lavaan_compat)
    lav0 <- lavaan::sem("x1 ~ 0*x2 + x3", d, meanstructure = TRUE, fixed.x = FALSE,
                        estimator = lavaan_compat)
    ref <- lavaan::lavTestLRT(lav1, lav0)
    a <- anova(h1, h0, lavaan_compat = lavaan_compat)
    expect_length(attr(a, "unavailable"), 0L)
    expect_equal(a$statistic, ref[["Chisq diff"]][2], tolerance = 2e-4)
    expect_equal(a$pvalue, ref[["Pr(>Chisq)"]][2], tolerance = 2e-4)
  }
})

test_that("defined parameters and cached lavaan_bundles retain their covariance", {
  d <- hs()
  model <- "f =~ x1 + a*x2 + b*x3 + x4\n difference := a - b"
  fit <- magmaan(model, d)
  lav <- lav_cfa(model, d, estimator = "MLM")
  ours <- coef(summary(fit, lavaan_compat = "MLM"))
  theirs <- lavaan::parameterEstimates(lav)
  expect_equal(ours$se[ours$op == ":="], theirs$se[theirs$op == ":="], tolerance = 2e-4)
  cached <- infer(fit, lavaan_compat = "MLM")
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(cached, path)
  expect_equal(vcov(readRDS(path), lavaan_compat = "MLM"), vcov(cached, lavaan_compat = "MLM"))
  expect_equal(vcov(readRDS(path), lavaan_compat = "ML"), vcov(cached, lavaan_compat = "ML"))
})

test_that("invalid and unavailable lavaan_bundles cannot change the fit or fall back", {
  d <- hs()
  fit <- magmaan(cfa, d)
  for (bad in list("magmaan", "WLSMV", "MLMV", NA_character_, c("ML", "MLR"), TRUE))
    expect_error(vcov(fit, lavaan_compat = bad), "lavaan_compat")
  expect_error(summary(fit, lavaan_compat = "WLSMV"), "incompatible")
  expect_error(confint(fit, lavaan_compat = "WLSMV"), "incompatible")
  expect_error(anova(fit, fit, lavaan_compat = "WLSMV"), "incompatible")
  failed <- fit
  failed$lab$converged <- FALSE
  err <- tryCatch(vcov(failed, lavaan_compat = "MLM"), magmaan_inference_unavailable = identity)
  expect_identical(err$reason, "not_converged")
  expect_output(print(summary(failed, lavaan_compat = "MLR")), "not_converged")
  d$x1[1:20] <- NA
  fiml <- magmaan(cfa, d, estimator = "FIML")
  expect_error(vcov(fiml, lavaan_compat = "MLM"), "incompatible")
  err <- tryCatch(vcov(fiml, lavaan_compat = "MLR"), magmaan_inference_unavailable = identity)
  expect_identical(err$reason, "unsupported_model")
  o <- ordinal_hs()
  model <- magmaan_model(cfa, prototype = o, ordered = paste0("x", 1:6))
  ord <- magmaan(model, o, estimator = "DWLS")
  a <- anova(ord, ord, lavaan_compat = "WLSMV")
  expect_match(attr(a, "unavailable")[[1]], "unsupported_model")
})

test_that("scaled saturated tests and penalized fits keep typed unavailability", {
  d <- hs()
  saturated <- magmaan("x1 ~~ x1", d)
  expect_true(all(is.finite(vcov(saturated, lavaan_compat = "MLR"))))
  s <- summary(saturated, lavaan_compat = "MLR")
  expect_null(s$tests)
  expect_identical(s$inference$status$reason, c("available", "saturated"))
  expect_output(print(s), "saturated")
  penalized <- suppressMessages(magmaan(cfa, d, covariance = barrier(.1)))
  err <- tryCatch(vcov(penalized, lavaan_compat = "MLM"), magmaan_inference_unavailable = identity)
  expect_identical(err$reason, "penalized")
  a <- anova(penalized, penalized, lavaan_compat = "MLR")
  expect_match(attr(a, "unavailable")[[1]], "penalized")
})
