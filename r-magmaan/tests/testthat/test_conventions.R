# Whole-bundle parity, with matching mean structure, group order and random X.
convention_reference <- function(fit, lav, convention, tolerance = 2e-4) {
  ours <- fit$lab$partable
  ours <- ours[ours$free > 0L & !duplicated(ours$free), , drop = FALSE]
  ours <- ours[order(ours$free), , drop = FALSE]
  theirs <- lavaan::parTable(lav)
  theirs <- theirs[theirs$free > 0L & !duplicated(theirs$free), , drop = FALSE]
  theirs <- theirs[order(theirs$free), , drop = FALSE]
  index <- match(.key(ours, fit$lab$group_labels), .key(theirs, .lav_labels(lav)))
  expect_false(anyNA(index))
  actual <- vcov(fit, convention = convention)
  expect_equal(as.vector(actual), as.vector(lavaan::lavInspect(lav, "vcov")[index, index]), tolerance = tolerance)
  s <- summary(fit, convention = convention)
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
  ci <- confint(fit, convention = convention)
  expect_equal(as.vector(ci[, 2] - ci[, 1]), as.vector(2 * qnorm(.975) * sqrt(diag(actual))))
  expect_identical(attr(actual, "convention"), convention)
  expect_identical(attr(ci, "convention"), convention)
  expect_output(print(s), paste0("lavaan ", convention))
}

test_that("ML bundles match lavaan on one retained fit", {
  d <- hs()
  fit <- magmaan(cfa, d)
  original <- fit$inference
  theta <- coef(fit)
  for (convention in c("ML", "MLM", "MLR")) {
    lav <- lav_cfa(cfa, d, estimator = convention, fixed.x = FALSE)
    convention_reference(fit, lav, convention)
  }
  expect_identical(fit$inference, original)
  expect_identical(summary(fit, convention = "MLM")$fit$inference, original)
  expect_identical(coef(fit), theta)
  cached <- infer(fit, convention = "MLM")
  expect_identical(cached$inference, original)
  expect_equal(vcov(cached, convention = "MLM"), vcov(fit, convention = "MLM"))
  expect_equal(vcov(cached), vcov(fit))
  deferred <- magmaan(cfa, d, inference = FALSE)
  expect_equal(vcov(deferred, convention = "ML"), vcov(fit, convention = "ML"))
  expect_null(deferred$inference)
})

test_that("grouped bundles match with constrained loadings and means", {
  d <- hs()
  eq <- c("loadings", "intercepts")
  model <- magmaan_model(cfa, prototype = d, group = "school", group.equal = eq)
  fit <- magmaan(model, d)
  for (convention in c("ML", "MLM", "MLR")) {
    lav <- lavaan::cfa(cfa, d, estimator = convention, meanstructure = TRUE,
      fixed.x = FALSE, group = "school", group.equal = eq, group.label = levels(d$school))
    convention_reference(fit, lav, convention)
  }
})

test_that("random covariates compare against lavaan with fixed.x FALSE", {
  d <- hs()
  # The original integer grade contains only a few distinct values, producing
  # lavaan's near-singular fourth-moment covariance warning. A continuous
  # covariate keeps the joint random-X regression gate away from that corner.
  model <- paste(cfa, "visual ~ ageyr", sep = "\n")
  fit <- magmaan(model, d)
  for (convention in c("ML", "MLM", "MLR")) {
    lav <- lav_cfa(model, d, estimator = convention, fixed.x = FALSE)
    convention_reference(fit, lav, convention)
  }
})

test_that("ordinal bundles retain the estimator and match lavaan", {
  d <- ordinal_hs()
  ord <- paste0("x", 1:6)
  for (parameterization in c("delta", "theta")) {
    model <- magmaan_model(cfa, prototype = d, ordered = ord, parameterization = parameterization)
    for (estimator in c("DWLS", "ULS", "WLS")) {
      fit <- magmaan(model, d, estimator = estimator)
      conventions <- switch(estimator, DWLS = c("DWLS", "WLSMV"),
                            ULS = c("ULS", "ULSMV"), WLS = "WLS")
      for (convention in conventions) {
        lav <- lavaan::cfa(cfa, d, ordered = ord, estimator = convention,
                          parameterization = parameterization)
        convention_reference(fit, lav, convention, tolerance = 2e-3)
      }
      s <- fit$inference$status
      if (estimator == "DWLS") {
        expect_equal(s$available, c(TRUE, TRUE, FALSE))
        expect_equal(s$reason[3L], "inapplicable")
      } else expect_true(all(s$reason == "unsupported_model"))
    }
  }
})

test_that("nested ML conventions match lavTestLRT defaults in either order", {
  d <- hs()
  restricted <- paste(cfa, "visual ~~ 0*textual", sep = "\n")
  h1 <- magmaan(cfa, d)
  h0 <- magmaan(restricted, d)
  for (convention in c("ML", "MLM", "MLR")) {
    lav1 <- lav_cfa(cfa, d, estimator = convention)
    lav0 <- lav_cfa(restricted, d, estimator = convention)
    ref <- lavaan::lavTestLRT(lav1, lav0)
    a <- anova(h1, h0, convention = convention)
    b <- anova(h0, h1, convention = convention)
    expect_equal(a$statistic, ref[["Chisq diff"]][2], tolerance = 2e-4)
    expect_equal(a$df, ref[["Df diff"]][2])
    expect_equal(a$pvalue, ref[["Pr(>Chisq)"]][2], tolerance = 2e-4)
    expect_equal(a$statistic, b$statistic)
    expect_identical(attr(a, "convention"), convention)
    expect_output(print(a), paste0("lavaan ", convention))
  }
})

test_that("grouped ordinal reporting uses each group's n minus one", {
  d <- ordinal_hs()
  ord <- paste0("x", 1:6)
  for (parameterization in c("delta", "theta")) {
    model <- magmaan_model(cfa, prototype = d, ordered = ord, group = "school",
                          group.equal = "loadings", parameterization = parameterization)
    fit <- magmaan(model, d, estimator = "DWLS")
    for (convention in c("DWLS", "WLSMV")) {
      lav <- lavaan::cfa(cfa, d, ordered = ord, estimator = convention,
        parameterization = parameterization, group = "school", group.equal = "loadings",
        group.label = levels(d$school))
      convention_reference(fit, lav, convention, tolerance = 2e-3)
    }
  }
})

test_that("nested robust comparisons retain a saturated alternative", {
  d <- hs()
  h1 <- magmaan("x1 ~ x2 + x3", d)
  h0 <- magmaan("x1 ~ 0*x2 + x3", d)
  for (convention in c("ML", "MLM", "MLR")) {
    lav1 <- lavaan::sem("x1 ~ x2 + x3", d, meanstructure = TRUE, fixed.x = FALSE,
                        estimator = convention)
    lav0 <- lavaan::sem("x1 ~ 0*x2 + x3", d, meanstructure = TRUE, fixed.x = FALSE,
                        estimator = convention)
    ref <- lavaan::lavTestLRT(lav1, lav0)
    a <- anova(h1, h0, convention = convention)
    expect_length(attr(a, "unavailable"), 0L)
    expect_equal(a$statistic, ref[["Chisq diff"]][2], tolerance = 2e-4)
    expect_equal(a$pvalue, ref[["Pr(>Chisq)"]][2], tolerance = 2e-4)
  }
})

test_that("defined parameters and cached conventions retain their covariance", {
  d <- hs()
  model <- "f =~ x1 + a*x2 + b*x3 + x4\n difference := a - b"
  fit <- magmaan(model, d)
  lav <- lav_cfa(model, d, estimator = "MLM")
  ours <- coef(summary(fit, convention = "MLM"))
  theirs <- lavaan::parameterEstimates(lav)
  expect_equal(ours$se[ours$op == ":="], theirs$se[theirs$op == ":="], tolerance = 2e-4)
  cached <- infer(fit, convention = "MLM")
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(cached, path)
  expect_equal(vcov(readRDS(path), convention = "MLM"), vcov(cached, convention = "MLM"))
  expect_equal(vcov(readRDS(path), convention = "ML"), vcov(cached, convention = "ML"))
})

test_that("invalid and unavailable conventions cannot change the fit or fall back", {
  d <- hs()
  fit <- magmaan(cfa, d)
  for (bad in list("WLSMV", "MLMV", NA_character_, c("ML", "MLR"), TRUE))
    expect_error(vcov(fit, convention = bad), "convention")
  expect_error(summary(fit, convention = "WLSMV"), "incompatible")
  expect_error(confint(fit, convention = "WLSMV"), "incompatible")
  expect_error(anova(fit, fit, convention = "WLSMV"), "incompatible")
  failed <- fit
  failed$lab$converged <- FALSE
  err <- tryCatch(vcov(failed, convention = "MLM"), magmaan_inference_unavailable = identity)
  expect_identical(err$reason, "not_converged")
  expect_output(print(summary(failed, convention = "MLR")), "not_converged")
  d$x1[1:20] <- NA
  fiml <- magmaan(cfa, d, estimator = "FIML")
  expect_error(vcov(fiml, convention = "MLM"), "incompatible")
  err <- tryCatch(vcov(fiml, convention = "MLR"), magmaan_inference_unavailable = identity)
  expect_identical(err$reason, "unsupported_model")
  o <- ordinal_hs()
  model <- magmaan_model(cfa, prototype = o, ordered = paste0("x", 1:6))
  ord <- magmaan(model, o, estimator = "DWLS")
  a <- anova(ord, ord, convention = "WLSMV")
  expect_match(attr(a, "unavailable")[[1]], "unsupported_model")
})

test_that("scaled saturated tests and penalized fits keep typed unavailability", {
  d <- hs()
  saturated <- magmaan("x1 ~~ x1", d)
  expect_true(all(is.finite(vcov(saturated, convention = "MLR"))))
  s <- summary(saturated, convention = "MLR")
  expect_null(s$tests)
  expect_identical(s$inference$status$reason, c("available", "saturated"))
  expect_output(print(s), "saturated")
  penalized <- suppressMessages(magmaan(cfa, d, covariance = barrier(.1)))
  err <- tryCatch(vcov(penalized, convention = "MLM"), magmaan_inference_unavailable = identity)
  expect_identical(err$reason, "penalized")
  a <- anova(penalized, penalized, convention = "MLR")
  expect_match(attr(a, "unavailable")[[1]], "penalized")
})
