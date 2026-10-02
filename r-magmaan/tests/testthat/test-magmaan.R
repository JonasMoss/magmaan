test_that("ML estimates come from magmaanlab and match lavaan", {
  d <- hs()
  fit <- magmaan(cfa, d)
  lab <- magmaanlab::fit_model(cfa, d, meanstructure = TRUE, fixed_x = FALSE)
  expect_s3_class(fit, "magmaan")
  expect_s3_class(as_lab_fit(fit), "magmaan_fit")
  expect_equal(unname(coef(fit)), lab$theta)
  expect_equal(nobs(fit), 301)
  expect_lavaan_estimates(fit, lav_cfa(cfa, d))
})

test_that("model options follow lavaan's meaning", {
  d <- hs()
  expect_lavaan_estimates(magmaan(magmaan_model(cfa, identification = "std.lv"), d),
                          lav_cfa(cfa, d, std.lv = TRUE))
  m <- magmaan_model(cfa, prototype = d, group = "school", group.equal = "loadings")
  fit <- magmaan(m, d)
  expect_lavaan_estimates(fit, lav_cfa(cfa, d, group = "school", group.equal = "loadings"))
  # Groups follow the prototype's factor levels.
  expect_equal(fit$rows$group, levels(d$school))
  expect_true("group" %in% names(coef(summary(fit))))
  intercept <- "visual =~ x1 + x2 + x3\n x1 ~ 1"
  expect_lavaan_estimates(magmaan(intercept, d), lavaan::cfa(intercept, d))
})

test_that("start = \"fabin3\" gives the former FABIN3 start", {
  d <- hs()
  fit <- magmaan(cfa, d, inference = FALSE)
  old <- magmaan(cfa, d, options = list(start = "fabin3"), inference = FALSE)
  expect_identical(as_lab_fit(fit)$ml_start_policy, "layered")
  expect_identical(as_lab_fit(old)$ml_start_policy, "transported-std-lv-fabin")
  lab <- magmaanlab::fit_model(cfa, d, meanstructure = TRUE, fixed_x = FALSE,
                               control = list(start = "scaled-fabin"))
  expect_identical(unname(coef(old)), lab$theta)
  expect_equal(coef(old), coef(fit), tolerance = 1e-4)
  gls <- magmaan(cfa, d, estimator = "GLS", options = list(start = "fabin3"), inference = FALSE)
  expect_identical(unname(coef(gls)),
                   magmaanlab::fit_model(cfa, d, estimator = "GLS", meanstructure = TRUE,
                                         fixed_x = FALSE, control = list(start = "fabin3"))$theta)
  psd <- magmaan(cfa, d, covariance = "psd", inference = FALSE)
  expect_identical(coef(magmaan(cfa, d, covariance = "psd", options = list(start = "fabin3"),
                                inference = FALSE)),
                   coef(psd))
  expect_identical(as_lab_fit(psd)$ml_start_policy, "transported-std-lv-fabin")
  expect_error(magmaan(cfa, d, options = list(start = "simple")), "must be one of")
  m <- magmaan_model(cfa, prototype = ordinal_hs(), ordered = paste0("x", 1:6))
  expect_error(magmaan(m, ordinal_hs(), estimator = "DWLS", options = list(start = "fabin3")),
               "not available for ordered variables")
})

test_that("start = a fit or a parameter table sets the start values", {
  d <- hs()
  fit <- magmaan(cfa, d, inference = FALSE)
  again <- magmaan(cfa, d, options = list(start = fit), inference = FALSE)
  expect_equal(as_lab_fit(again)$start$theta, unname(coef(fit)), tolerance = 1e-12)
  expect_equal(coef(again), coef(fit), tolerance = 1e-6)
  table <- coef(summary(fit))
  table$est[table$lhs == "visual" & table$rhs == "x2"] <- 0.3
  partial <- magmaan(cfa, d, options = list(start = table[table$op == "=~", ]),
                     inference = FALSE)
  x2 <- which(names(coef(fit)) == "visual=~x2")
  expect_equal(as_lab_fit(partial)$start$theta[x2], 0.3)
  psd <- magmaan(cfa, d, covariance = "psd", options = list(start = fit), inference = FALSE)
  expect_equal(as_lab_fit(psd)$start$theta, unname(coef(fit)), tolerance = 1e-12)
  expect_error(magmaan(cfa, d, options = list(start = data.frame(lhs = "visual"))),
               "start table needs")
})

test_that("bundled lavaan estimator names point to the plain estimator", {
  d <- hs()
  expect_error(magmaan(cfa, d, estimator = "MLR"), "Use estimator = \"ML\"")
  expect_error(magmaan(cfa, d, estimator = "WLSMV"), "Use estimator = \"DWLS\"")
  expect_error(magmaan(cfa, d, estimator = "ulsmv"), "Use estimator = \"ULS\"")
  expect_error(magmaan(cfa, d, estimator = "BAYES"), "unknown estimator")
})

test_that("estimator and data type must agree", {
  d <- hs()
  o <- ordinal_hs()
  ordinal <- magmaan_model(cfa, prototype = o, ordered = paste0("x", 1:6))
  expect_error(magmaan(cfa, d, estimator = "DWLS"), "declare them with magmaan_model")
  expect_error(magmaan(ordinal, o), "treats every variable as continuous")
  expect_error(magmaan(cfa, d, estimator = "WLS"), "continuous WLS")
  expect_error(magmaan_model(cfa, identification = "sphere"), "planned")
  expect_error(magmaan(cfa, as.matrix(d[paste0("x", 1:6)])), "data frame")
})

test_that("rows with missing values are deleted listwise and reported", {
  d <- hs()
  d$x1[1:5] <- NA
  fit <- magmaan(cfa, d)
  expect_equal(fit$rows$used, 296L)
  expect_equal(fit$rows$deleted, 5L)
  expect_output(print(fit), "296 used of 301 rows; 5 deleted listwise")
  fiml <- magmaan(cfa, d, estimator = "FIML")
  expect_equal(fiml$rows$used, 301L)
})

test_that("inference = FALSE defers the same policy to infer()", {
  d <- hs()
  later <- magmaan(cfa, d, inference = FALSE)
  expect_null(later$inference)
  err <- tryCatch(vcov(later), magmaan_inference_unavailable = function(e) e)
  expect_equal(err$reason, "not_computed")
  now <- infer(later)
  direct <- magmaan(cfa, d)
  expect_equal(vcov(now), vcov(direct))
  expect_equal(now$inference$global_lr, direct$inference$global_lr)
  expect_equal(unname(coef(now)), unname(coef(later)))
})

test_that("unsupported estimators keep their estimates and give a reason", {
  o <- ordinal_hs()
  m <- magmaan_model(cfa, prototype = o, ordered = paste0("x", 1:6))
  fit <- magmaan(m, o, estimator = "ULS")
  expect_true(all(fit$inference$status$reason == "unsupported_model"))
  err <- tryCatch(confint(fit), magmaan_inference_unavailable = function(e) e)
  expect_equal(err$reason, "unsupported_model")
  expect_true(all(is.na(coef(summary(fit))$se)))
  expect_output(print(summary(fit)), "Unavailable inference")
})

test_that("all-ordinal DWLS gets the estimated-weight covariance and one global test", {
  o <- ordinal_hs()
  m <- magmaan_model(cfa, prototype = o, ordered = paste0("x", 1:6))
  fit <- magmaan(m, o, estimator = "DWLS")
  s <- fit$inference$status
  expect_equal(s$available, c(TRUE, TRUE, FALSE))
  expect_equal(s$reason[s$component == "global_lr"], "inapplicable")
  lab <- as_lab_fit(fit)
  expect_equal(unname(vcov(fit)), unname(vcov(lab, regime = "sandwich_ij")), tolerance = 1e-12)
  tests <- coef(summary(fit))
  expect_false(anyNA(tests$se[tests$free]))
  g <- summary(fit)$tests
  expect_equal(g$test, "fit function")
  expect_true(is.finite(g$statistic) && g$df > 0)
  out <- capture.output(print(summary(fit)))
  expect_false(any(grepl("Unavailable inference", out)))
  expect_true(any(grepl("Note: DWLS has no likelihood", out)))
  expect_true(any(grepl("inference: +computed$", capture.output(print(fit)))))
})

test_that("anova() compares nested all-ordinal DWLS fits with the fit-function difference", {
  o <- ordinal_hs()
  ord <- paste0("x", 1:6)
  m1 <- magmaan_model(cfa, prototype = o, ordered = ord)
  m0 <- magmaan_model(paste(cfa, "visual =~ x1 + a*x2 + a*x3", sep = "\n"),
                      prototype = o, ordered = ord)
  f1 <- magmaan(m1, o, estimator = "DWLS")
  f0 <- magmaan(m0, o, estimator = "DWLS")
  a <- anova(f0, f1)
  expect_equal(a$test, c("score", "fit-function difference"))
  expect_equal(a$df[2], 1L)
  expect_true(is.finite(a$p.sb[2]) && is.finite(a$p.peba4[2]))
  expect_true(is.na(a$statistic[1]))
  lab <- magmaanlab::policy_nested(as_lab_fit(f1), as_lab_fit(f0))
  expect_equal(a$statistic[2], lab$lr$statistic, tolerance = 1e-12)
  expect_equal(lab$lr$label, "fit_function_difference")
  expect_equal(lab$score$reason, "unsupported_model")
  expect_output(print(a), "fit-function difference")
  # No likelihood-ratio row, so no likelihood-ratio caveat.
  expect_false(any(grepl("score test is primary", capture.output(print(a)))))
})

test_that("ordered variables use the categorical estimators", {
  o <- ordinal_hs()
  ord <- paste0("x", 1:6)
  m <- magmaan_model(cfa, prototype = o, ordered = ord)
  fit <- magmaan(m, o, estimator = "DWLS")
  expect_true(isTRUE(as_lab_fit(fit)$converged))
  expect_lavaan_estimates(fit, lavaan::cfa(cfa, o, ordered = ord, estimator = "DWLS"),
                          tolerance = 1e-3)
  # Grouped, from a zero-row schema whose factor levels declare everything.
  grouped <- magmaan_model(cfa, prototype = o[0, c(ord, "school")], ordered = ord,
                           group = "school")
  fit <- magmaan(grouped, o, estimator = "DWLS")
  expect_lavaan_estimates(fit, lavaan::cfa(cfa, o, ordered = ord, estimator = "DWLS",
                                           group = "school"), tolerance = 1e-3)
})

test_that("covariance = \"psd\" fits through the PSD-constrained estimator", {
  d <- hs()
  fit <- magmaan(cfa, d, covariance = "psd")
  expect_identical(fit$covariance$policy, "psd")
  expect_true(isTRUE(as_lab_fit(fit)$options$psd))
  expect_equal(unname(coef(fit)), unname(coef(magmaan(cfa, d))), tolerance = 1e-4)
  expect_output(print(fit), "PSD-constrained")
})

test_that("ML covariance is lavaan's observed-information sandwich", {
  d <- hs()
  fit <- magmaan(cfa, d)
  expect_equal(fit$inference$status$available, c(TRUE, TRUE, TRUE))
  lav <- lav_cfa(cfa, d, estimator = "MLR")
  expect_lavaan_se(fit, lav)
  p <- coef(summary(fit))
  pt <- fit$lab$partable
  free <- pt$free[pt$free > 0L & !pt$op %in% c("==", "<", ">")]
  expect_equal(p$se[p$free], unname(sqrt(diag(vcov(fit))))[free])
  ci <- confint(fit)
  expect_equal(unname(ci[, 2] - ci[, 1]), unname(2 * stats::qnorm(0.975) * sqrt(diag(vcov(fit)))))
})

test_that("the sandwich uses the fitted means, as lavaan's MLR does", {
  # Scalar invariance: the fitted means differ from the sample means.
  d <- hs()
  eq <- c("loadings", "intercepts")
  fit <- magmaan(magmaan_model(cfa, prototype = d, group = "school", group.equal = eq), d)
  # lavaan orders groups by appearance; the first group is the reference group
  # whose latent means are fixed, so give lavaan the model's order.
  lav <- lavaan::cfa(cfa, d, group = "school", group.equal = eq, estimator = "MLR",
                     group.label = levels(d$school))
  expect_lavaan_se(fit, lav)
})

test_that("anova() nests configural, metric and scalar invariance with released latent means", {
  d <- hs()
  fit <- function(eq) magmaan(magmaan_model(cfa, prototype = d, group = "school",
                                            group.equal = eq), d)
  configural <- fit(character())
  metric <- fit("loadings")
  scalar <- fit(c("loadings", "intercepts"))
  lav <- function(eq) lavaan::cfa(cfa, d, group = "school", group.equal = eq,
                                  group.label = levels(d$school))
  reference <- lavaan::lavTestLRT(lav(character()), lav("loadings"), lav(c("loadings", "intercepts")))
  for (pair in list(list(configural, metric, 2L), list(metric, scalar, 3L))) {
    a <- anova(pair[[2]], pair[[1]])
    expect_equal(a$test, c("score", "likelihood ratio"))
    expect_equal(a$statistic[2], reference[pair[[3]], "Chisq diff"], tolerance = 1e-5)
    expect_equal(a$df[2], reference[pair[[3]], "Df diff"])
    expect_true(is.finite(a$statistic[1]) && is.finite(a$p.sb[1]))
  }
})

test_that("the likelihood-ratio test's SB calibration is lavaan's Satorra-Bentler", {
  d <- hs()
  lr <- magmaan(cfa, d)$inference$global_lr
  fm <- lavaan::fitMeasures(lav_cfa(cfa, d, test = "satorra.bentler"),
                            c("chisq", "df", "chisq.scaling.factor", "pvalue.scaled"))
  expect_equal(lr$statistic, unname(fm["chisq"]), tolerance = 1e-6)
  expect_equal(lr$df, unname(fm["df"]))
  expect_equal(lr$sb_scale, unname(fm["chisq.scaling.factor"]), tolerance = 1e-5)
  expect_equal(lr$p_sb, unname(fm["pvalue.scaled"]), tolerance = 1e-5)
  score <- magmaan(cfa, d)$inference$global_score
  expect_true(is.finite(score$p_peba4) && is.finite(lr$p_peba4))
  expect_output(print(summary(magmaan(cfa, d))), "Global tests against the saturated model")
})

test_that("saturated intercepts change no other estimate, standard error or test", {
  d <- hs()
  fit <- magmaan(cfa, d)
  lab <- magmaanlab::fit_model(cfa, d)
  no_means <- magmaanlab::policy_inference(lab)
  pt <- as_lab_fit(fit)$partable
  expect_setequal(pt$lhs[pt$op == "~1" & pt$free > 0L], paste0("x", 1:6))
  rows <- names(coef(fit))[!grepl("~1$", names(coef(fit)))]
  expect_equal(unname(coef(fit)[rows]), lab$theta, tolerance = 1e-6)
  keep <- match(rows, names(coef(fit)))
  expect_equal(unname(vcov(fit)[keep, keep]), unname(no_means$covariance), tolerance = 1e-6)
  for (test in c("global_lr", "global_score")) {
    reference <- no_means[[if (test == "global_lr") "lr" else "score"]]
    expect_equal(fit$inference[[test]]$statistic, reference$statistic, tolerance = 1e-6)
    expect_equal(fit$inference[[test]]$df, reference$df)
    expect_equal(fit$inference[[test]]$p_sb, reference$p_sb, tolerance = 1e-6)
    expect_equal(fit$inference[[test]]$p_peba4, reference$p_peba4, tolerance = 1e-6)
  }
})

test_that("observed covariates are random, with policy inference", {
  d <- hs()
  fit <- magmaan("x1 ~ x2 + x3", d)
  expect_true(all(fit$inference$status$available[1]))
  p <- coef(summary(fit))
  expect_true(all(p$free[p$lhs %in% c("x2", "x3") & p$op == "~~"]))
  # The ML regression coefficients are those of the fixed-x fit.
  fixed <- magmaanlab::fit_model("x1 ~ x2 + x3", d)
  expect_equal(unname(coef(fit)[c("x1~x2", "x1~x3")]), fixed$theta[1:2], tolerance = 1e-6)
  expect_lavaan_se(fit, lavaan::sem("x1 ~ x2 + x3", d, meanstructure = TRUE,
                                    fixed.x = FALSE, estimator = "MLR"))
  # In an overidentified model, ML structural estimates are those of the
  # fixed-x fit but least-squares estimates are not (project/scope.md).
  latent <- "visual =~ x1 + x2 + x3\nvisual ~ x4 + x5"
  paths <- c("visual~x4", "visual~x5")
  fixed_est <- function(fit) fit$partable$est[fit$partable$op == "~"]
  ml <- magmaan(latent, d, inference = FALSE)
  expect_equal(unname(coef(ml)[paths]), fixed_est(magmaanlab::fit_model(latent, d)),
               tolerance = 1e-5)
  gls <- magmaan(latent, d, estimator = "GLS", inference = FALSE)
  # lavaan's GLS uses the N - 1 sample covariance, which rescales variances
  # but not these paths.
  lav_gls <- lavaan::sem(latent, d, estimator = "GLS", meanstructure = TRUE, fixed.x = FALSE)
  expect_equal(unname(coef(gls)[paths]), unname(lavaan::coef(lav_gls)[paths]), tolerance = 1e-5)
  fixed_gls <- magmaanlab::fit_model(latent, d, estimator = "GLS")
  expect_gt(max(abs(unname(coef(gls)[paths]) - fixed_est(fixed_gls))), 1e-3)
})

test_that("defined parameters use the policy covariance", {
  d <- hs()
  m <- "visual =~ x1 + a*x2 + b*x3\ntextual =~ x4 + x5 + x6\nab := a*b"
  p <- coef(summary(magmaan(m, d)))
  lav <- lavaan::parameterEstimates(lav_cfa(m, d, estimator = "MLR"))
  expect_equal(p$est[p$op == ":="], lav$est[lav$op == ":="], tolerance = 1e-5)
  expect_equal(p$se[p$op == ":="], lav$se[lav$op == ":="], tolerance = 1e-4)
})

test_that("saturated models have a covariance but no global test", {
  fit <- magmaan("visual =~ x1 + x2 + x3", hs())
  expect_equal(fit$inference$status$reason, c("available", "saturated", "saturated"))
  expect_equal(dim(vcov(fit)), c(9L, 9L))
})

test_that("PSD fits on the cone boundary get inference for an interior population", {
  set.seed(3)
  n <- 150
  f <- stats::rnorm(n)
  d <- data.frame(y1 = f + stats::rnorm(n, 0, 0.01), y2 = 0.5 * f + stats::rnorm(n),
                  y3 = 0.4 * f + stats::rnorm(n), y4 = 0.6 * f + stats::rnorm(n))
  m <- "F =~ y1 + y2 + y3 + y4"
  boundary <- suppressWarnings(magmaan(m, d, covariance = "psd"))
  expect_false(isTRUE(as_lab_fit(boundary)$diagnostics$newton_accuracy$covariance_interior))
  expect_true(all(boundary$inference$status$available))
  expect_true(boundary$inference$psd_boundary)
  expect_output(print(summary(boundary)), "assumes the population is interior")
  # The same components as the explicit lab composition at that estimate.
  lab <- magmaanlab::policy_inference(as_lab_fit(boundary))
  expect_equal(unname(vcov(boundary)), unname(lab$covariance))
  expect_true(lab$psd_boundary)
  interior <- magmaan(cfa, hs(), covariance = "psd")
  expect_true(all(interior$inference$status$available))
  expect_false(interior$inference$psd_boundary)
})

test_that("fitted() gives lavaan's model-implied moments", {
  d <- hs()
  one <- fitted(magmaan(cfa, d, inference = FALSE))
  lav <- lavaan::fitted(lav_cfa(cfa, d))
  expect_equal(one$cov, unclass(lav$cov), tolerance = 1e-5, ignore_attr = TRUE)
  expect_equal(one$mean, unclass(lav$mean), tolerance = 1e-5, ignore_attr = TRUE)
  m <- magmaan_model(cfa, prototype = d, group = "school")
  mg <- fitted(magmaan(m, d, inference = FALSE))
  lav <- lavaan::fitted(lav_cfa(cfa, d, group = "school"))
  expect_setequal(names(mg), names(lav))
  expect_equal(mg[["Pasteur"]]$mean, unclass(lav[["Pasteur"]]$mean), tolerance = 1e-5,
               ignore_attr = TRUE)
  expect_equal(rownames(mg[[1]]$cov), paste0("x", 1:6))
})

test_that("a fully specified model gives its population moments", {
  pop <- "f =~ 1*x1 + 0.8*x2 + 0.6*x3\nf ~~ 1*f\nx1 ~~ 1*x1\nx2 ~~ 1*x2\nx3 ~~ 1*x3\nx1 ~ 4*1\nx2 ~ 6*1\nx3 ~ 2*1"
  fit <- magmaan(pop, hs())
  expect_true(as_lab_fit(fit)$converged)
  expect_length(coef(fit), 0L)
  m <- fitted(fit)
  expect_equal(unname(m$cov), matrix(c(2, .8, .6, .8, 1.64, .48, .6, .48, 1.36), 3))
  expect_equal(unname(m$mean), c(4, 6, 2))
  expect_equal(unname(m$cov),
               unname(unclass(lavaan::fitted(lavaan::sem(pop, hs(), meanstructure = TRUE))$cov)),
               ignore_attr = TRUE)
  expect_equal(fit$inference$status$reason, rep("available", 3L))
  expect_equal(dim(vcov(fit)), c(0L, 0L))
  expect_equal(fit$inference$global_score$df, 9L)
  expect_equal(fit$inference$global_lr$df, 9L)
  # lavaan suppresses tests when no parameters are free; its ML objective
  # still supplies the independently fitted LR discrepancy (2 N f_min).
  lav <- lavaan::sem(pop, hs(), meanstructure = TRUE)
  expect_equal(fit$inference$global_lr$statistic,
               2 * nrow(hs()) * lavaan::lavInspect(lav, "optim")$fx,
               tolerance = 1e-8)
  # No fitted parameter directions: use the full saturated-normal score,
  # including the covariance contribution from the fixed-mean discrepancy.
  x <- as.matrix(hs()[, paste0("x", 1:3)])
  z <- sweep(x, 2, colMeans(x))
  sample <- crossprod(z) / nrow(x)
  shift <- colMeans(x) - m$mean
  inverse <- solve(m$cov)
  error <- sample + tcrossprod(shift) - m$cov
  score <- nrow(x) * (sum(shift * (inverse %*% shift)) +
                      .5 * sum(diag(inverse %*% error %*% inverse %*% error)))
  expect_equal(fit$inference$global_score$statistic, score, tolerance = 1e-8)
  expect_true(is.finite(fit$inference$global_score$p_sb))
  expect_true(is.finite(fit$inference$global_lr$p_peba4))
  deferred <- infer(magmaan(pop, hs(), inference = FALSE))
  expect_equal(deferred$inference, fit$inference)
})

test_that("anova() gives nested LR and score tests with SB and PEBA4", {
  d <- hs()
  m1 <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9"
  m0 <- "visual =~ x1 + a*x2 + a*x3\ntextual =~ x4 + b*x5 + b*x6\nspeed =~ x7 + x8 + x9"
  f1 <- magmaan(m1, d)
  f0 <- magmaan(m0, d)
  a <- anova(f0, f1)
  expect_s3_class(a, "magmaan_anova")
  expect_equal(attr(a, "restricted"), "f0")
  expect_equal(unclass(anova(f1, f0)), unclass(a), ignore_attr = TRUE)
  expect_equal(a$df, c(2L, 2L))
  # Both tests are the lab's hypothesis quadratics in the observed nested
  # geometry, calibrated explicitly.
  shared <- magmaanlab::prepare_inference_data(as_lab_fit(f1))
  h <- magmaanlab::prepare_hypothesis(magmaanlab::prepare_inference(as_lab_fit(f0), shared),
                                      magmaanlab::prepare_inference(as_lab_fit(f1), shared))
  for (k in 1:2) {
    test <- c("score", "lr")[k]
    cal <- magmaanlab::calibrate_quadratic(
      magmaanlab::inference_quadratic(h, test, geometry = "observed"), c("sb", "peba4"))
    expect_equal(a$statistic[k], cal$statistic[1], tolerance = 1e-10)
    expect_equal(c(a$p.sb[k], a$p.peba4[k]), cal$p_value, tolerance = 1e-10)
  }
  # The LR statistic is the normal-theory difference. In the expected
  # geometry the lab's SB is lavaan's Satorra (2000) with the exact
  # restriction map; the policy's observed geometry is not a lavaan method.
  nt <- lavaan::lavTestLRT(lav_cfa(m0, d), lav_cfa(m1, d))
  expect_equal(a$test, c("score", "likelihood ratio"))
  expect_equal(a$statistic[2], as.numeric(nt[2, "Chisq diff"]), tolerance = 1e-6)
  sb <- lavaan::lavTestLRT(lav_cfa(m0, d, estimator = "MLM"), lav_cfa(m1, d, estimator = "MLM"),
                           method = "satorra.2000", A.method = "exact", scaled.shifted = FALSE)
  lab_sb <- magmaanlab::calibrate_quadratic(magmaanlab::inference_quadratic(h, "lr"), "sb")
  expect_equal(lab_sb$p_value, as.numeric(sb[2, "Pr(>Chisq)"]), tolerance = 1e-5)
  expect_output(print(a), "Nested tests of f0 \\(restricted\\) against f1")
  expect_output(print(a), "The score test is primary")
  expect_output(print(summary(f1)), "The score test is primary")
})

test_that("confint() intervals are Wald, with likelihood-ratio inversion planned", {
  fit <- magmaan(cfa, hs())
  expect_identical(confint(fit, test = "wald"), confint(fit))
  expect_error(confint(fit, test = "lr"), "confint\\(\\): test = \"lr\" is planned")
  expect_error(confint(fit, test = "score"), "must be one of \"wald\"")
})

test_that("anova() refuses pairs it cannot compare", {
  d <- hs()
  f1 <- magmaan(cfa, d, inference = FALSE)
  other <- magmaan("visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nvisual ~~ 0*textual", d,
                   inference = FALSE)
  fixed <- anova(f1, other)
  expect_equal(fixed$df, c(1L, 1L))
  expect_true(all(is.finite(fixed$p.sb)))
  expect_error(anova(f1, magmaan(cfa, d[-1, ], inference = FALSE)), "same observations")
  expect_error(anova(f1, magmaan(cfa, d, covariance = "psd", inference = FALSE)),
               "covariance policy")
  expect_error(anova(f1), "exactly two")
  # A restriction written as a constraint on a labeled parameter is nested.
  labeled <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nvisual ~~ c*textual"
  zero <- paste(labeled, "c == 0", sep = "\n")
  z <- anova(magmaan(zero, d, inference = FALSE), magmaan(labeled, d, inference = FALSE))
  expect_equal(z$df, c(1L, 1L))
  expect_true(all(is.finite(z$p.sb)))
})
