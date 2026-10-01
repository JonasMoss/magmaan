hs <- function() {
  skip_if_not_installed("lavaan")
  lavaan::HolzingerSwineford1939
}
cfa <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"

# Same parameter rows as lavaan, and the same free estimates, aligned by
# (lhs, op, rhs, group).
expect_lavaan_estimates <- function(fit, lav, tolerance = 1e-4) {
  ours <- fit$lab$partable
  theirs <- lavaan::parTable(lav)
  key <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group)
  keep <- function(pt) pt[pt$op != "==", , drop = FALSE]
  expect_setequal(key(keep(ours)), key(keep(theirs)))
  theirs <- theirs[theirs$free > 0L, , drop = FALSE]
  idx <- match(key(theirs), key(ours))
  expect_false(anyNA(idx))
  expect_equal(ours$est[idx], theirs$est, tolerance = tolerance)
}

# Standard errors of every lavaan free row, aligned by (lhs, op, rhs, group).
expect_lavaan_se <- function(fit, lav, tolerance = 1e-4) {
  p <- coef(summary(fit))
  theirs <- lavaan::parTable(lav)
  theirs <- theirs[theirs$free > 0L, , drop = FALSE]
  group <- if (is.null(p$group)) 1L else p$group
  idx <- match(paste(theirs$lhs, theirs$op, theirs$rhs, theirs$group),
               paste(p$lhs, p$op, p$rhs, group))
  expect_false(anyNA(idx))
  expect_equal(p$se[idx], theirs$se, tolerance = tolerance)
}

test_that("ML estimates come from magmaanlab and match lavaan", {
  d <- hs()
  fit <- magmaan(cfa, d)
  lab <- magmaanlab::fit_model(cfa, d)
  expect_s3_class(fit, "magmaan")
  expect_s3_class(as_lab_fit(fit), "magmaan_fit")
  expect_equal(unname(coef(fit)), lab$theta)
  expect_equal(nobs(fit), 301)
  expect_lavaan_estimates(fit, lavaan::cfa(cfa, d))
})

test_that("identification and group options follow lavaan's meaning", {
  d <- hs()
  expect_lavaan_estimates(magmaan(cfa, d, identification = "std.lv"),
                          lavaan::cfa(cfa, d, std.lv = TRUE))
  fit <- magmaan(cfa, d, group = "school", group.equal = "loadings")
  expect_lavaan_estimates(fit, lavaan::cfa(cfa, d, group = "school",
                                           group.equal = "loadings"))
  expect_equal(fit$rows$group, unique(as.character(d$school)))
  expect_true("group" %in% names(coef(summary(fit))))
  intercept <- "visual =~ x1 + x2 + x3\n x1 ~ 1"
  expect_lavaan_estimates(magmaan(intercept, d), lavaan::cfa(intercept, d))
})

test_that("start = \"fabin3\" gives the former FABIN3 start", {
  d <- hs()
  fit <- magmaan(cfa, d, inference = FALSE)
  old <- magmaan(cfa, d, start = "fabin3", inference = FALSE)
  expect_identical(as_lab_fit(fit)$ml_start_policy, "layered")
  expect_identical(as_lab_fit(old)$ml_start_policy, "transported-std-lv-fabin")
  lab <- magmaanlab::fit_model(cfa, d, control = list(start = "scaled-fabin"))
  expect_identical(unname(coef(old)), lab$theta)
  expect_equal(coef(old), coef(fit), tolerance = 1e-4)
  gls <- magmaan(cfa, d, estimator = "GLS", start = "fabin3", inference = FALSE)
  expect_identical(unname(coef(gls)),
                   magmaanlab::fit_model(cfa, d, estimator = "GLS",
                                         control = list(start = "fabin3"))$theta)
  psd <- magmaan(cfa, d, psd = TRUE, inference = FALSE)
  expect_identical(coef(magmaan(cfa, d, psd = TRUE, start = "fabin3", inference = FALSE)),
                   coef(psd))
  expect_identical(as_lab_fit(psd)$ml_start_policy, "transported-std-lv-fabin")
  expect_error(magmaan(cfa, d, start = "simple"), "must be one of")
  expect_error(magmaan(cfa, d, start = "fabin3", cluster = "school"),
               "not available with `ordered` or `cluster`")
})

test_that("start = a fit or a parameter table sets the start values", {
  d <- hs()
  fit <- magmaan(cfa, d, inference = FALSE)
  again <- magmaan(cfa, d, start = fit, inference = FALSE)
  expect_equal(as_lab_fit(again)$start$theta, unname(coef(fit)), tolerance = 1e-12)
  expect_equal(coef(again), coef(fit), tolerance = 1e-6)
  table <- coef(summary(fit))
  table$est[table$lhs == "visual" & table$rhs == "x2"] <- 0.3
  partial <- magmaan(cfa, d, start = table[table$op == "=~", ], inference = FALSE)
  x2 <- which(names(coef(fit)) == "visual=~x2")
  expect_equal(as_lab_fit(partial)$start$theta[x2], 0.3)
  psd <- magmaan(cfa, d, psd = TRUE, start = fit, inference = FALSE)
  expect_equal(as_lab_fit(psd)$start$theta, unname(coef(fit)), tolerance = 1e-12)
  expect_error(magmaan(cfa, d, start = data.frame(lhs = "visual")), "start table needs")
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
  expect_error(magmaan(cfa, d, estimator = "DWLS"), "declare them with `ordered")
  expect_error(magmaan(cfa, d, ordered = "x1"), "treats every variable as continuous")
  expect_error(magmaan(cfa, d, estimator = "WLS"), "continuous WLS")
  expect_error(magmaan(cfa, d, missing = "pairwise"), "for ordered variables")
  expect_error(magmaan(cfa, d, identification = "sphere"), "planned")
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
  d <- hs()
  for (v in paste0("x", 1:6)) {
    d[[v]] <- cut(d[[v]], breaks = stats::quantile(d[[v]], c(0, 1 / 3, 2 / 3, 1)),
                  include.lowest = TRUE, labels = FALSE)
  }
  fit <- magmaan(cfa, d, estimator = "DWLS", ordered = paste0("x", 1:6))
  expect_true(all(fit$inference$status$reason == "unsupported_model"))
  err <- tryCatch(confint(fit), magmaan_inference_unavailable = function(e) e)
  expect_equal(err$reason, "unsupported_model")
  expect_true(all(is.na(coef(summary(fit))$se)))
  expect_output(print(summary(fit)), "Unavailable inference")
})

test_that("ordered variables use the categorical estimators", {
  d <- hs()
  for (v in paste0("x", 1:6)) {
    d[[v]] <- cut(d[[v]], breaks = stats::quantile(d[[v]], c(0, 1 / 3, 2 / 3, 1)),
                  include.lowest = TRUE, labels = FALSE)
  }
  ord <- paste0("x", 1:6)
  fit <- magmaan(cfa, d, estimator = "DWLS", ordered = ord)
  expect_true(isTRUE(as_lab_fit(fit)$converged))
  expect_lavaan_estimates(fit, lavaan::cfa(cfa, d, ordered = ord, estimator = "DWLS"),
                          tolerance = 1e-3)
})

test_that("psd = TRUE fits through the PSD-constrained estimator", {
  d <- hs()
  fit <- magmaan(cfa, d, psd = TRUE)
  expect_true(fit$psd)
  expect_true(isTRUE(as_lab_fit(fit)$options$psd))
  expect_equal(unname(coef(fit)), unname(coef(magmaan(cfa, d))), tolerance = 1e-4)
  expect_output(print(fit), "PSD-constrained")
})

test_that("ML covariance is lavaan's observed-information sandwich", {
  d <- hs()
  fit <- magmaan(cfa, d)
  expect_equal(fit$inference$status$available, c(TRUE, TRUE, TRUE))
  lav <- lavaan::cfa(cfa, d, estimator = "MLR")
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
  fit <- magmaan(cfa, d, group = "school", group.equal = eq)
  lav <- lavaan::cfa(cfa, d, group = "school", group.equal = eq, estimator = "MLR")
  expect_lavaan_se(fit, lav)
})

test_that("the likelihood-ratio test's SB calibration is lavaan's Satorra-Bentler", {
  d <- hs()
  lr <- magmaan(cfa, d)$inference$global_lr
  fm <- lavaan::fitMeasures(lavaan::cfa(cfa, d, test = "satorra.bentler"),
                            c("chisq", "df", "chisq.scaling.factor", "pvalue.scaled"))
  expect_equal(lr$statistic, unname(fm["chisq"]), tolerance = 1e-6)
  expect_equal(lr$df, unname(fm["df"]))
  expect_equal(lr$sb_scale, unname(fm["chisq.scaling.factor"]), tolerance = 1e-5)
  expect_equal(lr$p_sb, unname(fm["pvalue.scaled"]), tolerance = 1e-5)
  score <- magmaan(cfa, d)$inference$global_score
  expect_true(is.finite(score$p_peba4) && is.finite(lr$p_peba4))
  expect_output(print(summary(magmaan(cfa, d))), "Global tests against the saturated model")
})

test_that("defined parameters use the policy covariance", {
  d <- hs()
  m <- "visual =~ x1 + a*x2 + b*x3\ntextual =~ x4 + x5 + x6\nab := a*b"
  p <- coef(summary(magmaan(m, d)))
  lav <- lavaan::parameterEstimates(lavaan::cfa(m, d, estimator = "MLR"))
  expect_equal(p$est[p$op == ":="], lav$est[lav$op == ":="], tolerance = 1e-5)
  expect_equal(p$se[p$op == ":="], lav$se[lav$op == ":="], tolerance = 1e-4)
})

test_that("saturated models have a covariance but no global test", {
  fit <- magmaan("visual =~ x1 + x2 + x3", hs())
  expect_equal(fit$inference$status$reason, c("available", "saturated", "saturated"))
  expect_equal(dim(vcov(fit)), c(6L, 6L))
})

test_that("PSD fits on the cone boundary get inference for an interior population", {
  set.seed(3)
  n <- 150
  f <- stats::rnorm(n)
  d <- data.frame(y1 = f + stats::rnorm(n, 0, 0.01), y2 = 0.5 * f + stats::rnorm(n),
                  y3 = 0.4 * f + stats::rnorm(n), y4 = 0.6 * f + stats::rnorm(n))
  m <- "F =~ y1 + y2 + y3 + y4"
  boundary <- suppressWarnings(magmaan(m, d, psd = TRUE))
  expect_false(isTRUE(as_lab_fit(boundary)$diagnostics$newton_accuracy$covariance_interior))
  expect_true(all(boundary$inference$status$available))
  expect_true(boundary$inference$psd_boundary)
  expect_output(print(summary(boundary)), "assumes the population is interior")
  # The same components as the explicit lab composition at that estimate.
  lab <- magmaanlab::policy_inference(as_lab_fit(boundary))
  expect_equal(unname(vcov(boundary)), unname(lab$covariance))
  expect_true(lab$psd_boundary)
  interior <- magmaan(cfa, hs(), psd = TRUE)
  expect_true(all(interior$inference$status$available))
  expect_false(interior$inference$psd_boundary)
})

test_that("fitted() gives lavaan's model-implied moments", {
  d <- hs()
  expect_equal(fitted(magmaan(cfa, d, inference = FALSE))$cov,
               unclass(lavaan::fitted(lavaan::cfa(cfa, d))$cov), tolerance = 1e-5,
               ignore_attr = TRUE)
  mg <- fitted(magmaan(cfa, d, group = "school", meanstructure = TRUE, inference = FALSE))
  lav <- lavaan::fitted(lavaan::cfa(cfa, d, group = "school", meanstructure = TRUE))
  expect_equal(names(mg), names(lav))
  expect_equal(mg[[2]]$mean, unclass(lav[[2]]$mean), tolerance = 1e-5, ignore_attr = TRUE)
  expect_equal(rownames(mg[[1]]$cov), paste0("x", 1:6))
})

test_that("a fully specified model gives its population moments", {
  pop <- "f =~ 1*x1 + 0.8*x2 + 0.6*x3\nf ~~ 1*f\nx1 ~~ 1*x1\nx2 ~~ 1*x2\nx3 ~~ 1*x3\nx1 ~ 4*1\nx2 ~ 6*1\nx3 ~ 2*1"
  fit <- magmaan(pop, hs(), meanstructure = TRUE)
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
  deferred <- infer(magmaan(pop, hs(), meanstructure = TRUE, inference = FALSE))
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
  # LR: the normal-theory difference, and SB is lavaan's Satorra (2000) with
  # the exact restriction map.
  l1 <- lavaan::cfa(m1, d, estimator = "MLM")
  l0 <- lavaan::cfa(m0, d, estimator = "MLM")
  nt <- lavaan::lavTestLRT(lavaan::cfa(m0, d), lavaan::cfa(m1, d))
  expect_equal(a$statistic[1], as.numeric(nt[2, "Chisq diff"]), tolerance = 1e-6)
  sb <- lavaan::lavTestLRT(l0, l1, method = "satorra.2000", A.method = "exact",
                           scaled.shifted = FALSE)
  expect_equal(a$statistic[1] / a$sb.scale[1], as.numeric(sb[2, "Chisq diff"]), tolerance = 1e-6)
  expect_equal(a$p.sb[1], as.numeric(sb[2, "Pr(>Chisq)"]), tolerance = 1e-5)
  # Score: the lab's hypothesis quadratic, calibrated explicitly.
  shared <- magmaanlab::prepare_inference_data(as_lab_fit(f1))
  h <- magmaanlab::prepare_hypothesis(magmaanlab::prepare_inference(as_lab_fit(f0), shared),
                                      magmaanlab::prepare_inference(as_lab_fit(f1), shared))
  cal <- magmaanlab::calibrate_quadratic(magmaanlab::inference_quadratic(h, "score"),
                                         c("sb", "peba4"))
  expect_equal(a$statistic[2], cal$statistic[1], tolerance = 1e-10)
  expect_equal(c(a$p.sb[2], a$p.peba4[2]), cal$p_value, tolerance = 1e-10)
  expect_output(print(a), "Nested tests of f0 \\(restricted\\) against f1")
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
  expect_error(anova(f1, other), "not nested")
  expect_error(anova(f1, magmaan(cfa, d[-1, ], inference = FALSE)), "same observations")
  expect_error(anova(f1, magmaan(cfa, d, psd = TRUE, inference = FALSE)), "psd setting")
  expect_error(anova(f1), "exactly two")
  # A restriction written as a constraint on a labeled parameter is nested.
  labeled <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nvisual ~~ c*textual"
  zero <- paste(labeled, "c == 0", sep = "\n")
  z <- anova(magmaan(zero, d, inference = FALSE), magmaan(labeled, d, inference = FALSE))
  expect_equal(z$df, c(1L, 1L))
  expect_true(all(is.finite(z$p.sb)))
})
