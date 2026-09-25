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
  expect_true("group" %in% names(parameters(fit)))
  intercept <- "visual =~ x1 + x2 + x3\n x1 ~ 1"
  expect_lavaan_estimates(magmaan(intercept, d), lavaan::cfa(intercept, d))
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

test_that("unavailable inference is typed and never substituted", {
  d <- hs()
  later <- magmaan(cfa, d, inference = FALSE)
  expect_null(later$inference)
  err <- tryCatch(vcov(later), magmaan_inference_unavailable = function(e) e)
  expect_equal(err$reason, "not_computed")
  now <- infer(later)
  expect_equal(now$inference$status$component,
               c("covariance", "global_score", "global_lr"))
  err <- tryCatch(confint(now), magmaan_inference_unavailable = function(e) e)
  expect_equal(err$reason, "not_implemented")
  expect_equal(unname(coef(now)), unname(coef(later)))
  p <- parameters(now)
  expect_true(all(is.na(p$se)))
  expect_output(print(summary(now)), "Unavailable inference")
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
