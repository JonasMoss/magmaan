test_that("MI tables exclude identification releases across indicator units", {
  skip_if_not_installed("lavaan")
  set.seed(424242)
  eta <- rnorm(1200)
  x <- sapply(c(1, 0.8, 0.7, 0.9), function(loading) {
    loading * eta + rnorm(length(eta), sd = 0.8)
  })
  colnames(x) <- paste0("x", 1:4)
  syntax <- "f =~ x1+x2+x3+x4"
  key <- function(table) paste(table$lhs, table$op, table$rhs)
  for (missing in c(FALSE, TRUE)) {
    for (units in c(0.1, 1, 10)) {
      d <- as.data.frame(x)
      d$x1 <- units * d$x1
      if (missing) d$x2[seq(1L, nrow(d), 11L)] <- NA_real_
      fit <- fit_model(syntax, d, meanstructure = TRUE,
                       estimator = if (missing) "FIML" else "ML",
                       control = list(max_iter = 1000L, ftol = 1e-12, gtol = 1e-8))
      expect_true(fit$converged)
      oracle <- lavaan::cfa(syntax, d, meanstructure = TRUE,
                            missing = if (missing) "ml" else "listwise")
      expect_true(lavaan::lavInspect(oracle, "converged"))
      mi <- modification_indices(fit, data = d, bread = if (missing) "observed" else "expected",
                                 cov = "model_implied", estimated_weight = FALSE)
      reference <- lavaan::modindices(oracle,
                                      information = if (missing) "observed" else "expected")
      expect_equal(nrow(mi), 6L)
      expect_true(all(mi$op == "~~" & mi$lhs != mi$rhs))
      expect_setequal(key(mi), key(reference))
      expect_equal(mi$mi, reference$mi[match(key(mi), key(reference))], tolerance = 1e-4)
      robust <- modification_indices_robust(fit, data = d, bread = if (missing) "observed" else "expected", estimated_weight = FALSE)
      expect_setequal(key(robust), key(mi))
      if (missing) {
        expect_true(all(robust$reason == "available"))
        expect_equal(robust$mi.scaled, robust$score^2 / robust$v.eff, tolerance = 1e-10)
      } else expect_equal(robust$mi, mi$mi, tolerance = 1e-8)
    }
  }
})
