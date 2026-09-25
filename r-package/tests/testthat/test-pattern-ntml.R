pattern_ntml_data <- function(n = 220L, missing = FALSE) {
  lambda <- c(0.8, 0.7, 0.9, 0.75)
  sigma <- tcrossprod(lambda) + diag(1 - lambda^2)
  x <- matrix(rnorm(n * 4L), n, 4L) %*% chol(sigma)
  x <- sweep(x, 2L, c(0.2, -0.1, 0.3, 0), "+")
  colnames(x) <- paste0("y", 1:4)
  if (missing) {
    mask <- matrix(runif(n * 4L) < 0.25, n, 4L)
    mask[rowSums(!mask) == 0L, 1L] <- FALSE
    x[mask] <- NA_real_
  }
  as.data.frame(x)
}

test_that("frontier pattern NTML reduces to FIML on complete normal data", {
  set.seed(77)
  dat <- pattern_ntml_data()
  syntax <- "f =~ y1 + y2 + y3 + y4"
  model <- model_spec(syntax, meanstructure = TRUE)

  pntml <- frontier_fit_pattern_ntml(syntax, dat)
  fiml <- fit_model(model, dat, estimator = "FIML")

  expect_true(pntml$converged)
  expect_equal(pntml$theta, fiml$theta, tolerance = 2e-5)
  expect_identical(pntml$estimator, "PNTML")
  expect_true(pntml$meanstructure)
  expect_identical(pntml$stage2_objective, "pattern_ntml")
  expect_equal(pntml$pntml$eigvals,
               rep(1, length(pntml$pntml$eigvals)), tolerance = 5e-6)
})

test_that("frontier pattern NTML fits MCAR data and reuses Stage 1", {
  set.seed(78)
  dat <- pattern_ntml_data(missing = TRUE)
  model <- model_spec("f =~ y1 + y2 + y3 + y4", meanstructure = TRUE)
  raw <- df_to_fiml_data(dat, model)
  stage1 <- magmaan_core$estimate_saturated_em_moments(raw)
  target_only <- stage1[c("mean", "cov", "n_obs", "warnings")]

  fresh <- frontier_fit_pattern_ntml(model, raw)
  reused <- frontier_fit_pattern_ntml(model, raw, stage1 = target_only)

  expect_true(fresh$converged)
  expect_true(reused$converged)
  expect_equal(reused$theta, fresh$theta, tolerance = 1e-10)
  expect_equal(reused$pntml$eigvals, fresh$pntml$eigvals, tolerance = 1e-10)
  expect_true(all(is.finite(reused$se)))
  expect_true(all(reused$pntml$eigvals > 0))
  expect_true(is.finite(reused$chisq_scaled))
})
