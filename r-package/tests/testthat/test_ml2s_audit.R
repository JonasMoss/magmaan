ml2s_audit_example <- function() {
  set.seed(20261006)
  n <- 130L
  eta <- rnorm(n)
  X <- outer(eta, c(.85, .8, .7, .9)) + matrix(rnorm(n * 4L), n, 4L)
  X <- sweep(X, 2L, c(2, -1, .5, 4), "+")
  colnames(X) <- paste0("x", 1:4)
  list(spec = magmaanlab::model_spec("f =~ x1 + x2 + x3 + x4",
    fixed_x = FALSE, meanstructure = TRUE), raw = list(X = list(X)))
}

test_that("ML2S captures the fitting input and serialized audits own their artifacts", {
  ex <- ml2s_audit_example()
  core <- magmaanlab::magmaan_core
  sm <- core$estimate_saturated_em_moments(ex$raw)
  expect_error(core$estimate_saturated_em_moments(ex$raw, h_step = 0), "h_step must be > 0")
  fit <- core$fit_ml2s(ex$spec, ex$raw)
  reused <- core$fit_ml2s(ex$spec, ex$raw, stage1 = sm)
  expect_identical(fit$theta, reused$theta)
  expect_identical(fit$fmin, reused$fmin)
  expect_identical(fit$stage1, reused$stage1)
  expect_identical(fit$stage2_input$moments$cov, sm$cov)
  sm$cov[[1L]][1L, 1L] <- 10
  expect_false(identical(fit$stage2_input$moments$cov, sm$cov))
  audit <- core$frontier_ml2s_convergence_audit(unserialize(serialize(fit, NULL)))
  expect_identical(audit$status, "passed")
  expect_identical(audit$handoff$status, "passed")
  expect_identical(audit$solver_stop$status, "passed")
  rm(fit, reused)
  invisible(gc())
  expect_true(all(is.finite(audit$stage1$hessian)))
  expect_true(all(is.finite(audit$stage2$hessian)))
})

test_that("ML2S stages cannot supply a pass for absent or mismatched handoff evidence", {
  ex <- ml2s_audit_example()
  core <- magmaanlab::magmaan_core
  fit <- core$fit_ml2s(ex$spec, ex$raw, stage2_weight = "dls")
  wrong <- fit
  wrong$stage2_input$weight_blocks[[1L]][1L, 1L] <-
    2 * wrong$stage2_input$weight_blocks[[1L]][1L, 1L]
  audit <- core$frontier_ml2s_convergence_audit(wrong)
  expect_identical(audit$stage1_assessment$status, "passed")
  expect_identical(audit$stage2_assessment$status, "passed")
  expect_identical(audit$handoff$status, "failed")
  expect_identical(audit$status, "failed")
  missing <- fit
  missing$stage2_input$moments$acov <- NULL
  audit <- core$frontier_ml2s_convergence_audit(missing)
  expect_identical(audit$handoff$status, "unchecked")
  expect_identical(audit$status, "unchecked")
  legacy <- fit
  legacy$stage2_input <- NULL
  legacy$stage1 <- legacy$stage1[c("mean", "cov", "n_obs", "H", "J", "acov", "warnings")]
  audit <- core$frontier_ml2s_convergence_audit(legacy)
  expect_identical(audit$status, "unchecked")
  expect_identical(audit$solver_stop$status, "unchecked")
  expect_false(audit$stage1$value_recorded)
})
