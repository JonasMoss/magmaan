# Exact-fit raw data with x1 scaled by 1000 and x3 by 1/1000. lavaan 0.7.2
# (and the pinned preset) accepts a non-stationary endpoint here with
# fmin = 0.276; magmaan's own convergence check rejects it.
.disagreement_data <- function() {
  lambda <- c(1, .8, .6, .9)
  s0 <- tcrossprod(lambda) + diag(.7, 4)
  dk <- diag(c(1000, 1, 1 / 1000, 1))
  s <- dk %*% s0 %*% dk
  n <- 200
  set.seed(1)
  z <- scale(matrix(stats::rnorm(n * 4), n), scale = FALSE)
  z <- z %*% solve(chol(crossprod(z) / n))
  d <- as.data.frame(z %*% chol(s))
  names(d) <- paste0("x", 1:4)
  d
}
.printed <- function(x) paste(utils::capture.output(print(x)), collapse = "\n")

test_that("a lavaan-accepted fit that magmaan's check rejects carries a note", {
  d <- .disagreement_data()
  m <- "f =~ x1+x2+x3+x4"
  fit <- suppressWarnings(magmaan(m, d, options = list(preset = "lavaan-0.7.2")))
  conv <- fit$inference$convergence
  expect_identical(conv$rule, "lavaan-0.7.2")
  expect_true(conv$converged)
  expect_identical(conv$magmaan, "failed")
  expect_true(conv$disagree)
  # Inference follows the selected rule, as lavaan would report it.
  expect_false(any(fit$inference$status$reason == "not_converged"))
  expect_match(.printed(fit), "yes, by the lavaan-0.7.2 rule (magmaan's check: failed)", fixed = TRUE)
  expect_match(.printed(fit), "see the convergence note", fixed = TRUE)
  expect_match(.printed(summary(fit)), "rule accepted this fit, but magmaan's convergence", fixed = TRUE)

  plain <- suppressWarnings(magmaan(m, d))
  expect_identical(plain$inference$convergence$rule, "newton")
  expect_false(plain$inference$convergence$disagree)
  expect_no_match(.printed(summary(plain)), "convergence note|rule accepted")

  # anova() reports the disagreement when either fit has it.
  wider <- suppressWarnings(magmaan("f =~ x1+x2+x3+x4\nx2 ~~ x4", d,
      options = list(preset = "lavaan-0.7.2")))
  expect_match(.printed(anova(fit, wider)), "convergence check disagree", fixed = TRUE)
})

test_that("a fit the rule rejects but magmaan's check accepts carries the mirror note", {
  d <- lavaan::HolzingerSwineford1939
  fit <- magmaan("f =~ x1+x2+x3+x4", d, inference = FALSE,
                 options = list(preset = "lavaan-0.7.2"))
  expect_identical(as_lab_fit(fit)$diagnostics$verdict$status, "passed")
  # Construct the reverse verdict: the selected rule rejects the same point.
  fit$lab$converged <- FALSE
  fit$lab$verdict$status <- "failed"
  fit <- infer(fit)
  expect_true(fit$inference$convergence$disagree)
  expect_true(all(fit$inference$status$reason == "not_converged"))
  expect_match(.printed(fit), "no, by the lavaan-0.7.2 rule (magmaan's check: passed)", fixed = TRUE)
  expect_match(.printed(summary(fit)), "rule rejected this fit, but magmaan's convergence", fixed = TRUE)
})
