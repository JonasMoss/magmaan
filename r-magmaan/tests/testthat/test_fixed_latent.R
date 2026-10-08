test_that("fully fixed latent models support fitted moments and deferred inference", {
  set.seed(1299)
  vars <- c("y1", "y2", "y3", "x1", "x2", "x3")
  d <- as.data.frame(matrix(rnorm(600), 100, 6)); names(d) <- vars
  measurement <- "Y =~ 1*y1 + 0.8*y2 + 0.6*y3\nX =~ 1*x1 + 0.8*x2 + 0.6*x3"
  fixed <- paste(c(paste(vars, "~~ 1*", vars), paste(vars, "~ 0*1")), collapse = "\n")
  structures <- c("Y ~~ 1*Y\nX ~~ 1*X\nY ~~ 0*X",
                  "Y ~ 0.25*X\nY ~~ 1*Y\nX ~~ 1*X",
                  "G =~ 0.5*Y + 0.7*X\nG ~~ 1*G\nY ~~ 1*Y\nX ~~ 1*X\nY ~~ 0*X")
  loading <- matrix(0, 6, 2)
  loading[1:3, 1] <- loading[4:6, 2] <- c(1, .8, .6)
  latent <- list(diag(2), matrix(c(1.0625, .25, .25, 1), 2),
                 matrix(c(1.25, .35, .35, 1.49), 2))
  for (i in seq_along(structures)) {
    model <- paste(measurement, structures[i], fixed, sep = "\n")
    expected <- loading %*% latent[[i]] %*% t(loading) + diag(6)
    dimnames(expected) <- list(vars, vars)
    fit <- magmaan(model, d, inference = FALSE)
    expect_length(as_lab_fit(fit)$theta, 0)
    expect_true(as_lab_fit(fit)$converged)
    expect_equal(fitted(fit)$cov, expected, tolerance = 1e-12)
    expect_equal(fitted(fit)$mean, setNames(rep(0, 6), vars))
    pt <- coef(summary(fit))
    expect_true(all(is.finite(pt$est)))
    inferred <- infer(fit)
    expect_identical(inferred$inference$status$reason, rep("available", 3))
    expect_length(vcov(inferred), 0)
    expect_true(all(is.na(coef(summary(inferred))$se)))
    expect_equal(fitted(inferred), fitted(fit))
  }
})
