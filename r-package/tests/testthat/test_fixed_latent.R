test_that("fully fixed latent models return their implied moments", {
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
    fit <- fit_model(model, d, meanstructure = TRUE)
    expect_length(fit$theta, 0)
    expect_true(fit$converged)
    pt <- fit$partable
    expect_true(all(pt$free == 0L))
    expect_equal(pt$est, pt$ustart)
    implied <- magmaan_core$model_implied(fit)
    expect_equal(unname(implied$sigma[[1]]), unname(expected), tolerance = 1e-12)
    expect_equal(as.numeric(implied$mu[[1]]), rep(0, 6))
  }
})
