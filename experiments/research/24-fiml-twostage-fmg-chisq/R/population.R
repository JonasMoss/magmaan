# Single-group population for the FIML / two-stage-ML goodness-of-fit-statistic
# calibration study.
#
# One latent factor, six indicators, mean structure; the fitted one-factor model
# is the correct model, so every goodness-of-fit rejection is a Type-I error.
# That is the whole design: no misspecified arm, no nested test, no power. The
# only thing that varies the reference law is the marginal non-normality (the
# experiment-17 VM/IG/PL families) and the missing-data mechanism/rate. Identical
# structural parameters to the per-group block of experiment 21/23, so the two
# studies are directly comparable; this one strips the second group and the
# invariance ladder.

build_population <- function() {
  list(
    ov     = paste0("x", 1:6),
    lambda = c(1.00, 0.80, 0.90, 0.70, 1.10, 0.85),  # x1 is the marker
    nu     = c(0.50, 0.30, 0.40, 0.20, 0.60, 0.35),
    theta  = c(0.50, 0.60, 0.55, 0.65, 0.50, 0.60),  # residual variances
    psi    = 1.00,                                    # factor variance
    alpha  = 0.00                                     # factor mean
  )
}

# Implied mean/covariance of the correctly-specified one-factor population.
population_moments <- function(pop) {
  Sigma <- tcrossprod(pop$lambda) * pop$psi + diag(pop$theta)
  mu    <- pop$nu + pop$lambda * pop$alpha
  list(Sigma = Sigma, mu = mu)
}

# The estimated model: a one-factor CFA with a free mean structure (FIML and
# ML2S both require it). x1 is the unit-loading marker; the factor mean is fixed
# at 0 and the factor variance is free, so df = 27 moments - 18 free = 9.
model_syntax <- function() "f =~ x1 + x2 + x3 + x4 + x5 + x6"
