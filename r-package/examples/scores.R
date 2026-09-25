# Reusable score objects and covariance/Wald composition; no simulation run.
suppressPackageStartupMessages(library(magmaanlab))
set.seed(731)
x <- matrix(rnorm(400 * 6), 400, 6) + rnorm(400)
colnames(x) <- paste0('x', 1:6)
x <- as.data.frame(x)
syntax <- 'f =~ x1 + a*x2 + b*x3 + x4 + x5 + x6'
control <- list(max_iter = 3000L, ftol = 1e-12, gtol = 1e-8)
reject <- function(expr, pattern) {
  e <- tryCatch({force(expr); NULL}, error = identity)
  stopifnot(inherits(e, 'error'), grepl(pattern, conditionMessage(e)))
}
check_score <- function(fit, data = NULL) {
  context <- prepare_inference(fit, data)
  components <- score_components(context)
  projected <- project_scores(components, retain_rows = TRUE)
  reference <- score_spectrum(projected)
  p <- calibrate_quadratic(reference, c('sb', 'peba2', 'peba4'))
  old <- global_score_flip_test(fit, data, n_flips = 31L, seed = 19)
  stopifnot(abs(projected$statistic - old$statistic_effective) < 1e-6,
            max(abs(reference$eigenvalues - old$eigenvalues)) < 1e-6,
            abs(p$p_value[1] - old$p_mean_scaled) < 1e-7,
            resample_scores(projected, 31L, 19)$p_value == old$p_effective,
            identical(p, calibrate_quadratic(reference, c('sb', 'peba2', 'peba4'))),
            abs(calibrate_quadratic(projected, 'sb')$p_value - p$p_value[1]) < 1e-9)
  reject({reference$statistic <- 0}, 'locked')
  reject(resample_scores(project_scores(components), 31L), 'retain rows')
  reject(project_scores(unserialize(serialize(components, NULL))), 'prepare it again')
  if (fit$estimator != 'ML2S') {
    s <- scores(context, space = 'saturated')
    stopifnot(max(abs(s$score - colSums(s$rows))) < 1e-10,
              max(abs(s$score - components$score)) < 1e-7)
    I <- inference_information(context, 'expected')
    V <- parameter_covariance(context, I)
    # The same covariance is reusable for any number of Wald contrasts.
    R <- diag(length(context$theta))[1:2,,drop=FALSE]
    w <- wald_test(context, R, V)
    expected <- drop(crossprod(R %*% context$theta, solve(R %*% V %*% t(R), R %*% context$theta)))
    stopifnot(abs(w$chi2 - expected) < 1e-7)
    context2 <- prepare_inference(fit, data)
    reject(wald_test(context2, R, V), 'another fit snapshot')
    reject(parameter_covariance(context2, I), 'another fit snapshot')
    saved <- context$theta
    fit$theta[] <- 0
    stopifnot(identical(context$theta, saved))
  } else {
    stopifnot(components$influence_rows,
              max(abs(colSums(components$rows) - components$score)) > 1e-5)
  }
  invisible(context)
}
model <- prepare_model(syntax, meanstructure = TRUE)
data <- prepare_data(model, x)
fit <- estimate(model, data, control = control)
context <- check_score(fit, x)
# A prepared dataset and the fitting data retained by estimate are also accepted.
stopifnot(max(abs(score_components(prepare_inference(fit, data))$score - score_components(context)$score)) < 1e-8,
          max(abs(score_components(prepare_inference(fit))$score - score_components(context)$score)) < 1e-8)
y <- x; y$x1 <- y$x1 + .2
reject(prepare_inference(fit, y), 'match the fitted sample')
missing <- x; missing[seq(1,400,5),2] <- NA; missing[seq(3,400,7),5] <- NA
ff <- fit_model(syntax, missing, estimator = 'FIML', control = control)
check_score(ff)
f2 <- fit_model(syntax, missing, estimator = 'ML2S', control = control)
check_score(f2)

# The nested score uses an H1 model only, without an H1 fit.
h0 <- fit_model(paste(syntax, 'a == b', sep = '\n'), x, meanstructure = TRUE, control = control)
h1 <- model_spec(syntax, meanstructure = TRUE)
nc <- prepare_inference(h0, x)
np <- project_scores(score_components(nc, H1 = h1), retain_rows = TRUE)
old <- score_flip_test(h1, h0, x, n_flips = 31L, seed = 19, calibration = 'effective')
stopifnot(abs(np$statistic - old$statistic_effective) < 1e-6,
          resample_scores(np,31L,19)$p_value == old$p_effective)

# Singular meat remains valid for a mixture reference, but not a sandwich inverse.
q <- score_quadratic(c(1,2), diag(2), diag(c(1,0)))
e <- score_spectrum(q)
stopifnot(identical(as.numeric(e$eigenvalues), c(0,1)), e$df == 2L,
          is.finite(calibrate_quadratic(e,'sb')$p_value))
reject(score_sandwich(q), 'singular')
reject(score_quadratic(c(1,2), diag(c(1,-1))), 'positive definite')

# A precomputed LR/GOF spectrum uses the same calibration operation.
lr <- quadratic_reference(8, 6, rep(1,6))
p <- calibrate_quadratic(lr,c('std','sb','peba2','peba4'))
stopifnot(max(abs(p$p_value - pchisq(8,6,lower.tail=FALSE))) < 1e-8)
cat('Reusable score, inference and Wald checks passed.\n')

# Legacy GOF/LR consumers use the prepared native fit context, and their
# returned reference laws can be calibrated repeatedly without refitting.
gof <- fmg_tests(context, tests = 'peba4_ml')
stopifnot(abs(gof$p_value - fmg_tests(fit, data=x, tests='peba4_ml')$p_value) < 1e-8)
nested <- robust_nested_lrt(context, nc)
old_nested <- robust_nested_lrt(fit, h0, x)
stopifnot(abs(nested$T_diff - old_nested$T_diff) < 1e-8,
          max(abs(nested$eigenvalues - old_nested$eigenvalues)) < 1e-8)
nref <- quadratic_reference(nested$T_diff, nested$df_diff, nested$eigenvalues)
stopifnot(abs(calibrate_quadratic(nref,'sb')$p_value -
                fmg_nested(context,nc,tests='sb_ml')$p_value) < 1e-8)
# A modified extracted fit must invalidate cached structure/sample context.
changed <- context$fit
changed$nobs <- changed$nobs * 2
cached <- magmaanlab:::infer_information_expected(changed)
attr(changed,'magmaan_context') <- NULL
fresh <- magmaanlab:::infer_information_expected(changed)
stopifnot(max(abs(cached-fresh)) < 1e-8)
cat('Prepared GOF/LR compatibility and invalidation checks passed.\n')

# Reuse extracted ingredients for a supplied-matrix hypothesis construction.
parts <- score_components(context)
supplied <- score_components_from_matrices(parts$score, parts$rows,
  parts$sensitivity, parts$metric, parts$nuisance, parts$directions)
stopifnot(abs(project_scores(supplied)$statistic - project_scores(parts)$statistic) < 1e-10)
