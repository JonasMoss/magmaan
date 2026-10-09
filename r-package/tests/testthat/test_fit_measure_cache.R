test_that('FIML measure ingredients preserve every result across call orders', {
  set.seed(148)
  f <- rnorm(180)
  data <- as.data.frame(sapply(1:5, function(j) .7 * f + rnorm(180)))
  names(data) <- paste0('x', 1:5)
  data$x3[seq(3, 180, by = 7)] <- NA_real_
  fit <- fit_model(model_spec('f =~ x1+x2+x3+x4+x5', meanstructure = TRUE),
                   data, estimator = 'FIML')
  expect_true(fit$converged)
  uncached <- fit
  attr(uncached, 'policy_cache') <- NULL
  calls <- list(ml = function(x) convention_fit_measures(x, 'ML'),
                mlr = function(x) convention_fit_measures(x, 'MLR'),
                policy = policy_fit_measures, wald = fit_index_intervals_misspec)
  expected <- lapply(calls, function(call) call(uncached))
  for (order in list(names(calls), rev(names(calls)))) {
    attr(fit, 'policy_cache') <- new.env(parent = emptyenv())
    for (name in order) expect_identical(calls[[name]](fit), expected[[name]])
    for (name in order) expect_identical(calls[[name]](fit), expected[[name]])
  }
  expect_false(identical(fit_index_intervals_misspec(fit, .95)$upper,
                         expected$wald$upper))

  # Even aliases that share the environment must not reuse another fitted point.
  changed <- fit_model(model_spec('f =~ x1+x2+x3+x4+x5', meanstructure = TRUE),
                       transform(data, x1 = x1 + .2 * x2), estimator = 'FIML')
  changed_uncached <- changed
  attr(changed_uncached, 'policy_cache') <- NULL
  old <- attr(fit, 'policy_cache')$fit_measures$value$native
  attr(changed, 'policy_cache') <- attr(fit, 'policy_cache')
  for (name in names(calls)) expect_identical(calls[[name]](changed),
                                            calls[[name]](changed_uncached))
  expect_false(identical(attr(changed, 'policy_cache')$fit_measures$value$native, old))
  for (name in names(calls)) expect_identical(calls[[name]](fit), expected[[name]])

  # Serialized external pointers are invalid and require a fresh native cache.
  restored <- unserialize(serialize(fit, NULL))
  for (name in names(calls)) expect_identical(calls[[name]](restored), expected[[name]])
})
