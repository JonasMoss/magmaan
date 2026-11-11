.unfitted_cfa <- paste('visual =~ x1 + a*x2 + b*x3',
  'textual =~ x4 + x5 + x6', 'speed =~ x7 + x8 + x9', sep = '\n')
.unfitted_refs <- c('std', 'sb', 'mv', 'peba4', 'pall')

.unfitted_check <- function(null, model, alternative) {
  testthat::local_mocked_bindings(
    fit_model = function(...) stop("unexpected alternative fit"),
    refit_from_null = function(...) stop("unexpected refit recovery"),
    .package = "magmaanlab")
  one <- anova(null, model, references = .unfitted_refs)
  two <- anova(null, alternative, references = .unfitted_refs)
  plain <- function(x) as.data.frame(lapply(x, identity))
  score <- function(x) plain(x)[x$test == 'score', ]
  expect_equal(score(one), score(two), tolerance = 1e-10)
  expect_equal(attr(one, 'spectra')$score, attr(two, 'spectra')$score, tolerance = 1e-10)
  expect_null(attr(one, 'spectra')$lr)
  expect_identical(one$reason[one$test == 'lr'], 'alternative_not_fitted')
  expect_true(is.na(one$reference[one$test == 'lr']))
  expect_true(is.na(one$pvalue[one$test == 'lr']))
  expect_null(attr(one, 'refit'))
  expect_null(attr(one, 'reseed'))
  reverse <- anova(model, null, references = .unfitted_refs)
  expect_identical(plain(reverse), plain(one))
  expect_output(print(one), 'larger model was not fitted')
  # The model-spec lab path is also a public entry; it must share the policy.
  lab <- magmaanlab::policy_nested(model$spec, as_lab_fit(null))
  expect_equal(lab$score$statistic, one$statistic[one$test == 'score'][1], tolerance = 1e-10)
  one
}

test_that('unfitted metric-invariance alternative uses the restricted ML fit', {
  d <- hs()
  cfa <- sub('a*x2 + b*x3', 'x2 + x3', .unfitted_cfa, fixed = TRUE)
  free <- magmaan_model(cfa, prototype = d, group = 'school')
  equal <- magmaan_model(cfa, prototype = d, group = 'school', group.equal = 'loadings')
  metric <- magmaan(equal, d)
  configural <- magmaan(free, d)
  one <- .unfitted_check(metric, free, configural)
  explicit <- magmaanlab::nested_score_test(free$spec, as_lab_fit(metric), data = d,
    sensitivity = 'expected')
  expect_equal(one$statistic[one$test == 'score'][1], explicit$statistic_effective, tolerance = 1e-10)
  expect_equal(attr(one, 'spectra')$score, explicit$eigenvalues, tolerance = 1e-10)
  expect_equal(one$pvalue[one$reference %in% 'sb'], explicit$p_sb, tolerance = 1e-10)
  expect_equal(one$pvalue[one$reference %in% 'peba4'], explicit$p_peba4, tolerance = 1e-10)
  expect_identical(anova(metric, free)$reference, c('peba4', NA_character_))
})

test_that('unfitted equality alternative works under marker and std.lv', {
  d <- hs()
  results <- lapply(c('marker', 'std.lv'), function(id) {
    free <- magmaan_model(.unfitted_cfa, prototype = d, identification = id)
    equal <- magmaan_model(paste(.unfitted_cfa, 'a == b', sep = '\n'),
      prototype = d, identification = id)
    .unfitted_check(magmaan(equal, d), free, magmaan(free, d))
  })
  expect_equal(results[[1]]$statistic, results[[2]]$statistic, tolerance = 1e-6)
})

test_that('unfitted FIML alternative retains the observed score geometry', {
  d <- hs()
  set.seed(175)
  for (v in paste0('x', 1:9)) d[sample(nrow(d), floor(.2 * nrow(d))), v] <- NA
  free <- magmaan_model(.unfitted_cfa, prototype = d)
  equal <- magmaan_model(paste(.unfitted_cfa, 'a == b', sep = '\n'), prototype = d)
  .unfitted_check(magmaan(equal, d, estimator = 'FIML'), free,
    magmaan(free, d, estimator = 'FIML'))
})

test_that('unfitted comparisons refuse wrong roles, schemas and compatibility', {
  d <- hs()
  free <- magmaan_model(.unfitted_cfa, prototype = d)
  equal <- magmaan_model(paste(.unfitted_cfa, 'a == b', sep = '\n'), prototype = d)
  larger <- magmaan(free, d)
  expect_error(anova(larger, equal), 'score test is computed at the restricted fit')
  expect_error(anova(larger, equal, lavaan_compat = 'MLM'), 'both models fitted')
  other <- magmaan_model('visual =~ x1 + x2 + x3', prototype = d)
  expect_error(anova(larger, other), 'same data schema')
  grouped <- magmaan_model(.unfitted_cfa, prototype = d, group = 'school')
  expect_error(anova(larger, grouped), 'same data schema')
  ordered <- d
  for (v in paste0('x', 1:9)) ordered[[v]] <- ordered(cut(d[[v]],
    quantile(d[[v]], c(0, 1/3, 2/3, 1)), include.lowest = TRUE))
  m1 <- magmaan_model(.unfitted_cfa, prototype = ordered, ordered = paste0('x', 1:9))
  m0 <- magmaan_model(paste(.unfitted_cfa, 'a == b', sep = '\n'),
    prototype = ordered, ordered = paste0('x', 1:9))
  for (est in c('DWLS', 'RDWLS', 'ML')) {
    f <- magmaan(m0, ordered, estimator = est)
    table <- anova(f, m1)
    expect_true(all(table$reason == 'unsupported_model'))
  }
})

test_that('the printout of a score-only anova() omits the LR placeholder', {
  skip_if_not_installed('lavaan')
  d <- lavaan::HolzingerSwineford1939
  free <- magmaan_model(.unfitted_cfa, prototype = d)
  equal <- magmaan_model(paste(.unfitted_cfa, 'a == b', sep = '\n'), prototype = d)
  table <- anova(magmaan(equal, d), free)
  expect_true('lr' %in% table$test)
  out <- capture.output(print(table))
  expect_true(any(grepl('only the score test is reported', out, fixed = TRUE)))
  expect_false(any(grepl('likelihood ratio', out, fixed = TRUE)))
  expect_false(any(grepl('lr unavailable', out, fixed = TRUE)))
})
