mplus_ordinary_input <- function(model, analysis = '', variable = 'NAMES=y1 y2 y3 x1 x2;', data = 'FILE=x;')
  paste('DATA:', data, '\nVARIABLE:', variable, '\nANALYSIS:', analysis,
        '\nMODEL:', model, '\n')

mplus_parameter_key <- function(pt) {
  lhs <- as.character(pt$lhs); rhs <- as.character(pt$rhs)
  swap <- pt$op == '~~' & rhs < lhs
  old <- lhs[swap]; lhs[swap] <- rhs[swap]; rhs[swap] <- old
  paste(lhs, pt$op, rhs, if (is.null(pt$group)) 1L else pt$group)
}

test_that('explicit Mplus joint models match frozen Demo and lavaan ML/FIML gates', {
  skip_if_not_installed('lavaan')
  # Synthetic P-MS08b data, independently reconstructed from its published seed.
  set.seed(58052)
  f <- rnorm(500)
  invisible(rnorm(2000))
  d0 <- data.frame(y1=.7*f+rnorm(500), y2=.7*f+rnorm(500),
                   y3=.7*f+rnorm(500), x1=rnorm(500), x2=rnorm(500))
  missing_rows <- sample.int(500, 50)
  for (missing in c(FALSE, TRUE)) for (with in c(FALSE, TRUE)) for (means in c(FALSE, TRUE)) {
    input <- mplus_ordinary_input(paste('f BY y1-y3; f ON x1 x2; x1 x2;',
      if (with) 'x1 WITH x2;' else '', if (means) '[x1 x2];' else ''),
      analysis='ESTIMATOR=ML;')
    d <- d0
    if (missing) d$x1[missing_rows] <- NA_real_
    reference <- lavaan::lavaan(paste('f =~ 1*y1 + y2 + y3; f ~ x1 + x2;',
      'f ~~ f; f ~ 0*1; y1 ~~ y1; y2 ~~ y2; y3 ~~ y3;',
      'y1 ~ 1; y2 ~ 1; y3 ~ 1; x1 ~ 1; x2 ~ 1;',
      'x1 ~~ x1 + x2; x2 ~~ x2;'), data=d, fixed.x=FALSE,
      meanstructure=TRUE, missing=if (missing) 'ml' else 'listwise',
      auto.var=FALSE, auto.cov.lv.x=FALSE, auto.cov.y=FALSE)
    spec <- magmaanlab::mplus_model(input)
    expect_false(spec$options$fixed_x)
    model <- magmaan_model(spec)
    expect_true(model$fittable)
    expect_identical(model$spec$mplus_source, input)
    estimator <- if (anyNA(d)) 'FIML' else 'ML'
    fit <- magmaan(model, d, estimator)
    pt <- fit$lab$partable
    ref <- lavaan::parTable(reference)
    hit <- match(mplus_parameter_key(ref), mplus_parameter_key(pt))
    expect_false(anyNA(hit))
    expect_equal(pt$est[hit], ref$est, tolerance = 2e-5)
    fm <- magmaanlab::fit_measures(fit$lab)
    expect_equal(fm$df, 4)
    expect_equal(fm$chisq, unname(lavaan::fitMeasures(reference, 'chisq')), tolerance = 2e-5)
    expect_equal(fm$chisq, if (missing) 1.596 else 1.974, tolerance = .002)
    expect_true(any(fit$inference$status$available))
    expect_equal(sum(fit$rows$used), 500)
    if (!missing && !with && !means) {
      # Each named edit returns the same accepted model and numeric reference.
      no_mean_input <- sub('ESTIMATOR=ML;',
        'ESTIMATOR=ML; MODEL=NOMEANSTRUCTURE; INFORMATION=EXPECTED;', input, fixed=TRUE)
      no_mean <- magmaan_model(magmaanlab::mplus_model(no_mean_input))
      e <- tryCatch(magmaan(no_mean, d), error=identity)
      expect_identical(e$reason, 'nomeanstructure')
      repaired <- sub('MODEL=NOMEANSTRUCTURE;', '', no_mean_input, fixed=TRUE)
      expect_equal(magmaan(magmaanlab::mplus_model(repaired), d)$lab$theta,
                   fit$lab$theta, tolerance=1e-8)
      summary_input <- sub('FILE=x;', 'FILE=x; TYPE=COVARIANCE; NOBSERVATIONS=500;', input, fixed=TRUE)
      summary <- magmaan_model(magmaanlab::mplus_model(summary_input))
      e <- tryCatch(magmaan(summary, d), error=identity)
      expect_identical(e$reason, 'summary_without_means')
      raw_file <- tempfile(fileext='.dat')
      write.table(d, raw_file, row.names=FALSE, col.names=FALSE, quote=FALSE)
      repaired <- sub('FILE=x; TYPE=COVARIANCE; NOBSERVATIONS=500;',
                      paste0('FILE=', raw_file, ';'), summary_input, fixed=TRUE)
      raw_spec <- magmaanlab::mplus_model(repaired)
      raw_data <- magmaanlab::mplus_data(raw_spec)
      expect_equal(magmaan(raw_spec, raw_data)$lab$theta, fit$lab$theta, tolerance=1e-8)
      unlink(raw_file)
    }
    # The public prepared fit and the lab's reconstruction share one contract.
    fresh <- magmaanlab::fit_model(spec, d, estimator = estimator)
    expect_equal(fit$lab$theta, fresh$theta, tolerance = 1e-8)
  }
})

test_that('Mplus refusals preserve tables and name the input edit', {
  base <- 'f BY y1-y3; f ON x1 x2;'
  conditional <- magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input(base)))
  expect_false(conditional$fittable)
  expect_true(nrow(conditional$spec$partable) > 0)
  e <- tryCatch(magmaan(conditional, data.frame()), error = identity)
  expect_s3_class(e, 'magmaan_mplus_error')
  expect_identical(e$reason, 'conditional_x')
  expect_match(e$edit, 'x1 x2;', fixed = TRUE)
  edited <- magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input(paste(base, 'x1 x2;'))))
  expect_true(edited$fittable)
  expect_error(magmaanlab::mplus_model(mplus_ordinary_input(paste(base, 'x1;'))), 'x1 x2;', fixed = TRUE)
  no_mean <- magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input(
    'f BY y1-y3;', 'MODEL=NOMEANSTRUCTURE; INFORMATION=EXPECTED;')))
  e <- tryCatch(magmaan(no_mean, data.frame()), error = identity)
  expect_identical(e$reason, 'nomeanstructure')
  expect_match(e$edit, 'Remove NOMEANSTRUCTURE', fixed = TRUE)
  expect_true(magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input('f BY y1-y3;')))$fittable)
  summary <- magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input(
    'f BY y1-y3;', variable = 'NAMES=y1 y2 y3;',
    data = 'FILE=x; TYPE=COVARIANCE; NOBSERVATIONS=500;')))
  e <- tryCatch(magmaan(summary, list()), error = identity)
  expect_identical(e$reason, 'summary_without_means')
  expect_match(e$edit, 'raw observations via magmaanlab::mplus_data()', fixed = TRUE)
  expect_true(magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input(
    'f BY y1-y3;', variable = 'NAMES=y1 y2 y3;')))$fittable)
  expect_error(magmaan_model(mplus_ordinary_input(base))) # strings are lavaan only
})

test_that('Mplus errors list all edits and printing shows fittability', {
  base <- 'f BY y1-y3; f ON x1 x2;'
  model <- magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input(
    base, 'MODEL=NOMEANSTRUCTURE; INFORMATION=EXPECTED;')))
  e <- tryCatch(magmaan(model, data.frame()), error = identity)
  expect_s3_class(e, 'magmaan_mplus_error')
  expect_identical(e$reason, c('conditional_x', 'nomeanstructure'))
  expect_identical(names(e$edit), e$reason)
  expect_identical(unname(e$edit), unlist(model$mplus_refusals, use.names=FALSE))
  expect_identical(conditionMessage(e), paste0(
    'magmaan(): this Mplus input needs 2 edits before magmaan() can fit it:\n',
    '1) ', e$edit[[1]], '\n2) ', e$edit[[2]]))
  expect_output(print(model), 'Mplus input:    not fittable; 2 input edit(s) needed (see $mplus_refusals)', fixed=TRUE)
  expect_output(print(model), paste0('    1) ', e$edit[[1]]), fixed=TRUE)
  expect_output(print(model), paste0('    2) ', e$edit[[2]]), fixed=TRUE)
  conditional <- magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input(base)))
  single <- tryCatch(magmaan(conditional, data.frame()), error=identity)
  expect_identical(names(single$edit), single$reason)
  expect_identical(conditionMessage(single), paste0(
    'magmaan(): Mplus input is unfittable: ', single$edit))
  expect_output(print(conditional), 'not fittable; 1 input edit(s) needed', fixed=TRUE)
  summary <- magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input(
    base, data='FILE=x; TYPE=COVARIANCE; NOBSERVATIONS=500;')))
  summary_error <- tryCatch(magmaan(summary, list()), error=identity)
  expect_identical(summary_error$reason, c('conditional_x', 'summary_without_means'))
  expect_identical(names(summary_error$edit), summary_error$reason)
  expect_identical(conditionMessage(summary_error), paste0(
    'magmaan(): this Mplus input needs 2 edits before magmaan() can fit it:\n',
    '1) ', summary_error$edit[[1]], '\n2) ', summary_error$edit[[2]]))
  joint <- magmaan_model(magmaanlab::mplus_model(mplus_ordinary_input(paste(base, 'x1 x2;'))))
  expect_output(print(joint), '  Mplus input:    fittable', fixed=TRUE)
  ordinary <- magmaan_model('f =~ y1 + y2 + y3')
  expect_identical(capture.output(print(ordinary)), c('magmaan model',
    '  observed:       y1, y2, y3', '  identification: marker; identified'))
})

test_that('continuous grouped and categorical Mplus models rebuild in a fresh process', {
  set.seed(5701)
  f <- rnorm(300)
  d <- data.frame(y1 = f + rnorm(300), y2 = .8*f + rnorm(300), y3 = .7*f + rnorm(300))
  grouped <- transform(d, g = rep(c(2, 1), each = 150))
  ordinal <- as.data.frame(lapply(d, function(x) ordered(cut(x, c(-Inf,-.5,.5,Inf), labels=FALSE))))
  cases <- list(
    list(d, 'ML', mplus_ordinary_input('f BY y1-y3;', variable='NAMES=y1 y2 y3;')),
    list(grouped, 'ML', mplus_ordinary_input('f BY y1-y3;',
      variable='NAMES=y1 y2 y3 g; GROUPING=g(2=b 1=a);')),
    list(ordinal, 'DWLS', mplus_ordinary_input('f BY y1-y3;',
      analysis='ESTIMATOR=WLSMV;', variable='NAMES=y1 y2 y3; CATEGORICAL=y1-y3;')))
  for (case in cases) {
    model <- magmaan_model(magmaanlab::mplus_model(case[[3]]), prototype=case[[1]])
    fit <- magmaan(model, case[[1]], case[[2]])
    expect_true(model$fittable)
    expect_false(is.null(model$prepared_cache$handle))
    rebuilt <- magmaan_model(magmaanlab::mplus_model(model$spec$mplus_source), prototype=case[[1]])
    expect_equal(magmaan(rebuilt, case[[1]], case[[2]])$lab$theta, fit$lab$theta, tolerance=1e-8)
    path <- tempfile(fileext='.rds'); result <- tempfile(fileext='.rds'); script <- tempfile(fileext='.R')
    saveRDS(list(model=model, data=case[[1]], estimator=case[[2]]), path)
    writeLines(c(sprintf('.libPaths(%s)', deparse(.libPaths(), width.cutoff=500)),
      sprintf('x <- readRDS(%s)', deparse(path)),
      'fit <- magmaan::magmaan(x$model, x$data, x$estimator)',
      sprintf('saveRDS(list(theta=fit$lab$theta, status=fit$inference$status), %s)', deparse(result))), script)
    output <- system2(file.path(R.home('bin'), 'Rscript'), shQuote(script), stdout=TRUE, stderr=TRUE)
    expect_null(attr(output, 'status'), info=paste(output, collapse='\n'))
    if (file.exists(result)) {
      got <- readRDS(result)
      expect_equal(got$theta, fit$lab$theta, tolerance=1e-8)
      expect_equal(got$status, fit$inference$status)
    }
    unlink(c(path, result, script))
  }
})
