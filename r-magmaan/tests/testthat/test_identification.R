test_that("construction stores identification and fitting refuses ridges before data preparation", {
  ridge <- magmaan_model("f =~ NA*x1 + x2 + x3")
  report <- ridge$identification_report
  expect_identical(report$status, "unidentified")
  expect_false(report$directions_at_estimate)
  for (policy in c("unrestricted", "psd", "barrier")) {
    err <- tryCatch(magmaan(ridge, data.frame(), covariance = policy, inference = FALSE),
                    magmaan_identification_error = identity)
    expect_s3_class(err, "magmaan_identification_error")
    expect_identical(err$status, report$status)
    expect_identical(err$reason, report$reason)
    expect_identical(err$directions, report$null_directions)
    expect_true(all(c("f=~x1", "f~~f") %in% rownames(err$directions)))
    expect_match(conditionMessage(err), "Free directions")
  }
  expect_identical(capture.output(print(ridge)), c("magmaan model",
    "  observed:       x1, x2, x3", "  identification: marker; NOT identified"))
})

test_that("marker, std.lv and unchecked models remain fit-able", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  for (setting in c("marker", "std.lv")) {
    m <- magmaan_model("f =~ x1 + x2 + x3", identification = setting)
    expect_identical(m$identification_report$status, "identified")
    fit <- magmaan(m, d, inference = FALSE)
    expect_s3_class(fit, "magmaan")
    expect_true(as_lab_fit(fit)$converged)
    expect_identical(as_lab_fit(fit)$diagnostics$identification, m$identification_report)
  }
  unchecked <- magmaan_model("f =~ x1 + a*x2 + b*x3\nb == a*a")
  expect_identical(unchecked$identification_report$status, "unchecked")
  expect_output(print(unchecked), "unchecked (nonlinear_constraints)", fixed = TRUE)
  expect_s3_class(suppressWarnings(magmaan(unchecked, d, inference = FALSE)), "magmaan")
})

test_that("ordinary repeated datasets reuse the construction report", {
  skip_if_not_installed("lavaan")
  count <- magmaanlab:::prepared_identification_count_impl
  before <- count()
  m <- magmaan_model("f =~ x1 + x2 + x3")
  expect_equal(count(), before + 1)
  d <- lavaan::HolzingerSwineford1939
  for (rows in list(seq_len(nrow(d)), rev(seq_len(nrow(d))))) {
    fit <- magmaan(m, d[rows, ], inference = FALSE)
    expect_identical(as_lab_fit(fit)$diagnostics$identification, m$identification_report)
    expect_equal(count(), before + 1)
  }
})

test_that("ML2S fallback retains the prepared report in its Stage-2 diagnostics", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  m <- magmaan_model("f =~ x1 + x2 + x3")
  count <- magmaanlab:::prepared_identification_count_impl
  before <- count()
  for (policy in c("unrestricted", "psd", "barrier")) {
    fit <- suppressMessages(suppressWarnings(magmaan(m, d, estimator = "ML2S",
      covariance = policy, inference = FALSE)))
    expect_identical(as_lab_fit(fit)$diagnostics$identification, m$identification_report)
    expect_equal(count(), before)
  }
})

test_that("ordinary recovery refuses an unidentified saved alternative before refitting", {
  m <- magmaan_model("f =~ NA*x1 + x2 + x3")
  result <- list(score = list(reason = "not_converged",
    detail = "the alternative fits worse than the null"))
  alternative <- list(diagnostics = list(identification = m$identification_report))
  expect_error(magmaan:::.nested_recovery(result, alternative, NULL, NULL, "score"),
               class = "magmaan_identification_error")
})
