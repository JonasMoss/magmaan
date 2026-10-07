test_that("ordinary fit measures preserve the lab policy and stable rows", {
  d <- hs()
  codes <- c("rmsea", "cfi", "tli", "srmr", "crmr", "logl",
             "unrestricted_logl", "aic", "bic")
  for (estimator in c("ML", "FIML", "ULS", "GLS", "ML2S")) {
    data <- d
    if (estimator == "FIML") data$x2[seq(2, nrow(data), by = 7)] <- NA
    fit <- magmaan(cfa, data, estimator = estimator, inference = FALSE)
    expected <- magmaanlab::policy_fit_measures(as_lab_fit(fit))
    expected$index[expected$index == "unrestricted.logl"] <- "unrestricted_logl"
    got <- fit_measures(fit)
    expect_identical(got, expected)
    expect_identical(got$index, if (estimator %in% c("ML", "FIML")) codes[c(1:3, 6:9, 4)] else codes[1:4])
    expect_identical(names(got), c("index", "estimate", "reason"))
    expect_true(all(!is.na(got$reason[is.na(got$estimate)])))
  }
  for (group in c(FALSE, TRUE)) {
    data <- ordinal_hs()
    model <- magmaan_model(cfa, prototype = data, ordered = paste0("x", 1:6),
                           group = if (group) "school" else NULL)
    fit <- magmaan(model, data, estimator = "DWLS", inference = FALSE)
    expect_identical(fit_measures(fit), magmaanlab::policy_fit_measures(as_lab_fit(fit)))
    expect_identical(fit_measures(fit)$index, codes[c(1:3, 5, 4)])
  }
})

test_that("summary computes fit measures only on request and prints reasons", {
  fit <- magmaan(cfa, hs(), inference = FALSE)
  expect_null(summary(fit)$fit_measures)
  out <- summary(fit, fit_measures = TRUE)
  expect_identical(out$fit_measures, fit_measures(fit))
  expect_output(print(out), "Policy fit measures")
  fit$lab$converged <- FALSE
  got <- fit_measures(fit)
  expect_true(all(is.na(got$estimate)))
  expect_true(all(grepl("^not_converged:", got$reason)))
  expect_output(print(summary(fit, fit_measures = TRUE)), "not_converged")
  expect_error(fit_measures(fit, lavaan_compat = "MLR"), "convergence verdict")
  expect_error(summary(fit, fit_measures = TRUE, lavaan_compat = "MLR"), "convergence verdict")
  expect_error(fit_measures(list()), "supply a magmaan")
  expect_error(summary(fit, fit_measures = NA), "TRUE or FALSE")
})

test_that("ordinary compatibility measures preserve lab names and convention", {
  fit <- magmaan(cfa, hs(), inference=FALSE)
  for (convention in c("ML", "MLM", "MLR")) {
    actual <- fit_measures(fit, lavaan_compat=convention)
    expect_identical(actual, magmaanlab::convention_fit_measures(as_lab_fit(fit), convention))
    expect_identical(attr(actual,"lavaan_compat"), convention)
    expect_true("chisq" %in% actual$index)
    expect_identical(summary(fit, fit_measures=TRUE, lavaan_compat=convention)$fit_measures, actual)
  }
  expect_error(fit_measures(fit,lavaan_compat="WLSMV"),"different fitted estimator")
})


test_that("ordinary FIML compatibility preserves typed unavailable rows", {
  d <- hs()
  d$x2[seq(1, nrow(d), by=5)] <- NA
  fit <- magmaan(cfa, d, estimator="FIML", inference=FALSE)
  for (convention in c("ML", "MLR")) {
    actual <- fit_measures(fit, lavaan_compat=convention)
    expect_identical(actual, magmaanlab::convention_fit_measures(as_lab_fit(fit), convention))
    expect_true(all(is.na(actual$estimate)))
    expect_true(all(actual$reason == "unsupported_model: not yet validated against lavaan"))
  }
})
