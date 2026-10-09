test_that("objective modes reach continuous LS and preserve defaults", {
  set.seed(865121304)
  x <- matrix(rnorm(480), 120, 4) + rnorm(120)
  colnames(x) <- paste0("x", 1:4)
  spec <- magmaanlab::model_spec("f =~ x1 + x2 + x3 + x4", fixed_x = FALSE, meanstructure = TRUE)
  fit <- function(control = NULL) suppressWarnings(magmaanlab::fit_model(
    spec, as.data.frame(x), estimator = "ULS", control = control))
  baseline <- fit()
  explicit <- fit(list(coordinate_scaling = "information"))
  expect_identical(baseline$theta, explicit$theta)
  expect_identical(baseline$fmin, explicit$fmin)
  expect_identical(baseline$f_evals, explicit$f_evals)
  for (mode in c("objective", "objective_downward")) {
    ls <- fit(list(coordinate_scaling = mode))
    expect_identical(ls$coordinate_scaling, mode)
    expect_true(all(is.finite(ls$theta)))
    # The public typed error must survive its R string projection.
    expect_error(magmaanlab::fit_model(spec, as.data.frame(x),
      estimator = "ML", control = list(coordinate_scaling = mode)),
      "\\[Unsupported\\]")
    raw <- list(X = list(x))
    expect_error(magmaanlab::magmaan_core$fit_fiml(spec, raw,
      control = list(coordinate_scaling = mode)), "\\[Unsupported\\]")
    expect_error(magmaanlab::magmaan_core$fit_ml2s(spec, raw,
      control = list(coordinate_scaling = mode)), "\\[Unsupported\\]")
    expect_error(magmaanlab::fit_model(spec, as.data.frame(x), estimator = "ULS",
      optimizer = "port-nls", control = list(coordinate_scaling = mode)),
      "\\[Unsupported\\]")
    ml2s <- suppressWarnings(magmaanlab::magmaan_core$fit_ml2s(spec, raw,
      stage2_weight = "dls", control = list(coordinate_scaling = mode)))
    expect_identical(ml2s$coordinate_scaling, mode)
  }
})

test_that("ordinal and mixed LS reject explicit objective modes and preserve defaults", {
  skip_if_not_installed("lavaan")
  for (ordered in list(paste0("x", 1:6), paste0("x", 1:3))) {
    d <- lavaan::HolzingerSwineford1939
    for (v in ordered) d[[v]] <- as.integer(cut(d[[v]], 3))
    spec <- model_spec("f =~ x1 + x2 + x3\ng =~ x4 + x5 + x6",
                       ordered = ordered, fixed_x = FALSE)
    fit <- function(control = NULL) suppressWarnings(fit_model(
      spec, d, estimator = "DWLS", control = control))
    baseline <- fit()
    for (mode in c("none", "sample_units", "information")) {
      explicit <- fit(list(coordinate_scaling = mode))
      expect_identical(baseline$theta, explicit$theta)
      expect_identical(baseline$fmin, explicit$fmin)
      expect_identical(baseline$f_evals, explicit$f_evals)
    }
    for (mode in c("objective", "objective_downward")) {
      estimators <- if (length(ordered) == 6L) c("ULS", "DWLS", "WLS") else c("DWLS", "WLS")
      for (estimator in estimators) {
        expect_error(fit_model(spec, d, estimator = estimator,
          control = list(coordinate_scaling = mode)), "\\[Unsupported\\]")
      }
    }
  }
})
