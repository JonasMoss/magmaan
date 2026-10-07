test_that("ordinary fits emit the lab core scaling warning once", {
  set.seed(331053)
  dat <- data.frame(x = rnorm(200), y = 40 * rnorm(200))
  syntax <- "x ~~ x\ny ~~ y\nx ~~ y"
  for (estimator in "ULS") {
    messages <- character()
    fit <- withCallingHandlers(magmaan(syntax, dat, estimator = estimator,
                                       inference = FALSE), warning = function(w) {
      messages <<- c(messages, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
    expect_length(messages, 1)
    expect_match(messages, "observed variances differ by a factor.*consider rescaling")
    expect_no_warning(magmaan(syntax, as.data.frame(scale(dat)), estimator = estimator,
                              inference = FALSE))
  }
  expect_no_warning(magmaan(syntax, dat, estimator = "ML", inference = FALSE))
  dat$z <- as.integer(cut(rnorm(200), c(-Inf, 0, Inf)))
  model <- magmaan_model(paste(syntax, "z ~~ z\nz ~~ x + y", sep = "\n"),
                          ordered = "z", prototype = dat)
  expect_warning(magmaan(model, dat, estimator = "DWLS", inference = FALSE),
                  "observed variances differ by a factor.*consider rescaling")
})
