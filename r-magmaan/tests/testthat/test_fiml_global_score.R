test_that("FIML global score follows expected geometry with complete, MCAR and MAR data", {
  skip_if_not_installed("lavaan")
  syntax <- "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6"
  for (missing in c("complete", "mcar", "mar")) for (group in list(NULL, "school")) {
    d <- lavaan::HolzingerSwineford1939
    set.seed(172)
    for (v in c("x2", "x3", "x5", "x6")) {
      prob <- if (missing == "complete") rep(0, nrow(d)) else
        if (missing == "mcar") rep(.3, nrow(d)) else
          plogis(-.9 + as.numeric(scale(d$x1)))
      d[[v]][runif(nrow(d)) < prob] <- NA
    }
    fit <- suppressWarnings(magmaan(magmaan_model(syntax, prototype = d, group = group),
                                   d, estimator = "FIML"))
    got <- fit$inference$global_score
    explicit <- magmaanlab::global_score_flip_test(as_lab_fit(fit), n_flips = 3L,
      sensitivity = "expected", metric = "expected")
    expect_equal(got$statistic, explicit$statistic_effective, tolerance = 1e-8)
    expect_equal(got$df, explicit$df)
    expect_equal(got$eigenvalues, explicit$eigenvalues, tolerance = 1e-8)
    expect_equal(got$p_peba4, magmaanlab:::infer_fmg_test(got$statistic,
      got$df, got$eigenvalues, "peba", 4)$p_value, tolerance = 1e-8)
  }
})
