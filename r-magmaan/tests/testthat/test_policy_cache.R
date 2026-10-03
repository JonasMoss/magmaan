test_that("ordinary reporting and repeated nested sequences retain policy caches", {
  skip_if_not_installed("lavaan")
  base <- lavaan::HolzingerSwineford1939
  syntax <- c("visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6",
              "visual =~ x1 + a*x2 + a*x3\ntextual =~ x4 + x5 + x6",
              "visual =~ x1 + a*x2 + a*x3\ntextual =~ x4 + b*x5 + b*x6")
  for (estimator in c("FIML", "DWLS")) {
    d <- base; ord <- character()
    if (estimator == "FIML") d$x2[seq_len(nrow(d)) %% 7 == 0] <- NA_real_ else {
      ord <- paste0("x", 1:6)
      for (v in ord) d[[v]] <- ordered(cut(d[[v]], quantile(d[[v]], c(0, 1/3, 2/3, 1)),
                                         include.lowest = TRUE, labels = FALSE), levels = 1:3)
    }
    fits <- lapply(syntax, function(s) magmaan(magmaan_model(s, prototype = d,
      ordered = ord, parameterization = "theta"), d, estimator = estimator))
    sequence <- function(fs) list(anova(fs[[1]], fs[[2]]), anova(fs[[2]], fs[[3]]))
    value <- sequence(fits)
    expect_identical(sequence(fits), value)
    for (fit in fits) {
      expect_identical(vcov(fit), vcov(fit))
      expect_identical(summary(fit), summary(fit))
    }
    count <- function(fit) if (estimator == "DWLS") magmaanlab::inference_reuse(as_lab_fit(fit)) else
      magmaanlab::inference_reuse(magmaanlab::prepare_inference(as_lab_fit(fit)))
    expect_equal(vapply(fits, function(f) count(f)$ingredient_builds, numeric(1)),
                 if (estimator == "DWLS") c(2, 2, 1) else c(1, 1, 1))
    expect_identical(sequence(unserialize(serialize(fits, NULL))), value)
    if (.Platform$OS.type == "unix") {
      child <- parallel::mcparallel(sequence(fits), mc.set.seed = FALSE)
      expect_identical(parallel::mccollect(child)[[1]], value)
    }
  }
})
